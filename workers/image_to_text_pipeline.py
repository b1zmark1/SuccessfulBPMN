from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from graph_builder.pipeline import build_graph_from_ensemble
from narrator.orchestrator import run_narration

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXECUTABLE = sys.executable


def _repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


def _fix_mojibake(text: str) -> str:
    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text


def _read_image_from_url(image_url: str, workdir: Path) -> Path:
    parsed = urlparse(image_url)

    if image_url.startswith("data:image/"):
        header, encoded = image_url.split(",", 1)
        if ";base64" not in header:
            raise ValueError("Unsupported data URL format: expected base64 data URL")
        ext = "png"
        if "image/jpeg" in header:
            ext = "jpg"
        elif "image/webp" in header:
            ext = "webp"
        image_path = workdir / f"input.{ext}"
        image_path.write_bytes(base64.b64decode(encoded))
        return image_path

    if parsed.scheme in {"http", "https"}:
        image_path = workdir / "input_from_url.png"
        with urlopen(image_url, timeout=60) as response:
            image_path.write_bytes(response.read())
        return image_path

    local_path = Path(image_url)
    if not local_path.is_absolute():
        local_path = REPO_ROOT / local_path
    if not local_path.exists():
        raise FileNotFoundError(f"Image path not found: {local_path}")
    return local_path


async def _run_subprocess(cmd: list[str], cwd: Path) -> None:
    tag = Path(cmd[1]).name if len(cmd) > 1 else "subprocess"
    print(f"[pipeline] >>> {tag}: {' '.join(cmd)}", flush=True)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    out = stdout.decode("utf-8", errors="ignore")
    err = stderr.decode("utf-8", errors="ignore")
    if out.strip():
        for line in out.rstrip().splitlines():
            print(f"[{tag}] {line}", flush=True)
    if err.strip():
        for line in err.rstrip().splitlines():
            print(f"[{tag}:err] {line}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{out}\n{err}")


def _extract_text_from_ocr_json(ocr_json_path: Path) -> str:
    payload = json.loads(ocr_json_path.read_text(encoding="utf-8"))
    blocks = payload.get("blocks", [])
    lines: list[str] = []
    for block in blocks:
        text = block.get("text")
        if isinstance(text, str):
            cleaned = _fix_mojibake(text).strip()
            if cleaned:
                lines.append(cleaned)
    return "\n".join(lines)


def _narrator_output_format(mode: str) -> str:
    normalized = (mode or "").strip().lower()
    if normalized == "table":
        return "table"
    # UI mode "text" maps to narrator policy format "narrative".
    return "narrative"


def _prefer_detect_gpu() -> bool:
    raw = os.getenv("DETECT_RES_GPU", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


async def run_image_to_text_pipeline(
    image_url: str,
    *,
    narrator_mode: str = "table",
) -> dict[str, Any]:
    if not isinstance(image_url, str) or not image_url.strip():
        raise ValueError("image_url is required")

    with tempfile.TemporaryDirectory(prefix="job_image_to_text_") as tmp:
        use_gpu = _cuda_available()
        tmp_dir = Path(tmp)
        input_image = _read_image_from_url(image_url, tmp_dir)
        out_yolox = tmp_dir / "out_yolox"
        out_text = tmp_dir / "out_text"
        out_ocr = tmp_dir / "out_ocr"
        out_labeled = tmp_dir / "out_labeled"

        # Определяем размер картинки заранее — он решает, какой OCR-стек запускать.
        try:
            from PIL import Image as _PILImage  # type: ignore
            with _PILImage.open(str(input_image)) as _img_probe:
                _img_w, _img_h = _img_probe.size
        except Exception:
            _img_w, _img_h = 0, 0
        _max_side = max(_img_w, _img_h)
        _size_threshold = int(os.getenv("IMG_PIPELINE_SIZE_THRESHOLD", "2500"))
        # Маленькая картинка → старый legacy-pipeline (detect_res + tesseract + heavy + label_res),
        # он на маленьких давал значительно лучше качество чем full-image EasyOCR.
        # Большая → новый crop-based pipeline (ocr_full_sweep + label_aggregate).
        use_legacy_small_pipeline = _max_side > 0 and _max_side < _size_threshold
        forced_mode = os.getenv("IMG_PIPELINE_MODE", "").strip().lower()
        if forced_mode == "legacy":
            use_legacy_small_pipeline = True
        elif forced_mode == "modern":
            use_legacy_small_pipeline = False

        print(
            f"[pipeline] image_size={_img_w}x{_img_h} max_side={_max_side} "
            f"threshold={_size_threshold} -> "
            f"{'LEGACY (detect_res+tesseract+heavy+label_res)' if use_legacy_small_pipeline else 'MODERN (ocr_full_sweep+label_aggregate)'}",
            flush=True,
        )

        # ────────────────────────────────────────────────────────────────────
        # ШАГ 1: YOLOX detection — общий для обоих pipeline'ов.
        # ────────────────────────────────────────────────────────────────────
        ensemble_cmd = [
            PYTHON_EXECUTABLE,
            _repo_path("preprocanddetect", "ensemble_infer.py"),
            "--images",
            str(input_image),
            "--out",
            str(out_yolox),
            "--exp-file",
            _repo_path("results", "yolox_tiny_bpmn.py"),
            "--ckpt",
            _repo_path("results", "best_ckpt.pth"),
            "--device",
            "gpu" if use_gpu else "cpu",
            "--conf",
            os.getenv("YOLOX_CONF_THRESHOLD", "0.4"),
            "--lang",
            "ru",
            "--no-text",
        ]
        if use_gpu:
            ensemble_cmd.append("--fp16")
        await _run_subprocess(ensemble_cmd, cwd=REPO_ROOT)

        input_stem = input_image.stem
        ensemble_json_path = out_yolox / f"{input_stem}_ensemble.json"
        if not ensemble_json_path.exists():
            raise FileNotFoundError(f"Ensemble file not found: {ensemble_json_path}")

        if use_legacy_small_pipeline:
            # ────────────────────────────────────────────────────────────────
            # LEGACY pipeline для маленьких BPMN-картинок — В ИСХОДНОМ ВИДЕ
            # до моих правок: detect_res → tesseract_fast → label_res.
            # БЕЗ heavy_pass (EasyOCR full sweep на маленьких картинках
            # фрагментирует слова и портит результат Tesseract'а).
            # ────────────────────────────────────────────────────────────────
            detect_cmd = [
                PYTHON_EXECUTABLE,
                _repo_path("preprocanddetect", "detect_res.py"),
                "--input",
                str(input_image),
                "--outdir",
                str(out_text),
                "--lang",
                "ru",
                "--download-enabled",
            ]
            if _prefer_detect_gpu() and use_gpu:
                detect_cmd.append("--gpu")
            try:
                await _run_subprocess(detect_cmd, cwd=REPO_ROOT)
            except RuntimeError:
                if "--gpu" not in detect_cmd:
                    raise
                detect_cmd_no_gpu = [arg for arg in detect_cmd if arg != "--gpu"]
                await _run_subprocess(detect_cmd_no_gpu, cwd=REPO_ROOT)

            model_image_path = out_text / "00_model.png"
            blocks_path = out_text / "text_blocks.json"
            if not model_image_path.exists() or not blocks_path.exists():
                raise FileNotFoundError("Text detection artifacts not found after detect_res.py")

            tess_cmd = [
                PYTHON_EXECUTABLE,
                _repo_path("preprocanddetect", "ocr_tesseract_fast.py"),
                "--input",
                str(input_image),
                "--blocks",
                str(blocks_path),
                "--outdir",
                str(out_ocr),
                "--lang",
                "rus",
                "--max-side",
                "4096",
                "--upscale-factor",
                "2.0",
                "--pad-px",
                "10",
                "--psm-block",
                "6",
                "--psm-single",
                "7",
                "--psm-raw-line",
                "13",
                "--try-rotate-90",
                "--refine-text-bbox",
                "--cc-max-area-frac",
                "0.18",
                "--cc-min-area-px",
                "20",
                "--jobs",
                "4",
            ]
            await _run_subprocess(tess_cmd, cwd=REPO_ROOT)

            ocr_json_path = out_ocr / "ocr.json"
            if not ocr_json_path.exists():
                raise FileNotFoundError(f"OCR result not found: {ocr_json_path}")

            label_cmd = [
                PYTHON_EXECUTABLE,
                _repo_path("preprocanddetect", "label_res.py"),
                "--ensemble",
                str(ensemble_json_path),
                "--text-blocks",
                str(blocks_path),
                "--ocr",
                str(ocr_json_path),
                "--image",
                str(model_image_path),
                "--outdir",
                str(out_labeled),
            ]
            await _run_subprocess(label_cmd, cwd=REPO_ROOT)
        else:
            # ────────────────────────────────────────────────────────────────
            # MODERN pipeline для больших картинок.
            # ocr_full_sweep.py (crop-based по YOLOX-shape) + label_aggregate.py (containment).
            # ────────────────────────────────────────────────────────────────
            ocr_cmd = [
                PYTHON_EXECUTABLE,
                _repo_path("preprocanddetect", "ocr_full_sweep.py"),
                "--input",
                str(input_image),
                "--ensemble-json",
                str(ensemble_json_path),
                "--outdir",
                str(out_ocr),
                "--lang",
                os.getenv("OCR_LANG", "ru"),
                "--mode",
                os.getenv("OCR_MODE", "crop"),
                "--shape-pad",
                os.getenv("OCR_SHAPE_PAD", "8"),
                "--margin-ratio",
                os.getenv("OCR_MARGIN_RATIO", "0.1"),
            ]
            if use_gpu:
                ocr_cmd.append("--gpu")
            await _run_subprocess(ocr_cmd, cwd=REPO_ROOT)

            ocr_json_path = out_ocr / "ocr.json"
            if not ocr_json_path.exists():
                raise FileNotFoundError(f"OCR result not found: {ocr_json_path}")

            label_cmd = [
                PYTHON_EXECUTABLE,
                _repo_path("preprocanddetect", "label_aggregate.py"),
                "--ensemble",
                str(ensemble_json_path),
                "--ocr",
                str(ocr_json_path),
                "--outdir",
                str(out_labeled),
                "--min-overlap-ratio",
                os.getenv("LABEL_MIN_OVERLAP", "0.5"),
                "--min-text-conf",
                os.getenv("LABEL_MIN_TEXT_CONF", "0.3"),
            ]
            await _run_subprocess(label_cmd, cwd=REPO_ROOT)

        merged_ensemble_path = out_labeled / f"{input_stem}_ensemble_merged_labeled.json"
        if not merged_ensemble_path.exists():
            raise FileNotFoundError(f"Merged ensemble file not found: {merged_ensemble_path}")

        ensemble_payload = json.loads(merged_ensemble_path.read_text(encoding="utf-8"))

        # Запускаем pipeline пошагово (вместо build_graph_from_ensemble) чтобы
        # вытащить stats для диагностики ролей/лейнов. Сам serialize контракт-чистый.
        try:
            from graph_builder.normalize import normalize_ensemble_input
            from graph_builder.grouping import group_normalized_detections
            from graph_builder.direction import infer_process_direction
            from graph_builder.nodes import build_graph_nodes
            from graph_builder.containers import assign_container_hierarchy
            from graph_builder.edge_candidates import build_edge_candidates
            from graph_builder.edges import finalize_edges
            from graph_builder.text_merge import merge_adjacent_text_nodes
            from graph_builder.text_hooks import attach_text_placeholders_and_hooks
            from graph_builder.title_hints import assign_title_hints
            from graph_builder.lane_detection import detect_lanes_geometrically
            from graph_builder.diagnostics import build_uncertainty_and_diagnostics
            from graph_builder.serialize import serialize_graph_output

            _x = normalize_ensemble_input(ensemble_payload)
            _x = group_normalized_detections(_x)
            _x = infer_process_direction(_x)
            _x = build_graph_nodes(_x)
            _x = assign_container_hierarchy(_x)
            _x = build_edge_candidates(_x)
            _x = finalize_edges(_x)
            _x = merge_adjacent_text_nodes(_x)
            _x = assign_title_hints(_x)
            _x = attach_text_placeholders_and_hooks(_x)
            _x = detect_lanes_geometrically(_x)
            _x = build_uncertainty_and_diagnostics(_x)

            _ns = _x.get("node_stats", {})
            _lane = _x.get("lane_detection_stats", {})
            print(
                f"[pipeline] graph: total={_ns.get('total','?')} "
                f"shape={_ns.get('shape','?')} text={_ns.get('text','?')} "
                f"container={_ns.get('container','?')} | "
                f"lane: clusters={_lane.get('clusters_total','?')} "
                f"named={_lane.get('lanes_named','?')} "
                f"shapes_with_lane={_lane.get('shapes_with_lane','?')} "
                f"overrides={_lane.get('lane_role_overrides','?')} "
                f"cleared={_lane.get('lane_role_cleared','?')}",
                flush=True,
            )
            graph_payload = serialize_graph_output(_x)
        except Exception:
            # На случай если что-то поменялось в graph_builder — фолбэк на штатный путь.
            graph_payload = build_graph_from_ensemble(ensemble_payload)
        narration = run_narration(
            graph_payload=graph_payload,
            policy_overrides={
                "output_format": _narrator_output_format(narrator_mode),
                "missing_text_policy": "explicit_placeholder",
            },
            runtime_overrides={
                "model_path": os.getenv(
                    "NARRATOR_MODEL_PATH",
                    _repo_path("narrator", "qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf"),
                ),
                "n_ctx": int(os.getenv("NARRATOR_N_CTX", "4096")),
                "n_threads": int(os.getenv("NARRATOR_THREADS", "12")),
            },
        )

        return {
            "text": _fix_mojibake(str(narration.get("text", ""))),
            "narrator_mode": narrator_mode,
            "narrator_status": str(narration.get("status", "unknown")),
            "ocr_engine": "tesseract",
            "ocr_text": _extract_text_from_ocr_json(ocr_json_path),
        }


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Image-to-text pipeline worker entry.")
    parser.add_argument("--image-url", required=True, help="image URL, data URL, or local file path")
    parser.add_argument("--narrator-mode", choices=["text", "table"], default="table")
    args = parser.parse_args()
    result = await run_image_to_text_pipeline(args.image_url, narrator_mode=args.narrator_mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
