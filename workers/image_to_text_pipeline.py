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
        out_ocr = tmp_dir / "out_ocr"
        out_labeled = tmp_dir / "out_labeled"

        # ────────────────────────────────────────────────────────────────────
        # ШАГ 1/3: YOLOX detection (BPMN objects: tasks, gateways, events, lanes...).
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

        # ────────────────────────────────────────────────────────────────────
        # ШАГ 2/3: ONE OCR pass — EasyOCR на ОРИГИНАЛЬНОЙ картинке (cap 2500).
        # Coord transform: OCR_capped → ORIGINAL → MODEL (через resize_ratio из ensemble).
        # Заменяет detect_res.py + ocr_tesseract_fast.py + ocr_heavy_pass.py.
        # ────────────────────────────────────────────────────────────────────
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

        # ────────────────────────────────────────────────────────────────────
        # ШАГ 3/3: Containment-based aggregation OCR-блоков в YOLOX-объекты.
        # Заменяет label_res.py (1-to-1 матчинг).
        # ────────────────────────────────────────────────────────────────────
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
