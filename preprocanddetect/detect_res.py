from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from preprocess import PreprocessConfig, preprocess
from detect import DetectionConfig, detect_text_boxes, draw_text_boxes
from text_group import GroupConfig, build_text_blocks, draw_text_blocks


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _imwrite(path: str, img: np.ndarray) -> None:
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)

    ap.add_argument("--outdir", dest="outdir", default=None)
    ap.add_argument("--out-dir", dest="outdir", default=None)

    # preprocess
    ap.add_argument("--max-side", type=int, default=1800)
    ap.add_argument("--denoise-h", type=int, default=10)
    ap.add_argument("--adaptive-block", type=int, default=35)
    ap.add_argument("--adaptive-c", type=int, default=11)
    ap.add_argument("--geom-close-kernel", type=int, default=0)

    # detect (EasyOCR)
    ap.add_argument("--lang", type=str, default="ru")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--download-enabled", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--model-dir", type=str, default=None)

    ap.add_argument("--width-ths", type=float, default=0.12)
    ap.add_argument("--ycenter-ths", type=float, default=0.30)
    ap.add_argument("--height-ths", type=float, default=0.30)

    # grouping
    ap.add_argument("--gap-mult", type=float, default=1.6)
    ap.add_argument("--y-tol-mult", type=float, default=0.6)
    ap.add_argument("--line-gap-mult", type=float, default=1.3)

    args = ap.parse_args()
    if not args.outdir:
        raise SystemExit("Missing output dir. Use --outdir or --out-dir")
    return args


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    out_dir = Path(args.outdir).expanduser().resolve()
    _ensure_dir(str(out_dir))

    p_cfg = PreprocessConfig(
        max_side=args.max_side,
        denoise_h=args.denoise_h,
        adaptive_block=args.adaptive_block,
        adaptive_c=args.adaptive_c,
        geom_close_kernel=args.geom_close_kernel,
    )
    pre = preprocess(str(input_path), p_cfg)

    download_enabled = True
    if args.no_download:
        download_enabled = False
    elif args.download_enabled:
        download_enabled = True

    if not download_enabled and not args.model_dir:
        raise SystemExit("You used --no-download but did not set --model-dir with existing weights.")

    d_cfg = DetectionConfig(
        lang=args.lang,
        gpu=bool(args.gpu),
        download_enabled=bool(download_enabled),
        model_storage_directory=args.model_dir,
        width_ths=float(args.width_ths),
        ycenter_ths=float(args.ycenter_ths),
        height_ths=float(args.height_ths),
    )

    det = detect_text_boxes(pre["orig_bgr"], d_cfg)
    overlay_words = draw_text_boxes(pre["orig_bgr"], det)

    g_cfg = GroupConfig(
        gap_mult=float(args.gap_mult),
        y_tol_mult=float(args.y_tol_mult),
        line_gap_mult=float(args.line_gap_mult),
    )

    h, w = pre["orig_bgr"].shape[:2]
    blocks = build_text_blocks(det, image_w=w, image_h=h, cfg=g_cfg)
    overlay_blocks = draw_text_blocks(pre["orig_bgr"], blocks)

    mapping = {
        "00_original.png": pre["orig_bgr"],
        "01_text_detect_words.png": overlay_words,
        "02_text_detect_blocks.png": overlay_blocks,
        "03_ocr_bin.png": pre["ocr"],
        "04_geom.png": pre["geom"],
        "05_edges.png": pre["edges"],
    }

    for name, img in mapping.items():
        _imwrite(str(out_dir / name), img)

    (out_dir / "text_boxes.json").write_text(json.dumps(det, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "text_blocks.json").write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "input": str(input_path),
        "preprocess_config": asdict(p_cfg),
        "detect_config": det.get("config", {}),
        "group_config": asdict(g_cfg),
        "image_shape": list(pre["orig_bgr"].shape),
        "num_text_boxes": int(len(det.get("boxes", []))),
        "num_text_blocks": int(len(blocks.get("blocks", []))),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
