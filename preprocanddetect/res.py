from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from preprocess import PreprocessConfig, preprocess


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _imwrite(path: str, img: np.ndarray) -> None:
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)

    # поддержка двух имён параметра
    ap.add_argument("--outdir", dest="outdir", default=None)
    ap.add_argument("--out-dir", dest="outdir", default=None)

    ap.add_argument("--max-side", type=int, default=1800)
    ap.add_argument("--denoise-h", type=int, default=10)
    ap.add_argument("--adaptive-block", type=int, default=35)
    ap.add_argument("--adaptive-c", type=int, default=11)

    # 0 => авто-подбор (по умолчанию)
    ap.add_argument("--geom-close-kernel", type=int, default=0)

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

    cfg = PreprocessConfig(
        max_side=args.max_side,
        denoise_h=args.denoise_h,
        adaptive_block=args.adaptive_block,
        adaptive_c=args.adaptive_c,
        geom_close_kernel=args.geom_close_kernel,
    )

    result = preprocess(str(input_path), cfg)

    mapping = {
        "00_original.png": result["orig_bgr"],
        "01_gray.png": result["gray"],
        "02_gray_eq.png": result["gray_eq"],
        "03_denoised.png": result["denoised"],
        "04_ocr.png": result["ocr"],
        "05_cv_binary.png": result["cv_binary"],
        "06_geom.png": result["geom"],
        "07_edges.png": result["edges"],
    }

    for name, img in mapping.items():
        _imwrite(str(out_dir / name), img)

    meta = {
        "input": str(input_path),
        "config": asdict(cfg),
        "shapes": {k: list(v.shape) for k, v in result.items()},
        "dtypes": {k: str(v.dtype) for k, v in result.items()},
        "black_ratio_ocr": float(np.mean(result["ocr"] == 0)),
        "black_ratio_cv": float(np.mean(result["cv_binary"] == 0)),
        "black_ratio_geom": float(np.mean(result["geom"] == 0)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
