import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    max_side: int = 1800
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    denoise_h: int = 10
    adaptive_block: int = 35
    adaptive_c: int = 11
    close_kernel: int = 1


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _imwrite(path: str, img: np.ndarray) -> None:
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return img
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess(image_path: str, out_dir: str, cfg: PreprocessConfig) -> Tuple[str, str, str]:
    _ensure_dir(out_dir)

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    bgr = _resize_max_side(bgr, cfg.max_side)
    _imwrite(os.path.join(out_dir, "00_original.png"), bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _imwrite(os.path.join(out_dir, "01_gray.png"), gray)

    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(cfg.clahe_tile_grid_size, cfg.clahe_tile_grid_size),
    )
    gray_eq = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(gray_eq, h=cfg.denoise_h)
    _imwrite(os.path.join(out_dir, "02_denoised.png"), denoised)


    block = cfg.adaptive_block if cfg.adaptive_block % 2 == 1 else cfg.adaptive_block + 1
    ocr_bin = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        cfg.adaptive_c,
    )
    _imwrite(os.path.join(out_dir, "04_ocr_ready.png"), ocr_bin)

    _, cv_bin = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _imwrite(os.path.join(out_dir, "05_cv_binary.png"), cv_bin)

    inv = 255 - cv_bin
    k = cfg.close_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    geom = 255 - closed
    _imwrite(os.path.join(out_dir, "06_geom.png"), geom)

    return (
        os.path.join(out_dir, "04_ocr_ready.png"),
        os.path.join(out_dir, "05_cv_binary.png"),
        os.path.join(out_dir, "06_geom.png"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cfg = PreprocessConfig()
    ocr_path, cv_path, geom_path = preprocess(args.input, args.outdir, cfg)
    print(f"OCR-ready: {ocr_path}")
    print(f"CV-binary: {cv_path}")
    print(f"GEOM:      {geom_path}")


if __name__ == "__main__":
    main()
