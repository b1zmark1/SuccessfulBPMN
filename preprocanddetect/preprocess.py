from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np


class PreprocessError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreprocessConfig:
    # resize: по умолчанию только downscale
    max_side: int = 1800
    upscale_small: bool = False
    min_side_for_upscale: int = 1000
    keep_aspect: bool = True

    # CLAHE: включаем только если контраст низкий (p95-p05 < threshold)
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    clahe_min_contrast_p95_p05: int = 60

    # Denoise (NLM)
    use_nlm_denoise: bool = True
    denoise_h: int = 10
    denoise_template_window_size: int = 7
    denoise_search_window_size: int = 21

    # OCR bin
    adaptive_block: int = 35
    adaptive_c: int = 11

    # CV bin
    use_otsu_for_cv: bool = True

    # GEOM close:
    # 0 => авто-подбор (рекомендую)
    # 1 => выключено (без эффекта)
    # 3/5 => сильнее закрывает разрывы, но может склеивать мелочи
    geom_close_kernel: int = 0
    geom_close_iterations: int = 1

    # приводим к "черное на белом"
    force_black_on_white: bool = True

    # edges
    canny_sigma_low: float = 0.66
    canny_sigma_high: float = 1.33

    # geometry
    enable_deskew: bool = True
    deskew_min_deg: float = 0.5
    deskew_max_deg: float = 7.0
    deskew_hough_thresh: int = 120
    deskew_min_line_len: int = 60
    deskew_max_line_gap: int = 10


def preprocess(
    image: Union[str, bytes, np.ndarray],
    cfg: Optional[PreprocessConfig] = None,
) -> Dict[str, np.ndarray]:
    if cfg is None:
        cfg = PreprocessConfig()

    bgr_orig = _load_as_bgr(image)
    bgr_geom, geom_angle = _deskew_if_needed(bgr_orig, cfg)
    bgr, resize_ratio = _resize(bgr_geom, cfg)

    gray = _to_gray_uint8(bgr)
    gray_eq = _apply_clahe_if_needed(gray, cfg)
    denoised = _denoise(gray_eq, cfg)

    ocr_bin = _binarize_ocr(denoised, cfg)
    cv_bin = _binarize_cv(denoised, cfg)

    if cfg.force_black_on_white:
        ocr_bin = _ensure_black_on_white(ocr_bin)
        cv_bin = _ensure_black_on_white(cv_bin)

    geom = _make_geom(cv_bin, cfg)
    edges = _edges_from_gray(denoised, cfg)

    return {
        "orig_bgr": bgr_orig,
        "geom_bgr": bgr_geom,
        "model_bgr": bgr,
        "resize_ratio": np.array(resize_ratio, dtype=np.float32),
        "geom_angle_deg": np.array(geom_angle, dtype=np.float32),
        "gray": gray,
        "gray_eq": gray_eq,
        "denoised": denoised,
        "ocr": ocr_bin,
        "cv_binary": cv_bin,
        "geom": geom,
        "edges": edges,
    }


def _load_as_bgr(image: Union[str, bytes, np.ndarray]) -> np.ndarray:
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise PreprocessError(f"Failed to read image from path: {image}")
        return _drop_alpha_to_white(img)

    if isinstance(image, bytes):
        arr = np.frombuffer(image, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise PreprocessError("Failed to decode image from bytes")
        return _drop_alpha_to_white(img)

    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise PreprocessError("Empty numpy image array")
        return _drop_alpha_to_white(image)

    raise PreprocessError(f"Unsupported input type: {type(image)!r}")


def _drop_alpha_to_white(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim != 3:
        raise PreprocessError(f"Unexpected image shape: {img.shape}")

    ch = img.shape[2]
    if ch == 3:
        return img.astype(np.uint8, copy=False)

    if ch == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = (img[:, :, 3].astype(np.float32) / 255.0)[..., None]
        white = np.full_like(bgr, 255.0, dtype=np.float32)
        comp = bgr * alpha + white * (1.0 - alpha)
        return np.clip(comp, 0, 255).astype(np.uint8)

    raise PreprocessError(f"Unexpected channel count: {ch}")


def _resize(bgr: np.ndarray, cfg: PreprocessConfig) -> Tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    long_side = max(h, w)

    if long_side > cfg.max_side:
        scale = cfg.max_side / float(long_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA), float(scale)

    if cfg.upscale_small and long_side < cfg.min_side_for_upscale:
        scale = cfg.min_side_for_upscale / float(long_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC), float(scale)

    return bgr, 1.0


def _to_gray_uint8(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise PreprocessError(f"Expected BGR image, got shape: {bgr.shape}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _apply_clahe_if_needed(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if not cfg.use_clahe:
        return gray

    p05 = int(np.percentile(gray, 5))
    p95 = int(np.percentile(gray, 95))
    if (p95 - p05) >= int(cfg.clahe_min_contrast_p95_p05):
        return gray

    clahe = cv2.createCLAHE(
        clipLimit=float(cfg.clahe_clip_limit),
        tileGridSize=(int(cfg.clahe_tile_grid_size), int(cfg.clahe_tile_grid_size)),
    )
    return clahe.apply(gray)


def _denoise(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if not cfg.use_nlm_denoise:
        return gray

    return cv2.fastNlMeansDenoising(
        gray,
        None,
        h=int(cfg.denoise_h),
        templateWindowSize=int(cfg.denoise_template_window_size),
        searchWindowSize=int(cfg.denoise_search_window_size),
    )


def _binarize_ocr(denoised: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    block = int(cfg.adaptive_block)
    if block < 3:
        block = 3
    if block % 2 == 0:
        block += 1

    return cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        int(cfg.adaptive_c),
    )


def _binarize_cv(denoised: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if cfg.use_otsu_for_cv:
        _, cv_bin = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv_bin
    return _binarize_ocr(denoised, cfg)


def _ensure_black_on_white(binary: np.ndarray) -> np.ndarray:
    if binary.dtype != np.uint8:
        binary = np.clip(binary, 0, 255).astype(np.uint8)
    if float(np.mean(binary)) < 127.0:
        return cv2.bitwise_not(binary)
    return binary


def _make_geom(cv_binary_black_on_white: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    k = int(cfg.geom_close_kernel)
    if k == 0:
        k = _auto_select_close_kernel(cv_binary_black_on_white, cfg)

    inv = cv2.bitwise_not(cv_binary_black_on_white)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=int(cfg.geom_close_iterations))
    geom = cv2.bitwise_not(closed)

    if cfg.force_black_on_white:
        geom = _ensure_black_on_white(geom)
    return geom


def _auto_select_close_kernel(cv_binary_black_on_white: np.ndarray, cfg: PreprocessConfig) -> int:
    """
    Выбираем минимальный kernel из [1,3,5], который:
      - заметно уменьшает число компонент связности (разрывы линий),
      - но не сильно увеличивает долю черного (утолщение/склейка).
    """
    candidates = [1, 3, 5]

    base_cc = _count_black_components(cv_binary_black_on_white)
    base_black = float(np.mean(cv_binary_black_on_white == 0))

    # Если всё уже цельно — close не нужен
    if base_cc <= 200:
        return 1

    best_k = 1
    for k in candidates[1:]:
        inv = cv2.bitwise_not(cv_binary_black_on_white)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=int(cfg.geom_close_iterations))
        geom = cv2.bitwise_not(closed)

        cc = _count_black_components(geom)
        black = float(np.mean(geom == 0))

        # хотим уменьшить компонентность хотя бы на 3%
        cc_improved = (cc <= base_cc * 0.97)
        # и не нарастить черного больше чем на 2% абсолютных
        black_ok = (black <= base_black + 0.02)

        if cc_improved and black_ok:
            best_k = k
            break

    return best_k


def _count_black_components(binary_black_on_white: np.ndarray) -> int:
    fg = (binary_black_on_white == 0).astype(np.uint8)
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    return int(max(0, num_labels - 1))


def _edges_from_gray(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    med = float(np.median(gray))
    lower = int(max(0, cfg.canny_sigma_low * med))
    upper = int(min(255, cfg.canny_sigma_high * med))
    return cv2.Canny(gray, lower, upper)


def _deskew_if_needed(bgr: np.ndarray, cfg: PreprocessConfig) -> Tuple[np.ndarray, float]:
    if not cfg.enable_deskew:
        return bgr, 0.0

    gray = _to_gray_uint8(bgr)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(cfg.deskew_hough_thresh),
        minLineLength=int(cfg.deskew_min_line_len),
        maxLineGap=int(cfg.deskew_max_line_gap),
    )
    if lines is None or len(lines) == 0:
        return bgr, 0.0

    angles = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180
        angles.append(angle)

    if not angles:
        return bgr, 0.0

    median_angle = float(np.median(angles))
    if abs(median_angle) < cfg.deskew_min_deg:
        return bgr, 0.0
    if abs(median_angle) > cfg.deskew_max_deg:
        return bgr, 0.0

    h, w = bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    mat[0, 2] += (new_w / 2.0) - center[0]
    mat[1, 2] += (new_h / 2.0) - center[1]
    rotated = cv2.warpAffine(
        bgr,
        mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )
    return rotated, median_angle
