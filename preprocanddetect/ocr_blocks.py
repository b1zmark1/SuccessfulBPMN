from __future__ import annotations

# Изменения (по делу):
# 1) Ротации теперь не boolean, а строковые: rot0 / rot90_cw / rot90_ccw / rot180.
# 2) rot90 пробуем в обе стороны (CW и CCW), иначе вертикальные lane/role подписи часто читаются в мусор.
# 3) rot90 включаем только для "вертикальных" кропов (h >> w), чтобы не тратить время на обычный текст.
# 4) Исправлен debug-save: раньше проверялся best.rotation == "rot90", теперь используется строковая rotation.

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import re

import cv2
import numpy as np


_CYR_RE = re.compile(r"[\u0400-\u04FF]")
_WS_RE = re.compile(r"\s+")
_EDGE_GARBAGE_RE = re.compile(r"^[`\\|]+|[`\\|]+$")


@dataclass(frozen=True)
class OcrConfig:
    lang: str = "ru"

    # EasyOCR
    gpu: bool = False
    download_enabled: bool = True
    model_storage_directory: Optional[str] = None
    paragraph: bool = False
    decoder: str = "greedy"  # "greedy" | "beamsearch"
    beam_width: int = 5
    contrast_ths: float = 0.1
    adjust_contrast: float = 0.5

    # Crop / scaling
    pad_px: int = 6
    inner_crop_px: int = 1  # внутрь рамки (уменьшаем влияние границ)
    upscale_factor: float = 4.0
    min_crop_height: int = 28

    # Thresholding
    use_adaptive_bin: bool = True
    adaptive_block: int = 41
    adaptive_c: int = 11

    # Line removal
    remove_lines: bool = True
    line_h_kernel: int = 35
    line_v_kernel: int = 35
    line_iter: int = 1

    # Charset / filtering
    allowlist: str = (
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        "0123456789 .,:;!?-–—/()[]\"'№%+"
    )
    min_confidence: float = 0.1
    try_rotate_90: bool = True

    # Tesseract
    tesseract_enabled: bool = True
    tesseract_lang: str = "rus"
    tesseract_psm_single_line: int = 7
    tesseract_psm_block: int = 6

    # Debug
    save_raw: bool = False


@dataclass
class OcrCandidate:
    engine: str  # "easyocr" | "tesseract"
    variant: str  # "gray" | "clahe" | "otsu" | "adaptive" | etc
    rotation: str  # "rot0" | "rot90_cw" | "rot90_ccw" | "rot180"
    text: str
    conf: float  # 0..1 (если доступно)
    score: float
    psm: Optional[int] = None


def _clamp_xyxy(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def _as_int_xyxy(v: Any) -> Tuple[int, int, int, int]:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        raise ValueError(f"Invalid bbox: {v}")
    x1, y1, x2, y2 = v
    return int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))


def _pick_bbox_for_image(block: Dict[str, Any], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Автоматически выбираем bbox_* так, чтобы он помещался в текущее изображение.
    Это критично, если blocks.json хранит bbox в model-space, а на вход дали geom image (или наоборот).
    """
    cand_keys = []
    if "bbox_model_xyxy" in block:
        cand_keys.append("bbox_model_xyxy")
    if "bbox_geom_xyxy" in block:
        cand_keys.append("bbox_geom_xyxy")

    if not cand_keys:
        raise KeyError("Block has no bbox_model_xyxy / bbox_geom_xyxy")

    def fits(xyxy: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = xyxy
        return 0 <= x1 < x2 <= img_w and 0 <= y1 < y2 <= img_h

    best = None
    for k in cand_keys:
        xyxy = _as_int_xyxy(block[k])
        if fits(xyxy):
            best = xyxy
            break

    if best is None:
        # если ничего не влезает — берём первую и потом clamp
        best = _as_int_xyxy(block[cand_keys[0]])

    x1, y1, x2, y2 = best
    return _clamp_xyxy(x1, y1, x2, y2, img_w, img_h)


def _rotate_gray(gray: np.ndarray, rotation: str) -> np.ndarray:
    if rotation == "rot0":
        return gray
    if rotation == "rot90_cw":
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "rot90_ccw":
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "rot180":
        return cv2.rotate(gray, cv2.ROTATE_180)
    raise ValueError(f"Unknown rotation: {rotation}")


def _is_vertical_candidate(gray: np.ndarray, ratio: float = 1.35) -> bool:
    h, w = gray.shape[:2]
    return h > int(w * ratio)


def _clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\x0c", " ")
    s = s.strip()
    s = _EDGE_GARBAGE_RE.sub("", s).strip()
    s = s.replace("`", "").replace("\\", "").strip()
    s = _WS_RE.sub(" ", s)
    return s.strip()


def _text_quality(text: str, allowlist: str) -> float:
    """
    Эвристика качества: доля кириллицы, доля допустимых символов, штраф за мусор.
    0..1
    """
    t = text.strip()
    if not t:
        return 0.0

    allowed = set(allowlist)
    total = len(t)
    ok = sum(1 for c in t if c in allowed or c.isspace())
    ok_ratio = ok / max(1, total)

    letters = [c for c in t if c.isalpha()]
    if not letters:
        cyr_ratio = 1.0
    else:
        cyr_ratio = sum(1 for c in letters if _CYR_RE.match(c)) / max(1, len(letters))

    # штраф, если слишком много пунктуации/мусора
    punct = sum(1 for c in t if not c.isalnum() and not c.isspace())
    punct_ratio = punct / max(1, total)

    score = 0.55 * ok_ratio + 0.35 * cyr_ratio + 0.10 * (1.0 - punct_ratio)
    return max(0.0, min(1.0, score))


def _candidate_score(text: str, conf01: float, allowlist: str) -> float:
    """
    Итоговый скоринг кандидата: confidence + качество текста + длина.
    """
    q = _text_quality(text, allowlist)
    length = len(text.strip())
    len_bonus = min(1.0, length / 30.0)  # до ~30 символов растёт, дальше почти нет
    return (conf01 * 2.8) + (q * 1.6) + (len_bonus * 0.6)


def _make_variants(gray: np.ndarray, cfg: OcrConfig) -> List[Tuple[str, np.ndarray]]:
    variants: List[Tuple[str, np.ndarray]] = []

    variants.append(("gray", gray))

    # CLAHE часто помогает на бледном тексте
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    variants.append(("clahe", gray_clahe))

    # Otsu bin
    _, otsu = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))

    # Adaptive bin
    if cfg.use_adaptive_bin:
        blk = int(cfg.adaptive_block)
        if blk % 2 == 0:
            blk += 1
        blk = max(3, blk)
        adaptive = cv2.adaptiveThreshold(
            gray_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, int(cfg.adaptive_c)
        )
        variants.append(("adaptive", adaptive))

    # Remove lines (работает лучше на бинарном)
    if cfg.remove_lines:
        bin_src = adaptive if cfg.use_adaptive_bin else otsu
        rm = _remove_lines(bin_src, cfg)
        variants.append(("rm_lines", rm))

        # иногда лучше rm_lines + небольшое размывание
        rm_blur = cv2.GaussianBlur(rm, (3, 3), 0)
        variants.append(("rm_lines_blur", rm_blur))

    return variants


def _remove_lines(bin_img: np.ndarray, cfg: OcrConfig) -> np.ndarray:
    """
    Удаляем горизонтальные/вертикальные линии морфологией.
    Важно: это эвристика, может вредить если символы похожи на линии.
    """
    src = bin_img.copy()
    if len(src.shape) != 2:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Инверсия: линии/текст белые на чёрном часто лучше выделяются
    inv = 255 - src

    hk = max(3, int(cfg.line_h_kernel))
    vk = max(3, int(cfg.line_v_kernel))
    iter_n = max(1, int(cfg.line_iter))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel, iterations=iter_n)
    v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel, iterations=iter_n)

    lines = cv2.bitwise_or(h_lines, v_lines)

    # Убираем линии из inv
    cleaned = cv2.subtract(inv, lines)
    out = 255 - cleaned
    return out


def _ensure_upscale(gray: np.ndarray, upscale_factor: float) -> np.ndarray:
    if upscale_factor <= 1.0:
        return gray
    h, w = gray.shape[:2]
    nw = int(round(w * upscale_factor))
    nh = int(round(h * upscale_factor))
    nw = max(1, nw)
    nh = max(1, nh)
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC)


def _init_easyocr_reader(cfg: OcrConfig):
    import easyocr  # lazy import

    return easyocr.Reader(
        [cfg.lang],
        gpu=cfg.gpu,
        download_enabled=cfg.download_enabled,
        model_storage_directory=cfg.model_storage_directory,
    )


def _easyocr_read(reader, img_gray_or_bin: np.ndarray, cfg: OcrConfig) -> Tuple[str, float]:
    """
    Возвращает (text, conf01). conf — среднее по строкам, если доступно.
    """
    # easyocr ожидает uint8
    img = img_gray_or_bin
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    res = reader.readtext(
        img,
        detail=1,
        paragraph=cfg.paragraph,
        decoder=cfg.decoder,
        beamWidth=int(cfg.beam_width),
        contrast_ths=float(cfg.contrast_ths),
        adjust_contrast=float(cfg.adjust_contrast),
        allowlist=cfg.allowlist,
    )

    if not res:
        return "", 0.0

    # res: List[ [bbox, text, conf], ... ]
    texts = []
    confs = []
    for item in res:
        if len(item) >= 3:
            texts.append(str(item[1]))
            try:
                confs.append(float(item[2]))
            except Exception:
                pass

    text = "\n".join(t.strip() for t in texts if t and t.strip()).strip()
    if not confs:
        return text, 0.0
    conf01 = float(np.mean(confs))
    return text, max(0.0, min(1.0, conf01))


def _tesseract_read(img_gray_or_bin: np.ndarray, cfg: OcrConfig, psm: int) -> Tuple[str, float]:
    """
    Возвращает (text, conf01).
    """
    try:
        import pytesseract  # lazy import
    except Exception:
        return "", 0.0

    # pytesseract удобнее через PIL, но принимает и ndarray
    img = img_gray_or_bin
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    allow = cfg.allowlist.replace('"', '\\"')
    tess_cfg = (
        f'--oem 1 --psm {int(psm)} '
        f'-c tessedit_char_whitelist="{allow}" '
        f'-c preserve_interword_spaces=1 '
        f'--dpi 300'
    )

    # text
    text = pytesseract.image_to_string(img, lang=cfg.tesseract_lang, config=tess_cfg)

    # confidence: mean over words
    data = pytesseract.image_to_data(img, lang=cfg.tesseract_lang, config=tess_cfg, output_type=pytesseract.Output.DICT)
    confs: List[float] = []
    for c in data.get("conf", []):
        try:
            v = float(c)
        except Exception:
            continue
        if v >= 0:
            confs.append(v)

    if confs:
        conf01 = float(np.mean(confs) / 100.0)
    else:
        conf01 = 0.0

    return text.strip(), max(0.0, min(1.0, conf01))


def recognize_text_blocks(
    image_bgr: np.ndarray,
    blocks: List[Dict[str, Any]],
    cfg: OcrConfig,
    debug_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    blocks: список блоков (каждый содержит block_id и bbox_model_xyxy и/или bbox_geom_xyxy).
    """
    img_h, img_w = image_bgr.shape[:2]

    # easyocr init once
    reader = _init_easyocr_reader(cfg)

    crops_dir = None
    best_dir = None
    if debug_dir is not None:
        crops_dir = debug_dir / "crops"
        best_dir = debug_dir / "best"
        crops_dir.mkdir(parents=True, exist_ok=True)
        best_dir.mkdir(parents=True, exist_ok=True)

    out_blocks: List[Dict[str, Any]] = []

    for b in blocks:
        block_id = int(b.get("block_id", b.get("id", 0)) or 0)
        x1, y1, x2, y2 = _pick_bbox_for_image(b, img_w, img_h)

        # pad outward
        x1p = x1 - int(cfg.pad_px)
        y1p = y1 - int(cfg.pad_px)
        x2p = x2 + int(cfg.pad_px)
        y2p = y2 + int(cfg.pad_px)
        x1p, y1p, x2p, y2p = _clamp_xyxy(x1p, y1p, x2p, y2p, img_w, img_h)

        crop = image_bgr[y1p:y2p, x1p:x2p].copy()
        if crop.size == 0:
            out_blocks.append(
                {
                    "block_id": block_id,
                    "text": "",
                    "confidence": 0.0,
                    "confidence_available": False,
                    "raw": [] if cfg.save_raw else [],
                }
            )
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # inner crop (внутрь рамки) — снижает влияние границ и линий
        ic = max(0, int(cfg.inner_crop_px))
        if ic > 0 and gray.shape[0] > (2 * ic + 1) and gray.shape[1] > (2 * ic + 1):
            gray = gray[ic:-ic, ic:-ic]

        if gray.shape[0] < int(cfg.min_crop_height):
            # слишком мелко — всё равно попробуем, но часто будет шум
            pass

        gray = _ensure_upscale(gray, float(cfg.upscale_factor))

        rotations: List[str] = ["rot0"]
        if cfg.try_rotate_90 and _is_vertical_candidate(gray, ratio=1.35):
            rotations.extend(["rot90_cw", "rot90_ccw"])

        best: Optional[OcrCandidate] = None
        chosen: List[Dict[str, Any]] = []

        for rot_name in rotations:
            g = _rotate_gray(gray, rot_name)
            variants = _make_variants(g, cfg)

            for v_name, v_img in variants:
                # EasyOCR
                e_text, e_conf = _easyocr_read(reader, v_img, cfg)
                e_text = _clean_text(e_text)
                e_score = _candidate_score(e_text, e_conf, cfg.allowlist)
                cand_e = OcrCandidate(
                    engine="easyocr",
                    variant=v_name,
                    rotation=rot_name,
                    text=e_text,
                    conf=e_conf,
                    score=e_score,
                    psm=None,
                )
                if best is None or cand_e.score > best.score:
                    best = cand_e

                # Tesseract
                if cfg.tesseract_enabled:
                    # пробуем и single_line, и block
                    for psm in (int(cfg.tesseract_psm_single_line), int(cfg.tesseract_psm_block)):
                        t_text, t_conf = _tesseract_read(v_img, cfg, psm=psm)
                        t_text = _clean_text(t_text)
                        t_score = _candidate_score(t_text, t_conf, cfg.allowlist)
                        cand_t = OcrCandidate(
                            engine="tesseract",
                            variant=v_name,
                            rotation=rot_name,
                            text=t_text,
                            conf=t_conf,
                            score=t_score,
                            psm=psm,
                        )
                        if best is None or cand_t.score > best.score:
                            best = cand_t

        if best is None:
            best = OcrCandidate(engine="easyocr", variant="gray", rotation="rot0", text="", conf=0.0, score=0.0)

        # минимальный порог уверенности
        conf_avail = True
        if best.conf < float(cfg.min_confidence):
            # всё равно возвращаем текст (иногда полезно), но помечаем низкой уверенностью
            pass

        chosen.append(
            {
                "engine": best.engine,
                "variant": best.variant,
                "rotation": best.rotation,
                "conf": float(best.conf),
                "score": float(best.score),
                **({"psm": int(best.psm)} if best.psm is not None else {}),
            }
        )

        # debug saves
        if crops_dir is not None and best_dir is not None:
            crop_name = f"blk_{block_id:03d}_crop.png"
            cv2.imwrite(str(crops_dir / crop_name), crop)

            # сохраняем «лучшее» изображение (после upscale + предобработка)
            g0 = _rotate_gray(gray, best.rotation)
            vmap = dict(_make_variants(g0, cfg))
            best_img = vmap.get(best.variant, g0)
            cv2.imwrite(str(best_dir / f"blk_{block_id:03d}_{best.engine}_{best.variant}_{best.rotation}.png"), best_img)

        out_blocks.append(
            {
                "block_id": block_id,
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "text": best.text,
                "confidence": float(best.conf),
                "confidence_available": bool(conf_avail),
                "chosen": chosen,
                "raw": [] if not cfg.save_raw else [],
            }
        )

    return {
        "coord_space": "auto",
        "engine": "easyocr+tesseract" if cfg.tesseract_enabled else "easyocr",
        "ocr_config": asdict(cfg),
        "blocks": out_blocks,
    }
