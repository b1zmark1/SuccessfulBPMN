from __future__ import annotations

# Изменения (по делу):
# 1) По умолчанию ОТКЛЮЧЕН whitelist (часто ухудшает кириллицу и пробелы). Можно включить флагом.
# 2) Выбор лучшего кандидата: score = conf + бонусы за "качество текста" (алфанум, кириллица, длина),
#    чтобы не проигрывать случаям с conf=0.0, но нормальным текстом.
# 3) Otsu/Adaptive/Invert идут как фоллбек при плохом результате; базово предпочитаем gray (меньше слипаний).
# 4) Добавлен refine-text-bbox: с cv2 убираем крупные нетекстовые компоненты (иконки/рамки), чтобы OCR не путался.
# 5) Добавлен psm 13 (raw line) как доп.кандидат для коротких/узких блоков и низкого качества.
# 6) Постпроцессинг текста: выкидываем мусорные линии из одних тире/скобок, убираем "№00" как служебную метку и т.п.

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import pytesseract
from pytesseract import Output


try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False


@dataclass(frozen=True)
class OcrCfg:
    lang: str = "rus"
    pad_px: int = 10
    inner_crop_px: int = 0
    upscale_factor: float = 1.8
    max_side: int = 4096
    try_rotate_90: bool = True

    psm_block: int = 6
    psm_single: int = 7
    psm_raw_line: int = 13  # raw line

    conf_ok: float = 0.4
    jobs: int = 1

    refine_text_bbox: bool = True
    cc_max_area_frac: float = 0.18  # компоненты больше этой доли кадра считаем "нетекст" (иконки/рамки)
    cc_min_area_px: int = 20

    use_whitelist: bool = False


def _ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "crops"), exist_ok=True)


def _load_blocks(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        blocks = data.get("blocks", [])
        if not isinstance(blocks, list):
            raise ValueError("blocks field is not a list")
        return blocks
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported blocks json format")


def _get_bbox_xyxy(b: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    bbox = b.get("bbox_xyxy")
    if bbox is None:
        bbox = b.get("bbox")
    if bbox is None:
        return None
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    x1, y1, x2, y2 = bbox
    try:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _clip_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def _safe_inner_crop(img: Image.Image, inner: int) -> Image.Image:
    if inner <= 0:
        return img
    w, h = img.size
    if w <= 2 * inner + 24 or h <= 2 * inner + 18:
        return img
    return img.crop((inner, inner, w - inner, h - inner))


def _resize_with_cap(img: Image.Image, scale: float, max_side: int) -> Image.Image:
    if scale <= 0:
        return img
    if abs(scale - 1.0) < 1e-6:
        return img
    w, h = img.size
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    if max_side > 0 and max(nw, nh) > max_side:
        k = max_side / float(max(nw, nh))
        nw = max(1, int(round(nw * k)))
        nh = max(1, int(round(nh * k)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def _otsu_threshold(gray_u8: np.ndarray) -> np.ndarray:
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = gray_u8.size
    if total == 0:
        return gray_u8
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    var_max = -1.0
    thr = 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += float(t) * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > var_max:
            var_max = var_between
            thr = t
    bw = (gray_u8 >= thr).astype(np.uint8) * 255
    return bw


def _adaptive_threshold_cv(gray_u8: np.ndarray) -> Optional[np.ndarray]:
    if not _HAS_CV2:
        return None
    g = gray_u8
    # Adaptive лучше работает на неоднородном фоне и мелком шрифте
    bw = cv2.adaptiveThreshold(
        g,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    return bw


def _invert_l(img_l: Image.Image) -> Image.Image:
    if img_l.mode != "L":
        img_l = ImageOps.grayscale(img_l)
    return ImageOps.invert(img_l)


def _sanitize_whitelist_ru() -> str:
    ru = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    ru += ru.lower()
    digits = "0123456789"
    punct = ".,:;!?-–—/()[]№%+"
    return ru + digits + punct


def _mk_variants_base(crop_rgb: Image.Image) -> List[Tuple[str, Image.Image]]:
    # База: только gray. Бинаризации — позже, как фоллбек (уменьшает слипание слов).
    gray = ImageOps.grayscale(crop_rgb)
    return [("gray", gray)]


def _mk_variants_fallback(crop_rgb: Image.Image) -> List[Tuple[str, Image.Image]]:
    gray = ImageOps.grayscale(crop_rgb)
    g = np.array(gray, dtype=np.uint8)

    bw_otsu = _otsu_threshold(g)
    otsu = Image.fromarray(bw_otsu, mode="L")
    out: List[Tuple[str, Image.Image]] = [("otsu", otsu)]

    bw_ad = _adaptive_threshold_cv(g)
    if bw_ad is not None:
        out.append(("adaptive", Image.fromarray(bw_ad, mode="L")))
    return out


_CYR_RE = re.compile(r"[А-Яа-яЁё]")
_ALNUM_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]")


def _clean_text_ru(text: str) -> str:
    t = text.replace("\f", "").replace("\r", "")
    lines = [ln.strip() for ln in t.split("\n")]
    cleaned: List[str] = []
    for ln in lines:
        if not ln:
            continue
        # убираем строки из одних "черточек"/мусора
        if re.fullmatch(r"[-–—_=.]{2,}", ln):
            continue
        if re.fullmatch(r"[()\[\]{}<>]{1,3}", ln):
            continue
        # служебные метки типа "№00"
        if re.fullmatch(r"№\s*\d{1,4}", ln):
            continue
        cleaned.append(ln)

    if not cleaned:
        return ""

    # нормализация пробелов
    cleaned2 = []
    for ln in cleaned:
        ln = re.sub(r"[ \t]{2,}", " ", ln)
        cleaned2.append(ln)

    return "\n".join(cleaned2).strip()


def _text_quality_score(text: str, conf: float, rotation: str, prefer_rotation: bool) -> float:
    # conf в [0..1], но иногда 0 при нормальном тексте — поэтому добавляем эвристику.
    t = text.strip()
    if not t:
        return -1.0

    alnum = len(_ALNUM_RE.findall(t))
    cyr = len(_CYR_RE.findall(t))
    lines = t.count("\n") + 1
    # штраф за “мусорные” символы
    junk = sum(1 for ch in t if ch in "()[]{}<>|~`")

    score = float(conf)
    score += 0.02 * min(40, alnum)
    score += 0.01 * min(30, cyr)
    score += 0.03 * min(6, lines)
    score -= 0.05 * min(10, junk)

    if prefer_rotation and rotation == "rot90":
        score += 0.08  # небольшое предпочтение rot90 для узких вертикальных боксов

    return score


def _tesseract_data(img_l: Image.Image, lang: str, psm: int, whitelist: str) -> Tuple[str, float, int]:
    cfg_parts = [f"--oem 1 --psm {psm}", "-c preserve_interword_spaces=1"]
    if whitelist:
        cfg_parts.append(f"-c tessedit_char_whitelist={whitelist}")
    cfg = " ".join(cfg_parts)

    data = pytesseract.image_to_data(img_l, lang=lang, config=cfg, output_type=Output.DICT)

    n = len(data.get("text", []))
    if n == 0:
        return "", 0.0, 0

    lines: Dict[Tuple[int, int, int], List[str]] = {}
    confs: List[float] = []

    for i in range(n):
        txt = str(data["text"][i]).strip()
        if not txt:
            continue

        conf_raw = data.get("conf", ["-1"])[i]
        try:
            c = float(conf_raw)
        except Exception:
            c = -1.0
        if c >= 0:
            confs.append(c)

        key = (
            int(data.get("block_num", [1])[i]),
            int(data.get("par_num", [1])[i]),
            int(data.get("line_num", [1])[i]),
        )
        lines.setdefault(key, []).append(txt)

    if not lines:
        return "", 0.0, 0

    ordered_keys = sorted(lines.keys())
    text_lines = [" ".join(lines[k]).strip() for k in ordered_keys]
    text = "\n".join([t for t in text_lines if t]).strip()

    # conf: средний по словам; если слов много, ок. Если нет conf (все -1), даём мягкий fallback.
    conf = 0.0
    if confs:
        conf = float(np.mean(confs)) / 100.0
    else:
        # fallback: чуть-чуть за наличие "нормального" текста
        alnum = len(_ALNUM_RE.findall(text))
        conf = min(0.35, 0.05 * alnum)

    return text, float(conf), len(ordered_keys)


def _refine_crop_to_text_bbox(img_l: Image.Image, cfg: OcrCfg) -> Image.Image:
    if not (_HAS_CV2 and cfg.refine_text_bbox):
        return img_l
    try:
        g = np.array(img_l, dtype=np.uint8)
        # бинаризуем Otsu, потом инвертируем (текст=1)
        bw = _otsu_threshold(g)
        fg = (bw == 0).astype(np.uint8)  # black pixels

        h, w = fg.shape[:2]
        area = float(h * w)
        if area <= 0:
            return img_l

        # connected components
        num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)  # type: ignore
        if num <= 1:
            return img_l

        keep_boxes: List[Tuple[int, int, int, int]] = []
        max_area_px = int(cfg.cc_max_area_frac * area)

        for i in range(1, num):
            x, y, ww, hh, a = stats[i]
            if a < cfg.cc_min_area_px:
                continue
            if a > max_area_px:
                continue
            # выкидываем слишком "линейные" компоненты (часто рамки/линии)
            if ww >= 0.9 * w and hh <= 0.12 * h:
                continue
            if hh >= 0.9 * h and ww <= 0.12 * w:
                continue
            keep_boxes.append((x, y, x + ww, y + hh))

        if not keep_boxes:
            return img_l

        x1 = min(b[0] for b in keep_boxes)
        y1 = min(b[1] for b in keep_boxes)
        x2 = max(b[2] for b in keep_boxes)
        y2 = max(b[3] for b in keep_boxes)

        # небольшой запас, но без фанатизма
        m = 2
        x1 = max(0, x1 - m)
        y1 = max(0, y1 - m)
        x2 = min(w, x2 + m)
        y2 = min(h, y2 + m)

        # если bbox стал слишком маленьким — не трогаем
        if (x2 - x1) < 18 or (y2 - y1) < 12:
            return img_l

        return img_l.crop((x1, y1, x2, y2))
    except Exception:
        return img_l


def _draw_overlay(img_rgb: Image.Image, blocks: List[Dict[str, Any]], out_path: str) -> None:
    over = img_rgb.copy()
    dr = ImageDraw.Draw(over)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for b in blocks:
        bbox = b.get("bbox_xyxy") or b.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        conf = float(b.get("confidence", 0.0))
        bid = b.get("block_id", "")
        label = f"{bid}:{conf:.2f}"
        dr.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
        tx = x1
        ty = max(0, y1 - 12)
        dr.text((tx, ty), label, fill=(255, 0, 0), font=font)

    over.save(out_path)


def _write_report(blocks: List[Dict[str, Any]], out_csv: str) -> None:
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["block_id", "confidence", "text_len", "text"])
        for b in blocks:
            bid = b.get("block_id", "")
            conf = float(b.get("confidence", 0.0))
            txt = (b.get("text") or "").replace("\r", "")
            wr.writerow([bid, f"{conf:.4f}", len(txt), txt])


def _process_one_block(
    img_rgb: Image.Image,
    block: Dict[str, Any],
    cfg: OcrCfg,
    whitelist: str,
    crops_dir: str,
) -> Dict[str, Any]:
    out_block = dict(block)

    bbox = _get_bbox_xyxy(block)
    if bbox is None:
        out_block["bbox_xyxy"] = None
        out_block["text"] = ""
        out_block["confidence"] = 0.0
        out_block["confidence_available"] = True
        out_block["chosen"] = []
        out_block["raw"] = []
        return out_block

    w, h = img_rgb.size
    x1, y1, x2, y2 = bbox

    # динамический pad: мелкие боксы — больше pad, большие — меньше шанс захватить иконку
    bw = x2 - x1
    bh = y2 - y1
    pad = cfg.pad_px
    if min(bw, bh) < 22:
        pad += 4
    if min(bw, bh) < 16:
        pad += 6

    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    x1, y1, x2, y2 = _clip_bbox(x1, y1, x2, y2, w, h)

    crop = img_rgb.crop((x1, y1, x2, y2))
    crop = _safe_inner_crop(crop, cfg.inner_crop_px)

    # увеличение: мелкие боксы усиливаем
    cw, ch = crop.size
    eff_scale = cfg.upscale_factor
    if max(cw, ch) < 140:
        eff_scale = min(3.0, eff_scale * 1.25)
    crop = _resize_with_cap(crop, eff_scale, cfg.max_side)

    block_id = block.get("block_id", "na")
    crop_path = os.path.join(crops_dir, f"blk_{block_id}.png")
    crop.save(crop_path)

    # Нужно ли предпочесть rot90 (вертикальные подписи)
    prefer_rot90 = False
    if cfg.try_rotate_90:
        rw, rh = crop.size
        if rh > rw * 1.7:
            prefer_rot90 = True

    best_text = ""
    best_conf = 0.0
    best_score = -1e9
    best_meta: Dict[str, Any] = {}

    def try_candidate(img_l: Image.Image, variant_name: str, rotation: str, psm: int) -> None:
        nonlocal best_text, best_conf, best_score, best_meta
        if img_l.mode != "L":
            img_l2 = ImageOps.grayscale(img_l)
        else:
            img_l2 = img_l

        # refine crop (убрать иконки/рамки)
        img_l2 = _refine_crop_to_text_bbox(img_l2, cfg)

        txt, conf, _ = _tesseract_data(img_l2, cfg.lang, psm, whitelist)
        txt = _clean_text_ru(txt)

        score = _text_quality_score(txt, conf, rotation, prefer_rot90)
        if score > best_score:
            best_score = score
            best_text = txt
            best_conf = float(conf)
            best_meta = {"engine": "tesseract", "variant": variant_name, "rotation": rotation, "psm": int(psm), "conf": float(conf), "score": float(score)}

    # 1) База: gray + (psm 7,6) + psm13 для коротких/узких
    base_variants = _mk_variants_base(crop)
    for vname, vimg in base_variants:
        try_candidate(vimg, vname, "rot0", cfg.psm_single)
        try_candidate(vimg, vname, "rot0", cfg.psm_block)
        if prefer_rot90:
            r = vimg.rotate(90, expand=True)
            try_candidate(r, vname, "rot90", cfg.psm_single)
            try_candidate(r, vname, "rot90", cfg.psm_block)

        # raw-line иногда помогает коротким словам/титрам
        if max(cw, ch) < 220:
            try_candidate(vimg, vname, "rot0", cfg.psm_raw_line)
            if prefer_rot90:
                r = vimg.rotate(90, expand=True)
                try_candidate(r, vname, "rot90", cfg.psm_raw_line)

    # 2) Фоллбек: если результат слабый — добавляем бинаризации и инверсию
    if best_score < (cfg.conf_ok + 0.4):  # порог по score, не по conf
        fb = _mk_variants_fallback(crop)
        for vname, vimg in fb:
            try_candidate(vimg, vname, "rot0", cfg.psm_block)
            try_candidate(vimg, vname, "rot0", cfg.psm_single)
            if prefer_rot90:
                r = vimg.rotate(90, expand=True)
                try_candidate(r, vname, "rot90", cfg.psm_block)

            inv = _invert_l(vimg)
            try_candidate(inv, f"invert_{vname}", "rot0", cfg.psm_block)
            if prefer_rot90:
                r2 = inv.rotate(90, expand=True)
                try_candidate(r2, f"invert_{vname}", "rot90", cfg.psm_block)

            # psm13 в фоллбеке тоже
            if max(cw, ch) < 220:
                try_candidate(vimg, vname, "rot0", cfg.psm_raw_line)

    out_block["bbox_xyxy"] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
    out_block["text"] = best_text.strip()
    out_block["confidence"] = float(best_conf)
    out_block["confidence_available"] = True
    out_block["chosen"] = [best_meta] if best_meta else []
    out_block["raw"] = []
    return out_block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--lang", default="rus")
    ap.add_argument("--pad-px", type=int, default=10)
    ap.add_argument("--inner-crop-px", type=int, default=0)
    ap.add_argument("--upscale-factor", type=float, default=1.8)
    ap.add_argument("--max-side", type=int, default=4096)
    ap.add_argument("--try-rotate-90", action="store_true")

    ap.add_argument("--psm-block", type=int, default=6)
    ap.add_argument("--psm-single", type=int, default=7)
    ap.add_argument("--psm-raw-line", type=int, default=13)

    ap.add_argument("--conf-ok", type=float, default=0.4)
    ap.add_argument("--jobs", type=int, default=1)

    ap.add_argument("--refine-text-bbox", action="store_true")
    ap.add_argument("--cc-max-area-frac", type=float, default=0.18)
    ap.add_argument("--cc-min-area-px", type=int, default=20)

    ap.add_argument("--use-whitelist", action="store_true")

    args = ap.parse_args()

    cfg = OcrCfg(
        lang=str(args.lang),
        pad_px=int(args.pad_px),
        inner_crop_px=int(args.inner_crop_px),
        upscale_factor=float(args.upscale_factor),
        max_side=int(args.max_side),
        try_rotate_90=bool(args.try_rotate_90),
        psm_block=int(args.psm_block),
        psm_single=int(args.psm_single),
        psm_raw_line=int(args.psm_raw_line),
        conf_ok=float(args.conf_ok),
        jobs=int(args.jobs),
        refine_text_bbox=bool(args.refine_text_bbox),
        cc_max_area_frac=float(args.cc_max_area_frac),
        cc_min_area_px=int(args.cc_min_area_px),
        use_whitelist=bool(args.use_whitelist),
    )

    _ensure_outdir(args.outdir)
    crops_dir = os.path.join(args.outdir, "crops")

    img_rgb = Image.open(args.input).convert("RGB")
    blocks = _load_blocks(args.blocks)

    whitelist = _sanitize_whitelist_ru() if cfg.use_whitelist else ""

    out_blocks: List[Dict[str, Any]] = []
    if cfg.jobs and cfg.jobs > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=cfg.jobs) as ex:
            futs = [ex.submit(_process_one_block, img_rgb, b, cfg, whitelist, crops_dir) for b in blocks]
            for fut in as_completed(futs):
                out_blocks.append(fut.result())

        def _bid(x: Dict[str, Any]) -> int:
            try:
                return int(x.get("block_id", 0))
            except Exception:
                return 0

        out_blocks.sort(key=_bid)
    else:
        for b in blocks:
            out_blocks.append(_process_one_block(img_rgb, b, cfg, whitelist, crops_dir))

    out = {
        "coord_space": "auto",
        "engine": "tesseract_fast",
        "ocr_config": {
            "lang": cfg.lang,
            "pad_px": cfg.pad_px,
            "inner_crop_px": cfg.inner_crop_px,
            "upscale_factor": cfg.upscale_factor,
            "try_rotate_90": cfg.try_rotate_90,
            "psm_block": cfg.psm_block,
            "psm_single": cfg.psm_single,
            "psm_raw_line": cfg.psm_raw_line,
            "conf_ok": cfg.conf_ok,
            "jobs": cfg.jobs,
            "max_side": cfg.max_side,
            "refine_text_bbox": cfg.refine_text_bbox and _HAS_CV2,
            "cc_max_area_frac": cfg.cc_max_area_frac,
            "cc_min_area_px": cfg.cc_min_area_px,
            "use_whitelist": cfg.use_whitelist,
        },
        "blocks": out_blocks,
        "image": {"w": img_rgb.size[0], "h": img_rgb.size[1], "path": args.input},
    }

    out_json = os.path.join(args.outdir, "ocr.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    overlay_path = os.path.join(args.outdir, "overlay.png")
    _draw_overlay(img_rgb, out_blocks, overlay_path)

    report_csv = os.path.join(args.outdir, "report.csv")
    _write_report(out_blocks, report_csv)

    total = len(out_blocks)
    non_empty = sum(1 for b in out_blocks if (b.get("text") or "").strip())
    confs = [float(b.get("confidence", 0.0)) for b in out_blocks if (b.get("text") or "").strip()]
    avg_conf = float(np.mean(confs)) if confs else 0.0
    print(f"[ocr_fast] blocks_total={total} non_empty={non_empty} empty={total-non_empty} avg_conf_non_empty={avg_conf:.3f}")
    print(f"[ocr_fast] cv2_refine_available={_HAS_CV2}")
    print(f"[ocr_fast] saved: {out_json}")
    print(f"[ocr_fast] saved: {overlay_path}")
    print(f"[ocr_fast] saved: {report_csv}")


if __name__ == "__main__":
    main()
