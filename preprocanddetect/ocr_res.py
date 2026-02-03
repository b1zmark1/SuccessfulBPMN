from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import cv2
import pytesseract

try:
    import easyocr
except Exception:
    easyocr = None


# ----------------------------
# Utils
# ----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def _maybe_scale_bbox(bbox_xyxy: List[float], img_w: int, img_h: int) -> List[int]:
    """
    Supports:
    - pixel coords (e.g. [110, 8, 202, 20])
    - normalized coords (0..1), detected heuristically
    """
    x1, y1, x2, y2 = bbox_xyxy
    mx = max(abs(x1), abs(x2))
    my = max(abs(y1), abs(y2))
    # Heuristic: if coords are in [0..1.5], treat as normalized
    if mx <= 1.5 and my <= 1.5:
        x1 = int(round(x1 * img_w))
        x2 = int(round(x2 * img_w))
        y1 = int(round(y1 * img_h))
        y2 = int(round(y2 * img_h))
    else:
        x1 = int(round(x1))
        x2 = int(round(x2))
        y1 = int(round(y1))
        y2 = int(round(y2))
    return [x1, y1, x2, y2]


def _bbox_from_any(b: Dict[str, Any]) -> List[float]:
    """
    Accepts several formats:
    - bbox_xyxy / bbox_model_xyxy / bbox_geom_xyxy: [x1,y1,x2,y2]
    - bbox: can be [x1,y1,x2,y2] OR polygon points
      * [x1,y1,x2,y2] -> xyxy
      * [x1,y1,x2,y2,x3,y3,x4,y4] -> polygon -> xyxy
      * [[x,y], [x,y], ...] -> polygon -> xyxy
    """
    for k in ("bbox_xyxy", "bbox_model_xyxy", "bbox_geom_xyxy"):
        if k in b and isinstance(b[k], list) and len(b[k]) == 4:
            return [float(v) for v in b[k]]

    if "bbox" not in b:
        raise KeyError("Block has no bbox/bbox_xyxy/bbox_model_xyxy/bbox_geom_xyxy")

    bb = b["bbox"]
    if isinstance(bb, list) and len(bb) == 4 and all(isinstance(v, (int, float)) for v in bb):
        return [float(v) for v in bb]

    # polygon as flat list
    if isinstance(bb, list) and len(bb) >= 8 and all(isinstance(v, (int, float)) for v in bb):
        xs = [float(bb[i]) for i in range(0, len(bb), 2)]
        ys = [float(bb[i]) for i in range(1, len(bb), 2)]
        return [min(xs), min(ys), max(xs), max(ys)]

    # polygon as list of points
    if isinstance(bb, list) and len(bb) >= 3 and all(isinstance(p, list) and len(p) == 2 for p in bb):
        xs = [float(p[0]) for p in bb]
        ys = [float(p[1]) for p in bb]
        return [min(xs), min(ys), max(xs), max(ys)]

    raise ValueError(f"Unsupported bbox format in block_id={b.get('block_id')}: {type(bb)}")


def _normalize_text(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n+", "\n", s)
    return s.strip()


def _domain_postprocess(s: str) -> str:
    # минимально агрессивно: убрать мусор по краям и привести пробелы
    s = _normalize_text(s)
    s = s.strip("|`\\")
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s).strip()
    # удобнее оценивать — в отчёте всё равно покажем \n; но для BPMN чаще нужно в одну строку:
    # если хочешь оставлять переносы, закомментируй следующую строку
    s = " ".join(s.split())
    return s


def _resize_max_side(img_bgr: np.ndarray, max_side: Optional[int]) -> Tuple[np.ndarray, float]:
    if not max_side or max_side <= 0:
        return img_bgr, 1.0
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img_bgr, 1.0
    scale = max_side / float(m)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _rotate_bgr(img_bgr: np.ndarray, rot: str) -> np.ndarray:
    if rot == "rot0":
        return img_bgr
    if rot == "rot90":
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    if rot == "rot270":
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unknown rotation: {rot}")


# ----------------------------
# Preprocessing
# ----------------------------

def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _otsu(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _adaptive_bin(gray: np.ndarray, block: int, c: int) -> np.ndarray:
    # block должен быть нечётным
    if block % 2 == 0:
        block += 1
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, block, c)
    return bw


def _remove_lines_inpaint(gray: np.ndarray, h_kernel: int, v_kernel: int, iters: int) -> np.ndarray:
    # Создаём бинарку (инверсия чтобы линии стали белыми на чёрном)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    h_kernel = max(3, int(h_kernel))
    v_kernel = max(3, int(v_kernel))
    iters = max(1, int(iters))

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel))

    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk, iterations=iters)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk, iterations=iters)

    mask = cv2.bitwise_or(h_lines, v_lines)
    # inpaint работает по 8-bit single channel
    out = cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)
    return out


# ----------------------------
# OCR Engines
# ----------------------------

@dataclass
class OcrPick:
    engine: str
    variant: str
    rotation: str
    conf: float
    score: float
    text: str
    extra: Dict[str, Any]


def _score_text(text: str, conf: float) -> float:
    t = _normalize_text(text)
    # Небольшой бонус за "похоже на текст", но основное — confidence.
    len_bonus = min(len(t), 60) / 60.0
    return conf * 5.0 + len_bonus


def _tesseract_recognize(gray_or_bw: np.ndarray, lang: str, psm: int, allowlist: Optional[str]) -> Tuple[str, float]:
    cfg = f"--oem 3 --psm {int(psm)}"

    if allowlist:
        # CHANGED:
        # pytesseract разбирает config через shlex.split(); если в allowlist есть символ ",
        # то без экранирования получается "No closing quotation".
        safe = allowlist.replace("\\", "\\\\").replace('"', '\\"')
        cfg += f' -c tessedit_char_whitelist="{safe}"'

    data = pytesseract.image_to_data(
        gray_or_bw,
        lang=lang,
        config=cfg,
        output_type=pytesseract.Output.DICT
    )

    words: List[str] = []
    confs: List[float] = []

    n = len(data.get("text", []))
    for i in range(n):
        w = (data["text"][i] or "").strip()
        c = data.get("conf", ["-1"] * n)[i]
        try:
            c_val = float(c)
        except Exception:
            c_val = -1.0

        if w:
            words.append(w)
        if c_val >= 0:
            confs.append(c_val)

    text = " ".join(words).strip()
    if not confs:
        return text, 0.0

    conf = max(0.0, min(1.0, float(np.mean(confs)) / 100.0))
    return text, conf



def _easyocr_recognize(reader: "easyocr.Reader", img_rgb: np.ndarray, decoder: str, beam_width: int,
                       allowlist: Optional[str], paragraph: bool) -> Tuple[str, float]:
    # easyocr принимает RGB
    res = reader.readtext(img_rgb, detail=1, paragraph=paragraph, decoder=decoder, beamWidth=beam_width,
                          allowlist=allowlist)
    if not res:
        return "", 0.0
    # res: [ (bbox, text, conf), ... ]
    texts = [r[1] for r in res if isinstance(r, (list, tuple)) and len(r) >= 3]
    confs = [float(r[2]) for r in res if isinstance(r, (list, tuple)) and len(r) >= 3]

    text = " ".join([t.strip() for t in texts if t.strip()]).strip()
    conf = max(0.0, min(1.0, float(np.mean(confs)) if confs else 0.0))
    return text, conf


# ----------------------------
# Report
# ----------------------------

def _draw_overlay(image_bgr: np.ndarray, blocks: List[Dict[str, Any]], out_png: Path) -> None:
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)

    # попытка подобрать шрифт; если нет — будет дефолтный
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for b in blocks:
        bb = b.get("bbox_xyxy")
        if not bb or len(bb) != 4:
            continue
        x1, y1, x2, y2 = map(int, bb)
        conf = float(b.get("confidence", 0.0))
        text = f'blk:{b.get("block_id")} conf:{conf:.2f}'
        draw.rectangle([x1, y1, x2, y2], outline=(255, 128, 0), width=2)
        draw.text((x1 + 2, max(0, y1 - 16)), text, fill=(255, 128, 0), font=font)

    pil.save(out_png)


def _write_html_report(out_dir: Path, image_name: str, blocks: List[Dict[str, Any]]) -> None:
    report_dir = out_dir / "report"
    crops_dir = report_dir / "crops"
    _ensure_dir(crops_dir)

    # overlay already saved by caller
    html_path = report_dir / "index.html"

    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    rows = []
    for b in blocks:
        bid = b.get("block_id")
        conf = float(b.get("confidence", 0.0))
        txt = b.get("text", "")
        crop_rel = f"crops/blk_{bid}.png"
        rows.append(
            f"<tr>"
            f"<td>{bid}</td>"
            f"<td>{conf:.3f}</td>"
            f"<td><img src='{crop_rel}' style='max-width:320px; border:1px solid #ddd;'/></td>"
            f"<td style='white-space:pre-wrap; font-family:monospace;'>{esc(txt)}</td>"
            f"</tr>"
        )

    html = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<title>OCR report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 16px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
th {{ background: #f4f4f4; }}
.small {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
<h2>OCR report</h2>
<div class="small">Открой overlay.png и пройди по таблице: это самый быстрый способ оценить качество.</div>
<p><img src="overlay.png" style="max-width:100%; border:1px solid #ddd;"/></p>
<table>
<thead>
<tr><th>block_id</th><th>confidence</th><th>crop</th><th>text</th></tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def _save_block_crops(img_bgr: np.ndarray, blocks: List[Dict[str, Any]], out_dir: Path) -> None:
    crops_dir = out_dir / "report" / "crops"
    _ensure_dir(crops_dir)

    for b in blocks:
        bb = b.get("bbox_xyxy")
        if not bb or len(bb) != 4:
            continue
        bid = b.get("block_id")
        x1, y1, x2, y2 = map(int, bb)
        crop = img_bgr[y1:y2, x1:x2]
        out_path = crops_dir / f"blk_{bid}.png"
        cv2.imwrite(str(out_path), crop)


# ----------------------------
# Main OCR logic
# ----------------------------

def _prepare_variants(gray: np.ndarray, cfg: Dict[str, Any]) -> List[Tuple[str, np.ndarray]]:
    variants: List[Tuple[str, np.ndarray]] = []

    base_gray = gray.copy()
    variants.append(("gray", base_gray))
    variants.append(("otsu", _otsu(base_gray)))
    variants.append(("clahe", _clahe(base_gray)))
    variants.append(("clahe_otsu", _otsu(_clahe(base_gray))))

    if cfg.get("use_adaptive_bin", False):
        variants.append(("adaptive", _adaptive_bin(base_gray, int(cfg["adaptive_block"]), int(cfg["adaptive_c"]))))
        variants.append(("clahe_adaptive", _adaptive_bin(_clahe(base_gray), int(cfg["adaptive_block"]), int(cfg["adaptive_c"]))))

    if cfg.get("remove_lines", False):
        rm = _remove_lines_inpaint(base_gray, int(cfg["line_h_kernel"]), int(cfg["line_v_kernel"]), int(cfg["line_iter"]))
        variants.append(("gray_rm_lines", rm))
        variants.append(("otsu_rm_lines", _otsu(rm)))
        variants.append(("clahe_rm_lines", _clahe(rm)))
        variants.append(("clahe_otsu_rm_lines", _otsu(_clahe(rm))))

        rm_blur = cv2.GaussianBlur(rm, (3, 3), 0)
        variants.append(("gray_rm_lines_blur", rm_blur))
        variants.append(("otsu_rm_lines_blur", _otsu(rm_blur)))
        variants.append(("clahe_rm_lines_blur", _clahe(rm_blur)))
        variants.append(("clahe_otsu_rm_lines_blur", _otsu(_clahe(rm_blur))))

    return variants


def recognize_blocks(
    img_bgr: np.ndarray,
    blocks: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    debug: bool,
) -> List[Dict[str, Any]]:
    lang = cfg["lang"]
    allowlist = cfg.get("allowlist") or None
    min_crop_height = int(cfg["min_crop_height"])
    pad_px = int(cfg["pad_px"])
    inner_crop_px = int(cfg["inner_crop_px"])
    upscale_factor = float(cfg["upscale_factor"])
    try_rotate_90 = bool(cfg["try_rotate_90"])
    tesseract_enabled = bool(cfg["tesseract_enabled"])
    tesseract_lang = cfg["tesseract_lang"]
    psm_single = int(cfg["tesseract_psm_single_line"])
    psm_block = int(cfg["tesseract_psm_block"])
    domain_postprocess = bool(cfg.get("domain_postprocess", True))

    # init easyocr reader once
    reader = None
    if "easyocr" in cfg["engine"]:
        if easyocr is None:
            raise RuntimeError("easyocr is not installed but engine includes easyocr")
        reader = easyocr.Reader(
            [lang],
            gpu=bool(cfg["gpu"]),
            download_enabled=bool(cfg["download_enabled"]),
            model_storage_directory=cfg["model_storage_directory"],
        )

    h, w = img_bgr.shape[:2]
    out_blocks: List[Dict[str, Any]] = []

    rotations = ["rot0"]
    if try_rotate_90:
        rotations.append("rot90")

    for b in blocks:
        bid = b.get("block_id")
        bb_any = _bbox_from_any(b)
        bb_px = _maybe_scale_bbox(bb_any, w, h)
        x1, y1, x2, y2 = bb_px

        # pad bbox
        x1 -= pad_px
        y1 -= pad_px
        x2 += pad_px
        y2 += pad_px
        x1, y1, x2, y2 = _clamp_xyxy(x1, y1, x2, y2, w, h)

        # minimal crop height safeguard
        if (y2 - y1) < min_crop_height:
            center = (y1 + y2) // 2
            half = max(min_crop_height // 2, 1)
            y1 = max(0, center - half)
            y2 = min(h, center + half)

        crop_bgr = img_bgr[y1:y2, x1:x2].copy()

        # inner crop: remove frame pixels inside bbox (часто рамки/линии мешают)
        if inner_crop_px > 0 and (crop_bgr.shape[0] > 2 * inner_crop_px) and (crop_bgr.shape[1] > 2 * inner_crop_px):
            crop_bgr = crop_bgr[inner_crop_px:-inner_crop_px, inner_crop_px:-inner_crop_px].copy()

        # upscale
        if upscale_factor and abs(upscale_factor - 1.0) > 1e-6:
            ch, cw = crop_bgr.shape[:2]
            new_w = max(1, int(round(cw * upscale_factor)))
            new_h = max(1, int(round(ch * upscale_factor)))
            interp = cv2.INTER_CUBIC if upscale_factor >= 1.0 else cv2.INTER_AREA
            crop_bgr = cv2.resize(crop_bgr, (new_w, new_h), interpolation=interp)

        best: Optional[OcrPick] = None
        all_picks: List[Dict[str, Any]] = []

        for rot in rotations:
            rot_bgr = _rotate_bgr(crop_bgr, rot)
            gray = _to_gray(rot_bgr)
            variants = _prepare_variants(gray, cfg)

            # choose psm based on aspect ratio / height (simple heuristic)
            block_psm = psm_block
            if gray.shape[0] <= 40 or gray.shape[0] * 2 < gray.shape[1]:
                block_psm = psm_single

            for vname, vimg in variants:
                # tesseract expects gray or bw
                if tesseract_enabled:
                    t_txt, t_conf = _tesseract_recognize(vimg, tesseract_lang, block_psm, allowlist)
                    t_txt2 = _domain_postprocess(t_txt) if domain_postprocess else _normalize_text(t_txt)
                    t_score = _score_text(t_txt2, t_conf)
                    pick = OcrPick(
                        engine="tesseract",
                        variant=vname,
                        rotation=rot,
                        conf=float(t_conf),
                        score=float(t_score),
                        text=t_txt2,
                        extra={"psm": block_psm},
                    )
                    all_picks.append({
                        "engine": pick.engine, "variant": pick.variant, "rotation": pick.rotation,
                        "conf": pick.conf, "score": pick.score, **pick.extra
                    })
                    if best is None or pick.score > best.score:
                        best = pick

                # easyocr expects RGB
                if reader is not None:
                    rgb = cv2.cvtColor(rot_bgr, cv2.COLOR_BGR2RGB)
                    # если вариант бинарный/серый, применим его к RGB: заменим каналами
                    if len(vimg.shape) == 2:
                        rgb = cv2.cvtColor(vimg, cv2.COLOR_GRAY2RGB)
                    e_txt, e_conf = _easyocr_recognize(
                        reader=reader,
                        img_rgb=rgb,
                        decoder=str(cfg["decoder"]),
                        beam_width=int(cfg["beam_width"]),
                        allowlist=allowlist,
                        paragraph=bool(cfg["paragraph"]),
                    )
                    e_txt2 = _domain_postprocess(e_txt) if domain_postprocess else _normalize_text(e_txt)
                    e_score = _score_text(e_txt2, e_conf)
                    pick = OcrPick(
                        engine="easyocr",
                        variant=vname,
                        rotation=rot,
                        conf=float(e_conf),
                        score=float(e_score),
                        text=e_txt2,
                        extra={},
                    )
                    all_picks.append({
                        "engine": pick.engine, "variant": pick.variant, "rotation": pick.rotation,
                        "conf": pick.conf, "score": pick.score
                    })
                    if best is None or pick.score > best.score:
                        best = pick

        # fallback
        if best is None:
            best = OcrPick(engine="none", variant="none", rotation="rot0", conf=0.0, score=0.0, text="", extra={})

        out_block: Dict[str, Any] = {
            "block_id": bid,
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "text": best.text,
            "confidence": float(best.conf),
            "confidence_available": True,
        }

        if debug:
            out_block["chosen"] = all_picks
            out_block["raw"] = []  # оставлено для совместимости

        out_blocks.append(out_block)

    return out_blocks


def _load_blocks(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    coord_space = str(data.get("coord_space", "auto"))
    blocks = data.get("blocks", [])
    if not isinstance(blocks, list):
        raise ValueError("blocks must be a list")
    return coord_space, blocks


def build_summary(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    confs = [float(b.get("confidence", 0.0)) for b in blocks]
    avg = float(np.mean(confs)) if confs else 0.0
    low = [b["block_id"] for b in blocks if float(b.get("confidence", 0.0)) < 0.85]
    empty = [b["block_id"] for b in blocks if not (b.get("text") or "").strip()]
    return {
        "blocks_total": len(blocks),
        "avg_confidence": avg,
        "low_confidence_ids_lt_0_85": low,
        "empty_text_ids": empty,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--lang", default="ru")
    ap.add_argument("--engine", default="easyocr+tesseract", choices=["easyocr", "tesseract", "easyocr+tesseract"])

    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--download-enabled", action="store_true")
    ap.add_argument("--model-storage-directory", default=None)

    ap.add_argument("--max-side", type=int, default=0)
    ap.add_argument("--pad-px", type=int, default=6)
    ap.add_argument("--inner-crop-px", type=int, default=1)
    ap.add_argument("--upscale-factor", type=float, default=2.0)
    ap.add_argument("--min-crop-height", type=int, default=28)
    ap.add_argument("--min-confidence", type=float, default=0.1)

    ap.add_argument("--paragraph", action="store_true")
    ap.add_argument("--decoder", default="greedy", choices=["greedy", "beamsearch"])
    ap.add_argument("--beam-width", type=int, default=5)

    ap.add_argument("--contrast-ths", type=float, default=0.1)  # kept for compatibility (not used directly here)
    ap.add_argument("--adjust-contrast", type=float, default=0.5)  # kept for compatibility (not used directly here)

    ap.add_argument("--use-adaptive-bin", action="store_true")
    ap.add_argument("--adaptive-block", type=int, default=41)
    ap.add_argument("--adaptive-c", type=int, default=11)

    ap.add_argument("--remove-lines", action="store_true")
    ap.add_argument("--line-h-kernel", type=int, default=35)
    ap.add_argument("--line-v-kernel", type=int, default=35)
    ap.add_argument("--line-iter", type=int, default=1)

    ap.add_argument("--allowlist", default="АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789 .,:;!?-–—/()[]\"'№%+")
    ap.add_argument("--try-rotate-90", action="store_true")

    ap.add_argument("--tesseract-enabled", action="store_true")
    ap.add_argument("--tesseract-lang", default="rus")
    ap.add_argument("--tesseract-psm-single-line", type=int, default=7)
    ap.add_argument("--tesseract-psm-block", type=int, default=6)

    ap.add_argument("--domain-postprocess", action="store_true", default=True)

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--report", action="store_true")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    # Load image
    img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Cannot read image: {args.input}")

    img_bgr, scale = _resize_max_side(img_bgr, args.max_side if args.max_side > 0 else None)

    # Load blocks
    coord_space, blocks = _load_blocks(Path(args.blocks))

    cfg = {
        "coord_space": coord_space,
        "engine": args.engine,
        "lang": args.lang,
        "gpu": bool(args.gpu),
        "download_enabled": bool(args.download_enabled),
        "model_storage_directory": args.model_storage_directory,

        "max_side": int(args.max_side) if args.max_side else 0,
        "pad_px": int(args.pad_px),
        "inner_crop_px": int(args.inner_crop_px),
        "upscale_factor": float(args.upscale_factor),
        "min_crop_height": int(args.min_crop_height),
        "min_confidence": float(args.min_confidence),

        "paragraph": bool(args.paragraph),
        "decoder": str(args.decoder),
        "beam_width": int(args.beam_width),

        "contrast_ths": float(args.contrast_ths),
        "adjust_contrast": float(args.adjust_contrast),

        "use_adaptive_bin": bool(args.use_adaptive_bin),
        "adaptive_block": int(args.adaptive_block),
        "adaptive_c": int(args.adaptive_c),

        "remove_lines": bool(args.remove_lines),
        "line_h_kernel": int(args.line_h_kernel),
        "line_v_kernel": int(args.line_v_kernel),
        "line_iter": int(args.line_iter),

        "allowlist": str(args.allowlist) if args.allowlist else None,
        "try_rotate_90": bool(args.try_rotate_90),

        "tesseract_enabled": bool(args.tesseract_enabled) or ("tesseract" in args.engine),
        "tesseract_lang": str(args.tesseract_lang),
        "tesseract_psm_single_line": int(args.tesseract_psm_single_line),
        "tesseract_psm_block": int(args.tesseract_psm_block),

        "domain_postprocess": bool(args.domain_postprocess),
    }

    # If engine is tesseract-only, no easyocr needed
    if args.engine == "tesseract":
        cfg["engine"] = "tesseract"
    elif args.engine == "easyocr":
        cfg["engine"] = "easyocr"
    else:
        cfg["engine"] = "easyocr+tesseract"

    # OCR
    out_blocks = recognize_blocks(img_bgr=img_bgr, blocks=blocks, cfg=cfg, debug=bool(args.debug))
    summary = build_summary(out_blocks)

    ocr_json = {
        "coord_space": "auto",  # result coords are image pixel coords after max-side resize
        "engine": cfg["engine"],
        "ocr_config": cfg,
        "summary": summary,
        "blocks": out_blocks,
    }

    out_json_path = outdir / "ocr.json"
    out_json_path.write_text(json.dumps(ocr_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ocr_res] saved: {out_json_path}")
    print(f"[ocr_res] summary: {json.dumps(summary, ensure_ascii=False)}")

    if args.report:
        report_dir = outdir / "report"
        _ensure_dir(report_dir)

        overlay_path = report_dir / "overlay.png"
        _draw_overlay(img_bgr, out_blocks, overlay_path)
        _save_block_crops(img_bgr, out_blocks, outdir)
        _write_html_report(outdir, Path(args.input).name, out_blocks)

        print(f"[ocr_res] report overlay: {overlay_path}")
        print(f"[ocr_res] report html:    {report_dir / 'index.html'}")


if __name__ == "__main__":
    main()
