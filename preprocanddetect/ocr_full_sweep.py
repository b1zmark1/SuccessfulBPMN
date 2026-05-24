"""
OCR-шаг pipeline'а: crop-based EasyOCR.

ИДЕЯ: не запускать EasyOCR на всей картинке (это 100+ сек на CPU),
а нарезать ORIGINAL картинку по bbox'ам YOLOX-shape детектов и
прогонять EasyOCR на каждом маленьком кропе отдельно. Reader загружается
один раз, инференс на каждом кропе ~0.1-0.3 сек. Для 40-60 shapes ≈ 10-20 сек.

Дополнительно: сканируется левая полоса картинки (~10%) для вертикальных
лейн-заголовков (типа "Студофис"). Эти подписи не попадают ни в одну YOLOX-shape.

Все bbox'ы OCR-результата переводятся в MODEL-space (через resize_ratio из ensemble.json),
чтобы соответствовать YOLOX-координатам.

Заменяет: detect_res.py + ocr_tesseract_fast.py + ocr_heavy_pass.py + полный EasyOCR-проход.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image


# Какие YOLOX-классы нарезаем под OCR. text_annotation вне списка — обычно у них
# свой bbox, и нет смысла отдельно OCR'ить (его и так захватит соседняя task если есть).
SHAPE_CLASSES_FOR_OCR: Set[str] = {
    "start_event",
    "intermediate_event",
    "end_event",
    "task",
    "gateway_exclusive",
    "gateway_parallel",
    "gateway_inclusive",
    "subprocess",
    "data_object",
    "text_annotation",
    "pool",
    "lane",
}


def _polygon_to_xyxy(poly: Any) -> Optional[Tuple[int, int, int, int]]:
    if not poly:
        return None
    try:
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
    except Exception:
        return None
    if not xs or not ys:
        return None
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Original image path")
    ap.add_argument(
        "--ensemble-json",
        required=True,
        help="ensemble.json от ensemble_infer.py: даёт YOLOX-bbox'ы и resize_ratio",
    )
    ap.add_argument("--outdir", required=True, help="Куда писать ocr.json")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument(
        "--shape-pad",
        type=int,
        default=8,
        help="Расширение shape-bbox перед обрезкой (пиксели в ORIGINAL space)",
    )
    ap.add_argument(
        "--margin-ratio",
        type=float,
        default=0.1,
        help="Доля ширины слева для сканирования вертикальных лейн-заголовков. 0 = выключить",
    )
    ap.add_argument(
        "--min-crop-side",
        type=int,
        default=20,
        help="Минимальный размер кропа (по любой стороне) — мельче пропускаем",
    )
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input image not found: {args.input}")
    if not os.path.exists(args.ensemble_json):
        raise SystemExit(f"Ensemble JSON not found: {args.ensemble_json}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    with open(args.ensemble_json, "r", encoding="utf-8") as f:
        ensemble = json.load(f)
    img_meta = ensemble.get("image", {})
    resize_ratio = float(img_meta.get("resize_ratio", 1.0))
    model_w = int(img_meta.get("width", 0))
    model_h = int(img_meta.get("height", 0))
    if resize_ratio <= 0:
        resize_ratio = 1.0

    detections = ensemble.get("detections") or []

    # ORIGINAL image (без preprocess — на нём шрифт нормального размера)
    orig_pil = Image.open(args.input).convert("RGB")
    orig_w, orig_h = orig_pil.size
    orig_np = np.array(orig_pil)  # RGB

    inv_resize = 1.0 / resize_ratio  # model -> original

    # EasyOCR reader — грузится ОДИН раз
    import easyocr  # type: ignore

    reader = easyocr.Reader([args.lang], gpu=bool(args.gpu))

    print(
        f"[ocr_full] orig={orig_w}x{orig_h} model={model_w}x{model_h} "
        f"resize_ratio={resize_ratio:.3f} shapes_total={len(detections)} "
        f"lang={args.lang} gpu={args.gpu}",
        flush=True,
    )

    blocks: List[Dict[str, Any]] = []
    block_id = 0
    shapes_processed = 0
    shapes_with_text = 0
    crop_ocr_time = 0.0

    for det in detections:
        if not isinstance(det, dict):
            continue
        cls = str(det.get("class_name") or "").strip().lower()
        if cls not in SHAPE_CLASSES_FOR_OCR:
            continue
        bb = det.get("bbox_xyxy") or det.get("bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        try:
            mx1, my1, mx2, my2 = [float(v) for v in bb]
        except Exception:
            continue
        if mx2 <= mx1 or my2 <= my1:
            continue

        # MODEL -> ORIGINAL
        ox1 = int(round(mx1 * inv_resize)) - args.shape_pad
        oy1 = int(round(my1 * inv_resize)) - args.shape_pad
        ox2 = int(round(mx2 * inv_resize)) + args.shape_pad
        oy2 = int(round(my2 * inv_resize)) + args.shape_pad
        ox1 = _clip(ox1, 0, orig_w)
        oy1 = _clip(oy1, 0, orig_h)
        ox2 = _clip(ox2, 0, orig_w)
        oy2 = _clip(oy2, 0, orig_h)
        if (ox2 - ox1) < args.min_crop_side or (oy2 - oy1) < args.min_crop_side:
            continue

        crop = orig_np[oy1:oy2, ox1:ox2]
        shapes_processed += 1

        t_crop = time.time()
        try:
            results = reader.readtext(crop, detail=1, paragraph=False)
        except Exception as e:
            print(f"[ocr_full] crop error on det class={cls}: {e}", flush=True)
            continue
        crop_ocr_time += time.time() - t_crop

        if not results:
            continue

        added = 0
        for item in results:
            if not item:
                continue
            poly = item[0]
            text = str(item[1]).strip() if len(item) > 1 else ""
            if not text:
                continue
            conf = 0.0
            if len(item) >= 3:
                try:
                    conf = float(item[2])
                except Exception:
                    conf = 0.0

            cbox = _polygon_to_xyxy(poly)
            if cbox is None:
                continue
            cx1, cy1, cx2, cy2 = cbox

            # crop -> ORIGINAL -> MODEL
            ox1_t = ox1 + cx1
            oy1_t = oy1 + cy1
            ox2_t = ox1 + cx2
            oy2_t = oy1 + cy2
            bx1 = int(round(ox1_t * resize_ratio))
            by1 = int(round(oy1_t * resize_ratio))
            bx2 = int(round(ox2_t * resize_ratio))
            by2 = int(round(oy2_t * resize_ratio))

            block_id += 1
            blocks.append(
                {
                    "block_id": block_id,
                    "bbox_xyxy": [bx1, by1, bx2, by2],
                    "bbox_xywh": [bx1, by1, max(0, bx2 - bx1), max(0, by2 - by1)],
                    "text": text,
                    "confidence": conf,
                    "confidence_available": True,
                    "source": "easyocr_crop",
                }
            )
            added += 1

        if added:
            shapes_with_text += 1

    # Сканирование левой полосы для вертикальных лейн-заголовков
    margin_added = 0
    margin_time = 0.0
    if args.margin_ratio > 0:
        margin_w = int(orig_w * args.margin_ratio)
        if margin_w > 30:
            margin_crop = orig_np[:, :margin_w]
            t_m = time.time()
            try:
                results = reader.readtext(
                    margin_crop, detail=1, paragraph=False, rotation_info=[90, 270]
                )
            except TypeError:
                results = reader.readtext(margin_crop, detail=1, paragraph=False)
            except Exception as e:
                print(f"[ocr_full] margin scan error: {e}", flush=True)
                results = []
            margin_time = time.time() - t_m

            for item in results:
                if not item:
                    continue
                poly = item[0]
                text = str(item[1]).strip() if len(item) > 1 else ""
                if not text:
                    continue
                conf = 0.0
                if len(item) >= 3:
                    try:
                        conf = float(item[2])
                    except Exception:
                        conf = 0.0

                cbox = _polygon_to_xyxy(poly)
                if cbox is None:
                    continue
                cx1, cy1, cx2, cy2 = cbox

                bx1 = int(round(cx1 * resize_ratio))
                by1 = int(round(cy1 * resize_ratio))
                bx2 = int(round(cx2 * resize_ratio))
                by2 = int(round(cy2 * resize_ratio))

                block_id += 1
                blocks.append(
                    {
                        "block_id": block_id,
                        "bbox_xyxy": [bx1, by1, bx2, by2],
                        "bbox_xywh": [bx1, by1, max(0, bx2 - bx1), max(0, by2 - by1)],
                        "text": text,
                        "confidence": conf,
                        "confidence_available": True,
                        "source": "easyocr_margin",
                    }
                )
                margin_added += 1

    # Reading order
    blocks.sort(key=lambda b: (b["bbox_xyxy"][1], b["bbox_xyxy"][0]))

    out = {
        "coord_space": "model",
        "engine": "easyocr_crop+margin",
        "image": {
            "w": model_w,
            "h": model_h,
            "path": args.input,
        },
        "ocr_config": {
            "lang": args.lang,
            "shape_pad": args.shape_pad,
            "margin_ratio": args.margin_ratio,
            "min_crop_side": args.min_crop_side,
            "resize_ratio": resize_ratio,
        },
        "stats": {
            "shapes_total_in_ensemble": len(detections),
            "shapes_processed": shapes_processed,
            "shapes_with_text": shapes_with_text,
            "margin_blocks_added": margin_added,
            "total_blocks": len(blocks),
            "crop_ocr_time_sec": round(crop_ocr_time, 2),
            "margin_ocr_time_sec": round(margin_time, 2),
        },
        "blocks": blocks,
    }
    out_path = outdir / "ocr.json"
    _save_json(str(out_path), out)

    dt = time.time() - t0
    print(
        f"[ocr_full] blocks={len(blocks)} (crops={len(blocks) - margin_added}, margin={margin_added}) "
        f"elapsed={dt:.1f}s (crop_ocr={crop_ocr_time:.1f}s, margin={margin_time:.1f}s) "
        f"saved={out_path}",
        flush=True,
    )
    # Топ-5 sample
    for b in sorted(blocks, key=lambda x: -float(x.get("confidence", 0)))[:5]:
        print(
            f"[ocr_full] sample: text={b['text']!r} conf={b['confidence']:.3f} "
            f"src={b['source']} bbox={b['bbox_xyxy']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
