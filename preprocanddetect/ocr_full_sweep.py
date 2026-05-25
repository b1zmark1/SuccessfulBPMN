"""
MODERN OCR-шаг pipeline'а для больших BPMN-картинок.

Используется в `workers/image_to_text_pipeline.py` для картинок с
max_side >= IMG_PIPELINE_SIZE_THRESHOLD (≈ 2500 px по умолчанию). Для маленьких
картинок задействуется legacy-цепочка (detect_res + ocr_tesseract_fast + label_res),
которая на мелких диаграммах даёт лучшее качество.

Поддерживаемые режимы (--mode):
  - `crop` (дефолт для большой картинки): режем оригинал по YOLOX shape-bbox'ам
    и OCR'им каждый кроп отдельно. EasyOCR на полной картинке 5000+ px на CPU
    занимает 120+ сек — не лезет в бюджет.
  - `full`: один EasyOCR-проход по всей картинке (с опциональным cap по длинной
    стороне). Полезно для тестов / fallback'а.
  - `auto`: full если max_side < `--full-image-threshold`, иначе crop.

Дополнительно (в crop-режиме): отдельное сканирование левой полосы (~10% ширины)
с rotation_info=[90,270] — это ловит вертикально написанные lane-заголовки.

Все bbox'ы OCR-результата переводятся в MODEL-space (через resize_ratio из
ensemble.json), чтобы попадать в ту же систему координат что и YOLOX-детекты.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image


# Какие YOLOX-классы нарезаем в crop-режиме.
# ИСКЛЮЧЕНЫ pool/lane/subprocess: это большие контейнеры, физически содержащие
# мелкие task'и/gateway'и. Если OCR'ить и контейнер, и его содержимое — текст
# дублируется в N копий.
SHAPE_CLASSES_FOR_OCR: Set[str] = {
    "start_event",
    "intermediate_event",
    "end_event",
    "task",
    "gateway_exclusive",
    "gateway_parallel",
    "gateway_inclusive",
    "data_object",
    "text_annotation",
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


def _run_full_image_ocr(
    reader: Any,
    orig_np: np.ndarray,
    orig_w: int,
    orig_h: int,
    resize_ratio: float,
    cap: int,
    rotation_info: Optional[List[int]],
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Полный EasyOCR-проход по всей (опционально capped) картинке.
    Хорош для маленьких BPMN — нет фрагментации, текст читается целиком.
    """
    # Кэп длинной стороны (на всякий — обычно small image меньше cap'а)
    img_for_ocr = orig_np
    cap_scale = 1.0
    if cap > 0 and max(orig_w, orig_h) > cap:
        cap_scale = cap / float(max(orig_w, orig_h))
        new_w = max(1, int(round(orig_w * cap_scale)))
        new_h = max(1, int(round(orig_h * cap_scale)))
        img_for_ocr = cv2.resize(orig_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

    t = time.time()
    try:
        if rotation_info:
            results = reader.readtext(
                img_for_ocr, detail=1, paragraph=False, rotation_info=rotation_info
            )
        else:
            results = reader.readtext(img_for_ocr, detail=1, paragraph=False)
    except TypeError:
        results = reader.readtext(img_for_ocr, detail=1, paragraph=False)
    elapsed = time.time() - t

    inv_cap = 1.0 / cap_scale if abs(cap_scale - 1.0) > 1e-6 else 1.0
    total_scale = inv_cap * resize_ratio  # capped → original → model

    blocks: List[Dict[str, Any]] = []
    for idx, item in enumerate(results):
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

        bbox = _polygon_to_xyxy(poly)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox

        bx1 = int(round(x1 * total_scale))
        by1 = int(round(y1 * total_scale))
        bx2 = int(round(x2 * total_scale))
        by2 = int(round(y2 * total_scale))

        blocks.append(
            {
                "block_id": idx + 1,
                "bbox_xyxy": [bx1, by1, bx2, by2],
                "bbox_xywh": [bx1, by1, max(0, bx2 - bx1), max(0, by2 - by1)],
                "text": text,
                "confidence": conf,
                "confidence_available": True,
                "source": "easyocr_full",
            }
        )

    return blocks, elapsed


def _run_crop_based_ocr(
    reader: Any,
    orig_np: np.ndarray,
    orig_w: int,
    orig_h: int,
    resize_ratio: float,
    detections: List[Dict[str, Any]],
    shape_pad: int,
    min_crop_side: int,
    upscale_min_height: int,
    upscale_max_factor: float,
) -> Tuple[List[Dict[str, Any]], float, int, int]:
    """Crop по YOLOX-shape, OCR на каждом отдельно. Хорош для больших картинок."""
    inv_resize = 1.0 / resize_ratio
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

        ox1 = int(round(mx1 * inv_resize)) - shape_pad
        oy1 = int(round(my1 * inv_resize)) - shape_pad
        ox2 = int(round(mx2 * inv_resize)) + shape_pad
        oy2 = int(round(my2 * inv_resize)) + shape_pad
        ox1 = _clip(ox1, 0, orig_w)
        oy1 = _clip(oy1, 0, orig_h)
        ox2 = _clip(ox2, 0, orig_w)
        oy2 = _clip(oy2, 0, orig_h)
        if (ox2 - ox1) < min_crop_side or (oy2 - oy1) < min_crop_side:
            continue

        crop = orig_np[oy1:oy2, ox1:ox2]
        crop_h, crop_w = crop.shape[:2]
        shapes_processed += 1

        crop_for_ocr = crop
        ocr_upscale = 1.0
        if crop_h < upscale_min_height and crop_h > 0:
            factor = min(upscale_max_factor, upscale_min_height / float(crop_h))
            if factor > 1.05:
                new_w = max(1, int(round(crop_w * factor)))
                new_h = max(1, int(round(crop_h * factor)))
                crop_for_ocr = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                ocr_upscale = factor

        t_crop = time.time()
        try:
            results = reader.readtext(crop_for_ocr, detail=1, paragraph=False)
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

            if ocr_upscale != 1.0:
                inv_up = 1.0 / ocr_upscale
                cx1 = int(round(cx1 * inv_up))
                cy1 = int(round(cy1 * inv_up))
                cx2 = int(round(cx2 * inv_up))
                cy2 = int(round(cy2 * inv_up))

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

    return blocks, crop_ocr_time, shapes_processed, shapes_with_text


def _run_margin_scan(
    reader: Any,
    orig_np: np.ndarray,
    orig_w: int,
    margin_ratio: float,
    resize_ratio: float,
    start_block_id: int,
) -> Tuple[List[Dict[str, Any]], float]:
    """Сканирование левой полосы картинки с rotation для вертикальных лейн-заголовков."""
    blocks: List[Dict[str, Any]] = []
    if margin_ratio <= 0:
        return blocks, 0.0
    margin_w = int(orig_w * margin_ratio)
    if margin_w <= 30:
        return blocks, 0.0

    margin_crop = orig_np[:, :margin_w]
    t = time.time()
    try:
        results = reader.readtext(
            margin_crop, detail=1, paragraph=False, rotation_info=[90, 270]
        )
    except TypeError:
        results = reader.readtext(margin_crop, detail=1, paragraph=False)
    except Exception as e:
        print(f"[ocr_full] margin scan error: {e}", flush=True)
        return blocks, time.time() - t
    elapsed = time.time() - t

    block_id = start_block_id
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

    return blocks, elapsed


def _dedup_blocks(blocks: List[Dict[str, Any]], dist: float = 15.0) -> Tuple[List[Dict[str, Any]], int]:
    """Дедуп: одинаковый текст с центром в радиусе dist пикселей → оставляем тот что conf'ом выше."""
    if not blocks:
        return blocks, 0

    def _cx(b: Dict[str, Any]) -> float:
        x1, _, x2, _ = b["bbox_xyxy"]
        return (x1 + x2) / 2.0

    def _cy(b: Dict[str, Any]) -> float:
        _, y1, _, y2 = b["bbox_xyxy"]
        return (y1 + y2) / 2.0

    sorted_blocks = sorted(blocks, key=lambda b: -float(b.get("confidence", 0)))
    kept: List[Dict[str, Any]] = []
    for b in sorted_blocks:
        t_b = str(b.get("text", "")).strip().lower()
        cx_b, cy_b = _cx(b), _cy(b)
        is_dup = False
        for k in kept:
            if t_b != str(k.get("text", "")).strip().lower():
                continue
            if abs(cx_b - _cx(k)) <= dist and abs(cy_b - _cy(k)) <= dist:
                is_dup = True
                break
        if not is_dup:
            kept.append(b)
    return kept, len(blocks) - len(kept)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Original image path")
    ap.add_argument(
        "--ensemble-json",
        required=True,
        help="ensemble.json от ensemble_infer.py",
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument(
        "--mode",
        choices=["auto", "full", "crop"],
        default="auto",
        help="auto — выбор по размеру; full — всегда full-image; crop — всегда crop-based",
    )
    ap.add_argument(
        "--full-image-threshold",
        type=int,
        default=2500,
        help="В режиме auto: если max(orig_w, orig_h) < threshold → full-image OCR",
    )
    ap.add_argument(
        "--full-image-cap",
        type=int,
        default=2500,
        help="Кэп длинной стороны для full-image режима (защита от гигантских картинок)",
    )
    ap.add_argument(
        "--full-image-rotation",
        type=str,
        default="90,270",
        help="rotation_info в full-image режиме (для вертикального текста). Пусто — выкл.",
    )
    ap.add_argument(
        "--shape-pad",
        type=int,
        default=8,
        help="(crop-режим) Расширение shape-bbox перед обрезкой, ORIGINAL px",
    )
    ap.add_argument(
        "--margin-ratio",
        type=float,
        default=0.1,
        help="(crop-режим) Доля ширины слева для вертикальных лейн-подписей. 0 = выкл.",
    )
    ap.add_argument(
        "--min-crop-side",
        type=int,
        default=20,
    )
    ap.add_argument(
        "--upscale-min-height",
        type=int,
        default=96,
        help="(crop-режим) Если высота кропа меньше — апсемпл пропорционально",
    )
    ap.add_argument(
        "--upscale-max-factor",
        type=float,
        default=5.0,
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
    if resize_ratio <= 0:
        resize_ratio = 1.0
    model_w = int(img_meta.get("width", 0))
    model_h = int(img_meta.get("height", 0))
    detections = ensemble.get("detections") or []

    orig_pil = Image.open(args.input).convert("RGB")
    orig_w, orig_h = orig_pil.size
    orig_np = np.array(orig_pil)

    # Решаем режим
    max_side = max(orig_w, orig_h)
    if args.mode == "auto":
        chosen_mode = "full" if max_side < args.full_image_threshold else "crop"
    else:
        chosen_mode = args.mode

    # EasyOCR reader (один раз)
    import easyocr  # type: ignore

    reader = easyocr.Reader([args.lang], gpu=bool(args.gpu))

    print(
        f"[ocr_full] orig={orig_w}x{orig_h} model={model_w}x{model_h} "
        f"resize_ratio={resize_ratio:.3f} mode={chosen_mode} "
        f"shapes_in_ensemble={len(detections)} lang={args.lang} gpu={args.gpu}",
        flush=True,
    )

    blocks: List[Dict[str, Any]] = []
    full_time = 0.0
    crop_time = 0.0
    margin_time = 0.0
    shapes_processed = 0
    shapes_with_text = 0
    margin_added = 0

    if chosen_mode == "full":
        rotation_info: Optional[List[int]] = None
        if args.full_image_rotation.strip():
            try:
                rotation_info = [
                    int(x.strip())
                    for x in args.full_image_rotation.split(",")
                    if x.strip()
                ]
            except Exception:
                rotation_info = None
        blocks, full_time = _run_full_image_ocr(
            reader,
            orig_np,
            orig_w,
            orig_h,
            resize_ratio,
            cap=args.full_image_cap,
            rotation_info=rotation_info,
        )
    else:
        blocks, crop_time, shapes_processed, shapes_with_text = _run_crop_based_ocr(
            reader,
            orig_np,
            orig_w,
            orig_h,
            resize_ratio,
            detections,
            shape_pad=args.shape_pad,
            min_crop_side=args.min_crop_side,
            upscale_min_height=args.upscale_min_height,
            upscale_max_factor=args.upscale_max_factor,
        )
        margin_blocks, margin_time = _run_margin_scan(
            reader,
            orig_np,
            orig_w,
            args.margin_ratio,
            resize_ratio,
            start_block_id=len(blocks),
        )
        margin_added = len(margin_blocks)
        blocks.extend(margin_blocks)

    blocks, dedup_dropped = _dedup_blocks(blocks)
    blocks.sort(key=lambda b: (b["bbox_xyxy"][1], b["bbox_xyxy"][0]))

    out = {
        "coord_space": "model",
        "engine": f"easyocr_{chosen_mode}",
        "image": {"w": model_w, "h": model_h, "path": args.input},
        "ocr_config": {
            "mode": chosen_mode,
            "lang": args.lang,
            "full_image_threshold": args.full_image_threshold,
            "full_image_cap": args.full_image_cap,
            "shape_pad": args.shape_pad,
            "margin_ratio": args.margin_ratio,
            "resize_ratio": resize_ratio,
        },
        "stats": {
            "shapes_in_ensemble": len(detections),
            "shapes_processed": shapes_processed,
            "shapes_with_text": shapes_with_text,
            "margin_blocks_added": margin_added,
            "dedup_dropped": dedup_dropped,
            "total_blocks": len(blocks),
            "full_ocr_time_sec": round(full_time, 2),
            "crop_ocr_time_sec": round(crop_time, 2),
            "margin_ocr_time_sec": round(margin_time, 2),
        },
        "blocks": blocks,
    }
    out_path = outdir / "ocr.json"
    _save_json(str(out_path), out)

    dt = time.time() - t0
    print(
        f"[ocr_full] mode={chosen_mode} blocks={len(blocks)} "
        f"(dedup_dropped={dedup_dropped}, margin={margin_added}) "
        f"shapes_processed={shapes_processed}/{len(detections)} "
        f"elapsed={dt:.1f}s (full={full_time:.1f}s, crop={crop_time:.1f}s, margin={margin_time:.1f}s) "
        f"saved={out_path}",
        flush=True,
    )
    for b in sorted(blocks, key=lambda x: -float(x.get("confidence", 0)))[:5]:
        print(
            f"[ocr_full] sample: text={b['text']!r} conf={b['confidence']:.3f} "
            f"src={b['source']} bbox={b['bbox_xyxy']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
