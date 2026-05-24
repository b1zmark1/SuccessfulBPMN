from __future__ import annotations

# Полный EasyOCR-проход по всему изображению.
# Идея: bbox'ы fast-стадии (detect_res.py + EasyOCR detect) для русских BPMN-диаграмм
# часто фрагментируют слова, и Tesseract по этим фрагментам выдаёт мусор ("ати", "ри"...).
# EasyOCR в режиме detect+recognize группирует символы в слова/строки сам, поэтому
# вместо пост-обработки кусков мы полностью заменяем блоки результатом единого прогона.

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _polygon_to_xyxy(poly: Any) -> Tuple[int, int, int, int] | None:
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


def _run_easyocr_sweep(
    img_array: np.ndarray, lang: str, gpu: bool, paragraph: bool
) -> List[Dict[str, Any]]:
    import easyocr  # type: ignore

    reader = easyocr.Reader([lang], gpu=bool(gpu))
    # rotation_info=[90,270] — детектор дополнительно прогоняет картинку с поворотом,
    # чтобы поймать ВЕРТИКАЛЬНО подписанные swim-lane labels (типичный BPMN-кейс).
    try:
        results = reader.readtext(
            img_array, detail=1, paragraph=paragraph, rotation_info=[90, 270]
        )
    except TypeError:
        results = reader.readtext(img_array, detail=1, paragraph=paragraph)

    blocks: List[Dict[str, Any]] = []
    for idx, item in enumerate(results):
        if not item:
            continue
        # paragraph=False -> (poly, text, conf)
        # paragraph=True  -> (poly, text)   (без confidence)
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
        else:
            # paragraph-режим без conf — ставим консервативное значение
            conf = 0.6

        bbox = _polygon_to_xyxy(poly)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox

        blocks.append(
            {
                "block_id": idx + 1,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, max(0, x2 - x1), max(0, y2 - y1)],
                "text": text,
                "confidence": conf,
                "confidence_available": True,
                "source": "easyocr_full",
                "chosen": [{"engine": "easyocr_full", "conf": conf, "paragraph": paragraph}],
            }
        )

    # Сортировка в порядке чтения (сверху вниз, слева направо).
    blocks.sort(key=lambda b: (b["bbox_xyxy"][1], b["bbox_xyxy"][0]))
    return blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Original image path")
    ap.add_argument("--ocr-json", required=True, help="Fast OCR result JSON (will be overwritten)")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument(
        "--paragraph",
        action="store_true",
        help="Группировать строки в абзацы (для BPMN обычно НЕ нужно — лучше пословно)",
    )
    ap.add_argument(
        "--upscale",
        type=float,
        default=1.0,
        help="Апсемплинг всей картинки перед прогоном (1.5-2.0 помогает на мелком шрифте)",
    )
    # Старые аргументы оставлены для обратной совместимости с pipeline,
    # но в новом подходе они не используются.
    ap.add_argument("--conf-threshold", type=float, default=0.0)
    ap.add_argument("--pad-px", type=int, default=0)
    ap.add_argument("--accept-margin", type=float, default=0.0)

    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input image not found: {args.input}")
    if not os.path.exists(args.ocr_json):
        raise SystemExit(f"OCR JSON not found: {args.ocr_json}")

    data = _load_json(args.ocr_json)
    fast_blocks = data.get("blocks", [])
    if not isinstance(fast_blocks, list):
        raise SystemExit("OCR JSON does not contain a 'blocks' list")

    fast_backup = args.ocr_json.replace(".json", "_fast.json")
    if not os.path.exists(fast_backup):
        _save_json(fast_backup, data)
        print(f"[ocr_heavy] fast backup saved: {fast_backup}")

    img = Image.open(args.input).convert("RGB")

    # Сначала apply upscale (если хочется поднять резолюцию для мелкого шрифта),
    # ПОТОМ ограничиваем длинную сторону max_side, чтобы EasyOCR не задохнулся на CPU.
    # Для огромных BPMN-диаграмм (5000+ px) это критично — без cap'а легко уходим за 60 сек.
    if args.upscale > 1.0 + 1e-6:
        w, h = img.size
        img = img.resize(
            (max(1, int(w * args.upscale)), max(1, int(h * args.upscale))),
            Image.Resampling.LANCZOS,
        )

    max_side = int(os.getenv("OCR_HEAVY_MAX_SIDE", "2500"))
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )
        # Effective upscale изменится — учитываем при reverse-mapping bbox'ов ниже.
        effective_scale = float(args.upscale) * scale
    else:
        effective_scale = float(args.upscale)

    img_array = np.array(img)
    print(
        f"[ocr_heavy] running full EasyOCR sweep: lang={args.lang} "
        f"gpu={args.gpu} paragraph={args.paragraph} upscale={args.upscale} "
        f"size={img_array.shape[1]}x{img_array.shape[0]}"
    )

    new_blocks = _run_easyocr_sweep(img_array, args.lang, args.gpu, args.paragraph)

    # Возвращаем bbox'ы в координаты исходной картинки (учитывает и upscale, и max_side resize).
    if abs(effective_scale - 1.0) > 1e-6:
        inv = 1.0 / effective_scale
        for b in new_blocks:
            x1, y1, x2, y2 = b["bbox_xyxy"]
            x1s, y1s = int(round(x1 * inv)), int(round(y1 * inv))
            x2s, y2s = int(round(x2 * inv)), int(round(y2 * inv))
            b["bbox_xyxy"] = [x1s, y1s, x2s, y2s]
            b["bbox_xywh"] = [x1s, y1s, max(0, x2s - x1s), max(0, y2s - y1s)]

    print(
        f"[ocr_heavy] blocks: fast={len(fast_blocks)} -> easyocr_full={len(new_blocks)}"
    )

    # Полностью заменяем блоки результатом EasyOCR-проходом.
    data["blocks"] = new_blocks
    data["heavy_pass"] = {
        "strategy": "full_easyocr_sweep",
        "lang": args.lang,
        "paragraph": bool(args.paragraph),
        "upscale": float(args.upscale),
        "blocks_replaced": len(new_blocks),
        "fast_block_count": len(fast_blocks),
    }
    # Помечаем что engine изменился, чтобы downstream не путался.
    if isinstance(data.get("engine"), str):
        data["engine"] = data["engine"] + "+easyocr_full"
    else:
        data["engine"] = "easyocr_full"

    _save_json(args.ocr_json, data)

    # Топ-5 для быстрой проверки в логах.
    preview = sorted(new_blocks, key=lambda b: -float(b.get("confidence", 0)))[:5]
    for b in preview:
        text = b.get("text", "")
        conf = b.get("confidence", 0)
        bbox = b.get("bbox_xyxy")
        print(f"[ocr_heavy] sample: text={text!r} conf={conf:.3f} bbox={bbox}")

    print(f"[ocr_heavy] saved: {args.ocr_json}")


if __name__ == "__main__":
    main()
