"""
Containment-based aggregation OCR-текста в YOLOX-объекты.

Заменяет label_res.py (который делал 1-to-1 матчинг и терял текст).
Логика проще и точнее:
  - Для каждой YOLOX-shape ноды собираем ВСЕ OCR-блоки, чьи центры внутри bbox этого shape.
  - Сортируем в reading order (top-to-bottom, left-to-right).
  - Склеиваем через пробел в один полный label.

Это решает основную проблему фрагментации: "Поиск" + "доступных" + "поездов"
теперь становится единым "Поиск доступных поездов" в одной task-ноде.

Неприсвоенные OCR-блоки добавляются как `class_name="text"` детекции, чтобы:
  - graph_builder lane_detection мог использовать левые подписи как имена ролей
  - не потерять контекст (всё что вне shapes — text-аннотации/lane labels)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# Регулярки для отсева мусорных OCR-токенов: одиночные символы, чистые числа,
# короткие "слова" из цифр + пунктуации (типа "1 8 5"), одиночные пунктуационные знаки.
# Эти "tokens" появляются когда EasyOCR ошибочно распознаёт стрелки/линии/чёрточки.
_JUNK_ONLY_NON_LETTERS = re.compile(r"^[\d\W_]+$", re.UNICODE)
_HAS_RUS_OR_LAT_LETTER = re.compile(r"[A-Za-zА-Яа-яЁё]")


def _is_junk_text(text: str, min_letters: int = 2) -> bool:
    """True если текст похож на мусор (без букв или 1 буква + цифры/пунктуация)."""
    if not text:
        return True
    t = text.strip()
    if not t:
        return True
    # одиночные символы
    if len(t) == 1:
        return True
    # совсем нет букв (только цифры/пунктуация)
    if _JUNK_ONLY_NON_LETTERS.match(t):
        return True
    # букв меньше min_letters — скорее всего фрагмент стрелки или мусор
    letters = _HAS_RUS_OR_LAT_LETTER.findall(t)
    if len(letters) < min_letters:
        return True
    return False


NODE_CLASSES: Set[str] = {
    "start_event",
    "intermediate_event",
    "end_event",
    "task",
    "gateway_exclusive",
    "gateway_parallel",
    "gateway_inclusive",
    "subprocess",
    "pool",
    "lane",
    "data_object",
    "text_annotation",
}

EDGE_CLASSES: Set[str] = {"sequence_flow"}


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _as_bbox(v: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(x) for x in v]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _intersection_area(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _area(b: Tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _point_in_bbox(
    p: Tuple[float, float],
    b: Tuple[float, float, float, float],
) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--ocr", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.5,
        help="Fallback: если центр блока вне shape, но overlap >= этой доли площади блока — всё равно включаем",
    )
    ap.add_argument(
        "--min-text-conf",
        type=float,
        default=0.3,
        help="Минимальная confidence OCR-блока для включения в результат",
    )
    ap.add_argument(
        "--min-letters",
        type=int,
        default=2,
        help="Минимум букв (рус/лат) в OCR-блоке. Отсекает мусор типа '1 8 5', одиночные '1', 'Е'",
    )
    args = ap.parse_args()

    ensemble = _read_json(args.ensemble)
    ocr = _read_json(args.ocr)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detections = ensemble.get("detections")
    if not isinstance(detections, list):
        raise SystemExit("ensemble has no detections list")

    blocks = ocr.get("blocks")
    if not isinstance(blocks, list):
        blocks = []

    # Нормализуем блоки: фильтр по conf + по содержимому, парсим bbox.
    norm_blocks: List[Dict[str, Any]] = []
    junk_filtered = 0
    for b in blocks:
        if not isinstance(b, dict):
            continue
        text = str(b.get("text") or "").strip()
        if not text:
            continue
        if _is_junk_text(text, min_letters=args.min_letters):
            junk_filtered += 1
            continue
        try:
            conf = float(b.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        if conf < args.min_text_conf:
            continue
        bbox = _as_bbox(b.get("bbox_xyxy") or b.get("bbox"))
        if bbox is None:
            continue
        norm_blocks.append(
            {
                "block_id": b.get("block_id"),
                "bbox": bbox,
                "text": text,
                "conf": conf,
            }
        )

    used_block_ids: Set[Any] = set()
    detections_out: List[Dict[str, Any]] = []

    stats = {
        "shapes_total": 0,
        "shapes_labeled": 0,
        "shapes_empty": 0,
        "blocks_total": len(norm_blocks),
        "blocks_used": 0,
        "blocks_unused": 0,
    }

    for det in detections:
        if not isinstance(det, dict):
            continue
        cls = str(det.get("class_name") or "").strip().lower()
        det_bbox = _as_bbox(det.get("bbox_xyxy") or det.get("bbox"))

        new_det = dict(det)

        if cls in EDGE_CLASSES:
            # sequence_flow — текст не агрегируем тут (мог бы — но это отдельная задача)
            detections_out.append(new_det)
            continue

        if cls not in NODE_CLASSES or det_bbox is None:
            detections_out.append(new_det)
            continue

        stats["shapes_total"] += 1

        inside_blocks: List[Dict[str, Any]] = []
        for b in norm_blocks:
            bcenter = _center(b["bbox"])
            if _point_in_bbox(bcenter, det_bbox):
                inside_blocks.append(b)
                continue
            # Fallback: блок в основном внутри shape
            ix = _intersection_area(b["bbox"], det_bbox)
            ba = _area(b["bbox"])
            if ba > 0 and ix / ba >= args.min_overlap_ratio:
                inside_blocks.append(b)

        if inside_blocks:
            # Reading order: сначала сверху вниз, затем слева направо
            inside_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
            text = " ".join(b["text"] for b in inside_blocks).strip()
            confs = [b["conf"] for b in inside_blocks]
            avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
            block_ids = [b["block_id"] for b in inside_blocks if b["block_id"] is not None]

            new_det["text"] = text
            new_det["text_conf"] = avg_conf
            new_det["text_block_ids"] = block_ids
            new_det["match_score"] = 1.0  # совместимость со схемой
            stats["shapes_labeled"] += 1
            for b in inside_blocks:
                used_block_ids.add(id(b))  # уникальность по объекту
        else:
            new_det["text"] = None
            new_det["text_conf"] = 0.0
            new_det["text_block_ids"] = []
            stats["shapes_empty"] += 1

        detections_out.append(new_det)

    # Неиспользованные OCR-блоки -> отдельные text-детекции (нужны для lane_detection)
    appended = 0
    for b in norm_blocks:
        if id(b) in used_block_ids:
            continue
        x1, y1, x2, y2 = b["bbox"]
        detections_out.append(
            {
                "class_name": "text",
                "score": b["conf"],
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "source": "ocr",
                "text": b["text"],
                "text_block_ids": ([int(b["block_id"])] if isinstance(b["block_id"], int) else []),
                "text_conf": b["conf"],
            }
        )
        appended += 1

    stats["blocks_used"] = len(used_block_ids)
    stats["blocks_unused"] = appended

    merged = dict(ensemble)
    meta = dict(merged.get("meta") or {})
    meta["text_merge_source"] = "label_aggregate:containment"
    meta["text_nodes_labeled"] = int(stats["shapes_labeled"])
    meta["appended_text_detections"] = int(appended)
    merged["meta"] = meta
    merged["detections"] = detections_out

    input_stem = Path(args.ensemble).stem.replace("_ensemble", "")
    out_path = out_dir / f"{input_stem}_ensemble_merged_labeled.json"
    _write_json(str(out_path), merged)

    print(
        f"[label_agg] shapes={stats['shapes_total']} labeled={stats['shapes_labeled']} "
        f"empty={stats['shapes_empty']} blocks={stats['blocks_total']} "
        f"used={stats['blocks_used']} unused_appended={stats['blocks_unused']} "
        f"junk_filtered={junk_filtered}",
        flush=True,
    )
    print(f"[label_agg] saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
