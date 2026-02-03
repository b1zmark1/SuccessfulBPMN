from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class TextGroupError(RuntimeError):
    pass


@dataclass(frozen=True)
class GroupConfig:
    # Разрыв между словами в одной строке, после которого начинаем новый сегмент строки.
    # Это ключевой параметр против "lane-wide" склеивания.
    gap_mult: float = 1.8

    # Допуск по центру Y для принадлежности к одной строке.
    y_tol_mult: float = 0.6

    # Минимальный вертикальный overlap, чтобы слово попало в строку.
    min_y_overlap: float = 0.20

    # Объединение сегментов в многострочный блок
    line_gap_mult: float = 1.25
    min_x_overlap_for_block: float = 0.18

    # Ограничения, чтобы блоки никогда не становились "на весь lane"
    max_block_w_ratio: float = 0.55
    max_block_h_ratio: float = 0.35

    # Фильтры
    min_box_area: int = 25


@dataclass
class WordBox:
    box_id: int
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def w(self) -> int:
        return int(max(0, self.x2 - self.x1))

    @property
    def h(self) -> int:
        return int(max(0, self.y2 - self.y1))

    def area(self) -> int:
        return int(self.w * self.h)

    def cy(self) -> float:
        return float(self.y1 + self.h / 2.0)


@dataclass
class LineCluster:
    # временная "строка" до сегментации по X-gap
    word_ids: List[int]
    bbox: Tuple[int, int, int, int]
    cy: float


@dataclass
class LineSegment:
    seg_id: int
    word_ids: List[int]
    bbox: Tuple[int, int, int, int]


@dataclass
class TextBlock:
    block_id: int
    seg_ids: List[int]
    word_ids: List[int]
    bbox: Tuple[int, int, int, int]


def build_text_blocks(
    text_boxes_json: Dict[str, Any],
    image_w: int,
    image_h: int,
    cfg: Optional[GroupConfig] = None,
) -> Dict[str, Any]:
    """
    Вход: JSON из detect_text_boxes() -> {"boxes":[{"bbox":[x1,y1,x2,y2], ...}, ...]}
    Выход:
      {
        "meta": {...},
        "segments": [{seg_id, word_ids, bbox}],
        "blocks":   [{block_id, seg_ids, word_ids, bbox}]
      }
    """
    if cfg is None:
        cfg = GroupConfig()

    boxes_raw = text_boxes_json.get("boxes")
    if not isinstance(boxes_raw, list):
        raise TextGroupError("text_boxes_json['boxes'] must be a list")

    words: List[WordBox] = []
    for i, b in enumerate(boxes_raw, start=1):
        if not isinstance(b, dict):
            continue

        kind = b.get("kind", "horizontal")
        if kind != "horizontal":
            continue

        bb = b.get("bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue

        x1, y1, x2, y2 = [int(v) for v in bb]
        x1 = max(0, min(image_w - 1, x1))
        x2 = max(0, min(image_w - 1, x2))
        y1 = max(0, min(image_h - 1, y1))
        y2 = max(0, min(image_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        wb = WordBox(box_id=i, bbox=(x1, y1, x2, y2))
        if wb.area() < int(cfg.min_box_area):
            continue
        words.append(wb)

    if not words:
        return {
            "meta": {"num_words": 0, "num_line_clusters": 0, "num_segments": 0, "num_blocks": 0},
            "segments": [],
            "blocks": [],
        }

    median_h = _median_height(words)
    if median_h <= 0:
        median_h = 12

    # 1) Сначала слова -> кластеры строк по Y
    clusters = _group_words_to_line_clusters(words, median_h=median_h, cfg=cfg)

    # 2) Внутри каждой строки режем на сегменты по большим X-gap
    segments = _split_clusters_into_segments(clusters, words_by_id={w.box_id: w for w in words}, median_h=median_h, cfg=cfg)

    # 3) Сегменты -> блоки (многострочные подписи), с лимитами на размер
    blocks = _group_segments_to_blocks(segments, median_h=median_h, image_w=image_w, image_h=image_h, cfg=cfg)

    segments_out = [{"seg_id": int(s.seg_id), "word_ids": [int(x) for x in s.word_ids], "bbox": list(s.bbox)} for s in segments]
    blocks_out = [{"block_id": int(b.block_id), "seg_ids": [int(x) for x in b.seg_ids], "word_ids": [int(x) for x in b.word_ids], "bbox": list(b.bbox)} for b in blocks]

    return {
        "meta": {
            "median_word_height": int(median_h),
            "num_words": int(len(words)),
            "num_line_clusters": int(len(clusters)),
            "num_segments": int(len(segments)),
            "num_blocks": int(len(blocks)),
        },
        "segments": segments_out,
        "blocks": blocks_out,
    }


def draw_text_blocks(bgr: np.ndarray, blocks_json: Dict[str, Any]) -> np.ndarray:
    out = bgr.copy()
    blocks = blocks_json.get("blocks", [])
    if not isinstance(blocks, list):
        return out

    for b in blocks:
        if not isinstance(b, dict):
            continue
        bb = b.get("bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        x1, y1, x2, y2 = [int(v) for v in bb]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 180, 0), 2)
        bid = int(b.get("block_id", 0))
        cv2.putText(
            out,
            f"blk:{bid}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def _median_height(words: List[WordBox]) -> int:
    hs = [w.h for w in words if w.h > 0]
    if not hs:
        return 12
    return int(np.median(np.array(hs, dtype=np.int32)))


def _merge_bbox(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _y_overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ay1, ay2 = a[1], a[3]
    by1, by2 = b[1], b[3]
    inter = max(0, min(ay2, by2) - max(ay1, by1))
    denom = float(max(1, min(ay2 - ay1, by2 - by1)))
    return float(inter) / denom


def _x_overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ax2 = a[0], a[2]
    bx1, bx2 = b[0], b[2]
    inter = max(0, min(ax2, bx2) - max(ax1, bx1))
    denom = float(max(1, min(ax2 - ax1, bx2 - bx1)))
    return float(inter) / denom


def _group_words_to_line_clusters(words: List[WordBox], median_h: int, cfg: GroupConfig) -> List[LineCluster]:
    words_sorted = sorted(words, key=lambda w: (w.cy(), w.x1))
    y_tol = float(cfg.y_tol_mult) * float(max(1, median_h))

    clusters: List[LineCluster] = []

    for w in words_sorted:
        best_i = -1
        best_d = 1e9

        for i, c in enumerate(clusters):
            d = abs(w.cy() - c.cy)
            if d > y_tol:
                continue
            if d < best_d and _y_overlap_ratio(w.bbox, c.bbox) >= float(cfg.min_y_overlap):
                best_d = d
                best_i = i

        if best_i < 0:
            clusters.append(LineCluster(word_ids=[w.box_id], bbox=w.bbox, cy=w.cy()))
        else:
            c = clusters[best_i]
            c.word_ids.append(w.box_id)
            c.bbox = _merge_bbox(c.bbox, w.bbox)
            c.cy = float(c.bbox[1] + (c.bbox[3] - c.bbox[1]) / 2.0)

    clusters.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
    return clusters


def _split_clusters_into_segments(
    clusters: List[LineCluster],
    words_by_id: Dict[int, WordBox],
    median_h: int,
    cfg: GroupConfig,
) -> List[LineSegment]:
    gap_px = float(cfg.gap_mult) * float(max(1, median_h))

    segments: List[LineSegment] = []
    seg_id = 1

    for c in clusters:
        ws = [words_by_id[i] for i in c.word_ids if i in words_by_id]
        if not ws:
            continue

        ws.sort(key=lambda w: w.x1)

        cur_ids: List[int] = [ws[0].box_id]
        cur_box = ws[0].bbox

        for w in ws[1:]:
            prev = words_by_id[cur_ids[-1]]
            gap = float(w.x1 - prev.x2)

            # КЛЮЧЕВОЕ: если большой разрыв — это не один текстовый сегмент
            if gap > gap_px:
                segments.append(LineSegment(seg_id=seg_id, word_ids=cur_ids, bbox=cur_box))
                seg_id += 1
                cur_ids = [w.box_id]
                cur_box = w.bbox
            else:
                cur_ids.append(w.box_id)
                cur_box = _merge_bbox(cur_box, w.bbox)

        segments.append(LineSegment(seg_id=seg_id, word_ids=cur_ids, bbox=cur_box))
        seg_id += 1

    segments.sort(key=lambda s: (s.bbox[1], s.bbox[0]))
    # перенумерация по порядку чтения
    for i, s in enumerate(segments, start=1):
        s.seg_id = i

    return segments


def _group_segments_to_blocks(
    segments: List[LineSegment],
    median_h: int,
    image_w: int,
    image_h: int,
    cfg: GroupConfig,
) -> List[TextBlock]:
    if not segments:
        return []

    line_gap_max = float(cfg.line_gap_mult) * float(max(1, median_h))
    max_w = int(round(float(cfg.max_block_w_ratio) * float(image_w)))
    max_h = int(round(float(cfg.max_block_h_ratio) * float(image_h)))

    blocks: List[TextBlock] = []
    next_id = 1

    for s in segments:
        placed = False

        # ищем лучший существующий блок, куда можно добавить этот сегмент
        best_idx = -1
        best_score = -1.0

        for i, b in enumerate(blocks):
            vgap = float(s.bbox[1] - b.bbox[3])
            # допускаем маленькое отрицательное из-за bbox шумов
            if vgap < -0.35 * float(max(1, median_h)):
                continue
            if vgap > line_gap_max:
                continue

            xov = _x_overlap_ratio(s.bbox, b.bbox)
            if xov < float(cfg.min_x_overlap_for_block):
                continue

            merged = _merge_bbox(b.bbox, s.bbox)
            mw = merged[2] - merged[0]
            mh = merged[3] - merged[1]
            if mw > max_w or mh > max_h:
                continue

            score = float(xov) - 0.01 * max(0.0, vgap)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            b = blocks[best_idx]
            b.seg_ids.append(s.seg_id)
            b.word_ids.extend(s.word_ids)
            b.bbox = _merge_bbox(b.bbox, s.bbox)
            placed = True

        if not placed:
            blocks.append(
                TextBlock(
                    block_id=next_id,
                    seg_ids=[s.seg_id],
                    word_ids=list(s.word_ids),
                    bbox=s.bbox,
                )
            )
            next_id += 1

    # нормализуем word_ids
    for b in blocks:
        b.word_ids = sorted(set(b.word_ids))

    blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    for i, b in enumerate(blocks, start=1):
        b.block_id = i

    return blocks
