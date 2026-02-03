from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

_READER_CACHE: Dict[str, Any] = {}


class DetectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class DetectionConfig:
    # EasyOCR language code. Для русских диаграмм: "ru".
    lang: str = "ru"
    gpu: bool = False

    # Оффлайн/онлайн веса
    download_enabled: bool = True
    model_storage_directory: Optional[str] = None

    # Параметры CRAFT (детектор текста)
    min_size: int = 10
    text_threshold: float = 0.55
    low_text: float = 0.25
    link_threshold: float = 0.25
    canvas_size: int = 3840
    mag_ratio: float = 1.2

    # Мерджинг боксов (важно не склеивать в полосы)
    slope_ths: float = 0.10
    ycenter_ths: float = 0.30
    height_ths: float = 0.30
    width_ths: float = 0.12
    add_margin: float = 0.05

    # Пост-фильтры
    max_box_height_ratio: float = 0.25  # отсечь “гигантские” по высоте кадра
    min_box_area: int = 30

    # NMS дедуп
    nms_iou_threshold: float = 0.30


@dataclass(frozen=True)
class TextBox:
    poly: List[List[int]]  # 4 точки
    bbox: List[int]        # [x1,y1,x2,y2]
    kind: str              # "horizontal" | "free"


def detect_text_boxes(bgr: np.ndarray, cfg: Optional[DetectionConfig] = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = DetectionConfig()

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise DetectionError(f"Expected BGR image, got shape: {bgr.shape}")

    h, w = bgr.shape[:2]

    try:
        import easyocr  # type: ignore
    except Exception as e:
        raise DetectionError("EasyOCR is not installed. Install: python -m pip install easyocr") from e

    # EasyOCR ожидает RGB массив
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # detector=True, recognizer=False: only text detection
    reader = _get_or_create_reader(easyocr, cfg)

    det = reader.detect(
        rgb,
        min_size=int(cfg.min_size),
        text_threshold=float(cfg.text_threshold),
        low_text=float(cfg.low_text),
        link_threshold=float(cfg.link_threshold),
        canvas_size=int(cfg.canvas_size),
        mag_ratio=float(cfg.mag_ratio),
        slope_ths=float(cfg.slope_ths),
        ycenter_ths=float(cfg.ycenter_ths),
        height_ths=float(cfg.height_ths),
        width_ths=float(cfg.width_ths),
        add_margin=float(cfg.add_margin),
    )

    if not isinstance(det, (tuple, list)) or len(det) != 2:
        raise DetectionError(f"Unexpected EasyOCR.detect output: {type(det)!r}")

    horizontal_list, free_list = det

    # Важно: easyocr часто возвращает batch-структуру: [boxes_for_img]
    horizontal_list = _unwrap_batch_list(horizontal_list)
    free_list = _unwrap_batch_list(free_list)

    boxes: List[TextBox] = []

    # horizontal_list: [[x_min, x_max, y_min, y_max] ...] (иногда +score)
    if isinstance(horizontal_list, list):
        for item in horizontal_list:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            x_min, x_max, y_min, y_max = [int(round(float(v))) for v in item[:4]]

            x1 = max(0, min(w - 1, x_min))
            x2 = max(0, min(w - 1, x_max))
            y1 = max(0, min(h - 1, y_min))
            y2 = max(0, min(h - 1, y_max))
            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            if area < int(cfg.min_box_area):
                continue
            if (y2 - y1) > int(round(float(cfg.max_box_height_ratio) * float(h))):
                continue

            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            boxes.append(TextBox(poly=poly, bbox=[x1, y1, x2, y2], kind="horizontal"))

    # free_list: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]] ...] (иногда +score)
    if isinstance(free_list, list):
        for item in free_list:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue

            pts: List[List[int]] = []
            ok = True
            for p in item[:4]:
                if not isinstance(p, (list, tuple)) or len(p) != 2:
                    ok = False
                    break
                pts.append([int(round(float(p[0]))), int(round(float(p[1])))])
            if not ok:
                continue

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1 = max(0, min(w - 1, min(xs)))
            x2 = max(0, min(w - 1, max(xs)))
            y1 = max(0, min(h - 1, min(ys)))
            y2 = max(0, min(h - 1, max(ys)))
            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            if area < int(cfg.min_box_area):
                continue
            if (y2 - y1) > int(round(float(cfg.max_box_height_ratio) * float(h))):
                continue

            poly = [[max(0, min(w - 1, p[0])), max(0, min(h - 1, p[1]))] for p in pts]
            boxes.append(TextBox(poly=poly, bbox=[x1, y1, x2, y2], kind="free"))

    boxes = _nms_text_boxes(boxes, float(cfg.nms_iou_threshold))

    return {
        "config": asdict(cfg),
        "boxes": [asdict(b) for b in boxes],
    }


def _get_or_create_reader(easyocr_module: Any, cfg: DetectionConfig):
    key = "|".join(
        [
            str(cfg.lang),
            str(bool(cfg.gpu)),
            str(bool(cfg.download_enabled)),
            str(cfg.model_storage_directory),
        ]
    )
    if key in _READER_CACHE:
        return _READER_CACHE[key]
    reader = easyocr_module.Reader(
        [cfg.lang],
        gpu=bool(cfg.gpu),
        detector=True,
        recognizer=False,
        download_enabled=bool(cfg.download_enabled),
        model_storage_directory=cfg.model_storage_directory,
    )
    _READER_CACHE[key] = reader
    return reader


def draw_text_boxes(bgr: np.ndarray, detection: Dict[str, Any]) -> np.ndarray:
    out = bgr.copy()
    boxes = detection.get("boxes", [])
    for i, b in enumerate(boxes):
        poly = b.get("poly")
        if not isinstance(poly, list) or len(poly) != 4:
            continue
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        bbox = b.get("bbox", [0, 0, 0, 0])
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1 = int(bbox[0]), int(bbox[1])
            cv2.putText(
                out,
                f"txt:{i + 1}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return out


def _unwrap_batch_list(x: Any) -> Any:
    """
    EasyOCR.detect часто возвращает:
      horizontal_list = [ list_for_img ]
      free_list       = [ list_for_img ]
    Нам нужен именно list_for_img.
    """
    if isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
        return x[0]
    return x


def _iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _nms_text_boxes(boxes: List[TextBox], iou_th: float) -> List[TextBox]:
    if not boxes:
        return boxes
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i].bbox[2] - boxes[i].bbox[0]) * (boxes[i].bbox[3] - boxes[i].bbox[1]), reverse=True)

    keep: List[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        filtered: List[int] = []
        for j in order:
            if _iou(boxes[i].bbox, boxes[j].bbox) < float(iou_th):
                filtered.append(j)
        order = filtered

    return [boxes[i] for i in keep]
