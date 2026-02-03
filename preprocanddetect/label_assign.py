from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        return self.w() * self.h()

    def cx(self) -> float:
        return self.x1 + self.w() * 0.5

    def cy(self) -> float:
        return self.y1 + self.h() * 0.5


@dataclass(frozen=True)
class ElementDet:
    det_id: int
    class_name: str
    bbox: BBox
    score: float
    source: str


@dataclass(frozen=True)
class TextBlock:
    block_id: int
    bbox: BBox
    text: str
    confidence: float


@dataclass(frozen=True)
class AssignConfig:
    # node assignment
    min_containment: float = 0.55
    min_node_accept_score: float = 0.85
    min_node_det_score: float = 0.55

    # edge assignment
    min_edge_det_score: float = 0.55
    max_edge_dist_mult: float = 2.2
    min_edge_accept_score: float = 0.55

    # weights
    w_containment: float = 1.6
    w_iou: float = 0.6
    w_center_inside: float = 0.25

    # edge weights
    w_edge_dist: float = 1.0
    w_edge_axis_overlap: float = 0.45
    w_edge_intersection: float = 0.20

    edge_classes: Tuple[str, ...] = ("sequence_flow",)


_EDGE_TRIM_RE = re.compile(r"^[\s\|\)\(\]\[>\-–—_`\\]+|[\s\|\)\(\]\[<\-–—_`\\]+$")
_ONLY_SMALL_GARBAGE_RE = re.compile(r"^[\s\|\)\(\]\[>\-–—_`\\<>]+$")
_CYR_ALNUM_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]")


def _bbox_from_xyxy(xyxy: Any) -> Optional[BBox]:
    if not isinstance(xyxy, list) or len(xyxy) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _intersection_area(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _iou(a: BBox, b: BBox) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = a.area() + b.area() - inter
    return (inter / union) if union > 0 else 0.0


def _rect_distance(a: BBox, b: BBox) -> float:
    dx = 0.0
    if a.x2 < b.x1:
        dx = b.x1 - a.x2
    elif b.x2 < a.x1:
        dx = a.x1 - b.x2

    dy = 0.0
    if a.y2 < b.y1:
        dy = b.y1 - a.y2
    elif b.y2 < a.y1:
        dy = a.y1 - b.y2

    return math.hypot(dx, dy)


def _x_overlap_ratio(a: BBox, b: BBox) -> float:
    inter = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    denom = max(1.0, min(a.w(), b.w()))
    return float(inter / denom)


def _y_overlap_ratio(a: BBox, b: BBox) -> float:
    inter = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    denom = max(1.0, min(a.h(), b.h()))
    return float(inter / denom)


def clean_ocr_text(text: str) -> str:
    t = (text or "").replace("\r", "").replace("\f", "").strip()
    if not t:
        return ""

    lines = [ln.strip() for ln in t.split("\n")]
    lines2: List[str] = []
    for ln in lines:
        ln = _EDGE_TRIM_RE.sub("", ln).strip()
        if not ln:
            continue
        if _ONLY_SMALL_GARBAGE_RE.fullmatch(ln):
            continue
        # типичный мусор: "уу", "о", "о о" — выкидываем если мало букв/цифр
        alnum = len(_CYR_ALNUM_RE.findall(ln))
        if alnum <= 1 and len(ln) <= 3:
            continue
        lines2.append(ln)

    out = "\n".join(lines2).strip()

    # финальный фильтр: если в итоге почти нет букв/цифр — выкидываем
    if len(_CYR_ALNUM_RE.findall(out)) <= 1:
        return ""

    # нормализация пробелов (внутри строк)
    out_lines = []
    for ln in out.split("\n"):
        ln = re.sub(r"[ \t]{2,}", " ", ln).strip()
        out_lines.append(ln)
    return "\n".join(out_lines).strip()


def load_elements_from_ensemble(ensemble_json: Dict[str, Any], cfg: AssignConfig) -> List[ElementDet]:
    dets = ensemble_json.get("detections", [])
    if not isinstance(dets, list):
        raise ValueError("ensemble_json['detections'] must be a list")

    out: List[ElementDet] = []
    next_id = 1
    for d in dets:
        if not isinstance(d, dict):
            continue

        class_name = str(d.get("class_name", "")).strip()
        if not class_name or class_name == "text":
            continue

        bbox = _bbox_from_xyxy(d.get("bbox_xyxy"))
        if bbox is None:
            continue

        try:
            score = float(d.get("score", 0.0))
        except Exception:
            score = 0.0

        out.append(
            ElementDet(
                det_id=next_id,
                class_name=class_name,
                bbox=bbox,
                score=score,
                source=str(d.get("source", "unknown")),
            )
        )
        next_id += 1

    return out


def load_text_blocks_with_ocr(text_blocks_json: Dict[str, Any], ocr_json: Dict[str, Any]) -> List[TextBlock]:
    blocks = text_blocks_json.get("blocks", [])
    if not isinstance(blocks, list):
        raise ValueError("text_blocks_json['blocks'] must be a list")

    ocr_blocks = ocr_json.get("blocks", [])
    if not isinstance(ocr_blocks, list):
        raise ValueError("ocr_json['blocks'] must be a list")
