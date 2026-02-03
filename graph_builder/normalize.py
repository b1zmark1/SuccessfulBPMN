from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class NormalizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class NormalizationConfig:
    min_area: float = 4.0
    dedup_iou_threshold: float = 0.85
    dedup_across_sources: bool = False
    source_priority: Dict[str, int] = field(
        default_factory=lambda: {
            "easyocr": 2,
            "yolox": 1,
        }
    )


@dataclass(frozen=True)
class InternalDetection:
    det_id: str
    original_index: int
    class_name: str
    source: str
    score: float
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    area: float
    dropped: bool = False
    drop_reason: Optional[str] = None
    notes: Tuple[str, ...] = ()


def normalize_ensemble_input(
    payload: Dict[str, Any],
    cfg: Optional[NormalizationConfig] = None,
) -> Dict[str, Any]:
    """
    Parse + sanitize + deduplicate detections from ensemble JSON.
    This function follows the input contract from graph_builder/contracts/README.md.
    """
    if cfg is None:
        cfg = NormalizationConfig()

    image = payload.get("image")
    if not isinstance(image, dict):
        raise NormalizationError("Input must contain object field 'image'.")
    width = _as_positive_int(image.get("width"), "image.width")
    height = _as_positive_int(image.get("height"), "image.height")

    detections = payload.get("detections")
    if not isinstance(detections, list):
        raise NormalizationError("Input must contain array field 'detections'.")

    parsed: List[InternalDetection] = []
    dropped: List[InternalDetection] = []

    for idx, raw in enumerate(detections):
        if not isinstance(raw, dict):
            dropped.append(
                InternalDetection(
                    det_id=f"det_{idx:06d}",
                    original_index=idx,
                    class_name="unknown",
                    source="unknown",
                    score=0.0,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    center=(0.0, 0.0),
                    area=0.0,
                    dropped=True,
                    drop_reason="invalid_detection_object",
                )
            )
            continue

        try:
            det = _parse_and_sanitize_one(raw, idx, width, height, cfg)
            parsed.append(det)
        except NormalizationError as e:
            dropped.append(
                InternalDetection(
                    det_id=f"det_{idx:06d}",
                    original_index=idx,
                    class_name=_normalize_class_name(raw.get("class_name")),
                    source=_normalize_source(raw.get("source")),
                    score=_safe_score(raw.get("score")),
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    center=(0.0, 0.0),
                    area=0.0,
                    dropped=True,
                    drop_reason=str(e),
                )
            )

    kept, dedup_dropped = _deduplicate(parsed, cfg)
    dropped.extend(dedup_dropped)

    kept_sorted = sorted(kept, key=lambda d: d.original_index)
    dropped_sorted = sorted(dropped, key=lambda d: d.original_index)

    return {
        "image": {
            "file_name": image.get("file_name"),
            "path": image.get("path"),
            "width": width,
            "height": height,
        },
        "meta": payload.get("meta", {}),
        "detections_normalized": [_det_to_dict(d) for d in kept_sorted],
        "detections_dropped": [_det_to_dict(d) for d in dropped_sorted],
        "stats": {
            "total_input": len(detections),
            "kept": len(kept_sorted),
            "dropped": len(dropped_sorted),
        },
    }


def _parse_and_sanitize_one(
    raw: Dict[str, Any],
    idx: int,
    width: int,
    height: int,
    cfg: NormalizationConfig,
) -> InternalDetection:
    class_name = _normalize_class_name(raw.get("class_name"))
    if not class_name:
        raise NormalizationError("missing_class_name")

    source = _normalize_source(raw.get("source"))
    if not source:
        raise NormalizationError("missing_source")

    bbox_raw = raw.get("bbox_xyxy")
    bbox = _parse_bbox_xyxy(bbox_raw)
    clipped_bbox, was_clipped = _clip_bbox_xyxy(bbox, width, height)

    x1, y1, x2, y2 = clipped_bbox
    if x2 <= x1 or y2 <= y1:
        raise NormalizationError("invalid_or_empty_bbox")

    area = (x2 - x1) * (y2 - y1)
    if area < cfg.min_area:
        raise NormalizationError("bbox_area_below_min")

    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    score = _safe_score(raw.get("score"))
    notes: List[str] = []
    if was_clipped:
        notes.append("bbox_clipped_to_image")

    return InternalDetection(
        det_id=f"det_{idx:06d}",
        original_index=idx,
        class_name=class_name,
        source=source,
        score=score,
        bbox=(x1, y1, x2, y2),
        center=center,
        area=area,
        notes=tuple(notes),
    )


def _deduplicate(
    detections: List[InternalDetection],
    cfg: NormalizationConfig,
) -> Tuple[List[InternalDetection], List[InternalDetection]]:
    if not detections:
        return [], []

    # deterministic ranking: stronger source -> higher score -> larger area -> earlier input
    ranked = sorted(
        detections,
        key=lambda d: (
            -int(cfg.source_priority.get(d.source, 0)),
            -d.score,
            -d.area,
            d.original_index,
        ),
    )

    kept: List[InternalDetection] = []
    dropped: List[InternalDetection] = []

    for cand in ranked:
        duplicate_of: Optional[str] = None
        for prev in kept:
            if cand.class_name != prev.class_name:
                continue
            if (not cfg.dedup_across_sources) and cand.source != prev.source:
                continue
            if _iou(cand.bbox, prev.bbox) >= cfg.dedup_iou_threshold:
                duplicate_of = prev.det_id
                break

        if duplicate_of is None:
            kept.append(cand)
        else:
            dropped.append(
                InternalDetection(
                    det_id=cand.det_id,
                    original_index=cand.original_index,
                    class_name=cand.class_name,
                    source=cand.source,
                    score=cand.score,
                    bbox=cand.bbox,
                    center=cand.center,
                    area=cand.area,
                    dropped=True,
                    drop_reason=f"duplicate_of:{duplicate_of}",
                    notes=cand.notes,
                )
            )

    return kept, dropped


def _det_to_dict(det: InternalDetection) -> Dict[str, Any]:
    return {
        "det_id": det.det_id,
        "original_index": det.original_index,
        "class_name": det.class_name,
        "source": det.source,
        "score": det.score,
        "bbox_xyxy": [det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]],
        "center": [det.center[0], det.center[1]],
        "area": det.area,
        "dropped": det.dropped,
        "drop_reason": det.drop_reason,
        "notes": list(det.notes),
    }


def _as_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise NormalizationError(f"{field_name}_must_be_positive_int")
    return value


def _parse_bbox_xyxy(value: Any) -> Tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise NormalizationError("bbox_xyxy_must_be_list_of_4")
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except Exception as e:
        raise NormalizationError("bbox_xyxy_contains_non_numeric") from e
    return x1, y1, x2, y2


def _clip_bbox_xyxy(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[Tuple[float, float, float, float], bool]:
    x1, y1, x2, y2 = bbox
    cx1 = min(max(0.0, x1), float(width - 1))
    cy1 = min(max(0.0, y1), float(height - 1))
    cx2 = min(max(0.0, x2), float(width - 1))
    cy2 = min(max(0.0, y2), float(height - 1))
    clipped = (cx1, cy1, cx2, cy2)
    return clipped, clipped != bbox


def _normalize_class_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _normalize_source(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _safe_score(value: Any) -> float:
    try:
        s = float(value)
    except Exception:
        return 0.0
    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    return s


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union
