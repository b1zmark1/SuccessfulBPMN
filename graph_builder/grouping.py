from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


class GroupingError(RuntimeError):
    pass


@dataclass(frozen=True)
class GroupingConfig:
    process_shape_classes: Set[str] = field(
        default_factory=lambda: {
            "start_event",
            "intermediate_event",
            "end_event",
            "task",
            "gateway_exclusive",
            "gateway_parallel",
            "gateway_inclusive",
            "data_object",
        }
    )
    flow_classes: Set[str] = field(
        default_factory=lambda: {
            "sequence_flow",
            "text_annotation",
        }
    )
    container_classes: Set[str] = field(
        default_factory=lambda: {
            "pool",
            "lane",
            "subprocess",
        }
    )
    text_classes: Set[str] = field(
        default_factory=lambda: {
            "text",
        }
    )
    source_reliability: Dict[str, float] = field(
        default_factory=lambda: {
            "yolox": 0.85,
            "easyocr": 0.9,
        }
    )
    class_reliability: Dict[str, float] = field(
        default_factory=lambda: {
            "lane": 0.45,
            "pool": 0.5,
            "gateway_inclusive": 0.5,
            "sequence_flow": 0.55,
        }
    )
    default_reliability: float = 0.7


def group_normalized_detections(
    normalized_payload: Dict[str, Any],
    cfg: Optional[GroupingConfig] = None,
) -> Dict[str, Any]:
    """
    Build primitive groups from normalized detections.
    Input is expected to be output of normalize_ensemble_input().
    """
    if cfg is None:
        cfg = GroupingConfig()

    normalized = normalized_payload.get("detections_normalized")
    if not isinstance(normalized, list):
        raise GroupingError("normalized_payload must contain 'detections_normalized' list")

    grouped = {
        "process_shapes": [],
        "flows": [],
        "containers": [],
        "texts": [],
        "unknown": [],
    }

    for det in normalized:
        if not isinstance(det, dict):
            continue

        class_name = str(det.get("class_name", "")).strip().lower()
        source = str(det.get("source", "")).strip().lower()
        item = _to_group_item(det, class_name, source, cfg)

        if class_name in cfg.process_shape_classes:
            grouped["process_shapes"].append(item)
        elif class_name in cfg.flow_classes:
            grouped["flows"].append(item)
        elif class_name in cfg.container_classes:
            grouped["containers"].append(item)
        elif class_name in cfg.text_classes or source == "easyocr":
            grouped["texts"].append(item)
        else:
            grouped["unknown"].append(item)

    return {
        "image": normalized_payload.get("image", {}),
        "meta": normalized_payload.get("meta", {}),
        "groups": grouped,
        "group_stats": {k: len(v) for k, v in grouped.items()},
    }


def _to_group_item(
    det: Dict[str, Any],
    class_name: str,
    source: str,
    cfg: GroupingConfig,
) -> Dict[str, Any]:
    source_rel = float(cfg.source_reliability.get(source, cfg.default_reliability))
    class_rel = float(cfg.class_reliability.get(class_name, cfg.default_reliability))
    hint_reliability = max(0.0, min(1.0, source_rel * class_rel))

    return {
        "det_id": det.get("det_id"),
        "original_index": det.get("original_index"),
        "class_name": class_name,
        "source": source,
        "score": det.get("score", 0.0),
        "bbox_xyxy": det.get("bbox_xyxy"),
        "center": det.get("center"),
        "area": det.get("area"),
        "text": det.get("text"),
        "ocr_confidence": det.get("ocr_confidence"),
        "ocr_block_ids": det.get("ocr_block_ids", []),
        "match_score": det.get("match_score"),
        "class_hint_reliability": hint_reliability,
        "provenance": {
            "source": source,
            "original_index": det.get("original_index"),
            "det_id": det.get("det_id"),
            "notes": det.get("notes", []),
        },
    }
