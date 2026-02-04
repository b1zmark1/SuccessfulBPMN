from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


class NodeBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class NodeBuildConfig:
    role_start_classes: Set[str] = field(default_factory=lambda: {"start_event"})
    role_end_classes: Set[str] = field(default_factory=lambda: {"end_event"})
    role_event_classes: Set[str] = field(default_factory=lambda: {"intermediate_event"})  # ДОБАВЛЕНО
    role_decision_classes: Set[str] = field(
        default_factory=lambda: {
            "gateway_exclusive",
            "gateway_parallel",
            "gateway_inclusive",
        }
    )
    role_action_classes: Set[str] = field(
        default_factory=lambda: {
            "task",
            "data_object",
        }
    )
    role_annotation_classes: Set[str] = field(default_factory=lambda: {"text_annotation"})  # ДОБАВЛЕНО


def build_graph_nodes(
    payload: Dict[str, Any],
    cfg: Optional[NodeBuildConfig] = None,
) -> Dict[str, Any]:
    """
    Build graph nodes from grouped detections.
    This step creates only nodes and keeps container_id as null (hierarchy in next step).
    """
    if cfg is None:
        cfg = NodeBuildConfig()

    groups = payload.get("groups")
    if not isinstance(groups, dict):
        raise NodeBuildError("payload must contain 'groups' object")

    nodes: List[Dict[str, Any]] = []
    warnings: List[str] = []

    process_shapes = _safe_list(groups.get("process_shapes"))
    flows = _safe_list(groups.get("flows"))
    containers = _safe_list(groups.get("containers"))
    texts = _safe_list(groups.get("texts"))
    annotations = _safe_list(groups.get("annotations"))  # ДОБАВЛЕНО
    unknown = _safe_list(groups.get("unknown"))

    nodes.extend(_build_nodes_from_items(process_shapes, "shape", cfg))
    nodes.extend(_build_nodes_from_items(flows, "flow", cfg))
    nodes.extend(_build_nodes_from_items(containers, "container", cfg))
    nodes.extend(_build_nodes_from_items(texts, "text", cfg))
    # Аннотации оставляем как shape, но с ролью annotation (и потом исключим их из edge-candidates)
    nodes.extend(_build_nodes_from_items(annotations, "shape", cfg))
    nodes.extend(_build_nodes_from_items(unknown, "shape", cfg, force_role="unknown"))

    nodes.sort(key=lambda n: (n.get("original_index", 10**12), n["id"]))

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    existing_warnings = meta.get("warnings", [])
    if isinstance(existing_warnings, list):
        warnings.extend([str(x) for x in existing_warnings])

    if not nodes:
        warnings.append("node_builder: no nodes created from grouped detections")

    out_meta = dict(meta)
    out_meta["warnings"] = warnings

    out = dict(payload)
    out["meta"] = out_meta
    out["nodes"] = nodes
    out["node_stats"] = {
        "total": len(nodes),
        "shape": _count_by_type(nodes, "shape"),
        "flow": _count_by_type(nodes, "flow"),
        "container": _count_by_type(nodes, "container"),
        "text": _count_by_type(nodes, "text"),
    }
    return out


def _build_nodes_from_items(
    items: List[Dict[str, Any]],
    node_type: str,
    cfg: NodeBuildConfig,
    force_role: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in items:
        bb = item.get("bbox_xyxy")
        center = item.get("center")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        if not (isinstance(center, list) and len(center) == 2):
            continue

        class_name = str(item.get("class_name", "")).strip().lower()
        role = force_role if force_role is not None else _infer_role(class_name, node_type, cfg)

        det_id = item.get("det_id")
        if not isinstance(det_id, str) or not det_id:
            continue

        out.append(
            {
                "id": f"node_{det_id}",
                "type": node_type,
                "bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                "center": [float(center[0]), float(center[1])],
                "role": role,
                "container_id": None,
                "text": _normalize_node_text(item.get("text")),
                "class_name": class_name,
                "source": item.get("source"),
                "score": float(item.get("score", 0.0)),
                "det_id": det_id,
                "ocr_confidence": item.get("ocr_confidence"),
                "ocr_block_ids": item.get("ocr_block_ids", []),
                "match_score": item.get("match_score"),
                "original_index": int(item.get("original_index", 0)),
            }
        )
    return out


def _infer_role(class_name: str, node_type: str, cfg: NodeBuildConfig) -> str:
    if node_type in {"text", "container", "flow"}:
        return "unknown"
    if class_name in cfg.role_annotation_classes:
        return "annotation"
    if class_name in cfg.role_start_classes:
        return "start"
    if class_name in cfg.role_end_classes:
        return "end"
    if class_name in cfg.role_event_classes:
        return "event"
    if class_name in cfg.role_decision_classes:
        return "decision"
    if class_name in cfg.role_action_classes:
        return "action"
    return "unknown"


def _safe_list(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    return []


def _count_by_type(nodes: List[Dict[str, Any]], node_type: str) -> int:
    return sum(1 for n in nodes if n.get("type") == node_type)


def _normalize_node_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    t = value.strip()
    return t or None
