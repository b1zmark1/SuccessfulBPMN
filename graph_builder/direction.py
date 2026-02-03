from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class DirectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class DirectionConfig:
    default_direction: str = "LR"
    min_points: int = 2
    ambiguity_threshold: float = 0.12
    node_weight: float = 0.7
    flow_weight: float = 0.3


def infer_process_direction(
    grouped_payload: Dict[str, Any],
    cfg: Optional[DirectionConfig] = None,
) -> Dict[str, Any]:
    """
    Infer global process direction (LR/TB) from grouped detections.
    Returns payload with meta.direction and debug trace.
    """
    if cfg is None:
        cfg = DirectionConfig()

    groups = grouped_payload.get("groups")
    if not isinstance(groups, dict):
        raise DirectionError("grouped_payload must contain 'groups' object")

    process_shapes = groups.get("process_shapes", [])
    flows = groups.get("flows", [])
    if not isinstance(process_shapes, list) or not isinstance(flows, list):
        raise DirectionError("groups.process_shapes and groups.flows must be lists")

    warnings: List[str] = []
    meta = grouped_payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    existing_warnings = meta.get("warnings", [])
    if isinstance(existing_warnings, list):
        warnings.extend([str(x) for x in existing_warnings])

    node_score, node_trace = _node_axis_score(process_shapes)
    flow_score, flow_trace = _flow_axis_score(flows)

    has_nodes = node_trace["count"] >= cfg.min_points
    has_flows = flow_trace["count"] > 0

    if has_nodes and has_flows:
        combined = cfg.node_weight * node_score + cfg.flow_weight * flow_score
    elif has_nodes:
        combined = node_score
    elif has_flows:
        combined = flow_score
    else:
        combined = 0.0
        warnings.append("direction_inference: no usable process shapes/flows, fallback to LR")

    is_ambiguous = abs(combined) < cfg.ambiguity_threshold
    if is_ambiguous:
        direction = cfg.default_direction
        warnings.append("direction_inference: ambiguous geometry, fallback to LR")
    else:
        direction = "LR" if combined >= 0 else "TB"

    confidence = min(1.0, abs(combined))
    if is_ambiguous:
        confidence = min(confidence, 0.5)

    out_meta = dict(meta)
    out_meta["direction"] = direction
    out_meta["direction_confidence"] = confidence
    out_meta["warnings"] = warnings
    out_meta["direction_trace"] = {
        "node_score": node_score,
        "flow_score": flow_score,
        "combined_score": combined,
        "used_nodes": has_nodes,
        "used_flows": has_flows,
        "is_ambiguous": is_ambiguous,
        "config": {
            "default_direction": cfg.default_direction,
            "min_points": cfg.min_points,
            "ambiguity_threshold": cfg.ambiguity_threshold,
            "node_weight": cfg.node_weight,
            "flow_weight": cfg.flow_weight,
        },
        "node_trace": node_trace,
        "flow_trace": flow_trace,
    }

    out = dict(grouped_payload)
    out["meta"] = out_meta
    return out


def _node_axis_score(process_shapes: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    centers = []
    for item in process_shapes:
        c = item.get("center")
        if isinstance(c, list) and len(c) == 2:
            try:
                centers.append((float(c[0]), float(c[1])))
            except Exception:
                continue

    if len(centers) < 2:
        return 0.0, {"count": len(centers), "spread_x": 0.0, "spread_y": 0.0}

    xs = sorted([c[0] for c in centers])
    ys = sorted([c[1] for c in centers])
    spread_x = _robust_span(xs)
    spread_y = _robust_span(ys)

    denom = spread_x + spread_y
    score = 0.0 if denom <= 1e-9 else (spread_x - spread_y) / denom
    return score, {
        "count": len(centers),
        "spread_x": spread_x,
        "spread_y": spread_y,
    }


def _flow_axis_score(flows: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    vals: List[float] = []
    for item in flows:
        bb = item.get("bbox_xyxy")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bb]
        except Exception:
            continue
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        denom = w + h
        if denom <= 1e-9:
            continue
        vals.append((w - h) / denom)

    if not vals:
        return 0.0, {"count": 0, "mean_axis": 0.0}
    mean_axis = sum(vals) / len(vals)
    return mean_axis, {"count": len(vals), "mean_axis": mean_axis}


def _robust_span(sorted_vals: List[float]) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    lo = sorted_vals[int(0.1 * (n - 1))]
    hi = sorted_vals[int(0.9 * (n - 1))]
    return max(0.0, hi - lo)
