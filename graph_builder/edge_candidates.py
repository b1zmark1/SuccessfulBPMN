from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math


class EdgeCandidateError(RuntimeError):
    pass


@dataclass(frozen=True)
class EdgeCandidateConfig:
    max_neighbors_per_node: int = 3
    max_distance_factor: float = 3.0
    min_candidate_score: float = 0.15
    direction_axis_weight: float = 0.45
    distance_weight: float = 0.35
    flow_support_weight: float = 0.20
    min_flow_support_iou: float = 0.05


def build_edge_candidates(
    payload: Dict[str, Any],
    cfg: Optional[EdgeCandidateConfig] = None,
) -> Dict[str, Any]:
    """
    Build directed edge candidates from geometry and flow hints.
    This step only creates candidate edges (not final pruning).
    """
    if cfg is None:
        cfg = EdgeCandidateConfig()

    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise EdgeCandidateError("payload must contain 'nodes' list")

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    direction = str(meta.get("direction", "LR")).upper()
    if direction not in {"LR", "TB"}:
        direction = "LR"

    shape_nodes = [n for n in nodes if isinstance(n, dict) and n.get("type") == "shape"]
    flow_nodes = [n for n in nodes if isinstance(n, dict) and n.get("type") == "flow"]

    warnings: List[str] = []
    existing_warnings = meta.get("warnings", [])
    if isinstance(existing_warnings, list):
        warnings.extend([str(x) for x in existing_warnings])

    if len(shape_nodes) < 2:
        warnings.append("edge_candidates: not enough shape nodes for candidate generation")

    candidates: List[Dict[str, Any]] = []
    avg_diag = _average_node_diagonal(shape_nodes)
    max_dist = cfg.max_distance_factor * max(avg_diag, 1.0)

    for src in shape_nodes:
        src_id = src.get("id")
        src_center = _center(src.get("center"))
        src_box = _bbox(src.get("bbox"))
        if not (isinstance(src_id, str) and src_center and src_box):
            continue

        per_src: List[Tuple[float, Dict[str, Any]]] = []
        for dst in shape_nodes:
            dst_id = dst.get("id")
            if not isinstance(dst_id, str) or dst_id == src_id:
                continue
            dst_center = _center(dst.get("center"))
            dst_box = _bbox(dst.get("bbox"))
            if not (dst_center and dst_box):
                continue

            dir_ok, axis_score = _direction_axis_score(src_center, dst_center, direction)
            if not dir_ok:
                continue

            eu = _euclidean(src_center, dst_center)
            if eu > max_dist:
                continue
            dist_score = max(0.0, 1.0 - (eu / max_dist))

            flow_support = _flow_support(src_box, dst_box, flow_nodes, cfg.min_flow_support_iou)

            score = (
                cfg.direction_axis_weight * axis_score
                + cfg.distance_weight * dist_score
                + cfg.flow_support_weight * flow_support
            )
            if score < cfg.min_candidate_score:
                continue

            cand = {
                "from": src_id,
                "to": dst_id,
                "score": score,
                "features": {
                    "axis_score": axis_score,
                    "distance_score": dist_score,
                    "flow_support": flow_support,
                    "distance": eu,
                },
                "type_hint": "sequential" if flow_support > 0.0 else "unknown",
            }
            per_src.append((score, cand))

        per_src.sort(key=lambda x: x[0], reverse=True)
        for _, cand in per_src[: cfg.max_neighbors_per_node]:
            candidates.append(cand)

    # deterministic order for debug/reproducibility
    candidates.sort(key=lambda c: (-float(c["score"]), str(c["from"]), str(c["to"])))

    out_meta = dict(meta)
    out_meta["warnings"] = warnings

    out = dict(payload)
    out["meta"] = out_meta
    out["edge_candidates"] = candidates
    out["edge_candidate_stats"] = {
        "shape_nodes": len(shape_nodes),
        "flow_nodes": len(flow_nodes),
        "candidates_total": len(candidates),
        "direction": direction,
        "max_distance": max_dist,
    }
    return out


def _direction_axis_score(
    src: Tuple[float, float],
    dst: Tuple[float, float],
    direction: str,
) -> Tuple[bool, float]:
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    if direction == "LR":
        if dx <= 0:
            return False, 0.0
        denom = abs(dx) + abs(dy)
        if denom <= 1e-9:
            return False, 0.0
        return True, abs(dx) / denom
    # TB
    if dy <= 0:
        return False, 0.0
    denom = abs(dx) + abs(dy)
    if denom <= 1e-9:
        return False, 0.0
    return True, abs(dy) / denom


def _average_node_diagonal(shape_nodes: List[Dict[str, Any]]) -> float:
    vals: List[float] = []
    for n in shape_nodes:
        b = _bbox(n.get("bbox"))
        if not b:
            continue
        w = b[2] - b[0]
        h = b[3] - b[1]
        vals.append(math.sqrt(w * w + h * h))
    if not vals:
        return 1.0
    return sum(vals) / len(vals)


def _flow_support(
    src_box: Tuple[float, float, float, float],
    dst_box: Tuple[float, float, float, float],
    flow_nodes: List[Dict[str, Any]],
    min_iou: float,
) -> float:
    corridor = _corridor_box(src_box, dst_box)
    if corridor is None:
        return 0.0

    best = 0.0
    for f in flow_nodes:
        fb = _bbox(f.get("bbox"))
        if not fb:
            continue
        ov = _iou(corridor, fb)
        if ov > best:
            best = ov
    if best < min_iou:
        return 0.0
    # clamp to [0,1]
    return max(0.0, min(1.0, best / max(min_iou, 1e-9)))


def _corridor_box(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float, float, float]]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    x2 = max(ax2, bx2)
    y2 = max(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _bbox(v: Any) -> Optional[Tuple[float, float, float, float]]:
    if not (isinstance(v, list) and len(v) == 4):
        return None
    try:
        x1, y1, x2, y2 = [float(x) for x in v]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _center(v: Any) -> Optional[Tuple[float, float]]:
    if not (isinstance(v, list) and len(v) == 2):
        return None
    try:
        return (float(v[0]), float(v[1]))
    except Exception:
        return None


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _area(b: Tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    ua = _area(a) + _area(b) - inter
    if ua <= 0.0:
        return 0.0
    return inter / ua
