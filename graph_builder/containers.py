from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class ContainerAssignError(RuntimeError):
    pass


@dataclass(frozen=True)
class ContainerAssignConfig:
    min_node_inside_ratio: float = 0.6
    require_center_inside: bool = True
    conflict_iou_threshold: float = 0.5


def assign_container_hierarchy(
    payload: Dict[str, Any],
    cfg: Optional[ContainerAssignConfig] = None,
) -> Dict[str, Any]:
    """
    Assign container_id for nodes based on geometric containment.
    This step does not create edges and keeps output contract-compatible nodes.
    """
    if cfg is None:
        cfg = ContainerAssignConfig()

    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise ContainerAssignError("payload must contain 'nodes' list")

    parsed_nodes = [n for n in nodes if isinstance(n, dict)]
    container_nodes = [n for n in parsed_nodes if n.get("type") == "container"]
    non_container_nodes = [n for n in parsed_nodes if n.get("type") != "container"]

    warnings: List[str] = []
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    existing_warnings = meta.get("warnings", [])
    if isinstance(existing_warnings, list):
        warnings.extend([str(x) for x in existing_warnings])

    # Nested containers are not supported by product decision.
    container_conflicts = _detect_container_conflicts(container_nodes, cfg)
    for c in container_conflicts:
        warnings.append(
            f"container_conflict: {c['a']} vs {c['b']} iou={c['iou']:.3f}; nested containers disabled"
        )

    updated_nodes: List[Dict[str, Any]] = []
    assigned_count = 0

    # Containers stay on top level.
    for c in container_nodes:
        out_c = dict(c)
        out_c["container_id"] = None
        updated_nodes.append(out_c)

    for n in non_container_nodes:
        out_n = dict(n)
        best = _pick_best_container(out_n, container_nodes, cfg)
        if best is not None:
            out_n["container_id"] = best
            assigned_count += 1
        else:
            out_n["container_id"] = None
        updated_nodes.append(out_n)

    updated_nodes.sort(key=lambda n: (int(n.get("original_index", 10**12)), str(n.get("id", ""))))

    out_meta = dict(meta)
    out_meta["warnings"] = warnings

    out = dict(payload)
    out["meta"] = out_meta
    out["nodes"] = updated_nodes
    out["container_stats"] = {
        "containers_total": len(container_nodes),
        "nodes_total": len(parsed_nodes),
        "nodes_with_container": assigned_count,
        "container_conflicts": len(container_conflicts),
    }
    return out


def _pick_best_container(
    node: Dict[str, Any],
    containers: List[Dict[str, Any]],
    cfg: ContainerAssignConfig,
) -> Optional[str]:
    nb = _bbox(node.get("bbox"))
    nc = _center(node.get("center"))
    if nb is None:
        return None

    cands: List[Tuple[float, float, int, str]] = []
    for c in containers:
        cb = _bbox(c.get("bbox"))
        if cb is None:
            continue
        if cfg.require_center_inside and nc is not None and not _point_inside(nc, cb):
            continue
        inside_ratio = _intersection_area(nb, cb) / max(_area(nb), 1e-9)
        if inside_ratio < cfg.min_node_inside_ratio:
            continue
        c_area = _area(cb)
        c_idx = int(c.get("original_index", 10**9))
        c_id = str(c.get("id", ""))
        # rank: higher inside ratio, then smaller container area, then lower original index
        cands.append((inside_ratio, -c_area, -c_idx, c_id))

    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][3]


def _detect_container_conflicts(
    containers: List[Dict[str, Any]],
    cfg: ContainerAssignConfig,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(len(containers)):
        ai = containers[i]
        ab = _bbox(ai.get("bbox"))
        if ab is None:
            continue
        for j in range(i + 1, len(containers)):
            bj = containers[j]
            bb = _bbox(bj.get("bbox"))
            if bb is None:
                continue
            iou = _iou(ab, bb)
            if iou >= cfg.conflict_iou_threshold:
                out.append(
                    {
                        "a": str(ai.get("id", "")),
                        "b": str(bj.get("id", "")),
                        "iou": iou,
                    }
                )
    return out


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


def _point_inside(p: Tuple[float, float], b: Tuple[float, float, float, float]) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def _area(b: Tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    ua = _area(a) + _area(b) - inter
    if ua <= 0:
        return 0.0
    return inter / ua
