from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math


class TextHookError(RuntimeError):
    pass


@dataclass(frozen=True)
class TextHookConfig:
    max_anchor_distance_factor: float = 2.0
    prefer_overlap: bool = True
    include_flow_nodes: bool = False


def attach_text_placeholders_and_hooks(
    payload: Dict[str, Any],
    cfg: Optional[TextHookConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = TextHookConfig()

    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise TextHookError("payload must contain 'nodes' list")

    parsed_nodes = [n for n in nodes if isinstance(n, dict)]
    text_nodes = [n for n in parsed_nodes if n.get("type") == "text"]

    def _is_target(n: Dict[str, Any]) -> bool:
        t = n.get("type")
        if t == "shape" or t == "container":
            return True
        if cfg.include_flow_nodes and t == "flow":
            return True
        return False

    target_nodes = [n for n in parsed_nodes if _is_target(n)]
    avg_diag = _average_diag(target_nodes)
    max_dist = max(1.0, cfg.max_anchor_distance_factor * avg_diag)

    warnings: List[str] = []
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    old_w = meta.get("warnings", [])
    if isinstance(old_w, list):
        warnings.extend([str(x) for x in old_w])

    if not text_nodes:
        warnings.append("text_hooks: no text nodes found")

    # anchor map: target_id -> list[text_node_id]
    anchor_map: Dict[str, List[str]] = {}
    reverse_map: Dict[str, str] = {}

    for tnode in text_nodes:
        tid = tnode.get("id")
        tb = _bbox(tnode.get("bbox"))
        tc = _center(tnode.get("center"))
        if not (isinstance(tid, str) and tb and tc):
            continue

        best_id: Optional[str] = None
        best_key: Tuple[float, float] = (-1.0, float("inf"))  # (overlap, distance)

        for n in target_nodes:
            nid = n.get("id")
            nb = _bbox(n.get("bbox"))
            nc = _center(n.get("center"))
            if not (isinstance(nid, str) and nb and nc):
                continue

            overlap = _intersection_area(tb, nb)
            dist = _euclidean(tc, nc)
            if dist > max_dist and overlap <= 0.0:
                continue

            if cfg.prefer_overlap:
                key = (overlap, -dist)
            else:
                key = (0.0, -dist)

            if key > best_key:
                best_key = key
                best_id = nid

        if best_id is not None:
            anchor_map.setdefault(best_id, []).append(tid)
            reverse_map[tid] = best_id

    updated_nodes: List[Dict[str, Any]] = []
    for n in parsed_nodes:
        out_n = dict(n)

        # Contract requirement: every node has text field, nullable.
        if "text" not in out_n:
            out_n["text"] = None
        if out_n["text"] is not None and not isinstance(out_n["text"], str):
            out_n["text"] = None

        nid = out_n.get("id")
        if isinstance(nid, str):
            if nid in anchor_map:
                out_n["text_anchor_ids"] = sorted(anchor_map[nid])
            if nid in reverse_map:
                out_n["text_anchor_for"] = reverse_map[nid]

        updated_nodes.append(out_n)

    updated_nodes.sort(key=lambda n: (int(n.get("original_index", 10**12)), str(n.get("id", ""))))

    ocr_hooks = []
    for nid, text_ids in sorted(anchor_map.items()):
        ocr_hooks.append(
            {
                "node_id": nid,
                "text_node_ids": sorted(text_ids),
                "status": "pending_ocr",
            }
        )

    out_meta = dict(meta)
    out_meta["warnings"] = warnings

    out = dict(payload)
    out["meta"] = out_meta
    out["nodes"] = updated_nodes
    out["ocr_hooks"] = ocr_hooks
    out["text_hook_stats"] = {
        "text_nodes": len(text_nodes),
        "target_nodes": len(target_nodes),
        "anchored_text_nodes": len(reverse_map),
        "max_anchor_distance": max_dist,
    }
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


def _average_diag(nodes: List[Dict[str, Any]]) -> float:
    vals: List[float] = []
    for n in nodes:
        b = _bbox(n.get("bbox"))
        if not b:
            continue
        w = b[2] - b[0]
        h = b[3] - b[1]
        vals.append(math.sqrt(w * w + h * h))
    if not vals:
        return 100.0
    return sum(vals) / len(vals)


def _intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)
