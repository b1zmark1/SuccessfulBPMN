from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class TextMergeError(RuntimeError):
    pass


@dataclass(frozen=True)
class TextMergeConfig:
    max_primary_gap_factor: float = 0.35
    min_secondary_overlap_ratio: float = 0.3


def merge_adjacent_text_nodes(
    payload: Dict[str, Any],
    cfg: Optional[TextMergeConfig] = None,
) -> Dict[str, Any]:
    """
    Step 9.1:
    Merge adjacent text boxes that likely belong to one phrase.
    Deterministic, direction-aware (LR/TB), contract-safe.
    """
    if cfg is None:
        cfg = TextMergeConfig()

    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise TextMergeError("payload must contain 'nodes' list")

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    direction = str(meta.get("direction", "LR")).upper()
    if direction not in {"LR", "TB"}:
        direction = "LR"

    warnings: List[str] = []
    old_warnings = meta.get("warnings", [])
    if isinstance(old_warnings, list):
        warnings.extend([str(x) for x in old_warnings])

    text_nodes = [n for n in nodes if isinstance(n, dict) and n.get("type") == "text"]
    other_nodes = [n for n in nodes if not (isinstance(n, dict) and n.get("type") == "text")]

    if len(text_nodes) < 2:
        out_meta = dict(meta)
        out_meta["warnings"] = warnings
        out = dict(payload)
        out["meta"] = out_meta
        out["nodes"] = list(nodes)
        out["text_merge_stats"] = {
            "text_nodes_in": len(text_nodes),
            "text_nodes_out": len(text_nodes),
            "merged_clusters": 0,
            "merged_nodes_created": 0,
            "direction": direction,
        }
        return out

    dims = []
    for n in text_nodes:
        b = _bbox(n.get("bbox"))
        if b:
            dims.append((b[2] - b[0], b[3] - b[1]))
    if not dims:
        raise TextMergeError("text nodes do not contain valid bbox geometry")

    med_w = _median([d[0] for d in dims])
    med_h = _median([d[1] for d in dims])
    primary_gap = max(4.0, cfg.max_primary_gap_factor * (med_h if direction == "LR" else med_w))

    n = len(text_nodes)
    parent = list(range(n))
    reasons: Dict[Tuple[int, int], str] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for i in range(n):
        bi = _bbox(text_nodes[i].get("bbox"))
        if bi is None:
            continue
        for j in range(i + 1, n):
            bj = _bbox(text_nodes[j].get("bbox"))
            if bj is None:
                continue
            should, reason = _should_merge(bi, bj, direction, primary_gap, cfg.min_secondary_overlap_ratio)
            if should:
                union(i, j)
                reasons[(i, j)] = reason

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    merged_nodes: List[Dict[str, Any]] = []
    merged_clusters = 0
    merged_created = 0

    for root in sorted(clusters.keys()):
        idxs = sorted(clusters[root], key=lambda k: int(text_nodes[k].get("original_index", 10**12)))
        if len(idxs) == 1:
            merged_nodes.append(dict(text_nodes[idxs[0]]))
            continue

        merged_clusters += 1
        cluster_nodes = [text_nodes[k] for k in idxs]
        mnode, has_conflict = _merge_cluster_nodes(cluster_nodes, idxs, reasons)
        if has_conflict:
            warnings.append(f"text_merge: mixed container_id in merged text cluster {mnode['id']}")
        merged_nodes.append(mnode)
        merged_created += 1

    final_nodes = other_nodes + merged_nodes
    final_nodes.sort(key=lambda x: (int(x.get("original_index", 10**12)), str(x.get("id", ""))))

    out_meta = dict(meta)
    out_meta["warnings"] = warnings

    out = dict(payload)
    out["meta"] = out_meta
    out["nodes"] = final_nodes
    out["text_merge_stats"] = {
        "text_nodes_in": len(text_nodes),
        "text_nodes_out": len(merged_nodes),
        "merged_clusters": merged_clusters,
        "merged_nodes_created": merged_created,
        "direction": direction,
        "primary_gap_threshold": primary_gap,
    }
    return out


def _merge_cluster_nodes(
    cluster_nodes: List[Dict[str, Any]],
    idxs: List[int],
    reasons_map: Dict[Tuple[int, int], str],
) -> Tuple[Dict[str, Any], bool]:
    bboxes = [_bbox(n.get("bbox")) for n in cluster_nodes]
    bboxes = [b for b in bboxes if b is not None]
    if not bboxes:
        raise TextMergeError("cannot merge text cluster without bbox")

    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    node_ids = [str(n.get("id", "")) for n in cluster_nodes]
    det_ids = [str(n.get("det_id", "")) for n in cluster_nodes]
    original_idx = min(int(n.get("original_index", 10**12)) for n in cluster_nodes)
    source = str(cluster_nodes[0].get("source", "easyocr"))
    score = max(float(n.get("score", 0.0)) for n in cluster_nodes)

    cids = [n.get("container_id") for n in cluster_nodes]
    unique_cids = {c for c in cids if c is not None}
    container_id = None
    has_conflict = False
    if len(unique_cids) == 1:
        container_id = list(unique_cids)[0]
    elif len(unique_cids) > 1:
        has_conflict = True

    reasons = []
    idx_set = set(idxs)
    for (i, j), r in reasons_map.items():
        if i in idx_set and j in idx_set:
            reasons.append(r)
    reasons = sorted(set(reasons))

    merged = {
        "id": f"node_textmerge_{original_idx:06d}",
        "type": "text",
        "bbox": [x1, y1, x2, y2],
        "center": center,
        "role": "unknown",
        "container_id": container_id,
        "text": None,
        "class_name": "text",
        "source": source,
        "score": score,
        "det_id": f"textmerge_{original_idx:06d}",
        "original_index": original_idx,
        "merge_provenance": {
            "merged_from_node_ids": node_ids,
            "merged_from_det_ids": det_ids,
            "merge_reasons": reasons,
            "cluster_size": len(cluster_nodes),
        },
    }
    return merged, has_conflict


def _should_merge(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    direction: str,
    primary_gap_threshold: float,
    min_secondary_overlap: float,
) -> Tuple[bool, str]:
    if _intersection_area(a, b) > 0.0:
        return True, "overlap"
    if _touching(a, b):
        return True, "touching"

    if direction == "LR":
        gap = _axis_gap(a[0], a[2], b[0], b[2])
        secondary_overlap = _overlap_ratio((a[1], a[3]), (b[1], b[3]))
        if gap <= primary_gap_threshold and secondary_overlap >= min_secondary_overlap:
            return True, "adjacent_lr"
        return False, ""

    gap = _axis_gap(a[1], a[3], b[1], b[3])
    secondary_overlap = _overlap_ratio((a[0], a[2]), (b[0], b[2]))
    if gap <= primary_gap_threshold and secondary_overlap >= min_secondary_overlap:
        return True, "adjacent_tb"
    return False, ""


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


def _median(vals: List[float]) -> float:
    xs = sorted(vals)
    n = len(xs)
    if n == 0:
        return 0.0
    m = n // 2
    if n % 2 == 1:
        return xs[m]
    return (xs[m - 1] + xs[m]) / 2.0


def _intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _touching(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    # touching or 1px gap-like adjacency
    expand = 1.0
    ae = (a[0] - expand, a[1] - expand, a[2] + expand, a[3] + expand)
    return _intersection_area(ae, b) > 0.0


def _axis_gap(a1: float, a2: float, b1: float, b2: float) -> float:
    if a2 < b1:
        return b1 - a2
    if b2 < a1:
        return a1 - b2
    return 0.0


def _overlap_ratio(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    i1 = max(a[0], b[0])
    i2 = min(a[1], b[1])
    inter = max(0.0, i2 - i1)
    la = max(1e-9, a[1] - a[0])
    lb = max(1e-9, b[1] - b[0])
    return inter / min(la, lb)
