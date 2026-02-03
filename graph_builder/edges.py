from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class EdgeBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class EdgeBuildConfig:
    min_final_score: float = 0.2
    max_outgoing_per_node: int = 4
    keep_parallel_edges: bool = True


def finalize_edges(
    payload: Dict[str, Any],
    cfg: Optional[EdgeBuildConfig] = None,
) -> Dict[str, Any]:
    """
    Convert edge candidates to final edges with deterministic pruning.
    Produces `edges` field that matches graph output contract.
    """
    if cfg is None:
        cfg = EdgeBuildConfig()

    nodes = payload.get("nodes")
    candidates = payload.get("edge_candidates")
    if not isinstance(nodes, list):
        raise EdgeBuildError("payload must contain 'nodes' list")
    if not isinstance(candidates, list):
        raise EdgeBuildError("payload must contain 'edge_candidates' list")

    node_by_id = {
        n.get("id"): n
        for n in nodes
        if isinstance(n, dict) and isinstance(n.get("id"), str)
    }

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    warnings: List[str] = []
    existing_warnings = meta.get("warnings", [])
    if isinstance(existing_warnings, list):
        warnings.extend([str(x) for x in existing_warnings])

    # Stage 1: validate + score filtering
    valid: List[Dict[str, Any]] = []
    dropped_low_score = 0
    dropped_invalid = 0
    for c in candidates:
        if not isinstance(c, dict):
            dropped_invalid += 1
            continue
        src = c.get("from")
        dst = c.get("to")
        score = c.get("score")
        if not (isinstance(src, str) and isinstance(dst, str) and isinstance(score, (int, float))):
            dropped_invalid += 1
            continue
        if src == dst:
            dropped_invalid += 1
            continue
        if src not in node_by_id or dst not in node_by_id:
            dropped_invalid += 1
            continue
        if float(score) < cfg.min_final_score:
            dropped_low_score += 1
            continue
        valid.append(c)

    # Stage 2: cap fan-out per source
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for c in valid:
        grouped.setdefault(c["from"], []).append(c)

    capped: List[Dict[str, Any]] = []
    pruned_fanout = 0
    for src, items in grouped.items():
        items_sorted = sorted(
            items,
            key=lambda x: (
                -float(x.get("score", 0.0)),
                str(x.get("to", "")),
            ),
        )
        keep = items_sorted[: cfg.max_outgoing_per_node]
        pruned_fanout += max(0, len(items_sorted) - len(keep))
        capped.extend(keep)

    if pruned_fanout > 0:
        warnings.append(f"edge_finalize: pruned {pruned_fanout} candidates by fan-out cap")
    if dropped_low_score > 0:
        warnings.append(f"edge_finalize: dropped {dropped_low_score} low-score candidates")
    if dropped_invalid > 0:
        warnings.append(f"edge_finalize: dropped {dropped_invalid} invalid/self-loop candidates")

    # Stage 3: convert to final edge types
    edges_raw: List[Dict[str, Any]] = []
    for c in capped:
        src = c["from"]
        dst = c["to"]
        src_node = node_by_id.get(src, {})
        edge_type = _infer_edge_type(src_node, c)
        edges_raw.append({"from": src, "to": dst, "type": edge_type})

    # Stage 4: dedup policy (respect parallel-edge requirement)
    if cfg.keep_parallel_edges:
        # Keep duplicates with same from/to but different types.
        # Remove only exact duplicates.
        edges = _dedup_exact(edges_raw)
    else:
        # keep only one best structural edge per (from,to) by priority
        edges = _dedup_by_pair(edges_raw)

    edges.sort(key=lambda e: (str(e["from"]), str(e["to"]), str(e["type"])))

    out_meta = dict(meta)
    out_meta["warnings"] = warnings

    out = dict(payload)
    out["meta"] = out_meta
    out["edges"] = edges
    out["edge_stats"] = {
        "candidates_in": len(candidates),
        "valid_after_score": len(valid),
        "after_fanout": len(capped),
        "final_edges": len(edges),
    }
    return out


def _infer_edge_type(src_node: Dict[str, Any], cand: Dict[str, Any]) -> str:
    src_role = str(src_node.get("role", "unknown"))
    hint = str(cand.get("type_hint", "unknown"))
    if src_role == "decision":
        return "conditional"
    if hint == "sequential":
        return "sequential"
    return "unknown"


def _dedup_exact(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for e in edges:
        k = (e["from"], e["to"], e["type"])
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    return out


def _dedup_by_pair(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    priority = {"conditional": 3, "sequential": 2, "unknown": 1}
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in edges:
        k = (e["from"], e["to"])
        old = best.get(k)
        if old is None:
            best[k] = e
            continue
        if priority.get(e["type"], 0) > priority.get(old["type"], 0):
            best[k] = e
    return list(best.values())
