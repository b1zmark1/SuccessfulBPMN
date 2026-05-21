from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


class SemanticProjectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SemanticProjectionConfig:
    schema_version: str = "semantic-projection.v1"


def project_graph_to_semantic(
    graph_payload: Dict[str, Any],
    cfg: Optional[SemanticProjectionConfig] = None,
) -> Dict[str, Any]:
    """
    Deterministically convert graph-builder output to LLM-friendly semantic JSON.
    """
    if cfg is None:
        cfg = SemanticProjectionConfig()

    meta = graph_payload.get("meta")
    nodes = graph_payload.get("nodes")
    edges = graph_payload.get("edges")
    if not isinstance(meta, dict):
        raise SemanticProjectionError("graph payload must contain 'meta' object")
    if not isinstance(nodes, list):
        raise SemanticProjectionError("graph payload must contain 'nodes' list")
    if not isinstance(edges, list):
        raise SemanticProjectionError("graph payload must contain 'edges' list")

    step_nodes = _extract_step_nodes(nodes)
    adjacency, indegree = _build_step_adjacency(step_nodes, edges)
    ordered_ids = _stable_order(step_nodes, adjacency, indegree)
    order_by_id = {sid: idx + 1 for idx, sid in enumerate(ordered_ids)}

    steps: List[Dict[str, Any]] = []
    for sid in ordered_ids:
        n = step_nodes[sid]
        next_ids = adjacency.get(sid, [])
        step_obj: Dict[str, Any] = {
            "id": sid,
            "order": order_by_id[sid],
            "role": _normalize_role(str(n.get("role", "unknown")), len(next_ids)),
            "text": _as_optional_str(n.get("text")),
            "next_step_ids": list(next_ids),
        }
        lane_role = _as_optional_str(n.get("lane_role"))
        if lane_role:
            step_obj["lane"] = lane_role
        steps.append(step_obj)

    src_schema = str(meta.get("schema_version", "graph-builder.v1"))
    direction = str(meta.get("direction", "LR")).upper()
    if direction not in {"LR", "TB"}:
        direction = "LR"
    warnings_raw = meta.get("warnings", [])
    warnings = [str(w) for w in warnings_raw] if isinstance(warnings_raw, list) else []

    return {
        "meta": {
            "schema_version": cfg.schema_version,
            "source_graph_schema_version": src_schema,
            "direction": direction,
            "warnings": warnings,
        },
        "steps": steps,
    }


def _extract_step_nodes(nodes: List[Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        ntype = n.get("type")
        if not isinstance(nid, str) or not nid:
            continue
        # Only process-like nodes become semantic steps.
        if ntype != "shape":
            continue
        out[nid] = n
    if not out:
        raise SemanticProjectionError("graph payload has no process shape nodes for projection")
    return out


def _build_step_adjacency(
    step_nodes: Dict[str, Dict[str, Any]],
    edges: List[Any],
) -> tuple[Dict[str, List[str]], Dict[str, int]]:
    step_ids: Set[str] = set(step_nodes.keys())
    adjacency: Dict[str, Set[str]] = {sid: set() for sid in step_ids}
    indegree: Dict[str, int] = {sid: 0 for sid in step_ids}

    for e in edges:
        if not isinstance(e, dict):
            continue
        src = e.get("from")
        dst = e.get("to")
        if not (isinstance(src, str) and isinstance(dst, str)):
            continue
        if src not in step_ids or dst not in step_ids:
            continue
        if dst in adjacency[src]:
            continue
        adjacency[src].add(dst)
        indegree[dst] += 1

    adjacency_sorted: Dict[str, List[str]] = {}
    for sid, dsts in adjacency.items():
        adjacency_sorted[sid] = sorted(dsts)
    return adjacency_sorted, indegree


def _stable_order(
    step_nodes: Dict[str, Dict[str, Any]],
    adjacency: Dict[str, List[str]],
    indegree: Dict[str, int],
) -> List[str]:
    def node_rank(node_id: str) -> tuple[int, str]:
        role = str(step_nodes[node_id].get("role", "unknown"))
        priority = 1 if role == "start" else 2
        return (priority, node_id)

    roots = [sid for sid, deg in indegree.items() if deg == 0]
    queue = sorted(roots, key=node_rank)
    if not queue:
        queue = sorted(step_nodes.keys(), key=node_rank)

    ordered: List[str] = []
    seen: Set[str] = set()

    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        for nxt in adjacency.get(current, []):
            if nxt not in seen:
                queue.append(nxt)
        queue = sorted(set(queue), key=node_rank)

    for sid in sorted(step_nodes.keys(), key=node_rank):
        if sid not in seen:
            ordered.append(sid)
    return ordered


def _normalize_role(role: str, out_degree: int) -> str:
    role = role.strip().lower()
    if role in {"start", "decision", "end"}:
        return role
    if out_degree > 1:
        return "parallel"
    return "action"


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v or None
    return str(value)

