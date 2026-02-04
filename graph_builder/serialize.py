# graph_builder/serialize.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


class SerializeError(RuntimeError):
    pass


@dataclass(frozen=True)
class SerializeConfig:
    schema_version: str = "graph-builder.v1"


def serialize_graph_output(
    payload: Dict[str, Any],
    cfg: Optional[SerializeConfig] = None,
) -> Dict[str, Any]:
    """
    Step 11:
    Emit final notation-agnostic graph JSON with stable ordering.
    Contract:
      - nodes: id, type, bbox, center, role, container_id, text
      - edges: from, to, type
      - meta: direction, warnings, schema_version
    """
    if cfg is None:
        cfg = SerializeConfig()

    nodes = payload.get("nodes")
    edges = payload.get("edges")
    meta = payload.get("meta")
    if not isinstance(nodes, list):
        raise SerializeError("payload must contain 'nodes' list")
    if not isinstance(edges, list):
        raise SerializeError("payload must contain 'edges' list")
    if not isinstance(meta, dict):
        meta = {}

    out_nodes, node_stats = _normalize_nodes(nodes)
    node_ids = {n["id"] for n in out_nodes}

    out_edges, edge_stats = _normalize_edges(edges, node_ids)
    out_meta = _normalize_meta(meta, cfg)

    # Add serialize-stage warnings (deterministic)
    warnings = list(out_meta.get("warnings", []))
    if node_stats["dropped_invalid_nodes"] > 0:
        warnings.append(f"serialize: dropped {node_stats['dropped_invalid_nodes']} invalid nodes")
    if edge_stats["dropped_invalid_edges"] > 0:
        warnings.append(f"serialize: dropped {edge_stats['dropped_invalid_edges']} invalid edges")
    if edge_stats["dropped_dangling_edges"] > 0:
        warnings.append(f"serialize: dropped {edge_stats['dropped_dangling_edges']} dangling edges")
    out_meta["warnings"] = _unique_preserve_order([str(w) for w in warnings])

    # Stable deterministic ordering for diffs/regression
    out_nodes.sort(key=lambda n: str(n["id"]))
    out_edges.sort(key=lambda e: (str(e["from"]), str(e["to"]), str(e["type"])))

    return {
        "meta": out_meta,
        "nodes": out_nodes,
        "edges": out_edges,
    }


def _normalize_nodes(nodes: List[Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    out: List[Dict[str, Any]] = []
    dropped_invalid = 0

    for n in nodes:
        if not isinstance(n, dict):
            dropped_invalid += 1
            continue

        nid = n.get("id")
        ntype = n.get("type")
        bbox = n.get("bbox")
        center = n.get("center")

        if not isinstance(nid, str) or not nid:
            dropped_invalid += 1
            continue
        if ntype not in {"shape", "container", "text", "flow"}:
            dropped_invalid += 1
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4):
            dropped_invalid += 1
            continue
        if not (isinstance(center, list) and len(center) == 2):
            dropped_invalid += 1
            continue

        role = n.get("role")
        if role not in {"action", "decision", "start", "end", "unknown"}:
            role = "unknown"

        container_id = n.get("container_id")
        if container_id is not None and not isinstance(container_id, str):
            container_id = None

        text = n.get("text")
        if text is not None and not isinstance(text, str):
            # keep contract strict: non-str becomes None
            text = None

        out.append(
            {
                "id": nid,
                "type": ntype,
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "center": [float(center[0]), float(center[1])],
                "role": role,
                "container_id": container_id,
                "text": text,
            }
        )

    return out, {"dropped_invalid_nodes": dropped_invalid}


def _normalize_edges(edges: List[Any], node_ids: Set[str]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    out: List[Dict[str, Any]] = []
    dropped_invalid = 0
    dropped_dangling = 0

    for e in edges:
        if not isinstance(e, dict):
            dropped_invalid += 1
            continue

        src = e.get("from")
        dst = e.get("to")
        etype = e.get("type")

        if not (isinstance(src, str) and src):
            dropped_invalid += 1
            continue
        if not (isinstance(dst, str) and dst):
            dropped_invalid += 1
            continue
        if etype not in {"sequential", "conditional", "unknown"}:
            dropped_invalid += 1
            continue

        if src not in node_ids or dst not in node_ids:
            dropped_dangling += 1
            continue

        out.append({"from": src, "to": dst, "type": etype})

    return out, {
        "dropped_invalid_edges": dropped_invalid,
        "dropped_dangling_edges": dropped_dangling,
    }


def _normalize_meta(meta: Dict[str, Any], cfg: SerializeConfig) -> Dict[str, Any]:
    direction = str(meta.get("direction", "LR")).upper()
    if direction not in {"LR", "TB"}:
        direction = "LR"

    warnings = meta.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = []
    warnings_out = [str(w) for w in warnings]

    return {
        "schema_version": cfg.schema_version,
        "direction": direction,
        "warnings": warnings_out,
    }


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out
