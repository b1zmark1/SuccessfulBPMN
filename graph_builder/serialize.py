from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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

    out_nodes = _normalize_nodes(nodes)
    out_edges = _normalize_edges(edges)
    out_meta = _normalize_meta(meta, cfg)

    # Stable deterministic ordering for diffs/regression
    out_nodes.sort(key=lambda n: (str(n["id"])))
    out_edges.sort(key=lambda e: (str(e["from"]), str(e["to"]), str(e["type"])))

    return {
        "meta": out_meta,
        "nodes": out_nodes,
        "edges": out_edges,
    }


def _normalize_nodes(nodes: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    required = ["id", "type", "bbox", "center", "role", "container_id", "text"]
    for n in nodes:
        if not isinstance(n, dict):
            continue
        if not all(k in n for k in required):
            continue

        nid = n.get("id")
        ntype = n.get("type")
        bbox = n.get("bbox")
        center = n.get("center")
        role = n.get("role")
        container_id = n.get("container_id")
        text = n.get("text")

        if not isinstance(nid, str) or not nid:
            continue
        if ntype not in {"shape", "container", "text", "flow"}:
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        if not (isinstance(center, list) and len(center) == 2):
            continue
        if role not in {"action", "decision", "start", "end", "unknown"}:
            continue
        if container_id is not None and not isinstance(container_id, str):
            continue
        if text is not None and not isinstance(text, str):
            continue

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
    return out


def _normalize_edges(edges: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        src = e.get("from")
        dst = e.get("to")
        etype = e.get("type")
        if not (isinstance(src, str) and src):
            continue
        if not (isinstance(dst, str) and dst):
            continue
        if etype not in {"sequential", "conditional", "unknown"}:
            continue
        out.append({"from": src, "to": dst, "type": etype})
    return out


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
