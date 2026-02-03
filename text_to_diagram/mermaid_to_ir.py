from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


_FLOWCHART_RE = re.compile(r"^flowchart\s+(LR|TB)\s*$")
_SUBGRAPH_RE = re.compile(r'^subgraph\s+([A-Za-z0-9_]+)\["(.*)"\]\s*$')

_NODE_START_END_RE = re.compile(r'^([A-Za-z0-9_]+)\(\["(.*)"\]\)\s*$')
_NODE_ACTION_RE = re.compile(r'^([A-Za-z0-9_]+)\["(.*)"\]\s*$')
_NODE_DECISION_RE = re.compile(r'^([A-Za-z0-9_]+)\{"(.*)"\}\s*$')
_NODE_EVENT_RE = re.compile(r'^([A-Za-z0-9_]+)\(\("(.*)"\)\)\s*$')

_EDGE_COND_RE = re.compile(r"^([A-Za-z0-9_]+)\s+-->\|(.+)\|\s+([A-Za-z0-9_]+)\s*$")
_EDGE_SEQ_RE = re.compile(r"^([A-Za-z0-9_]+)\s+-->\s+([A-Za-z0-9_]+)\s*$")
_EDGE_DOTTED_RE = re.compile(r"^([A-Za-z0-9_]+)\s+-\.\s*(.*?)\s*\.->\s+([A-Za-z0-9_]+)\s*$")
_EDGE_ASSOC_RE = re.compile(r"^([A-Za-z0-9_]+)\s+-\.->\s+([A-Za-z0-9_]+)\s*$")


def mermaid_to_ir(mmd: str) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    if not isinstance(mmd, str) or not mmd.strip():
        return _fallback_ir([_issue("MERMAID_TO_IR_EMPTY", "error", "empty mermaid input")])

    lines = [x.rstrip() for x in mmd.splitlines() if x.strip()]
    direction = "LR"
    lanes: List[Dict[str, Any]] = []
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    lane_ctx: Optional[str] = None
    lane_order = 0
    edge_idx = 1

    for raw in lines:
        line = raw.strip()
        m = _FLOWCHART_RE.match(line)
        if m:
            direction = m.group(1)
            continue

        m = _SUBGRAPH_RE.match(line)
        if m:
            lane_id = m.group(1)
            lane_name = m.group(2).strip() or lane_id
            lanes.append({"id": lane_id, "name": lane_name, "order": lane_order})
            lane_order += 1
            lane_ctx = lane_id
            continue

        if line == "end":
            lane_ctx = None
            continue

        node = _parse_node(line, lane_ctx)
        if node is not None:
            node_id = node["id"]
            if node_id in nodes:
                issues.append(_issue("MERMAID_TO_IR_DUP_NODE", "warning", f"duplicate node '{node_id}'"))
            else:
                nodes[node_id] = node
            continue

        edge = _parse_edge(line, edge_idx)
        if edge is not None:
            edges.append(edge)
            edge_idx += 1
            continue

    if not nodes:
        return _fallback_ir([_issue("MERMAID_TO_IR_NO_NODES", "error", "no nodes parsed from mermaid")])

    node_ids = set(nodes.keys())
    valid_edges: List[Dict[str, Any]] = []
    for edge in edges:
        if edge["from"] not in node_ids or edge["to"] not in node_ids:
            issues.append(
                _issue(
                    "MERMAID_TO_IR_EDGE_BROKEN_REF",
                    "warning",
                    f"edge '{edge['id']}' has unresolved node reference",
                )
            )
            continue
        valid_edges.append(edge)
    edges = valid_edges

    _resolve_start_end_roles(nodes, edges)

    ir = {
        "nodes": sorted(nodes.values(), key=lambda x: x["id"]),
        "edges": sorted(edges, key=lambda x: (x["from"], x["to"], x["id"])),
        "lanes": sorted(lanes, key=lambda x: (x["order"], x["id"])),
        "meta": {
            "schema_version": "process-ir.v1",
            "direction": direction if direction in {"LR", "TB"} else "LR",
            "source": "text_to_diagram",
            "language": "ru",
        },
        "issues": issues,
    }
    return ir


def _parse_node(line: str, lane_id: Optional[str]) -> Optional[Dict[str, Any]]:
    m = _NODE_DECISION_RE.match(line)
    if m:
        return _node(m.group(1), "decision", m.group(2), lane_id)
    m = _NODE_EVENT_RE.match(line)
    if m:
        return _node(m.group(1), "event_intermediate", m.group(2), lane_id)
    m = _NODE_START_END_RE.match(line)
    if m:
        return _node(m.group(1), "start_end_candidate", m.group(2), lane_id)
    m = _NODE_ACTION_RE.match(line)
    if m:
        return _node(m.group(1), "action", m.group(2), lane_id)
    return None


def _parse_edge(line: str, idx: int) -> Optional[Dict[str, Any]]:
    m = _EDGE_COND_RE.match(line)
    if m:
        return {
            "id": f"e{idx}",
            "from": m.group(1),
            "to": m.group(3),
            "type": "conditional",
            "text": m.group(2).strip() or None,
        }
    m = _EDGE_SEQ_RE.match(line)
    if m:
        return {"id": f"e{idx}", "from": m.group(1), "to": m.group(2), "type": "sequential", "text": None}
    m = _EDGE_DOTTED_RE.match(line)
    if m:
        label = m.group(2).strip()
        return {
            "id": f"e{idx}",
            "from": m.group(1),
            "to": m.group(3),
            "type": "association",
            "text": label or None,
        }
    m = _EDGE_ASSOC_RE.match(line)
    if m:
        return {"id": f"e{idx}", "from": m.group(1), "to": m.group(2), "type": "association", "text": None}
    return None


def _resolve_start_end_roles(nodes: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
    indeg = {node_id: 0 for node_id in nodes}
    outdeg = {node_id: 0 for node_id in nodes}
    for edge in edges:
        outdeg[edge["from"]] = outdeg.get(edge["from"], 0) + 1
        indeg[edge["to"]] = indeg.get(edge["to"], 0) + 1

    candidates = [x for x in nodes.values() if x.get("role") == "start_end_candidate"]
    if not candidates:
        return

    starts = sorted([x for x in candidates if indeg.get(x["id"], 0) == 0], key=lambda x: x["id"])
    ends = sorted([x for x in candidates if outdeg.get(x["id"], 0) == 0], key=lambda x: x["id"])

    start_id = starts[0]["id"] if starts else candidates[0]["id"]
    end_id = ends[0]["id"] if ends else candidates[-1]["id"]

    for node in candidates:
        if node["id"] == start_id:
            node["role"] = "start"
        elif node["id"] == end_id:
            node["role"] = "end"
        else:
            node["role"] = "action"


def _node(node_id: str, role: str, text: str, lane_id: Optional[str]) -> Dict[str, Any]:
    return {
        "id": node_id,
        "type": "shape",
        "role": role,
        "text": text.strip() if isinstance(text, str) else None,
        "lane_id": lane_id,
        "container_id": None,
    }


def _issue(code: str, severity: str, message: str) -> Dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "entity_type": None,
        "entity_id": None,
    }


def _fallback_ir(extra_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "nodes": [
            {"id": "n1", "type": "shape", "role": "start", "text": "Старт", "lane_id": None, "container_id": None},
            {"id": "n2", "type": "shape", "role": "end", "text": "Завершение", "lane_id": None, "container_id": None},
        ],
        "edges": [{"id": "e1", "from": "n1", "to": "n2", "type": "sequential", "text": None}],
        "lanes": [],
        "meta": {
            "schema_version": "process-ir.v1",
            "direction": "LR",
            "source": "text_to_diagram",
            "language": "ru",
        },
        "issues": [
            _issue("MERMAID_TO_IR_FALLBACK", "warning", "fallback IR generated from mermaid"),
            *extra_issues,
        ],
    }

