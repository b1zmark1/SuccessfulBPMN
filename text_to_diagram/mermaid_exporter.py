from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


ALLOWED_DIRECTIONS = {"LR", "TB"}


def export_mermaid(ir: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    if not isinstance(ir, dict):
        return {
            "status": "degraded",
            "mmd": "flowchart LR\n",
            "issues": [_issue("MERMAID_INVALID_IR", "error", "IR must be object")],
            "meta": {"exporter": "mermaid", "direction": "LR"},
        }

    direction = ir.get("meta", {}).get("direction") if isinstance(ir.get("meta"), dict) else None
    if direction not in ALLOWED_DIRECTIONS:
        direction = "LR"
        issues.append(_issue("MERMAID_DEFAULT_DIRECTION", "warning", "direction defaulted to LR"))

    nodes_raw = ir.get("nodes")
    edges_raw = ir.get("edges")
    lanes_raw = ir.get("lanes")
    nodes_raw = nodes_raw if isinstance(nodes_raw, list) else []
    edges_raw = edges_raw if isinstance(edges_raw, list) else []
    lanes_raw = lanes_raw if isinstance(lanes_raw, list) else []

    lane_ids = {l.get("id") for l in lanes_raw if isinstance(l, dict) and isinstance(l.get("id"), str)}

    nodes = _sorted_nodes(nodes_raw)
    lanes = _sorted_lanes(lanes_raw)
    node_ids: Set[str] = set()
    node_lines: List[str] = []
    lane_nodes: Dict[Optional[str], List[str]] = {}

    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            issues.append(_issue("MERMAID_NODE_ID_INVALID", "warning", "node skipped due to invalid id"))
            continue
        if node_id in node_ids:
            issues.append(
                _issue(
                    "MERMAID_DUPLICATE_NODE_ID",
                    "warning",
                    f"duplicate node id '{node_id}'",
                    entity_type="node",
                    entity_id=node_id,
                )
            )
            continue
        node_ids.add(node_id)
        lane_id = node.get("lane_id") if isinstance(node.get("lane_id"), str) else None
        if lane_id is not None and lane_id not in lane_ids:
            issues.append(
                _issue(
                    "MERMAID_NODE_UNKNOWN_LANE",
                    "warning",
                    f"node '{node_id}' lane '{lane_id}' not found; placed in global scope",
                    entity_type="node",
                    entity_id=node_id,
                )
            )
            lane_id = None
        line, node_issues = _render_node(node)
        issues.extend(node_issues)
        lane_nodes.setdefault(lane_id, []).append(f"    {line}")
        node_lines.append(line)

    edges = _sorted_edges(edges_raw)
    edge_lines: List[str] = []
    for edge in edges:
        line, edge_issues = _render_edge(edge, node_ids)
        issues.extend(edge_issues)
        if line is not None:
            edge_lines.append(f"    {line}")

    lines: List[str] = [f"flowchart {direction}"]
    for lane in lanes:
        lane_id = lane.get("id")
        lane_name = _safe_label(lane.get("name") if isinstance(lane.get("name"), str) else lane_id)
        lines.append(f'    subgraph {lane_id}["{lane_name}"]')
        for node_line in lane_nodes.get(lane_id, []):
            lines.append(node_line)
        lines.append("    end")

    for node_line in lane_nodes.get(None, []):
        lines.append(node_line)
    lines.extend(edge_lines)

    status = "degraded" if issues else "ok"
    return {
        "status": status,
        "mmd": "\n".join(lines).rstrip() + "\n",
        "issues": issues,
        "meta": {
            "exporter": "mermaid",
            "direction": direction,
            "nodes_count": len(node_ids),
            "edges_count": len(edge_lines),
            "lanes_count": len(lanes),
        },
    }


def _render_node(node: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    node_id = node["id"]
    role = node.get("role")
    text = node.get("text")
    label = _safe_label(text if isinstance(text, str) and text.strip() else node_id)

    if role == "start":
        return f'{node_id}(["{label}"])', issues
    if role == "end":
        return f'{node_id}(["{label}"])', issues
    if role == "action":
        return f'{node_id}["{label}"]', issues
    if role == "decision":
        return f'{node_id}{{"{label}"}}', issues
    if role == "parallel":
        return f'{node_id}{{"+"}}', issues
    if role == "inclusive":
        return f'{node_id}{{"O"}}', issues
    if role == "event_intermediate":
        return f'{node_id}(("{label}"))', issues

    issues.append(
        _issue(
            "MERMAID_UNKNOWN_NODE_ROLE",
            "warning",
            f"node '{node_id}' role '{role}' exported as action",
            entity_type="node",
            entity_id=node_id,
        )
    )
    return f'{node_id}["{label}"]', issues


def _render_edge(edge: Dict[str, Any], node_ids: Set[str]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    edge_id = edge.get("id")
    src = edge.get("from")
    dst = edge.get("to")
    if not isinstance(src, str) or not isinstance(dst, str):
        issues.append(_issue("MERMAID_EDGE_INVALID", "warning", "edge skipped due to invalid from/to"))
        return None, issues
    if src not in node_ids or dst not in node_ids:
        issues.append(
            _issue(
                "MERMAID_EDGE_BROKEN_REF",
                "warning",
                f"edge '{edge_id}' skipped due to unresolved node reference",
                entity_type="edge",
                entity_id=edge_id if isinstance(edge_id, str) else None,
            )
        )
        return None, issues

    edge_type = edge.get("type")
    label = edge.get("text")
    safe_label = _safe_label(label) if isinstance(label, str) and label.strip() else None

    if edge_type == "sequential":
        return f"{src} --> {dst}", issues
    if edge_type == "conditional":
        if safe_label:
            return f"{src} -->|{safe_label}| {dst}", issues
        return f"{src} --> {dst}", issues
    if edge_type == "message":
        if safe_label:
            return f"{src} -. {safe_label} .-> {dst}", issues
        return f"{src} -.-> {dst}", issues
    if edge_type == "association":
        return f"{src} -.-> {dst}", issues
    if edge_type == "unknown":
        issues.append(
            _issue(
                "MERMAID_UNKNOWN_EDGE_TYPE",
                "warning",
                f"edge '{edge_id}' type unknown; exported as sequential",
                entity_type="edge",
                entity_id=edge_id if isinstance(edge_id, str) else None,
            )
        )
        return f"{src} --> {dst}", issues

    issues.append(
        _issue(
            "MERMAID_UNSUPPORTED_EDGE_TYPE",
            "warning",
            f"edge '{edge_id}' has unsupported type '{edge_type}'; exported as sequential",
            entity_type="edge",
            entity_id=edge_id if isinstance(edge_id, str) else None,
        )
    )
    return f"{src} --> {dst}", issues


def _sorted_lanes(lanes: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = [x for x in lanes if isinstance(x, dict)]
    return sorted(
        normalized,
        key=lambda x: (
            x.get("order") if isinstance(x.get("order"), int) else 0,
            str(x.get("id")),
        ),
    )


def _sorted_nodes(nodes: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = [x for x in nodes if isinstance(x, dict)]
    return sorted(normalized, key=lambda x: str(x.get("id")))


def _sorted_edges(edges: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = [x for x in edges if isinstance(x, dict)]
    return sorted(
        normalized,
        key=lambda x: (
            str(x.get("from")),
            str(x.get("to")),
            str(x.get("id")),
        ),
    )


def _safe_label(value: Any) -> str:
    s = str(value) if value is not None else ""
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", " ")
    s = s.replace("\r", " ")
    return " ".join(s.split())


def _issue(
    code: str,
    severity: str,
    message: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "entity_type": entity_type,
        "entity_id": entity_id,
    }
