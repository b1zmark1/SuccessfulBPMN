from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


ALLOWED_DIRECTIONS = {"LR", "TB"}


def export_plantuml(ir: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    if not isinstance(ir, dict):
        return {
            "status": "degraded",
            "puml": "@startuml\nleft to right direction\n@enduml\n",
            "issues": [_issue("PUML_INVALID_IR", "error", "IR must be object")],
            "meta": {"exporter": "plantuml", "profile": "state-flow.v1"},
        }

    direction = ir.get("meta", {}).get("direction") if isinstance(ir.get("meta"), dict) else None
    if direction not in ALLOWED_DIRECTIONS:
        direction = "LR"
        issues.append(_issue("PUML_DEFAULT_DIRECTION", "warning", "direction defaulted to LR"))

    lanes_raw = ir.get("lanes") if isinstance(ir.get("lanes"), list) else []
    nodes_raw = ir.get("nodes") if isinstance(ir.get("nodes"), list) else []
    edges_raw = ir.get("edges") if isinstance(ir.get("edges"), list) else []

    lanes = _sorted_lanes(lanes_raw)
    nodes = _sorted_nodes(nodes_raw)
    edges = _sorted_edges(edges_raw)

    lane_ids = {x["id"] for x in lanes}
    node_ids: Set[str] = set()
    lane_nodes: Dict[Optional[str], List[str]] = {}
    global_nodes: List[str] = []

    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            issues.append(_issue("PUML_NODE_ID_INVALID", "warning", "node skipped due to invalid id"))
            continue
        if node_id in node_ids:
            issues.append(
                _issue(
                    "PUML_DUPLICATE_NODE_ID",
                    "warning",
                    f"duplicate node id '{node_id}'",
                    entity_type="node",
                    entity_id=node_id,
                )
            )
            continue

        node_ids.add(node_id)
        node_line, node_issues = _render_node(node)
        issues.extend(node_issues)

        lane_id = node.get("lane_id") if isinstance(node.get("lane_id"), str) else None
        if lane_id is not None and lane_id not in lane_ids:
            issues.append(
                _issue(
                    "PUML_NODE_UNKNOWN_LANE",
                    "warning",
                    f"node '{node_id}' lane '{lane_id}' not found; placed in global scope",
                    entity_type="node",
                    entity_id=node_id,
                )
            )
            lane_id = None

        if lane_id is None:
            global_nodes.append(node_line)
        else:
            lane_nodes.setdefault(lane_id, []).append(node_line)

    edge_lines: List[str] = []
    for edge in edges:
        line, edge_issues = _render_edge(edge, node_ids)
        issues.extend(edge_issues)
        if line is not None:
            edge_lines.append(line)

    lines: List[str] = ["@startuml"]
    lines.append("left to right direction" if direction == "LR" else "top to bottom direction")

    for lane in lanes:
        lane_id = lane["id"]
        lane_name = _safe_label(lane.get("name"))
        lines.append(f'state "{lane_name}" as lane_{lane_id} {{')
        for node_line in lane_nodes.get(lane_id, []):
            lines.append(f"  {node_line}")
        lines.append("}")

    for node_line in global_nodes:
        lines.append(node_line)

    for edge_line in edge_lines:
        lines.append(edge_line)

    lines.append("@enduml")
    puml = "\n".join(lines).rstrip() + "\n"
    status = "degraded" if issues else "ok"
    return {
        "status": status,
        "puml": puml,
        "issues": issues,
        "meta": {
            "exporter": "plantuml",
            "profile": "state-flow.v1",
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
    label = _safe_label(node.get("text") if isinstance(node.get("text"), str) else node_id)
    if role == "start":
        return f'state "{label}" as {node_id} <<start>>', issues
    if role == "end":
        return f'state "{label}" as {node_id} <<end>>', issues
    if role == "action":
        return f'state "{label}" as {node_id}', issues
    if role == "decision":
        return f'state "{label}" as {node_id} <<decision>>', issues
    if role == "parallel":
        return f'state "{label}" as {node_id} <<parallel>>', issues
    if role == "inclusive":
        return f'state "{label}" as {node_id} <<inclusive>>', issues
    if role == "event_intermediate":
        return f'state "{label}" as {node_id} <<event>>', issues

    issues.append(
        _issue(
            "PUML_UNKNOWN_NODE_ROLE",
            "warning",
            f"node '{node_id}' role '{role}' exported as action state",
            entity_type="node",
            entity_id=node_id,
        )
    )
    return f'state "{label}" as {node_id}', issues


def _render_edge(edge: Dict[str, Any], node_ids: Set[str]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    edge_id = edge.get("id") if isinstance(edge.get("id"), str) else None
    src = edge.get("from")
    dst = edge.get("to")

    if not isinstance(src, str) or not isinstance(dst, str):
        issues.append(
            _issue(
                "PUML_EDGE_INVALID",
                "warning",
                "edge skipped due to invalid from/to",
                entity_type="edge",
                entity_id=edge_id,
            )
        )
        return None, issues
    if src not in node_ids or dst not in node_ids:
        issues.append(
            _issue(
                "PUML_EDGE_BROKEN_REF",
                "warning",
                f"edge '{edge_id}' skipped due to unresolved node reference",
                entity_type="edge",
                entity_id=edge_id,
            )
        )
        return None, issues

    edge_type = edge.get("type")
    label = _safe_label(edge.get("text")) if isinstance(edge.get("text"), str) else None
    if edge_type == "sequential":
        return f"{src} --> {dst}", issues
    if edge_type == "conditional":
        return f"{src} --> {dst} : {label}" if label else f"{src} --> {dst}", issues
    if edge_type == "message":
        return f"{src} ..> {dst} : {label}" if label else f"{src} ..> {dst}", issues
    if edge_type == "association":
        return f"{src} ..> {dst}", issues
    if edge_type == "unknown":
        issues.append(
            _issue(
                "PUML_UNKNOWN_EDGE_TYPE",
                "warning",
                f"edge '{edge_id}' type unknown; exported as sequential",
                entity_type="edge",
                entity_id=edge_id,
            )
        )
        return f"{src} --> {dst}", issues

    issues.append(
        _issue(
            "PUML_UNSUPPORTED_EDGE_TYPE",
            "warning",
            f"edge '{edge_id}' type '{edge_type}' exported as sequential",
            entity_type="edge",
            entity_id=edge_id,
        )
    )
    return f"{src} --> {dst}", issues


def _sorted_lanes(lanes: List[Any]) -> List[Dict[str, Any]]:
    normalized = [x for x in lanes if isinstance(x, dict) and isinstance(x.get("id"), str)]
    return sorted(
        normalized,
        key=lambda x: (
            x.get("order") if isinstance(x.get("order"), int) else 0,
            str(x.get("id")),
        ),
    )


def _sorted_nodes(nodes: List[Any]) -> List[Dict[str, Any]]:
    normalized = [x for x in nodes if isinstance(x, dict)]
    return sorted(normalized, key=lambda x: str(x.get("id")))


def _sorted_edges(edges: List[Any]) -> List[Dict[str, Any]]:
    normalized = [x for x in edges if isinstance(x, dict)]
    return sorted(normalized, key=lambda x: (str(x.get("from")), str(x.get("to")), str(x.get("id"))))


def _safe_label(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).replace('"', '\\"').replace("\n", " ").replace("\r", " ")
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

