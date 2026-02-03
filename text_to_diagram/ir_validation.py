from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


ALLOWED_NODE_TYPES: Set[str] = {"shape", "container", "text", "flow"}
ALLOWED_NODE_ROLES: Set[str] = {
    "start",
    "end",
    "action",
    "decision",
    "parallel",
    "inclusive",
    "event_intermediate",
    "unknown",
}
ALLOWED_EDGE_TYPES: Set[str] = {
    "sequential",
    "conditional",
    "message",
    "association",
    "unknown",
}


@dataclass(frozen=True)
class IRValidationPolicy:
    hard_fail_on_codes: Set[str] = field(default_factory=lambda: {"IR_ROOT_NOT_OBJECT"})


class IRValidationError(RuntimeError):
    pass


def validate_and_normalize_ir(
    ir: Any,
    policy: Optional[IRValidationPolicy] = None,
) -> Dict[str, Any]:
    if policy is None:
        policy = IRValidationPolicy()

    normalized, normalize_issues = normalize_ir(ir)
    validate_issues = validate_ir(normalized)

    all_issues = _merge_issues(normalized.get("issues", []), normalize_issues, validate_issues)
    normalized["issues"] = all_issues

    triggered_hard_fail_codes = sorted(
        {
            issue["code"]
            for issue in all_issues
            if issue.get("code") in policy.hard_fail_on_codes
        }
    )
    hard_fail = len(triggered_hard_fail_codes) > 0

    return {
        "status": "hard_fail" if hard_fail else ("degraded" if all_issues else "ok"),
        "hard_fail": hard_fail,
        "hard_fail_codes": triggered_hard_fail_codes,
        "normalized_ir": normalized,
        "issues": all_issues,
    }


def normalize_ir(ir: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    if not isinstance(ir, dict):
        issues.append(_issue("IR_ROOT_NOT_OBJECT", "error", "IR root must be object"))
        ir = {}

    root = dict(ir)

    nodes_raw = root.get("nodes")
    if not isinstance(nodes_raw, list):
        nodes_raw = []
        issues.append(_issue("IR_REPAIR_NODES", "warning", "nodes replaced with []"))

    edges_raw = root.get("edges")
    if not isinstance(edges_raw, list):
        edges_raw = []
        issues.append(_issue("IR_REPAIR_EDGES", "warning", "edges replaced with []"))

    lanes_raw = root.get("lanes")
    if not isinstance(lanes_raw, list):
        lanes_raw = []
        issues.append(_issue("IR_REPAIR_LANES", "warning", "lanes replaced with []"))

    meta_raw = root.get("meta")
    if not isinstance(meta_raw, dict):
        meta_raw = {}
        issues.append(_issue("IR_REPAIR_META", "warning", "meta replaced with {}"))

    issues_raw = root.get("issues")
    if not isinstance(issues_raw, list):
        issues_raw = []
        issues.append(_issue("IR_REPAIR_ISSUES", "warning", "issues replaced with []"))

    lanes, lane_map, lane_issues = _normalize_lanes(lanes_raw)
    issues.extend(lane_issues)
    nodes, node_map, node_issues = _normalize_nodes(nodes_raw, lane_map)
    issues.extend(node_issues)
    edges, edge_issues = _normalize_edges(edges_raw, node_map)
    issues.extend(edge_issues)

    meta = _normalize_meta(meta_raw)
    merged_root_issues = [x for x in issues_raw if isinstance(x, dict)]

    normalized = {
        "nodes": nodes,
        "edges": edges,
        "lanes": lanes,
        "meta": meta,
        "issues": merged_root_issues,
    }
    return normalized, issues


def validate_ir(ir: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    nodes = ir.get("nodes", [])
    edges = ir.get("edges", [])
    lanes = ir.get("lanes", [])

    node_ids = [n.get("id") for n in nodes if isinstance(n, dict)]
    edge_ids = [e.get("id") for e in edges if isinstance(e, dict)]
    lane_ids = [l.get("id") for l in lanes if isinstance(l, dict)]

    _check_unique_ids(node_ids, "node", issues)
    _check_unique_ids(edge_ids, "edge", issues)
    _check_unique_ids(lane_ids, "lane", issues)

    node_id_set = {x for x in node_ids if isinstance(x, str)}
    lane_id_set = {x for x in lane_ids if isinstance(x, str)}

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = edge.get("from")
        dst = edge.get("to")
        if src not in node_id_set or dst not in node_id_set:
            issues.append(
                _issue(
                    "IR_EDGE_BROKEN_REF",
                    "warning",
                    f"edge '{edge.get('id')}' has unresolved from/to reference",
                    entity_type="edge",
                    entity_id=edge.get("id"),
                )
            )
        if src == dst and isinstance(src, str):
            issues.append(
                _issue(
                    "IR_SELF_LOOP_EDGE",
                    "warning",
                    f"edge '{edge.get('id')}' references same node in from/to",
                    entity_type="edge",
                    entity_id=edge.get("id"),
                )
            )

    for node in nodes:
        if not isinstance(node, dict):
            continue
        lane_id = node.get("lane_id")
        if lane_id is not None and lane_id not in lane_id_set:
            issues.append(
                _issue(
                    "IR_NODE_BROKEN_LANE_REF",
                    "warning",
                    f"node '{node.get('id')}' references unknown lane_id",
                    entity_type="node",
                    entity_id=node.get("id"),
                )
            )

    start_count = sum(1 for n in nodes if isinstance(n, dict) and n.get("role") == "start")
    end_count = sum(1 for n in nodes if isinstance(n, dict) and n.get("role") == "end")
    if start_count == 0:
        issues.append(_issue("IR_MISSING_START", "warning", "IR has no start node"))
    if end_count == 0:
        issues.append(_issue("IR_MISSING_END", "warning", "IR has no end node"))

    cycle = _lane_parent_cycle(lanes)
    if cycle:
        issues.append(_issue("IR_LANE_PARENT_CYCLE", "warning", "lane hierarchy has cycle"))

    return issues


def _normalize_lanes(raw_lanes: List[Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    prepared: List[Tuple[str, Dict[str, Any]]] = []

    for idx, item in enumerate(raw_lanes):
        if not isinstance(item, dict):
            issues.append(_issue("IR_LANE_INVALID_ITEM", "warning", "lane item ignored"))
            continue
        old_id = _safe_id(item.get("id"), f"_lane_{idx + 1}")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            name = f"Lane {idx + 1}"
            issues.append(
                _issue(
                    "IR_LANE_DEFAULT_NAME",
                    "warning",
                    "lane name was missing and replaced",
                    entity_type="lane",
                    entity_id=old_id,
                )
            )
        order = item.get("order")
        if not isinstance(order, int) or order < 0:
            order = idx
            issues.append(
                _issue(
                    "IR_LANE_DEFAULT_ORDER",
                    "warning",
                    "lane order was missing and replaced",
                    entity_type="lane",
                    entity_id=old_id,
                )
            )
        prepared.append(
            (
                old_id,
                {
                    "id": old_id,
                    "name": name.strip(),
                    "order": order,
                    "pool_id": item.get("pool_id") if isinstance(item.get("pool_id"), str) else None,
                    "parent_lane_id": (
                        item.get("parent_lane_id") if isinstance(item.get("parent_lane_id"), str) else None
                    ),
                },
            )
        )

    prepared.sort(key=lambda x: (x[1]["order"], x[1]["name"], x[0]))
    lane_map: Dict[str, str] = {}
    new_lanes: List[Dict[str, Any]] = []
    for idx, (old_id, lane) in enumerate(prepared, start=1):
        new_id = f"l{idx}"
        lane_map[old_id] = new_id
        lane["id"] = new_id
        new_lanes.append(lane)

    for lane in new_lanes:
        parent_old = lane.get("parent_lane_id")
        if parent_old is None:
            continue
        parent_new = lane_map.get(parent_old)
        if parent_new is None:
            lane["parent_lane_id"] = None
            issues.append(
                _issue(
                    "IR_LANE_BROKEN_PARENT_REF",
                    "warning",
                    "lane parent reference removed during normalization",
                    entity_type="lane",
                    entity_id=lane["id"],
                )
            )
        else:
            lane["parent_lane_id"] = parent_new

    return new_lanes, lane_map, issues


def _normalize_nodes(
    raw_nodes: List[Any],
    lane_map: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    prepared: List[Tuple[str, Dict[str, Any]]] = []

    for idx, item in enumerate(raw_nodes):
        if not isinstance(item, dict):
            issues.append(_issue("IR_NODE_INVALID_ITEM", "warning", "node item ignored"))
            continue
        old_id = _safe_id(item.get("id"), f"_node_{idx + 1}")

        node_type = item.get("type")
        if node_type not in ALLOWED_NODE_TYPES:
            node_type = "shape"
            issues.append(
                _issue(
                    "IR_NODE_TYPE_DEFAULTED",
                    "warning",
                    "node type defaulted to 'shape'",
                    entity_type="node",
                    entity_id=old_id,
                )
            )

        role = item.get("role")
        if role not in ALLOWED_NODE_ROLES:
            role = "unknown"
            issues.append(
                _issue(
                    "IR_NODE_ROLE_DEFAULTED",
                    "warning",
                    "node role defaulted to 'unknown'",
                    entity_type="node",
                    entity_id=old_id,
                )
            )

        text = item.get("text")
        if not (isinstance(text, str) or text is None):
            text = None
            issues.append(
                _issue(
                    "IR_NODE_TEXT_DEFAULTED",
                    "warning",
                    "node text defaulted to null",
                    entity_type="node",
                    entity_id=old_id,
                )
            )

        lane_id = item.get("lane_id")
        if isinstance(lane_id, str):
            lane_id = lane_map.get(lane_id)
            if lane_id is None:
                issues.append(
                    _issue(
                        "IR_NODE_BROKEN_LANE_REF",
                        "warning",
                        "node lane reference removed during normalization",
                        entity_type="node",
                        entity_id=old_id,
                    )
                )
        else:
            lane_id = None

        container_id = item.get("container_id")
        if not (isinstance(container_id, str) or container_id is None):
            container_id = None

        prepared.append(
            (
                old_id,
                {
                    "id": old_id,
                    "type": node_type,
                    "role": role,
                    "text": text,
                    "lane_id": lane_id,
                    "container_id": container_id,
                },
            )
        )

    prepared.sort(key=lambda x: x[0])
    node_map: Dict[str, str] = {}
    new_nodes: List[Dict[str, Any]] = []
    for idx, (old_id, node) in enumerate(prepared, start=1):
        new_id = f"n{idx}"
        node_map[old_id] = new_id
        node["id"] = new_id
        new_nodes.append(node)

    return new_nodes, node_map, issues


def _normalize_edges(
    raw_edges: List[Any],
    node_map: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    prepared: List[Tuple[str, Dict[str, Any]]] = []

    for idx, item in enumerate(raw_edges):
        if not isinstance(item, dict):
            issues.append(_issue("IR_EDGE_INVALID_ITEM", "warning", "edge item ignored"))
            continue

        old_id = _safe_id(item.get("id"), f"_edge_{idx + 1}")
        src_old = item.get("from")
        dst_old = item.get("to")
        if not isinstance(src_old, str) or not isinstance(dst_old, str):
            issues.append(
                _issue(
                    "IR_EDGE_MISSING_ENDPOINT",
                    "warning",
                    "edge ignored due to missing from/to",
                    entity_type="edge",
                    entity_id=old_id,
                )
            )
            continue

        src = node_map.get(src_old)
        dst = node_map.get(dst_old)
        if src is None or dst is None:
            issues.append(
                _issue(
                    "IR_EDGE_BROKEN_REF",
                    "warning",
                    "edge removed due to unresolved node reference",
                    entity_type="edge",
                    entity_id=old_id,
                )
            )
            continue

        edge_type = item.get("type")
        if edge_type not in ALLOWED_EDGE_TYPES:
            edge_type = "unknown"
            issues.append(
                _issue(
                    "IR_EDGE_TYPE_DEFAULTED",
                    "warning",
                    "edge type defaulted to 'unknown'",
                    entity_type="edge",
                    entity_id=old_id,
                )
            )

        text = item.get("text")
        if not (isinstance(text, str) or text is None):
            text = None

        prepared.append(
            (
                old_id,
                {
                    "id": old_id,
                    "from": src,
                    "to": dst,
                    "type": edge_type,
                    "text": text,
                },
            )
        )

    prepared.sort(key=lambda x: (x[1]["from"], x[1]["to"], x[0]))
    new_edges: List[Dict[str, Any]] = []
    for idx, (_old_id, edge) in enumerate(prepared, start=1):
        edge["id"] = f"e{idx}"
        new_edges.append(edge)

    return new_edges, issues


def _normalize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    direction = meta.get("direction")
    if direction not in {"LR", "TB"}:
        direction = "LR"
    return {
        **meta,
        "schema_version": "process-ir.v1",
        "direction": direction,
        "source": "text_to_diagram",
        "language": "ru",
    }


def _check_unique_ids(ids: List[Any], entity_type: str, issues: List[Dict[str, Any]]) -> None:
    seen: Set[str] = set()
    for value in ids:
        if not isinstance(value, str):
            issues.append(
                _issue(
                    "IR_NON_STRING_ID",
                    "warning",
                    f"{entity_type} id must be string",
                    entity_type=entity_type,
                    entity_id=None,
                )
            )
            continue
        if value in seen:
            issues.append(
                _issue(
                    "IR_DUPLICATE_ID",
                    "warning",
                    f"duplicate {entity_type} id '{value}'",
                    entity_type=entity_type,
                    entity_id=value,
                )
            )
        seen.add(value)


def _lane_parent_cycle(lanes: List[Dict[str, Any]]) -> bool:
    graph: Dict[str, Optional[str]] = {
        lane.get("id"): lane.get("parent_lane_id")
        for lane in lanes
        if isinstance(lane, dict) and isinstance(lane.get("id"), str)
    }
    visiting: Set[str] = set()
    visited: Set[str] = set()

    def dfs(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        parent = graph.get(node)
        if isinstance(parent, str) and parent in graph and dfs(parent):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(dfs(node) for node in graph.keys())


def _safe_id(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _merge_issues(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[Any, Any, Any, Any, Any]] = set()
    for group in groups:
        for issue in group:
            if not isinstance(issue, dict):
                continue
            key = (
                issue.get("code"),
                issue.get("severity"),
                issue.get("message"),
                issue.get("entity_type"),
                issue.get("entity_id"),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "code": issue.get("code", "IR_UNKNOWN_ISSUE"),
                    "severity": issue.get("severity", "warning"),
                    "message": issue.get("message", ""),
                    "entity_type": issue.get("entity_type"),
                    "entity_id": issue.get("entity_id"),
                }
            )
    return out


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

