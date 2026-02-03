from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
CAMUNDA_NS = "http://camunda.org/schema/1.0/bpmn"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
BPMNDI_NS = "http://www.omg.org/spec/BPMN/20100524/DI"
DC_NS = "http://www.omg.org/spec/DD/20100524/DC"
DI_NS = "http://www.omg.org/spec/DD/20100524/DI"
XMLNS = {
    "bpmn": BPMN_NS,
    "camunda": CAMUNDA_NS,
    "xsi": XSI_NS,
    "bpmndi": BPMNDI_NS,
    "dc": DC_NS,
    "di": DI_NS,
}

ALLOWED_DIRECTIONS = {"LR", "TB"}


def export_bpmn(ir: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    if not isinstance(ir, dict):
        return {
            "status": "degraded",
            "bpmn_xml": _fallback_xml(),
            "issues": [_issue("BPMN_INVALID_IR", "error", "IR must be object")],
            "meta": {"exporter": "bpmn", "process_id": "Process_1"},
        }

    for k, v in XMLNS.items():
        ET.register_namespace(k, v)

    direction = ir.get("meta", {}).get("direction") if isinstance(ir.get("meta"), dict) else None
    if direction not in ALLOWED_DIRECTIONS:
        direction = "LR"
        issues.append(_issue("BPMN_DEFAULT_DIRECTION", "warning", "direction defaulted to LR"))

    nodes = _sorted_nodes(ir.get("nodes") if isinstance(ir.get("nodes"), list) else [])
    edges = _sorted_edges(ir.get("edges") if isinstance(ir.get("edges"), list) else [])
    lanes = _sorted_lanes(ir.get("lanes") if isinstance(ir.get("lanes"), list) else [])

    definitions = ET.Element(
        _bpmn("definitions"),
        {
            "id": "Definitions_1",
            "targetNamespace": "https://megaschool.local/text-to-diagram",
            f"{{{CAMUNDA_NS}}}modelerVersion": "mvp",
        },
    )
    process = ET.SubElement(
        definitions,
        _bpmn("process"),
        {
            "id": "Process_1",
            "name": "Generated Process",
            "isExecutable": "false",
        },
    )

    lane_ids = {x.get("id") for x in lanes if isinstance(x.get("id"), str)}
    node_elems: Dict[str, ET.Element] = {}
    node_lane: Dict[str, Optional[str]] = {}
    node_ids: Set[str] = set()
    flow_refs_by_lane: Dict[str, List[str]] = defaultdict(list)

    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            issues.append(_issue("BPMN_NODE_ID_INVALID", "warning", "node skipped due to invalid id"))
            continue
        if node_id in node_ids:
            issues.append(
                _issue(
                    "BPMN_DUPLICATE_NODE_ID",
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
            lane_id = None
            issues.append(
                _issue(
                    "BPMN_NODE_UNKNOWN_LANE",
                    "warning",
                    f"node '{node_id}' lane is unknown; node stays in process scope",
                    entity_type="node",
                    entity_id=node_id,
                )
            )
        node_lane[node_id] = lane_id
        if lane_id is not None:
            flow_refs_by_lane[lane_id].append(node_id)

        elem_name, role_issues = _bpmn_element_for_role(node.get("role"), node_id)
        issues.extend(role_issues)
        node_name = _safe_text(node.get("text")) or node_id
        node_el = ET.SubElement(process, _bpmn(elem_name), {"id": node_id, "name": node_name})
        node_elems[node_id] = node_el

    if lanes:
        lane_set = ET.SubElement(process, _bpmn("laneSet"), {"id": "LaneSet_1"})
        for lane in lanes:
            lane_id = lane["id"]
            lane_el = ET.SubElement(
                lane_set,
                _bpmn("lane"),
                {"id": lane_id, "name": _safe_text(lane.get("name")) or lane_id},
            )
            for node_id in sorted(flow_refs_by_lane.get(lane_id, [])):
                ET.SubElement(lane_el, _bpmn("flowNodeRef")).text = node_id

    incoming_by_node: Dict[str, List[str]] = defaultdict(list)
    outgoing_by_node: Dict[str, List[str]] = defaultdict(list)
    rendered_flow_ids: Set[str] = set()
    rendered_flow_endpoints: Dict[str, Tuple[str, str]] = {}

    for idx, edge in enumerate(edges, start=1):
        edge_id = edge.get("id") if isinstance(edge.get("id"), str) and edge.get("id") else f"e{idx}"
        if edge_id in rendered_flow_ids:
            edge_id = f"{edge_id}_{idx}"
        src = edge.get("from")
        dst = edge.get("to")
        if src not in node_ids or dst not in node_ids:
            issues.append(
                _issue(
                    "BPMN_EDGE_BROKEN_REF",
                    "warning",
                    f"edge '{edge_id}' skipped due to unresolved node reference",
                    entity_type="edge",
                    entity_id=edge_id,
                )
            )
            continue

        edge_type = edge.get("type")
        label = _safe_text(edge.get("text"))
        flow_tag, flow_issues = _bpmn_flow_for_edge_type(edge_type, edge_id)
        issues.extend(flow_issues)

        flow_el = ET.SubElement(
            process,
            _bpmn(flow_tag),
            {"id": edge_id, "sourceRef": src, "targetRef": dst},
        )
        if label:
            flow_el.set("name", label)
        if flow_tag == "sequenceFlow" and edge_type == "conditional" and label:
            cond = ET.SubElement(
                flow_el,
                _bpmn("conditionExpression"),
                {f"{{{XSI_NS}}}type": "bpmn:tFormalExpression"},
            )
            cond.text = label

        rendered_flow_ids.add(edge_id)
        rendered_flow_endpoints[edge_id] = (src, dst)
        outgoing_by_node[src].append(edge_id)
        incoming_by_node[dst].append(edge_id)

    for node_id in sorted(node_elems.keys()):
        node_el = node_elems[node_id]
        for flow_id in sorted(incoming_by_node.get(node_id, [])):
            ET.SubElement(node_el, _bpmn("incoming")).text = flow_id
        for flow_id in sorted(outgoing_by_node.get(node_id, [])):
            ET.SubElement(node_el, _bpmn("outgoing")).text = flow_id

    # BPMN DI section (required by bpmn-js renderer).
    node_di_bounds = _compute_node_layout(
        direction=direction,
        node_ids=sorted(node_elems.keys()),
        node_lane=node_lane,
        lane_order={lane["id"]: idx for idx, lane in enumerate(lanes)},
        node_elems=node_elems,
    )
    bpmn_diagram = ET.SubElement(definitions, _bpmndi("BPMNDiagram"), {"id": "BPMNDiagram_1"})
    bpmn_plane = ET.SubElement(
        bpmn_diagram,
        _bpmndi("BPMNPlane"),
        {"id": "BPMNPlane_1", "bpmnElement": "Process_1"},
    )

    for node_id in sorted(node_di_bounds.keys()):
        shape = ET.SubElement(
            bpmn_plane,
            _bpmndi("BPMNShape"),
            {"id": f"{node_id}_di", "bpmnElement": node_id},
        )
        b = node_di_bounds[node_id]
        ET.SubElement(
            shape,
            _dc("Bounds"),
            {
                "x": _fmt(b["x"]),
                "y": _fmt(b["y"]),
                "width": _fmt(b["w"]),
                "height": _fmt(b["h"]),
            },
        )

    for edge_idx, edge_id in enumerate(sorted(rendered_flow_endpoints.keys()), start=1):
        src, dst = rendered_flow_endpoints[edge_id]
        if src not in node_di_bounds or dst not in node_di_bounds:
            continue
        edge_di = ET.SubElement(
            bpmn_plane,
            _bpmndi("BPMNEdge"),
            {"id": f"{edge_id}_di", "bpmnElement": edge_id},
        )
        waypoints = _build_edge_waypoints(
            direction=direction,
            src=node_di_bounds[src],
            dst=node_di_bounds[dst],
            edge_idx=edge_idx,
        )
        for x, y in waypoints:
            ET.SubElement(edge_di, _di("waypoint"), {"x": _fmt(x), "y": _fmt(y)})

    issues.extend(
        _validate_di_section(
            definitions=definitions,
            flow_nodes_count=len(node_elems),
            rendered_edges_count=len(rendered_flow_ids),
        )
    )

    xml_text = ET.tostring(definitions, encoding="unicode")
    try:
        ET.fromstring(xml_text)
    except Exception:
        issues.append(_issue("BPMN_XML_NOT_WELL_FORMED", "error", "generated XML is not well formed"))

    status = "degraded" if issues else "ok"
    return {
        "status": status,
        "bpmn_xml": xml_text,
        "issues": issues,
        "meta": {
            "exporter": "bpmn",
            "process_id": "Process_1",
            "direction": direction,
            "nodes_count": len(node_ids),
            "edges_count": len(rendered_flow_ids),
            "lanes_count": len(lanes),
            "di_shapes_count": len(node_di_bounds),
            "di_edges_count": len(rendered_flow_endpoints),
        },
    }


def _bpmn_element_for_role(role: Any, node_id: str) -> Tuple[str, List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    if role == "start":
        return "startEvent", issues
    if role == "end":
        return "endEvent", issues
    if role == "action":
        return "task", issues
    if role == "decision":
        return "exclusiveGateway", issues
    if role == "parallel":
        return "parallelGateway", issues
    if role == "inclusive":
        return "inclusiveGateway", issues
    if role == "event_intermediate":
        return "intermediateThrowEvent", issues
    issues.append(
        _issue(
            "BPMN_UNKNOWN_NODE_ROLE",
            "warning",
            f"node '{node_id}' role '{role}' exported as task",
            entity_type="node",
            entity_id=node_id,
        )
    )
    return "task", issues


def _bpmn_flow_for_edge_type(edge_type: Any, edge_id: str) -> Tuple[str, List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    if edge_type in {"sequential", "conditional"}:
        return "sequenceFlow", issues
    if edge_type == "association":
        return "association", issues
    if edge_type == "message":
        issues.append(
            _issue(
                "BPMN_MESSAGE_FLOW_DOWNCAST",
                "warning",
                f"edge '{edge_id}' message flow exported as sequenceFlow in MVP",
                entity_type="edge",
                entity_id=edge_id,
            )
        )
        return "sequenceFlow", issues
    if edge_type == "unknown":
        issues.append(
            _issue(
                "BPMN_UNKNOWN_EDGE_TYPE",
                "warning",
                f"edge '{edge_id}' type unknown; exported as sequenceFlow",
                entity_type="edge",
                entity_id=edge_id,
            )
        )
        return "sequenceFlow", issues
    issues.append(
        _issue(
            "BPMN_UNSUPPORTED_EDGE_TYPE",
            "warning",
            f"edge '{edge_id}' type '{edge_type}' exported as sequenceFlow",
            entity_type="edge",
            entity_id=edge_id,
        )
    )
    return "sequenceFlow", issues


def _sorted_lanes(lanes: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = [x for x in lanes if isinstance(x, dict) and isinstance(x.get("id"), str)]
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
    return sorted(normalized, key=lambda x: (str(x.get("from")), str(x.get("to")), str(x.get("id"))))


def _bpmn(name: str) -> str:
    return f"{{{BPMN_NS}}}{name}"


def _bpmndi(name: str) -> str:
    return f"{{{BPMNDI_NS}}}{name}"


def _dc(name: str) -> str:
    return f"{{{DC_NS}}}{name}"


def _di(name: str) -> str:
    return f"{{{DI_NS}}}{name}"


def _safe_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = " ".join(value.replace("\n", " ").replace("\r", " ").split())
    return value if value else None


def _compute_node_layout(
    direction: str,
    node_ids: List[str],
    node_lane: Dict[str, Optional[str]],
    lane_order: Dict[str, int],
    node_elems: Dict[str, ET.Element],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    lane_step = 220.0
    main_step = 220.0
    origin_x = 120.0
    origin_y = 100.0

    for idx, node_id in enumerate(node_ids):
        lane_id = node_lane.get(node_id)
        lane_idx = lane_order.get(lane_id, 0) if lane_id is not None else 0
        node_tag = node_elems[node_id].tag
        w, h = _size_for_tag(node_tag)

        if direction == "LR":
            x = origin_x + idx * main_step
            y = origin_y + lane_idx * lane_step
        else:
            x = origin_x + lane_idx * lane_step
            y = origin_y + idx * main_step
        out[node_id] = {"x": x, "y": y, "w": w, "h": h}
    return out


def _size_for_tag(tag: str) -> Tuple[float, float]:
    if tag.endswith("startEvent") or tag.endswith("endEvent") or tag.endswith("intermediateThrowEvent"):
        return 36.0, 36.0
    if tag.endswith("exclusiveGateway") or tag.endswith("parallelGateway") or tag.endswith("inclusiveGateway"):
        return 50.0, 50.0
    return 140.0, 80.0


def _center_point(bounds: Dict[str, float]) -> Tuple[float, float]:
    return bounds["x"] + bounds["w"] / 2.0, bounds["y"] + bounds["h"] / 2.0


def _build_edge_waypoints(
    direction: str,
    src: Dict[str, float],
    dst: Dict[str, float],
    edge_idx: int,
) -> List[Tuple[float, float]]:
    sx, sy = _center_point(src)
    tx, ty = _center_point(dst)

    if direction == "TB":
        start = (sx, src["y"] + src["h"])
        end = (tx, dst["y"])
        # Forward edge (top->bottom): orthogonal elbow with a middle Y.
        if ty >= sy:
            mid_y = (start[1] + end[1]) / 2.0
            return [start, (sx, mid_y), (tx, mid_y), end]
        # Back edge: route to the right to avoid crossing node bodies.
        offset = 60.0 + (edge_idx % 5) * 24.0
        track_x = max(src["x"] + src["w"], dst["x"] + dst["w"]) + offset
        return [start, (track_x, start[1]), (track_x, end[1]), end]

    # LR layout by default.
    start = (src["x"] + src["w"], sy)
    end = (dst["x"], ty)
    # Forward edge (left->right): orthogonal elbow with a middle X.
    if tx >= sx:
        mid_x = (start[0] + end[0]) / 2.0
        return [start, (mid_x, start[1]), (mid_x, end[1]), end]

    # Back edge: route above nodes to keep diagram readable.
    offset = 50.0 + (edge_idx % 5) * 18.0
    track_y = min(src["y"], dst["y"]) - offset
    return [start, (start[0], track_y), (end[0], track_y), end]


def _fmt(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _validate_di_section(
    definitions: ET.Element,
    flow_nodes_count: int,
    rendered_edges_count: int,
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    diagram = definitions.find(f".//{{{BPMNDI_NS}}}BPMNDiagram")
    if diagram is None:
        issues.append(_issue("BPMN_DI_MISSING_DIAGRAM", "error", "BPMNDiagram element is missing"))
        return issues

    plane = diagram.find(f".//{{{BPMNDI_NS}}}BPMNPlane")
    if plane is None:
        issues.append(_issue("BPMN_DI_MISSING_PLANE", "error", "BPMNPlane element is missing"))
        return issues

    shape_count = len(plane.findall(f".//{{{BPMNDI_NS}}}BPMNShape"))
    edge_count = len(plane.findall(f".//{{{BPMNDI_NS}}}BPMNEdge"))
    if shape_count < flow_nodes_count:
        issues.append(
            _issue(
                "BPMN_DI_SHAPE_COUNT_MISMATCH",
                "error",
                f"BPMNShape count {shape_count} is less than flow nodes {flow_nodes_count}",
            )
        )
    if edge_count < rendered_edges_count:
        issues.append(
            _issue(
                "BPMN_DI_EDGE_COUNT_MISMATCH",
                "error",
                f"BPMNEdge count {edge_count} is less than rendered edges {rendered_edges_count}",
            )
        )
    return issues


def _fallback_xml() -> str:
    return (
        f'<bpmn:definitions xmlns:bpmn="{BPMN_NS}" xmlns:camunda="{CAMUNDA_NS}" '
        f'xmlns:bpmndi="{BPMNDI_NS}" xmlns:dc="{DC_NS}" xmlns:di="{DI_NS}" '
        'id="Definitions_1" targetNamespace="https://megaschool.local/text-to-diagram">'
        '<bpmn:process id="Process_1" name="Generated Process" isExecutable="false" />'
        "</bpmn:definitions>"
    )


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
