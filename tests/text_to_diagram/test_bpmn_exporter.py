from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from text_to_diagram.bpmn_exporter import BPMN_NS, BPMNDI_NS, export_bpmn


def test_export_bpmn_generates_well_formed_xml_with_lanes_and_flows():
    ir = {
        "nodes": [
            {"id": "n1", "type": "shape", "role": "start", "text": "Старт", "lane_id": "l1"},
            {"id": "n2", "type": "shape", "role": "action", "text": "Проверка", "lane_id": "l1"},
            {"id": "n3", "type": "shape", "role": "decision", "text": "ОК?", "lane_id": "l2"},
            {"id": "n4", "type": "shape", "role": "end", "text": "Финиш", "lane_id": "l2"},
        ],
        "edges": [
            {"id": "e1", "from": "n1", "to": "n2", "type": "sequential", "text": None},
            {"id": "e2", "from": "n2", "to": "n3", "type": "conditional", "text": "да"},
            {"id": "e3", "from": "n3", "to": "n4", "type": "sequential", "text": None},
        ],
        "lanes": [
            {"id": "l2", "name": "Менеджер", "order": 1},
            {"id": "l1", "name": "Клиент", "order": 0},
        ],
        "meta": {"direction": "LR"},
        "issues": [],
    }

    out = export_bpmn(ir)
    assert out["status"] == "ok"
    xml_text = out["bpmn_xml"]
    root = ET.fromstring(xml_text)

    assert root.tag == f"{{{BPMN_NS}}}definitions"
    process = root.find(f".//{{{BPMN_NS}}}process")
    assert process is not None
    assert process.attrib["id"] == "Process_1"
    assert process.find(f".//{{{BPMN_NS}}}laneSet") is not None
    assert process.find(f".//{{{BPMN_NS}}}startEvent") is not None
    assert process.find(f".//{{{BPMN_NS}}}task") is not None
    assert process.find(f".//{{{BPMN_NS}}}exclusiveGateway") is not None
    assert process.find(f".//{{{BPMN_NS}}}endEvent") is not None
    assert len(process.findall(f".//{{{BPMN_NS}}}sequenceFlow")) == 3
    diagram = root.find(f".//{{{BPMNDI_NS}}}BPMNDiagram")
    plane = root.find(f".//{{{BPMNDI_NS}}}BPMNPlane")
    assert diagram is not None
    assert plane is not None
    shapes = root.findall(f".//{{{BPMNDI_NS}}}BPMNShape")
    di_edges = root.findall(f".//{{{BPMNDI_NS}}}BPMNEdge")
    flow_nodes = (
        process.findall(f".//{{{BPMN_NS}}}startEvent")
        + process.findall(f".//{{{BPMN_NS}}}task")
        + process.findall(f".//{{{BPMN_NS}}}exclusiveGateway")
        + process.findall(f".//{{{BPMN_NS}}}parallelGateway")
        + process.findall(f".//{{{BPMN_NS}}}inclusiveGateway")
        + process.findall(f".//{{{BPMN_NS}}}intermediateThrowEvent")
        + process.findall(f".//{{{BPMN_NS}}}endEvent")
    )
    rendered_edges = process.findall(f".//{{{BPMN_NS}}}sequenceFlow") + process.findall(
        f".//{{{BPMN_NS}}}association"
    )
    assert len(shapes) >= len(flow_nodes)
    assert len(di_edges) >= len(rendered_edges)


def test_export_bpmn_reports_issues_for_unknown_role_and_edge_type():
    ir = {
        "nodes": [
            {"id": "n1", "type": "shape", "role": "start", "text": "Старт", "lane_id": None},
            {"id": "n2", "type": "shape", "role": "mystery", "text": "Шаг", "lane_id": None},
        ],
        "edges": [
            {"id": "e1", "from": "n1", "to": "n2", "type": "unknown", "text": None},
            {"id": "e2", "from": "n2", "to": "n3", "type": "message", "text": "msg"},
        ],
        "lanes": [],
        "meta": {"direction": "TB"},
        "issues": [],
    }
    out = export_bpmn(ir)
    assert out["status"] == "degraded"
    codes = {x["code"] for x in out["issues"]}
    assert "BPMN_UNKNOWN_NODE_ROLE" in codes
    assert "BPMN_UNKNOWN_EDGE_TYPE" in codes
    assert "BPMN_EDGE_BROKEN_REF" in codes


def test_export_bpmn_defaults_direction_for_invalid_meta():
    ir = {
        "nodes": [{"id": "n1", "type": "shape", "role": "start", "text": "Старт", "lane_id": None}],
        "edges": [],
        "lanes": [],
        "meta": {"direction": "DIAGONAL"},
        "issues": [],
    }
    out = export_bpmn(ir)
    assert out["meta"]["direction"] == "LR"
    assert any(x["code"] == "BPMN_DEFAULT_DIRECTION" for x in out["issues"])
