from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from text_to_diagram.plantuml_exporter import export_plantuml


def test_export_plantuml_generates_deterministic_puml_with_lanes():
    ir = {
        "nodes": [
            {"id": "n2", "type": "shape", "role": "action", "text": "Проверка", "lane_id": "l2"},
            {"id": "n1", "type": "shape", "role": "start", "text": "Старт", "lane_id": "l1"},
            {"id": "n3", "type": "shape", "role": "end", "text": "Финиш", "lane_id": "l2"},
        ],
        "edges": [
            {"id": "e2", "from": "n2", "to": "n3", "type": "sequential", "text": None},
            {"id": "e1", "from": "n1", "to": "n2", "type": "sequential", "text": None},
        ],
        "lanes": [
            {"id": "l2", "name": "Менеджер", "order": 1},
            {"id": "l1", "name": "Клиент", "order": 0},
        ],
        "meta": {"direction": "LR"},
        "issues": [],
    }
    out = export_plantuml(ir)
    text = out["puml"]
    assert out["status"] == "ok"
    assert text.startswith("@startuml")
    assert "left to right direction" in text
    assert 'state "Клиент" as lane_l1 {' in text
    assert 'state "Менеджер" as lane_l2 {' in text
    assert 'state "Старт" as n1 <<start>>' in text
    assert 'state "Проверка" as n2' in text
    assert 'state "Финиш" as n3 <<end>>' in text
    assert "n1 --> n2" in text
    assert "n2 --> n3" in text
    assert text.strip().endswith("@enduml")


def test_export_plantuml_reports_unknown_role_and_broken_edge():
    ir = {
        "nodes": [
            {"id": "n1", "type": "shape", "role": "start", "text": "Старт", "lane_id": None},
            {"id": "n2", "type": "shape", "role": "mystery", "text": "Шаг", "lane_id": None},
        ],
        "edges": [
            {"id": "e1", "from": "n1", "to": "n2", "type": "unknown", "text": None},
            {"id": "e2", "from": "n2", "to": "n3", "type": "sequential", "text": None},
        ],
        "lanes": [],
        "meta": {"direction": "TB"},
        "issues": [],
    }
    out = export_plantuml(ir)
    assert out["status"] == "degraded"
    codes = {x["code"] for x in out["issues"]}
    assert "PUML_UNKNOWN_NODE_ROLE" in codes
    assert "PUML_UNKNOWN_EDGE_TYPE" in codes
    assert "PUML_EDGE_BROKEN_REF" in codes
    assert "n2 --> n3" not in out["puml"]


def test_export_plantuml_defaults_direction_for_invalid_meta():
    ir = {
        "nodes": [{"id": "n1", "type": "shape", "role": "start", "text": "Старт", "lane_id": None}],
        "edges": [],
        "lanes": [],
        "meta": {"direction": "DIAGONAL"},
        "issues": [],
    }
    out = export_plantuml(ir)
    assert out["status"] == "degraded"
    assert "left to right direction" in out["puml"]
    assert any(x["code"] == "PUML_DEFAULT_DIRECTION" for x in out["issues"])

