from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from text_to_diagram.mermaid_exporter import export_mermaid


def test_export_mermaid_builds_deterministic_output_with_lanes():
    ir = {
        "nodes": [
            {"id": "n2", "type": "shape", "role": "action", "text": "Проверить", "lane_id": "l2"},
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

    out = export_mermaid(ir)
    assert out["status"] == "ok"
    text = out["mmd"]
    assert text.startswith("flowchart LR")
    assert 'subgraph l1["Клиент"]' in text
    assert 'subgraph l2["Менеджер"]' in text
    assert 'n1(["Старт"])' in text
    assert 'n2["Проверить"]' in text
    assert 'n3(["Финиш"])' in text
    assert "n1 --> n2" in text
    assert "n2 --> n3" in text


def test_export_mermaid_reports_unknown_role_and_broken_edge():
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

    out = export_mermaid(ir)
    assert out["status"] == "degraded"
    codes = {x["code"] for x in out["issues"]}
    assert "MERMAID_UNKNOWN_NODE_ROLE" in codes
    assert "MERMAID_UNKNOWN_EDGE_TYPE" in codes
    assert "MERMAID_EDGE_BROKEN_REF" in codes
    assert "n2 --> n3" not in out["mmd"]


def test_export_mermaid_defaults_direction():
    ir = {
        "nodes": [{"id": "n1", "type": "shape", "role": "start", "text": "Старт", "lane_id": None}],
        "edges": [],
        "lanes": [],
        "meta": {"direction": "DIAGONAL"},
        "issues": [],
    }
    out = export_mermaid(ir)
    assert out["mmd"].startswith("flowchart LR")
    assert any(x["code"] == "MERMAID_DEFAULT_DIRECTION" for x in out["issues"])

