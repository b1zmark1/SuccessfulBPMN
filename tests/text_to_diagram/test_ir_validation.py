from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from text_to_diagram.ir_validation import IRValidationPolicy, validate_and_normalize_ir


def test_validate_and_normalize_ir_canonicalizes_ids_and_meta():
    ir = {
        "nodes": [
            {"id": "A", "type": "shape", "role": "start", "text": "Старт", "lane_id": "L2"},
            {"id": "B", "type": "shape", "role": "end", "text": "Финиш", "lane_id": "L1"},
        ],
        "edges": [{"id": "X", "from": "A", "to": "B", "type": "sequential"}],
        "lanes": [
            {"id": "L1", "name": "Менеджер", "order": 1},
            {"id": "L2", "name": "Клиент", "order": 0},
        ],
        "meta": {"direction": "TB"},
        "issues": [],
    }

    out = validate_and_normalize_ir(ir)
    normalized = out["normalized_ir"]

    assert out["hard_fail"] is False
    assert normalized["meta"]["schema_version"] == "process-ir.v1"
    assert normalized["meta"]["source"] == "text_to_diagram"
    assert [x["id"] for x in normalized["lanes"]] == ["l1", "l2"]
    assert [x["id"] for x in normalized["nodes"]] == ["n1", "n2"]
    assert [x["id"] for x in normalized["edges"]] == ["e1"]
    assert normalized["edges"][0]["from"] == "n1"
    assert normalized["edges"][0]["to"] == "n2"


def test_validate_and_normalize_ir_drops_broken_edges_and_adds_issues():
    ir = {
        "nodes": [{"id": "N1", "type": "shape", "role": "start", "text": None, "lane_id": None}],
        "edges": [{"id": "E1", "from": "N1", "to": "N2", "type": "sequential"}],
        "lanes": [],
        "meta": {},
        "issues": [],
    }
    out = validate_and_normalize_ir(ir)
    normalized = out["normalized_ir"]

    assert len(normalized["edges"]) == 0
    codes = {x["code"] for x in out["issues"]}
    assert "IR_EDGE_BROKEN_REF" in codes
    assert "IR_MISSING_END" in codes


def test_validate_and_normalize_ir_hard_fails_on_root_type_with_default_policy():
    out = validate_and_normalize_ir("not-an-object")
    assert out["hard_fail"] is True
    assert out["status"] == "hard_fail"
    assert "IR_ROOT_NOT_OBJECT" in out["hard_fail_codes"]


def test_validate_and_normalize_ir_can_disable_hard_fail_policy():
    policy = IRValidationPolicy(hard_fail_on_codes=set())
    out = validate_and_normalize_ir("not-an-object", policy=policy)
    assert out["hard_fail"] is False
    assert out["status"] == "degraded"

