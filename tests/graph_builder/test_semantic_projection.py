from __future__ import annotations

from tests.graph_builder.utils import load_fixture

from graph_builder.pipeline import (
    build_graph_from_ensemble,
    run_graph_to_semantic_pipeline,
)
from graph_builder.semantic_contract import (
    check_semantic_projection_contract,
    validate_semantic_projection_contract,
)
from graph_builder.semantic_projection import project_graph_to_semantic


def test_graph_to_semantic_pipeline_contract():
    payload = load_fixture("bpmn_like_01.json")
    out = run_graph_to_semantic_pipeline(payload)

    assert set(out.keys()) == {"semantic_payload", "projection_meta"}
    assert out["projection_meta"]["status"] == "ok"
    assert out["projection_meta"]["contract_valid"] is True
    assert out["projection_meta"]["contract_errors"] == []

    semantic = out["semantic_payload"]
    assert set(semantic.keys()) == {"meta", "steps"}
    assert set(semantic["meta"].keys()) == {
        "schema_version",
        "source_graph_schema_version",
        "direction",
        "warnings",
    }
    assert semantic["meta"]["schema_version"] == "semantic-projection.v1"
    assert semantic["meta"]["direction"] in {"LR", "TB"}

    steps = semantic["steps"]
    assert isinstance(steps, list)
    assert len(steps) > 0

    expected_orders = list(range(1, len(steps) + 1))
    assert [s["order"] for s in steps] == expected_orders

    step_ids = {s["id"] for s in steps}
    for step in steps:
        assert set(step.keys()) == {"id", "order", "role", "text", "next_step_ids"}
        assert step["role"] in {"start", "action", "decision", "parallel", "end"}
        assert step["text"] is None or isinstance(step["text"], str)
        assert isinstance(step["next_step_ids"], list)
        assert len(step["next_step_ids"]) == len(set(step["next_step_ids"]))
        assert all(isinstance(nid, str) and nid in step_ids for nid in step["next_step_ids"])
        # semantic projection must not leak graph geometry
        assert "bbox" not in step
        assert "center" not in step


def test_projection_is_deterministic():
    payload = load_fixture("uml_like_01.json")
    graph = build_graph_from_ensemble(payload)

    first = project_graph_to_semantic(graph)
    second = project_graph_to_semantic(graph)
    assert first == second


def test_semantic_contract_validator_rejects_invalid_payload():
    payload = load_fixture("bpmn_like_01.json")
    graph = build_graph_from_ensemble(payload)
    semantic = project_graph_to_semantic(graph)
    semantic["steps"][0]["role"] = "unknown"

    ok, errors = check_semantic_projection_contract(semantic)
    assert ok is False
    assert any(".role must be one of" in e for e in errors)


def test_semantic_contract_validator_accepts_projection_output():
    payload = load_fixture("noisy_custom_01.json")
    graph = build_graph_from_ensemble(payload)
    semantic = project_graph_to_semantic(graph)
    validate_semantic_projection_contract(semantic)
