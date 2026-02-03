from __future__ import annotations

import pytest

from tests.graph_builder.utils import load_fixture, run_full_pipeline


def _assert_contract(out):
    assert set(out.keys()) == {"meta", "nodes", "edges"}
    assert set(out["meta"].keys()) == {"schema_version", "direction", "warnings"}
    assert out["meta"]["schema_version"] == "graph-builder.v1"
    assert out["meta"]["direction"] in {"LR", "TB"}
    assert isinstance(out["meta"]["warnings"], list)

    node_req = {"id", "type", "bbox", "center", "role", "container_id", "text"}
    edge_req = {"from", "to", "type"}
    for n in out["nodes"]:
        assert set(n.keys()) == node_req
        assert n["type"] in {"shape", "container", "text", "flow"}
        assert n["role"] in {"action", "decision", "start", "end", "unknown"}
    for e in out["edges"]:
        assert set(e.keys()) == edge_req
        assert e["type"] in {"sequential", "conditional", "unknown"}


@pytest.mark.parametrize(
    "fixture_name,expected_direction,expected_nodes,expected_edges",
    [
            ("bpmn_like_01.json", "LR", 13, 4),
            ("uml_like_01.json", "TB", 10, 5),
            ("noisy_custom_01.json", "LR", 7, 3),
    ],
)
def test_fixture_regression_metrics(fixture_name, expected_direction, expected_nodes, expected_edges):
    payload = load_fixture(fixture_name)
    out = run_full_pipeline(payload)
    _assert_contract(out)
    assert out["meta"]["direction"] == expected_direction
    assert len(out["nodes"]) == expected_nodes
    assert len(out["edges"]) == expected_edges
