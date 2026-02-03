from __future__ import annotations

from copy import deepcopy

from tests.graph_builder.utils import load_fixture, run_full_pipeline


def test_missing_flow_hints_warning():
    payload = load_fixture("bpmn_like_01.json")
    payload = deepcopy(payload)
    payload["detections"] = [d for d in payload["detections"] if d["class_name"] != "sequence_flow"]
    out = run_full_pipeline(payload)
    assert any("missing_flow_hints" in w for w in out["meta"]["warnings"])


def test_wrong_class_labels_do_not_break_pipeline():
    payload = load_fixture("bpmn_like_01.json")
    payload = deepcopy(payload)
    payload["detections"].append(
        {
            "class_id": 123,
            "class_name": "totally_unknown_class",
            "score": 0.8,
            "bbox_xyxy": [640, 120, 710, 200],
            "source": "yolox",
        }
    )
    out = run_full_pipeline(payload)
    # contract still valid and unknown class is absorbed into graph as shape/unknown role
    assert len(out["nodes"]) > 0
    assert any(n["role"] == "unknown" for n in out["nodes"])


def test_sparse_text_handling():
    payload = load_fixture("bpmn_like_01.json")
    payload = deepcopy(payload)
    payload["detections"] = [d for d in payload["detections"] if d["class_name"] != "text"]
    out = run_full_pipeline(payload)
    assert any("text_hooks: no text nodes found" in w for w in out["meta"]["warnings"])
    assert all(n["text"] is None for n in out["nodes"])


def test_overlapping_containers_warning():
    payload = load_fixture("noisy_custom_01.json")
    out = run_full_pipeline(payload)
    assert any("container_conflict" in w or "container_conflicts" in w for w in out["meta"]["warnings"])
