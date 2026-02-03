from __future__ import annotations

from copy import deepcopy

from graph_builder.normalize import normalize_ensemble_input
from tests.graph_builder.utils import load_fixture, run_full_pipeline


def test_ocr_text_is_propagated_into_graph_nodes():
    payload = deepcopy(load_fixture("bpmn_like_01.json"))

    # Inject OCR-assigned text into YOLOX detections (new upstream format).
    for det in payload["detections"]:
        if det.get("class_name") == "task":
            det["text"] = "Проверка заявки"
            det["text_conf"] = 0.91
            det["text_block_ids"] = [101, 102]
            det["match_score"] = 0.93
            break

    out = run_full_pipeline(payload)
    node_texts = [n.get("text") for n in out["nodes"] if n.get("text")]
    assert "Проверка заявки" in node_texts


def test_blank_ocr_text_is_normalized_to_null():
    payload = deepcopy(load_fixture("bpmn_like_01.json"))
    for det in payload["detections"]:
        if det.get("class_name") == "task":
            det["text"] = "   \n  "
            break

    out = run_full_pipeline(payload)
    # Contract remains nullable text; blank OCR should not leak as non-empty string.
    assert all((n["text"] is None) or (isinstance(n["text"], str) and n["text"].strip()) for n in out["nodes"])


def test_normalize_accepts_class_alias_and_infers_image_size():
    payload = {
        "image": {"path": "x.png"},
        "detections": [
            {
                "class": "task",
                "source": "yolox",
                "score": 0.8,
                "bbox": [10, 20, 110, 70],
                "text": "Текст шага",
                "text_conf": 0.92,
                "text_block_ids": [1, 2],
            }
        ],
    }
    out = normalize_ensemble_input(payload)
    assert out["image"]["width"] >= 111
    assert out["image"]["height"] >= 71
    det = out["detections_normalized"][0]
    assert det["class_name"] == "task"
    assert det["text"] == "Текст шага"
