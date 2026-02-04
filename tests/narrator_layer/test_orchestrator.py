from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import narrator.orchestrator as orchestrator
from narrator.orchestrator import run_narration, run_narration_from_ensemble
from tests.graph_builder.utils import load_fixture, run_full_pipeline


def _install_fake_llama_cpp() -> None:
    fake_module = types.ModuleType("llama_cpp")

    class FakeLlama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def create_chat_completion(self, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Итоговое описание процесса.",
                        }
                    }
                ]
            }

    fake_module.Llama = FakeLlama
    sys.modules["llama_cpp"] = fake_module


def test_run_narration_returns_unified_dto():
    _install_fake_llama_cpp()
    payload = load_fixture("bpmn_like_01.json")
    graph_payload = run_full_pipeline(payload)

    out = run_narration(
        graph_payload=graph_payload,
        runtime_overrides={"model_path": "dummy.gguf", "n_threads": 2},
    )

    assert set(out.keys()) == {"text", "status", "errors", "meta"}
    assert out["status"] == "ok"
    assert out["errors"] == []
    assert isinstance(out["text"], str) and out["text"]

    assert set(out["meta"].keys()) == {"projection", "narrator", "trace", "observability"}
    assert out["meta"]["projection"]["semantic_schema_version"] == "semantic-projection.v1"
    assert out["meta"]["narrator"]["single_call"] is True
    assert out["meta"]["narrator"]["applied_policy"]["max_sentences"] == 10
    assert out["meta"]["narrator"]["provider"] == "llama_cpp"
    assert any(t["stage"] == "llm_call" and t["status"] == "completed" for t in out["meta"]["trace"])
    assert any(t["stage"] == "postprocess" and t["status"] == "completed" for t in out["meta"]["trace"])
    assert out["meta"]["observability"]["status"] == "ok"
    assert out["meta"]["observability"]["provider"] == "llama_cpp"
    assert out["meta"]["observability"]["prompt_version"] == "narrator-prompt.v1"
    assert "text" not in out["meta"]["observability"]


def test_run_narration_from_ensemble_works():
    _install_fake_llama_cpp()
    payload = load_fixture("noisy_custom_01.json")

    out = run_narration_from_ensemble(
        ensemble_payload=payload,
        runtime_overrides={"model_path": "dummy.gguf", "n_threads": 2},
    )
    assert out["status"] == "ok"
    assert out["meta"]["narrator"]["single_call"] is True


def test_run_narration_returns_error_dto_on_invalid_graph():
    bad_graph_payload = {"meta": {}, "nodes": [], "edges": []}

    out = run_narration(graph_payload=bad_graph_payload)
    assert out["status"] == "degraded"
    assert "Техническая заглушка" in out["text"]
    assert len(out["errors"]) == 1
    assert out["errors"][0]["code"] == "PROJECTION_ERROR"
    assert any(t["status"] == "failed" for t in out["meta"]["trace"])
    assert out["meta"]["observability"]["status"] == "degraded"
    assert out["meta"]["observability"]["error_codes"] == ["PROJECTION_ERROR"]


def test_run_narration_timeout_returns_degraded_with_timeout_code(monkeypatch):
    _install_fake_llama_cpp()
    payload = load_fixture("bpmn_like_01.json")
    graph_payload = run_full_pipeline(payload)

    def _raise_timeout(*args, **kwargs):
        raise TimeoutError("generation timed out")

    monkeypatch.setattr(orchestrator, "run_single_llm_call", _raise_timeout)
    out = run_narration(
        graph_payload=graph_payload,
        runtime_overrides={"model_path": "dummy.gguf", "n_threads": 2},
    )
    assert out["status"] == "degraded"
    assert out["errors"][0]["code"] == "LLM_TIMEOUT"
    assert out["meta"]["observability"]["error_codes"] == ["LLM_TIMEOUT"]


def test_run_narration_invalid_output_returns_degraded(monkeypatch):
    _install_fake_llama_cpp()
    payload = load_fixture("bpmn_like_01.json")
    graph_payload = run_full_pipeline(payload)

    def _raise_invalid_output(*args, **kwargs):
        raise RuntimeError("non-string content")

    monkeypatch.setattr(orchestrator, "run_single_llm_call", _raise_invalid_output)
    out = run_narration(
        graph_payload=graph_payload,
        runtime_overrides={"model_path": "dummy.gguf", "n_threads": 2},
    )
    assert out["status"] == "degraded"
    assert out["errors"][0]["code"] == "LLM_INVALID_OUTPUT"


def test_run_narration_guardrail_violation_returns_degraded(monkeypatch):
    _install_fake_llama_cpp()
    payload = load_fixture("bpmn_like_01.json")
    graph_payload = run_full_pipeline(payload)

    def _json_like_output(*args, **kwargs):
        return {
            "text": '{"x": 1}',
            "narrator_meta": {
                "prompt_version": "narrator-prompt.v1",
                "provider": "llama_cpp",
                "model_path": "dummy.gguf",
                "single_call": True,
                "duration_ms": 1,
                "runtime": {
                    "n_ctx": 2048,
                    "n_threads": 2,
                    "n_batch": 256,
                    "temperature": 0.2,
                    "max_tokens": 64,
                    "timeout_sec": 120,
                    "cpu_only": True,
                },
            },
        }

    monkeypatch.setattr(orchestrator, "run_single_llm_call", _json_like_output)
    out = run_narration(
        graph_payload=graph_payload,
        runtime_overrides={"model_path": "dummy.gguf", "n_threads": 2},
    )
    assert out["status"] == "degraded"
    assert out["errors"][0]["code"] == "OUTPUT_GUARDRAIL_VIOLATION"
    assert "JSON_OUTPUT" in out["errors"][0]["message"]


def test_run_narration_supports_table_output_policy():
    _install_fake_llama_cpp()
    payload = load_fixture("bpmn_like_01.json")
    graph_payload = run_full_pipeline(payload)

    out = run_narration(
        graph_payload=graph_payload,
        policy_overrides={"output_format": "table"},
        runtime_overrides={"model_path": "dummy.gguf", "n_threads": 2},
    )
    assert out["status"] == "ok"
    assert out["meta"]["narrator"]["applied_policy"]["output_format"] == "table"


def test_role_hints_fallback_from_left_vertical_text_bands():
    graph_payload = {
        "meta": {"schema_version": "graph-builder.v1", "direction": "LR", "warnings": []},
        "nodes": [
            {
                "id": "role_top",
                "type": "text",
                "bbox": [5.0, 90.0, 20.0, 220.0],
                "center": [12.5, 155.0],
                "role": "unknown",
                "container_id": None,
                "text": "Инициатор",
            },
            {
                "id": "role_mid",
                "type": "text",
                "bbox": [6.0, 260.0, 22.0, 410.0],
                "center": [14.0, 335.0],
                "role": "unknown",
                "container_id": None,
                "text": "Координатор",
            },
            {
                "id": "s1",
                "type": "shape",
                "bbox": [120.0, 110.0, 260.0, 180.0],
                "center": [190.0, 145.0],
                "role": "action",
                "container_id": None,
                "text": "Шаг 1",
            },
            {
                "id": "s2",
                "type": "shape",
                "bbox": [120.0, 300.0, 260.0, 370.0],
                "center": [190.0, 335.0],
                "role": "action",
                "container_id": None,
                "text": "Шаг 2",
            },
        ],
        "edges": [{"from": "s1", "to": "s2", "type": "sequential"}],
    }

    semantic_payload = {
        "meta": {
            "schema_version": "semantic-projection.v1",
            "source_graph_schema_version": "graph-builder.v1",
            "direction": "LR",
            "warnings": [],
        },
        "steps": [
            {"id": "s1", "order": 1, "role": "action", "text": "Шаг 1", "next_step_ids": ["s2"]},
            {"id": "s2", "order": 2, "role": "action", "text": "Шаг 2", "next_step_ids": []},
        ],
    }

    hints = orchestrator._infer_step_role_hints_from_graph(graph_payload, semantic_payload)
    assert hints["s1"] == "Инициатор"
    assert hints["s2"] == "Координатор"
