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
