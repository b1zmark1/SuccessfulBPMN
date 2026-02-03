from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from narrator.e2e_validation import run_e2e_validation
from tests.graph_builder.utils import load_fixture


def _install_fake_llama_cpp_e2e() -> None:
    fake_module = types.ModuleType("llama_cpp")

    class FakeLlama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def create_chat_completion(self, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Процесс начинается, выполняются шаги, затем процесс завершается.",
                        }
                    }
                ]
            }

    fake_module.Llama = FakeLlama
    sys.modules["llama_cpp"] = fake_module


def test_e2e_validation_happy_path_metrics():
    _install_fake_llama_cpp_e2e()
    cases = [
        ("bpmn_like_01", load_fixture("bpmn_like_01.json")),
        ("uml_like_01", load_fixture("uml_like_01.json")),
        ("noisy_custom_01", load_fixture("noisy_custom_01.json")),
    ]
    report = run_e2e_validation(
        cases=cases,
        runtime_overrides={"model_path": "dummy.gguf", "n_threads": 2},
    )
    s = report["summary"]
    assert s["total_cases"] == 3
    assert s["ok_cases"] == 3
    assert s["degraded_cases"] == 0
    assert s["success_rate"] == 1.0
    assert s["fallback_rate"] == 0.0
    assert s["text_or_stub_rate"] == 1.0
    assert s["projection_valid_rate_ok"] == 1.0
    assert s["single_call_rate_ok"] == 1.0
    assert s["avg_total_duration_ms"] >= 0


def test_e2e_validation_degraded_still_returns_stub():
    _install_fake_llama_cpp_e2e()
    cases = [
        ("ok_case", load_fixture("bpmn_like_01.json")),
        ("bad_case", {}),
    ]
    report = run_e2e_validation(
        cases=cases,
        runtime_overrides={"model_path": "dummy.gguf", "n_threads": 2},
    )
    s = report["summary"]
    assert s["total_cases"] == 2
    assert s["ok_cases"] == 1
    assert s["degraded_cases"] == 1
    assert s["text_or_stub_rate"] == 1.0
    degraded = [r for r in report["results"] if r["result"]["status"] == "degraded"][0]
    assert "Техническая заглушка" in degraded["result"]["text"]

