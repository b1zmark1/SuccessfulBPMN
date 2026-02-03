from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from narrator.providers.base import BaseNarratorProvider
from text_to_diagram.orchestrator import run_text_to_diagram_use_case


class FakeProvider(BaseNarratorProvider):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return "{}"


def _valid_ir_json() -> str:
    return (
        '{"nodes":[{"id":"n1","type":"shape","role":"start","text":"Старт","lane_id":"l1"},'
        '{"id":"n2","type":"shape","role":"action","text":"Проверка","lane_id":"l1"},'
        '{"id":"n3","type":"shape","role":"end","text":"Финиш","lane_id":"l2"}],'
        '"edges":[{"id":"e1","from":"n1","to":"n2","type":"sequential","text":null},'
        '{"id":"e2","from":"n2","to":"n3","type":"sequential","text":null}],'
        '"lanes":[{"id":"l1","name":"Клиент","order":0},{"id":"l2","name":"Менеджер","order":1}],'
        '"meta":{"schema_version":"process-ir.v1","direction":"LR","source":"text_to_diagram","language":"ru"},'
        '"issues":[]}'
    )


def test_orchestrator_returns_all_text_artifacts_without_render():
    provider = FakeProvider([_valid_ir_json()])
    out = run_text_to_diagram_use_case(
        source_text="Клиент отправляет запрос, затем менеджер подтверждает.",
        llm_service=_service_with_provider(provider),
    )

    assert out["status"] == "ok"
    artifacts = out["artifacts"]
    assert isinstance(artifacts["ir_json"], dict)
    assert artifacts["mermaid_mmd"].startswith("flowchart LR")
    assert artifacts["bpmn_xml"].startswith("<bpmn:definitions")
    assert artifacts["plantuml_puml"].startswith("@startuml")
    assert artifacts["image_png_base64"] is None
    assert artifacts["image_jpg_base64"] is None
    assert provider.calls == 1
    assert "stage_durations_ms" in out["meta"]
    assert "trace" in out["meta"]
    assert out["meta"]["first_fail"] is None
    assert all(
        {"stage", "status", "duration_ms", "input_summary", "output_summary", "error_code", "error_message"}.issubset(
            set(x.keys())
        )
        for x in out["meta"]["trace"]
    )


def test_orchestrator_render_stage_mermaid_png(monkeypatch):
    provider = FakeProvider([_valid_ir_json()])

    def _fake_render(*args, **kwargs):
        png_b64 = base64.b64encode(b"\x89PNGdummy").decode("ascii")
        return {
            "status": "ok",
            "image_png_base64": png_b64,
            "image_jpg_base64": None,
            "issues": [],
            "meta": {"artifact_type": "mermaid", "image_format": "png", "duration_ms": 1},
        }

    import text_to_diagram.orchestrator as orch

    monkeypatch.setattr(orch, "render_artifact_to_image", _fake_render)

    out = run_text_to_diagram_use_case(
        source_text="Процесс с рендером.",
        render={"enabled": True, "artifact_type": "mermaid", "image_format": "png"},
        llm_service=_service_with_provider(provider),
    )
    assert out["status"] == "ok"
    assert out["artifacts"]["image_png_base64"] is not None
    assert any(x["stage"] == "render_artifact" and x["status"] == "completed" for x in out["meta"]["trace"])


def test_orchestrator_logs_bpmn_render_input_summary(monkeypatch):
    provider = FakeProvider([_valid_ir_json()])

    def _fake_render(*args, **kwargs):
        return {
            "status": "degraded",
            "image_png_base64": None,
            "image_jpg_base64": None,
            "issues": [{"code": "X", "severity": "warning", "message": "x", "entity_type": None, "entity_id": None}],
            "meta": {"artifact_type": "bpmn", "image_format": "png", "duration_ms": 1},
        }

    import text_to_diagram.orchestrator as orch

    monkeypatch.setattr(orch, "render_artifact_to_image", _fake_render)
    out = run_text_to_diagram_use_case(
        source_text="Процесс с BPMN рендером.",
        render={"enabled": True, "artifact_type": "bpmn", "image_format": "png"},
        llm_service=_service_with_provider(provider),
    )
    render_completed = [
        x for x in out["meta"]["trace"] if x["stage"] == "render_artifact" and x["status"] == "completed"
    ][0]
    inp = render_completed["input_summary"]
    assert "has_bpmn_diagram_tag" in inp
    assert "has_bpmn_plane_tag" in inp
    assert "bpmn_shape_count" in inp
    assert "bpmn_edge_count" in inp
    assert "process_node_count" in inp
    assert "process_edge_count" in inp


def test_orchestrator_degraded_when_llm_stage_fails():
    provider = FakeProvider(["not json", "still not json"])
    out = run_text_to_diagram_use_case(
        source_text="Плохой ответ модели.",
        llm_cfg_overrides={"max_reasks": 1, "repair_fallback_enabled": False},
        llm_service=_service_with_provider(provider),
    )
    assert out["status"] == "degraded"
    assert out["artifacts"]["mermaid_mmd"] is None
    assert any(x["code"] == "ORCH_LLM_STAGE_FAILED" for x in out["issues"])
    assert out["meta"]["first_fail"]["stage"] == "llm_text_to_ir"
    assert out["meta"]["first_fail"]["code"] == "ORCH_LLM_STAGE_FAILED"


def _service_with_provider(provider: BaseNarratorProvider):
    from text_to_diagram.llm_service import TextToDiagramLLMService

    return TextToDiagramLLMService(provider=provider)
