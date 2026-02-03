from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from narrator.providers.base import BaseNarratorProvider
from text_to_diagram.llm_service import TextToDiagramLLMService
from text_to_diagram.orchestrator import run_text_to_diagram_use_case


class FakeProvider(BaseNarratorProvider):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return self.responses.pop(0)


def test_e2e_text_to_artifacts_integration():
    provider = FakeProvider(
        [
            (
                '{"nodes":[{"id":"a","type":"shape","role":"start","text":"Старт","lane_id":"l1"},'
                '{"id":"b","type":"shape","role":"action","text":"Обработать заявку","lane_id":"l1"},'
                '{"id":"c","type":"shape","role":"end","text":"Завершение","lane_id":"l2"}],'
                '"edges":[{"id":"e1","from":"a","to":"b","type":"sequential","text":null},'
                '{"id":"e2","from":"b","to":"c","type":"sequential","text":null}],'
                '"lanes":[{"id":"l1","name":"Клиент","order":0},{"id":"l2","name":"Оператор","order":1}],'
                '"meta":{"schema_version":"process-ir.v1","direction":"LR","source":"text_to_diagram","language":"ru"},'
                '"issues":[]}'
            )
        ]
    )
    llm_service = TextToDiagramLLMService(provider=provider)

    out = run_text_to_diagram_use_case(
        source_text="Клиент отправляет заявку, оператор обрабатывает и завершает процесс.",
        llm_service=llm_service,
    )

    assert out["status"] == "ok"
    artifacts = out["artifacts"]
    assert isinstance(artifacts["ir_json"], dict)
    assert artifacts["mermaid_mmd"].startswith("flowchart LR")
    assert artifacts["plantuml_puml"].startswith("@startuml")
    ET.fromstring(artifacts["bpmn_xml"])
    assert provider.calls == 1
    assert isinstance(out["meta"]["total_duration_ms"], int)
    assert "llm_text_to_ir" in out["meta"]["stage_durations_ms"]
    assert "export_mermaid" in out["meta"]["stage_durations_ms"]

