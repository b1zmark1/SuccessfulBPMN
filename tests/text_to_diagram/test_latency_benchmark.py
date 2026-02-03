from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from narrator.providers.base import BaseNarratorProvider
from text_to_diagram.benchmark import run_latency_benchmark
from text_to_diagram.llm_service import TextToDiagramLLMService


class FakeProvider(BaseNarratorProvider):
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.responses.pop(0)


def _valid_ir_json() -> str:
    return (
        '{"nodes":[{"id":"n1","type":"shape","role":"start","text":"Старт","lane_id":null},'
        '{"id":"n2","type":"shape","role":"action","text":"Шаг","lane_id":null},'
        '{"id":"n3","type":"shape","role":"end","text":"Финиш","lane_id":null}],'
        '"edges":[{"id":"e1","from":"n1","to":"n2","type":"sequential","text":null},'
        '{"id":"e2","from":"n2","to":"n3","type":"sequential","text":null}],'
        '"lanes":[],"meta":{"schema_version":"process-ir.v1","direction":"LR","source":"text_to_diagram","language":"ru"},"issues":[]}'
    )


def test_latency_benchmark_meets_target_with_fake_provider():
    cases = []
    for i in range(5):
        service = TextToDiagramLLMService(provider=FakeProvider([_valid_ir_json()]))
        cases.append({"name": f"case_{i}", "source_text": "Короткий процесс.", "llm_service": service})

    report = run_latency_benchmark(cases, latency_target_ms=20_000)
    assert report["summary"]["total_cases"] == 5
    assert report["summary"]["all_within_target"] is True
    assert report["summary"]["p95_duration_ms"] <= 20_000

