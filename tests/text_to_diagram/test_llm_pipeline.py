from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from narrator.providers.base import BaseNarratorProvider
from text_to_diagram.llm_pipeline import TextToDiagramPipelineError, run_text_to_ir_pipeline


class FakeProvider(BaseNarratorProvider):
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return "{}"


def test_pipeline_reasks_and_returns_degraded_on_second_attempt():
    provider = FakeProvider(
        responses=[
            "broken json",
            '{"nodes":[],"edges":[],"lanes":[],"meta":{"schema_version":"process-ir.v1","direction":"LR","source":"text_to_diagram","language":"ru"},"issues":[]}',
        ]
    )
    out = run_text_to_ir_pipeline(
        source_text="Пользователь создает заявку, затем заявка закрывается.",
        provider=provider,
    )
    assert out["status"] == "degraded"
    assert out["meta"]["attempts"] == 2
    assert provider.calls == 2
    assert isinstance(out["ir"]["nodes"], list)


def test_pipeline_uses_fallback_ir_after_all_failed_attempts():
    provider = FakeProvider(responses=["bad", "still bad"])
    out = run_text_to_ir_pipeline(
        source_text="Короткий тестовый процесс.",
        provider=provider,
        llm_cfg_overrides={"max_reasks": 1},
    )
    assert out["status"] == "degraded"
    assert out["ir"]["meta"]["schema_version"] == "process-ir.v1"
    assert out["issues"][0]["code"] == "IR_PARSE_FAILED_FALLBACK"


def test_pipeline_rejects_too_long_input():
    provider = FakeProvider(responses=["{}"])
    with pytest.raises(TextToDiagramPipelineError):
        run_text_to_ir_pipeline(
            source_text="x" * 5000,
            provider=provider,
        )
