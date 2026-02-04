from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph_builder.pipeline import run_graph_to_semantic_pipeline
from tests.graph_builder.utils import load_fixture
from narrator.policies import resolve_narrator_policy
from narrator.prompts import NarratorPromptError, build_narrator_prompts, build_prompt_meta


def test_build_narrator_prompts_contract():
    payload = load_fixture("bpmn_like_01.json")
    semantic_payload = run_graph_to_semantic_pipeline(payload)["semantic_payload"]
    policy = resolve_narrator_policy()

    out = build_narrator_prompts(semantic_payload, policy)
    assert set(out.keys()) == {"prompt_version", "system_prompt", "user_prompt"}
    assert out["prompt_version"] == "narrator-prompt.v1"
    assert "Используй только данные из JSON" in out["system_prompt"]
    assert "max_sentences: 10" in out["user_prompt"]
    assert "branches_policy: cover_all" in out["user_prompt"]
    assert "output_format: narrative" in out["user_prompt"]
    assert "\"steps\"" in out["user_prompt"]


def test_build_narrator_prompts_rejects_invalid_semantic_payload():
    payload = load_fixture("bpmn_like_01.json")
    semantic_payload = run_graph_to_semantic_pipeline(payload)["semantic_payload"]
    semantic_payload["steps"][0]["role"] = "unknown"
    policy = resolve_narrator_policy()

    with pytest.raises(NarratorPromptError):
        build_narrator_prompts(semantic_payload, policy)


def test_build_prompt_meta():
    meta = build_prompt_meta("narrator-prompt.v1")
    assert meta == {"prompt_version": "narrator-prompt.v1"}


def test_build_narrator_prompts_table_mode_instructions():
    payload = load_fixture("bpmn_like_01.json")
    semantic_payload = run_graph_to_semantic_pipeline(payload)["semantic_payload"]
    policy = resolve_narrator_policy({"output_format": "table"})

    out = build_narrator_prompts(semantic_payload, policy)
    assert "output_format: table" in out["user_prompt"]
    assert "Шаг | Роль" in out["user_prompt"]
    assert "Если определить нельзя, укажи: Не указано." in out["user_prompt"]
