from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from text_to_diagram.llm_prompts import (
    TextToDiagramPromptError,
    build_text_to_ir_prompt_pack,
)


def test_build_prompt_pack_contains_json_only_requirements():
    out = build_text_to_ir_prompt_pack("Пользователь оформляет заказ и получает подтверждение.")
    assert out["prompt_version"] == "text-to-ir-prompt.v1"
    assert "ТОЛЬКО валидный JSON-объект" in out["system_prompt"]
    assert "nodes, edges, lanes, meta, issues" in out["system_prompt"]
    assert "ID должны быть стабильными строками" in out["user_prompt"]


def test_build_prompt_pack_rejects_empty_input():
    with pytest.raises(TextToDiagramPromptError):
        build_text_to_ir_prompt_pack("  ")

