from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from narrator.postprocess import postprocess_narration_text


def test_postprocess_normalizes_markdown_bullets_softly():
    src = "- Первый шаг\n- Второй шаг\n\n# Заголовок"
    out = postprocess_narration_text(src)
    assert out["hard_violation"] is False
    assert out["normalized"] is True
    assert "Первый шаг" in out["text"]
    assert "# " not in out["text"]


def test_postprocess_rejects_json_output():
    out = postprocess_narration_text('{"steps":[1,2]}')
    assert out["hard_violation"] is True
    assert out["violation_code"] == "JSON_OUTPUT"


def test_postprocess_rejects_notation_terms():
    out = postprocess_narration_text("Это описание BPMN процесса.")
    assert out["hard_violation"] is True
    assert out["violation_code"] == "NOTATION_TERM"


def test_postprocess_rejects_internal_model_phrase():
    out = postprocess_narration_text("Как языковая модель, я не могу...")
    assert out["hard_violation"] is True
    assert out["violation_code"] == "MODEL_SELF_REFERENCE"

