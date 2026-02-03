from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from text_to_diagram.llm_postprocess import TextToDiagramIRParseError, parse_and_repair_ir


def test_parse_and_repair_ir_parses_fenced_json():
    raw = """```json
{"nodes":[],"edges":[],"lanes":[],"meta":{"schema_version":"process-ir.v1","direction":"LR","source":"text_to_diagram","language":"ru"},"issues":[]}
```"""
    ir, repairs = parse_and_repair_ir(raw)
    assert isinstance(ir, dict)
    assert ir["meta"]["schema_version"] == "process-ir.v1"
    assert repairs == []


def test_parse_and_repair_ir_repairs_missing_sections():
    raw = '{"nodes": []}'
    ir, repairs = parse_and_repair_ir(raw)
    assert isinstance(ir["edges"], list)
    assert isinstance(ir["lanes"], list)
    assert isinstance(ir["meta"], dict)
    assert isinstance(ir["issues"], list)
    assert len(repairs) >= 1


def test_parse_and_repair_ir_fails_on_non_json():
    with pytest.raises(TextToDiagramIRParseError):
        parse_and_repair_ir("not a json")

