from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from text_to_diagram.mermaid_to_ir import mermaid_to_ir


def test_mermaid_to_ir_parses_nodes_edges_and_lanes():
    mmd = """flowchart LR
    subgraph l1["Клиент"]
    n1(["Старт"])
    n2["Проверка"]
    end
    n3(["Финиш"])
    n1 --> n2
    n2 --> n3
"""
    ir = mermaid_to_ir(mmd)
    assert len(ir["nodes"]) == 3
    assert len(ir["edges"]) == 2
    assert len(ir["lanes"]) == 1
    assert ir["meta"]["direction"] == "LR"


def test_mermaid_to_ir_fallback_on_empty_input():
    ir = mermaid_to_ir("")
    assert len(ir["nodes"]) == 2
    assert len(ir["edges"]) == 1
    assert any(x["code"] == "MERMAID_TO_IR_EMPTY" for x in ir["issues"])

