from __future__ import annotations

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph_builder.semantic_projection import project_graph_to_semantic
from narrator.acceptance import evaluate_acceptance_case
from narrator.orchestrator import run_narration


def _install_fake_llama_cpp_acceptance() -> None:
    fake_module = types.ModuleType("llama_cpp")

    class FakeLlama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def create_chat_completion(self, **kwargs):
            user_prompt = kwargs["messages"][1]["content"]
            marker = "Входной semantic JSON:\n"
            payload = json.loads(user_prompt.split(marker, 1)[1])
            steps = sorted(payload["steps"], key=lambda s: s["order"])
            by_id = {s["id"]: s for s in steps}

            parts = []
            for s in steps:
                txt = s["text"] if s["text"] else f"шаг {s['id']}"
                parts.append(f"Выполняется: {txt}.")
                next_ids = s.get("next_step_ids", [])
                if len(next_ids) > 1:
                    branch_labels = []
                    for nid in next_ids:
                        t = by_id[nid]["text"]
                        branch_labels.append(t if t else nid)
                    parts.append("Ветви: " + ", ".join(branch_labels) + ".")

            return {"choices": [{"message": {"content": " ".join(parts)}}]}

    fake_module.Llama = FakeLlama
    sys.modules["llama_cpp"] = fake_module


def _node(node_id: str, role: str, text: str | None) -> dict:
    return {
        "id": node_id,
        "type": "shape",
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "center": [5.0, 5.0],
        "role": role,
        "container_id": None,
        "text": text,
    }


def _graph_case_linear() -> dict:
    return {
        "meta": {"schema_version": "graph-builder.v1", "direction": "LR", "warnings": []},
        "nodes": [
            _node("s1", "start", "Начало"),
            _node("s2", "action", "Проверка"),
            _node("s3", "end", "Завершение"),
        ],
        "edges": [
            {"from": "s1", "to": "s2", "type": "sequential"},
            {"from": "s2", "to": "s3", "type": "sequential"},
        ],
    }


def _graph_case_decision() -> dict:
    return {
        "meta": {"schema_version": "graph-builder.v1", "direction": "LR", "warnings": []},
        "nodes": [
            _node("s1", "start", "Начало"),
            _node("s2", "decision", "Проверка условия"),
            _node("s3", "action", "Ветка А"),
            _node("s4", "action", "Ветка Б"),
            _node("s5", "end", "Завершение"),
        ],
        "edges": [
            {"from": "s1", "to": "s2", "type": "sequential"},
            {"from": "s2", "to": "s3", "type": "conditional"},
            {"from": "s2", "to": "s4", "type": "conditional"},
            {"from": "s3", "to": "s5", "type": "sequential"},
            {"from": "s4", "to": "s5", "type": "sequential"},
        ],
    }


def _graph_case_parallel() -> dict:
    return {
        "meta": {"schema_version": "graph-builder.v1", "direction": "TB", "warnings": []},
        "nodes": [
            _node("s1", "start", "Старт"),
            _node("s2", "action", "Параллельный запуск"),
            _node("s3", "action", "Обработка 1"),
            _node("s4", "action", "Обработка 2"),
            _node("s5", "end", "Финиш"),
        ],
        "edges": [
            {"from": "s1", "to": "s2", "type": "sequential"},
            {"from": "s2", "to": "s3", "type": "sequential"},
            {"from": "s2", "to": "s4", "type": "sequential"},
            {"from": "s3", "to": "s5", "type": "sequential"},
            {"from": "s4", "to": "s5", "type": "sequential"},
        ],
    }


def _graph_case_empty_text() -> dict:
    return {
        "meta": {"schema_version": "graph-builder.v1", "direction": "LR", "warnings": []},
        "nodes": [
            _node("s1", "start", None),
            _node("s2", "action", None),
            _node("s3", "end", None),
        ],
        "edges": [
            {"from": "s1", "to": "s2", "type": "sequential"},
            {"from": "s2", "to": "s3", "type": "sequential"},
        ],
    }


def _graph_case_noisy_incomplete() -> dict:
    return {
        "meta": {
            "schema_version": "graph-builder.v1",
            "direction": "LR",
            "warnings": ["noisy_input_detected"],
        },
        "nodes": [
            _node("s1", "unknown", "Неясный старт"),
            _node("s2", "action", "Основное действие"),
            _node("s3", "end", None),
            {
                "id": "junk",
                "type": "container",
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "center": [0.5, 0.5],
                "role": "unknown",
                "container_id": None,
                "text": None,
            },
        ],
        "edges": [
            {"from": "s1", "to": "s2", "type": "unknown"},
            {"from": "s2", "to": "s3", "type": "sequential"},
        ],
    }


def test_acceptance_suite_projection_to_narrator():
    _install_fake_llama_cpp_acceptance()
    cases = [
        ("linear_flow", _graph_case_linear()),
        ("multiple_branches", _graph_case_decision()),
        ("parallel_block", _graph_case_parallel()),
        ("empty_text", _graph_case_empty_text()),
        ("noise_incomplete", _graph_case_noisy_incomplete()),
    ]

    verdicts = []
    for case_name, graph_payload in cases:
        semantic = project_graph_to_semantic(graph_payload)
        out = run_narration(
            graph_payload=graph_payload,
            runtime_overrides={"model_path": "dummy.gguf", "n_threads": 2},
        )
        verdict = evaluate_acceptance_case(case_name, semantic, out)
        verdicts.append(verdict)

    assert len(verdicts) == 5
    assert all(v["passed"] for v in verdicts), verdicts

