from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from narrator.providers.base import BaseNarratorProvider
from text_to_diagram.acceptance import evaluate_text_to_diagram_case
from text_to_diagram.llm_service import TextToDiagramLLMService
from text_to_diagram.orchestrator import run_text_to_diagram_use_case


class FakeProvider(BaseNarratorProvider):
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.responses.pop(0)


def test_acceptance_suite_ru_scenarios():
    cases = [
        (
            "simple_flow_ru",
            "Клиент создает заявку. Оператор проверяет и закрывает заявку.",
            (
                '{"nodes":[{"id":"n1","type":"shape","role":"start","text":"Старт","lane_id":null},'
                '{"id":"n2","type":"shape","role":"action","text":"Проверка","lane_id":null},'
                '{"id":"n3","type":"shape","role":"end","text":"Финиш","lane_id":null}],'
                '"edges":[{"id":"e1","from":"n1","to":"n2","type":"sequential","text":null},'
                '{"id":"e2","from":"n2","to":"n3","type":"sequential","text":null}],'
                '"lanes":[],"meta":{"schema_version":"process-ir.v1","direction":"LR","source":"text_to_diagram","language":"ru"},"issues":[]}'
            ),
            {"require_lane": False, "require_branching": False},
        ),
        (
            "branching_flow_ru",
            "Система проверяет условие и идет по одной из двух веток.",
            (
                '{"nodes":[{"id":"n1","type":"shape","role":"start","text":"Старт","lane_id":null},'
                '{"id":"n2","type":"shape","role":"decision","text":"Условие","lane_id":null},'
                '{"id":"n3","type":"shape","role":"action","text":"Ветка А","lane_id":null},'
                '{"id":"n4","type":"shape","role":"action","text":"Ветка Б","lane_id":null},'
                '{"id":"n5","type":"shape","role":"end","text":"Финиш","lane_id":null}],'
                '"edges":[{"id":"e1","from":"n1","to":"n2","type":"sequential","text":null},'
                '{"id":"e2","from":"n2","to":"n3","type":"conditional","text":"да"},'
                '{"id":"e3","from":"n2","to":"n4","type":"conditional","text":"нет"},'
                '{"id":"e4","from":"n3","to":"n5","type":"sequential","text":null},'
                '{"id":"e5","from":"n4","to":"n5","type":"sequential","text":null}],'
                '"lanes":[],"meta":{"schema_version":"process-ir.v1","direction":"LR","source":"text_to_diagram","language":"ru"},"issues":[]}'
            ),
            {"require_lane": False, "require_branching": True},
        ),
        (
            "lane_flow_ru",
            "Клиент отправляет заявку, менеджер подтверждает и завершает.",
            (
                '{"nodes":[{"id":"n1","type":"shape","role":"start","text":"Старт","lane_id":"l1"},'
                '{"id":"n2","type":"shape","role":"action","text":"Отправить заявку","lane_id":"l1"},'
                '{"id":"n3","type":"shape","role":"action","text":"Подтвердить","lane_id":"l2"},'
                '{"id":"n4","type":"shape","role":"end","text":"Финиш","lane_id":"l2"}],'
                '"edges":[{"id":"e1","from":"n1","to":"n2","type":"sequential","text":null},'
                '{"id":"e2","from":"n2","to":"n3","type":"message","text":"запрос"},'
                '{"id":"e3","from":"n3","to":"n4","type":"sequential","text":null}],'
                '"lanes":[{"id":"l1","name":"Клиент","order":0},{"id":"l2","name":"Менеджер","order":1}],'
                '"meta":{"schema_version":"process-ir.v1","direction":"TB","source":"text_to_diagram","language":"ru"},"issues":[]}'
            ),
            {"require_lane": True, "require_branching": False},
        ),
    ]

    verdicts = []
    for case_name, source_text, ir_json, req in cases:
        service = TextToDiagramLLMService(provider=FakeProvider([ir_json]))
        result = run_text_to_diagram_use_case(source_text=source_text, llm_service=service)
        verdict = evaluate_text_to_diagram_case(case_name, result, **req)
        verdicts.append(verdict)

    assert len(verdicts) == 3
    assert all(v["passed"] for v in verdicts), verdicts

