from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional


class TextToDiagramPromptError(RuntimeError):
    pass


@dataclass(frozen=True)
class TextToDiagramPromptConfig:
    prompt_version: str = "text-to-ir-prompt.v1"


def build_text_to_ir_prompt_pack(
    source_text: str,
    cfg: Optional[TextToDiagramPromptConfig] = None,
    previous_output: Optional[str] = None,
    parse_error: Optional[str] = None,
) -> Dict[str, str]:
    if cfg is None:
        cfg = TextToDiagramPromptConfig()

    if not isinstance(source_text, str) or not source_text.strip():
        raise TextToDiagramPromptError("'source_text' must be non-empty string")

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(
        source_text=source_text.strip(),
        previous_output=previous_output,
        parse_error=parse_error,
    )
    return {
        "prompt_version": cfg.prompt_version,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def _build_system_prompt() -> str:
    return (
        "Ты преобразуешь текст процесса в формальный IR JSON.\n"
        "Верни ТОЛЬКО валидный JSON-объект без markdown, пояснений и префиксов.\n"
        "Структура корня: nodes, edges, lanes, meta, issues.\n"
        "Главная цель: сохранить смысл шагов из пользовательского текста.\n"
        "Ограничения:\n"
        "- nodes[].type: shape|container|text|flow\n"
        "- nodes[].role: start|end|action|decision|parallel|inclusive|event_intermediate|unknown\n"
        "- edges[].type: sequential|conditional|message|association|unknown\n"
        "- meta.schema_version: process-ir.v1\n"
        "- meta.source: text_to_diagram\n"
        "- meta.language: ru\n"
        "- issues[]: объекты {code, severity, message, entity_type, entity_id}\n"
        "- у каждого edges[] обязательно есть from и to\n"
        "- текст шагов в nodes[].text должен быть осмысленным (не 'n1', 'n2', ...)\n"
        "- lanes[] добавляй только если в тексте явно видны роли/участники\n"
        "Если данных не хватает, не выдумывай детали: фиксируй это в issues.\n"
    )


def _build_user_prompt(
    source_text: str,
    previous_output: Optional[str],
    parse_error: Optional[str],
) -> str:
    payload = {"input_text_ru": source_text}
    base = (
        "Преобразуй входной текст в IR JSON по контракту.\n"
        "Требования:\n"
        "1) Вывод строго JSON.\n"
        "2) ID должны быть стабильными строками: n1, n2, ...; e1, e2, ...; l1, l2, ...\n"
        "3) Минимум один start и один end узел.\n"
        "4) Поле issues всегда присутствует (может быть пустым массивом).\n"
        "5) Каждый edge обязан иметь from и to и ссылаться на существующие node id.\n"
        "6) Старайся переносить формулировки шагов из входного текста в nodes[].text.\n\n"
        "Вход:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    if previous_output is None:
        return base

    return (
        base
        + "\n\nПредыдущий ответ был невалидным JSON. Исправь его.\n"
        + f"Ошибка парсинга/валидации: {parse_error or 'unknown'}\n"
        + "Невалидный ответ:\n"
        + previous_output
    )
