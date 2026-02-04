from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from graph_builder.semantic_contract import (
    SemanticContractError,
    validate_semantic_projection_contract,
)
from narrator.policies import NarratorPolicyConfig


class NarratorPromptError(RuntimeError):
    pass


@dataclass(frozen=True)
class NarratorPromptConfig:
    prompt_version: str = "narrator-prompt.v1"


def build_narrator_prompts(
    semantic_payload: Dict[str, Any],
    policy: NarratorPolicyConfig,
    role_hints: Optional[Dict[str, str]] = None,
    cfg: Optional[NarratorPromptConfig] = None,
) -> Dict[str, str]:
    if cfg is None:
        cfg = NarratorPromptConfig()
    try:
        validate_semantic_projection_contract(semantic_payload)
    except SemanticContractError as exc:
        raise NarratorPromptError(f"invalid semantic payload for prompt build: {exc}") from exc

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(semantic_payload, policy, role_hints=role_hints)
    return {
        "prompt_version": cfg.prompt_version,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def build_prompt_meta(prompt_version: str) -> Dict[str, str]:
    return {"prompt_version": prompt_version}


def _build_system_prompt() -> str:
    return (
        "Ты Narrator-модуль. Твоя задача: по semantic JSON написать понятное человеку "
        "описание процесса на русском языке.\n"
        "Строго следуй правилам:\n"
        "1) Используй только данные из JSON, ничего не придумывай.\n"
        "2) Соблюдай порядок шагов по полю order и связи next_step_ids.\n"
        "3) Для decision и parallel покрывай все ветви.\n"
        "4) Не используй термины BPMN/UML и технический жаргон.\n"
        "5) Выводи только финальный текст в формате, который задан в пользовательских инструкциях, "
        "без JSON/markdown и служебных комментариев."
    )


def _build_user_prompt(
    semantic_payload: Dict[str, Any],
    policy: NarratorPolicyConfig,
    role_hints: Optional[Dict[str, str]] = None,
) -> str:
    payload_text = json.dumps(semantic_payload, ensure_ascii=False, indent=2, sort_keys=True)
    if policy.output_format == "table":
        output_rules = (
            "Формат вывода: table.\n"
            "Верни только таблицу в plain text (не markdown).\n"
            "Строка заголовка: Шаг | Роль\n"
            "Далее строки строго по шагам order:\n"
            "<номер>. <краткое название шага> | <роль>\n"
            "Для роли сначала используй lane/owner/participants, если эти поля есть во входе.\n"
            "Если роль не указана, попробуй аккуратно вывести ее по контексту шага и соседних шагов.\n"
            "Если определить нельзя, укажи: Не указано."
        )
    else:
        output_rules = (
            "Формат вывода: narrative.\n"
            "Верни связный человеко-читаемый текст."
        )

    role_hints_payload = role_hints if isinstance(role_hints, dict) else {}
    role_hints_text = json.dumps(role_hints_payload, ensure_ascii=False, indent=2, sort_keys=True)

    return (
        "Сгенерируй итоговое описание процесса для пользователя, не знакомого с нотациями.\n"
        "Применяй политики генерации:\n"
        f"- max_sentences: {policy.max_sentences}\n"
        f"- missing_text_policy: {policy.missing_text_policy}\n"
        f"- branches_policy: {policy.branches_policy}\n"
        f"- output_format: {policy.output_format}\n\n"
        f"{output_rules}\n\n"
        "Подсказки по ролям шагов (step_id -> роль), если доступны:\n"
        f"{role_hints_text}\n\n"
        "Входной semantic JSON:\n"
        f"{payload_text}"
    )
