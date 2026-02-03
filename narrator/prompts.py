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
    cfg: Optional[NarratorPromptConfig] = None,
) -> Dict[str, str]:
    if cfg is None:
        cfg = NarratorPromptConfig()
    try:
        validate_semantic_projection_contract(semantic_payload)
    except SemanticContractError as exc:
        raise NarratorPromptError(f"invalid semantic payload for prompt build: {exc}") from exc

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(semantic_payload, policy)
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
        "5) Выводи только финальный обычный текст, без JSON, markdown и служебных комментариев."
    )


def _build_user_prompt(
    semantic_payload: Dict[str, Any],
    policy: NarratorPolicyConfig,
) -> str:
    payload_text = json.dumps(semantic_payload, ensure_ascii=False, indent=2, sort_keys=True)
    return (
        "Сгенерируй итоговое описание процесса для пользователя, не знакомого с нотациями.\n"
        "Применяй политики генерации:\n"
        f"- max_sentences: {policy.max_sentences}\n"
        f"- missing_text_policy: {policy.missing_text_policy}\n"
        f"- branches_policy: {policy.branches_policy}\n\n"
        "Входной semantic JSON:\n"
        f"{payload_text}"
    )

