from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from narrator.policies import NarratorPolicyConfig


def render_table_from_semantic(
    semantic_payload: Dict[str, Any],
    policy: NarratorPolicyConfig,
    role_hints: Optional[Dict[str, str]] = None,
    node_meta_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    steps = semantic_payload.get("steps", [])
    if not isinstance(steps, list):
        steps = []

    role_hints_map = role_hints if isinstance(role_hints, dict) else {}
    node_meta_map = node_meta_by_id if isinstance(node_meta_by_id, dict) else {}

    ordered_steps = _sort_steps_by_order(steps)

    lines: List[str] = []
    lines.append("Шаг | Роль")

    idx = 1
    for step in ordered_steps:
        if not isinstance(step, dict):
            continue

        sid = step.get("id")
        sid_str = str(sid) if isinstance(sid, str) else None
        node_meta = node_meta_map.get(sid_str, {}) if sid_str else {}

        # Фильтр: в таблицу только "задачи".
        # 1) если в semantic есть текст — это почти всегда задача (оставляем)
        # 2) если текста нет — оставляем только если в graph node роль action
        step_text = _pick_step_text(step, policy)
        if not _is_task_like(step_text, node_meta):
            continue

        role = _pick_role(step, sid=sid_str, role_hints=role_hints_map, step_text=step_text, policy=policy)

        step_text = _sanitize_cell(step_text)
        role = _sanitize_cell(role)

        lines.append(f"{idx}. {step_text} | {role}")
        idx += 1

    if idx == 1:
        lines.append("1. [нет распознанных задач] | Не указано")

    return "\n".join(lines).strip()


def _is_task_like(step_text: str, node_meta: Dict[str, Any]) -> bool:
    # если распознан текст — считаем задачей
    if isinstance(step_text, str) and step_text.strip() and not step_text.strip().startswith("["):
        return True

    # иначе смотрим в graph meta
    ntype = str(node_meta.get("type", "")).lower()
    nrole = str(node_meta.get("role", "")).lower()

    # flows / start / end / decision из таблицы убираем
    if ntype == "flow":
        return False
    if nrole in {"start", "end", "decision"}:
        return False

    # оставляем только action без текста (редкий, но возможный случай)
    return nrole == "action"


def _sort_steps_by_order(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def keyfn(s: Dict[str, Any]) -> int:
        v = s.get("order")
        if isinstance(v, int):
            return v
        try:
            return int(v)
        except Exception:
            return 10**9

    out = [s for s in steps if isinstance(s, dict)]
    out.sort(key=keyfn)
    return out


def _pick_step_text(step: Dict[str, Any], policy: NarratorPolicyConfig) -> str:
    raw = step.get("text")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    # Для таблицы задач лучше НЕ тащить "плейсхолдеры" из control-нод.
    # Но если это реально action без текста — плейсхолдер останется.
    if policy.missing_text_policy == "explicit_placeholder":
        return "[без текста]"
    if policy.missing_text_policy == "skip":
        return "[шаг без текста]"
    return "Действие без названия"


def _pick_role(
    step: Dict[str, Any],
    sid: Optional[str],
    role_hints: Dict[str, str],
    step_text: str,
    policy: NarratorPolicyConfig,
) -> str:
    # 1) Явные поля из semantic (если они реально появятся)
    for key in ("lane", "owner", "executor", "assignee", "actor", "role_name"):
        v = step.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    participants = step.get("participants")
    if isinstance(participants, str) and participants.strip():
        return participants.strip()
    if isinstance(participants, list):
        parts = [p.strip() for p in participants if isinstance(p, str) and p.strip()]
        if parts:
            return ", ".join(parts)

    # 2) Хинт из графа (step_id -> роль), если lane/pool заработают
    if sid:
        hinted = role_hints.get(sid)
        if isinstance(hinted, str) and hinted.strip():
            return hinted.strip()

    # 3) Минимально безопасное извлечение роли ТОЛЬКО если явно указано в тексте
    low = step_text.lower()
    if re.search(r"(?<!\w)модератор(?!\w)", low):
        return "Модератор"

    # 4) Если хочешь “угадывать” роли — включай heuristic (см. policies.py ниже)
    if getattr(policy, "role_inference", "strict") == "heuristic":
        guessed = _heuristic_role_from_text(step_text)
        if guessed:
            return guessed

    return "Не указано"


def _heuristic_role_from_text(step_text: str) -> str:
    low = step_text.lower()

    # пользовательские действия (формы/кнопки/ввод)
    if any(x in low for x in ("загруз", "заполн", "ввод", "выбрать", "нажать", "опубликовать")):
        return "Пользователь"

    # системные действия (проверка/сохранение/авто-)
    if any(x in low for x in ("провер", "сохран", "автомодерац", "субтитр", "ограничение", "загрузка на", "сервер")):
        return "Система"

    return ""


def _sanitize_cell(value: str) -> str:
    if not isinstance(value, str):
        return "Не указано"
    s = value.replace("\r", "\n").replace("\t", " ").strip()
    s = " ".join([p for p in (x.strip() for x in s.split("\n")) if p])
    s = re.sub(r"[ ]{2,}", " ", s).strip()
    s = s.replace("|", "/")
    if not s:
        return "Не указано"
    return s
