from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def evaluate_acceptance_case(
    case_name: str,
    semantic_payload: Dict[str, Any],
    narration_result: Dict[str, Any],
) -> Dict[str, Any]:
    text = str(narration_result.get("text", "")).strip()
    status = narration_result.get("status")
    errors = narration_result.get("errors", [])

    checklist: Dict[str, bool] = {
        "status_supported": status in {"ok", "degraded"},
        "text_or_stub_present": bool(text),
        "russian_like_text": _looks_like_russian(text),
        "not_json_output": not _looks_like_json(text),
        "no_notation_terms": not _contains_notation_terms(text),
        "branches_covered": _branches_covered_by_text(semantic_payload, text),
    }

    reasons: List[str] = []
    if status == "degraded":
        if "Техническая заглушка" not in text:
            reasons.append("degraded status without technical stub text")
    if not checklist["branches_covered"] and status == "ok":
        reasons.append("not all branches are reflected in final text")
    if errors and status == "ok":
        reasons.append("errors are present for ok status")

    passed = all(checklist.values()) and (len(reasons) == 0)
    return {
        "case": case_name,
        "passed": passed,
        "checklist": checklist,
        "reasons": reasons,
    }


def _looks_like_russian(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"[А-Яа-яЁё]", text))


def _looks_like_json(text: str) -> bool:
    s = text.strip()
    if not s or s[0] not in "{[":
        return False
    try:
        parsed = json.loads(s)
        return isinstance(parsed, (dict, list))
    except Exception:
        return False


def _contains_notation_terms(text: str) -> bool:
    low = text.lower()
    return any(t in low for t in ("bpmn", "uml", "бпмн", "юмл"))


def _branches_covered_by_text(semantic_payload: Dict[str, Any], text: str) -> bool:
    steps = semantic_payload.get("steps", [])
    if not isinstance(steps, list):
        return False
    step_by_id = {s.get("id"): s for s in steps if isinstance(s, dict)}

    for s in steps:
        if not isinstance(s, dict):
            continue
        next_ids = s.get("next_step_ids", [])
        if not isinstance(next_ids, list) or len(next_ids) <= 1:
            continue
        for nid in next_ids:
            target = step_by_id.get(nid)
            if not isinstance(target, dict):
                return False
            target_text = target.get("text")
            if isinstance(target_text, str) and target_text.strip():
                if target_text.strip().lower() not in text.lower():
                    return False
            else:
                if str(nid) not in text:
                    return False
    return True

