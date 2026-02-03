from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class NarratorPostprocessError(RuntimeError):
    pass


@dataclass(frozen=True)
class NarratorPostprocessConfig:
    banned_notation_terms: tuple[str, ...] = ("bpmn", "uml", "бпмн", "юмл")
    banned_internal_phrases: tuple[str, ...] = (
        "я как модель",
        "как языковая модель",
        "как ии",
        "as an ai",
        "language model",
    )


def postprocess_narration_text(
    text: str,
    cfg: Optional[NarratorPostprocessConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = NarratorPostprocessConfig()
    if not isinstance(text, str):
        raise NarratorPostprocessError("narration text must be a string")

    normalized, normalization_warnings = _normalize_text(text)
    hard_error = _detect_hard_violation(normalized, cfg)

    return {
        "text": normalized,
        "normalized": len(normalization_warnings) > 0,
        "normalization_warnings": normalization_warnings,
        "hard_violation": hard_error is not None,
        "violation_code": hard_error["code"] if hard_error else None,
        "violation_message": hard_error["message"] if hard_error else None,
    }


def _normalize_text(text: str) -> tuple[str, List[str]]:
    warnings: List[str] = []
    t = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ").strip()
    before = t

    lines: List[str] = []
    for raw in t.split("\n"):
        line = raw.strip()
        line = re.sub(r"^\s{0,3}#+\s+", "", line)
        line = re.sub(r"^\s*[-*]\s+", "", line)
        lines.append(line)

    t = "\n".join(lines)
    t = re.sub(r"[ ]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    if t != before:
        warnings.append("text_normalized")
    return t, warnings


def _detect_hard_violation(
    text: str,
    cfg: NarratorPostprocessConfig,
) -> Optional[Dict[str, str]]:
    if not text:
        return {"code": "EMPTY_TEXT", "message": "empty text after normalization"}

    if "```" in text:
        return {"code": "MARKDOWN_OUTPUT", "message": "markdown code fence detected"}

    if _looks_like_json(text):
        return {"code": "JSON_OUTPUT", "message": "json-like output detected"}

    low = text.lower()
    for term in cfg.banned_notation_terms:
        if re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", low):
            return {"code": "NOTATION_TERM", "message": f"notation term detected: {term}"}

    for phrase in cfg.banned_internal_phrases:
        if phrase.lower() in low:
            return {"code": "MODEL_SELF_REFERENCE", "message": f"internal phrase detected: {phrase}"}

    return None


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] not in "{[":
        return False
    try:
        parsed = json.loads(stripped)
        return isinstance(parsed, (dict, list))
    except Exception:
        return False

