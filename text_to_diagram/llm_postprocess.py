from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


class TextToDiagramIRParseError(RuntimeError):
    pass


def parse_and_repair_ir(raw_text: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise TextToDiagramIRParseError("empty LLM output")

    obj = _parse_json_object(raw_text)
    ir, repair_issues = _repair_ir_shape(obj)
    return ir, repair_issues


def _parse_json_object(raw_text: str) -> Dict[str, Any]:
    cleaned = raw_text.strip()
    fenced = _extract_fenced_json(cleaned)
    if fenced is not None:
        cleaned = fenced

    decoder = json.JSONDecoder()
    # Fast path: direct JSON.
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Fallback: find first decodable JSON object in mixed text.
    for i, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(cleaned[i:])
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj

    raise TextToDiagramIRParseError("cannot decode JSON object from LLM output")


def _extract_fenced_json(text: str) -> Optional[str]:
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _repair_ir_shape(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not isinstance(obj, dict):
        raise TextToDiagramIRParseError("root JSON value must be an object")

    issues: List[Dict[str, Any]] = []
    ir: Dict[str, Any] = dict(obj)

    if not isinstance(ir.get("nodes"), list):
        ir["nodes"] = []
        issues.append(_issue("IR_REPAIR_NODES", "warning", "nodes was missing or invalid; replaced with []"))

    if not isinstance(ir.get("edges"), list):
        ir["edges"] = []
        issues.append(_issue("IR_REPAIR_EDGES", "warning", "edges was missing or invalid; replaced with []"))

    if not isinstance(ir.get("lanes"), list):
        ir["lanes"] = []
        issues.append(_issue("IR_REPAIR_LANES", "warning", "lanes was missing or invalid; replaced with []"))

    if not isinstance(ir.get("meta"), dict):
        ir["meta"] = {}
        issues.append(_issue("IR_REPAIR_META", "warning", "meta was missing or invalid; replaced with {}"))

    if not isinstance(ir.get("issues"), list):
        ir["issues"] = []
        issues.append(_issue("IR_REPAIR_ISSUES", "warning", "issues was missing or invalid; replaced with []"))

    meta = ir["meta"]
    meta.setdefault("schema_version", "process-ir.v1")
    meta.setdefault("direction", "LR")
    meta.setdefault("source", "text_to_diagram")
    meta.setdefault("language", "ru")

    ir["issues"].extend(issues)
    return ir, issues


def _issue(code: str, severity: str, message: str) -> Dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "entity_type": None,
        "entity_id": None,
    }

