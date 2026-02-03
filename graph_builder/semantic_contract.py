from __future__ import annotations

from typing import Any, Dict, List, Tuple


class SemanticContractError(RuntimeError):
    pass


ALLOWED_ROLES = {"start", "action", "decision", "parallel", "end"}
ALLOWED_DIRECTIONS = {"LR", "TB"}
REQUIRED_META_KEYS = {
    "schema_version",
    "source_graph_schema_version",
    "direction",
    "warnings",
}
REQUIRED_STEP_KEYS = {"id", "order", "role", "text", "next_step_ids"}
SEMANTIC_SCHEMA_VERSION = "semantic-projection.v1"


def validate_semantic_projection_contract(payload: Dict[str, Any]) -> None:
    ok, errors = check_semantic_projection_contract(payload)
    if not ok:
        raise SemanticContractError("; ".join(errors))


def check_semantic_projection_contract(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if not isinstance(payload, dict):
        return False, ["semantic payload must be an object"]

    if set(payload.keys()) != {"meta", "steps"}:
        errors.append("semantic payload must contain only 'meta' and 'steps'")
        return False, errors

    meta = payload.get("meta")
    steps = payload.get("steps")

    if not isinstance(meta, dict):
        errors.append("'meta' must be an object")
    if not isinstance(steps, list):
        errors.append("'steps' must be an array")
    if errors:
        return False, errors

    if set(meta.keys()) != REQUIRED_META_KEYS:
        errors.append(f"'meta' must contain exactly {sorted(REQUIRED_META_KEYS)}")
    else:
        schema_version = meta.get("schema_version")
        if schema_version != SEMANTIC_SCHEMA_VERSION:
            errors.append(f"'meta.schema_version' must be '{SEMANTIC_SCHEMA_VERSION}'")

        source_graph_schema_version = meta.get("source_graph_schema_version")
        if not isinstance(source_graph_schema_version, str) or not source_graph_schema_version:
            errors.append("'meta.source_graph_schema_version' must be a non-empty string")

        direction = meta.get("direction")
        if direction not in ALLOWED_DIRECTIONS:
            errors.append("'meta.direction' must be 'LR' or 'TB'")

        warnings = meta.get("warnings")
        if not isinstance(warnings, list) or not all(isinstance(w, str) for w in warnings):
            errors.append("'meta.warnings' must be an array of strings")

    if len(steps) == 0:
        errors.append("'steps' must not be empty")
        return False, errors

    ids: List[str] = []
    orders: List[int] = []
    for idx, step in enumerate(steps):
        path = f"steps[{idx}]"
        if not isinstance(step, dict):
            errors.append(f"{path} must be an object")
            continue

        if set(step.keys()) != REQUIRED_STEP_KEYS:
            errors.append(f"{path} must contain exactly {sorted(REQUIRED_STEP_KEYS)}")
            continue

        sid = step.get("id")
        if not isinstance(sid, str) or not sid:
            errors.append(f"{path}.id must be a non-empty string")
        else:
            ids.append(sid)

        order = step.get("order")
        if not isinstance(order, int) or order < 1:
            errors.append(f"{path}.order must be integer >= 1")
        else:
            orders.append(order)

        role = step.get("role")
        if role not in ALLOWED_ROLES:
            errors.append(f"{path}.role must be one of {sorted(ALLOWED_ROLES)}")

        text = step.get("text")
        if text is not None and not isinstance(text, str):
            errors.append(f"{path}.text must be string or null")

        next_step_ids = step.get("next_step_ids")
        if not isinstance(next_step_ids, list):
            errors.append(f"{path}.next_step_ids must be an array")
        elif not all(isinstance(nid, str) and nid for nid in next_step_ids):
            errors.append(f"{path}.next_step_ids must contain non-empty strings")
        elif len(next_step_ids) != len(set(next_step_ids)):
            errors.append(f"{path}.next_step_ids must contain unique values")

    if len(ids) != len(set(ids)):
        errors.append("step ids must be unique")

    if sorted(orders) != list(range(1, len(steps) + 1)):
        errors.append("step orders must be continuous sequence 1..N")

    step_id_set = set(ids)
    for idx, step in enumerate(steps):
        next_step_ids = step.get("next_step_ids")
        if isinstance(next_step_ids, list):
            for nid in next_step_ids:
                if isinstance(nid, str) and nid not in step_id_set:
                    errors.append(f"steps[{idx}].next_step_ids references unknown step id '{nid}'")

    return len(errors) == 0, errors

