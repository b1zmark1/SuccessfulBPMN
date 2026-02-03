from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional


def generate_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def build_observability_meta(
    trace_id: str,
    status: str,
    stage_durations_ms: Dict[str, int],
    total_duration_ms: int,
    projection_meta: Optional[Dict[str, Any]],
    narrator_meta: Optional[Dict[str, Any]],
    errors: List[Dict[str, str]],
) -> Dict[str, Any]:
    return {
        "trace_id": trace_id,
        "status": status,
        "degraded": status == "degraded",
        "stage_durations_ms": dict(stage_durations_ms),
        "total_duration_ms": int(total_duration_ms),
        "semantic_schema_version": (
            projection_meta.get("semantic_schema_version") if projection_meta else None
        ),
        "source_graph_schema_version": (
            projection_meta.get("source_graph_schema_version") if projection_meta else None
        ),
        "provider": narrator_meta.get("provider") if narrator_meta else None,
        "prompt_version": narrator_meta.get("prompt_version") if narrator_meta else None,
        "error_codes": [e.get("code", "UNKNOWN") for e in errors],
    }

