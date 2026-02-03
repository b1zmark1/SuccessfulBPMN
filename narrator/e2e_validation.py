from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from narrator.orchestrator import run_narration_from_ensemble


def run_e2e_validation(
    cases: List[Tuple[str, Dict[str, Any]]],
    runtime_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    for case_name, ensemble_payload in cases:
        out = run_narration_from_ensemble(
            ensemble_payload=ensemble_payload,
            runtime_overrides=runtime_overrides,
        )
        results.append({"case": case_name, "result": out})

    total = len(results)
    ok = 0
    degraded = 0
    text_or_stub = 0
    projection_valid_ok = 0
    single_call_ok = 0
    durations: List[int] = []

    for item in results:
        out = item["result"]
        status = out.get("status")
        text = str(out.get("text", "")).strip()
        meta = out.get("meta", {}) if isinstance(out.get("meta"), dict) else {}
        obs = meta.get("observability", {}) if isinstance(meta.get("observability"), dict) else {}
        projection = meta.get("projection", {}) if isinstance(meta.get("projection"), dict) else {}
        narrator_meta = meta.get("narrator", {}) if isinstance(meta.get("narrator"), dict) else {}

        if status == "ok":
            ok += 1
            if projection.get("semantic_schema_version") == "semantic-projection.v1":
                projection_valid_ok += 1
            if narrator_meta.get("single_call") is True:
                single_call_ok += 1
        elif status == "degraded":
            degraded += 1

        if status == "ok" and bool(text):
            text_or_stub += 1
        elif status == "degraded" and "Техническая заглушка" in text:
            text_or_stub += 1

        duration = obs.get("total_duration_ms")
        if isinstance(duration, int):
            durations.append(duration)

    success_rate = (ok / total) if total else 0.0
    fallback_rate = (degraded / total) if total else 0.0
    text_or_stub_rate = (text_or_stub / total) if total else 0.0
    projection_valid_rate_ok = (projection_valid_ok / ok) if ok else 0.0
    single_call_rate_ok = (single_call_ok / ok) if ok else 0.0
    avg_total_duration_ms = int(mean(durations)) if durations else 0

    summary = {
        "total_cases": total,
        "ok_cases": ok,
        "degraded_cases": degraded,
        "success_rate": success_rate,
        "fallback_rate": fallback_rate,
        "text_or_stub_rate": text_or_stub_rate,
        "projection_valid_rate_ok": projection_valid_rate_ok,
        "single_call_rate_ok": single_call_rate_ok,
        "avg_total_duration_ms": avg_total_duration_ms,
    }

    return {
        "summary": summary,
        "results": results,
    }

