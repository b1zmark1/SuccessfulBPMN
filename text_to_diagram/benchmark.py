from __future__ import annotations

import statistics
from typing import Any, Dict, List

from text_to_diagram.orchestrator import run_text_to_diagram_use_case


def run_latency_benchmark(
    cases: List[Dict[str, Any]],
    latency_target_ms: int = 20_000,
) -> Dict[str, Any]:
    durations: List[int] = []
    results: List[Dict[str, Any]] = []

    for case in cases:
        out = run_text_to_diagram_use_case(
            source_text=case["source_text"],
            llm_service=case.get("llm_service"),
            render=case.get("render"),
            llm_cfg_overrides=case.get("llm_cfg_overrides"),
            runtime_overrides=case.get("runtime_overrides"),
        )
        duration = int(out.get("meta", {}).get("total_duration_ms", 0))
        durations.append(duration)
        results.append(
            {
                "case": case.get("name", "unnamed"),
                "status": out.get("status"),
                "duration_ms": duration,
                "within_target": duration <= latency_target_ms,
            }
        )

    avg = int(statistics.mean(durations)) if durations else 0
    p95 = int(_percentile(durations, 95.0)) if durations else 0
    within_target = all(r["within_target"] for r in results)
    return {
        "summary": {
            "total_cases": len(results),
            "latency_target_ms": latency_target_ms,
            "avg_duration_ms": avg,
            "p95_duration_ms": p95,
            "all_within_target": within_target,
        },
        "results": results,
    }


def _percentile(values: List[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (p / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    w = rank - low
    return ordered[low] * (1.0 - w) + ordered[high] * w

