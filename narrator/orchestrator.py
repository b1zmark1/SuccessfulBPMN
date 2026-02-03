from __future__ import annotations

from typing import Any, Dict, List, Optional
import time

from graph_builder.pipeline import build_graph_from_ensemble
from graph_builder.semantic_contract import validate_semantic_projection_contract
from graph_builder.semantic_projection import (
    SemanticProjectionConfig,
    project_graph_to_semantic,
)
from narrator.config import resolve_runtime_config
from narrator.observability import build_observability_meta, generate_trace_id
from narrator.policies import build_narrator_meta, resolve_narrator_policy
from narrator.postprocess import postprocess_narration_text
from narrator.prompts import NarratorPromptConfig, build_narrator_prompts
from narrator.runtime import run_single_llm_call


class NarratorOrchestrationError(RuntimeError):
    pass


TECHNICAL_STUB_TEXT = (
    "Техническая заглушка: не удалось сгенерировать описание процесса из-за внутренней ошибки."
)


def run_narration(
    graph_payload: Dict[str, Any],
    policy_overrides: Optional[Dict[str, Any]] = None,
    runtime_overrides: Optional[Dict[str, Any]] = None,
    projection_cfg: Optional[SemanticProjectionConfig] = None,
    prompt_cfg: Optional[NarratorPromptConfig] = None,
) -> Dict[str, Any]:
    """
    Unified interface for stages: graph -> semantic -> text.
    Returns DTO: text, status, errors, meta.
    """
    trace: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []
    trace_id = generate_trace_id()
    started_at = time.perf_counter()
    stage_durations_ms: Dict[str, int] = {}
    projection_meta: Optional[Dict[str, Any]] = None
    policy_meta: Optional[Dict[str, Any]] = None

    trace.append({"stage": "projection", "status": "started"})
    stage_started = time.perf_counter()
    try:
        semantic_payload = project_graph_to_semantic(graph_payload, cfg=projection_cfg)
        stage_durations_ms["projection"] = int((time.perf_counter() - stage_started) * 1000)
        trace.append({"stage": "projection", "status": "completed"})
        projection_meta = {
            "semantic_schema_version": semantic_payload["meta"]["schema_version"],
            "source_graph_schema_version": semantic_payload["meta"]["source_graph_schema_version"],
            "steps_count": len(semantic_payload.get("steps", [])),
        }
    except Exception as exc:
        trace.append({"stage": "projection", "status": "failed"})
        return _degraded_response(
            trace=trace,
            errors=errors,
            code="PROJECTION_ERROR",
            exc=exc,
            projection_meta=projection_meta,
            policy_meta=policy_meta,
            trace_id=trace_id,
            stage_durations_ms=stage_durations_ms,
            total_duration_ms=int((time.perf_counter() - started_at) * 1000),
        )

    trace.append({"stage": "contract_validation", "status": "started"})
    stage_started = time.perf_counter()
    try:
        validate_semantic_projection_contract(semantic_payload)
        stage_durations_ms["contract_validation"] = int((time.perf_counter() - stage_started) * 1000)
        trace.append({"stage": "contract_validation", "status": "completed"})
    except Exception as exc:
        trace.append({"stage": "contract_validation", "status": "failed"})
        return _degraded_response(
            trace=trace,
            errors=errors,
            code="SEMANTIC_CONTRACT_ERROR",
            exc=exc,
            projection_meta=projection_meta,
            policy_meta=policy_meta,
            trace_id=trace_id,
            stage_durations_ms=stage_durations_ms,
            total_duration_ms=int((time.perf_counter() - started_at) * 1000),
        )

    trace.append({"stage": "policy_resolution", "status": "started"})
    stage_started = time.perf_counter()
    try:
        policy = resolve_narrator_policy(policy_overrides)
        policy_meta = build_narrator_meta(policy)
        stage_durations_ms["policy_resolution"] = int((time.perf_counter() - stage_started) * 1000)
        trace.append({"stage": "policy_resolution", "status": "completed"})
    except Exception as exc:
        trace.append({"stage": "policy_resolution", "status": "failed"})
        return _degraded_response(
            trace=trace,
            errors=errors,
            code="POLICY_CONFIG_ERROR",
            exc=exc,
            projection_meta=projection_meta,
            policy_meta=policy_meta,
            trace_id=trace_id,
            stage_durations_ms=stage_durations_ms,
            total_duration_ms=int((time.perf_counter() - started_at) * 1000),
        )

    trace.append({"stage": "prompt_build", "status": "started"})
    stage_started = time.perf_counter()
    try:
        prompt_pack = build_narrator_prompts(semantic_payload, policy=policy, cfg=prompt_cfg)
        stage_durations_ms["prompt_build"] = int((time.perf_counter() - stage_started) * 1000)
        trace.append({"stage": "prompt_build", "status": "completed"})
    except Exception as exc:
        trace.append({"stage": "prompt_build", "status": "failed"})
        return _degraded_response(
            trace=trace,
            errors=errors,
            code="PROMPT_BUILD_ERROR",
            exc=exc,
            projection_meta=projection_meta,
            policy_meta=policy_meta,
            trace_id=trace_id,
            stage_durations_ms=stage_durations_ms,
            total_duration_ms=int((time.perf_counter() - started_at) * 1000),
        )

    trace.append({"stage": "runtime_resolution", "status": "started"})
    stage_started = time.perf_counter()
    try:
        runtime_cfg = resolve_runtime_config(runtime_overrides)
        stage_durations_ms["runtime_resolution"] = int((time.perf_counter() - stage_started) * 1000)
        trace.append({"stage": "runtime_resolution", "status": "completed"})
    except Exception as exc:
        trace.append({"stage": "runtime_resolution", "status": "failed"})
        return _degraded_response(
            trace=trace,
            errors=errors,
            code="RUNTIME_CONFIG_ERROR",
            exc=exc,
            projection_meta=projection_meta,
            policy_meta=policy_meta,
            trace_id=trace_id,
            stage_durations_ms=stage_durations_ms,
            total_duration_ms=int((time.perf_counter() - started_at) * 1000),
        )

    trace.append({"stage": "llm_call", "status": "started"})
    stage_started = time.perf_counter()
    try:
        llm_result = run_single_llm_call(prompt_pack=prompt_pack, runtime_cfg=runtime_cfg)
        stage_durations_ms["llm_call"] = int((time.perf_counter() - stage_started) * 1000)
        trace.append({"stage": "llm_call", "status": "completed"})
    except Exception as exc:
        trace.append({"stage": "llm_call", "status": "failed"})
        return _degraded_response(
            trace=trace,
            errors=errors,
            code=_classify_llm_error_code(exc),
            exc=exc,
            projection_meta=projection_meta,
            policy_meta=policy_meta,
            trace_id=trace_id,
            stage_durations_ms=stage_durations_ms,
            total_duration_ms=int((time.perf_counter() - started_at) * 1000),
        )

    trace.append({"stage": "postprocess", "status": "started"})
    stage_started = time.perf_counter()
    try:
        postprocess = postprocess_narration_text(llm_result["text"])
        stage_durations_ms["postprocess"] = int((time.perf_counter() - stage_started) * 1000)
        if postprocess["hard_violation"]:
            trace.append({"stage": "postprocess", "status": "failed"})
            return _degraded_response(
                trace=trace,
                errors=errors,
                code="OUTPUT_GUARDRAIL_VIOLATION",
                exc=RuntimeError(
                    f"{postprocess['violation_code']}: {postprocess['violation_message']}"
                ),
                projection_meta=projection_meta,
                policy_meta={
                    **policy_meta,
                    "postprocess": {
                        "normalized": postprocess["normalized"],
                        "normalization_warnings": postprocess["normalization_warnings"],
                        "violation_code": postprocess["violation_code"],
                    },
                },
                trace_id=trace_id,
                stage_durations_ms=stage_durations_ms,
                total_duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
        trace.append({"stage": "postprocess", "status": "completed"})
    except Exception as exc:
        trace.append({"stage": "postprocess", "status": "failed"})
        return _degraded_response(
            trace=trace,
            errors=errors,
            code="POSTPROCESS_ERROR",
            exc=exc,
            projection_meta=projection_meta,
            policy_meta=policy_meta,
            trace_id=trace_id,
            stage_durations_ms=stage_durations_ms,
            total_duration_ms=int((time.perf_counter() - started_at) * 1000),
        )

    narrator_meta = {
        **policy_meta,
        **llm_result["narrator_meta"],
        "postprocess": {
            "normalized": postprocess["normalized"],
            "normalization_warnings": postprocess["normalization_warnings"],
        },
    }
    total_duration_ms = int((time.perf_counter() - started_at) * 1000)
    return {
        "text": postprocess["text"],
        "status": "ok",
        "errors": errors,
        "meta": {
            "projection": projection_meta,
            "narrator": narrator_meta,
            "trace": trace,
            "observability": build_observability_meta(
                trace_id=trace_id,
                status="ok",
                stage_durations_ms=stage_durations_ms,
                total_duration_ms=total_duration_ms,
                projection_meta=projection_meta,
                narrator_meta=narrator_meta,
                errors=errors,
            ),
        },
    }


def run_narration_from_ensemble(
    ensemble_payload: Dict[str, Any],
    policy_overrides: Optional[Dict[str, Any]] = None,
    runtime_overrides: Optional[Dict[str, Any]] = None,
    projection_cfg: Optional[SemanticProjectionConfig] = None,
    prompt_cfg: Optional[NarratorPromptConfig] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper for full flow: ensemble -> graph -> semantic -> text.
    """
    try:
        started_at = time.perf_counter()
        trace_id = generate_trace_id()
        graph_payload = build_graph_from_ensemble(ensemble_payload)
    except Exception as exc:
        trace = [
            {"stage": "graph_build", "status": "started"},
            {"stage": "graph_build", "status": "failed"},
        ]
        return _degraded_response(
            trace=trace,
            errors=[],
            code="PROJECTION_ERROR",
            exc=exc,
            projection_meta=None,
            policy_meta=None,
            trace_id=trace_id,
            stage_durations_ms={},
            total_duration_ms=int((time.perf_counter() - started_at) * 1000),
        )
    return run_narration(
        graph_payload=graph_payload,
        policy_overrides=policy_overrides,
        runtime_overrides=runtime_overrides,
        projection_cfg=projection_cfg,
        prompt_cfg=prompt_cfg,
    )


def _degraded_response(
    trace: List[Dict[str, str]],
    errors: List[Dict[str, str]],
    code: str,
    exc: Exception,
    projection_meta: Optional[Dict[str, Any]],
    policy_meta: Optional[Dict[str, Any]],
    trace_id: str,
    stage_durations_ms: Dict[str, int],
    total_duration_ms: int,
) -> Dict[str, Any]:
    errors.append({"code": code, "message": str(exc)})
    meta: Dict[str, Any] = {"trace": trace}
    if projection_meta is not None:
        meta["projection"] = projection_meta
    if policy_meta is not None:
        meta["narrator"] = policy_meta
    meta["observability"] = build_observability_meta(
        trace_id=trace_id,
        status="degraded",
        stage_durations_ms=stage_durations_ms,
        total_duration_ms=total_duration_ms,
        projection_meta=projection_meta,
        narrator_meta=policy_meta,
        errors=errors,
    )
    return {
        "text": f"{TECHNICAL_STUB_TEXT} Код: {code}.",
        "status": "degraded",
        "errors": errors,
        "meta": meta,
    }


def _classify_llm_error_code(exc: Exception) -> str:
    msg = str(exc).lower()
    if isinstance(exc, TimeoutError) or "timeout" in msg or "timed out" in msg:
        return "LLM_TIMEOUT"
    if "context" in msg or "max context" in msg or "too many tokens" in msg:
        return "LLM_CONTEXT_OVERFLOW"
    if "non-string" in msg or "invalid output" in msg:
        return "LLM_INVALID_OUTPUT"
    return "LLM_RUNTIME_ERROR"
