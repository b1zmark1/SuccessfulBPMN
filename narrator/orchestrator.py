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
from narrator.table_renderer import render_table_from_semantic


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

    # Роли считаем один раз (нужно и для таблицы, и для narrative).
    trace.append({"stage": "role_hints", "status": "started"})
    stage_started = time.perf_counter()
    try:
        role_hints = _infer_step_role_hints_from_graph(graph_payload, semantic_payload)
        stage_durations_ms["role_hints"] = int((time.perf_counter() - stage_started) * 1000)
        trace.append({"stage": "role_hints", "status": "completed"})
    except Exception:
        # role_hints не должен валить пайплайн
        role_hints = {}
        stage_durations_ms["role_hints"] = int((time.perf_counter() - stage_started) * 1000)
        trace.append({"stage": "role_hints", "status": "completed"})

    # КЛЮЧЕВОЕ: если нужен table — НЕ ЗОВЕМ LLM. Это решает "20 секунд" и стабильный формат.
    if policy.output_format == "table":
        trace.append({"stage": "table_render", "status": "started"})
        stage_started = time.perf_counter()
        try:
            node_meta_by_id = {
                str(n.get("id")): n
                for n in graph_payload.get("nodes", [])
                if isinstance(n, dict) and isinstance(n.get("id"), str)
            }

            table_text = render_table_from_semantic(
                semantic_payload=semantic_payload,
                policy=policy,
                role_hints=role_hints,
                node_meta_by_id=node_meta_by_id,
            )

            table_text = render_table_from_semantic(
                semantic_payload=semantic_payload,
                policy=policy,
                role_hints=role_hints,
            )
            stage_durations_ms["table_render"] = int((time.perf_counter() - stage_started) * 1000)
            trace.append({"stage": "table_render", "status": "completed"})
        except Exception as exc:
            trace.append({"stage": "table_render", "status": "failed"})
            return _degraded_response(
                trace=trace,
                errors=errors,
                code="TABLE_RENDER_ERROR",
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
            postprocess = postprocess_narration_text(table_text)
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
            "provider": "deterministic",
            "prompt_version": "table-renderer.v1",
            "single_call": False,
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

    # ===== narrative режим: оставляем как было (LLM) =====

    trace.append({"stage": "prompt_build", "status": "started"})
    stage_started = time.perf_counter()
    try:
        prompt_pack = build_narrator_prompts(
            semantic_payload,
            policy=policy,
            role_hints=role_hints,
            cfg=prompt_cfg,
        )
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


# ====== ВНИЗУ: роль-хинты без изменений (как у тебя) ======

def _infer_step_role_hints_from_graph(
    graph_payload: Dict[str, Any],
    semantic_payload: Dict[str, Any],
) -> Dict[str, str]:
    nodes = graph_payload.get("nodes")
    steps = semantic_payload.get("steps")
    if not isinstance(nodes, list) or not isinstance(steps, list):
        return {}

    containers: Dict[str, Dict[str, float]] = {}
    texts: List[Dict[str, Any]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        ntype = str(n.get("type", "")).lower()
        nid = n.get("id")
        bbox = n.get("bbox")
        if ntype == "container" and isinstance(nid, str) and _valid_bbox(bbox):
            containers[nid] = {"bbox": bbox}
        elif ntype == "text":
            texts.append(n)

    container_roles: Dict[str, str] = {}
    container_scores: Dict[str, float] = {}
    for t in texts:
        cid = t.get("container_id")
        if not isinstance(cid, str) or cid not in containers:
            continue
        tb = t.get("bbox")
        if not _valid_bbox(tb):
            continue
        label = _clean_role_label(t.get("text"))
        if not label:
            continue
        cb = containers[cid]["bbox"]
        inside = _inside_ratio(tb, cb)
        if inside < 0.7:
            continue
        if not _in_left_band(tb, cb, 0.22):
            continue
        orientation_ok = _is_horizontal(tb, 1.2) or _is_vertical(tb, 1.6)
        if not orientation_ok:
            continue
        left_dist = abs(float(tb[0]) - float(cb[0])) / max(1.0, float(cb[2]) - float(cb[0]))
        score = inside + (1.0 - left_dist)
        if score > container_scores.get(cid, -1.0):
            container_scores[cid] = score
            container_roles[cid] = label

    node_by_id: Dict[str, Dict[str, Any]] = {
        str(n.get("id")): n for n in nodes if isinstance(n, dict) and isinstance(n.get("id"), str)
    }
    step_role_hints: Dict[str, str] = {}
    if container_roles:
        for step in steps:
            if not isinstance(step, dict):
                continue
            sid = step.get("id")
            if not isinstance(sid, str):
                continue
            node = node_by_id.get(sid)
            if not node:
                continue
            cid = node.get("container_id")
            if isinstance(cid, str):
                role = container_roles.get(cid)
                if role:
                    step_role_hints[sid] = role

    if not step_role_hints:
        step_role_hints = _infer_role_hints_from_left_edge_text(nodes, steps, node_by_id)
    return step_role_hints


def _clean_role_label(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    parts = [p.strip() for p in value.replace("\r", "\n").split("\n")]
    out = " ".join([p for p in parts if p]).strip()
    return out


def _valid_bbox(v: Any) -> bool:
    if not (isinstance(v, list) and len(v) == 4):
        return False
    try:
        x1, y1, x2, y2 = [float(x) for x in v]
    except Exception:
        return False
    return x2 > x1 and y2 > y1


def _inside_ratio(inner: List[float], outer: List[float]) -> float:
    ix1 = max(float(inner[0]), float(outer[0]))
    iy1 = max(float(inner[1]), float(outer[1]))
    ix2 = min(float(inner[2]), float(outer[2]))
    iy2 = min(float(inner[3]), float(outer[3]))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_inner = max(1e-9, (float(inner[2]) - float(inner[0])) * (float(inner[3]) - float(inner[1])))
    return float(inter / area_inner)


def _in_left_band(text_box: List[float], container_box: List[float], left_band_ratio: float) -> bool:
    c_w = float(container_box[2]) - float(container_box[0])
    left_limit = float(container_box[0]) + left_band_ratio * c_w
    cx = (float(text_box[0]) + float(text_box[2])) / 2.0
    return cx <= left_limit


def _is_horizontal(b: List[float], min_ratio: float) -> bool:
    w = float(b[2]) - float(b[0])
    h = float(b[3]) - float(b[1])
    if h <= 1e-9:
        return True
    return (w / h) >= min_ratio


def _is_vertical(b: List[float], min_ratio: float) -> bool:
    w = float(b[2]) - float(b[0])
    h = float(b[3]) - float(b[1])
    if w <= 1e-9:
        return True
    return (h / w) >= min_ratio


def _infer_role_hints_from_left_edge_text(
    nodes: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    node_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    x1_all = []
    y1_all = []
    x2_all = []
    y2_all = []
    for n in nodes:
        bb = n.get("bbox")
        if _valid_bbox(bb):
            x1_all.append(float(bb[0]))
            y1_all.append(float(bb[1]))
            x2_all.append(float(bb[2]))
            y2_all.append(float(bb[3]))
    if not x1_all:
        return {}

    img_x1 = min(x1_all)
    img_y1 = min(y1_all)
    img_x2 = max(x2_all)
    img_y2 = max(y2_all)
    img_w = max(1.0, img_x2 - img_x1)
    img_h = max(1.0, img_y2 - img_y1)

    candidates: List[Dict[str, Any]] = []
    for n in nodes:
        if str(n.get("type", "")).lower() != "text":
            continue
        bb = n.get("bbox")
        if not _valid_bbox(bb):
            continue
        label = _clean_role_label(n.get("text"))
        if not _looks_like_role_text(label):
            continue
        cx = (float(bb[0]) + float(bb[2])) / 2.0
        if cx > img_x1 + 0.10 * img_w:
            continue
        if float(bb[3]) <= img_y1 + 0.08 * img_h:
            continue
        if not _is_vertical(bb, 1.6):
            continue
        candidates.append({"label": label, "bbox": bb})

    if not candidates:
        return {}

    candidates.sort(key=lambda x: float(x["bbox"][1]))
    role_bands: List[Dict[str, Any]] = []
    for c in candidates:
        cb = c["bbox"]
        merged = False
        for rb in role_bands:
            if _y_overlap_ratio(cb, rb["bbox"]) >= 0.35:
                if len(c["label"]) > len(rb["label"]):
                    rb["label"] = c["label"]
                rb["bbox"] = [
                    min(float(rb["bbox"][0]), float(cb[0])),
                    min(float(rb["bbox"][1]), float(cb[1])),
                    max(float(rb["bbox"][2]), float(cb[2])),
                    max(float(rb["bbox"][3]), float(cb[3])),
                ]
                merged = True
                break
        if not merged:
            role_bands.append({"label": c["label"], "bbox": [float(x) for x in cb]})

    out: Dict[str, str] = {}
    for s in steps:
        if not isinstance(s, dict):
            continue
        sid = s.get("id")
        if not isinstance(sid, str):
            continue
        node = node_by_id.get(sid)
        if not node:
            continue
        bb = node.get("bbox")
        if not _valid_bbox(bb):
            continue
        best_label = None
        best_score = 0.0
        for rb in role_bands:
            score = _y_overlap_ratio(bb, rb["bbox"])
            if score > best_score:
                best_score = score
                best_label = rb["label"]
        if best_label and best_score > 0.1:
            out[sid] = best_label
    return out


def _looks_like_role_text(label: str) -> bool:
    s = label.strip()
    if len(s) < 3:
        return False
    letters = sum(1 for ch in s if ch.isalpha())
    return letters >= 2


def _y_overlap_ratio(a: List[float], b: List[float]) -> float:
    ay1, ay2 = float(a[1]), float(a[3])
    by1, by2 = float(b[1]), float(b[3])
    inter = max(0.0, min(ay2, by2) - max(ay1, by1))
    amin = max(1e-9, ay2 - ay1)
    return inter / amin
