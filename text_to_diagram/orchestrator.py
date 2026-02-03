from __future__ import annotations

import time
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from text_to_diagram.bpmn_exporter import export_bpmn
from text_to_diagram.ir_validation import IRValidationPolicy, validate_and_normalize_ir
from text_to_diagram.llm_service import TextToDiagramLLMService
from text_to_diagram.mermaid_exporter import export_mermaid
from text_to_diagram.mermaid_to_ir import mermaid_to_ir
from text_to_diagram.plantuml_exporter import export_plantuml
from text_to_diagram.render_layer import render_artifact_to_image


def run_text_to_diagram_use_case(
    source_text: str,
    render: Optional[Dict[str, Any]] = None,
    llm_cfg_overrides: Optional[Dict[str, Any]] = None,
    runtime_overrides: Optional[Dict[str, Any]] = None,
    validation_policy: Optional[IRValidationPolicy] = None,
    llm_service: Optional[TextToDiagramLLMService] = None,
) -> Dict[str, Any]:
    if llm_service is None:
        llm_service = TextToDiagramLLMService(runtime_overrides=runtime_overrides)

    trace_id = _trace_id()
    trace: List[Dict[str, Any]] = []
    stage_durations_ms: Dict[str, int] = {}
    stage_details: Dict[str, Any] = {}
    started_at = time.perf_counter()
    issues: List[Dict[str, Any]] = []

    artifacts: Dict[str, Any] = {
        "ir_json": None,
        "mermaid_mmd": None,
        "bpmn_xml": None,
        "plantuml_puml": None,
        "image_png_base64": None,
        "image_jpg_base64": None,
    }

    # Stage 1: text -> IR (LLM).
    stage = "llm_text_to_ir"
    input_summary = {"source_text_len": len(source_text)}
    trace.append(_trace_entry(stage, "started", 0, input_summary, {}, None, None))
    t0 = time.perf_counter()
    issues_before = len(issues)
    try:
        llm_out = llm_service.generate_ir(source_text=source_text, llm_cfg_overrides=llm_cfg_overrides)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        stage_durations_ms[stage] = duration_ms
        stage_details["llm_text_to_ir"] = llm_out.get("meta", {})
        issues.extend(llm_out.get("issues", []))
        llm_error = _first_issue_code_msg(issues[issues_before:], severity="error")
        trace.append(
            _trace_entry(
                stage,
                "completed",
                duration_ms,
                input_summary,
                {
                    "attempts": llm_out.get("meta", {}).get("attempts"),
                    "parse_errors_count": len(llm_out.get("meta", {}).get("parse_errors", [])),
                    "ir_nodes": len(llm_out.get("ir", {}).get("nodes", [])) if isinstance(llm_out.get("ir"), dict) else 0,
                    "ir_edges": len(llm_out.get("ir", {}).get("edges", [])) if isinstance(llm_out.get("ir"), dict) else 0,
                    "issues_added": len(issues) - issues_before,
                    "error_issues_added": _issues_error_count(issues[issues_before:]),
                },
                llm_error["code"] if llm_error else None,
                llm_error["message"] if llm_error else None,
            )
        )
    except Exception as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        stage_durations_ms[stage] = duration_ms
        issues.append(_issue("ORCH_LLM_STAGE_FAILED", "error", str(exc)))
        trace.append(
            _trace_entry(
                stage,
                "failed",
                duration_ms,
                input_summary,
                {"issues_added": 1, "error_issues_added": 1},
                "ORCH_LLM_STAGE_FAILED",
                str(exc),
            )
        )
        return _build_response(
            status="degraded",
            artifacts=artifacts,
            issues=issues,
            trace=trace,
            stage_durations_ms=stage_durations_ms,
            stage_details=stage_details,
            started_at=started_at,
            trace_id=trace_id,
            runtime_overrides=runtime_overrides,
        )

    # Stage 2: validate + normalize.
    stage = "validate_normalize_ir"
    input_summary = _ir_summary(llm_out.get("ir"))
    trace.append(_trace_entry(stage, "started", 0, input_summary, {}, None, None))
    t0 = time.perf_counter()
    issues_before = len(issues)
    vn_out = validate_and_normalize_ir(
        llm_out.get("ir"),
        policy=validation_policy,
    )
    duration_ms = int((time.perf_counter() - t0) * 1000)
    stage_durations_ms[stage] = duration_ms
    stage_details["validate_normalize_ir"] = {
        "status": vn_out.get("status"),
        "hard_fail": vn_out.get("hard_fail"),
        "hard_fail_codes": vn_out.get("hard_fail_codes", []),
    }
    issues.extend(vn_out.get("issues", []))
    normalized_ir = vn_out["normalized_ir"]
    artifacts["ir_json"] = normalized_ir
    vn_error = _first_issue_code_msg(issues[issues_before:], severity="error")
    trace.append(
        _trace_entry(
            stage,
            "completed",
            duration_ms,
            input_summary,
            {
                **_ir_summary(normalized_ir),
                "issues_added": len(issues) - issues_before,
                "error_issues_added": _issues_error_count(issues[issues_before:]),
                "hard_fail": bool(vn_out.get("hard_fail")),
            },
            vn_error["code"] if vn_error else None,
            vn_error["message"] if vn_error else None,
        )
    )

    # Stage 3-5: deterministic exporters.
    stage = "export_mermaid"
    input_summary = _ir_summary(normalized_ir)
    trace.append(_trace_entry(stage, "started", 0, input_summary, {}, None, None))
    t0 = time.perf_counter()
    issues_before = len(issues)
    mermaid_out = export_mermaid(normalized_ir)
    duration_ms = int((time.perf_counter() - t0) * 1000)
    stage_durations_ms[stage] = duration_ms
    stage_details["export_mermaid"] = mermaid_out.get("meta", {})
    artifacts["mermaid_mmd"] = mermaid_out["mmd"]
    issues.extend(mermaid_out.get("issues", []))
    mermaid_error = _first_issue_code_msg(issues[issues_before:], severity="error")
    trace.append(
        _trace_entry(
            stage,
            "completed",
            duration_ms,
            input_summary,
            {
                "mmd_len": len(mermaid_out.get("mmd", "")),
                "issues_added": len(issues) - issues_before,
                "error_issues_added": _issues_error_count(issues[issues_before:]),
            },
            mermaid_error["code"] if mermaid_error else None,
            mermaid_error["message"] if mermaid_error else None,
        )
    )

    stage = "export_bpmn"
    bpmn_source_ir = mermaid_to_ir(artifacts["mermaid_mmd"])
    bpmn_source_issues_before = len(issues)
    issues.extend(bpmn_source_ir.get("issues", []))
    input_summary = {
        **_ir_summary(bpmn_source_ir),
        "source": "mermaid",
        "source_issues_added": len(issues) - bpmn_source_issues_before,
        "source_error_issues_added": _issues_error_count(issues[bpmn_source_issues_before:]),
    }
    trace.append(_trace_entry(stage, "started", 0, input_summary, {}, None, None))
    t0 = time.perf_counter()
    issues_before = len(issues)
    bpmn_out = export_bpmn(bpmn_source_ir)
    duration_ms = int((time.perf_counter() - t0) * 1000)
    stage_durations_ms[stage] = duration_ms
    stage_details["export_bpmn"] = bpmn_out.get("meta", {})
    artifacts["bpmn_xml"] = bpmn_out["bpmn_xml"]
    issues.extend(bpmn_out.get("issues", []))
    bpmn_error = _first_issue_code_msg(issues[issues_before:], severity="error")
    trace.append(
        _trace_entry(
            stage,
            "completed",
            duration_ms,
            input_summary,
            {
                "bpmn_len": len(bpmn_out.get("bpmn_xml", "")),
                "issues_added": len(issues) - issues_before,
                "error_issues_added": _issues_error_count(issues[issues_before:]),
                "di_shapes_count": bpmn_out.get("meta", {}).get("di_shapes_count"),
                "di_edges_count": bpmn_out.get("meta", {}).get("di_edges_count"),
            },
            bpmn_error["code"] if bpmn_error else None,
            bpmn_error["message"] if bpmn_error else None,
        )
    )

    stage = "export_plantuml"
    input_summary = _ir_summary(normalized_ir)
    trace.append(_trace_entry(stage, "started", 0, input_summary, {}, None, None))
    t0 = time.perf_counter()
    issues_before = len(issues)
    puml_out = export_plantuml(normalized_ir)
    duration_ms = int((time.perf_counter() - t0) * 1000)
    stage_durations_ms[stage] = duration_ms
    stage_details["export_plantuml"] = puml_out.get("meta", {})
    artifacts["plantuml_puml"] = puml_out["puml"]
    issues.extend(puml_out.get("issues", []))
    puml_error = _first_issue_code_msg(issues[issues_before:], severity="error")
    trace.append(
        _trace_entry(
            stage,
            "completed",
            duration_ms,
            input_summary,
            {
                "puml_len": len(puml_out.get("puml", "")),
                "issues_added": len(issues) - issues_before,
                "error_issues_added": _issues_error_count(issues[issues_before:]),
            },
            puml_error["code"] if puml_error else None,
            puml_error["message"] if puml_error else None,
        )
    )

    # Optional render stage.
    render_cfg = _resolve_render_cfg(render)
    stage = "render_artifact"
    render_source = render_cfg["source_artifact"]
    source_text_for_render = artifacts.get(render_source)
    input_summary = {
        "enabled": render_cfg["enabled"],
        "artifact_type": render_cfg["artifact_type"],
        "image_format": render_cfg["image_format"],
        "source_artifact": render_source,
        "source_len": len(source_text_for_render) if isinstance(source_text_for_render, str) else 0,
        **_bpmn_render_input_summary_if_needed(render_cfg["artifact_type"], artifacts.get("bpmn_xml")),
    }
    trace.append(_trace_entry(stage, "started", 0, input_summary, {}, None, None))
    t0 = time.perf_counter()
    issues_before = len(issues)
    if render_cfg["enabled"]:
        if not isinstance(source_text_for_render, str) or not source_text_for_render.strip():
            code = "ORCH_RENDER_SOURCE_MISSING"
            msg = f"render source artifact '{render_source}' is empty"
            issues.append(_issue(code, "warning", msg))
            duration_ms = int((time.perf_counter() - t0) * 1000)
            stage_durations_ms[stage] = duration_ms
            trace.append(
                _trace_entry(
                    stage,
                    "failed",
                    duration_ms,
                    input_summary,
                    {"issues_added": 1, "error_issues_added": 0},
                    code,
                    msg,
                )
            )
        else:
            render_out = render_artifact_to_image(
                artifact_type=render_cfg["artifact_type"],
                artifact_text=source_text_for_render,
                image_format=render_cfg["image_format"],
            )
            duration_ms = int((time.perf_counter() - t0) * 1000)
            stage_durations_ms[stage] = duration_ms
            artifacts["image_png_base64"] = render_out.get("image_png_base64")
            artifacts["image_jpg_base64"] = render_out.get("image_jpg_base64")
            issues.extend(render_out.get("issues", []))
            stage_details["render_artifact"] = render_out.get("meta", {})
            render_error = _first_issue_code_msg(issues[issues_before:], severity="error")
            trace.append(
                _trace_entry(
                    stage,
                    "completed",
                    duration_ms,
                    input_summary,
                    {
                        "has_png": artifacts["image_png_base64"] is not None,
                        "has_jpg": artifacts["image_jpg_base64"] is not None,
                        "issues_added": len(issues) - issues_before,
                        "error_issues_added": _issues_error_count(issues[issues_before:]),
                    },
                    render_error["code"] if render_error else None,
                    render_error["message"] if render_error else None,
                )
            )
    else:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        stage_durations_ms[stage] = duration_ms
        trace.append(
            _trace_entry(
                stage,
                "completed",
                duration_ms,
                input_summary,
                {"skipped": True, "issues_added": 0, "error_issues_added": 0},
                None,
                None,
            )
        )

    status = "degraded" if issues else "ok"
    return _build_response(
        status=status,
        artifacts=artifacts,
        issues=issues,
        trace=trace,
        stage_durations_ms=stage_durations_ms,
        stage_details=stage_details,
        started_at=started_at,
        trace_id=trace_id,
        runtime_overrides=runtime_overrides,
    )


def _resolve_render_cfg(render: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(render, dict):
        return {
            "enabled": False,
            "artifact_type": "mermaid",
            "source_artifact": "mermaid_mmd",
            "image_format": "png",
        }

    enabled = bool(render.get("enabled", False))
    artifact_type = render.get("artifact_type", "mermaid")
    image_format = render.get("image_format", "png")
    mapping = {
        "mermaid": "mermaid_mmd",
        "bpmn": "bpmn_xml",
        "plantuml": "plantuml_puml",
    }
    source_artifact = mapping.get(artifact_type, "mermaid_mmd")
    return {
        "enabled": enabled,
        "artifact_type": artifact_type,
        "source_artifact": source_artifact,
        "image_format": image_format,
    }


def _build_response(
    status: str,
    artifacts: Dict[str, Any],
    issues: List[Dict[str, Any]],
    trace: List[Dict[str, str]],
    stage_durations_ms: Dict[str, int],
    stage_details: Dict[str, Any],
    started_at: float,
    trace_id: str,
    runtime_overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    deduped_issues = _dedupe_issues(issues)
    first_fail = _find_first_fail(trace)
    return {
        "status": status,
        "artifacts": artifacts,
        "issues": deduped_issues,
        "meta": {
            "target_latency_sec": 20,
            "cpu_only": True,
            "trace_id": trace_id,
            "trace": trace,
            "stage_durations_ms": stage_durations_ms,
            "stage_details": stage_details,
            "total_duration_ms": int((time.perf_counter() - started_at) * 1000),
            "runtime_overrides": runtime_overrides or {},
            "error_codes": [x["code"] for x in deduped_issues if x.get("severity") in {"warning", "error"}],
            "first_fail": first_fail,
        },
    }


def _dedupe_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        key = (
            issue.get("code"),
            issue.get("severity"),
            issue.get("message"),
            issue.get("entity_type"),
            issue.get("entity_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "code": issue.get("code", "ORCH_UNKNOWN_ISSUE"),
                "severity": issue.get("severity", "warning"),
                "message": issue.get("message", ""),
                "entity_type": issue.get("entity_type"),
                "entity_id": issue.get("entity_id"),
            }
        )
    return out


def _issue(code: str, severity: str, message: str) -> Dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "entity_type": None,
        "entity_id": None,
    }


def _trace_id() -> str:
    return uuid.uuid4().hex[:12]


def _trace_entry(
    stage: str,
    status: str,
    duration_ms: int,
    input_summary: Dict[str, Any],
    output_summary: Dict[str, Any],
    error_code: Optional[str],
    error_message: Optional[str],
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "status": status,
        "duration_ms": duration_ms,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "error_code": error_code,
        "error_message": error_message,
    }


def _issues_error_count(stage_issues: List[Dict[str, Any]]) -> int:
    return sum(1 for x in stage_issues if isinstance(x, dict) and x.get("severity") == "error")


def _ir_summary(ir: Any) -> Dict[str, Any]:
    if not isinstance(ir, dict):
        return {"nodes": 0, "edges": 0, "lanes": 0}
    return {
        "nodes": len(ir.get("nodes", [])) if isinstance(ir.get("nodes"), list) else 0,
        "edges": len(ir.get("edges", [])) if isinstance(ir.get("edges"), list) else 0,
        "lanes": len(ir.get("lanes", [])) if isinstance(ir.get("lanes"), list) else 0,
    }


def _first_issue_code_msg(stage_issues: List[Dict[str, Any]], severity: str = "error") -> Optional[Dict[str, str]]:
    for it in stage_issues:
        if isinstance(it, dict) and it.get("severity") == severity:
            return {"code": str(it.get("code")), "message": str(it.get("message"))}
    return None


def _bpmn_render_input_summary_if_needed(artifact_type: str, bpmn_xml: Any) -> Dict[str, Any]:
    if artifact_type != "bpmn" or not isinstance(bpmn_xml, str):
        return {}
    out = {
        "has_bpmn_diagram_tag": "<bpmndi:BPMNDiagram" in bpmn_xml,
        "has_bpmn_plane_tag": "<bpmndi:BPMNPlane" in bpmn_xml,
        "bpmn_shape_count": 0,
        "bpmn_edge_count": 0,
        "process_node_count": 0,
        "process_edge_count": 0,
    }
    try:
        root = ET.fromstring(bpmn_xml)
        ns = {
            "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
            "bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
        }
        out["bpmn_shape_count"] = len(root.findall(".//bpmndi:BPMNShape", ns))
        out["bpmn_edge_count"] = len(root.findall(".//bpmndi:BPMNEdge", ns))
        process_nodes = (
            root.findall(".//bpmn:startEvent", ns)
            + root.findall(".//bpmn:task", ns)
            + root.findall(".//bpmn:exclusiveGateway", ns)
            + root.findall(".//bpmn:parallelGateway", ns)
            + root.findall(".//bpmn:inclusiveGateway", ns)
            + root.findall(".//bpmn:intermediateThrowEvent", ns)
            + root.findall(".//bpmn:endEvent", ns)
        )
        process_edges = root.findall(".//bpmn:sequenceFlow", ns) + root.findall(".//bpmn:association", ns)
        out["process_node_count"] = len(process_nodes)
        out["process_edge_count"] = len(process_edges)
    except Exception:
        # Keep boolean tag diagnostics even if XML parsing fails.
        pass
    return out


def _find_first_fail(trace: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    # Rule: first stage from top with failed status OR completed + error issues added.
    for entry in trace:
        if entry.get("status") == "failed":
            return {
                "stage": entry.get("stage"),
                "code": entry.get("error_code") or "STAGE_FAILED",
                "message": entry.get("error_message") or "stage failed",
            }
        if entry.get("status") == "completed":
            out = entry.get("output_summary", {})
            if isinstance(out, dict) and int(out.get("error_issues_added", 0)) > 0:
                return {
                    "stage": entry.get("stage"),
                    "code": entry.get("error_code") or "STAGE_ERROR_ISSUES",
                    "message": entry.get("error_message")
                    or f"stage completed with error issues ({out.get('error_issues_added')})",
                }
    return None
