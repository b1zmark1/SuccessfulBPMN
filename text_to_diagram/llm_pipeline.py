from __future__ import annotations

import time
import re
from typing import Any, Dict, List, Optional

from narrator.providers.base import BaseNarratorProvider
from narrator.providers.factory import create_narrator_provider

from text_to_diagram.llm_config import (
    TextToDiagramLLMConfig,
    resolve_text_to_diagram_llm_config,
    resolve_text_to_diagram_runtime_config,
)
from text_to_diagram.llm_postprocess import TextToDiagramIRParseError, parse_and_repair_ir
from text_to_diagram.llm_prompts import build_text_to_ir_prompt_pack


class TextToDiagramPipelineError(RuntimeError):
    pass


def run_text_to_ir_pipeline(
    source_text: str,
    llm_cfg_overrides: Optional[Dict[str, Any]] = None,
    runtime_overrides: Optional[Dict[str, Any]] = None,
    provider: Optional[BaseNarratorProvider] = None,
) -> Dict[str, Any]:
    llm_cfg = resolve_text_to_diagram_llm_config(llm_cfg_overrides)
    _validate_input_text(source_text, llm_cfg)

    runtime_cfg = resolve_text_to_diagram_runtime_config(runtime_overrides)
    llm = provider or create_narrator_provider(runtime_cfg)

    started_at = time.perf_counter()
    parse_errors: List[str] = []
    raw_outputs: List[str] = []

    for attempt in range(llm_cfg.max_reasks + 1):
        prompt_pack = build_text_to_ir_prompt_pack(
            source_text=source_text,
            previous_output=raw_outputs[-1] if raw_outputs else None,
            parse_error=parse_errors[-1] if parse_errors else None,
        )
        raw = llm.generate(
            system_prompt=prompt_pack["system_prompt"],
            user_prompt=prompt_pack["user_prompt"],
        )
        raw_outputs.append(raw)
        try:
            ir, repair_issues = parse_and_repair_ir(raw)
            quality_failures: List[str] = []
            if llm_cfg.quality_gate_enabled:
                quality_failures = _quality_failures(ir)
            if quality_failures:
                parse_errors.append("quality gate failed: " + "; ".join(quality_failures))
                continue
            return {
                "status": "ok" if not parse_errors else "degraded",
                "ir": ir,
                "issues": list(ir.get("issues", [])),
                "meta": {
                    "prompt_version": prompt_pack["prompt_version"],
                    "attempts": attempt + 1,
                    "parse_errors": parse_errors,
                    "repair_issues_count": len(repair_issues),
                    "quality_gate_enabled": llm_cfg.quality_gate_enabled,
                    "duration_ms": int((time.perf_counter() - started_at) * 1000),
                    "runtime": {
                        "provider": runtime_cfg.provider,
                        "temperature": runtime_cfg.temperature,
                        "max_tokens": runtime_cfg.max_tokens,
                        "n_ctx": runtime_cfg.n_ctx,
                        "n_threads": runtime_cfg.n_threads,
                        "cpu_only": runtime_cfg.cpu_only,
                    },
                },
            }
        except TextToDiagramIRParseError as exc:
            parse_errors.append(str(exc))

    if not llm_cfg.repair_fallback_enabled:
        raise TextToDiagramPipelineError(
            "failed to parse LLM JSON output after all attempts"
        )

    # Last fallback to keep pipeline usable and preserve soft-fail behavior.
    fallback_ir = _build_minimal_fallback_ir(parse_errors)
    return {
        "status": "degraded",
        "ir": fallback_ir,
        "issues": fallback_ir["issues"],
        "meta": {
            "prompt_version": "text-to-ir-prompt.v1",
            "attempts": llm_cfg.max_reasks + 1,
            "parse_errors": parse_errors,
            "repair_issues_count": 0,
            "quality_gate_enabled": llm_cfg.quality_gate_enabled,
            "duration_ms": int((time.perf_counter() - started_at) * 1000),
            "runtime": {
                "provider": runtime_cfg.provider,
                "temperature": runtime_cfg.temperature,
                "max_tokens": runtime_cfg.max_tokens,
                "n_ctx": runtime_cfg.n_ctx,
                "n_threads": runtime_cfg.n_threads,
                "cpu_only": runtime_cfg.cpu_only,
            },
        },
    }


def _validate_input_text(source_text: str, cfg: TextToDiagramLLMConfig) -> None:
    if not isinstance(source_text, str) or not source_text.strip():
        raise TextToDiagramPipelineError("'source_text' must be non-empty string")
    if len(source_text) > cfg.max_input_chars:
        raise TextToDiagramPipelineError(
            f"'source_text' exceeds max_input_chars={cfg.max_input_chars}"
        )


def _build_minimal_fallback_ir(parse_errors: List[str]) -> Dict[str, Any]:
    return {
        "nodes": [
            {"id": "n1", "type": "shape", "role": "start", "text": "Старт", "lane_id": None},
            {"id": "n2", "type": "shape", "role": "end", "text": "Завершение", "lane_id": None},
        ],
        "edges": [{"id": "e1", "from": "n1", "to": "n2", "type": "sequential", "text": None}],
        "lanes": [],
        "meta": {
            "schema_version": "process-ir.v1",
            "direction": "LR",
            "source": "text_to_diagram",
            "language": "ru",
        },
        "issues": [
            {
                "code": "IR_PARSE_FAILED_FALLBACK",
                "severity": "error",
                "message": "LLM output could not be parsed as JSON; fallback IR was generated",
                "entity_type": None,
                "entity_id": None,
            },
            {
                "code": "IR_PARSE_ERROR_DETAILS",
                "severity": "info",
                "message": "; ".join(parse_errors) if parse_errors else "unknown parse failure",
                "entity_type": None,
                "entity_id": None,
            },
        ],
    }


_PLACEHOLDER_RE = re.compile(r"^[nle]\d+$", flags=re.IGNORECASE)


def _quality_failures(ir: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    nodes = ir.get("nodes", [])
    edges = ir.get("edges", [])
    lanes = ir.get("lanes", [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []
    if not isinstance(lanes, list):
        lanes = []

    node_ids = {n.get("id") for n in nodes if isinstance(n, dict) and isinstance(n.get("id"), str)}

    if len(nodes) > 1 and len(edges) == 0:
        reasons.append("no edges for multi-node graph")

    for e in edges:
        if not isinstance(e, dict):
            continue
        src, dst = e.get("from"), e.get("to")
        if not isinstance(src, str) or not isinstance(dst, str):
            reasons.append("edge missing from/to")
            break
        if src not in node_ids or dst not in node_ids:
            reasons.append("edge references unknown node")
            break

    business_nodes = [
        n
        for n in nodes
        if isinstance(n, dict)
        and n.get("role") in {"action", "decision", "parallel", "inclusive", "event_intermediate"}
    ]
    if business_nodes:
        meaningful = 0
        for n in business_nodes:
            txt = n.get("text")
            if not isinstance(txt, str):
                continue
            t = txt.strip()
            if not t:
                continue
            if _PLACEHOLDER_RE.match(t):
                continue
            meaningful += 1
        if meaningful == 0:
            reasons.append("no meaningful business node text")

    if lanes:
        lane_names = [
            x.get("name")
            for x in lanes
            if isinstance(x, dict) and isinstance(x.get("name"), str) and x.get("name").strip()
        ]
        if lane_names and all(name.strip().lower().startswith("lane ") for name in lane_names):
            reasons.append("all lane names are generic placeholders")

    return reasons
