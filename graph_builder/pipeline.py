from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .containers import assign_container_hierarchy
from .diagnostics import build_uncertainty_and_diagnostics
from .direction import infer_process_direction
from .edge_candidates import build_edge_candidates
from .edges import finalize_edges
from .grouping import group_normalized_detections
from .nodes import build_graph_nodes
from .normalize import normalize_ensemble_input
from .semantic_projection import (
    SemanticProjectionConfig,
    project_graph_to_semantic,
)
from .semantic_contract import check_semantic_projection_contract
from .serialize import serialize_graph_output
from .text_hooks import attach_text_placeholders_and_hooks
from .text_merge import merge_adjacent_text_nodes
from .title_hints import assign_title_hints


@dataclass(frozen=True)
class ProjectionStageStatus:
    status: str = "ok"


def build_graph_from_ensemble(ensemble_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the deterministic graph-builder pipeline.
    """
    x = normalize_ensemble_input(ensemble_payload)
    x = group_normalized_detections(x)
    x = infer_process_direction(x)
    x = build_graph_nodes(x)
    x = assign_container_hierarchy(x)
    x = build_edge_candidates(x)
    x = finalize_edges(x)
    x = merge_adjacent_text_nodes(x)
    x = attach_text_placeholders_and_hooks(x)
    x = assign_title_hints(x)
    x = build_uncertainty_and_diagnostics(x)
    return serialize_graph_output(x)


def run_graph_to_semantic_pipeline(
    ensemble_payload: Dict[str, Any],
    projection_cfg: Optional[SemanticProjectionConfig] = None,
) -> Dict[str, Any]:
    """
    Fixed integration point between Graph Builder and LLM Narrator.

    Returns:
      - semantic_payload: input-ready compact semantic JSON for LLM
      - projection_meta: service metadata of this stage
    """
    graph_payload = build_graph_from_ensemble(ensemble_payload)
    semantic_payload = project_graph_to_semantic(graph_payload, cfg=projection_cfg)
    contract_valid, contract_errors = check_semantic_projection_contract(semantic_payload)

    return {
        "semantic_payload": semantic_payload,
        "projection_meta": {
            "status": ProjectionStageStatus().status,
            "contract_valid": contract_valid,
            "contract_errors": contract_errors,
            "semantic_schema_version": semantic_payload["meta"]["schema_version"],
            "source_graph_schema_version": semantic_payload["meta"]["source_graph_schema_version"],
            "steps_count": len(semantic_payload.get("steps", [])),
        },
    }
