from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from graph_builder.normalize import normalize_ensemble_input
from graph_builder.grouping import group_normalized_detections
from graph_builder.direction import infer_process_direction
from graph_builder.nodes import build_graph_nodes
from graph_builder.containers import assign_container_hierarchy
from graph_builder.edge_candidates import build_edge_candidates
from graph_builder.edges import finalize_edges
from graph_builder.text_merge import merge_adjacent_text_nodes
from graph_builder.text_hooks import attach_text_placeholders_and_hooks
from graph_builder.title_hints import assign_title_hints
from graph_builder.diagnostics import build_uncertainty_and_diagnostics
from graph_builder.serialize import serialize_graph_output


def load_fixture(name: str) -> Dict[str, Any]:
    p = Path(__file__).resolve().parents[1] / "fixtures" / "graph_builder" / name
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_full_pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
    x = normalize_ensemble_input(payload)
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
