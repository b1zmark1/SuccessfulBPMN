from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class DiagnosticsError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiagnosticsConfig:
    debug_mode: bool = True
    low_direction_confidence_threshold: float = 0.35
    weak_edge_score_threshold: float = 0.35
    max_warnings: int = 12


def build_uncertainty_and_diagnostics(
    payload: Dict[str, Any],
    cfg: Optional[DiagnosticsConfig] = None,
) -> Dict[str, Any]:
    """
    Step 10:
    - collect ambiguity/uncertainty notes
    - keep concise meta.warnings
    - optionally attach per-node/per-edge debug diagnostics
    """
    if cfg is None:
        cfg = DiagnosticsConfig()

    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise DiagnosticsError("payload must contain 'nodes' and 'edges' lists")

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}

    existing_warnings = meta.get("warnings", [])
    warnings: List[str] = []
    if isinstance(existing_warnings, list):
        warnings.extend([str(x) for x in existing_warnings])

    uncertainty_notes: List[Dict[str, Any]] = []

    # 1) Direction confidence uncertainty
    dir_conf = _to_float(meta.get("direction_confidence"), 0.0)
    if dir_conf < cfg.low_direction_confidence_threshold:
        warnings.append(f"low_direction_confidence:{dir_conf:.3f}")
        uncertainty_notes.append(
            {
                "code": "low_direction_confidence",
                "severity": "medium",
                "message": "Global process direction confidence is low.",
                "value": dir_conf,
            }
        )

    # 2) Missing/weak flow evidence
    ec_stats = payload.get("edge_candidate_stats", {})
    if isinstance(ec_stats, dict):
        flow_nodes = int(ec_stats.get("flow_nodes", 0))
        if flow_nodes == 0:
            warnings.append("missing_flow_hints: no flow nodes detected")
            uncertainty_notes.append(
                {
                    "code": "missing_flow_hints",
                    "severity": "high",
                    "message": "No flow-like detections were found; edges rely on geometry only.",
                }
            )

    # 3) Conflicting containers
    c_stats = payload.get("container_stats", {})
    if isinstance(c_stats, dict):
        conflicts = int(c_stats.get("container_conflicts", 0))
        if conflicts > 0:
            warnings.append(f"container_conflicts:{conflicts}")
            uncertainty_notes.append(
                {
                    "code": "container_conflicts",
                    "severity": "medium",
                    "message": "Overlapping container detections were found.",
                    "value": conflicts,
                }
            )

    # 4) Weak edge evidence
    candidate_index = _index_candidate_scores(payload.get("edge_candidates"))
    weak_edges = 0
    unknown_edges = 0
    for e in edges:
        if not isinstance(e, dict):
            continue
        et = str(e.get("type", "unknown"))
        if et == "unknown":
            unknown_edges += 1
        key = (str(e.get("from", "")), str(e.get("to", "")))
        sc = candidate_index.get(key, 0.0)
        if sc < cfg.weak_edge_score_threshold:
            weak_edges += 1
    if weak_edges > 0:
        warnings.append(f"weak_edge_evidence:{weak_edges}/{len(edges)}")
        uncertainty_notes.append(
            {
                "code": "weak_edge_evidence",
                "severity": "medium",
                "message": "Some edges were formed with weak geometric/flow support.",
                "value": {"weak_edges": weak_edges, "total_edges": len(edges)},
            }
        )
    if unknown_edges > 0:
        warnings.append(f"unknown_edge_types:{unknown_edges}/{len(edges)}")

    # Keep warnings concise and deterministic
    warnings = _unique_preserve_order(warnings)[: cfg.max_warnings]

    out_meta = dict(meta)
    out_meta["warnings"] = warnings
    out_meta["uncertainties"] = uncertainty_notes

    out = dict(payload)
    out["meta"] = out_meta
    out["diagnostics_summary"] = {
        "nodes_total": len([n for n in nodes if isinstance(n, dict)]),
        "edges_total": len([e for e in edges if isinstance(e, dict)]),
        "uncertainty_notes": len(uncertainty_notes),
        "warnings_kept": len(warnings),
    }

    if cfg.debug_mode:
        out["diagnostics_debug"] = {
            "node_debug": _build_node_debug(nodes),
            "edge_debug": _build_edge_debug(edges, candidate_index),
        }

    return out


def _build_node_debug(nodes: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        if not isinstance(nid, str):
            continue
        role = str(n.get("role", "unknown"))
        score = _to_float(n.get("score"), 0.0)
        # deterministic lightweight confidence proxy
        role_bonus = 0.1 if role != "unknown" else 0.0
        conf = max(0.0, min(1.0, score + role_bonus))
        reasons = [f"score={score:.3f}", f"role={role}"]
        out.append(
            {
                "id": nid,
                "confidence_proxy": conf,
                "reasons": reasons,
            }
        )
    out.sort(key=lambda x: x["id"])
    return out


def _build_edge_debug(
    edges: List[Any],
    candidate_index: Dict[Tuple[str, str], float],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        src = e.get("from")
        dst = e.get("to")
        et = e.get("type")
        if not (isinstance(src, str) and isinstance(dst, str) and isinstance(et, str)):
            continue
        score = candidate_index.get((src, dst), 0.0)
        reasons = [f"type={et}", f"candidate_score={score:.3f}"]
        out.append(
            {
                "from": src,
                "to": dst,
                "type": et,
                "candidate_score": score,
                "reasons": reasons,
            }
        )
    out.sort(key=lambda x: (x["from"], x["to"], x["type"]))
    return out


def _index_candidate_scores(edge_candidates: Any) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    if not isinstance(edge_candidates, list):
        return out
    for c in edge_candidates:
        if not isinstance(c, dict):
            continue
        src = c.get("from")
        dst = c.get("to")
        score = c.get("score")
        if not (isinstance(src, str) and isinstance(dst, str) and isinstance(score, (int, float))):
            continue
        key = (src, dst)
        sc = float(score)
        if key not in out or sc > out[key]:
            out[key] = sc
    return out


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out
