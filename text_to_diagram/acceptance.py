from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List


def evaluate_text_to_diagram_case(
    case_name: str,
    result: Dict[str, Any],
    require_lane: bool = False,
    require_branching: bool = False,
) -> Dict[str, Any]:
    artifacts = result.get("artifacts", {})
    issues = result.get("issues", [])
    status = result.get("status")

    ir_json = artifacts.get("ir_json")
    mermaid_mmd = artifacts.get("mermaid_mmd")
    bpmn_xml = artifacts.get("bpmn_xml")
    puml = artifacts.get("plantuml_puml")

    checklist: Dict[str, bool] = {
        "status_supported": status in {"ok", "degraded"},
        "artifacts_present": isinstance(artifacts, dict),
        "ir_sections_present": _ir_sections_present(ir_json),
        "mermaid_non_empty": isinstance(mermaid_mmd, str) and bool(mermaid_mmd.strip()),
        "bpmn_well_formed": _bpmn_well_formed(bpmn_xml),
        "plantuml_non_empty": isinstance(puml, str) and bool(puml.strip()),
        "issues_is_list": isinstance(issues, list),
    }

    if require_lane:
        checklist["lane_reflected"] = _lane_reflected(ir_json, mermaid_mmd, bpmn_xml, puml)
    if require_branching:
        checklist["branching_reflected"] = _branching_reflected(ir_json, mermaid_mmd, bpmn_xml, puml)

    reasons: List[str] = []
    if status == "ok" and _has_error_severity(issues):
        reasons.append("status=ok при наличии issues с severity=error")
    if not checklist["ir_sections_present"]:
        reasons.append("IR не содержит обязательные секции")

    passed = all(checklist.values()) and not reasons
    return {
        "case": case_name,
        "passed": passed,
        "checklist": checklist,
        "reasons": reasons,
    }


def _ir_sections_present(ir_json: Any) -> bool:
    if not isinstance(ir_json, dict):
        return False
    required = {"nodes", "edges", "lanes", "meta", "issues"}
    return required.issubset(set(ir_json.keys()))


def _bpmn_well_formed(xml_text: Any) -> bool:
    if not isinstance(xml_text, str) or not xml_text.strip():
        return False
    try:
        ET.fromstring(xml_text)
        return True
    except Exception:
        return False


def _lane_reflected(ir_json: Any, mermaid_mmd: Any, bpmn_xml: Any, puml: Any) -> bool:
    if not isinstance(ir_json, dict):
        return False
    lanes = ir_json.get("lanes", [])
    if not isinstance(lanes, list) or len(lanes) == 0:
        return False

    lane_names = [
        str(x.get("name")).strip()
        for x in lanes
        if isinstance(x, dict) and isinstance(x.get("name"), str) and x.get("name").strip()
    ]
    if not lane_names:
        return False

    m_ok = isinstance(mermaid_mmd, str) and "subgraph" in mermaid_mmd
    b_ok = isinstance(bpmn_xml, str) and "<bpmn:lane" in bpmn_xml
    p_ok = isinstance(puml, str) and re.search(r"\bstate\s+\".*\"\s+as\s+lane_", puml) is not None
    return m_ok and b_ok and p_ok


def _branching_reflected(ir_json: Any, mermaid_mmd: Any, bpmn_xml: Any, puml: Any) -> bool:
    if not isinstance(ir_json, dict):
        return False
    nodes = ir_json.get("nodes", [])
    if not isinstance(nodes, list):
        return False
    has_branch_node = any(
        isinstance(n, dict) and n.get("role") in {"decision", "parallel", "inclusive"} for n in nodes
    )
    if not has_branch_node:
        return False

    m_ok = isinstance(mermaid_mmd, str) and ("{" in mermaid_mmd)
    b_ok = isinstance(bpmn_xml, str) and (
        "<bpmn:exclusiveGateway" in bpmn_xml
        or "<bpmn:parallelGateway" in bpmn_xml
        or "<bpmn:inclusiveGateway" in bpmn_xml
    )
    p_ok = isinstance(puml, str) and any(tag in puml for tag in ("<<decision>>", "<<parallel>>", "<<inclusive>>"))
    return m_ok and b_ok and p_ok


def _has_error_severity(issues: Any) -> bool:
    if not isinstance(issues, list):
        return False
    for it in issues:
        if isinstance(it, dict) and it.get("severity") == "error":
            return True
    return False

