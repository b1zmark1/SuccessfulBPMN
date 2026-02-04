from __future__ import annotations

import argparse
import base64
import json
from typing import Any

from text_to_diagram.orchestrator import run_text_to_diagram_use_case
from text_to_diagram.render_layer import render_artifact_to_image


def _valid_base64_payload(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        base64.b64decode(value, validate=True)
        return True
    except Exception:
        return False


def _first_issue(issues: list[dict[str, Any]]) -> str:
    if not issues:
        return "no_issues_reported"
    first = issues[0]
    code = str(first.get("code", "unknown"))
    msg = str(first.get("message", ""))
    return f"{code}: {msg}".strip()


def run_text_to_image_pipeline(prompt: str) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt is required")

    out = run_text_to_diagram_use_case(
        source_text=prompt,
        runtime_overrides={
            "max_tokens": 2200,
            "n_ctx": 4096,
            "temperature": 0.0,
        },
        llm_cfg_overrides={
            "max_reasks": 2,
            "max_input_chars": 4000,
            "quality_gate_enabled": True,
        },
        render={
            "enabled": True,
            "artifact_type": "bpmn",
            "image_format": "png",
        },
    )

    artifacts = out.get("artifacts", {}) if isinstance(out.get("artifacts"), dict) else {}
    issues = out.get("issues", []) if isinstance(out.get("issues"), list) else []

    bpmn_xml = artifacts.get("bpmn_xml") if isinstance(artifacts.get("bpmn_xml"), str) else None
    mermaid_mmd = artifacts.get("mermaid_mmd") if isinstance(artifacts.get("mermaid_mmd"), str) else None
    plantuml_puml = artifacts.get("plantuml_puml") if isinstance(artifacts.get("plantuml_puml"), str) else None

    def _with_sources(image_url: str) -> dict[str, Any]:
        return {
            "image_url": image_url,
            "bpmn_xml": bpmn_xml,
            "mermaid_mmd": mermaid_mmd,
            "plantuml_puml": plantuml_puml,
        }

    # Primary path: orchestrator render output.
    png_b64 = artifacts.get("image_png_base64")
    if _valid_base64_payload(png_b64):
        return _with_sources(f"data:image/png;base64,{png_b64}")

    # Secondary path: use jpg if renderer returned only JPG.
    jpg_b64 = artifacts.get("image_jpg_base64")
    if _valid_base64_payload(jpg_b64):
        return _with_sources(f"data:image/jpeg;base64,{jpg_b64}")

    # Tertiary path: try re-rendering from textual artifacts with fallback artifact types.
    rerender_errors: list[str] = []
    for artifact_type, key in (("bpmn", "bpmn_xml"), ("mermaid", "mermaid_mmd"), ("plantuml", "plantuml_puml")):
        artifact_text = artifacts.get(key)
        if not isinstance(artifact_text, str) or not artifact_text.strip():
            continue
        rerender = render_artifact_to_image(
            artifact_type=artifact_type,
            artifact_text=artifact_text,
            image_format="png",
        )
        rerender_png = rerender.get("image_png_base64")
        if _valid_base64_payload(rerender_png):
            return _with_sources(f"data:image/png;base64,{rerender_png}")
        rerender_issues = rerender.get("issues", []) if isinstance(rerender.get("issues"), list) else []
        rerender_errors.append(f"{artifact_type}: {_first_issue(rerender_issues)}")

    first_issue = _first_issue(issues)
    rerender_part = "; ".join(rerender_errors) if rerender_errors else "rerender_not_attempted"
    raise RuntimeError(
        "Text-to-image pipeline did not return image output. "
        f"orchestrator_status={out.get('status', 'unknown')}; "
        f"orchestrator_issue={first_issue}; "
        f"rerender={rerender_part}"
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Text-to-image pipeline worker entry.")
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    result = run_text_to_image_pipeline(args.prompt)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
