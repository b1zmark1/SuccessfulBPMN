from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cairosvg
from PIL import Image


SUPPORTED_ARTIFACT_TYPES = {"mermaid", "plantuml", "bpmn"}
SUPPORTED_IMAGE_FORMATS = {"png", "jpg"}


def render_artifact_to_image(
    artifact_type: str,
    artifact_text: str,
    image_format: str = "png",
) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    started = time.perf_counter()
    diagnostics = _collect_renderer_diagnostics(artifact_type, artifact_text)

    if artifact_type not in SUPPORTED_ARTIFACT_TYPES:
        return _degraded_result(
            artifact_type=artifact_type,
            image_format=image_format,
            issues=[_issue("RENDER_UNSUPPORTED_ARTIFACT", "error", f"unsupported artifact_type '{artifact_type}'")],
            duration_ms=int((time.perf_counter() - started) * 1000),
            diagnostics=diagnostics,
        )
    if image_format not in SUPPORTED_IMAGE_FORMATS:
        return _degraded_result(
            artifact_type=artifact_type,
            image_format=image_format,
            issues=[_issue("RENDER_UNSUPPORTED_IMAGE_FORMAT", "error", f"unsupported image_format '{image_format}'")],
            duration_ms=int((time.perf_counter() - started) * 1000),
            diagnostics=diagnostics,
        )
    if not isinstance(artifact_text, str) or not artifact_text.strip():
        return _degraded_result(
            artifact_type=artifact_type,
            image_format=image_format,
            issues=[_issue("RENDER_EMPTY_ARTIFACT", "error", "artifact_text must be non-empty string")],
            duration_ms=int((time.perf_counter() - started) * 1000),
            diagnostics=diagnostics,
        )

    try:
        png_bytes = _render_png_bytes(artifact_type, artifact_text)
    except Exception as exc:
        code = "RENDER_FAILED"
        msg = f"renderer failed for '{artifact_type}': {exc}"
        low = str(exc).lower()
        if artifact_type == "bpmn" and "no diagram to display" in low:
            code = "BPMN_RENDER_NO_DIAGRAM"
            msg = (
                "bpmn renderer cannot display XML because BPMN DI is missing "
                "(no <bpmndi:BPMNDiagram>/<bpmndi:BPMNPlane>)"
            )
        issues.append(
            _issue(
                code,
                "warning",
                msg,
            )
        )
        issues.append(
            _issue(
                "RENDER_DEBUG_DETAILS",
                "info",
                str(exc),
            )
        )
        return _degraded_result(
            artifact_type=artifact_type,
            image_format=image_format,
            issues=issues,
            duration_ms=int((time.perf_counter() - started) * 1000),
            diagnostics=diagnostics,
        )

    png_b64 = base64.b64encode(png_bytes).decode("ascii")
    jpg_b64: Optional[str] = None
    if image_format == "jpg":
        try:
            jpg_bytes = _convert_png_to_jpg(png_bytes)
            jpg_b64 = base64.b64encode(jpg_bytes).decode("ascii")
        except Exception as exc:
            issues.append(
                _issue(
                    "RENDER_JPG_CONVERSION_FAILED",
                    "warning",
                    f"png->jpg conversion failed: {exc}",
                )
            )

    return {
        "status": "degraded" if issues else "ok",
        "image_png_base64": png_b64,
        "image_jpg_base64": jpg_b64,
        "issues": issues,
        "meta": {
            "artifact_type": artifact_type,
            "image_format": image_format,
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "diagnostics": diagnostics,
        },
    }


def _render_png_bytes(artifact_type: str, artifact_text: str) -> bytes:
    if artifact_type == "mermaid":
        return _render_mermaid_png_bytes(artifact_text)
    if artifact_type == "plantuml":
        return _render_plantuml_png_bytes(artifact_text)
    if artifact_type == "bpmn":
        return _render_bpmn_png_bytes(artifact_text)
    raise RuntimeError(f"unsupported artifact type '{artifact_type}'")


def _render_mermaid_png_bytes(mmd_text: str) -> bytes:
    mmdc = shutil.which("mmdc")
    if not mmdc:
        raise RuntimeError("mermaid-cli (mmdc) is not installed")

    with tempfile.TemporaryDirectory(prefix="t2d_mermaid_") as tmp:
        in_path = Path(tmp) / "diagram.mmd"
        out_path = Path(tmp) / "diagram.png"
        in_path.write_text(mmd_text, encoding="utf-8")

        _run_command([mmdc, "-i", str(in_path), "-o", str(out_path)])

        if not out_path.exists():
            raise RuntimeError("mmdc did not produce png output")
        return out_path.read_bytes()


def _render_plantuml_png_bytes(puml_text: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="t2d_plantuml_") as tmp:
        in_path = Path(tmp) / "diagram.puml"
        in_path.write_text(puml_text, encoding="utf-8")
        out_path = Path(tmp) / "diagram.png"

        plantuml_cmd = shutil.which("plantuml")
        if plantuml_cmd:
            _run_command([plantuml_cmd, "-tpng", str(in_path), "-o", str(Path(tmp))])
            if out_path.exists():
                return out_path.read_bytes()

        plantuml_jar = os.environ.get("PLANTUML_JAR")
        java_cmd = shutil.which("java")
        if plantuml_jar and java_cmd:
            _run_command([java_cmd, "-jar", plantuml_jar, "-tpng", str(in_path), "-o", str(Path(tmp))])
            if out_path.exists():
                return out_path.read_bytes()

        raise RuntimeError("PlantUML renderer is not available (plantuml CLI or PLANTUML_JAR+java)")


def _render_bpmn_png_bytes(bpmn_xml: str) -> bytes:
    node_cmd = shutil.which("node")
    if not node_cmd:
        raise RuntimeError("node is not installed for BPMN renderer")

    script = Path("tools") / "bpmn_dataset" / "render_bpmn_svg.js"
    if not script.exists():
        raise RuntimeError("BPMN renderer script not found: tools/bpmn_dataset/render_bpmn_svg.js")

    with tempfile.TemporaryDirectory(prefix="t2d_bpmn_") as tmp:
        in_path = Path(tmp) / "diagram.bpmn"
        svg_path = Path(tmp) / "diagram.svg"
        meta_path = Path(tmp) / "diagram_meta.json"
        in_path.write_text(bpmn_xml, encoding="utf-8")

        _run_command([node_cmd, str(script), str(in_path), str(svg_path), str(meta_path)])
        if not svg_path.exists():
            raise RuntimeError("BPMN renderer did not produce SVG output")

        svg_data = svg_path.read_text(encoding="utf-8")
        return cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))


def _convert_png_to_jpg(png_bytes: bytes) -> bytes:
    with tempfile.TemporaryDirectory(prefix="t2d_jpg_") as tmp:
        in_path = Path(tmp) / "in.png"
        out_path = Path(tmp) / "out.jpg"
        in_path.write_bytes(png_bytes)
        with Image.open(in_path) as img:
            rgb = img.convert("RGB")
            rgb.save(out_path, format="JPEG", quality=90, optimize=True)
        return out_path.read_bytes()


def _run_command(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(stderr or f"command failed: {' '.join(cmd)}")


def _degraded_result(
    artifact_type: str,
    image_format: str,
    issues: List[Dict[str, Any]],
    duration_ms: int,
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "status": "degraded",
        "image_png_base64": None,
        "image_jpg_base64": None,
        "issues": issues,
        "meta": {
            "artifact_type": artifact_type,
            "image_format": image_format,
            "duration_ms": duration_ms,
            "diagnostics": diagnostics,
        },
    }


def _collect_renderer_diagnostics(artifact_type: str, artifact_text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "artifact_type": artifact_type,
        "artifact_len": len(artifact_text) if isinstance(artifact_text, str) else 0,
        "tools": {
            "mmdc": shutil.which("mmdc"),
            "plantuml": shutil.which("plantuml"),
            "java": shutil.which("java"),
            "node": shutil.which("node"),
            "PLANTUML_JAR": os.environ.get("PLANTUML_JAR"),
        },
    }
    if artifact_type == "bpmn" and isinstance(artifact_text, str):
        out["bpmn"] = {
            "has_bpmn_diagram_tag": "<bpmndi:BPMNDiagram" in artifact_text,
            "has_bpmn_plane_tag": "<bpmndi:BPMNPlane" in artifact_text,
            "has_process_tag": "<bpmn:process" in artifact_text,
        }
    return out


def _issue(code: str, severity: str, message: str) -> Dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "entity_type": None,
        "entity_id": None,
    }
