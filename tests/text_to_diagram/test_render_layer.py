from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import text_to_diagram.render_layer as render_layer
from text_to_diagram.render_layer import render_artifact_to_image


def _tiny_png_bytes() -> bytes:
    buff = io.BytesIO()
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(buff, format="PNG")
    return buff.getvalue()


def test_render_layer_returns_png_base64(monkeypatch):
    monkeypatch.setattr(render_layer, "_render_png_bytes", lambda artifact_type, artifact_text: _tiny_png_bytes())

    out = render_artifact_to_image("mermaid", "flowchart LR\nA-->B\n", image_format="png")
    assert out["status"] == "ok"
    assert out["image_png_base64"] is not None
    assert out["image_jpg_base64"] is None
    decoded = base64.b64decode(out["image_png_base64"])
    assert decoded.startswith(b"\x89PNG")


def test_render_layer_png_to_jpg_conversion(monkeypatch):
    monkeypatch.setattr(render_layer, "_render_png_bytes", lambda artifact_type, artifact_text: _tiny_png_bytes())

    out = render_artifact_to_image("plantuml", "@startuml\n@enduml\n", image_format="jpg")
    assert out["status"] == "ok"
    assert out["image_png_base64"] is not None
    assert out["image_jpg_base64"] is not None
    decoded_jpg = base64.b64decode(out["image_jpg_base64"])
    assert decoded_jpg.startswith(b"\xff\xd8")


def test_render_layer_fallback_on_renderer_failure(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("renderer missing")

    monkeypatch.setattr(render_layer, "_render_png_bytes", _boom)

    out = render_artifact_to_image("bpmn", "<xml/>", image_format="png")
    assert out["status"] == "degraded"
    assert out["image_png_base64"] is None
    assert any(x["code"] == "RENDER_FAILED" for x in out["issues"])


def test_render_layer_rejects_unsupported_artifact_type():
    out = render_artifact_to_image("unknown", "x", image_format="png")
    assert out["status"] == "degraded"
    assert any(x["code"] == "RENDER_UNSUPPORTED_ARTIFACT" for x in out["issues"])

