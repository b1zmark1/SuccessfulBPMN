from __future__ import annotations

import argparse
import base64
import json
from typing import Any

from text_to_diagram.orchestrator import run_text_to_diagram_use_case


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

    png_b64 = out.get("artifacts", {}).get("image_png_base64")
    if not isinstance(png_b64, str) or not png_b64.strip():
        raise RuntimeError("Text-to-image pipeline did not return PNG output")

    # Validate base64 payload early.
    base64.b64decode(png_b64)
    return {"image_url": f"data:image/png;base64,{png_b64}"}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Text-to-image pipeline worker entry.")
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    result = run_text_to_image_pipeline(args.prompt)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
