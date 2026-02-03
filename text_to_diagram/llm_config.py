from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from narrator.config import NarratorRuntimeConfig, resolve_runtime_config


@dataclass(frozen=True)
class TextToDiagramLLMConfig:
    max_input_chars: int = 4000
    max_reasks: int = 2
    repair_fallback_enabled: bool = True
    quality_gate_enabled: bool = True


def resolve_text_to_diagram_llm_config(
    overrides: Optional[Dict[str, Any]] = None,
) -> TextToDiagramLLMConfig:
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise RuntimeError("llm config overrides must be an object")

    merged = asdict(TextToDiagramLLMConfig())
    merged.update(overrides)

    max_input_chars = merged.get("max_input_chars")
    if not isinstance(max_input_chars, int) or max_input_chars < 128:
        raise RuntimeError("'max_input_chars' must be integer >= 128")

    max_reasks = merged.get("max_reasks")
    if not isinstance(max_reasks, int) or max_reasks < 0 or max_reasks > 3:
        raise RuntimeError("'max_reasks' must be integer in range [0, 3]")

    repair_fallback_enabled = merged.get("repair_fallback_enabled")
    if not isinstance(repair_fallback_enabled, bool):
        raise RuntimeError("'repair_fallback_enabled' must be boolean")

    quality_gate_enabled = merged.get("quality_gate_enabled")
    if not isinstance(quality_gate_enabled, bool):
        raise RuntimeError("'quality_gate_enabled' must be boolean")

    return TextToDiagramLLMConfig(
        max_input_chars=max_input_chars,
        max_reasks=max_reasks,
        repair_fallback_enabled=repair_fallback_enabled,
        quality_gate_enabled=quality_gate_enabled,
    )


def resolve_text_to_diagram_runtime_config(
    overrides: Optional[Dict[str, Any]] = None,
) -> NarratorRuntimeConfig:
    """
    CPU-oriented deterministic profile for text -> IR JSON generation.
    Reuses existing narrator runtime/provider infrastructure.
    """
    tuned_defaults: Dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": 2200,
        "n_ctx": 4096,
        "timeout_sec": 30,
        "cpu_only": True,
    }
    if overrides:
        tuned_defaults.update(overrides)
    return resolve_runtime_config(tuned_defaults)
