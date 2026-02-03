from __future__ import annotations

from typing import Any, Dict, Optional

from narrator.providers.base import BaseNarratorProvider
from narrator.providers.factory import create_narrator_provider

from text_to_diagram.llm_config import resolve_text_to_diagram_runtime_config
from text_to_diagram.llm_pipeline import run_text_to_ir_pipeline


class TextToDiagramLLMService:
    """
    Reusable service layer for text -> IR generation.
    Keeps one provider instance for repeated calls in the same process.
    """

    def __init__(
        self,
        runtime_overrides: Optional[Dict[str, Any]] = None,
        provider: Optional[BaseNarratorProvider] = None,
    ) -> None:
        self._runtime_overrides = runtime_overrides or {}
        self._provider = provider

    def generate_ir(
        self,
        source_text: str,
        llm_cfg_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._provider is None:
            runtime_cfg = resolve_text_to_diagram_runtime_config(self._runtime_overrides)
            self._provider = create_narrator_provider(runtime_cfg)

        return run_text_to_ir_pipeline(
            source_text=source_text,
            llm_cfg_overrides=llm_cfg_overrides,
            runtime_overrides=self._runtime_overrides,
            provider=self._provider,
        )

