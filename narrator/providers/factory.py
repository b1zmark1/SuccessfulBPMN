from __future__ import annotations

from narrator.config import NarratorRuntimeConfig
from narrator.providers.base import BaseNarratorProvider
from narrator.providers.llama_cpp_provider import LlamaCppNarratorProvider


def create_narrator_provider(cfg: NarratorRuntimeConfig) -> BaseNarratorProvider:
    if cfg.provider == "llama_cpp":
        return LlamaCppNarratorProvider(cfg)
    raise RuntimeError(f"unsupported provider: {cfg.provider}")

