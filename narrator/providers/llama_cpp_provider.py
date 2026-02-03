from __future__ import annotations

from typing import Any

from narrator.config import NarratorRuntimeConfig
from narrator.providers.base import BaseNarratorProvider


class LlamaCppProviderError(RuntimeError):
    pass


class LlamaCppNarratorProvider(BaseNarratorProvider):
    def __init__(self, cfg: NarratorRuntimeConfig) -> None:
        self._cfg = cfg
        self._llm = self._build_client(cfg)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
        )
        try:
            text = response["choices"][0]["message"]["content"]
        except Exception as exc:  # pragma: no cover - defensive
            raise LlamaCppProviderError("unexpected llama_cpp response format") from exc

        if not isinstance(text, str):
            raise LlamaCppProviderError("llama_cpp returned non-string content")
        return text.strip()

    @staticmethod
    def _build_client(cfg: NarratorRuntimeConfig) -> Any:
        try:
            from llama_cpp import Llama
        except Exception as exc:
            raise LlamaCppProviderError(
                "llama_cpp is not installed; install llama-cpp-python for GGUF CPU runtime"
            ) from exc

        return Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_batch=cfg.n_batch,
            n_gpu_layers=0 if cfg.cpu_only else -1,
            verbose=False,
        )

