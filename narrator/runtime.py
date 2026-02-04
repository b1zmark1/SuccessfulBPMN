from __future__ import annotations

import time
from threading import Lock
from typing import Any, Dict, Tuple

from narrator.config import NarratorRuntimeConfig, build_runtime_meta
from narrator.providers.factory import create_narrator_provider


class NarratorRuntimeError(RuntimeError):
    pass


_PROVIDER_LOCK = Lock()
_PROVIDER_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _provider_cache_key(cfg: NarratorRuntimeConfig) -> Tuple[Any, ...]:
    return (
        cfg.provider,
        cfg.model_path,
        cfg.n_ctx,
        cfg.n_threads,
        cfg.n_batch,
        cfg.cpu_only,
    )


def _get_cached_provider(cfg: NarratorRuntimeConfig) -> Any:
    key = _provider_cache_key(cfg)
    with _PROVIDER_LOCK:
        provider = _PROVIDER_CACHE.get(key)
        if provider is None:
            provider = create_narrator_provider(cfg)
            _PROVIDER_CACHE[key] = provider
        return provider


def run_single_llm_call(
    prompt_pack: Dict[str, str],
    runtime_cfg: NarratorRuntimeConfig,
) -> Dict[str, Any]:
    required = {"prompt_version", "system_prompt", "user_prompt"}
    if set(prompt_pack.keys()) != required:
        raise NarratorRuntimeError(f"prompt_pack must contain exactly {sorted(required)}")

    provider = _get_cached_provider(runtime_cfg)

    t0 = time.perf_counter()
    text = provider.generate(
        system_prompt=prompt_pack["system_prompt"],
        user_prompt=prompt_pack["user_prompt"],
    )
    duration_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "text": text,
        "narrator_meta": {
            "prompt_version": prompt_pack["prompt_version"],
            **build_runtime_meta(runtime_cfg, duration_ms=duration_ms),
        },
    }
