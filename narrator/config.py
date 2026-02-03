from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class NarratorRuntimeConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class NarratorRuntimeConfig:
    provider: str = "llama_cpp"
    model_path: str = ""
    n_ctx: int = 2048
    n_threads: int = 8
    n_batch: int = 256
    temperature: float = 0.2
    max_tokens: int = 384
    timeout_sec: int = 120
    cpu_only: bool = True


def resolve_runtime_config(overrides: Optional[Dict[str, Any]] = None) -> NarratorRuntimeConfig:
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise NarratorRuntimeConfigError("runtime overrides must be an object")

    default_model = _default_model_path()
    merged: Dict[str, Any] = asdict(
        NarratorRuntimeConfig(
            model_path=default_model,
            n_threads=max(1, min(16, os.cpu_count() or 8)),
        )
    )
    merged.update(overrides)

    provider = merged.get("provider")
    if provider != "llama_cpp":
        raise NarratorRuntimeConfigError("'provider' must be 'llama_cpp'")

    model_path = merged.get("model_path")
    if not isinstance(model_path, str) or not model_path.strip():
        raise NarratorRuntimeConfigError("'model_path' must be a non-empty string")

    n_ctx = merged.get("n_ctx")
    if not isinstance(n_ctx, int) or n_ctx < 256:
        raise NarratorRuntimeConfigError("'n_ctx' must be integer >= 256")

    n_threads = merged.get("n_threads")
    if not isinstance(n_threads, int) or n_threads < 1:
        raise NarratorRuntimeConfigError("'n_threads' must be integer >= 1")

    n_batch = merged.get("n_batch")
    if not isinstance(n_batch, int) or n_batch < 1:
        raise NarratorRuntimeConfigError("'n_batch' must be integer >= 1")

    temperature = merged.get("temperature")
    if not isinstance(temperature, (int, float)) or temperature < 0:
        raise NarratorRuntimeConfigError("'temperature' must be number >= 0")

    max_tokens = merged.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise NarratorRuntimeConfigError("'max_tokens' must be integer >= 1")

    timeout_sec = merged.get("timeout_sec")
    if not isinstance(timeout_sec, int) or timeout_sec < 1:
        raise NarratorRuntimeConfigError("'timeout_sec' must be integer >= 1")

    cpu_only = merged.get("cpu_only")
    if cpu_only is not True:
        raise NarratorRuntimeConfigError("'cpu_only' must be true for current runtime profile")

    return NarratorRuntimeConfig(
        provider=provider,
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        temperature=float(temperature),
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        cpu_only=cpu_only,
    )


def build_runtime_meta(cfg: NarratorRuntimeConfig, duration_ms: int) -> Dict[str, Any]:
    return {
        "provider": cfg.provider,
        "model_path": cfg.model_path,
        "single_call": True,
        "duration_ms": duration_ms,
        "runtime": {
            "n_ctx": cfg.n_ctx,
            "n_threads": cfg.n_threads,
            "n_batch": cfg.n_batch,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "timeout_sec": cfg.timeout_sec,
            "cpu_only": cfg.cpu_only,
        },
    }


def _default_model_path() -> str:
    base = Path(__file__).resolve().parent
    preferred = base / "qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf"
    if preferred.exists():
        return str(preferred)
    return str(preferred)

