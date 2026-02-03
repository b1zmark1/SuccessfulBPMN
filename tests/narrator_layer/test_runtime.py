from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from narrator.config import NarratorRuntimeConfigError, resolve_runtime_config
from narrator.runtime import run_single_llm_call


def test_resolve_runtime_config_defaults():
    cfg = resolve_runtime_config()
    assert cfg.provider == "llama_cpp"
    assert cfg.cpu_only is True
    assert cfg.n_ctx >= 256
    assert cfg.n_threads >= 1


def test_resolve_runtime_config_invalid_values():
    with pytest.raises(NarratorRuntimeConfigError):
        resolve_runtime_config({"provider": "other"})
    with pytest.raises(NarratorRuntimeConfigError):
        resolve_runtime_config({"cpu_only": False})
    with pytest.raises(NarratorRuntimeConfigError):
        resolve_runtime_config({"max_tokens": 0})


def test_run_single_llm_call_returns_runtime_meta(monkeypatch):
    fake_module = types.ModuleType("llama_cpp")

    class FakeLlama:
        call_count = 0

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def create_chat_completion(self, **kwargs):
            FakeLlama.call_count += 1
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Готовое описание процесса.",
                        }
                    }
                ]
            }

    fake_module.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    cfg = resolve_runtime_config(
        {
            "model_path": "dummy.gguf",
            "n_threads": 2,
            "max_tokens": 64,
        }
    )
    prompt_pack = {
        "prompt_version": "narrator-prompt.v1",
        "system_prompt": "sys",
        "user_prompt": "usr",
    }

    out = run_single_llm_call(prompt_pack, cfg)
    assert out["text"] == "Готовое описание процесса."
    meta = out["narrator_meta"]
    assert meta["single_call"] is True
    assert meta["provider"] == "llama_cpp"
    assert meta["prompt_version"] == "narrator-prompt.v1"
    assert meta["runtime"]["cpu_only"] is True
    assert FakeLlama.call_count == 1

