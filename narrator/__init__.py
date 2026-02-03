"""Narrator package (reserved for LLM narration layer)."""

from .policies import (
    NarratorPolicyConfig,
    NarratorPolicyError,
    build_narrator_meta,
    resolve_narrator_policy,
)
from .config import (
    NarratorRuntimeConfig,
    NarratorRuntimeConfigError,
    build_runtime_meta,
    resolve_runtime_config,
)
from .prompts import (
    NarratorPromptConfig,
    NarratorPromptError,
    build_narrator_prompts,
    build_prompt_meta,
)
from .runtime import (
    NarratorRuntimeError,
    run_single_llm_call,
)
from .orchestrator import (
    NarratorOrchestrationError,
    run_narration,
    run_narration_from_ensemble,
)
from .postprocess import (
    NarratorPostprocessConfig,
    NarratorPostprocessError,
    postprocess_narration_text,
)
from .observability import (
    build_observability_meta,
    generate_trace_id,
)
from .acceptance import (
    evaluate_acceptance_case,
)
from .e2e_validation import (
    run_e2e_validation,
)

__all__ = [
    "NarratorPolicyConfig",
    "NarratorPolicyError",
    "resolve_narrator_policy",
    "build_narrator_meta",
    "NarratorRuntimeConfig",
    "NarratorRuntimeConfigError",
    "resolve_runtime_config",
    "build_runtime_meta",
    "NarratorPromptConfig",
    "NarratorPromptError",
    "build_narrator_prompts",
    "build_prompt_meta",
    "NarratorRuntimeError",
    "run_single_llm_call",
    "NarratorOrchestrationError",
    "run_narration",
    "run_narration_from_ensemble",
    "NarratorPostprocessConfig",
    "NarratorPostprocessError",
    "postprocess_narration_text",
    "generate_trace_id",
    "build_observability_meta",
    "evaluate_acceptance_case",
    "run_e2e_validation",
]
