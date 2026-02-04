from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


class NarratorPolicyError(RuntimeError):
    pass


ALLOWED_MISSING_TEXT_POLICIES = {"generalize", "skip", "explicit_placeholder"}
ALLOWED_BRANCHES_POLICIES = {"cover_all", "summarize"}
ALLOWED_OUTPUT_FORMATS = {"narrative", "table"}


@dataclass(frozen=True)
class NarratorPolicyConfig:
    max_sentences: int = 10
    missing_text_policy: str = "generalize"
    branches_policy: str = "cover_all"
    output_format: str = "narrative"


def resolve_narrator_policy(overrides: Optional[Dict[str, Any]] = None) -> NarratorPolicyConfig:
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise NarratorPolicyError("policy overrides must be an object")

    merged: Dict[str, Any] = asdict(NarratorPolicyConfig())
    merged.update(overrides)

    max_sentences = merged.get("max_sentences")
    if not isinstance(max_sentences, int) or max_sentences < 1:
        raise NarratorPolicyError("'max_sentences' must be integer >= 1")

    missing_text_policy = merged.get("missing_text_policy")
    if missing_text_policy not in ALLOWED_MISSING_TEXT_POLICIES:
        raise NarratorPolicyError(
            "'missing_text_policy' must be one of "
            f"{sorted(ALLOWED_MISSING_TEXT_POLICIES)}"
        )

    branches_policy = merged.get("branches_policy")
    if branches_policy not in ALLOWED_BRANCHES_POLICIES:
        raise NarratorPolicyError(
            "'branches_policy' must be one of "
            f"{sorted(ALLOWED_BRANCHES_POLICIES)}"
        )

    output_format = merged.get("output_format")
    if output_format not in ALLOWED_OUTPUT_FORMATS:
        raise NarratorPolicyError(
            "'output_format' must be one of "
            f"{sorted(ALLOWED_OUTPUT_FORMATS)}"
        )

    return NarratorPolicyConfig(
        max_sentences=max_sentences,
        missing_text_policy=missing_text_policy,
        branches_policy=branches_policy,
        output_format=output_format,
    )


def build_narrator_meta(policy: NarratorPolicyConfig) -> Dict[str, Any]:
    return {
        "applied_policy": asdict(policy),
    }
