from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from narrator.policies import (
    NarratorPolicyError,
    build_narrator_meta,
    resolve_narrator_policy,
)


def test_policy_defaults():
    cfg = resolve_narrator_policy()
    assert cfg.max_sentences == 10
    assert cfg.missing_text_policy == "generalize"
    assert cfg.branches_policy == "cover_all"


def test_policy_runtime_overrides():
    cfg = resolve_narrator_policy(
        {
            "max_sentences": 7,
            "missing_text_policy": "skip",
            "branches_policy": "cover_all",
        }
    )
    assert cfg.max_sentences == 7
    assert cfg.missing_text_policy == "skip"
    assert cfg.branches_policy == "cover_all"


def test_policy_invalid_values_raise():
    with pytest.raises(NarratorPolicyError):
        resolve_narrator_policy({"max_sentences": 0})

    with pytest.raises(NarratorPolicyError):
        resolve_narrator_policy({"missing_text_policy": "unknown"})

    with pytest.raises(NarratorPolicyError):
        resolve_narrator_policy({"branches_policy": "unknown"})


def test_narrator_meta_contains_applied_policy():
    cfg = resolve_narrator_policy({"max_sentences": 12})
    meta = build_narrator_meta(cfg)
    assert set(meta.keys()) == {"applied_policy"}
    assert meta["applied_policy"]["max_sentences"] == 12
    assert meta["applied_policy"]["missing_text_policy"] == "generalize"
    assert meta["applied_policy"]["branches_policy"] == "cover_all"
