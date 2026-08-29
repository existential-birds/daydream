"""Tests for the LOCKED_FIELDS resume guard and per-stage lineage digests (M16, M18)."""

from __future__ import annotations

from typing import Any

import pytest

from daydream.training.lineage import (
    LOCKED_FIELDS,
    ResumeAborted,
    RunIdentity,
    stage_digests,
    validate_resume,
)

_SPEC_FIELDS = {
    "base_model",
    "tokenizer_renderer",
    "max_seq_len",
    "lora_rank",
    "lora_targets",
    "optimizer",
    "learning_rate",
    "corpus_digest",
    "split_digest",
    "profile_policy",
    "reward_version",
    "reward_weights",
    "stack_pins",
}


def _defaults() -> dict[str, Any]:
    return {
        "tokenizer_renderer": "default",
        "max_seq_len": 32768,
        "lora_targets": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "split_digest": "split-d1",
        "profile_policy": "strict",
        "reward_version": "2026.05.28-2",
        "reward_weights": {"outcome": 1.0},
        "stack_pins": {"verifiers": "0.2.1", "prime-rl": "0.7.0"},
    }


def test_every_spec_field_is_locked() -> None:
    assert _SPEC_FIELDS <= set(LOCKED_FIELDS)  # AC5 list is the floor, not the ceiling


def test_resume_aborts_on_any_locked_change() -> None:
    prior = RunIdentity(base_model="Qwen/Qwen3.5-9B", lora_rank=64, corpus_digest="d1", **_defaults())
    changed = prior.__class__(**{**prior.to_dict(), "learning_rate": 2e-5})
    with pytest.raises(ResumeAborted, match="learning_rate"):
        validate_resume(prior, changed)  # loud abort naming the field


def test_resume_aborts_names_every_differing_field() -> None:
    prior = RunIdentity(base_model="Qwen/Qwen3.5-9B", lora_rank=64, corpus_digest="d1", **_defaults())
    changed = prior.__class__(**{**prior.to_dict(), "learning_rate": 2e-5, "max_seq_len": 16384})
    with pytest.raises(ResumeAborted) as excinfo:
        validate_resume(prior, changed)
    assert "learning_rate" in str(excinfo.value)
    assert "max_seq_len" in str(excinfo.value)


def test_resume_aborts_is_value_error() -> None:
    prior = RunIdentity(base_model="m", lora_rank=64, corpus_digest="d1", **_defaults())
    changed = prior.__class__(**{**prior.to_dict(), "corpus_digest": "d2"})
    assert isinstance(exc := ResumeAborted("x"), ValueError)  # noqa: F841
    with pytest.raises(ValueError):
        validate_resume(prior, changed)


def test_resume_passes_on_identical_identity() -> None:
    a = RunIdentity(base_model="Qwen/Qwen3.5-9B", lora_rank=64, corpus_digest="d1", **_defaults())
    b = RunIdentity(base_model="Qwen/Qwen3.5-9B", lora_rank=64, corpus_digest="d1", **_defaults())
    validate_resume(a, b)  # no raise


def test_identity_round_trips_through_dict() -> None:
    a = RunIdentity(base_model="Qwen/Qwen3.5-9B", lora_rank=64, corpus_digest="d1", **_defaults())
    assert RunIdentity.from_dict(a.to_dict()) == a


def test_identity_is_frozen() -> None:
    a = RunIdentity(base_model="Qwen/Qwen3.5-9B", lora_rank=64, corpus_digest="d1", **_defaults())
    with pytest.raises(Exception):
        a.base_model = "other"  # type: ignore[misc]


def test_stage_digests_emitted() -> None:
    outputs = {
        "stage1": {"records": [{"session_id": "s1"}, {"session_id": "s2"}]},
        "stage2": {"records": [{"session_id": "s1"}]},
    }
    d = stage_digests(outputs)
    for stage in ("stage1", "stage2"):
        assert d[stage]["split_digest"]
        assert d[stage]["lineage_digest"]  # M16: per-stage digests
    # Content-addressed: same records, same digest; different set, different digest.
    assert d["stage1"]["split_digest"] != d["stage2"]["split_digest"]


def test_stage_digests_deterministic() -> None:
    outputs = {"stage1": {"records": [{"session_id": "b"}, {"session_id": "a"}]}}
    again = {"stage1": {"records": [{"session_id": "a"}, {"session_id": "b"}]}}
    assert stage_digests(outputs)["stage1"] == stage_digests(again)["stage1"]


def test_stage_digests_covers_record_lineage_fields() -> None:
    """Lineage fields carried through from corpus records must affect the digest (M16)."""
    base = {
        "session_id": "s1",
        "evidence_tier": "gold",
        "base_sha": "aaa",
        "head_sha": "bbb",
        "daydream_version": "1.0",
        "reward_version": "2026.05.28-2",
        "split": "train",
    }
    mutated = {**base, "evidence_tier": "weak"}
    d1 = stage_digests({"stage1": {"records": [base]}})["stage1"]["lineage_digest"]
    d2 = stage_digests({"stage1": {"records": [mutated]}})["stage1"]["lineage_digest"]
    assert d1 != d2
