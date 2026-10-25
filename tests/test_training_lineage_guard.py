"""Tests for the LOCKED_FIELDS resume guard (M18)."""

from __future__ import annotations

from daydream.training.lineage import LOCKED_FIELDS

_SPEC_FIELDS = {
    "base_model", "tokenizer_renderer", "max_seq_len", "lora_rank", "lora_targets", "optimizer", "learning_rate",
    "corpus_digest", "split_digest", "profile_policy", "reward_version", "reward_weights", "stack_pins",
}


def test_every_spec_field_is_locked() -> None:
    assert _SPEC_FIELDS <= set(LOCKED_FIELDS)  # AC5 list is the floor, not the ceiling
