"""Completeness + invariant-immutability guards for the review profile (M14, M15)."""

import pytest

from daydream import review_profile as rp


def test_new_model_bearing_stage_without_strategy_and_classification_fails():
    # All production STAGE_KEYS carry a strategy key + explicit envelope
    # classification (M14 completeness guard is currently satisfied).
    registered = set(rp.STAGE_KEYS)
    missing_strategy = registered - set(rp.build_default_profile().strategies)
    missing_envelope = registered - set(rp.ENVELOPE_CLASSIFICATION)
    assert not missing_strategy, f"model-bearing stages missing a strategy key: {missing_strategy}"
    assert not missing_envelope, f"stages missing envelope classification: {missing_envelope}"

    # The SAME guard computation detects a brand-new stage that was added
    # without a strategy key + envelope classification.
    new_stage = {"discovery.brand_new_stage"}
    assert new_stage - set(rp.build_default_profile().strategies) == new_stage
    assert new_stage - set(rp.ENVELOPE_CLASSIFICATION) == new_stage


def test_profile_cannot_alter_invariant_host_owned_keys():
    for field in ("findings_schema", "verifier", "judge", "scoring", "gold", "skill_name", "model"):
        with pytest.raises(rp.ProfileError):
            rp.parse_profile(f'schema_version = 1\nname = "p"\n{field} = "x"', source="s")
