"""Unit tests for the gold-admission helper shared with the projection packages.

The legacy v1 admission/query helpers (``_is_admitted``, ``_build_record``,
``_annotation_reward``, ``_build_query``, ``CorpusFilters``) were deleted with
the legacy records-builder pipeline (#1093); only ``_is_admitted_outcome_gold``
survives in ``daydream.training.corpus`` and these tests pin its contract.
"""

from __future__ import annotations

from typing import Any

from daydream.training.corpus import _is_admitted_outcome_gold
from daydream.training.labeler_versions import LABELER_POLICY_VERSION

LEGACY: dict[str, Any] = {"label": "accepted", "has_posterior": True, "labeler_policy_version": None,
                          "decisive_mix": False, "decisive_only": True}
MIXED: dict[str, Any] = {**LEGACY, "labeler_policy_version": LABELER_POLICY_VERSION, "decisive_mix": True}
AMBIG: dict[str, Any] = {
    **LEGACY, "labeler_policy_version": LABELER_POLICY_VERSION, "decisive_mix": False, "decisive_only": False}
GOOD: dict[str, Any] = {**LEGACY, "labeler_policy_version": LABELER_POLICY_VERSION}


def test_gold_admission_rejects_legacy_observations() -> None:
    """Legacy reply-count/merge-presence rows are excluded from outcome-bearing gold (M16/M22)."""
    assert _is_admitted_outcome_gold(**LEGACY) is False


def test_gold_admission_rejects_mixed_and_ambiguous() -> None:
    assert _is_admitted_outcome_gold(**MIXED) is False
    assert _is_admitted_outcome_gold(**AMBIG) is False


def test_gold_admission_accepts_current_clean_evidence() -> None:
    assert _is_admitted_outcome_gold(**GOOD) is True
