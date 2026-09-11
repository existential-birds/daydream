"""Unit tests for the corpus projection's shared admission/query helpers.

These tests moved out of the deleted legacy ``tests/test_training_corpus.py``
(#1093): they test canonical helpers that survive the legacy records-builder
deletion — ``_is_admitted``, ``_is_admitted_outcome_gold``, ``_build_record``,
``_annotation_reward``, and ``_build_query`` — in isolation, with no archive
emission path involved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.training.corpus import (
    CorpusFilters,
    _annotation_reward,
    _build_query,
    _build_record,
    _is_admitted,
    _is_admitted_outcome_gold,
)
from daydream.training.labeler_versions import LABELER_POLICY_VERSION
from daydream.training.reward import PosteriorBreakdown, RewardBreakdown


def _ann_with_posterior_reward_json() -> dict[str, Any]:
    """Annotation row for a labeled (PR-outcome) run.

    ``reward_json`` is a real ``PosteriorBreakdown.to_dict()`` — it carries
    ``posterior_cost`` (the population discriminator), while ``composite`` /
    ``composite_reward`` remain the pure intrinsic score (C5: the posterior is
    a sibling, never folded into the composite).
    """
    breakdown = PosteriorBreakdown(
        correctness_per_finding=[1.0],
        grounding=0.9,
        format_valid=True,
        length_penalty=0.1,
        composite=0.6,
        axes_present={"correctness": True, "grounding": True, "length": True},
        reward_version="2026.05.28-1",
        false_positive_penalty=1.0,
        posterior_cost=0.5,
        outcome_prior=0.5,
        outcome_prior_n=12,
    )
    reward_dict = breakdown.to_dict()
    return {
        "session_id": "s-labeled",
        "labels": json.dumps(["rejected"]),
        "reward_json": json.dumps(reward_dict),
        "composite_reward": reward_dict["composite"],
        "valid_at": "2026-04-01T00:00:00+00:00",
    }


def _ann_with_intrinsic_reward_json() -> dict[str, Any]:
    """Annotation row for an unlabeled (no PR-outcome) run.

    ``reward_json`` is a real ``RewardBreakdown.to_dict()`` — it has no
    ``posterior_cost`` key, so the absence of that key on the emitted record is
    what marks the row as intrinsic-only.
    """
    breakdown = RewardBreakdown(
        correctness_per_finding=[1.0],
        grounding=0.9,
        format_valid=True,
        length_penalty=0.1,
        composite=0.6,
        axes_present={"correctness": True, "grounding": True, "length": True},
        reward_version="2026.05.28-1",
    )
    reward_dict = breakdown.to_dict()
    return {
        "session_id": "s-intrinsic",
        "labels": json.dumps([]),
        "reward_json": json.dumps(reward_dict),
        "composite_reward": reward_dict["composite"],
        "valid_at": "2026-04-01T00:00:00+00:00",
    }


def test_is_admitted_min_reward_compares_intrinsic_only() -> None:
    """``min_reward`` compares against the stored intrinsic composite (C5).

    The labeled row carries a ``posterior_cost`` of 0.5 in its breakdown, yet
    its stored ``composite_reward`` is the pure intrinsic 0.6 (the posterior is
    never folded in). So ``min_reward=0.6`` admits it on the intrinsic threshold
    even though its label (``rejected``) is not in ``labels``. If the stored
    scalar were intrinsic-minus-posterior (0.1), this would NOT admit — that is
    the mixing bug C5 prevents and this test pins.
    """
    assert (
        _is_admitted(
            label="rejected",
            composite_reward=0.6,
            filters=CorpusFilters(min_reward=0.6, include_all_labels=False, labels=()),
        )
        is True
    )


def test_build_record_emits_posterior_discriminator_only_for_labeled(tmp_path: Path) -> None:
    """``posterior_cost`` in ``record["reward"]`` is the population discriminator.

    ``reward_json`` is parsed via ``_annotation_reward`` and written verbatim
    (no transform), so a labeled annotation built from
    ``PosteriorBreakdown.to_dict()`` carries ``posterior_cost`` while an
    unlabeled one built from ``RewardBreakdown.to_dict()`` does not.
    """
    manifest_row = {"session_id": "s", "archive_path": str(tmp_path)}

    labeled_reward, labeled_composite = _annotation_reward(_ann_with_posterior_reward_json(), "s")
    rec_labeled = _build_record(
        manifest_row,
        trajectory={},
        stack=None,
        manifest=None,
        reward=labeled_reward,
        composite_reward=labeled_composite,
    )
    assert rec_labeled is not None
    intrinsic_reward, intrinsic_composite = _annotation_reward(_ann_with_intrinsic_reward_json(), "s")
    rec_intrinsic = _build_record(
        manifest_row,
        trajectory={},
        stack=None,
        manifest=None,
        reward=intrinsic_reward,
        composite_reward=intrinsic_composite,
    )
    assert rec_intrinsic is not None

    assert "posterior_cost" in rec_labeled["reward"]
    assert "posterior_cost" not in rec_intrinsic.get("reward", {})
    # The discriminator does not leak into the intrinsic composite scalar.
    assert rec_labeled["composite_reward"] == 0.6
    assert rec_intrinsic["composite_reward"] == 0.6


def test_query_builds_pipeline_status_gate_from_filters() -> None:
    """The pipeline_status knob flows into the WHERE clause as
    ``pipeline_status = ?``, mirroring the existing ``status = ?`` construction —
    the ''authoritative'' pipeline-outcome gate.
    """
    filters = CorpusFilters(status="complete")
    where, params = _build_query(filters=filters)
    # The status gate itself is unchanged...
    assert "status = ?" in where and "complete" in params
    # ...and with the pipeline knob applied the query gains the pipeline gate.
    from dataclasses import replace

    effective = replace(filters, pipeline_status="succeeded")
    where2, params2 = _build_query(filters=effective)
    assert "pipeline_status = ?" in where2 and "succeeded" in params2


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


def test_is_admitted_label_path_respects_filters_labels() -> None:
    """The label path requires both filters.labels membership and the gold guard.

    A ``labels=()`` filter admits nothing on the label path even when the row
    is current-policy gold — and a clean gold row still admits under the
    default ``("accepted",)`` filter, so the guard never narrows the default.
    """
    assert _is_admitted("accepted", None, CorpusFilters(labels=()), **
                        {k: v for k, v in GOOD.items() if k != "label"}) is False
    assert _is_admitted(**GOOD, composite_reward=None, filters=CorpusFilters()) is True


def test_min_reward_path_unaffected() -> None:
    """The intrinsic min_reward path keeps its existing contract (no creep)."""
    assert _is_admitted("rejected", 1.0, CorpusFilters(min_reward=0.5)) is True
