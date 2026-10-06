"""Tests for the shared per-finding annotation snapshot serializer (issue #1055, Task 1)."""

import json

import pytest

from daydream.training import labeler_versions
from daydream.training.adjudication.snapshot import (
    ANNOTATION_SNAPSHOT_SCHEMA_VERSION,
    build_canonical_record,
    record_evidence_digest,
)
from daydream.training.harvest_types import HarvestEvidence
from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    PerFindingResolution,
    PRMergeSignal,
)
from daydream.training.labeler_versions import reply_evidence_digest
from daydream.training.record_identity import record_finding_id
from daydream.training.reward import ScoringInputs
from daydream.training.rubric import Rubric


def _resolution(fp: str = "fp-1", digest: str = "d" * 32) -> PerFindingResolution:
    return PerFindingResolution(
        fingerprint=fp,
        comment_id=7,
        disposition="unanswered",
        evidence=[{"reply_id": 1, "body_sha256": "abc"}],
        evidence_digest=digest,
    )


def _mutable_and_frozen_resolution() -> tuple[
    PerFindingResolution,
    PerFindingResolution,
    HarvestEvidence,
]:
    source_evidence = [{"reply_id": 1, "body_sha256": "abc", "context": {"labels": ["accepted", "reviewed"]}}]
    resolution = PerFindingResolution(
        fingerprint="fp-1",
        comment_id=7,
        disposition="accepted",
        evidence=source_evidence,
        evidence_digest=reply_evidence_digest(source_evidence),
    )
    rubric = Rubric(
        pr_merge=PRMergeSignal(merged=True, merged_at="2026-01-01T00:00:00Z"),
        fix_applied=FixAppliedSignal(verdict="applied", hunks_applied=1, hunks_total=1, window_commits=["abc123"]),
        comment_resolution=CommentResolutionSignal(total=1, replied=1, unresolved=0),
        local_commit_applied=None,
        posterior_source="pr_review",
        per_finding_resolutions=[resolution],
    )
    harvest_evidence = HarvestEvidence(
        scoring_inputs=ScoringInputs(verifier_verdicts=None, format_valid=True, length=None),
        rubric=rubric,
    )
    frozen_resolutions = harvest_evidence.rubric.per_finding_resolutions
    assert frozen_resolutions is not None
    return resolution, frozen_resolutions[0], harvest_evidence


def _snapshot_session() -> dict[str, object]:
    return {
        "session_id": "s1",
        "trajectory_id": "s1-t",
        "segment_id": "s1-seg",
        "resolutions": [{"fingerprint": "fp-1", "item_uid": "item:1", "profile_name": "pr_review", "stack": "python"}],
    }


def test_build_canonical_record_pins_identity_and_provenance() -> None:
    session = _snapshot_session()
    record = build_canonical_record(session, _resolution(), evidence_observed_at="2026-01-01T00:00:00+00:00")
    # identity via record_identity.record_finding_id — recompute and compare, never trust a stored copy

    assert record["record_id"] == record_finding_id("s1", "s1-t", "s1-seg", "item:1")
    assert record["evidence_digest"] == "d" * 32
    assert record["disposition"] == "unanswered"
    assert record["profile"]["profile_name"] == "pr_review"
    assert record["stack"] == "python"

    assert record["classifier_version"] == labeler_versions.REPLY_CLASSIFIER_VERSION
    assert ANNOTATION_SNAPSHOT_SCHEMA_VERSION in record["schema_version"]


def test_build_canonical_record_serializes_frozen_harvest_evidence_canonically() -> None:
    mutable, frozen, _harvest_evidence = _mutable_and_frozen_resolution()
    session = _snapshot_session()
    mutable_record = build_canonical_record(session, mutable, evidence_observed_at="2026-01-01T00:00:00Z")
    frozen_record = build_canonical_record(session, frozen, evidence_observed_at="2026-01-01T00:00:00Z")

    assert json.dumps(frozen_record, sort_keys=True) == json.dumps(mutable_record, sort_keys=True)


def test_record_evidence_digest_matches_frozen_harvest_digest_with_nested_json() -> None:
    mutable, frozen, _harvest_evidence = _mutable_and_frozen_resolution()

    assert record_evidence_digest([frozen.evidence]) == record_evidence_digest([mutable.evidence])


def test_record_evidence_digest_flattens_and_orders_per_finding_evidence() -> None:
    ev_a = [{"reply_id": 1, "body_sha256": "aaa"}]
    ev_b = [{"reply_id": 2, "body_sha256": "bbb"}]

    shared = record_evidence_digest([ev_a, ev_b])
    assert shared == reply_evidence_digest(ev_a + ev_b)
    assert shared == record_evidence_digest([ev_b, ev_a])

    # Represent absent evidence as None rather than an empty digest to avoid identity collisions.
    assert record_evidence_digest([]) is None


def test_build_canonical_record_rejects_missing_digest() -> None:
    session = {"session_id": "s1", "trajectory_id": "t", "segment_id": "g", "resolutions": []}
    with pytest.raises(ValueError, match="evidence_digest"):
        build_canonical_record(session, _resolution(digest=""), evidence_observed_at="2026-01-01")


@pytest.mark.parametrize("rows", [[], [{"fingerprint": "fp-1"}] * 2])
def test_build_canonical_record_rejects_wrong_resolution_row_count(rows: list[dict[str, object]]) -> None:
    # Keep the digest valid to isolate the row-count guard.
    session = {"session_id": "s1", "trajectory_id": "t", "segment_id": "g", "resolutions": rows}
    with pytest.raises(ValueError, match="expected exactly 1"):
        build_canonical_record(session, _resolution(), evidence_observed_at="2026-01-01")


def test_build_canonical_record_as_of_passthrough() -> None:
    session = _snapshot_session()
    record = build_canonical_record(
        session,
        _resolution(),
        evidence_observed_at="2026-01-01T00:00:00+00:00",
        as_of="2026-02-01T00:00:00+00:00",
    )
    assert record["as_of"] == "2026-02-01T00:00:00+00:00"
    assert "as_of" not in build_canonical_record(
        session, _resolution(), evidence_observed_at="2026-01-01T00:00:00+00:00"
    )
