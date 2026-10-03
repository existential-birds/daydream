"""Tests for the shared per-finding annotation snapshot serializer (issue #1055, Task 1)."""

import json

import pytest

from daydream.training import labeler_versions
from daydream.training.adjudication.snapshot import (
    ANNOTATION_SNAPSHOT_SCHEMA_VERSION,
    FindingRecord,
    record_evidence_digest,
    snapshot_id,
)
from daydream.training.corpus_projection.identity import record_id
from daydream.training.harvest_types import HarvestEvidence
from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    PerFindingResolution,
    PRMergeSignal,
)
from daydream.training.labeler_versions import reply_evidence_digest
from daydream.training.reward import ScoringInputs
from daydream.training.rubric import Rubric


def _resolution(fp: str = "fp-1", digest: str = "d" * 32) -> PerFindingResolution:
    return PerFindingResolution(
        fingerprint=fp, comment_id=7, disposition="unanswered", evidence=[{"reply_id": 1, "body_sha256": "abc"}],
        evidence_digest=digest,
    )


def _mutable_and_frozen_resolution() -> tuple[PerFindingResolution, PerFindingResolution, HarvestEvidence,]:
    source_evidence = [{"reply_id": 1, "body_sha256": "abc", "context": {"labels": ["accepted", "reviewed"]}}]
    resolution = PerFindingResolution(
        fingerprint="fp-1", comment_id=7, disposition="accepted", evidence=source_evidence,
        evidence_digest=reply_evidence_digest(source_evidence),
    )
    rubric = Rubric(pr_merge=PRMergeSignal(merged=True, merged_at="2026-01-01T00:00:00Z"),
        fix_applied=FixAppliedSignal(verdict="applied", hunks_applied=1, hunks_total=1, window_commits=["abc123"]),
        comment_resolution=CommentResolutionSignal(total=1, replied=1, unresolved=0), local_commit_applied=None,
        posterior_source="pr_review", per_finding_resolutions=[resolution],
    )
    harvest_evidence = HarvestEvidence(
        scoring_inputs=ScoringInputs(verifier_verdicts=None,  format_valid=True, length=None), rubric=rubric,
    )
    frozen_resolutions = harvest_evidence.rubric.per_finding_resolutions
    assert frozen_resolutions is not None
    return resolution, frozen_resolutions[0], harvest_evidence


def _snapshot_session() -> dict[str, object]:
    return {"session_id": "s1", "trajectory_id": "s1-t", "segment_id": "s1-seg",
        "resolutions": [{"fingerprint": "fp-1", "profile_name": "pr_review", "stack": "python"}],
    }


def test_build_canonical_record_pins_identity_and_provenance() -> None:
    session = _snapshot_session()
    record = FindingRecord.snapshot(
        session, _resolution(), evidence_observed_at="2026-01-01T00:00:00+00:00",
    ).canonical()
    # identity via corpus_projection.identity.record_id — recompute and compare, never trust a stored copy

    assert record["record_id"] == record_id("s1", "s1-t", "s1-seg", "fp-1")
    assert record["evidence_digest"] == "d" * 32
    assert record["disposition"] == "unanswered"
    assert record["profile"]["profile_name"] == "pr_review"
    assert record["stack"] == "python"

    assert record["classifier_version"] == labeler_versions.REPLY_CLASSIFIER_VERSION
    assert ANNOTATION_SNAPSHOT_SCHEMA_VERSION in record["schema_version"]

def test_build_canonical_record_serializes_frozen_harvest_evidence_canonically() -> None:
    mutable, frozen, _harvest_evidence = _mutable_and_frozen_resolution()
    session = _snapshot_session()
    mutable_record = FindingRecord.snapshot(session, mutable, evidence_observed_at="2026-01-01T00:00:00Z").canonical()
    frozen_record = FindingRecord.snapshot(session, frozen, evidence_observed_at="2026-01-01T00:00:00Z").canonical()

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

def test_snapshot_id_is_content_addressed_and_order_stable() -> None:
    base = {"curation_id": "cur-1", "sanitized_hub_commit": "a" * 40, "source_hub_commit": "b" * 40,
        "archive_index_digest": "c" * 64, "evidence_observed_at": "2026-01-01T00:00:00+00:00",
        "as_of": "2026-02-01T00:00:00+00:00", "labeler_version": "v1", "rubric_version": "v1",
        "classifier_version": "v1",
    }
    sid_a = snapshot_id(base)
    sid_b = snapshot_id(dict(reversed(list(base.items()))))
    assert sid_a == sid_b
    assert len(sid_a) == 64
    changed = dict(base, as_of="2026-03-01T00:00:00+00:00")
    assert snapshot_id(changed) != sid_a  # any pin change => new id (AC 8)

def test_missing_pin_component_fails_closed() -> None:
    with pytest.raises(ValueError, match="curation_id"):
        snapshot_id({"sanitized_hub_commit": "a" * 40})

def test_build_canonical_record_rejects_missing_digest() -> None:
    session = {"session_id": "s1", "trajectory_id": "t", "segment_id": "g", "resolutions": []}
    with pytest.raises(ValueError, match="evidence_digest"):
        FindingRecord.snapshot(session, _resolution(digest=""), evidence_observed_at="2026-01-01").canonical()

@pytest.mark.parametrize("rows", [[], [{"fingerprint": "fp-1"}] * 2])
def test_build_canonical_record_rejects_wrong_resolution_row_count(rows: list[dict[str, object]]) -> None:
    # Keep the digest valid to isolate the row-count guard.
    session = {"session_id": "s1", "trajectory_id": "t", "segment_id": "g", "resolutions": rows}
    with pytest.raises(ValueError, match="expected exactly 1"):
        FindingRecord.snapshot(session, _resolution(), evidence_observed_at="2026-01-01").canonical()

def test_build_canonical_record_as_of_passthrough() -> None:
    session = _snapshot_session()
    record = FindingRecord.snapshot(
        session, _resolution(), evidence_observed_at="2026-01-01T00:00:00+00:00", as_of="2026-02-01T00:00:00+00:00",
    ).canonical()
    assert record["as_of"] == "2026-02-01T00:00:00+00:00"
    assert "as_of" not in FindingRecord.snapshot(
        session, _resolution(), evidence_observed_at="2026-01-01T00:00:00+00:00"
    ).canonical()

def test_snapshot_id_allows_empty_as_of_unpinned_edge() -> None:
    pin = {"curation_id": "cur-1", "sanitized_hub_commit": "a" * 40, "source_hub_commit": "b" * 40,
        "archive_index_digest": "c" * 64, "evidence_observed_at": "2026-01-01T00:00:00+00:00", "as_of": "",
        "labeler_version": "v1", "rubric_version": "v1", "classifier_version": "v1",
    }
    unpinned = snapshot_id(pin)
    assert len(unpinned) == 64
    del pin["as_of"]
    assert snapshot_id(pin) == unpinned
    assert snapshot_id(dict(pin, as_of="2026-02-01T00:00:00+00:00")) != unpinned
    with pytest.raises(ValueError, match="evidence_observed_at"):
        snapshot_id(dict(pin, evidence_observed_at=""))


def test_finding_snapshot_owns_source_and_separates_conflicted_wire_views() -> None:
    source = {"fingerprint": "fp-1", "profile_name": "pr_review", "extras": {"labels": ["original"]}}
    session = {"session_id": "s1", "trajectory_id": "s1-t", "segment_id": "s1-seg", "resolutions": [source]}
    finding = FindingRecord.snapshot(
        session, _resolution(), evidence_observed_at="2026-01-01T00:00:00Z", conflicting=True,
    )
    source["profile_name"] = "changed"
    source["extras"] = {"labels": ["changed"]}

    archive = finding.canonical()
    assert archive["profile"]["profile_name"] == "pr_review"
    assert archive["disposition"] == "unanswered"
    assert archive["resolutions"] == [{
        "fingerprint": "fp-1", "profile_name": "pr_review", "extras": {"labels": ["original"]},
    }]
    annotation = finding.canonical(project_conflict=True)
    assert annotation["disposition"] == annotation["resolutions"][0]["disposition"] == "ambiguous"
    annotation["resolutions"][0]["extras"]["labels"].append("mutated output")
    assert finding.canonical() == archive
