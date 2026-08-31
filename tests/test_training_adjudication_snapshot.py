"""Tests for the shared per-finding annotation snapshot serializer (issue #1055, Task 1)."""

import pytest

from daydream.training.adjudication.snapshot import (
    ANNOTATION_SNAPSHOT_SCHEMA_VERSION,
    build_canonical_record,
    snapshot_id,
)
from daydream.training.harvest import _reply_evidence_digest  # session-level twin
from daydream.training.labeler_signals import PerFindingResolution
from daydream.training.labeler_versions import reply_evidence_digest


def _resolution(fp: str = "fp-1", digest: str = "d" * 32) -> PerFindingResolution:
    return PerFindingResolution(
        fingerprint=fp,
        comment_id=7,
        disposition="unanswered",
        evidence=[{"reply_id": 1, "body_sha256": "abc"}],
        evidence_digest=digest,
    )


def test_build_canonical_record_pins_identity_and_provenance() -> None:
    session = {
        "session_id": "s1",
        "trajectory_id": "s1-t",
        "segment_id": "s1-seg",
        "resolutions": [{"fingerprint": "fp-1", "profile_name": "pr_review", "stack": "python"}],
    }
    record = build_canonical_record(
        session, _resolution(), evidence_observed_at="2026-01-01T00:00:00+00:00"
    )
    # identity via corpus_v2.identity.record_id — recompute and compare, never trust a stored copy
    from daydream.training.corpus_v2.identity import record_id

    assert record["record_id"] == record_id("s1", "s1-t", "s1-seg", "fp-1")
    assert record["evidence_digest"] == "d" * 32
    assert record["disposition"] == "unanswered"
    assert record["profile"]["profile_name"] == "pr_review"
    assert record["stack"] == "python"
    from daydream.training import labeler_versions

    assert record["classifier_version"] == labeler_versions.REPLY_CLASSIFIER_VERSION
    assert ANNOTATION_SNAPSHOT_SCHEMA_VERSION in record["schema_version"]


def test_record_evidence_digest_matches_harvest_row_digest() -> None:
    # K5 spike verdict: shared digest == training/harvest.py session-level digest
    ev_a = [{"reply_id": 1, "body_sha256": "aaa"}]
    ev_b = [{"reply_id": 2, "body_sha256": "bbb"}]

    from daydream.training.adjudication.snapshot import record_evidence_digest

    shared = record_evidence_digest([ev_a, ev_b])
    # the harvest twin flattens PerFindingResolution evidence in recorded order

    class _R:
        disposition = "accepted"
        evidence = ev_a

    class _R2:
        disposition = "rejected"
        evidence = ev_b

    class _Rubric:  # duck-typed twin of harvest.Rubric
        per_finding_resolutions = [_R(), _R2()]

    assert shared == _reply_evidence_digest(_Rubric())  # type: ignore[arg-type]
    assert shared == reply_evidence_digest(ev_a + ev_b)
    # order of the per-finding list must not matter (digest normalizes by reply_id)
    assert shared == record_evidence_digest([ev_b, ev_a])


def test_snapshot_id_is_content_addressed_and_order_stable() -> None:
    base = {
        "curation_id": "cur-1",
        "sanitized_hub_commit": "a" * 40,
        "source_hub_commit": "b" * 40,
        "archive_index_digest": "c" * 64,
        "evidence_observed_at": "2026-01-01T00:00:00+00:00",
        "as_of": "2026-02-01T00:00:00+00:00",
        "labeler_version": "v1",
        "rubric_version": "v1",
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
        build_canonical_record(session, _resolution(digest=""), evidence_observed_at="2026-01-01")
