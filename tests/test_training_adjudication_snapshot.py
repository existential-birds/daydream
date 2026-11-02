"""Tests for the shared per-finding annotation snapshot serializer (issue #1055, Task 1)."""

import hashlib
from collections.abc import Mapping
from typing import Any

import pytest

from daydream.training._immutable_json import freeze_json
from daydream.training.adjudication.snapshot import build_canonical_record, record_evidence_digest
from daydream.training.labeler_versions import reply_evidence_digest


def _resolution(fp: str = "fp-1", digest: str = "d" * 32) -> dict[str, Any]:
    return {
        "fingerprint": fp,
        "comment_id": 7,
        "disposition": "unanswered",
        "evidence": [{"reply_id": 1, "body_sha256": "abc"}],
        "evidence_digest": digest,
    }


def _mutable_and_frozen_resolution() -> tuple[dict[str, Any], Mapping[str, Any]]:
    source_evidence = [{"reply_id": 1, "body_sha256": "abc", "context": {"labels": ["accepted", "reviewed"]}}]
    resolution = {**_resolution(), "disposition": "accepted", "evidence": source_evidence,
                  "evidence_digest": reply_evidence_digest(source_evidence)}
    return resolution, freeze_json(resolution)


def _snapshot_session() -> dict[str, object]:
    return {
        "session_id": "s1",
        "trajectory_id": "s1-t",
        "segment_id": "s1-seg",
        "resolutions": [{"fingerprint": "fp-1", "item_uid": "item:1", "profile_name": "pr_review", "stack": "python"}],
    }


def test_record_evidence_digest_matches_frozen_shared_digest_with_nested_json() -> None:
    mutable, frozen = _mutable_and_frozen_resolution()

    canonical = (b'[{"body": "", "body_sha256": "abc", "context": {"labels": ["accepted", "reviewed"]}, '
                 b'"reply_id": 1}]')
    expected = hashlib.sha256(canonical).hexdigest()
    assert record_evidence_digest([frozen["evidence"]]) == expected
    assert record_evidence_digest([mutable["evidence"]]) == expected


def test_build_canonical_record_rejects_missing_digest() -> None:
    session = {"session_id": "s1", "trajectory_id": "t", "segment_id": "g", "resolutions": []}
    with pytest.raises(ValueError, match="evidence_digest"):
        build_canonical_record(session, _resolution(digest=""), evidence_observed_at="2026-01-01")


def test_build_canonical_record_rejects_invalid_required_resolution_fields() -> None:
    for field, value in (("fingerprint", ""), ("disposition", "other")):
        resolution = _resolution()
        resolution[field] = value
        with pytest.raises(ValueError, match=field):
            build_canonical_record(_snapshot_session(), resolution, evidence_observed_at="2026-01-01")
    # Keep the digest valid to isolate the row-count guard.
    rows = [{"fingerprint": "fp-1"}] * 2
    session = {"session_id": "s1", "trajectory_id": "t", "segment_id": "g", "resolutions": rows}
    with pytest.raises(ValueError, match="expected exactly 1"):
        build_canonical_record(session, _resolution(), evidence_observed_at="2026-01-01")
