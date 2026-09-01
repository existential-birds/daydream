"""Tests for the local-observation importer's pure core.

Task 2 covers ``link_session_identity``: Hub session identity linkage with
session_id primary rule, repo_slug+SHA fallback, and report-don't-drop
unmatched / identity-conflict buckets (M2, KD1).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daydream.archive.importer import link_session_identity

SID = "sess-abc123"
D = "d" * 64


def build_hydrated_index_with_session(tmp_path: Path, session_id: str, digest: str) -> dict[str, dict[str, str]]:
    """Hydrated-index shape follows the hydrate import-ledger join:
    ``{session_id: {"derivative_digest": ..., "record_id": ...}}``."""
    return {session_id: {"derivative_digest": digest, "record_id": f"rec-{session_id}"}}


def _record(
    session_id: str,
    digest: str = D,
    repo_slug: str | None = "org/repo",
    base_sha: str | None = "a" * 40,
    head_sha: str | None = "b" * 40,
) -> dict[str, str | None]:
    return {
        "session_id": session_id,
        "derivative_digest": digest,
        "repo_slug": repo_slug,
        "base_sha": base_sha,
        "head_sha": head_sha,
    }


def test_link_session_by_session_id(tmp_path: Path) -> None:
    idx = build_hydrated_index_with_session(tmp_path, SID, digest=D)
    result = link_session_identity([_record(SID)], hydrated_index=idx, repo_slug_sha_lookup={})
    entry = result["linked"][SID]
    assert entry["hub_session_id"] == SID
    assert entry["matched_by"] == "session_id"


def test_link_fallback_repo_slug_sha(tmp_path: Path) -> None:
    # session_id absent from the hydrated index; repo_slug+SHA fallback matches.
    lookup = {("org/repo", "a" * 40, "b" * 40): "hub-999"}
    result = link_session_identity(
        [_record("s1")], hydrated_index={}, repo_slug_sha_lookup=lookup
    )
    entry = result["linked"]["s1"]
    assert entry["hub_session_id"] == "hub-999"
    assert entry["matched_by"] == "repo_slug_sha"


def test_unmatched_and_conflict_buckets(tmp_path: Path) -> None:
    idx = build_hydrated_index_with_session(tmp_path, SID, digest=D)
    records = [
        _record("no-such-session"),  # no Hub entry, no fallback hit
        _record(SID, digest="e" * 64),  # conflicting derivative digest
    ]
    r = link_session_identity(records, hydrated_index=idx, repo_slug_sha_lookup={})
    assert "no-such-session" in r["unmatched"]
    assert SID in r["identity_conflict"]


def test_conflicting_digest_never_links(tmp_path: Path) -> None:
    idx = build_hydrated_index_with_session(tmp_path, SID, digest=D)
    r = link_session_identity(
        [_record(SID, digest="e" * 64)], hydrated_index=idx, repo_slug_sha_lookup={}
    )
    assert r["linked"] == {}
    assert SID in r["identity_conflict"]


def test_missing_fallback_fields_raises(tmp_path: Path) -> None:
    # session_id absent from the Hub index and the record lacks the session
    # fields required for the fallback -> ValueError naming the session_id.
    record = _record("s-broken", repo_slug=None, base_sha=None, head_sha=None)
    record.pop("derivative_digest")
    with pytest.raises(ValueError, match="s-broken"):
        link_session_identity([record], hydrated_index={}, repo_slug_sha_lookup={})
