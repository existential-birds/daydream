"""Tests for the local-observation importer's pure core.

Task 2 covers ``link_session_identity``: Hub session identity linkage with
session_id primary rule, repo_slug+SHA fallback, and report-don't-drop
unmatched / identity-conflict buckets (M2, KD1).

Task 3 covers ``dedupe_observations``: idempotent, byte-identical dedupe
across overlapping backups keyed on the writer's versioned auto-dedup tuple
plus the canonical-JSON payload digest, with content-conflict accounting
(M4, KD2/Assumption 3).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from daydream.archive.importer import dedupe_observations, link_session_identity
from daydream.archive.index import append_label_observation, upsert_run
from tests.harness.trajectory import make_manifest

SID = "sess-abc123"
D = "d" * 64

_OBSERVED_A = "2026-05-01T00:00:00+00:00"
_OBSERVED_B = "2026-05-02T00:00:00+00:00"

_OBSERVATION_COLUMNS = [
    "session_id",
    "observed_at",
    "labels",
    "pr_state",
    "labeler_version",
    "evidence_sha",
    "rubric_json",
    "valid_at",
    "reward_version",
    "reward_json",
    "composite_reward",
    "reviewer_logins",
    "has_posterior",
    "source",
    "labeler_policy_version",
    "reply_classifier_version",
    "reply_evidence_digest",
    "legacy",
]


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


# ---------------------------------------------------------------------------
# Task 3: dedupe across overlapping backups
# ---------------------------------------------------------------------------


def _seed_run(root: Path) -> None:
    upsert_run(
        root,
        make_manifest(
            session_id=SID,
            repo_slug="org/repo",
            head_sha="b" * 40,
            base_sha="a" * 40,
        ),
    )


def _append_observation(root: Path, *, observed_at: str, evidence_sha: str) -> None:
    append_label_observation(
        root,
        SID,
        labels=["accepted"],
        pr_state=None,
        labeler_version="980-rubric-r2",
        evidence_sha=evidence_sha,
        # Explicit valid_at: the writer collapses valid_at=None to observed_at,
        # which would make the two roots' payloads differ by capture stamp.
        valid_at="2026-04-30T00:00:00+00:00",
        reply_evidence_digest=None,
        reward_version=None,
        has_posterior=False,
        source="auto",
        observed_at=observed_at,
    )


def _force_insert_observation(root: Path, *, observed_at: str, evidence_sha: str) -> None:
    """Insert the exact same evidence payload under a different ``observed_at``.

    Mirrors the real overlap scenario — two independently captured backups of
    the same archive carry identical rows stamped at their own capture times.
    """
    write = sqlite3.connect(root / "index.db")
    write.execute(
        "INSERT INTO label_observations (session_id, observed_at, labels, pr_state, labeler_version,"
        " evidence_sha, rubric_json, valid_at, reward_version, reward_json, composite_reward,"
        " reviewer_logins, has_posterior, source, labeler_policy_version,"
        " reply_classifier_version, reply_evidence_digest, legacy) VALUES ("
        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            SID,
            observed_at,
            '["accepted"]',
            None,
            "980-rubric-r2",
            evidence_sha,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            "auto",
            "980-policy-r1",
            None,
            None,
            "auto",
        ),
    )
    write.commit()
    write.close()


def read_label_rows(root: Path) -> list[dict[str, Any]]:
    """Read-only inventory of one root's ``label_observations`` rows.

    Local test stand-in for the Task 1 inventory: a ``mode=ro`` SQLite read of
    every column, returned as plain dicts.
    """
    conn = sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        cols = ", ".join(_OBSERVATION_COLUMNS)
        rows = conn.execute(f"SELECT {cols} FROM label_observations ORDER BY observed_at").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def mk_backup_pair(tmp_path: Path, evidence_sha: str) -> tuple[Path, Path]:
    """Two backup roots holding one shared evidence payload under different
    ``observed_at`` stamps (root A at ``_OBSERVED_A``, root B at
    ``_OBSERVED_B``) — the overlapping-backup shape from the plan."""
    root_a = tmp_path / "backup-a"
    root_b = tmp_path / "backup-b"
    for root in (root_a, root_b):
        root.mkdir()
        _seed_run(root)
    _append_observation(root_a, observed_at=_OBSERVED_A, evidence_sha=evidence_sha)
    _append_observation(root_b, observed_at=_OBSERVED_B, evidence_sha=evidence_sha)
    # Each root is an independent capture: the writer's within-root auto-dedup
    # only compares the latest row, so both first-time appends insert — the
    # shared evidence payload now exists in both roots at different stamps.
    return root_a, root_b


def canonical_digest(rows: list[dict[str, Any]]) -> str:
    """Content digest over the merged rows — canonical JSON, sorted keys."""
    payload = json.dumps(rows, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def test_overlapping_backups_byte_identical(tmp_path: Path) -> None:
    src_a, src_b = mk_backup_pair(tmp_path, evidence_sha="c" * 64)
    inv_a = read_label_rows(src_a)
    inv_b = read_label_rows(src_b)
    merged = dedupe_observations([inv_a, inv_b])

    canon = canonical_digest(merged["rows"])
    merged_once = dedupe_observations([inv_a])
    assert canonical_digest(dedupe_observations([merged_once["rows"], inv_b])["rows"]) == canon
    # and re-running is a no-op:
    assert canonical_digest(dedupe_observations([merged["rows"], inv_b])["rows"]) == canon
    # inventory order must not affect the merged row set:
    assert canonical_digest(dedupe_observations([inv_b, inv_a])["rows"]) == canon


def test_no_duplicate_same_evidence_diff_observed_at(tmp_path: Path) -> None:
    src_a, src_b = mk_backup_pair(tmp_path, evidence_sha="c" * 64)
    merged = dedupe_observations([read_label_rows(src_a), read_label_rows(src_b)])
    keys = {(r["session_id"], r["evidence_sha"], r["labels"]) for r in merged["rows"]}
    assert len(keys) == len(merged["rows"])  # no dup evidence rows survived
    # exactly one row was deduped (the second capture of the same evidence)
    assert merged["deduped_count"] == 1
    # one surviving row, stamped with the earliest observed_at of the pair
    assert len(merged["rows"]) == 1
    assert merged["rows"][0]["observed_at"] == _OBSERVED_A


def test_every_generation_kept_never_keep_latest(tmp_path: Path) -> None:
    # Two *distinct* evidence generations (different policy versions) must both
    # survive — dedupe collapses identical evidence only, never keeps-latest.
    src_a, src_b = mk_backup_pair(tmp_path, evidence_sha="c" * 64)
    _append_observation(src_a, observed_at="2026-05-03T00:00:00+00:00", evidence_sha="e" * 64)
    inv_a = read_label_rows(src_a)
    inv_b = read_label_rows(src_b)
    merged = dedupe_observations([inv_a, inv_b])
    assert len(merged["rows"]) == 2
    assert {r["evidence_sha"] for r in merged["rows"]} == {"c" * 64, "e" * 64}


def test_human_rows_across_backups_kept(tmp_path: Path) -> None:
    # Human-sourced rows are never auto-deduped by the writer, so identical
    # human evidence captured at two different observed_at stamps is preserved
    # bitemporally; only byte-identical human rows collapse.
    src_a, src_b = mk_backup_pair(tmp_path, evidence_sha="c" * 64)
    for root, at in ((src_a, _OBSERVED_A), (src_b, _OBSERVED_B)):
        append_label_observation(
            root,
            SID,
            labels=["rejected"],
            pr_state=None,
            labeler_version="1055-human-r1",
            evidence_sha="f" * 64,
            source="human",
            observed_at=at,
        )
    merged = dedupe_observations([read_label_rows(src_a), read_label_rows(src_b)])
    human = [r for r in merged["rows"] if r["source"] == "human"]
    # Both human stamps survive (PK collisions in each root bump the stamp by
    # a microsecond, which does not matter) — they are never collapsed.
    assert len(human) == 2
    assert len({r["observed_at"] for r in human}) == 2


def test_identical_human_rows_collapse(tmp_path: Path) -> None:
    src_a, src_b = mk_backup_pair(tmp_path, evidence_sha="c" * 64)
    for root in (src_a, src_b):
        append_label_observation(
            root,
            SID,
            labels=["rejected"],
            pr_state=None,
            labeler_version="1055-human-r1",
            evidence_sha="f" * 64,
            source="human",
            observed_at="2026-05-04T00:00:00+00:00",
        )
    merged = dedupe_observations([read_label_rows(src_a), read_label_rows(src_b)])
    human = [r for r in merged["rows"] if r["source"] == "human"]
    assert len(human) == 1
    # one auto duplicate from the backup pair + one human duplicate
    assert merged["deduped_count"] == 2


def test_same_tuple_diff_payload_routes_to_content_conflict(tmp_path: Path) -> None:
    # The versioned dedup tuple matches but the immutable payload differs
    # (different rubric_json) — ambiguous evidence routes to content_conflict,
    # never silently keeps one.
    src_a, src_b = mk_backup_pair(tmp_path, evidence_sha="c" * 64)
    _force_insert_rubric_variant(src_b, observed_at="2026-05-05T00:00:00+00:00")
    merged = dedupe_observations([read_label_rows(src_a), read_label_rows(src_b)])
    assert merged["rows"] == []
    # every row sharing the ambiguous tuple is reported, none silently kept
    assert len(merged["content_conflict"]) == 3
    assert merged["deduped_count"] == 0


def _force_insert_rubric_variant(root: Path, *, observed_at: str) -> None:
    """Insert a row matching the tuple but carrying a different rubric_json."""
    write = sqlite3.connect(root / "index.db")
    write.execute(
        "INSERT INTO label_observations (session_id, observed_at, labels, pr_state, labeler_version,"
        " evidence_sha, rubric_json, valid_at, reward_version, reward_json, composite_reward,"
        " reviewer_logins, has_posterior, source, labeler_policy_version,"
        " reply_classifier_version, reply_evidence_digest, legacy) VALUES ("
        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            SID,
            observed_at,
            '["accepted"]',
            None,
            "980-rubric-r2",
            "c" * 64,
            '{"hash": "variant"}',
            None,
            None,
            None,
            None,
            None,
            0,
            "auto",
            "980-rubric-r2",
            None,
            None,
            "auto",
        ),
    )
    write.commit()
    write.close()


def test_empty_inventories(tmp_path: Path) -> None:
    merged = dedupe_observations([[], []])
    assert merged == {"rows": [], "deduped_count": 0, "content_conflict": []}
