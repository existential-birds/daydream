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

from daydream.archive.hydrate_rules import (
    HYDRATION_INDEX_SCHEMA_VERSION,
    REASON_CODE_IMPORT_DECISIVE_PER_FINDING,
    REASON_CODE_IMPORT_IDENTITY_CONFLICT,
    REASON_CODE_IMPORT_INVALID_VERSION,
    REASON_CODE_IMPORT_RUN_LEVEL_ONLY,
    REASON_CODE_IMPORT_STALE_EVIDENCE,
    REASON_CODE_IMPORT_UNMATCHED_SESSION,
)
from daydream.archive.importer import (
    IMPORT_REASON_CODES,
    accounting,
    build_import_ledger,
    canonical_payload_digest,
    classify_run_level,
    dedupe_observations,
    gold_eligible,
    link_session_identity,
    merge_imported_observations,
    run_pure_import,
)
from daydream.archive.index import (
    append_label_observation,
    label_observation_history,
    upsert_run,
)
from daydream.archive.known_versions import STALE_LEGACY
from daydream.training.labeler_versions import HUMAN_LABELER_VERSION
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


# --- Task 4: version-gated gold eligibility (M6, KD3) -----------------------


def make_observation(ver: str) -> dict[str, Any]:
    """Minimal observation dict with all version axes set to ``ver``."""
    return {
        "session_id": SID,
        "observed_at": _OBSERVED_A,
        "labels": ["accepted"],
        "labeler_version": ver,
        "labeler_policy_version": ver,
        "reply_classifier_version": ver,
        "legacy": "legacy" if ver == STALE_LEGACY else "auto",
    }


def test_gold_eligibility_three_fixtures() -> None:
    valid = make_observation(ver=HUMAN_LABELER_VERSION)  # gold participates
    stale = make_observation(ver=STALE_LEGACY)  # non-gold
    unknown = make_observation(ver="9999-future-r9")  # non-gold
    assert [obs for obs in (valid, stale, unknown) if gold_eligible(obs)] == [valid]


# --- Task 5: run-level labels stay run-level (M5, AC2) ----------------------


def make_session_row(
    *,
    session_id: str = SID,
    evidence_sha: str = "c" * 64,
    labels: str | list[str] | None = None,
    record_id: str | None = None,
) -> dict[str, Any]:
    """A run-level (session-scoped, no record_id) label observation row."""
    return {
        "session_id": session_id,
        "observed_at": _OBSERVED_A,
        "record_id": record_id,
        "evidence_sha": evidence_sha,
        "labels": json.dumps(labels) if isinstance(labels, list) else (labels or '["finding-accepted"]'),
        "source": "human",
    }


def test_run_level_label_never_fans_out() -> None:
    # A run-level label with no projected findings in the session: it is
    # emitted as run-level evidence only — never copied onto every finding
    # of the run (AC2) — and lands in the run_level_only bucket.
    rec = make_session_row()
    out = classify_run_level([rec], projector_findings={})
    assert out["per_finding"].get(SID) is None
    assert SID in out["run_level_only"]
    assert out["ambiguous_run_mapping"] == {}


def test_ambiguous_mapping_routes_to_queue() -> None:
    # Two candidate findings, neither matching the row's evidence digest:
    # the run<->finding mapping is ambiguous — the row routes to the
    # ambiguous_run_mapping bucket (the per-finding adjudication queue),
    # never a fan-out and never a silent per_finding substitution.
    rec = make_session_row()
    ambiguous = {
        SID: [
            {"record_id": "r1", "evidence_sha": "z" * 64},
            {"record_id": "r2", "evidence_sha": "y" * 64},
        ]
    }
    out = classify_run_level([rec], projector_findings=ambiguous)
    assert SID in out["ambiguous_run_mapping"]
    assert out["per_finding"].get(SID) is None
    assert SID not in out["run_level_only"]


def test_decisive_identity_and_digest_match_lands_per_finding() -> None:
    # Exactly one candidate finding matching identity + evidence digest:
    # the row is per-finding eligible (feeds the _is_admitted_outcome_gold
    # path through the existing decisive-only semantics).
    rec = make_session_row(evidence_sha="c" * 64)
    findings = {SID: [{"record_id": "r1", "evidence_sha": "c" * 64}]}
    out = classify_run_level([rec], projector_findings=findings)
    assert out["per_finding"][SID] == [rec]
    assert out["run_level_only"] == {}
    assert out["ambiguous_run_mapping"] == {}


def test_multiple_candidates_one_digest_match_is_decisive() -> None:
    # Multiple candidates but exactly one matches the evidence digest: the
    # identity+digest match is decisive, not ambiguous.
    rec = make_session_row(evidence_sha="c" * 64)
    findings = {
        SID: [
            {"record_id": "r1", "evidence_sha": "z" * 64},
            {"record_id": "r2", "evidence_sha": "c" * 64},
        ]
    }
    out = classify_run_level([rec], projector_findings=findings)
    assert out["per_finding"][SID] == [rec]


def test_malformed_labels_json_raises_naming_session() -> None:
    rec = make_session_row(labels="{not-json")
    with pytest.raises(ValueError, match=SID):
        classify_run_level([rec], projector_findings={})


def test_referenced_finding_missing_fields_raises() -> None:
    # A projector_findings entry referenced by the run is malformed (missing
    # record_id/evidence_sha): raise, never silently substitute.
    rec = make_session_row(evidence_sha="c" * 64)
    findings = {SID: [{"record_id": "r1"}]}  # no evidence_sha
    with pytest.raises(ValueError, match="evidence_sha"):
        classify_run_level([rec], projector_findings=findings)


def test_per_finding_row_is_not_run_level() -> None:
    # A row that already carries a record_id is per-finding evidence; it is
    # accounted for in per_finding, never routed through the run-level
    # buckets (M7: bucket sum == source row count).
    rec = make_session_row(record_id="r1")
    out = classify_run_level([rec], projector_findings={})
    assert out["per_finding"][SID] == [rec]
    assert out["run_level_only"] == {}
    assert out["ambiguous_run_mapping"] == {}


def test_buckets_account_for_every_row() -> None:
    # Deterministic partition: every input row lands in exactly one bucket.
    run_level = make_session_row()
    per_finding = make_session_row(record_id="r1", evidence_sha="c" * 64)
    out = classify_run_level([run_level, per_finding], projector_findings={})
    total = sum(len(v) for v in out["per_finding"].values())
    total += len(out["run_level_only"]) + len(out["ambiguous_run_mapping"])
    assert total == 2


# ---------------------------------------------------------------------------
# Task 7: reason-coded accounting (M7, KD5)
# ---------------------------------------------------------------------------

SIX_BUCKET_CODES = IMPORT_REASON_CODES


def _import_row(
    session_id: str,
    *,
    evidence_sha: str,
    derivative_digest: str = D,
    versions: str = "gold",
) -> dict[str, Any]:
    """One inventory-shaped observation row with distinct evidence per session.

    ``versions="gold"`` stamps the known-versions allowlist axes;
    anything else stamps an unknown version (non-gold, M6).
    """
    if versions == "gold":
        labeler, policy, classifier = "980-rubric-r2", "980-policy-r1", "980-classifier-r1"
    else:
        labeler = policy = classifier = "9999-future-r9"
    return {
        "session_id": session_id,
        "observed_at": _OBSERVED_A,
        "labels": '["accepted"]',
        "pr_state": None,
        "labeler_version": labeler,
        "evidence_sha": evidence_sha,
        "rubric_json": None,
        "valid_at": None,
        "reward_version": None,
        "reward_json": None,
        "composite_reward": None,
        "reviewer_logins": None,
        "has_posterior": 0,
        "source": "auto",
        "labeler_policy_version": policy,
        "reply_classifier_version": classifier,
        "reply_evidence_digest": None,
        "derivative_digest": derivative_digest,
        "repo_slug": "org/repo",
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
    }


def _six_row_fixture() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """One row per bucket, each session distinct (no dedupe overlap).

    Returns (inventory rows, hydrated_index, projector_findings) covering:
    unmatched, identity conflict, stale evidence (ambiguous run mapping),
    invalid version, decisive per-finding, and run-level-only.
    """
    rows = [
        _import_row("s-unmatched", evidence_sha="1" * 64),
        _import_row("s-conflict", evidence_sha="2" * 64),
        _import_row("s-stale", evidence_sha="3" * 64),
        _import_row("s-invalid", evidence_sha="4" * 64, versions="unknown"),
        _import_row("s-perfinding", evidence_sha="5" * 64),
        _import_row("s-runlevel", evidence_sha="6" * 64),
    ]
    hydrated_index = {
        "s-conflict": {"derivative_digest": "e" * 64, "record_id": "rec-conflict"},
        "s-stale": {"derivative_digest": D, "record_id": "rec-stale"},
        "s-invalid": {"derivative_digest": D, "record_id": "rec-invalid"},
        "s-perfinding": {"derivative_digest": D, "record_id": "rec-pf"},
        "s-runlevel": {"derivative_digest": D, "record_id": "rec-rl"},
    }
    projector_findings = {
        "s-stale": [{"record_id": "r1", "evidence_sha": "z" * 64}],  # no digest match
        "s-perfinding": [{"record_id": "r1", "evidence_sha": "5" * 64}],  # decisive
    }
    return rows, hydrated_index, projector_findings


def test_reason_codes_sum_to_source_row_count() -> None:
    rows, hydrated_index, projector_findings = _six_row_fixture()
    result = run_pure_import(
        [rows],
        hydrated_index=hydrated_index,
        repo_slug_sha_lookup={},
        projector_findings=projector_findings,
    )
    counts = [result["accounting"][code] for code in SIX_BUCKET_CODES]
    assert set(result["accounting"]) == set(SIX_BUCKET_CODES)
    assert sum(counts) == len(rows)  # M7: bucket sum == source row count


def test_each_bucket_reason_stable() -> None:
    rows, hydrated_index, projector_findings = _six_row_fixture()
    result = run_pure_import(
        [rows],
        hydrated_index=hydrated_index,
        repo_slug_sha_lookup={},
        projector_findings=projector_findings,
    )
    acc = result["accounting"]
    assert acc[REASON_CODE_IMPORT_UNMATCHED_SESSION] == 1
    assert acc[REASON_CODE_IMPORT_IDENTITY_CONFLICT] == 1
    assert acc[REASON_CODE_IMPORT_STALE_EVIDENCE] == 1
    assert acc[REASON_CODE_IMPORT_INVALID_VERSION] == 1
    assert acc[REASON_CODE_IMPORT_DECISIVE_PER_FINDING] == 1
    assert acc[REASON_CODE_IMPORT_RUN_LEVEL_ONLY] == 1


def test_ledger_mirrors_hydrate_import_ledger_shape() -> None:
    rows, hydrated_index, projector_findings = _six_row_fixture()
    result = run_pure_import(
        [rows],
        hydrated_index=hydrated_index,
        repo_slug_sha_lookup={},
        projector_findings=projector_findings,
    )
    ledger = build_import_ledger(result)
    assert ledger["schema_version"] == HYDRATION_INDEX_SCHEMA_VERSION
    assert set(ledger["accounting"]) == set(SIX_BUCKET_CODES)
    assert sum(ledger["accounting"].values()) == len(ledger["observations"])
    assert len(ledger["observations"]) == len(rows)
    by_sid = {e["session_id"]: e for e in ledger["observations"]}
    assert by_sid["s-unmatched"]["reason_code"] == REASON_CODE_IMPORT_UNMATCHED_SESSION
    assert by_sid["s-conflict"]["reason_code"] == REASON_CODE_IMPORT_IDENTITY_CONFLICT
    assert by_sid["s-stale"]["reason_code"] == REASON_CODE_IMPORT_STALE_EVIDENCE
    assert by_sid["s-invalid"]["reason_code"] == REASON_CODE_IMPORT_INVALID_VERSION
    assert by_sid["s-perfinding"]["reason_code"] == REASON_CODE_IMPORT_DECISIVE_PER_FINDING
    assert by_sid["s-runlevel"]["reason_code"] == REASON_CODE_IMPORT_RUN_LEVEL_ONLY
    for entry in ledger["observations"]:
        assert set(entry) == {"session_id", "observed_at", "source", "reason_code"}


def test_unclassifiable_row_raises_naming_it() -> None:
    # A merged row absent from every link/run-level result and carrying no
    # dedupe conflict: fail closed with a ValueError naming the row — never
    # an implicit drop (M7).
    row = _import_row("s-orphan", evidence_sha="7" * 64)
    with pytest.raises(ValueError, match="s-orphan"):
        accounting(
            [row],
            content_conflict=[],
            link_result={"linked": {}, "unmatched": {}, "identity_conflict": {}},
            run_level_result={"per_finding": {}, "run_level_only": {}, "ambiguous_run_mapping": {}},
        )


# ---------------------------------------------------------------------------
# Task 6: merge imported observations via the canonical-harvest seam
# ---------------------------------------------------------------------------


def _row_with_digest(row: dict[str, Any]) -> dict[str, Any]:
    """Attach the inventory-time payload digest a merge validates against."""
    row = dict(row)
    row["payload_digest"] = canonical_payload_digest(
        row, include_observed_at=row["source"] != "auto"
    )
    return row


def _seed_generation(root: Path, *, observed_at: str, evidence_sha: str, labels: list[str]) -> None:
    append_label_observation(
        root,
        SID,
        labels=labels,
        pr_state=None,
        labeler_version="980-rubric-r2",
        evidence_sha=evidence_sha,
        valid_at="2026-04-30T00:00:00+00:00",
        reply_evidence_digest=None,
        reward_version=None,
        has_posterior=False,
        source="auto",
        observed_at=observed_at,
    )


def test_bitemporal_history_preserved(tmp_path: Path) -> None:
    # Three evidence generations A->B->C captured in a surviving backup are
    # appended into the target archive in order, verbatim (M3).
    source_root = tmp_path / "backup"
    source_root.mkdir()
    _seed_run(source_root)
    for at, sha, labels in (
        (_OBSERVED_A, "e" * 64, ["accepted"]),
        (_OBSERVED_B, "f" * 64, ["rejected"]),
        ("2026-05-03T00:00:00+00:00", "0" * 64, ["accepted"]),
    ):
        _seed_generation(source_root, observed_at=at, evidence_sha=sha, labels=labels)
    imported = read_label_rows(source_root)

    target = tmp_path / "target"
    target.mkdir()
    _seed_run(target)
    merged = merge_imported_observations(target, [_row_with_digest(r) for r in imported])

    assert merged["appended"] == 3
    assert merged["deduped"] == 0
    hist = label_observation_history(target, SID)
    assert [r["observed_at"] for r in hist] == [
        _OBSERVED_A,
        _OBSERVED_B,
        "2026-05-03T00:00:00+00:00",
    ]
    assert hist[0]["labels"] == json.dumps(["accepted"])  # verbatim
    assert hist[1]["labels"] == json.dumps(["rejected"])


def test_newer_existing_observation_never_displaced(tmp_path: Path) -> None:
    # The target already holds a newer auto observation; importing an older
    # auto generation appends it but the precedence projection (recency within
    # source class) keeps the newer row as the winner in the denormalized
    # runs cache — import must never bypass the projection (M3).
    target = tmp_path / "target"
    target.mkdir()
    _seed_run(target)
    newer_at = "2026-06-01T00:00:00+00:00"
    _seed_generation(target, observed_at=newer_at, evidence_sha="f" * 64, labels=["accepted"])

    source_root = tmp_path / "backup"
    source_root.mkdir()
    _seed_run(source_root)
    _seed_generation(source_root, observed_at=_OBSERVED_A, evidence_sha="e" * 64, labels=["rejected"])
    imported = [_row_with_digest(r) for r in read_label_rows(source_root)]

    merged = merge_imported_observations(target, imported)
    assert merged["appended"] == 1
    hist = label_observation_history(target, SID)
    assert [r["observed_at"] for r in hist] == [_OBSERVED_A, newer_at]  # append-only
    # runs.outcome_labels cache still reflects the precedence projection:
    conn = sqlite3.connect(target / "index.db")
    try:
        conn.row_factory = sqlite3.Row
        run = conn.execute("SELECT outcome_labels, labeled_at FROM runs WHERE session_id = ?", (SID,)).fetchone()
    finally:
        conn.close()
    assert run["labeled_at"] == newer_at
    assert run["outcome_labels"] == json.dumps(["accepted"])


def test_idempotent_reimport_dedupes_auto_rows(tmp_path: Path) -> None:
    # Byte-identical re-import (M4): auto rows already present dedupe via the
    # existing writer; no new generations appear.
    source_root = tmp_path / "backup"
    source_root.mkdir()
    _seed_run(source_root)
    _seed_generation(source_root, observed_at=_OBSERVED_A, evidence_sha="e" * 64, labels=["accepted"])
    imported = [_row_with_digest(r) for r in read_label_rows(source_root)]

    target = tmp_path / "target"
    target.mkdir()
    _seed_run(target)
    merge_imported_observations(target, imported)
    before = label_observation_history(target, SID)

    merged_again = merge_imported_observations(target, imported)
    assert merged_again["appended"] == 0
    assert merged_again["deduped"] == 1
    assert label_observation_history(target, SID) == before


def test_drift_fails_closed_before_any_write(tmp_path: Path) -> None:
    # A row whose immutable payload changed between inventory and merge is
    # rejected with a ValueError naming the row; nothing is written (M9
    # fail-closed, mirroring the canonical-harvest drift gate).
    target = tmp_path / "target"
    target.mkdir()
    _seed_run(target)
    good: dict[str, Any] = {"session_id": SID, "source": "auto", "observed_at": _OBSERVED_A}
    row = dict(good)
    row.update(
        labels=["accepted"],
        pr_state=None,
        labeler_version="980-rubric-r2",
        evidence_sha="e" * 64,
        rubric_json=None,
        valid_at=None,
        reward_version=None,
        reward_json=None,
        composite_reward=None,
        reviewer_logins=None,
        has_posterior=0,
        labeler_policy_version="980-policy-r1",
        reply_classifier_version=None,
        reply_evidence_digest=None,
    )
    tampered = _row_with_digest(row)
    tampered["labels"] = ["smuggled"]
    with pytest.raises(ValueError, match=SID):
        merge_imported_observations(target, [tampered])
    assert label_observation_history(target, SID) == []


def test_dry_run_plans_without_writing(tmp_path: Path) -> None:
    source_root = tmp_path / "backup"
    source_root.mkdir()
    _seed_run(source_root)
    _seed_generation(source_root, observed_at=_OBSERVED_A, evidence_sha="e" * 64, labels=["accepted"])
    imported = [_row_with_digest(r) for r in read_label_rows(source_root)]

    target = tmp_path / "target"
    target.mkdir()
    _seed_run(target)
    merged = merge_imported_observations(target, imported, dry_run=True)

    assert merged["dry_run"] is True
    assert len(merged["planned"]) == 1
    assert merged["planned"][0]["session_id"] == SID
    assert merged["planned"][0]["observed_at"] == _OBSERVED_A
    assert merged["planned"][0]["source"] == "auto"
    assert label_observation_history(target, SID) == []


def test_human_source_appended_verbatim(tmp_path: Path) -> None:
    # Human evidence keeps its source class and explicit observed_at: the
    # writer never dedupes human rows (S0-1) and the bitemporal stamp is
    # preserved exactly.
    target = tmp_path / "target"
    target.mkdir()
    _seed_run(target)
    row = {
        "session_id": SID,
        "source": "human",
        "observed_at": _OBSERVED_B,
        "labels": ["rejected"],
        "pr_state": None,
        "labeler_version": HUMAN_LABELER_VERSION,
        "evidence_sha": None,
        "rubric_json": None,
        "valid_at": _OBSERVED_B,
        "reward_version": None,
        "reward_json": None,
        "composite_reward": None,
        "reviewer_logins": None,
        "has_posterior": 0,
        "labeler_policy_version": STALE_LEGACY,
        "reply_classifier_version": None,
        "reply_evidence_digest": None,
    }
    merged = merge_imported_observations(target, [_row_with_digest(row)])
    assert merged["appended"] == 1
    hist = label_observation_history(target, SID)
    assert len(hist) == 1
    assert hist[0]["source"] == "human"
    assert hist[0]["observed_at"] == _OBSERVED_B
    assert hist[0]["labels"] == json.dumps(["rejected"])

# ---------------------------------------------------------------------------
# Task 8: fail-closed secret scan + redaction before publication (M9, AC6)
# ---------------------------------------------------------------------------

from daydream.archive.hydrate_rules import (  # noqa: E402
    REASON_CODE_IMPORT_UNREDACTABLE_METADATA,
)
from daydream.archive.importer import REDACTED_PATH, redact_imported_metadata  # noqa: E402
from daydream.archive.scan import scan_run_dir  # noqa: E402


def _metadata_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "session_id": SID,
        "observed_at": _OBSERVED_A,
        "source": "human",
        "remote_url": "https://github.com/acme/widget.git",
        "source_path": None,
        "labels": [],
        "pr_state": None,
        "labeler_version": HUMAN_LABELER_VERSION,
        "evidence_sha": None,
        "rubric_json": None,
        "valid_at": _OBSERVED_A,
        "reward_version": None,
        "reward_json": None,
        "composite_reward": None,
        "reviewer_logins": None,
        "has_posterior": 0,
        "labeler_policy_version": STALE_LEGACY,
        "reply_classifier_version": None,
        "reply_evidence_digest": None,
    }
    row.update(overrides)
    return row


def test_clean_metadata_passes_scan(tmp_path: Path) -> None:
    scan_dir = tmp_path / "scan"
    result = redact_imported_metadata([_metadata_row()], scan_dir=scan_dir)
    assert result["blocked"] is False
    assert scan_dir.is_dir()
    assert scan_run_dir(scan_dir).clean
    # The payload file scanned is the redacted payload itself.
    payload = json.loads((scan_dir / "payload.json").read_text(encoding="utf-8"))
    assert payload == result["payload"]


def test_absolute_paths_redacted(tmp_path: Path) -> None:
    row = _metadata_row(
        remote_url="/Users/k/proj",
        source_path="/Users/k/proj/inner",
        rubric_json=json.dumps({"workdir": "/Users/k/proj/build", "note": "ok"}),
        reward_json=json.dumps([{"home": "/Users/k/proj/out"}]),
    )
    result = redact_imported_metadata([row], scan_dir=tmp_path / "scan")
    assert result["blocked"] is False
    encoded = json.dumps(result["payload"])
    assert "/Users/k" not in encoded
    out_row = result["payload"][0]
    assert REDACTED_PATH in out_row["remote_url"]
    assert REDACTED_PATH in out_row["source_path"]
    rubric = json.loads(out_row["rubric_json"])
    assert rubric["workdir"] == REDACTED_PATH
    assert rubric["note"] == "ok"
    reward = json.loads(out_row["reward_json"])
    assert reward[0]["home"] == REDACTED_PATH


def test_credential_url_redacted(tmp_path: Path) -> None:
    row = _metadata_row(remote_url="https://user:ghp_secret@github.com/acme/widget.git")
    result = redact_imported_metadata([row], scan_dir=tmp_path / "scan")
    assert result["blocked"] is False
    assert "ghp_secret" not in json.dumps(result["payload"])
    assert scan_run_dir(tmp_path / "scan").clean


def test_dirty_metadata_blocks_publish(tmp_path: Path) -> None:
    # A credential-shaped string in a free-text field the redaction pass does
    # not own cannot be made clean: the payload is flagged blocked — never
    # downgraded to a warning, never dropped silently.
    dirty = _metadata_row(notes="clone from https://user:secret1@github.com/x/y.git")
    result = redact_imported_metadata([_metadata_row(), dirty], scan_dir=tmp_path / "scan")
    assert scan_run_dir(tmp_path / "scan").clean is False
    assert result["blocked"] is True
    assert result["blocked_reasons"] == [REASON_CODE_IMPORT_UNREDACTABLE_METADATA]


def test_blocked_payload_carries_marker_skipping(tmp_path: Path) -> None:
    # Already-redacted markers are safe output, not secrets: a payload whose
    # only "suspicious" text is our own marker scans clean.
    row = _metadata_row(notes=f"workdir was {REDACTED_PATH}")
    result = redact_imported_metadata([row], scan_dir=tmp_path / "scan")
    assert result["blocked"] is False
    assert len(result["payload"]) == 1


def test_malformed_rubric_json_raises(tmp_path: Path) -> None:
    row = _metadata_row(rubric_json="{not json")
    with pytest.raises(ValueError, match="rubric_json"):
        redact_imported_metadata([row], scan_dir=tmp_path / "scan")
