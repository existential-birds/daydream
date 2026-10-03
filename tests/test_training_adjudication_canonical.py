"""Canonical harvest preserves pre-write drift refusal, three-tier precedence, and exactly-once observation
appends.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from daydream.archive.index import _get_connection, label_observation_history
from daydream.training.adjudication.canonical import AnnotationDriftError, run_canonical_harvest
from daydream.training.adjudication.materialize import run_materialize
from daydream.training.adjudication.snapshot import FindingRecord, record_evidence_digest
from daydream.training.corpus_projection.identity import record_id
from tests.harness.adjudication import make_hydrated_sqlite_index, seed_index_dispositions, write_sessions_jsonl

_PIN = {"curation_id": "cur-1", "sanitized_hub_commit": "a" * 40,
    "source_hub_commit": "b" * 40, "archive_index_digest": "c" * 64,
    "evidence_observed_at": "2026-01-01T00:00:00+00:00", "as_of": "2026-02-01T00:00:00+00:00",
    "labeler_version": "v1", "rubric_version": "v1", "classifier_version": "v1",
}


def _annotation_records(mat: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (mat / "annotations.jsonl").read_text().splitlines() if line.strip()]


def _index(
    tmp_path: Path, digest: str = "d" * 32, *, session_id: str = "s1", created_at: str = "2026-01-01T00:00:00+00:00",
) -> Path:
    root = tmp_path / "index"
    sessions = [{"session_id": session_id, "trajectory_id": f"{session_id}-t", "segment_id": f"{session_id}-seg",
        "resolutions": [{"fingerprint": "fp-1", "disposition": "unanswered",
            "evidence": [{"reply_id": 1, "body_sha256": "abc", "created_at": created_at}],
            "evidence_digest": digest, "profile": "pr_review", "stack": "python", "comment_id": 7,
        }],
    }]
    write_sessions_jsonl(root, sessions)
    (root / "index-revision.txt").write_text("a" * 40, encoding="utf-8")
    return root


def _stored_resolutions(rubric: dict[str, Any]) -> list[dict[str, Any]]:
    """Read a stored rubric's per-finding resolutions (either key spelling)."""
    stored = rubric.get("per_finding_outcomes") or rubric.get("per_finding_resolutions")
    return stored if isinstance(stored, list) else []


def _write_observation(path: Path, record_id: str, *, labeler: str, role: str, rationale: str, observed_at: str,
    disposition: str = "accepted", evidence_digest: str = "d" * 32,
) -> None:
    path.write_text(json.dumps({
        "record_id": record_id, "disposition": disposition, "evidence_digest": evidence_digest, "evidence": [],
        "labeler": labeler, "role": role, "rationale": rationale,
        "valid_at": "2026-01-02T00:00:00+00:00", "observed_at": observed_at, "rubric_version": "v1",
    }) + "\n", encoding="utf-8")


def _seed_archive(archive_dir: Path, session_id: str = "s1") -> None:
    conn = _get_connection(archive_dir)
    conn.execute("INSERT INTO runs (session_id, archived_at, run_flow, archive_path) "
        "VALUES (?, '2026-01-01T00:00:00+00:00', 'deep', ?)", (session_id, f"archive/{session_id}"),
    )
    conn.commit()
    conn.close()


def _materialized(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Index + seeded archive + materialize dir under the canonical pin."""
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    _seed_archive(archive)
    mat = tmp_path / "mat"
    run_materialize(root, mat, pin=_PIN)
    return root, archive, mat


def _harvest(root: Path, archive: Path, mat: Path, observations_path: Path | None = None,) -> dict[str, Any]:
    return run_canonical_harvest(
        index_root=root, materialize_dir=mat, archive_dir=archive, observations_path=observations_path,
    )


def test_canonical_harvest_appends_label_observation_exactly_once(tmp_path: Path) -> None:
    root, archive, mat = _materialized(tmp_path)
    out = _harvest(root, archive, mat, None)
    assert out["appended_sessions"] == 1
    history = label_observation_history(archive, "s1")
    assert len(history) == 1  # exactly once
    row = history[0]
    assert row["labeler_version"] == _PIN["labeler_version"]
    rubric = json.loads(row["rubric_json"])
    stored = _stored_resolutions(rubric)
    assert stored and stored[0]["evidence_digest"] == "d" * 32
    assert row["reply_evidence_digest"] == record_evidence_digest([stored[0]["evidence"]])
    out2 = _harvest(root, archive, mat, None)
    assert out2["appended_sessions"] == 0
    assert len(label_observation_history(archive, "s1")) == 1

def test_canonical_harvest_fails_closed_on_drift_before_any_write(tmp_path: Path) -> None:
    root, archive, mat = _materialized(tmp_path)
    sessions_path = root / "sessions.jsonl"
    s = json.loads(sessions_path.read_text().splitlines()[0])
    s["resolutions"][0]["evidence_digest"] = "f" * 32
    s["resolutions"][0]["evidence"][0]["body_sha256"] = "mut"
    sessions_path.write_text(json.dumps(s, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(AnnotationDriftError) as excinfo:
        _harvest(root, archive, mat, None)
    assert excinfo.value.requeued_record_ids  # named for requeue (AC 4/M5)
    assert label_observation_history(archive, "s1") == []  # nothing written
    assert not (tmp_path / "mat" / "annotations.jsonl").exists()

def test_canonical_harvest_fails_closed_when_record_absent_from_fresh_queue(tmp_path: Path,) -> None:
    """Exercise missing fresh-queue identity separately from digest drift; both must refuse before writing."""
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    run_materialize(root, tmp_path / "mat", pin=_PIN)
    # Remove the finding from the fresh index to exercise identity refusal.
    sessions = json.loads((root / "sessions.jsonl").read_text().splitlines()[0])
    sessions["session_id"] = "gone"
    (root / "sessions.jsonl").write_text(json.dumps(sessions, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent from the freshly built"):
        _harvest(root, archive, tmp_path / "mat", None)
    assert not (tmp_path / "mat" / "annotations.jsonl").exists()

def test_canonical_harvest_fails_closed_on_missing_materialized_outputs(tmp_path: Path,) -> None:
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    missing_manifest = tmp_path / "missing-manifest"
    with pytest.raises(FileNotFoundError, match="preview manifest not found"):
        _harvest(root, archive, missing_manifest, None)
    mat = tmp_path / "mat"
    mat.mkdir()
    (mat / "preview-manifest.json").write_text(json.dumps(_PIN, sort_keys=True), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="materialized preview snapshot not found"):
        _harvest(root, archive, mat, None)
    assert not (mat / "annotations.jsonl").exists()

def test_canonical_harvest_fails_closed_on_unreadable_materialized_outputs(tmp_path: Path,) -> None:
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    bad_manifest = tmp_path / "bad-manifest"
    bad_manifest.mkdir()
    (bad_manifest / "preview-manifest.json").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable preview manifest"):
        _harvest(root, archive, bad_manifest, None)
    mat = tmp_path / "mat"
    mat.mkdir()
    (mat / "preview-manifest.json").write_text(json.dumps(_PIN, sort_keys=True), encoding="utf-8")
    (mat / "sessions.jsonl").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable materialized snapshot"):
        _harvest(root, archive, mat, None)
    assert not (mat / "annotations.jsonl").exists()


def _seed_decisive_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Seed accepted, rejected, and unanswered findings so materialization and drift checks must retain
    decisive records.
    """
    root = tmp_path / "index"
    seed_index_dispositions(root)
    archive = tmp_path / "archive"
    _seed_archive(archive)
    for n in (2, 3):
        _seed_archive(archive, session_id=f"s{n}")
    mat = tmp_path / "mat"
    run_materialize(root, mat, pin=_PIN)
    return root, archive, mat


def test_canonical_harvest_merges_human_observations_by_precedence(tmp_path: Path) -> None:
    root, archive, mat = _materialized(tmp_path)
    rid = record_id("s1", "s1-t", "s1-seg", "fp-1")
    obs = tmp_path / "observations.jsonl"
    _write_observation(
        obs, rid, labeler="alice", role="rater", rationale="looked right", observed_at="2026-01-02T01:00:00+00:00",
    )
    out = _harvest(root, archive, mat, obs)
    assert out["human_adjudicated"] == 1
    history = label_observation_history(archive, "s1")
    rubric = json.loads(history[0]["rubric_json"])
    stored = _stored_resolutions(rubric)
    assert stored[0]["disposition"] == "accepted"

def test_canonical_harvest_rejects_unknown_observation_record_id(tmp_path: Path) -> None:
    root, archive, mat = _materialized(tmp_path)
    obs = tmp_path / "observations.jsonl"
    _write_observation(
        obs, "e" * 64, labeler="alice", role="rater", rationale="x", observed_at="2026-01-02T01:00:00+00:00",
    )
    with pytest.raises(ValueError, match="e" * 64):
        _harvest(root, archive, mat, obs)
    assert label_observation_history(archive, "s1") == []

def test_canonical_harvest_emits_annotations_jsonl_from_merged_records(tmp_path: Path) -> None:
    root, archive, mat = _materialized(tmp_path)
    out = _harvest(root, archive, mat, None)
    assert out["record_count"] == 1
    lines = (tmp_path / "mat" / "annotations.jsonl").read_text().splitlines()
    records = _annotation_records(tmp_path / "mat")
    assert [r["record_id"] for r in records] == sorted(r["record_id"] for r in records)
    assert records[0]["evidence_digest"] == "d" * 32
    assert records[0]["session_id"] == "s1"
    for line in lines:
        assert line == json.dumps(json.loads(line), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def test_canonical_harvest_flags_evidence_after_as_of(tmp_path: Path) -> None:
    """Keep evidence after as_of but mark it ineligible for gold; earlier evidence remains unflagged."""
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    _seed_archive(archive)
    _seed_archive(archive, session_id="s2")
    run_materialize(root, tmp_path / "mat", pin=_PIN)
    _harvest(root, archive, tmp_path / "mat", None)
    _index(tmp_path, session_id="s2", created_at="2026-03-01T00:00:00+00:00")
    run_materialize(root, tmp_path / "mat2", pin=_PIN)
    out = _harvest(root, archive, tmp_path / "mat2", None)
    assert out["evidence_after_as_of"] == [_annotation_records(tmp_path / "mat2")[0]["record_id"]]
    records = _annotation_records(tmp_path / "mat2")
    assert records[0]["evidence_after_as_of"] is True
    # Use the real pre-pin timestamp so a missing key cannot make the assertion pass.
    first = _annotation_records(tmp_path / "mat")
    assert first[0]["evidence_after_as_of"] is False

def test_canonical_harvest_changed_pin_appends_new_generation(tmp_path: Path) -> None:
    """A changed pin must append a generation and preserve identical pin flags in the archived rubric and
    emitted annotations.
    """
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    _seed_archive(archive)
    # Evidence observed after pin-a's as_of but before pin-b's: only the
    # second, re-pinned harvest may flag evidence_after_as_of.
    _index(tmp_path, created_at="2026-02-15T00:00:00+00:00")
    pin_a = dict(_PIN, as_of="2026-03-01T00:00:00+00:00")  # evidence before as_of
    pin_b = dict(_PIN, as_of="2026-02-01T00:00:00+00:00", rubric_version="v2")  # after
    run_materialize(root, tmp_path / "mat-a", pin=pin_a)
    out_a = _harvest(root, archive, tmp_path / "mat-a", None)
    assert out_a["appended_sessions"] == 1
    assert out_a["evidence_after_as_of"] == []
    run_materialize(root, tmp_path / "mat-b", pin=pin_b)
    out_b = _harvest(root, archive, tmp_path / "mat-b", None)
    assert out_b["appended_sessions"] == 1
    assert out_b["skipped_sessions"] == 0
    history = label_observation_history(archive, "s1")
    assert len(history) == 2
    latest = history[-1]
    # The dedup key omits rubric JSON, so evidence_sha must include snapshot and rubric content.
    # Archive and emitted pin flags must agree.
    manifest = json.loads((tmp_path / "mat-b" / "preview-manifest.json").read_text())
    assert latest["evidence_sha"] == hashlib.sha256(
        (manifest["snapshot_id"] + ":" + latest["rubric_json"]).encode("utf-8")
    ).hexdigest()
    rubric = json.loads(latest["rubric_json"])
    stored = _stored_resolutions(rubric)
    assert rubric["rubric_version"] == "v2"
    assert stored[0]["evidence_after_as_of"] is True
    emitted = _annotation_records(tmp_path / "mat-b")
    assert emitted[0]["evidence_after_as_of"] is True
    out_c = _harvest(root, archive, tmp_path / "mat-b", None)
    assert out_c["appended_sessions"] == 0
    assert len(label_observation_history(archive, "s1")) == 2

def test_canonical_harvest_label_preserving_overlay_change_skips_nothing(tmp_path: Path,) -> None:
    """Label-preserving overlay edits still append: evidence_sha must include rubric content because the
    dedup tuple omits rubric_json.
    """
    root, archive, mat = _materialized(tmp_path)

    rid = record_id("s1", "s1-t", "s1-seg", "fp-1")
    obs = tmp_path / "observations.jsonl"
    _write_observation(
        obs, rid, labeler="alice", role="rater", rationale="first pass", observed_at="2026-01-02T01:00:00+00:00",
    )
    out1 = _harvest(root, archive, mat, obs)
    assert out1["appended_sessions"] == 1
    # A new human observation with the same label changes only rubric content and must append a
    # generation.
    _write_observation(
        obs, rid, labeler="bob", role="adjudicator", rationale="second pass", observed_at="2026-01-02T02:00:00+00:00",
    )
    out2 = _harvest(root, archive, mat, obs)
    assert out2["appended_sessions"] == 1  # fresh generation, never a silent skip
    history = label_observation_history(archive, "s1")
    assert len(history) == 2
    rubric = json.loads(history[-1]["rubric_json"])
    stored = _stored_resolutions(rubric)
    assert stored[0]["human_labeler"] == "bob"
    emitted = _annotation_records(tmp_path / "mat")
    assert emitted[0]["human_labeler"] == "bob"
    out3 = _harvest(root, archive, mat, obs)
    assert out3["appended_sessions"] == 0
    assert len(label_observation_history(archive, "s1")) == 2

def test_canonical_harvest_complete_set_is_idempotent_and_exactly_once(tmp_path: Path,) -> None:
    """Re-derive the complete record set, preserve untouched automatic decisive labels, and append nothing
    on an identical re-harvest.
    """
    stage, archive, mat = _seed_decisive_fixture(tmp_path)
    run_canonical_harvest(stage, mat, archive, observations_path=None)
    first = (mat / "annotations.jsonl").read_bytes()
    dispositions = [json.loads(ln)["disposition"] for ln in first.splitlines() if ln]
    assert sorted(dispositions) == ["accepted", "rejected", "unanswered"]

    summary2 = run_canonical_harvest(stage, mat, archive, observations_path=None)
    assert (mat / "annotations.jsonl").read_bytes() == first
    assert summary2["appended_sessions"] == 0
    assert summary2["skipped_sessions"] == 3


def _hydrated_sqlite_index_with_conflict(tmp_path: Path) -> Path:
    """Two disagreeing generations share a session; the latest automatic row wins but remains conflicting."""
    return make_hydrated_sqlite_index(tmp_path,
        [("2026-01-02T00:00:00+00:00", '["finding-accepted"]', "e" * 64, "980-rubric-r2", "accepted"),
            ("2026-01-03T00:00:00+00:00", '["finding-rejected"]', "f" * 64, "980-rubric-r2", "accepted"),
        ],
    )


def test_conflicted_session_yields_no_decisive_label(tmp_path: Path) -> None:
    root = _hydrated_sqlite_index_with_conflict(tmp_path)  # two distinct dedup keys, s1
    mat = tmp_path / "mat"
    run_materialize(root, mat, pin=_PIN)
    record = json.loads((tmp_path / "mat" / "sessions.jsonl").read_text().splitlines()[0])
    assert record["conflicting"] is True  # surfaced, never silently merged
    # Materialization uses a neutral disposition for this non-gold queue entry; the archive retains
    # its decisive provenance.
    assert record["disposition"] == "ambiguous"
    archive = tmp_path / "archive"
    _seed_archive(archive)
    out = _harvest(root, archive, mat)
    assert out["appended_sessions"] == 1
    history = label_observation_history(archive, "s1")
    rubric = json.loads(history[0]["rubric_json"])
    assert rubric["per_finding_resolutions"][0]["conflicting"] is True
    assert "finding-accepted" not in history[0]["labels"]

def test_canonical_harvest_re_derives_conflict_after_materialize(tmp_path: Path) -> None:
    """Re-derive conflicts at harvest time: a disagreeing post-materialization import can pass evidence-
    digest checks yet must suppress decisive labels.
    """

    root = make_hydrated_sqlite_index(
        tmp_path, [("2026-01-02T00:00:00+00:00", '["finding-accepted"]', "e" * 64, "980-rubric-r2", "accepted")],
    )
    mat = tmp_path / "mat"
    run_materialize(root, mat, pin=_PIN)
    record = json.loads((tmp_path / "mat" / "sessions.jsonl").read_text().splitlines()[0])
    assert record.get("conflicting") is None  # not conflicting at materialize time
    # An old disagreeing generation changes conflict state without changing the winning digest.
    rubric_json = json.dumps({"posterior_source": "pr_review",
        "per_finding_resolutions": [{"fingerprint": "fp-1", "comment_id": 7, "disposition": "accepted",
            "evidence": [{"reply_id": 1, "body_sha256": "abc"}], "evidence_digest": "d" * 32}]})
    conn = _get_connection(root)
    conn.execute("INSERT INTO label_observations (session_id, observed_at, labels, labeler_version, "
        "evidence_sha, rubric_json, has_posterior, source, labeler_policy_version) "
        "VALUES ('s1', '2026-01-01T00:00:00+00:00', '[\"finding-rejected\"]', 'v1', "
        "'d' * 64, ?, 0, 'auto', '980-rubric-r2')", (rubric_json,),
    )
    conn.commit()
    conn.close()
    archive = tmp_path / "archive"
    _seed_archive(archive)
    out = _harvest(root, archive, mat)
    assert out["appended_sessions"] == 1
    history = label_observation_history(archive, "s1")
    assert "finding-accepted" not in history[0]["labels"]
    rows = _annotation_records(mat)
    assert rows[0]["disposition"] == "ambiguous"
    assert rows[0]["conflicting"] is True

def test_canonical_harvest_human_resolution_clears_session_conflict(tmp_path: Path,) -> None:
    """A decisive human judgment clears the conflict flag and restores the decisive archive label."""

    root = _hydrated_sqlite_index_with_conflict(tmp_path)  # two distinct dedup keys, s1
    mat = tmp_path / "mat"
    run_materialize(root, mat, pin=_PIN)
    archive = tmp_path / "archive"
    _seed_archive(archive)
    rid = record_id("s1", "s1", "s1", "fp-1")
    obs = tmp_path / "observations.jsonl"
    _write_observation(
        obs, rid, labeler="alice", role="adjudicator", rationale="operator resolved the disagreeing generations",
        observed_at="2026-01-02T01:00:00+00:00",
    )
    out = _harvest(root, archive, mat, obs)
    assert out["human_adjudicated"] == 1
    history = label_observation_history(archive, "s1")
    rubric = json.loads(history[0]["rubric_json"])
    record = rubric["per_finding_resolutions"][0]
    assert record.get("conflicting") is not True  # cleared by the human resolution
    assert record["disposition"] == "accepted"
    assert "finding-accepted" in history[0]["labels"]
    rows = _annotation_records(mat)
    assert rows[0]["disposition"] == "accepted"
    assert rows[0].get("conflicting") is not True

def test_conflicted_session_never_projects_gold(tmp_path: Path) -> None:
    """Projecting a conflicted annotation must remain non-gold with no outcome label even when its
    disposition and evidence are decisive.
    """

    root = _hydrated_sqlite_index_with_conflict(tmp_path)  # two distinct dedup keys, s1
    mat = tmp_path / "mat"
    run_materialize(root, mat, pin=_PIN)
    archive = tmp_path / "archive"
    _seed_archive(archive)
    _harvest(root, archive, mat)
    rows = _annotation_records(mat)
    assert any(row.get("conflicting") is True for row in rows)
    # The snapshot uses the session-resolution shape consumed at the projection boundary.
    session = {"session_id": "s1", "trajectory_id": "s1", "segment_id": "s1",
        "resolutions": [row for row in rows if row.get("session_id") == "s1"],
    }
    records = [finding.project() for finding in FindingRecord.from_session(session)]
    assert records  # the finding is still projected -- provenance preserved
    assert all(r["tier"] != "gold" for r in records)
    assert all(r["outcome_label"] is None for r in records)
