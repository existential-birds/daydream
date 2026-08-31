"""Canonical harvest tests (issue #1055, Task 4).

Drift gate fail-closed pre-write, three-tier precedence merge, and the
exactly-once ``label_observations`` append (M5/M8).
"""

import json
from pathlib import Path

import pytest

from daydream.archive.index import _get_connection, label_observation_history
from daydream.training.adjudication.canonical import AnnotationDriftError, run_canonical_harvest
from daydream.training.adjudication.materialize import run_materialize

_PIN = {
    "curation_id": "cur-1", "sanitized_hub_commit": "a" * 40,
    "source_hub_commit": "b" * 40, "archive_index_digest": "c" * 64,
    "evidence_observed_at": "2026-01-01T00:00:00+00:00",
    "as_of": "2026-02-01T00:00:00+00:00",
    "labeler_version": "v1", "rubric_version": "v1", "classifier_version": "v1",
}


def _index(tmp_path: Path, digest: str = "d" * 32) -> Path:
    root = tmp_path / "index"
    root.mkdir(exist_ok=True)
    sessions = [{
        "session_id": "s1", "trajectory_id": "s1-t", "segment_id": "s1-seg",
        "resolutions": [{
            "fingerprint": "fp-1", "disposition": "unanswered",
            "evidence": [{"reply_id": 1, "body_sha256": "abc",
                          "created_at": "2026-01-01T00:00:00+00:00"}],
            "evidence_digest": digest, "profile": "pr_review", "stack": "python",
            "comment_id": 7,
        }],
    }]
    (root / "sessions.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions), encoding="utf-8"
    )
    (root / "index-revision.txt").write_text("a" * 40, encoding="utf-8")
    return root


def _seed_archive(archive_dir: Path) -> None:
    # Real-path seeding: the project's own schema (index.db), one run row.
    conn = _get_connection(archive_dir)
    conn.execute(
        "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) "
        "VALUES ('s1', '2026-01-01T00:00:00+00:00', 'deep', 'archive/s1')"
    )
    conn.commit()
    conn.close()


def test_canonical_harvest_appends_label_observation_exactly_once(tmp_path: Path) -> None:
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    _seed_archive(archive)
    run_materialize(root, tmp_path / "mat", pin=_PIN)
    out = run_canonical_harvest(
        index_root=root, materialize_dir=tmp_path / "mat", archive_dir=archive,
        observations_path=None,
    )
    assert out["appended_sessions"] == 1
    history = label_observation_history(archive, "s1")
    assert len(history) == 1  # exactly once
    row = history[0]
    assert row["labeler_version"] == _PIN["labeler_version"]
    rubric = json.loads(row["rubric_json"])
    stored = rubric.get("per_finding_outcomes") or rubric.get("per_finding_resolutions")
    assert stored and stored[0]["evidence_digest"] == "d" * 32
    # session-level digest matches the shared serializer's (K5 spike)
    from daydream.training.adjudication.snapshot import record_evidence_digest
    assert row["reply_evidence_digest"] == record_evidence_digest(
        [stored[0]["evidence"]]
    )
    # re-run unchanged => idempotent, no duplicate row
    out2 = run_canonical_harvest(
        index_root=root, materialize_dir=tmp_path / "mat", archive_dir=archive,
        observations_path=None,
    )
    assert out2["appended_sessions"] == 0
    assert len(label_observation_history(archive, "s1")) == 1


def test_canonical_harvest_fails_closed_on_drift_before_any_write(tmp_path: Path) -> None:
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    _seed_archive(archive)
    run_materialize(root, tmp_path / "mat", pin=_PIN)
    # evidence drifts AFTER materialization: harvest must refuse pre-write
    sessions_path = root / "sessions.jsonl"
    s = json.loads(sessions_path.read_text().splitlines()[0])
    s["resolutions"][0]["evidence_digest"] = "f" * 32
    s["resolutions"][0]["evidence"][0]["body_sha256"] = "mut"
    sessions_path.write_text(json.dumps(s, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(AnnotationDriftError) as excinfo:
        run_canonical_harvest(
            index_root=root, materialize_dir=tmp_path / "mat", archive_dir=archive,
            observations_path=None,
        )
    assert excinfo.value.requeued_record_ids  # named for requeue (AC 4/M5)
    assert label_observation_history(archive, "s1") == []  # nothing written
    assert not (tmp_path / "mat" / "annotations.jsonl").exists()


def test_canonical_harvest_fails_closed_when_record_absent_from_fresh_queue(
    tmp_path: Path,
) -> None:
    """Sibling fail-closed gate: a materialized record_id absent from the freshly
    built queue raises ValueError before any write (the absent-from-queue branch,
    not the digest-drift branch)."""
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    run_materialize(root, tmp_path / "mat", pin=_PIN)
    # The index is swapped out wholesale: the materialized s1 record no longer
    # exists in the freshly built queue.
    sessions = json.loads((root / "sessions.jsonl").read_text().splitlines()[0])
    sessions["session_id"] = "gone"
    (root / "sessions.jsonl").write_text(
        json.dumps(sessions, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="absent from the freshly built"):
        run_canonical_harvest(
            index_root=root, materialize_dir=tmp_path / "mat", archive_dir=archive,
            observations_path=None,
        )
    assert not (tmp_path / "mat" / "annotations.jsonl").exists()


def test_canonical_harvest_fails_closed_on_missing_materialized_outputs(
    tmp_path: Path,
) -> None:
    """Sibling fail-closed gate: a materialize_dir missing the preview manifest or
    the materialized sessions.jsonl raises FileNotFoundError before any write."""
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    # No manifest at all: _load_pin refuses.
    missing_manifest = tmp_path / "missing-manifest"
    with pytest.raises(FileNotFoundError, match="preview manifest not found"):
        run_canonical_harvest(
            index_root=root, materialize_dir=missing_manifest, archive_dir=archive,
            observations_path=None,
        )
    # Manifest present but sessions.jsonl absent: _load_materialized_records refuses.
    mat = tmp_path / "mat"
    mat.mkdir()
    (mat / "preview-manifest.json").write_text(
        json.dumps(_PIN, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="materialized preview snapshot not found"):
        run_canonical_harvest(
            index_root=root, materialize_dir=mat, archive_dir=archive,
            observations_path=None,
        )
    assert not (mat / "annotations.jsonl").exists()


def test_canonical_harvest_fails_closed_on_unreadable_materialized_outputs(
    tmp_path: Path,
) -> None:
    """Sibling fail-closed gate: an unreadable preview manifest or materialized
    sessions.jsonl raises ValueError before any write."""
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    bad_manifest = tmp_path / "bad-manifest"
    bad_manifest.mkdir()
    (bad_manifest / "preview-manifest.json").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable preview manifest"):
        run_canonical_harvest(
            index_root=root, materialize_dir=bad_manifest, archive_dir=archive,
            observations_path=None,
        )
    mat = tmp_path / "mat"
    mat.mkdir()
    (mat / "preview-manifest.json").write_text(
        json.dumps(_PIN, sort_keys=True), encoding="utf-8"
    )
    (mat / "sessions.jsonl").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable materialized snapshot"):
        run_canonical_harvest(
            index_root=root, materialize_dir=mat, archive_dir=archive,
            observations_path=None,
        )
    assert not (mat / "annotations.jsonl").exists()


def test_canonical_harvest_merges_human_observations_by_precedence(tmp_path: Path) -> None:
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    _seed_archive(archive)
    run_materialize(root, tmp_path / "mat", pin=_PIN)
    from daydream.training.corpus_v2.identity import record_id
    rid = record_id("s1", "s1-t", "s1-seg", "fp-1")
    obs = tmp_path / "observations.jsonl"
    obs.write_text(json.dumps({
        "record_id": rid, "disposition": "accepted", "evidence_digest": "d" * 32,
        "evidence": [], "labeler": "alice", "role": "rater", "rationale": "looked right",
        "valid_at": "2026-01-02T00:00:00+00:00", "observed_at": "2026-01-02T01:00:00+00:00",
        "rubric_version": "v1",
    }) + "\n", encoding="utf-8")
    out = run_canonical_harvest(
        index_root=root, materialize_dir=tmp_path / "mat", archive_dir=archive,
        observations_path=obs,
    )
    assert out["human_adjudicated"] == 1
    history = label_observation_history(archive, "s1")
    rubric = json.loads(history[0]["rubric_json"])
    stored = rubric.get("per_finding_outcomes") or rubric.get("per_finding_resolutions")
    # the human disposition wins the stored resolution (M5 precedence merge)
    assert stored[0]["disposition"] == "accepted"


def test_canonical_harvest_rejects_unknown_observation_record_id(tmp_path: Path) -> None:
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    _seed_archive(archive)
    run_materialize(root, tmp_path / "mat", pin=_PIN)
    obs = tmp_path / "observations.jsonl"
    obs.write_text(json.dumps({
        "record_id": "e" * 64, "disposition": "accepted", "evidence_digest": "d" * 32,
        "evidence": [], "labeler": "alice", "role": "rater", "rationale": "x",
        "valid_at": "2026-01-02T00:00:00+00:00", "observed_at": "2026-01-02T01:00:00+00:00",
        "rubric_version": "v1",
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="e" * 64):
        run_canonical_harvest(
            index_root=root, materialize_dir=tmp_path / "mat", archive_dir=archive,
            observations_path=obs,
        )
    assert label_observation_history(archive, "s1") == []


def test_canonical_harvest_emits_annotations_jsonl_from_merged_records(tmp_path: Path) -> None:
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    _seed_archive(archive)
    run_materialize(root, tmp_path / "mat", pin=_PIN)
    out = run_canonical_harvest(
        index_root=root, materialize_dir=tmp_path / "mat", archive_dir=archive,
        observations_path=None,
    )
    assert out["record_count"] == 1
    lines = (tmp_path / "mat" / "annotations.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    # canonical JSON, sorted by record_id, shared serializer shape
    assert [r["record_id"] for r in records] == sorted(r["record_id"] for r in records)
    assert records[0]["evidence_digest"] == "d" * 32
    assert records[0]["session_id"] == "s1"
    for line in lines:
        assert line == json.dumps(json.loads(line), sort_keys=True,
                                  separators=(",", ":"), ensure_ascii=False)


def test_canonical_harvest_flags_evidence_after_as_of(tmp_path: Path) -> None:
    """Recorded-and-flagged as_of edge policy: evidence observed after the pin's
    ``as_of`` keeps its evidence but is stamped ``evidence_after_as_of=True``
    (never gold-eligible downstream); evidence before the pin stays False."""
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    _seed_archive(archive)
    conn = _get_connection(archive)
    conn.execute(
        "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) "
        "VALUES ('s2', '2026-01-01T00:00:00+00:00', 'deep', 'archive/s2')"
    )
    conn.commit()
    conn.close()
    run_materialize(root, tmp_path / "mat", pin=_PIN)
    run_canonical_harvest(
        index_root=root, materialize_dir=tmp_path / "mat", archive_dir=archive,
        observations_path=None,
    )
    # Second index whose evidence carries a created_at after the pin
    # (2026-02-01T00:00:00+00:00).
    sessions = [{
        "session_id": "s2", "trajectory_id": "s2-t", "segment_id": "s2-seg",
        "resolutions": [{
            "fingerprint": "fp-1", "disposition": "unanswered",
            "evidence": [{"reply_id": 1, "body_sha256": "abc",
                          "created_at": "2026-03-01T00:00:00+00:00"}],
            "evidence_digest": "d" * 32, "profile": "pr_review", "stack": "python",
            "comment_id": 7,
        }],
    }]
    (root / "sessions.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions), encoding="utf-8"
    )
    run_materialize(root, tmp_path / "mat2", pin=_PIN)
    out = run_canonical_harvest(
        index_root=root, materialize_dir=tmp_path / "mat2", archive_dir=archive,
        observations_path=None,
    )
    assert out["evidence_after_as_of"] == [json.loads(
        (tmp_path / "mat2" / "annotations.jsonl").read_text().splitlines()[0]
    )["record_id"]]
    records = [json.loads(line) for line in
               (tmp_path / "mat2" / "annotations.jsonl").read_text().splitlines()]
    assert records[0]["evidence_after_as_of"] is True
    # The first (pre-pin) harvest carries a before-pin created_at
    # (2026-01-01T00:00:00+00:00 < as_of 2026-02-01T00:00:00+00:00) and stays
    # unflagged — a real timestamp comparison, not the missing-key guard.
    first = [json.loads(line) for line in
             (tmp_path / "mat" / "annotations.jsonl").read_text().splitlines()]
    assert first[0]["evidence_after_as_of"] is False


def test_canonical_harvest_changed_pin_appends_new_generation(tmp_path: Path) -> None:
    """A re-harvest under a changed pin (new as_of/rubric_version ⇒ new
    snapshot_id) appends a fresh ``label_observations`` generation instead of
    silently dedup-skipping: the archived ``rubric_json`` must carry the new
    pin's flags exactly like the rewritten ``annotations.jsonl`` (no bitemporal
    divergence between the archive row and the emitted bundle)."""
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    _seed_archive(archive)
    # Evidence observed after pin-a's as_of but before pin-b's: only the
    # second, re-pinned harvest may flag evidence_after_as_of.
    sessions = [{
        "session_id": "s1", "trajectory_id": "s1-t", "segment_id": "s1-seg",
        "resolutions": [{
            "fingerprint": "fp-1", "disposition": "unanswered",
            "evidence": [{"reply_id": 1, "body_sha256": "abc",
                          "created_at": "2026-02-15T00:00:00+00:00"}],
            "evidence_digest": "d" * 32, "profile": "pr_review", "stack": "python",
            "comment_id": 7,
        }],
    }]
    (root / "sessions.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions), encoding="utf-8"
    )
    pin_a = dict(_PIN, as_of="2026-03-01T00:00:00+00:00")  # evidence before as_of
    pin_b = dict(_PIN, as_of="2026-02-01T00:00:00+00:00", rubric_version="v2")  # after
    run_materialize(root, tmp_path / "mat-a", pin=pin_a)
    out_a = run_canonical_harvest(
        index_root=root, materialize_dir=tmp_path / "mat-a", archive_dir=archive,
        observations_path=None,
    )
    assert out_a["appended_sessions"] == 1
    assert out_a["evidence_after_as_of"] == []
    run_materialize(root, tmp_path / "mat-b", pin=pin_b)
    out_b = run_canonical_harvest(
        index_root=root, materialize_dir=tmp_path / "mat-b", archive_dir=archive,
        observations_path=None,
    )
    # Changed pin ⇒ a fresh generation, never a silent dedup skip (the
    # evidence/labels/digest are all unchanged between the two harvests).
    assert out_b["appended_sessions"] == 1
    assert out_b["skipped_sessions"] == 0
    history = label_observation_history(archive, "s1")
    assert len(history) == 2
    latest = history[-1]
    # The archived row names the exact snapshot it was harvested under.
    manifest = json.loads((tmp_path / "mat-b" / "preview-manifest.json").read_text())
    assert latest["evidence_sha"] == manifest["snapshot_id"]
    rubric = json.loads(latest["rubric_json"])
    stored = rubric.get("per_finding_outcomes") or rubric.get("per_finding_resolutions")
    assert rubric["rubric_version"] == "v2"
    assert stored[0]["evidence_after_as_of"] is True
    # The archived row and the emitted bundle agree on the new pin's flag.
    emitted = [json.loads(line) for line in
               (tmp_path / "mat-b" / "annotations.jsonl").read_text().splitlines()
               if line.strip()]
    assert emitted[0]["evidence_after_as_of"] is True
    # Exactly-once still holds for an unchanged re-run under the same pin.
    out_c = run_canonical_harvest(
        index_root=root, materialize_dir=tmp_path / "mat-b", archive_dir=archive,
        observations_path=None,
    )
    assert out_c["appended_sessions"] == 0
    assert len(label_observation_history(archive, "s1")) == 2
