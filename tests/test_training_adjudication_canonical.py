"""Canonical harvest tests (issue #1055, Task 4).

Drift gate fail-closed pre-write, three-tier precedence merge, and the
exactly-once ``label_observations`` append (M5/M8).
"""

import hashlib
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


def _seed_decisive_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Index + archive + materialize-dir fixture whose index carries one
    ``accepted``, one ``rejected``, and one ``unanswered`` finding — the complete
    disposition set the widened materialize emits and the drift gate must
    re-derive via ``build_queue(..., include_decisive=True)``."""
    root = tmp_path / "index"
    root.mkdir(exist_ok=True)
    resolutions = [
        {
            "fingerprint": f"fp-{n}", "disposition": disposition,
            "evidence": [{"reply_id": n, "body_sha256": "abc",
                          "created_at": "2026-01-01T00:00:00+00:00"}],
            "evidence_digest": "d" * 32, "profile": "pr_review", "stack": "python",
            "comment_id": 7,
        }
        for n, disposition in enumerate(("accepted", "rejected", "unanswered"), start=1)
    ]
    sessions = [
        {
            "session_id": f"s{n}", "trajectory_id": f"s{n}-t", "segment_id": f"s{n}-seg",
            "resolutions": [resolution],
        }
        for n, resolution in enumerate(resolutions, start=1)
    ]
    (root / "sessions.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions), encoding="utf-8"
    )
    (root / "index-revision.txt").write_text("a" * 40, encoding="utf-8")
    archive = tmp_path / "archive"
    _seed_archive(archive)
    conn = _get_connection(archive)
    for n in (2, 3):
        conn.execute(
            "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) "
            f"VALUES ('s{n}', '2026-01-01T00:00:00+00:00', 'deep', 'archive/s{n}')"
        )
    conn.commit()
    conn.close()
    mat = tmp_path / "mat"
    run_materialize(root, mat, pin=_PIN)
    return root, archive, mat


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
    # The archived row names the exact generation it was harvested under: a
    # digest over the content-addressed snapshot_id plus the archived
    # rubric_json (the dedup tuple omits rubric_json, M14), so the digest
    # also proves the archive and the emitted bundle agree on the pin/flags.
    manifest = json.loads((tmp_path / "mat-b" / "preview-manifest.json").read_text())
    assert latest["evidence_sha"] == hashlib.sha256(
        (manifest["snapshot_id"] + ":" + latest["rubric_json"]).encode("utf-8")
    ).hexdigest()
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


def test_canonical_harvest_label_preserving_overlay_change_skips_nothing(
    tmp_path: Path,
) -> None:
    """Unchanged pin + a label-preserving observation-overlay edit (the
    disposition set stays put) still appends a fresh generation: the dedup
    tuple omits ``rubric_json``, so the rubric-content digest riding on
    ``evidence_sha`` is what prevents the archived rubric from going stale
    while ``annotations.jsonl`` is re-emitted from the new overlay (M14)."""
    root = _index(tmp_path)
    archive = tmp_path / "archive"
    _seed_archive(archive)
    run_materialize(root, tmp_path / "mat", pin=_PIN)
    from daydream.training.corpus_v2.identity import record_id

    rid = record_id("s1", "s1-t", "s1-seg", "fp-1")
    obs = tmp_path / "observations.jsonl"
    obs.write_text(json.dumps({
        "record_id": rid, "disposition": "accepted", "evidence_digest": "d" * 32,
        "evidence": [], "labeler": "alice", "role": "rater", "rationale": "first pass",
        "valid_at": "2026-01-02T00:00:00+00:00",
        "observed_at": "2026-01-02T01:00:00+00:00", "rubric_version": "v1",
    }) + "\n", encoding="utf-8")
    out1 = run_canonical_harvest(
        index_root=root, materialize_dir=tmp_path / "mat", archive_dir=archive,
        observations_path=obs,
    )
    assert out1["appended_sessions"] == 1
    # Same pin, same materialized snapshot, but a different human labeler
    # re-affirms the same decisive disposition: the archived labels set is
    # unchanged, so only the rubric-content digest can tell the generations
    # apart.
    obs.write_text(json.dumps({
        "record_id": rid, "disposition": "accepted", "evidence_digest": "d" * 32,
        "evidence": [], "labeler": "bob", "role": "adjudicator", "rationale": "second pass",
        "valid_at": "2026-01-02T00:00:00+00:00",
        "observed_at": "2026-01-02T02:00:00+00:00", "rubric_version": "v1",
    }) + "\n", encoding="utf-8")
    out2 = run_canonical_harvest(
        index_root=root, materialize_dir=tmp_path / "mat", archive_dir=archive,
        observations_path=obs,
    )
    assert out2["appended_sessions"] == 1  # fresh generation, never a silent skip
    history = label_observation_history(archive, "s1")
    assert len(history) == 2
    rubric = json.loads(history[-1]["rubric_json"])
    stored = rubric.get("per_finding_outcomes") or rubric.get("per_finding_resolutions")
    assert stored[0]["human_labeler"] == "bob"
    # The archived rubric matches the emitted bundle's overlay.
    emitted = [json.loads(line) for line in
               (tmp_path / "mat" / "annotations.jsonl").read_text().splitlines()
               if line.strip()]
    assert emitted[0]["human_labeler"] == "bob"
    # Unchanged everything (pin + rubric) stays exactly-once.
    out3 = run_canonical_harvest(
        index_root=root, materialize_dir=tmp_path / "mat", archive_dir=archive,
        observations_path=obs,
    )
    assert out3["appended_sessions"] == 0
    assert len(label_observation_history(archive, "s1")) == 2


def test_canonical_harvest_complete_set_is_idempotent_and_exactly_once(
    tmp_path: Path,
) -> None:
    """The drift gate re-derives the *complete* record set (decisive included)
    via ``build_queue(..., include_decisive=True)``: harvesting a widened
    materialization succeeds, preserves unchanged automatic decisive
    dispositions verbatim (no observation group ⇒ untouched), and an identical
    re-run is byte-identical with no new generation (exactly-once)."""
    stage, archive, mat = _seed_decisive_fixture(tmp_path)
    run_canonical_harvest(stage, mat, archive, observations_path=None)
    first = (mat / "annotations.jsonl").read_bytes()
    dispositions = [json.loads(ln)["disposition"] for ln in
                    first.splitlines() if ln]
    assert sorted(dispositions) == ["accepted", "rejected", "unanswered"]

    # Re-run with identical inputs: identical annotations.jsonl, no new generation
    summary2 = run_canonical_harvest(stage, mat, archive, observations_path=None)
    assert (mat / "annotations.jsonl").read_bytes() == first
    assert summary2["appended_sessions"] == 0
    assert summary2["skipped_sessions"] == 3


def _hydrated_sqlite_index_with_conflict(tmp_path: Path) -> Path:
    """Task 3's ``_hydrated_sqlite_index`` shape, extended with a second,
    distinct-dedup-key ``label_observations`` row for the same session — two
    harvester generations disagreeing (labels differ ⇒ dedup keys differ) on
    one ``s1``. The latest-observed auto row wins; the session is conflicting."""
    from daydream.archive.index import _get_connection

    root = tmp_path / "hydrated"
    conn = _get_connection(root)
    conn.execute(
        "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) "
        "VALUES ('s1', '2026-01-01T00:00:00+00:00', 'deep', 'archive/s1')"
    )
    rubric = {"posterior_source": "pr_review",
              "per_finding_resolutions": [{
                  "fingerprint": "fp-1", "comment_id": 7, "disposition": "accepted",
                  "evidence": [{"reply_id": 1, "body_sha256": "abc"}],
                  "evidence_digest": "d" * 32}]}
    rubric_json = json.dumps(rubric)
    for observed_at, labels, evidence_sha in (
        ("2026-01-02T00:00:00+00:00", '["finding-accepted"]', "e" * 64),
        ("2026-01-03T00:00:00+00:00", '["finding-rejected"]', "f" * 64),
    ):
        conn.execute(
            "INSERT INTO label_observations (session_id, observed_at, labels, labeler_version, "
            "evidence_sha, rubric_json, has_posterior, source, labeler_policy_version) "
            "VALUES ('s1', ?, ?, 'v1', ?, ?, 0, 'auto', '980-rubric-r2')",
            (observed_at, labels, evidence_sha, rubric_json),
        )
    conn.commit()
    conn.close()
    (root / "downloads" / ("a" * 40)).mkdir(parents=True)
    return root


def test_conflicted_session_yields_no_decisive_label(tmp_path: Path) -> None:
    """Two distinct harvester dedup keys on one session = conflicting
    observations: the winner's resolutions still materialize, but the
    canonical harvest must not emit a decisive finding label for it."""
    root = _hydrated_sqlite_index_with_conflict(tmp_path)  # two distinct dedup keys, s1
    mat = tmp_path / "mat"
    run_materialize(root, mat, pin=_PIN)
    record = json.loads((tmp_path / "mat" / "sessions.jsonl").read_text().splitlines()[0])
    assert record["conflicting"] is True  # surfaced, never silently merged
    # The conflicted disposition is neutralized so the operator queue routes
    # the finding to task-only adjudication and the bundle carries one
    # disposition (issue #336 item 7) -- the archive rubric_json restores the
    # real decisive disposition for provenance in the harvest below.
    assert record["disposition"] == "ambiguous"
    archive = tmp_path / "archive"
    _seed_archive(archive)
    out = run_canonical_harvest(
        index_root=root, materialize_dir=mat, archive_dir=archive,
    )
    assert out["appended_sessions"] == 1
    history = label_observation_history(archive, "s1")
    rubric = json.loads(history[0]["rubric_json"])
    assert rubric["per_finding_resolutions"][0]["conflicting"] is True
    # and no decisive label was projected from the conflicted disposition
    assert "finding-accepted" not in history[0]["labels"]


def test_canonical_harvest_re_derives_conflict_after_materialize(tmp_path: Path) -> None:
    """A session that becomes conflicting *after* materialize (runbook step-3b
    import appends a disagreeing generation to the same index.db) passes the
    evidence-digest drift gate but must not emit decisive labels: the conflict
    verdict is re-derived from the fresh sessions at harvest time, never
    trusted from the materialized snapshot's flags (issue #336 item 2)."""
    from daydream.archive.index import _get_connection

    root = tmp_path / "hydrated"
    conn = _get_connection(root)
    conn.execute(
        "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) "
        "VALUES ('s1', '2026-01-01T00:00:00+00:00', 'deep', 'archive/s1')"
    )
    rubric = {"posterior_source": "pr_review",
              "per_finding_resolutions": [{
                  "fingerprint": "fp-1", "comment_id": 7, "disposition": "accepted",
                  "evidence": [{"reply_id": 1, "body_sha256": "abc"}],
                  "evidence_digest": "d" * 32}]}
    rubric_json = json.dumps(rubric)
    conn.execute(
        "INSERT INTO label_observations (session_id, observed_at, labels, labeler_version, "
        "evidence_sha, rubric_json, has_posterior, source, labeler_policy_version) "
        "VALUES ('s1', '2026-01-02T00:00:00+00:00', '[\"finding-accepted\"]', 'v1', "
        "'e' * 64, ?, 0, 'auto', '980-rubric-r2')",
        (rubric_json,),
    )
    conn.commit()
    conn.close()
    (root / "downloads" / ("a" * 40)).mkdir(parents=True)
    mat = tmp_path / "mat"
    run_materialize(root, mat, pin=_PIN)
    record = json.loads((tmp_path / "mat" / "sessions.jsonl").read_text().splitlines()[0])
    assert record.get("conflicting") is None  # not conflicting at materialize time
    # Step-3b import: an older disagreeing generation lands in the same
    # index.db -- the winning row (and its evidence digest) is unchanged.
    conn = _get_connection(root)
    conn.execute(
        "INSERT INTO label_observations (session_id, observed_at, labels, labeler_version, "
        "evidence_sha, rubric_json, has_posterior, source, labeler_policy_version) "
        "VALUES ('s1', '2026-01-01T00:00:00+00:00', '[\"finding-rejected\"]', 'v1', "
        "'d' * 64, ?, 0, 'auto', '980-rubric-r2')",
        (rubric_json,),
    )
    conn.commit()
    conn.close()
    archive = tmp_path / "archive"
    _seed_archive(archive)
    out = run_canonical_harvest(index_root=root, materialize_dir=mat, archive_dir=archive)
    assert out["appended_sessions"] == 1
    history = label_observation_history(archive, "s1")
    # The freshly re-derived conflict suppressed the decisive label even
    # though the materialized snapshot had no conflicting flag.
    assert "finding-accepted" not in history[0]["labels"]
    rows = [
        json.loads(line)
        for line in (mat / "annotations.jsonl").read_text().splitlines()
        if line
    ]
    assert rows[0]["disposition"] == "ambiguous"
    assert rows[0]["conflicting"] is True


def test_canonical_harvest_human_resolution_clears_session_conflict(
    tmp_path: Path,
) -> None:
    """A decisive human adjudication on a conflicted finding resolves the
    conflict: the precedence merge clears the ``conflicting`` flag so the
    resolution is not suppressed to non-gold, and the archive row projects the
    decisive label (issue #336 item 7 -- a human override is never ignored)."""
    from daydream.training.corpus_v2.identity import record_id

    root = _hydrated_sqlite_index_with_conflict(tmp_path)  # two distinct dedup keys, s1
    mat = tmp_path / "mat"
    run_materialize(root, mat, pin=_PIN)
    archive = tmp_path / "archive"
    _seed_archive(archive)
    rid = record_id("s1", "s1", "s1", "fp-1")
    obs = tmp_path / "observations.jsonl"
    obs.write_text(json.dumps({
        "record_id": rid, "disposition": "accepted", "evidence_digest": "d" * 32,
        "evidence": [], "labeler": "alice", "role": "adjudicator",
        "rationale": "operator resolved the disagreeing generations",
        "valid_at": "2026-01-02T00:00:00+00:00",
        "observed_at": "2026-01-02T01:00:00+00:00", "rubric_version": "v1",
    }) + "\n", encoding="utf-8")
    out = run_canonical_harvest(
        index_root=root, materialize_dir=mat, archive_dir=archive,
        observations_path=obs,
    )
    assert out["human_adjudicated"] == 1
    history = label_observation_history(archive, "s1")
    rubric = json.loads(history[0]["rubric_json"])
    record = rubric["per_finding_resolutions"][0]
    assert record.get("conflicting") is not True  # cleared by the human resolution
    assert record["disposition"] == "accepted"
    assert "finding-accepted" in history[0]["labels"]
    rows = [
        json.loads(line)
        for line in (mat / "annotations.jsonl").read_text().splitlines()
        if line
    ]
    assert rows[0]["disposition"] == "accepted"
    assert rows[0].get("conflicting") is not True


def test_conflicted_session_never_projects_gold(tmp_path: Path) -> None:
    """The non-gold guarantee extends to the corpus-v2 projection: a
    conflicted session's annotations.jsonl row (full record, ``conflicting``
    flag intact) must never classify gold with ``outcome_label`` set even
    with a decisive disposition + evidence -- the same gate canonical.py
    applies to the archive labels column, enforced where ``classify_tier``
    flows into the projected record."""
    from daydream.training.corpus_v2.projector import project_findings

    root = _hydrated_sqlite_index_with_conflict(tmp_path)  # two distinct dedup keys, s1
    mat = tmp_path / "mat"
    run_materialize(root, mat, pin=_PIN)
    archive = tmp_path / "archive"
    _seed_archive(archive)
    run_canonical_harvest(index_root=root, materialize_dir=mat, archive_dir=archive)
    rows = [
        json.loads(line)
        for line in (mat / "annotations.jsonl").read_text().splitlines()
        if line
    ]
    # the flag rides through the harvest into the projection input verbatim
    assert any(row.get("conflicting") is True for row in rows)
    # run_build_corpus_v2's snapshot assembly (session-scoped resolutions) --
    # the boundary classify_tier reaches the corpus-v2 gold label through.
    session = {
        "session_id": "s1",
        "trajectory_id": "s1",
        "segment_id": "s1",
        "resolutions": [row for row in rows if row.get("session_id") == "s1"],
    }
    records = project_findings(session)
    assert records  # the finding is still projected -- provenance preserved
    assert all(r["tier"] != "gold" for r in records)
    assert all(r["outcome_label"] is None for r in records)
