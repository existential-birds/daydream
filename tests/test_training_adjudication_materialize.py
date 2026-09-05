import hashlib
import json
from pathlib import Path

import pytest

from daydream.archive.hydrate import HubUnavailableError
from daydream.training.adjudication.materialize import run_materialize


def _index(tmp_path: Path) -> Path:
    root = tmp_path / "index"
    root.mkdir()
    sessions = [{
        "session_id": "s1", "trajectory_id": "s1-t", "segment_id": "s1-seg",
        "resolutions": [{
            "fingerprint": "fp-1", "disposition": "unanswered",
            "evidence": [{"reply_id": 1, "body_sha256": "abc"}],
            "evidence_digest": "d" * 32, "profile": "pr_review", "stack": "python",
            "comment_id": 7,
        }],
    }]
    (root / "sessions.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions), encoding="utf-8"
    )
    (root / "index-revision.txt").write_text("a" * 40, encoding="utf-8")
    return root


_PIN = {
    "curation_id": "cur-1", "sanitized_hub_commit": "a" * 40,
    "source_hub_commit": "b" * 40, "archive_index_digest": "c" * 64,
    "evidence_observed_at": "2026-01-01T00:00:00+00:00",
    "as_of": "2026-02-01T00:00:00+00:00",
    "labeler_version": "v1", "rubric_version": "v1", "classifier_version": "v1",
}


def test_materialize_emits_deterministic_sessions_and_manifest(tmp_path: Path) -> None:
    root = _index(tmp_path)
    r1 = run_materialize(root, tmp_path / "out-a", pin=_PIN)
    r2 = run_materialize(root, tmp_path / "out-b", pin=_PIN)
    assert r1["snapshot_id"] == r2["snapshot_id"]  # identical inputs => identical id
    a = (tmp_path / "out-a" / "sessions.jsonl").read_bytes()
    b = (tmp_path / "out-b" / "sessions.jsonl").read_bytes()
    assert a == b  # byte-identical (C4)
    manifest = json.loads((tmp_path / "out-a" / "preview-manifest.json").read_text())
    for key in ("curation_id", "sanitized_hub_commit", "source_hub_commit",
                "archive_index_digest", "evidence_observed_at", "as_of"):
        assert manifest[key] == _PIN[key]
    assert manifest["snapshot_id"] == r1["snapshot_id"]
    record = json.loads(a.splitlines()[0])
    assert record["record_id"] and record["evidence_digest"] == "d" * 32
    assert record["disposition"] == "unanswered"


def test_materialize_never_writes_canonical_state(tmp_path: Path) -> None:
    root = _index(tmp_path)
    out = tmp_path / "out"
    run_materialize(root, out, pin=_PIN)
    # preview mode: no label_observations append, no resume-cache marker (AC 4)
    assert not (out / "harvest-resume.json").exists()
    assert not (out / "label_observations.jsonl").exists()
    assert not (root / "daydream.sqlite").exists()


def test_materialize_dry_run_validates_and_writes_nothing(tmp_path: Path) -> None:
    root = _index(tmp_path)
    out = tmp_path / "out"
    summary = run_materialize(root, out, pin=_PIN, dry_run=True)
    # dry-run validates everything: same summary a real run would produce
    full = run_materialize(root, tmp_path / "real", pin=_PIN)
    assert summary["snapshot_id"] == full["snapshot_id"]
    assert summary["index_revision"] == "a" * 40
    assert summary["record_count"] == full["record_count"]
    # ... and writes nothing: no out dir, no state side effects (AC 4)
    assert not out.exists()
    assert not (root / "daydream.sqlite").exists()


def test_materialize_missing_sessions_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(HubUnavailableError):
        run_materialize(tmp_path, tmp_path / "out", pin=_PIN)


def test_materialize_drift_yields_new_snapshot_id(tmp_path: Path) -> None:
    root = _index(tmp_path)
    r1 = run_materialize(root, tmp_path / "o1", pin=_PIN)
    # mutate evidence => digest changes => new snapshot id (AC 8)
    sessions_path = root / "sessions.jsonl"
    s = json.loads(sessions_path.read_text().splitlines()[0])
    s["resolutions"][0]["evidence"][0]["body_sha256"] = "zzz"
    s["resolutions"][0]["evidence_digest"] = "f" * 32
    sessions_path.write_text(json.dumps(s, sort_keys=True) + "\n", encoding="utf-8")
    r2 = run_materialize(root, tmp_path / "o2", pin=_PIN)
    assert r2["snapshot_id"] != r1["snapshot_id"]


def _index_all_dispositions(tmp_path: Path) -> Path:
    root = tmp_path / "index"
    root.mkdir()
    evidence = [{"reply_id": 1, "body_sha256": "abc"}]
    digest = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    dispositions = ["accepted", "rejected", "ambiguous", "unanswered", "missing"]
    resolutions = [
        {
            "fingerprint": f"fp-{i}", "disposition": d,
            "evidence": evidence, "evidence_digest": digest,
            "profile": "pr_review", "stack": "python", "comment_id": 7,
        }
        for i, d in enumerate(dispositions, start=1)
    ]
    sessions = [{
        "session_id": "s1", "trajectory_id": "s1-t", "segment_id": "s1-seg",
        "resolutions": resolutions,
    }]
    (root / "sessions.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions), encoding="utf-8"
    )
    (root / "index-revision.txt").write_text("a" * 40, encoding="utf-8")
    return root


def test_materialize_emits_every_disposition(tmp_path: Path) -> None:
    root = _index_all_dispositions(tmp_path)  # one session, five dispositions
    result = run_materialize(root, tmp_path / "out", pin=_PIN)
    assert result["record_count"] == 5
    rows = [json.loads(ln) for ln in
            (tmp_path / "out" / "sessions.jsonl").read_text().splitlines() if ln]
    assert {r["disposition"] for r in rows} == {
        "accepted", "rejected", "ambiguous", "unanswered", "missing"}
    assert len({r["record_id"] for r in rows}) == 5  # no silent dedup across classes


def _hydrated_sqlite_index(tmp_path: Path) -> Path:
    """Hydrated staging archive whose per-finding data lives ONLY in
    label_observations.rubric_json — no trajectory.json resolutions key."""
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
    conn.execute(
        "INSERT INTO label_observations (session_id, observed_at, labels, labeler_version, "
        "evidence_sha, rubric_json, has_posterior, source, labeler_policy_version) "
        "VALUES ('s1', '2026-01-02T00:00:00+00:00', '[\"finding-accepted\"]', 'v1', "
        "'e' * 64, ?, 0, 'auto', '980-rubric-r2')",
        (json.dumps(rubric),),
    )
    conn.commit()
    conn.close()
    (root / "downloads" / ("a" * 40)).mkdir(parents=True)
    return root


def test_materialize_reads_resolutions_from_sqlite_not_trajectory(tmp_path: Path) -> None:
    root = _hydrated_sqlite_index(tmp_path)
    assert not (root / "runs").exists()  # no trajectory anywhere
    summary = run_materialize(root, tmp_path / "out", pin=_PIN)
    assert summary["record_count"] == 1
    record = json.loads((tmp_path / "out" / "sessions.jsonl").read_text().splitlines()[0])
    assert record["fingerprint"] == "fp-1"
    assert record["disposition"] == "accepted"
    assert record["evidence_digest"] == "d" * 32


def _make_labels_only_rubric(root: Path) -> None:
    """Rewrite the session's winning rubric_json to the pre-#1095 labels-only
    shape (only ``per_finding_outcomes``, no ``per_finding_resolutions``) —
    exactly what the import path appends verbatim from a surviving old archive
    (``Rubric.to_dict`` before 7a2b580, ``docs`` commit 23c02d4)."""
    import sqlite3

    labels_only = json.dumps({"per_finding_outcomes": ["accepted"]})
    conn = sqlite3.connect(str(root / "index.db"))
    conn.execute("UPDATE label_observations SET rubric_json = ?", (labels_only,))
    conn.commit()
    conn.close()


def _seed_legacy_trajectory(root: Path, session_id: str = "s1") -> None:
    """Legacy hydrated trajectory: ``runs/<session_id>/trajectory.json`` with a
    ``resolutions`` key — the pre-#1095 materialization source."""
    trajectory_dir = root / "runs" / session_id
    trajectory_dir.mkdir(parents=True)
    trajectory_dir.joinpath("trajectory.json").write_text(json.dumps({
        "session_id": session_id,
        "trajectory_id": session_id,
        "segment_id": session_id,
        "resolutions": [{
            "fingerprint": "fp-1", "disposition": "accepted",
            "evidence": [{"reply_id": 1, "body_sha256": "abc"}],
            "evidence_digest": "d" * 32,
        }],
    }), encoding="utf-8")


def test_materialize_serves_legacy_labels_only_rows_from_trajectory(tmp_path: Path) -> None:
    """A pre-#1095 labels-only row (imported verbatim from a surviving old
    archive) must not brick the session: materialize serves it from the
    sanitized per-run trajectory — the pre-#1095 materialization source —
    instead of failing closed, so preview/materialize/harvest keep working
    after an import."""
    root = _hydrated_sqlite_index(tmp_path)
    _make_labels_only_rubric(root)
    _seed_legacy_trajectory(root)
    summary = run_materialize(root, tmp_path / "out", pin=_PIN)
    assert summary["record_count"] == 1
    record = json.loads((tmp_path / "out" / "sessions.jsonl").read_text().splitlines()[0])
    assert record["fingerprint"] == "fp-1"
    assert record["disposition"] == "accepted"
    assert record["evidence_digest"] == "d" * 32


def test_materialize_fails_closed_when_legacy_labels_only_rows_have_no_trajectory(
    tmp_path: Path,
) -> None:
    """A legacy labels-only session with no trajectory anywhere has no
    materializable resolutions: fail closed naming the session — never
    silently skipped, never a partial snapshot."""
    root = _hydrated_sqlite_index(tmp_path)
    _make_labels_only_rubric(root)
    with pytest.raises(Exception, match="s1"):
        run_materialize(root, tmp_path / "out", pin=_PIN)


def _hydrated_sqlite_index_agreeing_generations(tmp_path: Path) -> Path:
    """Task 3's ``_hydrated_sqlite_index`` shape, extended with a second
    generation for ``s1`` that agrees on the disposition (identical labels)
    but splits the dedup tuple on non-disposition members (evidence_sha and
    policy version) — exactly what ``append_label_observation`` produces on a
    policy-version bump or an edited-reply digest change."""
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
    for observed_at, policy_version, evidence_sha in (
        ("2026-01-02T00:00:00+00:00", "980-rubric-r1", "e" * 64),
        ("2026-01-03T00:00:00+00:00", "980-rubric-r2", "f" * 64),
    ):
        conn.execute(
            "INSERT INTO label_observations (session_id, observed_at, labels, labeler_version, "
            "evidence_sha, rubric_json, has_posterior, source, labeler_policy_version) "
            "VALUES ('s1', ?, '[\"finding-accepted\"]', 'v1', ?, ?, 0, 'auto', ?)",
            (observed_at, evidence_sha, rubric_json, policy_version),
        )
    conn.commit()
    conn.close()
    (root / "downloads" / ("a" * 40)).mkdir(parents=True)
    return root


def test_agreeing_generations_are_not_conflicting(tmp_path: Path) -> None:
    """Two generations with identical dispositions (same labels) split only on
    non-disposition dedup members (evidence_sha, policy version) are agreeing
    generations: the session must not be flagged conflicting so the next
    canonical harvest keeps projecting its decisive finding-accepted label
    instead of suppressing it."""
    root = _hydrated_sqlite_index_agreeing_generations(tmp_path)
    mat = tmp_path / "mat"
    run_materialize(root, mat, pin=_PIN)
    record = json.loads((tmp_path / "mat" / "sessions.jsonl").read_text().splitlines()[0])
    assert record.get("conflicting") is None
    assert record["disposition"] == "accepted"


def test_materialize_fails_loudly_on_uncheckpointed_wal(tmp_path: Path) -> None:
    """A crashed/interrupted writer between commit and close leaves committed
    rows in ``index.db-wal``; the ``immutable=1`` read-only adapters would
    silently skip them and serve fewer sessions with no error — the guard must
    fail loudly instead of serving a partial snapshot."""
    root = _hydrated_sqlite_index(tmp_path)
    import sqlite3
    # Simulate the crash window: a fresh writer commits rows into the WAL but
    # has not closed (checkpointed) yet.
    conn = sqlite3.connect(str(root / "index.db"))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) "
        "VALUES ('s2', '2026-01-01T00:00:00+00:00', 'deep', 'archive/s2')"
    )
    conn.commit()
    try:
        assert (root / "index.db-wal").is_file()  # committed but uncheckpointed
        with pytest.raises(HubUnavailableError, match="uncheckpointed WAL"):
            run_materialize(root, tmp_path / "out", pin=_PIN)
    finally:
        conn.close()
