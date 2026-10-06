import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from daydream.archive.hydrate import HubUnavailableError
from daydream.archive.index import append_label_observation
from daydream.training.adjudication.materialize import _trajectory_resolutions_readonly, run_materialize
from daydream.trajectory import run_directory, run_document_path
from tests.harness.adjudication import make_hydrated_sqlite_index, write_sessions_jsonl
from tests.test_training_adjudication_canonical import _PIN, _index


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
    assert not (out / "harvest-resume.json").exists()
    assert not (out / "label_observations.jsonl").exists()
    assert not (root / "daydream.sqlite").exists()

def test_materialize_dry_run_validates_and_writes_nothing(tmp_path: Path) -> None:
    root = _index(tmp_path)
    out = tmp_path / "out"
    summary = run_materialize(root, out, pin=_PIN, dry_run=True)
    full = run_materialize(root, tmp_path / "real", pin=_PIN)
    assert summary["snapshot_id"] == full["snapshot_id"]
    assert summary["index_revision"] == "a" * 40
    assert summary["record_count"] == full["record_count"]
    assert not out.exists()
    assert not (root / "daydream.sqlite").exists()

def test_materialize_missing_sessions_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(HubUnavailableError):
        run_materialize(tmp_path, tmp_path / "out", pin=_PIN)

def test_materialize_drift_yields_new_snapshot_id(tmp_path: Path) -> None:
    root = _index(tmp_path)
    r1 = run_materialize(root, tmp_path / "o1", pin=_PIN)
    sessions_path = root / "sessions.jsonl"
    s = json.loads(sessions_path.read_text().splitlines()[0])
    s["resolutions"][0]["evidence"][0]["body_sha256"] = "zzz"
    s["resolutions"][0]["evidence_digest"] = "f" * 32
    sessions_path.write_text(json.dumps(s, sort_keys=True) + "\n", encoding="utf-8")
    r2 = run_materialize(root, tmp_path / "o2", pin=_PIN)
    assert r2["snapshot_id"] != r1["snapshot_id"]


def _index_all_dispositions(tmp_path: Path) -> Path:
    root = tmp_path / "index"
    evidence = [{"reply_id": 1, "body_sha256": "abc"}]
    digest = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    dispositions = ["accepted", "rejected", "ambiguous", "unanswered", "missing"]
    resolutions = [{"fingerprint": f"fp-{i}", "disposition": d, "evidence": evidence, "evidence_digest": digest,
            "profile": "pr_review", "stack": "python", "comment_id": 7,
        }
        for i, d in enumerate(dispositions, start=1)
    ]
    sessions = [{"session_id": "s1", "trajectory_id": "s1-t", "segment_id": "s1-seg", "resolutions": resolutions}]
    write_sessions_jsonl(root, sessions)
    (root / "index-revision.txt").write_text("a" * 40, encoding="utf-8")
    return root


def test_materialize_emits_every_disposition(tmp_path: Path) -> None:
    root = _index_all_dispositions(tmp_path)  # one session, five dispositions
    result = run_materialize(root, tmp_path / "out", pin=_PIN)
    assert result["record_count"] == 5
    rows = [json.loads(ln) for ln in (tmp_path / "out" / "sessions.jsonl").read_text().splitlines() if ln]
    assert {r["disposition"] for r in rows} == {"accepted", "rejected", "ambiguous", "unanswered", "missing"}
    assert len({r["record_id"] for r in rows}) == 5  # no silent dedup across classes


def _hydrated_sqlite_index(tmp_path: Path) -> Path:
    """Hydrated staging archive whose per-finding data lives ONLY in
    label_observations.rubric_json — no trajectory.json resolutions key."""
    return make_hydrated_sqlite_index(
        tmp_path, [("2026-01-02T00:00:00+00:00", '["finding-accepted"]', "e" * 64, "980-rubric-r2", "accepted")],
    )


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
    """Model a legacy imported rubric with per_finding_outcomes but no per_finding_resolutions."""

    labels_only = json.dumps({"per_finding_outcomes": ["accepted"]})
    conn = sqlite3.connect(str(root / "index.db"))
    conn.execute("UPDATE label_observations SET rubric_json = ?", (labels_only,))
    conn.commit()
    conn.close()


def _seed_legacy_trajectory(root: Path, session_id: str = "s1") -> None:
    """Stage legacy trajectory resolutions through the same run-document layout surface as production
    readers.
    """
    trajectory_path = run_document_path(run_directory(root, session_id))
    trajectory_path.parent.mkdir(parents=True)
    trajectory_path.write_text(json.dumps({
        "session_id": session_id, "trajectory_id": session_id, "segment_id": session_id, "resolutions": [{
            "fingerprint": "fp-1", "disposition": "accepted", "evidence": [{"reply_id": 1, "body_sha256": "abc"}],
            "evidence_digest": "d" * 32,
        }],
    }), encoding="utf-8")


def test_materialize_serves_legacy_labels_only_rows_from_trajectory(tmp_path: Path) -> None:
    """Legacy labels-only imports recover resolutions from the sanitized trajectory so
    preview/materialize/harvest remain usable.
    """
    root = _hydrated_sqlite_index(tmp_path)
    _make_labels_only_rubric(root)
    _seed_legacy_trajectory(root)
    summary = run_materialize(root, tmp_path / "out", pin=_PIN)
    assert summary["record_count"] == 1
    record = json.loads((tmp_path / "out" / "sessions.jsonl").read_text().splitlines()[0])
    assert record["fingerprint"] == "fp-1"
    assert record["disposition"] == "accepted"
    assert record["evidence_digest"] == "d" * 32

def test_hydrated_readers_address_the_layout_run_directory(tmp_path: Path) -> None:

    root = _hydrated_sqlite_index(tmp_path)
    _make_labels_only_rubric(root)
    _seed_legacy_trajectory(root, "s1")

    assert run_document_path(run_directory(root, "s1")).is_file()
    assert _trajectory_resolutions_readonly(root, "s1") is not None

def test_materialize_skips_legacy_labels_only_sessions_without_trajectory(tmp_path: Path,) -> None:
    """A labels-only row without a trajectory is evidence-only and yields no resolutions; it must not abort
    the rest of the curation.
    """
    root = _hydrated_sqlite_index(tmp_path)
    _make_labels_only_rubric(root)
    summary = run_materialize(root, tmp_path / "out", pin=_PIN)
    assert summary["record_count"] == 0
    lines = [ln for ln in (tmp_path / "out" / "sessions.jsonl").read_text().splitlines() if ln]
    assert lines == []  # the session is served nothing, never fabricated


def _hydrated_sqlite_index_agreeing_generations(tmp_path: Path) -> Path:
    """Agreeing generations differ only in policy version and evidence digest, as after a policy bump or
    reply edit.
    """
    return make_hydrated_sqlite_index(tmp_path,
        [("2026-01-02T00:00:00+00:00", '["finding-accepted"]', "e" * 64, "980-rubric-r1", "accepted"),
            ("2026-01-03T00:00:00+00:00", '["finding-accepted"]', "f" * 64, "980-rubric-r2", "accepted"),
        ],
    )


def test_materialize_serves_human_labeled_session_from_trajectory(tmp_path: Path) -> None:
    """A winning human row with NULL rubric recovers trajectory resolutions and never creates a session
    conflict.
    """
    root = _hydrated_sqlite_index(tmp_path)
    _seed_legacy_trajectory(root)

    append_label_observation(
        root, "s1", labels=["finding-rejected"], pr_state=None, labeler_version="human", evidence_sha=None,
        source="human", observed_at="2026-01-04T00:00:00+00:00",
    )
    summary = run_materialize(root, tmp_path / "out", pin=_PIN)
    assert summary["record_count"] == 1
    record = json.loads((tmp_path / "out" / "sessions.jsonl").read_text().splitlines()[0])
    assert record["disposition"] == "accepted"  # served from the trajectory
    assert record.get("conflicting") is None  # human override is not a disagreement

def test_agreeing_generations_are_not_conflicting(tmp_path: Path) -> None:
    """Matching dispositions remain non-conflicting across policy/evidence generations and retain decisive
    labels.
    """
    root = _hydrated_sqlite_index_agreeing_generations(tmp_path)
    mat = tmp_path / "mat"
    run_materialize(root, mat, pin=_PIN)
    record = json.loads((tmp_path / "mat" / "sessions.jsonl").read_text().splitlines()[0])
    assert record.get("conflicting") is None
    assert record["disposition"] == "accepted"


def _hydrated_sqlite_index_evolving(tmp_path: Path) -> Path:
    """A later accepted generation resolves an earlier unanswered one without creating a disagreement."""
    return make_hydrated_sqlite_index(tmp_path,
        [("2026-01-02T00:00:00+00:00", '["finding-unanswered"]', "e" * 64, "980-rubric-r2", "unanswered"),
            ("2026-01-03T00:00:00+00:00", '["finding-accepted"]', "f" * 64, "980-rubric-r2", "accepted"),
        ],
    )


def test_resolved_unanswered_to_accepted_evolution_is_not_conflicting(tmp_path: Path,) -> None:
    """An unanswered-to-decisive evolution remains gold-eligible after materialization."""
    root = _hydrated_sqlite_index_evolving(tmp_path)
    mat = tmp_path / "mat"
    run_materialize(root, mat, pin=_PIN)
    record = json.loads((tmp_path / "mat" / "sessions.jsonl").read_text().splitlines()[0])
    assert record.get("conflicting") is None
    assert record["disposition"] == "accepted"

def test_materialize_fails_loudly_on_uncheckpointed_wal(tmp_path: Path) -> None:
    """An uncheckpointed WAL contains committed rows that immutable SQLite reads would miss; refuse rather
    than serve a partial snapshot.
    """
    root = _hydrated_sqlite_index(tmp_path)
    # Leave a committed writer open in WAL mode to simulate a crash.
    conn = sqlite3.connect(str(root / "index.db"))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("INSERT INTO runs (session_id, archived_at, run_flow, archive_path) "
        "VALUES ('s2', '2026-01-01T00:00:00+00:00', 'deep', 'archive/s2')"
    )
    conn.commit()
    try:
        assert (root / "index.db-wal").is_file()  # committed but uncheckpointed
        with pytest.raises(HubUnavailableError, match="uncheckpointed WAL"):
            run_materialize(root, tmp_path / "out", pin=_PIN)
    finally:
        conn.close()
