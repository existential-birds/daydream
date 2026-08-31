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
