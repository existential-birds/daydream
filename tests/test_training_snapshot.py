"""Tests for the corpus snapshot writer (Task 14)."""

from __future__ import annotations

import json
from pathlib import Path

from daydream.archive.index import append_label_observation, upsert_run
from daydream.archive.manifest import Manifest
from daydream.training.snapshot import SnapshotConfig, run_snapshot


def test_snapshot_writes_corpus_snapshot_json(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    upsert_run(
        archive_dir,
        Manifest(
            session_id="s-1",
            archived_at="2026-01-01T00:00:00Z",
            run_flow="normal",
            backend="claude",
            repo_slug="org/repo",
            archive_path=str(tmp_path / "run-1"),
        ),
    )
    append_label_observation(
        archive_dir,
        "s-1",
        labels=["accepted"],
        pr_state="merged",
        labeler_version="v1",
        evidence_sha=None,
    )

    out = tmp_path / "corpus-snapshot.json"
    config = SnapshotConfig(archive_dir=archive_dir, out_path=out)
    summary = run_snapshot(config)

    snapshot = json.loads(out.read_text())
    assert snapshot["archive_index_sha"]
    assert snapshot["as_of_ts"]
    assert snapshot["label_counts"]["accepted"] == 1
    assert snapshot["total_runs"] == 1
    assert summary["total_runs"] == 1


def test_snapshot_label_counts_aggregate_by_class(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    for idx, label in enumerate(["accepted", "accepted", "rejected", "contested", None]):
        sess = f"s-{idx}"
        upsert_run(
            archive_dir,
            Manifest(
                session_id=sess,
                archived_at="2026-01-01T00:00:00Z",
                run_flow="normal",
                backend="claude",
                archive_path=str(tmp_path / sess),
            ),
        )
        if label is not None:
            append_label_observation(
                archive_dir,
                sess,
                labels=[label],
                pr_state="merged",
                labeler_version="v1",
                evidence_sha=None,
            )

    out = tmp_path / "snap.json"
    run_snapshot(SnapshotConfig(archive_dir=archive_dir, out_path=out))

    snapshot = json.loads(out.read_text())
    assert snapshot["label_counts"] == {"accepted": 2, "rejected": 1, "contested": 1, "unlabeled": 1}
    assert snapshot["total_runs"] == 5
