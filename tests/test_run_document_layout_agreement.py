"""#1220: one run, every reader, the same documents.

The run tree is written directly (not through the layout surface) so this test
characterizes what the readers do today and keeps doing after the refactor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from daydream.archive import get_archive_dir
from daydream.archive.bundle import _project_documents
from daydream.eval.analyzer import _latest_main_trajectory, collect_trajectory_paths
from daydream.trajectory import (
    RunWriteSnapshot,
    TrajectoryDocumentSnapshot,
    default_trajectory_path,
    snapshot_trajectories,
)
from tests.harness.trajectory import trajectory_payload

SESSION = "11111111-2222-3333-4444-555555555555"
SIBLING = "deep-python.json"


def _write_run(target: Path) -> Path:
    root_document = default_trajectory_path(target, SESSION)
    root_document.parent.mkdir(parents=True, exist_ok=True)
    root_document.write_bytes(trajectory_payload(SESSION))
    sibling = root_document.parent / "trajectories" / SIBLING
    sibling.parent.mkdir(parents=True, exist_ok=True)
    sibling.write_bytes(trajectory_payload("fork-1"))
    return root_document.parent


def _snapshot(root_path: Path, status: Literal["complete", "partial"] = "complete") -> RunWriteSnapshot:
    return RunWriteSnapshot(status=status, cutoff_at="2026-01-01T00:00:00Z", root_trajectory_id=SESSION,
        documents=(TrajectoryDocumentSnapshot(SESSION, root_path, trajectory_payload(SESSION)),
            TrajectoryDocumentSnapshot(
                "fork-1", root_path.parent / "trajectories" / SIBLING, trajectory_payload("fork-1")
            ),
        ),
    )

def test_every_reader_resolves_the_same_document_set(tmp_path: Path) -> None:
    target = tmp_path / "proj"
    run_dir = _write_run(target)
    daydream_dir = target / ".daydream"
    canonical = {"trajectory.json", f"trajectories/{SIBLING}"}

    # 1. the analyzer, by run dir and by latest-run fallback
    assert {p.relative_to(run_dir).as_posix() for p in collect_trajectory_paths(run_dir)} == canonical
    assert {p.relative_to(run_dir).as_posix() for p in collect_trajectory_paths(daydream_dir)} == canonical
    assert _latest_main_trajectory(daydream_dir) == run_dir / "trajectory.json"
    # 2. the producer's frozen projection agrees with the analyzer on the same names
    frozen = snapshot_trajectories(_snapshot(run_dir / "trajectory.json"))
    assert frozen["main"]["_source_file"] == "trajectory.json"
    assert [d["_source_file"] for d in frozen["forked"]] == [SIBLING]

    # 3. the archive projection writes the same relative names under its own root
    archive_run_dir = get_archive_dir() / "runs" / SESSION
    archive_run_dir.mkdir(parents=True, exist_ok=True)
    _project_documents(_snapshot(run_dir / "trajectory.json"), archive_run_dir, session_id=SESSION)
    assert {p.relative_to(archive_run_dir).as_posix() for p in archive_run_dir.rglob("*.json")} == canonical

def test_the_partial_root_is_stripped_by_the_archive_and_ignored_by_readers(tmp_path: Path) -> None:
    target = tmp_path / "proj"
    run_dir = _write_run(target)
    (run_dir / "trajectory.json").unlink()
    partial = run_dir / "trajectory.json.partial"
    partial.write_bytes(trajectory_payload(SESSION))

    # readers stay partial-blind: no main document, sibling still resolved
    assert [p.name for p in collect_trajectory_paths(run_dir)] == [SIBLING]

    # the archive strips the suffix at its destination and keeps the frozen bytes
    archive_run_dir = get_archive_dir() / "runs" / SESSION
    archive_run_dir.mkdir(parents=True, exist_ok=True)
    _project_documents(_snapshot(partial, status="partial"), archive_run_dir, session_id=SESSION)
    assert (archive_run_dir / "trajectory.json").read_bytes() == trajectory_payload(SESSION)
