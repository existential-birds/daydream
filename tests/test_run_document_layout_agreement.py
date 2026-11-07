"""#1220: one run, every reader, the same documents; #1714: one document at a time.

The run tree is written directly (not through the layout surface) so this test
characterizes what the readers do today and keeps doing after the refactor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

import scripts.measure_finalization_memory as measure
from daydream.archive import get_archive_dir
from daydream.archive.bundle import _project_documents
from daydream.eval.analyzer import _latest_main_trajectory, collect_trajectory_paths
from daydream.trajectory import (
    RunWriteSnapshot,
    TrajectoryDocumentReference,
    TrajectoryDocumentSnapshot,
    default_trajectory_path,
    partial_document_path,
    summarize_run,
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
    """A prepared root plus a completed child referenced at a differently named private (frozen) copy.

    Both logical paths carry the ``.partial`` suffix that every reader strips."""
    private_child = root_path.parents[3] / "frozen-copy.json"
    private_child.write_bytes(trajectory_payload("fork-1"))
    return RunWriteSnapshot(status=status, cutoff_at="2026-01-01T00:00:00Z", root_trajectory_id=SESSION,
        documents=(TrajectoryDocumentSnapshot(SESSION, partial_document_path(root_path), trajectory_payload(SESSION)),
            TrajectoryDocumentReference.written(
                "fork-1", partial_document_path(root_path.parent / "trajectories" / SIBLING), private_child,
                trajectory_payload("fork-1"),
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
    # 2. the frozen run summary agrees with the analyzer on the same names and retains no step bodies
    summary = summarize_run(_snapshot(run_dir / "trajectory.json"))
    assert [(d.source_file, "steps" in d.metadata) for d in summary.documents] == [
        ("trajectory.json", False), (SIBLING, False),
    ]

    # 3. the archive projection writes the same relative names under its own root
    archive_run_dir = get_archive_dir() / "runs" / SESSION
    archive_run_dir.mkdir(parents=True, exist_ok=True)
    _project_documents(_snapshot(run_dir / "trajectory.json"), archive_run_dir, session_id=SESSION)
    assert {p.relative_to(archive_run_dir).as_posix() for p in archive_run_dir.rglob("*.json")} == canonical


def test_every_finalization_reader_holds_one_document_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live and CLI run info, archive with evaluation, and capture keep a flat heap peak as forks are added.

    A reader that retains every document, such as a list where a one-at-a-time pass belongs,
    grows by about one decoded document per added fork and fails this bound.
    """
    peaks = []
    for forks in (2, 8):  # The smaller run goes first, so one-time warm-up cannot inflate the larger one.
        monkeypatch.setenv("DAYDREAM_ARCHIVE_DIR", str(tmp_path / f"archive-{forks}"))
        peaks.append(measure._scenario(forks, 32, 2, 1))
    few, many = peaks
    for stage in ("runinfo", "summarize", "archive", "capture"):
        assert many[f"{stage}_peak"] - few[f"{stage}_peak"] < few["largest_mib"], (stage, few, many)
