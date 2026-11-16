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
    DOCUMENT_SUFFIX,
    RUN_DOCUMENT_NAME,
    SIBLINGS_DIRNAME,
    RunWriteSnapshot,
    TrajectoryDocumentReference,
    default_trajectory_path,
    summarize_run,
)
from tests.harness.trajectory import evidence_document

SESSION = "11111111-2222-3333-4444-555555555555"
SIBLING = f"deep-python{DOCUMENT_SUFFIX}"


def _write_run(target: Path) -> Path:
    root_document = default_trajectory_path(target, SESSION)
    root_document.parent.mkdir(parents=True, exist_ok=True)
    root_document.write_bytes(evidence_document(SESSION, session_id=SESSION))
    sibling = root_document.parent / SIBLINGS_DIRNAME / SIBLING
    sibling.parent.mkdir(parents=True, exist_ok=True)
    sibling.write_bytes(evidence_document("fork-1", session_id=SESSION))
    return root_document.parent


def _snapshot(root_path: Path, status: Literal["complete", "partial"] = "complete") -> RunWriteSnapshot:
    """The written root plus a completed child referenced at a differently named private (frozen) copy."""
    private_child = root_path.parents[3] / f"frozen-copy{DOCUMENT_SUFFIX}"
    child = evidence_document("fork-1", session_id=SESSION)
    private_child.write_bytes(child)
    return RunWriteSnapshot(status=status, cutoff_at="2026-01-01T00:00:00Z", root_trajectory_id=SESSION,
        documents=(TrajectoryDocumentReference.written(SESSION, root_path, root_path, root_path.read_bytes()),
            TrajectoryDocumentReference.written(
                "fork-1", root_path.parent / SIBLINGS_DIRNAME / SIBLING, private_child, child,
            ),
        ),
    )

def test_every_reader_resolves_the_same_document_set(tmp_path: Path) -> None:
    target = tmp_path / "proj"
    run_dir = _write_run(target)
    daydream_dir = target / ".daydream"
    canonical = {RUN_DOCUMENT_NAME, f"{SIBLINGS_DIRNAME}/{SIBLING}"}

    # 1. the analyzer, by run dir and by latest-run fallback
    assert {p.relative_to(run_dir).as_posix() for p in collect_trajectory_paths(run_dir)} == canonical
    assert {p.relative_to(run_dir).as_posix() for p in collect_trajectory_paths(daydream_dir)} == canonical
    assert _latest_main_trajectory(daydream_dir) == run_dir / RUN_DOCUMENT_NAME
    # 2. the frozen run summary agrees with the analyzer on the same names and retains no step bodies
    summary = summarize_run(_snapshot(run_dir / RUN_DOCUMENT_NAME))
    assert [(d.source_file, "steps" in d.metadata) for d in summary.documents] == [
        (RUN_DOCUMENT_NAME, False), (SIBLING, False),
    ]

    # 3. the archive projection writes the same relative names under its own root
    archive_run_dir = get_archive_dir() / "runs" / SESSION
    archive_run_dir.mkdir(parents=True, exist_ok=True)
    _project_documents(_snapshot(run_dir / RUN_DOCUMENT_NAME), archive_run_dir, session_id=SESSION)
    archived = archive_run_dir.rglob(f"*{DOCUMENT_SUFFIX}")
    assert {p.relative_to(archive_run_dir).as_posix() for p in archived} == canonical


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
