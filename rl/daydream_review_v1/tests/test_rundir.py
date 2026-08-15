"""Boundary guard: the run-dir collector never forwards golden trajectory payloads."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from verifiers.v1.runtimes.subprocess import SubprocessRuntime

from daydream_review_v1.rundir import fetch_run_dir

#: The deterministic scoring projection fetch_run_dir must return for the golden
#: run: the RUN_DIR_FILES allowlist members present in the fixture, plus the
#: deep/stack-*-records.json glob members. trajectory.json and any per-fork
#: trajectories/*.json are deliberately NOT here — untrusted, model-directed,
#: test-only data that must never reach model context through the collector.
_EXPECTED_SCORING_FILES: frozenset[str] = frozenset(
    {
        "manifest.json",
        "review-output.md",
        "deep/review-output.md",
        "deep/recommendation-verdicts.json",
        "deep/merged-items.json",
        "deep/test-verdict.json",
        "deep/stack-generic-records.json",
        "deep/stack-structure-records.json",
    }
)


async def test_fetch_run_dir_excludes_fixture_trajectories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: SubprocessRuntime,
    rundir_golden: Path,
) -> None:
    """The collector's projection excludes trajectory payloads from the golden run.

    Stage the real golden archive, install a guarded read that raises on any
    trajectory path, and prove fetch_run_dir returns exactly the 8-file scoring
    set — so captured operational text can never be forwarded into model context
    through the run-dir collector.
    """
    archive = tmp_path / "archive"
    staged = archive / "runs" / "session-1"
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(rundir_golden, staged)

    # The sole remaining untrusted trajectory shape in the fixture is staged.
    assert (staged / "trajectory.json").is_file()

    saved_read = runtime.read

    async def guarded_read(path: str) -> bytes:
        rel = Path(path).relative_to(str(staged)).as_posix()
        if rel == "trajectory.json" or rel.startswith("trajectories/"):
            raise AssertionError(f"trajectory path forwarded to collector: {rel}")
        return await saved_read(path)

    monkeypatch.setattr(runtime, "read", guarded_read)

    selected = await fetch_run_dir(runtime, tmp_path / "selected", archive_root=str(archive))

    assert selected is not None
    projected = {p.relative_to(selected).as_posix() for p in selected.rglob("*") if p.is_file()}
    assert projected == _EXPECTED_SCORING_FILES
    assert not (selected / "trajectory.json").exists()
    assert not (selected / "trajectories").exists()
