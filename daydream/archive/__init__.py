"""Centralized run archive for daydream.

Automatically copies the full artifact bundle (trajectory, review output,
deep artifacts, diff) to ``~/.daydream/archive/runs/{session_id}/`` after
every run, writes a ``manifest.json``, and indexes the run in a SQLite
database for cross-project querying.

Exports:
    archive_run: Top-level entry point called from the TrajectoryRecorder
        on_write callback.
    get_archive_dir: Returns the archive root directory, creating it on
        first access.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.archive.git_context import capture_git_context
from daydream.archive.index import upsert_run
from daydream.archive.manifest import build_manifest
from daydream.config import REVIEW_OUTPUT_FILE

if TYPE_CHECKING:
    from daydream.runner import RunConfig
    from daydream.trajectory import TrajectoryRecorder


def get_archive_dir() -> Path:
    """Return the archive root directory, creating it on first access.

    Respects ``DAYDREAM_ARCHIVE_DIR`` env var. Default: ``~/.daydream/archive/``.

    Returns:
        Path to the archive root directory.
    """
    env = os.environ.get("DAYDREAM_ARCHIVE_DIR")
    if env:
        archive_dir = Path(env)
    else:
        archive_dir = Path.home() / ".daydream" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "runs").mkdir(exist_ok=True)
    return archive_dir


def archive_run(
    *,
    recorder: TrajectoryRecorder,
    target_dir: Path,
    config: RunConfig,
    status: str = "complete",
    run_eval: bool = False,
) -> None:
    """Copy artifact bundle to archive and index in SQLite.

    Called from the ``on_write`` callback on ``TrajectoryRecorder``. Wraps
    its entire body in try/except so archive failure never affects the
    primary run.

    Args:
        recorder: The TrajectoryRecorder that produced the trajectory.
        target_dir: Target directory that was reviewed.
        config: The RunConfig for this run.
        status: Run status (``complete``, ``partial``, ``failed``).
        run_eval: Whether to run deterministic evaluation analysis.
    """
    try:
        _archive_run_inner(
            recorder=recorder,
            target_dir=target_dir,
            config=config,
            status=status,
            run_eval=run_eval,
        )
    except Exception:  # noqa: BLE001 - archive failure must never affect the run
        # Import lazily to avoid circular imports at module level
        try:
            from daydream.ui import create_console, print_warning

            print_warning(create_console(), "Run archive failed (non-fatal)")
        except Exception:  # noqa: BLE001
            pass


def _archive_run_inner(
    *,
    recorder: TrajectoryRecorder,
    target_dir: Path,
    config: RunConfig,
    status: str,
    run_eval: bool,
) -> None:
    """Core archive logic, not exception-wrapped."""
    archive_dir = get_archive_dir()
    run_dir = archive_dir / "runs" / recorder.session_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy artifact bundle
    _copy_bundle(target_dir, run_dir, recorder.session_id)

    # 2. Capture git context
    git_ctx = capture_git_context(target_dir)

    # 3. Optionally run deterministic evaluation
    evaluation: dict[str, Any] | None = None
    if run_eval:
        evaluation = _run_eval(target_dir, recorder.session_id, run_dir)

    # 4. Build and write manifest
    manifest = build_manifest(
        recorder=recorder,
        config=config,
        git_ctx=git_ctx,
        status=status,
        archive_path=run_dir,
        evaluation=evaluation,
    )
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    # 5. Index in SQLite
    upsert_run(archive_dir, manifest)


def _copy_bundle(target_dir: Path, run_dir: Path, session_id: str) -> None:
    """Copy ``.daydream/`` artifacts to the archive run directory.

    Copies trajectory files matching the session_id prefix, the
    ``trajectories/`` subdirectory, ``review-output.md``, ``deep/``
    artifacts, and ``diff.patch``. Missing files are silently skipped.
    """
    daydream_dir = target_dir / ".daydream"
    prefix = session_id[:8]

    # Main trajectory file(s) matching this session
    for traj_file in daydream_dir.glob(f"trajectory-*-{prefix}.json"):
        shutil.copy2(traj_file, run_dir / "trajectory.json")
        break  # Only one main trajectory per session

    # Also try the partial file
    for partial_file in daydream_dir.glob(f"trajectory-*-{prefix}.json.partial"):
        shutil.copy2(partial_file, run_dir / "trajectory.json.partial")
        break

    # Forked sub-trajectories
    trajectories_dir = daydream_dir / "trajectories"
    if trajectories_dir.is_dir():
        dest_traj = run_dir / "trajectories"
        # Copy only files matching this session's prefix
        matched = list(trajectories_dir.glob(f"{prefix}.*.json"))
        if matched:
            dest_traj.mkdir(exist_ok=True)
            for f in matched:
                shutil.copy2(f, dest_traj / f.name)

    # Review output (in target root, not .daydream/)
    review_output = target_dir / REVIEW_OUTPUT_FILE
    if review_output.is_file():
        shutil.copy2(review_output, run_dir / "review-output.md")

    # Deep artifacts directory
    deep_dir = daydream_dir / "deep"
    if deep_dir.is_dir():
        shutil.copytree(deep_dir, run_dir / "deep", dirs_exist_ok=True)

    # Diff patch
    diff_patch = daydream_dir / "diff.patch"
    if diff_patch.is_file():
        shutil.copy2(diff_patch, run_dir / "diff.patch")


def _run_eval(target_dir: Path, session_id: str, run_dir: Path) -> dict[str, Any] | None:
    """Run deterministic evaluation analysis and write results to the archive."""
    try:
        from daydream.eval.analyzer import analyze_session

        daydream_dir = target_dir / ".daydream"
        result = analyze_session(daydream_dir, session_id=session_id)
        eval_path = run_dir / "evaluation.json"
        eval_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    except Exception:  # noqa: BLE001 - eval failure should not block archive
        return None
