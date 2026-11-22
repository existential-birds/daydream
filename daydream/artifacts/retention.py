"""Bound one workspace's finished run directories (docs/runtime.md#artifacts-and-recovery).

Run directories are created and finalized only under the exclusive workspace lock,
so while a closing session still holds it every other run directory is finished.
Recovery reads transactions, never run directories; a run named by a retained or
quarantined transaction keeps every byte for the operator.
"""

from __future__ import annotations

from pathlib import Path

from daydream.artifacts import filesystem
from daydream.artifacts.models import _FROZEN_MANIFEST, _LIVE, ArtifactVisibilityError
from daydream.trajectory import RUNS_DIRNAME

_TRANSACTION_ROOTS = ("transactions", "unreconciled")


def _held_sessions(state_root: Path) -> set[str]:
    """Session ids named by ``<kind>-<session id>-<token>`` transactions that still hold evidence."""
    held: set[str] = set()
    for name in _TRANSACTION_ROOTS:
        root = state_root / name
        if root.is_dir() and not root.is_symlink():
            held.update(entry.name.partition("-")[2].rpartition("-")[0] for entry in root.iterdir())
    return held


def _finished_at(run: Path) -> tuple[int, str]:
    """Freeze time when frozen, else the directory's own time; neither changes when live is dropped."""
    manifest = run / _FROZEN_MANIFEST
    stamp = manifest if manifest.is_file() and not manifest.is_symlink() else run
    return stamp.lstat().st_mtime_ns, run.name


def retain_finished_runs(state_root: Path, *, current: str, retained: int) -> int:
    """Keep the *retained* newest runs, *current* included, and drop each kept frozen run's live copy.

    The caller holds the workspace lock and has reconciled *current*. Return how many
    runs could not be cleaned; an unsafe tree is refused whole and left in place.
    """
    runs = state_root / RUNS_DIRNAME
    held = _held_sessions(state_root)
    finished = sorted(
        (run for run in runs.iterdir()
         if run.name != current and run.name not in held and run.is_dir() and not run.is_symlink()),
        key=_finished_at, reverse=True,
    )
    kept = finished[:max(retained - 1, 0)]
    failures = 0
    for run in finished[len(kept):]:
        try:
            filesystem._remove_owned_tree(run, runs)
        except (OSError, ArtifactVisibilityError):
            failures += 1
    for run in ([] if current in held else [runs / current]) + kept:
        live = run / _LIVE
        manifest = run / _FROZEN_MANIFEST
        if manifest.is_file() and not manifest.is_symlink() and live.is_dir() and not live.is_symlink():
            try:
                filesystem._remove_owned_tree(live, run)
            except (OSError, ArtifactVisibilityError):
                failures += 1
    return failures
