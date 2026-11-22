"""Bound a workspace's finished runs, private and public (docs/runtime.md#artifacts-and-recovery).

Run directories are created and finalized only under the exclusive workspace lock,
so while a closing session still holds it every other run directory is finished.
Recovery reads transactions, never run directories; a run named by a retained or
quarantined transaction keeps every byte for the operator.
"""

from __future__ import annotations

import stat
from pathlib import Path

from daydream.artifacts import filesystem
from daydream.artifacts.models import _FROZEN_MANIFEST, _LIVE, _RUNS, ArtifactVisibilityError
from daydream.artifacts.ownership import SourceOwnership
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


def _bound(runs: Path, *, current: str, held: set[str], retained: int) -> tuple[list[Path], int]:
    """Keep the *retained* newest runs under *runs*, *current* and held runs aside, and delete the rest.

    Return the kept runs other than *current* and how many could not be deleted. Links are never
    followed, and an unsafe tree is refused whole and left in place.
    """
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
    return kept, failures


def retain_finished_runs(state_root: Path, *, current: str, retained: int) -> int:
    """Bound the private runs, dropping each kept frozen run's live copy; return the failure count.

    The caller holds the workspace lock and has reconciled *current*.
    """
    runs = state_root / RUNS_DIRNAME
    held = _held_sessions(state_root)
    kept, failures = _bound(runs, current=current, held=held, retained=retained)
    for run in ([] if current in held else [runs / current]) + kept:
        live = run / _LIVE
        manifest = run / _FROZEN_MANIFEST
        if manifest.is_file() and not manifest.is_symlink() and live.is_dir() and not live.is_symlink():
            try:
                filesystem._remove_owned_tree(live, run)
            except (OSError, ArtifactVisibilityError):
                failures += 1
    return failures


def retain_public_runs(source: SourceOwnership, state_root: Path, *, current: str, retained: int) -> int:
    """Bound the checkout's ``.daydream/runs/`` with the same rule (a public run is ordered by its directory's
    time: it never holds a frozen manifest); return the failure count.

    The caller holds the workspace lock and has reconciled *current*, so nothing is parked. A
    checkout without public run evidence, such as a hosted review's, is left alone.
    """
    runs = source.path / _RUNS
    try:
        metadata = runs.lstat()
    except FileNotFoundError:
        return 0
    if not stat.S_ISDIR(metadata.st_mode):
        return 1
    source.validate()
    # A run that published nothing, such as a rolled-back one, takes no slot from earlier runs.
    published = (runs / current).exists() or (runs / current).is_symlink()
    return _bound(runs, current=current, held=_held_sessions(state_root), retained=retained + (not published))[1]
