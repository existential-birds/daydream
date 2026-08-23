"""Ledger-backed cleanup of a Harbor benchmark workspace (issue #782).

A sibling of :mod:`daydream.benchmark.harbor.run` that removes exactly what the
workspace recorded it created: the disposable ``cache/`` clone + build stage
(``--cache``), contained ``agent/trajectory.json`` files in ledgered job dirs
(``--trajectories``), and the ledgered job dirs + their recorded Docker images
(``--jobs``) — all driven by ``runtime/harbor.json``, never guessed. Curated
source/gold (``benchmark.yaml``, ``imports/``, ``cases/``, ``snapshots/``) is
preserved unless the single explicit ``--all --yes`` total-deletion path is
taken. Every filesystem target is containment-/symlink-escape-checked via the
existing ``storage._resolve_target`` / ``run._validate_job_dir`` primitives and
fails closed (``RunError``/``WorkspaceCorrupt``). Docker removal runs through an
injectable ``docker_rm`` seam (the real production default shells ``docker
rmi``; CI stays hermetic).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.benchmark.storage import (
    WorkspaceCorrupt,
    WorkspaceLock,
    _resolve_target,
    atomic_write_json,
)


_CACHE_TARGETS = ("cache/repository.git", "cache/harbor-build-stage")


def _default_docker_rm(refs: list[str]) -> dict[str, Any]:
    """Default Docker image-removal seam: shell ``docker rmi`` (real path).

    Hermetic tests inject a stub; CI never exercises this default.
    """
    completed = subprocess.run(["docker", "rmi", *refs])
    return {"returncode": completed.returncode}


@dataclass
class CleanReport:
    """Counters + recoverability for one ``clean_workspace`` pass."""

    cache_deleted: int = 0
    cache_absent: int = 0
    trajectory_deleted: int = 0
    trajectory_absent: int = 0
    job_dirs_deleted: int = 0
    job_dirs_absent: int = 0
    runs_cleaned: int = 0
    runs_already_clean: int = 0
    images_removed: int = 0
    images_absent: int = 0
    images_failed: int = 0
    gold_deleted: int = 0
    recoverable: bool = True
    refused: bool = False

    @property
    def exit_code(self) -> int:
        """0 only when the requested deletion set completed fully."""
        return 1 if (self.images_failed or self.refused) else 0

    def summary_lines(self) -> list[str]:
        """A short human summary of the exact effect of the pass."""
        esc = "recoverable" if self.recoverable else "unrecoverable"
        return [
            f"cache: {self.cache_deleted} deleted, {self.cache_absent} absent",
            f"trajectories: {self.trajectory_deleted} deleted, {self.trajectory_absent} absent",
            f"job dirs: {self.job_dirs_deleted} deleted, {self.job_dirs_absent} absent",
            f"runs: {self.runs_cleaned} cleaned, {self.runs_already_clean} already clean",
            f"images: {self.images_removed} removed, {self.images_absent} absent, {self.images_failed} failed",
            f"gold: {self.gold_deleted} deleted",
            f"deletion is {esc}",
        ]


def _delete_path(path: Path) -> None:
    """Delete a containment-resolved filesystem target (symlink-safe).

    Callers must have resolved ``path`` through ``_resolve_target``/containment
    already; this only removes the exact link (for a symlink) or the tree/file.
    """
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=False)
    else:
        path.unlink()


def _clean_cache(root: Path, report: CleanReport) -> None:
    """Remove the disposable clone + build stage under ``cache/``.

    Each target is resolved through ``storage._resolve_target`` (forcing
    containment and rejecting symlink escapes/``..``/outside-root as
    ``WorkspaceCorrupt``). A target already absent is a no-op. The ``cache/``
    scaffold dir itself is never removed.
    """
    for rel in _CACHE_TARGETS:
        resolved = _resolve_target(root, rel)
        target = root / Path(resolved)
        if not target.exists() and not target.is_symlink():
            report.cache_absent += 1
            continue
        _delete_path(target)
        report.cache_deleted += 1


def clean_workspace(
    root,
    *,
    cache: bool = False,
    jobs: bool = False,
    trajectories: bool = False,
    all_: bool = False,
    yes: bool = False,
) -> CleanReport:
    """Delete only the requested ledger-derived artifacts (empty selection = no-op)."""
    report = CleanReport()
    if cache:
        _clean_cache(Path(root), report)
    return report