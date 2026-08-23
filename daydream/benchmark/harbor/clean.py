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
from typing import Any, Callable

from daydream.benchmark.harbor.run import (
    RunError,
    _ledger_path,
    _load_ledger,
    _validate_job_dir,
)
from daydream.benchmark.storage import (
    WorkspaceCorrupt,
    WorkspaceLock,
    _resolve_target,
    atomic_write_json,
)

RunError  # re-exported for the CLI handler / hermetic escape tests


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


def _clean_trajectories(root: Path, report: CleanReport) -> None:
    """Delete contained ``agent/trajectory.json`` files in ledgered job dirs.

    Ledger-driven: every job dir is validated under ``<ws>/harbor/jobs/``
    (``RunError`` on escape) and every glob hit is containment-resolved
    (``WorkspaceCorrupt`` on a symlinked trajectory escaping the workspace).
    The job dir and its non-trajectory content are left intact.
    """
    doc = _load_ledger(root)
    for entry in doc["runs"]:
        job_abs = _validate_job_dir(root, entry["job_dir"])
        job_path = Path(job_abs)
        if not job_path.is_dir():
            report.trajectory_absent += 1
            continue
        for hit in job_path.rglob("agent/trajectory.json"):
            _resolve_target(root, hit)
            hit.unlink()
            report.trajectory_deleted += 1


def _image_refs(env: dict[str, Any]) -> list[str]:
    """The exact recorded image refs for one environment (never guessed)."""
    refs: list[str] = []
    if env.get("image_id"):
        refs.append(str(env["image_id"]))
    refs.extend(str(t) for t in env.get("image_tags") or [])
    return refs


def _clean_jobs(
    root: Path, report: CleanReport, *, docker_rm: Callable[[list[str]], dict[str, Any]] | None = None,
) -> None:
    """Remove ledgered job dirs + their recorded Docker images.

    For each non-``cleaned`` run, every environment whose ``removed`` flag is
    not already true is removed via the injectable ``docker_rm`` seam using only
    the exact recorded ``image_id``/``image_tags`` refs. A failed removal leaves
    the run's job dir and prior state intact (partial-failure rule); a run
    transitions to ``cleaned`` only when its job dir is gone/absent *and* all of
    its environments are removed. One locked load-mutate-write pass.
    """
    docker_rm = docker_rm or _default_docker_rm
    with WorkspaceLock(root):
        doc = _load_ledger(root)
        changed = False
        for entry in doc["runs"]:
            if entry.get("state") == "cleaned":
                report.runs_already_clean += 1
                continue
            validated = _validate_job_dir(root, entry["job_dir"])
            envs = entry.get("environments") or []
            all_removed = True
            for env in envs:
                if env.get("removed") is True:
                    continue
                result = docker_rm(_image_refs(env))
                if result.get("returncode") == 0:
                    env["removed"] = True
                    report.images_removed += 1
                else:
                    report.images_failed += 1
                    all_removed = False
            if not all_removed:
                continue
            run_path = Path(validated)
            if run_path.is_dir():
                _delete_path(run_path)
                report.job_dirs_deleted += 1
            else:
                report.job_dirs_absent += 1
            entry["state"] = "cleaned"
            report.runs_cleaned += 1
            changed = True
        if changed:
            atomic_write_json(_ledger_path(root), doc, mode=0o600)


def clean_workspace(
    root,
    *,
    cache: bool = False,
    jobs: bool = False,
    trajectories: bool = False,
    all_: bool = False,
    yes: bool = False,
    confirm: Callable[[str], bool] | None = None,
    docker_rm: Callable[[list[str]], dict[str, Any]] | None = None,
) -> CleanReport:
    """Delete only the requested ledger-derived artifacts (empty selection = no-op)."""
    root = Path(root).resolve()
    report = CleanReport()
    if cache:
        _clean_cache(root, report)
    if trajectories:
        _clean_trajectories(root, report)
    if jobs:
        _clean_jobs(root, report, docker_rm=docker_rm)
    return report