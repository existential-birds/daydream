"""Remove only ledger-owned Harbor outputs with containment checks. Cache cleanup preserves
its scaffold; trajectory cleanup removes contained trajectory files; job cleanup uses
recorded directories and image references. Curated content survives unless explicit
--all --yes deletion is requested, and even then all derived cleanup must succeed first.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from daydream.benchmark.harbor.run import (
    RunError,
    _default_confirm,
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

__all__ = ["CleanReport", "clean_workspace", "RunError", "WorkspaceCorrupt"]


_CACHE_TARGETS = ("cache/repository.git", "cache/harbor-build-stage")
_CURATED_DIRS = ("imports", "cases", "snapshots")


def _default_docker_rm(refs: list[str]) -> dict[str, Any]:
    """Remove a recorded Docker image; treat an already-missing image as absent."""
    completed = subprocess.run(
        ["docker", "rmi", *refs],
        stderr=subprocess.PIPE,
        text=True,
    )
    result = {"returncode": completed.returncode}
    if completed.returncode != 0 and "No such image" in (completed.stderr or ""):
        result["absent"] = True
    return result


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
    """Delete a containment-resolved filesystem target (symlink-safe)."""
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=False)
    else:
        path.unlink()


def _clean_cache(root: Path, report: CleanReport) -> None:
    """Remove contained disposable cache targets, retaining the scaffold; absent targets
    are a no-op.
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
    """Delete only contained trajectory files from validated ledger job directories."""
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
    """Remove recorded images and job directories in one locked ledger update. Failures
    preserve the job and prior run state. Already-absent images count as removed; mark
    cleaned only after all images and the job are gone.
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
            run_path = Path(validated)
            envs = entry.get("environments") or []
            if not envs:
                # Missing image refs allow cleanup only when no job directory materialized.
                # Otherwise retain the run: deleting it would strand unaddressable Docker images.
                if run_path.is_dir():
                    report.images_failed += 1
                    continue
                report.job_dirs_absent += 1
                entry["state"] = "cleaned"
                report.runs_cleaned += 1
                changed = True
                continue
            all_removed = True
            for env in envs:
                if env.get("removed") is True:
                    continue
                result = docker_rm(_image_refs(env))
                if result.get("returncode") == 0 or result.get("absent"):
                    env["removed"] = True
                    changed = True
                    if result.get("returncode") == 0:
                        report.images_removed += 1
                    else:
                        report.images_absent += 1
                else:
                    report.images_failed += 1
                    all_removed = False
            if not all_removed:
                # Partial failure: persist any images actually removed so a
                # later pass does not re-attempt (and re-fail) already-removed
                # images, but keep the job dir and the run's pre-clean state.
                continue
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


def _clean_curated(root: Path, report: CleanReport) -> None:
    """Delete curated source/gold: the four curated paths (only under ``--all``)."""
    for rel in (*_CURATED_DIRS, "benchmark.yaml"):
        _resolve_target(root, rel)
        target = root / rel
        if target.exists() or target.is_symlink():
            _delete_path(target)
            report.gold_deleted += 1
    report.recoverable = False


def clean_workspace(
    root: Path,
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
    if all_ and not yes:
        confirm = confirm or _default_confirm
        if not confirm("Refusing unconfirmed total cleanup (--all)"):
            report.refused = True
            return report
    if not (cache or jobs or trajectories or all_):
        return report
    if cache or all_:
        _clean_cache(root, report)
    if trajectories or all_:
        _clean_trajectories(root, report)
    if jobs or all_:
        _clean_jobs(root, report, docker_rm=docker_rm)
    # Delete irreplaceable source/gold only after all derived cleanup succeeds.
    if all_ and report.exit_code == 0:
        _clean_curated(root, report)
    return report
