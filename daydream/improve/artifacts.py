"""Artifact path helpers for the improve advisor flow."""

from __future__ import annotations

from pathlib import Path

from daydream.artifact_visibility import artifact_dir_for


def improve_dir(target: Path) -> Path:
    """Return the target's ``.daydream/improve`` directory, creating it."""
    directory = artifact_dir_for(target) / "improve"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def recon_path(improve_dir_path: Path) -> Path:
    """Return the repository-reconnaissance artifact path."""
    return improve_dir_path / "recon.json"


def coverage_path(improve_dir_path: Path) -> Path:
    """Return the partition/group coverage-ledger artifact path."""
    return improve_dir_path / "coverage.json"


def vetted_findings_path(improve_dir_path: Path) -> Path:
    """Return the vetted-findings artifact path."""
    return improve_dir_path / "vetted-findings.json"


def report_path(improve_dir_path: Path) -> Path:
    """Return the rendered improve report path."""
    return improve_dir_path / "report.md"


def plan_write_diagnostics_path(improve_dir_path: Path) -> Path:
    """Return the sanitized plan-writer attempt diagnostics path."""
    return improve_dir_path / "plan-write-diagnostics.json"


def published_issues_path(improve_dir_path: Path) -> Path:
    """Return the GitHub issue-publication disposition artifact path."""
    return improve_dir_path / "published-issues.json"
