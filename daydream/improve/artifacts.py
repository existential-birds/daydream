"""Artifact path helpers for the improve advisor flow."""

from __future__ import annotations

from pathlib import Path


def improve_dir(target: Path) -> Path:
    """Return the target's ``.daydream/improve`` directory, creating it."""
    directory = target / ".daydream" / "improve"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def services_path(improve_dir_path: Path) -> Path:
    """Return the service-enumeration artifact path."""
    return improve_dir_path / "services.json"


def recon_path(improve_dir_path: Path) -> Path:
    """Return the repository-reconnaissance artifact path."""
    return improve_dir_path / "recon.json"


def audit_findings_path(
    improve_dir_path: Path, category: str, stack: str | None
) -> Path:
    """Return the findings path for one category/stack audit assignment."""
    suffix = f"-{stack}" if stack else ""
    return improve_dir_path / f"audit-{category}{suffix}-findings.json"


def vetted_findings_path(improve_dir_path: Path) -> Path:
    """Return the vetted-findings artifact path."""
    return improve_dir_path / "vetted-findings.json"


def report_path(improve_dir_path: Path) -> Path:
    """Return the rendered improve report path."""
    return improve_dir_path / "report.md"
