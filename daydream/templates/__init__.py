"""Packaged workflow templates shipped as data inside the daydream wheel.

The three #147 GitHub Actions workflow templates live under
``daydream/templates/workflows/`` so they are present in both editable and
built installs (``importlib.resources`` resolves them identically in either
case). ``workflow_template_files()`` is the single source both the CLI copier
and the browser guide reference for the canonical template set.

Templates install Daydream at ``RELEASE_COMMIT_PLACEHOLDER``; ``render_workflow``
replaces it with the full commit SHA the installed workflow pins.

Exports:
    workflow_template_files() -> list[Path]: The three ``*.yml`` workflow
        templates (excludes ``README.md``).
    render_workflow(template, commit) -> str: Template text pinned to ``commit``.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

RELEASE_COMMIT_PLACEHOLDER = "DAYDREAM_RELEASE_COMMIT"


def workflow_template_files() -> list[Path]:
    """Return the three packaged workflow ``*.yml`` templates.

    Discovers the templates via ``importlib.resources`` so they resolve the
    same way in editable and built installs. ``README.md`` is excluded — it is
    operator documentation, not a workflow to land.

    Returns:
        Sorted list of ``Path`` objects for each ``*.yml`` template.
    """
    workflows = files("daydream.templates.workflows")
    return sorted(
        Path(str(entry))
        for entry in workflows.iterdir()
        if entry.name.endswith(".yml")
    )


def render_workflow(template: Path, commit: str) -> str:
    """Return ``template`` with every Daydream install line pinned to ``commit``."""
    return template.read_text().replace(RELEASE_COMMIT_PLACEHOLDER, commit)
