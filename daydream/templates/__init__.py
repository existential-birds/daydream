"""Packaged workflow templates shipped as data inside the daydream wheel.

The three #147 GitHub Actions workflow templates live under
``daydream/templates/workflows/`` so they are present in both editable and
built installs (``importlib.resources`` resolves them identically in either
case). ``workflow_template_files()`` is the single source both the CLI copier
and the browser guide reference for the canonical template set.

Exports:
    workflow_template_files() -> list[Path]: The three ``*.yml`` workflow
        templates (excludes ``README.md``).
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


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
