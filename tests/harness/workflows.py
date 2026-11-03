"""Parse GitHub workflow YAML for workflow and gate tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def load_workflow(path: Path) -> dict[str, Any]:
    """Parse a workflow template into its YAML tree."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{path.name} did not parse to a mapping"
    return loaded


def job_steps(wf: dict[str, Any], job: str) -> list[dict[str, Any]]:
    """Return the steps list for ``job``."""
    steps = wf["jobs"][job]["steps"]
    assert isinstance(steps, list) and steps
    return steps


def workflow_triggers(wf: dict[str, Any]) -> dict[str, Any]:
    """Return the ``on:`` trigger map, normalizing PyYAML's boolean key."""
    on: Any = wf.get("on")
    if on is None:
        on = cast(dict[Any, Any], wf).get(True)
    return on if isinstance(on, dict) else {}
