"""Supervised Harbor runs behind the Oracle self-match gate (issue #781).

Task 1 clocks in the CLI surface; the full supervisor lands in later tasks.
"""

from __future__ import annotations

from pathlib import Path


def run_run(
    workspace: Path,
    *,
    oracle: bool = False,
    yes: bool = False,
    env: dict | None = None,
    spawn=None,
    docker_ok=None,
    confirm=None,
) -> int:
    """Run (supervised) — implemented in later tasks of issue #781."""
    raise NotImplementedError("run_run lands with the preflight tasks")