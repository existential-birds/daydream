"""Fixtures for training-pipeline config tests.

The dry-path tests need a prime-rl workspace checkout per rl/train/README.md
(prime-rl cannot be consumed as a dependency, so the entrypoint only exists
inside that checkout). Resolution order:

1. ``$PRIME_RL_WORKSPACE`` if set,
2. ``/home/exedev/prime-rl`` if it exists and has a synced ``.venv``.

Otherwise the test skips with the documented reason — Task 0's spike
(plan-notes.md) records the workspace recipe.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

DEFAULT_WORKSPACE = Path("/home/exedev/prime-rl")


def _usable_workspace() -> Path | None:
    candidate = os.environ.get("PRIME_RL_WORKSPACE")
    paths = [Path(candidate)] if candidate else [DEFAULT_WORKSPACE]
    for p in paths:
        if p.is_dir() and (p / ".venv").is_dir() and (p / "pyproject.toml").is_file():
            return p
    return None


@pytest.fixture(scope="session")
def prime_rl_workspace() -> Path:
    """Path to a synced prime-rl workspace checkout, or skip."""
    workspace = _usable_workspace()
    if workspace is None:
        pytest.skip(
            "prime-rl workspace checkout not available "
            "(set PRIME_RL_WORKSPACE or sync /home/exedev/prime-rl per rl/train/README.md)"
        )
    return workspace
