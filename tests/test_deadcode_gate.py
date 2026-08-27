"""Dead-code gate: vulture scans exit clean on the real trees."""

import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_root_deadcode_scan_exits_clean() -> None:
    """Root project (daydream + tests) has zero vulture findings at min_confidence."""
    proc = subprocess.run(
        ["uv", "run", "vulture", "--config", "pyproject.toml", "daydream", "tests"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, f"unexpected vulture findings:\n{proc.stdout}"


def test_rl_deadcode_scan_exits_clean() -> None:
    """RL project (daydream_review_v1 + tests) has zero vulture findings at min_confidence."""
    rl_root = _REPO_ROOT / "rl" / "daydream_review_v1"
    proc = subprocess.run(
        ["uv", "run", "vulture", "--config", "pyproject.toml", "daydream_review_v1", "tests"],
        capture_output=True,
        text=True,
        cwd=str(rl_root),
    )
    assert proc.returncode == 0, f"unexpected vulture findings:\n{proc.stdout}"
