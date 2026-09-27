"""Single-source guards for the required local gate (issue #1228).

The gate's ordered check set and its coverage floor each live in exactly one
place. These tests fail when a description of the gate restates a value the
declaration already owns.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# Files that describe the required gate to a contributor. The enforced coverage
# floor is a number in exactly one of them: pyproject.toml's fail_under.
# docs/coverage.md is deliberately absent: its baseline table is the recorded
# measurement history the ratchet procedure maintains, not a policy statement.
_GATE_FILES = (
    "Makefile",
    "pyproject.toml",
    ".github/workflows/ci.yml",
    "CONTRIBUTING.md",
    "CLAUDE.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
)

_FLOOR_PERCENT_RE = re.compile(r"(?<![\w.])8\d(?:\.\d+)?\s*%")
_FLOOR_SETTING_RE = re.compile(r"fail_under\s*=\s*\d+")


def test_coverage_floor_is_declared_once_numerically() -> None:
    """No gate file but pyproject.toml restates the floor as a number."""
    restatements = [
        rel
        for rel in _GATE_FILES
        if rel != "pyproject.toml"
        and (
            _FLOOR_PERCENT_RE.search((_ROOT / rel).read_text(encoding="utf-8"))
            or _FLOOR_SETTING_RE.search((_ROOT / rel).read_text(encoding="utf-8"))
        )
    ]

    assert restatements == [], f"files still restating the coverage floor: {restatements}"
