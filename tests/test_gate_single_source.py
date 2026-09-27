"""Single-source guards for the required local gate (issue #1228).

The gate's ordered check set and its coverage floor each live in exactly one
place. These tests fail when a description of the gate restates a value the
declaration already owns.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
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


def _gate_steps() -> list[str]:
    """The ordered gate steps the Makefile declares (the single authority)."""
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    for line in makefile.splitlines():
        match = re.match(r"^check:\s*(\S.*)$", line)
        if match:
            return match.group(1).split()
    raise AssertionError("the Makefile declares no `check:` gate target")


def _documented_gate_steps() -> dict[str, list[str]]:
    """The gate steps each contributor-facing enumeration claims, in order."""
    claude = (_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    claude_line = next(
        line for line in claude.splitlines() if line.startswith("make check") and "#" in line
    )
    contributing = (_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    contributing_block = (
        contributing.split("which runs, in order:", 1)[1].split("```text\n", 1)[1].split("\n```", 1)[0]
    )
    return {
        "CLAUDE.md": [step.strip() for step in claude_line.split("#", 1)[1].split(" (the gate)")[0].split("+")],
        "CONTRIBUTING.md": contributing_block.split(),
    }


def test_gate_prose_enumerations_match_the_declared_gate() -> None:
    """Every prose enumeration lists the gate's steps, in the declared order."""
    declared = _gate_steps()

    for rel, listed in _documented_gate_steps().items():
        assert listed == declared, f"{rel} enumerates {listed}, the gate declares {declared}"


def test_pr_checklist_does_not_claim_the_rl_gate_runs_under_make_check() -> None:
    """The checklist's `make check` line covers root + workflow checks only."""
    template = (_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")
    checklist = [line for line in template.splitlines() if line.startswith("- [ ]")]
    gate_line = next(line for line in checklist if "`make check`" in line)

    assert "RL" not in gate_line, gate_line
    assert any("`make rl-check`" in line for line in checklist), checklist


def test_actionlint_cannot_be_skipped_silently_when_required(tmp_path: Path) -> None:
    """`ACTIONLINT_REQUIRE_DOCKER` turns the missing-daemon skip into a failure."""
    make = shutil.which("make")
    assert make is not None
    env = {key: value for key, value in os.environ.items() if key != "ACTIONLINT_REQUIRE_DOCKER"}
    env["PATH"] = str(tmp_path)  # no docker on PATH

    optional = subprocess.run(
        [make, "actionlint"], cwd=_ROOT, env=env, capture_output=True, text=True, input=""
    )
    required = subprocess.run(
        [make, "actionlint"],
        cwd=_ROOT,
        env={**env, "ACTIONLINT_REQUIRE_DOCKER": "1"},
        capture_output=True,
        text=True,
        input="",
    )

    assert optional.returncode == 0 and "skipped" in optional.stdout
    assert required.returncode != 0, required.stdout + required.stderr
