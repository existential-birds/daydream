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
from typing import Any

from tests.test_workflow_templates import job_steps, load_workflow

_ROOT = Path(__file__).resolve().parents[1]

# Files that describe the required gate to a contributor. The enforced coverage
# floor is a number in exactly one of them: pyproject.toml's fail_under.
# docs/coverage.md is deliberately absent: its baseline table is the recorded
# measurement history the ratchet procedure maintains, not a policy statement.
_GATE_FILES = ("Makefile", "pyproject.toml", ".github/workflows/ci.yml", "CONTRIBUTING.md", "AGENTS.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
)

_FLOOR_PERCENT_RE = re.compile(r"(?<![\w.])8\d(?:\.\d+)?\s*%")
_FLOOR_SETTING_RE = re.compile(r"fail_under\s*=\s*\d+")

def test_coverage_floor_is_declared_once_numerically() -> None:
    """No gate file but pyproject.toml restates the floor as a number."""
    restatements = [rel
        for rel in _GATE_FILES
        if rel != "pyproject.toml"
        and (_FLOOR_PERCENT_RE.search((_ROOT / rel).read_text(encoding="utf-8"))
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
    contributing = (_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    contributing_block = (
        contributing.split("which runs, in order:", 1)[1].split("```text\n", 1)[1].split("\n```", 1)[0]
    )
    return {
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
    optional = subprocess.run([make, "actionlint"], cwd=_ROOT, env=env, capture_output=True, text=True, input="")
    required = subprocess.run(
        [make, "actionlint"], cwd=_ROOT, env={**env, "ACTIONLINT_REQUIRE_DOCKER": "1"}, capture_output=True, text=True,
        input="",
    )

    assert optional.returncode == 0 and "skipped" in optional.stdout
    assert required.returncode != 0, required.stdout + required.stderr


_CI_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"
_CHECK_JOB = "check"

# CI check-job steps that are deliberately not local gate steps: step name ->
# the one-line reason. Keep this list small and reasoned; a step that is
# neither declared here nor a gate invocation fails the guard.
_CI_ONLY_STEPS: dict[str, str] = {"Run vulture": (
        "root project only — the standalone RL scan runs in the separate rl-check job, "
        "so the local gate's wider dead-code scope is not restated here"
    ),
}

_MAKE_INVOCATION_RE = re.compile(r"^make\s+([A-Za-z0-9_.-]+)\s*$")
_ACTIONLINT_DIGEST_RE = re.compile(r"rhysd/actionlint:[\w.]+@sha256:[0-9a-f]{64}")
_WHITESPACE_RE = re.compile(r"\s+")


def _recipe_commands(recipe: str) -> list[str]:
    """The individual shell commands a dry-run recipe printed."""
    lines = [line for line in recipe.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    joined = "\n".join(lines).replace("\\\n", " ")
    commands: list[str] = []
    for chunk in re.split(r"&&|;|\n", joined):
        command = chunk.strip().lstrip("@").strip()
        if command and command not in {"then", "else", "fi"}:
            commands.append(_WHITESPACE_RE.sub(" ", command))
    return commands


def _observed_gate_commands() -> dict[str, str]:
    """Map each gate step's command text to the target that owns it.

    Observed by dry-running the real Makefile (`make -n` prints a recipe without
    executing it), so the recovery needs no Docker daemon and no network.
    `MAKEFLAGS` is scrubbed because make prints `Entering directory` lines on
    stdout under it, which would be parsed as commands.
    """
    env = {key: value for key, value in os.environ.items() if key not in ("MAKEFLAGS", "MFLAGS")}
    owners: dict[str, str] = {}
    for target in _gate_steps():
        proc = subprocess.run(["make", "-n", target], cwd=_ROOT, env=env, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"`make -n {target}` failed: {proc.stderr}"
        for command in _recipe_commands(proc.stdout):
            owners.setdefault(command, target)
    return owners


def _gate_violations(
    steps: list[dict[str, Any]], gate_steps: set[str], owners: dict[str, str], exemptions: dict[str, str],
) -> list[str]:
    """Every reason a check job's steps are not represented by the local gate."""
    violations: list[str] = []
    for step in steps:
        run = step.get("run")
        if not isinstance(run, str):
            continue  # setup / artifact steps carry no command
        name = str(step.get("name", "<unnamed step>"))
        invocation = _MAKE_INVOCATION_RE.match(run.strip())
        if invocation:
            target = invocation.group(1)
            if target not in gate_steps:
                violations.append(f"{name}: runs `make {target}`, which is not a declared gate step")
            continue
        if name in exemptions:
            if not exemptions[name].strip():
                violations.append(f"{name}: declared CI-only step carries no reason")
            continue
        owner = owners.get(_WHITESPACE_RE.sub(" ", run.strip()))
        if owner is not None:
            violations.append(f"{name}: restates `{run.strip()}`, the command owned by the `{owner}` gate step; "
                f"run `make {owner}` instead"
            )
        else:
            violations.append(f"{name}: neither an invocation of a declared gate step nor a declared CI-only step")
    return violations

def test_ci_check_job_is_represented_by_the_declared_gate() -> None:
    """The real check job invokes declared gate steps, plus stated CI-only ones."""
    steps = job_steps(load_workflow(_CI_WORKFLOW), _CHECK_JOB)
    violations = _gate_violations(steps, set(_gate_steps()), _observed_gate_commands(), _CI_ONLY_STEPS)

    assert violations == [], "\n".join(violations)

def test_guard_rejects_a_restated_gate_command() -> None:
    """An inlined gate command is caught, and names the step that owns it."""
    steps = [{"name": "Lint with ruff", "run": "uv run ruff check daydream tests"}]
    violations = _gate_violations(steps, set(_gate_steps()), _observed_gate_commands(), {})

    assert violations and "lint" in violations[0], violations

def test_guard_rejects_an_unrepresented_step() -> None:
    """A check-job command that matches no gate step and no exemption fails."""
    steps = [{"name": "Inline gate", "run": "bash scripts/inline-gate.sh"}]
    violations = _gate_violations(steps, set(_gate_steps()), _observed_gate_commands(), {})

    assert violations and "neither an invocation" in violations[0], violations

def test_guard_rejects_a_ci_only_step_without_a_reason() -> None:
    """Declaring a step CI-only is not enough; it must state why."""
    steps = [{"name": "Run vulture", "run": "uv run vulture --config pyproject.toml daydream tests"}]
    owners = _observed_gate_commands()
    allowed = _gate_violations(steps, set(_gate_steps()), owners, {"Run vulture": "root only"})
    rejected = _gate_violations(steps, set(_gate_steps()), owners, {"Run vulture": "  "})

    assert allowed == []
    assert rejected and "no reason" in rejected[0], rejected

def test_guard_accepts_a_new_gate_step() -> None:
    """The guard reads the declared gate, so adding or reordering a step stays green."""
    steps = [{"name": "New gate step", "run": "make newcheck"}]

    assert _gate_violations(steps, {"lint", "newcheck"}, {}, {}) == []

def test_pinned_actionlint_image_has_one_owner() -> None:
    """Only the gate's actionlint step declares the workflow-lint image digest."""
    owners = [rel for rel in _GATE_FILES if _ACTIONLINT_DIGEST_RE.search((_ROOT / rel).read_text(encoding="utf-8"))]

    assert owners == ["Makefile"], owners
