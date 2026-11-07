"""Single-source guards for the required local gate (issue #1228).

The gate's ordered check set and its coverage floor each live in exactly one
place. These tests fail when a description of the gate restates a value the
declaration already owns.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from tests.harness.workflows import job_steps, load_workflow, workflow_triggers

_ROOT = Path(__file__).resolve().parents[1]

# Files that describe the required gate to a contributor. The enforced coverage
# floor is a number in exactly one of them: pyproject.toml's fail_under.
# docs/coverage.md is deliberately absent: its baseline table is the recorded
# measurement history the ratchet procedure maintains, not a policy statement.
_GATE_FILES = ("Makefile", "pyproject.toml", ".github/workflows/ci.yml", "CONTRIBUTING.md", "AGENTS.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
)

_FLOOR_PERCENT_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*%")
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


def _static_steps() -> list[str]:
    """The checks the gate's `static` step runs in parallel."""
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(r"^STATIC_CHECKS = (.+)$", makefile, re.MULTILINE)
    assert match, "the Makefile declares no STATIC_CHECKS group"
    return match.group(1).split()


def _documented_gate_steps() -> dict[str, list[str]]:
    """The gate steps each contributor-facing enumeration claims, in order."""
    agent_guidance = (_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    guidance_line = next(line for line in agent_guidance.splitlines() if line.startswith("make check") and "#" in line)
    contributing = (_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    contributing_block = (
        contributing.split("which runs, in order:", 1)[1].split("```text\n", 1)[1].split("\n```", 1)[0]
    )
    return {
        "AGENTS.md": [step.strip() for step in guidance_line.split("#", 1)[1].split(" (the gate)")[0].split("+")],
        "CONTRIBUTING.md": contributing_block.split(),
    }

def test_gate_prose_enumerations_match_the_declared_gate() -> None:
    """Every prose enumeration lists the gate's steps, in the declared order."""
    declared = _gate_steps()
    for rel, listed in _documented_gate_steps().items():
        assert listed == declared, f"{rel} enumerates {listed}, the gate declares {declared}"

_CI_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"


def _execute_shard_plan(
    tmp_path: Path, *, dispatch: bool, requested_count: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    """Execute the shipped planner step, resolving only its Actions input context."""
    workflow = load_workflow(_CI_WORKFLOW)
    input_spec = workflow_triggers(workflow)["workflow_dispatch"]["inputs"]["test_jobs"]
    value = str(input_spec["default"]) if requested_count is None else requested_count
    inputs = {"test_jobs": value} if dispatch else {}
    step = next(step for step in job_steps(workflow, "plan") if step["name"] == "Plan pytest shards")
    output = tmp_path / "github-output"
    output.touch()
    env = {**os.environ, "GITHUB_OUTPUT": str(output)}
    for name, value in step.get("env", {}).items():
        # Resolve the runner-provided input value, not the planner's Python logic.
        expression = re.fullmatch(r"\$\{\{\s*inputs\.(\w+)(?:\s*\|\|\s*'([^']*)')?\s*\}\}", str(value))
        env[name] = str(inputs.get(expression[1]) or expression[2] or "") if expression else str(value)
    result = subprocess.run(["bash", "-eu", "-o", "pipefail", "-c", step["run"]], cwd=_ROOT, env=env,
                            capture_output=True, text=True, check=False)
    return result, dict(line.split("=", 1) for line in output.read_text().splitlines())


@pytest.mark.parametrize(("dispatch", "requested_count", "count", "workers"), [
    (False, None, 8, 4),
    (True, None, 8, 4),
    (True, "1", 1, 8),
    (True, "20", 20, 4),
], ids=["automatic-default", "dispatch-default", "baseline", "maximum"])
def test_ci_shard_planner_emits_complete_supported_matrices(
    tmp_path: Path, dispatch: bool, requested_count: str | None, count: int, workers: int,
) -> None:
    """A working CLI alone cannot prove CI's default, input binding or emitted matrix."""
    result, outputs = _execute_shard_plan(tmp_path, dispatch=dispatch, requested_count=requested_count)
    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["count"] == str(count)
    assert outputs["workers"] == str(workers)
    assert json.loads(outputs["matrix"]) == {"index": list(range(count))}
    input_spec = workflow_triggers(load_workflow(_CI_WORKFLOW))["workflow_dispatch"]["inputs"]["test_jobs"]
    if dispatch and requested_count is not None and input_spec["type"] == "choice":
        assert requested_count in input_spec["options"], "the workflow UI must offer the supported treatment"


@pytest.mark.parametrize("requested_count", ["0", "21", "1.5"], ids=["below-minimum", "above-cap", "fractional"])
def test_ci_shard_planner_rejects_counts_without_publishing_a_partial_matrix(
    tmp_path: Path, requested_count: str,
) -> None:
    """Direct dispatch values cannot create empty, over-cap or fractional matrices."""
    result, outputs = _execute_shard_plan(tmp_path, dispatch=True, requested_count=requested_count)
    assert result.returncode != 0
    assert outputs == {}

# CI check-job steps that are deliberately not local gate steps: step name ->
# the one-line reason. Keep this list small and reasoned; a step that is
# neither declared here nor a gate invocation fails the guard.
_CI_ONLY_STEPS: dict[str, str] = {
    "Install Pi for native output contract tests": (
        "provision the pinned native CLI on the disposable CI runner — the local gate uses contributor-installed "
        "prerequisites without globally installing or replacing them"
    ),
    "Plan pytest shards": "select supported shard counts for the manual benchmark and normal duration-weighted gate",
    "Run pytest shard": "partition the complete pytest collection and record checkout-bound execution and coverage",
    "Combine and enforce branch coverage": "validate every shard receipt before applying the aggregate coverage floor",
    "Require complete CI gate": "reject failed, cancelled, missing or unexpectedly skipped dependencies",
    "Plan CI gate": (
        "remote evidence determines whether CI needs to repeat a gate; local hooks always run the full gate"
    ),
    "Record successful merge-group tree": (
        "publish the verified queue checkout only after the complete remote gate passes; "
        "local gates publish no CI proof"
    ),
}

_MAKE_INVOCATION_RE = re.compile(r"^make\s+([A-Za-z0-9_.-]+)\s*$")


def test_ci_check_job_is_represented_by_the_declared_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real check job invokes declared gate steps, plus stated CI-only ones."""
    monkeypatch.setenv("MAKEFLAGS", "-w")
    monkeypatch.setenv("MFLAGS", "-w")
    workflow = load_workflow(_CI_WORKFLOW)
    steps = [step for job in workflow["jobs"].values() for step in job["steps"]]
    assert all(reason.strip() for reason in _CI_ONLY_STEPS.values())
    targets = {*_gate_steps(), *_static_steps()}
    for step in steps:
        command = step.get("run")
        if not isinstance(command, str) or step.get("name") in _CI_ONLY_STEPS:
            continue
        invocation = _MAKE_INVOCATION_RE.fullmatch(command.strip())
        assert invocation and invocation.group(1) in targets, (
            f"{step.get('name')}: invoke a declared make gate target or explain the CI-only exception"
        )


def test_merge_queue_workflow_keeps_complete_gate_on_github_hosted_runners() -> None:
    """Queue/manual checks retain all full gates; PR basic never publishes proof."""
    workflow = load_workflow(_CI_WORKFLOW)
    triggers = workflow_triggers(workflow)
    assert triggers["merge_group"]["types"] == ["checks_requested"]
    assert all(job["runs-on"] == "ubuntu-24.04" for job in workflow["jobs"].values())
    jobs = workflow["jobs"]
    assert jobs["check"]["if"] == "always()"
    assert set(jobs["check"]["needs"]) == {"plan", "static-checks", "test-shard", "coverage", "legacy-check"}
    assert jobs["static-checks"]["needs"] == "plan"
    assert jobs["static-checks"]["if"] == "needs.plan.outputs.count != '1'"
    assert jobs["legacy-check"]["if"] == "github.event_name == 'workflow_dispatch' && needs.plan.outputs.count == '1'"
    assert jobs["legacy-check"]["env"]["PYTEST_WORKERS"] == "8"
    assert jobs["legacy-check"]["env"]["PYTEST_REPORT_ARGS"] == "--junitxml=baseline-tests.xml"
    legacy = {step["name"]: step for step in job_steps(workflow, "legacy-check")}
    assert legacy["Run tests"]["run"] == "make test"
    assert "if" not in legacy["Run tests"]
    assert set(legacy["Upload coverage report"]["with"]["path"].split()) == {"coverage.xml", "baseline-tests.xml"}
    assert jobs["test-shard"]["needs"] == "plan"
    assert jobs["test-shard"]["if"] == "needs.plan.outputs.full == 'true' && needs.plan.outputs.count != '1'"
    assert jobs["test-shard"]["strategy"]["fail-fast"] is False
    assert set(jobs["coverage"]["needs"]) == {"plan", "test-shard"}
    assert jobs["coverage"]["if"] == (
        "always() && needs.plan.result == 'success' && needs.plan.outputs.full == 'true' "
        "&& needs.plan.outputs.count != '1'"
    )
    assert all(not job.get("continue-on-error", False) for job in jobs.values())
    assert all(not step.get("continue-on-error", False)
               for job in jobs.values() for step in job["steps"])
    static = {step["name"]: step for step in job_steps(workflow, "static-checks")}
    for name in ("Set up uv", "Verify uv.lock is in sync", "Install dependencies", "Lint with ruff"):
        assert static[name]["if"] == "needs.plan.outputs.mode != 'reuse'"
    for name in ("Run vulture", "Type check with mypy", "Run script unit tests"):
        assert static[name]["if"] == "needs.plan.outputs.full == 'true'"
    assert static["Run script unit tests"]["run"] == "make test-scripts"
    for name in ("Naming grep-gate", "Lint workflows with actionlint"):
        assert "if" not in static[name]
    shard = {step["name"]: step for step in job_steps(workflow, "test-shard")}
    assert shard["Run pytest shard"]["env"]["PYTEST_WORKERS"] == "${{ needs.plan.outputs.workers }}"
    assert "--index" in shard["Run pytest shard"]["run"]
    assert shard["Upload shard evidence"]["if"] == "always()"
    assert shard["Upload shard evidence"]["with"]["overwrite"] is True
    assert shard["Upload shard evidence"]["with"]["if-no-files-found"] == "error"
    coverage = {step["name"]: step for step in job_steps(workflow, "coverage")}
    assert "--diagnostic" in coverage["Combine and enforce branch coverage"]["run"]
    assert coverage["Download all shard evidence"]["with"]["pattern"] == "shard-*"
    assert not coverage["Download all shard evidence"]["with"].get("merge-multiple", False)
    assert coverage["Upload coverage report"]["with"]["overwrite"] is True
    assert coverage["Upload coverage report"]["if"] == "always()"
    final = {step["name"]: step for step in job_steps(workflow, "check")}
    assert list(final).index("Require complete CI gate") < list(final).index("Record successful merge-group tree")
    gate_env = final["Require complete CI gate"]["env"]
    assert set(gate_env.values()) >= {
        "${{ needs.plan.result }}", "${{ needs.static-checks.result }}",
        "${{ needs.test-shard.result }}", "${{ needs.coverage.result }}", "${{ needs.legacy-check.result }}",
    }
    for name in ("Record successful merge-group tree", "Upload successful merge-group tree"):
        assert final[name]["if"] == "github.event_name == 'merge_group' && needs.plan.outputs.full == 'true'"
    assert final["Upload successful merge-group tree"]["with"]["retention-days"] == 14
    assert final["Upload successful merge-group tree"]["with"]["overwrite"] is True
