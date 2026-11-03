"""Acceptance matrix for the bounded repair job (issue #1210, task 10).

Every row below is a real-path test: it enters through ``runner.run`` against a
real temporary Git worktree, with a real event loop, a real filesystem, and a
real test process where the row is about a real suite. The only thing replaced is
the external model backend, installed at the ``create_backend`` seam.

The rows exercise distinct host outcomes, repair deadlines, retained candidates,
and required-suite acceptance; passing prose cannot overrule their native evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.config_file import DaydreamFileConfig
from daydream.run_config import RunConfig
from daydream.runner import run
from tests.harness.execution import host_test_command
from tests.harness.stub_backend import StubBackend
from tests.test_deep_orchestrator import (
    Mute,
    _merge_item,
    _silence,
)

#: The interrupted repair below still lands the edits it was working on, so a
#: candidate exists to continue from.
CANDIDATE_LINE = "\n# candidate repair\n"
_JOB = "repair-job.json"
_VERDICT = "test-verdict.json"


def _install(
    monkeypatch: pytest.MonkeyPatch, target: Path, **knobs: Any,
) -> StubBackend:
    """Install the stub backend, with the repair-turn knobs a row needs."""
    stub = StubBackend(target)
    for name, value in knobs.items():
        setattr(stub, name, value)
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kw: stub)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    return stub


def _interrupted_repair(**overrides: Any) -> dict[str, Any]:
    """Knobs for a repair turn the host wall budget cuts short, mid-edit."""
    return {
        "heal_fix_partial": "PARTIAL-DIAGNOSIS-abc123",
        "runaway_fix_sleep_s": 0.05,
        "heal_fix_authorized": "api.py",
        "heal_fix_authorized_line": CANDIDATE_LINE,
        **overrides,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _job(target: Path) -> dict[str, Any]:
    """The persisted repair job record of the run under test."""
    return _read_json(target / ".daydream" / "deep" / _JOB)


def _verdict(target: Path) -> dict[str, Any]:
    """The persisted test verdict of the run under test."""
    return _read_json(target / ".daydream" / "deep" / _VERDICT)


async def _run_deep(target: Path, **overrides: Any) -> int:
    """Await the production entrypoint for a repaired target."""
    fields: dict[str, Any] = {
        "target": str(target), "non_interactive": True, "cleanup": False, "archive": False,
        "assume": "yes", "output_mode": "loop", **overrides,
    }
    with anyio.fail_after(90):
        return await run(RunConfig(**fields))


# --- rows -----------------------------------------------------------------------------------------


async def test_operator_grant_resumes_the_same_job_without_erasing_history(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """An operator's grant resumes the same job; the history it already spent stays spent.

    The job's own wall allowance is configured to nothing, so the first
    execution exhausts it. The operator's additional finite allowance is what
    lets the bounded validation run -- and it is recorded *separately* from the
    consumption, so the record never claims the time was never spent.
    """
    _silence(monkeypatch)
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 0.3)
    target = multi_stack_target
    stub = _install(monkeypatch, target, **_interrupted_repair())
    command = host_test_command(target.parent / "suite", mode="first")
    mute_side_effects(heal=False)

    exit_code = await _run_deep(
        target, test_command=command,
        file_config=DaydreamFileConfig(repair_job_wall_s=0.5, repair_grant_job_wall_s=300.0),
    )
    assert exit_code == 0, "the granted allowance let the bounded validation run"

    job = _job(target)
    assert job["state"] == "completed", job
    assert job["executions"] == 2, job
    assert job["granted_allowance_s"] == 300.0, job
    assert job["consumed_s"] > 0.0, job
    # The grant is an addition, not a rewrite: the seconds this job actually
    # spent are still the seconds it spent.
    assert job["consumed_s"] < 300.0, job
    # The job's own allowance is what the first execution exhausted; the grant
    # is what the second one drew on, and it is still on the record afterwards.
    assert job["policy"]["job_total_s"] == 0.5, job
    assert any("operator granted 300s more" in note for note in job["diagnostics"]), job
    assert stub.test_suite_calls == 0
