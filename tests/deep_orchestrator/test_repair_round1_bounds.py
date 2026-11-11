"""Round-1 review regressions: the repair job's four documented bounds are live.

Each test here pins a fix the round-1 deep review found inert (see
``deep/merged-items.json`` on the review fork). None of the fixes was covered by
the round's own regression tests, so the bounds could have stayed decorative:

* the scope-widening contract is only reachable if a turn's ``BEGIN SCOPE
  REQUEST`` block is parsed into ``RepairAttemptEvidence.scope_request``;
* ``repair_execution_wall_s`` must bound the repair turn, not just the record;
* ``_step_test`` must fail closed on a job that cannot report green;
* the job identity has exactly one definition.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream.deep.repair_job import (
    RepairJobPolicy,
    RepairJobRecord,
    RepairJobState,
    repair_job_id,
    write_repair_job_record,
)
from daydream.extensions.api import Stop
from daydream.phases.fix import (
    SCOPE_REQUEST_BEGIN,
    SCOPE_REQUEST_END,
    parse_fix_scope_request,
)
from daydream.phases.testing import _repair_turn_wall_budget_s
from tests.deep_orchestrator.support import (
    _base_repo,
    _direct_fix_context,
    _direct_fix_state,
)
from tests.harness.git_helpers import commit as _commit, git as _git
from tests.test_deep_orchestrator import _merge_item


def _scope_request_block(*paths: str) -> str:
    """A turn's final message ending in one well-formed scope-request block."""
    named = ", ".join(f'"{p}"' for p in paths)
    evidence = ", ".join(f'"{p}": "api.py:1 -- requires it"' for p in paths)
    return (
        "I need one more path.\n"
        f"{SCOPE_REQUEST_BEGIN}\n"
        f'{{"paths": [{named}], "evidence": {{{evidence}}}}}\n'
        f"{SCOPE_REQUEST_END}\n"
    )


# --- the scope-widening contract is reachable ----------------------------------------


def test_scope_request_block_is_parsed_or_grants_nothing() -> None:
    """The only wire shape the prompt documents is the one the host parses."""
    parsed = parse_fix_scope_request(_scope_request_block("daydream/other.py"))
    assert parsed is not None
    assert parsed["paths"] == ["daydream/other.py"]
    assert parsed["evidence"] == {"daydream/other.py": "api.py:1 -- requires it"}
    # ``evidence_source`` is host-assigned: a turn cannot assert its own authority.
    assert parsed["evidence_source"] == "repair_turn"
    # Fail closed: an unterminated or unusable block grants nothing.
    for output in (f"{SCOPE_REQUEST_BEGIN}\n{{\"paths\": [\"a.py\"]}}\n",
                   f"{SCOPE_REQUEST_BEGIN}\nnot json\n{SCOPE_REQUEST_END}\n",
                   f"{SCOPE_REQUEST_BEGIN}\n[\"a.py\"]\n{SCOPE_REQUEST_END}\n"):
        assert parse_fix_scope_request(output) is None, output


# --- the stored bounds are live bounds ------------------------------------------------


def test_repair_turn_wall_budget_follows_the_jobs_own_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once a record exists, the grant the job was given is the bound (findings 3/4)."""
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 1800.0)
    repo = _base_repo(tmp_path, "budget-stored")
    deep = repo / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    job = RepairJobRecord(
        job_id=repair_job_id("s1"),
        policy=RepairJobPolicy(execution_s=900.0, job_total_s=1000.0, reserve_s=60.0),
        consumed_s=500.0,
    )
    write_repair_job_record(deep, job)
    allowance = job.execution_allowance_s()
    assert 0.0 < allowance < 900.0, "the fixture must leave a live but reduced allowance"
    config = SimpleNamespace(file_config=None, repair_execution_wall_s=1800.0)
    # Not the configured 1800.0 ceiling, and not the policy's own 900.0 --
    # what the job has left after its persisted consumption and the reserve.
    assert _repair_turn_wall_budget_s(
        config,
        repo=repo,
        job_id=repair_job_id("s1"),
        artifact_session=None,
        allow_standalone=True,
    ) == allowance


# --- a job that cannot report green cannot report green -------------------------------


async def test_step_test_refuses_green_next_to_a_non_completed_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coordinator may hand back a passing result beside a blocked job.

    ``repair_coordinator`` short-circuits on a persisted blocked/exhausted record
    and returns the caller's result unchanged, so ``_step_test`` is the only place
    the documented fail-closed gate can be enforced -- before the retained tree is
    committed and pushed (findings 2/6).
    """
    from daydream.deep.fix_steps import _step_test
    from daydream.deep.repair_coordinator import RepairContinuationResult
    from daydream.phases import TestAndHealResult, TestAttemptEvidence

    repo = _base_repo(tmp_path, "fail-closed")
    items = [{**_merge_item(1, "a.py", "high"), "item_uid": "item:a"}]
    ctx = _direct_fix_context(repo, items, changed_files={"a.py"})
    state = _direct_fix_state(ctx, items, {"a.py"})
    (repo / "a.py").write_text("A = 2\n")
    _git(repo, "add", ".")
    _commit(repo, "fix")

    attempt = TestAttemptEvidence(
        session_id=state.session_id, kind="host", command=("pytest",), passed=True,
        input_tree_key="k", output_tree_key="k",
    )
    green = TestAndHealResult(True, 0, True, False, (attempt,))

    async def _green(*_a: Any, **_k: Any) -> TestAndHealResult:
        return green

    async def _blocked(**_k: Any) -> RepairContinuationResult:
        return RepairContinuationResult(
            continued=False,
            result=green,
            job=RepairJobRecord(
                job_id=repair_job_id(state.session_id),
                state=RepairJobState.BLOCKED,
                last_transition_reason="checkpoint_untrusted",
            ),
            reason="checkpoint_untrusted",
        )

    monkeypatch.setattr("daydream.deep.fix_steps.phase_test_and_heal", _green)
    monkeypatch.setattr("daydream.deep.fix_steps.continue_repair_job", _blocked)

    outcome = await _step_test(ctx)

    assert isinstance(outcome, Stop), "a non-completed job must stop the run"
    assert outcome.exit_code == 1
    verdict = json.loads((ctx.data["dd"] / "test-verdict.json").read_text())
    assert verdict["passed"] is False, "the persisted verdict must not report green"


# --- one identity, one definition ------------------------------------------------------


async def test_fix_steps_stop_closed_on_fix_scope_verify_and_configuration_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every fix-cycle step failure is a confinement or configuration stop, never a silent pass.

    Runner journeys reach these steps only through slow full pipelines; here each
    real step runs against real Git with the external phase call replaced.
    """
    from dataclasses import replace

    from daydream.config_file import DaydreamFileConfig
    from daydream.deep import fix_state
    from daydream.deep.fix_steps import _round_dispatch_items, _step_fix, _step_fix_verify, _step_test
    from daydream.extensions.api import BreakLoop
    from daydream.phases import TestAndHealResult, TestAttemptEvidence
    from daydream.test_execution import TestCommandDiagnostic

    items = [{**_merge_item(1, "a.py", "high"), "item_uid": "item:a"}]

    def fresh(name: str) -> tuple[Any, Any]:
        ctx = _direct_fix_context(_base_repo(tmp_path, name), items, changed_files={"a.py"})
        ctx.config = replace(ctx.config, file_config=DaydreamFileConfig(quality_gate_enabled=False))
        ctx.data["intent_path"] = ctx.data["dd"] / "intent.md"
        return ctx, _direct_fix_state(ctx, items, {"a.py"})

    async def failed_fix(*_a: Any, **_k: Any) -> dict[str, str]:
        raise RuntimeError("fix fan-out failed")

    ctx, _state = fresh("fix-raises")
    monkeypatch.setattr("daydream.deep.fix_steps.phase_fix_parallel", failed_fix)
    assert await _step_fix(ctx) == Stop(1)

    async def no_failures(*_a: Any, **_k: Any) -> dict[str, str]:
        return {}

    def scope_failure(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("scope enforcement failed")

    ctx, _state = fresh("scope-raises")
    monkeypatch.setattr("daydream.deep.fix_steps.phase_fix_parallel", no_failures)
    with monkeypatch.context() as patch:
        patch.setattr(fix_state, "_strict_scope_and_scrub", scope_failure)
        assert await _step_fix(ctx) == Stop(1)

    async def failed_verify(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise RuntimeError("verifier failed")

    ctx, _state = fresh("verify-raises")
    monkeypatch.setattr("daydream.deep.fix_steps.verify_retained_tree", failed_verify)
    assert await _step_fix_verify(ctx) == Stop(1)

    async def unresolved(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"item:a": {"issue_id": 1, "verdict": "unresolved", "reason": "still broken"}}

    async def regressed(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"item:a": {"issue_id": 1, "verdict": "regressed", "reason": "make lint fails",
                           "check_evidence": {"status": "failed"}}}

    # A substantiated regression blocks commit and push, persisting the verdict it acted on.
    ctx, _state = fresh("verify-regressed")
    ctx.data["iteration"] = 3
    monkeypatch.setattr("daydream.deep.fix_steps.verify_retained_tree", regressed)
    assert await _step_fix_verify(ctx) == Stop(1)
    persisted = json.loads((ctx.data["dd"] / "fix-outcomes.json").read_text())["outcomes"]
    assert persisted["item:a"]["verdict"] == "regressed"

    ctx, _state = fresh("verify-exhausted")
    ctx.data["iteration"] = 3
    monkeypatch.setattr("daydream.deep.fix_steps.verify_retained_tree", unresolved)
    assert isinstance(await _step_fix_verify(ctx), BreakLoop)
    # A later round carries the unresolved verdict and its reason into the next dispatch.
    ctx.data["iteration"] = 2
    assert [(item["fix_verify_verdict"], item["fix_verify_reason"]) for item in _round_dispatch_items(ctx, items)
            ] == [("unresolved", "still broken")]

    ctx, state = fresh("test-no-execution")
    diagnostic = TestCommandDiagnostic("unavailable", "cli")
    no_execution = TestAttemptEvidence(session_id=state.session_id, kind="no-execution", command=None, passed=False,
                                       input_tree_key="", output_tree_key="", diagnostic=diagnostic)

    async def unavailable(*_a: Any, **_k: Any) -> TestAndHealResult:
        return TestAndHealResult(False, 0, False, False, (no_execution,), (), diagnostic)

    monkeypatch.setattr("daydream.deep.fix_steps.phase_test_and_heal", unavailable)
    assert await _step_test(ctx) == Stop(1)
    verdict = json.loads((ctx.data["dd"] / "test-verdict.json").read_text())
    assert verdict["status"] == "no-execution" and verdict["attempts"][0]["kind"] == "no-execution"
    assert state.latest_test_evidence is None and state.latest_retained is None
