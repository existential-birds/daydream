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
    _build_fix_scope_clause,
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


def test_scope_request_block_is_parsed_into_the_structured_request() -> None:
    """The only wire shape the prompt documents is the one the host parses."""
    parsed = parse_fix_scope_request(_scope_request_block("daydream/other.py"))
    assert parsed is not None
    assert parsed["paths"] == ["daydream/other.py"]
    assert parsed["evidence"] == {"daydream/other.py": "api.py:1 -- requires it"}
    # ``evidence_source`` is host-assigned: a turn cannot assert its own authority.
    assert parsed["evidence_source"] == "repair_turn"


@pytest.mark.parametrize(
    "output",
    [
        pytest.param("", id="empty"),
        pytest.param("I need daydream/other.py edited, api.py:1 requires it.", id="prose_only"),
        pytest.param(
            f"{SCOPE_REQUEST_BEGIN}\n{{\"paths\": [\"a.py\"]}}\n", id="unterminated"
        ),
        pytest.param(
            f"{SCOPE_REQUEST_BEGIN}\nnot json\n{SCOPE_REQUEST_END}\n", id="not_json"
        ),
        pytest.param(
            f"{SCOPE_REQUEST_BEGIN}\n[\"a.py\"]\n{SCOPE_REQUEST_END}\n", id="not_an_object"
        ),
    ],
)
def test_unusable_scope_request_is_never_a_half_parsed_grant(output: str) -> None:
    """Fail closed in both directions: no block, or an unusable one, grants nothing."""
    assert parse_fix_scope_request(output) is None


def test_scope_clause_documents_the_exact_block_the_host_parses() -> None:
    """The prompt must not ask for prose the parser cannot read (finding 1)."""
    clause = _build_fix_scope_clause(frozenset({"a.py"}), frozenset({"a.py"}))
    assert SCOPE_REQUEST_BEGIN in clause
    assert SCOPE_REQUEST_END in clause
    assert "Name the path in your final message" not in clause


def test_phase_test_records_the_turns_own_scope_request(tmp_path: Path) -> None:
    """``RepairAttemptEvidence.scope_request`` is populated in production."""
    source = Path(__file__).resolve().parents[2] / "daydream" / "phases" / "testing.py"
    text = source.read_text(encoding="utf-8")
    assert "scope_request = parse_fix_scope_request(turn_output)" in text, (
        "the repair turn's own scope request is no longer parsed from its final message"
    )
    assert "scope_request=scope_request," in text, (
        "the parsed request is no longer carried on the repair evidence"
    )
    assert "scope_request_unreadable" in text, (
        "an unparseable block must be a named degradation, not silence"
    )


# --- the stored bounds are live bounds ------------------------------------------------


def test_repair_turn_wall_budget_is_the_configured_ceiling_before_a_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first execution of a job is bounded by ``repair_execution_wall_s``.

    The phase's own wall budget is a second, independent ceiling and the tighter
    of the two wins, so neither bound can be widened past the other.
    """
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 1800.0)
    repo = _base_repo(tmp_path, "budget-first")
    deep = repo / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    config = SimpleNamespace(file_config=None, repair_execution_wall_s=300.0)
    assert _repair_turn_wall_budget_s(
        config,
        repo=repo,
        job_id=repair_job_id("s1"),
        artifact_session=None,
        allow_standalone=True,
    ) == 300.0
    # A repair bound ABOVE the phase budget does not escape the phase budget.
    generous = SimpleNamespace(file_config=None, repair_execution_wall_s=99_000.0)
    assert _repair_turn_wall_budget_s(
        generous,
        repo=repo,
        job_id=repair_job_id("s1"),
        artifact_session=None,
        allow_standalone=True,
    ) == 1800.0


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
        state=RepairJobState.PAUSED,
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


def test_repair_turn_wall_budget_ignores_a_foreign_jobs_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record belonging to another job is not this job's budget."""
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 1800.0)
    repo = _base_repo(tmp_path, "budget-foreign")
    deep = repo / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    write_repair_job_record(
        deep,
        RepairJobRecord(
            job_id=repair_job_id("other"),
            state=RepairJobState.PAUSED,
            policy=RepairJobPolicy(execution_s=30.0, job_total_s=1000.0, reserve_s=60.0),
            consumed_s=500.0,
        ),
    )
    config = SimpleNamespace(file_config=None, repair_execution_wall_s=300.0)
    assert _repair_turn_wall_budget_s(
        config,
        repo=repo,
        job_id=repair_job_id("s1"),
        artifact_session=None,
        allow_standalone=True,
    ) == 300.0


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


async def test_step_test_keeps_green_for_a_completed_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is the job's state, not a blanket refusal of every continuation."""
    from daydream.deep.fix_steps import _step_test
    from daydream.deep.repair_coordinator import RepairContinuationResult
    from daydream.deep.repair_job import write_repair_job_record as _write
    from daydream.phases import TestAndHealResult, TestAttemptEvidence

    repo = _base_repo(tmp_path, "fail-closed-completed")
    items = [{**_merge_item(1, "a.py", "high"), "item_uid": "item:a"}]
    ctx = _direct_fix_context(repo, items, changed_files={"a.py"})
    state = _direct_fix_state(ctx, items, {"a.py"})
    (repo / "a.py").write_text("A = 2\n")
    _git(repo, "add", ".")
    _commit(repo, "fix")
    _write(
        ctx.data["dd"],
        RepairJobRecord(job_id=repair_job_id(state.session_id), state=RepairJobState.COMPLETED),
    )

    attempt = TestAttemptEvidence(
        session_id=state.session_id, kind="host", command=("pytest",), passed=True,
        input_tree_key="k", output_tree_key="k",
    )
    green = TestAndHealResult(True, 0, True, False, (attempt,))

    async def _green(*_a: Any, **_k: Any) -> TestAndHealResult:
        return green

    async def _completed(**_k: Any) -> RepairContinuationResult:
        return RepairContinuationResult(
            continued=False,
            result=green,
            job=RepairJobRecord(
                job_id=repair_job_id(state.session_id), state=RepairJobState.COMPLETED,
            ),
            reason="completed",
        )

    monkeypatch.setattr("daydream.deep.fix_steps.phase_test_and_heal", _green)
    monkeypatch.setattr("daydream.deep.fix_steps.continue_repair_job", _completed)

    await _step_test(ctx)

    verdict = json.loads((ctx.data["dd"] / "test-verdict.json").read_text())
    assert verdict["passed"] is True, "a completed job must not be barred from green"


# --- one identity, one definition ------------------------------------------------------


def test_repair_job_identity_has_exactly_one_definition() -> None:
    """Producer and consumer must agree byte-for-byte (finding 7)."""
    from daydream.deep import repair_coordinator

    assert repair_job_id("abc") == "repair-abc"
    # The coordinator re-exports the owner's function, not a second copy of it.
    assert repair_coordinator.repair_job_id is repair_job_id
    testing_source = (
        Path(__file__).resolve().parents[2] / "daydream" / "phases" / "testing.py"
    ).read_text(encoding="utf-8")
    assert 'f"repair-{session_id}"' not in testing_source, (
        "phases/testing.py rebuilds the identity locally instead of calling the owner"
    )
