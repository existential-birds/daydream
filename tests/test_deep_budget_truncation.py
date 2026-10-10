"""Deep non-wonder budget/truncation integration tests."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.backends import AgentEvent, ResultEvent, ToolStartEvent
from daydream.config_file import DaydreamFileConfig
from daydream.review_result import ReviewCoverage
from daydream.run_config import RunConfig
from daydream.runner import run
from tests.deep_orchestrator.support import _scan_trajectory_extra
from tests.harness.fake_clock import FakeClock
from tests.harness.review_profile import independent_alternatives_profile
from tests.harness.stub_backend import (
    StubBackend,
    install_stub_backend,
    review_stage_state,
    silence,
    stage_result,
)
from tests.test_deep_orchestrator import _pin_findings_pr, _profile_with_pipeline


async def test_budget_truncated_stack_lands_in_failed_stacks(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., 'RunConfig'],
    mute_side_effects: Callable[..., None],
) -> None:
    """A truncated per-stack review is recorded as a failure, not a success."""

    silence(monkeypatch)
    monkeypatch.setattr("daydream.config.DEFAULT_TOOL_CALL_BUDGET", 3)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.runaway_stack = "python"
    mute_side_effects()
    with anyio.fail_after(30):
        await run(make_config(
                multi_stack_target, trajectory_path=tmp_path / "trajectory.json", assume="yes", output_mode="loop",
            )
        )
    coverage = ReviewCoverage.from_dict(json.loads(
        (multi_stack_target / ".daydream/deep/review-coverage.json").read_text()))
    failures = coverage.unfinished_scopes
    assert "python" in failures, failures
    assert "budget" in failures["python"].lower()

async def test_runaway_test_turn_is_bounded_and_reaches_abort(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., 'RunConfig'],
    mute_side_effects: Callable[..., None],
) -> None:
    """A hung test turn is capped, so the run reaches the heal/abort path."""

    silence(monkeypatch)
    monkeypatch.setattr("daydream.config.DEFAULT_TOOL_CALL_BUDGET", 3)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.runaway_test = True
    mute_side_effects(heal=False)
    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"))
    assert exit_code != 0
    assert [c for c in stub.calls if "run the project's test suite" in c["prompt"].lower()]
    test_stop_reasons = _scan_trajectory_extra(multi_stack_target / ".daydream", traj, "stop_reason", phase="test")
    assert any("budget" in r for r in test_stop_reasons), test_stop_reasons

@pytest.mark.parametrize(
    "phase", ["intent", "alternatives", "per_stack", "merge", "arbiter", "suppression", "supervisor"],
)
@pytest.mark.parametrize("budget", ["wall", "tool_call"])
async def test_review_budget_stop_emits_partial_findings(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., RunConfig], phase: str,
    budget: str,
) -> None:
    """Real agent budget stops cannot strand completed findings before Phase B."""

    fake = FakeClock().install(monkeypatch)
    fragments = {"intent": "present your understanding concisely", "alternatives": "evaluate the implementation",
        "per_stack": "you are reviewing the python stack", "merge": "cross-stack merge agent",
        "arbiter": "you are the arbiter", "suppression": "you are the suppression reviewer",
        "supervisor": "supervisor adjudication",
    }

    class BudgetBackend(StubBackend):
        exhaust = True

        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any,) -> AsyncIterator[AgentEvent]:
            if self.exhaust and fragments[phase] in prompt.lower():
                stage = review_stage_state(prompt)
                for n in range(stage['remaining_tool_calls'] + 1 if stage is not None else 17):
                    if budget == "wall":
                        fake.advance(601)
                    yield ToolStartEvent(id=f"budget-{n}", name="Read", input={"file_path": "api.py"})
                    await anyio.sleep(0)
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    silence(monkeypatch)
    stub = BudgetBackend(multi_stack_target)
    if phase == "arbiter":
        stub.parse_severity = "high"
    elif phase == "suppression":
        stub.parse_severity = "low"
    file_config = DaydreamFileConfig(
        supervisor="llm" if phase == "supervisor" else "off", precision_mode=phase == "suppression",
    )
    monkeypatch.setattr("daydream.runner.create_backend", lambda *a, **kw: stub)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    monkeypatch.setattr("daydream.config.DEFAULT_TOOL_CALL_BUDGET", 3 if budget == "tool_call" else None)
    _pin_findings_pr(monkeypatch, multi_stack_target)
    out = multi_stack_target / "findings.json"
    with anyio.fail_after(30):
        code = await run(make_config(multi_stack_target, pr_number=7, findings_out=str(out), file_config=file_config,
            review_profile=independent_alternatives_profile() if phase == "alternatives" else None,
        ))
    assert code == 0
    artifact = json.loads(out.read_text())
    assert artifact["review_warnings"], "budget stop must be visible to the poster"
    assert any(f"{budget}_budget_exceeded" in warning for warning in artifact["review_warnings"])
    assert artifact["findings"], "completed reviewers' findings must survive"
    terminal = artifact["terminal_result"]
    assert terminal["analysis_state"] == "incomplete"
    expected_reason = ("host_wall_budget_exhaustion" if budget == "wall" else "host_tool_budget_exhaustion")
    assert expected_reason in terminal["reason_codes"]
    if phase == "per_stack":
        scope = next(row for row in terminal["stack_outcomes"] if row["scope_id"] == "python")
        assert scope["status"] == "incomplete"
    else:
        phase_name = {"supervisor": "supervision"}.get(phase, phase)
        outcome = next(row for row in terminal["phase_outcomes"] if row["phase"] == phase_name)
        assert outcome["status"] in {"incomplete", "failed"}
        assert expected_reason in outcome["reason_codes"]
    assert "incomplete" in (multi_stack_target / ".review-output.md").read_text().lower()

    if phase == "merge":
        stub.exhaust = False
        assert await run(make_config(multi_stack_target, pr_number=7, findings_out=str(out), start_at="merge",)) == 0
        assert not json.loads(out.read_text()).get("review_warnings")

async def test_single_stack_alternatives_timeout_still_emits_findings(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., RunConfig],
) -> None:
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, tiny_diff_target)
    stub.runaway_alternatives = True
    monkeypatch.setattr("daydream.config.DEFAULT_TOOL_CALL_BUDGET", 3)
    _pin_findings_pr(monkeypatch, tiny_diff_target)
    out = tiny_diff_target / "findings.json"
    assert await run(make_config(
        tiny_diff_target, pr_number=7, findings_out=str(out), review_profile=independent_alternatives_profile(),
    )) == 0
    artifact = json.loads(out.read_text())
    assert artifact["review_warnings"] == ["alternatives: tool_call_budget_exceeded"]

async def test_admitted_interaction_findings_survive_triage_cutoff_and_merge_resume(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., RunConfig],
) -> None:

    class CheckpointBackend(StubBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            state = review_stage_state(prompt)
            if state is not None and state['scope_id'] == 'structure' and state['stage'] == 'triage':
                yield ResultEvent(structured_output=stage_result(state, candidates=[
                    dict(item, disposition='rejected') for item in state['candidates']
                ]), continuation=None)
                for n in range(state['remaining_tool_calls'] + 1):
                    yield ToolStartEvent(id=f"extra-{n}", name="Read", input={"file_path": "api.py"})
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                if (state is not None and state['scope_id'] == 'structure'
                        and isinstance(event, ResultEvent)):
                    assert isinstance(event.structured_output, dict)
                    event.structured_output['candidates'].append({
                        'candidate_id': '', 'file': 'api.py', 'line': 2,
                        'trigger': 'hello is called', 'consequence': 'Greeting may conflict with callers',
                        'grounds': 'api.py:2 returns universe', 'disposition': 'open', 'finding': None,
                    })
                yield event

    silence(monkeypatch)
    backend = CheckpointBackend(multi_stack_target)
    backend.parse_severity = "high"
    backend.merge_echo_records = True
    monkeypatch.setattr("daydream.runner.create_backend", lambda *a, **kw: backend)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    _pin_findings_pr(monkeypatch, multi_stack_target)
    out = multi_stack_target / "findings.json"
    for start_at in (None, "merge"):
        assert await run(make_config(multi_stack_target, pr_number=7, findings_out=str(out), start_at=start_at,)) == 0
        saved = json.loads((multi_stack_target / ".daydream/deep/stack-structure-records.json").read_text())
        assert saved["issues"], "findings admitted before triage cutoff must survive adjudication and resume"
        assert saved["incomplete"] is True
        published = json.loads(out.read_text())
        assert published["findings"]
        terminal = published["terminal_result"]
        assert terminal["analysis_state"] == "incomplete"
        scope = next(row for row in terminal["stack_outcomes"] if row["scope_id"] == "structure")
        assert scope["status"] == "incomplete"
        assert scope["partial_evidence"] is True
        assert scope["reason_codes"] == ["host_tool_budget_exhaustion"]
        assert any("structure" in warning for warning in published["review_warnings"])

async def test_spent_pipeline_budget_still_publishes_explicitly_incomplete_artifact(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., RunConfig],
) -> None:
    silence(monkeypatch)
    backend = install_stub_backend(monkeypatch, multi_stack_target)
    _pin_findings_pr(monkeypatch, multi_stack_target)
    out = multi_stack_target / "findings.json"
    assert await run(make_config(multi_stack_target, pr_number=7, findings_out=str(out),
        review_profile=_profile_with_pipeline(review_wall_budget_s=0),
    )) == 0
    assert not backend.calls
    artifact = json.loads(out.read_text())
    assert artifact["findings"] == []
    assert artifact["terminal_result"]["analysis_state"] == "failed"
    assert "host_pipeline_budget_exhaustion" in artifact["terminal_result"]["reason_codes"]
    assert all(row["status"] == "uncovered" for row in artifact["terminal_result"]["stack_outcomes"])
    assert artifact["review_warnings"]
    assert "Review incomplete" in (multi_stack_target / ".review-output.md").read_text()
