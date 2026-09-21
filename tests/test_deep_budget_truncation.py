"""Deep non-wonder budget/truncation integration tests."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import anyio
import pytest

from daydream.runner import RunConfig
from tests.harness.review_profile import independent_alternatives_profile
from tests.harness.stub_backend import install_stub_backend, silence


def _test_step_stop_reasons(run_root: Path, traj: Path) -> list[str]:
    """stop_reason values on TEST-phase trajectory steps (empty list if none)."""
    values: list[str] = []
    for path in list(run_root.rglob("*.json")) + ([traj] if traj.exists() else []):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        for step in payload.get("steps", []):
            extra = step.get("extra") or {}
            if extra.get("daydream_phase") == "test":
                reason = extra.get("stop_reason")
                if reason:
                    values.append(str(reason))
    return values


async def test_budget_truncated_stack_lands_in_failed_stacks(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., 'RunConfig'],
    mute_side_effects: Callable[..., None],
) -> None:
    """A truncated per-stack review is recorded as a failure, not a success."""
    from daydream.runner import run

    silence(monkeypatch)
    monkeypatch.setattr("daydream.phases.DEFAULT_TOOL_CALL_BUDGET", 3)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.runaway_stack = "python"
    mute_side_effects()
    with anyio.fail_after(30):
        await run(
            make_config(
                multi_stack_target,
                trajectory_path=tmp_path / "trajectory.json",
                assume="yes",
                output_mode="loop",
            )
        )
    failures = json.loads((multi_stack_target / ".daydream" / "deep" / "per-stack-failures.json").read_text())
    assert "python" in failures, failures
    assert "budget" in failures["python"].lower()


async def test_runaway_test_turn_is_bounded_and_reaches_abort(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., 'RunConfig'],
    mute_side_effects: Callable[..., None],
) -> None:
    """A hung test turn is capped, so the run reaches the heal/abort path."""
    from daydream.runner import run

    silence(monkeypatch)
    monkeypatch.setattr("daydream.phases.DEFAULT_TOOL_CALL_BUDGET", 3)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.runaway_test = True
    mute_side_effects(heal=False)
    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"))
    assert exit_code != 0
    assert [c for c in stub.calls if "run the project's test suite" in c["prompt"].lower()]
    test_stop_reasons = _test_step_stop_reasons(multi_stack_target / ".daydream", traj)
    assert any("budget" in r for r in test_stop_reasons), test_stop_reasons


@pytest.mark.parametrize(
    "phase", ["intent", "alternatives", "per_stack", "merge", "arbiter", "suppression", "supervisor"],
)
@pytest.mark.parametrize("budget", ["wall", "tool_call"])
async def test_review_budget_stop_emits_partial_findings(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., RunConfig],
    phase: str,
    budget: str,
) -> None:
    """Real agent budget stops cannot strand completed findings before Phase B."""
    from collections.abc import AsyncIterator
    from typing import Any

    from daydream.backends import AgentEvent, ToolStartEvent
    from daydream.config_file import DaydreamFileConfig
    from daydream.runner import run
    from tests.harness.fake_clock import FakeClock
    from tests.harness.stub_backend import StubBackend
    from tests.test_deep_orchestrator import _pin_findings_pr

    fake = FakeClock().install(monkeypatch)
    fragments = {
        "intent": "present your understanding concisely",
        "alternatives": "evaluate the implementation",
        "per_stack": "you are reviewing the python stack",
        "merge": "cross-stack merge agent",
        "arbiter": "you are the arbiter",
        "suppression": "you are the suppression reviewer",
        "supervisor": "supervisor adjudication",
    }

    class BudgetBackend(StubBackend):
        exhaust = True

        async def execute(
            self, cwd: Path, prompt: str, *args: Any, **kwargs: Any,
        ) -> AsyncIterator[AgentEvent]:
            if self.exhaust and fragments[phase] in prompt.lower():
                for n in range(10):
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
        supervisor="llm" if phase == "supervisor" else "off",
        precision_mode=phase == "suppression",
    )
    monkeypatch.setattr("daydream.runner.create_backend", lambda *a, **kw: stub)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    monkeypatch.setattr("daydream.phases.DEFAULT_TOOL_CALL_BUDGET", 3 if budget == "tool_call" else None)
    _pin_findings_pr(monkeypatch, multi_stack_target)
    out = multi_stack_target / "findings.json"
    with anyio.fail_after(30):
        code = await run(make_config(
            multi_stack_target, pr_number=7, findings_out=str(out), file_config=file_config,
            review_profile=independent_alternatives_profile() if phase == "alternatives" else None,
        ))
    assert code == 0
    artifact = json.loads(out.read_text())
    assert artifact["review_warnings"], "budget stop must be visible to the poster"
    assert any(f"{budget}_budget_exceeded" in warning for warning in artifact["review_warnings"])
    assert artifact["findings"], "completed reviewers' findings must survive"
    assert "incomplete" in (multi_stack_target / ".review-output.md").read_text().lower()

    if phase == "merge":
        stub.exhaust = False
        assert await run(make_config(
            multi_stack_target, pr_number=7, findings_out=str(out), start_at="merge",
        )) == 0
        assert not json.loads(out.read_text()).get("review_warnings")


async def test_single_stack_alternatives_timeout_still_emits_findings(
    tiny_diff_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., RunConfig],
) -> None:
    from daydream.runner import run
    from tests.test_deep_orchestrator import _pin_findings_pr

    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, tiny_diff_target)
    stub.runaway_alternatives = True
    monkeypatch.setattr("daydream.phases.DEFAULT_TOOL_CALL_BUDGET", 3)
    _pin_findings_pr(monkeypatch, tiny_diff_target)
    out = tiny_diff_target / "findings.json"
    assert await run(make_config(
        tiny_diff_target, pr_number=7, findings_out=str(out), review_profile=independent_alternatives_profile(),
    )) == 0
    artifact = json.loads(out.read_text())
    assert artifact["review_warnings"] == ["Alternatives: tool_call_budget_exceeded"]


async def test_partial_checkpoint_survives_publication_and_merge_resume(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., RunConfig],
) -> None:
    from collections.abc import AsyncIterator
    from typing import Any

    from daydream.backends import AgentEvent, ToolStartEvent
    from daydream.runner import run
    from tests.harness.stub_backend import StubBackend
    from tests.test_deep_orchestrator import _pin_findings_pr

    class CheckpointBackend(StubBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event
            if "you are reviewing the python stack" in prompt.lower():
                for n in range(5):
                    yield ToolStartEvent(id=f"extra-{n}", name="Read", input={"file_path": "api.py"})

    silence(monkeypatch)
    backend = CheckpointBackend(multi_stack_target)
    backend.parse_severity = "high"
    backend.merge_echo_records = True
    monkeypatch.setattr("daydream.runner.create_backend", lambda *a, **kw: backend)
    monkeypatch.setattr("daydream.phases.DEFAULT_TOOL_CALL_BUDGET", 3)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    _pin_findings_pr(monkeypatch, multi_stack_target)
    out = multi_stack_target / "findings.json"
    for start_at in (None, "merge"):
        assert await run(make_config(
            multi_stack_target, pr_number=7, findings_out=str(out), start_at=start_at,
        )) == 0
        saved = json.loads((multi_stack_target / ".daydream/deep/stack-python-records.json").read_text())
        assert saved["issues"], "validated checkpoint must survive adjudication and resume"
        assert saved["incomplete"] is True
        published = json.loads(out.read_text())
        assert published["findings"]
        assert any("python" in warning for warning in published["review_warnings"])


async def test_spent_pipeline_budget_still_publishes_explicitly_incomplete_artifact(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., RunConfig],
) -> None:
    from daydream.runner import run
    from tests.test_deep_orchestrator import _pin_findings_pr, _profile_with_pipeline

    silence(monkeypatch)
    backend = install_stub_backend(monkeypatch, multi_stack_target)
    _pin_findings_pr(monkeypatch, multi_stack_target)
    out = multi_stack_target / "findings.json"
    assert await run(make_config(
        multi_stack_target, pr_number=7, findings_out=str(out),
        review_profile=_profile_with_pipeline(review_wall_budget_s=0),
    )) == 0
    assert not backend.calls
    artifact = json.loads(out.read_text())
    assert artifact["findings"] == []
    assert artifact["review_warnings"]
    assert "Review incomplete" in (multi_stack_target / ".review-output.md").read_text()
