"""Agent wall/tool budgets close only the current stream, record aborts, and retain partial output."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    ResultEvent,
    RetryPolicy,
    TextEvent,
    ToolStartEvent,
)
from daydream.backends.pi import PiBackend
from daydream.config import RETRY_CIRCUIT_FAILURE_THRESHOLD
from daydream.prompt_budget import prepare_sanctioned_inputs
from daydream.retry_policy import derive_retry_summary
from daydream.review_budget import ReviewInvestigationBudget, ReviewLimits
from daydream.run_context import InteractionPolicy, RunContext
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_clock import FakeClock, patch_retry_sleep
from tests.harness.pi_replay import make_mock_process, make_mock_process_from_fixture
from tests.harness.trajectory import make_recorder


class _RetryableBackendError(RuntimeError):
    """A retryable transport failure for the deadline-retry tests."""

    retryable = True


def _retryable_failing_backend(*, advance: Callable[[float], None], advance_s: float) -> ScriptedBackend:
    """A ScriptedBackend that advances an injected clock per attempt, then fails retryably."""
    async def responder(*args: Any, **kwargs: Any) -> Any:
        advance(advance_s)
        return [_RetryableBackendError("transient")]
    return ScriptedBackend(
        responder=responder, model="mock-model", fanout_concurrency=4,
        retry_policy=RetryPolicy(attempts=20, base_delay_s=0.0, max_delay_s=0.0),
    )


def _retryable_then_succeeding_backend(
    *, advance: Callable[[float], None], retry_advance_s: float, success_advance_s: float
) -> ScriptedBackend:
    """Attempt 1 fails retryably; attempt 2 spends a large, legitimate turn then succeeds."""
    attempts = 0
    async def responder(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            advance(retry_advance_s)
            return [_RetryableBackendError("transient")]
        advance(success_advance_s)
        return [TextEvent(text="done"), ResultEvent(structured_output=None, continuation=None)]
    return ScriptedBackend(
        responder=responder, model="mock-model", fanout_concurrency=4,
        retry_policy=RetryPolicy(attempts=20, base_delay_s=0.0, max_delay_s=0.0),
    )


def _agent_step_with_stop_reason(traj: dict[str, Any]) -> dict[str, Any]:
    agent_steps: list[dict[str, Any]] = [s for s in traj["steps"] if s["source"] == "agent"]
    for step in agent_steps:
        if step.get("extra", {}).get("stop_reason"):
            return step
    raise AssertionError(f"no agent step carried extra['stop_reason']: {agent_steps}")


def _budget_stop(recorder: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recorded trajectory plus the single ``agent_budget_stop`` metadata."""
    traj = json.loads(recorder.path.read_text(encoding="utf-8"))
    stops = [e for e in traj["extra"]["phase_events"] if e["event"] == "agent_budget_stop"]
    assert len(stops) == 1
    return traj, stops[0]["metadata"]


async def test_run_agent_tool_call_ceiling(tmp_path: Path) -> None:
    """A close fault cannot replace a budget stop or lose its already received tool evidence."""
    closed = False

    async def stream(*args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        nonlocal closed
        try:
            yield TextEvent(text="partial output")
            for index in range(8):
                yield ToolStartEvent(id=f"tool-{index}", name="Bash", input={"command": "ls"})
        finally:
            closed = True
            raise RuntimeError("stream close exploded")

    backend = ScriptedBackend(responder=stream, model="mock-model")
    recorder = make_recorder(tmp_path)
    with anyio.fail_after(5):
        async with recorder:
            output, continuation, reason = await run_agent(
                backend,
                tmp_path,
                "go",
                phase=DaydreamPhase.FIX,
                tool_call_budget=5,
                wall_budget_s=None,
            )
    assert (output, continuation, reason) == ("partial output", None, "tool_call_budget_exceeded")
    assert closed and backend.cancel_calls == 0
    trajectory = json.loads(recorder.path.read_text())
    step = _agent_step_with_stop_reason(trajectory)
    assert step["extra"]["stop_reason"] == "tool_call_budget_exceeded"
    assert step["message"] == "partial output"
    # The received over-limit start is retained as evidence even though it is not displayed or executed further.
    assert [call["tool_call_id"] for call in step["tool_calls"]] == [f"tool-{index}" for index in range(6)]
    assert trajectory["extra"]["partial"] is True


async def test_streaming_turn_is_cut_at_the_deadline_and_keeps_partial_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeClock(monotonic_value=1_000.0).install(monkeypatch)
    delivered = 0
    async def stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[AgentEvent]:
        nonlocal delivered
        for i in range(200):
            yield TextEvent(text=f"partial-output-sentinel-{i}")
            fake.advance(200.0)
            delivered += 1
            yield ToolStartEvent(id=f"t{i}", name="Bash", input={"command": "ls"})
    backend = ScriptedBackend(responder=stream, model="mock-model", fanout_concurrency=2)
    recorder = make_recorder(tmp_path)
    with anyio.fail_after(5):
        async with recorder:
            output, _, reason = await run_agent(
                backend, tmp_path, "go", phase=DaydreamPhase.FIX,
                wall_budget_s=600.0,          # deadline = 1000 + 600 = 1600
            )
    assert reason == "wall_budget_exceeded"
    assert delivered == 3             # 1000->1200->1400->1600: the 4th event is past it
    assert backend.cancel_calls == 0          # sibling isolation: no backend-wide cancel
    assert "partial-output-sentinel-2" in output   # text emitted before expiry survives
    traj, meta = _budget_stop(recorder)
    assert _agent_step_with_stop_reason(traj)["extra"]["stop_reason"] == "wall_budget_exceeded"
    # The deadline interrupted an in-flight attempt, so the stop record must say
    # its partials were kept -- the two deadline shapes are not interchangeable.
    assert meta["partial_edit_handling"] == "kept"
    assert meta["retry_stop_reason"] is None


async def test_expiry_cleanup_is_shielded_and_bounded_by_the_grace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeClock(monotonic_value=1_000.0)

    class _HangingCloseBackend:
        model = "mock-model"
        fanout_concurrency = 4
        def execute(self, *args: Any, **kwargs: Any) -> AsyncGenerator[AgentEvent, None]:
            async def _gen() -> AsyncGenerator[AgentEvent, None]:
                try:
                    yield TextEvent(text="partial-output-sentinel")
                    yield ToolStartEvent(id="t", name="Bash", input={"command": "ls"})
                    fake.advance(10.0)  # push the next event past the deadline
                    yield ToolStartEvent(id="t2", name="Bash", input={"command": "ls"})
                finally:
                    await anyio.sleep(3_600)  # teardown that never completes
            return _gen()
        async def cancel(self) -> None:
            pass

    fake.install(monkeypatch)
    monkeypatch.setattr("daydream.agent.BUDGET_CLEANUP_GRACE_S", 0.2)
    start = anyio.current_time()
    with anyio.fail_after(5):
        output, _, reason = await run_agent(
            _HangingCloseBackend(), tmp_path, "go", phase=DaydreamPhase.FIX, wall_budget_s=5.0
        )
    assert reason == "wall_budget_exceeded"
    assert "partial-output-sentinel" in output
    assert anyio.current_time() - start < 3.0  # bounded by the grace, not the teardown


async def test_aborting_invocation_does_not_cancel_shared_backend_sibling(tmp_path: Path) -> None:
    """An invocation budget abort closes its stream without cancelling a sibling."""
    sibling_started = anyio.Event()
    release_sibling = anyio.Event()
    closed_prompts: set[str] = set()
    backend_ref: list[ScriptedBackend] = []
    async def responder(cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> Any:
        async def _gen() -> AsyncGenerator[AgentEvent, None]:
            try:
                if prompt == "sibling":
                    sibling_started.set()
                    await release_sibling.wait()
                    if backend_ref[0].cancel_calls:
                        return
                    yield TextEvent(text="sibling completed")
                    yield ResultEvent(structured_output=None, continuation=None)
                    return
                await sibling_started.wait()
                yield ToolStartEvent(id="tool-0", name="Bash", input={"command": "ls"})
            finally:
                closed_prompts.add(prompt)
        return _gen()
    backend = ScriptedBackend(responder=responder, model="mock-model", fanout_concurrency=2)
    backend_ref.append(backend)
    results: dict[str, str | bool] = {}
    async def run_sibling() -> None:
        output, _, _ = await run_agent(backend, tmp_path, "sibling", phase=DaydreamPhase.REVIEW)
        results["sibling"] = output
    async def run_aborting_invocation() -> None:
        _, _, reason = await run_agent(backend, tmp_path, "abort", phase=DaydreamPhase.FIX, tool_call_budget=0)
        results["abort_reason"] = reason or ""
        results["abort_iterator_closed"] = "abort" in closed_prompts
        release_sibling.set()
    with anyio.fail_after(5):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(run_sibling)
            task_group.start_soon(run_aborting_invocation)
    assert results == {
        "sibling": "sibling completed", "abort_reason": "tool_call_budget_exceeded", "abort_iterator_closed": True,
    }
    assert backend.cancel_calls == 0


async def test_run_agent_cancellation_closes_only_its_invocation(tmp_path: Path) -> None:
    entered = asyncio.Event()
    closed = asyncio.Event()
    async def stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[AgentEvent]:
        try:
            entered.set()
            yield TextEvent(text="started")
            await asyncio.sleep(3600)
        finally:
            closed.set()
    backend = ScriptedBackend(responder=stream, model="mock-model")
    task = asyncio.create_task(run_agent(backend, tmp_path, "hi", phase=DaydreamPhase.REVIEW))
    await entered.wait()  # run_agent is inside the invocation
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert backend.cancel_calls == 0


async def test_retry_recovery_allowance_ends_the_ladder_without_dispatching_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeClock(monotonic_value=1_000.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent_retry._sample_retry_delay", lambda cap: cap)
    backend = _retryable_failing_backend(advance=fake.advance, advance_s=30.0)
    setattr(backend, "retry_policy", RetryPolicy(attempts=20, base_delay_s=60.0, max_delay_s=60.0))
    recorder = make_recorder(tmp_path)
    with pytest.raises(_RetryableBackendError):
        async with recorder:
            await run_agent(
                backend, tmp_path, "go", phase=DaydreamPhase.FIX,
                wall_budget_s=10_000.0, retry_recovery_allowance_s=60.0,
            )
    # 1000 (+30 attempt 1) -> backoff 60 -> 1090 (+30 attempt 2) -> allowance spent -> stop
    assert backend.call_count == 2
    assert slept == [60.0]
    assert fake.monotonic_value == 1_120.0
    _traj, meta = _budget_stop(recorder)
    assert meta["retry_stop_reason"] == "retry_recovery_allowance_exhausted"
    assert meta["partial_edit_handling"] == "discarded"
    assert meta["circuit_state"] == "closed"
    assert set(meta) >= {"attempts", "backend_s", "backoff_s", "elapsed_s"}
    assert str(fake.monotonic_value) not in recorder.path.read_text(encoding="utf-8")


async def test_retry_recovery_allowance_is_never_rebased_by_a_later_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent_retry._sample_retry_delay", lambda cap: cap)
    backend = _retryable_failing_backend(advance=fake.advance, advance_s=0.0)
    setattr(backend, "retry_policy", RetryPolicy(attempts=20, base_delay_s=40.0, max_delay_s=40.0))
    with pytest.raises(_RetryableBackendError):
        await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.FIX, retry_recovery_allowance_s=100.0)
    assert backend.call_count == 4  # 1 + 40 + 40 + 20, then the allowance is spent
    assert slept == [40.0, 40.0, 20.0]  # a re-basing implementation would sleep 40 forever


async def test_group_deadline_still_wins_over_a_larger_allowance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeClock(monotonic_value=1_000.0).install(monkeypatch)
    patch_retry_sleep(monkeypatch, fake)
    backend = _retryable_failing_backend(advance=fake.advance, advance_s=20.0)
    _, _, reason = await run_agent(
        backend, tmp_path, "go", phase=DaydreamPhase.FIX, deadline=1_030.0, retry_recovery_allowance_s=300.0,
    )
    assert reason == "wall_budget_exceeded"
    assert backend.call_count == 2  # stopped at the group deadline, not the allowance


async def test_a_healthy_invocation_is_never_capped_by_the_allowance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    patch_retry_sleep(monkeypatch, fake)
    backend = _retryable_then_succeeding_backend(advance=fake.advance, retry_advance_s=0.0, success_advance_s=1_200.0)
    output, _, reason = await run_agent(
        backend, tmp_path, "go", phase=DaydreamPhase.FIX, wall_budget_s=1_800.0, retry_recovery_allowance_s=300.0,
    )
    assert output == "done"
    assert reason is None  # 1200 s of legitimate post-retry work is not cancelled
    assert backend.call_count == 2


async def test_a_deadline_that_ends_a_retry_ladder_still_records_retry_telemetry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Deadline checks precede allowance checks but must still retain spent retry overhead."""
    fake = FakeClock(monotonic_value=5_000.0).install(monkeypatch)
    patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent_retry._sample_retry_delay", lambda cap: cap)
    # 5000 (+300 attempt 1) -> 5300 -> retry -> 5600 (+300 retry) -> deadline spent.
    backend = _retryable_failing_backend(advance=fake.advance, advance_s=300.0)
    recorder = make_recorder(tmp_path)
    with anyio.fail_after(5):
        async with recorder:
            output, _, reason = await run_agent(
                backend, tmp_path, "go", phase=DaydreamPhase.FIX, deadline=5_600.0, retry_recovery_allowance_s=300.0,
            )
    assert reason == "wall_budget_exceeded"
    assert backend.call_count == 2
    assert output == "" and fake.monotonic_value == 5_600.0
    traj, meta = _budget_stop(recorder)
    assert meta["elapsed_s"] == 600.0
    assert "5600.0" not in recorder.path.read_text(encoding="utf-8")
    assert meta["retry_stop_reason"] == "retry_deadline_exhausted"
    assert meta["limit_expired"] == "caller_deadline"
    # Retry-scoped counters: the one dispatched retry and its 300 s of backend
    # time -- never the first, useful-work attempt.
    assert meta["attempts"] == 1
    assert meta["backend_s"] == 300.0
    assert meta["backoff_s"] == 0.0
    assert meta["retry_recovery_spent_s"] == 300.0
    assert meta["partial_edit_handling"] == "discarded"
    summary = derive_retry_summary(traj["extra"]["phase_events"])
    assert summary is not None
    assert summary["stops"] == {"retry_deadline_exhausted": 1}
    assert summary["attempts"] == 1 and summary["backend_s"] == 300.0


async def test_a_deadline_that_cuts_the_ladder_during_backoff_is_still_a_ladder_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Backoff is retry overhead even when the deadline prevents the retry from dispatching."""
    fake = FakeClock(monotonic_value=1_000.0).install(monkeypatch)
    patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent_retry._sample_retry_delay", lambda cap: cap)
    # 1000 (+300 attempt 1) -> 1300 -> 100 s backoff -> 1400 = the deadline.
    backend = _retryable_failing_backend(advance=fake.advance, advance_s=300.0)
    setattr(backend, "retry_policy", RetryPolicy(attempts=20, base_delay_s=100.0, max_delay_s=100.0))
    recorder = make_recorder(tmp_path)
    with anyio.fail_after(5):
        async with recorder:
            _, _, reason = await run_agent(
                backend, tmp_path, "go", phase=DaydreamPhase.FIX, deadline=1_400.0, retry_recovery_allowance_s=300.0,
            )
    assert reason == "wall_budget_exceeded"
    assert backend.call_count == 1                  # the retry never got to dispatch
    traj, meta = _budget_stop(recorder)
    assert meta["retry_stop_reason"] == "retry_deadline_exhausted"
    assert meta["attempts"] == 0               # no retry attempt was dispatched
    assert meta["backend_s"] == 0.0
    assert meta["backoff_s"] == 100.0          # the sleep is the retry overhead
    assert meta["retry_recovery_spent_s"] == 100.0
    assert meta["partial_edit_handling"] == "discarded"
    assert derive_retry_summary(traj["extra"]["phase_events"]) is not None


async def test_a_zero_retry_ladder_stop_reports_no_retry_overhead(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Zero allowance records the stop reason without charging the initial attempt as retry overhead."""
    fake = FakeClock(monotonic_value=2_000.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    backend = _retryable_failing_backend(advance=fake.advance, advance_s=250.0)
    recorder = make_recorder(tmp_path)
    with anyio.fail_after(5):
        async with recorder:
            with pytest.raises(_RetryableBackendError):
                await run_agent(
                    backend, tmp_path, "go", phase=DaydreamPhase.FIX,
                    wall_budget_s=10_000.0, retry_recovery_allowance_s=0.0,
                )
    assert backend.call_count == 1 and slept == []
    traj, meta = _budget_stop(recorder)
    assert meta["retry_stop_reason"] == "retry_recovery_allowance_exhausted"
    assert meta["attempts"] == 0
    assert meta["backend_s"] == 0.0 and meta["backoff_s"] == 0.0
    assert meta["retry_recovery_spent_s"] == 0.0


async def test_initial_failed_attempt_deadline_records_discarded_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An initial failed attempt spending the wall budget has no retry stop reason."""
    fake = FakeClock(monotonic_value=1_000.0).install(monkeypatch)
    run_context = RunContext(InteractionPolicy(interactive=False))
    backend = _retryable_failing_backend(advance=fake.advance, advance_s=1_800.0)
    recorder = make_recorder(tmp_path)
    with anyio.fail_after(5):
        async with recorder:
            output, _, reason = await run_agent(
                backend, tmp_path, "go", phase=DaydreamPhase.FIX, wall_budget_s=1_800.0,
                retry_recovery_allowance_s=60.0, run_context=run_context,
            )
    assert output == "" and reason == "wall_budget_exceeded"
    assert backend.call_count == 1
    assert fake.monotonic_value == 2_800.0
    _traj, meta = _budget_stop(recorder)
    assert meta["retry_stop_reason"] is None
    assert meta["partial_edit_handling"] == "discarded"
    assert meta["circuit_state"] == "closed"
    assert set(meta) >= {"attempts", "backend_s", "backoff_s", "elapsed_s"}
    assert str(fake.monotonic_value) not in recorder.path.read_text(encoding="utf-8")  # no reusable monotonic


async def _expect_retryable_failure(backend: ScriptedBackend, tmp_path: Path, run_context: RunContext) -> None:
    """Drive one invocation to its circuit-suppressed failure."""
    with pytest.raises(_RetryableBackendError):
        await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.FIX, run_context=run_context)


async def test_concurrent_invocations_share_one_run_scoped_circuit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Failing siblings coordinate on one circuit instead of one ladder each."""
    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    # Checkpoint each fake sleep so sibling ladders interleave.
    inner_sleeper = anyio.sleep
    async def _yielding_sleeper(delay: float) -> None:
        await inner_sleeper(delay)
        await anyio.lowlevel.checkpoint()
    monkeypatch.setattr("daydream.agent.anyio.sleep", _yielding_sleeper)
    monkeypatch.setattr("daydream.agent_retry._sample_retry_delay", lambda cap: cap)
    run_context = RunContext(InteractionPolicy(interactive=False))
    backends = [
        _retryable_failing_backend(advance=fake.advance, advance_s=1.0) for _ in range(3)
    ]
    async with anyio.create_task_group() as tg:
        for backend in backends:
            tg.start_soon(_expect_retryable_failure, backend, tmp_path, run_context)
    assert sum(b.call_count for b in backends) <= 3 + 1  # threshold + the one probe
    assert len(slept) <= 3
    assert run_context.outage_circuit.state() in {"open", "half_open"}
    assert RunContext(InteractionPolicy(interactive=False)).outage_circuit.state() == "closed"


async def test_a_granted_half_open_probe_is_never_counted_as_its_own_failed_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A granted probe cannot fail before dispatch or release its token to another ladder.

    A zero probe interval isolates this grant invariant from interval waiting.
    """
    fake = FakeClock(monotonic_value=1_000.0).install(monkeypatch)
    patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent_retry._sample_retry_delay", lambda cap: cap)
    monkeypatch.setattr("daydream.config.RETRY_CIRCUIT_PROBE_INTERVAL_S", 0.0)
    run_context = RunContext(InteractionPolicy(interactive=False))
    # Open the circuit up front, then let the first ladder's retry be the probe.
    for _ in range(RETRY_CIRCUIT_FAILURE_THRESHOLD):
        run_context.outage_circuit.record_failure(fake.monotonic_value)
    assert run_context.outage_circuit.state() == "open"
    probe_ladder = _retryable_failing_backend(advance=fake.advance, advance_s=0.0)
    setattr(probe_ladder, "retry_policy", RetryPolicy(attempts=1, base_delay_s=0.0, max_delay_s=0.0))
    recorder = make_recorder(tmp_path)
    with anyio.fail_after(5):
        async with recorder:
            with contextlib.suppress(Exception):
                await run_agent(
                    probe_ladder, tmp_path, "go", phase=DaydreamPhase.FIX, retry_recovery_allowance_s=60.0,
                    run_context=run_context,
                )
    traj, meta = _budget_stop(recorder)
    assert probe_ladder.call_count == 2
    assert meta["retry_stop_reason"] == "retry_attempts_exhausted"
    # The probe remains outstanding after dispatch; its own grant cannot reopen the circuit.
    assert meta["circuit_state"] == "half_open"
    assert meta["partial_edit_handling"] == "discarded"
    assert set(meta) >= {"attempts", "backend_s", "backoff_s", "elapsed_s"}
    assert str(fake.monotonic_value) not in recorder.path.read_text(encoding="utf-8")
    # While that probe is outstanding, a second ladder gets no probe of its own.
    sibling = _retryable_failing_backend(advance=fake.advance, advance_s=0.0)
    setattr(sibling, "retry_policy", RetryPolicy(attempts=20, base_delay_s=0.0, max_delay_s=0.0))
    sibling_recorder = make_recorder(tmp_path)
    with anyio.fail_after(5):
        async with sibling_recorder:
            with contextlib.suppress(Exception):
                await run_agent(
                    sibling, tmp_path, "go", phase=DaydreamPhase.FIX, retry_recovery_allowance_s=60.0,
                    run_context=run_context,
                )
    _sibling_traj, sibling_meta = _budget_stop(sibling_recorder)
    assert sibling_meta["retry_stop_reason"] == "circuit_open"
    assert sibling_meta["partial_edit_handling"] == "discarded"
    assert sibling_meta["circuit_state"] == "open"
    assert set(sibling_meta) >= {"attempts", "backend_s", "backoff_s", "elapsed_s"}
    assert str(fake.monotonic_value) not in sibling_recorder.path.read_text(encoding="utf-8")
    assert sibling.call_count == 1  # suppressed before its own probe could dispatch


@pytest.mark.parametrize("bounded", [True, False], ids=["staged-deadline", "unbounded"])
async def test_pi_attempt_preamble_tracks_preparation_and_retry_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bounded: bool,
) -> None:
    """The shipped Pi system prompt describes enforcement, including a healthy retry."""
    fake = FakeClock(monotonic_value=1000).install(monkeypatch)
    patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent_retry._sample_retry_delay", lambda cap: cap)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "settings"))
    backend = PiBackend(model="fixture-model")
    backend.retry_attempts = 1
    backend.retry_base_delay_s = backend.retry_max_delay_s = 5
    source = tmp_path / "assigned.py"
    source.write_text("answer = 42\n")
    prepared = prepare_sanctioned_inputs(backend, tmp_path, {"assigned": source}, read_only=True)
    real_lstat = Path.lstat

    def delayed_lstat(path: Path, *args: Any, **kwargs: Any) -> Any:
        result = real_lstat(path, *args, **kwargs)
        if path == source:
            fake.advance(3)  # Real input revalidation consumes time before dispatch.
        return result

    monkeypatch.setattr(Path, "lstat", delayed_lstat)
    prompts: list[str] = []

    async def spawn(*args: str, **kwargs: Any) -> Any:
        system_arg = args[args.index("--append-system-prompt") + 1]
        prompts.append(Path(system_arg).read_text() if system_arg.startswith("/") else system_arg)
        if len(prompts) == 1:
            fake.advance(20)
            return make_mock_process([json.dumps(row) for row in [
                {"type": "session", "sessionId": "retry-fixture"},
                {"type": "turn_end", "message": {"role": "assistant", "content": [],
                 "stopReason": "error", "errorMessage": "503 Service Unavailable"}},
            ]])
        # The recovery allowance restricts the retry ladder, not healthy execution.
        fake.advance(100)
        return make_mock_process_from_fixture("simple_text.jsonl")

    monkeypatch.setattr("daydream.backends._transport.asyncio.create_subprocess_exec", spawn)
    budget = ReviewInvestigationBudget(ReviewLimits(960, 120, 96), deadline=1306.749)
    result, _, reason = await run_agent(
        backend, tmp_path, "Assigned work", phase=DaydreamPhase.DEEP, read_only=True,
        sanctioned_inputs=prepared, retry_recovery_allowance_s=10,
        wall_budget_s=3600 if bounded else None,
        investigation_budget=budget if bounded else None,
        deadline=1300 if bounded else None,  # Caller deadline wins over the staged 306.749s bound.
    )
    assert reason is None and result
    assert len(prompts) == 2
    if bounded:
        # Revalidation takes 3s each time; failed attempt + backoff takes another 25s.
        assert "effective wall allowance is 297 seconds" in prompts[0]
        assert "effective wall allowance is 269 seconds" in prompts[1]
    else:
        assert all("wall allowance is unbounded" in system for system in prompts)
        assert all("1800 seconds" not in system for system in prompts)


async def test_sanctioned_input_preparation_expiry_never_dispatches_pi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expiry while checking real input identity cannot launch a zero-allowance attempt."""
    fake = FakeClock(monotonic_value=1000).install(monkeypatch)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "settings"))
    backend = PiBackend(model="fixture-model")
    source = tmp_path / "assigned.py"
    source.write_text("answer = 42\n")
    prepared = prepare_sanctioned_inputs(backend, tmp_path, {"assigned": source}, read_only=True)
    real_lstat = Path.lstat

    def delayed_lstat(path: Path, *args: Any, **kwargs: Any) -> Any:
        result = real_lstat(path, *args, **kwargs)
        if path == source:
            fake.advance(5)
        return result

    monkeypatch.setattr(Path, "lstat", delayed_lstat)
    prompts: list[str] = []

    async def spawn(*args: str, **kwargs: Any) -> Any:
        prompts.append(str(args))
        return make_mock_process_from_fixture("simple_text.jsonl")

    monkeypatch.setattr("daydream.backends._transport.asyncio.create_subprocess_exec", spawn)
    recorder = make_recorder(tmp_path)
    async with recorder:
        assert (await run_agent(backend, tmp_path, "Earlier work", phase=DaydreamPhase.DEEP))[2] is None
        result, _, reason = await run_agent(backend, tmp_path, "Assigned work", phase=DaydreamPhase.DEEP,
            read_only=True, sanctioned_inputs=prepared, wall_budget_s=5)
    assert reason == "wall_budget_exceeded" and result == ""
    assert len(prompts) == 1  # Only the completed earlier invocation launched Pi.
    trajectory, stop = _budget_stop(recorder)
    assert [step["message"] for step in trajectory["steps"] if step["source"] == "user"] == ["Earlier work"]
    assert stop["attempts"] == 0 and stop["backend_s"] == 0
    assert stop["partial_edit_handling"] == "discarded"
