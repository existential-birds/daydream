"""Tests for run_agent wall-clock and tool-call budgets.

Both budgets live inside run_agent: the loop is wrapped in
``anyio.move_on_after(wall_budget_s)`` and a ToolStartEvent counter trips the
tool-call ceiling. On abort, the invocation's event iterator is closed, the
recorder Invocation is marked aborted via ``inv.mark_aborted(reason)``, and the
partial output is returned without cancelling sibling backend invocations.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    Backend,
    ResultEvent,
    RetryPolicy,
    TextEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.config import RETRY_CIRCUIT_FAILURE_THRESHOLD
from daydream.retry_policy import derive_retry_summary
from daydream.run_context import InteractionPolicy, RunContext
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_clock import FakeClock, patch_retry_sleep
from tests.harness.trajectory import make_recorder


def _burst_backend(*, count: int = 200, sleep_s: float = 0.0) -> ScriptedBackend:
    """A ScriptedBackend that streams many ToolStartEvents and never a ResultEvent.

    With no ``sleep_s`` the stream is a plain ``events=`` turn; a positive
    ``sleep_s`` streams through a responder so a wall-clock budget can trip.
    """
    if not sleep_s:
        events: list[AgentEvent] = [
            ToolStartEvent(id=f"tool-{i}", name="Bash", input={"command": "ls"}) for i in range(count)
        ]
        return ScriptedBackend(events=events, model="mock-model", fanout_concurrency=4)

    async def responder(*args: Any, **kwargs: Any) -> Any:
        async def _gen() -> AsyncGenerator[AgentEvent, None]:
            for i in range(count):
                await anyio.sleep(sleep_s)
                yield ToolStartEvent(id=f"tool-{i}", name="Bash", input={"command": "ls"})

        return _gen()

    return ScriptedBackend(responder=responder, model="mock-model", fanout_concurrency=4)


class _RetryableBackendError(RuntimeError):
    """A retryable transport failure for the deadline-retry tests."""

    retryable = True


def _retryable_failing_backend(*, advance: Callable[[float], None], advance_s: float) -> ScriptedBackend:
    """A ScriptedBackend that advances an injected clock per attempt, then fails retryably."""

    async def responder(*args: Any, **kwargs: Any) -> Any:
        advance(advance_s)
        return [_RetryableBackendError("transient")]

    return ScriptedBackend(
        responder=responder,
        model="mock-model",
        fanout_concurrency=4,
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
        responder=responder,
        model="mock-model",
        fanout_concurrency=4,
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
    """A 200-event burst with tool_call_budget=5 returns under budget, marked aborted."""
    backend = _burst_backend(count=200, sleep_s=0.0)
    recorder = make_recorder(tmp_path)

    with anyio.fail_after(5):
        async with recorder:
            result, _, _ = await run_agent(
                backend,
                tmp_path,
                "go",
                phase=DaydreamPhase.FIX,
                tool_call_budget=5,
                wall_budget_s=None,
            )

    assert isinstance(result, str)
    assert backend.cancel_calls == 0
    traj = json.loads(recorder.path.read_text(encoding="utf-8"))
    step = _agent_step_with_stop_reason(traj)
    assert step["extra"]["stop_reason"] == "tool_call_budget_exceeded"


async def test_run_agent_abort_swallows_event_stream_close_error(tmp_path: Path) -> None:
    """Invocation cleanup failures must not replace the successful abort result."""

    async def responder(*args: Any, **kwargs: Any) -> Any:
        async def _gen() -> AsyncGenerator[AgentEvent, None]:
            try:
                yield ToolStartEvent(id="tool-0", name="Bash", input={"command": "ls"})
            finally:
                raise RuntimeError("stream close exploded")

        return _gen()

    backend = ScriptedBackend(responder=responder, model="mock-model", fanout_concurrency=4)

    with anyio.fail_after(5):
        result, _, reason = await run_agent(
            backend,
            tmp_path,
            "go",
            phase=DaydreamPhase.FIX,
            tool_call_budget=0,
        )

    assert isinstance(result, str)
    assert reason == "tool_call_budget_exceeded"
    assert backend.cancel_calls == 0


async def test_run_agent_abort_records_reason_and_turn_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Abort bookkeeping preserves both the reason and synthetic turn boundary."""

    class _InvocationSpy:
        def __init__(self) -> None:
            self.abort_reasons: list[str] = []
            self.events: list[AgentEvent] = []

        async def __aenter__(self) -> _InvocationSpy:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        def observe_user_step(self, *, prompt: str) -> None:
            pass

        def mark_aborted(self, reason: str) -> None:
            self.abort_reasons.append(reason)

        def observe(self, event: AgentEvent) -> None:
            self.events.append(event)

    invocation = _InvocationSpy()

    class _RecorderSpy:
        def invocation(self, *, phase: DaydreamPhase) -> _InvocationSpy:
            return invocation

    monkeypatch.setattr("daydream.agent.get_current_recorder", lambda: _RecorderSpy())

    await run_agent(
        _burst_backend(),
        tmp_path,
        "go",
        phase=DaydreamPhase.FIX,
        tool_call_budget=0,
        progress_callback=lambda _: None,
    )

    assert invocation.abort_reasons == ["tool_call_budget_exceeded"]
    assert isinstance(invocation.events[-1], TurnEndEvent)


async def test_caller_deadline_bounds_attempts_and_is_not_restarted_by_a_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A caller deadline spans the whole retry ladder; a retry cannot restart it."""

    fake = FakeClock(monotonic_value=1_000.0).install(monkeypatch)
    # retry_policy: attempts=20, delays=0.0; advances the injected clock per attempt.
    backend = _retryable_failing_backend(advance=fake.advance, advance_s=400.0)

    output, _, reason = await run_agent(
        backend, tmp_path, "go",
        phase=DaydreamPhase.FIX,
        wall_budget_s=10_000.0,          # deliberately looser than the caller deadline
        deadline=1_600.0,                # absolute: fake clock starts at 1000.0
    )

    assert reason == "wall_budget_exceeded"
    assert backend.call_count == 2            # 1000 -> 1400 (attempt 1) -> 1800 (attempt 2), then spent
    assert fake.monotonic_value == 1_800.0
    assert output == ""


async def test_budget_stop_records_the_limit_and_durations_but_no_monotonic_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    fake = FakeClock(monotonic_value=5_000.0).install(monkeypatch)
    backend = _retryable_failing_backend(advance=fake.advance, advance_s=300.0)
    recorder = make_recorder(tmp_path)

    with anyio.fail_after(5):
        async with recorder:
            _, _, reason = await run_agent(
                backend, tmp_path, "go", phase=DaydreamPhase.FIX,
                wall_budget_s=10_000.0, deadline=5_600.0,
            )

    assert reason == "wall_budget_exceeded"
    traj, meta = _budget_stop(recorder)
    # The deadline is the limit that expired, and the ladder had already flown one
    # retry, so this is a retry-ladder stop: its counters are retry-scoped.
    assert meta["limit_expired"] == "caller_deadline"
    assert meta["retry_stop_reason"] == "retry_deadline_exhausted"
    assert meta["attempts"] == 1            # 5000 -> 5300 (attempt 1) -> 5600 (retry), then spent
    assert meta["elapsed_s"] == 600.0 and meta["backend_s"] == 300.0
    assert "5600.0" not in recorder.path.read_text(encoding="utf-8")  # no reusable monotonic


async def test_run_agent_wall_budget(tmp_path: Path) -> None:
    """A slow stream with wall_budget_s=0.2 returns, step marked wall_budget_exceeded."""
    backend = _burst_backend(count=200, sleep_s=0.05)
    recorder = make_recorder(tmp_path)

    with anyio.fail_after(5):
        async with recorder:
            result, _, _ = await run_agent(
                backend,
                tmp_path,
                "go",
                phase=DaydreamPhase.FIX,
                wall_budget_s=0.2,
                tool_call_budget=None,
            )

    assert isinstance(result, str)
    assert backend.cancel_calls == 0
    traj = json.loads(recorder.path.read_text(encoding="utf-8"))
    step = _agent_step_with_stop_reason(traj)
    assert step["extra"]["stop_reason"] == "wall_budget_exceeded"


async def test_streaming_turn_is_cut_at_the_deadline_and_keeps_partial_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    class _ClockAdvancingBurstBackend:
        """Streams text + tool starts, advancing the injected clock per event."""

        model = "mock-model"
        fanout_concurrency = 2

        def __init__(self, advance: Any, advance_s: float) -> None:
            self.advance, self.advance_s, self.delivered, self.cancel_calls = advance, advance_s, 0, 0

        def execute(self, *args: Any, **kwargs: Any) -> AsyncGenerator[AgentEvent, None]:
            async def _gen() -> AsyncGenerator[AgentEvent, None]:
                for i in range(200):
                    yield TextEvent(text=f"partial-output-sentinel-{i}")
                    self.advance(self.advance_s)
                    self.delivered += 1
                    yield ToolStartEvent(id=f"t{i}", name="Bash", input={"command": "ls"})

            return _gen()

        async def cancel(self) -> None:
            self.cancel_calls += 1

    fake = FakeClock(monotonic_value=1_000.0).install(monkeypatch)
    backend = _ClockAdvancingBurstBackend(fake.advance, 200.0)
    recorder = make_recorder(tmp_path)

    with anyio.fail_after(5):
        async with recorder:
            output, _, reason = await run_agent(
                backend, tmp_path, "go",
                phase=DaydreamPhase.FIX,
                wall_budget_s=600.0,          # deadline = 1000 + 600 = 1600
            )

    assert reason == "wall_budget_exceeded"
    assert backend.delivered == 3             # 1000->1200->1400->1600: the 4th event is past it
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
    assert anyio.current_time() - start < 3.0   # bounded by the grace, not the teardown


async def test_aborting_invocation_does_not_cancel_shared_backend_sibling(
    tmp_path: Path,
) -> None:
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
        output, _, _ = await run_agent(
            backend,
            tmp_path,
            "sibling",
            phase=DaydreamPhase.REVIEW,
        )
        results["sibling"] = output

    async def run_aborting_invocation() -> None:
        _, _, reason = await run_agent(
            backend,
            tmp_path,
            "abort",
            phase=DaydreamPhase.FIX,
            tool_call_budget=0,
        )
        results["abort_reason"] = reason or ""
        results["abort_iterator_closed"] = "abort" in closed_prompts
        release_sibling.set()

    with anyio.fail_after(5):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(run_sibling)
            task_group.start_soon(run_aborting_invocation)

    assert results == {
        "sibling": "sibling completed",
        "abort_reason": "tool_call_budget_exceeded",
        "abort_iterator_closed": True,
    }
    assert backend.cancel_calls == 0


class _RecordingCancelBackend:
    """Minimal Backend that records cancel() calls and blocks forever in execute."""

    model = "mock-model"

    def __init__(self) -> None:
        self.cancelled = False
        self.entered = asyncio.Event()

    async def execute(self, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        self.entered.set()
        yield TextEvent(text="started")
        await asyncio.sleep(3600)

    async def cancel(self) -> None:
        self.cancelled = True


async def test_run_agent_cancellation_awaits_backend_cancel(tmp_path: Path) -> None:
    """Task cancellation (SIGINT unwind) deterministically cancels the backend."""
    backend = _RecordingCancelBackend()

    task = asyncio.create_task(
        run_agent(
            cast(Backend, backend),
            tmp_path,
            "hi",
            phase=DaydreamPhase.REVIEW,
        )
    )
    await backend.entered.wait()  # run_agent is inside the invocation
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert backend.cancelled is True


async def test_retry_recovery_allowance_ends_the_ladder_without_dispatching_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    fake = FakeClock(monotonic_value=1_000.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent._sample_retry_delay", lambda cap: cap)
    backend = _retryable_failing_backend(advance=fake.advance, advance_s=30.0)
    setattr(backend, "retry_policy", RetryPolicy(attempts=20, base_delay_s=60.0, max_delay_s=60.0))

    with pytest.raises(_RetryableBackendError):
        await run_agent(
            backend, tmp_path, "go", phase=DaydreamPhase.FIX,
            wall_budget_s=10_000.0, retry_recovery_allowance_s=60.0,
        )

    # 1000 (+30 attempt 1) -> backoff 60 -> 1090 (+30 attempt 2) -> allowance spent -> stop
    assert backend.call_count == 2
    assert slept == [60.0]
    assert fake.monotonic_value == 1_120.0


async def test_retry_recovery_allowance_is_never_rebased_by_a_later_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent._sample_retry_delay", lambda cap: cap)
    backend = _retryable_failing_backend(advance=fake.advance, advance_s=0.0)
    setattr(backend, "retry_policy", RetryPolicy(attempts=20, base_delay_s=40.0, max_delay_s=40.0))

    with pytest.raises(_RetryableBackendError):
        await run_agent(
            backend, tmp_path, "go", phase=DaydreamPhase.FIX,
            retry_recovery_allowance_s=100.0,
        )

    assert backend.call_count == 4  # 1 + 40 + 40 + 20, then the allowance is spent
    assert slept == [40.0, 40.0, 20.0]  # a re-basing implementation would sleep 40 forever


async def test_group_deadline_still_wins_over_a_larger_allowance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    fake = FakeClock(monotonic_value=1_000.0).install(monkeypatch)
    patch_retry_sleep(monkeypatch, fake)
    backend = _retryable_failing_backend(advance=fake.advance, advance_s=20.0)
    setattr(backend, "retry_policy", RetryPolicy(attempts=20, base_delay_s=0.0, max_delay_s=0.0))

    _, _, reason = await run_agent(
        backend, tmp_path, "go", phase=DaydreamPhase.FIX,
        deadline=1_030.0, retry_recovery_allowance_s=300.0,
    )

    assert reason == "wall_budget_exceeded"
    assert backend.call_count == 2  # stopped at the group deadline, not the allowance


async def test_a_healthy_invocation_is_never_capped_by_the_allowance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    patch_retry_sleep(monkeypatch, fake)
    backend = _retryable_then_succeeding_backend(
        advance=fake.advance, retry_advance_s=0.0, success_advance_s=1_200.0
    )

    output, _, reason = await run_agent(
        backend, tmp_path, "go", phase=DaydreamPhase.FIX,
        wall_budget_s=1_800.0, retry_recovery_allowance_s=300.0,
    )

    assert output == "done"
    assert reason is None  # 1200 s of legitimate post-retry work is not cancelled
    assert backend.call_count == 2


async def test_a_deadline_that_ends_a_retry_ladder_still_records_retry_telemetry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ladder cut by the deadline after a retry keeps its retry summary.

    The deadline check runs before the allowance check inside the retry branch, so
    a ladder that already flew a retry and then overran the group wall used to end
    on a reason-less deadline stop: the manifest's retry summary was silently
    erased even though real retry overhead had been spent. The stop is now a
    retry-ladder stop naming the deadline, and the reducer folds it in.
    """

    fake = FakeClock(monotonic_value=5_000.0).install(monkeypatch)
    patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent._sample_retry_delay", lambda cap: cap)
    # 5000 (+300 attempt 1) -> 5300 -> retry -> 5600 (+300 retry) -> deadline spent.
    backend = _retryable_failing_backend(advance=fake.advance, advance_s=300.0)
    setattr(backend, "retry_policy", RetryPolicy(attempts=20, base_delay_s=0.0, max_delay_s=0.0))
    recorder = make_recorder(tmp_path)

    with anyio.fail_after(5):
        async with recorder:
            _, _, reason = await run_agent(
                backend, tmp_path, "go", phase=DaydreamPhase.FIX,
                deadline=5_600.0, retry_recovery_allowance_s=300.0,
            )

    assert reason == "wall_budget_exceeded"
    assert backend.call_count == 2
    traj, meta = _budget_stop(recorder)
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
    """A backoff sleep that overruns the deadline is retry overhead too.

    The ladder spent its allowance on a real backoff sleep and then found the
    deadline gone before it could dispatch the retry: the stop must name that
    ending rather than masquerading as a plain deadline stop that erased the
    retry summary.
    """

    fake = FakeClock(monotonic_value=1_000.0).install(monkeypatch)
    patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent._sample_retry_delay", lambda cap: cap)
    # 1000 (+300 attempt 1) -> 1300 -> 100 s backoff -> 1400 = the deadline.
    backend = _retryable_failing_backend(advance=fake.advance, advance_s=300.0)
    setattr(backend, "retry_policy", RetryPolicy(attempts=20, base_delay_s=100.0, max_delay_s=100.0))
    recorder = make_recorder(tmp_path)

    with anyio.fail_after(5):
        async with recorder:
            _, _, reason = await run_agent(
                backend, tmp_path, "go", phase=DaydreamPhase.FIX,
                deadline=1_400.0, retry_recovery_allowance_s=300.0,
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
    """``allowance = 0`` stops the ladder on the first failure with no retry cost.

    The stop still names its reason (the manifest must show why recovery ended),
    but every counter is zero: the failed attempt is useful work, not retry
    overhead, so the summary cannot claim a retry that never happened.
    """

    fake = FakeClock(monotonic_value=2_000.0).install(monkeypatch)
    backend = _retryable_failing_backend(advance=fake.advance, advance_s=250.0)
    recorder = make_recorder(tmp_path)

    with anyio.fail_after(5):
        async with recorder:
            with pytest.raises(_RetryableBackendError):
                await run_agent(
                    backend, tmp_path, "go", phase=DaydreamPhase.FIX,
                    wall_budget_s=10_000.0, retry_recovery_allowance_s=0.0,
                )

    assert backend.call_count == 1
    traj, meta = _budget_stop(recorder)
    assert meta["retry_stop_reason"] == "retry_recovery_allowance_exhausted"
    assert meta["attempts"] == 0
    assert meta["backend_s"] == 0.0 and meta["backoff_s"] == 0.0
    assert meta["retry_recovery_spent_s"] == 0.0


def _ending_backend(
    monkeypatch: pytest.MonkeyPatch, ending: str
) -> tuple[Any, ScriptedBackend, RunContext]:
    """Build ``(fake_clock, backend, run_context)`` for one ladder ending."""

    fake = FakeClock(monotonic_value=1_000.0).install(monkeypatch)
    patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent._sample_retry_delay", lambda cap: cap)
    run_context = RunContext(InteractionPolicy(interactive=False))
    if ending == "deadline":
        # The 1_800 s wall budget is spent inside the first dispatched attempt.
        backend = _retryable_failing_backend(advance=fake.advance, advance_s=1_800.0)
    elif ending == "allowance":
        # The 60 s allowance is spent by the first 60 s backoff sleep.
        backend = _retryable_failing_backend(advance=fake.advance, advance_s=30.0)
        setattr(backend, "retry_policy", RetryPolicy(attempts=20, base_delay_s=60.0, max_delay_s=60.0))
    elif ending == "attempts":
        backend = _retryable_failing_backend(advance=fake.advance, advance_s=0.0)
        setattr(backend, "retry_policy", RetryPolicy(attempts=1, base_delay_s=0.0, max_delay_s=0.0))
    elif ending == "circuit":
        # Open the one run-scoped circuit first; the ladder is then suppressed.
        backend = _retryable_failing_backend(advance=fake.advance, advance_s=0.0)
        circuit = run_context.outage_circuit
        for _ in range(RETRY_CIRCUIT_FAILURE_THRESHOLD):
            circuit.record_failure(fake.monotonic_value)
    else:  # pragma: no cover - parametrization is closed
        raise AssertionError(ending)
    return fake, backend, run_context


@pytest.mark.parametrize(
    ("ending", "expected_stop", "expected_partial"),
    [
        # The 1800 s wall budget is spent by the first attempt's own advance, so
        # the ladder ends before any retry is dispatched: the deadline stop is
        # reason-less and the discarded-partials shape (the loop-top reset already
        # wiped the failed attempt's partial output).
        pytest.param("deadline", None, "discarded", id="deadline"),
        pytest.param("allowance", "retry_recovery_allowance_exhausted", "discarded", id="allowance"),
        pytest.param("attempts", "retry_attempts_exhausted", "discarded", id="attempts"),
        pytest.param("circuit", "circuit_open", "discarded", id="circuit"),
    ],
)
async def test_every_ladder_ending_records_one_budget_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ending: str, expected_stop: str | None, expected_partial: str
) -> None:
    """Each way a retry ladder ends leaves exactly one budget-stop record."""
    fake, backend, run_context = _ending_backend(monkeypatch, ending)
    recorder = make_recorder(tmp_path)

    with anyio.fail_after(5):
        async with recorder:
            with contextlib.suppress(Exception):
                await run_agent(
                    backend,
                    tmp_path,
                    "go",
                    phase=DaydreamPhase.FIX,
                    wall_budget_s=1_800.0,
                    retry_recovery_allowance_s=60.0,
                    run_context=run_context,
                )

    traj, meta = _budget_stop(recorder)
    assert meta["retry_stop_reason"] == expected_stop
    assert meta["partial_edit_handling"] == expected_partial
    assert meta["circuit_state"] in {"closed", "open", "half_open"}
    assert set(meta) >= {"attempts", "backend_s", "backoff_s", "elapsed_s"}
    assert str(fake.monotonic_value) not in recorder.path.read_text(encoding="utf-8")  # no reusable monotonic


async def _expect_retryable_failure(
    backend: ScriptedBackend, tmp_path: Path, run_context: RunContext
) -> None:
    """Drive one invocation to its circuit-suppressed failure."""
    with pytest.raises(_RetryableBackendError):
        await run_agent(
            backend, tmp_path, "go", phase=DaydreamPhase.FIX, run_context=run_context
        )


async def test_concurrent_invocations_share_one_run_scoped_circuit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Failing siblings coordinate on one circuit instead of one ladder each."""

    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    # Yield to the event loop after each injected sleep so the sibling
    # invocations actually interleave; without the checkpoint the first task
    # would run its whole ladder before any sibling starts.
    inner_sleeper = anyio.sleep

    async def _yielding_sleeper(delay: float) -> None:
        await inner_sleeper(delay)
        await anyio.lowlevel.checkpoint()

    monkeypatch.setattr("daydream.agent.anyio.sleep", _yielding_sleeper)
    monkeypatch.setattr("daydream.agent._sample_retry_delay", lambda cap: cap)
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


async def test_a_granted_half_open_probe_is_never_counted_as_its_own_failed_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ladder granted the half-open probe must not re-open the circuit itself.

    The probe has not dispatched yet when the grant is made, so recording that
    same failure as a *failed probe* re-opens the circuit on the spot, clears the
    probe token (letting a concurrent ladder fly a second probe) and restarts the
    interval — the documented "exactly one probe per interval" contract could then
    never execute. ``RETRY_CIRCUIT_PROBE_INTERVAL_S = 0`` isolates the grant from
    the interval wait: an open circuit admits a probe on the very next retry.
    """

    fake = FakeClock(monotonic_value=1_000.0).install(monkeypatch)
    patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent._sample_retry_delay", lambda cap: cap)
    monkeypatch.setattr("daydream.config.RETRY_CIRCUIT_PROBE_INTERVAL_S", 0.0)
    run_context = RunContext(InteractionPolicy(interactive=False))
    # Open the circuit up front, then let the first ladder's retry be the probe.

    for _ in range(RETRY_CIRCUIT_FAILURE_THRESHOLD):
        run_context.outage_circuit.record_failure(fake.monotonic_value)

    probe_ladder = _retryable_failing_backend(advance=fake.advance, advance_s=0.0)
    setattr(probe_ladder, "retry_policy", RetryPolicy(attempts=1, base_delay_s=0.0, max_delay_s=0.0))
    recorder = make_recorder(tmp_path)

    with anyio.fail_after(5):
        async with recorder:
            with contextlib.suppress(Exception):
                await run_agent(
                    probe_ladder,
                    tmp_path,
                    "go",
                    phase=DaydreamPhase.FIX,
                    retry_recovery_allowance_s=60.0,
                    run_context=run_context,
                )

    traj, meta = _budget_stop(recorder)
    assert meta["retry_stop_reason"] == "retry_attempts_exhausted"
    # The half-open grant survives its own dispatch: the probe is still outstanding,
    # so the state reported at the next stop is the probe's, not a re-opened circuit.
    assert meta["circuit_state"] == "half_open"

    # While that probe is outstanding, a second ladder gets no probe of its own.
    sibling = _retryable_failing_backend(advance=fake.advance, advance_s=0.0)
    setattr(sibling, "retry_policy", RetryPolicy(attempts=20, base_delay_s=0.0, max_delay_s=0.0))
    sibling_recorder = make_recorder(tmp_path)

    with anyio.fail_after(5):
        async with sibling_recorder:
            with contextlib.suppress(Exception):
                await run_agent(
                    sibling,
                    tmp_path,
                    "go",
                    phase=DaydreamPhase.FIX,
                    retry_recovery_allowance_s=60.0,
                    run_context=run_context,
                )

    _sibling_traj, sibling_meta = _budget_stop(sibling_recorder)
    assert sibling_meta["retry_stop_reason"] == "circuit_open"
    assert sibling.call_count == 1  # suppressed before its own probe could dispatch


async def test_a_fresh_run_starts_closed(tmp_path: Path) -> None:
    assert (
        RunContext(InteractionPolicy(interactive=False)).outage_circuit.state()
        == "closed"
    )
