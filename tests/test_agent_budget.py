"""Tests for run_agent wall-clock and tool-call budgets.

Both budgets live inside run_agent: the loop is wrapped in
``anyio.move_on_after(wall_budget_s)`` and a ToolStartEvent counter trips the
tool-call ceiling. On abort, the invocation's event iterator is closed, the
recorder Invocation is marked aborted via ``inv.mark_aborted(reason)``, and the
partial output is returned without cancelling sibling backend invocations.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    Backend,
    ContinuationToken,
    ResultEvent,
    RetryPolicy,
    TextEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    TrajectoryRecorder,
    _reset_recorder_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_recorder() -> Any:
    _reset_recorder_for_tests()
    yield
    _reset_recorder_for_tests()


@dataclass
class _BurstBackend:
    """Backend that yields many ToolStartEvents and never a ResultEvent.

    Optionally sleeps between events so a wall-clock budget can trip.
    """

    model = "mock-model"
    fanout_concurrency: int = 4
    count: int = 200
    sleep_s: float = 0.0
    cancel_calls: int = 0

    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
        persist_session: bool = True,
    ) -> AsyncGenerator[AgentEvent, None]:
        count = self.count
        sleep_s = self.sleep_s

        async def _gen() -> AsyncGenerator[AgentEvent, None]:
            for i in range(count):
                if sleep_s:
                    await anyio.sleep(sleep_s)
                yield ToolStartEvent(id=f"tool-{i}", name="Bash", input={"command": "ls"})

        return _gen()

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _RetryableBackendError(RuntimeError):
    """A retryable transport failure for the deadline-retry tests."""

    retryable = True


@dataclass
class _RetryableFailingBackend:
    """Backend that advances an injected clock per attempt, then fails retryably."""

    advance: Callable[[float], None]
    advance_s: float
    model = "mock-model"
    fanout_concurrency: int = 4
    calls: int = 0
    retry_policy: RetryPolicy = field(
        default_factory=lambda: RetryPolicy(attempts=20, base_delay_s=0.0, max_delay_s=0.0)
    )

    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
        persist_session: bool = True,
    ) -> AsyncGenerator[AgentEvent, None]:
        async def _gen() -> AsyncGenerator[AgentEvent, None]:
            self.calls += 1
            self.advance(self.advance_s)
            raise _RetryableBackendError("transient")
            yield  # pragma: no cover - unreachable, marks this a generator

        return _gen()

    async def cancel(self) -> None:
        pass


def _make_recorder(tmp_path: Path) -> TrajectoryRecorder:
    return TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id="test",
    )


def _agent_step_with_stop_reason(traj: dict[str, Any]) -> dict[str, Any]:
    agent_steps: list[dict[str, Any]] = [s for s in traj["steps"] if s["source"] == "agent"]
    for step in agent_steps:
        if step.get("extra", {}).get("stop_reason"):
            return step
    raise AssertionError(f"no agent step carried extra['stop_reason']: {agent_steps}")


async def test_run_agent_tool_call_ceiling(tmp_path: Path) -> None:
    """A 200-event burst with tool_call_budget=5 returns under budget, marked aborted."""
    backend = _BurstBackend(count=200, sleep_s=0.0)
    recorder = _make_recorder(tmp_path)

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

    class _RaisingCloseBackend(_BurstBackend):
        def execute(
            self,
            cwd: Path,
            prompt: str,
            output_schema: dict[str, Any] | None = None,
            continuation: ContinuationToken | None = None,
            agents: dict[str, Any] | None = None,
            max_turns: int | None = None,
            read_only: bool = False,
            persist_session: bool = True,
        ) -> AsyncGenerator[AgentEvent, None]:
            async def _gen() -> AsyncGenerator[AgentEvent, None]:
                try:
                    yield ToolStartEvent(id="tool-0", name="Bash", input={"command": "ls"})
                finally:
                    raise RuntimeError("stream close exploded")

            return _gen()

    backend = _RaisingCloseBackend()

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
        _BurstBackend(),
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
    from tests.harness.fake_clock import FakeClock

    fake = FakeClock(monotonic_value=1_000.0).install(monkeypatch)
    # retry_policy: attempts=20, delays=0.0; advances the injected clock per attempt.
    backend = _RetryableFailingBackend(advance=fake.advance, advance_s=400.0)

    output, _, reason = await run_agent(
        backend, tmp_path, "go",
        phase=DaydreamPhase.FIX,
        wall_budget_s=10_000.0,          # deliberately looser than the caller deadline
        deadline=1_600.0,                # absolute: fake clock starts at 1000.0
    )

    assert reason == "wall_budget_exceeded"
    assert backend.calls == 2            # 1000 -> 1400 (attempt 1) -> 1800 (attempt 2), then spent
    assert fake.monotonic_value == 1_800.0
    assert output == ""


async def test_budget_stop_records_the_limit_and_durations_but_no_monotonic_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.harness.fake_clock import FakeClock

    fake = FakeClock(monotonic_value=5_000.0).install(monkeypatch)
    backend = _RetryableFailingBackend(advance=fake.advance, advance_s=300.0)
    recorder = _make_recorder(tmp_path)

    with anyio.fail_after(5):
        async with recorder:
            _, _, reason = await run_agent(
                backend, tmp_path, "go", phase=DaydreamPhase.FIX,
                wall_budget_s=10_000.0, deadline=5_600.0,
            )

    assert reason == "wall_budget_exceeded"
    traj = json.loads(recorder.path.read_text(encoding="utf-8"))
    stops = [e for e in traj["extra"]["phase_events"] if e["event"] == "agent_budget_stop"]
    assert len(stops) == 1
    meta = stops[0]["metadata"]
    assert meta["limit_expired"] == "caller_deadline"
    assert meta["attempts"] == 2            # 5000 -> 5300 (attempt 1) -> 5600 (attempt 2), then spent
    assert meta["elapsed_s"] == 600.0 and meta["backend_s"] == 600.0
    assert "5600.0" not in recorder.path.read_text(encoding="utf-8")  # no reusable monotonic


async def test_run_agent_wall_budget(tmp_path: Path) -> None:
    """A slow stream with wall_budget_s=0.2 returns, step marked wall_budget_exceeded."""
    backend = _BurstBackend(count=200, sleep_s=0.05)
    recorder = _make_recorder(tmp_path)

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
    from tests.harness.fake_clock import FakeClock

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
    recorder = _make_recorder(tmp_path)

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
    traj = json.loads(recorder.path.read_text(encoding="utf-8"))
    assert _agent_step_with_stop_reason(traj)["extra"]["stop_reason"] == "wall_budget_exceeded"


async def test_aborting_invocation_does_not_cancel_shared_backend_sibling(
    tmp_path: Path,
) -> None:
    """An invocation budget abort closes its stream without cancelling a sibling."""

    class _SharedBackend:
        model = "mock-model"
        fanout_concurrency = 2

        def __init__(self) -> None:
            self.cancel_calls = 0
            self.closed_prompts: set[str] = set()
            self.sibling_started = anyio.Event()
            self.release_sibling = anyio.Event()

        def execute(
            self,
            cwd: Path,
            prompt: str,
            output_schema: dict[str, Any] | None = None,
            continuation: ContinuationToken | None = None,
            agents: dict[str, Any] | None = None,
            max_turns: int | None = None,
            read_only: bool = False,
            persist_session: bool = True,
        ) -> AsyncGenerator[AgentEvent, None]:
            async def _gen() -> AsyncGenerator[AgentEvent, None]:
                try:
                    if prompt == "sibling":
                        self.sibling_started.set()
                        await self.release_sibling.wait()
                        if self.cancel_calls:
                            return
                        yield TextEvent(text="sibling completed")
                        yield ResultEvent(structured_output=None, continuation=None)
                        return

                    await self.sibling_started.wait()
                    yield ToolStartEvent(id="tool-0", name="Bash", input={"command": "ls"})
                finally:
                    self.closed_prompts.add(prompt)

            return _gen()

        async def cancel(self) -> None:
            self.cancel_calls += 1

    backend = _SharedBackend()
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
        results["abort_iterator_closed"] = "abort" in backend.closed_prompts
        backend.release_sibling.set()

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
