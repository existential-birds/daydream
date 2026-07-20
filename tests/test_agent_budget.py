"""Tests for run_agent wall-clock + tool-call budgets (Task 3).

Both budgets live inside run_agent: the loop is wrapped in
``anyio.move_on_after(wall_budget_s)`` and a ToolStartEvent counter trips the
tool-call ceiling. On abort, the invocation's event iterator is closed, the
recorder Invocation is marked aborted via ``inv.mark_aborted(reason)``, and the
partial output is returned without cancelling sibling backend invocations.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    ResultEvent,
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

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


def _make_recorder(tmp_path: Path) -> TrajectoryRecorder:
    return TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id="test",
    )


def _agent_step_with_stop_reason(traj: dict[str, Any]) -> dict[str, Any]:
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
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


async def test_run_agent_abort_does_not_call_backend_cancel(tmp_path: Path) -> None:
    """An abort must not invoke the backend-wide cancellation API."""

    class _RaisingCancelBackend(_BurstBackend):
        async def cancel(self) -> None:
            self.cancel_calls += 1
            raise RuntimeError("cancel exploded")

    backend = _RaisingCancelBackend()
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


async def test_run_agent_closes_event_stream_before_abort_callback(tmp_path: Path) -> None:
    """Invocation resources are released before abort notification can block."""

    class _CloseTrackingBackend(_BurstBackend):
        stream_closed = False

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
                    self.stream_closed = True

            return _gen()

    backend = _CloseTrackingBackend()
    closed_during_abort_callback: list[bool] = []

    def progress_callback(message: Any) -> None:
        if "aborted" in str(message):
            closed_during_abort_callback.append(backend.stream_closed)

    await run_agent(
        backend,
        tmp_path,
        "go",
        phase=DaydreamPhase.FIX,
        tool_call_budget=0,
        progress_callback=progress_callback,
    )

    assert closed_during_abort_callback == [True]


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

        def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
            return f"/{skill_key}"

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
