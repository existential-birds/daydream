"""Agent wall/tool budgets close only the current stream, record aborts, and retain partial output."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import anyio
import httpx
import pytest

from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    ResultEvent,
    RetryPolicy,
    TextEvent,
    ToolStartEvent,
)
from daydream.backends.codex import CodexBackend
from daydream.backends.pi import PiBackend
from daydream.config import RETRY_CIRCUIT_FAILURE_THRESHOLD
from daydream.host_tools import NATIVE_SUBMIT, HostToolBinding, HostToolFatal, HostToolRejected
from daydream.phases.review_prompts import build_review_stage_system_instruction
from daydream.prompt_budget import prepare_sanctioned_inputs
from daydream.prompts.grounding import REVIEW_STOPPING_GUIDANCE
from daydream.retry_policy import RetryRecoveryBudget, derive_retry_summary
from daydream.review_budget import ReviewInvestigationBudget, ReviewLimits
from daydream.run_context import InteractionPolicy, RunContext
from daydream.supervision import RuleBasedToolSupervisor, apply_findings_verdicts
from daydream.trajectory import DaydreamPhase, RunWriteSnapshot
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_cli_process import FakeCliScript, install_fake_cli_process
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

async def test_run_agent_abort_records_reason_and_turn_end(tmp_path: Path) -> None:
    snapshots: list[RunWriteSnapshot] = []
    recorder = make_recorder(tmp_path, on_write=lambda _recorder, snapshot: snapshots.append(snapshot))

    async def stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[AgentEvent]:
        try:
            yield ToolStartEvent(id="tool-0", name="Bash", input={"command": "ls"})
        finally:
            recorder.write_partial()

    backend = ScriptedBackend(responder=stream, model="mock-model")
    async with recorder:
        _, _, reason = await run_agent(
            backend, tmp_path, "go", phase=DaydreamPhase.FIX, tool_call_budget=0, progress_callback=lambda _: None,
        )
    assert reason == "tool_call_budget_exceeded"
    partial = next(snapshot for snapshot in snapshots if snapshot.status == "partial")
    cleanup_document = json.loads(partial.documents[0].read_bytes())
    final_document = json.loads(recorder.path.read_bytes())
    for document in (cleanup_document, final_document):
        step = _agent_step_with_stop_reason(document)
        assert step["extra"]["stop_reason"] == "tool_call_budget_exceeded"
        assert "partial_step" not in step["extra"]
        assert [tool["tool_call_id"] for tool in step["tool_calls"]] == ["tool-0"]
        interruption, = step["observation"]["results"]
        assert interruption["extra"] == {"is_error": True, "status": "interrupted"}
        assert document["extra"]["partial"] is True


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


@pytest.mark.parametrize(("bound", "backoff"), [(False, False), (True, False), (True, True)],
                         ids=["ordinary-consumer", "host-tools-consumer", "host-tools-backoff"])
async def test_run_agent_cancellation_closes_only_its_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound: bool, backoff: bool,
) -> None:
    """Unlike cancellation inside iteration, consumer/backoff waits need host-owned closure before joining."""
    entered = asyncio.Event()
    closed = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_state: list[tuple[bool, bool]] = []
    binding = HostToolBinding(deadline=anyio.current_time() + 60, accept=lambda payload: payload) if bound else None
    if backoff:
        monkeypatch.setattr("daydream.agent_retry._sample_retry_delay", lambda cap: cap)

    async def stream(*_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent, None]:
        async with binding.session() if binding is not None else contextlib.nullcontext():
            try:
                if binding is not None:
                    binding.admit("read-1", "read", {"path": "assigned.py"})
                yield ToolStartEvent(id="read-1", name="read", input={"path": "assigned.py"})
                if backoff:
                    raise _RetryableBackendError("retry before cancellation")
            finally:
                if backoff:
                    assert binding is not None
                    binding.close_attempt()
                elif binding is not None:
                    cleanup_state.append((binding.authority_open, binding.closed))
                cleanup_started.set()
                if not backoff:
                    await release_cleanup.wait()
                closed.set()

    async def progress(text: Any) -> None:
        if backoff:
            if text.plain.lstrip().startswith("[retry]"):
                entered.set()  # Returning leaves the actual backoff sleep as the next scheduling yield.
            return
        entered.set()
        await asyncio.sleep(3600)

    backend = ScriptedBackend(model="mock-model", supports_host_tools=True,
                              retry_policy=RetryPolicy(attempts=1, base_delay_s=30, max_delay_s=30))
    monkeypatch.setattr(backend, "execute", stream)
    with anyio.fail_after(5):
        task = asyncio.create_task(run_agent(
            backend, tmp_path, "hi", phase=DaydreamPhase.REVIEW, host_tools=binding,
            progress_callback=progress, run_context=RunContext(InteractionPolicy(log_mode=False)),
        ))
        await entered.wait()
        task.cancel()
        await cleanup_started.wait()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert closed.is_set()
    assert backend.cancel_calls == 0
    assert cleanup_state == ([(False, True)] if bound and not backoff else [])
    if binding is not None:
        assert binding.closed
        with pytest.raises(HostToolFatal, match="authority is closed"):
            async with binding.session():
                pytest.fail("cancelled authority reopened")


async def test_staged_conversations_cannot_renew_retry_recovery_spend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Single-call ladders miss renewal between conversations and charging healthy challenge work as recovery."""
    fake = FakeClock(monotonic_value=1_000).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent_retry._sample_retry_delay", lambda cap: cap)
    budget = ReviewInvestigationBudget(ReviewLimits(tool_calls=10), deadline=1_100)

    async def responder(_cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> None:
        if prompt.splitlines()[0] == "challenge":
            fake.advance(3)  # Fresh useful work cannot consume recovery activated by discovery.

    failed = [_RetryableBackendError("transient")]
    done: list[AgentEvent] = [TextEvent(text="done"), ResultEvent(structured_output=None, continuation=None)]
    backend = ScriptedBackend(
        script=[failed, done, done, failed, done, failed], responder=responder,
        retry_policy=RetryPolicy(attempts=1, base_delay_s=4, max_delay_s=4),
    )
    recorder = make_recorder(tmp_path)
    async with recorder:
        for name, allowance in (("discovery", 6), ("challenge", 60), ("rechallenge", 60)):
            output, _, reason = await run_agent(
                backend, tmp_path, name, phase=DaydreamPhase.REVIEW,
                investigation_budget=budget, retry_recovery_allowance_s=allowance,
            )
            assert (output, reason) == ("done", None)
        with pytest.raises(_RetryableBackendError):
            await run_agent(
                backend, tmp_path, "final", phase=DaydreamPhase.REVIEW,
                investigation_budget=budget, retry_recovery_allowance_s=60,
            )
    assert slept == [4, 2]
    assert [prompt.splitlines()[0] for prompt in backend.prompts] == [
        "discovery", "discovery", "challenge", "rechallenge", "rechallenge", "final",
    ]
    _, stop = _budget_stop(recorder)
    assert stop["retry_recovery_spent_s"] == 0  # Earlier conversations' spend is not billed twice.
    assert stop["retry_stop_reason"] == "retry_recovery_allowance_exhausted"
    assert (stop["attempts"], stop["backoff_s"], stop["partial_edit_handling"]) == (0, 0.0, "discarded")


async def test_retry_recovery_allowance_is_never_rebased_by_a_later_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent_retry._sample_retry_delay", lambda cap: cap)
    backend = ScriptedBackend(events=[_RetryableBackendError("transient")],
                              retry_policy=RetryPolicy(attempts=20, base_delay_s=40.0, max_delay_s=40.0))
    with pytest.raises(_RetryableBackendError):
        await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.FIX, retry_recovery_allowance_s=100.0)
    assert backend.call_count == 4  # 1 + 40 + 40 + 20, then the allowance is spent
    assert slept == [40.0, 40.0, 20.0]  # a re-basing implementation would sleep 40 forever


async def test_host_admitted_starts_remain_charged_when_native_events_are_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unlike installed retry journeys, HTTP admissions lose native starts before the next conversation."""
    deadline = anyio.current_time() + 60
    budget = ReviewInvestigationBudget(ReviewLimits(tool_calls=3), deadline=deadline)
    bindings = [HostToolBinding(deadline=deadline, accept=lambda payload: payload) for _ in range(3)]
    residual: list[int | None] = []

    async def stream(
        *_args: Any, host_tools: HostToolBinding, **_kwargs: Any,
    ) -> AsyncGenerator[AgentEvent, None]:
        schema = {"type": "object"}
        host_tools.configure_schema(schema)
        async with host_tools.session() as server:
            residual.append(host_tools.remaining_tool_calls)
            async with httpx.AsyncClient(base_url=server.url, headers={
                "Authorization": server.packet()["authorization"],
            }, timeout=3) as client:
                ready = await client.post("/ready", json={"tool_name": NATIVE_SUBMIT, "schema": schema})
                assert ready.json()["ready"]
                response = await client.post("/admit", json={
                    "id": "read-1", "name": "read", "arguments": {"path": "assigned.py"},
                })
                assert response.json()["allowed"]
            if host_tools is bindings[0]:
                raise _RetryableBackendError("native stream ended before the admitted start")
            yield ToolStartEvent(id="read-1", name="read", input={"path": "assigned.py"})
            yield TextEvent(text="done")
            yield ResultEvent(structured_output=None, continuation=None)

    backend = ScriptedBackend(model="mock-model", supports_host_tools=True,
                              retry_policy=RetryPolicy(attempts=1, base_delay_s=0, max_delay_s=0))
    monkeypatch.setattr(backend, "execute", stream)
    recorder = make_recorder(tmp_path)
    with anyio.fail_after(10):
        async with recorder:
            for index, binding in enumerate(bindings):
                with pytest.raises(_RetryableBackendError) if index == 0 else contextlib.nullcontext():
                    output, _, reason = await run_agent(
                        backend, tmp_path, "review", phase=DaydreamPhase.REVIEW,
                        investigation_budget=budget, host_tools=binding, tool_call_budget=2 if index == 0 else None,
                    )
                    assert reason == ("tool_call_budget_exceeded" if index == 2 else None)
                    if index == 1:
                        assert output == "done"
    assert residual == [2, 1, 1]
    assert [binding.calls for binding in bindings] == [2, 1, 0]
    assert backend.cancel_calls == 0
    assert budget.observed_tool_starts == 3
    trajectory = json.loads(recorder.path.read_text())
    calls = [call for step in trajectory["steps"] for call in step.get("tool_calls") or []]
    assert [call["tool_call_id"] for call in calls] == ["read-1"]


@pytest.mark.parametrize("retry", [False, True], ids=["one-attempt", "retry-after-denial"])
async def test_protected_submission_denials_do_not_trip_the_stream_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry: bool,
) -> None:
    """Denied protected-slot starts are recorded once without aborting the submission attempt.

    The native compaction journey proves the stream ceiling end to end; this owns the
    cases it cannot force: a denied start lost from the stream is reconciled exactly
    once, and a retryable failure after denials still reaches the protected start. A retry
    after denials previously stopped at the budget's observed-start gate although the
    protected submission start was still unused.
    """
    deadline = anyio.current_time() + 60
    budget = ReviewInvestigationBudget(ReviewLimits(tool_calls=3), deadline=deadline)
    binding = HostToolBinding(deadline=deadline, accept=lambda payload: payload, protected_submission=True)

    attempts: list[int] = []

    async def stream(*_args: Any, host_tools: HostToolBinding, **_kwargs: Any) -> AsyncGenerator[AgentEvent, None]:
        attempts.append(1)
        host_tools.configure_schema({"type": "object"})
        async with host_tools.session():
            # Pi awaits /admit before writing each start, so admission precedes its event.
            for call_id in ("read-1", "read-2", "read-3", "read-lost") if len(attempts) == 1 else ():
                try:
                    host_tools.admit(call_id, "read", None)
                except HostToolRejected as exc:
                    assert exc.code == "capacity_protected"
                if call_id != "read-lost":
                    yield ToolStartEvent(id=call_id, name="read", input={})
            if retry and len(attempts) == 1:
                raise _RetryableBackendError("transient native failure after denials")
            host_tools.admit("submit", NATIVE_SUBMIT, {}, terminal=True)
            yield ToolStartEvent(id="submit", name=NATIVE_SUBMIT, input={})
            host_tools.submit("submit", {})
            yield TextEvent(text="done")
            yield ResultEvent(structured_output=None, continuation=None)

    backend = ScriptedBackend(model="mock-model", supports_host_tools=True,
                              retry_policy=RetryPolicy(attempts=2, base_delay_s=0.0, max_delay_s=0.0))
    monkeypatch.setattr(backend, "execute", stream)
    with anyio.fail_after(10):
        output, _, reason = await run_agent(
            backend, tmp_path, "review", phase=DaydreamPhase.REVIEW,
            investigation_budget=budget, host_tools=binding, tool_call_budget=3,
        )
    assert (output, reason) == ("done", None) and binding.accepted and len(attempts) == 1 + retry
    assert (binding.calls, binding.denied_starts, binding.charged_starts) == (5, 2, 3)
    assert budget.observed_tool_starts == 5


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
    events = traj["extra"]["phase_events"]
    summary = derive_retry_summary(events)
    assert summary is not None
    # A repeated stop folds counts once per stop without repeating its circuit state.
    stop = next(event for event in events if event.get("event") == "agent_budget_stop"
                and event.get("metadata", {}).get("retry_stop_reason"))
    assert derive_retry_summary([stop, stop]) == derive_retry_summary([stop, {**stop, "metadata": {
        **stop["metadata"], "circuit_state": None}}])


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


@pytest.mark.parametrize("bounded", [False, True], ids=["unbounded", "staged-deadline"])
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


def test_rule_based_supervision_holds_findings_and_vetoes_denied_tools() -> None:
    """Held findings leave the kept set with an event; denied writes and Bash commands are vetoed."""
    kept, held, events = apply_findings_verdicts(
        [{"id": 1, "file": "a.py"}, {"id": 2, "file": "b.py"}],
        {1: {"id": 1, "action": "hold", "reason": "needs owner"}, 2: {"id": 2, "action": "unknown"}},
    )
    assert [item["id"] for item in kept] == [2] and [item["id"] for item in held] == [1]
    assert events == [(1, "hold", "needs owner")]
    supervisor = RuleBasedToolSupervisor(deny_globs=["*.lock"], bash_deny=[r"\brm\b"])
    review = DaydreamPhase.REVIEW
    assert supervisor("Edit", {"file_path": "deps/uv.lock"}, phase=review).veto
    assert not supervisor("Write", {"path": "src/app.py"}, phase=review).veto
    denied = supervisor("Bash", {"command": "rm -rf build"}, phase=review)
    assert denied.veto and denied.reason == "Bash command denied by regex '\\brm\\b'"
    assert not supervisor("Bash", {"command": "git status"}, phase=review).veto
    assert not supervisor("Read", {"file_path": "deps/uv.lock"}, phase=review).veto


async def test_staged_assistant_text_is_bounded_per_turn(tmp_path: Path) -> None:
    """Staged review keeps at most 128 KiB of assistant text per turn instead of growing without bound."""
    limit = 128 * 1024
    backend = ScriptedBackend(events=[TextEvent(text="x" * limit), TextEvent(text="overflow-tail"),
                                      ResultEvent(structured_output=None, continuation=None)])
    output, _, reason = await run_agent(
        backend, tmp_path, "review", phase=DaydreamPhase.REVIEW, output_schema={"type": "object"},
        investigation_budget=ReviewInvestigationBudget.from_limits(ReviewLimits()),
    )
    assert reason is None
    assert isinstance(output, str) and len(output) == limit and "overflow-tail" not in output


async def test_cancelling_one_codex_invocation_leaves_its_shared_backend_sibling_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attempt cancellation closes only its own transport; backend-wide cancellation belongs to run shutdown."""
    finished = [json.dumps(row) for row in (
        {"type": "thread.started", "thread_id": "sibling"}, {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "sibling done"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    )]
    spawner = install_fake_cli_process(monkeypatch, "codex", lines=[], scripts=[
        FakeCliScript(lines=[json.dumps({"type": "thread.started", "thread_id": "cancelled"})], hang=True),
        FakeCliScript(lines=finished, hang=True),
    ])
    backend = CodexBackend(model="fixture-model")
    async with asyncio.TaskGroup() as tasks:
        cancelled = tasks.create_task(run_agent(backend, tmp_path, "first", phase=DaydreamPhase.REVIEW))
        with anyio.fail_after(5):
            await spawner.wait_for_read(0)
        sibling = tasks.create_task(run_agent(backend, tmp_path, "second", phase=DaydreamPhase.REVIEW))
        with anyio.fail_after(5):
            await spawner.wait_for_read(1)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert spawner.procs[0].reaped and spawner.procs[1].terminate_calls == 0
        spawner.procs[1].finish([])
        with anyio.fail_after(5):
            output, _, reason = await sibling
    assert (output, reason) == ("sibling done", None)


async def test_pi_review_system_is_scoped_and_preserves_retry_and_stage_spend(tmp_path: Path) -> None:
    """Review policy reaches only review invocations; a retry snapshot reports the reduced remaining allocation."""
    commands: list[tuple[str, ...]] = []
    system_prompts: list[str] = []
    async def spawn(*args: str, **kwargs: Any) -> Any:
        commands.append(args)
        value = args[args.index("--append-system-prompt") + 1]
        system_prompts.append(Path(value).read_text() if value.startswith("/") else value)
        if len(commands) == 3:
            return make_mock_process([json.dumps(event) for event in [
                {"type": "session", "sessionId": "failed-review"},
                *[{"type": "tool_execution_start", "toolCallId": path, "toolName": "read", "args": {"path": path}}
                  for path in ("api.py", "App.tsx")],
                {"type": "turn_end", "message": {"role": "assistant", "content": [],
                 "stopReason": "error", "errorMessage": "503 Service Unavailable"}},
            ]])
        return make_mock_process_from_fixture("simple_text.jsonl")
    backend = PiBackend(model="fixture-model", reasoning_effort="high")
    backend.retry_attempts = 1
    backend.retry_base_delay_s = backend.retry_max_delay_s = 0
    budget = ReviewInvestigationBudget.from_limits(ReviewLimits(17, 3, 4))
    access_guide = json.dumps({'supporting_catalog': {'path': str(tmp_path / 'catalog.json'),
                                                   'status': 'partial', 'omitted_part_count': 1}})
    invocations: list[tuple[DaydreamPhase, dict[str, Any]]] = [
        (DaydreamPhase.DEEP, {"review_limits": ReviewLimits(17, 3, 3)}),
        (DaydreamPhase.FIX, {}),
        *[(DaydreamPhase.DEEP, {
            "investigation_budget": budget,
            "tool_call_budget": 4,
            "review_system_instructions": build_review_stage_system_instruction({
                "stage": stage,
                "assigned_target_ids": [target],
                "access_guide": access_guide,
            }),
            **({'continuation': ContinuationToken('pi', {'session_id': 'retained-navigation'})}
               if stage == 'integration' else {}),
        }) for stage, target in (("first_pass", "target-1"), ("integration", "whole-change"))],
    ]
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", side_effect=spawn):
        for phase, options in invocations:
            assert (await run_agent(backend, tmp_path, "Assigned work", phase=phase,
                                    progress_callback=lambda _: None, **options))[2] is None
    review_system, fix_system, initial, retry, integration = system_prompts
    paths = [args[args.index("--append-system-prompt") + 1] for args in commands]
    assert all(not Path(path).exists() for path in paths if path.startswith("/"))
    assert all(args[args.index("--thinking") + 1] == "high" for args in commands)
    assert commands[-1][commands[-1].index('--session-id') + 1] == 'retained-navigation'
    assert "at most 17 seconds and 3 tool calls" in review_system
    assert REVIEW_STOPPING_GUIDANCE in review_system
    assert "planted" not in review_system
    assert "hidden evaluation" not in review_system
    assert "Reopen a rejected candidate only when new evidence" in review_system
    assert "an empty findings result is valid" in review_system
    assert "Closed candidate decisions stay closed" not in review_system
    assert "repository-scoped" in review_system
    assert "Search before you read" not in review_system
    assert "typically 50" not in review_system
    assert REVIEW_STOPPING_GUIDANCE not in fix_system
    assert "Investigation allowance" not in fix_system
    assert budget.observed_tool_starts == 2 and budget.remaining_tool_calls == 2
    assert "Reviewer dispatch-time snapshot:" in initial
    assert "current invocation allocation: 4 tool calls" in initial
    assert "Retry dispatch-time snapshot (supersedes the earlier invocation allocation):" in retry
    assert "current invocation allocation: 2 tool calls" in retry
    assert "Failed-attempt calls remain charged; future-stage reserves remain excluded" in retry
    assert "current invocation allocation: 2 tool calls" in integration
    for instructions, objective, target in (
        (initial, "Review assigned changed behavior", "target-1"),
        (retry, "Review assigned changed behavior", "target-1"),
        (integration, "Review whole-change interactions and boundaries", "whole-change"),
    ):
        assignment_line = next(line for line in instructions.splitlines()
                               if line.startswith("Daydream final-submission assignment: "))
        assert json.loads(assignment_line.removeprefix("Daydream final-submission assignment: ")) == {
            "objective": objective, "target_ids": [target], "candidate_ids": [],
        }
        assert 'Persistent exact access guide (supporting metadata): ' + access_guide in instructions
        assert f'"targets": ["{target}"]' in instructions
    for instructions in (initial, retry, integration):
        assert "Cumulative observed starts:" not in instructions
        assert "remaining cumulative tool starts" not in instructions
        assert "at most one final-submission warning per native attempt" in instructions
        assert "not a recurring counter" in instructions
        assert "refreshed" not in instructions
        assert "Unused capacity is expected" in instructions
        assert "submit when those" in instructions
        assert "planted" not in instructions
        assert "hidden evaluation" not in instructions
        assert REVIEW_STOPPING_GUIDANCE not in instructions
        assert "Closed decisions stay closed" in instructions
        assert "Return exactly assigned target or triage candidate IDs" in instructions
        assert "host publishes terminal findings" in instructions


async def test_shared_retry_recovery_spans_work_items_without_renewal_or_early_clamp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Committed review shares one recovery ledger, activated against the review deadline.

    Challenge previously built a fresh budget and renewed recovery. Merely sharing the
    ledger instead let the first failing local work clamp the allowance to its own short
    deadline, starving later required work of any retry; and stop evidence computed as the
    shared ledger's delta attributed earlier or concurrent work's retries to this invocation.
    """
    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    patch_retry_sleep(monkeypatch, fake)
    recovery = RetryRecoveryBudget(math.inf)
    recovery.activate(0.0, 1_000.0)  # As committed review's schedule does for its whole window.
    for deadline, expected in [(30.0, "wall_budget_exceeded"), (1_000.0, None)]:
        budget = ReviewInvestigationBudget(ReviewLimits(tool_calls=10), deadline=deadline,
                                           retry_recovery_budget=recovery)
        sibling = [5.0] if expected is None else []

        def advance(seconds: float, sibling: list[float] = sibling) -> None:
            fake.advance(seconds)
            if sibling:
                recovery.charge(sibling.pop())  # A concurrent sibling charges the shared ledger mid-invocation.

        backend = _retryable_failing_backend(advance=advance, advance_s=20.0)
        if expected is not None:
            _, _, reason = await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.REVIEW,
                                           investigation_budget=budget, retry_recovery_allowance_s=100.0)
            assert reason == expected and recovery.spent_s == pytest.approx(20.0)
            continue
        recorder = make_recorder(tmp_path)
        with pytest.raises(_RetryableBackendError):
            async with recorder:
                await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.REVIEW,
                                investigation_budget=budget, retry_recovery_allowance_s=100.0)
    # The later work retries only within what remains of the one cumulative allowance.
    assert backend.call_count == 4 and recovery.spent_s == pytest.approx(85.0) and recovery.spent_s <= 100.0
    # Its stop evidence reports only its own retries, not the earlier work's shared-ledger charge.
    _traj, meta = _budget_stop(recorder)
    assert meta["retry_recovery_spent_s"] == pytest.approx(60.0)
