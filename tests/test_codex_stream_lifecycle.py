"""Synthetic public JSONL through real Codex executables and production transports."""

import asyncio
import errno
import json
import os
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest

from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    BackendExecutionInput,
    CostEvent,
    MetricsEvent,
    RequestEvent,
    ResultEvent,
    TextEvent,
)
from daydream.backends._subprocess import StreamStalledError
from daydream.backends.codex import CodexBackend, CodexError
from daydream.backends.lifecycle import BackendLifecycleCollector
from daydream.host_tools import HostToolBinding, HostToolFatal
from daydream.review_budget import BudgetLedger
from daydream.trajectory import DaydreamPhase
from tests.harness.codex_lifecycle_cli import CodexLifecycleCli, LifecycleMode, install_codex_lifecycle_cli
from tests.harness.codex_replay import make_mock_process
from tests.harness.processes import wait_for_process_group_gone
from tests.harness.trajectory import make_recorder


def _bounded_codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: LifecycleMode,
                   **allowances: Any) -> tuple[CodexLifecycleCli, CodexBackend]:
    """Put the fixture on PATH, then shorten postterminal allowances through the run-owned input."""
    fixture = install_codex_lifecycle_cli(tmp_path / "cli", mode=mode)
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    execution = BackendExecutionInput.from_environment(dict(os.environ), backend="codex")
    return fixture, CodexBackend(model="fixture-model", execution_input=replace(execution, **allowances))


@pytest.mark.parametrize("mode,category,expected_metrics", [
    ("completed_next_turn", None, 2), ("unfinished_next_turn", "TRANSPORT", 1),
])
async def test_delayed_consumer_drains_finished_child_without_skipping_latest_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: LifecycleMode,
    category: str | None, expected_metrics: int,
) -> None:
    """Consumer latency cannot invent drain failure or hide buffered latest-turn events."""
    fixture, backend = _bounded_codex(tmp_path, monkeypatch, mode, codex_terminal_drain_timeout_s=0.2)
    seen: list[AgentEvent] = []
    error: CodexError | None = None
    collector = BackendLifecycleCollector(attempt=1, started_at=asyncio.get_running_loop().time(),
                                          wall_allowance_s=None, deadline=None)
    token = collector.bind()
    try:
        async with asyncio.timeout(3):
            try:
                async for event in backend.execute(tmp_path, "synthetic lifecycle request"):
                    seen.append(event)
                    if isinstance(event, MetricsEvent) and sum(isinstance(row, MetricsEvent) for row in seen) == 1:
                        await asyncio.sleep(0.5)
            except CodexError as exc:
                error = exc
    finally:
        await collector.stop_sampling()
        snapshot = collector.finalize()
        collector.reset(token)
    assert (error.category if error else None) == category
    assert snapshot is not None and snapshot.process_returncode == 0
    assert snapshot.stdout_eof_s is not None and snapshot.expiry_stage is None
    assert snapshot.terminal_received_s is not None and snapshot.cleanup_expired is False
    assert snapshot.teardown_started_s is not None and snapshot.teardown_completed_s is not None
    assert [(row.prompt_tokens, row.completion_tokens) for row in seen if isinstance(row, MetricsEvent)] == (
        [(10, 5)] * expected_metrics
    )
    assert [(row.input_tokens, row.output_tokens) for row in seen if isinstance(row, CostEvent)] == (
        [(10, 5)] * expected_metrics
    )
    assert len([row for row in seen if isinstance(row, ResultEvent)]) == (1 if category is None else 0)
    assert any(isinstance(row, TextEvent) and "Synthetic lifecycle response" in row.text for row in seen)
    if mode == "unfinished_next_turn":
        assert error is not None and error.retryable is False
    else:
        assert any(isinstance(row, TextEvent) and row.text == "Latest public turn" for row in seen)
    for observation in fixture.observations():
        await wait_for_process_group_gone(observation["pid"], timeout_s=5)


async def test_finished_child_postdeadline_chatter_cannot_remove_the_drain_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even a zero-exit child cannot force unlimited buffered event processing after expiry."""
    _, backend = _bounded_codex(tmp_path, monkeypatch, "completed_chatter", codex_terminal_drain_timeout_s=0.2)
    seen: list[AgentEvent] = []
    async with asyncio.timeout(3):
        with pytest.raises(CodexError, match="terminal_drain") as failure:
            async for event in backend.execute(tmp_path, "synthetic lifecycle request"):
                seen.append(event)
                if isinstance(event, MetricsEvent):
                    await asyncio.sleep(0.5)
    assert failure.value.retryable is False
    assert "child exit 0 observed" in str(failure.value)
    assert not any(isinstance(event, ResultEvent) for event in seen)


@pytest.mark.parametrize("mode,stage", [
    ("terminal_eof_live_child", "process_reap"),
    ("repeated_terminal", "terminal_drain"),  # repeated terminal events cannot extend the tail allowance
])
async def test_completed_codex_turn_has_bounded_failure_and_retains_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: LifecycleMode, stage: str,
) -> None:
    """Drain/reap expiry retains public terminal metrics without reporting success."""
    fixture, backend = _bounded_codex(tmp_path, monkeypatch, mode, codex_terminal_drain_timeout_s=0.15,
                                      codex_process_reap_timeout_s=0.15, codex_cleanup_timeout_s=0.5)
    seen: list[AgentEvent] = []
    terminal_observed = False
    # Cold fixture startup is separate from the bounded postterminal behavior.
    async with asyncio.timeout(15) as watchdog:
        with pytest.raises(CodexError) as failure:
            async for event in backend.execute(tmp_path, "synthetic lifecycle request"):
                seen.append(event)
                if not terminal_observed and isinstance(event, MetricsEvent):
                    terminal_observed = True
                    watchdog.reschedule(asyncio.get_running_loop().time() + 3)
    assert failure.value.category == "TIMEOUT"
    assert failure.value.retryable is False
    assert stage in str(failure.value)
    assert "timeout after 0.15s" in str(failure.value)
    assert ("stdout EOF was not observed" if stage == "terminal_drain"
            else "actual child exit was not observed") in str(failure.value)
    assert any(isinstance(event, TextEvent) and "Synthetic lifecycle response" in event.text for event in seen)
    metrics = [event for event in seen if isinstance(event, MetricsEvent)]
    costs = [event for event in seen if isinstance(event, CostEvent)]
    assert len(metrics) == 1 and metrics[0].prompt_tokens == 10 and metrics[0].completion_tokens == 5
    assert len(costs) == 1 and costs[0].input_tokens == 10 and costs[0].output_tokens == 5
    assert not any(isinstance(event, ResultEvent) for event in seen)
    for observation in fixture.observations():
        await wait_for_process_group_gone(observation["pid"], timeout_s=5)


async def test_codex_process_outcomes_require_completion_and_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Success needs a completed latest turn and an observed zero exit.

    A completed public turn followed by a nonzero exit is a process failure with no result,
    and a real silent child cannot turn its initial commentary into a completed result.
    A submission the host accepted over MCP survives that exit unsettled, a turn that completes
    with nothing accepted is a retryable truncation, and Codex-owned starts stay bounded after acceptance.
    Unlisted host tools, a host call Codex failed itself, a Codex-owned start past the allowance
    and instructions too large to pass stop the invocation with what happened.
    """
    completed = [json.dumps(row) for row in (
        {"type": "thread.started", "thread_id": "exit-check"}, {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}},
    )]
    process = make_mock_process(completed)
    process.wait.return_value = process.returncode = 7
    nonzero_events: list[AgentEvent] = []
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=process):
        with pytest.raises(CodexError, match="return code 7") as exit_failure:
            async for event in CodexBackend(model="fixture-model").execute(tmp_path, "exit check"):
                nonzero_events.append(event)
    assert exit_failure.value.category == "PROCESS_EXIT"
    assert not any(isinstance(event, ResultEvent) for event in nonzero_events)
    fixture = install_codex_lifecycle_cli(tmp_path / "cli", mode="preterminal_silence")
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DAYDREAM_STREAM_IDLE_TIMEOUT_S", "1")
    seen: list[AgentEvent] = []
    async with asyncio.timeout(3):
        with pytest.raises(StreamStalledError) as failure:
            async for event in CodexBackend(model="fixture-model").execute(tmp_path, "synthetic lifecycle request"):
                seen.append(event)
    assert failure.value.cli == "codex" and failure.value.timeout_s == 1.0 and failure.value.retryable is True
    assert not any(isinstance(event, (ResultEvent, MetricsEvent)) for event in seen)
    assert any(isinstance(event, TextEvent) and "initial commentary" in event.text for event in seen)
    for observation in fixture.observations():
        await wait_for_process_group_gone(observation["pid"], timeout_s=5)

    def binding(limit: int) -> HostToolBinding:
        ledger = BudgetLedger("host", deadline=asyncio.get_running_loop().time() + 30, limit=limit, start_floor=1)
        return HostToolBinding(budget=ledger, accept=lambda value: {"final": value})

    # Codex reads the instructions as a TOML string; every character must survive that reading.
    instructions = 'policy "quoted" back\\slash\ttab\r\nline \x01\x7f \u00e9 \U0001f600 \'\'\' # ${HOME}'
    monkeypatch.delenv("DAYDREAM_STREAM_IDLE_TIMEOUT_S")
    environment = dict(os.environ)
    outcomes = {}
    rows: tuple[tuple[str, LifecycleMode, str, int], ...] = (
        ("settled", "healthy", '{"ok": true}', 4), ("failed exit", "nonzero_exit", '{"ok": true}', 4),
        ("revoked", "healthy", '{"ok": true}', 4), ("overrun", "healthy", '{"ok": true, "shells": 2}', 2))
    for name, mode, prompt, limit in rows:
        served = install_codex_lifecycle_cli(tmp_path / mode / name, mode=mode)
        monkeypatch.setenv("PATH", f"{served.bin_dir}{os.pathsep}{environment['PATH']}")
        bound, events, stop = binding(limit), [], None
        async with asyncio.timeout(15):
            try:
                async for event in CodexBackend(model="fixture-model").execute(
                    tmp_path, prompt, {"type": "object"}, read_only=True,
                    review_instructions=instructions, host_tools=bound,
                ):
                    events.append(event)
                    if name == "revoked" and isinstance(event, RequestEvent):
                        bound.close_attempt()  # as a consumer that stops reading does
            except (CodexError, HostToolFatal) as exc:
                stop = getattr(exc, "code", None) or getattr(exc, "category", None)
        (observation,) = served.observations()
        await wait_for_process_group_gone(observation["pid"], timeout_s=5)
        # The request carries exactly the instructions, the child decoded exactly them, the host
        # environment kept neither the secret nor a changed exemption, and the endpoint is gone.
        assert [event.system_prompt for event in events if isinstance(event, RequestEvent)] == [instructions]
        assert observation["host"]["instructions"] == instructions
        monkeypatch.setenv("PATH", environment["PATH"])
        assert dict(os.environ) == environment
        with pytest.raises(ConnectionRefusedError):
            socket.create_connection(
                ("127.0.0.1", urlsplit(observation["host"]["overrides"]["mcp_servers.daydream.url"]).port))
        outcomes[name] = (bound.accepted_result, bound.native_settled, bound.cleanup_complete, bound.closed, stop,
                          [event.structured_output for event in events if isinstance(event, ResultEvent)],
                          (bound.budget.charged, bound.budget.observed))
    accepted = {"final": {"ok": True}}
    assert outcomes == {
        # The accepted submission, not the turn's native result field, is the settled result.
        "settled": (accepted, True, True, True, None, [accepted], (1, 1)),
        # The failed exit keeps its accepted submission unsettled.
        "failed exit": (accepted, False, True, True, "PROCESS_EXIT", [], (1, 1)),
        # A callback after the attempt was revoked changes nothing, so the completed turn submitted
        # nothing: a retryable truncation, and the binding may open another attempt.
        "revoked": (None, False, True, False, "STREAM_TRUNCATION", [None], (0, 0)),
        # Codex-owned starts after acceptance stay charged; one past the allowance stops the invocation.
        "overrun": ({"final": {"ok": True, "shells": 2}}, False, True, True, "capacity_exhausted", [], (2, 3)),
    }
    native_failure = {"type": "item.completed", "item": {
        "id": "item_1", "type": "mcp_tool_call", "server": "daydream", "tool": "structured_output", "arguments": {},
        "result": None, "error": {"message": "tool call requires approval"}, "status": "failed"}}
    shell = {"type": "item.started", "item": {"type": "command_execution", "id": "shell", "command": "ls"}}
    patched = {"type": "item.completed", "item": {"type": "file_change", "id": "patch", "file_path": "a.py"}}
    stops: tuple[tuple[list[dict[str, Any]] | OSError, str | None, str], ...] = (
        # Codex's own error text is reported as what happened, and the binding stays retryable.
        ([native_failure], None, "Codex failed a host tool call itself before acceptance.*requires approval"),
        ([{"type": "turn.started"}], "native_readiness", "did not list"),
        # Codex-owned starts cannot be denied, so one past the allowance stops the invocation.
        ([shell, patched], "capacity_exhausted", "allocation exhausted"),
        # The instructions travel in one argument, whose size the platform bounds.
        (OSError(errno.E2BIG, "Argument list too long"), None, "exceed this platform's single-argument size"),
    )
    for lines, code, reason in stops:
        bound = binding(1)
        spawn = {"side_effect": lines} if isinstance(lines, OSError) else {"return_value": make_mock_process(
            [json.dumps(row) for row in ({"type": "thread.started", "thread_id": "t"}, *lines)])}
        with patch("daydream.backends._transport.asyncio.create_subprocess_exec", **spawn):
            with pytest.raises((CodexError, HostToolFatal), match=reason) as stopped:
                async for _ in CodexBackend(model="fixture-model").execute(
                    tmp_path, "stopped", {"type": "object"}, read_only=True, review_instructions="policy",
                    host_tools=bound,
                ):
                    pass
        assert (getattr(stopped.value, "code", None), bound.fatal_error) == (
            code, stopped.value if code else None)
        assert bound.closed is (code is not None) and bound.cleanup_complete and not bound.possibly_accepted


async def test_codex_cleanup_expiry_preserves_primary_drain_error_and_recorded_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second cleanup expiry cannot replace a completed turn's drain failure or evidence."""
    fixture, backend = _bounded_codex(tmp_path, monkeypatch, "terminal_open_stdout",
                                      codex_terminal_drain_timeout_s=0.1, codex_cleanup_timeout_s=0.000001)
    recorder = make_recorder(tmp_path)
    async with asyncio.timeout(3):
        with pytest.raises(CodexError, match="terminal_drain") as failure:
            async with recorder:
                await run_agent(backend, tmp_path, "synthetic lifecycle request", phase=DaydreamPhase.FIX)
    assert failure.value.retryable is False
    trajectory = json.loads(recorder.path.read_text())
    assert trajectory["extra"]["partial"] is True
    attempts = trajectory["extra"]["subtrajectories"]
    assert len(attempts) == 1
    lifecycle = attempts[0]["backend_lifecycle"]
    assert lifecycle["expiry_stage"] == "terminal_drain" and lifecycle["cleanup_expired"] is True
    assert lifecycle["terminal_received_s"] is not None
    assert lifecycle["teardown_started_s"] is not None and lifecycle["teardown_completed_s"] is None
    agent_steps = [step for step in trajectory["steps"] if step["source"] == "agent"]
    assert any("Synthetic lifecycle response" in step.get("message", "") for step in agent_steps)
    assert any(step.get("metrics", {}).get("prompt_tokens") == 10 for step in agent_steps)
    for observation in fixture.observations():
        await wait_for_process_group_gone(observation["pid"], timeout_s=5)
