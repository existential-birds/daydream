"""Synthetic public JSONL through real Codex executables and production transports."""

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream.agent import run_agent
from daydream.backends import AgentEvent, BackendExecutionInput, CostEvent, MetricsEvent, ResultEvent, TextEvent
from daydream.backends._subprocess import StreamStalledError
from daydream.backends.codex import CodexBackend, CodexError
from daydream.backends.lifecycle import BackendLifecycleCollector
from daydream.trajectory import DaydreamPhase
from tests.harness.codex_lifecycle_cli import LifecycleMode, install_codex_lifecycle_cli
from tests.harness.codex_replay import make_mock_process
from tests.harness.processes import wait_for_process_group_gone
from tests.harness.trajectory import make_recorder


def _bounded_codex(**allowances: Any) -> CodexBackend:
    """Shorten postterminal allowances through the run-owned input after the fixture is on PATH."""
    execution = BackendExecutionInput.from_environment(dict(os.environ), backend="codex")
    return CodexBackend(model="fixture-model", execution_input=replace(execution, **allowances))


@pytest.mark.parametrize("mode,category,expected_metrics", [
    ("completed_next_turn", None, 2), ("unfinished_next_turn", "TRANSPORT", 1),
])
async def test_delayed_consumer_drains_finished_child_without_skipping_latest_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: LifecycleMode,
    category: str | None, expected_metrics: int,
) -> None:
    """Consumer latency cannot invent drain failure or hide buffered latest-turn events."""
    fixture = install_codex_lifecycle_cli(tmp_path / "cli", mode=mode)
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    backend = _bounded_codex(codex_terminal_drain_timeout_s=0.2)
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
    fixture = install_codex_lifecycle_cli(tmp_path / "cli", mode="completed_chatter")
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    seen: list[AgentEvent] = []
    backend = _bounded_codex(codex_terminal_drain_timeout_s=0.2)
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
    fixture = install_codex_lifecycle_cli(tmp_path / "cli", mode=mode)
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    seen: list[AgentEvent] = []
    backend = _bounded_codex(codex_terminal_drain_timeout_s=0.15, codex_process_reap_timeout_s=0.15,
                             codex_cleanup_timeout_s=0.5)
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


async def test_codex_cleanup_expiry_preserves_primary_drain_error_and_recorded_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second cleanup expiry cannot replace a completed turn's drain failure or evidence."""
    fixture = install_codex_lifecycle_cli(tmp_path / "cli", mode="terminal_open_stdout")
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    recorder = make_recorder(tmp_path)
    backend = _bounded_codex(codex_terminal_drain_timeout_s=0.1, codex_cleanup_timeout_s=0.000001)
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
