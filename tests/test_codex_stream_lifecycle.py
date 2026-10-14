"""Synthetic public JSONL through real Codex executables and production transports."""

import asyncio
import json
import os
from pathlib import Path

import pytest

from daydream.agent import run_agent
from daydream.backends import AgentEvent, BackendExecutionInput, MetricsEvent, ResultEvent, TextEvent
from daydream.backends._subprocess import StreamStalledError
from daydream.backends.codex import CodexBackend, CodexError
from daydream.trajectory import DaydreamPhase
from tests.harness.codex_lifecycle_cli import LifecycleMode, install_codex_lifecycle_cli
from tests.harness.processes import wait_for_process_group_gone
from tests.harness.trajectory import make_recorder


@pytest.mark.parametrize("mode,stage", [
    ("terminal_open_stdout", "terminal_drain"),
    ("terminal_eof_live_child", "process_reap"),
    ("descendant_stdout", "terminal_drain"),
    ("repeated_terminal", "terminal_drain"),
])
async def test_completed_codex_turn_has_bounded_failure_and_retains_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: LifecycleMode, stage: str,
) -> None:
    """Drain/reap expiry retains public terminal metrics without reporting success."""
    fixture = install_codex_lifecycle_cli(tmp_path / "cli", mode=mode)
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DAYDREAM_CODEX_TERMINAL_DRAIN_TIMEOUT_S", "0.15")
    monkeypatch.setenv("DAYDREAM_CODEX_PROCESS_REAP_TIMEOUT_S", "0.15")
    monkeypatch.setenv("DAYDREAM_CODEX_CLEANUP_TIMEOUT_S", "0.5")
    seen: list[AgentEvent] = []
    backend = CodexBackend(model="fixture-model")
    started = asyncio.get_running_loop().time()
    async with asyncio.timeout(3):
        with pytest.raises(CodexError) as failure:
            async for event in backend.execute(tmp_path, "synthetic lifecycle request"):
                seen.append(event)
    assert failure.value.category == "TIMEOUT"
    assert failure.value.retryable is False
    assert stage in str(failure.value)
    assert asyncio.get_running_loop().time() - started < 2
    assert any(isinstance(event, TextEvent) and "Synthetic lifecycle response" in event.text for event in seen)
    assert any(isinstance(event, MetricsEvent) and event.prompt_tokens == 10 for event in seen)
    assert not any(isinstance(event, ResultEvent) for event in seen)
    for observation in fixture.observations():
        await wait_for_process_group_gone(observation["pid"], timeout_s=5)


@pytest.mark.parametrize("mode", ["healthy", "nonzero_exit", "preterminal_silence"])
async def test_codex_process_outcomes_require_completion_and_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: LifecycleMode,
) -> None:
    """Silence and a completed crashing process cannot become successful outcomes."""
    fixture = install_codex_lifecycle_cli(tmp_path / "cli", mode=mode)
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DAYDREAM_STREAM_IDLE_TIMEOUT_S", "1")
    seen: list[AgentEvent] = []
    error: Exception | None = None
    async with asyncio.timeout(3):
        try:
            async for event in CodexBackend(model="fixture-model").execute(tmp_path, "synthetic lifecycle request"):
                seen.append(event)
        except (CodexError, StreamStalledError) as exc:
            error = exc
    results = [event for event in seen if isinstance(event, ResultEvent)]
    metrics = [event for event in seen if isinstance(event, MetricsEvent)]
    if mode == "healthy":
        assert error is None and len(results) == 1 and len(metrics) == 1
    elif mode == "nonzero_exit":
        assert isinstance(error, CodexError) and error.category == "PROCESS_EXIT"
        assert "return code 7" in str(error)
        assert not results and len(metrics) == 1
    else:
        assert isinstance(error, StreamStalledError) and error.retryable is True
        assert not results and not metrics
        assert any(isinstance(event, TextEvent) and "initial commentary" in event.text for event in seen)
    for observation in fixture.observations():
        await wait_for_process_group_gone(observation["pid"], timeout_s=5)


async def test_codex_observed_exit_after_eof_can_complete_within_reap_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EOF is independent of actual exit; releasing a live child permits truthful success."""
    fixture = install_codex_lifecycle_cli(tmp_path / "cli", mode="terminal_eof_live_child")
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    seen: list[AgentEvent] = []

    async def consume() -> None:
        async for event in CodexBackend(model="fixture-model").execute(tmp_path, "synthetic lifecycle request"):
            seen.append(event)

    task = asyncio.create_task(consume())
    try:
        observation = await fixture.wait_for_stage("stdout_closed")
        assert not any(isinstance(event, ResultEvent) for event in seen)
        fixture.release(observation["pid"])
        async with asyncio.timeout(3):
            await task
        assert len([event for event in seen if isinstance(event, ResultEvent)]) == 1
        await wait_for_process_group_gone(observation["pid"], timeout_s=5)
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.parametrize("frozen_input", [False, True])
async def test_codex_lifecycle_allowance_is_frozen_before_environment_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_input: bool,
) -> None:
    """Direct construction and host execution inputs retain the configured tail bound."""
    fixture = install_codex_lifecycle_cli(tmp_path / "cli", mode="terminal_open_stdout")
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DAYDREAM_CODEX_TERMINAL_DRAIN_TIMEOUT_S", "0.15")
    execution_input = BackendExecutionInput.from_environment(os.environ, backend="codex") if frozen_input else None
    backend = CodexBackend(model="fixture-model", execution_input=execution_input)
    monkeypatch.setenv("DAYDREAM_CODEX_TERMINAL_DRAIN_TIMEOUT_S", "60")
    started = asyncio.get_running_loop().time()
    async with asyncio.timeout(3):
        with pytest.raises(CodexError, match="terminal_drain") as failure:
            [event async for event in backend.execute(tmp_path, "synthetic lifecycle request")]
    assert failure.value.retryable is False
    assert asyncio.get_running_loop().time() - started < 2
    for observation in fixture.observations():
        await wait_for_process_group_gone(observation["pid"], timeout_s=5)


async def test_cancelling_codex_agent_does_not_kill_shared_backend_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invocation cancellation reaps only its real process while a sibling can publish completion."""
    blocked = install_codex_lifecycle_cli(tmp_path / "blocked", mode="preterminal_silence")
    sibling = install_codex_lifecycle_cli(tmp_path / "sibling", mode="terminal_open_stdout")
    monkeypatch.setenv("DAYDREAM_STREAM_IDLE_TIMEOUT_S", "0")
    monkeypatch.setenv("PATH", f"{blocked.bin_dir}{os.pathsep}{os.environ['PATH']}")
    backend = CodexBackend(model="fixture-model")
    cancelled = asyncio.create_task(run_agent(backend, tmp_path, "synthetic blocked request", phase=DaydreamPhase.FIX))
    healthy: asyncio.Task[tuple[object, object, str | None]] | None = None
    try:
        blocked_observation = await blocked.wait_for_stage("initial")
        monkeypatch.setenv("PATH", f"{sibling.bin_dir}{os.pathsep}{os.environ['PATH']}")
        healthy = asyncio.create_task(run_agent(
            backend, tmp_path, "synthetic sibling request", phase=DaydreamPhase.FIX))
        sibling_observation = await sibling.wait_for_stage("terminal")
        cancelled.cancel()
        async with asyncio.timeout(3):
            with pytest.raises(asyncio.CancelledError):
                await cancelled
        await wait_for_process_group_gone(blocked_observation["pid"], timeout_s=5)
        sibling.release(sibling_observation["pid"])
        async with asyncio.timeout(3):
            output, _, reason = await healthy
        assert isinstance(output, str) and output.endswith("Synthetic lifecycle response") and reason is None
        await wait_for_process_group_gone(sibling_observation["pid"], timeout_s=5)
    finally:
        for task in (cancelled, healthy):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task


async def test_codex_cleanup_expiry_preserves_primary_drain_error_and_recorded_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second cleanup expiry cannot replace a completed turn's drain failure or evidence."""
    fixture = install_codex_lifecycle_cli(tmp_path / "cli", mode="terminal_open_stdout")
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DAYDREAM_CODEX_TERMINAL_DRAIN_TIMEOUT_S", "0.1")
    monkeypatch.setenv("DAYDREAM_CODEX_CLEANUP_TIMEOUT_S", "0.000001")
    recorder = make_recorder(tmp_path)
    async with asyncio.timeout(3):
        with pytest.raises(CodexError, match="terminal_drain") as failure:
            async with recorder:
                await run_agent(CodexBackend(model="fixture-model"), tmp_path,
                                "synthetic lifecycle request", phase=DaydreamPhase.FIX)
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
