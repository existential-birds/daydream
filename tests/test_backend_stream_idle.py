"""Test CLI stalls, retries, wall budgets, and shielded process cleanup.

FakeCliProcess models permanent silence at the spawn boundary: only the timer
under test can fire, so load cannot invert outcomes. Final wiring tests use
real subprocesses that print and exit without timing races. Assert failures,
spawn/reap counts, ATIF state, and the OS process table.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

from daydream.agent import run_agent
from daydream.backends import ResultEvent, TextEvent
from daydream.backends._subprocess import (
    DEFAULT_PI_RESPONSE_IDLE_TIMEOUT_S,
    DEFAULT_STREAM_IDLE_TIMEOUT_S,
    STREAM_IDLE_TIMEOUT_ENV,
    StreamStalledError,
    readline_with_idle_timeout,
    stream_idle_timeout_s,
)
from daydream.backends.codex import CodexBackend
from daydream.backends.pi import PiBackend
from daydream.config import DEFAULT_WALL_BUDGET_S
from daydream.trajectory import DaydreamPhase
from tests.harness.fake_cli_process import (
    SIGKILL_RC,
    SIGTERM_RC,
    FakeCliProcess,
    install_fake_cli_process,
)
from tests.harness.trajectory import make_recorder

# A complete, valid stream for each CLI.
PI_LINES = [
    json.dumps({"type": "session", "id": "sess-idle-1"}), json.dumps({"type": "agent_start"}),
    json.dumps(
        {
            "type": "message_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "slow but alive"}]},
        }
    ),
    json.dumps(
        {
            "type": "turn_end",
            "message": {
                "role": "assistant", "stopReason": "end_turn",
                "usage": {"input": 10, "output": 4, "cost": {"total": 0.01}},
            },
        }
    ), json.dumps({"type": "agent_end"}),
]

CODEX_LINES = [
    json.dumps({"type": "thread.started", "thread_id": "thr-idle-1"}), json.dumps({"type": "turn.started"}),
    json.dumps({"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "slow but alive"}}),
    json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 4}}),
]

# Short windows only accelerate permanent-silence cases; they must not create a timing race.
TINY_WINDOW = "0.05"

def assert_stalled_and_reaped(spawner: Any, *, expected_spawns: int = 1) -> FakeCliProcess:
    assert len(spawner.procs) == expected_spawns, (
        f"expected {expected_spawns} subprocess launch(es), saw {len(spawner.procs)}"
    )
    proc = spawner.procs[-1]
    assert proc.returncode is not None, "subprocess was never killed — leaked"
    assert proc.reaped, "subprocess was killed but never wait()ed — zombie"
    return cast(FakeCliProcess, proc)

async def run_pi_wall_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, ignore_sigterm: bool = False
) -> FakeCliProcess:
    spawner = install_fake_cli_process(monkeypatch, "pi", lines=PI_LINES[:2], hang=True, ignore_sigterm=ignore_sigterm)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "3600")
    output, _, budget_reason = await run_agent(
        PiBackend(model="test-model"), tmp_path, "review", phase=DaydreamPhase.REVIEW, wall_budget_s=1.0,
    )
    assert budget_reason == "wall_budget_exceeded"
    assert output == ""
    return assert_stalled_and_reaped(spawner)

async def drain(backend: Any, cwd: Path) -> list[Any]:
    return [event async for event in backend.execute(cwd, "do the thing")]

# A silent stream trips the idle timeout; the subprocess is torn down.

@pytest.mark.parametrize(
    ("cli", "backend_cls", "lines", "expected_timeout"),
    [
        pytest.param("pi", PiBackend, PI_LINES[:2], float(TINY_WINDOW), id="pi"),
        pytest.param("codex", CodexBackend, CODEX_LINES[:2], None, id="codex"),
    ],
)
async def test_silent_stream_trips_idle_timeout_and_reaps_subprocess(
    cli: str, backend_cls: Any, lines: list[str], expected_timeout: float | None, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CLI that emits two lines then goes silent forever ends the turn. The pi case also proves the env override
    reaches the armed window: the raised error carries the exact configured value."""
    spawner = install_fake_cli_process(monkeypatch, cli, lines=lines, hang=True)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, TINY_WINDOW)
    with pytest.raises(StreamStalledError) as excinfo:
        await drain(backend_cls(model="test-model"), tmp_path)
    assert excinfo.value.cli == cli
    if expected_timeout is not None:
        assert excinfo.value.timeout_s == expected_timeout
    assert excinfo.value.retryable is True
    assert_stalled_and_reaped(spawner)

async def test_pi_stalls_before_first_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spawner = install_fake_cli_process(monkeypatch, "pi", lines=[], hang=True)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, TINY_WINDOW)
    with pytest.raises(StreamStalledError):
        await drain(PiBackend(model="test-model"), tmp_path)
    assert_stalled_and_reaped(spawner)

async def test_pi_response_stall_uses_shorter_default_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        json.dumps({"type": "session", "id": "sess-response-stall"}), json.dumps({"type": "agent_start"}),
        json.dumps({"type": "turn_start"}),
        json.dumps(
            {"type": "tool_execution_start", "toolCallId": "read-1", "toolName": "read", "args": {"path": "src/lib.rs"}}
        ),
        json.dumps(
            {
                "type": "tool_execution_end", "toolCallId": "read-1",
                "result": {"content": [{"type": "text", "text": "done"}]},
            }
        ),
    ]
    spawner = install_fake_cli_process(monkeypatch, "pi", lines=lines, hang=True)
    monkeypatch.delenv(STREAM_IDLE_TIMEOUT_ENV, raising=False)
    monkeypatch.setattr("daydream.backends.pi.DEFAULT_PI_RESPONSE_IDLE_TIMEOUT_S", 0.05, raising=False)
    with anyio.fail_after(1):
        with pytest.raises(StreamStalledError) as excinfo:
            await drain(PiBackend(model="test-model"), tmp_path)
    assert excinfo.value.timeout_s == pytest.approx(0.05)
    assert_stalled_and_reaped(spawner)

async def test_pi_active_tool_keeps_long_subprocess_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An output-silent tool may legitimately outlive the response window."""
    lines = [
        json.dumps({"type": "session", "id": "sess-active-tool"}), json.dumps({"type": "agent_start"}),
        json.dumps({"type": "turn_start"}),
        json.dumps(
            {
                "type": "tool_execution_start", "toolCallId": "cargo-1", "toolName": "bash",
                "args": {"command": "cargo build --quiet"},
            }
        ),
    ]
    spawner = install_fake_cli_process(monkeypatch, "pi", lines=lines, hang=True)
    monkeypatch.delenv(STREAM_IDLE_TIMEOUT_ENV, raising=False)
    monkeypatch.setattr("daydream.backends.pi.DEFAULT_PI_RESPONSE_IDLE_TIMEOUT_S", 0.05, raising=False)
    monkeypatch.setattr("daydream.backends.pi.DEFAULT_STREAM_IDLE_TIMEOUT_S", 0.2, raising=False)
    with anyio.fail_after(1):
        with pytest.raises(StreamStalledError) as excinfo:
            await drain(PiBackend(model="test-model"), tmp_path)
    assert excinfo.value.timeout_s == pytest.approx(0.2)
    assert_stalled_and_reaped(spawner)

# Available data resets the idle window independently of elapsed wall time.

@pytest.mark.parametrize(
    ("cli", "backend_cls", "lines"),
    [pytest.param("pi", PiBackend, PI_LINES, id="pi"), pytest.param("codex", CodexBackend, CODEX_LINES, id="codex")],
)
async def test_flowing_stream_does_not_trip_a_tiny_window(
    cli: str, backend_cls: Any, lines: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawner = install_fake_cli_process(monkeypatch, cli, lines=lines)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, TINY_WINDOW)
    events = await drain(backend_cls(model="test-model"), tmp_path)
    assert [e.text for e in events if isinstance(e, TextEvent)] == ["slow but alive"]
    assert any(isinstance(e, ResultEvent) for e in events)
    assert spawner.procs[0].reaped

async def test_idle_window_restarts_after_each_line() -> None:
    """Re-arm the same timeout after each line: successful reads do not consume the next idle window."""
    reader = asyncio.StreamReader()
    window = float(TINY_WINDOW)
    reader.feed_data(b"one\n")
    assert await readline_with_idle_timeout(reader, cli="pi", timeout_s=window) == b"one\n"
    reader.feed_data(b"two\n")
    assert await readline_with_idle_timeout(reader, cli="pi", timeout_s=window) == b"two\n"
    with pytest.raises(StreamStalledError):
        await readline_with_idle_timeout(reader, cli="pi", timeout_s=window)

# Operator configuration.

async def test_zero_disables_idle_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_cli_process(monkeypatch, "pi", lines=PI_LINES)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "0")
    assert stream_idle_timeout_s() is None
    events = await drain(PiBackend(model="test-model"), tmp_path)
    assert [e.text for e in events if isinstance(e, TextEvent)] == ["slow but alive"]

@pytest.mark.parametrize("raw", ["not-a-number", "-5", "nan", "inf"])
def test_malformed_override_falls_back_to_default(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A garbage/negative/non-finite value never disables or shortens the window."""
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, raw)
    assert stream_idle_timeout_s() == DEFAULT_STREAM_IDLE_TIMEOUT_S

def test_default_windows_straddle_the_wall_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pi response silence must preempt the wall budget; output-silent tools and Codex generations retain a
    longer idle window.
    """
    monkeypatch.delenv(STREAM_IDLE_TIMEOUT_ENV, raising=False)
    assert stream_idle_timeout_s() == DEFAULT_STREAM_IDLE_TIMEOUT_S
    assert DEFAULT_PI_RESPONSE_IDLE_TIMEOUT_S == 600.0
    assert (
        DEFAULT_PI_RESPONSE_IDLE_TIMEOUT_S
        < DEFAULT_WALL_BUDGET_S
        < DEFAULT_STREAM_IDLE_TIMEOUT_S
    )

# Exercise stalled retries through the real run_agent path.

async def test_stall_retry_is_limited_below_backend_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawner = install_fake_cli_process(monkeypatch, "pi", lines=PI_LINES[:2], hang=True)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, TINY_WINDOW)
    monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0.01")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_MAX_DELAY_S", "0.01")
    backend = PiBackend(model="test-model")
    assert backend.retry_attempts == 3, "test must exercise a backend that does retry"
    trajectory_path = tmp_path / ".daydream" / "trajectory.json"
    recorder = make_recorder(tmp_path, path=trajectory_path, agent_model_name="test-model", session_id="stall-test")
    with pytest.raises(StreamStalledError):
        async with recorder:
            await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW)
    assert_stalled_and_reaped(spawner, expected_spawns=2)
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert trajectory["extra"]["partial"] is True
    errored = [
        step for step in trajectory["steps"]
        if (step.get("extra") or {}).get("error_subtype") == "StreamStalledError"
    ]
    assert errored, "the stall was not recorded on any trajectory step"

async def test_wall_budget_still_aborts_while_blocked_in_the_idle_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outer AnyIO cancellation must abort an inner asyncio idle read and reap the child.

    Only the wall timer can fire: the stream stays silent and idle timeout is much longer.
    """
    proc = await run_pi_wall_budget(tmp_path, monkeypatch)
    assert proc.returncode == SIGTERM_RC, "a cooperative child needs no SIGKILL"

async def test_cancelled_teardown_still_escalates_to_sigkill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancelled generator still shields SIGTERM/grace/SIGKILL/reap for an ignoring child.

    Zero grace makes escalation immediate; an unshielded first await would leak the process.
    """
    monkeypatch.setattr("daydream.backends._subprocess.TERMINATE_GRACE_S", 0.0)
    proc = await run_pi_wall_budget(tmp_path, monkeypatch, ignore_sigterm=True)
    assert proc.terminate_calls == 1
    assert proc.kill_calls == 1
    assert proc.returncode == SIGKILL_RC

# A real printing CLI exercises PATH, pipes, decoding, and reaping without relying on timing races.

_WIRING_CLI = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import os, sys
    with open(os.environ["FAKE_CLI_PID_LOG"], "a") as fh:
        fh.write(str(os.getpid()) + "\\n")
    with open(os.environ["FAKE_CLI_LINES"], encoding="utf-8") as fh:
        sys.stdout.write(fh.read())
    """
)

def install_wiring_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, name: str, lines: list[str]) -> Path:
    """Put a real fake *name* executable on ``$PATH``; return the pid-log path."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / name
    # Pin the interpreter to bypass PATH shims.
    script.write_text(_WIRING_CLI.replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1), encoding="utf-8")
    script.chmod(0o755)
    lines_file = tmp_path / f"{name}-lines.jsonl"
    lines_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pid_log = tmp_path / f"{name}-pids.txt"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_CLI_LINES", str(lines_file))
    monkeypatch.setenv("FAKE_CLI_PID_LOG", str(pid_log))
    return pid_log

def assert_pid_reaped(pid_log: Path) -> int:
    """Require the child already gone when drain returns; polling would hide a missed reap."""
    pids = [int(line) for line in pid_log.read_text(encoding="utf-8").split() if line]
    assert len(pids) == 1, f"expected one subprocess, saw {pids}"
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)
    return pids[0]

@pytest.mark.parametrize(
    ("cli", "backend_cls", "lines"),
    [pytest.param("pi", PiBackend, PI_LINES, id="pi"), pytest.param("codex", CodexBackend, CODEX_LINES, id="codex")],
)
async def test_real_subprocess_wiring(
    cli: str, backend_cls: Any, lines: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_log = install_wiring_cli(tmp_path, monkeypatch, name=cli, lines=lines)
    monkeypatch.delenv(STREAM_IDLE_TIMEOUT_ENV, raising=False)
    events = await drain(backend_cls(model="test-model"), tmp_path)
    assert [e.text for e in events if isinstance(e, TextEvent)] == ["slow but alive"]
    assert any(isinstance(e, ResultEvent) for e in events)
    assert_pid_reaped(pid_log)
