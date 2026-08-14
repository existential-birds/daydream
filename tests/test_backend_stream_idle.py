"""Idle-stall detection and teardown for the pi/codex subprocess backends.

Deterministic by construction — no test here races two clocks. The stall,
retry, wall-budget, and SIGKILL-escalation tests drive the real backend
code (spawn call, readline loop, idle window, shielded teardown) against an
in-process :class:`~tests.harness.fake_cli_process.FakeCliProcess` at the
``asyncio.create_subprocess_exec`` boundary. Silence is modeled as a
``readline()`` that never resolves, so a timer under test is the ONLY timer
in the test: it may fire late under load, but the outcome cannot flip.

The two wiring tests at the end spawn one REAL subprocess each to prove the
production spawn plumbing (argv resolution, pipe wiring, decode, reap) against
a fake CLI that unconditionally prints its stream and exits — no hang, no
scripted cadence, no timeout in play, hence equally deterministic.

Assertions are on observable outcomes: the exception that terminates the
turn, whether the child was killed and reaped, how many children were
launched, the written ATIF trajectory, and (for the wiring tests) the OS
process table.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from daydream.agent import run_agent
from daydream.backends import ResultEvent, TextEvent
from daydream.backends._subprocess import (
    DEFAULT_STREAM_IDLE_TIMEOUT_S,
    STREAM_IDLE_TIMEOUT_ENV,
    StreamStalledError,
    readline_with_idle_timeout,
    stream_idle_timeout_s,
)
from daydream.backends.codex import CodexBackend
from daydream.backends.pi import PiBackend
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder
from tests.harness.fake_cli_process import (
    SIGKILL_RC,
    SIGTERM_RC,
    FakeCliProcess,
    install_fake_cli_process,
)

# A complete, valid stream for each CLI.
PI_LINES = [
    json.dumps({"type": "session", "id": "sess-idle-1"}),
    json.dumps({"type": "agent_start"}),
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
                "role": "assistant",
                "stopReason": "end_turn",
                "usage": {"input": 10, "output": 4, "cost": {"total": 0.01}},
            },
        }
    ),
    json.dumps({"type": "agent_end"}),
]

CODEX_LINES = [
    json.dumps({"type": "thread.started", "thread_id": "thr-idle-1"}),
    json.dumps({"type": "turn.started"}),
    json.dumps(
        {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "slow but alive"}}
    ),
    json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 4}}),
]

# Small only for speed. Correctness never depends on the value: wherever a
# test arms this window, the fake stream is PERMANENTLY silent, so the timer
# firing late (a loaded host) cannot change the outcome — only delay it.
TINY_WINDOW = "0.05"


def assert_stalled_and_reaped(spawner: Any, *, expected_spawns: int = 1) -> FakeCliProcess:
    """Assert exactly *expected_spawns* children ran and the last one was torn down."""
    assert len(spawner.procs) == expected_spawns, (
        f"expected {expected_spawns} subprocess launch(es), saw {len(spawner.procs)}"
    )
    proc = spawner.procs[-1]
    assert proc.returncode is not None, "subprocess was never killed — leaked"
    assert proc.reaped, "subprocess was killed but never wait()ed — zombie"
    return proc


async def drain(backend: Any, cwd: Path) -> list[Any]:
    return [event async for event in backend.execute(cwd, "do the thing")]


# --------------------------------------------------------------------------
# A silent stream trips the idle timeout; the subprocess is torn down.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pi_silent_stream_trips_idle_timeout_and_reaps_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``pi`` that emits two lines then goes silent forever ends the turn.

    Also proves the env override reaches the armed window: the raised error
    carries the exact configured value.
    """
    spawner = install_fake_cli_process(monkeypatch, "pi", lines=PI_LINES[:2], hang=True)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, TINY_WINDOW)

    with pytest.raises(StreamStalledError) as excinfo:
        await drain(PiBackend(model="test-model"), tmp_path)

    assert excinfo.value.cli == "pi"
    assert excinfo.value.timeout_s == float(TINY_WINDOW)
    assert excinfo.value.retryable is True
    assert_stalled_and_reaped(spawner)


@pytest.mark.asyncio
async def test_codex_silent_stream_trips_idle_timeout_and_reaps_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``codex`` that emits two lines then goes silent forever ends the turn."""
    spawner = install_fake_cli_process(monkeypatch, "codex", lines=CODEX_LINES[:2], hang=True)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, TINY_WINDOW)

    with pytest.raises(StreamStalledError) as excinfo:
        await drain(CodexBackend(model="test-model"), tmp_path)

    assert excinfo.value.cli == "codex"
    assert excinfo.value.retryable is True
    assert_stalled_and_reaped(spawner)


@pytest.mark.asyncio
async def test_pi_stalls_before_first_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CLI that never writes anything at all is still caught (startup hang)."""
    spawner = install_fake_cli_process(monkeypatch, "pi", lines=[], hang=True)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, TINY_WINDOW)

    with pytest.raises(StreamStalledError):
        await drain(PiBackend(model="test-model"), tmp_path)

    assert_stalled_and_reaped(spawner)


# --------------------------------------------------------------------------
# A stream with data flowing must NOT trip, however small the window. This is
# the regression that keeps a genuinely slow-but-alive model from being killed
# mid-turn: data availability, not elapsed time, is what feeds the window.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pi_flowing_stream_does_not_trip_a_tiny_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawner = install_fake_cli_process(monkeypatch, "pi", lines=PI_LINES)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, TINY_WINDOW)

    events = await drain(PiBackend(model="test-model"), tmp_path)

    assert [e.text for e in events if isinstance(e, TextEvent)] == ["slow but alive"]
    assert any(isinstance(e, ResultEvent) for e in events)
    assert spawner.procs[0].reaped


@pytest.mark.asyncio
async def test_codex_flowing_stream_does_not_trip_a_tiny_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawner = install_fake_cli_process(monkeypatch, "codex", lines=CODEX_LINES)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, TINY_WINDOW)

    events = await drain(CodexBackend(model="test-model"), tmp_path)

    assert [e.text for e in events if isinstance(e, TextEvent)] == ["slow but alive"]
    assert any(isinstance(e, ResultEvent) for e in events)
    assert spawner.procs[0].reaped


@pytest.mark.asyncio
async def test_idle_window_restarts_after_each_line() -> None:
    """The window bounds each inter-line gap, not the total elapsed stream.

    Two successful reads through the same tiny window, then the SAME window
    value trips on the first gap with no data — the window is re-armed per
    line, so only full silence can trip it.
    """
    reader = asyncio.StreamReader()
    window = float(TINY_WINDOW)

    reader.feed_data(b"one\n")
    assert await readline_with_idle_timeout(reader, cli="pi", timeout_s=window) == b"one\n"
    reader.feed_data(b"two\n")
    assert await readline_with_idle_timeout(reader, cli="pi", timeout_s=window) == b"two\n"

    with pytest.raises(StreamStalledError):
        await readline_with_idle_timeout(reader, cli="pi", timeout_s=window)


# --------------------------------------------------------------------------
# Operator configuration.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_disables_idle_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DAYDREAM_STREAM_IDLE_TIMEOUT_S=0`` opts out; the stream still completes."""
    install_fake_cli_process(monkeypatch, "pi", lines=PI_LINES)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "0")

    assert stream_idle_timeout_s() is None
    events = await drain(PiBackend(model="test-model"), tmp_path)

    assert [e.text for e in events if isinstance(e, TextEvent)] == ["slow but alive"]


@pytest.mark.parametrize("raw", ["not-a-number", "-5", "nan", "inf"])
def test_malformed_override_falls_back_to_default(
    raw: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A garbage/negative/non-finite value never disables or shortens the window."""
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, raw)
    assert stream_idle_timeout_s() == DEFAULT_STREAM_IDLE_TIMEOUT_S


def test_default_exceeds_the_wall_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """The idle window must never act as a shorter, second turn cap.

    Budgeted phases are bounded by ``DEFAULT_WALL_BUDGET_S``; the idle timeout is
    a backstop for the turns nothing else bounds (the improve phases run with no
    wall budget). Keeping it strictly above the wall budget is what guarantees
    this change cannot shorten any phase that works today.
    """
    from daydream.config import DEFAULT_WALL_BUDGET_S

    monkeypatch.delenv(STREAM_IDLE_TIMEOUT_ENV, raising=False)
    assert stream_idle_timeout_s() == DEFAULT_STREAM_IDLE_TIMEOUT_S
    assert DEFAULT_STREAM_IDLE_TIMEOUT_S > DEFAULT_WALL_BUDGET_S


# --------------------------------------------------------------------------
# Retryable — driven through run_agent, the production call site. A stalled
# stream consumes the backend's bounded retry budget after the full idle window.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stall_is_retried_within_bounded_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persistently-stalling ``pi`` consumes only the bounded retry budget."""
    spawner = install_fake_cli_process(monkeypatch, "pi", lines=PI_LINES[:2], hang=True)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, TINY_WINDOW)
    monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0.01")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_MAX_DELAY_S", "0.01")

    backend = PiBackend(model="test-model")
    assert backend.retry_attempts == 3, "test must exercise a backend that does retry"

    trajectory_path = tmp_path / ".daydream" / "trajectory.json"
    recorder = TrajectoryRecorder(
        path=trajectory_path,
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="test-model",
        session_id="stall-test",
    )

    with pytest.raises(StreamStalledError):
        async with recorder:
            await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW)

    assert_stalled_and_reaped(spawner, expected_spawns=4)

    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert trajectory["extra"]["partial"] is True
    errored = [
        step for step in trajectory["steps"]
        if (step.get("extra") or {}).get("error_subtype") == "StreamStalledError"
    ]
    assert errored, "the stall was not recorded on any trajectory step"


@pytest.mark.asyncio
async def test_wall_budget_still_aborts_while_blocked_in_the_idle_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wall budget must still fire while the read sits inside the idle window.

    The idle window is an ``asyncio.timeout`` on the reading task; ``run_agent``'s
    wall budget is an anyio cancel scope around the same task. This drives the
    nesting that matters — outer anyio cancel delivered while the inner asyncio
    timeout is armed — and asserts the abort path stays intact: no exception
    escapes, the turn is marked aborted, and the subprocess is still reaped.
    The wall budget is the only timer that can fire: the fake stream is
    permanently silent and the idle window is far larger.
    """
    spawner = install_fake_cli_process(monkeypatch, "pi", lines=PI_LINES[:2], hang=True)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "3600")

    output, _, budget_reason = await run_agent(
        PiBackend(model="test-model"),
        tmp_path,
        "review",
        phase=DaydreamPhase.REVIEW,
        wall_budget_s=1.0,
    )

    assert budget_reason == "wall_budget_exceeded"
    assert output == ""
    proc = assert_stalled_and_reaped(spawner)
    assert proc.returncode == SIGTERM_RC, "a cooperative child needs no SIGKILL"


@pytest.mark.asyncio
async def test_cancelled_teardown_still_escalates_to_sigkill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child that ignores SIGTERM is still SIGKILLed when the wall budget fires.

    The subprocess is read inside the backend generator, so a wall-budget cancel
    lands straight in the teardown ``finally`` with the parent scope still
    cancelled. That teardown must run to completion regardless: SIGTERM, wait the
    grace, then SIGKILL. If it is not shielded from the cancellation, the first
    await re-raises before the SIGKILL, and a SIGTERM-ignoring child stays alive
    — a real leaked process. The grace is zero so the escalation is immediate:
    ``wait_for(..., timeout=0)`` raises without sleeping on an unexited child.
    """
    monkeypatch.setattr("daydream.backends._subprocess.TERMINATE_GRACE_S", 0.0)
    spawner = install_fake_cli_process(
        monkeypatch, "pi", lines=PI_LINES[:2], hang=True, ignore_sigterm=True
    )
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "3600")

    output, _, budget_reason = await run_agent(
        PiBackend(model="test-model"),
        tmp_path,
        "review",
        phase=DaydreamPhase.REVIEW,
        wall_budget_s=1.0,
    )

    assert budget_reason == "wall_budget_exceeded"
    assert output == ""
    proc = assert_stalled_and_reaped(spawner)
    assert proc.terminate_calls == 1
    assert proc.kill_calls == 1
    assert proc.returncode == SIGKILL_RC


# --------------------------------------------------------------------------
# Real-subprocess wiring. One test per backend proves the production spawn
# plumbing — PATH resolution, pipe wiring, real byte decode, real wait/reap —
# against a fake CLI that unconditionally prints its stream and exits. No
# hang, no cadence, no timer in play: nothing here races anything.
# --------------------------------------------------------------------------

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


def install_wiring_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, name: str, lines: list[str]
) -> Path:
    """Put a real fake *name* executable on ``$PATH``; return the pid-log path."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / name
    # Pin the shebang to the interpreter running the suite so PATH-resolved
    # launcher shims (e.g. pyenv) never sit between the backend and the fake.
    script.write_text(
        _WIRING_CLI.replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1),
        encoding="utf-8",
    )
    script.chmod(0o755)

    lines_file = tmp_path / f"{name}-lines.jsonl"
    lines_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pid_log = tmp_path / f"{name}-pids.txt"

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_CLI_LINES", str(lines_file))
    monkeypatch.setenv("FAKE_CLI_PID_LOG", str(pid_log))
    return pid_log


def assert_pid_reaped(pid_log: Path) -> int:
    """Assert exactly one child ran and is already gone from the process table.

    No polling: the backend ``await``s the child's exit before yielding its
    final events, so by the time ``drain`` returns the pid must be gone.
    """
    pids = [int(line) for line in pid_log.read_text(encoding="utf-8").split() if line]
    assert len(pids) == 1, f"expected one subprocess, saw {pids}"
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)
    return pids[0]


@pytest.mark.asyncio
async def test_pi_real_subprocess_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_log = install_wiring_cli(tmp_path, monkeypatch, name="pi", lines=PI_LINES)
    monkeypatch.delenv(STREAM_IDLE_TIMEOUT_ENV, raising=False)

    events = await drain(PiBackend(model="test-model"), tmp_path)

    assert [e.text for e in events if isinstance(e, TextEvent)] == ["slow but alive"]
    assert any(isinstance(e, ResultEvent) for e in events)
    assert_pid_reaped(pid_log)


@pytest.mark.asyncio
async def test_codex_real_subprocess_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_log = install_wiring_cli(tmp_path, monkeypatch, name="codex", lines=CODEX_LINES)
    monkeypatch.delenv(STREAM_IDLE_TIMEOUT_ENV, raising=False)

    events = await drain(CodexBackend(model="test-model"), tmp_path)

    assert [e.text for e in events if isinstance(e, TextEvent)] == ["slow but alive"]
    assert any(isinstance(e, ResultEvent) for e in events)
    assert_pid_reaped(pid_log)
