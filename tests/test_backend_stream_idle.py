"""Idle-stall detection for the pi/codex subprocess backends.

Every test here spawns a REAL subprocess: a fake ``pi`` / ``codex`` executable is
placed on ``$PATH`` and the genuine ``PiBackend`` / ``CodexBackend`` launches it
through ``asyncio.create_subprocess_exec``, reads its real stdout pipe, and tears
it down through the real ``finally`` path. Only the CLI binary — the external
seam — is faked; the pipe, the readline loop, the idle window, and the
SIGTERM/SIGKILL teardown are all production code.

Assertions are on observable outcomes: the exception that terminates the turn,
the OS process table (a reaped pid answers ``ESRCH``), the on-disk spawn log
(how many subprocesses were actually launched), and the written ATIF trajectory.
"""

from __future__ import annotations

import json
import os
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from daydream.agent import run_agent
from daydream.backends import ResultEvent, TextEvent
from daydream.backends._subprocess import (
    DEFAULT_STREAM_IDLE_TIMEOUT_S,
    STREAM_IDLE_TIMEOUT_ENV,
    StreamStalledError,
    stream_idle_timeout_s,
)
from daydream.backends.codex import CodexBackend
from daydream.backends.pi import PiBackend
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder

# A complete, valid stream for each CLI. Replayed line by line at a configurable
# cadence so a test can make the stream slow without making it silent.
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

# Fake CLI: logs its pid (one line per launch — this is also the spawn counter),
# replays the scripted lines at a fixed cadence, then either exits or hangs
# forever with stdout still open. Hanging with an open pipe is exactly the live
# pathology: readline() blocks with no EOF and no data.
_FAKE_CLI = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import os, sys, time

    with open(os.environ["FAKE_CLI_PID_LOG"], "a") as fh:
        fh.write(str(os.getpid()) + "\\n")
        fh.flush()

    delay = float(os.environ.get("FAKE_CLI_DELAY", "0"))
    with open(os.environ["FAKE_CLI_LINES"], encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().split("\\n") if ln]

    for line in lines:
        time.sleep(delay)
        sys.stdout.write(line + "\\n")
        sys.stdout.flush()

    if os.environ.get("FAKE_CLI_HANG") == "1":
        while True:
            time.sleep(3600)
    """
)


def install_fake_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    lines: list[str],
    delay: float = 0.0,
    hang: bool = False,
) -> Path:
    """Put a fake *name* executable on ``$PATH``; return the spawn/pid log path."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / name
    script.write_text(_FAKE_CLI, encoding="utf-8")
    script.chmod(0o755)

    lines_file = tmp_path / f"{name}-lines.jsonl"
    lines_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pid_log = tmp_path / f"{name}-pids.txt"

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_CLI_LINES", str(lines_file))
    monkeypatch.setenv("FAKE_CLI_PID_LOG", str(pid_log))
    monkeypatch.setenv("FAKE_CLI_DELAY", str(delay))
    monkeypatch.setenv("FAKE_CLI_HANG", "1" if hang else "0")
    return pid_log


def spawned_pids(pid_log: Path) -> list[int]:
    """Pids of every fake-CLI process that actually started, in launch order."""
    if not pid_log.exists():
        return []
    return [int(line) for line in pid_log.read_text(encoding="utf-8").split() if line]


def assert_reaped(pid: int, timeout: float = 10.0) -> None:
    """Assert *pid* is gone from the process table (terminated AND reaped).

    A terminated-but-unreaped child stays a zombie and ``os.kill(pid, 0)``
    still succeeds, so this fails on a leaked process as well as a live one.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail(f"fake CLI pid {pid} is still in the process table — subprocess leaked")


async def drain(backend: Any, cwd: Path) -> list[Any]:
    return [event async for event in backend.execute(cwd, "do the thing")]


# --------------------------------------------------------------------------
# A silent stream trips the idle timeout; the subprocess is torn down.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pi_silent_stream_trips_idle_timeout_and_reaps_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``pi`` that emits two lines then goes silent forever ends the turn."""
    pid_log = install_fake_cli(
        tmp_path, monkeypatch, name="pi", lines=PI_LINES[:2], hang=True
    )
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "1.0")
    backend = PiBackend(model="test-model")

    with pytest.raises(StreamStalledError) as excinfo:
        await drain(backend, tmp_path)

    assert excinfo.value.cli == "pi"
    assert excinfo.value.timeout_s == 1.0
    assert excinfo.value.retryable is False
    pids = spawned_pids(pid_log)
    assert len(pids) == 1, f"expected one pi subprocess, saw {pids}"
    assert_reaped(pids[0])


@pytest.mark.asyncio
async def test_codex_silent_stream_trips_idle_timeout_and_reaps_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``codex`` that emits two lines then goes silent forever ends the turn."""
    pid_log = install_fake_cli(
        tmp_path, monkeypatch, name="codex", lines=CODEX_LINES[:2], hang=True
    )
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "1.0")
    backend = CodexBackend(model="test-model")

    with pytest.raises(StreamStalledError) as excinfo:
        await drain(backend, tmp_path)

    assert excinfo.value.cli == "codex"
    assert excinfo.value.retryable is False
    pids = spawned_pids(pid_log)
    assert len(pids) == 1, f"expected one codex subprocess, saw {pids}"
    assert_reaped(pids[0])


@pytest.mark.asyncio
async def test_pi_stalls_before_first_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CLI that never writes anything at all is still caught (startup hang)."""
    pid_log = install_fake_cli(tmp_path, monkeypatch, name="pi", lines=[], hang=True)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "1.0")

    with pytest.raises(StreamStalledError):
        await drain(PiBackend(model="test-model"), tmp_path)

    assert_reaped(spawned_pids(pid_log)[0])


# --------------------------------------------------------------------------
# A slow-but-emitting stream must NOT trip. This is the regression that keeps a
# genuinely slow model from being killed mid-turn.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pi_slow_but_continuous_stream_does_not_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five lines at 0.4s apart under a 1.0s window: total 2s+, no single gap 1s+."""
    install_fake_cli(tmp_path, monkeypatch, name="pi", lines=PI_LINES, delay=0.4)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "1.0")

    events = await drain(PiBackend(model="test-model"), tmp_path)

    assert [e.text for e in events if isinstance(e, TextEvent)] == ["slow but alive"]
    assert any(isinstance(e, ResultEvent) for e in events)


@pytest.mark.asyncio
async def test_codex_slow_but_continuous_stream_does_not_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same for codex: elapsed time far exceeds the window, inter-line gaps don't."""
    install_fake_cli(tmp_path, monkeypatch, name="codex", lines=CODEX_LINES, delay=0.4)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "1.0")

    events = await drain(CodexBackend(model="test-model"), tmp_path)

    assert [e.text for e in events if isinstance(e, TextEvent)] == ["slow but alive"]
    assert any(isinstance(e, ResultEvent) for e in events)


# --------------------------------------------------------------------------
# Operator configuration.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_env_override_governs_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same 0.6s-cadence stream trips under 0.25s and completes under 5s."""
    install_fake_cli(tmp_path, monkeypatch, name="pi", lines=PI_LINES, delay=0.6)

    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "0.25")
    with pytest.raises(StreamStalledError):
        await drain(PiBackend(model="test-model"), tmp_path)

    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "5")
    events = await drain(PiBackend(model="test-model"), tmp_path)
    assert [e.text for e in events if isinstance(e, TextEvent)] == ["slow but alive"]


@pytest.mark.asyncio
async def test_zero_disables_idle_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DAYDREAM_STREAM_IDLE_TIMEOUT_S=0`` opts out; a slow stream still completes."""
    install_fake_cli(tmp_path, monkeypatch, name="pi", lines=PI_LINES, delay=0.3)
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "0")

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
# Terminal, not retryable — driven through run_agent, the production call site.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stall_is_terminal_and_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One stall costs one idle window, not ``retry_attempts + 1`` of them.

    ``run_agent``'s retry loop re-arms the backend per attempt, so a retryable
    stall would multiply the dead air by the attempt count. With three retries
    configured, exactly one ``pi`` subprocess must ever be launched, and the
    error must reach the caller and land in the ATIF trajectory.
    """
    pid_log = install_fake_cli(
        tmp_path, monkeypatch, name="pi", lines=PI_LINES[:2], hang=True
    )
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "1.0")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0.01")

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

    pids = spawned_pids(pid_log)
    assert len(pids) == 1, f"stall was retried — {len(pids)} pi subprocesses launched: {pids}"
    assert_reaped(pids[0])

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
    """
    pid_log = install_fake_cli(
        tmp_path, monkeypatch, name="pi", lines=PI_LINES[:2], hang=True
    )
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "60")  # far outside the wall budget

    output, _, budget_reason = await run_agent(
        PiBackend(model="test-model"),
        tmp_path,
        "review",
        phase=DaydreamPhase.REVIEW,
        wall_budget_s=1.0,
    )

    assert budget_reason == "wall_budget_exceeded"
    assert output == ""
    assert_reaped(spawned_pids(pid_log)[0])
