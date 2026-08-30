"""Transport contract tests: real subprocess, no mocks on the happy path."""

from __future__ import annotations

import json
import sys

import pytest

from daydream.backends._subprocess import (
    StreamStalledError,
    stream_idle_timeout_s,
)
from daydream.backends._transport import (
    CliTransport,
    StderrPolicy,
    StdinMode,
    TransportExitError,
)

LIMIT = 2**16


def emit_lines(*lines: str, exit_code: int = 0, to_stderr: str = "") -> str:
    body = "".join(f"print({line!r})\n" for line in lines)
    stderr_part = f"print({to_stderr!r}, file=sys.stderr)" if to_stderr else ""
    return "import sys\n" + body + stderr_part + f"\nraise SystemExit({exit_code})\n"


async def test_transport_streams_jsonl_lines_with_exit_code() -> None:
    t = CliTransport(cli="fake", limit=LIMIT, argv=[sys.executable, "-c", emit_lines('{"a":1}', '{"b":2}')])
    await t.start()
    events = [json.loads(line) async for line in t.lines(timeout_for_line=lambda: 5.0)]
    assert events == [{"a": 1}, {"b": 2}]
    assert await t.wait() == 0  # exit code surfaced, transport reaped
    assert t.returncode == 0


async def test_transport_writes_stdin_then_closes() -> None:
    child = (
        "import sys\n"
        'data = sys.stdin.read()\n'
        'print(json.dumps({"echo": data}))\n'
        "raise SystemExit(0)\n"
    ).replace("import sys\n", "import json, sys\n", 1)
    t = CliTransport(
        limit=LIMIT,
        cli="fake",
        argv=[sys.executable, "-c", child],
        stdin_mode=StdinMode.PIPE,
        stdin_data=b"prompt\n",
    )
    await t.start()
    events = [json.loads(line) async for line in t.lines(timeout_for_line=lambda: 5.0)]
    assert events == [{"echo": "prompt\n"}]
    assert await t.wait() == 0
    assert t.stdin_closed is True


async def test_transport_nonzero_exit_raises_exit_error_with_diagnostics() -> None:
    diagnostics: list[str] = []
    t = CliTransport(
        limit=LIMIT,
        cli="fake",
        argv=[sys.executable, "-c", emit_lines("not json", exit_code=3)],
        diagnostics_sink=diagnostics.append,
    )
    await t.start()
    seen: list[str] = []
    async for line in t.lines(timeout_for_line=lambda: 5.0):
        seen.append(line)
        if not line.startswith("{"):
            # Backend-side protocol mapping: non-JSON stdout goes to the sink.
            t.note_diagnostic(line)
    assert seen == ["not json"]
    with pytest.raises(TransportExitError) as excinfo:
        await t.wait()
    assert excinfo.value.returncode == 3
    assert diagnostics == ["not json"]
    assert excinfo.value.diagnostics == ["not json"]
    assert t.returncode == 3


async def test_transport_stderr_drain_task_feeds_sink() -> None:
    diagnostics: list[str] = []
    t = CliTransport(
        limit=LIMIT,
        cli="fake",
        argv=[sys.executable, "-c", emit_lines('{"a":1}', to_stderr="boom")],
        stderr_policy=StderrPolicy.DRAIN_TASK,
        stderr_sink=diagnostics.append,
    )
    await t.start()
    events = [json.loads(line) async for line in t.lines(timeout_for_line=lambda: 5.0)]
    assert events == [{"a": 1}]
    assert await t.wait() == 0
    await t.drain_finished()
    assert diagnostics == ["boom"]


_HANGING_CLI = "import time\ntime.sleep(60)\n"

# Mirrors _HOLDER_SCRIPT in tests/test_subprocess_lifecycle.py: the CLI forks a
# `sleep` grandchild (which inherits the stdout pipe), reports readiness by
# printing "UP", then hangs. A python CLI is used because bash defers SIGTERM
# while a child runs, which would stall every test on TERMINATE_GRACE_S.
_GROUP_HOLDER_CLI = (
    "import subprocess, sys, time; "
    "subprocess.Popen(['sleep', '60']); "
    "print('UP', flush=True); "
    "time.sleep(1000)"
)


async def _wait_for_group_gone(pgid: int, *, timeout_s: float = 30.0) -> None:
    """Await *pgid*'s disappearance (mirrors tests/test_subprocess_lifecycle.py).

    A group-signal kill reaps the direct child synchronously, but a grandchild
    reparented to PID 1 lingers as a zombie until init reaps it — and a zombie
    still answers ``killpg(pgid, 0)``. Polling until the group is gone makes
    the assertion deterministic: the observable outcome is "no process remains
    in the group", not "the group vanished by the next instruction".
    """
    import asyncio as _asyncio
    import os as _os

    loop = _asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        try:
            _os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        if loop.time() > deadline:
            raise TimeoutError(f"process group {pgid} still alive after {timeout_s}s")
        await _asyncio.sleep(0.01)


async def test_transport_idle_timeout_fires_on_silent_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stream that goes silent for the window raises StreamStalledError.

    Feed the transport a real hanging child (never writes), with the env
    override shrinking the window — the same env contract as production.
    """
    monkeypatch.setenv("DAYDREAM_STREAM_IDLE_TIMEOUT_S", "0.2")
    t = CliTransport(cli="fake", limit=LIMIT, argv=[sys.executable, "-c", _HANGING_CLI])
    await t.start()
    with pytest.raises(StreamStalledError) as exc:
        [line async for line in t.lines(timeout_for_line=lambda: stream_idle_timeout_s())]
    assert "fake CLI produced no output for 0.2s" in str(exc.value)
    await t.terminate()  # teardown reaps the hung child
    assert t.returncode is not None


async def test_transport_teardown_is_idempotent_and_group_signalling() -> None:
    """Double terminate() must not raise, and the grandchild dies with the group."""
    import os

    t = CliTransport(cli="fake", limit=LIMIT, argv=[sys.executable, "-c", _GROUP_HOLDER_CLI])
    await t.start()
    proc = t.processes[0]
    pgid = os.getpgid(proc.pid)
    assert pgid == proc.pid  # start_new_session => session leader => pid is the pgid
    it = t.lines(timeout_for_line=lambda: 5.0).__aiter__()
    assert await it.__anext__() == "UP"
    await t.terminate()
    await t.terminate()  # double-call must not raise
    assert t.returncode is not None
    await _wait_for_group_gone(pgid)  # grandchild must be gone too


async def test_transport_cancel_all_is_shielded() -> None:
    """Cancelling the surrounding scope while lines() pends still reaps the group.

    Mirrors the shielded-teardown shape in tests/test_subprocess_lifecycle.py:
    the cancel fires mid-read, unwinds into the generator's caller, and the
    ``finally`` teardown (cancel_all) must run to completion despite the still-
    cancelled scope.
    """
    import os

    import anyio

    t = CliTransport(cli="fake", limit=LIMIT, argv=[sys.executable, "-c", _GROUP_HOLDER_CLI])
    await t.start()
    pgid = os.getpgid(t.processes[0].pid)

    async def consume() -> None:
        try:
            async for _ in t.lines(timeout_for_line=lambda: 5.0):
                pass
        finally:
            await CliTransport.cancel_all([t])

    with anyio.move_on_after(0.2) as scope:
        await consume()
    assert scope.cancelled_caught  # cancel fired while lines() was pending
    assert t.returncode is not None
    await _wait_for_group_gone(pgid)
