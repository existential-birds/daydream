"""Transport contract tests: real subprocess, no mocks on the happy path."""

from __future__ import annotations

import json
import os
import sys

import anyio
import pytest

from daydream.backends._subprocess import (
    StreamStalledError,
    stream_idle_timeout_s,
)
from daydream.backends._transport import (
    CliTransport,
)
from tests.harness.processes import GROUP_HOLDER_CLI, wait_for_process_group_gone

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
    t = CliTransport(limit=LIMIT, cli="fake", argv=[sys.executable, "-c", child],
        stdin_data=b"prompt\n",
    )
    await t.start()
    events = [json.loads(line) async for line in t.lines(timeout_for_line=lambda: 5.0)]
    assert events == [{"echo": "prompt\n"}]
    assert await t.wait() == 0
    assert t.stdin_closed is True

async def test_transport_nonzero_exit_returns_status() -> None:
    t = CliTransport(limit=LIMIT, cli="fake", argv=[sys.executable, "-c", emit_lines("not json", exit_code=3)],)
    await t.start()
    seen: list[str] = []
    async for line in t.lines(timeout_for_line=lambda: 5.0):
        seen.append(line)
    assert seen == ["not json"]
    assert await t.wait() == 3
    assert t.returncode == 3

async def test_transport_stderr_drain_task_feeds_sink() -> None:
    diagnostics: list[str] = []
    t = CliTransport(limit=LIMIT, cli="fake", argv=[sys.executable, "-c", emit_lines('{"a":1}', to_stderr="boom")],
        stderr_sink=diagnostics.append,
    )
    await t.start()
    events = [json.loads(line) async for line in t.lines(timeout_for_line=lambda: 5.0)]
    assert events == [{"a": 1}]
    assert await t.wait() == 0
    await t.drain_finished()
    assert diagnostics == ["boom"]


_HANGING_CLI = "import time\ntime.sleep(60)\n"

async def test_transport_idle_timeout_fires_on_silent_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stream that goes silent for the window raises StreamStalledError.

    Feed the transport a real hanging child (never writes), with the env override shrinking the window — the same
    env contract as production."""
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

    t = CliTransport(cli="fake", limit=LIMIT, argv=[sys.executable, "-c", GROUP_HOLDER_CLI])
    await t.start()
    proc = t._proc
    assert proc is not None
    pgid = os.getpgid(proc.pid)
    assert pgid == proc.pid  # start_new_session => session leader => pid is the pgid
    it = t.lines(timeout_for_line=lambda: 5.0).__aiter__()
    assert await it.__anext__() == "UP"
    await t.terminate()
    await t.terminate()  # double-call must not raise
    assert t.returncode is not None
    await wait_for_process_group_gone(pgid)  # grandchild must be gone too

async def test_transport_cancel_all_is_shielded() -> None:
    """Cancelling the surrounding scope while lines() pends still reaps the group.

    Mirrors the shielded-teardown shape in tests/test_subprocess_lifecycle.py: the cancel fires mid-read, unwinds
    into the generator's caller, and the ``finally`` teardown (cancel_all) must run to completion despite the
    still- cancelled scope."""


    t = CliTransport(cli="fake", limit=LIMIT, argv=[sys.executable, "-c", GROUP_HOLDER_CLI])
    await t.start()
    assert t._proc is not None
    pgid = os.getpgid(t._proc.pid)

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
    await wait_for_process_group_gone(pgid)
