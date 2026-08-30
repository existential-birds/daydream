"""Transport contract tests: real subprocess, no mocks on the happy path."""

from __future__ import annotations

import json
import sys

import pytest

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
