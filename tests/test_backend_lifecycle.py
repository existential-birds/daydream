"""#1221: one shared reap / exit-check / teardown across the CLI backends."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from daydream.backends import (
    effective_fanout_concurrency,
    resolve_fanout_concurrency,
)
from daydream.backends.codex import CodexBackend, CodexError
from daydream.backends.osprey import OspreyBackend, OspreyError
from daydream.backends.pi import PiBackend, PiError
from tests.harness.fake_cli_process import FakeCliProcess
from tests.harness.osprey_jsonl import osprey_session

DIAG = [f"diag-{i:02d}" for i in range(1, 26)]  # 25 > every capture window (codex/pi 20, osprey 10)


# Real drivers with only OS process spawning replaced.


async def _drive(
    backend: Any, stdout_lines: list[str], *, exit_code: int = 0, stderr_lines: list[str] | None = None,
) -> tuple[list[Any], FakeCliProcess]:
    """Return events and the fake child. Osprey receives JSONL stdout separately from diagnostic stderr;
    Codex/Pi merge their streams.
    """
    proc = FakeCliProcess(stdout_lines, exit_code=exit_code, stderr_lines=stderr_lines)
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=proc):
        events = [event async for event in backend.execute(Path("/tmp"), "p")]
    return events, proc


def _osprey_stream() -> list[str]:
    return [json.dumps(event) for event in osprey_session()]


async def test_codex_process_exit_message_anchor_and_count() -> None:
    with pytest.raises(CodexError) as exc_info:
        await _drive(CodexBackend(model="fixture-model"), DIAG, exit_code=1)
    assert str(exc_info.value) == (
        "Codex CLI exited with return code 1.\nCodex CLI output (last 10 non-JSON lines):\n"
        + "\n".join(DIAG[15:25])  # rolling tail-20 keeps 6..25; the excerpt is the stream's last ten
    )
    assert exc_info.value.category == "PROCESS_EXIT"
    assert not hasattr(exc_info.value, "retryable")  # codex passes no retryable= kwarg


@pytest.mark.parametrize("returncode,retryable", [
    (-9, True),
    (137, True),
    (1, False),
])
async def test_pi_process_exit_message_anchor_and_count(returncode: int, retryable: bool) -> None:
    proc = FakeCliProcess(DIAG, exit_code=returncode)
    backend = PiBackend(model="glm-5.2")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=proc):
        with pytest.raises(PiError) as exc_info:
            async for _ in backend.execute(Path("/tmp"), "p"):
                pass
    assert str(exc_info.value) == (
        f"Pi CLI exited with return code {returncode}.\nPi CLI output (last 10 non-JSON lines):\n"
        + "\n".join(DIAG[10:20])
    )
    assert exc_info.value.category == "PROCESS_EXIT" and exc_info.value.retryable is retryable
    assert proc.reaped and proc.stdin.closed and proc._transport.closed
    assert backend._transports == []


async def test_osprey_process_exit_message_anchor_and_count() -> None:
    with pytest.raises(OspreyError) as exc_info:
        await _drive(
            OspreyBackend(osprey_binary="fake"),
            _osprey_stream(),  # valid protocol/session_start/…/session_end JSONL
            exit_code=1,
            stderr_lines=DIAG,  # 25 drained stderr lines; the sink caps at 10
        )
    assert str(exc_info.value) == (
        "Osprey CLI exited with return code 1: " + "\n".join(DIAG[:10])
    )
    assert exc_info.value.category == "PROCESS_EXIT"
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("raw,expected", [("6", 6), ("0", 10)])
def test_endpoint_fanout_admission_and_workflow_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    raw: str,
    expected: int,
) -> None:
    """The shared positive rule owns admission; actual backend journeys own defaults and forwarding."""
    monkeypatch.setenv("DAYDREAM_PI_FANOUT_CONCURRENCY", raw)
    hint = resolve_fanout_concurrency("DAYDREAM_PI_FANOUT_CONCURRENCY", 10)
    assert hint == expected
    assert effective_fanout_concurrency(2, SimpleNamespace(fanout_concurrency=hint)) == 2
    assert ("must be positive; using default 10" in caplog.text) is (raw == "0")
