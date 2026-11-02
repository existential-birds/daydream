"""#1221: one shared reap / exit-check / teardown across the CLI backends."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from daydream.backends import (
    effective_fanout_concurrency,
    resolve_fanout_concurrency,
)
from tests.harness.fake_cli_process import FakeCliProcess

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


@pytest.mark.parametrize("raw,expected", [("0", 10)])
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
