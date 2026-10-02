"""Shared subprocess mocks for Codex/Pi JSONL replay through the transport seam, including stdin,
stdout, wait, returncode, terminate, and kill.
"""

import asyncio
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from daydream.backends import AgentEvent, Backend
from tests.harness.fake_cli_process import ImmediateStdout, cli_process


async def replay_process(
    backend: Backend, process: MagicMock, cwd: Path, prompt: str, **options: Any,
) -> tuple[list[AgentEvent], AsyncMock]:
    """Collect a canned execution and expose spawn arguments. Tests observing partial output,
    cancellation, or concurrency drive execute directly to retain explicit observation timing.
    """
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=process) as spawn:
        events = [event async for event in backend.execute(cwd, prompt, **options)]
    return events, spawn


class GapThenBlockingStdout:
    """A stdout that yields one parser-gap line, then blocks forever."""

    def __init__(self) -> None:
        self.sent_gap = False

    async def readline(self) -> bytes:
        if not self.sent_gap:
            self.sent_gap = True
            return b'{"type":"future.before.stall"}\n'
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def make_mock_process(lines: list[str], *, writable_stdin: bool) -> MagicMock:
    """Replay newline-free JSONL lines through stdout.readline, followed by b"" EOF. writable_stdin
    selects Codex's prompt pipe; Pi passes prompts positionally and uses DEVNULL/stdin=None. Return
    a Process-shaped MagicMock.
    """
    process = cli_process(ImmediateStdout(lines))
    if not writable_stdin:
        process.stdin = None
    return process


def make_mock_process_from_fixture(fixtures_dir: Path, name: str, *, writable_stdin: bool) -> MagicMock:
    """Build a mock process replaying a recorded JSONL fixture file."""
    fixture_path = fixtures_dir / name
    lines = fixture_path.read_text().strip().split("\n")
    return make_mock_process(lines, writable_stdin=writable_stdin)


def bind_replay(fixtures_dir: Path, *, writable_stdin: bool
) -> tuple[Callable[[list[str]], MagicMock], Callable[[str], MagicMock]]:
    """Bind one backend's fixture dir and stdin mode to the replay builders."""
    return (partial(make_mock_process, writable_stdin=writable_stdin),
        partial(make_mock_process_from_fixture, fixtures_dir, writable_stdin=writable_stdin),
    )
