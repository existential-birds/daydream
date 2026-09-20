"""Shared subprocess mock builder for JSONL replay backends.

The returned ``MagicMock`` reproduces the exact
``stdout``/``stdin``/``wait``/``returncode``/``terminate``/``kill`` shape that
the Codex and Pi backends drive via
``daydream.backends._transport.asyncio.create_subprocess_exec``.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


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
    """Build an async-subprocess stand-in that yields *lines* through stdout.

    Args:
        lines: JSONL lines (without trailing newlines) to replay through
            ``stdout.readline()``. After the last line, ``readline()`` returns
            ``b""`` to signal EOF.
        writable_stdin: Whether the backend writes to the process's stdin. Codex
            prompts over stdin; Pi passes the prompt positionally and opens the
            process with ``stdin=DEVNULL`` (``stdin=None``).

    Returns:
        A ``MagicMock`` mimicking ``asyncio.subprocess.Process``.
    """

    class _MockStdout:
        def __init__(self) -> None:
            self._lines = iter(lines)

        async def readline(self) -> bytes:
            try:
                line = next(self._lines)
                return (line + "\n").encode()
            except StopIteration:
                return b""

    process = MagicMock()
    process.stdout = _MockStdout()
    if writable_stdin:
        process.stdin = MagicMock()
        process.stdin.write = MagicMock()
        process.stdin.close = MagicMock()
    else:
        process.stdin = None
    process.wait = AsyncMock(return_value=0)
    process.returncode = 0
    process.terminate = MagicMock()
    process.kill = MagicMock()
    return process


def make_mock_process_from_fixture(fixtures_dir: Path, name: str, *, writable_stdin: bool) -> MagicMock:
    """Build a mock process replaying a recorded JSONL fixture file."""
    fixture_path = fixtures_dir / name
    lines = fixture_path.read_text().strip().split("\n")
    return make_mock_process(lines, writable_stdin=writable_stdin)
