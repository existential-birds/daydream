"""Shared Codex mock-process builder.

The returned ``MagicMock`` reproduces the
exact ``stdout``/``stdin``/``wait``/``returncode``/``terminate``/``kill`` shape
that ``CodexBackend.execute`` drives via
``daydream.backends._transport.asyncio.create_subprocess_exec``.
"""

from pathlib import Path
from unittest.mock import MagicMock

from tests.harness.process_replay import (
    GapThenBlockingStdout as GapThenBlockingStdout,
    make_mock_process as _make_mock_process,
    make_mock_process_from_fixture as _make_mock_process_from_fixture,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "codex_jsonl"


def make_mock_process(lines: list[str]) -> MagicMock:
    """Build a Codex async-subprocess stand-in over *lines*."""
    return _make_mock_process(lines, writable_stdin=True)


def make_mock_process_from_fixture(name: str) -> MagicMock:
    """Build a mock process replaying a recorded Codex JSONL fixture file."""
    return _make_mock_process_from_fixture(FIXTURES_DIR, name, writable_stdin=True)
