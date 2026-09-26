"""Shared Pi mock-process builder.

Mirrors :mod:`tests.harness.codex_replay`, differing only in the subprocess
shape the backend drives: Pi's prompt is a positional argument (not stdin like
Codex), so ``PiBackend.execute`` opens the process with ``stdin=DEVNULL`` and
never writes, and the mock sets ``stdin=None`` accordingly.
"""

from pathlib import Path
from unittest.mock import MagicMock

from tests.harness.process_replay import (
    make_mock_process as _make_mock_process,
    make_mock_process_from_fixture as _make_mock_process_from_fixture,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "pi_jsonl"


def make_mock_process(lines: list[str]) -> MagicMock:
    """Build a Pi async-subprocess stand-in over *lines*."""
    return _make_mock_process(lines, writable_stdin=False)


def make_mock_process_from_fixture(name: str) -> MagicMock:
    """Build a mock process replaying a recorded Pi JSONL fixture file."""
    return _make_mock_process_from_fixture(FIXTURES_DIR, name, writable_stdin=False)
