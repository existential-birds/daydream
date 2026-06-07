"""Shared Codex mock-process builder.

Consolidates the two formerly-duplicated ``_make_mock_process`` helpers
(``tests/contract/_loaders.py`` lines variant and ``tests/test_backend_codex.py``
fixture variant) into one builder. The returned ``MagicMock`` reproduces the
exact ``stdout``/``stdin``/``wait``/``returncode``/``terminate``/``kill`` shape
that ``CodexBackend.execute`` drives via
``daydream.backends.codex.asyncio.create_subprocess_exec``.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "codex_jsonl"


def make_mock_process(lines: list[str]) -> MagicMock:
    """Build an async-subprocess stand-in that yields *lines* through stdout.

    Args:
        lines: JSONL lines (without trailing newlines) to replay through
            ``stdout.readline()``. After the last line, ``readline()`` returns
            ``b""`` to signal EOF.

    Returns:
        A ``MagicMock`` mimicking ``asyncio.subprocess.Process`` with the
        ``stdout``/``stdin``/``wait``/``returncode``/``terminate``/``kill``
        shape ``CodexBackend.execute`` relies on.
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
    process.stdin = MagicMock()
    process.stdin.write = MagicMock()
    process.stdin.close = MagicMock()
    process.wait = AsyncMock(return_value=0)
    process.returncode = 0
    process.terminate = MagicMock()
    process.kill = MagicMock()
    return process


def make_mock_process_from_fixture(name: str) -> MagicMock:
    """Build a mock process replaying a recorded JSONL fixture file.

    Args:
        name: Filename under ``tests/fixtures/codex_jsonl/`` to read.

    Returns:
        The result of :func:`make_mock_process` over the fixture's lines.
    """
    fixture_path = FIXTURES_DIR / name
    lines = fixture_path.read_text().strip().split("\n")
    return make_mock_process(lines)
