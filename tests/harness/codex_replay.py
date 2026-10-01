"""Shared Codex mock-process builder.

The returned ``MagicMock`` reproduces the
exact ``stdout``/``stdin``/``wait``/``returncode``/``terminate``/``kill`` shape
that ``CodexBackend.execute`` drives via
``daydream.backends._transport.asyncio.create_subprocess_exec``.
"""

from pathlib import Path

from tests.harness.process_replay import (
    GapThenBlockingStdout as GapThenBlockingStdout,
    bind_replay,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "codex_jsonl"

make_mock_process, make_mock_process_from_fixture = bind_replay(FIXTURES_DIR, writable_stdin=True)
