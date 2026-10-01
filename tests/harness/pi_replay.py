"""Shared Pi mock-process builder.

Mirrors :mod:`tests.harness.codex_replay`, differing only in the subprocess
shape the backend drives: Pi's prompt travels through a positional @file attachment (not stdin like
Codex), so ``PiBackend.execute`` opens the process with ``stdin=DEVNULL`` and
never writes, and the mock sets ``stdin=None`` accordingly.
"""

from pathlib import Path

from tests.harness.process_replay import bind_replay

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "pi_jsonl"

make_mock_process, make_mock_process_from_fixture = bind_replay(FIXTURES_DIR, writable_stdin=False)
