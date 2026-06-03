# tests/test_read_only_enforcement.py
"""Cross-backend gate: both backends construct a non-mutating profile under read_only.

In-process proof that each backend, given ``read_only=True``, builds a profile
that refuses mutation and permits read-only inspection. The *real* CLI/SDK
denial equivalence is the standing proof of the Task 0 spike
(`.beagle/concepts/handoff-accuracy-redesign/task0-spike-notes.md`); this gate
fails the plan if either backend's profile construction regresses.
"""

from pathlib import Path
from unittest.mock import patch

import pytest


def test_claude_read_only_profile_refuses_mutation():
    """Claude's observable refusal is the Bash-guard decision."""
    from daydream.backends.claude import _is_read_only_command

    # Mutating commands denied; read-only inspection allowed.
    assert _is_read_only_command("git commit -m x") is False
    assert _is_read_only_command("git reset --hard HEAD") is False
    assert _is_read_only_command("rm -rf build") is False
    # Newline/carriage-return command separators are bash chaining that shlex
    # elides as whitespace; they must be rejected on the raw string.
    assert _is_read_only_command("ls \nrm -rf /") is False
    assert _is_read_only_command("cat foo\rrm x") is False
    assert _is_read_only_command("git log") is True
    assert _is_read_only_command("git blame -L 1,1 f.py") is True


@pytest.mark.asyncio
async def test_claude_read_only_guard_blocks_write_tool():
    """Under read_only, the guard denies the Write tool outright (not just Bash)."""
    from daydream.backends.claude import _read_only_guard

    decision = await _read_only_guard(
        {"tool_name": "Write", "tool_input": {"file_path": "x", "content": "y"}}, None, {},
    )
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.asyncio
async def test_codex_read_only_profile_refuses_mutation():
    """Codex passes --sandbox read-only (never danger-full-access) under read_only."""
    from daydream.backends.codex import CodexBackend

    backend = CodexBackend(model="gpt-x")

    captured: dict[str, object] = {}

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        raise RuntimeError("stop after argv capture")

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", fake_exec):
        with pytest.raises(RuntimeError):
            async for _ in backend.execute(Path("/tmp"), "p", read_only=True):
                pass

    flat_args = list(captured["args"])  # type: ignore[arg-type]
    assert "read-only" in flat_args
    assert "danger-full-access" not in flat_args
