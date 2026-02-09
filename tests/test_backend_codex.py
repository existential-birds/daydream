# tests/test_backend_codex.py
"""Tests for CodexBackend with canned JSONL fixtures."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daydream.backends import (
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.codex import CodexBackend, CodexError, _unwrap_shell_command

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "codex_jsonl"


def _make_mock_process(fixture_name: str):
    """Create a mock asyncio.subprocess.Process from a JSONL fixture file."""
    fixture_path = FIXTURES_DIR / fixture_name
    lines = fixture_path.read_text().strip().split("\n")

    class MockStdout:
        def __init__(self):
            self._lines = iter(lines)

        async def readline(self):
            try:
                line = next(self._lines)
                return (line + "\n").encode()
            except StopIteration:
                return b""

    process = MagicMock()
    process.stdout = MockStdout()
    process.stdin = MagicMock()
    process.stdin.write = MagicMock()
    process.stdin.close = MagicMock()
    process.wait = AsyncMock(return_value=0)
    process.returncode = 0
    process.terminate = MagicMock()
    process.kill = MagicMock()
    return process


@pytest.mark.asyncio
async def test_simple_text_events():
    backend = CodexBackend(model="gpt-5.2-codex")
    mock_proc = _make_mock_process("simple_text.jsonl")

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Say hello"):
            events.append(event)

    text_events = [e for e in events if isinstance(e, TextEvent)]
    cost_events = [e for e in events if isinstance(e, CostEvent)]
    result_events = [e for e in events if isinstance(e, ResultEvent)]

    assert len(text_events) == 1
    assert text_events[0].text == "Hello from Codex"
    assert len(cost_events) == 1
    assert cost_events[0].cost_usd is None
    assert cost_events[0].input_tokens == 100
    assert cost_events[0].output_tokens == 50
    assert len(result_events) == 1
    assert result_events[0].continuation is not None
    assert result_events[0].continuation.backend == "codex"
    assert result_events[0].continuation.data["thread_id"] == "th_abc123"


@pytest.mark.asyncio
async def test_tool_use_events():
    backend = CodexBackend()
    mock_proc = _make_mock_process("tool_use.jsonl")

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Run ls"):
            events.append(event)

    thinking = [e for e in events if isinstance(e, ThinkingEvent)]
    tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    texts = [e for e in events if isinstance(e, TextEvent)]

    # Reasoning → ThinkingEvent
    assert len(thinking) == 1
    assert thinking[0].text == "Let me run a command"

    # command_execution → ToolStart + ToolResult
    assert any(ts.name == "shell" and ts.input == {"command": "ls -la"} for ts in tool_starts)
    assert any(tr.output == "file.py\ntest.py" and not tr.is_error for tr in tool_results)

    # file_change → synthetic ToolStart("patch") + ToolResult
    assert any(ts.name == "patch" for ts in tool_starts)
    assert any("main.py" in tr.output for tr in tool_results)

    # agent_message → TextEvent
    assert any(t.text == "Done!" for t in texts)


@pytest.mark.asyncio
async def test_structured_output():
    backend = CodexBackend()
    mock_proc = _make_mock_process("structured_output.jsonl")
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Parse", output_schema=schema):
            events.append(event)

    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output == {
        "issues": [{"id": 1, "description": "Fix type hints", "file": "app.py", "line": 5}]
    }


@pytest.mark.asyncio
async def test_turn_failed_raises():
    backend = CodexBackend()
    mock_proc = _make_mock_process("turn_failed.jsonl")

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(CodexError, match="Model returned an error"):
            async for _ in backend.execute(Path("/tmp"), "Fail"):
                pass


@pytest.mark.asyncio
async def test_continuation_token_resumes():
    """Test that continuation token is passed as 'resume' argument."""
    from daydream.backends import ContinuationToken

    backend = CodexBackend()
    mock_proc = _make_mock_process("simple_text.jsonl")
    token = ContinuationToken(backend="codex", data={"thread_id": "th_prev"})

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "Continue", continuation=token):
            pass

        # Verify 'resume' and thread_id appear in the args
        call_args = mock_exec.call_args
        flat_args = list(call_args.args) if call_args.args else []
        # asyncio.create_subprocess_exec takes *args
        assert "resume" in flat_args
        assert "th_prev" in flat_args


@pytest.mark.asyncio
async def test_streamed_structured_output_via_item_updated():
    """Text delivered via item.updated deltas (item.completed has empty content)."""
    backend = CodexBackend()
    mock_proc = _make_mock_process("streamed_structured_output.jsonl")
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Parse", output_schema=schema):
            events.append(event)

    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output == {
        "issues": [{"id": 1, "description": "Missing type hint", "file": "app.py", "line": 10}]
    }

    # Text should also be yielded as a TextEvent
    text_events = [e for e in events if isinstance(e, TextEvent)]
    assert len(text_events) == 1


@pytest.mark.asyncio
async def test_output_text_content_blocks():
    """agent_message with output_text content blocks (schema-constrained)."""
    backend = CodexBackend()
    mock_proc = _make_mock_process("output_text_blocks.jsonl")
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Parse", output_schema=schema):
            events.append(event)

    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output == {
        "issues": [{"id": 1, "description": "Bad import", "file": "main.py", "line": 3}]
    }


@pytest.mark.asyncio
async def test_toplevel_text_field():
    """Real Codex format: text directly on item, not in content blocks."""
    backend = CodexBackend()
    mock_proc = _make_mock_process("toplevel_text.jsonl")
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Parse", output_schema=schema):
            events.append(event)

    # reasoning with top-level text → ThinkingEvent
    thinking = [e for e in events if isinstance(e, ThinkingEvent)]
    assert len(thinking) == 1
    assert "read the review" in thinking[0].text

    # agent_message with top-level text containing JSON → structured output
    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output == {
        "issues": [{"id": 1, "description": "Missing yield for non-result events", "file": "agents/architect.py", "line": 134}]
    }

    # Also emitted as TextEvent
    text_events = [e for e in events if isinstance(e, TextEvent)]
    assert len(text_events) == 1


@pytest.mark.asyncio
async def test_turn_completed_result_field():
    """Structured output returned in turn.completed result field."""
    backend = CodexBackend()
    mock_proc = _make_mock_process("turn_completed_result.jsonl")
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Parse", output_schema=schema):
            events.append(event)

    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output == {
        "issues": [{"id": 1, "description": "Unused variable", "file": "utils.py", "line": 22}]
    }


def test_format_skill_invocation():
    backend = CodexBackend()
    # Should strip namespace prefix and use $ syntax
    result = backend.format_skill_invocation("beagle-python:review-python")
    assert result == "$review-python"


def test_format_skill_invocation_with_args():
    backend = CodexBackend()
    result = backend.format_skill_invocation("beagle-core:fetch-pr-feedback", "--pr 42 --bot mybot")
    assert result == "$fetch-pr-feedback --pr 42 --bot mybot"


def test_format_skill_invocation_no_namespace():
    backend = CodexBackend()
    result = backend.format_skill_invocation("commit-push")
    assert result == "$commit-push"


class TestUnwrapShellCommand:
    """Tests for _unwrap_shell_command helper."""

    def test_zsh_wrapper_with_cd(self):
        cmd = '/bin/zsh -lc "cd /home/user/project && make test"'
        assert _unwrap_shell_command(cmd) == "make test"

    def test_bash_wrapper_with_cd(self):
        cmd = '/bin/bash -lc "cd /tmp/work && pytest -x"'
        assert _unwrap_shell_command(cmd) == "pytest -x"

    def test_sh_wrapper_with_cd(self):
        cmd = '/bin/sh -lc "cd /app && echo hello"'
        assert _unwrap_shell_command(cmd) == "echo hello"

    def test_wrapper_without_cd(self):
        cmd = '/bin/zsh -lc "ls -la"'
        assert _unwrap_shell_command(cmd) == "ls -la"

    def test_plain_command_passthrough(self):
        assert _unwrap_shell_command("ls -la") == "ls -la"

    def test_empty_command(self):
        assert _unwrap_shell_command("") == ""

    def test_single_quotes(self):
        cmd = "/bin/zsh -lc 'cd /project && git status'"
        assert _unwrap_shell_command(cmd) == "git status"

    def test_unquoted_simple(self):
        """Real Codex format: no quotes around simple commands."""
        assert _unwrap_shell_command("/bin/zsh -lc ls") == "ls"

    def test_single_quoted_git_diff(self):
        """Real Codex format: single-quoted multi-word command."""
        cmd = "/bin/zsh -lc 'git diff main...HEAD'"
        assert _unwrap_shell_command(cmd) == "git diff main...HEAD"

    def test_double_quoted_sed(self):
        """Real Codex format: double-quoted command with inner single quotes."""
        cmd = """/bin/zsh -lc "sed -n '1,260p' amelia/agents/architect.py\""""
        assert _unwrap_shell_command(cmd) == "sed -n '1,260p' amelia/agents/architect.py"
