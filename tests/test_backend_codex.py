# tests/test_backend_codex.py
"""Tests for CodexBackend with canned JSONL fixtures."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daydream.backends import (
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.backends.codex import (
    _CODEX_STDOUT_LIMIT_BYTES,
    CodexBackend,
    CodexError,
    _unwrap_shell_command,
)
from tests.harness.codex_replay import make_mock_process_from_fixture

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "codex_jsonl"


@pytest.mark.asyncio
async def test_simple_text_events():
    backend = CodexBackend(model="gpt-5.3-codex")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

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
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("tool_use.jsonl")

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
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("structured_output.jsonl")
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
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("turn_failed.jsonl")

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(CodexError, match="Model returned an error"):
            async for _ in backend.execute(Path("/tmp"), "Fail"):
                pass


@pytest.mark.asyncio
async def test_continuation_token_resumes():
    """Test that continuation token is passed as 'resume' argument."""
    from daydream.backends import ContinuationToken

    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
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
async def test_codex_read_only_uses_read_only_sandbox():
    """read_only=True selects --sandbox read-only; danger-full-access absent."""
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "p", read_only=True):
            pass

        flat_args = list(mock_exec.call_args.args)
        assert "read-only" in flat_args
        assert "danger-full-access" not in flat_args
        # --sandbox immediately precedes the mode
        assert flat_args[flat_args.index("--sandbox") + 1] == "read-only"


@pytest.mark.asyncio
async def test_codex_default_uses_full_access_sandbox():
    """read_only=False (default) keeps the existing danger-full-access sandbox."""
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "p"):
            pass

        flat_args = list(mock_exec.call_args.args)
        assert flat_args[flat_args.index("--sandbox") + 1] == "danger-full-access"
        assert "read-only" not in flat_args


@pytest.mark.asyncio
async def test_codex_stdout_limit_allows_large_jsonl_events() -> None:
    backend = CodexBackend(model="fixture-model")
    large_text = "x" * (70 * 1024)
    large_line = (
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": large_text}}) + "\n"
    ).encode()
    lines = [
        large_line,
        b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
    ]
    captured_kwargs: dict[str, object] = {}

    class _LimitAwareStdout:
        def __init__(self, limit: int) -> None:
            self._limit = limit
            self._lines = iter(lines)

        async def readline(self) -> bytes:
            try:
                line = next(self._lines)
            except StopIteration:
                return b""
            if len(line) > self._limit:
                raise ValueError("Separator is found, but chunk is longer than limit")
            return line

    async def fake_exec(*args: object, **kwargs: object) -> MagicMock:
        captured_kwargs.update(kwargs)
        raw_limit = kwargs.get("limit", 64 * 1024)
        limit = raw_limit if isinstance(raw_limit, int) else 64 * 1024
        process = MagicMock()
        process.stdout = _LimitAwareStdout(limit)
        process.stdin = MagicMock()
        process.stdin.write = MagicMock()
        process.stdin.close = MagicMock()
        process.wait = AsyncMock(return_value=0)
        process.returncode = 0
        process.terminate = MagicMock()
        process.kill = MagicMock()
        return process

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", fake_exec):
        events = [event async for event in backend.execute(Path("/tmp"), "large event")]

    text_events = [e for e in events if isinstance(e, TextEvent)]
    assert text_events[0].text == large_text
    assert captured_kwargs["limit"] == _CODEX_STDOUT_LIMIT_BYTES
    assert _CODEX_STDOUT_LIMIT_BYTES > len(large_line)


@pytest.mark.asyncio
async def test_streamed_structured_output_via_item_updated():
    """Text delivered via item.updated deltas (item.completed has empty content)."""
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("streamed_structured_output.jsonl")
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
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("output_text_blocks.jsonl")
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
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("toplevel_text.jsonl")
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
        "issues": [
            {
                "id": 1,
                "description": "Missing yield for non-result events",
                "file": "agents/architect.py",
                "line": 134,
            }
        ]
    }

    # Also emitted as TextEvent
    text_events = [e for e in events if isinstance(e, TextEvent)]
    assert len(text_events) == 1


@pytest.mark.asyncio
async def test_turn_completed_result_field():
    """Structured output returned in turn.completed result field."""
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("turn_completed_result.jsonl")
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


@pytest.mark.asyncio
async def test_turn_completed_cached_input_tokens():
    """Codex emits cached_input_tokens on turn.completed.usage; surface it on
    MetricsEvent and CostEvent so cache-hit ratios work for the Codex backend
    (refs #65, K4 — fix for the historical hardcoded cached_tokens=None)."""
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("turn_completed_cached_tokens.jsonl")

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Cached"):
            events.append(event)

    metrics_events = [e for e in events if isinstance(e, MetricsEvent)]
    cost_events = [e for e in events if isinstance(e, CostEvent)]

    assert len(metrics_events) == 1
    assert metrics_events[0].prompt_tokens == 300
    assert metrics_events[0].completion_tokens == 150
    assert metrics_events[0].cached_tokens == 200
    assert metrics_events[0].cost_usd is None

    assert len(cost_events) == 1
    assert cost_events[0].input_tokens == 300
    assert cost_events[0].output_tokens == 150
    assert cost_events[0].cached_tokens == 200


@pytest.mark.asyncio
async def test_codex_backend_emits_turn_end_after_each_agent_message() -> None:
    """One TurnEndEvent per item.completed of type agent_message."""
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("two_agent_turns.jsonl")

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Two turns"):
            events.append(event)

    texts = [e for e in events if isinstance(e, TextEvent)]
    turn_ends = [e for e in events if isinstance(e, TurnEndEvent)]
    assert len(texts) == 2
    assert len(turn_ends) == 2
    assert all(e.message_id == "" for e in turn_ends)


@pytest.mark.asyncio
async def test_concurrent_execute_calls_do_not_share_stdout_reader() -> None:
    """Overlapping runs on one backend must keep reading their own process."""
    backend = CodexBackend(model="fixture-model")

    class _ImmediateStdout:
        def __init__(self, lines: list[str]) -> None:
            self._lines = iter(lines)

        async def readline(self) -> bytes:
            try:
                return (next(self._lines) + "\n").encode()
            except StopIteration:
                return b""

    class _BlockingStdout:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self._waiting = False

        async def readline(self) -> bytes:
            if self._waiting:
                raise RuntimeError("readuntil() called while another coroutine is already waiting for incoming data")
            self._waiting = True
            self.entered.set()
            try:
                await self.release.wait()
                return b""
            finally:
                self._waiting = False

    def _proc(stdout: object) -> MagicMock:
        process = MagicMock()
        process.stdout = stdout
        process.stdin = MagicMock()
        process.stdin.write = MagicMock()
        process.stdin.close = MagicMock()
        process.wait = AsyncMock(return_value=0)
        process.returncode = 0
        process.terminate = MagicMock()
        process.kill = MagicMock()
        return process

    first_proc = _proc(
        _ImmediateStdout(
            [
                '{"type":"item.completed","item":{"type":"agent_message","text":"first"}}',
                '{"type":"turn.completed","usage":{}}',
            ]
        )
    )
    second_stdout = _BlockingStdout()
    second_proc = _proc(second_stdout)
    procs = iter([first_proc, second_proc])

    async def fake_exec(*args: object, **kwargs: object) -> MagicMock:
        return next(procs)

    async def consume_second() -> list[object]:
        return [event async for event in backend.execute(Path("/tmp"), "second")]

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", fake_exec):
        first_iter = backend.execute(Path("/tmp"), "first")
        first_event = await anext(first_iter)
        assert isinstance(first_event, TextEvent)

        second_task = asyncio.create_task(consume_second())
        await second_stdout.entered.wait()

        try:
            turn_end = await anext(first_iter)
            assert isinstance(turn_end, TurnEndEvent)
            next_first_event = await anext(first_iter)
            assert isinstance(next_first_event, CostEvent)
        finally:
            second_stdout.release.set()
            await second_task


def test_format_skill_invocation():
    backend = CodexBackend(model="fixture-model")
    # Should strip namespace prefix and use $ syntax
    result = backend.format_skill_invocation("beagle-python:review-python")
    assert result == "$review-python"


def test_format_skill_invocation_with_args():
    backend = CodexBackend(model="fixture-model")
    result = backend.format_skill_invocation("beagle-core:fetch-pr-feedback", "--pr 42 --bot mybot")
    assert result == "$fetch-pr-feedback --pr 42 --bot mybot"


def test_format_skill_invocation_no_namespace():
    backend = CodexBackend(model="fixture-model")
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


@pytest.mark.asyncio
async def test_execute_raises_on_agents():
    """CodexBackend refuses agents= with NotImplementedError (Plan 02-04)."""
    backend = CodexBackend(model="fixture-model")
    mock_agent = {"description": "test", "prompt": "test"}

    with pytest.raises(NotImplementedError, match="Codex backend does not support exploration"):
        async for _ in backend.execute(Path("/tmp"), "Test", agents={"explorer": mock_agent}):
            pass
