# tests/test_backend_pi.py
"""Tests for PiBackend with canned JSONL fixtures.

Mirrors ``tests/test_backend_codex.py``: the subprocess is mocked via
``tests.harness.pi_replay`` and each test drives ``PiBackend.execute`` against
a scripted JSONL stream, asserting the exact ``AgentEvent`` sequence and
payloads.
"""

import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daydream.backends import (
    ContinuationToken,
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.backends.pi import (
    _PI_STDOUT_LIMIT_BYTES,
    _SKILL_TOKEN_RE,
    PiBackend,
    PiError,
    _render_tool_result,
    _resolve_skill_dir,
    _schema_instruction,
)
from tests.harness.pi_replay import make_mock_process, make_mock_process_from_fixture


@pytest.mark.asyncio
async def test_simple_text_events():
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Say hello"):
            events.append(event)

    text_events = [e for e in events if isinstance(e, TextEvent)]
    metrics_events = [e for e in events if isinstance(e, MetricsEvent)]
    cost_events = [e for e in events if isinstance(e, CostEvent)]
    result_events = [e for e in events if isinstance(e, ResultEvent)]

    assert len(text_events) == 1
    assert text_events[0].text == "Hello from Pi"

    assert len(metrics_events) == 1
    assert metrics_events[0].prompt_tokens == 100
    assert metrics_events[0].completion_tokens == 50
    assert metrics_events[0].cached_tokens == 10
    assert metrics_events[0].cost_usd == 0.0003
    assert metrics_events[0].message_id == ""

    assert len(cost_events) == 1
    assert cost_events[0].cost_usd == 0.0003
    assert cost_events[0].input_tokens == 100
    assert cost_events[0].output_tokens == 50
    assert cost_events[0].cached_tokens == 10
    assert cost_events[0].model_name == "glm-5.2"

    assert len(result_events) == 1
    assert result_events[0].continuation is not None
    assert result_events[0].continuation.backend == "pi"
    assert result_events[0].continuation.data["session_id"] == "pi_ses_simple"


@pytest.mark.asyncio
async def test_thinking_and_tool_use_events():
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("tool_use.jsonl")

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Read the file"):
            events.append(event)

    thinking = [e for e in events if isinstance(e, ThinkingEvent)]
    tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    texts = [e for e in events if isinstance(e, TextEvent)]

    assert len(thinking) == 1
    assert thinking[0].text == "Let me read the file"

    assert len(tool_starts) == 1
    assert tool_starts[0].id == "t1"
    assert tool_starts[0].name == "read"
    assert tool_starts[0].input == {"path": "/x"}

    assert len(tool_results) == 1
    assert tool_results[0].id == "t1"
    assert tool_results[0].output == "file.py\ntest.py"
    assert tool_results[0].is_error is False

    # Text emitted from message_end before the tool-execution events.
    assert any(t.text == "Looking now" for t in texts)


@pytest.mark.asyncio
async def test_structured_output():
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("structured_output.jsonl")
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        events = []
        async for event in backend.execute(Path("/tmp"), "Parse", output_schema=schema):
            events.append(event)

    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output == {
        "issues": [{"id": 1, "description": "Fix type hints", "file": "app.py", "line": 5}]
    }

    # Schema is emulated via prompt appendix (not a CLI flag) — verify the
    # positional prompt argument carries the schema instruction.
    flat_args = list(mock_exec.call_args.args)
    positional = flat_args[-1]
    assert "JSON schema" in positional
    assert json.dumps(schema) in positional


@pytest.mark.asyncio
async def test_multi_turn_emits_turn_end_per_turn_and_aggregates_cost():
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("multi_turn.jsonl")

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Two turns"):
            events.append(event)

    texts = [e for e in events if isinstance(e, TextEvent)]
    turn_ends = [e for e in events if isinstance(e, TurnEndEvent)]
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    cost_events = [e for e in events if isinstance(e, CostEvent)]

    assert [t.text for t in texts] == ["First turn body", "Second turn body"]
    assert len(turn_ends) == 2
    assert all(e.message_id == "" for e in turn_ends)

    # One MetricsEvent per turn_end (both carry usage).
    assert len(metrics) == 2

    # CostEvent fires once at agent_end, aggregating both turns.
    assert len(cost_events) == 1
    assert cost_events[0].input_tokens == 200  # 150 + 50
    assert cost_events[0].output_tokens == 100  # 75 + 25
    assert cost_events[0].cost_usd == pytest.approx(0.00015)  # 0.0001 + 0.00005


@pytest.mark.asyncio
async def test_error_turn_raises_pi_error():
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("error_turn.jsonl")

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(PiError, match="Model returned an error"):
            async for _ in backend.execute(Path("/tmp"), "Fail"):
                pass


@pytest.mark.asyncio
async def test_continuation_token_uses_session_id_flag():
    """A pi continuation token maps to --session-id <id> (not --no-session)."""
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    token = ContinuationToken(backend="pi", data={"session_id": "pi_resume_me"})

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "Continue", continuation=token):
            pass

        flat_args = list(mock_exec.call_args.args)
        assert "--session-id" in flat_args
        assert flat_args[flat_args.index("--session-id") + 1] == "pi_resume_me"
        assert "--no-session" not in flat_args


@pytest.mark.asyncio
async def test_fresh_run_uses_session_id_not_no_session():
    """No continuation → --session-id <uuid> (persistent); never --no-session.

    Fresh runs must not use --no-session: that flag is ephemeral (pi docs:
    "Don't save session (ephemeral)"), so the session id harvested from the run
    cannot be resumed later — resuming with --session-id <id> creates an empty
    session. Generating a UUID up front and passing --session-id <uuid> makes
    the returned continuation token genuinely resumable, which matters because
    phase_test_and_heal feeds the token back into its retry loop.
    """
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "Fresh"):
            pass

        flat_args = list(mock_exec.call_args.args)
        assert "--no-session" not in flat_args
        assert "--session-id" in flat_args
        passed_id = flat_args[flat_args.index("--session-id") + 1]
        # A genuine UUID: the token must name a resumable persistent session.
        uuid.UUID(passed_id)


@pytest.mark.asyncio
async def test_read_only_restricts_tools():
    """read_only=True adds --tools read,find,ls,grep (excludes mutating tools)."""
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "p", read_only=True):
            pass

        flat_args = list(mock_exec.call_args.args)
        assert flat_args[flat_args.index("--tools") + 1] == "read,find,ls,grep"
    # read_only=False by default → no --tools flag.
    mock_proc2 = make_mock_process_from_fixture("simple_text.jsonl")
    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc2) as mock_exec2:
        async for _ in backend.execute(Path("/tmp"), "p"):
            pass
        assert "--tools" not in list(mock_exec2.call_args.args)


@pytest.mark.asyncio
async def test_env_overrides_forwarded_as_flags(monkeypatch):
    """PI_PROVIDER / PI_API_KEY / PI_THINKING env vars become CLI flags."""
    monkeypatch.setenv("PI_PROVIDER", "zai")
    monkeypatch.setenv("PI_API_KEY", "secret-key")
    monkeypatch.setenv("PI_THINKING", "medium")

    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "p"):
            pass

        flat_args = list(mock_exec.call_args.args)
        assert flat_args[flat_args.index("--provider") + 1] == "zai"
        assert flat_args[flat_args.index("--api-key") + 1] == "secret-key"
        assert flat_args[flat_args.index("--thinking") + 1] == "medium"


@pytest.mark.asyncio
async def test_cwd_passed_to_subprocess():
    """The target dir is passed as the process cwd (Pi reads it natively)."""
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/some/repo"), "p"):
            pass

        assert mock_exec.call_args.kwargs["cwd"] == "/some/repo"
        assert mock_exec.call_args.kwargs["limit"] == _PI_STDOUT_LIMIT_BYTES


@pytest.mark.asyncio
async def test_execute_raises_on_agents():
    """PiBackend refuses agents= with NotImplementedError (plan §5)."""
    backend = PiBackend(model="glm-5.2")
    mock_agent = {"description": "test", "prompt": "test"}

    with pytest.raises(NotImplementedError, match="Pi backend does not support exploration"):
        async for _ in backend.execute(Path("/tmp"), "Test", agents={"explorer": mock_agent}):
            pass


@pytest.mark.asyncio
async def test_agent_end_always_finalizes_when_stream_ends_without_it():
    """Guard (plan §10): stream ending mid-turn still emits Cost + Result."""
    backend = PiBackend(model="glm-5.2")
    # Stream ends after a turn_end but with NO agent_end line.
    lines = [
        '{"type":"session","sessionId":"pi_ses_truncated"}',
        '{"type":"agent_start"}',
        '{"type":"turn_start"}',
        '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]}}',
        '{"type":"turn_end","message":{"role":"assistant","content":[{"type":"text","text":"hi"}],'
        '"usage":{"input":5,"output":3,"cost":{"total":0.0001}},"stopReason":"stop"}}',
        # EOF — no agent_end.
    ]
    mock_proc = make_mock_process(lines)

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Truncated"):
            events.append(event)

    cost_events = [e for e in events if isinstance(e, CostEvent)]
    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(cost_events) == 1
    assert len(result_events) == 1
    assert result_events[0].continuation.data["session_id"] == "pi_ses_truncated"


@pytest.mark.asyncio
async def test_cancel_terminates_then_kills():
    """cancel() sends SIGTERM to all tracked processes, SIGKILL on timeout."""
    backend = PiBackend(model="glm-5.2")

    proc = MagicMock()
    proc.returncode = None
    proc.wait = AsyncMock(side_effect=[asyncio.TimeoutError(), 0])
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    backend._processes = [proc]

    await backend.cancel()

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_no_op_when_no_processes():
    backend = PiBackend(model="glm-5.2")
    backend._processes = []
    await backend.cancel()  # Must not raise.


@pytest.mark.asyncio
async def test_stdout_limit_allows_large_jsonl_events():
    """Large message_end lines must not trip asyncio's chunk-length guard."""
    backend = PiBackend(model="glm-5.2")
    large_text = "x" * (70 * 1024)
    large_line = (
        json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": large_text}],
                },
            }
        )
        + "\n"
    ).encode()
    lines = [
        b'{"type":"session","sessionId":"pi_ses_big"}\n',
        b'{"type":"agent_start"}\n',
        b'{"type":"turn_start"}\n',
        large_line,
        b'{"type":"turn_end","message":{"role":"assistant","content":[{"type":"text","text":"x"}],'
        b'"usage":{"input":1,"output":1},"stopReason":"stop"}}\n',
        b'{"type":"agent_end","messages":[]}\n',
    ]
    captured: dict[str, object] = {}

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
        captured.update(kwargs)
        raw_limit = kwargs.get("limit", 64 * 1024)
        limit = raw_limit if isinstance(raw_limit, int) else 64 * 1024
        process = MagicMock()
        process.stdout = _LimitAwareStdout(limit)
        process.wait = AsyncMock(return_value=0)
        process.returncode = 0
        process.terminate = MagicMock()
        process.kill = MagicMock()
        return process

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", fake_exec):
        events = [event async for event in backend.execute(Path("/tmp"), "large")]

    text_events = [e for e in events if isinstance(e, TextEvent)]
    assert text_events[0].text == large_text
    assert captured["limit"] == _PI_STDOUT_LIMIT_BYTES
    assert _PI_STDOUT_LIMIT_BYTES > len(large_line)


@pytest.mark.asyncio
async def test_missing_usage_skips_metrics_but_keeps_turn_end():
    """A turn_end without usage emits no MetricsEvent but still closes the step."""
    backend = PiBackend(model="glm-5.2")
    lines = [
        '{"type":"session","sessionId":"pi_ses_nousage"}',
        '{"type":"agent_start"}',
        '{"type":"turn_start"}',
        '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]}}',
        '{"type":"turn_end","message":{"role":"assistant","content":[{"type":"text","text":"hi"}],"stopReason":"stop"}}',
        '{"type":"agent_end","messages":[]}',
    ]
    mock_proc = make_mock_process(lines)

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "p"):
            events.append(event)

    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    turn_ends = [e for e in events if isinstance(e, TurnEndEvent)]
    cost_events = [e for e in events if isinstance(e, CostEvent)]
    assert metrics == []
    assert len(turn_ends) == 1
    assert len(cost_events) == 1
    assert cost_events[0].cost_usd is None
    assert cost_events[0].input_tokens == 0


@pytest.mark.asyncio
async def test_nonzero_exit_raises_with_captured_output():
    """Non-zero exit surfaces pi's diagnostic output in the PiError message."""
    backend = PiBackend(model="glm-5.2")
    lines = [
        "Error: authentication required. Run `pi login` to authenticate.",
        "fatal: could not connect to API endpoint",
    ]
    mock_proc = make_mock_process(lines)
    mock_proc.returncode = 1

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(PiError, match="return code 1") as exc_info:
            async for _ in backend.execute(Path("/tmp"), "p"):
                pass

    # The error must include the captured diagnostic lines, not just the
    # return code — otherwise debugging a crashed pi is impossible.
    msg = str(exc_info.value)
    assert "authentication required" in msg
    assert "could not connect" in msg


@pytest.mark.asyncio
async def test_nonzero_exit_with_no_output_still_informative():
    """If pi crashes with zero output, the error says so explicitly."""
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process([])
    mock_proc.returncode = 1

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(PiError, match="return code 1") as exc_info:
            async for _ in backend.execute(Path("/tmp"), "p"):
                pass

    assert "no non-JSON output captured" in str(exc_info.value)


@pytest.mark.asyncio
async def test_concurrent_execute_calls_do_not_share_stdout_reader():
    """Overlapping runs on one backend keep reading their own process."""
    backend = PiBackend(model="glm-5.2")

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
                raise RuntimeError("readuntil() called while another coroutine is already waiting")
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
        process.wait = AsyncMock(return_value=0)
        process.returncode = 0
        process.terminate = MagicMock()
        process.kill = MagicMock()
        return process

    first_proc = _proc(
        _ImmediateStdout(
            [
                '{"type":"session","sessionId":"s1"}',
                '{"type":"agent_start"}',
                '{"type":"turn_start"}',
                '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"first"}]}}',
                '{"type":"turn_end","message":{"role":"assistant","content":[],"stopReason":"stop"}}',
                '{"type":"agent_end","messages":[]}',
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

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", fake_exec):
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


def test_render_tool_result_text_blocks():
    result = {"content": [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]}
    assert _render_tool_result(result) == "line1line2"


def test_render_tool_result_string_content():
    assert _render_tool_result({"content": "raw"}) == "raw"


def test_render_tool_result_falls_back_to_details():
    assert _render_tool_result({"details": {"note": "x"}}) == "{'note': 'x'}"


def test_render_tool_result_non_dict():
    assert _render_tool_result("plain") == "plain"
    assert _render_tool_result(None) == ""


def test_schema_instruction_contains_schema_json():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    instruction = _schema_instruction(schema)
    assert "JSON schema" in instruction
    assert json.dumps(schema) in instruction


def test_resolve_skill_dir_returns_none_when_absent(tmp_path, monkeypatch):
    # Isolate from the real ~/.claude and ~/.agents by repointing home and cwd
    # into an empty tmp_path.
    monkeypatch.setattr("daydream.backends.pi.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DAYDREAM_SKILLS_DIR", str(tmp_path / "nope"))
    assert _resolve_skill_dir("beagle-python:review-python") is None


def test_resolve_skill_dir_finds_slug(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "review-python"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# review-python\n")
    monkeypatch.setattr("daydream.backends.pi.Path.home", lambda: tmp_path)
    monkeypatch.setenv("DAYDREAM_SKILLS_DIR", str(skills_root))
    monkeypatch.chdir(tmp_path)
    resolved = _resolve_skill_dir("beagle-python:review-python")
    assert resolved == skill_dir


def test_format_skill_invocation_emits_native_command():
    backend = PiBackend(model="glm-5.2")
    result = backend.format_skill_invocation("beagle-python:review-python")
    assert result == "/skill:review-python"
    assert "beagle-python" not in result


def test_format_skill_invocation_preserves_args():
    backend = PiBackend(model="glm-5.2")
    result = backend.format_skill_invocation("beagle-core:review-structure", "--pr 7")
    assert result == "/skill:review-structure --pr 7"


def test_format_skill_invocation_bare_slug_unchanged():
    backend = PiBackend(model="glm-5.2")
    assert backend.format_skill_invocation("review-go") == "/skill:review-go"


def test_skill_token_re_grammar_matches_pi_name_rules():
    # Pi skill names are lowercase letters, digits, and hyphens.
    admitted = ["review-python", "review-frontend", "review-go", "2-thing", "a"]
    rejected = ["Review_Python", "PDF-Processing", "foo_bar", "CamelCase"]
    for slug in admitted:
        assert _SKILL_TOKEN_RE.findall(f"/skill:{slug}") == [slug], slug
    for slug in rejected:
        # No token (or a truncated one ending before the disallowed char) is
        # captured as a *complete* match of the invalid slug.
        assert slug not in _SKILL_TOKEN_RE.findall(f"/skill:{slug}"), slug


@pytest.mark.asyncio
async def test_format_skill_invocation_token_is_consumed_by_execute(tmp_path, monkeypatch):
    # The formatted token must be recognized by execute's scanner and registered.
    skills_root = tmp_path / "skills"
    py_dir = skills_root / "review-python"
    py_dir.mkdir(parents=True)
    (py_dir / "SKILL.md").write_text("# review-python\n")
    monkeypatch.setattr("daydream.backends.pi.Path.home", lambda: tmp_path)
    monkeypatch.setenv("DAYDREAM_SKILLS_DIR", str(skills_root))
    monkeypatch.chdir(tmp_path)

    backend = PiBackend(model="glm-5.2")
    prompt = f"Review this.\n\n{backend.format_skill_invocation('beagle-python:review-python')}"

    mock_proc = make_mock_process(['{"id": "s1"}'])
    with patch(
        "daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc
    ) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), prompt):
            pass

    flat_args = list(mock_exec.call_args.args)
    skill_indices = [i for i, a in enumerate(flat_args) if a == "--skill"]
    assert len(skill_indices) == 1, (
        f"producer/consumer pairing broken: token from format_skill_invocation "
        f"was not scanned by execute; args: {flat_args}"
    )
    assert flat_args[skill_indices[0] + 1] == str(py_dir)


@pytest.mark.asyncio
async def test_execute_registers_resolved_skills_with_skill_flag(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    py_dir = skills_root / "review-python"
    py_dir.mkdir(parents=True)
    (py_dir / "SKILL.md").write_text("# review-python\n")
    monkeypatch.setattr("daydream.backends.pi.Path.home", lambda: tmp_path)
    monkeypatch.setenv("DAYDREAM_SKILLS_DIR", str(skills_root))
    monkeypatch.chdir(tmp_path)

    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process(['{"id": "s1"}'])
    # Duplicate references are de-duplicated; unresolved skills are skipped.
    prompt = "Review this.\n\n/skill:review-python\n/skill:review-python\n/skill:review-go"

    with patch(
        "daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc
    ) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), prompt):
            pass

    flat_args = list(mock_exec.call_args.args)
    skill_indices = [i for i, a in enumerate(flat_args) if a == "--skill"]
    assert len(skill_indices) == 1, f"expected one --skill flag, got args: {flat_args}"
    assert flat_args[skill_indices[0] + 1] == str(py_dir)
    # The --skill flag precedes the positional prompt (last arg).
    assert skill_indices[0] < len(flat_args) - 1


@pytest.mark.asyncio
async def test_execute_no_skill_flag_when_unresolvable(tmp_path, monkeypatch):
    # Unresolvable skills do not add flags or fail the run.
    monkeypatch.setattr("daydream.backends.pi.Path.home", lambda: tmp_path)
    monkeypatch.setenv("DAYDREAM_SKILLS_DIR", str(tmp_path / "nope"))
    monkeypatch.chdir(tmp_path)

    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process(['{"id": "s1"}'])

    with patch(
        "daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc
    ) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "Review.\n\n/skill:review-rust"):
            pass

    assert "--skill" not in list(mock_exec.call_args.args)


def test_create_backend_pi_returns_pi_backend_with_default_model():
    from daydream.backends import create_backend
    from daydream.config import DEFAULT_PI_MODEL

    backend = create_backend("pi")
    assert isinstance(backend, PiBackend)
    assert backend.model == DEFAULT_PI_MODEL


def test_create_backend_pi_custom_model():
    from daydream.backends import create_backend

    backend = create_backend("pi", model="glm-4.5-air")
    assert isinstance(backend, PiBackend)
    assert backend.model == "glm-4.5-air"


def test_create_backend_invalid_includes_pi_in_message():
    from daydream.backends import create_backend

    with pytest.raises(ValueError, match="pi"):
        create_backend("invalid")


# Truly opt-in live smoke test (plan §8). Gating on `shutil.which("pi")`
# alone is NOT enough: with `pi` installed but z.ai unconfigured, `pi --mode
# json` blocks waiting for /login, which would hang `make test` and the
# pre-push hook for the 60s timeout below and then fail. So the test also
# requires DAYDREAM_PI_LIVE=1 — it is skipped by default and only runs when a
# human explicitly opts in (mirroring the benchmark e2e "spends money" gate).
_PI_AVAILABLE = shutil.which("pi") is not None
_PI_LIVE_OPT_IN = os.environ.get("DAYDREAM_PI_LIVE") == "1"


@pytest.mark.skipif(
    not (_PI_AVAILABLE and _PI_LIVE_OPT_IN),
    reason="live pi smoke test; set DAYDREAM_PI_LIVE=1 (and ensure `pi` is on $PATH and logged in) to run",
)
@pytest.mark.asyncio
async def test_live_pi_smoke():
    """Smoke test against a real `pi` binary (opt-in via DAYDREAM_PI_LIVE=1).

    Asserts an observable success signal (actual assistant text), not mere
    event arrival: the backend's finalization path always emits CostEvent +
    ResultEvent on EOF, so on an auth/model failure (empty stdout, error on
    stderr) those two events still arrive — the text assertion is what
    distinguishes a real reply from a bare EOF. The wait_for timeout converts
    a hang (e.g. pi blocking on /login when z.ai creds are absent) into a
    failure rather than an infinite stall.
    """
    backend = PiBackend(model="glm-5.2")
    events = []

    async def _collect() -> None:
        async for event in backend.execute(Path("/tmp"), "Reply with exactly: pong"):
            events.append(event)

    await asyncio.wait_for(_collect(), timeout=60.0)

    # Observable success: the agent must actually have replied. Asserting only
    # ResultEvent/CostEvent is a false green — they are unconditionally emitted
    # at EOF by the finalization path, so they survive an auth/model failure.
    text = "".join(e.text for e in events if isinstance(e, TextEvent))
    assert "pong" in text.lower(), f"no assistant text emitted (auth/model failure?): events={events!r}"
    # A real run must finalize with CostEvent + ResultEvent.
    assert any(isinstance(e, ResultEvent) for e in events)
    assert any(isinstance(e, CostEvent) for e in events)


@pytest.mark.asyncio
async def test_pi_trajectory_is_valid_atif_v1_6(tmp_path: Path):
    """A Pi-driven run must produce a trajectory.json that passes the ATIF v1.6
    validator (plan §8.3) — the replay/trajectory proof."""
    from daydream.atif import validate
    from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder

    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("tool_use.jsonl")
    traj_path = tmp_path / "trajectory.json"

    recorder = TrajectoryRecorder(
        path=traj_path,
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="glm-5.2",
        session_id="00000000-0000-0000-0000-0000000000aa",
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc):
                async for event in backend.execute(tmp_path, "Review"):
                    inv.observe(event)

    # The trajectory file must be valid ATIF v1.6.
    assert traj_path.is_file()
    assert validate(traj_path, validate_images=False)

    # And must contain the expected agent step content. The CostEvent at
    # agent_end opens a trailing empty step (matches the Claude/Codex recorder
    # behavior) — assert on the first content-bearing agent step.
    agent_steps = [s for s in recorder.steps if s.source == "agent"]
    assert len(agent_steps) >= 1
    step = agent_steps[0]
    assert step.message == "Looking now"
    assert step.reasoning_content == "Let me read the file"
    assert [tc.tool_call_id for tc in (step.tool_calls or [])] == ["t1"]
    obs = {r.source_call_id: r.content for r in (step.observation.results if step.observation else [])}
    assert obs == {"t1": "file.py\ntest.py"}
    # Pi reports real cost (unlike Codex) — metrics must be populated.
    assert step.metrics is not None
    assert step.metrics.prompt_tokens == 200
    assert step.metrics.completion_tokens == 100
    assert step.metrics.cost_usd == 0.0005


# ---------------------------------------------------------------------------
# Default --provider is zai; PI_PROVIDER overrides (extension-based provider)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_provider_is_zai(monkeypatch):
    """Without PI_PROVIDER, --provider defaults to ``zai`` (matches DEFAULT_PI_MODEL).

    The z.ai provider is registered by the ``~/.pi/extensions/zai-provider``
    pi extension; daydream must always point pi at it so the default GLM model
    resolves without relying on a user-configured models.json entry.
    """
    monkeypatch.delenv("PI_PROVIDER", raising=False)

    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "p"):
            pass

        flat_args = list(mock_exec.call_args.args)
        assert flat_args[flat_args.index("--provider") + 1] == "zai"


@pytest.mark.asyncio
async def test_pi_provider_override(monkeypatch):
    """PI_PROVIDER overrides the z.ai default."""
    monkeypatch.setenv("PI_PROVIDER", "my-proxy")

    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "p"):
            pass

        flat_args = list(mock_exec.call_args.args)
        assert flat_args[flat_args.index("--provider") + 1] == "my-proxy"


# ---------------------------------------------------------------------------
# System prompt preamble (--append-system-prompt)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_append_system_prompt_preamble_in_args():
    """The tool-efficiency preamble is passed via --append-system-prompt.

    Pi's built-in system prompt is minimal compared to Claude Code / Codex; the
    GLM model needs the budget-awareness guidance appended or it exhausts its
    tool-call budget during exploration. The flag must appear in every run,
    not gated on env vars or read_only.
    """
    from daydream.backends.pi import _PI_SYSTEM_PREAMBLE

    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "p"):
            pass

        flat_args = list(mock_exec.call_args.args)
        assert "--append-system-prompt" in flat_args
        preamble = flat_args[flat_args.index("--append-system-prompt") + 1]
        assert preamble == _PI_SYSTEM_PREAMBLE
        # Preamble must actually carry the budget-awareness guidance, not be
        # an empty stub a future refactor could silently collapse to.
        assert "tool-call budget" in preamble
        assert "grep" in preamble.lower()

