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
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
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
from daydream.backends._subprocess import StreamStalledError
from daydream.backends.pi import (
    _PI_DEFAULT_RETRY_ATTEMPTS,
    _PI_DEFAULT_RETRY_BASE_DELAY,
    _PI_DEFAULT_RETRY_MAX_DELAY,
    _PI_STDOUT_LIMIT_BYTES,
    PiBackend,
    PiError,
    _is_retryable_error_message,
    _is_retryable_exit_code,
    _is_stream_truncation_message,
    _pi_error_category,
    _pi_retry_attempts,
    _pi_retry_base_delay,
    _pi_retry_max_delay,
    _render_tool_result,
    _schema_instruction,
)
from tests.harness.pi_replay import make_mock_process, make_mock_process_from_fixture
from tests.harness.stub_backend import force_interactive as _force_interactive
from tests.harness.stub_backend import silence as _silence

if TYPE_CHECKING:
    from daydream.runner import RunConfig

MakeConfig = Callable[..., "RunConfig"]
Mute = Callable[..., None]


async def _run_and_capture_args(
    backend: Any,
    prompt: Any="p",
    *,
    fixture: Any="simple_text.jsonl",
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Drive ``execute`` over a canned fixture and return the subprocess argv.

    Consolidates the recurring pattern of patching ``create_subprocess_exec``,
    draining the event stream, and reading ``mock_exec.call_args``. Returns the
    ``(flat_args, mock_exec)`` pair so callers can also assert on kwargs.
    """
    mock_proc = make_mock_process_from_fixture(fixture)
    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), prompt, **kwargs):
            pass
    return list(mock_exec.call_args.args), mock_exec


@pytest.mark.asyncio
async def test_simple_text_events() -> None:
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
async def test_thinking_and_tool_use_events() -> None:
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
async def test_structured_output() -> None:
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
async def test_multi_turn_emits_turn_end_per_turn_and_aggregates_cost() -> None:
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
async def test_error_turn_raises_pi_error() -> None:
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("error_turn.jsonl")

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(PiError, match="Model returned an error"):
            async for _ in backend.execute(Path("/tmp"), "Fail"):
                pass


@pytest.mark.asyncio
async def test_continuation_token_uses_session_id_flag() -> None:
    """A pi continuation token maps to --session-id <id> (not --no-session)."""
    backend = PiBackend(model="glm-5.2")
    token = ContinuationToken(backend="pi", data={"session_id": "pi_resume_me"})

    flat_args, _ = await _run_and_capture_args(backend, "Continue", continuation=token)
    assert "--session-id" in flat_args
    assert flat_args[flat_args.index("--session-id") + 1] == "pi_resume_me"
    assert "--no-session" not in flat_args


@pytest.mark.asyncio
async def test_fresh_run_uses_session_id_not_no_session() -> None:
    """No continuation → --session-id <uuid> (persistent); never --no-session.

    Fresh runs must not use --no-session: that flag is ephemeral (pi docs:
    "Don't save session (ephemeral)"), so the session id harvested from the run
    cannot be resumed later — resuming with --session-id <id> creates an empty
    session. Generating a UUID up front and passing --session-id <uuid> makes
    the returned continuation token genuinely resumable, which matters because
    phase_test_and_heal feeds the token back into its retry loop.
    """
    backend = PiBackend(model="glm-5.2")

    flat_args, _ = await _run_and_capture_args(backend, "Fresh")
    assert "--no-session" not in flat_args
    assert "--session-id" in flat_args
    passed_id = flat_args[flat_args.index("--session-id") + 1]
    # A genuine UUID: the token must name a resumable persistent session.
    uuid.UUID(passed_id)


@pytest.mark.asyncio
async def test_ephemeral_pi_call_uses_no_session() -> None:
    backend = PiBackend(model="glm-5.2")

    flat_args, _ = await _run_and_capture_args(
        backend,
        persist_session=False,
    )
    assert "--no-session" in flat_args
    assert "--session-id" not in flat_args

    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    with patch(
        "daydream.backends.pi.asyncio.create_subprocess_exec",
        return_value=mock_proc,
    ):
        result_events = [
            event
            async for event in backend.execute(
                Path("/tmp"),
                "p",
                persist_session=False,
            )
            if isinstance(event, ResultEvent)
        ]
    assert result_events[0].continuation is None


@pytest.mark.asyncio
async def test_read_only_restricts_tools() -> None:
    """read_only=True adds --tools read,find,ls,grep (excludes mutating tools)."""
    backend = PiBackend(model="glm-5.2")

    flat_args, _ = await _run_and_capture_args(backend, read_only=True)
    assert flat_args[flat_args.index("--tools") + 1] == "read,find,ls,grep"
    # read_only=False by default → no --tools flag.
    flat_args_default, _ = await _run_and_capture_args(backend)
    assert "--tools" not in flat_args_default


@pytest.mark.asyncio
async def test_pi_api_key_never_enters_process_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = "synthetic-pi-api-key-sentinel"
    monkeypatch.setenv("PI_PROVIDER", "zai")
    monkeypatch.setenv("PI_API_KEY", sentinel)
    monkeypatch.setenv("PI_THINKING", "medium")

    backend = PiBackend(model="glm-5.2")

    flat_args, mock_exec = await _run_and_capture_args(backend)
    assert flat_args[flat_args.index("--provider") + 1] == "zai"
    assert sentinel not in flat_args
    assert "--api-key" not in flat_args
    assert mock_exec.call_args.kwargs["env"]["ZAI_API_KEY"] == sentinel
    assert flat_args[flat_args.index("--thinking") + 1] == "medium"


@pytest.mark.asyncio
async def test_pi_api_key_unknown_provider_warns_and_skips(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unmapped provider warns and proceeds; the key never reaches argv or any env var."""
    sentinel = "synthetic-unknown-provider-key"
    monkeypatch.setenv("PI_PROVIDER", "custom-provider")
    monkeypatch.setenv("PI_API_KEY", sentinel)
    backend = PiBackend(model="custom-model")

    with caplog.at_level("WARNING"):
        flat_args, mock_exec = await _run_and_capture_args(backend)

    # No hard failure — the run proceeded to launch the subprocess.
    assert flat_args[flat_args.index("--provider") + 1] == "custom-provider"
    # The key never reaches argv...
    assert sentinel not in flat_args
    assert "--api-key" not in flat_args
    # ...nor any env var handed to the child.
    child_env = mock_exec.call_args.kwargs["env"]
    assert sentinel not in child_env.values()
    assert "PI_API_KEY" not in child_env
    # And the user is warned (without the key value leaking into the log).
    assert any("PI_API_KEY" in r.getMessage() for r in caplog.records)
    assert sentinel not in caplog.text


@pytest.mark.asyncio
async def test_cwd_passed_to_subprocess() -> None:
    """The target dir is passed as the process cwd (Pi reads it natively)."""
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/some/repo"), "p"):
            pass

        assert mock_exec.call_args.kwargs["cwd"] == "/some/repo"
        assert mock_exec.call_args.kwargs["limit"] == _PI_STDOUT_LIMIT_BYTES


@pytest.mark.asyncio
async def test_spawn_uses_start_new_session() -> None:
    """CLI spawns create a new session so the process group is killable."""
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    with patch(
        "daydream.backends.pi.asyncio.create_subprocess_exec",
        return_value=mock_proc,
    ) as mock_exec:
        events = []
        async for event in backend.execute(Path("/tmp"), "hello"):
            events.append(event)
    assert mock_exec.call_args.kwargs["start_new_session"] is True


@pytest.mark.asyncio
async def test_execute_finally_closes_transport_after_process_exit() -> None:
    """Even when the CLI already exited, the finally reaps the process group.

    The helper runs unconditionally when ``proc`` is not ``None``: its first
    group signal fires regardless of ``returncode``, so a grandchild that
    outlived the CLI is still signalled and the pipe fds are still released by
    the transport close.
    """
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    mock_proc.returncode = 0  # process already exited
    mock_proc._transport = MagicMock()
    terminated: list[asyncio.subprocess.Process] = []

    from daydream.backends._subprocess import terminate_process as real_terminate

    async def recording_terminate(proc: asyncio.subprocess.Process, timeout: float | None = None) -> None:
        terminated.append(proc)
        await real_terminate(proc, timeout)

    with (
        patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc),
        patch("daydream.backends.pi.terminate_process", side_effect=recording_terminate),
    ):
        events = []
        async for event in backend.execute(Path("/tmp"), "hello"):
            events.append(event)

    assert terminated == [mock_proc]
    mock_proc._transport.close.assert_called_once()


@pytest.mark.asyncio
async def test_execute_raises_on_agents() -> None:
    """PiBackend refuses agents= with NotImplementedError (plan §5)."""
    backend = PiBackend(model="glm-5.2")
    mock_agent = {"description": "test", "prompt": "test"}

    with pytest.raises(NotImplementedError, match="Pi backend does not support exploration"):
        async for _ in backend.execute(Path("/tmp"), "Test", agents={"explorer": mock_agent}):
            pass


@pytest.mark.asyncio
async def test_agent_end_always_finalizes_when_stream_ends_without_it() -> None:
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
    cont = result_events[0].continuation
    assert cont is not None
    assert cont.data["session_id"] == "pi_ses_truncated"


@pytest.mark.asyncio
async def test_cancel_terminates_then_kills() -> None:
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
async def test_cancel_no_op_when_no_processes() -> None:
    backend = PiBackend(model="glm-5.2")
    backend._processes = []
    await backend.cancel()  # Must not raise.


@pytest.mark.asyncio
async def test_stdout_limit_allows_large_jsonl_events() -> None:
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
async def test_missing_usage_skips_metrics_but_keeps_turn_end() -> None:
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
async def test_nonzero_exit_raises_with_captured_output() -> None:
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
async def test_nonzero_exit_with_no_output_still_informative() -> None:
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
async def test_concurrent_execute_calls_do_not_share_stdout_reader() -> None:
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


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"content": [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]}, "line1line2"),
        ({"content": "raw"}, "raw"),
        ({"details": {"note": "x"}}, "{'note': 'x'}"),
        ("plain", "plain"),
        (None, ""),
    ],
    ids=["text-blocks", "string-content", "details-fallback", "non-dict-str", "non-dict-none"],
)
def test_render_tool_result(result: Any, expected: Any) -> None:
    assert _render_tool_result(result) == expected


def test_schema_instruction_contains_schema_json() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    instruction = _schema_instruction(schema)
    assert "JSON schema" in instruction
    assert json.dumps(schema) in instruction


@pytest.mark.asyncio
async def test_execute_always_passes_no_skills_never_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """M15/M16: Pi disables skills even when ambient skill mirrors exist (#727)."""
    monkeypatch.setattr("daydream.backends.pi.Path.home", lambda: tmp_path)
    monkeypatch.setenv("DAYDREAM_SKILLS_DIR", str(tmp_path / "env-skills"))
    (tmp_path / ".agents" / "skills" / "x" / "SKILL.md").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".agents" / "skills" / "x" / "SKILL.md").write_text("# x\n")
    (tmp_path / ".claude" / "skills" / "y" / "SKILL.md").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".claude" / "skills" / "y" / "SKILL.md").write_text("# y\n")
    (tmp_path / "env-skills" / "z" / "SKILL.md").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "env-skills" / "z" / "SKILL.md").write_text("# z\n")

    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process(['{"id": "s1"}'])
    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(tmp_path, "Review the change."):
            pass

    args = list(mock_exec.call_args.args)
    assert args.count("--no-skills") == 1
    assert "--skill" not in args


def test_create_backend_pi_returns_pi_backend_with_default_model() -> None:
    from daydream.backends import create_backend
    from daydream.config import DEFAULT_PI_MODEL

    backend = create_backend("pi")
    assert isinstance(backend, PiBackend)
    assert backend.model == DEFAULT_PI_MODEL

    custom = create_backend("pi", model="glm-4.5-air")
    assert isinstance(custom, PiBackend)
    assert custom.model == "glm-4.5-air"


def test_create_backend_invalid_includes_pi_in_message() -> None:
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
async def test_live_pi_smoke() -> None:
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
async def test_pi_trajectory_is_valid_atif_v1_7(tmp_path: Path) -> None:
    """A Pi-driven run must produce a trajectory.json that passes the ATIF v1.7
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

    # The trajectory file must be valid ATIF v1.7.
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
# Default --provider is nous; PI_PROVIDER overrides (extension-based provider)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env_provider", "expected"),
    [(None, "nous"), ("my-proxy", "my-proxy")],
    ids=["default-nous", "PI_PROVIDER-override"],
)
@pytest.mark.asyncio
async def test_provider_flag(monkeypatch: pytest.MonkeyPatch, env_provider: Any, expected: Any) -> None:
    """--provider defaults to ``nous`` (matches DEFAULT_PI_MODEL) unless PI_PROVIDER overrides it.

    The nous provider is configured via pi's ``~/.pi/agent/models.json``
    custom-provider registry; daydream must always point pi at it so the
    default DeepSeek model resolves without relying on a user-configured
    models.json entry.
    """
    if env_provider is None:
        monkeypatch.delenv("PI_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("PI_PROVIDER", env_provider)

    backend = PiBackend(model="glm-5.2")
    flat_args, _ = await _run_and_capture_args(backend)
    assert flat_args[flat_args.index("--provider") + 1] == expected


@pytest.mark.asyncio
async def test_default_model_does_not_override_pi_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Pi-configured default wins when daydream did not select a model."""
    settings = tmp_path / ".pi" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(
        '{"defaultProvider": "openai", "defaultModel": "gpt-5.6-luna"}'
    )
    monkeypatch.delenv("PI_PROVIDER", raising=False)

    backend = PiBackend()
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    with patch(
        "daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc
    ) as mock_exec:
        async for _ in backend.execute(tmp_path, "Reply"):
            pass

    flat_args = list(mock_exec.call_args.args)
    assert "--model" not in flat_args
    assert "--provider" not in flat_args


def test_public_model_reflects_pi_settings_before_execute(tmp_path: Path) -> None:
    """The public model is resolved from the target workspace at construction."""
    settings = tmp_path / ".pi" / "settings.json"
    settings.parent.mkdir()
    settings.write_text('{"defaultProvider": "openai", "defaultModel": "gpt-5.6-luna"}')

    backend = PiBackend(cwd=tmp_path)

    assert backend.model == "gpt-5.6-luna"


@pytest.mark.asyncio
async def test_explicit_model_overrides_pi_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit daydream model still wins over Pi's configured default."""
    settings = tmp_path / ".pi" / "settings.json"
    settings.parent.mkdir()
    settings.write_text('{"defaultProvider": "openai", "defaultModel": "gpt-5.6-luna"}')
    monkeypatch.delenv("PI_PROVIDER", raising=False)

    backend = PiBackend(model="custom-model")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    with patch(
        "daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc
    ) as mock_exec:
        async for _ in backend.execute(tmp_path, "Reply"):
            pass

    flat_args = list(mock_exec.call_args.args)
    assert flat_args[flat_args.index("--model") + 1] == "custom-model"


@pytest.mark.asyncio
async def test_nous_deepseek_is_pi_fallback_when_no_model_is_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepSeek on the nous provider remains the fallback when neither daydream nor Pi selects a model."""
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi-agent"))

    backend = PiBackend()
    flat_args, _ = await _run_and_capture_args(backend)

    assert flat_args[flat_args.index("--model") + 1] == "deepseek/deepseek-v4-flash-0731"
    assert flat_args[flat_args.index("--provider") + 1] == "nous"


# ---------------------------------------------------------------------------
# Migration guards: GLM-pin and provider/model mismatch warnings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_glm_pin_warning_fires_with_unset_provider(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An explicit glm-* model with PI_PROVIDER unset warns about the zai->nous default change."""
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.setattr("daydream.backends.pi._warned_migration_mismatches", set())

    backend = PiBackend(model="glm-5.2")
    with caplog.at_level("WARNING"):
        flat_args, _ = await _run_and_capture_args(backend)

    assert any("z.ai-hosted GLM" in r.getMessage() for r in caplog.records)
    # The run still proceeds, pairing the pinned model with the nous default.
    assert flat_args[flat_args.index("--model") + 1] == "glm-5.2"
    assert flat_args[flat_args.index("--provider") + 1] == "nous"


@pytest.mark.asyncio
async def test_glm_pin_warning_silent_when_provider_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """PI_PROVIDER=zai opts back into the zai provider: no warning, provider honored."""
    monkeypatch.setenv("PI_PROVIDER", "zai")
    monkeypatch.setattr("daydream.backends.pi._warned_migration_mismatches", set())

    backend = PiBackend(model="glm-5.2")
    with caplog.at_level("WARNING"):
        flat_args, _ = await _run_and_capture_args(backend)

    assert not any("z.ai-hosted GLM" in r.getMessage() for r in caplog.records)
    assert flat_args[flat_args.index("--provider") + 1] == "zai"


@pytest.mark.asyncio
async def test_glm_pin_warning_fires_once_across_executes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The migration warning is once-guarded, not re-logged per phase or retry."""
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.setattr("daydream.backends.pi._warned_migration_mismatches", set())

    backend = PiBackend(model="glm-5.2")
    with caplog.at_level("WARNING"):
        await _run_and_capture_args(backend)
        await _run_and_capture_args(backend)

    glm_warnings = [r for r in caplog.records if "z.ai-hosted GLM" in r.getMessage()]
    assert len(glm_warnings) == 1


@pytest.mark.asyncio
async def test_zai_provider_with_fallback_model_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """PI_PROVIDER=zai with no configured model pairs the old provider with the new fallback model and warns."""
    monkeypatch.setenv("PI_PROVIDER", "zai")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi-agent"))
    monkeypatch.setattr("daydream.backends.pi._warned_migration_mismatches", set())

    backend = PiBackend()
    with caplog.at_level("WARNING"):
        flat_args, _ = await _run_and_capture_args(backend)

    assert any("no configured model" in r.getMessage() for r in caplog.records)
    # The stale pairing is still passed through (warn-and-continue).
    assert flat_args[flat_args.index("--provider") + 1] == "zai"
    assert flat_args[flat_args.index("--model") + 1] == "deepseek/deepseek-v4-flash-0731"


# ---------------------------------------------------------------------------
# Real-path through runner.run: real PiBackend, only the pi subprocess mocked
# ---------------------------------------------------------------------------


def _capture_pi_subprocess(monkeypatch: pytest.MonkeyPatch, captured: list[list[str]]) -> None:
    """Replace only the pi subprocess spawn; keep the real PiBackend/create_backend.

    Each pi spawn captures its argv and replays a canned no-op session so the
    deep flow can complete without a real pi CLI. The argv is the observable
    contract under test: which --model/--provider daydream hands pi.

    pi.py does a plain ``import asyncio``, so the patch lands on the shared
    asyncio module. Only pi invocations (first argv element ``pi``) are
    intercepted; any other ``create_subprocess_exec`` caller falls through to
    the real executor so the patch never reshapes non-pi spawns.
    """

    # Type the fallthrough as Any: the real create_subprocess_exec signature is
    # keyword-typed, so an opaque *args/**kwargs passthrough would otherwise fail
    # mypy even though it is exactly the forwarding this helper needs.
    real_exec: Any = asyncio.create_subprocess_exec

    async def _fake_exec(*args: object, **kwargs: object) -> MagicMock:
        if args and args[0] == "pi":
            captured.append([str(a) for a in args])
            return make_mock_process_from_fixture("simple_text.jsonl")
        return cast(MagicMock, await real_exec(*args, **kwargs))

    monkeypatch.setattr(
        "daydream.backends.pi.asyncio.create_subprocess_exec", _fake_exec
    )


def _assert_pi_model_and_provider(captured: list[list[str]], *, model: str, provider: str) -> None:
    """Every spawned pi invocation carries the expected --model and --provider."""
    assert captured, "expected at least one pi subprocess spawn"
    for argv in captured:
        assert argv[argv.index("--model") + 1] == model
        assert argv[argv.index("--provider") + 1] == provider


@pytest.mark.parametrize(
    ("env_provider", "expected_provider"),
    [(None, "nous"), ("custom-proxy", "custom-proxy")],
    ids=["default-nous", "PI_PROVIDER-override"],
)
@pytest.mark.asyncio
async def test_runner_real_path_pi_provider_axis(
    tiny_diff_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
    env_provider: str | None,
    expected_provider: str,
) -> None:
    """Real runner path: no model selected → pi gets the fallback --model and the default/overridden --provider.

    Runs ``runner.run`` with the real ``create_backend`` (real PiBackend) on a
    real git worktree, mocking ONLY the pi subprocess spawn. The pi agent dir
    is isolated to an empty temp dir so no settings.json exists and the
    code-level fallback fires.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects()
    if env_provider is None:
        monkeypatch.delenv("PI_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("PI_PROVIDER", env_provider)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tiny_diff_target / "pi-agent"))

    captured: list[list[str]] = []
    _capture_pi_subprocess(monkeypatch, captured)

    rc = await run(
        make_config(
            tiny_diff_target,
            backend="pi",
            assume="yes",
        )
    )
    assert rc == 0
    _assert_pi_model_and_provider(
        captured,
        model="deepseek/deepseek-v4-flash-0731",
        provider=expected_provider,
    )


# ---------------------------------------------------------------------------
# System prompt preamble (--append-system-prompt)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_append_system_prompt_preamble_in_args() -> None:
    """The tool-efficiency preamble is passed via --append-system-prompt.

    Pi's built-in system prompt is minimal compared to Claude Code / Codex; the
    default DeepSeek model needs the budget-awareness guidance appended or it
    exhausts its tool-call budget during exploration. The flag must appear in
    every run, not gated on env vars or read_only.
    """
    from daydream.backends.pi import _PI_SYSTEM_PREAMBLE

    backend = PiBackend(model="glm-5.2")
    flat_args, _ = await _run_and_capture_args(backend)
    assert "--append-system-prompt" in flat_args
    preamble = flat_args[flat_args.index("--append-system-prompt") + 1]
    assert preamble == _PI_SYSTEM_PREAMBLE
    # Preamble must actually carry the budget-awareness guidance, not be
    # an empty stub a future refactor could silently collapse to.
    assert "tool-call budget" in preamble
    assert "grep" in preamble.lower()


# ---------------------------------------------------------------------------
# PiError.retryable attribute
# ---------------------------------------------------------------------------


def test_pierror_retryable_default_and_kwarg_and_message() -> None:
    assert PiError("something went wrong").retryable is False
    assert PiError("429 rate limit", retryable=True).retryable is True
    assert str(PiError("auth failed", retryable=False)) == "auth failed"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("429 Too Many Requests", "RATE_LIMIT"),
        ("502 status code (no body)", "SERVER_ERROR"),
        ("503 status code (no body)", "SERVER_ERROR"),
        ("service unavailable", "SERVER_ERROR"),
        ("request timed out after 30 seconds", "TIMEOUT"),
        ("socket hang up", "STREAM_DROP"),
        ("Pi CLI exited with return code 1", "PROCESS_EXIT"),
        ("authentication required", "AUTH_CONFIG"),
        ("synthetic opaque failure", "UNKNOWN"),
    ],
    ids=[
        "rate-limit",
        "server-error-502",
        "server-error-503",
        "server-error-service-unavailable",
        "timeout",
        "stream-drop",
        "process-exit",
        "auth-config",
        "unknown",
    ],
)
def test_pi_error_categories_are_stable_host_codes(
    message: str,
    expected: str,
) -> None:
    assert _pi_error_category(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "503 status code (no body)",
        "Stream ended without finish_reason",
        "request timeout while waiting for response",
    ],
)
def test_pi_transient_failures_are_retryable(message: str) -> None:
    assert _is_retryable_error_message(message) is True


def test_is_stream_truncation_message() -> None:
    assert _is_stream_truncation_message("stream ended without finish_reason") is True


@pytest.mark.asyncio
async def test_stream_eof_without_finish_reason_is_retryable_pi_error() -> None:
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process(
        [
            '{"type":"session","sessionId":"pi_ses_truncated"}',
            '{"type":"agent_start"}',
            '{"type":"turn_start"}',
        ]
    )
    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(PiError, match="finish_reason") as raised:
            async for _ in backend.execute(Path("/tmp"), "Truncated"):
                pass
    assert raised.value.retryable is True
    assert raised.value.category == "STREAM_TRUNCATION"


@pytest.mark.asyncio
async def test_stream_eof_after_completed_earlier_turn_is_retryable_pi_error() -> None:
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process(
        [
            '{"type":"session","sessionId":"pi_ses_truncated"}',
            '{"type":"agent_start"}',
            '{"type":"turn_start"}',
            '{"type":"turn_end","message":{"role":"assistant","stopReason":"stop"}}',
            '{"type":"turn_start"}',
        ]
    )
    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(PiError, match="finish_reason") as raised:
            async for _ in backend.execute(Path("/tmp"), "Truncated"):
                pass
    assert raised.value.retryable is True
    assert raised.value.category == "STREAM_TRUNCATION"


@pytest.mark.asyncio
async def test_pi_stream_timeout_is_retryable() -> None:
    backend = PiBackend(model="glm-5.2")
    mock_proc = MagicMock()
    mock_proc.stdout = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)
    mock_proc.returncode = 0
    mock_proc.terminate = MagicMock()
    mock_proc.kill = MagicMock()
    with (
        patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc),
        patch(
            "daydream.backends.pi.readline_with_idle_timeout",
            side_effect=StreamStalledError("pi", 1.0),
        ),
    ):
        with pytest.raises(StreamStalledError) as raised:
            async for _ in backend.execute(Path("/tmp"), "Timeout"):
                pass
    assert raised.value.retryable is True


# ---------------------------------------------------------------------------
# _is_retryable_error_message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("429 Too Many Requests", True),
        ("502 status code (no body)", True),
        ("502 Bad Gateway", True),
        ("503 status code (no body)", True),
        ("503 Service Unavailable", True),
        ("Service is currently overloaded", True),
        ("rate limit exceeded", True),
        ("rate_limit hit", True),
        ("Capacity unavailable", True),
        ("too many requests from this IP", True),
        ("Request throttled", True),
        ("throttling in effect", True),
        ("OVERLOAD detected", True),  # case-insensitive
        # Stream-drop signatures (z.ai/GLM connection drops).
        ("terminated", True),
        ("ECONNRESET", True),
        ("connection reset", True),
        ("socket hang up", True),
        ("premature close", True),
        ("EPIPE", True),
        ("auth failed", False),
        ("service is not overloaded", False),
        ("capacity planning required", False),
        ("", False),
        ("Unknown Pi error", False),
    ],
    ids=[
        "429",
        "502-status-code",
        "502-bad-gateway",
        "503-status-code",
        "503-service-unavailable",
        "overload",
        "rate-limit-space",
        "rate_limit-underscore",
        "capacity",
        "too-many-requests",
        "throttle",
        "throttling",
        "case-insensitive",
        "stream-drop-terminated",
        "stream-drop-econnreset",
        "stream-drop-connection-reset",
        "stream-drop-socket-hang-up",
        "stream-drop-premature-close",
        "stream-drop-epipe",
        "auth-failed",
        "not-overloaded",
        "capacity-planning",
        "empty",
        "unknown",
    ],
)
def test_is_retryable_error_message(message: Any, expected: Any) -> None:
    assert _is_retryable_error_message(message) is expected


# ---------------------------------------------------------------------------
# _is_retryable_exit_code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [(-9, True), (137, True), (1, False), (2, False), (0, False)],
    ids=["sigkill", "oom-137", "exit-1", "exit-2", "zero"],
)
def test_is_retryable_exit_code(code: Any, expected: Any) -> None:
    assert _is_retryable_exit_code(code) is expected


# ---------------------------------------------------------------------------
# Error turn uses retryable classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error_message", "expected_retryable"),
    [
        ("429 Too Many Requests - rate limit exceeded", True),
        ("502 status code (no body)", True),
        ("authentication required", False),
    ],
    ids=["429-retryable", "502-retryable", "auth-non-retryable"],
)
@pytest.mark.asyncio
async def test_error_turn_sets_retryable_via_classifier(error_message: Any, expected_retryable: Any) -> None:
    """The turn_end errorMessage is run through the retryable classifier."""
    backend = PiBackend(model="glm-5.2")
    lines = [
        '{"type":"session","sessionId":"pi_ses_err"}',
        '{"type":"agent_start"}',
        '{"type":"turn_start"}',
        '{"type":"turn_end","message":{"role":"assistant","content":[],'
        f'"stopReason":"error","errorMessage":{json.dumps(error_message)}}}}}',
    ]
    mock_proc = make_mock_process(lines)

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(PiError) as exc_info:
            async for _ in backend.execute(Path("/tmp"), "p"):
                pass

    assert exc_info.value.retryable is expected_retryable


@pytest.mark.parametrize(
    ("returncode", "output_lines", "expected_retryable"),
    [
        (-9, [], True),  # SIGKILL/OOM
        (1, ["Error: not authenticated"], False),  # auth/config error
    ],
    ids=["oom-sigkill-retryable", "exit-1-non-retryable"],
)
@pytest.mark.asyncio
async def test_nonzero_exit_sets_retryable_via_exit_code(
    returncode: Any,
    output_lines: Any,
    expected_retryable: Any,
) -> None:
    """The subprocess return code is run through the exit-code retryable classifier."""
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process(output_lines)
    mock_proc.returncode = returncode

    with patch("daydream.backends.pi.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(PiError) as exc_info:
            async for _ in backend.execute(Path("/tmp"), "p"):
                pass

    assert exc_info.value.retryable is expected_retryable


# ---------------------------------------------------------------------------
# Retry env knobs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, _PI_DEFAULT_RETRY_ATTEMPTS), ("5", 5), ("", _PI_DEFAULT_RETRY_ATTEMPTS)],
    ids=["default", "env-override", "empty-warns-and-falls-back"],
)
def test_pi_retry_attempts(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    env_value: str,
    expected: Any,
) -> None:
    if env_value is not None:
        monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", env_value)
    assert _pi_retry_attempts() == expected
    if env_value == "":
        assert (
            f"is not a valid integer; using default {_PI_DEFAULT_RETRY_ATTEMPTS}"
            in caplog.text
        )


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, _PI_DEFAULT_RETRY_BASE_DELAY),
        ("0.5", 0.5),
        ("nan", _PI_DEFAULT_RETRY_BASE_DELAY),
        ("inf", _PI_DEFAULT_RETRY_BASE_DELAY),
        ("", _PI_DEFAULT_RETRY_BASE_DELAY),
    ],
    ids=["default", "env-override", "nan-falls-back", "inf-falls-back", "empty-warns"],
)
def test_pi_retry_base_delay(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    env_value: str,
    expected: Any,
) -> None:
    if env_value is not None:
        monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", env_value)
    assert _pi_retry_base_delay() == pytest.approx(expected)
    if env_value == "":
        assert (
            f"is not a valid float; using default {_PI_DEFAULT_RETRY_BASE_DELAY:g}"
            in caplog.text
        )


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, _PI_DEFAULT_RETRY_MAX_DELAY),
        ("45.5", 45.5),
        ("not-a-float", _PI_DEFAULT_RETRY_MAX_DELAY),
        ("-1", _PI_DEFAULT_RETRY_MAX_DELAY),
        ("", _PI_DEFAULT_RETRY_MAX_DELAY),
    ],
    ids=["default", "env-override", "invalid-warns", "negative-warns", "empty-warns"],
)
def test_pi_retry_max_delay(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    env_value: Any,
    expected: Any,
) -> None:
    if env_value is not None:
        monkeypatch.setenv("DAYDREAM_PI_RETRY_MAX_DELAY_S", env_value)
    assert _pi_retry_max_delay() == pytest.approx(expected)
    if env_value is not None and expected == _PI_DEFAULT_RETRY_MAX_DELAY:
        assert f"using default {_PI_DEFAULT_RETRY_MAX_DELAY:g}" in caplog.text


# ---------------------------------------------------------------------------
# fanout_concurrency
# ---------------------------------------------------------------------------


def test_pi_fanout_concurrency_defaults_to_ten(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAYDREAM_PI_FANOUT_CONCURRENCY", raising=False)
    backend = PiBackend(model="glm-5.2")
    assert backend.fanout_concurrency == 10


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("6", 6),
        ("0", 10),
        ("-1", 10),
        ("invalid", 10),
        ("", 10),
    ],
)
def test_pi_fanout_concurrency_env_validation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    env_value: Any,
    expected: int,
) -> None:
    monkeypatch.setenv("DAYDREAM_PI_FANOUT_CONCURRENCY", env_value)
    assert PiBackend(model="glm-5.2").fanout_concurrency == expected
    if expected == 10:
        assert "using default 10" in caplog.text


@pytest.mark.asyncio
async def test_pi_reasoning_effort_forwards_as_thinking_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The resolved per-phase effort arrives as ``--thinking <level>``."""
    monkeypatch.delenv("PI_THINKING", raising=False)
    backend = PiBackend(model="glm-5.2", reasoning_effort="max")

    flat_args, _ = await _run_and_capture_args(backend)
    assert flat_args[flat_args.index("--thinking") + 1] == "max"


@pytest.mark.asyncio
async def test_pi_reasoning_effort_outranks_pi_thinking_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit per-phase level beats Pi's ambient PI_THINKING default."""
    monkeypatch.setenv("PI_THINKING", "low")
    backend = PiBackend(model="glm-5.2", reasoning_effort="xhigh")

    flat_args, _ = await _run_and_capture_args(backend)
    assert flat_args[flat_args.index("--thinking") + 1] == "xhigh"
    assert "low" not in flat_args


@pytest.mark.asyncio
async def test_pi_falls_back_to_pi_thinking_when_no_effort_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_THINKING", "high")
    backend = PiBackend(model="glm-5.2")

    flat_args, _ = await _run_and_capture_args(backend)
    assert flat_args[flat_args.index("--thinking") + 1] == "high"
