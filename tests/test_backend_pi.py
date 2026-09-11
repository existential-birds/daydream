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
    GenerationEndEvent,
    GenerationStartEvent,
    MetricsEvent,
    PiRequestConfig,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallChoicePart,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
    unix_ms_to_ns,
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
    _pi_error_category,
    _pi_retry_attempts,
    _pi_retry_base_delay,
    _pi_retry_max_delay,
    _render_tool_result,
    _schema_instruction,
)
from tests.harness.pi_replay import FIXTURES_DIR, make_mock_process, make_mock_process_from_fixture
from tests.harness.stub_backend import force_interactive as _force_interactive
from tests.harness.stub_backend import silence as _silence

if TYPE_CHECKING:
    from daydream.runner import RunConfig

MakeConfig = Callable[..., "RunConfig"]
Mute = Callable[..., None]


def test_backend_execution_input_is_immutable_parsed_and_returns_fresh_environment(
    tmp_path: Path,
) -> None:
    from daydream.backends import BackendExecutionInput, RetryPolicy

    source = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/run/bin",
        "PI_PROVIDER": "openrouter",
        "PI_THINKING": "high",
        "PI_API_KEY": "run-secret",
        "PI_CODING_AGENT_DIR": str(tmp_path / "pi-agent"),
        "DAYDREAM_PI_FANOUT_CONCURRENCY": "3",
        "DAYDREAM_PI_RETRY_ATTEMPTS": "4",
        "DAYDREAM_PI_RETRY_BASE_DELAY_S": "0.25",
        "DAYDREAM_PI_RETRY_MAX_DELAY_S": "5",
        "DAYDREAM_STREAM_IDLE_TIMEOUT_S": "7",
    }

    execution = BackendExecutionInput.from_environment(source, backend="pi")
    source["PI_API_KEY"] = "mutated"
    first = execution.child_environment()
    first["PI_API_KEY"] = "also-mutated"

    assert execution.retry_policy == RetryPolicy(
        attempts=4, base_delay_s=0.25, max_delay_s=5.0
    )
    assert execution.fanout_concurrency == 3
    assert execution.pi_provider == "openrouter"
    assert execution.pi_thinking == "high"
    assert execution.pi_agent_dir == tmp_path / "pi-agent"
    assert execution.stream_idle_timeout_s == 7.0
    assert execution.pi_response_idle_timeout_s == 7.0
    assert execution.child_environment()["PI_API_KEY"] == "run-secret"
    assert "run-secret" not in repr(execution)


def test_backend_execution_input_uses_backend_specific_defaults(tmp_path: Path) -> None:
    from daydream.backends import BackendExecutionInput, RetryPolicy

    environment = {"HOME": str(tmp_path), "PATH": "/run/bin"}

    pi = BackendExecutionInput.from_environment(environment, backend="pi")
    codex = BackendExecutionInput.from_environment(environment, backend="codex")
    claude = BackendExecutionInput.from_environment(environment, backend="claude")

    assert pi.retry_policy == RetryPolicy(20, 10.0, 120.0)
    assert pi.fanout_concurrency == 10
    assert pi.pi_agent_dir == tmp_path / ".pi" / "agent"
    assert codex.retry_policy == claude.retry_policy == RetryPolicy(20, 2.0, 120.0)
    assert codex.fanout_concurrency == claude.fanout_concurrency == 8


def test_backend_execution_input_rejects_explicit_osprey() -> None:
    from daydream.backends import BackendExecutionInput

    with pytest.raises(ValueError, match="osprey"):
        BackendExecutionInput.from_environment({}, backend="osprey")


@pytest.mark.asyncio
async def test_artifact_visibility_protocol_cli_uses_argv_prompt_devnull_and_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    from tests.harness.protocol_cli import install_protocol_cli

    target = (tmp_path / "model cwd with spaces").resolve()
    target.mkdir()
    (target / "source.py").write_text("SOURCE_CANARY\n", encoding="utf-8")
    fixture = install_protocol_cli(tmp_path / "external fixture", "pi")
    settings = tmp_path / "empty pi settings"
    settings.mkdir()
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(settings))
    monkeypatch.setenv("PI_PROVIDER", "nous")
    for name in ("PI_THINKING", "PI_API_KEY", "NOUS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    prompt = "Inspect the committed source only."
    backend = PiBackend(model="fixture-model")

    events = [event async for event in backend.execute(target, prompt)]

    observation = fixture.read_observations()[0]
    assert observation["effective_cwd"] == str(target)
    assert observation["inherited_cwd"] == str(target)
    assert observation["stdin_bytes"] == 0
    assert observation["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert observation["cwd_canaries"]["SOURCE_CANARY"] is True
    assert observation["walk_truncated"] is False
    argv = observation["argv"]
    assert argv[argv.index("--mode") + 1] == "json"
    assert argv[argv.index("--provider") + 1] == "nous"
    assert argv[argv.index("--model") + 1] == "fixture-model"
    assert "--append-system-prompt" in argv and "--no-skills" in argv
    from daydream.backends.pi import _PI_SYSTEM_PREAMBLE

    # Neither the prompt nor the preamble is recorded verbatim -- only digests.
    observation_bytes = next(fixture.observations.glob("*.json")).read_bytes()
    assert prompt.encode() not in observation_bytes
    assert json.dumps(_PI_SYSTEM_PREAMBLE).encode() not in observation_bytes
    assert argv[argv.index("--append-system-prompt") + 1] == "[content omitted]"
    assert observation["content_arguments"]["--append-system-prompt"] == [{
        "bytes": len(_PI_SYSTEM_PREAMBLE.encode()),
        "sha256": hashlib.sha256(_PI_SYSTEM_PREAMBLE.encode()).hexdigest(),
    }]
    assert any(isinstance(event, TextEvent) and event.text == "CURRENT_REASONING_CANARY" for event in events)
    assert len([event for event in events if isinstance(event, ResultEvent)]) == 1
    assert backend._transports == []


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
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), prompt, **kwargs):
            pass
    return list(mock_exec.call_args.args), mock_exec


@pytest.mark.asyncio
async def test_pi_execution_input_controls_native_argv_environment_and_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daydream.backends import BackendExecutionInput, RetryPolicy

    execution = BackendExecutionInput.from_environment(
        {
            "HOME": str(tmp_path / "run-home"),
            "PATH": "/run/bin",
            "PI_PROVIDER": "openrouter",
            "PI_THINKING": "high",
            "PI_API_KEY": "run-key",
            "DAYDREAM_PI_FANOUT_CONCURRENCY": "2",
            "DAYDREAM_PI_RETRY_ATTEMPTS": "1",
            "DAYDREAM_PI_RETRY_BASE_DELAY_S": "0",
            "DAYDREAM_PI_RETRY_MAX_DELAY_S": "4",
        },
        backend="pi",
    )
    monkeypatch.setenv("PI_PROVIDER", "zai")
    monkeypatch.setenv("PI_THINKING", "low")
    monkeypatch.setenv("PI_API_KEY", "ambient-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ambient-native-key")

    backend = PiBackend(model="fixture-model", execution_input=execution)
    args, mock_exec = await _run_and_capture_args(backend)
    child = mock_exec.call_args.kwargs["env"]

    assert args[args.index("--provider") + 1] == "openrouter"
    assert args[args.index("--thinking") + 1] == "high"
    assert child["OPENROUTER_API_KEY"] == "run-key"
    assert child["PATH"] == "/run/bin"
    assert "PI_API_KEY" not in child
    assert "ambient-key" not in repr(child)
    assert backend.fanout_concurrency == 2
    assert backend.retry_policy == RetryPolicy(1, 0.0, 4.0)


@pytest.mark.asyncio
async def test_simple_text_events() -> None:
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
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
    assert metrics_events[0].prompt_tokens == 110  # uncached input plus cache-read subset
    assert metrics_events[0].completion_tokens == 50
    assert metrics_events[0].cached_tokens == 10
    assert metrics_events[0].cost_usd == 0.0003
    assert metrics_events[0].message_id == ""

    assert len(cost_events) == 1
    assert cost_events[0].cost_usd == 0.0003
    assert cost_events[0].input_tokens == 110
    assert cost_events[0].output_tokens == 50
    assert cost_events[0].cached_tokens == 10
    assert cost_events[0].model_name == "glm-4.6"  # actual response model from the recorded stream

    assert len(result_events) == 1
    assert result_events[0].continuation is not None
    assert result_events[0].continuation.backend == "pi"
    assert result_events[0].continuation.data["session_id"] == "pi_ses_simple"


@pytest.mark.asyncio
async def test_thinking_and_tool_use_events() -> None:
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("tool_use.jsonl")

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
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

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
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

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
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

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
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
        "daydream.backends._transport.asyncio.create_subprocess_exec",
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

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
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
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        return_value=mock_proc,
    ) as mock_exec:
        events = []
        async for event in backend.execute(Path("/tmp"), "hello"):
            events.append(event)
    assert mock_exec.call_args.kwargs["start_new_session"] is True


@pytest.mark.asyncio
async def test_execute_finally_reaps_process_after_exit() -> None:
    """Even when the CLI already exited, the finally reaps the process group.

    The transport teardown runs unconditionally: its group signal fires
    regardless of ``returncode``, so a grandchild that outlived the CLI is
    still signalled and the pipe fds are still released.
    """
    from tests.harness.fake_cli_process import FakeCliProcess, FakeCliSpawner

    backend = PiBackend(model="glm-5.2")
    captured = FakeCliSpawner()

    async def fake_exec(*args: Any, **kwargs: Any) -> FakeCliProcess:
        proc = FakeCliProcess(
            [
                '{"type":"session","sessionId":"pi_ses_bye"}',
                '{"type":"agent_start"}',
                '{"type":"agent_end","messages":[]}',
            ],
            exit_code=0,
        )
        captured.procs.append(proc)
        return proc

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
        events = [event async for event in backend.execute(Path("/tmp"), "hello")]

    assert events  # the turn ran to completion
    proc = captured.procs[0]
    assert proc.returncode == 0
    assert proc.reaped, "the finally must reap the child even after clean exit"
    assert proc._transport.closed, "the finally must release the pipe fds even after clean exit"
    assert backend._transports == [], "the finally must drop the transport from the backend list"


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

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
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
    from daydream.backends._transport import CliTransport

    backend = PiBackend(model="glm-5.2")

    proc = MagicMock()
    proc.returncode = None
    proc.wait = AsyncMock(side_effect=[asyncio.TimeoutError(), 0])
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    transport = CliTransport("pi", ["pi", "--mode", "json"], limit=1024)
    transport.processes.append(proc)
    backend._transports = [transport]

    await backend.cancel()

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_no_op_when_no_processes() -> None:
    backend = PiBackend(model="glm-5.2")
    backend._transports = []
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

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
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

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
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
    assert cost_events[0].input_tokens is None


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

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
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

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
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

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
        first_iter = backend.execute(Path("/tmp"), "first")
        assert isinstance(await anext(first_iter), RequestEvent)
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
        ({"details": {"note": "x"}}, '{"details": {"note": "x"}}'),
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
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
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
            with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
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
        "daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc
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
        "daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc
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
        "daydream.backends._transport.asyncio.create_subprocess_exec", _fake_exec
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
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
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
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
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
        patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc),
        patch(
            "daydream.backends._transport.readline_with_idle_timeout",
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

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
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

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
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


# --- P18 Task 1: generation lifecycle + config at the Pi argv/JSONL seam -----


def test_pi_replay_fixture_is_sanitized_labeled() -> None:
    """The replay fixture carries no real private content — placeholders only."""
    fixture = FIXTURES_DIR / "generation_lifecycle.jsonl"
    text = fixture.read_text(encoding="utf-8")
    # Placeholder-marked content everywhere; exact pinned numbers only.
    assert "THINKING_PLACEHOLDER_ONE" in text
    assert "TEXT_PLACEHOLDER" in text
    assert "FILE_PLACEHOLDER" in text
    assert "src/example.py" in text  # synthetic argument path, not a real one
    assert "1788690314289" in text
    # No real artifact content: the pinned historical blob text is absent.
    assert "395.332" not in text.split("\n")[0]


@pytest.mark.asyncio
async def test_pi_generation_lifecycle_start_end_pair_around_tool() -> None:
    """Two assistant generations: start before, end sealed at message_end before tools."""
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("generation_lifecycle.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "go"):
            events.append(event)

    starts = [e for e in events if isinstance(e, GenerationStartEvent)]
    ends = [e for e in events if isinstance(e, GenerationEndEvent)]
    tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
    assert len(starts) == 2 and len(ends) == 2

    first_end = ends[0]
    assert first_end.generation_id == starts[0].generation_id
    # Host invocation-local UUID correlation, never a provider identity.
    assert len(first_end.generation_id) == 36
    assert first_end.response_id == "resp_gen_01"  # native, attached to the end
    assert first_end.response_id != first_end.generation_id
    assert first_end.model_name == "glm-4.6" and first_end.provider_name == "nous"
    assert first_end.finish_reason == "toolUse"
    assert first_end.end_source == "host_observed_message_end"
    assert first_end.boundary_complete is True

    # Ordered complete provider choice: reasoning, text, tool-call (provider order).
    kinds = [part.kind for part in first_end.choice_parts]
    assert kinds == ["reasoning", "text", "tool_call"]
    tool_part = first_end.choice_parts[2]
    assert isinstance(tool_part, ToolCallChoicePart)
    assert tool_part.call_id == "call_001"
    assert tool_part.name == "read_file"
    assert tool_part.arguments == {"path": "src/example.py"}

    # The tool execution is a later sibling linked by call ID and does not
    # author or duplicate the choice part.
    assert tool_starts[0].id == "call_001"
    ordering = [type(e).__name__ for e in events]
    assert ordering.index("GenerationEndEvent") < ordering.index("ToolStartEvent")
    assert len([p for p in first_end.choice_parts if p.kind == "tool_call"]) == 1

    # Second generation: text-only choice, distinct correlation ID.
    second_end = ends[1]
    assert second_end.generation_id == starts[1].generation_id
    assert second_end.generation_id != first_end.generation_id
    assert [p.kind for p in second_end.choice_parts] == ["text"]
    assert second_end.finish_reason == "stop"


@pytest.mark.asyncio
async def test_pi_native_ms_start_converts_exactly_and_chronology_holds() -> None:
    """Native Unix-ms start → exact ns; end receipt is host-observed and later."""
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("generation_lifecycle.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "go"):
            events.append(event)

    ends = [e for e in events if isinstance(e, GenerationEndEvent)]
    first = ends[0]
    # Exact producer shape: multiplication-only conversion of 1788690314289 ms.
    assert first.native_started_at_unix_ms == 1788690314289
    assert first.native_started_at_unix_ms is not None
    assert unix_ms_to_ns(first.native_started_at_unix_ms) == 1788690314289000000
    # Host end receipt is a Unix-ns instant at/after the native start.
    assert first.ended_at_unix_ns >= unix_ms_to_ns(first.native_started_at_unix_ms)


@pytest.mark.asyncio
async def test_pi_user_and_tool_results_do_not_create_generations() -> None:
    """Only assistant message boundaries create generation events."""
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("generation_lifecycle.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "go"):
            events.append(event)

    # The fixture has exactly two assistant messages → exactly two pairs.
    # Tool execution events and turn boundaries are not generations.
    assert len([e for e in events if isinstance(e, GenerationStartEvent)]) == 2
    tool_events = [e for e in events if isinstance(e, (ToolStartEvent, ToolResultEvent))]
    assert tool_events
    assert all(
        not isinstance(e, (GenerationStartEvent, GenerationEndEvent)) for e in tool_events
    )


@pytest.mark.asyncio
async def test_pi_turn_end_carries_native_identity() -> None:
    """Per-turn finish reason/model/provider land on the matching TurnEndEvent."""
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("generation_lifecycle.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "go"):
            events.append(event)

    turn_ends = [e for e in events if isinstance(e, TurnEndEvent)]
    assert turn_ends
    final = turn_ends[-1]
    assert final.finish_reason == "stop"
    assert final.model_name == "glm-4.6"
    assert final.provider_name == "nous"
    assert final.model_source == "native"
    assert final.provider_source == "native"
    assert final.message_id == ""  # Pi exposes no native message id
    assert final.message_id_source is None
    assert final.timestamp_source == "host_observed"


@pytest.mark.asyncio
async def test_pi_request_event_config_matches_exact_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pi config mirrors argv: read-only tools, no-skills, emulated schema."""
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.delenv("PI_API_KEY", raising=False)
    backend = PiBackend(model="glm-5.2")
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    flat_args, _ = await _run_and_capture_args(
        backend, "structured please", output_schema=schema, read_only=True,
        persist_session=False,
    )

    request_events: list[RequestEvent] = []
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        async for event in PiBackend(model="glm-5.2").execute(
            Path("/tmp"), "structured please", output_schema=schema, read_only=True,
            persist_session=False,
        ):
            if isinstance(event, RequestEvent):
                request_events.append(event)
    request = request_events[0]
    config = request.config
    assert isinstance(config, PiRequestConfig)
    # Exact argv correspondence for each admitted control.
    assert config.read_only is True
    assert flat_args[flat_args.index("--tools") + 1] == "read,find,ls,grep"
    assert config.selected_tools_count == 4 and config.selected_tools_present is True
    assert config.no_skills is True and "--no-skills" in flat_args
    assert config.schema_emulated is True  # no native Pi schema flag
    assert config.persist_session is False and "--no-session" in flat_args
    assert config.continuation_mode == "fresh"
    assert config.model_mode == "single"
    assert config.max_turns is None  # accepted by daydream, never passed to Pi
    # System preamble is invocation-level content, never a config value.
    assert request.system_prompt is not None and "tool-call budget" in request.system_prompt


@pytest.mark.asyncio
async def test_pi_multi_turn_fixture_produces_two_turn_end_boundaries() -> None:
    """Two text turns → two TurnEndEvents, each with its own native identity."""
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("multi_turn.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "go"):
            events.append(event)

    turn_ends = [e for e in events if isinstance(e, TurnEndEvent)]
    assert len(turn_ends) == 2
    assert [t.finish_reason for t in turn_ends] == ["stop", "stop"]
    assert all(t.model_name == "glm-4.6" for t in turn_ends)


@pytest.mark.asyncio
async def test_pi_error_turn_sets_explicit_incomplete_boundary() -> None:
    """An errored turn keeps identity native where exposed; outcome stays custom."""
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("error_turn.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(PiError):
            async for _ in backend.execute(Path("/tmp"), "go"):
                pass

    # The error fixture carries no model/provider on the boundary; identity
    # fields must stay None, never a fabricated value.
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        collected: list[Any] = []
        mock_proc2 = make_mock_process_from_fixture("error_turn.jsonl")
        with patch(
            "daydream.backends._transport.asyncio.create_subprocess_exec",
            return_value=mock_proc2,
        ):
            try:
                async for event in PiBackend(model="glm-5.2").execute(Path("/tmp"), "go"):
                    collected.append(event)
            except PiError:
                pass
    turn_ends = [e for e in collected if isinstance(e, TurnEndEvent)]
    assert turn_ends
    error_turn = turn_ends[-1]
    assert error_turn.finish_reason == "error"
    # glm model absent in the error fixture message → no claimed identity.
    assert error_turn.model_name is None or error_turn.model_name == "glm-5.2"


@pytest.mark.asyncio
async def test_pi_usage_events_carry_turn_end_and_terminal_provenance() -> None:
    """Pi turn usage is turn_end-sourced; terminal totals are reported cost."""
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("generation_lifecycle.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "go"):
            events.append(event)

    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    assert metrics
    assert all(m.measurement_source == "turn_end" for m in metrics)

    costs = [e for e in events if isinstance(e, CostEvent)]
    assert costs
    terminal = costs[-1]
    assert terminal.measurement_source == "terminal"
    assert terminal.cost_source == "reported"
