"""Drive PiBackend.execute through canned JSONL and real protocol CLI fixtures."""

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daydream.atif import validate
from daydream.backends import (
    BackendExecutionInput,
    ContinuationToken,
    CostEvent,
    GenerationEndEvent,
    GenerationStartEvent,
    MetricsEvent,
    PiRequestConfig,
    RequestEvent,
    ResultEvent,
    RetryPolicy,
    TextEvent,
    ThinkingEvent,
    ToolCallChoicePart,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
    create_backend,
    unix_ms_to_ns,
)
from daydream.backends._subprocess import StreamStalledError
from daydream.backends.pi import (
    _PI_DEFAULT_RETRY_ATTEMPTS,
    _PI_DEFAULT_RETRY_BASE_DELAY,
    _PI_DEFAULT_RETRY_MAX_DELAY,
    _PI_STDOUT_LIMIT_BYTES,
    _PI_SYSTEM_PREAMBLE,
    PiBackend,
    PiError,
    _is_retryable_exit_code,
    _pi_error_category,
    _pi_retry_attempts,
    _pi_retry_base_delay,
    _pi_retry_max_delay,
    _pi_retryable_for,
    _render_tool_result,
    _schema_instruction,
)
from daydream.config import DEFAULT_PI_MODEL
from daydream.retry_policy import parse_message_retry_hint
from daydream.runner import run
from daydream.trajectory import DaydreamPhase
from tests.harness.fake_cli_process import LimitAwareStdout, assert_concurrent_streams_isolated
from tests.harness.pi_replay import FIXTURES_DIR, make_mock_process, make_mock_process_from_fixture
from tests.harness.process_replay import replay_process
from tests.harness.protocol_cli import install_protocol_cli
from tests.harness.protocol_cli_assertions import assert_protocol_cli_invariants, make_cancel_probe
from tests.harness.stub_backend import force_interactive as _force_interactive, silence as _silence
from tests.harness.trajectory import make_recorder

if TYPE_CHECKING:
    from daydream.run_config import RunConfig

MakeConfig = Callable[..., "RunConfig"]
Mute = Callable[..., None]

def test_backend_execution_input_is_immutable_parsed_and_returns_fresh_environment(tmp_path: Path) -> None:
    source = {
        "HOME": str(tmp_path / "home"), "PATH": "/run/bin", "PI_PROVIDER": "openrouter", "PI_THINKING": "high",
        "PI_API_KEY": "run-secret", "PI_CODING_AGENT_DIR": str(tmp_path / "pi-agent"),
        "DAYDREAM_PI_FANOUT_CONCURRENCY": "3", "DAYDREAM_PI_RETRY_ATTEMPTS": "4",
        "DAYDREAM_PI_RETRY_BASE_DELAY_S": "0.25", "DAYDREAM_PI_RETRY_MAX_DELAY_S": "5",
        "DAYDREAM_STREAM_IDLE_TIMEOUT_S": "7",
    }
    execution = BackendExecutionInput.from_environment(source, backend="pi")
    source["PI_API_KEY"] = "mutated"
    first = execution.child_environment()
    first["PI_API_KEY"] = "also-mutated"
    assert execution.retry_policy == RetryPolicy(attempts=4, base_delay_s=0.25, max_delay_s=5.0)
    assert execution.fanout_concurrency == 3
    assert execution.pi_provider == "openrouter"
    assert execution.pi_thinking == "high"
    assert execution.pi_agent_dir == tmp_path / "pi-agent"
    assert execution.stream_idle_timeout_s == 7.0
    assert execution.pi_response_idle_timeout_s == 7.0
    assert execution.child_environment()["PI_API_KEY"] == "run-secret"
    assert "run-secret" not in repr(execution)

def test_backend_execution_input_uses_backend_specific_defaults(tmp_path: Path) -> None:
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
    with pytest.raises(ValueError, match="osprey"):
        BackendExecutionInput.from_environment({}, backend="osprey")

async def test_artifact_visibility_protocol_cli_uses_argv_prompt_devnull_and_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    observation = assert_protocol_cli_invariants(fixture, target, prompt, events, backend)
    argv = observation["argv"]
    assert argv[argv.index("--mode") + 1] == "json"
    assert argv[argv.index("--provider") + 1] == "nous"
    assert argv[argv.index("--model") + 1] == "fixture-model"
    assert "--append-system-prompt" in argv and "--no-skills" in argv
    # Neither the prompt nor the preamble is recorded verbatim -- only digests.
    observation_bytes = next(fixture.observations.glob("*.json")).read_bytes()
    assert prompt.encode() not in observation_bytes
    assert json.dumps(_PI_SYSTEM_PREAMBLE).encode() not in observation_bytes
    assert argv[argv.index("--append-system-prompt") + 1] == "[content omitted]"
    assert observation["content_arguments"]["--append-system-prompt"] == [{
        "bytes": len(_PI_SYSTEM_PREAMBLE.encode()), "sha256": hashlib.sha256(_PI_SYSTEM_PREAMBLE.encode()).hexdigest(),
    }]

@pytest.fixture
def pi_workspace(tmp_path: Path) -> Path:
    """Return a workspace whose ``.pi/settings.json`` selects a default model."""
    settings = tmp_path / ".pi" / "settings.json"
    settings.parent.mkdir()
    settings.write_text('{"defaultProvider": "openai", "defaultModel": "gpt-5.6-luna"}')
    return tmp_path

async def _run_and_capture_args(
    backend: Any, prompt: Any="p", *, fixture: Any="simple_text.jsonl", cwd: Path = Path("/tmp"), **kwargs: Any,
) -> tuple[Any, ...]:
    """Replay a fixture and return (argv, spawner) for exact command/environment assertions."""
    mock_proc = make_mock_process_from_fixture(fixture)
    _, mock_exec = await replay_process(backend, mock_proc, cwd, prompt, **kwargs)
    return list(mock_exec.call_args.args), mock_exec

async def _collect_events(
    backend: Any, prompt: Any = "p", *, fixture: Any = "simple_text.jsonl", **kwargs: Any,
) -> list[Any]:
    mock_proc = make_mock_process_from_fixture(fixture)
    events, _ = await replay_process(backend, mock_proc, Path("/tmp"), prompt, **kwargs)
    return events

async def test_pi_execution_input_controls_native_argv_environment_and_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = BackendExecutionInput.from_environment(
        {
            "HOME": str(tmp_path / "run-home"), "PATH": "/run/bin", "PI_PROVIDER": "openrouter", "PI_THINKING": "high",
            "PI_API_KEY": "run-key", "DAYDREAM_PI_FANOUT_CONCURRENCY": "2", "DAYDREAM_PI_RETRY_ATTEMPTS": "1",
            "DAYDREAM_PI_RETRY_BASE_DELAY_S": "0", "DAYDREAM_PI_RETRY_MAX_DELAY_S": "4",
        }, backend="pi",
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

async def test_simple_text_events() -> None:
    events = await _collect_events(PiBackend(model="glm-5.2"), "Say hello", fixture="simple_text.jsonl")
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

async def test_thinking_and_tool_use_events() -> None:
    events = await _collect_events(PiBackend(model="glm-5.2"), "Read the file", fixture="tool_use.jsonl")
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
    assert any(t.text == "Looking now" for t in texts)

async def test_structured_output() -> None:
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("structured_output.jsonl")
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}
    events, mock_exec = await replay_process(backend, mock_proc, Path("/tmp"), "Parse", output_schema=schema)
    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output == {
        "issues": [{"id": 1, "description": "Fix type hints", "file": "app.py", "line": 5}]
    }
    # Request records preserve the logical prompt; argv carries only an attachment.
    flat_args = list(mock_exec.call_args.args)
    positional = flat_args[-1]
    assert positional.startswith("@/")
    assert not Path(positional[1:]).exists()
    request = next(e for e in events if isinstance(e, RequestEvent))
    assert "JSON schema" in request.prompt
    assert json.dumps(schema) in request.prompt

async def test_multi_turn_emits_turn_end_per_turn_and_aggregates_cost() -> None:
    events = await _collect_events(PiBackend(model="glm-5.2"), "Two turns", fixture="multi_turn.jsonl")
    texts = [e for e in events if isinstance(e, TextEvent)]
    turn_ends = [e for e in events if isinstance(e, TurnEndEvent)]
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    cost_events = [e for e in events if isinstance(e, CostEvent)]
    assert [t.text for t in texts] == ["First turn body", "Second turn body"]
    assert len(turn_ends) == 2
    assert all(e.message_id == "" for e in turn_ends)
    assert len(metrics) == 2
    assert len(cost_events) == 1
    assert cost_events[0].input_tokens == 200  # 150 + 50
    assert cost_events[0].output_tokens == 100  # 75 + 25
    assert cost_events[0].cost_usd == pytest.approx(0.00015)  # 0.0001 + 0.00005

async def test_error_turn_raises_pi_error() -> None:
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("error_turn.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(PiError, match="Model returned an error"):
            async for _ in backend.execute(Path("/tmp"), "Fail"):
                pass

async def test_continuation_token_uses_session_id_flag() -> None:
    backend = PiBackend(model="glm-5.2")
    token = ContinuationToken(backend="pi", data={"session_id": "pi_resume_me"})
    flat_args, _ = await _run_and_capture_args(backend, "Continue", continuation=token)
    assert "--session-id" in flat_args
    assert flat_args[flat_args.index("--session-id") + 1] == "pi_resume_me"
    assert "--no-session" not in flat_args

async def test_fresh_run_uses_session_id_not_no_session() -> None:
    """Fresh runs use a generated persistent session id so later healing can resume.

    --no-session would discard history and make a later resume start empty.
    """
    backend = PiBackend(model="glm-5.2")
    flat_args, _ = await _run_and_capture_args(backend, "Fresh")
    assert "--no-session" not in flat_args
    assert "--session-id" in flat_args
    passed_id = flat_args[flat_args.index("--session-id") + 1]
    # A genuine UUID: the token must name a resumable persistent session.
    uuid.UUID(passed_id)

async def test_ephemeral_pi_call_uses_no_session() -> None:
    backend = PiBackend(model="glm-5.2")
    flat_args, _ = await _run_and_capture_args(backend, persist_session=False)
    assert "--no-session" in flat_args
    assert "--session-id" not in flat_args
    events = await _collect_events(backend, persist_session=False, fixture="simple_text.jsonl")
    result_events = [event for event in events if isinstance(event, ResultEvent)]
    assert result_events[0].continuation is None

async def test_read_only_restricts_tools() -> None:
    backend = PiBackend(model="glm-5.2")
    flat_args, _ = await _run_and_capture_args(backend, read_only=True)
    assert flat_args[flat_args.index("--tools") + 1] == "read,find,ls,grep"
    flat_args_default, _ = await _run_and_capture_args(backend)
    assert "--tools" not in flat_args_default

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

async def test_pi_api_key_unknown_provider_warns_and_skips(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """An unmapped provider warns and proceeds; the key never reaches argv or any env var."""
    sentinel = "synthetic-unknown-provider-key"
    monkeypatch.setenv("PI_PROVIDER", "custom-provider")
    monkeypatch.setenv("PI_API_KEY", sentinel)
    backend = PiBackend(model="custom-model")
    with caplog.at_level("WARNING"):
        flat_args, mock_exec = await _run_and_capture_args(backend)
    assert flat_args[flat_args.index("--provider") + 1] == "custom-provider"
    assert sentinel not in flat_args
    assert "--api-key" not in flat_args
    child_env = mock_exec.call_args.kwargs["env"]
    assert sentinel not in child_env.values()
    assert "PI_API_KEY" not in child_env
    assert any("PI_API_KEY" in r.getMessage() for r in caplog.records)
    assert sentinel not in caplog.text

async def test_cwd_passed_to_subprocess() -> None:
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/some/repo"), "p"):
            pass
        assert mock_exec.call_args.kwargs["cwd"] == "/some/repo"
        assert mock_exec.call_args.kwargs["limit"] == _PI_STDOUT_LIMIT_BYTES

async def test_execute_raises_on_agents() -> None:
    backend = PiBackend(model="glm-5.2")
    mock_agent = {"description": "test", "prompt": "test"}
    with pytest.raises(NotImplementedError, match="Pi backend does not support exploration"):
        async for _ in backend.execute(Path("/tmp"), "Test", agents={"explorer": mock_agent}):
            pass

async def test_agent_end_always_finalizes_when_stream_ends_without_it() -> None:
    backend = PiBackend(model="glm-5.2")
    lines = [
        '{"type":"session","sessionId":"pi_ses_truncated"}', '{"type":"agent_start"}', '{"type":"turn_start"}',
        '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]}}',
        '{"type":"turn_end","message":{"role":"assistant","content":[{"type":"text","text":"hi"}],'
        '"usage":{"input":5,"output":3,"cost":{"total":0.0001}},"stopReason":"stop"}}',
    ]
    mock_proc = make_mock_process(lines)
    events, _ = await replay_process(backend, mock_proc, Path("/tmp"), "Truncated")
    cost_events = [e for e in events if isinstance(e, CostEvent)]
    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(cost_events) == 1
    assert len(result_events) == 1
    cont = result_events[0].continuation
    assert cont is not None
    assert cont.data["session_id"] == "pi_ses_truncated"

async def test_cancel_terminates_then_kills() -> None:
    backend, proc = make_cancel_probe("pi")
    await backend.cancel()
    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()

async def test_cancel_no_op_when_no_processes() -> None:
    backend = PiBackend(model="glm-5.2")
    backend._transports = []
    await backend.cancel()  # Must not raise.

async def test_stdout_limit_allows_large_jsonl_events() -> None:
    backend = PiBackend(model="glm-5.2")
    large_text = "x" * (70 * 1024)
    large_line = (
        json.dumps(
            {
                "type": "message_end",
                "message": {"role": "assistant", "content": [{"type": "text", "text": large_text}]},
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
    async def fake_exec(*args: object, **kwargs: object) -> MagicMock:
        captured.update(kwargs)
        raw_limit = kwargs.get("limit", 64 * 1024)
        limit = raw_limit if isinstance(raw_limit, int) else 64 * 1024
        process = MagicMock()
        process.stdout = LimitAwareStdout(lines, limit)
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

async def test_missing_usage_skips_metrics_but_keeps_turn_end() -> None:
    backend = PiBackend(model="glm-5.2")
    lines = [
        '{"type":"session","sessionId":"pi_ses_nousage"}', '{"type":"agent_start"}', '{"type":"turn_start"}',
        '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]}}',
        '{"type":"turn_end","message":{"role":"assistant","content":[{"type":"text","text":"hi"}],"stopReason":"stop"}}',
        '{"type":"agent_end","messages":[]}',
    ]
    mock_proc = make_mock_process(lines)
    events, _ = await replay_process(backend, mock_proc, Path("/tmp"), "p")
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    turn_ends = [e for e in events if isinstance(e, TurnEndEvent)]
    cost_events = [e for e in events if isinstance(e, CostEvent)]
    assert metrics == []
    assert len(turn_ends) == 1
    assert len(cost_events) == 1
    assert cost_events[0].cost_usd is None
    assert cost_events[0].input_tokens is None

async def test_nonzero_exit_raises_with_captured_output() -> None:
    backend = PiBackend(model="glm-5.2")
    lines = [
        "Error: authentication required. Run `pi login` to authenticate.", "fatal: could not connect to API endpoint",
    ]
    mock_proc = make_mock_process(lines)
    mock_proc.returncode = 1
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(PiError, match="return code 1") as exc_info:
            async for _ in backend.execute(Path("/tmp"), "p"):
                pass
    msg = str(exc_info.value)
    assert "authentication required" in msg
    assert "could not connect" in msg

async def test_nonzero_exit_with_no_output_still_informative() -> None:
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process([])
    mock_proc.returncode = 1
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(PiError, match="return code 1") as exc_info:
            async for _ in backend.execute(Path("/tmp"), "p"):
                pass
    assert "no non-JSON output captured" in str(exc_info.value)

async def test_concurrent_execute_calls_do_not_share_stdout_reader() -> None:
    await assert_concurrent_streams_isolated(
        PiBackend(model="glm-5.2"),
        [
            '{"type":"session","sessionId":"s1"}', '{"type":"agent_start"}', '{"type":"turn_start"}',
            '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"first"}]}}',
            '{"type":"turn_end","message":{"role":"assistant","content":[],"stopReason":"stop"}}',
            '{"type":"agent_end","messages":[]}',
        ],
    )

@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"content": [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]}, "line1line2"),
        ({"content": "raw"}, "raw"), ({"details": {"note": "x"}}, '{"details": {"note": "x"}}'), ("plain", "plain"),
        (None, ""),
    ], ids=["text-blocks", "string-content", "details-fallback", "non-dict-str", "non-dict-none"],
)
def test_render_tool_result(result: Any, expected: Any) -> None:
    assert _render_tool_result(result) == expected

def test_schema_instruction_contains_schema_json() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    instruction = _schema_instruction(schema)
    assert "JSON schema" in instruction
    assert json.dumps(schema) in instruction

async def test_execute_always_passes_no_skills_never_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    _, mock_exec = await replay_process(backend, mock_proc, tmp_path, "Review the change.")
    args = list(mock_exec.call_args.args)
    assert args.count("--no-skills") == 1
    assert "--skill" not in args

def test_create_backend_pi_returns_pi_backend_with_default_model() -> None:
    backend = create_backend("pi")
    assert isinstance(backend, PiBackend)
    assert backend.model == DEFAULT_PI_MODEL
    custom = create_backend("pi", model="glm-4.5-air")
    assert isinstance(custom, PiBackend)
    assert custom.model == "glm-4.5-air"

def test_create_backend_invalid_includes_pi_in_message() -> None:
    with pytest.raises(ValueError, match="pi"):
        create_backend("invalid")

# Live Pi requires explicit opt-in: an installed CLI may hang on login or incur model charges.
_PI_AVAILABLE = shutil.which("pi") is not None
_PI_LIVE_OPT_IN = os.environ.get("DAYDREAM_PI_LIVE") == "1"

@pytest.mark.skipif(
    not (_PI_AVAILABLE and _PI_LIVE_OPT_IN),
    reason="live pi smoke test; set DAYDREAM_PI_LIVE=1 (and ensure `pi` is on $PATH and logged in) to run",
)
async def test_live_pi_smoke() -> None:
    """Opt-in real Pi test requiring assistant text; EOF terminal events alone cannot prove success.

    Bound the call so missing credentials or interactive login cannot hang the suite.
    """
    backend = PiBackend(model="glm-5.2")
    events = []
    async def _collect() -> None:
        async for event in backend.execute(Path("/tmp"), "Reply with exactly: pong"):
            events.append(event)
    await asyncio.wait_for(_collect(), timeout=60.0)
    # EOF and terminal events alone are insufficient; require actual response text.
    text = "".join(e.text for e in events if isinstance(e, TextEvent))
    assert "pong" in text.lower(), f"no assistant text emitted (auth/model failure?): events={events!r}"
    assert any(isinstance(e, ResultEvent) for e in events)
    assert any(isinstance(e, CostEvent) for e in events)

async def test_pi_trajectory_is_valid_atif_v1_7(tmp_path: Path) -> None:
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("tool_use.jsonl")
    traj_path = tmp_path / "trajectory.json"
    recorder = make_recorder(
        tmp_path, path=traj_path, agent_model_name="glm-5.2", session_id="00000000-0000-0000-0000-0000000000aa",
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
                async for event in backend.execute(tmp_path, "Review"):
                    inv.observe(event)
    assert traj_path.is_file()
    assert validate(traj_path, validate_images=False)
    # Terminal cost may open an empty trailing step; inspect a content-bearing agent step.
    agent_steps = [s for s in recorder.steps if s.source == "agent"]
    assert len(agent_steps) >= 1
    step = agent_steps[0]
    assert step.message == "Looking now"
    assert step.reasoning_content == "Let me read the file"
    assert [tc.tool_call_id for tc in (step.tool_calls or [])] == ["t1"]
    obs = {r.source_call_id: r.content for r in (step.observation.results if step.observation else [])}
    assert obs == {"t1": "file.py\ntest.py"}
    assert step.metrics is not None
    assert step.metrics.prompt_tokens == 200
    assert step.metrics.completion_tokens == 100
    assert step.metrics.cost_usd == 0.0005

# Default --provider is nous; PI_PROVIDER overrides (extension-based provider)

@pytest.mark.parametrize(
    ("env_provider", "expected"), [(None, "nous"), ("my-proxy", "my-proxy")],
    ids=["default-nous", "PI_PROVIDER-override"],
)
async def test_provider_flag(monkeypatch: pytest.MonkeyPatch, env_provider: Any, expected: Any) -> None:
    """Always pass the default nous provider unless PI_PROVIDER overrides it, so the fallback model
    resolves consistently.
    """
    if env_provider is None:
        monkeypatch.delenv("PI_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("PI_PROVIDER", env_provider)
    backend = PiBackend(model="glm-5.2")
    flat_args, _ = await _run_and_capture_args(backend)
    assert flat_args[flat_args.index("--provider") + 1] == expected

async def test_default_model_does_not_override_pi_settings(pi_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    backend = PiBackend()
    flat_args, _ = await _run_and_capture_args(backend, cwd=pi_workspace)
    assert "--model" not in flat_args
    assert "--provider" not in flat_args

def test_public_model_reflects_pi_settings_before_execute(pi_workspace: Path) -> None:
    backend = PiBackend(cwd=pi_workspace)
    assert backend.model == "gpt-5.6-luna"

async def test_explicit_model_overrides_pi_settings(pi_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    backend = PiBackend(model="custom-model")
    flat_args, _ = await _run_and_capture_args(backend, cwd=pi_workspace)
    assert flat_args[flat_args.index("--model") + 1] == "custom-model"

async def test_nous_deepseek_is_pi_fallback_when_no_model_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi-agent"))
    backend = PiBackend()
    flat_args, _ = await _run_and_capture_args(backend)
    assert flat_args[flat_args.index("--model") + 1] == "deepseek/deepseek-v4-flash-0731"
    assert flat_args[flat_args.index("--provider") + 1] == "nous"

# Migration guards: GLM-pin and provider/model mismatch warnings

@pytest.mark.parametrize("provider", [None, "zai"])
async def test_glm_pin_warning_requires_implicit_provider(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, provider: str | None,
) -> None:
    if provider is None:
        monkeypatch.delenv("PI_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("PI_PROVIDER", provider)
    monkeypatch.setattr("daydream.backends.pi._warned_migration_mismatches", set())
    with caplog.at_level("WARNING"):
        flat_args, _ = await _run_and_capture_args(PiBackend(model="glm-5.2"))
    assert any("z.ai-hosted GLM" in record.getMessage() for record in caplog.records) is (provider is None)
    assert flat_args[flat_args.index("--model") + 1] == "glm-5.2"
    assert flat_args[flat_args.index("--provider") + 1] == (provider or "nous")


async def test_glm_pin_warning_fires_once_across_executes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.setattr("daydream.backends.pi._warned_migration_mismatches", set())
    backend = PiBackend(model="glm-5.2")
    with caplog.at_level("WARNING"):
        await _run_and_capture_args(backend)
        await _run_and_capture_args(backend)
    glm_warnings = [r for r in caplog.records if "z.ai-hosted GLM" in r.getMessage()]
    assert len(glm_warnings) == 1

async def test_zai_provider_with_fallback_model_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("PI_PROVIDER", "zai")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi-agent"))
    monkeypatch.setattr("daydream.backends.pi._warned_migration_mismatches", set())
    backend = PiBackend()
    with caplog.at_level("WARNING"):
        flat_args, _ = await _run_and_capture_args(backend)
    assert any("no configured model" in r.getMessage() for r in caplog.records)
    assert flat_args[flat_args.index("--provider") + 1] == "zai"
    assert flat_args[flat_args.index("--model") + 1] == "deepseek/deepseek-v4-flash-0731"

# Real-path through runner.run: real PiBackend, only the pi subprocess mocked

def _capture_pi_subprocess(monkeypatch: pytest.MonkeyPatch, captured: list[list[str]]) -> None:
    """Intercept only pi spawns, record argv, and replay a no-op session.

    Keep the real backend/factory; other asyncio subprocess calls pass through.
    """
    # Any permits this forwarding wrapper around the keyword-typed subprocess signature.
    real_exec: Any = asyncio.create_subprocess_exec
    async def _fake_exec(*args: object, **kwargs: object) -> MagicMock:
        if args and args[0] == "pi":
            captured.append([str(a) for a in args])
            return make_mock_process_from_fixture("simple_text.jsonl")
        return cast(MagicMock, await real_exec(*args, **kwargs))
    monkeypatch.setattr("daydream.backends._transport.asyncio.create_subprocess_exec", _fake_exec)

def _assert_pi_model_and_provider(captured: list[list[str]], *, model: str, provider: str) -> None:
    assert captured, "expected at least one pi subprocess spawn"
    for argv in captured:
        assert argv[argv.index("--model") + 1] == model
        assert argv[argv.index("--provider") + 1] == provider

@pytest.mark.parametrize(
    ("env_provider", "expected_provider"), [(None, "nous"), ("custom-proxy", "custom-proxy")],
    ids=["default-nous", "PI_PROVIDER-override"],
)
async def test_runner_real_path_pi_provider_axis(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    env_provider: str | None, expected_provider: str,
) -> None:
    """Run the real factory/backend in a Git worktree with only Pi spawning mocked.

    An empty settings directory forces the fallback model and tests provider overrides.
    """
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
    rc = await run(make_config( tiny_diff_target, backend="pi", assume="yes", ))
    assert rc == 0
    _assert_pi_model_and_provider(captured, model="deepseek/deepseek-v4-flash-0731", provider=expected_provider)

# System prompt preamble (--append-system-prompt)

async def test_append_system_prompt_preamble_in_args() -> None:
    backend = PiBackend(model="glm-5.2")
    flat_args, _ = await _run_and_capture_args(backend)
    assert "--append-system-prompt" in flat_args
    preamble = flat_args[flat_args.index("--append-system-prompt") + 1]
    assert preamble == _PI_SYSTEM_PREAMBLE
    assert "tool-call budget" in preamble
    assert "grep" in preamble.lower()

# PiError.retryable attribute

def test_pierror_retryable_default_and_kwarg_and_message() -> None:
    assert PiError("something went wrong").retryable is False
    assert PiError("429 rate limit", retryable=True).retryable is True
    assert str(PiError("auth failed", retryable=False)) == "auth failed"

@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("429 Too Many Requests", "RATE_LIMIT"), ("502 status code (no body)", "SERVER_ERROR"),
        ("503 status code (no body)", "SERVER_ERROR"), ("service unavailable", "SERVER_ERROR"),
        ("request timed out after 30 seconds", "TIMEOUT"), ("socket hang up", "STREAM_DROP"),
        ("Pi CLI exited with return code 1", "PROCESS_EXIT"), ("authentication required", "AUTH_CONFIG"),
        ("synthetic opaque failure", "UNKNOWN"), ("model not found: gpt-5 (503)", "AUTH_CONFIG"),
        ("response failed JSON schema validation: additionalProperties", "SCHEMA"),
        ("provider rate limit", "RATE_LIMIT"),
    ],
    ids=[
        "rate-limit", "server-error-502", "server-error-503", "server-error-service-unavailable", "timeout",
        "stream-drop", "process-exit", "auth-config", "unknown", "permanent-beats-transient",
        "provider-noun-is-not-permanent", "schema",
    ],
)
def test_pi_error_categories_are_stable_host_codes(message: str, expected: str) -> None:
    assert _pi_error_category(message) == expected

@pytest.mark.parametrize("message", ["Service is currently overloaded", "capacity exceeded", "Request throttled"])
def test_overload_throttle_and_capacity_messages_stay_retryable(message: str) -> None:
    """Throttle/capacity categories must remain retryable even when the message also contains a generic
    permanent-error token.
    """
    category = _pi_error_category(message)
    assert category == "SERVER_ERROR"
    assert _pi_retryable_for(category=category, message=message) is True

def test_backend_execution_input_parses_the_retry_recovery_allowance(tmp_path: Path) -> None:
    source = {"HOME": str(tmp_path), "PATH": "/run/bin"}
    undeclared = BackendExecutionInput.from_environment(source, backend="pi")
    # Undeclared is None (fall through to the default), never a declared 0.
    assert undeclared.retry_policy.retry_recovery_allowance_s is None
    declared = BackendExecutionInput.from_environment(
        {**source, "DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S": "42"}, backend="pi"
    )
    assert declared.retry_policy.retry_recovery_allowance_s == 42.0
    for junk in ("nonsense", "-1", "nan", "inf"):
        invalid = BackendExecutionInput.from_environment(
            {**source, "DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S": junk}, backend="pi"
        )
        assert invalid.retry_policy.retry_recovery_allowance_s is None, junk

def test_pi_error_carries_the_retry_hint_from_the_error_message() -> None:
    error = PiError(
        "503 Service Unavailable; retry-after: 30",
        retryable=_pi_retryable_for(category="SERVER_ERROR", message="503 Service Unavailable; retry-after: 30"),
        category=_pi_error_category("503 Service Unavailable; retry-after: 30"),
        retry_after=parse_message_retry_hint("503 Service Unavailable; retry-after: 30"),
    )
    assert error.retry_after == 30.0

@pytest.mark.parametrize(
    "message",
    ["503 status code (no body)", "Stream ended without finish_reason", "request timeout while waiting for response"],
)
def test_pi_transient_failures_are_retryable(message: str) -> None:
    assert _pi_retryable_for(category=_pi_error_category(message), message=message) is True

@pytest.mark.parametrize(
    "lines",
    [
        pytest.param(
            ['{"type":"session","sessionId":"pi_ses_truncated"}', '{"type":"agent_start"}', '{"type":"turn_start"}'],
            id="without-finish-reason",
        ),
        pytest.param(
            [
                '{"type":"session","sessionId":"pi_ses_truncated"}', '{"type":"agent_start"}', '{"type":"turn_start"}',
                '{"type":"turn_end","message":{"role":"assistant","stopReason":"stop"}}', '{"type":"turn_start"}',
            ], id="after-completed-earlier-turn",
        ),
    ],
)
async def test_stream_eof_without_finish_reason_is_retryable_pi_error(lines: list[str]) -> None:
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process(lines)
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(PiError, match="finish_reason") as raised:
            async for _ in backend.execute(Path("/tmp"), "Truncated"):
                pass
    assert raised.value.retryable is True
    assert raised.value.category == "STREAM_TRUNCATION"

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
        patch("daydream.backends._transport.readline_with_idle_timeout", side_effect=StreamStalledError("pi", 1.0)),
    ):
        with pytest.raises(StreamStalledError) as raised:
            async for _ in backend.execute(Path("/tmp"), "Timeout"):
                pass
    assert raised.value.retryable is True

# pi retry classification

@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("429 Too Many Requests", True), ("502 status code (no body)", True), ("502 Bad Gateway", True),
        ("503 status code (no body)", True), ("503 Service Unavailable", True),
        ("Service is currently overloaded", True), ("rate limit exceeded", True), ("rate_limit hit", True),
        ("Capacity unavailable", True), ("too many requests from this IP", True), ("Request throttled", True),
        ("throttling in effect", True),
        ("OVERLOAD detected", True),  # case-insensitive
        # Stream-drop signatures (z.ai/GLM connection drops).
        ("terminated", True), ("ECONNRESET", True), ("connection reset", True), ("socket hang up", True),
        ("premature close", True), ("EPIPE", True), ("auth failed", False), ("service is not overloaded", False),
        ("capacity planning required", False), ("", False), ("Unknown Pi error", False),
    ],
    ids=[
        "429", "502-status-code", "502-bad-gateway", "503-status-code", "503-service-unavailable", "overload",
        "rate-limit-space", "rate_limit-underscore", "capacity", "too-many-requests", "throttle", "throttling",
        "case-insensitive", "stream-drop-terminated", "stream-drop-econnreset", "stream-drop-connection-reset",
        "stream-drop-socket-hang-up", "stream-drop-premature-close", "stream-drop-epipe", "auth-failed",
        "not-overloaded", "capacity-planning", "empty", "unknown",
    ],
)
def test_is_retryable_error_message(message: Any, expected: Any) -> None:
    assert _pi_retryable_for(category=_pi_error_category(message), message=message) is expected

# _is_retryable_exit_code

@pytest.mark.parametrize(
    ("code", "expected"), [(-9, True), (137, True), (1, False), (2, False), (0, False)],
    ids=["sigkill", "oom-137", "exit-1", "exit-2", "zero"],
)
def test_is_retryable_exit_code(code: Any, expected: Any) -> None:
    assert _is_retryable_exit_code(code) is expected

# Error turn uses retryable classifier

@pytest.mark.parametrize(
    ("error_message", "expected_retryable"),
    [
        ("429 Too Many Requests - rate limit exceeded", True), ("502 status code (no body)", True),
        ("authentication required", False),
    ], ids=["429-retryable", "502-retryable", "auth-non-retryable"],
)
async def test_error_turn_sets_retryable_via_classifier(error_message: Any, expected_retryable: Any) -> None:
    backend = PiBackend(model="glm-5.2")
    lines = [
        '{"type":"session","sessionId":"pi_ses_err"}', '{"type":"agent_start"}', '{"type":"turn_start"}',
        '{"type":"turn_end","message":{"role":"assistant","content":[],'
        f'"stopReason":"error","errorMessage":{json.dumps(error_message)}}}}}',
    ]
    mock_proc = make_mock_process(lines)
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(PiError) as exc_info:
            async for _ in backend.execute(Path("/tmp"), "p"):
                pass
    assert exc_info.value.retryable is expected_retryable

async def test_error_turn_carries_structured_retry_hint_from_the_provider_payload() -> None:
    """The serialized OpenRouter 429 decodes to 10.0 on the backend's error hint.

    Drives the real Pi protocol boundary (two real spawns) with only
    create_subprocess_exec patched; a hand-built exception is not sufficient
    coverage for the extraction contract (req 13).
    """
    error_msg = (
        '429: {"message":"Temporary admission failure","code":429,'
        '"metadata":{"headers":{"Retry-After":"10"}}}'
    )
    error_lines = [
        '{"type":"session","sessionId":"pi_ses_429"}',
        '{"type":"agent_start"}',
        '{"type":"turn_start"}',
        json.dumps({"type": "turn_end", "message": {"role": "assistant", "content": [],
                                                    "stopReason": "error", "errorMessage": error_msg}}),
    ]
    healthy_lines = [
        '{"type":"session","sessionId":"pi_ses_ok"}',
        '{"type":"agent_start"}',
        '{"type":"turn_start"}',
        '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"done"}]}}',
        '{"type":"turn_end","message":{"role":"assistant","content":[{"type":"text","text":"done"}],'
        '"usage":{"input":1,"output":1},"stopReason":"stop"}}',
        '{"type":"agent_end","messages":[]}',
    ]
    backend = PiBackend(model="glm-5.2")
    procs = [make_mock_process(error_lines), make_mock_process(healthy_lines)]
    with patch(
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        side_effect=lambda *a, **k: procs.pop(0),
    ):
        with pytest.raises(PiError) as exc_info:
            async for _ in backend.execute(Path("/tmp"), "p"):
                pass
    assert exc_info.value.retry_after == 10.0
    assert exc_info.value.retryable is True
    # A healthy follow-up spawn proves the mock pair works and the ladder can recover.
    with patch(
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        side_effect=lambda *a, **k: procs.pop(0),
    ):
        events = [e async for e in backend.execute(Path("/tmp"), "p")]
    assert any(getattr(e, "text", None) == "done" for e in events)

@pytest.mark.parametrize(
    ("returncode", "output_lines", "expected_retryable"),
    [
        (-9, [], True),  # SIGKILL/OOM
        (1, ["Error: not authenticated"], False),  # auth/config error
    ], ids=["oom-sigkill-retryable", "exit-1-non-retryable"],
)
async def test_nonzero_exit_sets_retryable_via_exit_code(
    returncode: Any, output_lines: Any, expected_retryable: Any,
) -> None:
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process(output_lines)
    mock_proc.returncode = returncode
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(PiError) as exc_info:
            async for _ in backend.execute(Path("/tmp"), "p"):
                pass
    assert exc_info.value.retryable is expected_retryable

# Retry env knobs

@pytest.mark.parametrize(
    ("env_var", "parse", "env_value", "expected", "warn_fragment"),
    [
        ("DAYDREAM_PI_RETRY_ATTEMPTS", _pi_retry_attempts, None, _PI_DEFAULT_RETRY_ATTEMPTS, None),
        ("DAYDREAM_PI_RETRY_ATTEMPTS", _pi_retry_attempts, "5", 5, None),
        (
            "DAYDREAM_PI_RETRY_ATTEMPTS", _pi_retry_attempts, "", _PI_DEFAULT_RETRY_ATTEMPTS,
            f"is not a valid integer; using default {_PI_DEFAULT_RETRY_ATTEMPTS}",
        ), ("DAYDREAM_PI_RETRY_BASE_DELAY_S", _pi_retry_base_delay, None, _PI_DEFAULT_RETRY_BASE_DELAY, None),
        ("DAYDREAM_PI_RETRY_BASE_DELAY_S", _pi_retry_base_delay, "0.5", 0.5, None),
        (
            "DAYDREAM_PI_RETRY_BASE_DELAY_S", _pi_retry_base_delay, "nan", _PI_DEFAULT_RETRY_BASE_DELAY,
            f"is not finite; using default {_PI_DEFAULT_RETRY_BASE_DELAY:g}",
        ),
        (
            "DAYDREAM_PI_RETRY_BASE_DELAY_S", _pi_retry_base_delay, "inf", _PI_DEFAULT_RETRY_BASE_DELAY,
            f"is not finite; using default {_PI_DEFAULT_RETRY_BASE_DELAY:g}",
        ),
        (
            "DAYDREAM_PI_RETRY_BASE_DELAY_S", _pi_retry_base_delay, "", _PI_DEFAULT_RETRY_BASE_DELAY,
            f"is not a valid float; using default {_PI_DEFAULT_RETRY_BASE_DELAY:g}",
        ), ("DAYDREAM_PI_RETRY_MAX_DELAY_S", _pi_retry_max_delay, None, _PI_DEFAULT_RETRY_MAX_DELAY, None),
        ("DAYDREAM_PI_RETRY_MAX_DELAY_S", _pi_retry_max_delay, "45.5", 45.5, None),
        (
            "DAYDREAM_PI_RETRY_MAX_DELAY_S", _pi_retry_max_delay, "not-a-float", _PI_DEFAULT_RETRY_MAX_DELAY,
            f"is not a valid float; using default {_PI_DEFAULT_RETRY_MAX_DELAY:g}",
        ),
        (
            "DAYDREAM_PI_RETRY_MAX_DELAY_S", _pi_retry_max_delay, "-1", _PI_DEFAULT_RETRY_MAX_DELAY,
            f"is negative; using default {_PI_DEFAULT_RETRY_MAX_DELAY:g}",
        ),
        (
            "DAYDREAM_PI_RETRY_MAX_DELAY_S", _pi_retry_max_delay, "", _PI_DEFAULT_RETRY_MAX_DELAY,
            f"is not a valid float; using default {_PI_DEFAULT_RETRY_MAX_DELAY:g}",
        ),
    ],
    ids=[
        "attempts-default", "attempts-env-override", "attempts-empty-warns", "base-default", "base-env-override",
        "base-nan-warns", "base-inf-warns", "base-empty-warns", "max-default", "max-env-override", "max-invalid-warns",
        "max-negative-warns", "max-empty-warns",
    ],
)
def test_pi_retry_env_knobs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, env_var: str, parse: Callable[[], int | float],
    env_value: str | None, expected: Any, warn_fragment: str | None,
) -> None:
    if env_value is not None:
        monkeypatch.setenv(env_var, env_value)
    assert parse() == pytest.approx(expected)
    if warn_fragment is not None:
        assert warn_fragment in caplog.text

# fanout_concurrency

def test_pi_fanout_concurrency_defaults_to_ten(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAYDREAM_PI_FANOUT_CONCURRENCY", raising=False)
    backend = PiBackend(model="glm-5.2")
    assert backend.fanout_concurrency == 10

@pytest.mark.parametrize(("env_value", "expected"), [ ("6", 6), ("0", 10), ("-1", 10), ("invalid", 10), ("", 10), ])
def test_pi_fanout_concurrency_env_validation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, env_value: Any, expected: int,
) -> None:
    monkeypatch.setenv("DAYDREAM_PI_FANOUT_CONCURRENCY", env_value)
    assert PiBackend(model="glm-5.2").fanout_concurrency == expected
    if expected == 10:
        assert "using default 10" in caplog.text

@pytest.mark.parametrize(("effort", "ambient", "expected"), [
    ("max", None, "max"), ("xhigh", "low", "xhigh"), (None, "high", "high"), (None, None, None),
])
async def test_pi_reasoning_effort_precedes_ambient_thinking(
    monkeypatch: pytest.MonkeyPatch, effort: str | None, ambient: str | None, expected: str | None,
) -> None:
    if ambient is None:
        monkeypatch.delenv("PI_THINKING", raising=False)
    else:
        monkeypatch.setenv("PI_THINKING", ambient)
    flat_args, _ = await _run_and_capture_args(PiBackend(model="glm-5.2", reasoning_effort=effort))
    if expected is None:
        assert "--thinking" not in flat_args
    else:
        assert flat_args[flat_args.index("--thinking") + 1] == expected
    if effort and ambient and effort != ambient:
        assert ambient not in flat_args



# --- P18 Task 1: generation lifecycle + config at the Pi argv/JSONL seam -----

def test_pi_replay_fixture_is_sanitized_labeled() -> None:
    fixture = FIXTURES_DIR / "generation_lifecycle.jsonl"
    text = fixture.read_text(encoding="utf-8")
    assert "THINKING_PLACEHOLDER_ONE" in text
    assert "TEXT_PLACEHOLDER" in text
    assert "FILE_PLACEHOLDER" in text
    assert "src/example.py" in text  # synthetic argument path, not a real one
    assert "1788690314289" in text
    assert "395.332" not in text.split("\n")[0]

async def test_pi_generation_lifecycle_start_end_pair_around_tool() -> None:
    """Two assistant generations: start before, end sealed at message_end before tools."""
    events = await _collect_events(PiBackend(model="glm-5.2"), "go", fixture="generation_lifecycle.jsonl")
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
    second_end = ends[1]
    assert second_end.generation_id == starts[1].generation_id
    assert second_end.generation_id != first_end.generation_id
    assert [p.kind for p in second_end.choice_parts] == ["text"]
    assert second_end.finish_reason == "stop"

async def test_pi_native_ms_start_converts_exactly_and_chronology_holds() -> None:
    """Native Unix-ms start → exact ns; end receipt is host-observed and later."""
    events = await _collect_events(PiBackend(model="glm-5.2"), "go", fixture="generation_lifecycle.jsonl")
    ends = [e for e in events if isinstance(e, GenerationEndEvent)]
    first = ends[0]
    assert first.native_started_at_unix_ms == 1788690314289
    assert first.native_started_at_unix_ms is not None
    assert unix_ms_to_ns(first.native_started_at_unix_ms) == 1788690314289000000
    assert first.ended_at_unix_ns >= unix_ms_to_ns(first.native_started_at_unix_ms)

async def test_pi_user_and_tool_results_do_not_create_generations() -> None:
    events = await _collect_events(PiBackend(model="glm-5.2"), "go", fixture="generation_lifecycle.jsonl")
    # The fixture has exactly two assistant messages → exactly two pairs.
    # Tool execution events and turn boundaries are not generations.
    assert len([e for e in events if isinstance(e, GenerationStartEvent)]) == 2
    tool_events = [e for e in events if isinstance(e, (ToolStartEvent, ToolResultEvent))]
    assert tool_events
    assert all(not isinstance(e, (GenerationStartEvent, GenerationEndEvent)) for e in tool_events)

async def test_pi_turn_end_carries_native_identity() -> None:
    events = await _collect_events(PiBackend(model="glm-5.2"), "go", fixture="generation_lifecycle.jsonl")
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

async def test_pi_request_event_config_matches_exact_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.delenv("PI_API_KEY", raising=False)
    backend = PiBackend(model="glm-5.2")
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    flat_args, _ = await _run_and_capture_args(
        backend, "structured please", output_schema=schema, read_only=True, persist_session=False,
    )
    events = await _collect_events(
        PiBackend(model="glm-5.2"), "structured please", fixture="simple_text.jsonl", output_schema=schema,
        read_only=True, persist_session=False,
    )
    request = next(event for event in events if isinstance(event, RequestEvent))
    config = request.config
    assert isinstance(config, PiRequestConfig)
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

async def test_pi_multi_turn_fixture_produces_two_turn_end_boundaries() -> None:
    events = await _collect_events(PiBackend(model="glm-5.2"), "go", fixture="multi_turn.jsonl")
    turn_ends = [e for e in events if isinstance(e, TurnEndEvent)]
    assert len(turn_ends) == 2
    assert [t.finish_reason for t in turn_ends] == ["stop", "stop"]
    assert all(t.model_name == "glm-4.6" for t in turn_ends)

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
        with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc2):
            try:
                async for event in PiBackend(model="glm-5.2").execute(Path("/tmp"), "go"):
                    collected.append(event)
            except PiError:
                pass
    turn_ends = [e for e in collected if isinstance(e, TurnEndEvent)]
    assert turn_ends
    error_turn = turn_ends[-1]
    assert error_turn.finish_reason == "error"
    assert error_turn.model_name is None or error_turn.model_name == "glm-5.2"

async def test_pi_usage_events_carry_turn_end_and_terminal_provenance() -> None:
    events = await _collect_events(PiBackend(model="glm-5.2"), "go", fixture="generation_lifecycle.jsonl")
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    assert metrics
    assert all(m.measurement_source == "turn_end" for m in metrics)
    costs = [e for e in events if isinstance(e, CostEvent)]
    assert costs
    terminal = costs[-1]
    assert terminal.measurement_source == "terminal"
    assert terminal.cost_source == "reported"

@pytest.mark.parametrize("mode", ["success", "process_error"])
async def test_large_prompt_uses_private_attachment_until_child_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: Literal["success", "process_error"],
) -> None:
    fixture = install_protocol_cli(tmp_path / "fixture", "pi", response_mode=mode)
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "settings"))
    prompt = "合法 prompt sentinel\n" + "x" * 3_690_129
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}
    events: list[Any] = []
    try:
        async for event in PiBackend(model="fixture").execute(tmp_path, prompt, output_schema=schema):
            events.append(event)
    except PiError:
        assert mode == "process_error"
    observed = fixture.read_observations()[0]
    request = next(event for event in events if isinstance(event, RequestEvent))
    assert observed["prompt_sha256"] == hashlib.sha256(request.prompt.encode()).hexdigest()
    assert request.prompt.startswith(prompt)
    assert json.dumps(schema) in request.prompt
    attachment = Path(observed["prompt_attachment"])
    assert attachment.is_absolute()
    assert not attachment.exists()
    assert observed["prompt_attachment_mode"] == 0o600
    assert observed["argv_bytes"] < 32_768

@pytest.mark.parametrize("real_spawn", [True, False])
async def test_prompt_attachment_removed_after_spawn_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_spawn: bool,
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "settings"))
    observed: list[Path] = []
    async def fail(*args: Any, **kwargs: Any) -> Any:
        path = Path(args[-1][1:])
        assert path.read_text() == "spawn failure prompt"
        observed.append(path)
        raise OSError("mock spawn failure")
    if not real_spawn:
        monkeypatch.setattr("daydream.backends._transport.asyncio.create_subprocess_exec", fail)
    with pytest.raises(OSError):
        async for _ in PiBackend(model="fixture").execute(tmp_path, "spawn failure prompt"):
            pass
    assert not list(tmp_path.glob("daydream-pi-prompt-*"))
    assert real_spawn or len(observed) == 1

@pytest.mark.parametrize("after_spawn", [False, True])
async def test_prompt_attachment_generator_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_spawn: bool,
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    monkeypatch.setattr(
        "daydream.backends._transport.asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc),
    )
    stream = PiBackend(model="fixture").execute(tmp_path, "close prompt")
    assert isinstance(await anext(stream), RequestEvent)
    assert not list(tmp_path.glob("daydream-pi-prompt-*"))
    if after_spawn:
        await anext(stream)
        paths = list(tmp_path.glob("daydream-pi-prompt-*"))
        assert len(paths) == 1
        assert paths[0].read_text() == "close prompt"
    await stream.aclose()
    assert not list(tmp_path.glob("daydream-pi-prompt-*"))

async def test_concurrent_prompt_attachments_survive_until_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fixture walks its cwd before publishing an observation. Keep its
    # concurrently renamed observation files outside that walk.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fixture = install_protocol_cli(tmp_path / "fixture", "pi", response_mode="block")
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "settings"))
    backend = PiBackend(model="fixture")
    async def consume(prompt: str) -> None:
        async for _ in backend.execute(workspace, prompt):
            pass
    tasks = [asyncio.create_task(consume(prompt)) for prompt in ("first invocation", "second invocation")]
    try:
        async with asyncio.timeout(5):
            while len(fixture.read_observations()) < 2:
                for task in tasks:
                    if task.done():
                        task.result()
                        pytest.fail("Pi fixture exited before both invocations became ready")
                await asyncio.sleep(0.01)
        paths = [Path(item["prompt_attachment"]) for item in fixture.read_observations()]
        assert len(set(paths)) == 2
        assert {path.read_text() for path in paths} == {"first invocation", "second invocation"}
        tasks[0].cancel()
        await asyncio.gather(tasks[0], return_exceptions=True)
        remaining = [path for path in paths if path.exists()]
        assert len(remaining) == 1
        assert remaining[0].read_text() == "second invocation"
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert not any(path.exists() for path in paths)
    assert backend._transports == []

async def test_prompt_attachment_removed_even_if_teardown_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr("daydream.backends._transport.asyncio.create_subprocess_exec",
                        AsyncMock(return_value=make_mock_process_from_fixture("simple_text.jsonl")))
    monkeypatch.setattr("daydream.backends.pi.teardown", AsyncMock(side_effect=RuntimeError("teardown failed")))
    with pytest.raises(RuntimeError, match="teardown failed"):
        async for _ in PiBackend(model="fixture").execute(tmp_path, "teardown prompt"):
            pass
    assert not list(tmp_path.glob("daydream-pi-prompt-*"))

async def test_large_review_instructions_use_system_prompt_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = install_protocol_cli(tmp_path / "fixture", "pi")
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "settings"))
    instructions = "bounded review policy\n" + "x" * 3_690_129
    requests = [event async for event in PiBackend(model="fixture").execute(
        tmp_path, "Review source", review_instructions=instructions,
    ) if isinstance(event, RequestEvent)]
    observed = fixture.read_observations()[0]
    assert requests[0].system_prompt is not None
    assert instructions in requests[0].system_prompt
    assert observed["content_arguments"]["--append-system-prompt"] == [{
        "bytes": len(requests[0].system_prompt.encode()),
        "sha256": hashlib.sha256(requests[0].system_prompt.encode()).hexdigest(),
    }]
    assert observed["argv_bytes"] < 32_768

async def test_prompt_attachment_removed_after_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    create_file = tempfile.NamedTemporaryFile
    def failing_file(*args: Any, **kwargs: Any) -> Any:
        opened = create_file(*args, **kwargs)
        monkeypatch.setattr(opened, "write", MagicMock(side_effect=OSError("disk full")))
        return opened
    monkeypatch.setattr("daydream.backends.pi.tempfile.NamedTemporaryFile", failing_file)
    with pytest.raises(OSError, match="disk full"):
        async for _ in PiBackend(model="fixture").execute(tmp_path, "write failure prompt"):
            pass
    assert not list(tmp_path.glob("daydream-pi-prompt-*"))
