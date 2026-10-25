"""Drive PiBackend.execute through canned JSONL and real protocol CLI fixtures."""

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import httpx
import pytest

from daydream.agent import run_agent
from daydream.atif import validate
from daydream.backends import (
    BackendExecutionInput,
    CostEvent,
    DiagnosticEvent,
    MetricsEvent,
    PiRequestConfig,
    RequestEvent,
    ResultEvent,
    RetryPolicy,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
    create_backend,
)
from daydream.backends._subprocess import StreamStalledError
from daydream.backends.pi import (
    _PI_STDOUT_LIMIT_BYTES,
    _PI_SYSTEM_PREAMBLE,
    PiBackend,
    PiError,
)
from daydream.config_file import DaydreamFileConfig
from daydream.phases.schemas import PER_STACK_RECORD_SCHEMA
from daydream.runner import run
from daydream.trajectory import DaydreamPhase
from tests.conftest import ExtDir
from tests.harness.fake_cli_process import (
    LimitAwareStdout,
    assert_concurrent_streams_isolated,
    install_fake_cli_process,
)
from tests.harness.fake_clock import FakeClock
from tests.harness.pi_replay import FIXTURES_DIR, make_mock_process, make_mock_process_from_fixture
from tests.harness.process_replay import replay_process
from tests.harness.protocol_cli import install_protocol_cli
from tests.harness.protocol_cli_assertions import assert_protocol_cli_invariants, make_cancel_probe
from tests.harness.review_profile import independent_alternatives_profile
from tests.harness.stub_backend import (
    review_stage_state,
    silence as _silence,
    stage_result,
)
from tests.harness.trajectory import make_recorder


def test_ordinary_pi_fresh_process_does_not_load_host_admission_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary Pi runs without a host endpoint; all-extras imports can hide unwanted eager endpoint loading."""
    fixture = install_protocol_cli(tmp_path / "external", "pi")
    target = tmp_path / "repo"
    target.mkdir()
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PI_PROVIDER", "nous")
    monkeypatch.delenv("PI_API_KEY", raising=False)
    script = '''
import asyncio
import sys
from pathlib import Path
# Forbid admission endpoint loading in a fresh ordinary invocation.
assert not any(name == "daydream" or name.startswith("daydream.") for name in sys.modules)
sys.modules["daydream.host_tools_server"] = None
from daydream.backends.pi import PiBackend
from daydream.backends import ResultEvent, TextEvent

async def main():
    events = [event async for event in PiBackend(model="fixture-model").execute(
        Path(sys.argv[1]), "Ordinary Pi call", persist_session=False)]
    assert any(isinstance(event, ResultEvent) for event in events)
    assert [event.text for event in events if isinstance(event, TextEvent)] == ["CURRENT_REASONING_CANARY"]

asyncio.run(main())
assert sys.modules["daydream.host_tools_server"] is None
'''
    result = subprocess.run([sys.executable, "-c", script, str(target)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    observation, = fixture.read_observations()
    assert observation["process_outcome"] == "success"


@pytest.mark.parametrize("kind", ["text", "thinking", "toolcall", "tool", "diagnostic"])
async def test_interrupted_pi_updates_retain_only_admitted_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    """The installed delta-only schema survives interruption without turning diagnostics into progress.

    Runner journeys own end-to-end export proof; this seam protects Pi frame admission,
    including missing outer message, empty/malformed deltas, and tool updates.
    """
    frames: list[dict[str, Any]] = [
        {"type": "session", "id": "private-session-canary"},
        {"type": "message_start", "message": {"role": "assistant"}},
        {"type": "message_update", "assistantMessageEvent": {"type": "heartbeat", "contentIndex": 0,
                                                             "delta": "private-canary"}},
        {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": ""}},
        {"type": "message_update", "assistantMessageEvent": {"type": "thinking_delta", "contentIndex": True,
                                                             "delta": "private-canary"}},
        {"type": "message_update", "assistantMessageEvent": {"type": "toolcall_delta", "contentIndex": -1,
                                                             "delta": "private-canary"}},
        {"type": "tool_execution_update", "partialResult": {"content": [{"text": "private-canary"}]}},
    ]
    if kind == "tool":
        frames.extend([
            {"type": "message_end", "message": {"role": "assistant", "stopReason": "toolUse", "content": [
                {"type": "toolCall", "id": "active-tool", "name": "read", "arguments": {}},
            ]}},
            {"type": "tool_execution_start", "toolCallId": "active-tool", "toolName": "read", "args": {}},
            {"type": "tool_execution_update", "partialResult": "invalid"},
            {"type": "tool_execution_update", "partialResult": {"content": [{"text": "private-canary"}]}},
        ])
    elif kind != "diagnostic":
        frames.append({"type": "message_update", "assistantMessageEvent": {
            "type": f"{kind}_delta", "contentIndex": 0, "delta": "private-canary",
        }})
    lines = ["", "private stderr-canary", *[json.dumps(frame) for frame in frames]]
    spawner = install_fake_cli_process(monkeypatch, "pi", lines=lines, hang=True)
    monkeypatch.setenv("DAYDREAM_STREAM_IDLE_TIMEOUT_S", "0")
    clock = FakeClock().install(monkeypatch)
    wall_scopes: list[anyio.CancelScope] = []

    def staged_wall_scope(delay: float, *, shield: bool = False) -> anyio.CancelScope:
        scope = anyio.move_on_after(delay if shield else None, shield=shield)
        if not shield:
            wall_scopes.append(scope)
        return scope

    # This case owns admission of every scripted frame before interruption.
    # Stage only the agent's wall scope until its actual reader blocks for more
    # bytes; setup/scheduling latency must not truncate the admission scenario.
    # Cleanup scopes retain their ordinary real-time bounds.
    monkeypatch.setattr("daydream.agent.anyio", SimpleNamespace(move_on_after=staged_wall_scope, sleep=anyio.sleep))
    recorder = make_recorder(tmp_path, backend_name="pi")
    async with recorder:
        async with asyncio.TaskGroup() as tasks:
            turn = tasks.create_task(run_agent(
                PiBackend(model="test-model"), tmp_path, "review", phase=DaydreamPhase.REVIEW, wall_budget_s=0.05,
            ))
            with anyio.fail_after(5):
                await spawner.wait_for_read()
                assert len(wall_scopes) == 1
                clock.advance(0.05)
                wall_scopes[0].deadline = anyio.current_time()
                output, _, reason = await turn
    assert reason == "wall_budget_exceeded"
    assert output == ""
    trajectory = json.loads(recorder.path.read_text())
    evidence = trajectory["extra"]["subtrajectories"][0]["backend_lifecycle"]
    assert evidence["state"] == ("native_event" if kind == "tool" else "generating")
    assert evidence["generation_count"] == 1 and evidence["generation_open"] is (kind != "tool")
    assert evidence["idle_detection_enabled"] is False
    assert evidence["response_idle_timeout_s"] == 0.0 and evidence["tool_idle_timeout_s"] == 0.0
    assert evidence["raw_line_count"] == len(lines)
    for field, admitted_kind in (("text_delta_count", "text"), ("thinking_delta_count", "thinking"),
                                 ("toolcall_delta_count", "toolcall"), ("tool_progress_count", "tool")):
        assert evidence[field] == int(kind == admitted_kind)
    assert (evidence["last_progress_s"] is None) == (kind == "diagnostic")
    assert evidence["active_tool_count"] == int(kind == "tool")
    assert "private" not in json.dumps(evidence)
    assert spawner.procs[0].reaped and spawner.procs[0].returncode == -15


if TYPE_CHECKING:
    from daydream.run_config import RunConfig

MakeConfig = Callable[..., "RunConfig"]


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


async def test_structured_output() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("structured_output.jsonl")
    events, mock_exec = await replay_process(backend, mock_proc, Path("/tmp"), "Parse",
                                             output_schema=schema, finalization=True)
    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output is None
    assert json.loads(next(event.text for event in events if isinstance(event, TextEvent))) == {
        "issues": [{"id": 1, "description": "Fix type hints", "file": "app.py", "line": 5}]}
    # Request records preserve the logical prompt; argv carries only an attachment.
    flat_args = list(mock_exec.call_args.args)
    positional = flat_args[-1]
    assert positional.startswith("@/")
    assert not Path(positional[1:]).exists()
    request = next(e for e in events if isinstance(e, RequestEvent))
    assert "JSON schema" in request.prompt
    assert json.dumps(schema) in request.prompt


async def test_text_transport_forwards_incidental_json_and_empty_result() -> None:
    """The host receives the entire final text, including incidental JSON (issue #1445)."""
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("issue1445_empty_results.jsonl")
    events, _ = await replay_process(backend, mock_proc, Path("/tmp"), "Parse",
                                     output_schema=PER_STACK_RECORD_SCHEMA, finalization=True)
    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output is None
    text = next(event.text for event in events if isinstance(event, TextEvent))
    assert '["module", "moduleVersion", "surface"]' in text and '{"issues": []}' in text


@pytest.mark.parametrize('fault', ['none', 'domain-truncation', 'failed-last', 'mixed-read', 'mixed-read-first',
                                  'null-details', 'oversized-details', 'wrong-id', 'unsettled', 'late-exit'])
async def test_native_output_uses_finalized_transcript_order_and_requires_complete_settlement(fault: str) -> None:
    """Daydream selects settled successful submissions independently of execution completion order."""
    calls: list[dict[str, Any]] = [
        {'type': 'toolCall', 'id': call_id, 'name': 'structured_output', 'arguments': {'answer': value}}
             for call_id, value in [('first', 'first'), ('last', 'last')]]
    if fault == 'domain-truncation':
        calls[-1]['arguments']['truncation'] = {'truncated': True, 'firstLineExceedsLimit': True}
    rows: list[dict[str, Any]] = [{'type': 'agent_start'}, {'type': 'turn_start'},
        {'type': 'message_end', 'message': {'role': 'assistant', 'content': calls, 'stopReason': 'toolUse'}},
        *[{'type': 'tool_execution_start', 'toolCallId': call['id'], 'toolName': 'structured_output',
           'args': call['arguments']} for call in calls],
        # Parallel completion arrives in the reverse order; finalized messages remain transcript ordered.
        *[{'type': 'tool_execution_end', 'toolCallId': call['id'], 'toolName': 'structured_output',
           'isError': False, 'result': {'content': [{'type': 'text', 'text': 'Submitted.'}],
                                      'details': call['arguments']}} for call in reversed(calls)],
        *[{'type': 'message_end', 'message': {'role': 'toolResult', 'toolCallId': call['id'],
           'toolName': 'structured_output', 'isError': False,
           'content': [{'type': 'text', 'text': 'Submitted.'}], 'details': call['arguments']}} for call in calls],
        {'type': 'turn_end', 'message': {'role': 'assistant', 'content': calls, 'stopReason': 'toolUse'}},
        {'type': 'agent_end', 'messages': []}, {'type': 'agent_settled'}]
    if fault == 'failed-last':
        rows[5]['isError'] = True
        rows[-4]['message']['isError'] = True
    elif fault in {'mixed-read', 'mixed-read-first'}:
        read_call = {'type': 'toolCall', 'id': 'read', 'name': 'read', 'arguments': {'path': 'api.py'}}
        calls.insert(0 if fault == 'mixed-read-first' else len(calls), read_call)
        rows[3:3] = [{'type': 'tool_execution_start', 'toolCallId': 'read', 'toolName': 'read',
                     'args': read_call['arguments']},
                    {'type': 'tool_execution_end', 'toolCallId': 'read', 'toolName': 'read',
                     'isError': False, 'result': {'content': [{'type': 'text', 'text': 'VALUE = 1'}]}}]
    elif fault == 'null-details':
        rows[-4]['message']['details'] = None
    elif fault == 'oversized-details':
        rows[-4]['message']['details'] = {'answer': 'x' * (129 * 1024)}
    elif fault == 'wrong-id':
        rows[-4]['message']['toolCallId'] = 'unassociated'
    elif fault == 'unsettled':
        rows.pop()
    process = make_mock_process([json.dumps(row) for row in rows])
    if fault == 'late-exit':
        process.wait.return_value = process.returncode = 7
    schema = {'type': 'object', 'properties': {'answer': {'type': 'string'}, 'truncation': {
        'type': 'object', 'properties': {'truncated': {'type': 'boolean'},
                                          'firstLineExceedsLimit': {'type': 'boolean'}}}}, 'required': ['answer']}
    if fault in {'none', 'domain-truncation', 'failed-last', 'mixed-read', 'mixed-read-first'}:
        events, _ = await replay_process(PiBackend(model='fixture'), process, Path('/tmp'), 'Answer',
                                         output_schema=schema)
        result = next(event for event in events if isinstance(event, ResultEvent))
        expected = {'answer': 'first' if fault == 'failed-last' else 'last',
                    **({'truncation': {'truncated': True, 'firstLineExceedsLimit': True}}
                                       if fault == 'domain-truncation' else {})}
        assert result.structured_output == expected
        assert all(not event.truncated for event in events if isinstance(event, ToolResultEvent))
        assert [event.id for event in events if isinstance(event, ToolStartEvent)] == (
            ['read', 'first', 'last'] if fault in {'mixed-read', 'mixed-read-first'} else ['first', 'last'])
    else:
        with pytest.raises(PiError):
            await replay_process(PiBackend(model='fixture'), process, Path('/tmp'), 'Answer', output_schema=schema)


@pytest.mark.parametrize("bound", [False, True])
async def test_pi_api_key_never_enters_process_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound: bool,
) -> None:
    sentinel = "synthetic-pi-api-key-sentinel"
    monkeypatch.setenv("PI_PROVIDER", "zai")
    monkeypatch.setenv("PI_API_KEY", sentinel)
    monkeypatch.setenv("PI_THINKING", "medium")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi-agent"))
    backend = PiBackend(model="glm-5.2")
    if bound:
        backend.bind_review_context(tmp_path)
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


async def test_cancel_terminates_then_kills(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daydream.backends._subprocess.TERMINATE_GRACE_S", 0.0)
    backend, proc = make_cancel_probe("pi")
    await backend.cancel()
    assert proc.terminate_calls == 1 and proc.kill_calls == 1
    assert proc.returncode == -9 and proc.reaped


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


# Default --provider is nous; PI_PROVIDER overrides (extension-based provider)


async def test_default_model_does_not_override_pi_settings(pi_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    backend = PiBackend(cwd=pi_workspace)
    assert backend.model == "gpt-5.6-luna"
    flat_args, _ = await _run_and_capture_args(backend, cwd=pi_workspace)
    assert "--model" not in flat_args
    assert "--provider" not in flat_args



async def test_bound_pi_context_preserves_native_configuration_and_auth_refresh(
    pi_workspace: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configuration context survives relocation without resolving the native model.

    Ordinary settings tests cannot detect binding the CLI to the captured cwd or
    accepting changed run configuration. Authentication must remain refreshable.
    """
    agent_dir = pi_workspace / "pi-agent"
    agent_dir.mkdir()
    settings = pi_workspace / ".pi/settings.json"
    snapshot = pi_workspace / "snapshot"
    snapshot.mkdir()
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.delenv("PI_THINKING", raising=False)
    backend = PiBackend(cwd=pi_workspace)
    identity = backend.bind_review_context(pi_workspace)
    assert backend.bind_review_context(pi_workspace) == identity
    assert backend.model == "pi-native"
    monkeypatch.setenv("PI_PROVIDER", "other-provider")
    monkeypatch.setenv("NOUS_API_KEY", "refreshed-loopback-credential")
    (agent_dir / "auth.json").write_text('{"refreshed": true}')
    args, process = await _run_and_capture_args(backend, cwd=snapshot)
    assert "--model" not in args and "--provider" not in args
    assert "--thinking" not in args
    assert process.call_args.kwargs["cwd"] == str(pi_workspace)
    assert process.call_args.kwargs["env"]["PI_CODING_AGENT_DIR"] == str(agent_dir)
    assert process.call_args.kwargs["env"]["NOUS_API_KEY"] == "refreshed-loopback-credential"
    settings.write_text(json.dumps({"defaultProvider": "another-run"}))
    with pytest.raises(PiError, match="start a new run"):
        async for _ in backend.execute(snapshot, "Review captured source"):
            pass


async def test_explicit_model_overrides_pi_settings(pi_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    backend = PiBackend(model="custom-model")
    flat_args, _ = await _run_and_capture_args(backend, cwd=pi_workspace)
    assert flat_args[flat_args.index("--model") + 1] == "custom-model"


# Migration guards: GLM-pin and provider/model mismatch warnings


async def test_glm_pin_warning_requires_implicit_provider(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("PI_PROVIDER", "zai")
    monkeypatch.setattr("daydream.backends.pi._warned_migration_mismatches", set())
    with caplog.at_level("WARNING"):
        flat_args, _ = await _run_and_capture_args(PiBackend(model="glm-5.2"))
    assert not any("z.ai-hosted GLM" in record.getMessage() for record in caplog.records)
    assert flat_args[flat_args.index("--model") + 1] == "glm-5.2"
    assert flat_args[flat_args.index("--provider") + 1] == "zai"


async def test_glm_pin_warning_fires_once_across_executes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.setattr("daydream.backends.pi._warned_migration_mismatches", set())
    backend = PiBackend(model="glm-5.2")
    with caplog.at_level("WARNING"):
        first_args, _ = await _run_and_capture_args(backend)
        await _run_and_capture_args(backend)
    assert first_args[first_args.index("--model") + 1] == "glm-5.2"
    assert first_args[first_args.index("--provider") + 1] == "nous"
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


def _capture_pi_subprocess(
    monkeypatch: pytest.MonkeyPatch, captured: list[list[str]],
    response: Callable[[list[str]], MagicMock] | None = None,
) -> None:
    """Replay only native process bytes while keeping required HTTP admission real.

    Runner configuration/recording tests must not imply domain acceptance by
    prewriting successful native receipts. Installed-Pi tests own actual extension
    execution; this external-process seam drives the same callback and returns its
    canonical details before the adapter receives the corresponding native rows.
    """
    real_exec: Any = asyncio.create_subprocess_exec
    async def _fake_exec(*args: object, **kwargs: object) -> MagicMock:
        if args and args[0] == "pi":
            argv = [str(a) for a in args]
            if argv == ["pi", "--version"]:
                return make_mock_process(["1.4.0"])
            captured.append(argv)
            process = response(argv) if response else make_mock_process_from_fixture("simple_text.jsonl")
            environment = cast(dict[str, str], kwargs.get("env") or {})
            packet_path = environment.get("DAYDREAM_PI_HOST_PACKET")
            if packet_path is None:
                return process
            packet = json.loads(Path(packet_path).read_text())
            async def request(path: str, value: dict[str, Any]) -> dict[str, Any]:
                async with httpx.AsyncClient(base_url=packet["url"],
                    headers={"Authorization": packet["authorization"]}) as client:
                    result = await client.post(path, json=value)
                    assert result.status_code == 200, result.text
                    return cast(dict[str, Any], result.json())
            assert (await request("/ready", {"tool_name": packet["tool_name"], "schema": packet["schema"]}))["ready"]
            stdout: Any = process.stdout
            original_readline = stdout.readline
            ready_pending = True
            terminal_call: str | None = None
            arguments: dict[str, Any] = {}
            permissions: dict[str, dict[str, Any]] = {}
            finalized: dict[str, dict[str, Any]] = {}
            pending_rows: list[bytes] = []
            async def readline() -> bytes:
                nonlocal ready_pending, terminal_call
                if pending_rows:
                    return pending_rows.pop(0)
                if ready_pending:
                    ready_pending = False
                    return b'{"type":"daydream_host_tools_ready"}\n'
                raw: bytes = await original_readline()
                if not raw:
                    return raw
                row = json.loads(raw)
                event_type = row.get("type")
                message = row.get("message") or {}
                if event_type == "message_end" and message.get("role") == "assistant":
                    calls = [part for part in message.get("content", []) if part.get("type") == "toolCall"]
                    terminal_call = (calls[0]["id"] if len(calls) == 1
                        and calls[0]["name"] == "structured_output"
                        and message.get("stopReason") in {"stop", "toolUse"} else None)
                elif event_type == "tool_execution_start":
                    call_id = row["toolCallId"]
                    arguments[call_id] = row.get("args")
                    permissions[call_id] = await request("/admit", {"id": call_id, "name": row["toolName"],
                        "arguments": row.get("args"), "terminal": terminal_call == call_id})
                    receipt = permissions[call_id]
                    if receipt.get("outcome") in {"fatal", "indeterminate"}:
                        pending_rows.append(raw)
                        return (json.dumps({"type": "daydream_host_tool_fatal", "id": call_id,
                            "outcome": receipt["outcome"], "code": receipt["code"], "error": receipt["error"]})
                            + "\n").encode()
                elif event_type == "tool_execution_end" and row.get("toolName") == "structured_output":
                    call_id = row["toolCallId"]
                    receipt = permissions[call_id]
                    if receipt.get("allowed") and row.get("isError") is False:
                        receipt = await request("/submit", {"id": call_id, "arguments": arguments[call_id]})
                        assert receipt["id"] == call_id
                    if receipt.get("accepted"):
                        native = {"isError": False, "content": [{"type": "text", "text": "Review result accepted."}],
                                  "details": receipt["result"]}
                    else:
                        native = {"isError": True, "content": [{"type": "text",
                                  "text": receipt.get("error", "Rejected native terminal call")}]}
                        if receipt.get("outcome") in {"fatal", "indeterminate"}:
                            native["details"] = {"host_submission_status": receipt["outcome"], "code": receipt["code"]}
                    finalized[call_id] = native
                    row.update(isError=native["isError"], result={key: val for key, val in native.items()
                                                               if key != "isError"})
                elif event_type == "message_end" and message.get("role") == "toolResult":
                    if message.get("toolCallId") in finalized:
                        row["message"] = {**message, **finalized[message["toolCallId"]]}
                return (json.dumps(row) + "\n").encode()
            stdout.readline = readline
            return process
        return cast(MagicMock, await real_exec(*args, **kwargs))
    monkeypatch.setattr("daydream.backends._transport.asyncio.create_subprocess_exec", _fake_exec)


def _assert_pi_model_and_provider(captured: list[list[str]], *, model: str, provider: str) -> None:
    assert captured, "expected at least one pi subprocess spawn"
    for argv in captured:
        assert argv[argv.index("--model") + 1] == model
        assert argv[argv.index("--provider") + 1] == provider


@pytest.mark.parametrize(
    ("ambient", "cli_effort", "file_global", "file_phase", "expected", "env_provider", "expected_provider"), [
        (None, None, None, None, "medium", None, "nous"),
        ("high", None, None, None, "medium", "custom-proxy", "custom-proxy"),
        ("high", None, "low", None, "low", None, "nous"),
        ("high", None, "low", "xhigh", "xhigh", "custom-proxy", "custom-proxy"),
        ("high", "low", "high", "xhigh", "low", None, "nous"),
    ], ids=["default", "ambient-high", "file-global", "file-phase", "cli-global"],
)
async def test_runner_pi_stack_review_effort_reaches_native_submission(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
    ambient: str | None, cli_effort: str | None, file_global: str | None, file_phase: str | None, expected: str,
    env_provider: str | None, expected_provider: str,
) -> None:
    """Built-in effort precedence and default/override provider/model reach native Pi argv."""
    _silence(monkeypatch)
    if ambient is None:
        monkeypatch.delenv("PI_THINKING", raising=False)
    else:
        monkeypatch.setenv("PI_THINKING", ambient)
    if env_provider is None:
        monkeypatch.delenv("PI_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("PI_PROVIDER", env_provider)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi-agent"))
    captured: list[list[str]] = []
    reviewer_args: list[list[str]] = []
    wall_allowances: list[float] = []
    fake = FakeClock(monotonic_value=1000).install(monkeypatch)

    def response(argv: list[str]) -> MagicMock:
        prompt = Path(argv[-1][1:]).read_text()
        stage = review_stage_state(prompt)
        if len(captured) == 1:
            fake.advance(1)  # Preparation/earlier phases spend the shared allowance.
        if stage is None and "--extension" not in argv:
            return make_mock_process_from_fixture("simple_text.jsonl")
        if stage is not None:
            reviewer_args.append(argv)
            system_arg = argv[argv.index("--append-system-prompt") + 1]
            system = Path(system_arg).read_text() if system_arg.startswith("/") else system_arg
            assert "Attempt dispatch-time snapshot:" in system
            assert ("Each call counts, including failed calls, parallel batch members and structured submissions"
                    in system)
            match = re.search(r"effective wall allowance is ([0-9.]+) seconds", system)
            assert match is not None
            wall_allowances.append(float(match.group(1)))
        payload = stage_result(stage) if stage is not None else {"issues": []}
        call = {"type": "toolCall", "id": "submit", "name": "structured_output", "arguments": payload}
        assistant = {"role": "assistant", "content": [call], "stopReason": "toolUse"}
        tool_result = {"content": [{"type": "text", "text": "Submitted."}], "details": payload}
        return make_mock_process([json.dumps(row) for row in [
            {"type": "session", "sessionId": "review-fixture"},
            {"type": "agent_start"}, {"type": "turn_start"},
            {"type": "message_end", "message": assistant},
            {"type": "tool_execution_start", "toolCallId": "submit", "toolName": "structured_output",
             "args": payload},
            {"type": "tool_execution_end", "toolCallId": "submit", "toolName": "structured_output",
             "isError": False, "result": tool_result},
            {"type": "message_end", "message": {"role": "toolResult", "toolCallId": "submit",
             "toolName": "structured_output", "isError": False, **tool_result}},
            {"type": "turn_end", "message": assistant},
            {"type": "agent_end", "messages": []}, {"type": "agent_settled"},
        ]])

    _capture_pi_subprocess(monkeypatch, captured, response)
    phases = {"per_stack_review": {"reasoning_effort": file_phase}} if file_phase else {}
    profile = independent_alternatives_profile()
    profile = replace(profile, profile=replace(profile.profile,
        pipeline=replace(profile.profile.pipeline, review_wall_budget_s=385)))
    config = make_config(multi_stack_target, backend="pi", reasoning_effort=cli_effort,
        file_config=DaydreamFileConfig(reasoning_effort=file_global, phases=phases),
        output_mode="review", assume="yes", diagram="off", latency_profile="forensic",
        review_profile=profile,
        shallow_fanout_threshold=0, review_cache_enabled=False, run_eval=False)
    assert await run(config) == 0
    _assert_pi_model_and_provider(captured, model="deepseek/deepseek-v4-flash-0731", provider=expected_provider)
    assert reviewer_args
    # The host nominal ceiling is 3600s; 385s shared minus 77s reserve and 1s spent leaves 307s.
    assert wall_allowances and all(0 < wall <= 307 for wall in wall_allowances)
    assert {(argv[argv.index("--thinking") + 1] if "--thinking" in argv else None)
            for argv in reviewer_args} == {expected}
    coverage = json.loads((multi_stack_target / ".daydream/deep/review-coverage.json").read_text())
    assert {scope["status"] for scope in coverage["stack_outcomes"]} == {"complete"}
    assert (multi_stack_target / ".review-output.md").is_file()
    # The independently dispatched wonder keeps ambient effort under the forensic route.
    other_args = [argv for argv in captured if argv not in reviewer_args]
    assert other_args
    other_effort = cli_effort or file_global or ambient
    for argv in other_args:
        assert (argv[argv.index("--thinking") + 1] if "--thinking" in argv else None) == other_effort


@pytest.mark.parametrize(
    ("env_provider", "expected_provider"), [(None, "nous"), ("custom-proxy", "custom-proxy")],
    ids=["default-nous", "PI_PROVIDER-override"],
)
async def test_runner_custom_flow_pi_reports_unbounded_allowance(
    tiny_diff_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, ext_dir: ExtDir,
    env_provider: str | None, expected_provider: str,
) -> None:
    """One real Pi turn preserves provider selection, fallback model and unbounded allowance."""
    _silence(monkeypatch)
    if env_provider is None:
        monkeypatch.delenv("PI_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("PI_PROVIDER", env_provider)
    # No user settings may replace the production fallback model/provider.
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi-agent"))
    ext_dir.write_module(
        "from daydream.agent import run_agent\n"
        "from daydream.extensions import FlowStep\n"
        "from daydream.trajectory import DaydreamPhase\n"
        "async def unbounded(ctx):\n"
        "    result, _, reason = await run_agent(ctx.backend_for('unbounded'), ctx.work.repo, 'Unbounded task',\n"
        "        phase=DaydreamPhase.REVIEW, wall_budget_s=None, run_context=ctx.run_context)\n"
        "    assert reason is None and ctx.artifacts is not None\n"
        "    ctx.artifacts.review_output.write_text(result)\n"
        "def register(registry):\n"
        "    registry.register_phase(FlowStep(name='unbounded', run=unbounded))\n"
        "    registry.set_flow('unbounded', ['unbounded'])\n"
    )
    system_prompts: list[str] = []

    def response(argv: list[str]) -> MagicMock:
        system_arg = argv[argv.index("--append-system-prompt") + 1]
        system_prompts.append(Path(system_arg).read_text() if system_arg.startswith("/") else system_arg)
        return make_mock_process_from_fixture("simple_text.jsonl")

    captured: list[list[str]] = []
    _capture_pi_subprocess(monkeypatch, captured, response)
    assert await run(make_config(tiny_diff_target, backend="pi", flow_name="unbounded", run_eval=False)) == 0
    _assert_pi_model_and_provider(captured, model="deepseek/deepseek-v4-flash-0731", provider=expected_provider)
    assert (tiny_diff_target / ".review-output.md").read_text() == "Hello from Pi"
    assert len(system_prompts) == 1
    assert "Attempt dispatch-time snapshot:" in system_prompts[0]
    assert "tool-call allocation is uncapped" in system_prompts[0]
    assert "wall allowance is unbounded" in system_prompts[0]
    assert "1800 seconds" not in system_prompts[0]


# System prompt preamble (--append-system-prompt)


# PiError.retryable attribute


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


# _is_retryable_exit_code


# Error turn uses retryable classifier


@pytest.mark.parametrize(
    ("error_message", "expected_retryable", "expected_category"),
    [
    ("502 status code (no body)", True, "SERVER_ERROR"),
    ("request timed out after 30 seconds", True, "TIMEOUT"),
    ("Stream ended without finish_reason", True, "STREAM_TRUNCATION"),
    ("socket hang up", True, "STREAM_DROP"),
    ("Pi CLI exited with return code 1", False, "PROCESS_EXIT"),
    ("model not found: gpt-5 (503)", False, "AUTH_CONFIG"),
    ("JSON schema validation failed: the model is temporarily at capacity.", False, "SCHEMA"),
    ("provider rate limit", True, "RATE_LIMIT"),
    (
            "Upstream error from Wafer: The model is temporarily at capacity. Please retry shortly.",
            True,
            "SERVER_ERROR",
        ),
    ("The MODEL is TEMPORARILY\nAT CAPACITY.", True, "SERVER_ERROR"),
    ("The model is not temporarily at capacity.", False, "UNKNOWN"),
    ("Capacity planning: the model is temporarily at capacity.", False, "UNKNOWN"),
    ("If the model is temporarily at capacity, retry later", False, "UNKNOWN"),
    ("The model is temporarily at capacity?", False, "UNKNOWN"),
    ("Service is currently overloaded", True, "SERVER_ERROR"),
    ("capacity exceeded", True, "SERVER_ERROR"),
    ("Request throttled", True, "SERVER_ERROR"),
    ("service is not overloaded", False, "UNKNOWN"),
],
)
async def test_error_turn_sets_retryable_via_classifier(
    error_message: str,
    expected_retryable: bool,
    expected_category: str,
) -> None:
    """Public terminal errors retain diagnosis without altering classification or totals."""
    backend = PiBackend(model="glm-5.2")
    lines = [
        '{"type":"session","sessionId":"pi_ses_err"}', '{"type":"agent_start"}', '{"type":"turn_start"}',
        '{"type":"turn_end","message":{"role":"assistant","content":[],'
        '"model":"native-model","provider":"native-provider",'
        f'"stopReason":"error","rawStopReason":"error","errorMessage":{json.dumps(error_message)},'
        '"usage":{"input":50,"output":10,"cacheRead":20,"cacheWrite":30,"cost":{"total":0.01}}}}',
    ]
    mock_proc = make_mock_process(lines)
    events = []
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as spawn:
        with pytest.raises(PiError) as exc_info:
            async for event in backend.execute(Path("/tmp"), "p"):
                events.append(event)
    assert exc_info.value.retryable is expected_retryable
    assert exc_info.value.category == expected_category and str(exc_info.value) == error_message
    assert spawn.call_count == 1
    diagnostics = [event for event in events if isinstance(event, DiagnosticEvent)]
    assert len(diagnostics) == 1 and diagnostics[0].message == error_message
    assert diagnostics[0].metadata == {
        "category": expected_category, "retryable": expected_retryable, "finish_reason": "error",
        "raw_stop_reason": "error", "message_source": "native",
    }
    assert events.index(diagnostics[0]) < next(i for i, event in enumerate(events) if isinstance(event, TurnEndEvent))
    totals = [event for event in events if isinstance(event, CostEvent)]
    assert len(totals) == 1 and totals[0].input_tokens == 100 and totals[0].output_tokens == 10
    assert totals[0].cost_usd == 0.01
    assert (totals[0].cached_tokens, totals[0].cache_creation_tokens) == (20, 30)
    metric = next(event for event in events if isinstance(event, MetricsEvent))
    assert (metric.model_name, metric.provider_name, metric.usage_scope, metric.cache_creation_tokens) == (
        "native-model", "native-provider", "message", 30,
    )
    result = next(event for event in events if isinstance(event, ResultEvent))
    assert (result.model_name, result.provider_name, result.session_id, result.finish_reason, result.continuation) == (
        "native-model", "native-provider", "pi_ses_err", "error", None,
    )
    assert mock_proc.returncode is not None


@pytest.mark.parametrize("native", [
    {}, {"errorMessage": None, "rawStopReason": {"not": "a string"}},
    {"errorMessage": ["invalid"], "rawStopReason": 7},
    {"errorMessage": 'failure: -----BEGIN PRIVATE KEY-----\n' + 'x' * 5000
     + '\n-----END PRIVATE KEY-----\n' + 'safe ' * 2000,
     "rawStopReason": 'TOKEN=hidden-secret; ' + 'reason ' * 1000},
], ids=["absent", "null", "malformed", "oversized-secret"])
async def test_pi_terminal_diagnostic_is_bounded_and_redacted_before_persistence(
    tmp_path: Path, native: dict[str, Any],
) -> None:
    """Public optional fields cannot crash error handling or leak cut-off secret fragments."""
    backend = PiBackend(model="fixture-model")
    backend.retry_attempts = 0
    message = {"role": "assistant", "content": [], "stopReason": "error", **native}
    recorder = make_recorder(tmp_path)
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=make_mock_process([
        json.dumps({"type": "turn_end", "message": message}),
    ])) as spawn:
        async with recorder:
            with pytest.raises(PiError) as failure:
                await run_agent(backend, tmp_path, "Review", phase=DaydreamPhase.REVIEW,
                                progress_callback=lambda _: None)
    assert failure.value.category == "UNKNOWN" and failure.value.retryable is False
    assert spawn.call_count == 1
    document = json.loads(recorder.path.read_text())
    diagnostics = [diagnostic for step in document["steps"]
                   for diagnostic in step.get("extra", {}).get("backend_diagnostics", [])]
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert len(diagnostic["message"]) <= 4096
    if isinstance(native.get("errorMessage"), str):
        assert diagnostic["metadata"]["message_source"] == "native"
        assert diagnostic["metadata"]["message_truncated"] is True
        assert "[REDACTED_PEM_KEY]" in diagnostic["message"]
        assert "x" * 20 not in json.dumps(diagnostics) and "hidden-secret" not in json.dumps(diagnostics)
        assert "[REDACTED_ENV_VAR]" in diagnostic["metadata"]["raw_stop_reason"]
        assert len(diagnostic["metadata"]["raw_stop_reason"]) == 256
        assert diagnostic["metadata"]["raw_stop_reason_truncated"] is True
    else:
        assert diagnostic["message"] == "" and diagnostic["metadata"]["message_source"] == "unavailable"
        assert "raw_stop_reason" not in diagnostic["metadata"]


@pytest.mark.parametrize("redaction_fault", ["raises", "nontext"])
async def test_pi_diagnostic_redaction_fault_preserves_native_failure_and_terminal_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, redaction_fault: str,
) -> None:
    """A failed evidence sanitizer must neither replace PiError nor lose terminal billing."""
    def broken_redactor(_: str) -> Any:
        if redaction_fault == "raises":
            raise RuntimeError("redactor unavailable")
        return None

    monkeypatch.setattr("daydream.backends.pi.redact_structured_text", broken_redactor)
    backend = PiBackend(model="fixture-model")
    backend.retry_attempts = 0
    recorder = make_recorder(tmp_path)
    native = {"role": "assistant", "stopReason": "error", "errorMessage": "Provider finish_reason: error",
              "rawStopReason": "TOKEN=never-persist", "usage": {"input": 100, "output": 10,
                                                                     "cost": {"total": 0.01}}}
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=make_mock_process([
        json.dumps({"type": "turn_end", "message": native}),
    ])) as spawn:
        async with recorder:
            with pytest.raises(PiError) as failure:
                await run_agent(backend, tmp_path, "Review", phase=DaydreamPhase.REVIEW,
                                progress_callback=lambda _: None)
    assert failure.value.category == "UNKNOWN" and failure.value.retryable is False
    assert spawn.call_count == 1
    document = json.loads(recorder.path.read_text())
    diagnostics = [diagnostic for step in document["steps"]
                   for diagnostic in step.get("extra", {}).get("backend_diagnostics", [])]
    assert len(diagnostics) == 1
    assert diagnostics[0]["message"] == "[DIAGNOSTIC_REDACTION_FAILED]"
    assert diagnostics[0]["metadata"]["raw_stop_reason"] == "[DIAGNOSTIC_REDACTION_FAILED]"
    assert "never-persist" not in json.dumps(document)
    assert document["final_metrics"]["total_prompt_tokens"] == 100
    assert document["final_metrics"]["total_completion_tokens"] == 10
    assert document["final_metrics"]["total_cost_usd"] == 0.01


# Retry env knobs


# fanout_concurrency


@pytest.mark.parametrize(("environment", "expected", "warnings"), [
    ({}, RetryPolicy(20, 10.0, 120.0), []),
    ({"DAYDREAM_PI_RETRY_ATTEMPTS": "5", "DAYDREAM_PI_RETRY_BASE_DELAY_S": "0.5",
      "DAYDREAM_PI_RETRY_MAX_DELAY_S": "45.5"}, RetryPolicy(5, 0.5, 45.5), []),
    ({"DAYDREAM_PI_RETRY_ATTEMPTS": "-1", "DAYDREAM_PI_RETRY_BASE_DELAY_S": "nan",
      "DAYDREAM_PI_RETRY_MAX_DELAY_S": "not-a-float"}, RetryPolicy(20, 10.0, 120.0), [
        "DAYDREAM_PI_RETRY_ATTEMPTS='-1' is negative; using default 20",
        "DAYDREAM_PI_RETRY_BASE_DELAY_S='nan' is not finite; using default 10",
        "DAYDREAM_PI_RETRY_MAX_DELAY_S='not-a-float' is not a valid float; using default 120",
    ]),
    ({"DAYDREAM_PI_RETRY_ATTEMPTS": "", "DAYDREAM_PI_RETRY_BASE_DELAY_S": "-1"},
     RetryPolicy(20, 10.0, 120.0), [
        "DAYDREAM_PI_RETRY_ATTEMPTS='' is not a valid integer; using default 20",
        "DAYDREAM_PI_RETRY_BASE_DELAY_S='-1' is negative; using default 10",
    ]),
])
def test_pi_retry_policy_is_built_from_ambient_environment(
    environment: dict[str, str], expected: RetryPolicy, warnings: list[str], caplog: pytest.LogCaptureFixture,
) -> None:
    """One actual constructor admits or diagnoses all retry settings from the same captured environment."""
    with patch.dict(os.environ, environment, clear=True), caplog.at_level("WARNING"):
        backend = PiBackend(model="fixture-model")
    assert (backend.retry_attempts, backend.retry_base_delay_s, backend.retry_max_delay_s) == (
        expected.attempts, expected.base_delay_s, expected.max_delay_s,
    )
    assert [record.getMessage() for record in caplog.records] == warnings


# --- P18 Task 1: generation lifecycle + config at the Pi argv/JSONL seam -----


@pytest.mark.parametrize('mode', ['native', 'no-schema', 'disabled', 'finalization', 'validation-opt-out'])
async def test_pi_request_event_config_matches_exact_argv(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.delenv("PI_API_KEY", raising=False)
    backend = PiBackend(model="glm-5.2")
    schema = None if mode == "no-schema" else {"type": "object", "properties": {"answer": {"type": "string"}}}
    process = make_mock_process((FIXTURES_DIR / 'simple_text.jsonl').read_text().splitlines()
                                + ['{"type":"agent_settled"}'])
    if mode == 'disabled':
        process.stdin = MagicMock()
    events, spawn = await replay_process(
        backend, process, Path("/tmp"), "structured please",
        output_schema=schema, tools_disabled=mode == "disabled", finalization=mode == "finalization",
        validate_structured_output=mode != "validation-opt-out",
        read_only=True, persist_session=False,
    )
    flat_args = list(spawn.call_args.args)
    request = next(event for event in events if isinstance(event, RequestEvent))
    config = request.config
    assert isinstance(config, PiRequestConfig)
    assert config.read_only is True
    if mode in {'disabled', 'finalization'}:
        assert '--no-tools' in flat_args and config.selected_tools_count == 0
    else:
        assert flat_args[flat_args.index('--tools') + 1] == 'read,find,ls,grep' + (
            ',structured_output' if mode == 'native' else '')
        assert config.selected_tools_count == (5 if mode == 'native' else 4)
        assert config.selected_tools_present is True
    assert config.no_skills is True and "--no-skills" in flat_args
    assert config.schema_emulated is (mode not in {"native", "no-schema"})
    assert config.persist_session is False and "--no-session" in flat_args
    assert "--session-id" not in flat_args and request.output_schema == schema
    terminal = next(event for event in events if isinstance(event, ResultEvent))
    assert terminal.session_id == "pi_ses_simple" and terminal.continuation is None
    assert config.continuation_mode == "fresh"
    assert config.model_mode == "single"
    assert config.max_turns is None  # accepted by daydream, never passed to Pi
    # System preamble is invocation-level content, never a config value.
    assert request.system_prompt is not None
    if mode != "finalization":
        assert "Attempt dispatch-time snapshot:" in request.system_prompt
        assert "tool-call allocation is uncapped" in request.system_prompt


async def test_pi_error_turn_sets_explicit_incomplete_boundary() -> None:
    """An errored turn keeps identity native where exposed; outcome stays custom."""
    backend = PiBackend(model="glm-5.2")
    mock_proc = make_mock_process_from_fixture("error_turn.jsonl")
    collected: list[Any] = []
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(PiError):
            async for event in backend.execute(Path("/tmp"), "go"):
                collected.append(event)
    # The error fixture carries no model/provider on the boundary; identity
    # fields must stay None, never a fabricated value.
    turn_ends = [e for e in collected if isinstance(e, TurnEndEvent)]
    assert turn_ends
    error_turn = turn_ends[-1]
    assert error_turn.finish_reason == "error"
    assert error_turn.model_name is None or error_turn.model_name == "glm-5.2"


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
    assert request.output_schema == schema and "structured_output" in request.prompt
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


async def test_native_pi_tool_batch_and_resumed_turn_record_literal_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native conversation preserves heterogeneous observations and partial-turn telemetry."""
    monkeypatch.delenv("DAYDREAM_PI_FANOUT_CONCURRENCY", raising=False)
    batch = [
        ("blocks", {"content": [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]}),
        ("content", {"content": "raw"}),
        ("details", {"details": {"note": "x"}}),
        ("plain", "plain"),
        ("empty", None),
        (
            "image",
            {
                "content": [
                    {"type": "text", "text": "caption"},
                    {"type": "image", "data": "public-image", "mimeType": "image/png"},
                ],
                "details": {"count": 1},
            },
        ),
        ("failed", {"content": "permission denied"}),
    ]
    assistant = {
        "role": "assistant",
        "model": "pi-model",
        "provider": "native-provider",
        "content": [
            {"type": "thinking", "thinking": "Inspect these results"},
            {"type": "text", "text": "Inspecting"},
            *[
                {"type": "toolCall", "id": call_id, "name": "read", "arguments": {"path": call_id}}
                for call_id, _ in batch
            ],
        ],
    }
    rows = [
        {"type": "session", "sessionId": "pi-native"},
        {"type": "agent_start"},
        {"type": "turn_start"},
        {"type": "message_start", "message": assistant},
        {"type": "message_end", "message": assistant},
    ]
    for call_id, result in batch:
        rows.extend(
            [
                {"type": "tool_execution_start", "toolCallId": call_id, "toolName": "read", "args": {"path": call_id}},
                {
                    "type": "tool_execution_end",
                    "toolCallId": call_id,
                    "toolName": "read",
                    "result": result,
                    "isError": call_id == "failed",
                },
            ]
        )
    final = {
        "role": "assistant",
        "model": "pi-model",
        "provider": "native-provider",
        "content": [{"type": "text", "text": "Complete"}],
        "stopReason": "stop",
        "usage": {"input": 150, "output": 100, "cacheRead": 20, "cacheWrite": 30, "cost": {"total": 0.0}},
    }
    rows.extend(
        [
            {"type": "turn_end", "message": {"role": "assistant", "stopReason": "stop"}},
            {"type": "turn_start"},
            {"type": "message_end", "message": final},
            {"type": "turn_end", "message": final},
            {"type": "agent_end", "messages": []},
        ]
    )
    resumed = {
        "role": "assistant",
        "model": "pi-model",
        "provider": "native-provider",
        "content": [{"type": "text", "text": "Fixed"}],
        "stopReason": "stop",
        "usage": {"input": 75, "output": 50, "cacheRead": 25, "cost": {"total": 0.001}},
    }
    processes = [
        make_mock_process([json.dumps(row) for row in rows]),
        make_mock_process(
            [
                json.dumps(row)
                for row in [
                    {"type": "session", "sessionId": "pi-native"},
                    {"type": "agent_start"},
                    {"type": "turn_start"},
                    {"type": "message_end", "message": resumed},
                    {"type": "turn_end", "message": resumed},
                    {"type": "agent_end", "messages": []},
                ]
            ]
        ),
    ]
    backend = PiBackend(model="fixture-model")
    assert backend.fanout_concurrency == 10
    recorder = make_recorder(tmp_path, agent_model_name="pi")
    events: list[Any] = []
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", side_effect=processes) as spawn:
        async with recorder:
            async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
                inv.observe_user_step(prompt="Review")
                async for event in backend.execute(tmp_path, "Review", read_only=True):
                    events.append(event)
                    inv.observe(event)
            token = next(event for event in events if isinstance(event, ResultEvent)).continuation
            assert token is not None and token.data["session_id"] == "pi-native"
            async with recorder.invocation(phase=DaydreamPhase.FIX) as inv:
                inv.observe_user_step(prompt="Fix")
                async for event in backend.execute(tmp_path, "Fix", continuation=token):
                    events.append(event)
                    inv.observe(event)
    first, second = [list(call.args) for call in spawn.call_args_list]
    uuid.UUID(first[first.index("--session-id") + 1])
    assert "--no-session" not in first and second[second.index("--session-id") + 1] == "pi-native"
    assert [event.text for event in events if isinstance(event, TextEvent)] == ["Inspecting", "Complete", "Fixed"]
    assert [
        (event.message_id, event.prompt_tokens, event.completion_tokens)
        for event in events
        if isinstance(event, MetricsEvent)
    ] == [("", 200, 100), ("", 100, 50)]
    assert [event.cost_usd for event in events if isinstance(event, CostEvent)] == [0.0, 0.001]
    assert [(event.model_name, event.provider_name, event.usage_scope) for event in events
            if isinstance(event, MetricsEvent)] == [("pi-model", "native-provider", "message")] * 2
    assert [(event.session_id, event.model_name, event.provider_name, event.finish_reason) for event in events
            if isinstance(event, ResultEvent)] == [("pi-native", "pi-model", "native-provider", "stop")] * 2
    assert [(event.cached_tokens, event.cache_creation_tokens) for event in events
            if isinstance(event, CostEvent)] == [(20, 30), (25, None)]
    first_end = next(index for index, event in enumerate(events) if isinstance(event, TurnEndEvent))
    assert not any(isinstance(event, MetricsEvent) for event in events[:first_end])
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    expected = {
        "blocks": "line1line2",
        "content": "raw",
        "details": '{"details": {"note": "x"}}',
        "plain": "plain",
        "empty": "",
        "image": '{"content": [{"type": "text", "text": "caption"}, '
        '{"type": "image", "data": "public-image", "mimeType": "image/png"}], "details": {"count": 1}}',
        "failed": "permission denied",
    }
    assert {event.id: event.output for event in results} == expected
    assert [event.id for event in results if event.is_error] == ["failed"]
    document = json.loads(recorder.path.read_text())
    agent_steps = [step for step in document["steps"] if step["source"] == "agent"]
    recorded = next(step for step in agent_steps if step["message"] == "Inspecting")
    assert recorded["reasoning_content"] == "Inspect these results"
    assert [call["tool_call_id"] for call in recorded["tool_calls"]] == list(expected)
    assert {result["source_call_id"]: result["content"] for result in recorded["observation"]["results"]} == expected
    assert document["final_metrics"]["total_prompt_tokens"] == 300
    assert document["final_metrics"]["total_completion_tokens"] == 150
    assert [step["message"] for step in agent_steps] == ["Inspecting", "Complete", "Fixed"]
    billed = [step["metrics"] for step in agent_steps if step.get("metrics")]
    assert [(m["prompt_tokens"], m["completion_tokens"], m["cost_usd"]) for m in billed] == [
        (200, 100, 0.0), (100, 50, 0.001),
    ]
    assert document["final_metrics"]["total_cached_tokens"] == 45
    assert document["final_metrics"]["total_cost_usd"] == pytest.approx(0.001)
    assert [call.kwargs["stdin"] for call in spawn.call_args_list] == [asyncio.subprocess.DEVNULL] * 2
    for process in processes:
        assert process.returncode == 0 and process.stdin is None
        process.wait.assert_awaited()
        process._transport.close.assert_called_once()
    assert backend._transports == []
    assert validate(recorder.path, validate_images=False)
