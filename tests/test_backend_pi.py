"""Drive PiBackend.execute through canned JSONL and real protocol CLI fixtures."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
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
from daydream.backends.pi import (
    PiBackend,
    PiError,
)
from daydream.trajectory import DaydreamPhase
from tests.harness.fake_cli_process import (
    install_fake_cli_process,
)
from tests.harness.fake_clock import FakeClock
from tests.harness.pi_replay import FIXTURES_DIR, make_mock_process, make_mock_process_from_fixture
from tests.harness.process_replay import replay_process
from tests.harness.protocol_cli import install_protocol_cli
from tests.harness.protocol_cli_assertions import make_cancel_probe
from tests.harness.trajectory import make_recorder


def _case_dir(tmp_path: Path, index: int) -> Path:
    case = tmp_path / f"case-{index}"
    case.mkdir()
    return case


async def _interrupted_pi_updates_retain_only_admitted_progress_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    """Content-free Pi progress retains malformed/non-progress guards through interruption."""
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
    elif kind == "generation":
        frames.extend({"type": "message_update", "assistantMessageEvent": {
            "type": f"{delta_kind}_delta", "contentIndex": index, "delta": delta,
        }} for index, (delta_kind, deltas) in enumerate((
            ("text", ["private-text-canary"]),
            ("thinking", ["private-", "thinking-canary"]),
            ("toolcall", ['{"path":', '"private-toolcall-canary"', '}']),
        )) for delta in deltas)
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

    # Interrupt only after the real reader consumes every frame and blocks for more bytes.
    # Only the agent wall scope is staged; cleanup keeps its ordinary real-time bounds.
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
    assert (evidence["text_delta_count"], evidence["thinking_delta_count"], evidence["toolcall_delta_count"]) == (
        (1, 2, 3) if kind == "generation" else (0, 0, 0))
    assert evidence["tool_progress_count"] == int(kind == "tool")
    assert (evidence["last_progress_s"] is None) == (kind == "diagnostic")
    assert evidence["active_tool_count"] == int(kind == "tool")
    assert "private" not in json.dumps(evidence)
    assert spawner.procs[0].reaped and spawner.procs[0].returncode == -15


async def test_interrupted_pi_updates_retain_only_admitted_progress(tmp_path: Path) -> None:
    for index, kind in enumerate(["generation", "tool", "diagnostic"]):
        with pytest.MonkeyPatch.context() as mp:
            await _interrupted_pi_updates_retain_only_admitted_progress_case(
                monkeypatch=mp, tmp_path=_case_dir(tmp_path, index), kind=kind
            )


if TYPE_CHECKING:
    from daydream.run_config import RunConfig

MakeConfig = Callable[..., "RunConfig"]


def test_ordinary_pi_fresh_process_does_not_load_host_admission_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary Pi runs without a host endpoint; all-extras imports can hide unwanted eager endpoint loading."""
    fixture = install_protocol_cli(tmp_path / "external", "pi")
    target = tmp_path / "repo"
    target.mkdir()
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PI_PROVIDER", "nous")
    monkeypatch.delenv("PI_API_KEY", raising=False)
    script = """
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
"""
    result = subprocess.run([sys.executable, "-c", script, str(target)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    (observation,) = fixture.read_observations()
    assert observation["process_outcome"] == "success"


def test_backend_execution_input_rejects_explicit_osprey() -> None:
    with pytest.raises(ValueError, match="osprey"):
        BackendExecutionInput.from_environment({}, backend="osprey")


@pytest.fixture
def pi_workspace(tmp_path: Path) -> Path:
    """Return a workspace whose ``.pi/settings.json`` selects a default model."""
    settings = tmp_path / ".pi" / "settings.json"
    settings.parent.mkdir()
    settings.write_text('{"defaultProvider": "openai", "defaultModel": "gpt-5.6-luna"}')
    return tmp_path


async def _run_and_capture_args(
    backend: Any,
    prompt: Any = "p",
    *,
    fixture: Any = "simple_text.jsonl",
    cwd: Path = Path("/tmp"),
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Replay a fixture and return (argv, spawner) for exact command/environment assertions."""
    mock_proc = make_mock_process_from_fixture(fixture)
    _, mock_exec = await replay_process(backend, mock_proc, cwd, prompt, **kwargs)
    return list(mock_exec.call_args.args), mock_exec


async def _native_output_uses_finalized_transcript_order_and_requires_complete_settlement_case(fault: str) -> None:
    """Daydream selects settled successful submissions independently of execution completion order."""
    calls: list[dict[str, Any]] = [
        {"type": "toolCall", "id": call_id, "name": "structured_output", "arguments": {"answer": value}}
        for call_id, value in [("first", "first"), ("last", "last")]
    ]
    if fault == "domain-truncation":
        calls[-1]["arguments"]["truncation"] = {"truncated": True, "firstLineExceedsLimit": True}
    rows: list[dict[str, Any]] = [
        {"type": "agent_start"},
        {"type": "turn_start"},
        {"type": "message_end", "message": {"role": "assistant", "content": calls, "stopReason": "toolUse"}},
        *[
            {
                "type": "tool_execution_start",
                "toolCallId": call["id"],
                "toolName": "structured_output",
                "args": call["arguments"],
            }
            for call in calls
        ],
        # Parallel completion arrives in the reverse order; finalized messages remain transcript ordered.
        *[
            {
                "type": "tool_execution_end",
                "toolCallId": call["id"],
                "toolName": "structured_output",
                "isError": False,
                "result": {"content": [{"type": "text", "text": "Submitted."}], "details": call["arguments"]},
            }
            for call in reversed(calls)
        ],
        *[
            {
                "type": "message_end",
                "message": {
                    "role": "toolResult",
                    "toolCallId": call["id"],
                    "toolName": "structured_output",
                    "isError": False,
                    "content": [{"type": "text", "text": "Submitted."}],
                    "details": call["arguments"],
                },
            }
            for call in calls
        ],
        {"type": "turn_end", "message": {"role": "assistant", "content": calls, "stopReason": "toolUse"}},
        {"type": "agent_end", "messages": []},
        {"type": "agent_settled"},
    ]
    if fault == "failed-last":
        rows[5]["isError"] = True
        rows[-4]["message"]["isError"] = True
    elif fault in {"mixed-read", "mixed-read-first"}:
        read_call = {"type": "toolCall", "id": "read", "name": "read", "arguments": {"path": "api.py"}}
        calls.insert(0 if fault == "mixed-read-first" else len(calls), read_call)
        rows[3:3] = [
            {"type": "tool_execution_start", "toolCallId": "read", "toolName": "read", "args": read_call["arguments"]},
            {
                "type": "tool_execution_end",
                "toolCallId": "read",
                "toolName": "read",
                "isError": False,
                "result": {"content": [{"type": "text", "text": "VALUE = 1"}]},
            },
        ]
    elif fault == "null-details":
        rows[-4]["message"]["details"] = None
    elif fault == "oversized-details":
        rows[-4]["message"]["details"] = {"answer": "x" * (129 * 1024)}
    elif fault == "wrong-id":
        rows[-4]["message"]["toolCallId"] = "unassociated"
    elif fault == "unsettled":
        rows.pop()
    process = make_mock_process([json.dumps(row) for row in rows])
    if fault == "late-exit":
        process.wait.return_value = process.returncode = 7
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "truncation": {
                "type": "object",
                "properties": {"truncated": {"type": "boolean"}, "firstLineExceedsLimit": {"type": "boolean"}},
            },
        },
        "required": ["answer"],
    }
    if fault in {"none", "domain-truncation", "failed-last", "mixed-read", "mixed-read-first"}:
        events, _ = await replay_process(
            PiBackend(model="fixture"), process, Path("/tmp"), "Answer", output_schema=schema
        )
        result = next(event for event in events if isinstance(event, ResultEvent))
        expected = {
            "answer": "first" if fault == "failed-last" else "last",
            **(
                {"truncation": {"truncated": True, "firstLineExceedsLimit": True}}
                if fault == "domain-truncation"
                else {}
            ),
        }
        assert result.structured_output == expected
        assert all(not event.truncated for event in events if isinstance(event, ToolResultEvent))
        assert [event.id for event in events if isinstance(event, ToolStartEvent)] == (
            ["read", "first", "last"] if fault in {"mixed-read", "mixed-read-first"} else ["first", "last"]
        )
    else:
        with pytest.raises(PiError):
            await replay_process(PiBackend(model="fixture"), process, Path("/tmp"), "Answer", output_schema=schema)


async def test_native_output_uses_finalized_transcript_order_and_requires_complete_settlement() -> None:
    for fault in [
        "none",
        "domain-truncation",
        "failed-last",
        "mixed-read",
        "mixed-read-first",
        "null-details",
        "oversized-details",
        "wrong-id",
        "unsettled",
        "late-exit",
    ]:
        await _native_output_uses_finalized_transcript_order_and_requires_complete_settlement_case(fault=fault)


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


async def test_cancel_terminates_then_kills(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daydream.backends._subprocess.TERMINATE_GRACE_S", 0.0)
    backend, proc = make_cancel_probe("pi")
    await backend.cancel()
    assert proc.terminate_calls == 1 and proc.kill_calls == 1
    assert proc.returncode == -9 and proc.reaped


def test_create_backend_invalid_includes_pi_in_message() -> None:
    with pytest.raises(ValueError, match="pi"):
        create_backend("invalid")


# Default --provider is nous; PI_PROVIDER overrides (extension-based provider)


async def test_default_model_does_not_override_pi_settings(pi_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    backend = PiBackend(cwd=pi_workspace)
    assert backend.model == "gpt-5.6-luna"
    flat_args, _ = await _run_and_capture_args(backend, cwd=pi_workspace)
    assert "--model" not in flat_args
    assert "--provider" not in flat_args


async def test_bound_pi_context_preserves_native_configuration_and_auth_refresh(
    pi_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
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


# Migration guards: GLM-pin and provider/model mismatch warnings


async def test_zai_provider_with_fallback_model_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
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


# System prompt preamble (--append-system-prompt)


# PiError.retryable attribute


# pi retry classification


# _is_retryable_exit_code


# Error turn uses retryable classifier


async def _error_turn_sets_retryable_via_classifier_case(
    error_message: str,
    expected_retryable: bool,
    expected_category: str,
) -> None:
    """Public terminal errors retain diagnosis without altering classification or totals."""
    backend = PiBackend(model="glm-5.2")
    lines = [
        '{"type":"session","sessionId":"pi_ses_err"}',
        '{"type":"agent_start"}',
        '{"type":"turn_start"}',
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
    diagnostics = [
        event for event in events if isinstance(event, DiagnosticEvent) and event.code == "pi_terminal_error"
    ]
    assert len(diagnostics) == 1 and diagnostics[0].message == error_message
    assert diagnostics[0].metadata == {
        "category": expected_category,
        "retryable": expected_retryable,
        "finish_reason": "error",
        "raw_stop_reason": "error",
        "message_source": "native",
    }
    assert events.index(diagnostics[0]) < next(i for i, event in enumerate(events) if isinstance(event, TurnEndEvent))
    totals = [event for event in events if isinstance(event, CostEvent)]
    assert len(totals) == 1 and totals[0].input_tokens == 100 and totals[0].output_tokens == 10
    assert totals[0].cost_usd == 0.01
    assert (totals[0].cached_tokens, totals[0].cache_creation_tokens) == (20, 30)
    metric = next(event for event in events if isinstance(event, MetricsEvent))
    assert (metric.model_name, metric.provider_name, metric.usage_scope, metric.cache_creation_tokens) == (
        "native-model",
        "native-provider",
        "message",
        30,
    )
    result = next(event for event in events if isinstance(event, ResultEvent))
    assert (result.model_name, result.provider_name, result.session_id, result.finish_reason, result.continuation) == (
        "native-model",
        "native-provider",
        "pi_ses_err",
        "error",
        None,
    )
    assert mock_proc.returncode is not None


async def test_error_turn_sets_retryable_via_classifier() -> None:
    for error_message, expected_retryable, expected_category in [
        ("request timed out after 30 seconds", True, "TIMEOUT"),
        ("Stream ended without finish_reason", True, "STREAM_TRUNCATION"),
        ("Pi CLI exited with return code 1", False, "PROCESS_EXIT"),
        ("model not found: gpt-5 (503)", False, "AUTH_CONFIG"),
        ("JSON schema validation failed: the model is temporarily at capacity.", False, "SCHEMA"),
        ("capacity exceeded", True, "SERVER_ERROR"),
        ("service is not overloaded", False, "UNKNOWN"),
    ]:
        await _error_turn_sets_retryable_via_classifier_case(error_message, expected_retryable, expected_category)


async def _pi_terminal_diagnostic_is_bounded_and_redacted_before_persistence_case(
    tmp_path: Path,
    native: dict[str, Any],
) -> None:
    """Public optional fields cannot crash error handling or leak cut-off secret fragments."""
    backend = PiBackend(model="fixture-model")
    backend.retry_attempts = 0
    message = {"role": "assistant", "content": [], "stopReason": "error", **native}
    recorder = make_recorder(tmp_path)
    with patch(
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        return_value=make_mock_process(
            [
                json.dumps({"type": "turn_end", "message": message}),
            ]
        ),
    ) as spawn:
        async with recorder:
            with pytest.raises(PiError) as failure:
                await run_agent(
                    backend, tmp_path, "Review", phase=DaydreamPhase.REVIEW, progress_callback=lambda _: None
                )
    assert failure.value.category == "UNKNOWN" and failure.value.retryable is False
    assert spawn.call_count == 1
    document = json.loads(recorder.path.read_text())
    diagnostics = [
        diagnostic
        for step in document["steps"]
        for diagnostic in step.get("extra", {}).get("backend_diagnostics", [])
        if diagnostic["code"] == "pi_terminal_error"
    ]
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


async def test_pi_terminal_diagnostic_is_bounded_and_redacted_before_persistence(tmp_path: Path) -> None:
    natives: list[dict[str, Any]] = [
        {},
        {"errorMessage": None, "rawStopReason": {"not": "a string"}},
        {"errorMessage": ["invalid"], "rawStopReason": 7},
        {
            "errorMessage": "failure: -----BEGIN PRIVATE KEY-----\n"
            + "x" * 5000
            + "\n-----END PRIVATE KEY-----\n"
            + "safe " * 2000,
            "rawStopReason": "TOKEN=hidden-secret; " + "reason " * 1000,
        },
    ]
    for index, native in enumerate(natives):
        await _pi_terminal_diagnostic_is_bounded_and_redacted_before_persistence_case(
            tmp_path=_case_dir(tmp_path, index), native=native
        )


async def _pi_diagnostic_redaction_fault_preserves_native_failure_and_terminal_usage_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    redaction_fault: str,
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
    native = {
        "role": "assistant",
        "stopReason": "error",
        "errorMessage": "Provider finish_reason: error",
        "rawStopReason": "TOKEN=never-persist",
        "usage": {"input": 100, "output": 10, "cost": {"total": 0.01}},
    }
    with patch(
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        return_value=make_mock_process(
            [
                json.dumps({"type": "turn_end", "message": native}),
            ]
        ),
    ) as spawn:
        async with recorder:
            with pytest.raises(PiError) as failure:
                await run_agent(
                    backend, tmp_path, "Review", phase=DaydreamPhase.REVIEW, progress_callback=lambda _: None
                )
    assert failure.value.category == "UNKNOWN" and failure.value.retryable is False
    assert spawn.call_count == 1
    document = json.loads(recorder.path.read_text())
    diagnostics = [
        diagnostic
        for step in document["steps"]
        for diagnostic in step.get("extra", {}).get("backend_diagnostics", [])
        if diagnostic["code"] == "pi_terminal_error"
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0]["message"] == "[DIAGNOSTIC_REDACTION_FAILED]"
    assert diagnostics[0]["metadata"]["raw_stop_reason"] == "[DIAGNOSTIC_REDACTION_FAILED]"
    assert "never-persist" not in json.dumps(document)
    assert document["final_metrics"]["total_prompt_tokens"] == 100
    assert document["final_metrics"]["total_completion_tokens"] == 10
    assert document["final_metrics"]["total_cost_usd"] == 0.01


async def test_pi_diagnostic_redaction_fault_preserves_native_failure_and_terminal_usage(tmp_path: Path) -> None:
    for index, redaction_fault in enumerate(["raises", "nontext"]):
        with pytest.MonkeyPatch.context() as mp:
            await _pi_diagnostic_redaction_fault_preserves_native_failure_and_terminal_usage_case(
                monkeypatch=mp, tmp_path=_case_dir(tmp_path, index), redaction_fault=redaction_fault
            )


async def _pi_response_diagnostics_admit_only_bounded_redacted_fields_case(status: Any) -> None:
    """Optional helper data cannot leak headers or replace a native terminal failure.

    Installed-native journeys exercise ordinary HTTP headers; they cannot emit
    malformed helper frames or put a secret before a truncated header boundary.
    """
    native = {
        "type": "daydream_provider_response",
        "status": status,
        "headers": {
            "x-request-id": "TOKEN=private-response-canary; " + "safe " * 1000,
            "retry-after": {"malformed": True},
            "authorization": "private-auth-canary",
        },
    }
    process = make_mock_process(
        [
            json.dumps(frame)
            for frame in [
                {"type": "session", "id": "diagnostics"},
                native,
                {"type": "message_start", "message": {"role": "assistant"}},
                {
                    "type": "turn_end",
                    "message": {"role": "assistant", "stopReason": "error", "errorMessage": "terminated"},
                },
            ]
        ]
    )
    events = []
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=process):
        with pytest.raises(PiError, match="^terminated$") as failure:
            async for event in PiBackend(model="fixture-model").execute(Path("/tmp"), "Review"):
                events.append(event)
    assert failure.value.category == "STREAM_DROP" and failure.value.retryable
    responses = [
        event for event in events if isinstance(event, DiagnosticEvent) and event.code == "pi_provider_response"
    ]
    assert len(responses) == int(type(status) is int and status == 200)
    if responses:
        metadata = responses[0].metadata
        assert metadata["status"] == 200 and metadata["generation_id"] is not None
        assert metadata["headers_truncated"] is True and set(metadata["headers"]) == {"x-request-id"}
        value = metadata["headers"]["x-request-id"]
        assert len(value) <= 512 and "[REDACTED_ENV_VAR]" in value
        assert "private-response-canary" not in json.dumps(metadata)
        assert "private-auth-canary" not in json.dumps(metadata)


async def test_pi_response_diagnostics_admit_only_bounded_redacted_fields() -> None:
    for status in [None, True, 99, 600, "200", 200]:
        await _pi_response_diagnostics_admit_only_bounded_redacted_fields_case(status=status)


async def test_pi_response_without_generation_is_retained_without_previous_attribution() -> None:
    """An interrupted second request cannot inherit the previous generation's ID.

    Complete native responses always emit message_start; only a partial transport
    replay can isolate headers followed by EOF before that boundary.
    """
    message = {"role": "assistant", "content": [], "stopReason": "stop"}
    process = make_mock_process(
        [
            json.dumps(frame)
            for frame in [
                {"type": "session", "id": "diagnostics"},
                {"type": "turn_start"},
                {"type": "message_start", "message": {"role": "assistant"}},
                {"type": "message_end", "message": message},
                {"type": "turn_end", "message": message},
                {"type": "turn_start"},
                {
                    "type": "daydream_provider_response",
                    "status": 200,
                    "headers": {"x-request-id": "interrupted-request"},
                },
            ]
        ]
    )
    events = []
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=process):
        with pytest.raises(PiError, match="Stream ended without finish_reason"):
            async for event in PiBackend(model="fixture-model").execute(Path("/tmp"), "Review"):
                events.append(event)
    response = next(
        event for event in events if isinstance(event, DiagnosticEvent) and event.code == "pi_provider_response"
    )
    assert response.metadata["generation_id"] is None
    assert response.metadata["headers"] == {"x-request-id": "interrupted-request"}


# Retry env knobs


# fanout_concurrency


def test_pi_retry_policy_is_built_from_ambient_environment(caplog: pytest.LogCaptureFixture) -> None:
    """One actual constructor admits or diagnoses all retry settings from the same captured environment."""
    cases: list[tuple[dict[str, str], RetryPolicy, list[str]]] = [
        (
            {
                "DAYDREAM_PI_RETRY_ATTEMPTS": "-1",
                "DAYDREAM_PI_RETRY_BASE_DELAY_S": "nan",
                "DAYDREAM_PI_RETRY_MAX_DELAY_S": "not-a-float",
            },
            RetryPolicy(20, 10.0, 120.0),
            [
                "DAYDREAM_PI_RETRY_ATTEMPTS='-1' is negative; using default 20",
                "DAYDREAM_PI_RETRY_BASE_DELAY_S='nan' is not finite; using default 10",
                "DAYDREAM_PI_RETRY_MAX_DELAY_S='not-a-float' is not a valid float; using default 120",
            ],
        ),
        (
            {"DAYDREAM_PI_RETRY_ATTEMPTS": "", "DAYDREAM_PI_RETRY_BASE_DELAY_S": "-1"},
            RetryPolicy(20, 10.0, 120.0),
            [
                "DAYDREAM_PI_RETRY_ATTEMPTS='' is not a valid integer; using default 20",
                "DAYDREAM_PI_RETRY_BASE_DELAY_S='-1' is negative; using default 10",
            ],
        ),
    ]
    for environment, expected, warnings in cases:
        caplog.clear()
        with patch.dict(os.environ, environment, clear=True), caplog.at_level("WARNING"):
            backend = PiBackend(model="fixture-model")
        assert (backend.retry_attempts, backend.retry_base_delay_s, backend.retry_max_delay_s) == (
            expected.attempts,
            expected.base_delay_s,
            expected.max_delay_s,
        )
        assert [record.getMessage() for record in caplog.records] == warnings


# --- P18 Task 1: generation lifecycle + config at the Pi argv/JSONL seam -----


@pytest.mark.parametrize(
    "mode",
    [
        "finalization",
    ],
)
async def test_pi_request_event_config_matches_exact_argv(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.delenv("PI_API_KEY", raising=False)
    backend = PiBackend(model="glm-5.2")
    schema = None if mode == "no-schema" else {"type": "object", "properties": {"answer": {"type": "string"}}}
    process = make_mock_process(
        (FIXTURES_DIR / "simple_text.jsonl").read_text().splitlines() + ['{"type":"agent_settled"}']
    )
    if mode == "disabled":
        process.stdin = MagicMock()
    events, spawn = await replay_process(
        backend,
        process,
        Path("/tmp"),
        "structured please",
        output_schema=schema,
        tools_disabled=mode == "disabled",
        finalization=mode == "finalization",
        validate_structured_output=mode != "validation-opt-out",
        read_only=True,
        persist_session=False,
    )
    flat_args = list(spawn.call_args.args)
    request = next(event for event in events if isinstance(event, RequestEvent))
    config = request.config
    assert isinstance(config, PiRequestConfig)
    assert config.read_only is True
    if mode in {"disabled", "finalization"}:
        assert "--no-tools" in flat_args and config.selected_tools_count == 0
    else:
        assert flat_args[flat_args.index("--tools") + 1] == "read,find,ls,grep" + (
            ",structured_output" if mode == "native" else ""
        )
        assert config.selected_tools_count == (5 if mode == "native" else 4)
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


@pytest.mark.parametrize("real_spawn", [False])
async def test_prompt_attachment_removed_after_spawn_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_spawn: bool,
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


async def _prompt_attachment_generator_close_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_spawn: bool,
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    monkeypatch.setattr(
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        AsyncMock(return_value=mock_proc),
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


async def test_prompt_attachment_generator_close(tmp_path: Path) -> None:
    for index, after_spawn in enumerate([False, True]):
        with pytest.MonkeyPatch.context() as mp:
            await _prompt_attachment_generator_close_case(
                monkeypatch=mp, tmp_path=_case_dir(tmp_path, index), after_spawn=after_spawn
            )


async def test_prompt_attachment_removed_even_if_teardown_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        AsyncMock(return_value=make_mock_process_from_fixture("simple_text.jsonl")),
    )
    monkeypatch.setattr("daydream.backends.pi.teardown", AsyncMock(side_effect=RuntimeError("teardown failed")))
    with pytest.raises(RuntimeError, match="teardown failed"):
        async for _ in PiBackend(model="fixture").execute(tmp_path, "teardown prompt"):
            pass
    assert not list(tmp_path.glob("daydream-pi-prompt-*"))


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
    assert [
        (event.model_name, event.provider_name, event.usage_scope)
        for event in events
        if isinstance(event, MetricsEvent)
    ] == [("pi-model", "native-provider", "message")] * 2
    assert [
        (event.session_id, event.model_name, event.provider_name, event.finish_reason)
        for event in events
        if isinstance(event, ResultEvent)
    ] == [("pi-native", "pi-model", "native-provider", "stop")] * 2
    assert [(event.cached_tokens, event.cache_creation_tokens) for event in events if isinstance(event, CostEvent)] == [
        (20, 30),
        (25, None),
    ]
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
        (200, 100, 0.0),
        (100, 50, 0.001),
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
