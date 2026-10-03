"""Hermetic contract tests for the additive Osprey backend boundary."""

from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    CostEvent,
    MetricsEvent,
    OspreyRequestConfig,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
    create_backend,
)
from daydream.backends.osprey import (
    OspreyBackend,
    OspreyConfig,
    OspreyError,
    OspreyProtocolError,
    OspreyTerminalError,
    OspreyUnsupportedOption,
    _optional_non_negative_int,
    _required_non_negative_int,
    _stderr_diagnostic_sink,
)
from daydream.trajectory import DaydreamPhase
from tests.harness.fake_cli_process import FakeCliProcess, FakeCliSpawner
from tests.harness.osprey_jsonl import osprey_session
from tests.harness.protocol_cli import install_protocol_cli
from tests.harness.protocol_cli_assertions import assert_protocol_cli_invariants, make_cancel_probe
from tests.harness.trajectory import make_recorder


@pytest.mark.parametrize("sandbox", [False, True])
async def test_artifact_visibility_protocol_cli_preserves_sandbox_roots_and_terminal_envelope(
    tmp_path: Path, sandbox: bool,
) -> None:
    target = (tmp_path / "model cwd with spaces").resolve()
    target.mkdir()
    (target / "source.py").write_text("SOURCE_CANARY\n", encoding="utf-8")
    fixture = install_protocol_cli(tmp_path / "external fixture", "osprey")
    allowed = str(target / "source.py")
    backend = OspreyBackend(OspreyConfig(
        model="fixture-model", osprey_binary=str(fixture.executable),
        sandbox=sandbox, allowed_roots=[allowed] if sandbox else (),
    ))
    prompt = "Inspect the committed source only."
    events = [event async for event in backend.execute(target, prompt)]
    observation = assert_protocol_cli_invariants(fixture, target, prompt, events, backend)
    argv = observation["argv"]
    assert argv[:2] == ["agent", "--events-jsonl"]
    assert ("--sandbox" in argv) is sandbox
    if sandbox:
        assert argv[argv.index("--allowed-root") + 1] == allowed
    else:
        assert "--allowed-root" not in argv

def _over_limit_reader(payload: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader(limit=64)
    reader.feed_data(payload)
    reader.feed_eof()
    return reader

async def _collect(
    backend: OspreyBackend, lines: list[dict[str, object]], *, returncode: int = 0,
    output_schema: dict[str, Any] | None = None, continuation: ContinuationToken | None = None,
    agents: dict[str, Any] | None = None, max_turns: int | None = None, read_only: bool = False,
    persist_session: bool = True, stderr_lines: list[str] | None = None, stderr_held_open: bool = False,
    stdout_reader: asyncio.StreamReader | None = None, stderr_reader: asyncio.StreamReader | None = None,
    workspace: Path = Path("/repo"),
) -> tuple[list[AgentEvent], FakeCliSpawner]:
    spawner = FakeCliSpawner()
    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        proc = FakeCliProcess(
            [json.dumps(line) for line in lines], exit_code=returncode, stderr_lines=stderr_lines,
            stderr_held_open=stderr_held_open, stdout_reader=stdout_reader, stderr_reader=stderr_reader,
        )
        spawner.procs.append(proc)
        spawner.argvs.append(tuple(str(a) for a in args))
        spawner.kwargs.append(kwargs)
        return proc
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
        events = [
            event
            async for event in backend.execute(
                workspace, "prompt", output_schema=output_schema, continuation=continuation, agents=agents,
                max_turns=max_turns, read_only=read_only, persist_session=persist_session,
            )
        ]
    return events, spawner

async def test_oversized_stdout_line_is_categorized_as_protocol_error() -> None:
    stdout = _over_limit_reader(b"x" * 65)
    backend = OspreyBackend(OspreyConfig(osprey_binary="fake"))
    with pytest.raises(OspreyError, match=r"(?i)stdout.*limit") as exc_info:
        await _collect(backend, [], stdout_reader=stdout)
    assert exc_info.value.category == "PROTOCOL"
    assert backend._transports == []

async def test_non_utf8_stdout_line_is_non_json_protocol_error_not_over_limit() -> None:
    """Replacement decoding may still leave invalid JSON; diagnose that as non-JSON rather than an
    oversized line.
    """
    stdout = _over_limit_reader(b'{"event"' + b"\xff" + b"}\n")
    backend = OspreyBackend(OspreyConfig(osprey_binary="fake"))
    with pytest.raises(OspreyError, match=r"non-JSON line in JSONL mode") as exc_info:
        await _collect(backend, [], stdout_reader=stdout)
    assert exc_info.value.category == "PROTOCOL"
    assert "byte limit" not in str(exc_info.value)
    assert "\ufffd" in str(exc_info.value)
    assert backend._transports == []

async def test_non_utf8_byte_inside_json_event_is_repaired_and_session_completes() -> None:
    lines = osprey_session({"event": "text_delta", "content": "x"})
    payload: list[bytes] = []
    for event in lines:
        raw = json.dumps(event).encode()
        if event.get("event") == "text_delta":
            raw = raw.replace(b'"x"', b'"x\xff"')
        payload.append(raw)
    stdout = asyncio.StreamReader(limit=2_048)
    stdout.feed_data(b"\n".join(payload) + b"\n")
    stdout.feed_eof()
    events, _ = await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), [], stdout_reader=stdout)
    text = next(event for event in events if isinstance(event, TextEvent))
    assert text.text == "x\ufffd"
    assert type(events[-1]) is ResultEvent

async def test_oversized_stderr_diagnostic_does_not_fail_successful_run() -> None:
    lines = osprey_session()
    stderr = _over_limit_reader(b"diagnostic" * 8)
    events, _ = await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), lines, stderr_reader=stderr)
    assert [type(event) for event in events] == [RequestEvent, CostEvent, ResultEvent]

async def test_oversized_stderr_diagnostic_is_reported_on_process_failure() -> None:
    lines = osprey_session()
    stderr = _over_limit_reader(b"provider authentication failed" * 3)
    with pytest.raises(OspreyError, match="stderr diagnostic line exceeded stream limit"):
        await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), lines, returncode=1, stderr_reader=stderr)

async def test_stderr_diagnostics_are_redacted_and_capped_while_draining() -> None:
    secret = "sk-realvalue123"
    diagnostics: list[str] = []
    stderr = asyncio.StreamReader(limit=2_048)
    stderr.feed_data((f"OPENAI_API_KEY={secret} " + "x" * 1_024 + "\n").encode())
    sink = _stderr_diagnostic_sink(diagnostics)
    for _ in range(12):
        sink("OPENAI_API_KEY=" + secret + " " + "x" * 1_024)
    assert len(diagnostics) == 10
    assert len(diagnostics[0]) <= 500
    assert secret not in diagnostics[0]
    assert "[REDACTED_ENV_VAR]" in diagnostics[0]

async def test_stderr_is_drained_separately_from_jsonl_stdout() -> None:
    lines = osprey_session()
    events, spawner = await _collect(
        OspreyBackend(OspreyConfig(osprey_binary="fake")), lines,
        stderr_lines=["2026-08-27T23:07:54Z INFO osprey_cli::headless: " "restoring remembered model preference"],
    )
    assert [type(event) for event in events] == [RequestEvent, CostEvent, ResultEvent]
    assert spawner.kwargs[0]["stderr"] is asyncio.subprocess.PIPE

async def test_execute_spawns_detached_and_reaps_on_success(tmp_path: Path) -> None:
    """Even successful children must be reaped, their pipes closed, and their transport removed."""
    lines = osprey_session()
    backend = OspreyBackend(OspreyConfig(osprey_binary="fake"))
    events, spawner = await _collect(backend, lines, workspace=tmp_path)
    assert [type(event) for event in events] == [RequestEvent, CostEvent, ResultEvent]
    kwargs = spawner.kwargs[0]
    assert kwargs["start_new_session"] is True
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["stderr"] is asyncio.subprocess.PIPE
    assert len(spawner.procs) == 1
    proc = spawner.procs[0]
    assert proc.returncode == 0
    assert proc.reaped, "the transport wait must reap the child even after a clean exit"
    assert proc.stdin.closed, "the transport terminate must close the stdin pipe"
    assert proc._transport.closed, "the transport terminate must release the pipe fds even after a clean exit"
    assert backend._transports == [], "the finally must drop the transport from the backend list"

async def test_process_cleanup_releases_inherited_stderr_before_waiting_for_eof() -> None:
    lines = osprey_session()
    events, _ = await asyncio.wait_for(
        _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), lines, stderr_held_open=True), timeout=0.5,
    )
    assert [type(event) for event in events] == [RequestEvent, CostEvent, ResultEvent]

def test_factory_builds_verified_osprey_jsonl_command() -> None:
    backend = create_backend("osprey", model="test-model", osprey_binary="fake-osprey")
    assert isinstance(backend, OspreyBackend)
    assert backend.model == "test-model"
    assert backend.config.prepare("hello")[0] == [
        "fake-osprey", "agent", "--events-jsonl", "--observation-budget-update-bytes", "65536",
        "--observation-budget-inline-bytes", "262144", "--observation-budget-admission-bytes", "2097152", "--model",
        "test-model", "hello",
    ]

async def test_translates_text_thinking_tool_identity_metrics_and_result() -> None:
    lines = osprey_session(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
        {"event": "thinking_delta", "content": "checking"}, {"event": "text_delta", "content": "done"},
        {"event": "tool_call", "tool_call_id": "c-1", "tool_name": "tool_search", "arguments": {"query": "MCP"}},
        {
            "event": "tool_result", "tool_call_id": "c-1", "tool_name": "mcp.fetch", "status": "success",
            "content": "payload", "duration_ms": 4,
        },
        {
            "event": "turn_end", "turn_id": "t-1", "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": None,
            "usage_reported": False, "duration_ms": 4,
        },
    )
    events, _ = await _collect(OspreyBackend(OspreyConfig(model="custom-model", osprey_binary="fake")), lines)
    assert [type(event) for event in events] == [
        RequestEvent, ThinkingEvent, TextEvent, ToolStartEvent, ToolResultEvent, TurnEndEvent, CostEvent, ResultEvent,
    ]
    tool_start = next(event for event in events if isinstance(event, ToolStartEvent))
    tool_result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert tool_start.name == "tool_search"
    assert tool_start.id == "c-1"
    assert tool_result.id == "c-1"
    assert tool_result.output == "payload"
    assert tool_result.is_error is False
    assert not any(isinstance(event, MetricsEvent) for event in events)

async def test_coalesces_streaming_thinking_deltas_before_text() -> None:
    lines = osprey_session(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"}, {"event": "thinking_delta", "content": "Let"},
        {"event": "thinking_delta", "content": " me"}, {"event": "thinking_delta", "content": " think."},
        {"event": "text_delta", "content": "Done."}, {"event": "turn_end", "turn_id": "t-1", "usage_reported": False},
    )
    events, _ = await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), lines)
    assert [type(event) for event in events] == [
        RequestEvent, ThinkingEvent, TextEvent, TurnEndEvent, CostEvent, ResultEvent,
    ]
    assert [event.text for event in events if isinstance(event, ThinkingEvent)] == ["Let me think."]

async def test_ignores_blank_thinking_deltas() -> None:
    lines = osprey_session(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"}, {"event": "thinking_delta", "content": ""},
        {"event": "thinking_delta", "content": "  \n"},
        {"event": "text_delta", "content": "Done."}, {"event": "turn_end", "turn_id": "t-1", "usage_reported": False},
    )
    events, _ = await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), lines)
    assert not any(isinstance(event, ThinkingEvent) for event in events)

async def test_result_exposes_session_model_without_usage_metrics() -> None:
    lines = osprey_session(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"}, {"event": "text_delta", "content": "Done."},
        {"event": "turn_end", "turn_id": "t-1", "usage_reported": False},
    )
    backend = OspreyBackend(OspreyConfig(osprey_binary="fake"))
    assert backend.model == "unknown"
    events, _ = await _collect(backend, lines)
    result = next(event for event in events if isinstance(event, ResultEvent))
    assert result.model_name == "custom-model"
    assert backend.model == "custom-model"

async def test_separates_thinking_runs_at_protocol_boundaries() -> None:
    lines = osprey_session(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
        {"event": "thinking_delta", "content": "first attempt"}, {"event": "driver_retry"},
        {"event": "thinking_delta", "content": "second attempt"}, {"event": "text_delta", "content": "Done."},
        {"event": "turn_end", "turn_id": "t-1", "usage_reported": False},
    )
    events, _ = await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), lines)
    assert [event.text for event in events if isinstance(event, ThinkingEvent)] == ["first attempt", "second attempt"]

async def test_trajectory_records_coalesced_thinking_as_prose(tmp_path: Path) -> None:
    lines = osprey_session(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"}, {"event": "thinking_delta", "content": "Let"},
        {"event": "thinking_delta", "content": " me"}, {"event": "thinking_delta", "content": " think."},
        {"event": "message_end", "messages": [{"type": "result", "data": {"content": "Done."}}]},
        {"event": "turn_end", "turn_id": "t-1", "usage_reported": False},
    )
    recorder = make_recorder(
        tmp_path, path=tmp_path / "trajectory.json", agent_model_name="osprey", session_id="daydream-session",
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            invocation.observe_user_step("prompt")
            events, _ = await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), lines)
            for event in events:
                invocation.observe(event)
    trajectory = recorder.build_trajectory().model_dump(exclude_none=True)
    assert trajectory["steps"][1]["reasoning_content"] == "Let me think."
    assert trajectory["agent"]["model_name"] == "custom-model"
    assert trajectory["steps"][1]["model_name"] == "custom-model"

async def test_usage_and_structured_output_preserve_optional_metrics() -> None:
    lines = osprey_session(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
        {
            "event": "turn_end", "turn_id": "t-1", "prompt_tokens": 12, "completion_tokens": 7, "cached_tokens": None,
            "cache_write_tokens": None, "usage_reported": True, "provider": "openai-compatible",
            "model": "custom-model", "duration_ms": 0, "thinking_tokens": 3, "tool_calls": 0, "cost_usd": "0.125",
        },
    )
    lines[-1]["structured_output"] = {"ok": True}
    events, _ = await _collect(OspreyBackend(OspreyConfig(model="custom-model", osprey_binary="fake")), lines)
    metrics = next(event for event in events if isinstance(event, MetricsEvent))
    result = next(event for event in events if isinstance(event, ResultEvent))
    assert metrics.prompt_tokens == 12
    assert metrics.completion_tokens == 7
    assert metrics.reasoning_tokens == 3
    assert metrics.cached_tokens is None
    assert metrics.cost_usd == pytest.approx(0.125)
    assert result.structured_output == {"ok": True}
    assert result.continuation is not None
    assert result.continuation.data["session_id"] == "s-137"
    assert result.continuation.data == {
        "session_id": "s-137", "provider": "openai-compatible", "model": "custom-model", "outcome": "completed",
        "exit_code": 0,
    }

async def test_unreported_usage_permits_omitted_optional_telemetry() -> None:
    lines = osprey_session(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
        {"event": "turn_end", "turn_id": "t-1", "usage_reported": False},
    )
    events, _ = await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), lines)
    assert [event.message_id for event in events if isinstance(event, TurnEndEvent)] == ["t-1"]
    assert not any(isinstance(event, MetricsEvent) for event in events)

async def test_message_end_reconstructs_result_when_no_text_delta_arrives() -> None:
    lines = osprey_session(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
        {"event": "message_end", "messages": [{"type": "result", "data": {"content": "from message_end"}}]},
        {
            "event": "turn_end", "turn_id": "t-1", "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": None,
            "usage_reported": False, "duration_ms": 0,
        },
    )
    events, _ = await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), lines)
    assert [event.text for event in events if isinstance(event, TextEvent)] == ["from message_end"]

async def test_protocol_version_and_unknown_events_fail_closed() -> None:
    backend = OspreyBackend(OspreyConfig(osprey_binary="fake"))
    bad_version = [{"event": "protocol", "version": 99}]
    with pytest.raises(OspreyProtocolError, match="unsupported Osprey JSONL protocol version"):
        await _collect(backend, bad_version)
    unknown: list[dict[str, object]] = [
        {"event": "protocol", "version": 2},
        {
            "event": "session_start", "session_id": "s", "started_at": "2026-08-15T00:00:00Z", "model": "m",
            "provider": "p",
        }, {"event": "not-a-real-event"},
    ]
    with pytest.raises(OspreyProtocolError, match="unknown Osprey JSONL event"):
        await _collect(backend, unknown)

def test_command_forwards_verified_policy_resume_fork_and_schema_flags(tmp_path: Path) -> None:
    backend = OspreyBackend(OspreyConfig(
        model="m", osprey_binary="fake", approval_mode="deny-untrusted", sandbox=True, allowed_roots=[tmp_path],
    ))
    command, _request = backend.config.prepare(
        "prompt", output_schema_path=tmp_path / "schema.json",
        continuation=ContinuationToken("osprey", {"session_id": "s", "mode": "fork"}), max_turns=3, read_only=True,
    )
    assert command[:11] == [
        "fake", "agent", "--events-jsonl", "--observation-budget-update-bytes", "65536",
        "--observation-budget-inline-bytes", "262144", "--observation-budget-admission-bytes", "2097152", "--model",
        "m",
    ]
    assert "--read-only" in command
    assert command[command.index("--fork-from") + 1] == "s"
    assert command[command.index("--output-schema") + 1].endswith("schema.json")
    assert command[command.index("--max-turns") + 1] == "3"
    assert command[command.index("--approval") + 1] == "deny-untrusted"

async def test_output_schema_is_temp_file_forwarded_and_cleaned() -> None:
    lines = osprey_session()
    _, spawner = await _collect(
        OspreyBackend(OspreyConfig(osprey_binary="fake")), lines, output_schema={"type": "object"},
    )
    command = list(spawner.argvs[0])
    schema_path = Path(command[command.index("--output-schema") + 1])
    assert not schema_path.exists()

async def test_output_schema_temp_file_is_cleaned_when_serialization_fails(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.touch()
    handle = MagicMock()
    handle.name = str(schema_path)
    with (
        patch("daydream.backends._transport.tempfile.NamedTemporaryFile", return_value=handle),
        patch("daydream.backends._transport.json.dump", side_effect=TypeError("not serializable")),
        pytest.raises(TypeError, match="not serializable"),
    ):
        await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), [], output_schema={"type": object})
    assert not schema_path.exists()

def test_unsupported_policy_and_tool_search_options_fail_closed() -> None:
    with pytest.raises(OspreyUnsupportedOption, match="interactive approver"):
        OspreyConfig(approval_mode=cast(Any, "on-request")).prepare("prompt")
    with pytest.raises(OspreyUnsupportedOption, match="no corresponding flag"):
        OspreyConfig().prepare("prompt", tool_search_mode="off")
    with pytest.raises(OspreyUnsupportedOption, match="ephemeral-session"):
        OspreyConfig().prepare("prompt", persist_session=False)

async def test_non_success_terminal_outcome_is_not_reported_as_success() -> None:
    lines = osprey_session()
    lines[-1]["outcome"] = "budget_expired"
    with pytest.raises(OspreyTerminalError, match="budget_expired") as exc_info:
        await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), lines)
    assert exc_info.value.outcome == "budget_expired"

async def test_non_success_terminal_outcome_with_nonzero_process_exit_is_process_failure() -> None:
    lines = osprey_session()
    lines[-1]["outcome"] = "budget_expired"
    with pytest.raises(OspreyError, match="return code 1") as exc_info:
        await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), lines, returncode=1)
    assert exc_info.value.category == "PROCESS_EXIT"

async def test_nonzero_process_exit_includes_stderr_diagnostics() -> None:
    lines = osprey_session()
    with pytest.raises(OspreyError, match="provider authentication failed"):
        await _collect(
            OspreyBackend(OspreyConfig(osprey_binary="fake")), lines, returncode=1,
            stderr_lines=["provider authentication failed"],
        )

@pytest.mark.parametrize(
    "events, message",
    [
        ([{"event": "turn_start", "turn_id": "t-1", "timestamp": "now"}], "active turn"),
        (
            [{ "event": "tool_call", "tool_call_id": "c-1", "tool_name": "tool_search", "arguments": {}, }],
            "pending tool calls",
        ),
    ],
)
async def test_successful_session_end_requires_a_quiescent_stream(
    events: list[dict[str, object]], message: str,
) -> None:
    lines = osprey_session(*events)
    with pytest.raises(OspreyError, match=message):
        await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), lines)

async def test_terminal_exit_code_must_match_process_status() -> None:
    lines = osprey_session()
    lines[-1]["exit_code"] = 1
    with pytest.raises(OspreyError, match="exit_code"):
        await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), lines)

def test_non_negative_int_helpers_reject_below_zero() -> None:
    event = {"event": "turn_end", "k": -1}
    with pytest.raises(OspreyProtocolError, match="has negative 'k'"):
        _required_non_negative_int(event, "k")
    with pytest.raises(OspreyProtocolError, match="has negative 'k'"):
        _optional_non_negative_int(event, "k")
    assert _optional_non_negative_int({"event": "turn_end", "k": None}, "k") is None
    assert _required_non_negative_int({"event": "turn_end", "k": 0}, "k") == 0
    assert _optional_non_negative_int({"event": "turn_end", "k": 0}, "k") == 0

async def test_negative_session_total_cost_is_rejected() -> None:
    lines = osprey_session()
    lines[-1]["total_cost_usd"] = "-0.125"
    with pytest.raises(OspreyError, match="total_cost_usd"):
        await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), lines)

@pytest.mark.parametrize(
    "events, field",
    [
        ([{"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
          {"event": "turn_end", "turn_id": "t-1", "usage_reported": True,
           "duration_ms": -1, "prompt_tokens": 0, "completion_tokens": 0}], "duration_ms"),
        ([{"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
          {"event": "turn_end", "turn_id": "t-1", "usage_reported": False, "duration_ms": -1}], "duration_ms"),
        ([{"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
          {"event": "turn_end", "turn_id": "t-1", "usage_reported": True,
           "duration_ms": 0, "prompt_tokens": -1, "completion_tokens": 0}], "prompt_tokens"),
        ([{"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
          {"event": "turn_end", "turn_id": "t-1", "usage_reported": True,
           "duration_ms": 0, "prompt_tokens": 0, "completion_tokens": -1}], "completion_tokens"),
        ([{"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
          {"event": "turn_end", "turn_id": "t-1", "usage_reported": True,
           "duration_ms": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": -1}], "cached_tokens"),
        ([{"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
          {"event": "turn_end", "turn_id": "t-1", "usage_reported": True,
           "duration_ms": 0, "prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": -1}], "thinking_tokens"),
        ([{"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
          {"event": "turn_end", "turn_id": "t-1", "usage_reported": True,
           "duration_ms": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": "-0.125"}], "cost_usd"),
    ],
)
async def test_negative_osprey_telemetry_is_rejected(events: list[dict[str, object]], field: str) -> None:
    lines = osprey_session(*events)
    with pytest.raises(OspreyError, match=field):
        await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), lines)

@pytest.mark.parametrize(
    "events, message",
    [
        (
            [
                {
                    "event": "turn_end", "turn_id": "t-1", "prompt_tokens": 0, "completion_tokens": 0,
                    "usage_reported": False, "duration_ms": 0,
                }
            ], "turn_end without turn_start",
        ),
        (
            [
                {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
                {"event": "turn_start", "turn_id": "t-2", "timestamp": "now"},
            ], "turn_start before prior turn_end",
        ),
    ],
)
async def test_invalid_turn_event_order_fails_closed(events: list[dict[str, object]], message: str) -> None:
    lines = osprey_session(*events)
    with pytest.raises(OspreyProtocolError, match=message):
        await _collect(OspreyBackend(OspreyConfig(osprey_binary="fake")), lines)

async def test_trajectory_preserves_tool_identity(tmp_path: Path) -> None:
    lines = osprey_session(
        {"event": "tool_call", "tool_call_id": "c-2", "tool_name": "mcp.fetch", "arguments": {"id": "7"}},
        {
            "event": "tool_result", "tool_call_id": "c-2", "tool_name": "mcp.fetch", "status": "success",
            "content": "payload", "duration_ms": 1,
        },
    )
    recorder = make_recorder(
        tmp_path, path=tmp_path / "trajectory.json", agent_model_name="osprey", session_id="daydream-session",
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            invocation.observe_user_step("prompt")
            events, _ = await _collect(OspreyBackend(OspreyConfig(model="requested", osprey_binary="fake")), lines)
            for event in events:
                invocation.observe(event)
    trajectory = recorder.build_trajectory().model_dump(exclude_none=True)
    tool_call = trajectory["steps"][1]["tool_calls"][0]
    assert tool_call["function_name"] == "mcp.fetch"
    assert tool_call["tool_call_id"] == "c-2"
    assert trajectory["steps"][1]["observation"]["results"][0]["source_call_id"] == "c-2"
    assert "cost_usd" not in trajectory["final_metrics"]

async def test_cancel_delegates_to_shared_transport_lifecycle() -> None:
    backend, proc = make_cancel_probe("osprey")
    await backend.cancel()
    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()

# --- P18 Task 1: effective request-config admission at the Osprey argv seam --

def _p18_osprey_events() -> list[dict[str, object]]:
    """Session events for a temperature-0, persona+toolset Osprey run."""
    return [
        {"event": "turn_start", "turn_id": "turn-1", "timestamp": "2026-08-15T00:00:01Z"},
        {"event": "text_delta", "content": "T0 answer"},
        {
            "event": "turn_end", "turn_id": "turn-1", "usage_reported": True,
            "duration_ms": 10, "prompt_tokens": 5, "completion_tokens": 3, "model": "custom-model",
        },
    ]

async def test_osprey_request_event_temperature_and_label_presence_rules() -> None:
    """Explicit 0.0 is admitted; labels reduce to presence; mode closed values."""
    backend = OspreyBackend(OspreyConfig(
        model="custom-model", osprey_binary="fake", temperature=0.0, persona="PATTERN_PERSONA_PLACEHOLDER",
        toolset="TOOLSET_PLACEHOLDER", approval_mode="deny-untrusted",
        sandbox=True, immutable_surface=True, compress_context=False,
        ultracode=True, max_subagents=3, llm_rpm=0, vars=(("k", "v"), ("k2", "v2")),
    ))
    events, spawner = await _collect(backend, osprey_session(*_p18_osprey_events()))
    argv = list(spawner.argvs[-1])
    request = next(e for e in events if isinstance(e, RequestEvent))
    config = request.config
    assert isinstance(config, OspreyRequestConfig)
    assert config.temperature == 0.0
    assert argv[argv.index("--temperature") + 1] == "0.0"
    # Arbitrary labels belong in argv; telemetry records only their presence.
    assert argv[argv.index("--persona") + 1] == "PATTERN_PERSONA_PLACEHOLDER"
    assert argv[argv.index("--toolset") + 1] == "TOOLSET_PLACEHOLDER"
    assert config.persona_present is True and config.toolset_present is True
    assert not hasattr(config, "persona") and not hasattr(config, "toolset")
    assert config.approval_mode == "deny-untrusted"
    assert config.sandbox is True
    assert config.immutable_surface is True
    assert config.compress_context is False
    assert config.ultracode is True
    assert config.max_subagents == 3
    assert config.llm_rpm == 0  # zero retained
    assert config.vars_count == 2
    assert config.continuation_mode == "fresh"
    assert config.model_mode == "single"
    assert request.model_source == "native"
    assert request.provider_source == "native"
    assert request.session_source == "native"
    assert request.timestamp_source == "native"

@pytest.mark.parametrize("temperature", [None, 0.7])
async def test_osprey_temperature_telemetry_matches_explicit_option(temperature: float | None) -> None:
    backend = OspreyBackend(OspreyConfig(model="custom-model", osprey_binary="fake", temperature=temperature))
    events, spawner = await _collect(backend, osprey_session(*_p18_osprey_events()))
    argv = list(spawner.argvs[-1])
    if temperature is None:
        assert "--temperature" not in argv
    else:
        assert argv[argv.index("--temperature") + 1] == "0.7"
    request = next(e for e in events if isinstance(e, RequestEvent))
    config = request.config
    assert isinstance(config, OspreyRequestConfig)
    if temperature is None:
        assert config.temperature is None
    else:
        assert config.temperature == 0.7

async def test_osprey_turn_end_model_override_is_native() -> None:
    backend = OspreyBackend(OspreyConfig(model="custom-model", osprey_binary="fake"))
    events, _spawner = await _collect(backend, osprey_session(*_p18_osprey_events()))
    turn_ends = [e for e in events if isinstance(e, TurnEndEvent)]
    assert len(turn_ends) == 1
    turn_end = turn_ends[0]
    assert turn_end.message_id == "turn-1"
    assert turn_end.model_name == "custom-model"
    assert turn_end.provider_name == "openai-compatible"
    assert turn_end.model_source == "native"
    assert turn_end.provider_source == "native"
    # Session outcome is not a model finish reason — stays unset.
    assert turn_end.finish_reason is None

async def test_osprey_session_end_usage_is_session_sourced() -> None:
    backend = OspreyBackend(OspreyConfig(model="custom-model", osprey_binary="fake"))
    events, _spawner = await _collect(backend, osprey_session(*_p18_osprey_events()))
    costs = [e for e in events if isinstance(e, CostEvent)]
    assert len(costs) == 1
    assert costs[0].measurement_source == "session"
    # The base fixture session_end carries no cost -> no provenance claim.
    assert costs[0].cost_source is None

async def test_osprey_resume_and_fork_continuation_modes() -> None:
    for mode, flag in (("resume", "--resume"), ("fork", "--fork-from")):
        backend = OspreyBackend(OspreyConfig(model="custom-model", osprey_binary="fake"))
        token = ContinuationToken(backend="osprey", data={"session_id": "s-9", "mode": mode})
        events, spawner = await _collect(backend, osprey_session(*_p18_osprey_events()), continuation=token)
        argv = list(spawner.argvs[-1])
        assert flag in argv
        assert argv[argv.index(flag) + 1] == "s-9"  # token session drives argv
        request = next(e for e in events if isinstance(e, RequestEvent))
        config = request.config
        assert isinstance(config, OspreyRequestConfig)
        assert config.continuation_mode == mode
        # Native handshake identity wins over the configured resume token.
        assert request.session_id == "s-137"
        assert request.session_source == "native"


async def test_native_config_private_fields_never_enter_serialized_request_metadata(tmp_path: Path) -> None:
    """Native CLI inputs are retained, while serialized metadata gets the exact safe schema."""
    secret = "OSPREY_PRIVATE_CONFIG_CANARY"
    native = OspreyConfig(
        model=secret + "_model", reasoning_effort=secret + "_thinking",
        osprey_binary=secret + "_binary", persona=secret + "_persona", toolset=secret + "_toolset",
        allowed_roots=[tmp_path / secret / "root"], atif_output=tmp_path / secret / "trajectory.json",
        atif_system_prompt_plaintext=True, tool_result_raw_dir=tmp_path / secret / "raw",
        vars=[(secret + "_key", secret + "_value")], effort=secret + "_effort",
        osprey_home=tmp_path / secret / "home",
    )
    events, spawner = await _collect(OspreyBackend(native), osprey_session())
    request = next(event for event in events if isinstance(event, RequestEvent))
    assert type(request.config) is OspreyRequestConfig
    assert secret not in json.dumps(asdict(request.config))
    assert request.config.persona_present and request.config.toolset_present
    assert request.config.vars_count == 1
    argv = spawner.argvs[0]
    assert argv[0] == native.osprey_binary
    for flag, value in (
        ("--model", native.model), ("--persona", native.persona), ("--toolset", native.toolset),
        ("--allowed-root", str(tmp_path / secret / "root")), ("--atif-output", str(native.atif_output)),
        ("--tool-result-raw-dir", str(native.tool_result_raw_dir)),
        ("--var", secret + "_key=" + secret + "_value"), ("--effort", native.effort),
    ):
        assert argv[argv.index(flag) + 1] == value
    assert spawner.kwargs[0]["env"]["OSPREY_HOME"] == str(native.osprey_home)


def test_native_config_freezes_sequences_and_separates_invocation_metadata(tmp_path: Path) -> None:
    roots = [tmp_path / "initial"]
    variables = [("initial", "value")]
    config = OspreyConfig(allowed_roots=roots, vars=variables, sandbox=True)
    roots.append(tmp_path / "later")
    variables.append(("later", "value"))
    first_argv, first = config.prepare(
        "first", max_turns=3, read_only=True,
        continuation=ContinuationToken("osprey", {"session_id": "first-session", "mode": "fork"}),
    )
    second_argv, second = config.prepare("second", max_turns=7)
    assert first is not second and type(first) is type(second) is OspreyRequestConfig
    assert first.max_turns == 3 and first.read_only and first.continuation_mode == "fork"
    assert second.max_turns == 7 and not second.read_only and second.continuation_mode == "fresh"
    assert first.vars_count == second.vars_count == 1
    assert str(tmp_path / "later") not in first_argv + second_argv
    assert "later=value" not in first_argv + second_argv
    with pytest.raises(FrozenInstanceError):
        config.sandbox = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.max_turns = 99  # type: ignore[misc]


async def test_native_model_identity_does_not_replace_requested_model_on_next_invocation() -> None:
    backend = OspreyBackend(OspreyConfig(model="requested-model", osprey_binary="fake"))
    first, _ = await _collect(backend, osprey_session())
    assert backend.model == "custom-model"
    second, spawner = await _collect(backend, osprey_session())
    argv = spawner.argvs[0]
    assert argv[argv.index("--model") + 1] == "requested-model"
    assert next(event for event in first if isinstance(event, RequestEvent)).config is not next(
        event for event in second if isinstance(event, RequestEvent)
    ).config


async def test_foreign_continuation_is_fresh_in_both_argv_and_request_metadata() -> None:
    events, spawner = await _collect(
        OspreyBackend(OspreyConfig(osprey_binary="fake")), osprey_session(),
        continuation=ContinuationToken("pi", {"session_id": "foreign", "mode": "fork"}),
    )
    assert "--resume" not in spawner.argvs[0] and "--fork-from" not in spawner.argvs[0]
    request = next(event for event in events if isinstance(event, RequestEvent))
    assert request.config.continuation_mode == "fresh"


@pytest.mark.parametrize("option", ["provider", "base_url"])
def test_unmapped_native_config_options_fail_closed(option: str) -> None:
    with pytest.raises(OspreyUnsupportedOption, match=option):
        if option == "provider":
            OspreyConfig(provider="unsupported")
        else:
            OspreyConfig(base_url="unsupported")


def test_scalar_native_controls_preserve_cli_group_order_and_request_values() -> None:
    config = OspreyConfig(
        osprey_binary="fake", turn_timeout=0, stream_idle_timeout_secs=1, streaming_timeout_secs=2,
        empty_completion_threshold=3, driver_max_retries=4, compress_context=False, compress_min_bytes=5,
        tool_result_cap=6, tool_result_head=7, tool_result_tail=8, tool_result_max_lines=9,
        retry_failure_threshold=10, no_progress_family_threshold=11, no_progress_family_window=12,
        no_progress_artifact_threshold=13, no_progress_suppression_window=14, max_subagents=15, llm_rpm=16,
    )
    argv, request = config.prepare("prompt", max_turns=19, read_only=True)
    assert argv[9:] == [
        "--max-turns", "19", "--turn-timeout", "0", "--stream-idle-timeout-secs", "1",
        "--streaming-timeout-secs", "2", "--empty-completion-threshold", "3", "--driver-max-retries", "4",
        "--read-only", "--compress-context=false", "--compress-min-bytes", "5", "--tool-result-cap", "6",
        "--tool-result-head", "7", "--tool-result-tail", "8", "--tool-result-max-lines", "9",
        "--retry-failure-threshold", "10", "--no-progress-family-threshold", "11", "--no-progress-family-window", "12",
        "--no-progress-artifact-threshold", "13", "--no-progress-suppression-window", "14",
        "--max-subagents", "15", "--llm-rpm", "16", "prompt",
    ]
    assert (request.turn_timeout, request.stream_idle_timeout_secs, request.streaming_timeout_secs,
            request.empty_completion_threshold, request.driver_max_retries, request.compress_min_bytes,
            request.tool_result_cap, request.tool_result_head, request.tool_result_tail, request.tool_result_max_lines,
            request.retry_failure_threshold, request.no_progress_family_threshold, request.no_progress_family_window,
            request.no_progress_artifact_threshold, request.no_progress_suppression_window,
            request.max_subagents, request.llm_rpm) == tuple(range(17))
