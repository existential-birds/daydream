"""Hermetic contract tests for the additive Osprey backend boundary."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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
    OspreyError,
    OspreyProtocolError,
    OspreyTerminalError,
    OspreyUnsupportedOption,
    _optional_non_negative_int,
    _required_non_negative_int,
    _stderr_diagnostic_sink,
)
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder
from tests.harness.fake_cli_process import FakeCliProcess, FakeCliSpawner


@pytest.mark.asyncio
@pytest.mark.parametrize("sandbox", [False, True])
async def test_artifact_visibility_protocol_cli_preserves_sandbox_roots_and_terminal_envelope(
    tmp_path: Path, sandbox: bool,
) -> None:
    import hashlib

    from tests.harness.protocol_cli import install_protocol_cli

    target = (tmp_path / "model cwd with spaces").resolve()
    target.mkdir()
    (target / "source.py").write_text("SOURCE_CANARY\n", encoding="utf-8")
    fixture = install_protocol_cli(tmp_path / "external fixture", "osprey")
    allowed = str(target / "source.py")
    backend = OspreyBackend(
        model="fixture-model", osprey_binary=str(fixture.executable),
        sandbox=sandbox, allowed_roots=[allowed] if sandbox else (),
    )
    prompt = "Inspect the committed source only."

    events = [event async for event in backend.execute(target, prompt)]

    observation = fixture.read_observations()[0]
    assert observation["effective_cwd"] == str(target)
    assert observation["inherited_cwd"] == str(target)
    assert observation["stdin_bytes"] == 0
    assert observation["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert observation["cwd_canaries"]["SOURCE_CANARY"] is True
    assert observation["walk_truncated"] is False
    argv = observation["argv"]
    assert argv[:2] == ["agent", "--events-jsonl"]
    assert ("--sandbox" in argv) is sandbox
    if sandbox:
        assert argv[argv.index("--allowed-root") + 1] == allowed
    else:
        assert "--allowed-root" not in argv
    assert any(isinstance(event, TextEvent) and event.text == "CURRENT_REASONING_CANARY" for event in events)
    assert len([event for event in events if isinstance(event, ResultEvent)]) == 1
    assert backend._transports == []


def _stream(
    *events: dict[str, object], returncode: int = 0
) -> tuple[list[dict[str, object]], int]:
    return [
        {"event": "protocol", "version": 2},
        {
            "event": "session_start",
            "session_id": "s-137",
            "started_at": "2026-08-15T00:00:00Z",
            "model": "custom-model",
            "provider": "openai-compatible",
        },
        *events,
        {
            "event": "session_end",
            "total_turns": 1,
            "session_wallclock_ms": 15,
            "total_cost_usd": None,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cached_tokens": None,
            "total_cache_write_tokens": None,
            "total_thinking_tokens": 0,
            "total_oom_kills": 0,
            "p50_turn_ms": 15,
            "p99_turn_ms": 15,
            "avg_turn_cost_usd": None,
            "structured_output": None,
            "outcome": "completed",
            "verification": None,
            "exit_code": 0,
        },
    ], returncode


def _over_limit_reader(payload: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader(limit=64)
    reader.feed_data(payload)
    reader.feed_eof()
    return reader


async def _collect(
    backend: OspreyBackend,
    lines: list[dict[str, object]],
    *,
    returncode: int = 0,
    output_schema: dict[str, Any] | None = None,
    continuation: ContinuationToken | None = None,
    agents: dict[str, Any] | None = None,
    max_turns: int | None = None,
    read_only: bool = False,
    persist_session: bool = True,
    stderr_lines: list[str] | None = None,
    stderr_held_open: bool = False,
    stdout_reader: asyncio.StreamReader | None = None,
    stderr_reader: asyncio.StreamReader | None = None,
) -> tuple[list[AgentEvent], FakeCliSpawner]:
    spawner = FakeCliSpawner()

    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        proc = FakeCliProcess(
            [json.dumps(line) for line in lines],
            exit_code=returncode,
            stderr_lines=stderr_lines,
            stderr_held_open=stderr_held_open,
            stdout_reader=stdout_reader,
            stderr_reader=stderr_reader,
        )
        spawner.procs.append(proc)
        spawner.argvs.append(tuple(str(a) for a in args))
        spawner.kwargs.append(kwargs)
        return proc

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
        events = [
            event
            async for event in backend.execute(
                Path("/repo"),
                "prompt",
                output_schema=output_schema,
                continuation=continuation,
                agents=agents,
                max_turns=max_turns,
                read_only=read_only,
                persist_session=persist_session,
            )
        ]
    return events, spawner


@pytest.mark.asyncio
async def test_oversized_stdout_line_is_categorized_as_protocol_error() -> None:
    stdout = _over_limit_reader(b"x" * 65)
    backend = OspreyBackend(osprey_binary="fake")

    with pytest.raises(OspreyError, match=r"(?i)stdout.*limit") as exc_info:
        await _collect(
            backend,
            [],
            stdout_reader=stdout,
        )

    assert exc_info.value.category == "PROTOCOL"
    assert backend._transports == []


@pytest.mark.asyncio
async def test_non_utf8_stdout_line_is_non_json_protocol_error_not_over_limit() -> None:
    """A non-UTF-8 byte must surface as a non-JSON line, not an over-limit line.

    Osprey decodes stdout with errors="replace" (its historical contract), so
    a 0xff byte in a line that still fails JSON parsing after repair is
    diagnosed as non-JSON — never mislabeled as the 10485760-byte-limit error.
    """
    stdout = _over_limit_reader(b'{"event"' + b"\xff" + b"}\n")
    backend = OspreyBackend(osprey_binary="fake")

    with pytest.raises(OspreyError, match=r"non-JSON line in JSONL mode") as exc_info:
        await _collect(backend, [], stdout_reader=stdout)

    assert exc_info.value.category == "PROTOCOL"
    assert "byte limit" not in str(exc_info.value)
    assert "\ufffd" in str(exc_info.value)
    assert backend._transports == []


@pytest.mark.asyncio
async def test_non_utf8_byte_inside_json_event_is_repaired_and_session_completes() -> None:
    """A non-UTF-8 byte inside a JSON string is repaired with U+FFFD and the
    event is processed normally (the historical errors="replace" contract),
    never hard-failed as a protocol error.
    """
    lines, _ = _stream({"event": "text_delta", "content": "x"})
    payload: list[bytes] = []
    for event in lines:
        raw = json.dumps(event).encode()
        if event.get("event") == "text_delta":
            raw = raw.replace(b'"x"', b'"x\xff"')
        payload.append(raw)
    stdout = asyncio.StreamReader(limit=2_048)
    stdout.feed_data(b"\n".join(payload) + b"\n")
    stdout.feed_eof()

    events, _ = await _collect(
        OspreyBackend(osprey_binary="fake"),
        [],
        stdout_reader=stdout,
    )

    text = next(event for event in events if isinstance(event, TextEvent))
    assert text.text == "x\ufffd"
    assert type(events[-1]) is ResultEvent


@pytest.mark.asyncio
async def test_oversized_stderr_diagnostic_does_not_fail_successful_run() -> None:
    lines, _ = _stream()
    stderr = _over_limit_reader(b"diagnostic" * 8)

    events, _ = await _collect(
        OspreyBackend(osprey_binary="fake"),
        lines,
        stderr_reader=stderr,
    )

    assert [type(event) for event in events] == [RequestEvent, CostEvent, ResultEvent]


@pytest.mark.asyncio
async def test_oversized_stderr_diagnostic_is_reported_on_process_failure() -> None:
    lines, _ = _stream()
    stderr = _over_limit_reader(b"provider authentication failed" * 3)

    with pytest.raises(OspreyError, match="stderr diagnostic line exceeded stream limit"):
        await _collect(
            OspreyBackend(osprey_binary="fake"),
            lines,
            returncode=1,
            stderr_reader=stderr,
        )


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_stderr_is_drained_separately_from_jsonl_stdout() -> None:
    lines, _ = _stream()

    events, spawner = await _collect(
        OspreyBackend(osprey_binary="fake"),
        lines,
        stderr_lines=[
            "2026-08-27T23:07:54Z INFO osprey_cli::headless: "
            "restoring remembered model preference"
        ],
    )

    assert [type(event) for event in events] == [RequestEvent, CostEvent, ResultEvent]
    assert spawner.kwargs[0]["stderr"] is asyncio.subprocess.PIPE


@pytest.mark.asyncio
async def test_process_cleanup_releases_inherited_stderr_before_waiting_for_eof() -> None:
    lines, _ = _stream()

    events, _ = await asyncio.wait_for(
        _collect(
            OspreyBackend(osprey_binary="fake"),
            lines,
            stderr_held_open=True,
        ),
        timeout=0.5,
    )

    assert [type(event) for event in events] == [RequestEvent, CostEvent, ResultEvent]


def test_factory_builds_verified_osprey_jsonl_command() -> None:
    backend = create_backend(
        "osprey",
        model="test-model",
        osprey_binary="fake-osprey",
    )
    assert isinstance(backend, OspreyBackend)

    assert backend.model == "test-model"
    assert backend.build_command("hello") == [
        "fake-osprey",
        "agent",
        "--events-jsonl",
        "--observation-budget-update-bytes",
        "65536",
        "--observation-budget-inline-bytes",
        "262144",
        "--observation-budget-admission-bytes",
        "2097152",
        "--model",
        "test-model",
        "hello",
    ]


@pytest.mark.asyncio
async def test_translates_text_thinking_tool_identity_metrics_and_result() -> None:
    lines, _ = _stream(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
        {"event": "thinking_delta", "content": "checking"},
        {"event": "text_delta", "content": "done"},
        {
            "event": "tool_call",
            "tool_call_id": "c-1",
            "tool_name": "tool_search",
            "arguments": {"query": "MCP"},
        },
        {
            "event": "tool_result",
            "tool_call_id": "c-1",
            "tool_name": "mcp.fetch",
            "status": "success",
            "content": "payload",
            "duration_ms": 4,
        },
        {
            "event": "turn_end",
            "turn_id": "t-1",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": None,
            "usage_reported": False,
            "duration_ms": 4,
        },
    )
    events, _ = await _collect(OspreyBackend(model="custom-model", osprey_binary="fake"), lines)

    assert [type(event) for event in events] == [
        RequestEvent,
        ThinkingEvent,
        TextEvent,
        ToolStartEvent,
        ToolResultEvent,
        TurnEndEvent,
        CostEvent,
        ResultEvent,
    ]
    tool_start = next(event for event in events if isinstance(event, ToolStartEvent))
    tool_result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert tool_start.name == "tool_search"
    assert tool_start.id == "c-1"
    assert tool_result.id == "c-1"
    assert tool_result.output == "payload"
    assert tool_result.is_error is False
    assert not any(isinstance(event, MetricsEvent) for event in events)


@pytest.mark.asyncio
async def test_coalesces_streaming_thinking_deltas_before_text() -> None:
    lines, _ = _stream(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
        {"event": "thinking_delta", "content": "Let"},
        {"event": "thinking_delta", "content": " me"},
        {"event": "thinking_delta", "content": " think."},
        {"event": "text_delta", "content": "Done."},
        {
            "event": "turn_end",
            "turn_id": "t-1",
            "usage_reported": False,
        },
    )

    events, _ = await _collect(OspreyBackend(osprey_binary="fake"), lines)

    assert [type(event) for event in events] == [
        RequestEvent,
        ThinkingEvent,
        TextEvent,
        TurnEndEvent,
        CostEvent,
        ResultEvent,
    ]
    assert [event.text for event in events if isinstance(event, ThinkingEvent)] == [
        "Let me think."
    ]


@pytest.mark.asyncio
async def test_ignores_blank_thinking_deltas() -> None:
    lines, _ = _stream(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
        {"event": "thinking_delta", "content": ""},
        {"event": "thinking_delta", "content": "  \n"},
        {"event": "text_delta", "content": "Done."},
        {"event": "turn_end", "turn_id": "t-1", "usage_reported": False},
    )

    events, _ = await _collect(OspreyBackend(osprey_binary="fake"), lines)

    assert not any(isinstance(event, ThinkingEvent) for event in events)


@pytest.mark.asyncio
async def test_result_exposes_session_model_without_usage_metrics() -> None:
    lines, _ = _stream(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
        {"event": "text_delta", "content": "Done."},
        {"event": "turn_end", "turn_id": "t-1", "usage_reported": False},
    )

    backend = OspreyBackend(osprey_binary="fake")
    assert backend.model == "unknown"
    events, _ = await _collect(backend, lines)
    result = next(event for event in events if isinstance(event, ResultEvent))

    assert result.model_name == "custom-model"
    assert backend.model == "custom-model"


@pytest.mark.asyncio
async def test_separates_thinking_runs_at_protocol_boundaries() -> None:
    lines, _ = _stream(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
        {"event": "thinking_delta", "content": "first attempt"},
        {"event": "driver_retry"},
        {"event": "thinking_delta", "content": "second attempt"},
        {"event": "text_delta", "content": "Done."},
        {
            "event": "turn_end",
            "turn_id": "t-1",
            "usage_reported": False,
        },
    )

    events, _ = await _collect(OspreyBackend(osprey_binary="fake"), lines)

    assert [event.text for event in events if isinstance(event, ThinkingEvent)] == [
        "first attempt",
        "second attempt",
    ]


@pytest.mark.asyncio
async def test_trajectory_records_coalesced_thinking_as_prose(tmp_path: Path) -> None:
    lines, _ = _stream(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
        {"event": "thinking_delta", "content": "Let"},
        {"event": "thinking_delta", "content": " me"},
        {"event": "thinking_delta", "content": " think."},
        {
            "event": "message_end",
            "messages": [{"type": "result", "data": {"content": "Done."}}],
        },
        {
            "event": "turn_end",
            "turn_id": "t-1",
            "usage_reported": False,
        },
    )
    recorder = TrajectoryRecorder(
        path=tmp_path / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="osprey",
        session_id="daydream-session",
    )

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            invocation.observe_user_step("prompt")
            events, _ = await _collect(OspreyBackend(osprey_binary="fake"), lines)
            for event in events:
                invocation.observe(event)

    trajectory = recorder.build_trajectory().model_dump(exclude_none=True)
    assert trajectory["steps"][1]["reasoning_content"] == "Let me think."
    assert trajectory["agent"]["model_name"] == "custom-model"
    assert trajectory["steps"][1]["model_name"] == "custom-model"


@pytest.mark.asyncio
async def test_usage_and_structured_output_preserve_optional_metrics() -> None:
    lines, _ = _stream(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
        {
            "event": "turn_end",
            "turn_id": "t-1",
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "cached_tokens": None,
            "cache_write_tokens": None,
            "usage_reported": True,
            "provider": "openai-compatible",
            "model": "custom-model",
            "duration_ms": 0,
            "thinking_tokens": 3,
            "tool_calls": 0,
            "cost_usd": "0.125",
        },
    )
    lines[-1]["structured_output"] = {"ok": True}
    events, _ = await _collect(OspreyBackend(model="custom-model", osprey_binary="fake"), lines)
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
        "session_id": "s-137",
        "provider": "openai-compatible",
        "model": "custom-model",
        "outcome": "completed",
        "exit_code": 0,
    }


@pytest.mark.asyncio
async def test_unreported_usage_permits_omitted_optional_telemetry() -> None:
    lines, _ = _stream(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
        {
            "event": "turn_end",
            "turn_id": "t-1",
            "usage_reported": False,
        },
    )

    events, _ = await _collect(OspreyBackend(osprey_binary="fake"), lines)

    assert [event.message_id for event in events if isinstance(event, TurnEndEvent)] == ["t-1"]
    assert not any(isinstance(event, MetricsEvent) for event in events)


@pytest.mark.asyncio
async def test_message_end_reconstructs_result_when_no_text_delta_arrives() -> None:
    lines, _ = _stream(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
        {
            "event": "message_end",
            "messages": [{"type": "result", "data": {"content": "from message_end"}}],
        },
        {
            "event": "turn_end",
            "turn_id": "t-1",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": None,
            "usage_reported": False,
            "duration_ms": 0,
        },
    )
    events, _ = await _collect(OspreyBackend(osprey_binary="fake"), lines)
    assert [event.text for event in events if isinstance(event, TextEvent)] == ["from message_end"]


@pytest.mark.asyncio
async def test_protocol_version_and_unknown_events_fail_closed() -> None:
    backend = OspreyBackend(osprey_binary="fake")
    bad_version = [{"event": "protocol", "version": 99}]
    with pytest.raises(
        OspreyProtocolError, match="unsupported Osprey JSONL protocol version"
    ):
        await _collect(backend, bad_version)

    unknown: list[dict[str, object]] = [
        {"event": "protocol", "version": 2},
        {
            "event": "session_start",
            "session_id": "s",
            "started_at": "2026-08-15T00:00:00Z",
            "model": "m",
            "provider": "p",
        },
        {"event": "not-a-real-event"},
    ]
    with pytest.raises(OspreyProtocolError, match="unknown Osprey JSONL event"):
        await _collect(backend, unknown)


def test_command_forwards_verified_policy_resume_fork_and_schema_flags(tmp_path: Path) -> None:
    backend = OspreyBackend(
        model="m",
        osprey_binary="fake",
        approval="deny-untrusted",
        sandbox=True,
        allowed_roots=[tmp_path],
    )
    command = backend.build_command(
        "prompt",
        output_schema_path=tmp_path / "schema.json",
        continuation=ContinuationToken("osprey", {"session_id": "s", "mode": "fork"}),
        max_turns=3,
        read_only=True,
    )
    assert command[:11] == [
        "fake",
        "agent",
        "--events-jsonl",
        "--observation-budget-update-bytes",
        "65536",
        "--observation-budget-inline-bytes",
        "262144",
        "--observation-budget-admission-bytes",
        "2097152",
        "--model",
        "m",
    ]
    assert "--read-only" in command
    assert command[command.index("--fork-from") + 1] == "s"
    assert command[command.index("--output-schema") + 1].endswith("schema.json")
    assert command[command.index("--max-turns") + 1] == "3"
    assert command[command.index("--approval") + 1] == "deny-untrusted"


@pytest.mark.asyncio
async def test_output_schema_is_temp_file_forwarded_and_cleaned(tmp_path: Path) -> None:
    lines, _ = _stream()
    _, spawner = await _collect(
        OspreyBackend(osprey_binary="fake"), lines, output_schema={"type": "object"}
    )
    command = list(spawner.argvs[0])
    schema_path = Path(command[command.index("--output-schema") + 1])
    assert not schema_path.exists()


@pytest.mark.asyncio
async def test_output_schema_temp_file_is_cleaned_when_serialization_fails(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.touch()
    handle = MagicMock()
    handle.name = str(schema_path)
    with (
        patch("daydream.backends.osprey.tempfile.NamedTemporaryFile", return_value=handle),
        patch("daydream.backends.osprey.json.dump", side_effect=TypeError("not serializable")),
        pytest.raises(TypeError, match="not serializable"),
    ):
        await _collect(OspreyBackend(osprey_binary="fake"), [], output_schema={"type": object})
    assert not schema_path.exists()


def test_unsupported_policy_and_tool_search_options_fail_closed() -> None:
    with pytest.raises(OspreyUnsupportedOption, match="interactive approver"):
        OspreyBackend(approval="on-request").build_command("prompt")
    with pytest.raises(OspreyUnsupportedOption, match="no corresponding flag"):
        OspreyBackend().build_command("prompt", tool_search_mode="off")
    with pytest.raises(OspreyUnsupportedOption, match="ephemeral-session"):
        OspreyBackend().build_command("prompt", persist_session=False)


@pytest.mark.asyncio
async def test_non_success_terminal_outcome_is_not_reported_as_success() -> None:
    lines, _ = _stream()
    lines[-1]["outcome"] = "budget_expired"
    with pytest.raises(OspreyTerminalError, match="budget_expired") as exc_info:
        await _collect(OspreyBackend(osprey_binary="fake"), lines)
    assert exc_info.value.outcome == "budget_expired"


@pytest.mark.asyncio
async def test_non_success_terminal_outcome_with_nonzero_process_exit_is_process_failure() -> None:
    lines, _ = _stream()
    lines[-1]["outcome"] = "budget_expired"
    with pytest.raises(OspreyError, match="return code 1") as exc_info:
        await _collect(OspreyBackend(osprey_binary="fake"), lines, returncode=1)
    assert exc_info.value.category == "PROCESS_EXIT"


@pytest.mark.asyncio
async def test_nonzero_process_exit_includes_stderr_diagnostics() -> None:
    lines, _ = _stream()

    with pytest.raises(OspreyError, match="provider authentication failed"):
        await _collect(
            OspreyBackend(osprey_binary="fake"),
            lines,
            returncode=1,
            stderr_lines=["provider authentication failed"],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events, message",
    [
        (
            [{"event": "turn_start", "turn_id": "t-1", "timestamp": "now"}],
            "active turn",
        ),
        (
            [
                {
                    "event": "tool_call",
                    "tool_call_id": "c-1",
                    "tool_name": "tool_search",
                    "arguments": {},
                }
            ],
            "pending tool calls",
        ),
    ],
)
async def test_successful_session_end_requires_a_quiescent_stream(
    events: list[dict[str, object]],
    message: str,
) -> None:
    lines, _ = _stream(*events)

    with pytest.raises(OspreyError, match=message):
        await _collect(OspreyBackend(osprey_binary="fake"), lines)


@pytest.mark.asyncio
async def test_terminal_exit_code_must_match_process_status() -> None:
    lines, _ = _stream()
    lines[-1]["exit_code"] = 1

    with pytest.raises(OspreyError, match="exit_code"):
        await _collect(OspreyBackend(osprey_binary="fake"), lines)


def test_non_negative_int_helpers_reject_below_zero() -> None:
    event = {"event": "turn_end", "k": -1}
    with pytest.raises(OspreyProtocolError, match="has negative 'k'"):
        _required_non_negative_int(event, "k")
    with pytest.raises(OspreyProtocolError, match="has negative 'k'"):
        _optional_non_negative_int(event, "k")
    assert _optional_non_negative_int({"event": "turn_end", "k": None}, "k") is None
    assert _required_non_negative_int({"event": "turn_end", "k": 0}, "k") == 0
    assert _optional_non_negative_int({"event": "turn_end", "k": 0}, "k") == 0


@pytest.mark.asyncio
async def test_negative_session_total_cost_is_rejected() -> None:
    lines, _ = _stream()
    lines[-1]["total_cost_usd"] = "-0.125"

    with pytest.raises(OspreyError, match="total_cost_usd"):
        await _collect(OspreyBackend(osprey_binary="fake"), lines)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events, field",
    [
        ([{"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
          {"event": "turn_end", "turn_id": "t-1", "usage_reported": True,
           "duration_ms": -1, "prompt_tokens": 0, "completion_tokens": 0}], "duration_ms"),
        ([{"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
          {"event": "turn_end", "turn_id": "t-1", "usage_reported": False,
           "duration_ms": -1}], "duration_ms"),
        ([{"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
          {"event": "turn_end", "turn_id": "t-1", "usage_reported": True,
           "duration_ms": 0, "prompt_tokens": -1, "completion_tokens": 0}], "prompt_tokens"),
        ([{"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
          {"event": "turn_end", "turn_id": "t-1", "usage_reported": True,
           "duration_ms": 0, "prompt_tokens": 0, "completion_tokens": -1}], "completion_tokens"),
        ([{"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
          {"event": "turn_end", "turn_id": "t-1", "usage_reported": True,
           "duration_ms": 0, "prompt_tokens": 0, "completion_tokens": 0,
           "cached_tokens": -1}], "cached_tokens"),
        ([{"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
          {"event": "turn_end", "turn_id": "t-1", "usage_reported": True,
           "duration_ms": 0, "prompt_tokens": 0, "completion_tokens": 0,
           "thinking_tokens": -1}], "thinking_tokens"),
        ([{"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
          {"event": "turn_end", "turn_id": "t-1", "usage_reported": True,
           "duration_ms": 0, "prompt_tokens": 0, "completion_tokens": 0,
           "cost_usd": "-0.125"}], "cost_usd"),
    ],
)
async def test_negative_osprey_telemetry_is_rejected(
    events: list[dict[str, object]], field: str
) -> None:
    lines, _ = _stream(*events)

    with pytest.raises(OspreyError, match=field):
        await _collect(OspreyBackend(osprey_binary="fake"), lines)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events, message",
    [
        (
            [
                {
                    "event": "turn_end",
                    "turn_id": "t-1",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "usage_reported": False,
                    "duration_ms": 0,
                }
            ],
            "turn_end without turn_start",
        ),
        (
            [
                {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
                {"event": "turn_start", "turn_id": "t-2", "timestamp": "now"},
            ],
            "turn_start before prior turn_end",
        ),
    ],
)
async def test_invalid_turn_event_order_fails_closed(
    events: list[dict[str, object]], message: str
) -> None:
    lines, _ = _stream(*events)
    with pytest.raises(OspreyProtocolError, match=message):
        await _collect(OspreyBackend(osprey_binary="fake"), lines)


@pytest.mark.asyncio
async def test_trajectory_preserves_tool_identity(tmp_path: Path) -> None:
    lines, _ = _stream(
        {
            "event": "tool_call",
            "tool_call_id": "c-2",
            "tool_name": "mcp.fetch",
            "arguments": {"id": "7"},
        },
        {
            "event": "tool_result",
            "tool_call_id": "c-2",
            "tool_name": "mcp.fetch",
            "status": "success",
            "content": "payload",
            "duration_ms": 1,
        },
    )
    recorder = TrajectoryRecorder(
        path=tmp_path / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="osprey",
        session_id="daydream-session",
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            invocation.observe_user_step("prompt")
            events, _ = await _collect(
                OspreyBackend(model="requested", osprey_binary="fake"), lines
            )
            for event in events:
                invocation.observe(event)

    trajectory = recorder.build_trajectory().model_dump(exclude_none=True)
    tool_call = trajectory["steps"][1]["tool_calls"][0]
    assert tool_call["function_name"] == "mcp.fetch"
    assert tool_call["tool_call_id"] == "c-2"
    assert trajectory["steps"][1]["observation"]["results"][0]["source_call_id"] == "c-2"
    assert "cost_usd" not in trajectory["final_metrics"]


@pytest.mark.asyncio
async def test_cancel_delegates_to_shared_transport_lifecycle() -> None:
    """cancel() reaps every tracked transport's process group and pipes."""
    from daydream.backends._transport import CliTransport

    backend = OspreyBackend(osprey_binary="fake")

    proc = MagicMock()
    proc.returncode = None
    proc.wait = AsyncMock(side_effect=[asyncio.TimeoutError(), 0])
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    transport = CliTransport("osprey", ["osprey", "agent"], limit=1024)
    transport.processes.append(proc)
    backend._transports = [transport]

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
            "duration_ms": 10, "prompt_tokens": 5, "completion_tokens": 3,
            "model": "custom-model",
        },
    ]


def _p18_osprey_stream(events: list[dict[str, object]]) -> list[str]:
    """Wrap events with protocol header/session_start/session_end JSONL lines."""
    lines, _rc = _stream(*events)
    return [json.dumps(line) for line in lines]


@pytest.mark.asyncio
async def test_osprey_request_event_temperature_and_label_presence_rules() -> None:
    """Explicit 0.0 is admitted; labels reduce to presence; mode closed values."""
    backend = OspreyBackend(
        model="custom-model", osprey_binary="fake",
        temperature=0.0, persona="PATTERN_PERSONA_PLACEHOLDER",
        toolset="TOOLSET_PLACEHOLDER", approval="deny-untrusted",
        sandbox=True, immutable_runtime_surface=True, compress_context=False,
        ultracode=True, max_subagents=3, llm_rpm=0,
        vars=(("k", "v"), ("k2", "v2")),
    )
    lines = _p18_osprey_stream(_p18_osprey_events())
    fake = FakeCliProcess(lines)
    with patch(
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        side_effect=lambda *a, **k: fake,
    ) as mock_exec:
        events = [event async for event in backend.execute(Path("/tmp"), "prompt here")]

    argv = [str(a) for a in mock_exec.call_args.args]
    request = next(e for e in events if isinstance(e, RequestEvent))
    config = request.config
    assert isinstance(config, OspreyRequestConfig)

    # Temperature: exact zero preserved (the argv carries it).
    assert config.temperature == 0.0
    assert argv[argv.index("--temperature") + 1] == "0.0"

    # Persona/toolset: exact argv values present in argv, presence booleans in
    # telemetry, and NO label-carrying field exists.
    assert argv[argv.index("--persona") + 1] == "PATTERN_PERSONA_PLACEHOLDER"
    assert argv[argv.index("--toolset") + 1] == "TOOLSET_PLACEHOLDER"
    assert config.persona_present is True and config.toolset_present is True
    assert not hasattr(config, "persona") and not hasattr(config, "toolset")

    # Closed modes and booleans from exact argv.
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

    # Provenance: all four identity fields are native (session_start).
    assert request.model_source == "native"
    assert request.provider_source == "native"
    assert request.session_source == "native"
    assert request.timestamp_source == "native"


@pytest.mark.asyncio
async def test_osprey_hidden_temperature_stays_absent_in_telemetry() -> None:
    """Without explicit --temperature, the config carries temperature=None."""
    backend = OspreyBackend(model="custom-model", osprey_binary="fake")
    lines = _p18_osprey_stream(_p18_osprey_events())
    fake = FakeCliProcess(lines)
    with patch(
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        side_effect=lambda *a, **k: fake,
    ) as mock_exec:
        events = [event async for event in backend.execute(Path("/tmp"), "p")]

    argv = [str(a) for a in mock_exec.call_args.args]
    assert "--temperature" not in argv
    request = next(e for e in events if isinstance(e, RequestEvent))
    config = request.config
    assert isinstance(config, OspreyRequestConfig)
    assert config.temperature is None  # config-resolved temperature is hidden


@pytest.mark.asyncio
async def test_osprey_nonzero_temperature_is_admitted_verbatim() -> None:
    """An explicit nonzero temperature passes through exactly."""
    backend = OspreyBackend(model="custom-model", osprey_binary="fake", temperature=0.7)
    lines = _p18_osprey_stream(_p18_osprey_events())
    fake = FakeCliProcess(lines)
    with patch(
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        side_effect=lambda *a, **k: fake,
    ) as mock_exec:
        events = [event async for event in backend.execute(Path("/tmp"), "p")]

    argv = [str(a) for a in mock_exec.call_args.args]
    assert argv[argv.index("--temperature") + 1] == "0.7"
    request = next(e for e in events if isinstance(e, RequestEvent))
    config = request.config
    assert isinstance(config, OspreyRequestConfig)
    assert config.temperature == 0.7


@pytest.mark.asyncio
async def test_osprey_turn_end_model_override_is_native() -> None:
    """turn_end model becomes the native TurnEnd identity; no finish reason."""
    backend = OspreyBackend(model="custom-model", osprey_binary="fake")
    lines = _p18_osprey_stream(_p18_osprey_events())
    fake = FakeCliProcess(lines)
    with patch(
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        side_effect=lambda *a, **k: fake,
    ):
        events = [event async for event in backend.execute(Path("/tmp"), "p")]

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


@pytest.mark.asyncio
async def test_osprey_session_end_usage_is_session_sourced() -> None:
    """Terminal totals carry session measurement source and reported cost."""
    backend = OspreyBackend(model="custom-model", osprey_binary="fake")
    lines = _p18_osprey_stream(_p18_osprey_events())
    fake = FakeCliProcess(lines)
    with patch(
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        side_effect=lambda *a, **k: fake,
    ):
        events = [event async for event in backend.execute(Path("/tmp"), "p")]

    costs = [e for e in events if isinstance(e, CostEvent)]
    assert len(costs) == 1
    assert costs[0].measurement_source == "session"
    # The base fixture session_end carries no cost -> no provenance claim.
    assert costs[0].cost_source is None


@pytest.mark.asyncio
async def test_osprey_resume_and_fork_continuation_modes() -> None:
    """Resume and fork continuation tokens map to closed config modes."""
    for mode, flag in (("resume", "--resume"), ("fork", "--fork-from")):
        backend = OspreyBackend(model="custom-model", osprey_binary="fake")
        token = ContinuationToken(
            backend="osprey", data={"session_id": "s-9", "mode": mode},
        )
        lines = _p18_osprey_stream(_p18_osprey_events())
        fake = FakeCliProcess(lines)
        with patch(
            "daydream.backends._transport.asyncio.create_subprocess_exec",
            side_effect=lambda *a, **k: fake,
        ) as mock_exec:
            events = [
                event
                async for event in backend.execute(Path("/tmp"), "p", continuation=token)
            ]
        argv = [str(a) for a in mock_exec.call_args.args]
        assert flag in argv
        assert argv[argv.index(flag) + 1] == "s-9"  # token session drives argv
        request = next(e for e in events if isinstance(e, RequestEvent))
        config = request.config
        assert isinstance(config, OspreyRequestConfig)
        assert config.continuation_mode == mode
        # Native handshake identity wins over the configured resume token.
        assert request.session_id == "s-137"
        assert request.session_source == "native"
