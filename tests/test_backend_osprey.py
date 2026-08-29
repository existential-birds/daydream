"""Hermetic contract tests for the additive Osprey backend boundary."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    MetricsEvent,
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
    OspreyTerminalError,
    OspreyUnsupportedOption,
    _drain_stderr,
)
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder


class _FakeStdout:
    def __init__(self, lines: list[dict[str, object]]) -> None:
        import json

        self._lines = iter(json.dumps(line).encode() + b"\n" for line in lines)

    async def readline(self) -> bytes:
        try:
            return next(self._lines)
        except StopIteration:
            return b""


class _FakeStderr:
    def __init__(self, lines: list[str], held_open: asyncio.Event | None = None) -> None:
        self._lines = iter(line.encode() + b"\n" for line in lines)
        self._held_open = held_open

    async def readline(self) -> bytes:
        try:
            return next(self._lines)
        except StopIteration:
            if self._held_open is not None:
                await self._held_open.wait()
            return b""


class _FakeProcess:
    def __init__(
        self,
        lines: list[dict[str, object]],
        returncode: int = 0,
        stderr_lines: list[str] | None = None,
        stderr_held_open: bool = False,
        stdout_reader: asyncio.StreamReader | None = None,
        stderr_reader: asyncio.StreamReader | None = None,
    ) -> None:
        self.stdout = stdout_reader or _FakeStdout(lines)
        self._stderr_eof = asyncio.Event() if stderr_held_open else None
        self.stderr = stderr_reader or _FakeStderr(stderr_lines or [], self._stderr_eof)
        self.returncode = returncode
        self.pid = 137

    async def wait(self) -> int:
        return self.returncode

    def close_stderr(self) -> None:
        if self._stderr_eof is not None:
            self._stderr_eof.set()


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
) -> tuple[list[AgentEvent], AsyncMock]:
    process = _FakeProcess(
        lines,
        returncode=returncode,
        stderr_lines=stderr_lines,
        stderr_held_open=stderr_held_open,
        stdout_reader=stdout_reader,
        stderr_reader=stderr_reader,
    )
    exec_mock = AsyncMock(return_value=process)
    terminate_mock = AsyncMock(side_effect=lambda _process: process.close_stderr())
    with (
        patch("daydream.backends.osprey.asyncio.create_subprocess_exec", exec_mock),
        patch("daydream.backends.osprey.terminate_process", new=terminate_mock),
    ):
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
    return events, exec_mock


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
    assert backend._processes == []


@pytest.mark.asyncio
async def test_oversized_stderr_diagnostic_does_not_fail_successful_run() -> None:
    lines, _ = _stream()
    stderr = _over_limit_reader(b"diagnostic" * 8)

    events, _ = await _collect(
        OspreyBackend(osprey_binary="fake"),
        lines,
        stderr_reader=stderr,
    )

    assert [type(event) for event in events] == [ResultEvent]


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
    stderr.feed_eof()

    await _drain_stderr(stderr, diagnostics)

    assert len(diagnostics) == 1
    assert len(diagnostics[0]) <= 500
    assert secret not in diagnostics[0]
    assert "[REDACTED_ENV_VAR]" in diagnostics[0]


@pytest.mark.asyncio
async def test_stderr_is_drained_separately_from_jsonl_stdout() -> None:
    lines, _ = _stream()

    events, exec_mock = await _collect(
        OspreyBackend(osprey_binary="fake"),
        lines,
        stderr_lines=[
            "2026-08-27T23:07:54Z INFO osprey_cli::headless: "
            "restoring remembered model preference"
        ],
    )

    assert [type(event) for event in events] == [ResultEvent]
    assert exec_mock.call_args.kwargs["stderr"] is asyncio.subprocess.PIPE


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

    assert [type(event) for event in events] == [ResultEvent]


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
        ThinkingEvent,
        TextEvent,
        ToolStartEvent,
        ToolResultEvent,
        TurnEndEvent,
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
        ThinkingEvent,
        TextEvent,
        TurnEndEvent,
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
    with pytest.raises(Exception, match="unsupported Osprey JSONL protocol version"):
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
    with pytest.raises(Exception, match="unknown Osprey JSONL event"):
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
    _, exec_mock = await _collect(
        OspreyBackend(osprey_binary="fake"), lines, output_schema={"type": "object"}
    )
    command = list(exec_mock.call_args.args)
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


@pytest.mark.asyncio
async def test_negative_thinking_tokens_are_rejected() -> None:
    lines, _ = _stream(
        {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"},
        {
            "event": "turn_end",
            "turn_id": "t-1",
            "usage_reported": True,
            "duration_ms": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thinking_tokens": -1,
        },
    )

    with pytest.raises(OspreyError, match="thinking_tokens"):
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
    with pytest.raises(Exception, match=message):
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
async def test_cancel_delegates_to_shared_process_lifecycle() -> None:
    backend = OspreyBackend(osprey_binary="fake")
    process = MagicMock()
    backend._processes.append(process)
    with patch("daydream.backends.osprey.cancel_processes", new=AsyncMock()) as cancel:
        await backend.cancel()
    cancel.assert_awaited_once_with([process])
