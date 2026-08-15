"""Hermetic contract tests for the additive Osprey backend boundary."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daydream.backends import (
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
    OspreyTerminalError,
    OspreyUnsupportedOption,
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


class _FakeProcess:
    def __init__(self, lines: list[dict[str, object]], returncode: int = 0) -> None:
        self.stdout = _FakeStdout(lines)
        self.returncode = returncode
        self.pid = 137

    async def wait(self) -> int:
        return self.returncode


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


async def _collect(
    backend: OspreyBackend,
    lines: list[dict[str, object]],
    *,
    returncode: int = 0,
    **kwargs: object,
) -> tuple[list[object], MagicMock]:
    process = _FakeProcess(lines, returncode=returncode)
    exec_mock = AsyncMock(return_value=process)
    with (
        patch("daydream.backends.osprey.asyncio.create_subprocess_exec", exec_mock),
        patch("daydream.backends.osprey.terminate_process", new=AsyncMock()),
    ):
        events = [event async for event in backend.execute(Path("/repo"), "prompt", **kwargs)]
    return events, exec_mock


def test_factory_builds_verified_osprey_jsonl_command() -> None:
    backend = create_backend(
        "osprey",
        model="test-model",
        osprey_binary="fake-osprey",
    )

    assert backend.model == "test-model"
    assert backend.build_command("hello") == [
        "fake-osprey",
        "agent",
        "--events-jsonl",
        "--model",
        "test-model",
        "hello",
    ]


@pytest.mark.asyncio
async def test_translates_text_thinking_tool_identity_metrics_and_result() -> None:
    lines, _ = _stream(
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
    assert events[2].name == "tool_search"
    assert events[2].id == "c-1"
    assert events[3].id == "c-1"
    assert events[3].output == "payload"
    assert events[3].is_error is False
    assert not any(isinstance(event, MetricsEvent) for event in events)


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
            "thinking_tokens": 0,
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
    assert metrics.cached_tokens is None
    assert metrics.cost_usd == pytest.approx(0.125)
    assert result.structured_output == {"ok": True}
    assert result.continuation is not None
    assert result.continuation.data["session_id"] == "s-137"


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

    unknown = [
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
    assert command[:5] == ["fake", "agent", "--events-jsonl", "--model", "m"]
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
