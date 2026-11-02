"""Hermetic contract tests for the additive Osprey backend boundary."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    CostEvent,
    MetricsEvent,
    OspreyRequestConfig,
    RequestEvent,
    ResultEvent,
)
from daydream.backends.osprey import (
    OspreyBackend,
    OspreyError,
    OspreyProtocolError,
    OspreyTerminalError,
    OspreyUnsupportedOption,
)
from daydream.trajectory import DaydreamPhase
from tests.harness.fake_cli_process import FakeCliProcess, FakeCliSpawner, install_fake_cli_process
from tests.harness.osprey_jsonl import osprey_session
from tests.harness.protocol_cli import install_protocol_cli
from tests.harness.protocol_cli_assertions import assert_protocol_cli_invariants
from tests.harness.trajectory import make_recorder


@pytest.mark.parametrize("sandbox", [True])
async def test_artifact_visibility_protocol_cli_preserves_sandbox_roots_and_terminal_envelope(
    tmp_path: Path, sandbox: bool,
) -> None:
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
    with pytest.MonkeyPatch.context() as monkeypatch:
        spawner = install_fake_cli_process(
            monkeypatch, backend.osprey_binary, lines=[json.dumps(line) for line in lines],
            exit_code=returncode, stderr_lines=stderr_lines, stderr_held_open=stderr_held_open,
            stdout_reader=stdout_reader, stderr_reader=stderr_reader,
        )
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
    backend = OspreyBackend(osprey_binary="fake")
    with pytest.raises(OspreyError, match=r"(?i)stdout.*limit") as exc_info:
        await _collect(backend, [], stdout_reader=stdout)
    assert exc_info.value.category == "PROTOCOL"
    assert backend._transports == []


async def test_non_utf8_stdout_line_is_non_json_protocol_error_not_over_limit() -> None:
    """Replacement decoding may still leave invalid JSON; diagnose that as non-JSON rather than an
    oversized line.
    """
    stdout = _over_limit_reader(b'{"event"' + b"\xff" + b"}\n")
    backend = OspreyBackend(osprey_binary="fake")
    with pytest.raises(OspreyError, match=r"non-JSON line in JSONL mode") as exc_info:
        await _collect(backend, [], stdout_reader=stdout)
    assert exc_info.value.category == "PROTOCOL"
    assert "byte limit" not in str(exc_info.value)
    assert "\ufffd" in str(exc_info.value)
    assert backend._transports == []


async def test_oversized_stderr_diagnostic_is_reported_on_process_failure() -> None:
    lines = osprey_session()
    stderr = _over_limit_reader(b"provider authentication failed" * 3)
    with pytest.raises(OspreyError, match="stderr diagnostic line exceeded stream limit"):
        await _collect(OspreyBackend(osprey_binary="fake"), lines, returncode=1, stderr_reader=stderr)


async def test_process_failure_without_stderr_reports_absent_diagnostics() -> None:
    with pytest.raises(OspreyError, match="return code 1: no diagnostic output captured"):
        await _collect(OspreyBackend(osprey_binary="fake"), osprey_session(), returncode=1)


async def test_osprey_turns_retain_thinking_boundaries_and_native_identity(tmp_path: Path) -> None:
    """Retry-separated thinking and a later blank turn survive native transport into persisted steps."""
    rows = osprey_session(
        {"event": "turn_start", "turn_id": "first", "timestamp": "2026-08-15T00:00:01Z"},
        {"event": "thinking_delta", "content": "Let"},
        {"event": "thinking_delta", "content": " me"},
        {"event": "thinking_delta", "content": " think."},
        {"event": "driver_retry"},
        {"event": "thinking_delta", "content": "second attempt"},
        {"event": "text_delta", "content": "Done."},
        {"event": "turn_end", "turn_id": "first", "usage_reported": False},
        {"event": "turn_start", "turn_id": "second", "timestamp": "2026-08-15T00:00:02Z"},
        {"event": "thinking_delta", "content": ""},
        {"event": "thinking_delta", "content": "  \n"},
        {"event": "message_end", "messages": [{"type": "result", "data": {"content": "from message_end"}}]},
        {"event": "turn_end", "turn_id": "second", "usage_reported": False},
    )
    rows[-1]["total_turns"] = 2
    process = FakeCliProcess([json.dumps(row) for row in rows], stderr_lines=["restoring native model"])
    backend = OspreyBackend(osprey_binary="fake")
    assert backend.model == "unknown"
    recorder = make_recorder(tmp_path, agent_model_name="osprey")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=process) as spawn:
        async with recorder:
            output, continuation, reason = await run_agent(backend, tmp_path, "prompt", phase=DaydreamPhase.REVIEW)
    assert (output, reason) == ("Done.from message_end", None)
    assert continuation is not None and continuation.data["session_id"] == "s-137"
    assert backend.model == "custom-model" and process.reaped
    assert spawn.call_args.kwargs["start_new_session"] is True
    assert spawn.call_args.kwargs["cwd"] == str(tmp_path)
    assert spawn.call_args.kwargs["stderr"] is asyncio.subprocess.PIPE
    assert process.returncode == 0 and process.stdin.closed and process._transport.closed
    assert backend._transports == []
    trajectory = json.loads(recorder.path.read_text())
    steps = [step for step in trajectory["steps"] if step["source"] == "agent"]
    assert [step["message"] for step in steps] == ["Done.", "from message_end"]
    assert steps[0]["reasoning_content"] == "Let me think.\nsecond attempt"
    assert steps[1].get("reasoning_content") is None
    assert all(step.get("metrics") is None and step["model_name"] == "custom-model" for step in steps)
    assert trajectory["agent"]["model_name"] == "custom-model"
    assert "total_cost_usd" not in trajectory["final_metrics"]


async def test_protocol_version_and_unknown_events_fail_closed() -> None:
    backend = OspreyBackend(osprey_binary="fake")
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
        await _collect(OspreyBackend(osprey_binary="fake"), [], output_schema={"type": object})
    assert not schema_path.exists()


def test_unsupported_policy_and_tool_search_options_fail_closed() -> None:
    with pytest.raises(OspreyUnsupportedOption, match="interactive approver"):
        OspreyBackend(approval="on-request").build_command("prompt")
    with pytest.raises(OspreyUnsupportedOption, match="no corresponding flag"):
        OspreyBackend().build_command("prompt", tool_search_mode="off")
    with pytest.raises(OspreyUnsupportedOption, match="ephemeral-session"):
        OspreyBackend().build_command("prompt", persist_session=False)


async def test_non_success_terminal_outcome_is_not_reported_as_success() -> None:
    lines = osprey_session()
    lines[-1]["outcome"] = "budget_expired"
    with pytest.raises(OspreyTerminalError, match="budget_expired") as exc_info:
        await _collect(OspreyBackend(osprey_binary="fake"), lines)
    assert exc_info.value.outcome == "budget_expired"


async def test_non_success_terminal_outcome_with_nonzero_process_exit_is_process_failure() -> None:
    lines = osprey_session(
        {"event": "turn_start", "turn_id": "failed-turn", "timestamp": "2026-08-15T00:00:01Z"},
        {"event": "turn_end", "turn_id": "failed-turn", "usage_reported": True, "duration_ms": 80,
         "prompt_tokens": 60, "completion_tokens": 5, "cached_tokens": 20, "cache_write_tokens": 30,
         "thinking_tokens": 2, "cost_usd": "0.02"},
    )
    lines[-1].update(outcome="budget_expired", exit_code=1, total_prompt_tokens=60, total_completion_tokens=5,
                     total_cached_tokens=20, total_cache_write_tokens=30, total_thinking_tokens=2,
                     total_cost_usd="0.02", session_wallclock_ms=120)
    secret = "sk-realvalue123"
    process = FakeCliProcess(
        [json.dumps(line) for line in lines], exit_code=1,
        stderr_lines=[f"provider authentication failed OPENAI_API_KEY={secret} " + "x" * 1_024] * 12,
    )
    backend = OspreyBackend(osprey_binary="fake")
    events: list[AgentEvent] = []
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=process):
        with pytest.raises(OspreyError, match="return code 1") as exc_info:
            async for event in backend.execute(Path("/repo"), "prompt"):
                events.append(event)
    assert exc_info.value.category == "PROCESS_EXIT"
    diagnostics = str(exc_info.value).split(": ", 1)[1].splitlines()
    assert len(diagnostics) == 10
    assert all(len(line) <= 500 and "[REDACTED_ENV_VAR]" in line for line in diagnostics)
    assert diagnostics[0].startswith("provider authentication failed OPENAI_API_KEY=")
    assert secret not in str(exc_info.value)
    metric = next(event for event in events if isinstance(event, MetricsEvent))
    assert (metric.started_at, metric.duration_ms, metric.cache_creation_tokens) == (
        "2026-08-15T00:00:01Z", 80, 30,
    )
    cost = next(event for event in events if isinstance(event, CostEvent))
    assert (cost.input_tokens, cost.output_tokens, cost.cached_tokens, cost.cache_creation_tokens,
            cost.reasoning_tokens, cost.cost_usd) == (60, 5, 20, 30, 2, 0.02)
    result = next(event for event in events if isinstance(event, ResultEvent))
    assert (result.session_id, result.provider_name, result.finish_reason, result.duration_ms) == (
        "s-137", "openai-compatible", "budget_expired", 120,
    )
    assert process.reaped and process.returncode == 1 and backend._transports == []


async def test_osprey_session_stream_violations_fail_closed() -> None:
    """Non-quiescent ends, mismatched exits, negative telemetry and turn disorder are not success."""
    start: dict[str, object] = {"event": "turn_start", "turn_id": "t-1", "timestamp": "now"}
    def turn_end(**fields: object) -> dict[str, object]:
        return {"event": "turn_end", "turn_id": "t-1", "usage_reported": True, "duration_ms": 0,
                "prompt_tokens": 0, "completion_tokens": 0, **fields}
    cases: list[tuple[list[dict[str, object]], dict[str, object], type[Exception], str]] = [
        ([start], {}, OspreyError, "active turn"),
        ([{"event": "tool_call", "tool_call_id": "c-1", "tool_name": "tool_search", "arguments": {}}], {},
         OspreyError, "pending tool calls"),
        ([], {"exit_code": 1}, OspreyError, "exit_code"),
        ([], {"total_cost_usd": "-0.125"}, OspreyError, "total_cost_usd"),
        ([start, turn_end(duration_ms=-1)], {}, OspreyError, "duration_ms"),
        ([start, {"event": "turn_end", "turn_id": "t-1", "usage_reported": False, "duration_ms": -1}], {},
         OspreyError, "duration_ms"),
        ([start, turn_end(prompt_tokens=-1)], {}, OspreyError, "prompt_tokens"),
        ([start, turn_end(completion_tokens=-1)], {}, OspreyError, "completion_tokens"),
        ([start, turn_end(cached_tokens=-1)], {}, OspreyError, "cached_tokens"),
        ([start, turn_end(thinking_tokens=-1)], {}, OspreyError, "thinking_tokens"),
        ([start, turn_end(cost_usd="-0.125")], {}, OspreyError, "cost_usd"),
        ([turn_end(usage_reported=False)], {}, OspreyProtocolError, "turn_end without turn_start"),
        ([start, {**start, "turn_id": "t-2"}], {}, OspreyProtocolError, "turn_start before prior turn_end"),
    ]
    for events, terminal, error, message in cases:
        lines = osprey_session(*events)
        lines[-1].update(terminal)
        with pytest.raises(error, match=message):
            await _collect(OspreyBackend(osprey_binary="fake"), lines)


# --- P18 Task 1: effective request-config admission at the Osprey argv seam --


def _p18_osprey_events() -> list[dict[str, object]]:
    """Session events for a temperature-0, persona+toolset Osprey run."""
    return [
        {"event": "turn_start", "turn_id": "turn-1", "timestamp": "2026-08-15T00:00:01Z"},
        {"event": "text_delta", "content": "T0 answer"},
        {
            "event": "turn_end", "turn_id": "turn-1", "usage_reported": True,
            "duration_ms": 10, "prompt_tokens": 5, "completion_tokens": 3, "cached_tokens": 0, "model": "custom-model",
        },
    ]


async def test_osprey_request_event_temperature_and_label_presence_rules() -> None:
    """Explicit 0.0 is admitted; labels reduce to presence; mode closed values."""
    backend = OspreyBackend(
        model="custom-model", osprey_binary="fake", temperature=0.0, persona="PATTERN_PERSONA_PLACEHOLDER",
        toolset="TOOLSET_PLACEHOLDER", approval="deny-untrusted",
        sandbox=True, immutable_runtime_surface=True, compress_context=False,
        ultracode=True, max_subagents=3, llm_rpm=0, vars=(("k", "v"), ("k2", "v2")),
    )
    events, spawner = await _collect(backend, osprey_session(*_p18_osprey_events()))
    argv = list(spawner.argvs[-1])
    request = next(e for e in events if isinstance(e, RequestEvent))
    config = request.config
    assert isinstance(config, OspreyRequestConfig)
    assert config.temperature == 0.0
    assert next(event for event in events if isinstance(event, MetricsEvent)).cached_tokens == 0
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


async def test_osprey_resume_and_fork_continuation_modes() -> None:
    for mode, flag in (("resume", "--resume"), ("fork", "--fork-from")):
        backend = OspreyBackend(model="custom-model", osprey_binary="fake", approval="deny-untrusted")
        token = ContinuationToken(backend="osprey", data={"session_id": "s-9", "mode": mode})
        schema = {"type": "object"}
        events, spawner = await _collect(
            backend,
            osprey_session(*_p18_osprey_events()),
            continuation=token,
            output_schema=schema,
            max_turns=3,
            read_only=True,
        )
        argv = list(spawner.argvs[-1])
        assert flag in argv
        assert argv[argv.index(flag) + 1] == "s-9"  # token session drives argv
        assert argv[:3] == ["fake", "agent", "--events-jsonl"]
        assert "--read-only" in argv and argv[argv.index("--max-turns") + 1] == "3"
        assert argv[argv.index("--approval") + 1] == "deny-untrusted"
        assert not Path(argv[argv.index("--output-schema") + 1]).exists()
        request = next(e for e in events if isinstance(e, RequestEvent))
        config = request.config
        assert isinstance(config, OspreyRequestConfig)
        assert config.continuation_mode == mode and request.output_schema == schema
        # Native handshake identity wins over the configured resume token.
        assert request.session_id == "s-137"
        assert request.session_source == "native"
