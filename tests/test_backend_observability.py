"""Native backend streams preserve the public observability contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock, ToolResultBlock, UserMessage

from daydream.backends import AgentEvent, CostEvent, MetricsEvent, RequestEvent, ResultEvent, ToolResultEvent
from daydream.backends.claude import ClaudeAgentError, ClaudeBackend
from daydream.backends.codex import CodexBackend
from daydream.backends.osprey import OspreyBackend, OspreyError
from daydream.backends.pi import PiBackend, PiError, _render_tool_result
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder
from tests.harness.claude_sdk import scripted_client
from tests.harness.fake_cli_process import FakeCliProcess


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_claude_terminal_metadata_survives_billed_failure(
    monkeypatch: pytest.MonkeyPatch, failed: bool,
) -> None:
    usage = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 20,
             "cache_creation_input_tokens": 30}
    messages = [
        AssistantMessage(content=[TextBlock("partial")], model="actual-model", usage=usage),
        UserMessage(content=[ToolResultBlock("call", [{"type": "text", "text": "structured tool output"}])]),
        ResultMessage(subtype="error_during_execution" if failed else "success", duration_ms=120,
                      duration_api_ms=100, is_error=failed, num_turns=1, session_id="native-session",
                      stop_reason="error" if failed else "end_turn", total_cost_usd=0.02, usage=usage,
                      result="provider failed" if failed else "done"),
    ]
    monkeypatch.setattr("daydream.backends.claude.ClaudeSDKClient", scripted_client(messages))
    events: list[Any] = []
    try:
        async for event in ClaudeBackend("requested-model").execute(Path("/tmp"), "actual prompt",
                                                                   persist_session=False):
            events.append(event)
    except ClaudeAgentError:
        assert failed
    else:
        assert not failed
    cost = next(e for e in events if isinstance(e, CostEvent))
    assert cost.input_tokens == 60
    assert cost.cache_creation_tokens == 30
    assert cost.cost_usd == 0.02
    result = next(e for e in events if isinstance(e, ResultEvent))
    assert result.session_id == "native-session"
    assert result.finish_reason == ("error" if failed else "end_turn")
    assert result.continuation is None
    assert result.duration_ms == 120
    assert result.duration_api_ms == 100
    tool = next(e for e in events if isinstance(e, ToolResultEvent))
    assert json.loads(tool.output) == [{"type": "text", "text": "structured tool output"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_pi_actual_identity_cache_totals_and_failed_billing(failed: bool) -> None:
    msg = {"role": "assistant", "model": "actual-model", "provider": "actual-provider",
           "content": [{"type": "text", "text": "done"}], "stopReason": "error" if failed else "stop",
           "errorMessage": "provider failed", "usage": {"input": 10, "output": 5, "cacheRead": 20,
                                                          "cacheWrite": 30, "cost": {"total": 0.02}}}
    native = [{"type": "session", "id": "native-session"}, {"type": "turn_start"},
              {"type": "message_end", "message": msg}, {"type": "turn_end", "message": msg}]
    proc = FakeCliProcess([json.dumps(item) for item in native])
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    events: list[Any] = []
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=proc) as spawn:
        try:
            async for event in PiBackend("requested-model").execute(Path("/tmp"), "actual prompt", schema,
                                                                   persist_session=False):
                events.append(event)
        except PiError:
            assert failed
        else:
            assert not failed
    metric = next(e for e in events if isinstance(e, MetricsEvent))
    assert metric.prompt_tokens == 60
    assert metric.cache_creation_tokens == 30
    assert metric.provider_name == "actual-provider"
    assert metric.model_name == "actual-model"
    assert metric.usage_scope == "message"
    cost = next(e for e in events if isinstance(e, CostEvent))
    assert (cost.input_tokens, cost.cached_tokens, cost.cache_creation_tokens) == (60, 20, 30)
    result = next(e for e in events if isinstance(e, ResultEvent))
    assert (result.model_name, result.provider_name, result.session_id) == (
        "actual-model", "actual-provider", "native-session")
    assert result.continuation is None
    assert result.finish_reason == ("error" if failed else "stop")
    request = next(e for e in events if isinstance(e, RequestEvent))
    assert request.prompt == spawn.call_args.args[-1]
    assert request.prompt.startswith("actual prompt")
    assert request.output_schema == schema
    assert request.system_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("native_identity", "expected_model", "expected_provider"),
    [
        ({}, "requested-model", None),
        ({"model": "custom-response-model", "provider": "custom-provider"},
         "custom-response-model", "custom-provider"),
    ],
    ids=["provider-unavailable", "native-custom-provider"],
)
async def test_codex_usage_scope_and_native_session(
    native_identity: dict[str, str], expected_model: str, expected_provider: str | None,
) -> None:
    native = [{"type": "thread.started", "thread_id": "native-thread"},
              {"type": "turn.completed", **native_identity,
               "usage": {"input_tokens": 60, "output_tokens": 5, "cached_input_tokens": 20}}]
    proc = FakeCliProcess([json.dumps(item) for item in native])
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=proc):
        events = [event async for event in CodexBackend("requested-model").execute(
            Path("/tmp"), "actual prompt", persist_session=False)]
    metric = next(e for e in events if isinstance(e, MetricsEvent))
    assert metric.usage_scope == "invocation"
    result = next(e for e in events if isinstance(e, ResultEvent))
    assert result.session_id == "native-thread"
    request = next(e for e in events if isinstance(e, RequestEvent))
    assert request.model_name == "requested-model"
    assert request.provider_name is None  # Neither model names nor ambient config prove the provider.
    cost = next(e for e in events if isinstance(e, CostEvent))
    for response in (metric, cost, result):
        assert response.model_name == expected_model
        assert response.provider_name == expected_provider


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_osprey_terminal_usage_and_native_tool_metadata(failed: bool) -> None:
    native = [
        {"event": "protocol", "version": 2},
        {"event": "session_start", "session_id": "native-session", "started_at": "2026-09-05T12:00:00Z",
         "model": "actual-model", "provider": "actual-provider"},
        {"event": "turn_start", "turn_id": "turn", "timestamp": "2026-09-05T12:00:01Z"},
        {"event": "tool_call", "tool_call_id": "call", "tool_name": "read", "arguments": {}},
        {"event": "tool_result", "tool_call_id": "call", "tool_name": "read", "status": "success",
         "content": "output", "duration_ms": 37},
        {"event": "turn_end", "turn_id": "turn", "usage_reported": True, "duration_ms": 80,
         "prompt_tokens": 60, "completion_tokens": 5, "cached_tokens": 20, "cache_write_tokens": 30,
         "thinking_tokens": 2, "cost_usd": "0.02"},
        {"event": "session_end", "outcome": "failed" if failed else "completed",
         "exit_code": 1 if failed else 0, "total_cost_usd": "0.02", "total_prompt_tokens": 60,
         "total_completion_tokens": 5, "total_cached_tokens": 20, "total_cache_write_tokens": 30,
         "total_thinking_tokens": 2, "session_wallclock_ms": 120},
    ]
    proc = FakeCliProcess([json.dumps(item) for item in native], exit_code=1 if failed else 0)
    events: list[Any] = []
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=proc):
        try:
            async for event in OspreyBackend(osprey_binary="fake").execute(Path("/tmp"), "actual prompt"):
                events.append(event)
        except OspreyError:
            assert failed
        else:
            assert not failed
    cost = next(e for e in events if isinstance(e, CostEvent))
    assert (cost.input_tokens, cost.output_tokens, cost.cached_tokens, cost.cache_creation_tokens) == (60, 5, 20, 30)
    assert cost.reasoning_tokens == 2
    assert cost.provider_name == "actual-provider"
    metric = next(e for e in events if isinstance(e, MetricsEvent))
    assert metric.started_at == "2026-09-05T12:00:01Z"
    assert metric.duration_ms == 80
    assert metric.cache_creation_tokens == 30
    tool = next(e for e in events if isinstance(e, ToolResultEvent))
    assert tool.duration_ms == 37
    assert tool.status == "success"
    result = next(e for e in events if isinstance(e, ResultEvent))
    assert result.session_id == "native-session"
    assert result.provider_name == "actual-provider"
    assert result.finish_reason == ("failed" if failed else "completed")
    assert result.duration_ms == 120
    request = next(e for e in events if isinstance(e, RequestEvent))
    assert request.timestamp == "2026-09-05T12:00:00Z"


@pytest.mark.asyncio
async def test_claude_preserves_selected_native_model_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = ResultMessage(
        subtype="success", duration_ms=10, duration_api_ms=8, is_error=False, num_turns=1,
        session_id="session", model_usage={
            "provider-model": {"inputTokens": 10, "outputTokens": 5, "cacheReadInputTokens": 20,
                               "cacheCreationInputTokens": 30, "webSearchRequests": 0, "costUSD": 0.02,
                               "contextWindow": 1000, "maxOutputTokens": 100,
                               "canonicalModel": "actual-model", "provider": "bedrock"},
        },
    )
    monkeypatch.setattr("daydream.backends.claude.ClaudeSDKClient", scripted_client([terminal]))
    events = [e async for e in ClaudeBackend("requested-model").execute(Path("/tmp"), "prompt")]
    cost = next(e for e in events if isinstance(e, CostEvent))
    assert cost.model_usage is not None
    usage = cost.model_usage["provider-model"]
    assert usage.model_name == "actual-model"
    assert usage.provider_name == "bedrock"
    assert (usage.input_tokens, usage.cached_tokens, usage.cache_creation_tokens) == (60, 20, 30)
    assert usage.cost_usd == 0.02
    result = next(e for e in events if isinstance(e, ResultEvent))
    assert result.model_name == "actual-model"
    assert result.provider_name == "bedrock"


def test_pi_preserves_mixed_tool_blocks_and_structured_details() -> None:
    result = {"content": [{"type": "text", "text": "caption"},
                          {"type": "image", "data": "public-image", "mimeType": "image/png"}],
              "details": {"count": 1}}
    assert json.loads(_render_tool_result(result)) == result


@pytest.mark.asyncio
async def test_request_event_does_not_fabricate_a_trajectory_step(tmp_path: Path) -> None:
    request = RequestEvent(prompt="effective prompt", output_schema={"type": "object"})
    assert isinstance(request, AgentEvent)
    recorder = TrajectoryRecorder(path=tmp_path / "trajectory.json", run_flow=DaydreamRunFlow.NORMAL,
                                  target_dir=tmp_path, agent_model_name="model", session_id="session")
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(request)
    assert recorder.steps == []
