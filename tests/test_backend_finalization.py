"""Invocation-local native finalization controls on cached backend instances."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from daydream.backends import (
    ClaudeRequestConfig,
    PiRequestConfig,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ToolStartEvent,
)
from daydream.backends.claude import ClaudeBackend
from daydream.backends.codex import CodexBackend
from daydream.backends.pi import PiBackend
from daydream.extensions.registry import Registry
from daydream.observability.config import ObservabilityConfig
from daydream.observability.runtime import trace_run
from daydream.observability.spans import agent_scope, attempt_scope
from tests.harness.claude_sdk import (
    MockAssistantMessage,
    MockResultMessage,
    MockToolUseBlock,
    patch_claude_sdk,
    scripted_client,
)
from tests.harness.codex_replay import make_mock_process as codex_process
from tests.harness.pi_replay import make_mock_process as pi_process

_SCHEMA = {
    "type": "object", "properties": {"findings": {"type": "array", "items": {"type": "string"}}},
    "required": ["findings"], "additionalProperties": False,
}


def _case_dir(tmp_path: Path, index: int) -> Path:
    case = tmp_path / f"case-{index}"
    case.mkdir()
    return case


async def _cli_finalization_controls_are_local_to_overlapping_calls_case(
    tmp_path: Path, kind: str, effort: str | None,
) -> None:
    backend = (PiBackend if kind == "pi" else CodexBackend)(model="fixture-model", reasoning_effort=effort)
    started = asyncio.Event()
    commands: list[tuple[str, ...]] = []
    async def spawn(*args: str, **kwargs: Any) -> Any:
        commands.append(args)
        if len(commands) == 2:
            started.set()
        await asyncio.wait_for(started.wait(), timeout=3)
        if kind == "pi":
            message = {"role": "assistant", "content": [{"type": "text", "text": '{"findings":[]}'}],
                       "stopReason": "stop"}
            return pi_process([json.dumps(event) for event in [
                {"type": "turn_start"}, {"type": "message_end", "message": message},
                {"type": "turn_end", "message": message}, {"type": "agent_end", "messages": []},
                {"type": "agent_settled"}]])
        return codex_process([json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})])
    async def run(finalization: bool) -> list[Any]:
        return [event async for event in backend.execute(
            tmp_path, "finalize" if finalization else "discover",
            output_schema=_SCHEMA, finalization=finalization,
        )]
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", side_effect=spawn):
        final_events, normal_events = await asyncio.gather(run(True), run(False))
    final = next(event for event in final_events if isinstance(event, RequestEvent))
    normal = next(event for event in normal_events if isinstance(event, RequestEvent))
    expected = effort if effort in {"none", "off", "minimal", "low"} else "low"
    assert final.reasoning_effort == expected
    assert normal.reasoning_effort == effort
    assert backend.reasoning_effort == effort
    assert final.config.finalization is True
    assert normal.config.finalization is False
    if kind == "pi":
        final_cmd = next(args for args in commands if "--no-tools" in args)
        normal_cmd = next(args for args in commands if "--no-tools" not in args)
        assert "--tools" not in final_cmd
        assert "--system-prompt" in final_cmd
        assert "--append-system-prompt" not in final_cmd
        assert "--append-system-prompt" in normal_cmd
        assert final_cmd[final_cmd.index("--thinking") + 1] == expected
        assert isinstance(final.config, PiRequestConfig)
        assert final.config.no_tools is True
        assert final.config.selected_tools_count == 0
        assert final.config.schema_emulated is True
        assert isinstance(normal.config, PiRequestConfig) and normal.config.schema_emulated is False
        assert next(event for event in final_events if isinstance(event, ResultEvent)).structured_output is None
        assert [event.text for event in final_events if isinstance(event, TextEvent)] == ['{"findings":[]}']
        assert next(event for event in normal_events if isinstance(event, ResultEvent)).structured_output is None
    else:
        assert any(f'model_reasoning_effort="{expected}"' in args for args in commands)
        assert all("--no-tools" not in args for args in commands)


async def test_cli_finalization_controls_are_local_to_overlapping_calls(tmp_path: Path) -> None:
    for index, (kind, effort) in enumerate([("pi", "off"), ("pi", None), ("codex", "high"), ("codex", "none")]):
        try:
            await _cli_finalization_controls_are_local_to_overlapping_calls_case(
                tmp_path=_case_dir(tmp_path, index), kind=kind, effort=effort)
        except BaseException as exc:
            exc.add_note(f"case: {kind}/{effort}")
            raise


async def test_claude_finalization_keeps_schema_and_guards_without_shared_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    effort = "high"
    captured: dict[str, Any] = {}
    started = asyncio.Event()
    base = scripted_client([
        MockAssistantMessage(content=[MockToolUseBlock(id="schema", name="StructuredOutput", input={"findings": []})]),
        MockResultMessage(structured_output={"findings": []}),
    ])
    class Client(base):  # type: ignore[misc,valid-type]
        async def query(self, prompt: str) -> None:
            captured[prompt] = self.options
            if len(captured) == 2:
                started.set()
            await asyncio.wait_for(started.wait(), timeout=3)
    patch_claude_sdk(monkeypatch, Client)
    backend = ClaudeBackend(model="fixture-model", reasoning_effort=effort)
    async def run(finalization: bool) -> list[Any]:
        return [event async for event in backend.execute(
            tmp_path, "finalize" if finalization else "discover",
            output_schema=_SCHEMA, finalization=finalization, read_only=True,
        )]
    final_events, normal_events = await asyncio.gather(run(True), run(False))
    final_opts, normal_opts = captured["finalize"], captured["discover"]
    assert final_opts.effort == "low"
    assert normal_opts.effort == effort
    assert backend.reasoning_effort == effort
    assert final_opts.tools == []
    assert normal_opts.tools is None
    assert final_opts.strict_mcp_config is True
    assert final_opts.mcp_servers == {}
    assert final_opts.output_format == {"type": "json_schema", "schema": _SCHEMA}
    # Verify the actual installed SDK command builder, not only options fields.
    final_opts.cli_path = "/fixture/claude"
    command = SubprocessCLITransport(prompt="finalize", options=final_opts)._build_command()
    assert command[command.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in command
    assert json.loads(command[command.index("--json-schema") + 1]) == _SCHEMA
    callbacks = [hook for matcher in final_opts.hooks["PreToolUse"] for hook in matcher.hooks]
    assert {hook.__name__ for hook in callbacks} == {
        "_dangerous_command_guard", "_background_bash_guard", "_read_only_guard", "_finalization_guard",
    }
    for callback in callbacks:
        assert await callback({"tool_name": "StructuredOutput"}, "schema", {}) == {}
    final_guard = callbacks[-1]
    for tool in ["Read", "Bash", "Grep", "mcp__external__query", "Agent", None]:
        decision = await final_guard({"tool_name": tool}, "blocked", {})
        assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert not any(isinstance(event, ToolStartEvent) for event in final_events)
    assert next(event for event in final_events if isinstance(event, ResultEvent)).structured_output == {"findings": []}
    final_request = next(event for event in final_events if isinstance(event, RequestEvent))
    normal_request = next(event for event in normal_events if isinstance(event, RequestEvent))
    assert final_request.reasoning_effort == "low"
    assert normal_request.reasoning_effort == effort
    assert isinstance(final_request.config, ClaudeRequestConfig)
    assert final_request.config.tools_count == 0

async def test_effective_finalization_configuration_reaches_exported_telemetry() -> None:
    """A custom config subclass exports no request-config fields, so private fields cannot leak."""
    @dataclass(frozen=True)
    class CustomConfig(PiRequestConfig):
        private_data: str = "PRIVATE CONFIG"
    request_config = CustomConfig(finalization=True, no_tools=True, selected_tools_count=0)
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    config = ObservabilityConfig(destinations=("memory",), capture_content=False)
    async with trace_run(config, registry, flow="review") as run:
        with agent_scope("review-finalize", backend="pi", model="fixture-model"):
            async with attempt_scope(1) as attempt:
                attempt.observe(RequestEvent(prompt="PRIVATE PROMPT", reasoning_effort="low", config=request_config, ))
        run.finish(0)
    span = next(span for span in exporter.get_finished_spans()
                if (span.attributes or {}).get("daydream.span.kind") == "attempt")
    attrs = dict(span.attributes or {})
    assert not any(key.startswith("daydream.request.config.") for key in attrs)
    assert attrs["gen_ai.request.reasoning_effort"] == "low"
    assert "PRIVATE PROMPT" not in str(attrs)
    assert "PRIVATE CONFIG" not in str(attrs)
