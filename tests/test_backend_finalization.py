"""Invocation-local native finalization controls on cached backend instances."""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from daydream.backends import ClaudeRequestConfig, PiRequestConfig, RequestEvent, ResultEvent, ToolStartEvent
from daydream.backends.claude import ClaudeBackend
from daydream.backends.codex import CodexBackend
from daydream.backends.pi import PiBackend
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
    "type": "object",
    "properties": {"findings": {"type": "array", "items": {"type": "string"}}},
    "required": ["findings"],
    "additionalProperties": False,
}


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "effort"), [
    *[(kind, effort) for kind in ("pi", "codex") for effort in (None, "high", "low", "minimal")],
    ("pi", "off"),
    ("codex", "none"),
])
async def test_cli_finalization_controls_are_local_to_overlapping_calls(
    tmp_path: Path, kind: str, effort: str | None,
) -> None:
    backend = (PiBackend if kind == "pi" else CodexBackend)(
        model="fixture-model", reasoning_effort=effort,
    )
    started = asyncio.Event()
    commands: list[tuple[str, ...]] = []

    async def spawn(*args: str, **kwargs: Any) -> Any:
        commands.append(args)
        if len(commands) == 2:
            started.set()
        await asyncio.wait_for(started.wait(), timeout=3)
        if kind == "pi":
            return pi_process([
                json.dumps({"type": "agent_end", "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": '{"findings":[]}'}]},
                ]}),
            ])
        return codex_process([
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}),
        ])

    async def run(finalization: bool) -> list[Any]:
        return [event async for event in backend.execute(
            tmp_path, "finalize" if finalization else "discover",
            output_schema=_SCHEMA, finalization=finalization, max_turns=1,
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
    # Neither CLI implements max_turns, including when finalizing.
    assert final.config.max_turns is None
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
    else:
        assert any(f'model_reasoning_effort="{expected}"' in args for args in commands)
        assert all("--no-tools" not in args for args in commands)


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", [None, "high", "low"])
async def test_claude_finalization_keeps_schema_and_guards_without_shared_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effort: str | None,
) -> None:
    captured: dict[str, Any] = {}
    started = asyncio.Event()
    base = scripted_client([
        MockAssistantMessage(content=[MockToolUseBlock(
            id="schema", name="StructuredOutput", input={"findings": []},
        )]),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("custom_config", [False, True])
async def test_effective_finalization_configuration_reaches_exported_telemetry(custom_config: bool) -> None:
    from dataclasses import dataclass

    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from daydream.extensions.registry import Registry
    from daydream.observability.config import ObservabilityConfig
    from daydream.observability.runtime import trace_run
    from daydream.observability.spans import agent_scope, attempt_scope

    @dataclass(frozen=True)
    class CustomConfig(PiRequestConfig):
        private_data: str = "PRIVATE CONFIG"

    request_config = (CustomConfig if custom_config else PiRequestConfig)(
        finalization=True, no_tools=True, selected_tools_count=0,
    )
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    config = ObservabilityConfig(destinations=("memory",), capture_content=False)
    async with trace_run(config, registry, flow="review") as run:
        with agent_scope("review-finalize", backend="pi", model="fixture-model"):
            async with attempt_scope(1) as attempt:
                attempt.observe(RequestEvent(
                    prompt="PRIVATE PROMPT", reasoning_effort="low",
                    config=request_config,
                ))
        run.finish(0)
    span = next(span for span in exporter.get_finished_spans()
                if (span.attributes or {}).get("daydream.span.kind") == "attempt")
    attrs = dict(span.attributes or {})
    if custom_config:
        assert not any(key.startswith("daydream.request.config.") for key in attrs)
    else:
        assert attrs["daydream.request.config.finalization"] is True
        assert attrs["daydream.request.config.no_tools"] is True
        assert attrs["daydream.request.config.selected_tools_count"] == 0
    assert attrs["gen_ai.request.reasoning_effort"] == "low"
    assert "daydream.request.config.max_turns" not in attrs
    assert "PRIVATE PROMPT" not in str(attrs)
    assert "PRIVATE CONFIG" not in str(attrs)


def test_new_request_controls_preserve_existing_positional_config_arguments() -> None:
    from dataclasses import fields

    from daydream.backends import CodexRequestConfig, EffectiveRequestConfig, OspreyRequestConfig

    expected = {
        EffectiveRequestConfig: [
            "temperature", "max_turns", "read_only", "persist_session", "continuation_mode", "model_mode",
        ],
        ClaudeRequestConfig: [
            "permission_mode", "allowed_tools_count", "allowed_tools_present", "audit_tools_count",
            "audit_tools_present", "setting_sources_present", "native_output_format", "buffer_limit_bytes",
            "hooks_enabled",
        ],
        CodexRequestConfig: [
            "sandbox_mode", "experimental_json", "native_output_schema", "read_only_isolation",
        ],
        PiRequestConfig: [
            "selected_tools_count", "selected_tools_present", "no_skills", "schema_emulated",
        ],
    }
    common = expected[EffectiveRequestConfig]
    for config_type, names in expected.items():
        positional = [item.name for item in fields(config_type) if not item.kw_only]
        if config_type is EffectiveRequestConfig:
            assert positional == names
        else:
            assert positional == common + names
    assert EffectiveRequestConfig(0.5, 2, True).temperature == 0.5
    assert PiRequestConfig(0.5, 2, True, False, "fresh", "single", 4, True, True, False).no_skills is True
    osprey = OspreyRequestConfig(0.5, 2, True, False, "fresh", "single", True, False)
    assert osprey.persona_present is True
    assert osprey.toolset_present is False
