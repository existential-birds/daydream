"""Real Claude/Codex/Pi/Osprey acceptance through runner and loopback OTLP.

Only external SDK/CLI boundaries are doubled: SDK-shaped Claude messages,
public Codex JSONL, labeled canned Pi JSONL, and strict Osprey process fixtures.
Check options, hierarchy, aggregate/capability evidence, usage, two turns, and
privacy. Backend-specific canaries expose cross-task inheritance. Generic gRPC
and all destination HTTP transports retain their real encoders.

The sanitized Pi replay contains placeholder text with pinned native start
1788690314289ms, 86,936 normalized input, 8,401 output, and $0.00402781."""

from __future__ import annotations

import json
import os
import textwrap
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk.types import AgentDefinition

from daydream import runner
from daydream.backends import GenerationEndEvent, GenerationStartEvent, RequestEvent
from daydream.backends.claude import ClaudeBackend
from daydream.run_config import RunConfig
from tests.conftest import ExtDir
from tests.harness.claude_sdk import (
    MockAssistantMessage,
    MockResultMessage,
    MockTextBlock,
    MockToolResultBlock,
    MockToolUseBlock,
    MockUserMessage,
    patch_claude_sdk,
    scripted_client,
)
from tests.harness.fake_cli_process import install_fake_cli_process
from tests.harness.otlp import TraceCollector, attributes, kind_of as _kind, otlp_collector

# Distinct backend canaries expose cross-task, fan-out, and destination leaks.
_CANARIES = {
    "claude": "canary-claude-opus-main-7f3a", "codex": "canary-codex-golden-91bd", "pi": "canary-pi-replay-53ce",
    "osprey": "canary-osprey-strict-2ab8",
}

_FLOW_IMPORTS = """
import json
from daydream.agent import run_agent
from daydream.extensions import FlowStep, Stop
from daydream.trajectory import DaydreamPhase
"""

_SINGLE_AGENT_BODY = """
output, _, aborted = await run_agent(
    ctx.backend_for("review"), ctx.work.repo, "inspect the sample", phase=DaydreamPhase.REVIEW,
)
(ctx.data["daydream_dir"] / "protocol-result.json").write_text(json.dumps({"output": str(output)}))
return Stop(1) if aborted else None
"""


def _flow(ext_dir: ExtDir, body: str = _SINGLE_AGENT_BODY) -> None:
    source = _FLOW_IMPORTS + "\nasync def probe(ctx):\n"
    source += textwrap.indent(body, "    ")
    source += "\ndef register(r):\n"
    source += "    r.register_phase(FlowStep(name='protocol-acceptance', run=probe))\n"
    source += "    r.set_flow('protocol-acceptance', ['protocol-acceptance'])\n"
    ext_dir.write_module(source)


def _flow_config(make_config: Callable[..., RunConfig], repo: Path, *, backend: str,) -> RunConfig:
    """Pin review_backend so the runner selects the intended real adapter instead of default Claude."""
    return make_config(repo, flow_name="protocol-acceptance", review_backend=backend)


def _configure_otlp(monkeypatch: pytest.MonkeyPatch, endpoint: str, *, protocol: str = "http/protobuf") -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", endpoint)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", protocol)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT", "2")
    monkeypatch.setenv("DAYDREAM_TRACE_TO", "otlp")


def _attempt(spans: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = _kind(spans, "attempt")
    assert len(attempts) == 1
    return attempts[0]


def _assert_portable_hierarchy(spans: list[dict[str, Any]]) -> None:
    """One root, one trace, complete chain: run > step > agent > attempt."""
    roots = [span for span in spans if not span.get("parentSpanId")]
    assert len(roots) == 1
    assert len({span["traceId"] for span in spans}) == 1
    by_id = {span["spanId"]: span for span in spans}
    attempt = _attempt(spans)
    agent = by_id[attempt["parentSpanId"]]
    assert attributes(agent)["daydream.span.kind"] == "agent"
    assert attributes(agent)["gen_ai.agent.name"] == "review"
    phase = by_id[agent["parentSpanId"]]
    assert attributes(phase)["daydream.span.kind"] == "step"
    assert phase["parentSpanId"] == roots[0]["spanId"]


def _assert_leak_free(receiver: TraceCollector, *canaries: str) -> None:
    payload = json.dumps([request["body"] for request in receiver.requests])
    for canary in canaries:
        assert canary in payload, f"canary {canary} missing from wire payload"
    home = os.environ.get("HOME", "/home/operator")
    assert home not in payload
    daydream_prices = os.environ.get("DAYDREAM_PRICES_FILE")
    if daydream_prices:
        assert daydream_prices not in payload


def _spawner(spawner: Any) -> tuple[tuple[str, ...], dict[str, Any]]:
    argvs = spawner.argvs
    assert len(argvs) == 1
    return argvs[0], spawner.kwargs[0]


# Claude: real ClaudeBackend through the actual SDK option surface


def _claude_messages(canary: str) -> list[Any]:
    usage = {"input_tokens": 60, "output_tokens": 12, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 5}
    return [MockAssistantMessage(
            content=[MockToolUseBlock(id="tool-claude-1", name="Read", input={"path": "src/main.py"})],
            model="claude-opus-4-5-20250901", usage=usage, message_id="msg-claude-1",
        ),
        MockUserMessage(
            content=[MockToolResultBlock(tool_use_id="tool-claude-1", content=f"tool result {canary}", is_error=False)]
        ),
        MockAssistantMessage(content=[MockTextBlock(f"done {canary}")], model="claude-opus-4-5-20250901", usage=usage,
            message_id="msg-claude-2",
        ),
        MockResultMessage(
            subtype="success", duration_ms=150, duration_api_ms=120, is_error=False, session_id="native-claude-session",
            stop_reason="end_turn", total_cost_usd=0.021, usage=usage, result=f"done {canary}",
        ),
    ]


class _RecordingClaudeBackend(ClaudeBackend):
    """Real ClaudeBackend that additionally records RequestEvents for assertion."""

    def __init__(self, recorded: list[RequestEvent], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._recorded = recorded

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            if isinstance(event, RequestEvent):
                self._recorded.append(event)
            yield event

async def test_claude_specialist_agents_make_aggregate_multi_model_without_claiming_single(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Specialists make the opaque SDK aggregate multi-model while retaining main-model provenance."""
    canary = _CANARIES["claude"]
    captured: dict[str, Any] = {}
    requests: list[RequestEvent] = []
    scripted = scripted_client(_claude_messages(canary), captured=captured)
    patch_claude_sdk(monkeypatch, scripted)
    agents = {"pattern-scanner": AgentDefinition(description="scan patterns", prompt="scan", model="sonnet",)}
    backend = _RecordingClaudeBackend(requests, model="claude-opus-5")
    events = []
    async for event in backend.execute(Path("/tmp"), f"scan {canary}", agents=agents):
        events.append(event)
    options = captured["options"]
    assert options.agents == agents
    assert options.model == "claude-opus-5"
    request = requests[0]
    assert request.config.model_mode == "multi_or_dynamic"
    assert request.model_name == "claude-opus-5"
    assert request.model_source == "configured"
    assert not any(isinstance(e, (GenerationStartEvent, GenerationEndEvent)) for e in events)


# Codex: real CodexBackend replaying committed public JSONL + multi-turn fake

# Pi: real PiBackend with the explicitly labeled long-generation replay


# Osprey: real OspreyBackend with a strict fake-process protocol fixture


def _osprey_lines(canary: str) -> list[str]:
    return [json.dumps({"event": "protocol", "version": 2}),
        json.dumps({
                "event": "session_start", "session_id": "native-osprey-session", "started_at": "2026-09-09T12:00:00Z",
                "model": "osprey-native-model", "provider": "osprey-native-provider",
            }
        ), json.dumps({"event": "turn_start", "turn_id": "t1", "timestamp": "2026-09-09T12:00:01Z"}),
        json.dumps({"event": "thinking_delta", "content": f"osprey thinking {canary}"}),
        json.dumps({"event": "text_delta", "content": f"osprey answer {canary}"}),
        json.dumps({"event": "tool_call", "tool_call_id": "call_osp_1", "tool_name": "read",
                "arguments": {"path": "src/osp.py"},
            }
        ), json.dumps({"event": "tool_result", "tool_call_id": "call_osp_1", "tool_name": "read", "status": "success",
                "content": "file body", "duration_ms": 41,
            }
        ),
        json.dumps({
                "event": "turn_end", "turn_id": "t1", "usage_reported": True, "duration_ms": 90, "prompt_tokens": 30,
                "completion_tokens": 6, "cached_tokens": 4, "cache_write_tokens": 2, "thinking_tokens": 3,
                "cost_usd": "0.007",
            }
        ), json.dumps({"event": "session_end", "outcome": "completed", "exit_code": 0, "total_cost_usd": "0.007",
                "total_prompt_tokens": 30, "total_completion_tokens": 6, "total_cached_tokens": 4,
                "total_cache_write_tokens": 2, "total_thinking_tokens": 3, "session_wallclock_ms": 120,
            }
        ),
    ]

async def test_osprey_strict_protocol_fixture_through_runner(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real OspreyBackend via strict fake process: argv, session usage, no gen child."""
    canary = _CANARIES["osprey"]
    _flow(ext_dir)
    spawner = install_fake_cli_process(monkeypatch, "osprey", lines=_osprey_lines(canary))
    with otlp_collector() as receiver:
        _configure_otlp(monkeypatch, receiver.base_url + "/v1/traces")
        assert await runner.run(_flow_config(make_config, feature_branch_repo, backend="osprey")) == 0

    # External argv shape, recorded at the real transport spawn seam.
    argv, _spawn_kwargs = _spawner(spawner)
    assert argv[0] == "osprey" and "agent" in argv and "--events-jsonl" in argv
    # Hidden/default temperature stays absent from the external surface.
    assert "--temperature" not in argv

    spans = receiver.spans
    _assert_portable_hierarchy(spans)
    attempt = _attempt(spans)
    billed = attributes(attempt)
    # Session-sourced authoritative totals keep the structural bill.
    assert billed["daydream.billing.owner"] == "structural_attempt"
    assert billed["gen_ai.usage.input_tokens"] == 30
    assert billed["gen_ai.usage.output_tokens"] == 6
    assert billed["gen_ai.usage.reasoning.output_tokens"] == 3
    assert billed["gen_ai.usage.cost"] == pytest.approx(0.007)
    assert billed["gen_ai.conversation.id"] == "native-osprey-session"
    tool = _kind(spans, "tool")[0]
    assert attributes(tool)["gen_ai.tool.name"] == "read"
    assert attributes(tool)["daydream.tool.duration_ms"] == 41
    assert attributes(tool)["daydream.tool.status"] == "success"
    # No generation child: Osprey is not native_generation_interval.
    assert _kind(spans, "generation") == []
    # Only sealed provider generations are CLIENT; local attempts remain INTERNAL.
    assert attempt["kind"] == "SPAN_KIND_INTERNAL"
    _assert_leak_free(receiver, canary)

# Real generic HTTP/protobuf + gRPC integration for the destinations (plan 791)


# Step 3 lifecycle matrix: real-runner retry with a failed billed attempt
