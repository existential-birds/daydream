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
import time
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

import anyio
import jsonschema
import pytest
from claude_agent_sdk.types import AgentDefinition

from daydream import runner
from daydream.backends import GenerationEndEvent, GenerationStartEvent, RequestEvent
from daydream.backends.claude import ClaudeBackend
from daydream.backends.osprey import OspreyBackend, OspreyConfig
from daydream.backends.pi import PiBackend
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
from tests.harness.otlp import TraceCollector, attributes, kind_of as _kind, otlp_collector, otlp_grpc_collector

# Distinct backend canaries expose cross-task, fan-out, and destination leaks.
_CANARIES = {
    "claude": "canary-claude-opus-main-7f3a", "codex": "canary-codex-golden-91bd", "pi": "canary-pi-replay-53ce",
    "osprey": "canary-osprey-strict-2ab8",
}

_FLOW_IMPORTS = """
import json
from daydream.agent import run_agent
from daydream.extensions import FlowStep
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


def _replay_lines(fixture_name: str, replacements: dict[str, str]) -> list[str]:
    fixture = Path(__file__).parent / "fixtures" / "pi_jsonl" / fixture_name
    lines = fixture.read_text(encoding="utf-8").strip().splitlines()
    for old, new in replacements.items():
        lines = [line.replace(old, new) for line in lines]
    return lines


def _pin_first_message_end_receipt(monkeypatch: pytest.MonkeyPatch, pinned_ns: int) -> None:
    """Pin the first Pi message-end receipt; later paired receipts advance monotonically.

    Capture the real clock before patching the shared stdlib time module to avoid recursion."""
    clock_state: dict[str, int] = {"reads": 0, "first_end_real": 0}
    real_time_ns = time.time_ns

    def pinned_time_ns() -> int:
        now = real_time_ns()
        reads = clock_state["reads"] + 1
        clock_state["reads"] = reads
        if reads < 2:
            # Pin the generation-start receipt one nanosecond before its end.
            return pinned_ns - 1
        if reads == 2:
            # First message_end host receipt — the exact historical instant.
            clock_state["first_end_real"] = now
            return pinned_ns
        # Later receipts advance by real elapsed ns (monotonic, never backward).
        return pinned_ns + (now - clock_state["first_end_real"])

    monkeypatch.setattr("daydream.backends.pi.time.time_ns", pinned_time_ns)


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

async def test_claude_real_backend_runner_trace_sdk_options_and_config(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner-driven Claude without specialists must retain configured single-model provenance."""
    canary = _CANARIES["claude"]
    _flow(ext_dir)
    captured: dict[str, Any] = {}
    messages = _claude_messages(canary)
    scripted = scripted_client(messages, captured=captured)
    patch_claude_sdk(monkeypatch, scripted)

    requests: list[RequestEvent] = []
    backend = _RecordingClaudeBackend(requests, model="claude-opus-5")
    install_backend(backend)

    with otlp_collector() as receiver:
        _configure_otlp(monkeypatch, receiver.base_url + "/v1/traces")
        assert await runner.run(_flow_config(make_config, feature_branch_repo, backend="claude")) == 0

    assert json.loads((feature_branch_repo / ".daydream/protocol-result.json").read_text())["output"].endswith(
        f"done {canary}"
    )
    # External SDK option surface, exactly what the backend built.
    options = captured["options"]
    assert options.model == "claude-opus-5"
    assert options.permission_mode == "bypassPermissions"
    assert options.cwd == str(feature_branch_repo)
    assert options.agents is None  # runner-driven attempt passes no agents
    # RequestEvent truth: configured main-model provenance survives.
    request = requests[0]
    assert request.model_name == "claude-opus-5"
    assert request.model_source == "configured"
    assert request.timestamp_source == "host_observed"
    assert request.config.model_mode == "single"
    # Wire hierarchy + structural aggregate classification.
    spans = receiver.spans
    _assert_portable_hierarchy(spans)
    attempt = _attempt(spans)
    billed = attributes(attempt)
    assert billed["gen_ai.operation.name"] == "invoke_agent"
    assert billed["daydream.invocation.aggregate"] is True
    assert billed["gen_ai.request.model"] == "claude-opus-5"
    # Configured-model provenance lives on the logical agent scope.
    assert attributes(_kind(spans, "agent")[0])["daydream.configured.model"] == "claude-opus-5"
    # Claude's three input buckets fold into the true total input.
    assert billed["gen_ai.usage.input_tokens"] == 60 + 20 + 5
    assert billed["gen_ai.usage.output_tokens"] == 12
    assert billed["gen_ai.usage.cost"] == 0.021
    assert billed["daydream.billing.owner"] == "structural_attempt"
    assert billed["daydream.models"] == ["claude-opus-4-5-20250901"]
    assert billed["gen_ai.response.finish_reasons"] == ["end_turn"]
    assert billed["gen_ai.conversation.id"] == "native-claude-session"
    # Local CLI/SDK attempts are INTERNAL; only sealed provider generations are CLIENT.
    assert attempt["kind"] == "SPAN_KIND_INTERNAL"
    assert "gen_ai.provider.name" not in billed or billed["gen_ai.provider.name"]
    _assert_leak_free(receiver, canary)

@pytest.mark.parametrize("content_mode", ["full", "metadata"])
@pytest.mark.parametrize("model_template", ["/{}/model", "api_key={}", "{}\nmodel", "{}" * 20])
async def test_claude_private_request_model_never_reaches_real_export_wire(
    content_mode: str, model_template: str, ext_dir: ExtDir, feature_branch_repo: Path,
    make_config: Callable[..., RunConfig], monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = "private-claude-model-4fca"
    model = model_template.format(*([private] * model_template.count("{}")))
    _flow(ext_dir, f"""
from daydream.backends.claude import ClaudeBackend
await run_agent(ClaudeBackend(model={model!r}), ctx.work.repo, "inspect sample", phase=DaydreamPhase.REVIEW)
""")
    captured: dict[str, Any] = {}
    patch_claude_sdk(monkeypatch, scripted_client(_claude_messages("ordinary reply"), captured=captured))
    with otlp_collector() as receiver:
        _configure_otlp(monkeypatch, receiver.base_url + "/v1/traces")
        monkeypatch.setenv("DAYDREAM_TRACE_CONTENT", content_mode)
        assert await runner.run(_flow_config(make_config, feature_branch_repo, backend="claude")) == 0
    assert captured["options"].model == model
    payload = json.dumps([request["body"] for request in receiver.requests])
    assert receiver.requests and private not in payload
    attempt = attributes(_attempt(receiver.spans))
    agent = attributes(_kind(receiver.spans, "agent")[0])
    assert "gen_ai.request.model" not in attempt
    assert "daydream.configured.model" not in agent
    assert attempt["daydream.request.model.diagnostic"] == agent["daydream.configured.model.diagnostic"]
    assert attempt["daydream.models"] == ["claude-opus-4-5-20250901"]


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

async def test_codex_real_backend_replays_public_golden_shape_through_runner(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig], monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = _CANARIES["codex"]
    _flow(ext_dir)
    # The sanitized one-turn golden capture has no canary; inject it in this prompt/answer.
    lines = ['{"type":"thread.started","thread_id":"th_golden_public"}', '{"type":"turn.started"}',
        '{"type":"item.started","item":{"id":"item_0","type":"command_execution",'
        '"command":"/bin/zsh -lc \'sed -n 1p README.md\'","status":"in_progress"}}',
        '{"type":"item.completed","item":{"id":"item_0","type":"command_execution",'
        '"command":"/bin/zsh -lc \'sed -n 1p README.md\'","aggregated_output":"sample",'
        '"exit_code":0,"status":"completed"}}',
        f'{{"type":"item.completed","item":{{"id":"item_1","type":"agent_message","text":"Codex answer {canary}"}}}}',
        '{"type":"turn.completed","usage":{"input_tokens":52101,"cached_input_tokens":39040,'
        '"output_tokens":225,"reasoning_output_tokens":79}}',
    ]
    spawner = install_fake_cli_process(monkeypatch, "codex", lines=lines)

    with otlp_collector() as receiver:
        _configure_otlp(monkeypatch, receiver.base_url + "/v1/traces")
        assert await runner.run(_flow_config(make_config, feature_branch_repo, backend="codex")) == 0

    # Codex omits cwd when it equals the caller’s directory.
    argv, _ = _spawner(spawner)
    assert argv[:4] == ("codex", "exec", "--experimental-json", "--model")
    assert "--sandbox" in argv and "--cd" in argv

    spans = receiver.spans
    _assert_portable_hierarchy(spans)
    attempt = _attempt(spans)
    billed = attributes(attempt)
    # Invocation-scoped turn_end totals land on the attempt.
    assert billed["gen_ai.usage.input_tokens"] == 52101
    assert billed["gen_ai.usage.output_tokens"] == 225
    assert billed["gen_ai.usage.cache_read.input_tokens"] == 39040
    assert billed["gen_ai.usage.reasoning.output_tokens"] == 79
    # Synthesized turn_end cost is not authoritative: no billing owner may close.
    assert billed["daydream.billing.owner"] == "unresolved"
    # The observed review default uses the pinned gpt-5.6-sol price.
    assert billed["daydream.models"] == ["gpt-5.6-sol"]
    assert billed["gen_ai.usage.cost"] == pytest.approx(
        5.00 * (52101 - 39040) / 1_000_000 + 0.50 * 39040 / 1_000_000 + 30.0 * 225 / 1_000_000
    )
    assert billed["gen_ai.conversation.id"] == "th_golden_public"
    tools = _kind(spans, "tool")
    assert len(tools) == 1
    tool_attrs = attributes(tools[0])
    assert tool_attrs["gen_ai.tool.name"] == "shell"
    assert tool_attrs["daydream.tool.error"] is False
    # Only sealed provider generations are CLIENT; local attempts remain INTERNAL.
    assert attempt["kind"] == "SPAN_KIND_INTERNAL"
    _assert_leak_free(receiver, canary)

async def test_codex_multi_turn_replay_yields_two_tool_spans_and_isolated_turns(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-turn protocol shape the golden does not cover (plan §789)."""
    canary = _CANARIES["codex"]
    _flow(ext_dir)
    install_fake_cli_process(monkeypatch, "codex",
        lines=['{"type":"thread.started","thread_id":"th_two_turns"}', '{"type":"turn.started"}',
            '{"type":"item.started","item":{"id":"item_0","type":"command_execution",'
            '"command":"/bin/zsh -lc \'cat a.txt\'","status":"in_progress"}}',
            '{"type":"item.completed","item":{"id":"item_0","type":"command_execution",'
            '"command":"/bin/zsh -lc \'cat a.txt\'","aggregated_output":"A","exit_code":0,"status":"completed"}}',
            f'{{"type":"item.completed","item":{{"id":"item_1","type":"agent_message","text":"first {canary}"}}}}',
            '{"type":"turn.completed","usage":{"input_tokens":150,"output_tokens":75}}', '{"type":"turn.started"}',
            '{"type":"item.started","item":{"id":"item_2","type":"command_execution",'
            '"command":"/bin/zsh -lc \'cat b.txt\'","status":"in_progress"}}',
            '{"type":"item.completed","item":{"id":"item_2","type":"command_execution",'
            '"command":"/bin/zsh -lc \'cat b.txt\'","aggregated_output":"B","exit_code":0,"status":"completed"}}',
            f'{{"type":"item.completed","item":{{"id":"item_3","type":"agent_message","text":"second {canary}"}}}}',
            '{"type":"turn.completed","usage":{"input_tokens":200,"output_tokens":100}}',
        ],
    )
    with otlp_collector() as receiver:
        _configure_otlp(monkeypatch, receiver.base_url + "/v1/traces")
        assert await runner.run(_flow_config(make_config, feature_branch_repo, backend="codex")) == 0
    spans = receiver.spans
    tools = _kind(spans, "tool")
    assert len(tools) == 2
    assert len({attributes(tool)["gen_ai.tool.call.id"] for tool in tools}) == 2
    billed = attributes(_attempt(spans))
    # Invocation totals use the last turn.completed value, never a sum.
    assert billed["gen_ai.usage.input_tokens"] == 200
    assert billed["gen_ai.usage.output_tokens"] == 100
    _assert_leak_free(receiver, canary)


# Pi: real PiBackend with the explicitly labeled long-generation replay

async def test_pi_replay_exact_native_timing_choice_and_billing_through_runner(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the 395.332s replay’s ordered provider choice and exact usage/timing.

    End the generation once at historical pre-tool time after resolving child billing;
    retain response/reasoning evidence without inventing child input history."""
    canary = _CANARIES["pi"]
    _flow(ext_dir)
    # Inject the run’s canary so isolation checks have positive content evidence.
    lines = _replay_lines("long_generation_replay.jsonl",
        {"REPLAY_TEXT_ONE": f"replay one {canary}", "REPLAY_TEXT_TWO": f"replay two {canary}"},
    )
    spawner = install_fake_cli_process(monkeypatch, "pi", lines=lines)
    # Pin message_end at 1788690709621000000 ns to prove the 395.332-second interval.
    _pin_first_message_end_receipt(monkeypatch, 1788690709621000000)

    with otlp_collector() as receiver:
        _configure_otlp(monkeypatch, receiver.base_url + "/v1/traces")
        assert await runner.run(_flow_config(make_config, feature_branch_repo, backend="pi")) == 0

    # External argv shape, recorded at the real transport spawn seam.
    argv, spawn_kwargs = _spawner(spawner)
    assert argv[0] == "pi" and argv[argv.index("--mode") + 1] == "json"
    assert "--no-skills" in argv
    assert spawn_kwargs.get("cwd") == str(feature_branch_repo)

    spans = receiver.spans
    generations = _kind(spans, "generation")
    assert len(generations) == 2
    first = generations[0]
    gen_attrs = attributes(first)
    # Exact native numeric-ms start → exact ns; host message_end receipt end.
    assert gen_attrs["daydream.generation.native_started_at_unix_ms"] == 1788690314289
    assert gen_attrs["daydream.generation.native_started_at_unix_ns"] == 1788690314289000000
    assert gen_attrs["daydream.generation.sealed_end_unix_ns"] == 1788690709621000000
    assert gen_attrs["daydream.generation.duration_ns"] == 395332000000
    assert int(first["startTimeUnixNano"]) == 1788690314289000000
    assert int(first["endTimeUnixNano"]) == 1788690709621000000
    # Chronology: the sealed generation ends BEFORE the later tool sibling.
    attempt = _attempt(spans)
    billed = attributes(attempt)
    tools = _kind(spans, "tool")
    assert len(tools) == 1
    assert int(tools[0]["startTimeUnixNano"]) >= int(first["endTimeUnixNano"])
    assert first["parentSpanId"] == attempt["spanId"]
    # Provider choice order precedes tool execution; typed kind becomes wire type.
    choice = json.loads(gen_attrs["daydream.generation.choice_parts"])
    assert [part["type"] for part in choice] == ["reasoning", "text", "tool_call"]
    assert choice[2]["id"] == "call_replay_001"
    assert choice[2]["name"] == "read_file"
    assert choice[2]["arguments"] == {"path": "src/replay.py"}
    # Native response identity differs from invocation-local correlation identity.
    assert gen_attrs["daydream.generation.id"] != "resp_replay_01"
    # The structural attempt owns billing; unbilled children retain custom evidence only.
    assert gen_attrs["daydream.generation.billed"] is False
    assert "gen_ai.response.model" not in gen_attrs
    assert "gen_ai.provider.name" not in gen_attrs
    assert "gen_ai.response.finish_reasons" not in gen_attrs
    assert billed["gen_ai.response.model"] == "glm-5.3-flash"
    assert billed["gen_ai.provider.name"] == "nous"
    assert billed["gen_ai.response.finish_reasons"] == ["stop"]
    # Invocation prompts must not masquerade as unavailable generation input history.
    assert "gen_ai.input.messages" not in gen_attrs
    # Uncorrelated turn_end usage bills the authoritative attempt, not generation children.
    assert billed["daydream.billing.owner"] == "structural_attempt"
    assert gen_attrs["daydream.generation.billed"] is False
    assert "gen_ai.usage.input_tokens" not in gen_attrs
    # Normalized input is 10,392+76,544=86,936; historical cost remains telemetry.
    assert billed["gen_ai.usage.input_tokens"] == 86936
    assert billed["gen_ai.usage.output_tokens"] == 8401
    assert billed["gen_ai.usage.cost"] == pytest.approx(0.00402781)
    assert "daydream.generation.duration_ns" not in billed
    # Only sealed provider generations are CLIENT; local attempts remain INTERNAL.
    assert attempt["kind"] == "SPAN_KIND_INTERNAL"
    _assert_leak_free(receiver, canary)

@pytest.mark.parametrize("vendor", ["otlp", "honeyhive", "langsmith"])
async def test_pi_metadata_mode_omits_generation_choice_content(
    vendor: str, ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = _CANARIES["pi"]
    _flow(ext_dir)
    lines = _replay_lines("long_generation_replay.jsonl",
        {"REPLAY_TEXT_ONE": f"replay one {canary}", "REPLAY_TEXT_TWO": f"replay two {canary}"},
    )
    install_fake_cli_process(monkeypatch, "pi", lines=lines)
    _pin_first_message_end_receipt(monkeypatch, 1788690709621000000)

    expected_paths = {"otlp": "/v1/traces", "honeyhive": "/opentelemetry/v1/traces", "langsmith": "/otel/v1/traces"}
    with otlp_collector() as receiver:
        _vendor_env(monkeypatch, vendor, receiver.base_url)
        monkeypatch.setenv("DAYDREAM_TRACE_TO", vendor)
        monkeypatch.setenv("DAYDREAM_TRACE_CONTENT", "metadata")
        assert await runner.run(_flow_config(make_config, feature_branch_repo, backend="pi")) == 0

    spans = receiver.spans
    generations = _kind(spans, "generation")
    assert len(generations) == 2
    # Preserve each native start; pin the first end receipt and advance later receipts monotonically.
    native_starts = sorted(
        attributes(generation)["daydream.generation.native_started_at_unix_ms"] for generation in generations
    )
    assert native_starts == [1788690314289, 1788690314500]
    ends = sorted(int(generation["endTimeUnixNano"]) for generation in generations)
    assert ends[0] == 1788690709621000000
    assert ends[1] > ends[0]
    for generation in generations:
        gen_attrs = attributes(generation)
        # Content omitted in metadata mode...
        assert "daydream.generation.choice_parts" not in gen_attrs
        assert "gen_ai.output.messages" not in gen_attrs
        # ...while counts/identity/timing evidence survives.
        assert gen_attrs["daydream.generation.billed"] is False
    attempt = _attempt(spans)
    billed = attributes(attempt)
    assert billed["gen_ai.usage.input_tokens"] == 86936
    assert billed["gen_ai.usage.cost"] == pytest.approx(0.00402781)
    payload = json.dumps([request["body"] for request in receiver.requests])
    for content in (canary, "replay one", "replay two", "src/replay.py"):
        assert content not in payload, f"metadata mode leaked: {content}"
    assert {request["path"] for request in receiver.requests} == {expected_paths[vendor]}
    home = os.environ.get("HOME", "/home/operator")
    assert home not in payload

async def test_pi_generation_lifecycle_fixture_two_generations_around_one_tool(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig], monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = _CANARIES["pi"]
    _flow(ext_dir)
    install_fake_cli_process(
        monkeypatch, "pi", lines=_replay_lines("generation_lifecycle.jsonl", {"src/example.py": f"src/{canary}.py"}),
    )
    # Pin gen0’s host receipt strictly between the two native generation starts.
    _pin_first_message_end_receipt(monkeypatch, 1788690315000000000)
    with otlp_collector() as receiver:
        _configure_otlp(monkeypatch, receiver.base_url + "/v1/traces")
        assert await runner.run(_flow_config(make_config, feature_branch_repo, backend="pi")) == 0
    spans = receiver.spans
    generations = _kind(spans, "generation")
    assert len(generations) == 2
    assert int(generations[0]["startTimeUnixNano"]) == 1788690314289000000
    assert int(generations[0]["endTimeUnixNano"]) == 1788690315000000000
    assert int(generations[1]["startTimeUnixNano"]) > int(generations[0]["endTimeUnixNano"])
    tools = _kind(spans, "tool")
    assert len(tools) == 1
    tool_input = json.loads(attributes(tools[0])["traceloop.entity.input"])
    assert tool_input["path"] == f"src/{canary}.py"
    assert attributes(tools[0])["gen_ai.tool.call.id"] == "call_001"
    choice = json.loads(attributes(generations[0])["daydream.generation.choice_parts"])
    assert choice[2]["arguments"] == {"path": f"src/{canary}.py"}
    _assert_leak_free(receiver, canary)


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

@pytest.mark.parametrize("content_mode", ["full", "metadata"])
async def test_osprey_private_config_never_reaches_real_export_wire(
    content_mode: str, ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = "private-osprey-native-config-9d03"
    _flow(ext_dir, f"""
from pathlib import Path
from daydream.backends.osprey import OspreyBackend, OspreyConfig
backend = OspreyBackend(OspreyConfig(
    model={"/" + private + "/model"!r}, osprey_binary={private + "-binary"!r},
    persona={private + "-persona"!r}, toolset={private + "-toolset"!r},
    allowed_roots=[Path({"/" + private + "/root"!r})],
    atif_output=Path({"/" + private + "/atif.json"!r}),
    tool_result_raw_dir=Path({"/" + private + "/raw"!r}),
    osprey_home=Path({"/" + private + "/home"!r}),
    vars=[({private + "-key"!r}, {private + "-value"!r})],
))
await run_agent(backend, ctx.work.repo, "inspect sample", phase=DaydreamPhase.REVIEW)
""")
    spawner = install_fake_cli_process(monkeypatch, private + "-binary", lines=_osprey_lines("ordinary reply"))
    with otlp_collector() as receiver:
        _configure_otlp(monkeypatch, receiver.base_url + "/v1/traces")
        monkeypatch.setenv("DAYDREAM_TRACE_CONTENT", content_mode)
        assert await runner.run(_flow_config(make_config, feature_branch_repo, backend="osprey")) == 0
    argv, kwargs = _spawner(spawner)
    assert argv[0] == private + "-binary"
    assert kwargs["env"]["OSPREY_HOME"] == "/" + private + "/home"
    payload = json.dumps([request["body"] for request in receiver.requests])
    assert receiver.requests and private not in payload
    attrs = attributes(_attempt(receiver.spans))
    assert attrs["daydream.request.config.persona_present"] is True
    assert attrs["daydream.request.config.toolset_present"] is True
    assert attrs["daydream.request.config.vars_count"] == 1


async def test_osprey_explicit_zero_temperature_reaches_argv_and_config(
    monkeypatch: pytest.MonkeyPatch, feature_branch_repo: Path,
) -> None:
    canary = _CANARIES["osprey"]
    spawner = install_fake_cli_process(monkeypatch, "osprey", lines=_osprey_lines(canary))
    backend = OspreyBackend(OspreyConfig(osprey_binary="osprey", temperature=0.0))
    events = []
    async for event in backend.execute(feature_branch_repo, f"prompt {canary}"):
        events.append(event)
    argv, _kwargs = _spawner(spawner)
    assert "--temperature" in argv
    assert argv[argv.index("--temperature") + 1] == "0.0"
    request = next(e for e in events if isinstance(e, RequestEvent))
    assert request.config.temperature == 0.0  # zero retained, never omitted
    assert request.model_source == "native"


# Schema-valid content on the wire (pinned semconv message schemas)

async def test_attempt_input_messages_validate_against_pinned_schema(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig], monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = _CANARIES["pi"]
    _flow(ext_dir)
    install_fake_cli_process(
        monkeypatch, "pi", lines=_replay_lines("simple_text.jsonl", {"Hello from Pi": f"pi reply {canary}"}),
    )
    with otlp_collector() as receiver:
        _configure_otlp(monkeypatch, receiver.base_url + "/v1/traces")
        assert await runner.run(_flow_config(make_config, feature_branch_repo, backend="pi")) == 0
    schema = json.loads(
        (Path(__file__).parent / "fixtures" / "observability_semconv" / "gen-ai-input-messages.json").read_text()
    )
    messages = json.loads(attributes(_attempt(receiver.spans))["gen_ai.input.messages"])
    jsonschema.validate(messages, schema)


# Real generic HTTP/protobuf + gRPC integration for the destinations (plan 791)


def _vendor_env(monkeypatch: pytest.MonkeyPatch, vendor: str, base: str) -> None:
    if vendor == "honeyhive":
        monkeypatch.setenv("HH_API_URL", base)
        monkeypatch.setenv("HH_API_KEY", "opaque-vendor-key")
    elif vendor == "langsmith":
        monkeypatch.setenv("LANGSMITH_ENDPOINT", base)
        monkeypatch.setenv("LANGSMITH_API_KEY", "opaque-vendor-key")
    else:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", base + "/v1/traces")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT", "2")

@pytest.mark.parametrize("vendor", ["otlp", "honeyhive", "langsmith"])
async def test_generic_http_protobuf_reaches_every_destination(
    vendor: str, ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = _CANARIES["pi"]
    _flow(ext_dir)
    install_fake_cli_process(
        monkeypatch, "pi", lines=_replay_lines("simple_text.jsonl", {"Hello from Pi": f"pi reply {canary}"}),
    )
    expected_paths = {"otlp": "/v1/traces", "honeyhive": "/opentelemetry/v1/traces", "langsmith": "/otel/v1/traces"}
    with otlp_collector() as receiver:
        _vendor_env(monkeypatch, vendor, receiver.base_url)
        monkeypatch.setenv("DAYDREAM_TRACE_TO", vendor)
        assert await runner.run(_flow_config(make_config, feature_branch_repo, backend="pi")) == 0
    assert receiver.requests
    assert {request["path"] for request in receiver.requests} == {expected_paths[vendor]}
    headers = receiver.requests[0]["headers"]
    assert headers["content-type"] == "application/x-protobuf"
    if vendor == "honeyhive":
        assert headers.get("authorization") == "Bearer opaque-vendor-key"
    elif vendor == "langsmith":
        assert headers.get("x-api-key") == "opaque-vendor-key"
        assert "langsmith-project" in headers
    spans = receiver.spans
    assert _attempt(spans)
    _assert_leak_free(receiver, canary)

async def test_generic_grpc_transport_reaches_real_loopback_server(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig], monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = _CANARIES["osprey"]
    _flow(ext_dir)
    install_fake_cli_process(monkeypatch, "osprey", lines=_osprey_lines(canary))
    with otlp_grpc_collector() as receiver:
        # Explicit http:// selects plaintext; scheme-less host:port would attempt TLS.
        _configure_otlp(monkeypatch, f"http://{receiver.base_url}", protocol="grpc")
        assert await runner.run(_flow_config(make_config, feature_branch_repo, backend="osprey")) == 0
    assert receiver.requests
    assert _attempt(receiver.spans)
    metadata = receiver.requests[0]["headers"]
    # gRPC hides HTTP/2 content-type metadata; inspect its user-agent and decoded protobuf instead.
    assert "user-agent" in metadata
    assert "grpc" in metadata["user-agent"]
    assert receiver.requests[0]["path"] == "/grpc"
    _assert_leak_free(receiver, canary)

async def test_grpc_outage_fails_open_and_review_completes(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig], monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _flow(ext_dir)
    install_fake_cli_process(monkeypatch, "osprey", lines=_osprey_lines(_CANARIES["osprey"]))
    with otlp_grpc_collector(reject=True) as receiver:
        # Plaintext http:// ensures the failure is genuine UNAVAILABLE rather than a TLS error.
        _configure_otlp(monkeypatch, f"http://{receiver.base_url}", protocol="grpc")
        with anyio.fail_after(30):
            assert await runner.run(_flow_config(make_config, feature_branch_repo, backend="osprey")) == 0
    assert json.loads((feature_branch_repo / ".daydream/protocol-result.json").read_text())["output"].endswith(
        _CANARIES["osprey"]
    )
    assert "Trace export failed" in caplog.text


# Step 3 lifecycle matrix: real-runner retry with a failed billed attempt


_PI_RETRY_FAILED_LINES = [
    json.dumps({"type": "session", "sessionId": "pi_ses_retry"}), json.dumps({"type": "agent_start"}),
    json.dumps({"type": "turn_start"}),
    json.dumps(
        {"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "partial"}]}}
    ), json.dumps({"type": "turn_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "partial"}], "stopReason": "error",
                "errorMessage": "429 too many requests",
                "usage": {"input": 500, "output": 20, "cacheRead": 0, "cost": {"total": 0.003}},
            },
        }
    ), json.dumps({"type": "agent_end", "messages": []}),
]

async def test_runner_failed_billed_attempt_wears_its_own_bill_real_pi(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing real Pi attempt must export ERROR and its own structural usage before raising."""
    _flow(ext_dir)
    # Disable Pi’s 20-retry default here; runner-level retry separation has its own test.
    backend = PiBackend()
    backend.retry_attempts = 0
    install_backend(backend)
    install_fake_cli_process(monkeypatch, "pi", lines=_PI_RETRY_FAILED_LINES)
    with otlp_collector() as receiver:
        _configure_otlp(monkeypatch, receiver.base_url + "/v1/traces")
        with pytest.raises(Exception, match="429 too many requests"):
            await runner.run(_flow_config(make_config, feature_branch_repo, backend="pi"))

    spans = receiver.spans
    attempts = _kind(spans, "attempt")
    assert len(attempts) == 1
    failed = attributes(attempts[0])
    assert attempts[0]["status"]["code"] == "STATUS_CODE_ERROR"
    assert failed["gen_ai.usage.input_tokens"] == 500
    assert failed["gen_ai.usage.output_tokens"] == 20
    assert failed["gen_ai.usage.cost"] == pytest.approx(0.003)
    assert failed["daydream.billing.owner"] == "structural_attempt"
    # No result file: the failure surfaced to the caller.
    assert not (feature_branch_repo / ".daydream/protocol-result.json").exists()


@pytest.mark.parametrize("content_mode", ["full", "metadata"])
@pytest.mark.parametrize("source", ["assistant", "usage-key", "usage-model", "usage-provider"])
@pytest.mark.parametrize("template", ["/{}/model", "api_key={}", "{}\nmodel", "{}" * 20])
async def test_claude_response_identity_admission_on_real_export_wire(
    content_mode: str, source: str, template: str, ext_dir: ExtDir, feature_branch_repo: Path,
    make_config: Callable[..., RunConfig], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from claude_agent_sdk.types import AssistantMessage, ModelUsage, ResultMessage, TextBlock

    private = "private-native-response-8fe2"
    label = template.format(*([private] * template.count("{}")))
    _flow(ext_dir)
    usage = {"input_tokens": 60, "output_tokens": 12}
    native_model = label if source == "assistant" else "ordinary-response-model"
    per_model = ModelUsage(
        inputTokens=60, outputTokens=12, cacheReadInputTokens=0, cacheCreationInputTokens=0,
        webSearchRequests=0, costUSD=0.021, contextWindow=100000, maxOutputTokens=1000,
        canonicalModel=label if source == "usage-model" else "ordinary-canonical-model",
        provider=label if source == "usage-provider" else "ordinary-provider",
    )
    messages = [
        AssistantMessage(content=[TextBlock("ordinary answer")], model=native_model, usage=usage),
        ResultMessage(
            subtype="success", duration_ms=150, duration_api_ms=120, is_error=False, num_turns=1,
            session_id="ordinary-session", total_cost_usd=0.021, usage=usage, result="ordinary answer",
            model_usage={label if source == "usage-key" else "ordinary-bucket": per_model},
        ),
    ]
    captured: dict[str, Any] = {}
    client = scripted_client(messages, captured=captured)
    monkeypatch.setattr("daydream.backends.claude.ClaudeSDKClient", client)
    monkeypatch.setattr("daydream.backends.claude._RunLocalClaudeSDKClient", lambda **kwargs: client(kwargs["options"]))
    with otlp_collector() as receiver:
        _configure_otlp(monkeypatch, receiver.base_url + "/v1/traces")
        monkeypatch.setenv("DAYDREAM_TRACE_CONTENT", content_mode)
        assert await runner.run(_flow_config(make_config, feature_branch_repo, backend="claude")) == 0
    assert captured["prompt"] == "inspect the sample"
    assert json.loads((feature_branch_repo / ".daydream/protocol-result.json").read_text()) == {
        "output": "ordinary answer",
    }
    payload = json.dumps([request["body"] for request in receiver.requests])
    assert receiver.requests and private not in payload
    attempt = attributes(_attempt(receiver.spans))
    assert attempt["gen_ai.usage.input_tokens"] == 60
    assert attempt["gen_ai.usage.output_tokens"] == 12
    assert attempt["gen_ai.usage.cost"] == 0.021
    assert attempt["daydream.billing.owner"] == "structural_attempt"
    assert attempt["daydream.backend_diagnostic.codes"]
    if source == "assistant":
        assert "gen_ai.response.model" not in attempt
        assert "daydream.models" not in attempt
        assert json.loads(attempt["daydream.message_usage"])[0]["model"] is None
    else:
        assert attempt["gen_ai.response.model"] == "ordinary-response-model"
        assert attempt["daydream.models"] == ["ordinary-response-model"]
    model_usage = json.loads(attempt["daydream.model_usage"])
    if source in ("usage-key", "usage-model"):
        assert model_usage == {}
    else:
        assert model_usage["ordinary-bucket"]["model_name"] == "ordinary-canonical-model"
        assert model_usage["ordinary-bucket"]["provider_name"] == (
            None if source == "usage-provider" else "ordinary-provider"
        )
        assert model_usage["ordinary-bucket"]["cost_usd"] == 0.021
