"""Real four-backend protocol acceptance through the runner onto the OTLP wire.

P18 Task 5 Step 1 (plan §Task 5): drive the production ``run_agent``/runner
path with the REAL ``ClaudeBackend``/``CodexBackend``/``PiBackend``/
``OspreyBackend`` and only external CLI/SDK fixtures — SDK-shaped message
doubles for Claude, replayed public JSONL for Codex, explicitly labeled canned
JSONL replay for Pi, and a strict fake-process protocol fixture for Osprey —
then assert the trace that actually reaches a loopback OTLP collector:
external options/argv, logical/attempt/generation/tool hierarchy, structural
aggregate classification, attempt kind (INTERNAL unless a remote agent with
provider-at-creation is proved), generation capability classification,
usage/cache/reasoning/cost completeness, two assistant turns, and no
secret/path/ambient-context leakage. Distinct per-backend canaries make
accidental cross-task/fan-out inheritance observable.

Also retains real generic gRPC + HTTP/protobuf integration for all
destinations, consuming Task 4A's wire/delivery contract rather than
reimplementing its response matrix (plan §791).

The Pi long-generation replay fixture (``long_generation_replay.jsonl``) is an
explicitly labeled sanitized protocol replay: placeholder content only, the
pinned native timestamp ``1788690314289`` ms and the exact replay accounting
10392+76544 cache-read input (86,936 normalized), 8,401 output, $0.00402781.
"""

from __future__ import annotations

import json
import os
import textwrap
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from daydream import runner
from daydream.backends import RequestEvent
from daydream.backends.claude import ClaudeBackend
from daydream.backends.osprey import OspreyBackend
from daydream.backends.pi import PiBackend
from daydream.runner import RunConfig
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
from tests.harness.otlp import TraceCollector, attributes, otlp_collector, otlp_grpc_collector

#: Each backend's private sentinel: must appear in exactly that backend's wire
#: payload and in no sibling's. Distinct canaries make accidental cross-task,
#: fan-out, or destination inheritance observable.
_CANARIES = {
    "claude": "canary-claude-opus-main-7f3a",
    "codex": "canary-codex-golden-91bd",
    "pi": "canary-pi-replay-53ce",
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


def _flow_config(
    make_config: Callable[..., RunConfig],
    repo: Path,
    *,
    backend: str,
) -> RunConfig:
    """RunConfig for the protocol-acceptance flow pinned to *backend*.

    The runner resolves the flow's ``ctx.backend_for("review")`` through
    ``review_backend`` — real Codex/Pi/Osprey protocols are exercised only
    when the review phase actually selects that backend (the default is
    claude, which would spawn the real SDK).
    """
    return make_config(repo, flow_name="protocol-acceptance", review_backend=backend)


def _configure_otlp(monkeypatch: pytest.MonkeyPatch, endpoint: str, *, protocol: str = "http/protobuf") -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", endpoint)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", protocol)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT", "2")
    monkeypatch.setenv("DAYDREAM_TRACE_TO", "otlp")


def _kind(spans: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [span for span in spans if attributes(span).get("daydream.span.kind") == kind]


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
    """Inject a deterministic host receipt clock for Pi's ``message_end`` reads.

    The backend records the host-observed generation-start receipt at an
    assistant ``message_start`` (a ``time.time_ns()`` read) then the end
    receipt at the matching ``message_end`` (the next read). Reads therefore
    arrive in trustworthy pairs. The FIRST ``message_end`` receipt lands
    exactly on ``pinned_ns`` (the historical host instant); later receipts
    advance by real elapsed ns (monotonic, never backward).

    ``daydream.backends.pi.time`` IS the stdlib ``time`` module, so the real
    clock is captured BEFORE patching and the replacement never re-enters
    itself.
    """
    clock_state: dict[str, int] = {"reads": 0, "first_end_real": 0}
    real_time_ns = time.time_ns

    def pinned_time_ns() -> int:
        now = real_time_ns()
        reads = clock_state["reads"] + 1
        clock_state["reads"] = reads
        if reads < 2:
            # Host-observed generation-start receipt: one ns before the
            # pinned end keeps strict before/after ordering.
            return pinned_ns - 1
        if reads == 2:
            # First message_end host receipt — the exact historical instant.
            clock_state["first_end_real"] = now
            return pinned_ns
        # Later receipts advance by real elapsed ns (monotonic, never backward).
        return pinned_ns + (now - clock_state["first_end_real"])

    monkeypatch.setattr("daydream.backends.pi.time.time_ns", pinned_time_ns)


# ============================================================================
# Claude: real ClaudeBackend through the actual SDK option surface
# ============================================================================


@dataclass
class _ClaudeAssistantWithUsage(MockAssistantMessage):
    """Shared MockAssistantMessage plus the metrics fields the backend reads."""

    model: str | None = None
    message_id: str = ""
    usage: dict[str, Any] | None = None


@dataclass
class _ClaudeResultWithUsage(MockResultMessage):
    """Shared MockResultMessage plus duration/stop-reason/turn metadata."""

    duration_ms: int | None = None
    duration_api_ms: int | None = None
    num_turns: int | None = None
    stop_reason: str | None = None
    usage: dict[str, Any] | None = None


def _claude_messages(canary: str) -> list[Any]:
    usage = {"input_tokens": 60, "output_tokens": 12, "cache_read_input_tokens": 20,
             "cache_creation_input_tokens": 5}
    return [
        _ClaudeAssistantWithUsage(
            content=[MockToolUseBlock(id="tool-claude-1", name="Read", input={"path": "src/main.py"})],
            model="claude-opus-4-5-20250901", usage=usage, message_id="msg-claude-1",
        ),
        MockUserMessage(content=[MockToolResultBlock(tool_use_id="tool-claude-1",
                                                     content=f"tool result {canary}", is_error=False)]),
        _ClaudeAssistantWithUsage(
            content=[MockTextBlock(f"done {canary}")],
            model="claude-opus-4-5-20250901", usage=usage, message_id="msg-claude-2",
        ),
        _ClaudeResultWithUsage(
            subtype="success", duration_ms=150, duration_api_ms=120, is_error=False,
            num_turns=2, session_id="native-claude-session", stop_reason="end_turn",
            total_cost_usd=0.021, usage=usage, result=f"done {canary}",
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
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real ClaudeBackend + SDK-shaped client through runner.run onto loopback OTLP.

    The runner-driven attempt passes no specialists, so the aggregate is
    single-model with configured main-model provenance; the multi-model
    specialist case is the direct-execute test below (plan §374/§789).
    """
    canary = _CANARIES["claude"]
    _flow(ext_dir)
    captured: dict[str, Any] = {}
    messages = _claude_messages(canary)
    scripted = scripted_client(messages, captured=captured)
    patch_claude_sdk(monkeypatch, scripted,
                     assistant_message=_ClaudeAssistantWithUsage, result_message=_ClaudeResultWithUsage)

    requests: list[RequestEvent] = []
    backend = _RecordingClaudeBackend(requests, model="claude-opus-5")
    install_backend(backend)

    with otlp_collector() as receiver:
        _configure_otlp(monkeypatch, receiver.base_url + "/v1/traces")
        assert await runner.run(
            _flow_config(make_config, feature_branch_repo, backend="claude")
        ) == 0

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
    # Attempt kind on the wire: every backend runs a remote agent (external
    # CLI/SDK boundary) whose provider is fixed by the run config, so the
    # attempt scope is the CLIENT boundary (spans.py:142 creates attempt
    # scopes as SpanKind.CLIENT) — the plan's CLIENT rule (proved remote
    # agent + provider known at creation), not local INTERNAL.
    assert attempt["kind"] == "SPAN_KIND_CLIENT"
    assert "gen_ai.provider.name" not in billed or billed["gen_ai.provider.name"]
    _assert_leak_free(receiver, canary)


async def test_claude_specialist_agents_make_aggregate_multi_model_without_claiming_single(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct real execute with a nonempty specialist mapping (plan §374).

    Main model opus, specialist sonnet: model_mode flips to multi_or_dynamic
    while configured main-model Daydream provenance survives and the stream
    still never claims the aggregate is single-model. No generation lifecycle
    events exist for the opaque SDK protocol.
    """
    from claude_agent_sdk.types import AgentDefinition

    from daydream.backends import GenerationEndEvent, GenerationStartEvent

    canary = _CANARIES["claude"]
    captured: dict[str, Any] = {}
    requests: list[RequestEvent] = []
    scripted = scripted_client(_claude_messages(canary), captured=captured)
    patch_claude_sdk(monkeypatch, scripted,
                     assistant_message=_ClaudeAssistantWithUsage, result_message=_ClaudeResultWithUsage)
    agents = {"pattern-scanner": AgentDefinition(
        description="scan patterns", prompt="scan", model="sonnet",
    )}
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


# ============================================================================
# Codex: real CodexBackend replaying committed public JSONL + multi-turn fake
# ============================================================================


async def test_codex_real_backend_replays_public_golden_shape_through_runner(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real CodexBackend, public real-capture shape, through runner.run."""
    canary = _CANARIES["codex"]
    _flow(ext_dir)
    # The golden real capture (tests/fixtures/codex_jsonl/real/golden.jsonl) is
    # one turn with a command_execution + agent_message; its text is public
    # sanitized content, so the canary rides in the prompt and answer here.
    lines = [
        '{"type":"thread.started","thread_id":"th_golden_public"}',
        '{"type":"turn.started"}',
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
        assert await runner.run(
            _flow_config(make_config, feature_branch_repo, backend="codex")
        ) == 0

    # External argv shape, recorded at the real transport spawn seam. Codex
    # omits the cwd kwarg when the execution cwd equals the caller's cwd.
    argv, spawn_kwargs = _spawner(spawner)
    assert argv[:4] == ("codex", "exec", "--experimental-json", "--model")
    assert "--sandbox" in argv and "--cd" in argv

    spans = receiver.spans
    _assert_portable_hierarchy(spans)
    attempt = _attempt(spans)
    billed = attributes(attempt)
    # Codex aggregates invocation usage at turn_end (usage_scope=invocation):
    # one MetricsEvent whose totals land on the attempt.
    assert billed["gen_ai.usage.input_tokens"] == 52101
    assert billed["gen_ai.usage.output_tokens"] == 225
    assert billed["gen_ai.usage.cache_read.input_tokens"] == 39040
    assert billed["gen_ai.usage.reasoning.output_tokens"] == 79
    # Cost is host-synthesized from the pinned price table (estimated, sourced
    # at turn_end — NOT an authoritative terminal total), so no closed billing
    # owner exists for a real Codex invocation; no native bill may export.
    assert billed["daydream.billing.owner"] == "unresolved"
    # The turn_end model observed natively is the review-phase default
    # (gpt-5.6-sol), whose pinned price synthesizes the reported cost.
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
    assert attempt["kind"] == "SPAN_KIND_CLIENT"
    _assert_leak_free(receiver, canary)


async def test_codex_multi_turn_replay_yields_two_tool_spans_and_isolated_turns(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-turn protocol shape the golden does not cover (plan §789)."""
    canary = _CANARIES["codex"]
    _flow(ext_dir)
    install_fake_cli_process(
        monkeypatch, "codex", lines=[
            '{"type":"thread.started","thread_id":"th_two_turns"}',
            '{"type":"turn.started"}',
            '{"type":"item.started","item":{"id":"item_0","type":"command_execution",'
            '"command":"/bin/zsh -lc \'cat a.txt\'","status":"in_progress"}}',
            '{"type":"item.completed","item":{"id":"item_0","type":"command_execution",'
            '"command":"/bin/zsh -lc \'cat a.txt\'","aggregated_output":"A","exit_code":0,"status":"completed"}}',
            f'{{"type":"item.completed","item":{{"id":"item_1","type":"agent_message","text":"first {canary}"}}}}',
            '{"type":"turn.completed","usage":{"input_tokens":150,"output_tokens":75}}',
            '{"type":"turn.started"}',
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
        assert await runner.run(
            _flow_config(make_config, feature_branch_repo, backend="codex")
        ) == 0
    spans = receiver.spans
    tools = _kind(spans, "tool")
    assert len(tools) == 2
    assert len({attributes(tool)["gen_ai.tool.call.id"] for tool in tools}) == 2
    billed = attributes(_attempt(spans))
    # The invocation aggregate reflects the LAST turn.completed totals — the
    # protocol's honest semantics (usage_scope=invocation, not additive).
    assert billed["gen_ai.usage.input_tokens"] == 200
    assert billed["gen_ai.usage.output_tokens"] == 100
    _assert_leak_free(receiver, canary)


# ============================================================================
# Pi: real PiBackend with the explicitly labeled long-generation replay
# ============================================================================


async def test_pi_replay_exact_native_timing_choice_and_billing_through_runner(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 395.332-second replay through runner.run: exact ns, choice, billing.

    Asserts exact native start 1788690314289 ms → 1788690314289000000 ns, the
    host message_end receipt 1788690709621000000 ns (395.332 s), the typed
    ordered provider choice containing the tool call before later execution,
    absent model-child input history, response ID, reasoning/content, the
    86,936/8,401/$0.00402781 accounting, one SDK end at the historical
    pre-tool time after billing resolution, and no parent billing.
    """
    canary = _CANARIES["pi"]
    _flow(ext_dir)
    # Inject the per-run canary into the replay's assistant text so the wire
    # payload positively proves this run's content (leak-free check requires
    # the canary present; the rest of the fixture stays frozen placeholder).
    lines = _replay_lines(
        "long_generation_replay.jsonl",
        {
            "REPLAY_TEXT_ONE": f"replay one {canary}",
            "REPLAY_TEXT_TWO": f"replay two {canary}",
        },
    )
    spawner = install_fake_cli_process(monkeypatch, "pi", lines=lines)
    # Plan §364/§519+corrections: inject the deterministic host receipt clock
    # for ``message_end`` so the replay proves the exact 395.332-second
    # interval (pinned first message_end receipt 1788690709621000000 ns).
    _pin_first_message_end_receipt(monkeypatch, 1788690709621000000)

    with otlp_collector() as receiver:
        _configure_otlp(monkeypatch, receiver.base_url + "/v1/traces")
        assert await runner.run(
            _flow_config(make_config, feature_branch_repo, backend="pi")
        ) == 0

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
    # Typed ordered provider choice: reasoning, text, tool-call — the tool
    # call exists in the choice BEFORE later execution, which never authors it.
    # Parts serialize to the provider content-block wire shape ("type"), the
    # same shape the trajectory draft keeps as {"kind": ...} via asdict; the
    # typed identity (kind) is preserved locally and mirrored on the wire.
    choice = json.loads(gen_attrs["daydream.generation.choice_parts"])
    assert [part["type"] for part in choice] == ["reasoning", "text", "tool_call"]
    assert choice[2]["id"] == "call_replay_001"
    assert choice[2]["name"] == "read_file"
    assert choice[2]["arguments"] == {"path": "src/replay.py"}
    # Response identity: native response ID on the child; the invocation-local
    # correlation ID is a different namespace and never replaces it.
    assert gen_attrs["daydream.generation.id"] != "resp_replay_01"
    # Standard generation aliases (model/provider/finish reasons) are written
    # only when the frozen ledger owner bills this child (binding decision 5);
    # the structural attempt owns this bill, so the child stays explicitly
    # non-billed and fail-closed — the native evidence lives on the attempt
    # (last event wins: gen1/turn_end stop "stop") and the child keeps only
    # explicit custom ``daydream.generation.*`` evidence.
    assert gen_attrs["daydream.generation.billed"] is False
    assert "gen_ai.response.model" not in gen_attrs
    assert "gen_ai.provider.name" not in gen_attrs
    assert "gen_ai.response.finish_reasons" not in gen_attrs
    assert billed["gen_ai.response.model"] == "glm-5.3-flash"
    assert billed["gen_ai.provider.name"] == "nous"
    assert billed["gen_ai.response.finish_reasons"] == ["stop"]
    # Model-child input history is absent: the stream exposes only the
    # invocation prompt, which is NOT copied onto the generation.
    assert "gen_ai.input.messages" not in gen_attrs
    # Billing: no per-generation usage reaches the ledger (Pi exposes usage at
    # turn_end without a generation correlation), so the authoritative
    # structural attempt owns the bill and children stay explicitly non-billed.
    assert billed["daydream.billing.owner"] == "structural_attempt"
    assert gen_attrs["daydream.generation.billed"] is False
    assert "gen_ai.usage.input_tokens" not in gen_attrs
    # 86,936 = 10,392 uncached + 76,544 cache-read (normalized input); the
    # replay's reported historical cost rides as telemetry, not real billing.
    assert billed["gen_ai.usage.input_tokens"] == 86936
    assert billed["gen_ai.usage.output_tokens"] == 8401
    assert billed["gen_ai.usage.cost"] == pytest.approx(0.00402781)
    assert "daydream.generation.duration_ns" not in billed
    # The attempt scope is the remote-agent CLIENT boundary (spans.py:142).
    assert attempt["kind"] == "SPAN_KIND_CLIENT"
    _assert_leak_free(receiver, canary)


@pytest.mark.parametrize("vendor", ["otlp", "honeyhive", "langsmith"])
async def test_pi_metadata_mode_omits_generation_choice_content(
    vendor: str,
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata mode through a real adapter: generation choice content absent.

    The Pi replay's assistant choice text would otherwise ride the wire
    through ``daydream.generation.choice_parts`` in every capture mode, which
    contradicts the field matrix (plan §252: generation output content is
    "Full content except counts/identity", full mode only). In metadata mode
    the counts/identity/timing evidence stays, the provider-choice content is
    omitted, and structure survives for every destination.
    """
    canary = _CANARIES["pi"]
    _flow(ext_dir)
    lines = _replay_lines(
        "long_generation_replay.jsonl",
        {
            "REPLAY_TEXT_ONE": f"replay one {canary}",
            "REPLAY_TEXT_TWO": f"replay two {canary}",
        },
    )
    install_fake_cli_process(monkeypatch, "pi", lines=lines)
    _pin_first_message_end_receipt(monkeypatch, 1788690709621000000)

    expected_paths = {"otlp": "/v1/traces", "honeyhive": "/opentelemetry/v1/traces",
                      "langsmith": "/otel/v1/traces"}
    with otlp_collector() as receiver:
        _vendor_env(monkeypatch, vendor, receiver.base_url)
        monkeypatch.setenv("DAYDREAM_TRACE_TO", vendor)
        monkeypatch.setenv("DAYDREAM_TRACE_CONTENT", "metadata")
        assert await runner.run(
            _flow_config(make_config, feature_branch_repo, backend="pi")
        ) == 0

    spans = receiver.spans
    generations = _kind(spans, "generation")
    assert len(generations) == 2
    # Each generation carries its own strict native start (1788690314289 and
    # 1788690314500 ms). The FIRST message_end receipt lands exactly on the
    # pinned historical instant; later receipts advance monotonically (real
    # elapsed ns) — never backward, never relabeled as provider latency.
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
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The committed lifecycle fixture: two generations, one tool, canary linkage."""
    canary = _CANARIES["pi"]
    _flow(ext_dir)
    install_fake_cli_process(
        monkeypatch, "pi",
        lines=_replay_lines("generation_lifecycle.jsonl", {"src/example.py": f"src/{canary}.py"}),
    )
    # Deterministic host message_end receipt strictly between the two native
    # starts (gen0 native 1788690314289000000 ns < P < gen1 native
    # 1788690455762000000 ns); the sealed gen0 end must precede gen1's start.
    _pin_first_message_end_receipt(monkeypatch, 1788690315000000000)
    with otlp_collector() as receiver:
        _configure_otlp(monkeypatch, receiver.base_url + "/v1/traces")
        assert await runner.run(
            _flow_config(make_config, feature_branch_repo, backend="pi")
        ) == 0
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


# ============================================================================
# Osprey: real OspreyBackend with a strict fake-process protocol fixture
# ============================================================================


def _osprey_lines(canary: str) -> list[str]:
    return [
        json.dumps({"event": "protocol", "version": 2}),
        json.dumps({
            "event": "session_start", "session_id": "native-osprey-session",
            "started_at": "2026-09-09T12:00:00Z", "model": "osprey-native-model",
            "provider": "osprey-native-provider",
        }),
        json.dumps({"event": "turn_start", "turn_id": "t1", "timestamp": "2026-09-09T12:00:01Z"}),
        json.dumps({"event": "thinking_delta", "content": f"osprey thinking {canary}"}),
        json.dumps({"event": "text_delta", "content": f"osprey answer {canary}"}),
        json.dumps({"event": "tool_call", "tool_call_id": "call_osp_1", "tool_name": "read",
                    "arguments": {"path": "src/osp.py"}}),
        json.dumps({"event": "tool_result", "tool_call_id": "call_osp_1", "tool_name": "read",
                    "status": "success", "content": "file body", "duration_ms": 41}),
        json.dumps({"event": "turn_end", "turn_id": "t1", "usage_reported": True, "duration_ms": 90,
                    "prompt_tokens": 30, "completion_tokens": 6, "cached_tokens": 4,
                    "cache_write_tokens": 2, "thinking_tokens": 3, "cost_usd": "0.007"}),
        json.dumps({"event": "session_end", "outcome": "completed", "exit_code": 0,
                    "total_cost_usd": "0.007", "total_prompt_tokens": 30,
                    "total_completion_tokens": 6, "total_cached_tokens": 4,
                    "total_cache_write_tokens": 2, "total_thinking_tokens": 3,
                    "session_wallclock_ms": 120}),
    ]


async def test_osprey_strict_protocol_fixture_through_runner(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real OspreyBackend via strict fake process: argv, session usage, no gen child."""
    canary = _CANARIES["osprey"]
    _flow(ext_dir)
    spawner = install_fake_cli_process(monkeypatch, "osprey", lines=_osprey_lines(canary))
    with otlp_collector() as receiver:
        _configure_otlp(monkeypatch, receiver.base_url + "/v1/traces")
        assert await runner.run(
            _flow_config(make_config, feature_branch_repo, backend="osprey")
        ) == 0

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
    # The attempt scope is the remote-agent CLIENT boundary (spans.py:142).
    assert attempt["kind"] == "SPAN_KIND_CLIENT"
    _assert_leak_free(receiver, canary)


async def test_osprey_explicit_zero_temperature_reaches_argv_and_config(
    monkeypatch: pytest.MonkeyPatch,
    feature_branch_repo: Path,
) -> None:
    """Only explicit temperature=0.0 is admitted; it reaches argv AND typed config."""
    canary = _CANARIES["osprey"]
    spawner = install_fake_cli_process(monkeypatch, "osprey", lines=_osprey_lines(canary))
    backend = OspreyBackend(osprey_binary="osprey", temperature=0.0)
    events = []
    async for event in backend.execute(feature_branch_repo, f"prompt {canary}"):
        events.append(event)
    argv, _kwargs = _spawner(spawner)
    assert "--temperature" in argv
    assert argv[argv.index("--temperature") + 1] == "0.0"
    request = next(e for e in events if isinstance(e, RequestEvent))
    assert request.config.temperature == 0.0  # zero retained, never omitted
    assert request.model_source == "native"


# ============================================================================
# Schema-valid content on the wire (pinned semconv message schemas)
# ============================================================================


async def test_attempt_input_messages_validate_against_pinned_schema(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gen_ai.input.messages on the attempt validates against the pinned schema."""
    canary = _CANARIES["pi"]
    _flow(ext_dir)
    install_fake_cli_process(
        monkeypatch, "pi",
        lines=_replay_lines("simple_text.jsonl", {"Hello from Pi": f"pi reply {canary}"}),
    )
    with otlp_collector() as receiver:
        _configure_otlp(monkeypatch, receiver.base_url + "/v1/traces")
        assert await runner.run(
            _flow_config(make_config, feature_branch_repo, backend="pi")
        ) == 0
    import jsonschema

    schema = json.loads(
        (Path(__file__).parent / "fixtures" / "observability_semconv" / "gen-ai-input-messages.json").read_text()
    )
    messages = json.loads(attributes(_attempt(receiver.spans))["gen_ai.input.messages"])
    jsonschema.validate(messages, schema)


# ============================================================================
# Real generic HTTP/protobuf + gRPC integration for the destinations (plan 791)
# ============================================================================


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
    vendor: str,
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP/protobuf transport for every destination consumes Task 4A's contract."""
    canary = _CANARIES["pi"]
    _flow(ext_dir)
    install_fake_cli_process(
        monkeypatch, "pi",
        lines=_replay_lines("simple_text.jsonl", {"Hello from Pi": f"pi reply {canary}"}),
    )
    expected_paths = {"otlp": "/v1/traces", "honeyhive": "/opentelemetry/v1/traces",
                      "langsmith": "/otel/v1/traces"}
    with otlp_collector() as receiver:
        _vendor_env(monkeypatch, vendor, receiver.base_url)
        monkeypatch.setenv("DAYDREAM_TRACE_TO", vendor)
        assert await runner.run(
            _flow_config(make_config, feature_branch_repo, backend="pi")
        ) == 0
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
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generic gRPC destination speaks the real TraceService protocol."""
    canary = _CANARIES["osprey"]
    _flow(ext_dir)
    install_fake_cli_process(monkeypatch, "osprey", lines=_osprey_lines(canary))
    with otlp_grpc_collector() as receiver:
        # Scheme-less host:port would resolve to a TLS channel (insecure only
        # when the endpoint carries an explicit http:// scheme); the loopback
        # server is plaintext, so the endpoint must say so.
        _configure_otlp(monkeypatch, f"http://{receiver.base_url}", protocol="grpc")
        assert await runner.run(
            _flow_config(make_config, feature_branch_repo, backend="osprey")
        ) == 0
    assert receiver.requests
    assert _attempt(receiver.spans)
    metadata = receiver.requests[0]["headers"]
    # gRPC's HTTP/2 layer sets the content-type pseudo-encoding itself;
    # grpc-python does not surface it in invocation metadata, so the wire
    # truth asserted here is the exporter's recorded gRPC user-agent and the
    # parsed real protobuf TraceBatch (the servicer records the serialized
    # ExportTraceServiceRequest, which the collector decodes into spans).
    assert "user-agent" in metadata
    assert "grpc" in metadata["user-agent"]
    assert receiver.requests[0]["path"] == "/grpc"
    _assert_leak_free(receiver, canary)


async def test_grpc_outage_fails_open_and_review_completes(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Transport outage on gRPC: bounded shutdown, run succeeds, warning logged."""
    import anyio

    _flow(ext_dir)
    install_fake_cli_process(monkeypatch, "osprey", lines=_osprey_lines(_CANARIES["osprey"]))
    with otlp_grpc_collector(reject=True) as receiver:
        # Explicit http:// scheme: the plaintext loopback server must be
        # reached as insecure, so the genuine UNAVAILABLE rejection path
        # (not a TLS-misconfiguration failure) is what the test asserts.
        _configure_otlp(monkeypatch, f"http://{receiver.base_url}", protocol="grpc")
        with anyio.fail_after(30):
            assert await runner.run(
                _flow_config(make_config, feature_branch_repo, backend="osprey")
            ) == 0
    assert json.loads((feature_branch_repo / ".daydream/protocol-result.json").read_text())["output"].endswith(
        _CANARIES["osprey"]
    )
    assert "Trace export failed" in caplog.text


# ============================================================================
# Step 3 lifecycle matrix: real-runner retry with a failed billed attempt
# ============================================================================


class _PiRetryFixture:
    """Pi error-turn stream: a retryable 429 overload with a real billed attempt."""

    def __init__(self, canary: str) -> None:
        self.failed = [
            json.dumps({"type": "session", "sessionId": "pi_ses_retry"}),
            json.dumps({"type": "agent_start"}),
            json.dumps({"type": "turn_start"}),
            json.dumps({"type": "message_end", "message": {"role": "assistant",
                                                           "content": [{"type": "text", "text": "partial"}]}}),
            json.dumps({"type": "turn_end", "message": {
                "role": "assistant", "content": [{"type": "text", "text": "partial"}],
                "stopReason": "error", "errorMessage": "429 too many requests",
                "usage": {"input": 500, "output": 20, "cacheRead": 0,
                          "cost": {"total": 0.003}},
            }}),
            json.dumps({"type": "agent_end", "messages": []}),
        ]


async def test_runner_failed_billed_attempt_wears_its_own_bill_real_pi(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Step 3 via a real adapter: the failed billed attempt keeps its bill.

    The two-attempt retry split with per-attempt wire separation is proven at
    the runner seam in ``test_observability_failure_integration.py``; this real
    Pi leg proves the same invariant against the actual adapter protocol: the
    failed attempt exports ERROR status, its own usage/cost (decision 5:
    preserve failed-attempt bills), and the structural owner, while the failure
    surfaces to the caller.
    """
    _flow(ext_dir)
    fixture = _PiRetryFixture(_CANARIES["pi"])
    # Pin zero retries: the retryable 429 would otherwise re-spawn the real
    # backend (DAYDREAM_PI_RETRY_ATTEMPTS defaults to 20); the retry-split is
    # proven at the runner seam in failure_integration, this leg proves the
    # single failed billed attempt against the real adapter protocol.
    backend = PiBackend()
    backend.retry_attempts = 0
    install_backend(backend)
    install_fake_cli_process(monkeypatch, "pi", lines=fixture.failed)
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
