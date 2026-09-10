"""Failure, fanout and cancellation traces through the production runner and OTLP wire."""

from __future__ import annotations

import json
import os
import textwrap
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream import runner
from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    CostEvent,
    GenerationEndEvent,
    GenerationStartEvent,
    MetricsEvent,
    RequestEvent,
    ResultEvent,
    TextChoicePart,
    TextEvent,
    ToolCallChoicePart,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.codex import CodexBackend, CodexError
from daydream.observability.config import ObservabilityConfig
from daydream.runner import RunConfig
from tests.conftest import ExtDir
from tests.harness.backend import ScriptedBackend
from tests.harness.otlp import TraceCollector, attributes, otlp_collector

_RESULT_FILE = ".daydream/trace-failure-result.json"
_FLOW_IMPORTS = """
import anyio
import json
from daydream.agent import run_agent
from daydream.extensions import FlowStep, Stop, ToolDecision
from daydream.trajectory import DaydreamPhase
"""
_SINGLE_AGENT_BODY = """
output, _, aborted = await run_agent(
    ctx.backend_for("review"), ctx.work.repo, "inspect sample", phase=DaydreamPhase.REVIEW,
    {controls}
)
(ctx.data["daydream_dir"] / "trace-failure-result.json").write_text(
    json.dumps({{"output": output, "aborted": aborted}})
)
return Stop(1) if aborted else None
"""


@pytest.fixture(autouse=True)
def isolated_operator_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in os.environ:
        if name.startswith(("OTEL_", "DAYDREAM_TRACE_", "_OTEL_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT", "0.5")


def _flow(ext_dir: ExtDir, *, body: str | None = None, controls: str = "", veto: bool = False) -> None:
    source = _FLOW_IMPORTS + "\nasync def probe(ctx):\n"
    source += textwrap.indent(body or _SINGLE_AGENT_BODY.format(controls=controls), "    ")
    source += "\ndef register(r):\n"
    source += "    r.register_phase(FlowStep(name='trace-failures', run=probe))\n"
    source += "    r.set_flow('trace-failures', ['trace-failures'])\n"
    if veto:
        source += "    r.register_tool_supervisor(lambda name, args, *, phase: ToolDecision(True, 'protected path'))\n"
    ext_dir.write_module(source)


def _config(
    make_config: Callable[..., RunConfig], repo: Path, endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> RunConfig:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", endpoint + "/v1/traces")
    return make_config(repo, flow_name="trace-failures", observability=ObservabilityConfig(destinations=("otlp",)))


def _kind(spans: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [span for span in spans if attributes(span).get("daydream.span.kind") == kind]


class _RetryableFailure(RuntimeError):
    retryable = True


async def test_runner_retry_keeps_failed_billed_attempt_separate_from_success(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _flow(ext_dir)
    backend = ScriptedBackend(
        script=[
            [
                RequestEvent("failed attempt request"),
                TextEvent("failed partial response"),
                CostEvent(0.03, 30, 3),
                _RetryableFailure("retry this transport failure"),
            ],
            [
                RequestEvent("successful attempt request"),
                TextEvent("successful final response"),
                CostEvent(0.02, 20, 2),
                ResultEvent(None, None),
            ],
        ],
        retry_attempts=1,
        retry_base_delay_s=0,
        retry_max_delay_s=0,
    )
    install_backend(backend)
    with otlp_collector() as collector:
        assert await runner.run(_config(make_config, feature_branch_repo, collector.base_url, monkeypatch)) == 0
    output = json.loads((feature_branch_repo / _RESULT_FILE).read_text())
    assert output == {"output": "successful final response", "aborted": None}
    assert backend.call_count == 2
    attempts = sorted(_kind(collector.spans, "attempt"), key=lambda span: int(span["startTimeUnixNano"]))
    assert len(attempts) == 2
    assert [span["status"]["code"] for span in attempts] == ["STATUS_CODE_ERROR", "STATUS_CODE_OK"]
    assert [attributes(span)["gen_ai.usage.input_tokens"] for span in attempts] == [30, 20]
    assert [attributes(span)["gen_ai.usage.cost"] for span in attempts] == [0.03, 0.02]
    assert json.loads(attributes(attempts[1])["traceloop.entity.input"])["prompt"] == "successful attempt request"
    assert "inspect sample" not in attributes(attempts[1])["gen_ai.input.messages"]
    assert "failed partial response" in attributes(attempts[0])["gen_ai.output.messages"]
    assert "failed partial response" not in attributes(attempts[1])["gen_ai.output.messages"]
    agent = _kind(collector.spans, "agent")[0]
    assert {span["parentSpanId"] for span in attempts} == {agent["spanId"]}
    assert json.loads(attributes(agent)["traceloop.entity.output"]) == "successful final response"
    assert "gen_ai.usage.cost" not in attributes(agent)
    root = _kind(collector.spans, "run")[0]
    assert root["status"]["code"] == "STATUS_CODE_OK"


class _FanoutBackend:
    model = "shared-model"

    def __init__(self) -> None:
        self.entered = 0
        self.both_active = anyio.Event()

    async def execute(self, _cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent]:
        yield RequestEvent(prompt)
        yield ToolStartEvent("shared-tool-id", "Read", {"path": f"{prompt}.py"})
        self.entered += 1
        if self.entered == 2:
            self.both_active.set()
        await self.both_active.wait()
        yield ToolResultEvent("shared-tool-id", f"content for {prompt}", False)
        yield TextEvent(f"answer for {prompt}")
        yield CostEvent(0.01, 10, 1)
        yield ResultEvent(None, None)

    async def cancel(self) -> None:
        pass


async def test_runner_fanout_shared_backend_and_tool_ids_keep_sibling_content_isolated(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _flow(
        ext_dir,
        body="""
backend = ctx.backend_for("review")
outputs = {}
async def child(label):
    result, _, _ = await run_agent(backend, ctx.work.repo, label, phase=DaydreamPhase.REVIEW)
    outputs[label] = result
async with anyio.create_task_group() as group:
    group.start_soon(child, "left")
    group.start_soon(child, "right")
(ctx.data["daydream_dir"] / "trace-failure-result.json").write_text(json.dumps(outputs))
""",
    )
    backend = _FanoutBackend()
    install_backend(backend)
    with otlp_collector() as collector:
        with anyio.fail_after(15):
            assert await runner.run(_config(make_config, feature_branch_repo, collector.base_url, monkeypatch)) == 0
    assert json.loads((feature_branch_repo / _RESULT_FILE).read_text()) == {
        "left": "answer for left",
        "right": "answer for right",
    }
    assert backend.entered == 2
    spans = collector.spans
    by_id = {span["spanId"]: span for span in spans}
    agents = _kind(spans, "agent")
    assert len(agents) == 2
    assert len({span["parentSpanId"] for span in agents}) == 1
    tools = _kind(spans, "tool")
    assert len(tools) == 2
    assert len({span["parentSpanId"] for span in tools}) == 2
    for tool in tools:
        tool_input = json.loads(attributes(tool)["traceloop.entity.input"])
        label = tool_input["path"].removesuffix(".py")
        other = "right" if label == "left" else "left"
        assert json.loads(attributes(tool)["traceloop.entity.output"]) == f"content for {label}"
        attempt = by_id[tool["parentSpanId"]]
        assert f"answer for {label}" in attributes(attempt)["gen_ai.output.messages"]
        assert f"answer for {other}" not in attributes(attempt)["gen_ai.output.messages"]
        assert by_id[attempt["parentSpanId"]] in agents
    assert len({span["traceId"] for span in spans}) == 1


class _ActiveToolBackend:
    model = "active-tool-model"

    def __init__(self) -> None:
        self.active = anyio.Event()
        self.cancelled = False
        self.finished = False

    async def execute(self, _cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent]:
        try:
            yield RequestEvent(prompt)
            yield TextEvent("partial response before interruption")
            yield CostEvent(0.005, 5, 1)
            yield ToolStartEvent("open-tool", "Write", {"path": "protected.py"})
            self.active.set()
            await anyio.sleep_forever()
        finally:
            self.finished = True

    async def cancel(self) -> None:
        self.cancelled = True


async def test_runner_cancellation_reaches_caller_and_flushes_root_and_active_tool(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _flow(ext_dir)
    backend = _ActiveToolBackend()
    install_backend(backend)
    cancellation_reached_caller = False
    with otlp_collector() as collector:
        config = _config(make_config, feature_branch_repo, collector.base_url, monkeypatch)

        async def launch() -> None:
            nonlocal cancellation_reached_caller
            try:
                await runner.run(config)
                pytest.fail("cancelled run must not return a successful result")
            except anyio.get_cancelled_exc_class():
                cancellation_reached_caller = True
                raise

        with anyio.fail_after(15):
            async with anyio.create_task_group() as group:
                group.start_soon(launch)
                await backend.active.wait()
                group.cancel_scope.cancel()
    assert cancellation_reached_caller and backend.cancelled and backend.finished
    assert not (feature_branch_repo / _RESULT_FILE).exists()
    spans = collector.spans
    assert len(spans) == 5
    assert all(span["status"]["code"] == "STATUS_CODE_ERROR" for span in spans)
    root = _kind(spans, "run")[0]
    tool = _kind(spans, "tool")[0]
    assert attributes(root)["daydream.outcome"] == "cancelled"
    assert attributes(tool)["daydream.outcome"] == "cancelled"
    assert all(int(span["endTimeUnixNano"]) >= int(span["startTimeUnixNano"]) for span in spans)


async def test_runner_export_outage_warns_and_preserves_review_result_and_output(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _flow(ext_dir)
    install_backend(ScriptedBackend(events=[TextEvent("review completed during outage"), ResultEvent(None, None)]))
    with otlp_collector(status=503) as collector:
        with anyio.fail_after(15):
            rc = await runner.run(_config(make_config, feature_branch_repo, collector.base_url, monkeypatch))
    assert rc == 0
    assert json.loads((feature_branch_repo / _RESULT_FILE).read_text()) == {
        "output": "review completed during outage",
        "aborted": None,
    }
    assert collector.requests
    assert attributes(_kind(collector.spans, "run")[0])["daydream.exit_code"] == 0
    assert "Trace export failed; review execution continues" in caplog.text


@pytest.mark.parametrize("interruption", ["supervisor", "tools", "wall"])
async def test_runner_partial_outcome_preserves_veto_and_budget_reasons(
    interruption: str,
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = {"supervisor": "", "tools": "tool_call_budget=0,", "wall": "wall_budget_s=0.05,"}[interruption]
    reason = {"supervisor": "tool_vetoed:Write", "tools": "tool_call_budget_exceeded", "wall": "wall_budget_exceeded"}[
        interruption
    ]
    _flow(ext_dir, controls=controls, veto=interruption == "supervisor")
    backend = _ActiveToolBackend()
    install_backend(backend)
    with otlp_collector() as collector:
        with anyio.fail_after(15):
            assert await runner.run(_config(make_config, feature_branch_repo, collector.base_url, monkeypatch)) == 1
    assert backend.finished
    assert json.loads((feature_branch_repo / _RESULT_FILE).read_text()) == {
        "output": "partial response before interruption",
        "aborted": reason,
    }
    spans = collector.spans
    assert len(spans) == 5
    assert all(span["status"]["code"] == "STATUS_CODE_ERROR" for span in spans)
    assert attributes(_kind(spans, "tool")[0])["daydream.outcome"] == reason.split(":")[0]
    assert attributes(_kind(spans, "attempt")[0])["gen_ai.usage.input_tokens"] == 5
    assert attributes(_kind(spans, "attempt")[0])["gen_ai.usage.cost"] == 0.005
    assert attributes(_kind(spans, "run")[0])["daydream.exit_code"] == 1


@pytest.mark.parametrize("terminal_result", [False, True])
@pytest.mark.parametrize("last_duration_source", ["tool", "message"])
async def test_runner_attempt_duration_requires_terminal_result(
    terminal_result: bool,
    last_duration_source: str,
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _flow(ext_dir)
    emitted = anyio.Event()
    timings: list[AgentEvent] = [
        MetricsEvent("message-one", 5, 1, None, None, duration_ms=123),
        ToolResultEvent("timed-tool", "source content", False, duration_ms=42),
    ]
    if last_duration_source == "message":
        timings.reverse()

    class TimedBackend:
        model = "timed-model"

        async def execute(self, _cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent]:
            yield RequestEvent(prompt)
            yield ToolStartEvent("timed-tool", "Read", {"path": "source.py"})
            for event in timings:
                yield event
            yield TextEvent("observed answer")
            if terminal_result:
                yield ResultEvent(None, None, duration_ms=500, duration_api_ms=400)
            else:
                emitted.set()
                await anyio.sleep_forever()

        async def cancel(self) -> None:
            pass

    install_backend(TimedBackend())
    with otlp_collector() as collector:
        config = _config(make_config, feature_branch_repo, collector.base_url, monkeypatch)
        with anyio.fail_after(15):
            if terminal_result:
                assert await runner.run(config) == 0
            else:
                async with anyio.create_task_group() as group:
                    group.start_soon(runner.run, config)
                    await emitted.wait()
                    group.cancel_scope.cancel()

    attempt = _kind(collector.spans, "attempt")[0]
    metadata = attributes(attempt)
    if terminal_result:
        assert metadata["daydream.duration_ms"] == 500
        assert metadata["daydream.duration_api_ms"] == 400
        assert attempt["status"]["code"] == "STATUS_CODE_OK"
        assert json.loads((feature_branch_repo / _RESULT_FILE).read_text())["output"] == "observed answer"
    else:
        assert "daydream.duration_ms" not in metadata
        assert "daydream.duration_api_ms" not in metadata
        assert attempt["status"]["code"] == "STATUS_CODE_ERROR"
        assert metadata["daydream.outcome"] == "cancelled"
        assert not (feature_branch_repo / _RESULT_FILE).exists()
    assert attributes(_kind(collector.spans, "tool")[0])["daydream.tool.duration_ms"] == 42
    assert json.loads(metadata["daydream.message_usage"])[0]["duration_ms"] == 123


async def test_codex_prelaunch_failure_does_not_export_an_effective_request(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _flow(ext_dir, body='''
from daydream.backends import ContinuationToken
await run_agent(
    ctx.backend_for("review"), ctx.work.repo, "original logical prompt",
    phase=DaydreamPhase.REVIEW, read_only=True,
    continuation=ContinuationToken("codex", {"thread_id": "previous-thread"}),
)
''')
    install_backend(CodexBackend(model="fixture-model"))
    with otlp_collector() as collector:
        with pytest.raises(CodexError, match="cannot be resumed"):
            await runner.run(_config(make_config, feature_branch_repo, collector.base_url, monkeypatch))
    attempt = _kind(collector.spans, "attempt")[0]
    metadata = attributes(attempt)
    assert attempt["status"]["code"] == "STATUS_CODE_ERROR"
    assert metadata["error.type"] == "CodexError"
    assert "daydream.request.timestamp" not in metadata
    assert "gen_ai.input.messages" not in metadata
    assert "traceloop.entity.input" not in metadata
    agent = _kind(collector.spans, "agent")[0]
    assert attempt["parentSpanId"] == agent["spanId"]
    assert json.loads(attributes(agent)["traceloop.entity.input"])["prompt"] == "original logical prompt"


# ============================================================================
# Step 3 lifecycle matrix (plan §808): late/missing/duplicate/contradictory
# authoritative metrics, zero-generation totals, exact allocation, residual
# totals, pending caps, tool error after a sealed generation, and resume with
# a native conversation distinct from the Daydream session — all through the
# production runner and the real OTLP wire. Billing-owner semantics mirror the
# T2 ledger unit tests (tests/test_trajectory_generation_lifecycle.py) but are
# proven here at the runner/wire boundary, per plan §781/§808.
# ============================================================================


def _gen_start(generation_id: str) -> GenerationStartEvent:
    return GenerationStartEvent(generation_id=generation_id, observed_at_unix_ns=1_778_000_000_000_000_000)


def _gen_end(
    generation_id: str,
    text: str,
    *,
    native_started_at_unix_ms: int | None = 1_778_000_000_000,
    ended_at_unix_ns: int = 1_778_000_000_500_000_000,
    response_id: str | None = None,
) -> GenerationEndEvent:
    return GenerationEndEvent(
        generation_id=generation_id,
        native_started_at_unix_ms=native_started_at_unix_ms,
        ended_at_unix_ns=ended_at_unix_ns,
        end_source="host_observed_message_end",
        choice_parts=(TextChoicePart(text=text), ToolCallChoicePart(call_id=f"{generation_id}-call",
                                                                   name="Read", arguments={"path": "src/x.py"})),
        response_id=response_id,
        model_name="gen-model",
        provider_name="gen-provider",
        finish_reason="stop",
    )


def _read_subtrajectory(repo: Path) -> dict[str, Any]:
    paths = list((repo / ".daydream/runs").glob("*/trajectory.json"))
    assert len(paths) == 1, f"expected exactly one trajectory, found {paths}"
    trajectory = json.loads(paths[0].read_text())
    subs = trajectory["extra"]["subtrajectories"]
    assert len(subs) == 1
    lifecycle = subs[0]["generation_lifecycle"]
    assert isinstance(lifecycle, dict)
    return lifecycle


async def _run_lifecycle_flow(
    ext_dir: ExtDir,
    repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
    backend: object,
    events: list[AgentEvent],
    body: str | None = None,
) -> TraceCollector:
    """One real runner.run() over a scripted backend, returning the collector."""
    _flow(ext_dir, body=body) if body is not None else _flow(ext_dir)
    install_backend(backend)
    with otlp_collector() as collector:
        _config(make_config, repo, collector.base_url, monkeypatch)
        assert await runner.run(make_config(repo, flow_name="trace-failures",
                                            observability=ObservabilityConfig(destinations=("otlp",)))) == 0
    return collector


class _ExactAllocationBackend:
    """Two complete generations, late per-generation usage, terminal total."""

    model = "lifecycle-model"

    def __init__(self) -> None:
        self.ended = False

    async def execute(self, _cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent]:
        yield RequestEvent(prompt)
        yield _gen_start("gen-a")
        yield _gen_end("gen-a", "first sealed choice", response_id="resp-alpha")
        yield MetricsEvent("gen-a", 100, 20, None, 0.002, reasoning_tokens=5, generation_id="gen-a")
        yield ToolStartEvent("gen-a-call", "Read", {"path": "src/x.py"})
        yield ToolResultEvent("gen-a-call", "tool output", False)
        yield _gen_start("gen-b")
        yield _gen_end("gen-b", "second sealed choice", response_id="resp-beta")
        yield MetricsEvent("gen-b", 50, 10, None, 0.001, reasoning_tokens=2, generation_id="gen-b")
        yield CostEvent(0.003, 150, 30, measurement_source="terminal")
        yield TextEvent("final answer")
        yield ResultEvent(None, None)
        self.ended = True

    async def cancel(self) -> None:
        pass


async def test_runner_exact_allocation_bills_children_and_chain_matches_wire(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete children + late usage matching the terminal total: chain bills.

    Ledger decision 5 at the wire boundary: owner generation_children, both
    children carry standard aliases + usage + OK, and child token sums equal
    the attempt aggregate exactly.
    """
    backend = _ExactAllocationBackend()
    collector = await _run_lifecycle_flow(
        ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch, backend, []
    )
    lifecycle = _read_subtrajectory(feature_branch_repo)
    assert lifecycle["billing_owner"] == "generation_children"
    assert {draft["generation_id"]: draft["billed"] for draft in lifecycle["drafts"]} == {
        "gen-a": True,
        "gen-b": True,
    }
    spans = collector.spans
    gens = {attributes(span)["daydream.generation.id"]: span for span in _kind(spans, "generation")}
    assert set(gens) == {"gen-a", "gen-b"}
    attempt = _kind(spans, "attempt")[0]
    for generation_id, (input_tokens, output_tokens, cost) in {
        "gen-a": (100, 20, 0.002),
        "gen-b": (50, 10, 0.001),
    }.items():
        meta = attributes(gens[generation_id])
        assert gens[generation_id]["status"]["code"] == "STATUS_CODE_OK"
        assert meta["daydream.generation.billed"] is True
        assert meta["gen_ai.response.id"] == {"gen-a": "resp-alpha", "gen-b": "resp-beta"}[generation_id]
        assert meta["gen_ai.usage.input_tokens"] == input_tokens
        assert meta["gen_ai.usage.output_tokens"] == output_tokens
        assert meta["gen_ai.usage.cost"] == pytest.approx(cost)
        assert meta["gen_ai.usage.reasoning.output_tokens"] == {"gen-a": 5, "gen-b": 2}[generation_id]
        # One SDK end at the sealed historical end, never the host receipt.
        assert int(gens[generation_id]["endTimeUnixNano"]) == 1_778_000_000_500_000_000
    billed = attributes(attempt)
    assert billed["daydream.billing.owner"] == "generation_children"
    assert billed["gen_ai.usage.input_tokens"] == 150
    assert billed["gen_ai.usage.output_tokens"] == 30
    assert billed["gen_ai.usage.cost"] == pytest.approx(0.003)
    assert gens["gen-a"]["parentSpanId"] == attempt["spanId"]


class _LateMissingMetricsBackend:
    """Terminal total without any per-generation usage: partial evidence."""

    model = "lifecycle-model"

    async def execute(self, _cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent]:
        yield RequestEvent(prompt)
        yield _gen_start("gen-late")
        yield _gen_end("gen-late", "choice arrived, usage never did")
        yield CostEvent(0.02, 200, 40, measurement_source="terminal")
        yield TextEvent("answer")
        yield ResultEvent(None, None)

    async def cancel(self) -> None:
        pass


async def test_runner_late_missing_metrics_bill_chain_children_stay_custom(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Late/missing per-generation usage with a terminal total: chain bills only.

    Children keep explicit non-billed custom generation evidence (no invented
    usage, no standard usage aliases) while the attempt owns the bill.
    """
    collector = await _run_lifecycle_flow(
        ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch,
        _LateMissingMetricsBackend(), [],
    )
    lifecycle = _read_subtrajectory(feature_branch_repo)
    assert lifecycle["billing_owner"] == "structural_attempt"
    assert lifecycle["drafts"][0]["billed"] is False
    spans = collector.spans
    generation = _kind(spans, "generation")[0]
    meta = attributes(generation)
    assert meta["daydream.generation.billed"] is False
    assert "gen_ai.response.id" not in meta  # standard alias withheld unbilled
    assert "gen_ai.usage.input_tokens" not in meta  # usage never invented
    assert "gen_ai.usage.output_tokens" not in meta
    assert "gen_ai.usage.cost" not in meta
    assert meta["daydream.generation.choice_parts"]  # custom evidence retained
    billed = attributes(_kind(spans, "attempt")[0])
    assert billed["daydream.billing.owner"] == "structural_attempt"
    assert billed["gen_ai.usage.input_tokens"] == 200
    assert billed["gen_ai.usage.cost"] == pytest.approx(0.02)


class _DuplicateMetricsBackend:
    """Per-generation usage whose sum exceeds the terminal total."""

    model = "lifecycle-model"

    async def execute(self, _cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent]:
        yield RequestEvent(prompt)
        yield _gen_start("gen-x")
        yield _gen_end("gen-x", "choice")
        yield MetricsEvent("gen-x", 80, 30, None, 0.004, generation_id="gen-x")
        yield CostEvent(0.002, 70, 25, measurement_source="terminal")  # contradiction
        yield TextEvent("answer")
        yield ResultEvent(None, None)

    async def cancel(self) -> None:
        pass


async def test_runner_contradictory_metrics_fail_closed_without_rewriting(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Child sum contradicting the terminal total: owner none, nothing billed.

    Neither side is rewritten; the attempt carries no standard usage and the
    fixed contradiction diagnostic lands in the trajectory lifecycle only.
    """
    collector = await _run_lifecycle_flow(
        ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch,
        _DuplicateMetricsBackend(), [],
    )
    lifecycle = _read_subtrajectory(feature_branch_repo)
    assert lifecycle["billing_owner"] == "none"
    assert lifecycle["drafts"][0]["billed"] is False
    assert lifecycle["authoritative_total"]["input_tokens"] == 70  # first total kept
    assert lifecycle["diagnostics"] == ["billing_contradiction:child_sum_mismatches_terminal_total"]
    attempt = _kind(collector.spans, "attempt")[0]
    meta = attributes(attempt)
    assert meta["daydream.billing.owner"] == "none"
    # Measured evidence still reports the authoritative total; owner "none"
    # means the vendor chain bills nothing (children stay custom).
    assert meta["gen_ai.usage.input_tokens"] == 70
    assert meta["gen_ai.usage.cost"] == pytest.approx(0.002)
    payload = json.dumps([request["body"] for request in collector.requests])
    assert "billing_contradiction" not in payload  # diagnostic stays local


class _DuplicateIdempotentBackend:
    """Identical terminal totals twice: idempotent, children bill."""

    model = "lifecycle-model"

    async def execute(self, _cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent]:
        yield RequestEvent(prompt)
        yield _gen_start("gen-dup")
        yield _gen_end("gen-dup", "choice")
        yield MetricsEvent("gen-dup", 60, 15, None, 0.003, generation_id="gen-dup")
        yield CostEvent(0.003, 60, 15, measurement_source="terminal")
        yield CostEvent(0.003, 60, 15, measurement_source="terminal")  # exact duplicate
        yield TextEvent("answer")
        yield ResultEvent(None, None)

    async def cancel(self) -> None:
        pass


async def test_runner_duplicate_identical_totals_are_idempotent(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = await _run_lifecycle_flow(
        ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch,
        _DuplicateIdempotentBackend(), [],
    )
    lifecycle = _read_subtrajectory(feature_branch_repo)
    assert lifecycle["billing_owner"] == "generation_children"
    assert lifecycle["diagnostics"] == []
    attempt = _kind(collector.spans, "attempt")[0]
    assert attributes(attempt)["gen_ai.usage.input_tokens"] == 60
    generation = _kind(collector.spans, "generation")[0]
    assert attributes(generation)["daydream.generation.billed"] is True


class _ResidualTotalBackend:
    """Terminal total above the per-message sum: residual folds to the chain."""

    model = "lifecycle-model"

    async def execute(self, _cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent]:
        yield RequestEvent(prompt)
        yield TextEvent("streamed answer")
        yield MetricsEvent("", 40, 8, None, 0.001)  # per-message, no generation
        yield CostEvent(0.004, 90, 18, measurement_source="terminal")  # residual 50/10
        yield TextEvent(" more")
        yield ResultEvent(None, None)

    async def cancel(self) -> None:
        pass


async def test_runner_residual_unallocated_total_folds_onto_chain(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal total exceeding the per-message sum: attempt bills the whole.

    The per-dimension take-max residual (issue #747) keeps the wire aggregate
    authoritative; no generation evidence exists so the structural chain owns
    the complete bill.
    """
    collector = await _run_lifecycle_flow(
        ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch,
        _ResidualTotalBackend(), [],
    )
    attempt = _kind(collector.spans, "attempt")[0]
    meta = attributes(attempt)
    assert meta["daydream.billing.owner"] == "structural_attempt"
    assert meta["gen_ai.usage.input_tokens"] == 90
    assert meta["gen_ai.usage.output_tokens"] == 18
    assert meta["gen_ai.usage.cost"] == pytest.approx(0.004)
    message_usage = json.loads(meta["daydream.message_usage"])
    assert message_usage[0]["usage"]["input_tokens"] == 40


class _ToolErrorAfterSealBackend:
    """Tool error after a sealed completed generation."""

    model = "lifecycle-model"

    async def execute(self, _cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent]:
        yield RequestEvent(prompt)
        yield _gen_start("gen-tool-err")
        yield _gen_end("gen-tool-err", "sealed before the tool failed", response_id="resp-tool-err")
        yield MetricsEvent("gen-tool-err", 30, 6, None, 0.0005, generation_id="gen-tool-err")
        yield ToolStartEvent("failing-tool", "Bash", {"command": "exit 3"})
        yield ToolResultEvent("failing-tool", "boom", True)
        yield CostEvent(0.0005, 30, 6, measurement_source="terminal")
        yield TextEvent("recovered answer")
        yield ResultEvent(None, None)

    async def cancel(self) -> None:
        pass


async def test_runner_tool_error_after_sealed_generation_keeps_allocation(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later tool error never unseals or unbills the completed generation.

    The tool span closes ERROR with the sanitized message, the sealed
    generation keeps its billing allocation with historical end, and the run
    still completes successfully.
    """
    collector = await _run_lifecycle_flow(
        ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch,
        _ToolErrorAfterSealBackend(), [],
    )
    spans = collector.spans
    tool = _kind(spans, "tool")[0]
    assert tool["status"]["code"] == "STATUS_CODE_ERROR"
    assert attributes(tool)["daydream.outcome"] == "error"
    assert attributes(tool)["daydream.tool.error"] is True
    generation = _kind(spans, "generation")[0]
    meta = attributes(generation)
    assert meta["daydream.generation.billed"] is True
    assert meta["gen_ai.response.id"] == "resp-tool-err"
    assert generation["status"]["code"] == "STATUS_CODE_OK"
    attempt = _kind(spans, "attempt")[0]
    assert attributes(attempt)["daydream.billing.owner"] == "generation_children"
    assert attributes(_kind(spans, "run")[0])["daydream.exit_code"] == 0


class _PendingCapBackend:
    """513 sealed generations trip the 512-draft cap on the final seal."""

    model = "lifecycle-model"

    async def execute(self, _cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent]:
        yield RequestEvent(prompt)
        for index in range(513):
            generation_id = f"gen-{index:04d}"
            yield _gen_start(generation_id)
            yield _gen_end(generation_id, f"choice {index}", native_started_at_unix_ms=None)
        yield CostEvent(0.01, 100, 10, measurement_source="terminal")
        yield TextEvent("done")
        yield ResultEvent(None, None)

    async def cancel(self) -> None:
        pass


async def test_runner_pending_count_cap_drains_children_stay_unbilled(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """512-draft cap at the wire: owner structural, zero billed children.

    Mirrors the T2 cap unit test through the real runner: all drafts drain
    ended, the fixed count-only cap diagnostic is stored locally, and no
    child carries standard usage aliases.
    """
    collector = await _run_lifecycle_flow(
        ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch,
        _PendingCapBackend(), [],
    )
    lifecycle = _read_subtrajectory(feature_branch_repo)
    assert lifecycle["billing_owner"] == "structural_attempt"
    assert lifecycle["children_after_cap"] is True
    assert len(lifecycle["drafts"]) == 513
    assert all(draft["ended"] for draft in lifecycle["drafts"])
    assert all(draft["billed"] is False for draft in lifecycle["drafts"])
    cap = [d for d in lifecycle["diagnostics"] if "cap" in d]
    assert len(cap) == 1 and "512" in cap[0]
    generations = _kind(collector.spans, "generation")
    assert len(generations) == 513
    assert all("gen_ai.usage.input_tokens" not in attributes(span) for span in generations)
    assert attributes(_kind(collector.spans, "attempt")[0])["daydream.billing.owner"] == "structural_attempt"
    payload = json.dumps([request["body"] for request in collector.requests])
    assert "generation_pending_cap" not in payload  # diagnostics never on the wire


async def test_runner_resume_native_conversation_distinct_from_daydream_session(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume with a native conversation distinct from the Daydream session.

    The second invocation resumes the first's native continuation (the token
    the backend minted reaches the resumed execute call) while every span's
    session identity — trajectory session id and traceloop association — stays
    the one Daydream session. The native conversation id surfaces only on the
    resumed attempt via gen_ai.conversation.id.
    """
    backend = ScriptedBackend(
        script=[
            [
                RequestEvent("original invocation"),
                TextEvent("first answer"),
                ResultEvent(None, ContinuationToken("scripted", {"thread_id": "native-thread-77"})),
            ],
            [
                RequestEvent("resumed invocation"),
                TextEvent("resumed answer"),
                ResultEvent(None, None, session_id="native-thread-77"),
            ],
        ],
    )
    collector = await _run_lifecycle_flow(
        ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch, backend, [],
        body="""
first_output, continuation, _ = await run_agent(
    ctx.backend_for("review"), ctx.work.repo, "original logical prompt",
    phase=DaydreamPhase.REVIEW,
)
resumed_output, _, _ = await run_agent(
    ctx.backend_for("review"), ctx.work.repo, "follow-up prompt",
    phase=DaydreamPhase.REVIEW, continuation=continuation,
)
(ctx.data["daydream_dir"] / "trace-failure-result.json").write_text(
    json.dumps({"first": first_output, "resumed": resumed_output})
)
return None
""",
    )
    assert backend.continuations[0] is None
    assert backend.continuations[1].data == {"thread_id": "native-thread-77"}
    assert json.loads((feature_branch_repo / _RESULT_FILE).read_text()) == {
        "first": "first answer",
        "resumed": "resumed answer",
    }
    trajectory_paths = list((feature_branch_repo / ".daydream/runs").glob("*/trajectory.json"))
    assert len(trajectory_paths) == 1
    trajectory = json.loads(trajectory_paths[0].read_text())
    spans = collector.spans
    attempts = sorted(_kind(spans, "attempt"), key=lambda span: int(span["startTimeUnixNano"]))
    assert len(attempts) == 2
    assert attributes(attempts[0]).get("gen_ai.conversation.id") is None
    assert attributes(attempts[1])["gen_ai.conversation.id"] == "native-thread-77"
    for span in attempts:
        identity = attributes(span)
        assert identity["daydream.session.id"] == trajectory["session_id"]
        assert identity["traceloop.association.properties.session_id"] == trajectory["session_id"]
    payload = json.dumps([request["body"] for request in collector.requests])
    assert payload.count("native-thread-77") == 1  # only the resumed attempt's conversation


async def test_runner_codex_unresumable_continuation_fails_before_effective_request(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Codex continuation its backend cannot resume fails before any request.

    Complements the protocols-file pre-launch case with the failure seam on
    the production runner: the attempt exports ERROR with no effective
    request evidence, and the real CodexError surfaces to the caller.
    """
    _flow(ext_dir, body='''
from daydream.backends import ContinuationToken
await run_agent(
    ctx.backend_for("review"), ctx.work.repo, "original logical prompt",
    phase=DaydreamPhase.REVIEW, read_only=True,
    continuation=ContinuationToken("codex", {"thread_id": "unresumable-thread"}),
)
''')
    install_backend(CodexBackend(model="fixture-model"))
    with otlp_collector() as collector:
        _config(make_config, feature_branch_repo, collector.base_url, monkeypatch)
        with pytest.raises(CodexError, match="cannot be resumed"):
            await runner.run(
                make_config(feature_branch_repo, flow_name="trace-failures",
                            observability=ObservabilityConfig(destinations=("otlp",)))
            )
    attempt = _kind(collector.spans, "attempt")[0]
    meta = attributes(attempt)
    assert attempt["status"]["code"] == "STATUS_CODE_ERROR"
    assert meta["error.type"] == "CodexError"
    assert "daydream.request.timestamp" not in meta
    assert "gen_ai.input.messages" not in meta
