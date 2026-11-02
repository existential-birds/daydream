"""Failure, fanout and cancellation traces through the production runner and OTLP wire."""

from __future__ import annotations

import json
import textwrap
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

import anyio
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from daydream import runner
from daydream.backends import (
    AgentEvent,
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
from daydream.backends.pi import PiBackend
from daydream.dataset import LocalRecordStore
from daydream.extensions import Registry, build_registry
from daydream.observability.config import ObservabilityConfig
from daydream.run_config import RunConfig
from tests.conftest import ExtDir
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_cli_process import FakeCliScript, install_fake_cli_process
from tests.harness.otlp import TraceCollector, attributes, kind_of as _kind, otlp_collector

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


class _NoCancelBackend:
    """Backend stub for tests that never exercise cancellation."""

    model = "lifecycle-model"


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

async def test_pi_runner_invocation_timeout_preserves_healthy_sibling(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local timeout/aclose reaps one invocation while its peer completes and bills once."""
    _flow(ext_dir, body='''
results = {}
async def invoke(label, allowance):
    output, _, aborted = await run_agent(ctx.backend_for("review"), ctx.work.repo, label,
        phase=DaydreamPhase.REVIEW, wall_budget_s=allowance)
    results[label] = {"output": output, "aborted": aborted}
async with anyio.create_task_group() as group:
    group.start_soon(invoke, "timed-out", 1)
    group.start_soon(invoke, "healthy", 60)
(ctx.data["daydream_dir"] / "trace-failure-result.json").write_text(json.dumps(results))
''')
    install_backend(PiBackend(model="test-model"))
    start = json.dumps({"type": "message_start", "message": {"role": "assistant"}})
    spawner = install_fake_cli_process(monkeypatch, "pi", lines=[], scripts=[
        FakeCliScript(lines=[start], hang=True), FakeCliScript(lines=[start], hang=True),
    ])
    monkeypatch.setenv("DAYDREAM_TRACE_CONTENT", "metadata")
    completed: list[int] = []
    with otlp_collector() as collector:
        config = _config(make_config, feature_branch_repo, collector.base_url, monkeypatch)
        async def launch() -> None:
            completed.append(await runner.run(config))
        # Generous outer and healthy allowances: only the 0.05s sibling may expire, even under CPU load.
        with anyio.fail_after(60):
            async with anyio.create_task_group() as group:
                group.start_soon(launch)
                # Both children must be live before the 1s allowance expires; spawn order is not the contract.
                await spawner.wait_for_read(0)
                await spawner.wait_for_read(1)
                while all(proc.returncode is None for proc in spawner.procs):
                    await anyio.sleep(0.01)
                reaped, = [proc for proc in spawner.procs if proc.returncode is not None]
                healthy, = [proc for proc in spawner.procs if proc is not reaped]
                assert await reaped.wait() == -15
                assert healthy.returncode is None
                healthy.finish([json.dumps(frame) for frame in [
                    {"type": "message_update", "assistantMessageEvent": {
                        "type": "text_delta", "contentIndex": 0, "delta": "private-delta-content",
                    }},
                    {"type": "message_end", "message": {"role": "assistant", "model": "test-model",
                        "content": [{"type": "text", "text": "healthy result"}], "stopReason": "stop",
                        "usage": {"input": 10, "output": 5, "cost": {"total": 0.01}},
                    }},
                    {"type": "turn_end", "message": {"role": "assistant", "model": "test-model",
                        "usage": {"input": 10, "output": 5, "cost": {"total": 0.01}}, "stopReason": "stop",
                    }}, {"type": "agent_end"},
                ]])
    assert completed == [0]
    result = json.loads((feature_branch_repo / _RESULT_FILE).read_text())
    assert result == {"timed-out": {"output": "", "aborted": "wall_budget_exceeded"},
                      "healthy": {"output": "healthy result", "aborted": None}}
    paths = list((feature_branch_repo / ".daydream/runs").glob("*/trajectory.json"))
    assert len(paths) == 1
    trajectory = json.loads(paths[0].read_text())
    # Subtrajectories are registered in completion order, which races between the
    # reaped and the healthy sibling; pair each lifecycle with its own outcome instead.
    evidence = [attempt["backend_lifecycle"] for attempt in trajectory["extra"]["subtrajectories"]]
    assert sorted((entry["text_delta_count"], entry["generation_open"]) for entry in evidence) == [
        (0, True), (1, False),
    ]
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 10
    assert trajectory["final_metrics"]["total_completion_tokens"] == 5
    assert trajectory["final_metrics"]["total_cost_usd"] == 0.01
    attempts = _kind(collector.spans, "attempt")
    assert len(attempts) == 2
    assert {attributes(span)["daydream.outcome"] for span in attempts} == {"wall_budget_exceeded", "success"}
    generations = _kind(collector.spans, "generation")
    assert len(generations) == 1
    assert attributes(generations[0])["daydream.generation.billed"] is False
    assert "gen_ai.usage.cost" not in attributes(generations[0])
    billed_attempts = [attributes(span) for span in attempts if "gen_ai.usage.cost" in attributes(span)]
    assert len(billed_attempts) == 1
    assert billed_attempts[0]["daydream.billing.owner"] == "structural_attempt"
    assert billed_attempts[0]["gen_ai.usage.cost"] == 0.01
    assert "private-delta-content" not in json.dumps(collector.requests)


async def test_runner_cleanup_outage_preserves_captured_root_and_review_output(
    ext_dir: ExtDir, feature_branch_repo: Path,
    make_config: Callable[..., RunConfig], install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    _flow(ext_dir)
    install_backend(ScriptedBackend(events=[TextEvent("review completed during outage"), ResultEvent(None, None)]))
    exporter = InMemorySpanExporter()
    def with_memory_exporter() -> Registry:
        registry = build_registry()
        registry.register_trace_exporter("memory", lambda _: exporter)
        return registry

    monkeypatch.setattr(runner, "build_registry", with_memory_exporter)
    def fail_shutdown(_exporter: InMemorySpanExporter) -> None:
        raise RuntimeError("exporter unavailable")
    monkeypatch.setattr(InMemorySpanExporter, "shutdown", fail_shutdown)
    config = make_config(feature_branch_repo, flow_name="trace-failures", dataset_capture=True,
        dataset_store_path=feature_branch_repo.parent / "records",
        observability=ObservabilityConfig(destinations=("memory",)))
    with anyio.fail_after(15):
        assert await runner.run(config) == 0
    assert "Trace exporter shutdown failed" in caplog.text
    assert json.loads((feature_branch_repo / _RESULT_FILE).read_text()) == {
        "output": "review completed during outage", "aborted": None,
    }
    root, = [span for span in exporter.get_finished_spans()
             if (span.attributes or {}).get("daydream.span.kind") == "run"]
    assert root.context is not None
    trace_id = f"{root.context.trace_id:032x}"
    assert len(trace_id) == 32 and int(trace_id, 16) != 0
    assert config.dataset_store_path is not None
    store = LocalRecordStore(config.dataset_store_path)
    records = store.read_snapshot(store.select_snapshot(observed_before="2100-01-01T00:00:00Z"))
    record, = records.runs
    assert record["outcome"] == "success" and record["trace_id"] == trace_id
    assert "review completed during outage" in json.dumps(record["trajectories"]["value"])


@pytest.mark.parametrize("interruption", ["wall"])
async def test_runner_partial_outcome_preserves_veto_and_budget_reasons(
    interruption: str, ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = {"supervisor": "", "tools": "tool_call_budget=0,", "wall": "wall_budget_s=0.5,"}[interruption]
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
        "output": "partial response before interruption", "aborted": reason,
    }
    spans = collector.spans
    root, = _kind(spans, "run")
    tool, = _kind(spans, "tool")
    attempt, = _kind(spans, "attempt")
    agent, = _kind(spans, "agent")
    review, = [span for span in _kind(spans, "step") if attributes(span)["daydream.step"] == "trace-failures"]
    review_chain = (root, review, agent, attempt, tool)
    assert all(span["status"]["code"] == "STATUS_CODE_ERROR" for span in review_chain)
    for parent, child in zip(review_chain, review_chain[1:]):
        assert child["parentSpanId"] == parent["spanId"]
    finalization, = [span for span in spans if attributes(span).get("daydream.step") == "artifact-finalization"]
    assert finalization["parentSpanId"] == root["spanId"]
    assert finalization["status"]["code"] == "STATUS_CODE_OK"
    assert attributes(tool)["daydream.outcome"] == reason.split(":")[0]
    assert tool["status"]["message"] == reason.split(":")[0]
    assert tool["events"][-1]["name"] == "exception"
    assert attributes(attempt)["gen_ai.usage.input_tokens"] == 5
    assert attributes(attempt)["gen_ai.usage.cost"] == 0.005
    assert attributes(root)["daydream.exit_code"] == 1

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
    generation_id: str, text: str, *, native_started_at_unix_ms: int | None = 1_778_000_000_000,
    ended_at_unix_ns: int = 1_778_000_000_500_000_000, response_id: str | None = None,
) -> GenerationEndEvent:
    return GenerationEndEvent(
        generation_id=generation_id, native_started_at_unix_ms=native_started_at_unix_ms,
        ended_at_unix_ns=ended_at_unix_ns, end_source="host_observed_message_end",
        choice_parts=(TextChoicePart(text=text), ToolCallChoicePart(call_id=f"{generation_id}-call",
                                                                   name="Read", arguments={"path": "src/x.py"})),
        response_id=response_id, model_name="gen-model", provider_name="gen-provider", finish_reason="stop",
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
    ext_dir: ExtDir, repo: Path, make_config: Callable[..., RunConfig], install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch, backend: object, body: str | None = None,
) -> TraceCollector:
    """One real runner.run() over a scripted backend, returning the collector."""
    _flow(ext_dir, body=body) if body is not None else _flow(ext_dir)
    install_backend(backend)
    with otlp_collector() as collector:
        config = _config(make_config, repo, collector.base_url, monkeypatch)
        assert await runner.run(config) == 0
    return collector


class _ExactAllocationBackend(_NoCancelBackend):
    """Two complete generations, late per-generation usage, terminal total."""

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
        yield CostEvent(0.003, 150, 30, measurement_source="terminal")
        yield TextEvent("final answer")
        yield ResultEvent(None, None)

async def test_runner_exact_allocation_bills_children_and_chain_matches_wire(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _ExactAllocationBackend()
    collector = await _run_lifecycle_flow(
        ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch, backend
    )
    lifecycle = _read_subtrajectory(feature_branch_repo)
    assert lifecycle["billing_owner"] == "generation_children"
    assert lifecycle["diagnostics"] == []
    assert lifecycle["authoritative_total"] == {"input_tokens": 150, "output_tokens": 30, "cost_usd": 0.003}
    assert {draft["generation_id"]: draft["billed"] for draft in lifecycle["drafts"]} == {
        "gen-a": True, "gen-b": True,
    }
    spans = collector.spans
    gens = {attributes(span)["daydream.generation.id"]: span for span in _kind(spans, "generation")}
    assert set(gens) == {"gen-a", "gen-b"}
    attempt = _kind(spans, "attempt")[0]
    for generation_id, (input_tokens, output_tokens, cost) in {
        "gen-a": (100, 20, 0.002), "gen-b": (50, 10, 0.001),
    }.items():
        meta = attributes(gens[generation_id])
        assert gens[generation_id]["status"]["code"] == "STATUS_CODE_OK"
        assert meta["daydream.generation.billed"] is True
        assert meta["gen_ai.response.id"] == {"gen-a": "resp-alpha", "gen-b": "resp-beta"}[generation_id]
        assert meta["gen_ai.response.model"] == "gen-model"
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


class _LateMissingMetricsBackend(_NoCancelBackend):
    """Terminal total without any per-generation usage: partial evidence."""

    async def execute(self, _cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent]:
        yield RequestEvent(prompt)
        yield _gen_start("gen-late")
        yield _gen_end("gen-late", "choice arrived, usage never did")
        yield _gen_start("gen-bool")
        yield _gen_end("gen-bool", "bool start", native_started_at_unix_ms=True)
        yield _gen_start("gen-missing")
        yield _gen_end("gen-missing", "missing start", native_started_at_unix_ms=None)
        yield _gen_start("gen-reversed")
        yield _gen_end("gen-reversed", "reversed start", native_started_at_unix_ms=1_778_000_001_000)
        yield CostEvent(0.02, 200, 40, measurement_source="terminal")
        yield TextEvent("answer")
        yield ResultEvent(None, None)

async def test_runner_late_missing_metrics_bill_chain_children_stay_custom(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = await _run_lifecycle_flow(
        ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch, _LateMissingMetricsBackend(),
    )
    lifecycle = _read_subtrajectory(feature_branch_repo)
    assert lifecycle["billing_owner"] == "structural_attempt"
    assert {draft["generation_id"]: (draft["native_started_at_unix_ns"], draft["duration_ns"], draft["start_fallback"])
            for draft in lifecycle["drafts"]} == {
        "gen-late": (1_778_000_000_000_000_000, 500_000_000, None),
        "gen-bool": (None, None, "invalid"), "gen-missing": (None, None, "missing"),
        "gen-reversed": (None, None, "reversed"),
    }
    assert all(draft["billed"] is False and "usage" not in draft for draft in lifecycle["drafts"])
    spans = collector.spans
    generations = {attributes(span)["daydream.generation.id"]: span for span in _kind(spans, "generation")}
    assert set(generations) == {"gen-late", "gen-bool", "gen-missing", "gen-reversed"}
    generation = generations["gen-late"]
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
