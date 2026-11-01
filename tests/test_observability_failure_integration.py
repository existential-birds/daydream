"""Failure, fanout and cancellation traces through the production runner and OTLP wire."""

from __future__ import annotations

import base64
import json
import logging
import textwrap
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from pathlib import Path
from typing import Any

import anyio
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

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
from daydream.backends.pi import PiBackend
from daydream.dataset import LocalRecordStore
from daydream.extensions import Registry, build_registry
from daydream.observability.config import ObservabilityConfig
from daydream.prompts.grounding import REPOSITORY_SEARCH_GUIDANCE
from daydream.run_config import RunConfig
from tests.conftest import ExtDir
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_cli_process import install_fake_cli_process
from tests.harness.git_helpers import commit, write_and_stage
from tests.harness.otlp import TraceCollector, attributes, kind_of as _kind, otlp_collector
from tests.harness.review_result import saved_coverage
from tests.harness.stub_backend import review_stage_state

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


class _RetryableFailure(RuntimeError):
    retryable = True

async def test_runner_retry_keeps_failed_billed_attempt_separate_from_success(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _flow(ext_dir)
    backend = ScriptedBackend(
        script=[
            [
                RequestEvent("failed attempt request"), TextEvent("failed partial response"), CostEvent(0.03, 30, 3),
                _RetryableFailure("retry this transport failure"),
            ],
            [
                RequestEvent("successful attempt request"), TextEvent("successful final response"),
                CostEvent(0.02, 20, 2), ResultEvent(None, None),
            ],
        ], retry_attempts=1, retry_base_delay_s=0, retry_max_delay_s=0,
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


class _NoCancelBackend:
    """Backend stub for tests that never exercise cancellation."""

    model = "lifecycle-model"


class _FanoutBackend(_NoCancelBackend):
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

async def test_runner_fanout_shared_backend_and_tool_ids_keep_sibling_content_isolated(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
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
        "left": "answer for left", "right": "answer for right",
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


@pytest.mark.parametrize("attribution", ["detected", "explicit", "sharded"])
async def test_runner_detected_review_stacks_attribute_only_their_own_subtrees(
    multi_stack_target: Path, tmp_path: Path, install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, attribution: str,
) -> None:
    """Real detection attributes overlapping reviewers without guessing shared-step stacks."""
    sharded = attribution == "sharded"
    selected_stack = "python" if attribution == "explicit" else None
    if sharded:
        write_and_stage(multi_stack_target, "other.py", "VALUE = 1\n")
        commit(multi_stack_target, "add another Python review scope")
    overlapping = {"python#0" if sharded else "python", "react"}
    expected_scopes = {"python#0", "python#1", "react", "generic", "structure"} if sharded else {
        "python", "react", "generic", "structure",
    }
    ready, release = anyio.Event(), anyio.Event()
    active: set[str] = set()

    class TracedReviewBackend(EmptyReviewBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            stage = review_stage_state(prompt)
            scope = stage["scope_id"] if stage is not None else None
            yield RequestEvent(prompt)
            yield _gen_start("shared-generation-id")
            if scope is not None:
                yield ToolStartEvent("shared-tool-id", "Read", {"review_scope": scope})
                if scope in overlapping:
                    active.add(scope)
                    if active == overlapping:
                        ready.set()
                    await release.wait()
                    active.remove(scope)
                yield ToolResultEvent("shared-tool-id", f"source for {scope}", False)
            yield _gen_end("shared-generation-id", f"reviewed {scope or 'shared step'}")
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    backend = TracedReviewBackend(multi_stack_target)
    install_backend(backend)
    trajectory_path = tmp_path / "trajectory.json"
    with otlp_collector() as collector:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", collector.base_url + "/v1/traces")
        config = empty_review_config(
            multi_stack_target, trajectory_path, stack=selected_stack,
            deep_shard_enabled=sharded, deep_shard_max_files=1,
            observability=ObservabilityConfig(destinations=("otlp",)),
        )

        async def drive() -> None:
            assert await runner.run(config) == 0

        with anyio.fail_after(60):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(drive)
                await ready.wait()
                await anyio.wait_all_tasks_blocked()
                assert active == overlapping
                release.set()

    coverage = saved_coverage(multi_stack_target / ".daydream/deep")
    assert set(coverage.scopes) == expected_scopes
    assert all(scope["status"] == "complete" for scope in coverage.scopes.values())
    report = (multi_stack_target / ".review-output.md").read_text()
    assert report and report == (multi_stack_target / ".daydream/deep/review-output.md").read_text()
    trajectory = json.loads(trajectory_path.read_text())
    spans = collector.spans
    root_run_id = attributes(_kind(spans, "run")[0])["daydream.run.id"]
    by_id = {span["spanId"]: span for span in spans}
    reviewer_ids: set[str] = set()
    seen: set[str] = set()
    for tool in _kind(spans, "tool"):
        scope = json.loads(attributes(tool)["traceloop.entity.input"])["review_scope"]
        seen.add(scope)
        assert json.loads(attributes(tool)["traceloop.entity.output"]) == f"source for {scope}"
        attempt = by_id[tool["parentSpanId"]]
        agent = by_id[attempt["parentSpanId"]]
        step = by_id[agent["parentSpanId"]]
        assert attributes(step)["daydream.span.kind"] == "step"
        generations = [span for span in _kind(spans, "generation") if span["parentSpanId"] == attempt["spanId"]]
        assert len(generations) == 1
        for span, kind in ((tool, "tool"), (attempt, "attempt"), (agent, "agent"), (generations[0], "generation")):
            metadata = attributes(span)
            assert metadata["daydream.span.kind"] == kind
            expected_stack = selected_stack if selected_stack is not None else scope.split("#")[0]
            assert metadata.get("daydream.stack") == expected_stack
            assert metadata["daydream.session.id"] == trajectory["session_id"]
            assert metadata["traceloop.association.properties.session_id"] == trajectory["session_id"]
            assert metadata["daydream.run.id"] == root_run_id
            assert metadata["traceloop.association.properties.daydream_run_id"] == root_run_id
            assert metadata["daydream.trajectory.descriptor"] == f"deep-{scope}"
            assert metadata["daydream.trajectory.id"] == f"{trajectory['session_id']}:deep-{scope}"
            reviewer_ids.add(span["spanId"])
    assert seen == set(coverage.scopes)
    assert len({span["traceId"] for span in spans}) == 1
    shared = [span for span in spans if span["spanId"] not in reviewer_ids]
    assert any(attributes(span).get("daydream.span.kind") == "generation" for span in shared)
    if selected_stack is None:
        assert all("daydream.stack" not in attributes(span) for span in shared)
    else:
        assert all(attributes(span).get("daydream.stack") == selected_stack for span in shared
                   if attributes(span)["daydream.span.kind"] != "run")
    assert not [record for record in caplog.records if "Invalid type NoneType" in record.getMessage()]


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
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
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
    assert attributes(root)["daydream.outcome"] == "cancelled"
    assert attributes(tool)["daydream.outcome"] == "cancelled"
    assert tool["status"]["message"] == "cancelled"
    assert tool["events"][-1]["name"] == "exception"
    assert all(int(span["endTimeUnixNano"]) >= int(span["startTimeUnixNano"]) for span in spans)


async def test_pi_runner_external_cancellation_flushes_progress_and_reaps_all_children(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run cancellation preserves native deltas and retains backend-wide shutdown semantics."""
    _flow(ext_dir, body='''
async with anyio.create_task_group() as group:
    for label in ("first", "second"):
        async def invoke(label=label):
            await run_agent(ctx.backend_for("review"), ctx.work.repo, label, phase=DaydreamPhase.REVIEW)
        group.start_soon(invoke)
''')
    install_backend(PiBackend(model="test-model"))
    lines = [json.dumps({"type": "message_start", "message": {"role": "assistant"}})]
    lines.extend(json.dumps({"type": "message_update", "assistantMessageEvent": {
        "type": "thinking_delta", "contentIndex": 0, "delta": "private-interrupted-reasoning" + "x" * 512,
    }}) for _ in range(512))
    spawner = install_fake_cli_process(monkeypatch, "pi", lines=lines, hang=True)
    monkeypatch.setenv("DAYDREAM_STREAM_IDLE_TIMEOUT_S", "0")
    monkeypatch.setenv("DAYDREAM_TRACE_CONTENT", "metadata")
    cancelled = False
    with otlp_collector() as collector:
        config = _config(make_config, feature_branch_repo, collector.base_url, monkeypatch)
        async def launch() -> None:
            nonlocal cancelled
            try:
                await runner.run(config)
                pytest.fail("cancelled runner returned")
            except anyio.get_cancelled_exc_class():
                cancelled = True
                raise
        with anyio.fail_after(15):
            async with anyio.create_task_group() as group:
                group.start_soon(launch)
                await spawner.wait_for_read(0)
                await spawner.wait_for_read(1)
                group.cancel_scope.cancel()
    assert cancelled and len(spawner.procs) == 2
    assert all(proc.reaped and proc.returncode == -15 for proc in spawner.procs)
    paths = list((feature_branch_repo / ".daydream/runs").glob("*/trajectory.json"))
    assert len(paths) == 1
    trajectory = json.loads(paths[0].read_text())
    attempts = trajectory["extra"]["subtrajectories"]
    assert len(attempts) == 2
    for attempt in attempts:
        evidence = attempt["backend_lifecycle"]
        assert evidence["state"] == "generating"
        assert evidence["thinking_delta_count"] == 512
        assert evidence["generation_open"] is True
        assert evidence["response_idle_timeout_s"] == 0
        assert evidence["last_progress_age_s"] >= 0
        assert len(json.dumps(evidence)) < 2000
        assert attempt["generation_lifecycle"]["billing_owner"] == "none"
        assert attempt["generation_lifecycle"]["drafts"][0]["sealed"] is False
    assert any(step.get("extra", {}).get("stop_reason") == "cancelled" for step in trajectory["steps"])
    exported = _kind(collector.spans, "attempt")
    assert len(exported) == 2
    for span in exported:
        attrs = attributes(span)
        assert attrs["daydream.outcome"] == "cancelled"
        assert attrs["daydream.backend_lifecycle.thinking_delta_count"] == 512
        assert attrs["daydream.backend_lifecycle.generation_open"] is True
        assert not any(key.startswith("gen_ai.usage.") for key in attrs)
    assert not _kind(collector.spans, "generation")
    assert "private-interrupted-reasoning" not in json.dumps(collector.requests)


async def test_pi_runner_idle_retry_retains_each_attempt_evidence_and_original_deadline(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native silence exports both original attempts under the existing one-stall-retry cap."""
    _flow(ext_dir, body='''
from daydream.backends._subprocess import StreamStalledError
try:
    await run_agent(ctx.backend_for("review"), ctx.work.repo, "silent review",
        phase=DaydreamPhase.REVIEW, wall_budget_s=5)
except StreamStalledError:
    (ctx.data["daydream_dir"] / "trace-failure-result.json").write_text(json.dumps({"error": "StreamStalledError"}))
    return Stop(1)
''')
    monkeypatch.setenv("DAYDREAM_STREAM_IDLE_TIMEOUT_S", "0.02")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_MAX_DELAY_S", "0")
    install_backend(PiBackend(model="test-model"))
    spawner = install_fake_cli_process(monkeypatch, "pi", lines=[], hang=True)
    monkeypatch.setenv("DAYDREAM_TRACE_CONTENT", "metadata")
    with otlp_collector() as collector:
        assert await runner.run(_config(make_config, feature_branch_repo, collector.base_url, monkeypatch)) == 1
    assert json.loads((feature_branch_repo / _RESULT_FILE).read_text()) == {"error": "StreamStalledError"}
    assert len(spawner.procs) == 2
    assert all(proc.reaped and proc.returncode == -15 for proc in spawner.procs)
    paths = list((feature_branch_repo / ".daydream/runs").glob("*/trajectory.json"))
    assert len(paths) == 1
    trajectory = json.loads(paths[0].read_text())
    evidence = [attempt["backend_lifecycle"] for attempt in trajectory["extra"]["subtrajectories"]]
    assert [entry["attempt"] for entry in evidence] == [1, 2]
    assert all(entry["state"] == "launched" and entry["response_idle_timeout_s"] == 0.02 for entry in evidence)
    assert evidence[1]["wall_allowance_s"] < evidence[0]["wall_allowance_s"]
    assert evidence[1]["remaining_wall_s"] < evidence[0]["remaining_wall_s"]
    attempts = sorted(_kind(collector.spans, "attempt"), key=lambda span: int(span["startTimeUnixNano"]))
    assert len(attempts) == 2
    for span, entry in zip(attempts, evidence, strict=True):
        attrs = attributes(span)
        assert attrs["daydream.outcome"] == "error"
        assert span["events"][-1]["attributes"]
        assert attrs["daydream.backend_lifecycle.attempt"] == entry["attempt"]
        assert attrs["daydream.backend_lifecycle.response_idle_timeout_s"] == 0.02
        assert not any(key.startswith("gen_ai.usage.") for key in attrs)

async def test_runner_export_outage_warns_and_preserves_review_result_and_output(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    _flow(ext_dir)
    install_backend(ScriptedBackend(events=[TextEvent("review completed during outage"), ResultEvent(None, None)]))
    with otlp_collector(status=503) as collector:
        config = _config(make_config, feature_branch_repo, collector.base_url, monkeypatch)
        config.dataset_capture = True
        config.dataset_store_path = feature_branch_repo.parent / "records"
        with anyio.fail_after(15):
            rc = await runner.run(config)
    assert rc == 0
    assert json.loads((feature_branch_repo / _RESULT_FILE).read_text()) == {
        "output": "review completed during outage", "aborted": None,
    }
    assert collector.requests
    assert attributes(_kind(collector.spans, "run")[0])["daydream.exit_code"] == 0
    assert "Trace export failed; review execution continues" in caplog.text
    store = LocalRecordStore(config.dataset_store_path)
    records = store.read_snapshot(store.select_snapshot(observed_before="2100-01-01T00:00:00Z"))
    assert len(records.runs) == 1
    assert records.runs[0]["outcome"] == "success"
    trace_id, = {base64.b64decode(root["traceId"]).hex() for root in _kind(collector.spans, "run")}
    assert len(trace_id) == 32 and int(trace_id, 16) != 0
    assert records.runs[0]["trace_id"] == trace_id
    assert records.runs[0]["trajectories"]["status"] == "available"
    assert "review completed during outage" in json.dumps(records.runs[0]["trajectories"]["value"])


@pytest.mark.parametrize(("outage", "warning"), [
    ("flush", "Trace processor flush failed"),
    ("shutdown", "Trace exporter shutdown failed"),
])
async def test_runner_cleanup_outage_preserves_captured_root_and_review_output(
    outage: str, warning: str, ext_dir: ExtDir, feature_branch_repo: Path,
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
    if outage == "flush":
        def fail_flush(_provider: TracerProvider, timeout_millis: int = 30000) -> bool:
            raise RuntimeError("processor unavailable")
        monkeypatch.setattr(TracerProvider, "force_flush", fail_flush)
    else:
        def fail_shutdown(_exporter: InMemorySpanExporter) -> None:
            raise RuntimeError("exporter unavailable")
        monkeypatch.setattr(InMemorySpanExporter, "shutdown", fail_shutdown)
    config = make_config(feature_branch_repo, flow_name="trace-failures", dataset_capture=True,
        dataset_store_path=feature_branch_repo.parent / "records",
        observability=ObservabilityConfig(destinations=("memory",)))
    with anyio.fail_after(15):
        assert await runner.run(config) == 0
    assert warning in caplog.text
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


@pytest.mark.parametrize("interruption", ["supervisor", "tools", "wall"])
async def test_runner_partial_outcome_preserves_veto_and_budget_reasons(
    interruption: str, ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
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

@pytest.mark.parametrize("terminal_result", [False, True])
async def test_runner_attempt_duration_requires_terminal_result(
    terminal_result: bool, ext_dir: ExtDir, feature_branch_repo: Path,
    make_config: Callable[..., RunConfig], install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _flow(ext_dir)
    emitted = anyio.Event()
    timings: list[AgentEvent] = [
        MetricsEvent("message-one", 5, 1, None, None, duration_ms=123),
        ToolResultEvent("timed-tool", "source content", False, duration_ms=42),
    ]

    class TimedBackend(_NoCancelBackend):
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
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
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
    assert json.loads(attributes(agent)["traceloop.entity.input"])["prompt"] == (
        "original logical prompt\n\n" + REPOSITORY_SEARCH_GUIDANCE
    )


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


async def test_runner_default_non_loop_generation_omits_optional_scope_attributes_without_warnings(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = ScriptedBackend(script=[[
        RequestEvent("inspect sample"), _gen_start("gen-default"), _gen_end("gen-default", "sealed choice"),
        TextEvent("final answer"), ResultEvent(None, None),
    ]])
    with caplog.at_level(logging.WARNING, logger="opentelemetry.attributes"):
        collector = await _run_lifecycle_flow(
            ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch, backend,
        )

    assert json.loads((feature_branch_repo / _RESULT_FILE).read_text()) == {"output": "final answer", "aborted": None}
    lifecycle = _read_subtrajectory(feature_branch_repo)
    assert [draft["generation_id"] for draft in lifecycle["drafts"]] == ["gen-default"]
    assert lifecycle["billing_owner"] == "none"
    assert lifecycle["drafts"][0]["billed"] is False and "usage" not in lifecycle["drafts"][0]
    spans = collector.spans
    by_kind = {kind: _kind(spans, kind) for kind in ("run", "agent", "attempt", "generation")}
    assert all(len(values) == 1 for values in by_kind.values())
    assert by_kind["run"][0]["status"]["code"] == "STATUS_CODE_OK"
    assert attributes(by_kind["generation"][0])["daydream.generation.id"] == "gen-default"
    review, = [span for span in _kind(spans, "step") if attributes(span)["daydream.step"] == "trace-failures"]
    review_chain = (by_kind["run"][0], review, *(by_kind[kind][0] for kind in ("agent", "attempt", "generation")))
    for parent, child in zip(review_chain, review_chain[1:]):
        assert child["parentSpanId"] == parent["spanId"]
    finalization, = [span for span in spans if attributes(span).get("daydream.step") == "artifact-finalization"]
    assert finalization["parentSpanId"] == by_kind["run"][0]["spanId"]
    assert finalization["status"]["code"] == "STATUS_CODE_OK"
    for span in spans:
        attrs = attributes(span)
        assert "daydream.stack" not in attrs
        assert "daydream.iteration" not in attrs
    assert not [
        record for record in caplog.records
        if record.name == "opentelemetry.attributes" and "Invalid type NoneType" in record.getMessage()
    ]


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


class _DuplicateMetricsBackend(_NoCancelBackend):
    """Per-generation usage whose sum exceeds the terminal total."""

    async def execute(self, _cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent]:
        yield RequestEvent(prompt)
        yield _gen_start("gen-x")
        yield _gen_end("gen-x", "choice")
        yield MetricsEvent("gen-x", 80, 30, None, 0.004, generation_id="gen-x")
        yield CostEvent(0.002, 70, 25, measurement_source="terminal")  # contradiction
        yield TextEvent("answer")
        yield ResultEvent(None, None)

async def test_runner_contradictory_metrics_fail_closed_without_rewriting(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contradictory totals bill neither side and retain the fixed local diagnostic without rewriting."""
    collector = await _run_lifecycle_flow(
        ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch, _DuplicateMetricsBackend(),
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


class _DuplicateIdempotentBackend(_NoCancelBackend):
    """A billed child without response identity retains idempotent terminal totals."""

    async def execute(self, _cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent]:
        yield RequestEvent(prompt)
        yield _gen_start("gen-dup")
        yield _gen_end("gen-dup", "choice")
        yield MetricsEvent("gen-dup", 60, 15, None, 0.003, generation_id="gen-dup")
        yield CostEvent(0.003, 60, 15, measurement_source="terminal")
        yield CostEvent(0.003, 60, 15, measurement_source="terminal")  # exact duplicate
        yield TextEvent("answer")
        yield ResultEvent(None, None)

async def test_runner_duplicate_identical_totals_are_idempotent(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A billed native child keeps missing response identity absent on the wire."""
    collector = await _run_lifecycle_flow(
        ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch, _DuplicateIdempotentBackend(),
    )
    lifecycle = _read_subtrajectory(feature_branch_repo)
    assert lifecycle["billing_owner"] == "generation_children"
    assert lifecycle["diagnostics"] == []
    attempt = _kind(collector.spans, "attempt")[0]
    assert attributes(attempt)["gen_ai.usage.input_tokens"] == 60
    generation, = _kind(collector.spans, "generation")
    meta = attributes(generation)
    assert generation["status"]["code"] == "STATUS_CODE_OK"
    assert meta["daydream.generation.id"] == "gen-dup"
    assert meta["daydream.generation.billed"] is True
    assert meta["gen_ai.usage.input_tokens"] == 60
    assert meta["gen_ai.usage.output_tokens"] == 15
    assert meta["gen_ai.usage.cost"] == pytest.approx(0.003)
    assert "gen_ai.response.id" not in meta


class _ResidualTotalBackend(_NoCancelBackend):
    """Terminal total above the per-message sum: residual folds to the chain."""

    async def execute(self, _cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent]:
        yield RequestEvent(prompt)
        yield TextEvent("streamed answer")
        yield MetricsEvent("", 40, 8, None, 0.001)  # per-message, no generation
        yield CostEvent(0.004, 90, 18, measurement_source="terminal")  # residual 50/10
        yield TextEvent(" more")
        yield ResultEvent(None, None)

async def test_runner_residual_unallocated_total_folds_onto_chain(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = await _run_lifecycle_flow(
        ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch, _ResidualTotalBackend(),
    )
    attempt = _kind(collector.spans, "attempt")[0]
    meta = attributes(attempt)
    assert meta["daydream.billing.owner"] == "structural_attempt"
    assert meta["gen_ai.usage.input_tokens"] == 90
    assert meta["gen_ai.usage.output_tokens"] == 18
    assert meta["gen_ai.usage.cost"] == pytest.approx(0.004)
    message_usage = json.loads(meta["daydream.message_usage"])
    assert message_usage[0]["usage"]["input_tokens"] == 40


class _ToolErrorAfterSealBackend(_NoCancelBackend):
    """Tool error after a sealed completed generation."""

    async def execute(self, _cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent]:
        yield RequestEvent(prompt)
        yield _gen_start("gen-tool-err")
        yield _gen_end("gen-tool-err", "sealed before the tool failed", response_id="resp-tool-err")
        yield MetricsEvent("gen-tool-err", 30, 6, None, 0.0005, generation_id="gen-tool-err")
        yield ToolStartEvent("failing-tool", "Bash", {"command": "exit 3"}, timestamp="2026-09-05T10:00:00Z")
        yield ToolResultEvent("failing-tool", "read failed opaque-secret", True,
                              timestamp="2026-09-05T10:00:02Z", duration_ms=2000)
        yield CostEvent(0.0005, 30, 6, measurement_source="terminal")
        yield TextEvent("recovered answer")
        yield ResultEvent(None, None)

async def test_runner_tool_error_after_sealed_generation_keeps_allocation(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later tool ERROR keeps sealed generation billing and historical timing intact."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "opaque-secret")
    collector = await _run_lifecycle_flow(
        ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch, _ToolErrorAfterSealBackend(),
    )
    spans = collector.spans
    tool = _kind(spans, "tool")[0]
    assert tool["status"]["code"] == "STATUS_CODE_ERROR"
    assert attributes(tool)["daydream.outcome"] == "error"
    assert attributes(tool)["daydream.tool.error"] is True
    assert tool["status"]["message"] == "error" and tool["events"][-1]["name"] == "exception"
    error = attributes(tool["events"][-1])
    assert error["exception.type"] == "ToolError" and "read failed" in error["exception.message"]
    assert "opaque-secret" not in json.dumps(tool)
    assert attributes(tool)["daydream.tool.started_at"] == "2026-09-05T10:00:00Z"
    assert attributes(tool)["daydream.tool.completed_at"] == "2026-09-05T10:00:02Z"
    generation = _kind(spans, "generation")[0]
    meta = attributes(generation)
    assert meta["daydream.generation.billed"] is True
    assert meta["gen_ai.response.id"] == "resp-tool-err"
    assert generation["status"]["code"] == "STATUS_CODE_OK"
    attempt = _kind(spans, "attempt")[0]
    assert attributes(attempt)["daydream.billing.owner"] == "generation_children"
    assert attributes(_kind(spans, "run")[0])["daydream.exit_code"] == 0


class _PendingCapBackend(_NoCancelBackend):
    """513 sealed generations trip the 512-draft cap on the final seal."""

    async def execute(self, _cwd: Path, prompt: str, *_args: Any, **_kwargs: Any) -> AsyncGenerator[AgentEvent]:
        yield RequestEvent(prompt)
        for index in range(513):
            generation_id = f"gen-{index:04d}"
            yield _gen_start(generation_id)
            yield _gen_end(generation_id, f"choice {index}", native_started_at_unix_ms=None)
            yield MetricsEvent(generation_id, 1, 1, None, None, generation_id=generation_id)
        yield CostEvent(0.01, 513, 513, measurement_source="terminal")
        yield TextEvent("done")
        yield ResultEvent(None, None)

async def test_runner_pending_count_cap_drains_children_stay_unbilled(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At the 512-draft cap, end every draft, bill the attempt, and retain only a count diagnostic."""
    # Allow encoding and loopback delivery of a full 512-span batch under CI load.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT", "5")
    collector = await _run_lifecycle_flow(
        ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch, _PendingCapBackend(),
    )
    lifecycle = _read_subtrajectory(feature_branch_repo)
    assert lifecycle["billing_owner"] == "structural_attempt"
    assert lifecycle["children_after_cap"] is True
    assert len(lifecycle["drafts"]) == 513
    assert all(draft["ended"] for draft in lifecycle["drafts"])
    assert all(draft["billed"] is False for draft in lifecycle["drafts"])
    assert all(draft["usage"] == {"input_tokens": 1, "output_tokens": 1} for draft in lifecycle["drafts"])
    assert lifecycle["authoritative_total"] == {"input_tokens": 513, "output_tokens": 513, "cost_usd": 0.01}
    cap = [d for d in lifecycle["diagnostics"] if "cap" in d]
    assert len(cap) == 1 and "512" in cap[0]
    generations = _kind(collector.spans, "generation")
    assert len(generations) == 513
    assert all("gen_ai.usage.input_tokens" not in attributes(span) for span in generations)
    assert attributes(_kind(collector.spans, "attempt")[0])["daydream.billing.owner"] == "structural_attempt"
    assert attributes(_kind(collector.spans, "attempt")[0])["gen_ai.usage.input_tokens"] == 513
    payload = json.dumps([request["body"] for request in collector.requests])
    assert "generation_pending_cap" not in payload  # diagnostics never on the wire

async def test_runner_resume_native_conversation_distinct_from_daydream_session(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume the backend’s native continuation while preserving the Daydream session.

    Only the resumed attempt carries the native gen_ai.conversation.id."""
    backend = ScriptedBackend(
        script=[
            [
                RequestEvent("original invocation"), TextEvent("first answer"),
                ResultEvent(None, ContinuationToken("scripted", {"thread_id": "native-thread-77"})),
            ],
            [
                RequestEvent("resumed invocation"), TextEvent("resumed answer"),
                ResultEvent(None, None, session_id="native-thread-77"),
            ],
        ],
    )
    collector = await _run_lifecycle_flow(
        ext_dir, feature_branch_repo, make_config, install_backend, monkeypatch, backend,
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
        "first": "first answer", "resumed": "resumed answer",
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
