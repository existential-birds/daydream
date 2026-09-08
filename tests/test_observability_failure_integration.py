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
    CostEvent,
    MetricsEvent,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.codex import CodexBackend, CodexError
from daydream.observability.config import ObservabilityConfig
from daydream.runner import RunConfig
from tests.conftest import ExtDir
from tests.harness.backend import ScriptedBackend
from tests.harness.otlp import attributes, otlp_collector

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
