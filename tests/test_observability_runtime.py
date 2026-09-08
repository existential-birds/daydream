"""Owned tracing lifecycle and event reconciliation through real SDK spans."""

import json
import logging
import threading
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
from typing import Any

import anyio
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, StatusCode, TraceFlags

from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    CostEvent,
    DiagnosticEvent,
    MetricsEvent,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.extensions import ToolDecision, get_registry, set_registry
from daydream.extensions.registry import Registry
from daydream.observability.config import ObservabilityConfig, ObservabilityError
from daydream.observability.privacy import PrivacyPolicy, diagnostic_scope
from daydream.observability.runtime import trace_run
from daydream.observability.spans import agent_scope, attempt_scope, step_scope
from daydream.trajectory import DaydreamPhase


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_owned_span_tree_usage_and_content() -> None:
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    original_provider = trace.get_tracer_provider()
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review") as run:
        with step_scope("review", iteration=2, stack="python"):
            with agent_scope("review", backend="pi", model="requested") as agent:
                async with attempt_scope(1) as attempt:
                    attempt.observe(RequestEvent("effective prompt", model_name="effective"))
                    attempt.observe(TextEvent("response"))
                    attempt.observe(MetricsEvent("a", 10, 2, 3, 0.1, model_name="actual"))
                    attempt.observe(MetricsEvent("b", 20, 4, None, 0.2))
                    attempt.observe(CostEvent(None, 0, None))
                    attempt.observe(ToolStartEvent("a", "Read", {"file": "one.py"}))
                    attempt.observe(ToolStartEvent("b", "Read", {"file": "two.py"}))
                    attempt.observe(ToolResultEvent("b", "second", False))
                    attempt.observe(ToolResultEvent("a", "first", False))
                agent.output({"accepted": True})
        run.finish(0)
    assert trace.get_tracer_provider() is original_provider
    spans = exporter.get_finished_spans()
    by_kind = {(span.attributes or {})["daydream.span.kind"]: span for span in spans}
    for child, parent in (("step", "run"), ("agent", "step"), ("attempt", "agent"), ("tool", "attempt")):
        parent_context = by_kind[parent].context
        child_parent = by_kind[child].parent
        assert parent_context is not None and child_parent is not None
        assert child_parent.span_id == parent_context.span_id
    attrs = dict(by_kind["attempt"].attributes or {})
    assert attrs["gen_ai.usage.input_tokens"] == 0
    assert attrs["gen_ai.usage.output_tokens"] == 6
    assert attrs["gen_ai.usage.cost"] == pytest.approx(0.3)
    assert attrs["gen_ai.request.model"] == "effective"
    assert attrs["gen_ai.response.model"] == "actual"
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["daydream.invocation.aggregate"] is True
    assert "response" in str(attrs["gen_ai.output.messages"])
    for kind in ("run", "step", "agent", "tool"):
        local = by_kind[kind].attributes or {}
        assert "gen_ai.request.model" not in local
        assert "daydream.invocation.aggregate" not in local
        assert not any(key.startswith("gen_ai.usage.") for key in local)
    assert (by_kind["agent"].attributes or {})["daydream.configured.model"] == "requested"
    tool_attrs = by_kind["tool"].attributes or {}
    assert tool_attrs["gen_ai.operation.name"] == "execute_tool"
    for key, value in {
        "daydream.flow": "review",
        "daydream.step": "review",
        "daydream.phase": "review",
        "daydream.iteration": 2,
        "daydream.stack": "python",
        "daydream.backend": "pi",
        "daydream.attempt": 1,
    }.items():
        assert tool_attrs[key] == value


@pytest.mark.anyio
async def test_attempt_and_nested_step_do_not_inherit_unobserved_request_metadata() -> None:
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
        with agent_scope("review", backend="pi", model="configured-only"):
            async with attempt_scope(1):
                with step_scope("prepare"):
                    pass

    spans = {(span.attributes or {})["daydream.span.kind"]: span for span in exporter.get_finished_spans()}
    for kind in ("attempt", "step"):
        attrs = spans[kind].attributes or {}
        assert "gen_ai.request.model" not in attrs
        assert "daydream.configured.model" not in attrs
    nested = spans["step"].attributes or {}
    assert "gen_ai.operation.name" not in nested
    assert "daydream.invocation.aggregate" not in nested
    assert nested["daydream.backend"] == "pi"
    assert nested["daydream.attempt"] == 1


@pytest.mark.anyio
async def test_attempt_records_only_scrubbed_diagnostic_codes_and_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "opaque-diagnostic-secret"
    monkeypatch.setenv("DAYDREAM_TEST_TOKEN", secret)
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)

    async with trace_run(
        ObservabilityConfig(destinations=("memory",), capture_content=True),
        registry,
        flow="review",
    ):
        with agent_scope("review", backend="codex"):
            async with attempt_scope(1) as attempt:
                attempt.observe(
                    DiagnosticEvent(
                        code=f"parser_{secret}",
                        message=f"do not export {secret}",
                        metadata={"api_key": secret, "count": 99},
                    )
                )
                attempt.observe(
                    DiagnosticEvent(
                        code=f"parser_{secret}",
                        message="another message",
                        metadata={"raw": "not observable"},
                    )
                )
                attempt.observe(
                    DiagnosticEvent(
                        code="codex_transport_coverage",
                        message="transport detail",
                        metadata={"occurrences": 4},
                    )
                )

    spans = exporter.get_finished_spans()
    assert sorted(str((span.attributes or {})["daydream.span.kind"]) for span in spans) == [
        "agent",
        "attempt",
        "run",
    ]
    attempt_span = next(
        span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "attempt"
    )
    attrs = dict(attempt_span.attributes or {})
    assert attrs["daydream.backend_diagnostic.codes"] == (
        "parser_[REDACTED_CREDENTIAL]",
        "codex_transport_coverage",
    )
    assert attrs["daydream.backend_diagnostic.counts"] == (2, 1)
    encoded = str(spans)
    assert secret not in encoded
    assert "do not export" not in encoded
    assert "not observable" not in encoded
    for span in spans:
        if span is not attempt_span:
            assert not any(
                key.startswith("daydream.backend_diagnostic")
                for key in (span.attributes or {})
            )


def test_diagnostics_scrub_formatted_arguments_and_exception(caplog: pytest.LogCaptureFixture) -> None:
    policy = PrivacyPolicy(environ={"LANGSMITH_API_KEY": "opaque-value"})
    with caplog.at_level(logging.WARNING), diagnostic_scope(policy):
        try:
            raise RuntimeError("opaque-value")
        except RuntimeError:
            logging.getLogger("opentelemetry.exporter.test").warning("failed %s", "opaque-value", exc_info=True)
    assert "opaque-value" not in caplog.text
    assert "REDACTED" in caplog.text


def test_flag_valued_secret_env_vars_are_not_harvested_as_credentials() -> None:
    """A boolean/numeric flag under a secret-named env var is not a credential.

    Harvesting e.g. ``HERMES_REDACT_SECRETS=true`` as a literal secret
    literal-replaced every ``true`` in span content, corrupting JSON payloads
    (``{"ok":true}`` read back as ``{"ok":[REDACTED_CREDENTIAL]}``).
    """
    policy = PrivacyPolicy(environ={"HERMES_REDACT_SECRETS": "true", "LANGSMITH_API_KEY": "opaque-value"})
    assert json.loads(policy.json({"ok": True})) == {"ok": True}
    assert json.loads(policy.json({"flag": "true", "n": 1, "off": False})) == {"flag": "true", "n": 1, "off": False}
    assert "opaque-value" not in policy.text("carries opaque-value inside")


@pytest.mark.anyio
async def test_run_spans_survive_flag_valued_secret_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real-path: trace_run builds its policy from the ambient environment, so a
    flag-valued secret-named var must not corrupt the agent span's JSON output."""
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review") as run:
        with agent_scope("review", backend="pi", model="requested") as agent:
            agent.output({"ok": True})
        run.finish(0)
    agent_span = next(
        span for span in exporter.get_finished_spans() if (span.attributes or {}).get("daydream.span.kind") == "agent"
    )
    assert json.loads(str((agent_span.attributes or {})["traceloop.entity.output"])) == {"ok": True}


@pytest.mark.anyio
@pytest.mark.parametrize("stall_during", ["export", "shutdown"])
async def test_shutdown_deadline_is_total_and_shutdown_once(
    stall_during: str, caplog: pytest.LogCaptureFixture,
) -> None:
    release = threading.Event()

    class BlockedExporter(InMemorySpanExporter):
        def __init__(self) -> None:
            super().__init__()
            self.shutdowns = 0
            self.blocked = threading.Event()
            self.shutdown_complete = threading.Event()

        def shutdown(self) -> None:
            self.shutdowns += 1
            if stall_during == "shutdown":
                self.blocked.set()
                release.wait()
            super().shutdown()
            self.shutdown_complete.set()

        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            if stall_during == "export":
                self.blocked.set()
                release.wait()
            return super().export(spans)

    exporters = [BlockedExporter(), BlockedExporter()]
    registry = Registry()
    registry.register_trace_exporter("first", lambda _: exporters[0])
    registry.register_trace_exporter("second", lambda _: exporters[1])
    returned = anyio.Event()

    async def run() -> None:
        async with trace_run(
            ObservabilityConfig(destinations=("first", "second")),
            registry,
            flow="review",
            cleanup_timeout_s=0.03,
        ):
            pass
        returned.set()

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(run)
            try:
                assert await anyio.to_thread.run_sync(exporters[0].blocked.wait, 10, abandon_on_cancel=True)
                await returned.wait()
                assert not release.is_set()
                assert all(not exporter.shutdown_complete.is_set() for exporter in exporters)
                assert "Trace cleanup exceeded its deadline" in caplog.text
            finally:
                release.set()
            for exporter in exporters:
                assert await anyio.to_thread.run_sync(exporter.shutdown_complete.wait, 10, abandon_on_cancel=True)
    assert [exporter.shutdowns for exporter in exporters] == [1, 1]


class _ScriptedBackend:
    model = "requested"
    retry_attempts = 1
    retry_base_delay_s = 0
    retry_max_delay_s = 0

    def __init__(self, attempts: list[list[AgentEvent | BaseException]], *, stall: bool = False) -> None:
        self.attempts = iter(attempts)
        self.stall = stall
        self.cancelled = False

    async def execute(self, *args: Any, **kwargs: Any) -> AsyncGenerator[AgentEvent]:
        for event in next(self.attempts):
            if isinstance(event, BaseException):
                raise event
            yield event
        if self.stall:
            await anyio.sleep_forever()

    async def cancel(self) -> None:
        self.cancelled = True


@pytest.mark.anyio
async def test_retry_keeps_failed_billing_and_agent_records_salvaged_result(tmp_path: Path) -> None:
    class RetryableError(RuntimeError):
        retryable = True

    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    backend = _ScriptedBackend(
        [
            [TextEvent("failed content"), CostEvent(0.05, 4, 2), ResultEvent(None, None), RetryableError("retry")],
            [
                RequestEvent("effective prompt", system_prompt="effective system"),
                TextEvent('{"ok":true}'),
                ResultEvent({"wrong": True}, None),
                MetricsEvent("", 30, 6, 4, 0.2, usage_scope="invocation"),
                CostEvent(None, 0, None),
            ],
        ]
    )
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
        result = await run_agent(
            backend,
            tmp_path,
            "original",
            phase=DaydreamPhase.REVIEW,
            output_schema={"type": "object", "required": ["ok"]},
        )
    assert result[0] == {"ok": True}
    attempts = [
        span for span in exporter.get_finished_spans() if (span.attributes or {}).get("daydream.span.kind") == "attempt"
    ]
    assert [span.status.status_code for span in attempts] == [StatusCode.ERROR, StatusCode.OK]
    first, second = [dict(span.attributes or {}) for span in attempts]
    assert first["gen_ai.usage.cost"] == 0.05
    assert second["gen_ai.usage.input_tokens"] == 0
    assert second["gen_ai.usage.output_tokens"] == 6
    assert "effective system" in str(second["gen_ai.input.messages"])
    assert "failed content" not in str(second)
    agent = next(
        span for span in exporter.get_finished_spans() if (span.attributes or {}).get("daydream.span.kind") == "agent"
    )
    assert json.loads(str((agent.attributes or {})["traceloop.entity.output"])) == {"ok": True}


@pytest.mark.anyio
async def test_metadata_policy_omits_all_content_and_exception_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "opaque-secret")
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    backend = _ScriptedBackend(
        [
            [
                RequestEvent("PRIVATE PROMPT", system_prompt="PRIVATE SYSTEM"),
                TextEvent("PRIVATE TEXT"),
                ThinkingEvent("PRIVATE REASON"),
                ToolStartEvent("tool", "Read", {"arg": "PRIVATE INPUT"}),
                ToolResultEvent("tool", "PRIVATE OUTPUT", True),
                CostEvent(0.1, 2, 3),
                RuntimeError("PRIVATE ERROR opaque-secret"),
            ]
        ]
    )
    with pytest.raises(RuntimeError, match="PRIVATE ERROR"):
        async with trace_run(
            ObservabilityConfig(destinations=("memory",), capture_content=False, service_name="service opaque-secret"),
            registry,
            flow="review",
        ):
            await run_agent(backend, tmp_path, "PRIVATE ORIGINAL", phase=DaydreamPhase.REVIEW)
    for span in exporter.get_finished_spans():
        encoded = str(span.attributes) + str(span.events) + str(span.resource.attributes) + str(span.status)
        assert "PRIVATE" not in encoded and "opaque-secret" not in encoded
        assert not any(
            key.startswith(
                (
                    "traceloop.entity.input",
                    "traceloop.entity.output",
                    "gen_ai.input.messages",
                    "gen_ai.output.messages",
                    "daydream.error.message",
                )
            )
            for key in span.attributes or {}
        )
    attempt = next(
        span for span in exporter.get_finished_spans() if (span.attributes or {}).get("daydream.span.kind") == "attempt"
    )
    assert (attempt.attributes or {})["gen_ai.usage.output_tokens"] == 3
    assert attempt.status.description == "error"
    assert attempt.events[-1].name == "exception"
    assert (attempt.events[-1].attributes or {})["exception.type"] == "RuntimeError"
    assert "exception.message" not in (attempt.events[-1].attributes or {})


@pytest.mark.anyio
async def test_cancelled_run_closes_open_tool_and_exporter(tmp_path: Path) -> None:
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    backend = _ScriptedBackend([[ToolStartEvent("tool", "Read", {})]], stall=True)
    with anyio.move_on_after(0.03) as cancellation:
        async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
            await run_agent(backend, tmp_path, "inspect", phase=DaydreamPhase.REVIEW)
    assert cancellation.cancelled_caught and backend.cancelled
    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    assert all(span.status.status_code == StatusCode.ERROR for span in spans)
    tool = next(span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "tool")
    assert (tool.attributes or {})["daydream.outcome"] == "cancelled"
    assert tool.status.description == "cancelled"
    assert tool.events[-1].name == "exception"


@pytest.mark.anyio
async def test_partial_initialization_cleans_first_destination_once() -> None:
    class CountedExporter(InMemorySpanExporter):
        shutdowns = 0

        def shutdown(self) -> None:
            self.shutdowns += 1

    exporter = CountedExporter()
    registry = Registry()
    registry.register_trace_exporter("first", lambda _: exporter)

    def failing(config: ObservabilityConfig) -> InMemorySpanExporter:
        raise RuntimeError("a sensitive diagnostic")

    registry.register_trace_exporter("broken", failing)
    with pytest.raises(ObservabilityError, match="broken"):
        async with trace_run(ObservabilityConfig(destinations=("first", "broken")), registry, flow="review"):
            pytest.fail("initialization must precede agent work")
    assert exporter.shutdowns == 1


@pytest.mark.anyio
@pytest.mark.parametrize("budget", ["wall", "tools", "supervisor"])
async def test_budget_and_supervision_close_tools_with_reason(tmp_path: Path, budget: str) -> None:
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)

    def veto(tool_name: str, tool_input: dict[str, Any], *, phase: DaydreamPhase) -> ToolDecision:
        return ToolDecision(True, "deny this tool")

    if budget == "supervisor":
        registry.register_tool_supervisor(veto)
    previous_registry = get_registry()
    set_registry(registry)
    try:
        backend = _ScriptedBackend([[ToolStartEvent("tool", "Write", {})]], stall=True)
        async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
            result = await run_agent(
                backend,
                tmp_path,
                "inspect",
                phase=DaydreamPhase.REVIEW,
                wall_budget_s=0.02 if budget == "wall" else None,
                tool_call_budget=0 if budget == "tools" else None,
            )
    finally:
        set_registry(previous_registry)
    reason = {"wall": "wall_budget_exceeded", "tools": "tool_call_budget_exceeded", "supervisor": "tool_vetoed:Write"}[
        budget
    ]
    assert result[2] == reason
    tool = next(
        span for span in exporter.get_finished_spans() if (span.attributes or {}).get("daydream.span.kind") == "tool"
    )
    assert (tool.attributes or {})["daydream.outcome"] == reason.split(":")[0]
    assert tool.status.status_code == StatusCode.ERROR
    assert tool.status.description == reason.split(":")[0]
    assert tool.events[-1].name == "exception"


@pytest.mark.anyio
async def test_concurrent_runs_and_fanout_are_isolated_from_ambient_span() -> None:
    exporters: list[InMemorySpanExporter] = []
    registry = Registry()

    def factory(config: ObservabilityConfig) -> InMemorySpanExporter:
        exporter = InMemorySpanExporter()
        exporters.append(exporter)
        return exporter

    registry.register_trace_exporter("memory", factory)

    async def child(number: int) -> None:
        with agent_scope(f"agent-{number}", backend="pi"):
            async with attempt_scope(1) as observer:
                observer.observe(TextEvent(f"answer-{number}"))
                await anyio.sleep(0)

    async def one_run() -> None:
        async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
            with step_scope("review"):
                async with anyio.create_task_group() as group:
                    group.start_soon(child, 1)
                    group.start_soon(child, 2)

    ambient = NonRecordingSpan(SpanContext(123, 456, False, TraceFlags(TraceFlags.SAMPLED)))
    with trace.use_span(ambient):
        async with anyio.create_task_group() as group:
            group.start_soon(one_run)
            group.start_soon(one_run)
    trace_ids: set[int] = set()
    for exporter in exporters:
        spans = exporter.get_finished_spans()
        assert len(spans) == 6
        roots = [span for span in spans if span.parent is None]
        assert len(roots) == 1 and roots[0].context is not None
        trace_ids.add(roots[0].context.trace_id)
        agents = [span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "agent"]
        assert agents[0].parent == agents[1].parent
        assert len({(span.attributes or {})["daydream.run.id"] for span in spans}) == 1
    assert len(trace_ids) == 2 and 123 not in trace_ids


@pytest.mark.anyio
async def test_hydration_ignores_ambient_length_limit_and_serialization_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "2")
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    text = "long source and result " * 1000
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
        async with attempt_scope(1) as observer:
            observer.observe(RequestEvent(text))
            observer.observe(TextEvent(text))
            observer.observe(ResultEvent(object(), None))
    attempt = next(
        span for span in exporter.get_finished_spans() if (span.attributes or {}).get("daydream.span.kind") == "attempt"
    )
    attrs = dict(attempt.attributes or {})
    assert text in str(attrs["gen_ai.input.messages"])
    assert text in str(attrs["gen_ai.output.messages"])
    assert json.loads(str(attrs["traceloop.entity.output"])) == "[UNSERIALIZABLE]"


@pytest.mark.anyio
async def test_failed_tool_exports_native_error_and_sanitized_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "opaque-secret")
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
        async with attempt_scope(1) as observer:
            observer.observe(ToolStartEvent("tool", "Read", {}, timestamp="2026-09-05T10:00:00Z"))
            observer.observe(
                ToolResultEvent(
                    "tool", "read failed opaque-secret", True, timestamp="2026-09-05T10:00:02Z", duration_ms=2000
                )
            )
    tool = next(
        span for span in exporter.get_finished_spans() if (span.attributes or {}).get("daydream.span.kind") == "tool"
    )
    assert tool.status.description == "error"
    assert tool.events[-1].name == "exception"
    assert (tool.events[-1].attributes or {})["exception.type"] == "ToolError"
    assert "read failed" in str((tool.events[-1].attributes or {})["exception.message"])
    assert "opaque-secret" not in str(tool.events)
    assert (tool.attributes or {})["daydream.tool.started_at"] == "2026-09-05T10:00:00Z"
    assert (tool.attributes or {})["daydream.tool.completed_at"] == "2026-09-05T10:00:02Z"
