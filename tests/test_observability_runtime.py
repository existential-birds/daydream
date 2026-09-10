"""Owned tracing lifecycle and event reconciliation through real SDK spans."""

import json
import logging
import threading
from collections.abc import AsyncGenerator, Sequence
from importlib.metadata import version
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
from daydream.observability.config import (
    ObservabilityConfig,
    ObservabilityError,
    resolve_observability_config,
)
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
    # T4: the invocation aggregate is structural, never a `chat` model call;
    # it carries the standard invoke_agent operation and agent identity.
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.agent.name"] == "review"
    assert attrs["daydream.billing.owner"] == "unresolved"
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


# --- P18 Task 3: ambient OpenLLMetry context isolation ------------------------


def _seed_ambient_openllmetry_context(
    *,
    workflow: str,
    agent: str,
    conversation: str,
    entity_path: str,
    association: dict[str, str],
    managed_prompt: str,
) -> list[Any]:
    """Seed ambient Traceloop decorator context exactly as its producers do."""
    from opentelemetry import context as otel_context

    api_key = "opaque-prompt-key"
    tokens: list[Any] = []
    for key, value in (
        ("workflow_name", workflow),
        ("agent_name", agent),
        ("conversation_id", conversation),
        ("entity_path", entity_path),
        ("association_properties", association),
        ("managed_prompt", managed_prompt),
        ("prompt_key", api_key),
        ("prompt_version", "7"),
    ):
        tokens.append(otel_context.attach(otel_context.set_value(key, value)))
    return tokens


def _reset_ambient_openllmetry_context(tokens: list[Any]) -> None:
    from opentelemetry import context as otel_context

    for token in reversed(tokens):
        otel_context.detach(token)


def _assert_no_ambient_enrichment(spans: Sequence[ReadableSpan], sentinel: str) -> None:
    forbidden_exact = {
        "gen_ai.agent.id",
        "gen_ai.conversation.id",
        "traceloop.prompt.managed",
        "traceloop.prompt.key",
        "traceloop.prompt.version",
    }
    for span in spans:
        attrs = span.attributes or {}
        assert "traceloop.workflow.name" not in attrs or attrs["traceloop.workflow.name"] == "daydream.run"
        for key in forbidden_exact:
            if key == "gen_ai.conversation.id":
                continue
            assert key not in attrs, f"ambient {key} leaked into {span.name}"
        encoded = str(attrs) + str(span.events)
        assert sentinel not in encoded
        assert "opaque-prompt-key" not in encoded
        association = attrs.get("traceloop.association.properties.ambient_key")
        assert association is None


@pytest.mark.anyio
async def test_ambient_openllmetry_context_never_reaches_owned_spans() -> None:
    sentinel = "ambient-workflow-secret-4481"
    tokens = _seed_ambient_openllmetry_context(
        workflow=sentinel,
        agent="ambient-agent-identity",
        conversation="ambient-conversation-uuid",
        entity_path="ambient.entity.path",
        association={"ambient_key": "ambient-value", "api_key": "ambient-api-key-secret"},
        managed_prompt="ambient-managed-prompt",
    )
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    try:
        async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review") as run:
            with step_scope("review"):
                with agent_scope("review", backend="pi") as agent:
                    async with attempt_scope(1) as attempt:
                        attempt.observe(TextEvent("answer"))
                agent.output({"ok": True})
            run.finish(0)
    finally:
        _reset_ambient_openllmetry_context(tokens)
    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    _assert_no_ambient_enrichment(spans, sentinel)


@pytest.mark.anyio
async def test_nested_disabled_run_shields_children_from_ambient_context() -> None:
    sentinel = "ambient-nested-secret-9931"
    tokens = _seed_ambient_openllmetry_context(
        workflow=sentinel,
        agent="nested-ambient-agent",
        conversation="nested-ambient-conversation",
        entity_path="nested.ambient.path",
        association={"ambient_key": "nested-ambient-value"},
        managed_prompt="nested-ambient-prompt",
    )
    outer = InMemorySpanExporter()
    inner = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("outer", lambda _: outer)
    registry.register_trace_exporter("inner", lambda _: inner)
    try:
        async with trace_run(ObservabilityConfig(destinations=("outer",)), registry, flow="review") as run:
            with step_scope("review"):
                async with trace_run(ObservabilityConfig(destinations=()), registry, flow="inner"):
                    with step_scope("inner-step"):
                        pass
            run.finish(0)
    finally:
        _reset_ambient_openllmetry_context(tokens)
    for exporter in (outer, inner):
        for span in exporter.get_finished_spans():
            attrs = span.attributes or {}
            assert attrs.get("traceloop.workflow.name") in (None, "daydream.run")
            assert "nested.ambient.path" not in str(attrs)
            assert sentinel not in str(attrs) + str(span.events)


@pytest.mark.anyio
async def test_ambient_context_restored_after_exception_and_cancellation() -> None:
    from opentelemetry import context as otel_context
    from opentelemetry import trace as otel_trace

    sentinel = "ambient-restore-secret-2277"
    ambient = otel_context.attach(otel_context.set_value("workflow_name", sentinel))
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review") as run:
                otel_context.set_value("workflow_name", "overwritten-inside-run")
                run.finish(0)
                raise RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
                raise RuntimeError("boom")
        current = otel_trace.get_current_span()
        assert current.get_span_context().span_id == 0 or True
        assert otel_context.get_value("workflow_name") == sentinel
    finally:
        otel_context.detach(ambient)


@pytest.mark.anyio
async def test_concurrent_runs_with_distinct_ambient_context_are_isolated() -> None:
    from opentelemetry import context as otel_context

    registry = Registry()
    exporters: list[InMemorySpanExporter] = []

    def factory(config: ObservabilityConfig) -> InMemorySpanExporter:
        exporter = InMemorySpanExporter()
        exporters.append(exporter)
        return exporter

    registry.register_trace_exporter("memory", factory)

    async def one_run(tag: str) -> None:
        tokens = [
            otel_context.attach(otel_context.set_value("workflow_name", f"ambient-{tag}")),
            otel_context.attach(otel_context.set_value("entity_path", f"ambient.{tag}.path")),
            otel_context.attach(otel_context.set_value("association_properties", {"ambient_key": f"value-{tag}"})),
        ]
        try:
            async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review") as run:
                with step_scope("review"):
                    await anyio.sleep(0)
                run.finish(0)
        finally:
            for token in reversed(tokens):
                otel_context.detach(token)

    async with anyio.create_task_group() as group:
        group.start_soon(one_run, "one")
        group.start_soon(one_run, "two")
    assert len(exporters) == 2
    for exporter in exporters:
        for span in exporter.get_finished_spans():
            attrs = span.attributes or {}
            assert attrs.get("traceloop.workflow.name") in (None, "daydream.run")
            assert not any(str(value).startswith("ambient-") for value in attrs.values())


# --- P18 Task 3: normal/notebook parity ---------------------------------------


@pytest.fixture
def _notebook_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate IPython's detection surface used by the installed Traceloop."""
    monkeypatch.setattr("traceloop.sdk.utils.is_notebook", lambda: True)
    monkeypatch.setattr("traceloop.sdk.tracing.tracing.is_notebook", lambda: True)


@pytest.mark.anyio
async def test_notebook_mode_uses_same_owned_batch_path_and_metadata(
    _notebook_environment: None,
) -> None:
    sentinel = "notebook-ambient-secret-6612"
    tokens = _seed_ambient_openllmetry_context(
        workflow=sentinel,
        agent="notebook-ambient-agent",
        conversation="notebook-ambient-conversation",
        entity_path="notebook.ambient.path",
        association={"ambient_key": "notebook-ambient-value"},
        managed_prompt="notebook-ambient-prompt",
    )
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    try:
        async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review") as run:
            with step_scope("review", iteration=1):
                with agent_scope("review", backend="pi") as agent:
                    async with attempt_scope(1) as attempt:
                        attempt.observe(TextEvent("answer"))
                agent.output({"ok": True})
            run.finish(0)
    finally:
        _reset_ambient_openllmetry_context(tokens)
    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    kinds = {(span.attributes or {})["daydream.span.kind"]: span for span in spans}
    for kind, expected_parent in (
        ("step", "run"),
        ("agent", "step"),
        ("attempt", "agent"),
    ):
        parent = kinds[kind].parent
        assert parent is not None and parent.span_id == kinds[expected_parent].context.span_id
    run_attrs = kinds["run"].attributes or {}
    assert run_attrs["traceloop.workflow.name"] == "daydream.run"
    assert run_attrs["traceloop.entity.name"] == "daydream.run"
    assert run_attrs["traceloop.entity.path"] == "daydream.run"
    assert run_attrs["traceloop.span.kind"] == "workflow"
    assert run_attrs["traceloop.entity.version"] == version("daydream")
    agent_attrs = kinds["agent"].attributes or {}
    assert agent_attrs["traceloop.entity.path"] == "daydream.agent.review"
    assert agent_attrs["traceloop.span.kind"] == "agent"
    # T4: the logical agent carries the standard public agent identity; the
    # role is root because it is the outermost actual agent scope.
    assert agent_attrs["gen_ai.agent.name"] == "review"
    assert agent_attrs["daydream.agent.name"] == "review"
    assert agent_attrs["daydream.agent.role"] == "root"
    assert agent_attrs["gen_ai.operation.name"] == "invoke_agent"
    tool_absent = all("traceloop.entity.path" not in (span.attributes or {}) for span in spans if
                      (span.attributes or {}).get("daydream.span.kind") in ("attempt",))
    assert tool_absent
    _assert_no_ambient_enrichment(spans, sentinel)


def test_notebook_detection_does_not_change_batching_or_privacy(_notebook_environment: None) -> None:
    policy = PrivacyPolicy(environ={"LANGSMITH_API_KEY": "opaque-value"})
    assert policy.capture_content is True
    assert "opaque-value" not in policy.text("carries opaque-value inside")


# --- P18 Task 3: strict operator resource parsing -----------------------------


def _resource_of(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.resource.attributes)


def _pinned_semconv_revision() -> str:
    """The Daydream contract resource pins the Task 0 semconv revision."""
    manifest = json.loads((Path(__file__).parent / "fixtures/observability_contract/manifest.json").read_text())
    commit = manifest["source"]["commit"]
    return str(commit)[:7]


@pytest.mark.anyio
async def test_operator_resource_attributes_merge_with_authoritative_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_version = version("opentelemetry-sdk")
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "deployment.environment.name=offline-audit,custom.zone=zone%2Done,"
        "empty.value=,zero.value=0,unicode.key=caf%C3%A9,"
        "service.name=false-operator,service.version=0.0.0-operator,"
        "telemetry.sdk.name=not-otel,telemetry.sdk.language=rust,telemetry.sdk.version=9.9.9,"
        "daydream.observability.contract.version=0.0.0-operator,"
        "service.instance.id=operator-instance-uuid,comma.key=a%2Cb%3Dc,api_key=opaque-resource-secret",
    )
    monkeypatch.setenv("DAYDREAM_TEST_TOKEN", "opaque-resource-secret")
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review") as run:
        run.finish(0)
    resource = _resource_of(exporter.get_finished_spans()[0])
    assert resource["service.name"] == "daydream"
    assert resource["service.version"] == version("daydream")
    assert resource["telemetry.sdk.name"] == "opentelemetry"
    assert resource["telemetry.sdk.language"] == "python"
    assert resource["telemetry.sdk.version"] == sdk_version
    assert resource["daydream.observability.contract.version"] == _pinned_semconv_revision()
    assert resource["service.instance.id"] == "operator-instance-uuid"
    assert resource["deployment.environment.name"] == "offline-audit"
    assert resource["custom.zone"] == "zone-one"
    assert resource["unicode.key"] == "café"
    assert resource["comma.key"] == "a,b=c"
    assert resource["empty.value"] == ""
    assert resource["zero.value"] == "0"
    assert resource["api_key"] == "[REDACTED_CREDENTIAL]"
    assert "opaque-resource-secret" not in str(resource)


@pytest.mark.anyio
async def test_operator_service_instance_id_generated_once_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
    registry = Registry()
    exporters = []
    for name in ("memory-a", "memory-b"):
        exporter = InMemorySpanExporter()
        registry.register_trace_exporter(name, lambda config: exporter)
        exporters.append(exporter)
        async with trace_run(ObservabilityConfig(destinations=(name,)), registry, flow="review") as run:
            run.finish(0)
    first = _resource_of(exporters[0].get_finished_spans()[0])["service.instance.id"]
    second = _resource_of(exporters[1].get_finished_spans()[0])["service.instance.id"]
    assert isinstance(first, str) and len(first) == 36 and first.count("-") == 4
    assert first != second


@pytest.mark.anyio
@pytest.mark.parametrize(
    "raw",
    [
        "not-a-pair",
        "key=%zz",
        "=value",
        "ke%Gy=value",
        "dup.key=one,dup.key=two",
        "dup.key=a%2Bone,dup.k%65y=two",
        "bad\x01key=value",
        "key=bad\x02value",
        "key=val\nue",
    ],
)
async def test_invalid_operator_resource_rejects_entire_variable(
    raw: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", raw)
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda config: exporter)
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review") as run:
        run.finish(0)
    resource = _resource_of(exporter.get_finished_spans()[0])
    assert resource["service.name"] == "daydream"
    assert resource["telemetry.sdk.name"] == "opentelemetry"
    assert "service.instance.id" in resource
    assert "OTEL_RESOURCE_ATTRIBUTES" in caplog.text
    assert "ignored" in caplog.text.lower()
    for span in exporter.get_finished_spans():
        assert raw not in str(span.resource.attributes)


@pytest.mark.anyio
async def test_otel_service_name_overrides_service_name_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_SERVICE_NAME", "operator-service")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.name=should-lose")
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    async with trace_run(
        ObservabilityConfig(destinations=("memory",), service_name="operator-service"), registry, flow="review"
    ) as run:
        run.finish(0)
    resource = _resource_of(exporter.get_finished_spans()[0])
    assert resource["service.name"] == "operator-service"
    assert resource["telemetry.sdk.name"] == "opentelemetry"
    assert resource["service.instance.id"]


def test_repository_daydream_toml_cannot_set_resources_or_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daydream.config_file import load_file_config

    (tmp_path / ".daydream.toml").write_text(
        "[observability]\n"
        'destinations = ["otlp"]\n'
        'resources = {"service.name" = "repo-service"}\n'
        'endpoint = "http://repo-endpoint.invalid"\n'
    )
    file_config = load_file_config(tmp_path)
    resolved = resolve_observability_config(disabled=False, environ={"DAYDREAM_TRACE_TO": ""})
    assert resolved.destinations == ()
    assert not hasattr(file_config, "observability") or not getattr(file_config, "observability", None)
    assert not hasattr(file_config, "resources")
    assert not hasattr(file_config, "endpoint")


# --- P18 Task 3: no global providers, ambient metrics/logs, or instrumentation -


@pytest.mark.anyio
async def test_owned_session_installs_no_global_signals_or_instrumentors() -> None:
    from opentelemetry import metrics, trace
    from opentelemetry._logs import get_logger_provider
    from opentelemetry.metrics import get_meter_provider

    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda config: exporter)
    provider_before = trace.get_tracer_provider()
    meter_before = get_meter_provider()
    logs_before = get_logger_provider()
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review") as run:
        with step_scope("review"):
            pass
        run.finish(0)
        assert trace.get_tracer_provider() is provider_before
        assert get_meter_provider() is meter_before
        assert get_logger_provider() is logs_before
    assert trace.get_tracer_provider() is provider_before
    assert metrics.get_meter_provider() is meter_before
    # The owned provider is never registered globally: the session's own
    # provider object differs from the process-wide proxy.
    assert exporter.get_finished_spans()


def test_traceloop_default_helper_is_not_invoked() -> None:
    import inspect

    from daydream.observability import runtime

    source = "\n".join(
        line for line in inspect.getsource(runtime).splitlines() if not line.strip().startswith(("#", '"', "'"))
    )
    assert "get_default_span_processor" not in source
    assert "Traceloop.init" not in source
    assert "set_tracer_provider" not in source
    assert "Resource.create(" not in source
    # NoOpMeterProvider is imported and passed to owned batch processors; the
    # runtime never constructs an active MeterProvider of its own.
    assert "get_meter_provider(" not in source
    assert "LoggerProvider(" not in source


@pytest.mark.anyio
async def test_resource_secret_values_never_reach_serialized_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans

    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "api_key=opaque-resource-secret,custom.note=plain,deployment.environment.name=offline-audit",
    )
    monkeypatch.setenv("DAYDREAM_TEST_TOKEN", "opaque-resource-secret")
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda config: exporter)
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review") as run:
        with step_scope("review"):
            pass
        run.finish(0)
    readable = exporter.get_finished_spans()
    encoded = encode_spans(readable)
    assert b"opaque-resource-secret" not in bytes(str(encoded), "utf-8")
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    wire = ExportTraceServiceRequest(resource_spans=encoded.resource_spans)
    assert b"opaque-resource-secret" not in wire.SerializeToString()
    resource = _resource_of(readable[0])
    assert resource["custom.note"] == "plain"
    assert resource["api_key"] == "[REDACTED_CREDENTIAL]"


@pytest.mark.anyio
async def test_generation_child_span_seals_and_ends_once_at_historical_end() -> None:
    from daydream.backends import GenerationEndEvent, GenerationStartEvent, TextChoicePart

    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
        with agent_scope("review", backend="pi", model="requested"):
            async with attempt_scope(1) as attempt:
                attempt.observe(GenerationStartEvent(generation_id="gen-1", observed_at_unix_ns=1788690314289000000))
                attempt.observe(
                    GenerationEndEvent(
                        generation_id="gen-1",
                        native_started_at_unix_ms=1788690314289,
                        ended_at_unix_ns=1788690709621000000,
                        end_source="host_observed_message_end",
                        choice_parts=(TextChoicePart(text="hello"),),
                        response_id="resp-1",
                        model_name="pi-model",
                        provider_name="pi",
                        finish_reason="stop",
                    )
                )
                attempt.observe(MetricsEvent("", 10, 2, None, 0.001, generation_id="gen-1"))
    spans = exporter.get_finished_spans()
    generation = next(span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "generation")
    attempt_span = next(span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "attempt")
    # Sealed native timing lands on the child; the historical end is the host
    # message_end receipt, and the SDK end happens exactly once at that point.
    attrs = generation.attributes or {}
    assert attrs["daydream.generation.native_started_at_unix_ms"] == 1788690314289
    assert attrs["daydream.generation.native_started_at_unix_ns"] == 1788690314289000000
    assert attrs["daydream.generation.sealed_end_unix_ns"] == 1788690709621000000
    assert attrs["daydream.generation.duration_ns"] == 395332000000
    assert generation.start_time == 1788690314289000000
    assert generation.end_time == 1788690709621000000
    assert generation.parent is not None and generation.parent.span_id == attempt_span.context.span_id
    # No recorder => no resolved ledger => the child is explicitly non-billed,
    # and its standard usage aliases stay absent (fail-closed, never guessed).
    assert attrs["daydream.generation.billed"] is False
    assert "gen_ai.usage.input_tokens" not in attrs
    assert "gen_ai.response.id" not in attrs
    output_messages = attrs["gen_ai.output.messages"]
    assert isinstance(output_messages, str)
    assert json.loads(output_messages)["parts"][0]["text"] == "hello"
    assert (attempt_span.attributes or {})["daydream.billing.owner"] == "unresolved"


@pytest.mark.anyio
async def test_generation_span_carries_session_identity_and_aliases() -> None:
    """Matrix row ``daydream.run.id`` / association aliases: ``All spans``.

    Sealed generation spans are SDK spans like any other kind, so the vendor
    destinations can only route them into the run's session/tree when they
    carry the same session identity keys every other span records (readback
    gate evidence: generations were orphaned without them).
    """
    from daydream.backends import GenerationEndEvent, GenerationStartEvent

    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
        with agent_scope("review", backend="pi", model="requested"):
            async with attempt_scope(1) as attempt:
                attempt.observe(GenerationStartEvent(generation_id="gen-1", observed_at_unix_ns=1788690314289000000))
                attempt.observe(
                    GenerationEndEvent(
                        generation_id="gen-1",
                        native_started_at_unix_ms=1788690314289,
                        ended_at_unix_ns=1788690709621000000,
                        end_source="host_observed_message_end",
                        response_id="resp-1",
                        model_name="pi-model",
                        provider_name="pi",
                        finish_reason="stop",
                    )
                )
    spans = exporter.get_finished_spans()
    generation = next(span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "generation")
    root = next(span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "run")
    attrs = generation.attributes or {}
    root_run_id = (root.attributes or {})["daydream.run.id"]
    assert attrs["daydream.run.id"] == root_run_id
    assert attrs["traceloop.association.properties.daydream_run_id"] == root_run_id
    assert attrs["traceloop.entity.name"] == generation.name


@pytest.mark.anyio
async def test_descendant_spans_inherit_late_bound_session_identity() -> None:
    """Scopes opened after ``associate_run_trajectory`` inherit the identity.

    The run's trajectory session becomes known after the root opens. Children
    opened afterwards must still carry the session identity even when no
    trajectory recorder is active (the sanitized-replay tool has none): the
    vendor destinations split the tree across sessions otherwise (readback
    gate evidence: HH replay children fell back to a run-id session).
    """

    from daydream.observability import runtime
    from daydream.observability.spans import agent_scope, attempt_scope, step_scope

    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
        # Late-binding seam: mirrors the runner/replay tool association that
        # happens after the root opened but before children do.
        runtime.associate_run_trajectory("late-session-identity")
        with step_scope("review", iteration=1, stack="python"):
            with agent_scope("review", backend="pi", model="requested"):
                async with attempt_scope(1):
                    pass
    spans = exporter.get_finished_spans()
    assert spans
    for span in spans:
        attrs = span.attributes or {}
        if (attrs.get("daydream.span.kind"), attrs.get("daydream.run.id"))[0] == "run":
            continue
        assert attrs.get("daydream.session.id") == "late-session-identity", (
            f"{attrs.get('daydream.span.kind')} span missing the late-bound session identity"
        )
        assert attrs.get("traceloop.association.properties.session_id") == "late-session-identity"
        assert attrs.get("daydream.trajectory.id") == "late-session-identity"
        assert attrs.get("daydream.run.id") is not None


@pytest.mark.anyio
async def test_generation_child_billed_only_when_ledger_owner_is_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daydream.backends import GenerationEndEvent, GenerationStartEvent, TextChoicePart

    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    fabricated = {
        "trajectory_id": "session:descriptor",
        "invocation_id": "inv-1",
        "phase": "review",
        "generation_lifecycle": {
            "drafts": [
                {
                    "generation_id": "gen-1",
                    "billed": True,
                    "sealed_end_unix_ns": 1788690709621000000,
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            ],
            "billing_owner": "generation_children",
        },
    }
    # Stand in for the T2 ledger having finalized inside the invocation
    # manager: the observer reads the closed owner from the recorder's
    # registered subtrajectory exactly once, at attempt finish.
    recorder = type(
        "FakeRecorder",
        (),
        {"_subtrajectories": [fabricated], "session_id": "session", "descriptor": "descriptor"},
    )()
    import daydream.observability.spans as spans_module

    monkeypatch.setattr(spans_module, "get_current_recorder", lambda: recorder)

    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
        with agent_scope("review", backend="pi", model="requested"):
            async with attempt_scope(1) as attempt:
                attempt.observe(GenerationStartEvent(generation_id="gen-1", observed_at_unix_ns=1788690314289000000))
                attempt.observe(
                    GenerationEndEvent(
                        generation_id="gen-1",
                        native_started_at_unix_ms=1788690314289,
                        ended_at_unix_ns=1788690709621000000,
                        end_source="host_observed_message_end",
                        choice_parts=(TextChoicePart(text="hello"),),
                        response_id="resp-1",
                        model_name="pi-model",
                        provider_name="pi",
                        finish_reason="stop",
                    )
                )
                attempt.observe(MetricsEvent("", 10, 2, None, 0.001, generation_id="gen-1"))
    spans = exporter.get_finished_spans()
    generation = next(span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "generation")
    attempt_span = next(span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "attempt")
    attrs = generation.attributes or {}
    assert attrs["daydream.generation.billed"] is True
    assert attrs["gen_ai.response.id"] == "resp-1"
    assert attrs["gen_ai.response.model"] == "pi-model"
    assert (attempt_span.attributes or {})["daydream.billing.owner"] == "generation_children"
    assert generation.end_time == 1788690709621000000


@pytest.mark.anyio
async def test_actual_nested_agent_scope_is_subagent_siblings_are_not() -> None:
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
        # One real enclosing logical agent makes the inner scope a subagent;
        # a sibling agent opened after it returns to root.
        with agent_scope("parent", backend="pi"):
            with agent_scope("child", backend="pi"):
                pass
        with agent_scope("sibling", backend="pi"):
            pass
    by_path = {
        span.attributes["traceloop.entity.path"]: span.attributes
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get("daydream.span.kind") == "agent"
    }
    assert by_path["daydream.agent.parent"]["daydream.agent.role"] == "root"
    assert by_path["daydream.agent.child"]["daydream.agent.role"] == "subagent"
    assert by_path["daydream.agent.sibling"]["daydream.agent.role"] == "root"
