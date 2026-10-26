"""Owned tracing lifecycle and event reconciliation through real SDK spans."""

import json
import logging
import threading
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import anyio
import pytest
from opentelemetry import context as otel_context, metrics, trace, trace as otel_trace
from opentelemetry._logs import get_logger_provider
from opentelemetry.metrics import get_meter_provider
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, StatusCode, TraceFlags

from daydream import clock
from daydream.agent import run_agent
from daydream.backends import (
    CostEvent,
    DiagnosticEvent,
    GenerationEndEvent,
    GenerationStartEvent,
    MetricsEvent,
    RequestEvent,
    ResultEvent,
    TextChoicePart,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.lifecycle import BackendLifecycleCollector
from daydream.extensions.registry import Registry
from daydream.observability import runtime
from daydream.observability.config import (
    ObservabilityConfig,
    ObservabilityError,
    resolve_observability_config,
)
from daydream.observability.privacy import PrivacyPolicy, diagnostic_scope
from daydream.observability.runtime import trace_run
from daydream.observability.spans import agent_scope, attempt_scope, stack_scope, step_scope
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _memory_tracing() -> tuple[InMemorySpanExporter, Registry]:
    """An in-memory span exporter registered as the active ``memory`` destination."""
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    return exporter, registry


def _observe_generation(attempt: Any, *, text: bool = False, metrics: bool = False) -> None:
    """Emit the canonical generation start/end pair shared by the generation tests."""
    attempt.observe(GenerationStartEvent(generation_id="gen-1", observed_at_unix_ns=1788690314289000000))
    attempt.observe(GenerationEndEvent(
            generation_id="gen-1", native_started_at_unix_ms=1788690314289, ended_at_unix_ns=1788690709621000000,
            end_source="host_observed_message_end", choice_parts=(TextChoicePart(text="hello"),) if text else (),
            response_id="resp-1", model_name="pi-model", provider_name="pi", finish_reason="stop",
        )
    )
    if metrics:
        attempt.observe(MetricsEvent("", 10, 2, None, 0.001, generation_id="gen-1"))

@pytest.mark.anyio
async def test_owned_span_tree_usage_and_content() -> None:
    exporter, registry = _memory_tracing()
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
    # Invocation aggregates are structural invoke_agent spans, never chat model calls.
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
        "daydream.flow": "review", "daydream.step": "review", "daydream.phase": "review", "daydream.iteration": 2,
        "daydream.stack": "python", "daydream.backend": "pi", "daydream.attempt": 1,
    }.items():
        assert tool_attrs[key] == value

@pytest.mark.anyio
async def test_attempt_and_nested_step_do_not_inherit_unobserved_request_metadata() -> None:
    exporter, registry = _memory_tracing()
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
async def test_attempt_records_only_scrubbed_diagnostic_codes_and_counts(monkeypatch: pytest.MonkeyPatch,) -> None:
    secret = "opaque-diagnostic-secret"
    monkeypatch.setenv("DAYDREAM_TEST_TOKEN", secret)
    exporter, registry = _memory_tracing()

    async with trace_run(ObservabilityConfig(destinations=("memory",), capture_content=True), registry, flow="review",
    ):
        with agent_scope("review", backend="codex"):
            async with attempt_scope(1) as attempt:
                attempt.observe(DiagnosticEvent(code=f"parser_{secret}", message=f"do not export {secret}",
                        metadata={"api_key": secret, "count": 99},
                    )
                )
                attempt.observe(DiagnosticEvent(
                        code=f"parser_{secret}", message="another message", metadata={"raw": "not observable"},
                    )
                )
                attempt.observe(DiagnosticEvent(
                        code="codex_transport_coverage", message="transport detail", metadata={"occurrences": 4},
                    )
                )

    spans = exporter.get_finished_spans()
    assert sorted(str((span.attributes or {})["daydream.span.kind"]) for span in spans) == ["agent", "attempt", "run"]
    attempt_span = next(span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "attempt")
    attrs = dict(attempt_span.attributes or {})
    assert attrs["daydream.backend_diagnostic.codes"] == ("parser_[REDACTED_CREDENTIAL]", "codex_transport_coverage",)
    assert attrs["daydream.backend_diagnostic.counts"] == (2, 1)
    encoded = str(spans)
    assert secret not in encoded
    assert "do not export" not in encoded
    assert "not observable" not in encoded
    for span in spans:
        if span is not attempt_span:
            assert not any(key.startswith("daydream.backend_diagnostic") for key in (span.attributes or {}))


@pytest.mark.anyio
@pytest.mark.parametrize("response_idle_timeout_s", [600.0, 0.0])
async def test_interrupted_generation_lifecycle_exports_only_attempt_evidence(
    monkeypatch: pytest.MonkeyPatch, response_idle_timeout_s: float,
) -> None:
    """Unsealed native progress must not become a model completion or billing evidence."""
    monkeypatch.setenv("DAYDREAM_TEST_TOKEN", "private-lifecycle-secret")
    monkeypatch.setattr(clock, "monotonic", lambda: 100.0)
    lifecycle = BackendLifecycleCollector(attempt=1, started_at=100.0, wall_allowance_s=0.1, deadline=100.1)
    lifecycle.activate(response_idle_timeout_s=response_idle_timeout_s, tool_idle_timeout_s=2700.0)
    lifecycle.spawn_pending()
    lifecycle.spawn_completed()
    lifecycle.native_event()
    lifecycle.generation_started()
    lifecycle.progress("thinking")
    monkeypatch.setattr(clock, "monotonic", lambda: 100.1)
    snapshot = lifecycle.finalize()
    assert snapshot is not None

    exporter, registry = _memory_tracing()
    async with trace_run(
        ObservabilityConfig(destinations=("memory",), capture_content=False), registry, flow="review",
    ):
        with agent_scope("review", backend="pi"):
            async with attempt_scope(1) as attempt:
                attempt.observe(RequestEvent("PRIVATE PROMPT private-lifecycle-secret"))
                attempt.observe(GenerationStartEvent(generation_id="private-generation", observed_at_unix_ns=123))
                attempt.observe(DiagnosticEvent(
                    code="pi_protocol", message="PRIVATE diagnostic",
                    metadata={"delta": "PRIVATE reasoning", "token": "private-lifecycle-secret"},
                ))
                attempt.record_backend_lifecycle(snapshot)
                attempt.abort("wall_budget_exceeded")

    spans = exporter.get_finished_spans()
    assert {span.attributes["daydream.span.kind"] for span in spans if span.attributes} == {"run", "agent", "attempt"}
    attempt_span = next(span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "attempt")
    attrs = dict(attempt_span.attributes or {})
    assert attrs["daydream.outcome"] == "wall_budget_exceeded"
    assert attrs["daydream.backend_lifecycle.state"] == "generating"
    assert attrs["daydream.backend_lifecycle.clock"] == "monotonic_elapsed"
    assert attrs["daydream.backend_lifecycle.timestamp_source"] == "host_observed"
    assert attrs["daydream.backend_lifecycle.response_idle_timeout_s"] == response_idle_timeout_s
    assert attrs["daydream.backend_lifecycle.tool_idle_timeout_s"] == 2700.0
    assert attrs["daydream.backend_lifecycle.idle_detection_enabled"] is bool(response_idle_timeout_s)
    assert attrs["daydream.backend_lifecycle.wall_allowance_s"] == 0.1
    assert attrs["daydream.backend_lifecycle.remaining_wall_s"] == 0.0
    assert attrs["daydream.backend_lifecycle.generation_open"] is True
    assert attrs["daydream.backend_lifecycle.generation_count"] == 1
    assert attrs["daydream.backend_lifecycle.thinking_delta_count"] == 1
    assert attrs["daydream.backend_lifecycle.last_progress_age_s"] == pytest.approx(0.1)
    assert "daydream.backend_lifecycle.first_raw_line_s" not in attrs
    assert attrs["daydream.billing.owner"] == "unresolved"
    assert not any(key.startswith("gen_ai.usage.") for key in attrs)
    assert "private-lifecycle-secret" not in str(spans)
    assert "PRIVATE" not in str(spans)
    assert "private-generation" not in str(spans)
    for span in spans:
        if span is not attempt_span:
            assert not any(key.startswith("daydream.backend_lifecycle.") for key in (span.attributes or {}))

def test_diagnostics_scrub_formatted_arguments_and_exception(caplog: pytest.LogCaptureFixture) -> None:
    policy = PrivacyPolicy(environ={"LANGSMITH_API_KEY": "opaque-value"})
    with caplog.at_level(logging.WARNING), diagnostic_scope(policy):
        try:
            raise RuntimeError("opaque-value")
        except RuntimeError:
            logging.getLogger("opentelemetry.exporter.test").warning("failed %s", "opaque-value", exc_info=True)
    assert "opaque-value" not in caplog.text
    assert "REDACTED" in caplog.text


@pytest.mark.anyio
async def test_run_spans_survive_flag_valued_secret_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real-path: trace_run builds its policy from the ambient environment, so a
    flag-valued secret-named var must not corrupt the agent span's JSON output."""
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
    exporter, registry = _memory_tracing()
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
async def test_shutdown_deadline_is_total_and_shutdown_once(stall_during: str, caplog: pytest.LogCaptureFixture,
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
            ObservabilityConfig(destinations=("first", "second")), registry, flow="review", cleanup_timeout_s=0.03,
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


@pytest.mark.anyio
async def test_retry_keeps_failed_billing_and_agent_records_salvaged_result(tmp_path: Path) -> None:
    class RetryableError(RuntimeError):
        retryable = True

    exporter, registry = _memory_tracing()
    backend = ScriptedBackend(script=[
            [TextEvent("failed content"), CostEvent(0.05, 4, 2), ResultEvent(None, None), RetryableError("retry")],
            [RequestEvent("effective prompt", system_prompt="effective system"), TextEvent('{"ok":true}'),
                ResultEvent({"wrong": True}, None), MetricsEvent("", 30, 6, 4, 0.2, usage_scope="invocation"),
                CostEvent(None, 0, None),
            ],
        ], model="requested", retry_attempts=1, retry_base_delay_s=0, retry_max_delay_s=0,
    )
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
        result = await run_agent(backend, tmp_path, "original", phase=DaydreamPhase.REVIEW,
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
async def test_metadata_policy_omits_all_content_and_exception_strings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "opaque-secret")
    exporter, registry = _memory_tracing()
    backend = ScriptedBackend(script=[[
                RequestEvent("PRIVATE PROMPT", system_prompt="PRIVATE SYSTEM"), TextEvent("PRIVATE TEXT"),
                ThinkingEvent("PRIVATE REASON"), ToolStartEvent("tool", "Read", {"arg": "PRIVATE INPUT"}),
                ToolResultEvent("tool", "PRIVATE OUTPUT", True), CostEvent(0.1, 2, 3),
                RuntimeError("PRIVATE ERROR opaque-secret"),
            ]
        ]
    )
    with pytest.raises(RuntimeError, match="PRIVATE ERROR"):
        async with trace_run(
            ObservabilityConfig(destinations=("memory",), capture_content=False, service_name="service opaque-secret"),
            registry, flow="review",
        ):
            await run_agent(backend, tmp_path, "PRIVATE ORIGINAL", phase=DaydreamPhase.REVIEW)
    for span in exporter.get_finished_spans():
        encoded = str(span.attributes) + str(span.events) + str(span.resource.attributes) + str(span.status)
        assert "PRIVATE" not in encoded and "opaque-secret" not in encoded
        assert not any(key.startswith(("traceloop.entity.input", "traceloop.entity.output", "gen_ai.input.messages",
                    "gen_ai.output.messages", "daydream.error.message",
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
async def test_hydration_ignores_ambient_length_limit_and_serialization_is_fail_closed(monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "2")
    exporter, registry = _memory_tracing()
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


# --- P18 Task 3: ambient OpenLLMetry context isolation ------------------------


def _seed_ambient_openllmetry_context(
    *, workflow: str, agent: str, conversation: str, entity_path: str, association: dict[str, str], managed_prompt: str,
) -> list[Any]:
    """Seed ambient Traceloop decorator context exactly as its producers do."""

    api_key = "opaque-prompt-key"
    tokens: list[Any] = []
    for key, value in (("workflow_name", workflow), ("agent_name", agent), ("conversation_id", conversation),
        ("entity_path", entity_path), ("association_properties", association), ("managed_prompt", managed_prompt),
        ("prompt_key", api_key), ("prompt_version", "7"),
    ):
        tokens.append(otel_context.attach(otel_context.set_value(key, value)))
    return tokens


def _reset_ambient_openllmetry_context(tokens: list[Any]) -> None:

    for token in reversed(tokens):
        otel_context.detach(token)


def _assert_no_ambient_enrichment(spans: Sequence[ReadableSpan], sentinel: str) -> None:
    forbidden_exact = {"gen_ai.agent.id", "gen_ai.conversation.id", "traceloop.prompt.managed", "traceloop.prompt.key",
        "traceloop.prompt.version",
    }
    for span in spans:
        attrs = span.attributes or {}
        assert "traceloop.workflow.name" not in attrs or attrs["traceloop.workflow.name"] == "daydream.run"
        for key in forbidden_exact:
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
        workflow=sentinel, agent="ambient-agent-identity", conversation="ambient-conversation-uuid",
        entity_path="ambient.entity.path",
        association={"ambient_key": "ambient-value", "api_key": "ambient-api-key-secret"},
        managed_prompt="ambient-managed-prompt",
    )
    exporter, registry = _memory_tracing()
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
        workflow=sentinel, agent="nested-ambient-agent", conversation="nested-ambient-conversation",
        entity_path="nested.ambient.path", association={"ambient_key": "nested-ambient-value"},
        managed_prompt="nested-ambient-prompt",
    )
    outer = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("outer", lambda _: outer)
    try:
        async with trace_run(ObservabilityConfig(destinations=("outer",)), registry, flow="review") as run:
            with step_scope("review"):
                async with trace_run(ObservabilityConfig(destinations=()), registry, flow="inner"):
                    with step_scope("inner-step"):
                        with agent_scope("inner-agent", backend="pi"):
                            async with attempt_scope(1) as attempt:
                                _observe_generation(attempt, text=True)
                with agent_scope("resumed", backend="pi"):
                    async with attempt_scope(1) as attempt:
                        attempt.observe(TextEvent("outer answer"))
            run.finish(0)
    finally:
        _reset_ambient_openllmetry_context(tokens)
    spans = outer.get_finished_spans()
    assert sorted(str((span.attributes or {})["daydream.span.kind"]) for span in spans) == [
        "agent", "attempt", "run", "step",
    ]
    attempt_span = next(span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "attempt")
    assert "outer answer" in str((attempt_span.attributes or {})["gen_ai.output.messages"])
    assert "hello" not in str((attempt_span.attributes or {})["gen_ai.output.messages"])
    _assert_no_ambient_enrichment(spans, sentinel)

@pytest.mark.anyio
@pytest.mark.parametrize("interruption", ["exception", "cancellation"])
async def test_ambient_context_restored_after_exception_and_cancellation(interruption: str) -> None:

    sentinel = "ambient-restore-secret-2277"
    ambient = otel_context.attach(otel_context.set_value("workflow_name", sentinel))
    exporter, registry = _memory_tracing()
    try:
        if interruption == "exception":
            with pytest.raises(RuntimeError, match="boom"):
                async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
                    raise RuntimeError("boom")
        else:
            with anyio.CancelScope() as cancellation:
                async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
                    cancellation.cancel()
                    await anyio.sleep_forever()
            assert cancellation.cancelled_caught
        current = otel_trace.get_current_span()
        # Failed runs must detach, restoring INVALID rather than leaving a stale run span.
        assert current.get_span_context().span_id == 0
        assert not current.is_recording()
        assert otel_context.get_value("workflow_name") == sentinel
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.ERROR
        assert (spans[0].attributes or {})["daydream.outcome"] == (
            "error" if interruption == "exception" else "cancelled"
        )
    finally:
        otel_context.detach(ambient)

@pytest.mark.anyio
async def test_concurrent_runs_with_distinct_ambient_context_are_isolated() -> None:

    registry = Registry()
    exporters: list[InMemorySpanExporter] = []

    def factory(config: ObservabilityConfig) -> InMemorySpanExporter:
        exporter = InMemorySpanExporter()
        exporters.append(exporter)
        return exporter

    registry.register_trace_exporter("memory", factory)

    async def one_run(tag: str) -> None:
        tokens = [otel_context.attach(otel_context.set_value("workflow_name", f"ambient-{tag}")),
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
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        _assert_no_ambient_enrichment(spans, "ambient-one")
        _assert_no_ambient_enrichment(spans, "ambient-two")
        for span in spans:
            assert "ambient.one.path" not in str(span.attributes)
            assert "ambient.two.path" not in str(span.attributes)


# --- P18 Task 3: normal/notebook parity ---------------------------------------


@pytest.fixture
def _notebook_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate IPython's detection surface used by the installed Traceloop."""
    monkeypatch.setattr("traceloop.sdk.utils.is_notebook", lambda: True)
    monkeypatch.setattr("traceloop.sdk.tracing.tracing.is_notebook", lambda: True)

@pytest.mark.anyio
async def test_notebook_mode_uses_same_owned_batch_path_and_metadata(_notebook_environment: None,) -> None:
    sentinel = "notebook-ambient-secret-6612"
    tokens = _seed_ambient_openllmetry_context(
        workflow=sentinel, agent="notebook-ambient-agent", conversation="notebook-ambient-conversation",
        entity_path="notebook.ambient.path", association={"ambient_key": "notebook-ambient-value"},
        managed_prompt="notebook-ambient-prompt",
    )
    exporter, registry = _memory_tracing()
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
    for kind, expected_parent in (("step", "run"), ("agent", "step"), ("attempt", "agent"),):
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
    # The outermost logical agent owns standard identity and root role.
    assert agent_attrs["gen_ai.agent.name"] == "review"
    assert agent_attrs["daydream.agent.name"] == "review"
    assert agent_attrs["daydream.agent.role"] == "root"
    assert agent_attrs["gen_ai.operation.name"] == "invoke_agent"
    tool_absent = all("traceloop.entity.path" not in (span.attributes or {})
        for span in spans
        if (span.attributes or {}).get("daydream.span.kind") in ("attempt",)
    )
    assert tool_absent
    _assert_no_ambient_enrichment(spans, sentinel)


# --- P18 Task 3: strict operator resource parsing -----------------------------


def _resource_of(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.resource.attributes)


@pytest.mark.anyio
async def test_operator_service_instance_id_generated_per_run_when_absent(monkeypatch: pytest.MonkeyPatch,) -> None:
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
@pytest.mark.parametrize("raw", [
    'key=%', 'key=%0', 'key=%FF',
    'not-a-pair', '=value', 'ke%Gy=value', 'dup.key=a%2Bone,dup.k%65y=two', 'bad\x01key=value', 'key=bad\x02value',
])
async def test_invalid_operator_resource_rejects_entire_variable(
    raw: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", f"valid.sibling=discard-me,{raw}")
    exporter, registry = _memory_tracing()
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review") as run:
        run.finish(0)
    resource = _resource_of(exporter.get_finished_spans()[0])
    assert resource["service.name"] == "daydream"
    assert resource["telemetry.sdk.name"] == "opentelemetry"
    assert "service.instance.id" in resource
    assert "valid.sibling" not in resource
    assert "dup.key" not in resource
    assert "OTEL_RESOURCE_ATTRIBUTES" in caplog.text
    assert "ignored" in caplog.text.lower()
    for span in exporter.get_finished_spans():
        assert raw not in str(span.resource.attributes)

@pytest.mark.anyio
async def test_otel_service_name_overrides_service_name_resource(monkeypatch: pytest.MonkeyPatch,) -> None:
    monkeypatch.setenv("OTEL_SERVICE_NAME", "operator-service")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.name=should-lose")
    exporter, registry = _memory_tracing()
    config = resolve_observability_config(destinations=("memory",))
    async with trace_run(config, registry, flow="review") as run:
        run.finish(0)
    resource = _resource_of(exporter.get_finished_spans()[0])
    assert resource["service.name"] == "operator-service"
    assert resource["telemetry.sdk.name"] == "opentelemetry"
    assert resource["service.instance.id"]


# --- P18 Task 3: no global providers, ambient metrics/logs, or instrumentation -

@pytest.mark.anyio
async def test_owned_session_installs_no_global_signals_or_instrumentors() -> None:

    exporter, registry = _memory_tracing()
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
    # Owned providers must differ from the global proxy.
    assert exporter.get_finished_spans()


@pytest.mark.anyio
@pytest.mark.parametrize(("stack", "iteration"), [("", False)])
async def test_generation_optional_scope_attributes_never_warn(
    stack: str | None, iteration: int | None, caplog: pytest.LogCaptureFixture,
) -> None:
    exporter, registry = _memory_tracing()
    with caplog.at_level(logging.WARNING, logger="opentelemetry.attributes"):
        async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
            with step_scope("review", stack=stack, iteration=iteration):
                with agent_scope("review", backend="pi"):
                    async with attempt_scope(1) as attempt:
                        _observe_generation(attempt)

    spans = exporter.get_finished_spans()
    by_kind = {(span.attributes or {})["daydream.span.kind"]: span for span in spans}
    assert set(by_kind) == {"run", "step", "agent", "attempt", "generation"}
    assert len(spans) == len(by_kind)
    for child, parent in (("step", "run"), ("agent", "step"), ("attempt", "agent"), ("generation", "attempt")):
        parent_context = by_kind[parent].context
        child_parent = by_kind[child].parent
        assert parent_context is not None and child_parent is not None
        assert child_parent.span_id == parent_context.span_id
    for kind in ("step", "agent", "attempt", "generation"):
        attrs = by_kind[kind].attributes or {}
        for key, supplied in (("daydream.stack", stack), ("daydream.iteration", iteration)):
            if supplied is None:
                assert key not in attrs
            else:
                assert attrs[key] == supplied
                assert type(attrs[key]) is type(supplied)
    assert not [
        record for record in caplog.records
        if record.name == "opentelemetry.attributes" and "Invalid type NoneType" in record.getMessage()
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("inherited_stack", "exit_path"), [(None, "normal"), ("python", "exception"), ("", "cancellation")]
)
async def test_stack_binding_preserves_precedence_and_restores_context_on_every_exit(
    inherited_stack: str | None, exit_path: str,
) -> None:
    """Spanless attribution restores the caller even after a cancelled reviewer."""
    exporter, registry = _memory_tracing()
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
        with step_scope("review", stack=inherited_stack, iteration=0):
            try:
                with anyio.CancelScope() as cancellation:
                    with stack_scope("react"):
                        with stack_scope("generic"):
                            with agent_scope("bound", backend="pi"):
                                async with attempt_scope(1) as attempt:
                                    _observe_generation(attempt)
                        if exit_path == "exception":
                            raise RuntimeError("review failed")
                        if exit_path == "cancellation":
                            cancellation.cancel()
                            await anyio.sleep(0)
            except RuntimeError as error:
                assert str(error) == "review failed"
            with agent_scope("restored", backend="pi"):
                async with attempt_scope(1) as attempt:
                    _observe_generation(attempt)

    spans = exporter.get_finished_spans()
    assert len(spans) == 8  # run + step + two agent/attempt/generation subtrees, no binding spans
    generations = [span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "generation"]
    assert len(generations) == 2
    bound, restored = generations
    expected = inherited_stack if inherited_stack is not None else "react"
    assert (bound.attributes or {})["daydream.stack"] == expected
    if inherited_stack is None:
        assert "daydream.stack" not in (restored.attributes or {})
    else:
        assert (restored.attributes or {})["daydream.stack"] == inherited_stack
    assert all((span.attributes or {})["daydream.iteration"] == 0 for span in generations)


@pytest.mark.anyio
async def test_default_nested_scope_clears_optional_attributes_and_restores_outer_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    exporter, registry = _memory_tracing()
    with caplog.at_level(logging.WARNING, logger="opentelemetry.attributes"):
        async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
            with step_scope("outer", stack="python", iteration=0):
                with step_scope("default-nested"):
                    runtime.associate_run_trajectory("nested-session")
                    with agent_scope("nested", backend="pi"):
                        async with attempt_scope(1) as attempt:
                            _observe_generation(attempt)
                with agent_scope("restored", backend="pi"):
                    async with attempt_scope(1) as attempt:
                        _observe_generation(attempt)
            with step_scope("default-sibling"):
                with agent_scope("sibling", backend="pi"):
                    async with attempt_scope(1) as attempt:
                        _observe_generation(attempt)

    spans = exporter.get_finished_spans()
    by_id = {span.context.span_id: span for span in spans if span.context is not None}
    generations = [span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "generation"]
    assert len(generations) == 3
    assert {(span.attributes or {})["daydream.step"] for span in generations} == {
        "outer", "default-nested", "default-sibling",
    }
    for generation in generations:
        assert generation.parent is not None
        assert (by_id[generation.parent.span_id].attributes or {})["daydream.span.kind"] == "attempt"
    for span in spans:
        attrs = span.attributes or {}
        if attrs.get("daydream.step") == "outer":
            assert attrs["daydream.stack"] == "python"
            assert attrs["daydream.iteration"] == 0
        elif attrs.get("daydream.step") in {"default-nested", "default-sibling"}:
            assert "daydream.stack" not in attrs
            assert "daydream.iteration" not in attrs
        if attrs.get("daydream.step") == "default-nested" and attrs["daydream.span.kind"] != "step":
            assert attrs["daydream.session.id"] == "nested-session"
            assert attrs["traceloop.association.properties.session_id"] == "nested-session"
    assert not [
        record for record in caplog.records
        if record.name == "opentelemetry.attributes" and "Invalid type NoneType" in record.getMessage()
    ]


@pytest.mark.anyio
async def test_generation_child_span_seals_and_ends_once_at_historical_end() -> None:

    exporter, registry = _memory_tracing()
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
        with agent_scope("review", backend="pi", model="requested"):
            async with attempt_scope(1) as attempt:
                _observe_generation(attempt, text=True, metrics=True)
    spans = exporter.get_finished_spans()
    generations = [span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "generation"]
    assert len(generations) == 1
    generation = generations[0]
    attempt_span = next(span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "attempt")
    # End once at the sealed historical message_end, preserving native child timing.
    attrs = generation.attributes or {}
    root = next(span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "run")
    root_run_id = (root.attributes or {})["daydream.run.id"]
    assert attrs["daydream.run.id"] == root_run_id
    assert attrs["traceloop.association.properties.daydream_run_id"] == root_run_id
    assert attrs["traceloop.entity.name"] == generation.name
    assert attrs["daydream.generation.native_started_at_unix_ms"] == 1788690314289
    assert attrs["daydream.generation.native_started_at_unix_ns"] == 1788690314289000000
    assert attrs["daydream.generation.sealed_end_unix_ns"] == 1788690709621000000
    assert attrs["daydream.generation.duration_ns"] == 395332000000
    assert generation.start_time == 1788690314289000000
    assert generation.end_time == 1788690709621000000
    assert generation.parent is not None and generation.parent.span_id == attempt_span.context.span_id
    # Without a recorder’s frozen ledger, children remain unbilled with no usage aliases.
    assert attrs["daydream.generation.billed"] is False
    assert "gen_ai.usage.input_tokens" not in attrs
    assert "gen_ai.response.id" not in attrs
    output_messages = attrs["gen_ai.output.messages"]
    assert isinstance(output_messages, str)
    assert json.loads(output_messages)["parts"][0]["text"] == "hello"
    assert (attempt_span.attributes or {})["daydream.billing.owner"] == "unresolved"


@pytest.mark.anyio
async def test_actual_nested_agent_scope_is_subagent_siblings_are_not() -> None:
    exporter, registry = _memory_tracing()
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
        # Nesting makes a subagent; a subsequent sibling returns to root.
        with agent_scope("parent", backend="pi"):
            with agent_scope("child", backend="pi"):
                pass
        with agent_scope("sibling", backend="pi"):
            pass
    by_path = {span.attributes["traceloop.entity.path"]: span.attributes
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get("daydream.span.kind") == "agent"
    }
    assert by_path["daydream.agent.parent"]["daydream.agent.role"] == "root"
    assert by_path["daydream.agent.child"]["daydream.agent.role"] == "subagent"
    assert by_path["daydream.agent.sibling"]["daydream.agent.role"] == "root"


@pytest.mark.parametrize("returncode", [0, -9])
async def test_codex_lifecycle_export_retains_exit_without_creating_usage(returncode: int) -> None:
    from daydream.backends.lifecycle import BackendLifecycleSnapshot

    exporter, registry = _memory_tracing()
    snapshot = BackendLifecycleSnapshot(
        attempt=1, backend="codex", last_native_event_kind="turn_completed", terminal_received_s=0.0,
        process_exit_s=0.1, process_returncode=returncode, expiry_stage="terminal_drain",
        teardown_started_s=0.2, teardown_completed_s=0.3, loop_sample_count=4, loop_max_lag_s=0.01,
    )
    async with trace_run(
        ObservabilityConfig(destinations=("memory",), capture_content=False), registry, flow="review",
    ):
        with agent_scope("review", backend="codex"):
            async with attempt_scope(1) as attempt:
                attempt.record_backend_lifecycle(snapshot)
                attempt.abort("backend_error")
    span = next(span for span in exporter.get_finished_spans()
                if (span.attributes or {}).get("daydream.span.kind") == "attempt")
    attrs = dict(span.attributes or {})
    assert attrs["daydream.backend_lifecycle.process_returncode"] == returncode
    assert attrs["daydream.backend_lifecycle.terminal_received_s"] == 0.0
    assert attrs["daydream.backend_lifecycle.expiry_stage"] == "terminal_drain"
    assert attrs["daydream.backend_lifecycle.loop_sample_count"] == 4
    assert "daydream.backend_lifecycle.stdout_eof_s" not in attrs
    assert not any(key.startswith("gen_ai.usage.") for key in attrs)
