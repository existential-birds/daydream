"""Owned tracing lifecycle and event reconciliation through real SDK spans."""

import json
import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, StatusCode, TraceFlags

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
from daydream.extensions.registry import Registry
from daydream.observability.config import (
    ObservabilityConfig,
    resolve_observability_config,
)
from daydream.observability.privacy import PrivacyPolicy, diagnostic_scope
from daydream.observability.runtime import current_run_trace_id, current_session, trace_run
from daydream.observability.spans import agent_scope, attempt_scope, stack_scope, step_scope
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend
from tests.harness.trajectory import make_recorder


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
async def test_attempt_records_scrubbed_diagnostic_events_in_observed_order(monkeypatch: pytest.MonkeyPatch,) -> None:
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
                        code=f"parser_{secret}", message="another message", metadata={"raw": "observable"},
                    )
                )
                attempt.observe(DiagnosticEvent(
                        code="codex_transport_coverage", message="transport detail", metadata={"occurrences": 4},
                    )
                )

    spans = exporter.get_finished_spans()
    assert sorted(str((span.attributes or {})["daydream.span.kind"]) for span in spans) == ["agent", "attempt", "run"]
    attempt_span = next(span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "attempt")
    assert [(event.name, dict(event.attributes or {})) for event in attempt_span.events] == [
        ("daydream.backend_diagnostic", {"daydream.backend_diagnostic.code": code,
            "daydream.backend_diagnostic.message": message, "daydream.backend_diagnostic.metadata": metadata})
        for code, message, metadata in [
            ("parser_[REDACTED_CREDENTIAL]", "do not export [REDACTED_CREDENTIAL]",
                '{"api_key":"[REDACTED_CREDENTIAL]","count":99}'),
            ("parser_[REDACTED_CREDENTIAL]", "another message", '{"raw":"observable"}'),
            ("codex_transport_coverage", "transport detail", '{"occurrences":4}'),
        ]
    ]
    for span in spans:
        assert not any(key.startswith("daydream.backend_diagnostic") for key in (span.attributes or {}))
        assert span is attempt_span or not span.events


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
async def test_shutdown_deadline_is_total_and_shutdown_once( caplog: pytest.LogCaptureFixture,
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
            super().shutdown()
            self.shutdown_complete.set()

        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
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
async def test_startup_failure_and_retry_keep_dispatch_facts_failed_billing_and_salvaged_result(tmp_path: Path) -> None:
    class RetryableError(RuntimeError):
        retryable = True

    exporter, registry = _memory_tracing()
    backend = ScriptedBackend(script=[
            [RetryableError("startup")],
            [TextEvent("failed content"), CostEvent(0.05, 4, 2), ResultEvent(None, None), RetryableError("retry")],
            [RequestEvent("effective prompt", system_prompt="effective system"), TextEvent('{"ok":true}'),
                ResultEvent({"wrong": True}, None), MetricsEvent("", 30, 6, 4, 0.2, usage_scope="invocation"),
                CostEvent(None, 0, None),
            ],
        ], model="requested", retry_attempts=2, retry_base_delay_s=0, retry_max_delay_s=0,
    )
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
        result = await run_agent(backend, tmp_path, "original", phase=DaydreamPhase.REVIEW,
            output_schema={"type": "object", "required": ["ok"]},
        )
    assert result[0] == {"ok": True}
    attempts = [
        span for span in exporter.get_finished_spans() if (span.attributes or {}).get("daydream.span.kind") == "attempt"
    ]
    assert [span.status.status_code for span in attempts] == [StatusCode.ERROR, StatusCode.ERROR, StatusCode.OK]
    startup, first, second = [dict(span.attributes or {}) for span in attempts]
    for attrs in (startup, first, second):
        assert {key: value for key, value in attrs.items() if key.startswith("daydream.host_dispatch.")} == {
            "daydream.host_dispatch.user_utf8_bytes": len("original".encode()),
            "daydream.host_dispatch.system_disposition": "unsupported",
        }
    assert not any(key.startswith("gen_ai.usage.") for key in startup)
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
                DiagnosticEvent("parser_gap", "PRIVATE DIAGNOSTIC", {"raw": "PRIVATE METADATA"}),
                ThinkingEvent("PRIVATE REASON"), ToolStartEvent("tool", "Read", {"arg": "PRIVATE INPUT"}),
                ToolResultEvent("tool", "PRIVATE OUTPUT", True), CostEvent(0.1, 2, 3),
                ResultEvent({"verdict": "PRIVATE STRUCTURED"}, None), RuntimeError("PRIVATE ERROR opaque-secret"),
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
        encoded = (str(span.attributes) + str([event.attributes for event in span.events])
                   + str(span.resource.attributes) + str(span.status.description))
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
    events = [dict(event.attributes or {}) for event in attempt.events if event.name == "daydream.backend_diagnostic"]
    assert events == [{"daydream.backend_diagnostic.code": "parser_gap"}]
    assert attempt.status.description == "error"
    assert attempt.events[-1].name == "exception"
    assert (attempt.events[-1].attributes or {})["exception.type"] == "RuntimeError"
    assert "exception.message" not in (attempt.events[-1].attributes or {})

@pytest.mark.anyio
async def test_hydration_ignores_ambient_length_limit_and_serialization_is_fail_closed(monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "2")
    exporter, registry = _memory_tracing()
    text = "long source and result " * 1000
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
        async with attempt_scope(1) as observer:
            observer.observe(RequestEvent(text))
            # Streamed deltas join one assistant text part, so the whole output survives hydration.
            observer.observe(TextEvent(text[:7]))
            observer.observe(TextEvent(text[7:]))
            observer.observe(ResultEvent(object(), None))
    attempt = next(
        span for span in exporter.get_finished_spans() if (span.attributes or {}).get("daydream.span.kind") == "attempt"
    )
    attrs = dict(attempt.attributes or {})
    assert text in str(attrs["gen_ai.input.messages"])
    assert text in str(attrs["gen_ai.output.messages"])
    assert json.loads(str(attrs["traceloop.entity.output"])) == "[UNSERIALIZABLE]"


# --- P18 Task 3: ambient OpenLLMetry context isolation ------------------------


# --- P18 Task 3: normal/notebook parity ---------------------------------------


# --- P18 Task 3: strict operator resource parsing -----------------------------


def _resource_of(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.resource.attributes)


@pytest.mark.anyio
async def test_invalid_operator_resource_rejects_entire_variable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    for raw in ('key=%FF', 'not-a-pair', 'ke%Gy=value', 'dup.key=a%2Bone,dup.k%65y=two', 'bad\x01key=value'):
        caplog.clear()
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
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.name=should-lose,deployment.api_key=opaque-resource-key,"
                       "service.instance.id=operator-instance")
    exporter, registry = _memory_tracing()
    config = resolve_observability_config(destinations=("memory",))
    async with trace_run(config, registry, flow="review") as run:
        run.finish(0)
    resource = _resource_of(exporter.get_finished_spans()[0])
    assert resource["service.name"] == "operator-service"
    assert resource["telemetry.sdk.name"] == "opentelemetry"
    assert resource["service.instance.id"] == "operator-instance"  # Operator identity wins over the run id.
    assert resource["deployment.api_key"] == "[REDACTED_CREDENTIAL]"


@pytest.mark.anyio
async def test_run_trace_identity_and_delivery_failures_never_block_the_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Capture identifies only the owned root; failed export and flush only warn."""

    class BrokenContext(NonRecordingSpan):
        def get_span_context(self) -> SpanContext:
            raise RuntimeError("owned context unavailable")

    class RejectingExporter(InMemorySpanExporter):
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            return SpanExportResult.FAILURE

    registry = Registry()
    registry.register_trace_exporter("rejecting", lambda _: RejectingExporter())
    ambient = NonRecordingSpan(SpanContext(123, 456, False, TraceFlags(TraceFlags.SAMPLED)))
    with trace.use_span(ambient):
        async with trace_run(ObservabilityConfig(destinations=("rejecting",)), registry, flow="review") as run:
            owned = current_run_trace_id()
            assert owned is not None and int(owned, 16) not in (0, 123)
            session = current_session()
            assert session is not None
            for unavailable in (NonRecordingSpan(SpanContext(0, 0, False)), BrokenContext(SpanContext(0, 0, False))):
                with monkeypatch.context() as patch:
                    patch.setattr(session, "root_scope", SimpleNamespace(span=unavailable))
                    assert current_run_trace_id() is None
            def fail_flush(_provider: TracerProvider, timeout_millis: int = 30000) -> bool:
                raise RuntimeError("processor unavailable")
            monkeypatch.setattr(TracerProvider, "force_flush", fail_flush)  # Cleanup flushes after the body.
            run.finish(0)
    assert current_run_trace_id() is None
    assert "Trace export failed; review execution continues" in caplog.text
    assert "Trace processor flush failed" in caplog.text


# --- P18 Task 3: no global providers, ambient metrics/logs, or instrumentation -

@pytest.mark.anyio
async def test_stack_binding_preserves_precedence_and_restores_context_on_every_exit(tmp_path: Path) -> None:
    """Spanless attribution restores the caller after a normal exit and after a cancelled reviewer."""
    for inherited_stack, exit_path in ((None, "normal"), ("", "cancellation")):
        exporter, registry = _memory_tracing()
        async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
            with step_scope("review", stack=inherited_stack, iteration=0):
                with anyio.CancelScope() as cancellation:
                    with stack_scope("react"):
                        with stack_scope("generic"):
                            with agent_scope("bound", backend="pi"):
                                async with attempt_scope(1) as attempt:
                                    _observe_generation(attempt)
                        if exit_path == "cancellation":
                            cancellation.cancel()
                            await anyio.sleep(0)
                with agent_scope("restored", backend="pi"):
                    async with attempt_scope(1) as attempt:
                        _observe_generation(attempt)

        spans = exporter.get_finished_spans()
        assert len(spans) == 8, exit_path  # run + step + two agent/attempt/generation subtrees, no binding spans
        generations = [span for span in spans if (span.attributes or {}).get("daydream.span.kind") == "generation"]
        assert len(generations) == 2, exit_path
        bound, restored = generations
        expected = inherited_stack if inherited_stack is not None else "react"
        assert (bound.attributes or {})["daydream.stack"] == expected, exit_path
        if inherited_stack is None:
            assert "daydream.stack" not in (restored.attributes or {})
        else:
            assert (restored.attributes or {})["daydream.stack"] == inherited_stack, exit_path
        assert all((span.attributes or {})["daydream.iteration"] == 0 for span in generations)

    # Spans opened inside a forked recorder carry that child's trajectory identity.
    exporter, registry = _memory_tracing()
    root = make_recorder(tmp_path)
    async with root, root.fork("stack-python") as child:
        async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review"):
            with step_scope("review", stack=None, iteration=0):
                pass
    step, = [span for span in exporter.get_finished_spans() if span.name == "daydream.step.review"]
    assert (step.attributes or {})["daydream.trajectory.descriptor"] == "stack-python"
    assert (step.attributes or {})["daydream.trajectory.id"] == f"{child.session_id}:stack-python"


async def test_codex_lifecycle_export_retains_exit_without_creating_usage() -> None:
    returncode = -9
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
