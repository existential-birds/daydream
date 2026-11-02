"""Destination factories tested against the real OTLP encoder and loopback HTTP transport."""

from __future__ import annotations

from typing import Any

import grpc
import pytest
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from daydream.observability.config import ObservabilityConfig, ObservabilityError
from daydream.observability.exporters import CompatSpanExporter, honeyhive_exporter, langsmith_exporter, otlp_exporter
from daydream.observability.otlp_compat import GrpcBridge
from daydream.observability.privacy import PrivacyPolicy, diagnostic_scope
from tests.harness.otlp import exporting_provider, grpc_trace_server, otlp_collector


def _emit(exporter: SpanExporter) -> None:
    with exporting_provider(exporter, resource=Resource({"service.name": "daydream-test"})) as provider:
        with provider.get_tracer("daydream-test").start_as_current_span("attempt",
            attributes={"daydream.span.kind": "attempt", "daydream.billing.owner": "structural_attempt",
                "gen_ai.usage.input_tokens": 100, "gen_ai.usage.output_tokens": 10, "gen_ai.usage.total_tokens": 110,
                "gen_ai.usage.cache_read.input_tokens": 40, "gen_ai.usage.cache_creation.input_tokens": 20,
                "gen_ai.usage.reasoning.output_tokens": 5, "gen_ai.usage.cost": 0.123,
                "gen_ai.input.messages": '[{"role":"user","parts":[{"type":"text","content":"hello"}]}]',
            },
        ):
            pass


def _emit_spans() -> list[ReadableSpan]:
    """One attributable attempt span for direct exporter.export() calls."""
    return [ReadableSpan("attempt", resource=Resource({"service.name": "daydream-test"}),
            attributes={"daydream.span.kind": "attempt", "daydream.billing.owner": "structural_attempt",
                "gen_ai.usage.input_tokens": 100,
            },
            context=trace_api.SpanContext(
                trace_id=0x11111111111111111111111111111111, span_id=0x2222222222222222, is_remote=False
            ),
        )
    ]


def test_setup_errors_are_actionable_and_do_not_echo_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    cases: list[tuple[Any, dict[str, str]]] = [
        (honeyhive_exporter, {}), (honeyhive_exporter, {"HH_API_KEY": "opaque-secret"}),
        (langsmith_exporter, {"LANGSMITH_API_KEY": "opaque-secret", "LANGSMITH_ENDPOINT": "https://host?opaque-secret"}),
        (langsmith_exporter, {"LANGSMITH_API_KEY": "opaque-secret\nforged: header"}),
        (otlp_exporter, {"OTEL_EXPORTER_OTLP_PROTOCOL": "unsupported-opaque-secret"}),
        (otlp_exporter, {"OTEL_EXPORTER_OTLP_TIMEOUT": "0"}),
        (otlp_exporter, {"OTEL_EXPORTER_OTLP_ENDPOINT": "https://user:opaque-secret@host"}),
    ]
    for factory, env in cases:
        with monkeypatch.context() as patch:
            for key, value in env.items():
                patch.setenv(key, value)
            with pytest.raises(ObservabilityError) as exc:
                factory(ObservabilityConfig())
        assert "opaque-secret" not in str(exc.value)
        assert str(exc.value)


def test_generic_grpc_exporter_uses_standard_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[ExportTraceServiceRequest] = []
    headers: list[tuple[str, str | bytes]] = []

    def receive(request: ExportTraceServiceRequest, context: grpc.ServicerContext) -> ExportTraceServiceResponse:
        received.append(request)
        headers.extend((item[0], item[1]) for item in context.invocation_metadata())
        return ExportTraceServiceResponse()

    with grpc_trace_server(receive) as port:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "grpc")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"http://127.0.0.1:{port}")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "x-custom=grpc-credential")
        monkeypatch.setenv("OTEL_PYTHON_SDK_INTERNAL_METRICS_ENABLED", "true")
        exporter = otlp_exporter(ObservabilityConfig())
        assert isinstance(exporter, CompatSpanExporter)
        bridge = exporter._transport
        assert isinstance(bridge, GrpcBridge)
        delegate_metrics = bridge._delegate._metrics
        inflight_metric = getattr(delegate_metrics, "_inflight", None)
        assert type(inflight_metric).__name__ == "NoOpUpDownCounter"
        _emit(exporter)
        exporter.shutdown()
        exporter.shutdown()
        assert exporter.export(_emit_spans()) == SpanExportResult.FAILURE
        assert exporter.delivery_snapshot()["delivered"] == 1
        assert len(received) == 1
        span = received[0].resource_spans[0].scope_spans[0].spans[0]
        assert span.name == "attempt"
        assert any(
            attr.key == "gen_ai.usage.input_tokens" and attr.value.int_value == 100 for attr in span.attributes
        )
        assert ("x-custom", "grpc-credential") in headers


def test_grpc_bridge_drift_refuses_send_and_closes_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    for drift in ("1.45.3", "missing-surface"):
        with monkeypatch.context() as patch:
            _assert_grpc_bridge_drift_refused(patch, drift)


def _assert_grpc_bridge_drift_refused(monkeypatch: pytest.MonkeyPatch, drift: str) -> None:
    """Unreviewed SDK versions or missing private fields must release the real channel before failing."""
    delegates: list[GrpcExporter] = []
    received: list[ExportTraceServiceRequest] = []

    def receive(request: ExportTraceServiceRequest, context: grpc.ServicerContext) -> ExportTraceServiceResponse:
        received.append(request)
        return ExportTraceServiceResponse()

    def create_delegate(**kwargs: Any) -> GrpcExporter:
        delegate = GrpcExporter(**kwargs)
        delegates.append(delegate)
        if drift == "missing-surface":
            delattr(delegate, "_timeout")
        return delegate

    monkeypatch.setattr("daydream.observability.exporters.GrpcExporter", create_delegate)
    if drift != "missing-surface":
        monkeypatch.setattr("importlib.metadata.version", lambda _distribution: drift)
    with grpc_trace_server(receive) as port:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "grpc")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"http://127.0.0.1:{port}")
        try:
            with pytest.raises(ObservabilityError, match="OpenTelemetry"):
                otlp_exporter(ObservabilityConfig())
            assert len(delegates) == 1
            client = delegates[0]._client
            assert client is not None
            with pytest.raises(ValueError, match="closed"):
                client.Export(request=ExportTraceServiceRequest(), timeout=1)
            assert received == []
        finally:
            for delegate in delegates:
                delegate.shutdown()


@pytest.mark.parametrize("destination", ["langsmith", ])
def test_presets_do_not_forward_redirected_payloads_or_credentials(
    destination: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    with otlp_collector() as second:
        with otlp_collector(status=302, response_headers={"Location": second.base_url + "/stolen"}) as first:
            monkeypatch.setenv("LANGSMITH_ENDPOINT", first.base_url)
            monkeypatch.setenv("LANGSMITH_API_KEY", "opaque-key")
            monkeypatch.setenv("LANGSMITH_WORKSPACE_ID", "workspace-7")
            monkeypatch.setenv("HH_API_URL", first.base_url)
            monkeypatch.setenv("HH_API_KEY", "opaque-key")
            factory = langsmith_exporter if destination == "langsmith" else honeyhive_exporter
            exporter = factory(ObservabilityConfig())
            with diagnostic_scope(PrivacyPolicy()):
                assert exporter.export(_emit_spans()) != SpanExportResult.SUCCESS
                exporter.shutdown()
            assert len(first.requests) == 1
            assert second.requests == []
            if destination == "langsmith":
                assert first.requests[0]["headers"]["x-tenant-id"] == "workspace-7"
            assert "redirected" in caplog.text


def test_malformed_otlp_headers_never_log_credential_fragments(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-valid=normal,opaque-broken-credential")
    with pytest.raises(ObservabilityError):
        otlp_exporter(ObservabilityConfig())
    assert "opaque-broken-credential" not in caplog.text
