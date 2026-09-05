"""Destination factories tested against the real OTLP encoder and loopback HTTP transport."""

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import grpc
import pytest
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from daydream.observability.config import ObservabilityConfig, ObservabilityError
from daydream.observability.exporters import LangSmithExporter, honeyhive_exporter, langsmith_exporter, otlp_exporter
from daydream.observability.privacy import PrivacyPolicy, diagnostic_scope
from tests.harness.otlp import attributes, otlp_collector


@pytest.fixture(autouse=True)
def isolated_trace_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    import os

    for key in os.environ:
        if key.startswith(("OTEL_", "LANGSMITH_", "HH_", "_OTEL_")):
            monkeypatch.delenv(key)
    yield


def _emit(exporter: SpanExporter) -> None:
    provider = TracerProvider(resource=Resource({"service.name": "daydream-test"}), shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        with provider.get_tracer("daydream-test").start_as_current_span(
            "attempt",
            attributes={
                "daydream.span.kind": "attempt",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 10,
                "gen_ai.usage.total_tokens": 110,
                "gen_ai.usage.cache_read.input_tokens": 40,
                "gen_ai.usage.cache_creation.input_tokens": 20,
                "gen_ai.usage.reasoning.output_tokens": 5,
                "gen_ai.usage.cost": 0.123,
                "gen_ai.input.messages": '[{"role":"user","parts":[{"type":"text","content":"hello"}]}]',
            },
        ):
            pass
    finally:
        provider.shutdown()


@pytest.mark.parametrize("destination", ["langsmith", "honeyhive", "otlp"])
def test_destinations_emit_portable_otlp(destination: str, monkeypatch: pytest.MonkeyPatch) -> None:
    with otlp_collector() as receiver:
        monkeypatch.setenv("LANGSMITH_ENDPOINT", receiver.base_url)
        monkeypatch.setenv("LANGSMITH_API_KEY", "langsmith-opaque-key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "my project")
        monkeypatch.setenv("LANGSMITH_WORKSPACE_ID", "workspace-id")
        monkeypatch.setenv("HH_API_URL", receiver.base_url)
        monkeypatch.setenv("HH_API_KEY", "honeyhive-opaque-key")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", receiver.base_url)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-custom=generic-key")
        factory = {"langsmith": langsmith_exporter, "honeyhive": honeyhive_exporter, "otlp": otlp_exporter}[destination]
        _emit(factory(ObservabilityConfig()))
        assert len(receiver.spans) == 1
        span = attributes(receiver.spans[0])
        assert span["gen_ai.usage.input_tokens"] == 100
        assert span["gen_ai.usage.cost"] == 0.123
        assert "hello" in span["gen_ai.input.messages"]
        request = receiver.requests[0]
        expected_paths = {"langsmith": "/otel/v1/traces", "honeyhive": "/opentelemetry/v1/traces", "otlp": "/v1/traces"}
        assert request["path"] == expected_paths[destination]
        assert attributes(request["body"]["resourceSpans"][0]["resource"])["service.name"] == "daydream-test"
        headers = request["headers"]
        if destination == "langsmith":
            assert headers["x-api-key"] == "langsmith-opaque-key"
            assert headers["langsmith-project"] == "my project"
            assert headers["x-tenant-id"] == "workspace-id"
            assert span["langsmith.span.kind"] == "llm"
            assert json.loads(span["langsmith.usage_metadata"]) == {
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "total_cost": 0.123,
                "input_token_details": {"cache_read": 40, "cache_creation": 20},
                "output_token_details": {"reasoning": 5},
            }
        elif destination == "honeyhive":
            assert headers["authorization"] == "Bearer honeyhive-opaque-key"
        else:
            assert headers["x-custom"] == "generic-key"
        if destination != "otlp":
            assert "x-custom" not in headers


def test_langsmith_mapping_preserves_original_spans_and_only_attempts_own_usage() -> None:
    sink = InMemorySpanExporter()
    exporter = LangSmithExporter(sink)
    spans = [
        ReadableSpan(
            "scope", resource=Resource({}), attributes={"daydream.span.kind": kind, "gen_ai.usage.input_tokens": 0}
        )
        for kind in ("run", "phase", "agent", "attempt", "tool")
    ]
    assert exporter.export(spans) == SpanExportResult.SUCCESS
    result = sink.get_finished_spans()
    assert [span.attributes["langsmith.span.kind"] for span in result if span.attributes] == [
        "chain",
        "chain",
        "chain",
        "llm",
        "tool",
    ]
    for original, mapped in zip(spans, result, strict=True):
        assert original.attributes is not None and mapped.attributes is not None
        assert "langsmith.span.kind" not in original.attributes
        assert all(mapped.attributes[key] == value for key, value in original.attributes.items())
        assert ("langsmith.usage_metadata" in mapped.attributes) == (
            original.attributes["daydream.span.kind"] == "attempt"
        )
    assert exporter.force_flush()
    exporter.shutdown()


@pytest.mark.parametrize(
    "factory,env",
    [
        (langsmith_exporter, {}),
        (honeyhive_exporter, {}),
        (honeyhive_exporter, {"HH_API_KEY": "opaque-secret"}),
        (
            langsmith_exporter,
            {"LANGSMITH_API_KEY": "opaque-secret", "LANGSMITH_ENDPOINT": "https://user:opaque-secret@host"},
        ),
        (
            langsmith_exporter,
            {"LANGSMITH_API_KEY": "opaque-secret", "LANGSMITH_ENDPOINT": "https://host?opaque-secret"},
        ),
        (langsmith_exporter, {"LANGSMITH_API_KEY": "opaque-secret\nforged: header"}),
        (otlp_exporter, {"OTEL_EXPORTER_OTLP_PROTOCOL": "unsupported-opaque-secret"}),
        (otlp_exporter, {"OTEL_EXPORTER_OTLP_TIMEOUT": "nan"}),
        (otlp_exporter, {"OTEL_EXPORTER_OTLP_TIMEOUT": "0"}),
        (otlp_exporter, {"OTEL_EXPORTER_OTLP_ENDPOINT": "https://user:opaque-secret@host"}),
    ],
)
def test_setup_errors_are_actionable_and_do_not_echo_credentials(
    factory: Any,
    env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ObservabilityError) as exc:
        factory(ObservabilityConfig())
    assert "opaque-secret" not in str(exc.value)
    assert str(exc.value)


def test_generic_trace_specific_environment_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    with otlp_collector() as receiver:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://unreachable.invalid")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", receiver.base_url + "/custom")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-generic=unused")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "x-selected=active")
        _emit(otlp_exporter(ObservabilityConfig()))
        assert receiver.requests[0]["path"] == "/custom"
        assert receiver.requests[0]["headers"]["x-selected"] == "active"
        assert "x-generic" not in receiver.requests[0]["headers"]


def test_generic_grpc_exporter_uses_standard_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[ExportTraceServiceRequest] = []
    headers: list[tuple[str, str | bytes]] = []

    def receive(request: ExportTraceServiceRequest, context: grpc.ServicerContext) -> bytes:
        received.append(request)
        headers.extend((item[0], item[1]) for item in context.invocation_metadata())
        return ExportTraceServiceResponse().SerializeToString()

    with ThreadPoolExecutor(max_workers=1) as pool:
        server = grpc.server(pool)
        server.add_generic_rpc_handlers(
            (
                grpc.method_handlers_generic_handler(
                    "opentelemetry.proto.collector.trace.v1.TraceService",
                    {
                        "Export": grpc.unary_unary_rpc_method_handler(
                            receive,
                            request_deserializer=ExportTraceServiceRequest.FromString,
                        ),
                    },
                ),
            )
        )
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "grpc")
            monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"http://127.0.0.1:{port}")
            monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "x-custom=grpc-credential")
            exporter = otlp_exporter(ObservabilityConfig())
            assert isinstance(exporter, GrpcExporter)
            _emit(exporter)
            assert len(received) == 1
            span = received[0].resource_spans[0].scope_spans[0].spans[0]
            assert span.name == "attempt"
            assert any(
                attr.key == "gen_ai.usage.input_tokens" and attr.value.int_value == 100 for attr in span.attributes
            )
            assert ("x-custom", "grpc-credential") in headers
        finally:
            server.stop(grace=0).wait(timeout=2)


@pytest.mark.parametrize("destination", ["langsmith", "honeyhive"])
def test_presets_do_not_forward_redirected_payloads_or_credentials(
    destination: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with otlp_collector() as second:
        with otlp_collector(status=302, response_headers={"Location": second.base_url + "/stolen"}) as first:
            monkeypatch.setenv("LANGSMITH_ENDPOINT", first.base_url)
            monkeypatch.setenv("LANGSMITH_API_KEY", "opaque-key")
            monkeypatch.setenv("HH_API_URL", first.base_url)
            monkeypatch.setenv("HH_API_KEY", "opaque-key")
            factory = langsmith_exporter if destination == "langsmith" else honeyhive_exporter
            with diagnostic_scope(PrivacyPolicy()):
                _emit(factory(ObservabilityConfig()))
            assert len(first.requests) == 1
            assert second.requests == []
            assert "redirected" in caplog.text


@pytest.mark.parametrize("destination", ["langsmith", "honeyhive"])
def test_presets_ignore_ambient_auth_and_generic_transport_settings(
    destination: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_netrc(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("preset must never consult ambient netrc authentication")

    monkeypatch.setattr("requests.sessions.get_netrc_auth", forbidden_netrc)
    monkeypatch.setenv("_OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER", "nonexistent-provider")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE", "/does-not-exist.pem")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_CLIENT_CERTIFICATE", "/does-not-exist-client.pem")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_CLIENT_KEY", "/does-not-exist-key.pem")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT", "not-a-number")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_COMPRESSION", "invalid-compression")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://invalid.local")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "authorization=wrong")
    with otlp_collector() as receiver:
        monkeypatch.setenv("LANGSMITH_ENDPOINT", receiver.base_url)
        monkeypatch.setenv("LANGSMITH_API_KEY", "real-key")
        monkeypatch.setenv("HH_API_URL", receiver.base_url)
        monkeypatch.setenv("HH_API_KEY", "real-key")
        factory = langsmith_exporter if destination == "langsmith" else honeyhive_exporter
        _emit(factory(ObservabilityConfig()))
        headers = receiver.requests[0]["headers"]
        if destination == "langsmith":
            assert headers["x-api-key"] == "real-key"
            assert "authorization" not in headers
        else:
            assert headers["authorization"] == "Bearer real-key"
        assert "content-encoding" not in headers


def test_malformed_otlp_headers_never_log_credential_fragments(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-valid=normal,opaque-broken-credential")
    exporter = otlp_exporter(ObservabilityConfig())
    exporter.shutdown()
    assert "opaque-broken-credential" not in caplog.text


def test_sdk_http_reason_is_sanitized_within_runtime_diagnostic_boundary(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "opaque-http-response-secret")
    with otlp_collector(status=400, reason="opaque-http-response-secret") as receiver:
        monkeypatch.setenv("LANGSMITH_ENDPOINT", receiver.base_url)
        with diagnostic_scope(PrivacyPolicy()):
            _emit(langsmith_exporter(ObservabilityConfig()))
        assert "opaque-http-response-secret" not in caplog.text
        assert "Failed to export" in caplog.text
