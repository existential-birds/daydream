"""Destination factories tested against the real OTLP encoder and loopback HTTP transport."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import grpc
import pytest
from opentelemetry import trace as trace_api
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from daydream.observability.config import ObservabilityConfig, ObservabilityError
from daydream.observability.exporters import (
    GrpcCompatExporter,
    LangSmithExporter,
    honeyhive_exporter,
    langsmith_exporter,
    otlp_exporter,
)
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
                "daydream.billing.owner": "structural_attempt",
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


def _emit_spans() -> list[ReadableSpan]:
    """One attributable attempt span for direct exporter.export() calls."""
    from opentelemetry.trace import SpanContext

    return [
        ReadableSpan(
            "attempt",
            resource=Resource({"service.name": "daydream-test"}),
            attributes={
                "daydream.span.kind": "attempt",
                "daydream.billing.owner": "structural_attempt",
                "gen_ai.usage.input_tokens": 100,
            },
            context=SpanContext(
                trace_id=0x11111111111111111111111111111111, span_id=0x2222222222222222, is_remote=False
            ),
        )
    ]


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
            assert span["langsmith.span.kind"] == "chain"
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
            assert span["honeyhive_event_type"] == "chain"
            assert span["honeyhive_metadata.cost"] == 0.123
            assert span["honeyhive_metadata.cache_read_input_tokens"] == 40
            assert span["honeyhive_metadata.cache_write_input_tokens"] == 20
            assert span["honeyhive_metadata.reasoning_tokens"] == 5
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
        for kind in ("run", "phase", "agent", "tool")
    ]
    spans.append(
        ReadableSpan(
            "attempt",
            resource=Resource({}),
            attributes={
                "daydream.span.kind": "attempt",
                "daydream.billing.owner": "structural_attempt",
                "gen_ai.usage.input_tokens": 0,
            },
        )
    )
    assert exporter.export(spans) == SpanExportResult.SUCCESS
    result = sink.get_finished_spans()
    assert [span.attributes["langsmith.span.kind"] for span in result if span.attributes] == [
        "chain",
        "chain",
        "chain",
        "tool",
        "chain",
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


@pytest.mark.parametrize("known_usage", [False, True])
@pytest.mark.parametrize("has_session", [False, True])
def test_honeyhive_mapping_preserves_spans_and_limits_native_usage_to_attempts(
    known_usage: bool, has_session: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    originals = InMemorySpanExporter()
    run_id = "8dc328f9-f3af-4293-a8c3-19f8627999cd"
    session_id = "f55b21e4-4693-4795-a625-64b1a2d969ee"
    with otlp_collector() as receiver:
        monkeypatch.setenv("HH_API_URL", receiver.base_url)
        monkeypatch.setenv("HH_API_KEY", "opaque-key")
        exporter = honeyhive_exporter(ObservabilityConfig())
        provider = TracerProvider(shutdown_on_exit=False)
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        provider.add_span_processor(SimpleSpanProcessor(originals))
        try:
            for kind in ("run", "step", "agent", "tool"):
                span_attributes: dict[str, Any] = {
                    "daydream.span.kind": kind,
                    "daydream.run.id": run_id,
                    "daydream.flow": "review",
                }
                if has_session:
                    span_attributes["traceloop.association.properties.session_id"] = session_id
                if known_usage:
                    span_attributes.update({
                        "gen_ai.usage.input_tokens": 0,
                        "gen_ai.usage.output_tokens": 0,
                        "gen_ai.usage.cost": 0.0,
                        "gen_ai.usage.cache_read.input_tokens": 0,
                        "gen_ai.usage.cache_creation.input_tokens": 0,
                        "gen_ai.usage.reasoning.output_tokens": 0,
                    })
                with provider.get_tracer("daydream-test").start_as_current_span(kind, attributes=span_attributes):
                    pass
            with provider.get_tracer("daydream-test").start_as_current_span(
                "attempt",
                attributes={
                    "daydream.span.kind": "attempt",
                    "daydream.billing.owner": "structural_attempt",
                    "daydream.run.id": run_id,
                    "daydream.flow": "review",
                    **({"traceloop.association.properties.session_id": session_id} if has_session else {}),
                    **(
                        {
                            "gen_ai.usage.input_tokens": 0,
                            "gen_ai.usage.output_tokens": 0,
                            "gen_ai.usage.cost": 0.0,
                            "gen_ai.usage.cache_read.input_tokens": 0,
                            "gen_ai.usage.cache_creation.input_tokens": 0,
                            "gen_ai.usage.reasoning.output_tokens": 0,
                        }
                        if known_usage
                        else {}
                    ),
                },
            ):
                pass
            assert provider.force_flush()
            recorded = originals.get_finished_spans()
        finally:
            provider.shutdown()
    assert len(receiver.spans) == len(recorded) == 5
    assert [attributes(span)["honeyhive_event_type"] for span in receiver.spans] == [
        "chain", "chain", "chain", "tool", "chain"
    ]
    for original, mapped in zip(recorded, receiver.spans, strict=True):
        assert original.attributes is not None and original.context is not None
        native = attributes(mapped)
        assert base64.b64decode(mapped["spanId"]).hex() == f"{original.context.span_id:016x}"
        assert all(native[key] == value for key, value in original.attributes.items())
        assert not any(key.startswith("honeyhive") for key in original.attributes)
        assert native["honeyhive.session_id"] == (session_id if has_session else run_id)
        assert native["honeyhive.session_auto_create"] is True
        assert native["honeyhive.session_name"] == "daydream.review"
        expected = known_usage and original.attributes["daydream.span.kind"] == "attempt"
        for key in (
            "prompt_tokens", "completion_tokens", "cost", "cache_read_input_tokens",
            "cache_write_input_tokens", "reasoning_tokens",
        ):
            assert (f"honeyhive_metadata.{key}" in native) == expected
            if expected:
                assert native[f"honeyhive_metadata.{key}"] == 0


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
            assert isinstance(exporter, GrpcCompatExporter)
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
            exporter = factory(ObservabilityConfig())
            with diagnostic_scope(PrivacyPolicy()):
                assert exporter.export(_emit_spans()) != SpanExportResult.SUCCESS
                exporter.shutdown()
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
    with pytest.raises(ObservabilityError):
        otlp_exporter(ObservabilityConfig())
    assert "opaque-broken-credential" not in caplog.text


def test_sdk_http_reason_is_sanitized_within_runtime_diagnostic_boundary(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "opaque-http-response-secret")
    with otlp_collector(status=400, reason="opaque-http-response-secret") as receiver:
        monkeypatch.setenv("LANGSMITH_ENDPOINT", receiver.base_url)
        exporter = langsmith_exporter(ObservabilityConfig())
        with diagnostic_scope(PrivacyPolicy()):
            assert exporter.export(_emit_spans()) != SpanExportResult.SUCCESS
            exporter.shutdown()
        assert "opaque-http-response-secret" not in caplog.text
        assert "Failed to export" in caplog.text


@pytest.mark.parametrize(
    "kind,expected_hh,expected_ls",
    [
        ("run", "chain", "chain"),
        ("step", "chain", "chain"),
        ("agent", "chain", "chain"),
        ("attempt", "chain", "chain"),
        ("generation", "model", "llm"),
        ("tool", "tool", "tool"),
    ],
)
def test_vendor_chain_model_tool_classification(
    kind: str, expected_hh: str, expected_ls: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with otlp_collector() as honeyhive, otlp_collector() as langsmith:
        monkeypatch.setenv("HH_API_URL", honeyhive.base_url)
        monkeypatch.setenv("HH_API_KEY", "opaque-key")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", langsmith.base_url)
        monkeypatch.setenv("LANGSMITH_API_KEY", "opaque-key")
        for exporter in (honeyhive_exporter(ObservabilityConfig()), langsmith_exporter(ObservabilityConfig())):
            sink = InMemorySpanExporter()
            provider = TracerProvider(shutdown_on_exit=False)
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            provider.add_span_processor(SimpleSpanProcessor(sink))
            group: list[dict[str, Any]] = [
                {"daydream.span.kind": kind, "daydream.run.id": "run-1", "daydream.flow": "review"},
                {"daydream.span.kind": "tool", "daydream.run.id": "run-1", "daydream.flow": "review"},
            ]
            if kind == "attempt":
                group[0]["daydream.billing.owner"] = "structural_attempt"
            elif kind == "generation":
                group[0]["daydream.generation.billed"] = True
            for attributes0 in group:
                with provider.get_tracer("daydream-test").start_as_current_span("span", attributes=attributes0):
                    pass
            provider.shutdown()
        hh_spans = [attributes(span) for span in honeyhive.spans if span["name"] == "span"]
        ls_spans = [attributes(span) for span in langsmith.spans if span["name"] == "span"]
        assert hh_spans[0]["honeyhive_event_type"] == expected_hh
        assert ls_spans[0]["langsmith.span.kind"] == expected_ls
        assert hh_spans[1]["honeyhive_event_type"] == "tool"
        assert ls_spans[1]["langsmith.span.kind"] == "tool"


def test_generation_billed_owner_carries_native_usage_only_on_resolved_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with otlp_collector() as honeyhive, otlp_collector() as langsmith:
        monkeypatch.setenv("HH_API_URL", honeyhive.base_url)
        monkeypatch.setenv("HH_API_KEY", "opaque-key")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", langsmith.base_url)
        monkeypatch.setenv("LANGSMITH_API_KEY", "opaque-key")
        shared = TracerProvider(shutdown_on_exit=False)
        for exporter in (honeyhive_exporter(ObservabilityConfig()), langsmith_exporter(ObservabilityConfig())):
            shared.add_span_processor(SimpleSpanProcessor(exporter))
        with shared.get_tracer("daydream-test").start_as_current_span(
            "attempt",
            attributes={
                "daydream.span.kind": "attempt",
                "daydream.billing.owner": "generation_children",
                "gen_ai.usage.input_tokens": 100,
            },
        ) as attempt:
            with shared.get_tracer("daydream-test").start_as_current_span(
                "generation",
                attributes={
                    "daydream.span.kind": "generation",
                    "daydream.generation.billed": True,
                    "gen_ai.usage.input_tokens": 60,
                    "gen_ai.usage.output_tokens": 10,
                    "gen_ai.usage.cost": 0.05,
                },
                context=trace_api.set_span_in_context(attempt),
            ):
                pass
        shared.shutdown()
    hh = [attributes(span) for span in honeyhive.spans]
    ls = [attributes(span) for span in langsmith.spans]
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for native in hh:
        by_kind.setdefault(native["daydream.span.kind"], []).append(native)
    # The structural chain is never billed when the frozen owner is
    # generation_children: no native usage/cost aliases on the attempt.
    attempt_native = by_kind["attempt"][0]
    generation_native = by_kind["generation"][0]
    assert attempt_native["daydream.billing.owner"] == "generation_children"
    assert not any(key.startswith("honeyhive_metadata.") for key in attempt_native)
    assert generation_native["honeyhive_event_type"] == "model"
    assert generation_native["honeyhive_metadata.prompt_tokens"] == 60
    assert generation_native["honeyhive_metadata.cost"] == 0.05
    ls_by_kind: dict[str, list[dict[str, Any]]] = {}
    for native in ls:
        ls_by_kind.setdefault(native["daydream.span.kind"], []).append(native)
    assert ls_by_kind["attempt"][0]["langsmith.span.kind"] == "chain"
    assert "langsmith.usage_metadata" not in ls_by_kind["attempt"][0]
    assert ls_by_kind["generation"][0]["langsmith.span.kind"] == "llm"
    assert json.loads(ls_by_kind["generation"][0]["langsmith.usage_metadata"]) == {
        "input_tokens": 60,
        "output_tokens": 10,
        "total_cost": 0.05,
    }


def test_owner_none_and_unbilled_generation_get_no_native_usage() -> None:
    sink = InMemorySpanExporter()
    exporter = LangSmithExporter(sink)
    spans = [
        ReadableSpan(
            "attempt",
            resource=Resource({}),
            attributes={
                "daydream.span.kind": "attempt",
                "daydream.billing.owner": "none",
                "gen_ai.usage.input_tokens": 100,
            },
        ),
        ReadableSpan(
            "generation",
            resource=Resource({}),
            attributes={
                "daydream.span.kind": "generation",
                "daydream.generation.billed": False,
                "gen_ai.usage.input_tokens": 60,
            },
        ),
    ]
    assert exporter.export(spans) == SpanExportResult.SUCCESS
    result = sink.get_finished_spans()
    for span in result:
        attrs = span.attributes or {}
        assert "langsmith.usage_metadata" not in attrs
        if attrs["daydream.span.kind"] == "attempt":
            assert attrs["langsmith.span.kind"] == "chain"
    assert result[1].attributes is not None
    assert result[1].attributes["langsmith.span.kind"] == "llm"


def test_honeyhive_agent_name_uses_standard_attribute_no_underscore_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with otlp_collector() as receiver:
        monkeypatch.setenv("HH_API_URL", receiver.base_url)
        monkeypatch.setenv("HH_API_KEY", "opaque-key")
        exporter = honeyhive_exporter(ObservabilityConfig())
        provider = TracerProvider(shutdown_on_exit=False)
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        with provider.get_tracer("daydream-test").start_as_current_span(
            "agent",
            attributes={
                "daydream.span.kind": "agent",
                "gen_ai.agent.name": "review",
                "daydream.agent.name": "review",
                "daydream.run.id": "run-1",
                "daydream.flow": "review",
            },
        ):
            pass
        provider.shutdown()
    native = attributes(receiver.spans[0])
    # The standard public attribute is preserved without any guessed
    # underscore-prefixed derived field (issue #1156 / AC-25).
    assert native["gen_ai.agent.name"] == "review"
    assert not any(key.startswith(("_", "honeyhive_metadata.agent_")) for key in native)


def test_langsmith_ls_agent_type_only_on_actual_agent_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    with otlp_collector() as receiver:
        monkeypatch.setenv("LANGSMITH_ENDPOINT", receiver.base_url)
        monkeypatch.setenv("LANGSMITH_API_KEY", "opaque-key")
        exporter = langsmith_exporter(ObservabilityConfig())
        provider = TracerProvider(shutdown_on_exit=False)
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        for kind, role in (("agent", "root"), ("agent", "subagent"), ("attempt", "root"), ("step", None)):
            attrs: dict[str, Any] = {"daydream.span.kind": kind, "daydream.run.id": "run-1"}
            if role is not None:
                attrs["daydream.agent.role"] = role
            with provider.get_tracer("daydream-test").start_as_current_span(kind, attributes=attrs):
                pass
        provider.shutdown()
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for span in receiver.spans:
        span_attrs = attributes(span)
        by_kind.setdefault(span_attrs["daydream.span.kind"], []).append(span_attrs)
    assert by_kind["agent"][0]["langsmith.metadata.ls_agent_type"] == "root"
    assert by_kind["agent"][1]["langsmith.metadata.ls_agent_type"] == "subagent"
    for native in by_kind["attempt"]:
        assert "langsmith.metadata.ls_agent_type" not in native
    for native in by_kind["step"]:
        assert "langsmith.metadata.ls_agent_type" not in native


def test_destination_copies_preserve_dropped_attribute_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    with otlp_collector() as receiver:
        monkeypatch.setenv("HH_API_URL", receiver.base_url)
        monkeypatch.setenv("HH_API_KEY", "opaque-key")
        exporter = honeyhive_exporter(ObservabilityConfig())
        provider = TracerProvider(shutdown_on_exit=False, span_limits=SpanLimits(max_attributes=4))
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        with provider.get_tracer("daydream-test").start_as_current_span(
            "attempt", attributes={"a": "1", "b": "2", "c": "3", "d": "4", "e": "5", "f": "6"}
        ):
            pass
        provider.shutdown()
    native = receiver.spans[0]
    # The HoneyHive clone keeps the exact original dropped-attribute count
    # (2 attributes dropped at the source) while adding vendor metadata.
    assert native["droppedAttributesCount"] == 2


def test_honeyhive_session_derives_only_from_daydream_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Session ID/name/autocreate never come from a native conversation id."""
    with otlp_collector() as receiver:
        monkeypatch.setenv("HH_API_URL", receiver.base_url)
        monkeypatch.setenv("HH_API_KEY", "opaque-key")
        exporter = honeyhive_exporter(ObservabilityConfig())
        provider = TracerProvider(shutdown_on_exit=False)
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        with provider.get_tracer("daydream-test").start_as_current_span(
            "attempt",
            attributes={
                "daydream.span.kind": "attempt",
                "daydream.billing.owner": "structural_attempt",
                "daydream.run.id": "run-abc",
                "daydream.flow": "review",
                "gen_ai.conversation.id": "native-conversation-7",
            },
        ):
            pass
        provider.shutdown()
    native = attributes(receiver.spans[0])
    assert native["honeyhive.session_id"] == "run-abc"
    assert native["honeyhive.session_name"] == "daydream.review"
    assert native["honeyhive.session_auto_create"] is True
    assert native["gen_ai.conversation.id"] == "native-conversation-7"
    assert native["honeyhive.session_id"] != "native-conversation-7"
