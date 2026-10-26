"""Real HTTP/gRPC protobuf delivery against scripted loopback collectors.

Exercise acknowledgment, retry, whole-operation deadlines, and outcome accounting;
patch clocks and external failure seams rather than production decisions."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from email.utils import formatdate
from typing import cast

import grpc
import httpx
import pytest
from google.protobuf import any_pb2
from google.rpc import error_details_pb2  # type: ignore[import-untyped]  # proto-only, no stubs
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext

from daydream.observability.config import ObservabilityConfig, ObservabilityError
from daydream.observability.exporters import (
    CompatSpanExporter,
    langsmith_exporter,
    otlp_exporter,
)
from daydream.observability.otlp_compat import HttpxOtlpTransport, classify_http_ack
from tests.harness.otlp import (
    GrpcReceiver,
    ScriptedResponse,
    TrickleServer,
    grpc_trace_server,
    otlp_collector,
    scripted_otlp_collector,
)

# The private requests-session credential provider settings the whole-operation
# deadline architecture must reject (Task 0 spike gate 4, binding).
_CREDENTIAL_PROVIDER_VARS = (
    "_OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER", "_OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER",
)


def _spans(count: int = 1) -> list[ReadableSpan]:
    return [ReadableSpan(name=f"span-{index}",
            context=SpanContext(trace_id=0x11111111111111111111111111111111, span_id=index + 1, is_remote=False),
            resource=Resource({"service.name": "daydream-test"}), attributes={},
        )
        for index in range(count)
    ]


def _generic_http(monkeypatch: pytest.MonkeyPatch, base_url: str, **extra: str) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", base_url + "/v1/traces")
    for key, value in extra.items():
        monkeypatch.setenv(key, value)


def _partial_response(rejected: int) -> ScriptedResponse:
    response = ExportTraceServiceResponse()
    response.partial_success.rejected_spans = rejected
    return ScriptedResponse(body=response.SerializeToString())


# ------------------------------------------------------------------ HTTP acknowledgment matrix

def test_http_connection_refusal_is_sanitized_unverified_delivery(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    requests: list[httpx.Request] = []

    async def refuse(_transport: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectError("opaque-transport-secret", request=request)

    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)
    _generic_http(monkeypatch, "http://127.0.0.1:9",
                  OTEL_EXPORTER_OTLP_TRACES_HEADERS="x-private=opaque-transport-secret")
    exporter = cast(CompatSpanExporter, otlp_exporter(ObservabilityConfig()))
    try:
        start = time.monotonic()
        assert exporter.export(_spans()) == SpanExportResult.FAILURE
        assert time.monotonic() - start < 5
        assert len(requests) == 1
        assert requests[0].headers["x-private"] == "opaque-transport-secret"
        assert requests[0].headers["content-type"] == "application/x-protobuf"
        request = ExportTraceServiceRequest.FromString(requests[0].content)
        assert request.resource_spans[0].scope_spans[0].spans[0].name == "span-0"
        assert exporter.delivery_snapshot() == {
            "delivered": 0, "rejected": 0, "unverified": 1, "warning": False,
            "diagnostics": ["OTLP_CONNECTION_FAILED"],
        }
        assert "opaque-transport-secret" not in caplog.text
    finally:
        exporter.shutdown()

@pytest.mark.parametrize(("response", "warning", "rejected"), [
    pytest.param(ScriptedResponse(headers={}), False, None, id="empty-no-content-type"),
    pytest.param(
        ScriptedResponse(headers={"Content-Type": "Application/JSON; charset=utf-8"}, body=b'{"success": true}'),
        True, None, id="case-insensitive-media-type",
    ),
    pytest.param(_partial_response(0), True, 0, id="partial-zero-rejected"),
])
def test_http_success_ack_is_delivered_once(
    monkeypatch: pytest.MonkeyPatch, response: ScriptedResponse, warning: bool, rejected: int | None,
) -> None:
    """Empty 200 acknowledgments ignore media type; vendor/partial acknowledgments warn."""
    content_types: list[str | None] = []
    with scripted_otlp_collector([response], capture_content_type=content_types) as receiver:
        _generic_http(monkeypatch, receiver.base_url)
        exporter = cast(CompatSpanExporter, otlp_exporter(ObservabilityConfig()))
        assert exporter.export(_spans()) == SpanExportResult.SUCCESS
        snapshot = exporter.delivery_snapshot()
        assert snapshot["delivered"] == 1
        assert snapshot["unverified"] == 0
        assert snapshot["warning"] is warning
        if rejected is not None:
            assert snapshot["rejected"] == rejected
        exporter.shutdown()
        assert len(receiver.requests) == 1
        assert content_types == [response.headers.get("Content-Type")]


@pytest.mark.parametrize(("response", "secret"), [
    pytest.param(ScriptedResponse(body=b"\x00opaque-garbage-body\x01"), "opaque-garbage", id="undecodable-protobuf"),
    pytest.param(
        ScriptedResponse(headers={"Content-Type": "application/json"}, body=b'{"success": false, "error": "boom"}'),
        "boom", id="json-error",
    ),
])
def test_http_terminal_ack_is_unverified_without_logging_body(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    response: ScriptedResponse, secret: str | None,
) -> None:
    with scripted_otlp_collector([response]) as receiver:
        _generic_http(monkeypatch, receiver.base_url)
        exporter = cast(CompatSpanExporter, otlp_exporter(ObservabilityConfig()))
        assert exporter.export(_spans()) == SpanExportResult.FAILURE
        exporter.shutdown()
        assert len(receiver.requests) == 1
        snapshot = exporter.delivery_snapshot()
        assert snapshot["delivered"] == 0
        assert snapshot["unverified"] == 1
        if secret is not None:
            assert secret not in caplog.text


def test_http_oversized_body_is_bounded_discard(monkeypatch: pytest.MonkeyPatch) -> None:
    """A decoded acknowledgment over 4 MiB fails once and never retries."""
    big = b"\n\x02\x08\x03" + b"x" * (4 * 1024 * 1024)  # > 4 MiB decoded budget
    with scripted_otlp_collector([ScriptedResponse(body=big), ScriptedResponse()]) as receiver:
        _generic_http(monkeypatch, receiver.base_url)
        exporter = cast(CompatSpanExporter, otlp_exporter(ObservabilityConfig()))
        assert exporter.export(_spans()) == SpanExportResult.FAILURE
        exporter.shutdown()
        assert len(receiver.requests) == 1
        snapshot = exporter.delivery_snapshot()
        assert snapshot["delivered"] == 0 and snapshot["unverified"] == 1


def test_http_partial_success_positive_rejection_is_terminal_no_retry(monkeypatch: pytest.MonkeyPatch,) -> None:
    with scripted_otlp_collector([_partial_response(3), ScriptedResponse()]) as receiver:
        _generic_http(monkeypatch, receiver.base_url)
        exporter = cast(CompatSpanExporter, otlp_exporter(ObservabilityConfig()))
        assert exporter.export(_spans(count=5)) != SpanExportResult.SUCCESS
        exporter.shutdown()
        assert len(receiver.requests) == 1  # never retried
        snapshot = exporter.delivery_snapshot()
        assert snapshot["rejected"] == 3
        assert snapshot["delivered"] == 0

@pytest.mark.parametrize(("status", "retryable"), [(408, False), (500, False), (401, False)])
def test_http_retry_only_for_429_502_503_504(monkeypatch: pytest.MonkeyPatch, status: int, retryable: bool) -> None:
    with scripted_otlp_collector([ScriptedResponse(status=status), ScriptedResponse()]) as receiver:
        _generic_http(monkeypatch, receiver.base_url)
        exporter = cast(CompatSpanExporter, otlp_exporter(ObservabilityConfig()))
        result = exporter.export(_spans())
        assert result == (SpanExportResult.SUCCESS if retryable else SpanExportResult.FAILURE), status
        exporter.shutdown()
        assert len(receiver.requests) == (2 if retryable else 1), status

def test_http_retry_after_seconds_and_http_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """One owned retry loop preserves the batch through every admitted response class."""
    responses = [ScriptedResponse(status=429, headers={"Retry-After": "0"}), ScriptedResponse(status=502),
        ScriptedResponse(status=503, headers={"Retry-After": formatdate(time.time(), usegmt=True)}),
        ScriptedResponse(status=504), ScriptedResponse()]
    with scripted_otlp_collector(responses) as receiver:
        _generic_http(monkeypatch, receiver.base_url, OTEL_EXPORTER_OTLP_TRACES_TIMEOUT="5")
        exporter = cast(CompatSpanExporter, otlp_exporter(ObservabilityConfig()))
        assert exporter.export(_spans()) == SpanExportResult.SUCCESS
        assert len(receiver.requests) == 5
        assert all(request["body"] == receiver.requests[0]["body"] for request in receiver.requests)
        snapshot = exporter.delivery_snapshot()
        assert snapshot["delivered"] == 1 and snapshot["unverified"] == 0
        exporter.shutdown()

def test_http_one_deadline_covers_send_read_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [ScriptedResponse(status=429, headers={"Retry-After": "30"}) for _ in range(3)]
    with scripted_otlp_collector(responses) as receiver:
        _generic_http(monkeypatch, receiver.base_url, OTEL_EXPORTER_OTLP_TRACES_TIMEOUT="0.5")
        exporter = cast(CompatSpanExporter, otlp_exporter(ObservabilityConfig()))
        start = time.monotonic()
        assert exporter.export(_spans()) == SpanExportResult.FAILURE
        elapsed = time.monotonic() - start
        exporter.shutdown()
        assert elapsed < 2.5  # bounded by the one deadline, not 30s of Retry-After
        assert len(receiver.requests) <= 2

def test_http_trickle_overruns_inactivity_timeout_but_respects_whole_deadline(monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The spike proof: per-read inactivity stays silent; the outer deadline fires."""
    server = TrickleServer(gap_s=0.04)
    try:
        _generic_http(monkeypatch, server.base_url, OTEL_EXPORTER_OTLP_TRACES_TIMEOUT="0.15")
        exporter = cast(CompatSpanExporter, otlp_exporter(ObservabilityConfig()))
        start = time.monotonic()
        result = exporter.export(_spans())
        elapsed = time.monotonic() - start
        exporter.shutdown()
        assert elapsed < 1.5  # returns at the deadline, not after the ~0.4s trickle
        assert result != SpanExportResult.SUCCESS
        # The peer observes the close once the transfer window elapses; bounded
        # grace instead of an unbounded wait on the server's shutdown path.
        deadline = time.monotonic() + 2.0
        while not server.peer_observed_close and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.peer_observed_close  # the client actually closed the connection
        snapshot = exporter.delivery_snapshot()
        assert snapshot["unverified"] == 1  # ambiguous post-send outcome
        assert snapshot["delivered"] == 0
    finally:
        server.close()

def test_http_presets_reach_the_destination_despite_ambient_proxy(monkeypatch: pytest.MonkeyPatch,) -> None:
    """Preset trust_env=False: HTTP(S)_PROXY/netrc never capture preset traffic."""
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    with otlp_collector() as receiver:
        monkeypatch.setenv("LANGSMITH_ENDPOINT", receiver.base_url)
        monkeypatch.setenv("LANGSMITH_API_KEY", "real-key")
        exporter = langsmith_exporter(ObservabilityConfig())
        assert exporter.export(_spans()) == SpanExportResult.SUCCESS
        exporter.shutdown()
        assert len(receiver.requests) == 1

def test_http_64mib_encode_bound_refuses_to_send(monkeypatch: pytest.MonkeyPatch) -> None:
    with otlp_collector() as receiver:
        _generic_http(monkeypatch, receiver.base_url)
        exporter = cast(CompatSpanExporter, otlp_exporter(ObservabilityConfig()))
        oversized = _spans(count=1)
        oversized[0] = ReadableSpan(name="huge", context=SpanContext(trace_id=1, span_id=1, is_remote=False),
            resource=Resource({"service.name": "daydream-test"}), attributes={"huge": "x" * (64 * 1024 * 1024 + 1024)},
        )
        assert exporter.export(oversized) == SpanExportResult.FAILURE
        exporter.shutdown()
        assert receiver.requests == []  # refused before any send


# ------------------------------------------------------------------ credential-provider rejection

def test_classify_http_ack_incomplete_empty_read_is_not_success() -> None:
    verdict, rejected = classify_http_ack(status=200, content_type=None, body=b"", complete=False)
    assert (verdict, rejected) == ("oversized", 0)

def test_classify_http_ack_none_body_is_not_success() -> None:
    verdict, rejected = classify_http_ack(status=200, content_type="application/x-protobuf", body=None, complete=True)
    assert (verdict, rejected) == ("malformed", 0)

def test_classify_http_ack_json_without_positive_success_is_malformed() -> None:
    verdict, rejected = classify_http_ack(
        status=200, content_type="application/json", body=b'{"ok": false}', complete=True
    )
    assert (verdict, rejected) == ("malformed", 0)

@pytest.mark.parametrize("env_var", _CREDENTIAL_PROVIDER_VARS)
def test_private_credential_provider_rejected_before_any_send(monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
    with scripted_otlp_collector([ScriptedResponse()]) as receiver:
        _generic_http(monkeypatch, receiver.base_url)
        monkeypatch.setenv(env_var, "tests.fixtures.opaque_provider:make_session")
        with pytest.raises(ObservabilityError):
            otlp_exporter(ObservabilityConfig())
        assert receiver.requests == []  # fail closed before client construction/send


# ------------------------------------------------------------------ gRPC matrix


@contextmanager
def _generic_grpc(
    monkeypatch: pytest.MonkeyPatch, receive: GrpcReceiver, **extra: str,
) -> Iterator[CompatSpanExporter]:
    with grpc_trace_server(receive) as port:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "grpc")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"http://127.0.0.1:{port}")
        for key, value in extra.items():
            monkeypatch.setenv(key, value)
        exporter = otlp_exporter(ObservabilityConfig())
        assert isinstance(exporter, CompatSpanExporter)
        try:
            yield exporter
        finally:
            exporter.shutdown()


@pytest.mark.parametrize("persistent", [False, True], ids=["recovers", "retry-exhausted"])
def test_grpc_resource_exhausted_retries_only_with_valid_retry_info(
    monkeypatch: pytest.MonkeyPatch, persistent: bool,
) -> None:
    """Valid RetryInfo allows one retry; repeated exhaustion cannot enter SDK backoff."""
    attempts = {"n": 0}

    def receive(request: ExportTraceServiceRequest, context: grpc.ServicerContext) -> ExportTraceServiceResponse:
        attempts["n"] += 1
        if attempts["n"] == 1 or persistent:
            retry = error_details_pb2.RetryInfo()
            retry.retry_delay.FromSeconds(0)
            detail = any_pb2.Any()
            detail.Pack(retry)
            context.send_initial_metadata(())
            context.set_trailing_metadata((("grpc-status-details-bin", detail.SerializeToString()),))
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "exhausted")
        return ExportTraceServiceResponse()

    with _generic_grpc(monkeypatch, receive) as exporter:
        expected = SpanExportResult.FAILURE if persistent else SpanExportResult.SUCCESS
        assert exporter.export(_spans()) == expected
        exporter.shutdown()
        assert attempts["n"] == 2  # one owned retry under valid RetryInfo
        snapshot = exporter.delivery_snapshot()
        assert snapshot["delivered"] == (0 if persistent else 1)
        assert snapshot["unverified"] == (1 if persistent else 0)
        assert snapshot["diagnostics"] == (["OTLP_GRPC_RETRY_BUDGET_EXHAUSTED"] if persistent else [])

@pytest.mark.parametrize(
    ("status", "message"), [(grpc.StatusCode.RESOURCE_EXHAUSTED, "exhausted"), (grpc.StatusCode.UNAVAILABLE, "down")],
)
def test_grpc_terminal_status_is_not_retried(monkeypatch: pytest.MonkeyPatch, status: grpc.StatusCode, message: str
) -> None:
    attempts = {"n": 0}

    def receive(request: ExportTraceServiceRequest, context: grpc.ServicerContext) -> ExportTraceServiceResponse:
        attempts["n"] += 1
        context.abort(status, message)
        return ExportTraceServiceResponse()  # pragma: no cover

    with _generic_grpc(monkeypatch, receive) as exporter:
        assert exporter.export(_spans()) != SpanExportResult.SUCCESS
        exporter.shutdown()
        assert attempts["n"] == 1  # owned policy: single Export per batch, no stock retry loop

def test_grpc_scheme_precedence_over_insecure_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    def receive(request: ExportTraceServiceRequest, context: grpc.ServicerContext) -> ExportTraceServiceResponse:
        return ExportTraceServiceResponse()

    with _generic_grpc(monkeypatch, receive, OTEL_EXPORTER_OTLP_INSECURE="false") as exporter:
        assert exporter.export(_spans()) == SpanExportResult.SUCCESS  # explicit http:// won


# ------------------------------------------------------------------ shared lifecycle

def test_exactly_once_shutdown_refuses_new_sends(monkeypatch: pytest.MonkeyPatch) -> None:
    with otlp_collector() as receiver:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", receiver.base_url + "/otlp-prefix")
        exporter = cast(CompatSpanExporter, otlp_exporter(ObservabilityConfig()))
        try:
            assert exporter.export(_spans()) == SpanExportResult.SUCCESS
            assert len(receiver.spans) == 1
            assert receiver.requests[-1]["path"] == "/otlp-prefix/v1/traces"
            # Reach into the transport seam the shutdown contract owns.
            transport = exporter._transport
            assert isinstance(transport, HttpxOtlpTransport)
            client = transport._client
            assert client is not None
        finally:
            exporter.shutdown()
        assert client.is_closed
        assert transport._state == "CLOSED"
        exporter.shutdown()  # idempotent, no error
        assert transport._state == "CLOSED"
        assert exporter.export(_spans()) == SpanExportResult.FAILURE  # CLOSED refuses sends
        assert len(receiver.requests) == 1

def test_pre_send_deadline_exhaustion_records_unverified(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both retry-backoff exhaustion and a later zero-budget send must retain exact ledger diagnostics."""
    with scripted_otlp_collector([ScriptedResponse(status=503)]) as receiver:
        _generic_http(monkeypatch, receiver.base_url, OTEL_EXPORTER_OTLP_TRACES_TIMEOUT="0.3")
        exporter = cast(CompatSpanExporter, otlp_exporter(ObservabilityConfig()))
        try:
            exporter.export(_spans())
            snapshot = exporter.delivery_snapshot()
            assert snapshot["delivered"] == 0
            assert snapshot["unverified"] >= 1
            assert "OTLP_RETRY_BUDGET_EXHAUSTED" in snapshot["diagnostics"]
            assert exporter.force_flush() is True
            exporter.export(_spans())
            snapshot = exporter.delivery_snapshot()
            assert snapshot["unverified"] >= 2
            assert "OTLP_RETRY_BUDGET_EXHAUSTED" in snapshot["diagnostics"]
        finally:
            exporter.shutdown()
