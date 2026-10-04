"""Local OTLP/HTTP receiver for tests of the real exporter wire contract."""

from __future__ import annotations

import gzip
import threading
import time
import zlib
from collections.abc import Callable, Iterator, Mapping
from concurrent import futures
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter


@contextmanager
def exporting_provider(*exporters: SpanExporter, **options: Any) -> Iterator[TracerProvider]:
    """Own synchronous span processors and always shut down their provider."""
    provider = TracerProvider(shutdown_on_exit=False, **options)
    try:
        for exporter in exporters:
            provider.add_span_processor(SimpleSpanProcessor(exporter))
        yield provider
    finally:
        provider.shutdown()


class ScriptedResponse:
    """One scripted HTTP response; the default body is the empty protobuf ack."""

    def __init__(self, *, status: int = 200, headers: Mapping[str, str] | None = None, body: bytes = b"",
        reason: str | None = None, delay_s: float = 0.0,
    ) -> None:
        # Default to the canonical protobuf ack content type; an explicitly empty
        # headers mapping sends no Content-Type at all (the LangSmith ack shape).
        self.headers = dict(headers) if headers is not None else {"Content-Type": "application/x-protobuf"}
        self.status = status
        self.body = body
        self.reason = reason
        self.delay_s = delay_s


@dataclass
class TraceCollector:
    """Captured requests and decoded spans from a loopback collector."""

    base_url: str = ""
    requests: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def spans(self) -> list[dict[str, Any]]:
        with self._lock:
            return [span
                for request in self.requests
                for resource in request["body"].get("resourceSpans", [])
                for scope in resource.get("scopeSpans", [])
                for span in scope.get("spans", [])
            ]

    def capture(self, path: str, headers: Mapping[str, str | bytes], body: bytes) -> None:
        encoding = headers.get("content-encoding")
        if encoding == "gzip":
            body = gzip.decompress(body)
        elif encoding == "deflate":
            body = zlib.decompress(body)
        # A body that is not a protobuf request (a scripted JSON or garbage ack,
        # a partially-success response) is still recorded, with body=None.
        decoded: dict[str, Any] | None = None
        try:
            payload = ExportTraceServiceRequest()
            payload.ParseFromString(body)
            decoded = MessageToDict(payload)
        except Exception:
            decoded = None
        with self._lock:
            self.requests.append({"path": path, "headers": headers, "body": decoded})


@contextmanager
def _loopback_http_server(handler: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    """Bind a loopback HTTP server, yield its base URL, and tear it down."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class _QuietHTTPHandler(BaseHTTPRequestHandler):
    """BaseHTTPRequestHandler that keeps test output free of request logs."""

    def log_message(self, format: str, *args: Any) -> None:
        pass


@contextmanager
def otlp_collector(*, status: int = 200, response_headers: Mapping[str, str] | None = None, reason: str | None = None,
) -> Iterator[TraceCollector]:
    """Receive real protobuf exports and answer one fixed acknowledgment."""
    ack = ScriptedResponse(
        status=status,
        headers={"Content-Type": "application/x-protobuf", **(response_headers or {})},
        reason=reason,
    )
    with scripted_otlp_collector([ack]) as collector:
        yield collector


@contextmanager
def scripted_otlp_collector(responses: list[ScriptedResponse], *, capture_content_type: list[str | None] | None = None,
) -> Iterator[TraceCollector]:
    """Collector serving scripted responses in order; extra requests get the last one.

    The plain ``otlp_collector`` always answers 200 with an empty protobuf body,
    which cannot exercise acknowledgment classification. This variant scripts
    the exact status/content-type/body sequence per request and records each
    ack's Content-Type for the strict acknowledgment tests. Bodies that are
    valid protobuf requests are additionally decoded through the normal
    capture path so partial-success bodies stay observable.
    """

    collector = TraceCollector()

    class Handler(_QuietHTTPHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            headers = {key.lower(): value for key, value in self.headers.items()}
            response = responses[min(len(collector.requests), len(responses) - 1)]
            if capture_content_type is not None:
                # Record the ack's Content-Type, exactly as the exporter observes it.
                capture_content_type.append(response.headers.get("Content-Type"))
            if response.delay_s > 0:
                time.sleep(response.delay_s)
            # One entry per request; a non-protobuf body is recorded with body=None.
            collector.capture(self.path, headers, body)
            self.send_response(response.status, message=response.reason)
            ack_content_type = response.headers.get("Content-Type")
            if ack_content_type is not None:
                self.send_header("Content-Type", ack_content_type)
            self.send_header("Content-Length", str(len(response.body)))
            for key, value in response.headers.items():
                if key != "Content-Type":
                    self.send_header(key, value)
            self.end_headers()
            if response.body:
                self.wfile.write(response.body)

    with _loopback_http_server(Handler) as collector.base_url:
        yield collector


class TrickleServer:
    """Loopback HTTP peer that trickles a response just inside inactivity timeouts.

    Each one-byte body chunk arrives ``gap_s`` apart, so a per-read inactivity
    timeout never fires while the whole transfer overruns any wall-clock budget
    smaller than ``gap_s * chunks``. ``peer_observed_close`` records that the
    client actually closed the connection (whole-operation deadline proof).
    """

    def __init__(self, *, gap_s: float = 0.04, chunks: int = 9) -> None:
        self.gap_s = gap_s
        self.chunks = chunks
        self.peer_observed_close = False
        self.headers_sent = threading.Event()
        self._close = threading.Event()
        host = "127.0.0.1"

        outer = self

        class Handler(_QuietHTTPHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-protobuf")
                    self.send_header("Content-Length", str(outer.chunks))
                    self.end_headers()
                    outer.headers_sent.set()
                    for _ in range(outer.chunks):
                        if outer._close.is_set():
                            outer.peer_observed_close = True
                            return
                        self.wfile.write(b"x")
                        self.wfile.flush()
                        outer._close.wait(outer.gap_s)
                    outer._close.wait(outer.gap_s * 3)
                    outer.peer_observed_close = outer._close.is_set()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    outer.peer_observed_close = True

        self._server = ThreadingHTTPServer((host, 0), Handler)
        self.base_url = f"http://{host}:{self._server.server_port}"
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._close.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


def attributes(span: dict[str, Any]) -> dict[str, Any]:
    """Decode OTLP JSON attributes into ordinary Python values."""
    return {item["key"]: _value(item["value"]) for item in span.get("attributes", [])}


def kind_of(spans: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Filter decoded spans by the ``daydream.span.kind`` attribute."""
    return [span for span in spans if attributes(span).get("daydream.span.kind") == kind]


def _value(value: dict[str, Any]) -> Any:
    if "intValue" in value:
        return int(value["intValue"])
    if "arrayValue" in value:
        return [_value(item) for item in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {item["key"]: _value(item["value"]) for item in value["kvlistValue"].get("values", [])}
    return next(iter(value.values()), None)


GrpcReceiver = Callable[[ExportTraceServiceRequest, grpc.ServicerContext], ExportTraceServiceResponse]


@contextmanager
def grpc_trace_server(receive: GrpcReceiver, *, workers: int = 1) -> Iterator[int]:
    """Serve a real loopback Export RPC; close its server and thread pool together."""
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        server = grpc.server(pool)
        server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
            "opentelemetry.proto.collector.trace.v1.TraceService",
            {"Export": grpc.unary_unary_rpc_method_handler(
                receive, request_deserializer=ExportTraceServiceRequest.FromString,
                response_serializer=ExportTraceServiceResponse.SerializeToString,
            )},
        ),))
        port = server.add_insecure_port("127.0.0.1:0")
        if port == 0:  # pragma: no cover - loopback bind failure is fatal
            raise RuntimeError("gRPC loopback server failed to bind")
        server.start()
        try:
            yield port
        finally:
            server.stop(grace=0).wait(timeout=5)


@contextmanager
def otlp_grpc_collector(*, reject: bool = False) -> Iterator[TraceCollector]:
    """Capture real gRPC requests/headers; optionally answer UNAVAILABLE after capture."""
    collector = TraceCollector()

    def receive(request: ExportTraceServiceRequest, context: grpc.ServicerContext) -> ExportTraceServiceResponse:
        metadata = {key.lower(): value for key, value in context.invocation_metadata()}
        collector.capture("/grpc", metadata, request.SerializeToString())
        if reject:
            context.abort(grpc.StatusCode.UNAVAILABLE, "scripted collector outage")
        return ExportTraceServiceResponse()

    with grpc_trace_server(receive, workers=2) as port:
        collector.base_url = f"127.0.0.1:{port}"
        yield collector
