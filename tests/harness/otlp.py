"""Local OTLP/HTTP receiver for tests of the real exporter wire contract."""

from __future__ import annotations

import gzip
import threading
import time
import zlib
from collections.abc import Iterator, Mapping
from concurrent import futures
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
    TraceServiceServicer,
    add_TraceServiceServicer_to_server,
)


class ScriptedResponse:
    """One scripted HTTP response; the default body is the empty protobuf ack."""

    def __init__(
        self,
        *,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
        body: bytes = b"",
        reason: str | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.status = status
        self.headers = dict(headers or {})
        self.body = body
        self.reason = reason
        self.delay_s = delay_s


@dataclass
class TraceCollector:
    """Captured requests and decoded spans from a loopback collector."""

    base_url: str = ""
    status: int = 200
    requests: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def spans(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                span
                for request in self.requests
                for resource in request["body"].get("resourceSpans", [])
                for scope in resource.get("scopeSpans", [])
                for span in scope.get("spans", [])
            ]

    def capture(self, path: str, headers: dict[str, str], body: bytes) -> None:
        encoding = headers.get("content-encoding")
        if encoding == "gzip":
            body = gzip.decompress(body)
        elif encoding == "deflate":
            body = zlib.decompress(body)
        payload = ExportTraceServiceRequest()
        payload.ParseFromString(body)
        with self._lock:
            self.requests.append({"path": path, "headers": headers, "body": MessageToDict(payload)})


@contextmanager
def otlp_collector(
    *,
    status: int = 200,
    response_headers: Mapping[str, str] | None = None,
    reason: str | None = None,
) -> Iterator[TraceCollector]:
    """Receive real protobuf exports; close all listener resources on exit."""
    collector = TraceCollector(status=status)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            collector.capture(self.path, {key.lower(): value for key, value in self.headers.items()}, body)
            self.send_response(collector.status, message=reason)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", "0")
            for key, value in (response_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            """Keep test output free of HTTP request logs."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    collector.base_url = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield collector
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@contextmanager
def scripted_otlp_collector(
    responses: list[ScriptedResponse],
    *,
    capture_content_type: list[str | None] | None = None,
) -> Iterator[TraceCollector]:
    """Collector serving scripted responses in order; extra requests get the last one.

    The plain ``otlp_collector`` always answers 200 with an empty protobuf body,
    which cannot exercise acknowledgment classification. This variant scripts
    the exact status/content-type/body sequence per request and records each
    request's Content-Type for the strict acknowledgment tests. Bodies that are
    valid protobuf requests are additionally decoded through the normal
    capture path so partial-success bodies stay observable.
    """

    collector = TraceCollector()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            headers = {key.lower(): value for key, value in self.headers.items()}
            if capture_content_type is not None:
                capture_content_type.append(self.headers.get("Content-Type"))
            response = responses[min(len(collector.requests), len(responses) - 1)]
            if response.delay_s > 0:
                time.sleep(response.delay_s)
            # One entry per request; decode only when the body is a protobuf request.
            entry: dict[str, Any] = {"path": self.path, "headers": headers, "body": None}
            if response.body:
                try:
                    payload = ExportTraceServiceRequest()
                    payload.ParseFromString(body)
                    entry["body"] = MessageToDict(payload)
                except Exception:
                    pass  # Non-protobuf bodies are never decodable requests.
            with collector._lock:
                collector.requests.append(entry)
            self.send_response(response.status, message=response.reason)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/x-protobuf"))
            self.send_header("Content-Length", str(len(response.body)))
            for key, value in response.headers.items():
                if key != "Content-Type":
                    self.send_header(key, value)
            self.end_headers()
            if response.body:
                self.wfile.write(response.body)

        def log_message(self, format: str, *args: Any) -> None:
            """Keep test output free of HTTP request logs."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    collector.base_url = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield collector
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


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

        class Handler(BaseHTTPRequestHandler):
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

            def log_message(self, format: str, *args: Any) -> None:
                """Keep test output free of HTTP request logs."""

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


def _value(value: dict[str, Any]) -> Any:
    if "intValue" in value:
        return int(value["intValue"])
    if "arrayValue" in value:
        return [_value(item) for item in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {item["key"]: _value(item["value"]) for item in value["kvlistValue"].get("values", [])}
    return next(iter(value.values()), None)


class _GrpcServicer(TraceServiceServicer):
    """Receive real protobuf ``Export`` calls and record request headers."""

    def __init__(self, collector: "TraceCollector", reject: bool) -> None:
        self._collector = collector
        self._reject = reject

    def Export(self, request: Any, context: Any) -> Any:
        metadata = {key.lower(): value for key, value in context.invocation_metadata()}
        self._collector.capture("/grpc", metadata, request.SerializeToString())
        if self._reject:
            context.abort(grpc.StatusCode.UNAVAILABLE, "scripted collector outage")
        return ExportTraceServiceRequest()


@contextmanager
def otlp_grpc_collector(*, reject: bool = False) -> Iterator[TraceCollector]:
    """Receive real gRPC OTLP exports on a loopback port (P18 Task 5).

    Serves the real ``opentelemetry.proto.collector.trace.v1`` TraceService
    over an insecure loopback channel so the generic gRPC destination is
    driven through its actual transport. Every call's metadata is recorded
    so tests can assert the exact wire headers that reached the child;
    ``reject=True`` answers UNAVAILABLE after capture so outage fail-open
    behavior can be observed end to end.
    """

    collector = TraceCollector()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    add_TraceServiceServicer_to_server(_GrpcServicer(collector, reject), server)  # type: ignore[no-untyped-call]  # generated grpc stub is untyped
    port = server.add_insecure_port("127.0.0.1:0")
    if port == 0:  # pragma: no cover - loopback bind failure is fatal
        raise RuntimeError("gRPC loopback server failed to bind")
    server.start()
    collector.base_url = f"127.0.0.1:{port}"
    try:
        yield collector
    finally:
        server.stop(grace=None).wait(timeout=5)

