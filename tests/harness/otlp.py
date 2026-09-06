"""Local OTLP/HTTP receiver for tests of the real exporter wire contract."""

from __future__ import annotations

import gzip
import threading
import zlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest


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
    *, status: int = 200, response_headers: Mapping[str, str] | None = None, reason: str | None = None,
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
