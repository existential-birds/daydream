"""Bounded OTLP transport compatibility boundary (P18 Task 4A).

Implements the Task-0-frozen composition contract: the pinned SDK encoder with
an owned protobuf fidelity repair, an HTTPX/AnyIO whole-operation-deadline HTTP
transport, a version-guarded gRPC delegate bridge, a shared acknowledgment
decision contract, and per-destination delivery outcome ledgers.

This is a bounded compatibility layer, not an SDK fork or global patch: no
``os.environ`` mutation, no global encoder replacement, and no nested retry
loops.
"""

from __future__ import annotations

import calendar
import gzip
import logging
import ssl
import threading
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any, Literal

import anyio
from google.protobuf.message import DecodeError
from google.rpc import error_details_pb2  # type: ignore[import-untyped]  # proto-only, no stubs
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans as sdk_encode_spans
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import TraceFlags

from daydream.observability.config import ObservabilityError

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Pinned OTLP protobuf flags masks (opentelemetry-proto 1.44.0).
from opentelemetry.proto.trace.v1.trace_pb2 import (  # noqa: E402
    SPAN_FLAGS_CONTEXT_HAS_IS_REMOTE_MASK,
    SPAN_FLAGS_CONTEXT_IS_REMOTE_MASK,
    SPAN_FLAGS_TRACE_FLAGS_MASK,
)

_FLAGS_HAS_IS_REMOTE = int(SPAN_FLAGS_CONTEXT_HAS_IS_REMOTE_MASK)
_FLAGS_IS_REMOTE = int(SPAN_FLAGS_CONTEXT_IS_REMOTE_MASK)
_FLAGS_TRACE_MASK = int(SPAN_FLAGS_TRACE_FLAGS_MASK)

#: Request/decode ceilings from the plan (uncompressed request; decoded ack).
_MAX_ENCODE_BYTES = 64 * 1024 * 1024
_MAX_DECODE_BYTES = 4 * 1024 * 1024

#: The private requests-session credential provider settings the deadline
#: architecture must reject (Task 0 spike gate 4). An arbitrary custom Session
#: cannot be translated to HTTPX without losing the whole-operation deadline.
_CREDENTIAL_PROVIDER_VARS = (
    "_OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER",
    "_OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER",
)

#: The exact OTel 1.44.0 gRPC delegate surface this bridge may touch. Drift is
#: an initialization failure, never a fallback to stock retry loops.
_GRPC_BRIDGE_SURFACE = ("_client", "_channel", "_headers", "_timeout", "_shutdown", "_initialize_channel_and_stub")

_ACK_OK = "ok"
_ACK_EMPTY_OK = "empty_ok"
_ACK_PARTIAL = "partial"
_ACK_MALFORMED = "malformed"
_ACK_OVERSIZED = "oversized"
_ACK_RETRYABLE = "retryable"

#: Fixed sanitized diagnostics; never reflect endpoint/header/exception text.
_VERDICT_DIAGNOSTIC = {
    _ACK_MALFORMED: "OTLP_MALFORMED_ACK",
    _ACK_OVERSIZED: "OTLP_OVERSIZED_ACK",
}

_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Delivery outcome ledger
# --------------------------------------------------------------------------


@dataclass
class DeliverySnapshot:
    """Per-destination observable delivery outcomes."""

    delivered: int = 0
    accepted: int = 0
    rejected: int = 0
    unverified: int = 0
    warning: bool = False
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "delivered": self.delivered,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "unverified": self.unverified,
            "warning": self.warning,
            "diagnostics": list(self.diagnostics),
        }


class DeliveryLedger:
    """Thread-safe per-destination delivered/accepted/rejected/unverified record."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = DeliverySnapshot()

    def record_delivered(self, *, accepted: int, warning: bool = False) -> None:
        with self._lock:
            self._snapshot.delivered += 1
            self._snapshot.accepted += accepted
            self._snapshot.warning = self._snapshot.warning or warning

    def record_rejected(self, rejected: int, accepted: int) -> None:
        with self._lock:
            self._snapshot.rejected += rejected
            self._snapshot.accepted += accepted

    def record_unverified(self, diagnostic: str) -> None:
        with self._lock:
            self._snapshot.unverified += 1
            self._snapshot.diagnostics = (*self._snapshot.diagnostics[-7:], diagnostic)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot.as_dict()


# --------------------------------------------------------------------------
# Wire codec: pinned encoder + bounded fidelity repair
# --------------------------------------------------------------------------


def encode_batch(spans: Sequence[ReadableSpan]) -> ExportTraceServiceRequest:
    """Encode once with the pinned SDK encoder, then repair proved losses.

    Repairs only the verified protobuf fidelity gaps in OTel 1.44.0: the low
    trace flags bits (sampled) on spans and links, and link trace state. Every
    other field is left exactly as the pinned encoder wrote it. One-to-one
    span/link identity and position are validated before any repair.
    """
    request = sdk_encode_spans(list(spans))
    wire_spans = [
        wire_span for resource in request.resource_spans for scope in resource.scope_spans for wire_span in scope.spans
    ]
    if len(wire_spans) != len(spans):
        raise ObservabilityError("OTLP encoder changed span count; refusing to repair a mismatched batch")
    for source, wire in zip(spans, wire_spans, strict=True):
        context = source.get_span_context()
        if context is None:
            raise ObservabilityError("OTLP encoder dropped span context; refusing a mismatched batch")
        if wire.trace_id != context.trace_id.to_bytes(16, "big") or wire.span_id != context.span_id.to_bytes(8, "big"):
            raise ObservabilityError("OTLP encoder changed span identity; refusing to repair a mismatched batch")
        _repair_flags(wire, context.trace_flags or TraceFlags(0x00), source.parent)
        if len(wire.links) != len(source.links):
            raise ObservabilityError("OTLP encoder changed link count; refusing to repair a mismatched batch")
        for source_link, wire_link in zip(source.links, wire.links, strict=True):
            if wire_link.trace_id != source_link.context.trace_id.to_bytes(
                16, "big"
            ) or wire_link.span_id != source_link.context.span_id.to_bytes(8, "big"):
                raise ObservabilityError("OTLP encoder changed link identity; refusing to repair a mismatched batch")
            _repair_link(wire_link, source_link)
    return request


def _repair_flags(wire_span: Any, trace_flags: Any, parent: Any) -> None:
    """Restore the full 8-bit trace flags alongside the remote-parent masks."""
    flags = _FLAGS_HAS_IS_REMOTE
    if parent is not None and parent.is_remote:
        flags |= _FLAGS_IS_REMOTE
    flags |= int(trace_flags) & _FLAGS_TRACE_MASK
    wire_span.flags = flags


def _repair_link(wire_link: Any, source_link: Any) -> None:
    flags = _FLAGS_HAS_IS_REMOTE
    if source_link.context.is_remote:
        flags |= _FLAGS_IS_REMOTE
    flags |= int(source_link.context.trace_flags) & _FLAGS_TRACE_MASK
    wire_link.flags = flags
    state = source_link.context.trace_state
    if state is not None:
        wire_link.trace_state = ",".join(f"{key}={value}" for key, value in state.items())


# --------------------------------------------------------------------------
# Acknowledgment decision contract (pure, shared by HTTP and gRPC)
# --------------------------------------------------------------------------


def classify_http_ack(
    *,
    status: int,
    content_type: str | None,
    body: bytes | None,
    complete: bool,
) -> tuple[str, int, int]:
    """Classify one HTTP acknowledgment. Returns (verdict, accepted, rejected).

    Binding decision 8: HTTP 200 with the protobuf content type and a complete
    zero-byte body is the canonical full success. Positive rejection is a
    terminal partial result; zero rejection with a warning is accepted with
    warning. Neither partial form is ever retried.
    """
    if status != 200:
        return (_ACK_MALFORMED, 0, 0)
    if content_type is None or content_type.split(";")[0].strip() != "application/x-protobuf":
        return (_ACK_MALFORMED, 0, 0)
    if body is None or not complete:
        return (_ACK_OVERSIZED if body is not None else _ACK_MALFORMED, 0, 0)
    if len(body) > _MAX_DECODE_BYTES:
        return (_ACK_OVERSIZED, 0, 0)
    if not body:
        return (_ACK_EMPTY_OK, 0, 0)
    response = ExportTraceServiceResponse()
    try:
        response.ParseFromString(body)
    except DecodeError:
        return (_ACK_MALFORMED, 0, 0)
    rejected = int(response.partial_success.rejected_spans)
    if rejected > 0:
        return (_ACK_PARTIAL, 0, rejected)
    if response.HasField("partial_success"):
        return (_ACK_PARTIAL, 0, 0)  # zero-rejected warning form
    return (_ACK_OK, 0, 0)


def classify_grpc_ack(payload: ExportTraceServiceResponse | None) -> tuple[str, int, int]:
    """Classify a decoded gRPC acknowledgment payload."""
    if payload is None:
        return (_ACK_OK, 0, 0)
    rejected = int(payload.partial_success.rejected_spans)
    if rejected > 0:
        return (_ACK_PARTIAL, 0, rejected)
    if payload.HasField("partial_success"):
        return (_ACK_PARTIAL, 0, 0)
    return (_ACK_OK, 0, 0)


_VERDICT_RESULT = {
    _ACK_OK: SpanExportResult.SUCCESS,
    _ACK_EMPTY_OK: SpanExportResult.SUCCESS,
    _ACK_PARTIAL: SpanExportResult.FAILURE,
    _ACK_MALFORMED: SpanExportResult.FAILURE,
    _ACK_OVERSIZED: SpanExportResult.FAILURE,
    _ACK_RETRYABLE: SpanExportResult.FAILURE,
}


def verdict_result(verdict: str) -> SpanExportResult:
    """Map an acknowledgment verdict to its non-retry SpanExportResult."""
    return _VERDICT_RESULT[verdict]


def parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    """Retry-After seconds or HTTP-date; None when absent/unparseable."""
    if not value:
        return None
    text = value.strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            target = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        moment = now if now is not None else time.time()
        delay = calendar.timegm(target.utctimetuple()) - moment
        return max(0.0, delay)
    if seconds < 0:
        return None
    return seconds


# --------------------------------------------------------------------------
# HTTP transport: HTTPX/AnyIO whole-operation deadline composition
# --------------------------------------------------------------------------


def reject_credential_provider_settings(environ: Any) -> None:
    """Fail closed when the private requests-session provider is configured."""
    for name in _CREDENTIAL_PROVIDER_VARS:
        if environ.get(name):
            raise ObservabilityError(
                "The private OTLP requests-session credential provider cannot preserve the "
                "whole-operation export deadline; configure explicit OTLP headers, CA, "
                "client certificate, endpoint or proxy settings instead"
            )


@dataclass(frozen=True)
class HttpTransportConfig:
    """Resolved, immutable HTTP transport configuration."""

    endpoint: str
    headers: tuple[tuple[str, str], ...]
    timeout_s: float
    compression: Literal["none", "gzip"]
    trust_env: bool
    verify: bool | ssl.SSLContext = True
    cert: tuple[str, str] | str | None = None
    follow_redirects: bool = False


def resolve_ssl_context(
    *,
    certificate_file: str | None,
    client_cert: str | None,
    client_key: str | None,
) -> ssl.SSLContext | bool:
    """Build the one SSLContext for the resolved CA and client chain."""
    if certificate_file is None and client_cert is None:
        return True
    context = ssl.create_default_context(cafile=certificate_file)
    if client_cert is not None:
        context.load_cert_chain(client_cert, client_key)
    return context


class HttpxOtlpTransport:
    """Owned HTTP transport with one whole-operation deadline per export.

    One persistent AnyIO BlockingPortal and one HTTPX AsyncClient are created
    before batches and closed exactly once at shutdown. Each attempt encloses
    send, the bounded decoded response read, and backoff inside one outer
    ``fail_after(remaining)`` scope. A post-deadline ambiguous delivery is
    recorded as unverified and never retried.
    """

    def __init__(
        self,
        config: HttpTransportConfig,
        ledger: DeliveryLedger,
        *,
        encode_bound: int = _MAX_ENCODE_BYTES,
        decode_bound: int = _MAX_DECODE_BYTES,
        backoff_s: float = 0.05,
        clock: Any = None,
    ) -> None:
        import httpx

        self._config = config
        self._ledger = ledger
        self._encode_bound = encode_bound
        self._decode_bound = decode_bound
        self._backoff_s = backoff_s
        self._clock = clock or time.monotonic
        self._client_factory = httpx.AsyncClient
        self._state = "OPEN"
        self._lock = threading.Lock()
        self._portal_cm: Any = None
        self._portal: Any = None
        self._client: Any = None
        self._portal_cm = anyio.from_thread.start_blocking_portal()
        self._portal = self._portal_cm.__enter__()
        self._client = self._portal.call(self._create_client)

    def _create_client(self) -> Any:
        return self._client_factory(
            headers=dict(self._config.headers),
            timeout=self._config.timeout_s,
            verify=self._config.verify,
            cert=self._config.cert,
            trust_env=self._config.trust_env,
            follow_redirects=self._config.follow_redirects,
        )

    @property
    def state(self) -> str:
        return self._state

    def export(self, request: ExportTraceServiceRequest, *, timeout_s: float) -> SpanExportResult:
        """One owned export: encode/compress inside the budget, then attempt."""
        if self._state != "OPEN":
            return SpanExportResult.FAILURE
        payload = request.SerializeToString()
        if len(payload) > self._encode_bound:
            self._ledger.record_unverified("OTLP_ENCODE_BOUND_EXCEEDED")
            return SpanExportResult.FAILURE
        body = gzip.compress(payload) if self._config.compression == "gzip" else payload
        remaining = timeout_s - (self._clock() - self._start)
        if remaining <= 0:
            self._ledger.record_unverified("OTLP_ENCODE_DEADLINE_EXHAUSTED")
            return SpanExportResult.FAILURE
        try:
            result: SpanExportResult = self._portal.call(self._attempt_loop, body, remaining)
            return result
        except anyio.WouldBlock:  # pragma: no cover - portal lifecycle guard
            return SpanExportResult.FAILURE

    def export_batch(self, spans: Sequence[ReadableSpan], *, timeout_s: float) -> SpanExportResult:
        """Encode inside the monotonic budget, then run the attempt loop."""
        if self._state != "OPEN":
            return SpanExportResult.FAILURE
        self._start = self._clock()
        request = encode_batch(spans)
        return self.export(request, timeout_s=timeout_s)

    async def _attempt_loop(self, body: bytes, remaining: float) -> SpanExportResult:
        deadline = anyio.current_time() + remaining
        attempt = 0
        last: SpanExportResult = SpanExportResult.FAILURE
        while True:
            now = anyio.current_time()
            left = deadline - now
            if left <= 0:
                # Encode/budget exhaustion before (or between) sends: record
                # the outcome like every other failure exit instead of
                # dropping it silently from the delivery ledger.
                self._ledger.record_unverified(
                    "OTLP_RETRY_BUDGET_EXHAUSTED" if attempt else "OTLP_DEADLINE_EXCEEDED_BEFORE_SEND"
                )
                return last
            try:
                with anyio.fail_after(left):
                    verdict, accepted, rejected, retry_after = await self._attempt(body)
            except TimeoutError:
                self._ledger.record_unverified("OTLP_DEADLINE_EXCEEDED_AFTER_SEND")
                return SpanExportResult.FAILURE  # ambiguous: no retry
            except Exception:  # noqa: BLE001 - sanitized transport failure seam
                self._ledger.record_unverified("OTLP_CONNECTION_FAILED")
                return SpanExportResult.FAILURE
            if verdict in (_ACK_OK, _ACK_EMPTY_OK, _ACK_PARTIAL):
                if verdict == _ACK_PARTIAL:
                    if rejected > 0:
                        self._ledger.record_rejected(rejected, accepted)
                        return SpanExportResult.FAILURE  # terminal partial result
                    self._ledger.record_delivered(accepted=accepted, warning=True)
                    return SpanExportResult.SUCCESS  # accepted with warning, no retry
                self._ledger.record_delivered(accepted=accepted)
                return verdict_result(verdict)
            if verdict != _ACK_RETRYABLE:
                # Malformed/oversized acknowledgment: terminal, never retried.
                _logger.warning("Failed to export trace batch: %s", _VERDICT_DIAGNOSTIC[verdict])
                self._ledger.record_unverified(_VERDICT_DIAGNOSTIC[verdict])
                return verdict_result(verdict)
            # Retryable malformed/status path: exponential backoff with jitter
            # inside the same deadline, then reclassify as unverified on exit.
            last = SpanExportResult.FAILURE
            wait = self._backoff_s * (2**attempt)
            if retry_after is not None:
                wait = max(wait, min(retry_after, max(0.0, deadline - anyio.current_time())))
            if wait <= 0 or anyio.current_time() + wait >= deadline:
                self._ledger.record_unverified("OTLP_RETRY_BUDGET_EXHAUSTED")
                return SpanExportResult.FAILURE
            await anyio.sleep(wait)
            attempt += 1

    async def _attempt(self, body: bytes) -> tuple[str, int, int, float | None]:
        assert self._client is not None
        headers = {"Content-Type": "application/x-protobuf"}
        if self._config.compression == "gzip":
            headers["Content-Encoding"] = "gzip"
        async with self._client.stream("POST", self._config.endpoint, content=body, headers=headers) as response:
            complete = True
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > self._decode_bound + 1:
                    complete = False
                    break  # bounded discard; the context closes the response
                chunks.append(chunk)
            if not complete:
                return (_ACK_OVERSIZED, 0, 0, None)
            content_type = response.headers.get("Content-Type")
            status = response.status_code
        if status in (301, 302, 303, 307, 308):
            _logger.warning("Trace destination redirected; configure its final endpoint")
            return (_ACK_MALFORMED, 0, 0, None)
        if status == 200:
            verdict, accepted, rejected = classify_http_ack(
                status=status, content_type=content_type, body=b"".join(chunks), complete=True
            )
            return (verdict, accepted, rejected, None)
        if status in (429, 502, 503, 504):
            retry_after = parse_retry_after(response.headers.get("Retry-After"))
            return (_ACK_RETRYABLE, 0, 0, retry_after)
        return (_ACK_MALFORMED, 0, 0, None)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Nothing is buffered here; flush success is not delivery acceptance."""
        return True

    def shutdown(self) -> None:
        """OPEN -> CLOSING -> CLOSED exactly once; client closed via the portal."""
        with self._lock:
            if self._state != "OPEN":
                self._state = "CLOSED"
                return
            self._state = "CLOSING"
            client, portal_cm = self._client, self._portal_cm
            self._client = None
        try:
            if portal_cm is not None and client is not None:
                # Close the owned AsyncClient through the portal BEFORE the
                # portal stops: the pool's keep-alive connections must be
                # closed by the client itself, and once the portal's loop
                # stops it can no longer run async cleanup. A close failure
                # is logged and never blocks the exactly-once portal stop.
                try:
                    self._portal.call(client.aclose)
                except Exception:  # noqa: BLE001 - sanitized transport failure seam
                    _logger.warning("Failed to close owned OTLP HTTP client cleanly", exc_info=True)
                portal_cm.__exit__(None, None, None)
            elif portal_cm is not None:
                portal_cm.__exit__(None, None, None)
        finally:
            self._state = "CLOSED"


def build_http_transport(
    *,
    endpoint: str,
    headers: tuple[tuple[str, str], ...],
    timeout_s: float,
    compression: Literal["none", "gzip"],
    trust_env: bool,
    certificate_file: str | None = None,
    client_cert: str | None = None,
    client_key: str | None = None,
    follow_redirects: bool = False,
    environ: Any = None,
    ledger: DeliveryLedger | None = None,
) -> HttpxOtlpTransport:
    """Resolve the typed HTTP config and construct the owned transport."""
    reject_credential_provider_settings(environ if environ is not None else _default_environ())
    verify = resolve_ssl_context(certificate_file=certificate_file, client_cert=client_cert, client_key=client_key)
    cert = (client_cert, client_key) if client_cert and client_key else client_cert
    config = HttpTransportConfig(
        endpoint=endpoint,
        headers=headers,
        timeout_s=timeout_s,
        compression=compression,
        trust_env=trust_env,
        verify=verify,
        cert=cert,
        follow_redirects=follow_redirects,
    )
    return HttpxOtlpTransport(config, ledger or DeliveryLedger())


def _default_environ() -> Any:
    import os

    return os.environ


# --------------------------------------------------------------------------
# gRPC bridge over the pinned 1.44.0 delegate
# --------------------------------------------------------------------------


def _grpc_retry_delay(exc: Any) -> float | None:
    """Extract a valid RetryInfo retry_delay from a gRPC RpcError, if present."""
    try:
        metadata = dict(exc.trailing_metadata() or ())
    except Exception:  # noqa: BLE001
        return None
    packed = metadata.get("grpc-status-details-bin")
    if not packed:
        return None
    try:
        from google.protobuf import any_pb2

        detail = any_pb2.Any()
        detail.ParseFromString(packed)
        retry_info = error_details_pb2.RetryInfo()
        retry_info.ParseFromString(detail.value)
        delay = retry_info.retry_delay.seconds + retry_info.retry_delay.nanos / 1.0e9
    except Exception:  # noqa: BLE001 - malformed details are not recoverability evidence
        return None
    return delay if delay >= 0 else None


class GrpcBridge:
    """Fail-closed versioned bridge over the pinned OTel gRPC delegate.

    Uses only the frozen private surface and never calls the delegate's
    ``export()``/``_export()`` retry loop: each owned retry performs exactly one
    unary ``Export(remaining)``. Version/surface drift is an initialization
    failure, not a fallback.
    """

    def __init__(self, delegate: GrpcExporter, ledger: DeliveryLedger) -> None:
        import grpc

        self._grpc = grpc
        missing = [name for name in _GRPC_BRIDGE_SURFACE if not hasattr(delegate, name)]
        if missing:
            raise ObservabilityError(
                "The installed OpenTelemetry gRPC exporter does not expose the pinned "
                "1.44.0 bridge surface; refusing to fall back to unowned retries"
            )
        from importlib.metadata import version

        otel_version = version("opentelemetry-exporter-otlp-proto-grpc")
        if not otel_version.startswith("1.44"):
            raise ObservabilityError(
                f"OpenTelemetry gRPC exporter {otel_version} is outside the pinned 1.44 bridge surface"
            )
        self._delegate = delegate
        self._ledger = ledger
        self._state = "OPEN"
        self._attempt_budget = 1  # exactly one Export per owned retry decision

    @property
    def delegate(self) -> GrpcExporter:
        return self._delegate

    def export(self, spans: Sequence[ReadableSpan], *, timeout_s: float) -> SpanExportResult:
        """One owned export with a single-retry policy for RESOURCE_EXHAUSTED.

        Exactly one unary ``Export(remaining)`` per attempt, never the delegate's
        retry loop. Only RESOURCE_EXHAUSTED with a parseable RetryInfo earns one
        in-deadline retry; every other failure is terminal.
        """
        if self._state != "OPEN" or getattr(self._delegate, "_shutdown", False):
            return SpanExportResult.FAILURE
        request = encode_batch(spans)
        deadline = time.monotonic() + timeout_s
        attempt = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._ledger.record_unverified("OTLP_GRPC_DEADLINE_EXHAUSTED")
                return SpanExportResult.FAILURE
            client = self._delegate._client
            if client is None:
                self._ledger.record_unverified("OTLP_GRPC_CLIENT_UNAVAILABLE")
                return SpanExportResult.FAILURE
            try:
                payload = client.Export(
                    request=request,
                    metadata=tuple(self._delegate._headers),
                    timeout=remaining,
                )
            except self._grpc.RpcError as exc:
                # Valid recoverability evidence only: a parseable standard
                # RetryInfo in the richer gRPC status details.
                retry_delay = _grpc_retry_delay(exc)
                if exc.code() == self._grpc.StatusCode.RESOURCE_EXHAUSTED and retry_delay is not None:
                    attempt += 1
                    if attempt > 1 or retry_delay > remaining:
                        self._ledger.record_unverified("OTLP_GRPC_RETRY_BUDGET_EXHAUSTED")
                        return SpanExportResult.FAILURE
                    time.sleep(retry_delay)
                    continue  # exactly one owned retry with remaining budget
                self._ledger.record_unverified("OTLP_GRPC_RPC_FAILED")
                del exc
                return SpanExportResult.FAILURE
            verdict, accepted, rejected = classify_grpc_ack(payload if payload is not None else None)
            if verdict in (_ACK_OK, _ACK_EMPTY_OK):
                self._ledger.record_delivered(accepted=accepted)
                return SpanExportResult.SUCCESS
            if verdict == _ACK_PARTIAL:
                if rejected > 0:
                    self._ledger.record_rejected(rejected, accepted)
                    return SpanExportResult.FAILURE
                self._ledger.record_delivered(accepted=accepted, warning=True)
                return SpanExportResult.SUCCESS  # accepted with warning
            self._ledger.record_unverified("OTLP_GRPC_MALFORMED_ACK")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Close the then-current channel exactly once via the delegate."""
        if self._state == "CLOSED":
            return
        self._state = "CLOSED"
        self._delegate.shutdown()
