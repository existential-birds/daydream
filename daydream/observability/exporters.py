"""Owned OTLP destinations and native aliases. Presets isolate auth/TLS; generic OTLP honors signal-
over-shared settings. Each runtime owns delivery state.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import grpc
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.util.types import AttributeValue

from daydream.observability.config import ObservabilityConfig, ObservabilityError, validate_endpoint
from daydream.observability.otlp_compat import (
    DeliveryLedger,
    GrpcBridge,
    build_http_transport,
)
from daydream.observability.privacy import PrivacyPolicy, diagnostic_scope

_PRESET_TIMEOUT_SECONDS = 5.0
_HTTP_COMPRESSIONS = {"none": None, "gzip": True}


def _header_value(setting: str, *, default: str | None = None, required: bool = True) -> str | None:
    value = os.environ.get(setting, default)
    if value is None or not value.strip():
        if not required:
            return None
        raise ObservabilityError(f"{setting} is required for this trace destination")
    if value != value.strip() or any(ord(char) < 32 or ord(char) > 255 or ord(char) == 127 for char in value):
        raise ObservabilityError(f"{setting} must be a valid HTTP header value without control characters")
    return value


def _snapshot_of(exporter: Any) -> dict[str, Any]:
    """Return the owned transport's delivery snapshot when it exposes one."""
    snapshot = getattr(exporter, "delivery_snapshot", None)
    if not callable(snapshot):
        raise ObservabilityError("The owned transport does not expose a delivery snapshot")
    return cast(dict[str, Any], snapshot())


def _preset_transport(
    endpoint: str,
    headers: dict[str, str],
) -> SpanExporter:
    """Build preset HTTP transport with explicit credentials/TLS and no ambient settings.

    Generic credential-provider rejection applies only to the generic resolver.
    """
    ledger = DeliveryLedger()
    try:
        transport = build_http_transport(
            endpoint=endpoint,
            headers=tuple(headers.items()),
            timeout_s=_PRESET_TIMEOUT_SECONDS,
            compression="none",
            trust_env=False,
            follow_redirects=False,
            environ={},
            ledger=ledger,
        )
    except ObservabilityError:
        raise
    except Exception:
        raise ObservabilityError(
            "Could not initialize trace destination HTTP transport; check its operator settings"
        ) from None
    return transport


def langsmith_exporter(config: ObservabilityConfig) -> SpanExporter:
    """LangSmith OTLP using its API key, project, region and optional workspace."""
    key = _header_value("LANGSMITH_API_KEY")
    project = _header_value("LANGSMITH_PROJECT", default="daydream")
    workspace = _header_value("LANGSMITH_WORKSPACE_ID", required=False)
    endpoint = validate_endpoint(
        os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"), "LANGSMITH_ENDPOINT"
    )
    assert key is not None and project is not None
    headers = {"x-api-key": key, "Langsmith-Project": project}
    if workspace is not None:
        headers["x-tenant-id"] = workspace
    return LangSmithExporter(_preset_transport(endpoint + "/otel/v1/traces", headers))


def honeyhive_exporter(config: ObservabilityConfig) -> SpanExporter:
    """HoneyHive v2 OTLP: deployment URL and a project-scoped API key."""
    key = _header_value("HH_API_KEY")
    base = os.environ.get("HH_API_URL", "")
    if not base:
        raise ObservabilityError("HH_API_URL is required; use your HoneyHive deployment's API base URL")
    endpoint = validate_endpoint(base, "HH_API_URL")
    return HoneyHiveExporter(
        _preset_transport(endpoint + "/opentelemetry/v1/traces", {"Authorization": f"Bearer {key}"})
    )


def _resolve_otlp_timeout(setting_traces: str, setting_shared: str) -> float:
    try:
        timeout = float(os.environ.get(setting_traces, os.environ.get(setting_shared, "10")))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError
    except ValueError:
        raise ObservabilityError(
            f"{setting_traces} / {setting_shared} must be a finite positive number of seconds"
        ) from None
    return timeout


def _otlp_http_headers_from_env(setting_traces: str, setting_shared: str) -> dict[str, str]:
    """Strict/literal OTEL header parsing: pairs split on '=', keys and values literal."""
    raw = os.environ.get(setting_traces, os.environ.get(setting_shared, "")).strip()
    headers: dict[str, str] = {}
    if not raw:
        return headers
    for pair in raw.split(","):
        name, separator, value = pair.partition("=")
        name = name.strip()
        value = value.strip()
        if not separator or not name or any(ord(char) < 32 or ord(char) == 127 for char in name + value):
            raise ObservabilityError(
                f"{setting_traces} / {setting_shared} must be 'key=value' pairs without control characters"
            )
        headers[name] = value
    return headers


def _otlp_compression_setting() -> Literal["none", "gzip"]:
    value = os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_COMPRESSION", os.environ.get("OTEL_EXPORTER_OTLP_COMPRESSION", "none")
    )
    if value not in _HTTP_COMPRESSIONS:
        raise ObservabilityError(
            "OTEL_EXPORTER_OTLP_TRACES_COMPRESSION / OTEL_EXPORTER_OTLP_COMPRESSION must be 'none' or 'gzip'"
        )
    return "gzip" if value == "gzip" else "none"


def _traces_or_shared_endpoint_setting() -> str:
    return (
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
        if "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" in os.environ
        else "OTEL_EXPORTER_OTLP_ENDPOINT"
    )


def otlp_exporter(config: ObservabilityConfig) -> SpanExporter:
    """Generic OTLP exporter honoring standard signal-specific and shared OTEL settings."""
    protocol = os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    )
    if protocol not in ("http/protobuf", "grpc"):
        raise ObservabilityError(
            "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL / OTEL_EXPORTER_OTLP_PROTOCOL must be 'http/protobuf' or 'grpc'"
        )
    endpoint_setting = _traces_or_shared_endpoint_setting()
    endpoint = os.environ.get(endpoint_setting)
    if endpoint is not None:
        validate_endpoint(endpoint, endpoint_setting, grpc=protocol == "grpc")
    timeout = _resolve_otlp_timeout("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT", "OTEL_EXPORTER_OTLP_TIMEOUT")
    if protocol == "grpc":
        return _grpc_generic_exporter(timeout, config)
    return _http_generic_exporter(timeout, config)


def _http_generic_exporter(timeout: float, config: ObservabilityConfig) -> SpanExporter:
    """Owned HTTP transport via the typed signal-over-shared resolver."""
    endpoint_setting = _traces_or_shared_endpoint_setting()
    raw_endpoint = os.environ.get(endpoint_setting, "http://localhost:4318")
    # Shared endpoints append /v1/traces even after an existing path; signal
    # endpoints stay verbatim (pinned OTel 1.44 behavior).
    is_signal_specific = endpoint_setting == "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    if is_signal_specific:
        endpoint = raw_endpoint
    else:
        endpoint = raw_endpoint.rstrip("/") + "/v1/traces"
    header_pairs = _otlp_http_headers_from_env("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "OTEL_EXPORTER_OTLP_HEADERS")
    headers = tuple(header_pairs.items())
    certificate = os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE", os.environ.get("OTEL_EXPORTER_OTLP_CERTIFICATE")
    )
    client_certificate = os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_CLIENT_CERTIFICATE", os.environ.get("OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE")
    )
    client_key = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_CLIENT_KEY", os.environ.get("OTEL_EXPORTER_OTLP_CLIENT_KEY"))
    if client_certificate and not client_key:
        raise ObservabilityError("A client certificate requires its client key")
    compression = _otlp_compression_setting()
    ledger = DeliveryLedger()
    with diagnostic_scope(PrivacyPolicy(capture_content=config.capture_content)):
        transport = build_http_transport(
            endpoint=endpoint,
            headers=headers,
            timeout_s=timeout,
            compression=compression,
            trust_env=True,  # generic OTLP keeps ambient proxy/CA behavior
            certificate_file=certificate,
            client_cert=client_certificate,
            client_key=client_key,
            environ=os.environ,
            ledger=ledger,
        )
    return transport


def _grpc_generic_exporter(timeout: float, config: ObservabilityConfig) -> SpanExporter:
    """Corrected scheme/insecure+compression resolution, then the pinned bridge."""
    endpoint_setting = _traces_or_shared_endpoint_setting()
    endpoint = os.environ.get(endpoint_setting)
    if endpoint is not None:
        validate_endpoint(endpoint, endpoint_setting, grpc=True)
        parsed = urlsplit(endpoint if "://" in endpoint else "//" + endpoint)
        if parsed.scheme == "http":
            insecure = True
        elif parsed.scheme == "https":
            insecure = False
        else:
            insecure = None
    else:
        parsed = urlsplit("http://localhost:4317")
        insecure = None
    compression = grpc.Compression.Gzip if _otlp_compression_setting() == "gzip" else grpc.Compression.NoCompression
    from opentelemetry.metrics import NoOpMeterProvider

    ledger = DeliveryLedger()
    with diagnostic_scope(PrivacyPolicy(capture_content=config.capture_content)):
        delegate = GrpcExporter(
            endpoint=endpoint,
            insecure=insecure,
            timeout=timeout,
            compression=compression,
            meter_provider=NoOpMeterProvider(),
        )
        bridge = GrpcBridge(delegate, ledger)
    return bridge


def _langsmith_usage(attributes: Mapping[str, AttributeValue]) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for portable, native in (
        ("gen_ai.usage.input_tokens", "input_tokens"),
        ("gen_ai.usage.output_tokens", "output_tokens"),
        ("gen_ai.usage.total_tokens", "total_tokens"),
        ("gen_ai.usage.cost", "total_cost"),
    ):
        if portable in attributes:
            usage[native] = attributes[portable]
    for group, entries in (
        (
            "input_token_details",
            (
                ("gen_ai.usage.cache_read.input_tokens", "cache_read"),
                ("gen_ai.usage.cache_creation.input_tokens", "cache_creation"),
            ),
        ),
        ("output_token_details", (("gen_ai.usage.reasoning.output_tokens", "reasoning"),)),
    ):
        details = {native: attributes[portable] for portable, native in entries if portable in attributes}
        if details:
            usage[group] = details
    return usage


def _copy_span(span: ReadableSpan, attributes: Mapping[str, AttributeValue]) -> ReadableSpan:
    """Copy spans while sharing untouched event/link containers, preserving ordering and dropped counts."""
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=attributes,
        events=getattr(span, "_events", span.events),
        links=getattr(span, "_links", span.links),
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def _preserved_attributes(
    span: ReadableSpan, additions: Mapping[str, AttributeValue]
) -> BoundedAttributes | dict[str, AttributeValue]:
    """Add keys without dropping existing attributes; preserve original bounded dropped counts."""
    original = getattr(span, "_attributes", None)
    if isinstance(original, BoundedAttributes):
        capacity = max(original.maxlen or 0, len(original) + len(additions))
        merged = BoundedAttributes(maxlen=capacity, attributes=dict(original.items()), immutable=False)
        merged.dropped = original.dropped
        for key, value in additions.items():
            merged[key] = value
        return merged
    plain: dict[str, AttributeValue] = dict(span.attributes or {})
    plain.update(additions)
    return plain


class _NativeSpanExporter(SpanExporter):
    """Shared delegation and billing-owner resolution for native exporters."""

    def __init__(self, exporter: SpanExporter) -> None:
        self._exporter = exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._exporter.export([self._adapt(span) for span in spans])

    @staticmethod
    def _adapt(span: ReadableSpan) -> ReadableSpan:
        raise NotImplementedError

    @staticmethod
    def _is_billed_owner(attributes: Mapping[str, AttributeValue]) -> bool:
        """Bill the resolved structural attempt or a generation explicitly marked billed.

        Unresolved/none ownership and other span kinds receive no native billing aliases.
        """
        kind = attributes.get("daydream.span.kind")
        if kind == "generation":
            return bool(attributes.get("daydream.generation.billed"))
        if kind == "attempt":
            return attributes.get("daydream.billing.owner") == "structural_attempt"
        return False

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def delivery_snapshot(self) -> dict[str, Any]:
        return _snapshot_of(self._exporter)


# Structural aggregates (run/step/logical agent/attempt) are chains in every
# vendor's native vocabulary and tool calls are tool events; only the approved
# generation kind differs per vendor.
_STRUCTURAL_EVENT_TYPES = {
    "run": "chain",
    "step": "chain",
    "agent": "chain",
    "attempt": "chain",
    "tool": "tool",
}


class HoneyHiveExporter(_NativeSpanExporter):
    """Add native event types and billed metadata without changing portable spans."""

    _EVENT_TYPES = {**_STRUCTURAL_EVENT_TYPES, "generation": "model"}

    @staticmethod
    def _adapt(span: ReadableSpan) -> ReadableSpan:
        attributes = dict(span.attributes or {})
        kind = attributes.get("daydream.span.kind")
        attributes["honeyhive_event_type"] = HoneyHiveExporter._EVENT_TYPES.get(str(kind), "chain")
        session_id = attributes.get("traceloop.association.properties.session_id") or attributes.get("daydream.run.id")
        if session_id:
            attributes["honeyhive.session_id"] = session_id
            attributes["honeyhive.session_auto_create"] = True
            attributes["honeyhive.session_name"] = f"daydream.{attributes.get('daydream.flow', 'run')}"
        # Standard gen_ai.agent identity already maps to native agent metadata.
        if HoneyHiveExporter._is_billed_owner(attributes):
            for portable, native in (
                ("gen_ai.usage.input_tokens", "prompt_tokens"),
                ("gen_ai.usage.output_tokens", "completion_tokens"),
                ("gen_ai.usage.cost", "cost"),
                ("gen_ai.usage.cache_read.input_tokens", "cache_read_input_tokens"),
                ("gen_ai.usage.cache_creation.input_tokens", "cache_write_input_tokens"),
                ("gen_ai.usage.reasoning.output_tokens", "reasoning_tokens"),
            ):
                if portable in attributes:
                    attributes[f"honeyhive_metadata.{native}"] = attributes[portable]
        return _copy_span(span, _preserved_attributes(span, attributes))


class LangSmithExporter(_NativeSpanExporter):
    """Add LangSmith compatibility attributes to copies; keep portable spans untouched."""

    _RUN_TYPES = {**_STRUCTURAL_EVENT_TYPES, "generation": "llm"}

    @staticmethod
    def _adapt(span: ReadableSpan) -> ReadableSpan:
        attributes = dict(span.attributes or {})
        kind = attributes.get("daydream.span.kind")
        attributes["langsmith.span.kind"] = LangSmithExporter._RUN_TYPES.get(str(kind), "chain")
        # Native agent type reflects only actual nested agent scopes.
        role = attributes.get("daydream.agent.role")
        if kind == "agent" and role in ("root", "subagent"):
            attributes["langsmith.metadata.ls_agent_type"] = role
        if LangSmithExporter._is_billed_owner(attributes):
            if kind == "attempt":
                # LangSmith's documented aggregation hook for the structural chain.
                attributes["langsmith.metadata.invocation_aggregate"] = True
            usage = _langsmith_usage(attributes)
            if usage:
                attributes["langsmith.usage_metadata"] = json.dumps(usage, separators=(",", ":"))
        return _copy_span(span, _preserved_attributes(span, attributes))
