"""OTLP destination factories and the small LangSmith compatibility adapter.

The runtime owns each returned exporter. Presets isolate authentication and TLS
from generic OTLP environment settings; the generic destination honors the SDK's
standard HTTP/protobuf and gRPC transport configuration.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import requests
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcExporter
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HttpExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.util.types import AttributeValue

from daydream.observability.config import ObservabilityConfig, ObservabilityError
from daydream.observability.privacy import PrivacyPolicy, diagnostic_scope

_PRESET_TIMEOUT_SECONDS = 5.0


def _validated_endpoint(endpoint: str, setting: str, *, grpc: bool = False) -> str:
    """Validate URLs without including their potentially sensitive contents in errors."""
    try:
        parsed = urlsplit("//" + endpoint if grpc and "://" not in endpoint else endpoint)
        valid = (
            bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and not any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in endpoint)
            and (parsed.scheme in ("http", "https") or (grpc and not parsed.scheme))
            and (not grpc or parsed.path in ("", "/"))
        )
        # Accessing .port validates malformed and out-of-range ports too.
        parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ObservabilityError(
            f"{setting} must be a valid {'HTTP(S) URL or host:port' if grpc else 'HTTP(S) URL'} "
            "without credentials, query, fragment or whitespace"
        )
    return endpoint.rstrip("/")


def _header_value(setting: str, *, default: str | None = None, required: bool = True) -> str | None:
    value = os.environ.get(setting, default)
    if value is None or not value.strip():
        if not required:
            return None
        raise ObservabilityError(f"{setting} is required for this trace destination")
    if value != value.strip() or any(ord(char) < 32 or ord(char) > 255 or ord(char) == 127 for char in value):
        raise ObservabilityError(f"{setting} must be a valid HTTP header value without control characters")
    return value


class _PresetSession(requests.Session):
    """Owned preset transport: no netrc/proxy auth, client certs, or redirects."""

    def __init__(self) -> None:
        super().__init__()
        self.trust_env = False

    def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
        # OTel's constructor reads generic certificate env even when passed an
        # explicit session. Enforce the preset's transport policy at send time.
        kwargs.update(allow_redirects=False, verify=True, cert=None)
        response = super().send(request, **kwargs)
        if 300 <= response.status_code < 400:
            response.close()
            raise requests.exceptions.InvalidURL("Trace destination redirected; configure its final endpoint")
        return response


def _preset_exporter(endpoint: str, headers: dict[str, str], config: ObservabilityConfig) -> SpanExporter:
    session = _PresetSession()
    try:
        with diagnostic_scope(PrivacyPolicy(capture_content=config.capture_content)):
            return HttpExporter(
                endpoint=endpoint,
                headers=headers,
                session=session,
                timeout=_PRESET_TIMEOUT_SECONDS,
                compression=Compression.NoCompression,
            )
    except Exception:
        session.close()
        raise ObservabilityError(
            "Could not initialize trace destination HTTP transport; check its operator settings"
        ) from None


def langsmith_exporter(config: ObservabilityConfig) -> SpanExporter:
    """LangSmith OTLP using its API key, project, region and optional workspace."""
    key = _header_value("LANGSMITH_API_KEY")
    project = _header_value("LANGSMITH_PROJECT", default="daydream")
    workspace = _header_value("LANGSMITH_WORKSPACE_ID", required=False)
    endpoint = _validated_endpoint(
        os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"), "LANGSMITH_ENDPOINT"
    )
    assert key is not None and project is not None
    headers = {"x-api-key": key, "Langsmith-Project": project}
    if workspace is not None:
        headers["x-tenant-id"] = workspace
    return LangSmithExporter(_preset_exporter(endpoint + "/otel/v1/traces", headers, config))


def honeyhive_exporter(config: ObservabilityConfig) -> SpanExporter:
    """HoneyHive v2 OTLP: deployment URL and a project-scoped API key."""
    key = _header_value("HH_API_KEY")
    base = os.environ.get("HH_API_URL", "")
    if not base:
        raise ObservabilityError("HH_API_URL is required; use your HoneyHive deployment's API base URL")
    endpoint = _validated_endpoint(base, "HH_API_URL")
    return _preset_exporter(endpoint + "/opentelemetry/v1/traces", {"Authorization": f"Bearer {key}"}, config)


def otlp_exporter(config: ObservabilityConfig) -> SpanExporter:
    """Generic OTLP exporter honoring standard signal-specific and shared OTEL settings."""
    protocol = os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    )
    if protocol not in ("http/protobuf", "grpc"):
        raise ObservabilityError(
            "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL / OTEL_EXPORTER_OTLP_PROTOCOL must be 'http/protobuf' or 'grpc'"
        )
    endpoint_setting = (
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
        if "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" in os.environ
        else "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    endpoint = os.environ.get(endpoint_setting)
    if endpoint is not None:
        _validated_endpoint(endpoint, endpoint_setting, grpc=protocol == "grpc")
    try:
        timeout = float(
            os.environ.get("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT", os.environ.get("OTEL_EXPORTER_OTLP_TIMEOUT", "10"))
        )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError
    except ValueError:
        raise ObservabilityError(
            "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT / OTEL_EXPORTER_OTLP_TIMEOUT must be a finite positive number of seconds"
        ) from None
    try:
        with diagnostic_scope(PrivacyPolicy(capture_content=config.capture_content)):
            return GrpcExporter(timeout=timeout) if protocol == "grpc" else HttpExporter(timeout=timeout)
    except Exception:
        raise ObservabilityError(
            "Could not initialize OTLP transport; check OTEL exporter headers, TLS, "
            "compression and credential-provider settings"
        ) from None


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


class LangSmithExporter(SpanExporter):
    """Add LangSmith compatibility attributes to copies; keep portable spans untouched."""

    def __init__(self, exporter: SpanExporter) -> None:
        self._exporter = exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._exporter.export([self._adapt(span) for span in spans])

    @staticmethod
    def _adapt(span: ReadableSpan) -> ReadableSpan:
        attributes = dict(span.attributes or {})
        kind = attributes.get("daydream.span.kind")
        attributes["langsmith.span.kind"] = "llm" if kind == "attempt" else "tool" if kind == "tool" else "chain"
        if kind == "attempt":
            attributes["langsmith.metadata.invocation_aggregate"] = True
            usage = _langsmith_usage(attributes)
            if usage:
                attributes["langsmith.usage_metadata"] = json.dumps(usage, separators=(",", ":"))
        return ReadableSpan(
            name=span.name,
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=attributes,
            events=span.events,
            links=span.links,
            kind=span.kind,
            status=span.status,
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._exporter.shutdown()
