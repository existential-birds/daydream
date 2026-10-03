"""Immutable operator tracing settings; secrets stay in the environment, never reviewed repository files."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import SpanExporter

_DESTINATION_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")


class ObservabilityError(ValueError):
    """An invalid operator tracing setting, reported before agent work."""


def validate_endpoint(endpoint: str, setting: str, *, grpc: bool = False) -> str:
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
    except (ValueError, AttributeError):
        valid = False
    if not valid:
        raise ObservabilityError(
            f"{setting} must be a valid {'HTTP(S) URL or host:port' if grpc else 'HTTP(S) URL'} "
            "without credentials, query, fragment or whitespace"
        )
    return endpoint.rstrip("/")


def validate_destination_name(name: str) -> None:
    """Reject ambiguous names without reflecting potentially secret input."""
    if not isinstance(name, str) or not _DESTINATION_NAME.fullmatch(name):
        raise ObservabilityError(
            "Trace destination names must start with a lowercase letter and contain only "
            "lowercase letters, digits, dots, underscores or hyphens (maximum 64 characters)"
        )


@dataclass(frozen=True)
class ObservabilityConfig:
    """Destinations selected by the operator; an empty tuple disables tracing."""

    destinations: tuple[str, ...] = ()
    capture_content: bool = True
    service_name: str = "daydream"

    def __post_init__(self) -> None:
        for name in self.destinations:
            validate_destination_name(name)
        if len(set(self.destinations)) != len(self.destinations):
            raise ObservabilityError("Trace destinations must be unique")
        if (
            not self.service_name.strip()
            or len(self.service_name) > 255
            or any(ord(char) < 32 or ord(char) == 127 for char in self.service_name)
        ):
            raise ObservabilityError(
                "OTEL_SERVICE_NAME must be nonempty, at most 255 characters, without control characters"
            )


class TraceExporterFactory(Protocol):
    """Build an owned exporter only at run start; runtime flushes/closes even partial setup.
    Registration and ext validate never call factories.
    """

    def __call__(self, config: ObservabilityConfig) -> SpanExporter: ...


def resolve_observability_config(
    *,
    destinations: Sequence[str] | None = None,
    disabled: bool = False,
    content: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ObservabilityConfig:
    """Resolve CLI-over-environment settings, with explicit off taking priority."""
    if disabled:
        return ObservabilityConfig()
    env = os.environ if environ is None else environ
    if destinations is None:
        raw = env.get("DAYDREAM_TRACE_TO", "").strip()
        destinations = tuple(name.strip() for name in raw.split(",")) if raw else ()
    policy = content if content is not None else env.get("DAYDREAM_TRACE_CONTENT", "full")
    if policy not in ("full", "metadata"):
        raise ObservabilityError(
            "Trace content must be 'full' or 'metadata' (--trace-content or DAYDREAM_TRACE_CONTENT)"
        )
    return ObservabilityConfig(
        destinations=tuple(destinations),
        capture_content=policy == "full",
        service_name=env.get("OTEL_SERVICE_NAME", "daydream"),
    )
