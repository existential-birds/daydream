"""Per-run provider ownership, context propagation and bounded cleanup."""

from __future__ import annotations

import inspect
import logging
import os
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from importlib.metadata import version
from typing import TYPE_CHECKING

import anyio
from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

from daydream.observability.config import ObservabilityConfig, ObservabilityError
from daydream.observability.privacy import (
    RESOURCE_DIAGNOSTIC,
    PrivacyPolicy,
    diagnostic_scope,
    parse_operator_resource_attributes,
    sanitize_operator_resource_attributes,
)

if TYPE_CHECKING:
    from daydream.extensions.registry import Registry
    from daydream.observability.spans import SpanScope

_logger = logging.getLogger(__name__)
_active_session: ContextVar[TraceSession | None] = ContextVar("daydream_trace_session", default=None)

#: SDK identity from the OTel resource spec's SDK-provided defaults; the pinned
#: distribution version comes from importlib.metadata, never an invented value.
_SDK_NAME = "opentelemetry"
_SDK_LANGUAGE = "python"

#: The Daydream field contract revision recorded as a resource attribute. It
#: pins the checked-in semconv fixture manifest's source commit; the upstream
#: schema URL is a documented omission (dev snapshot, not a released schema).
CONTRACT_VERSION = "94f432d"


def current_session() -> TraceSession | None:
    """Return the owned provider for this asynchronous run, if enabled."""
    return _active_session.get()


def associate_run_trajectory(session_id: str) -> None:
    """Link the early run span once its root trajectory identity is available."""
    session = current_session()
    if session is not None and session.root_scope is not None:
        session.root_scope.attrs({
            "daydream.session.id": session_id,
            "daydream.trajectory.id": session_id,
            "traceloop.association.properties.session_id": session_id,
        })
        # Every scope opened after this association must inherit the session
        # identity even without an active trajectory recorder (the replay and
        # interpreter-less paths have none). Without the seed, descendants
        # would carry only the run id and vendors would split the tree across
        # sessions. The root scope's exit resets the ContextVar token, so the
        # seed never leaks into a later run.
        from daydream.observability.spans import associate_trajectory_identity

        associate_trajectory_identity(session_id)


class _SafeExporter(SpanExporter):
    """Contain extension and SDK failures and sanitize their worker diagnostics."""

    def __init__(self, exporter: SpanExporter, policy: PrivacyPolicy) -> None:
        self.exporter = exporter
        self.policy = policy
        self._shutdown = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with diagnostic_scope(self.policy):
            try:
                result = self.exporter.export(spans)
                if result != SpanExportResult.SUCCESS:
                    _logger.warning("Trace export failed; review execution continues")
                return result
            except Exception:
                _logger.warning("Trace exporter raised an exception; review execution continues")
                return SpanExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        with diagnostic_scope(self.policy):
            try:
                return self.exporter.force_flush(timeout_millis)
            except Exception:
                _logger.warning("Trace exporter flush failed")
                return False

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        with diagnostic_scope(self.policy):
            try:
                self.exporter.shutdown()
            except Exception:
                _logger.warning("Trace exporter shutdown failed")


class TraceSession:
    """One owned OTel provider; never installs globals or automatic instrumentors."""

    def __init__(self, config: ObservabilityConfig, *, cleanup_timeout_s: float = 10) -> None:
        self.policy = PrivacyPolicy(config.capture_content)
        self.run_id = str(uuid.uuid4())
        self.root_scope: SpanScope | None = None
        self.cleanup_timeout_s = cleanup_timeout_s
        self._closed = False
        self._unowned: list[_SafeExporter] = []
        self.provider = TracerProvider(
            resource=self._build_resource(config),
            shutdown_on_exit=False,
            # Hydration must not inherit ambient OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT.
            span_limits=SpanLimits(
                max_attributes=512,
                max_span_attributes=512,
                max_events=100000,
                max_attribute_length=SpanLimits.UNSET,
                max_span_attribute_length=SpanLimits.UNSET,
                max_event_attributes=128,
            ),
        )
        self.tracer = self.provider.get_tracer("daydream", version("daydream"))

    @staticmethod
    def _sdk_resource_version() -> str:
        return version("opentelemetry-sdk")

    def _build_resource(self, config: ObservabilityConfig) -> Resource:
        """Assemble the immutable session resource from declared sources only.

        The operator variable is parsed strictly once per session without
        mutating ``os.environ``; any parse/decode/duplicate rejection discards
        the whole variable behind one fixed diagnostic. The authoritative
        application identity overlays every reserved operator collision except
        a valid operator ``service.instance.id``, which stays authoritative.
        ``Resource.create`` is intentionally avoided so ambient entry-point
        detectors and the unsanitized environment never re-enter the resource.
        """
        attributes: dict[str, str] = {}
        raw = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
        with diagnostic_scope(self.policy):
            try:
                operator = parse_operator_resource_attributes(raw)
            except Exception:
                _logger.warning(RESOURCE_DIAGNOSTIC)
            else:
                attributes.update(sanitize_operator_resource_attributes(operator, self.policy))
        reserved_overrides = {
            "service.name": self.policy.text(config.service_name),
            "service.version": version("daydream"),
            "telemetry.sdk.name": _SDK_NAME,
            "telemetry.sdk.language": _SDK_LANGUAGE,
            "telemetry.sdk.version": self._sdk_resource_version(),
            "daydream.observability.contract.version": CONTRACT_VERSION,
        }
        for key, value in reserved_overrides.items():
            attributes.pop(key, None)
            attributes[key] = value
        if "service.instance.id" not in attributes:
            attributes["service.instance.id"] = self.run_id
        return Resource(attributes)

    def configure(self, config: ObservabilityConfig, registry: Registry) -> None:
        """Resolve all names before opening any exporter; retain cleanup ownership."""
        factories = []
        for name in config.destinations:
            try:
                factories.append((name, registry.trace_exporter(name)))
            except Exception:
                raise ObservabilityError(f"Unknown trace exporter '{name}'") from None
        with diagnostic_scope(self.policy):
            try:
                from traceloop.sdk import Traceloop  # noqa: F401  (dependency presence probe)
            except ImportError:
                raise ObservabilityError(
                    "Tracing dependencies are missing or incompatible. From the Daydream clone, run "
                    "'uv tool install --reinstall --editable .' to refresh the installed command, "
                    "or use 'uv run daydream ...' to run with the project's dependencies. "
                    "If the error persists, check 'command -v daydream' for an older installation on PATH."
                ) from None

            for name, factory in factories:
                try:
                    raw_exporter = factory(config)
                except Exception as exc:
                    detail = self.policy.text(str(exc)) if isinstance(exc, ObservabilityError) else type(exc).__name__
                    raise ObservabilityError(f"Trace exporter '{name}': {detail}") from None
                if not isinstance(raw_exporter, SpanExporter):
                    if inspect.iscoroutine(raw_exporter):
                        raw_exporter.close()
                    raise ObservabilityError(f"Trace exporter '{name}' must return an OpenTelemetry SpanExporter")
                exporter = _SafeExporter(raw_exporter, self.policy)
                self._unowned.append(exporter)
                # One owned batch path for every destination and both normal and
                # notebook environments. The Traceloop default processor's
                # on_start callback reads ambient workflow/agent/conversation/
                # entity-path/association/managed-prompt context that Daydream
                # does not own; owned Traceloop aliases are authored explicitly
                # on the run scope instead. IPython detection never changes the
                # processor type, callback, limits, or privacy. An explicit
                # NoOpMeterProvider keeps the processor's internal metrics
                # inactive regardless of the ambient enabling variable.
                processor = BatchSpanProcessor(exporter, meter_provider=NoOpMeterProvider())
                self.provider.add_span_processor(processor)
                self._unowned.remove(exporter)

    async def close(self) -> None:
        """Bound waiting once; a worker owns eventual exactly-once shutdown."""
        if self._closed:
            return
        self._closed = True

        def cleanup() -> None:
            with diagnostic_scope(self.policy):
                try:
                    self.provider.force_flush(timeout_millis=max(1, int(self.cleanup_timeout_s * 1000)))
                except Exception:
                    _logger.warning("Trace processor flush failed")
                finally:
                    try:
                        self.provider.shutdown()
                    except Exception:
                        _logger.warning("Trace processor shutdown failed")
                    finally:
                        for exporter in self._unowned:
                            exporter.shutdown()

        with anyio.move_on_after(self.cleanup_timeout_s, shield=True) as deadline:
            await anyio.to_thread.run_sync(cleanup, abandon_on_cancel=True)
        if deadline.cancel_called:
            _logger.warning("Trace cleanup exceeded its deadline; cleanup continues in the background")


@asynccontextmanager
async def trace_run(
    config: ObservabilityConfig,
    registry: Registry,
    *,
    flow: str,
    cleanup_timeout_s: float = 10,
) -> AsyncIterator[SpanScope]:
    """Own a run workflow and export resources, or provide a zero-export off path."""
    from daydream.observability.spans import SpanScope

    if not config.destinations:
        # Explicit off also isolates nested runs from an enclosing active session.
        token = _active_session.set(None)
        try:
            yield SpanScope(None, "daydream.run", "run")
        finally:
            _active_session.reset(token)
        return
    session = TraceSession(config, cleanup_timeout_s=cleanup_timeout_s)
    try:
        try:
            await anyio.to_thread.run_sync(session.configure, config, registry)
        except Exception as exc:
            detail = (
                session.policy.text(str(exc))
                if isinstance(exc, (ObservabilityError, ValueError))
                else type(exc).__name__
            )
            raise ObservabilityError(f"Cannot initialize tracing: {detail}") from None
        token = _active_session.set(session)
        try:
            with SpanScope(session, "daydream.run", "run", {"daydream.flow": flow}) as root:
                session.root_scope = root
                try:
                    yield root
                finally:
                    session.root_scope = None
        finally:
            _active_session.reset(token)
    finally:
        await session.close()
