"""Per-run provider ownership, context propagation and bounded cleanup."""

from __future__ import annotations

import inspect
import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from importlib.metadata import version
from typing import TYPE_CHECKING

import anyio
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

from daydream.observability.config import ObservabilityConfig, ObservabilityError
from daydream.observability.privacy import PrivacyPolicy, diagnostic_scope

if TYPE_CHECKING:
    from daydream.extensions.registry import Registry
    from daydream.observability.spans import SpanScope

_logger = logging.getLogger(__name__)
_active_session: ContextVar[TraceSession | None] = ContextVar("daydream_trace_session", default=None)


def current_session() -> TraceSession | None:
    """Return the owned provider for this asynchronous run, if enabled."""
    return _active_session.get()


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
        self.cleanup_timeout_s = cleanup_timeout_s
        self._closed = False
        self._unowned: list[_SafeExporter] = []
        self.provider = TracerProvider(
            resource=Resource(
                {
                    "service.name": self.policy.text(config.service_name),
                    "service.version": version("daydream"),
                }
            ),
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

    def configure(self, config: ObservabilityConfig, registry: Registry) -> None:
        """Resolve all names before opening any exporter; retain cleanup ownership."""
        factories = []
        for name in config.destinations:
            try:
                factories.append((name, registry.trace_exporter(name)))
            except Exception:
                raise ObservabilityError(f"Unknown trace exporter '{name}'") from None
        with diagnostic_scope(self.policy):
            from traceloop.sdk import Traceloop

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
                processor = Traceloop.get_default_span_processor(exporter=exporter, headers={}, disable_batch=False)
                if not isinstance(processor, BatchSpanProcessor):
                    # The public helper uses SimpleSpanProcessor under IPython.
                    # No export has occurred yet; transfer the exporter to batching.
                    processor = BatchSpanProcessor(exporter)
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
                yield root
        finally:
            _active_session.reset(token)
    finally:
        await session.close()
