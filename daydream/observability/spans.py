"""Portable workflow scopes and a reducer for Daydream's public backend events."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict
from types import TracebackType
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind, Status, StatusCode

from daydream.backends import (
    AgentEvent,
    CostEvent,
    MetricsEvent,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.observability.runtime import current_session
from daydream.trajectory import get_current_recorder

if TYPE_CHECKING:
    from daydream.observability.runtime import TraceSession

_logger = logging.getLogger(__name__)
_scope_attributes: ContextVar[dict[str, Any]] = ContextVar("daydream_trace_attributes", default={})


def _reason_code(reason: str | None) -> str:
    candidate = (reason or "error").split(":", 1)[0]
    return (
        candidate
        if candidate
        in {
            "error",
            "cancelled",
            "interrupted",
            "tool_vetoed",
            "wall_budget_exceeded",
            "tool_call_budget_exceeded",
        }
        else "error"
    )


def _record_failure(
    span: trace.Span,
    session: TraceSession,
    *,
    reason: str,
    error_type: str,
    message: str | None = None,
) -> None:
    """Use OTel's portable error surface without implicit raw exception capture."""
    code = _reason_code(reason)
    span.set_status(Status(StatusCode.ERROR, description=code))
    safe_type = session.policy.text(error_type)
    span.set_attribute("error.type", safe_type)
    attributes = {"exception.type": safe_type}
    if session.policy.capture_content:
        attributes["exception.message"] = session.policy.text(message if message is not None else code)
    span.add_event("exception", attributes)


class SpanScope:
    """Small fail-open scope; application exceptions are never implicitly recorded."""

    def __init__(
        self,
        session: TraceSession | None,
        name: str,
        kind: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.session = session
        self.name = name
        self.kind = kind
        self.attributes = attributes or {}
        self.span: trace.Span | None = None
        self._context: Any = None
        self._token: Token[dict[str, Any]] | None = None

    def __enter__(self) -> SpanScope:
        if self.session is None:
            return self
        try:
            common = {**({} if self.kind == "run" else _scope_attributes.get()), **self.attributes}
            common["daydream.run.id"] = self.session.run_id
            common["traceloop.association.properties.daydream_run_id"] = self.session.run_id
            recorder = get_current_recorder()
            if recorder is not None:
                common["daydream.session.id"] = recorder.session_id
                common["traceloop.association.properties.session_id"] = recorder.session_id
                common["daydream.trajectory.id"] = (
                    f"{recorder.session_id}:{recorder.descriptor}" if recorder.descriptor else recorder.session_id
                )
                if recorder.descriptor:
                    common["daydream.trajectory.descriptor"] = recorder.descriptor
            self._token = _scope_attributes.set(common)
            self.span = self.session.tracer.start_span(
                self.session.policy.text(self.name),
                kind=SpanKind.CLIENT if self.kind == "attempt" else SpanKind.INTERNAL,
                context=Context() if self.kind == "run" else None,
            )
            self._context = trace.use_span(
                self.span, end_on_exit=False, record_exception=False, set_status_on_exception=False
            )
            self._context.__enter__()
            self.attrs(
                {
                    **common,
                    "daydream.span.kind": self.kind,
                    "traceloop.entity.name": self.name,
                    "traceloop.span.kind": {"run": "workflow", "agent": "agent", "tool": "tool"}.get(self.kind, "task"),
                }
            )
        except Exception:
            _logger.warning("Trace span initialization failed; execution continues")
        return self

    def attrs(self, attributes: dict[str, Any]) -> None:
        if self.span is None or self.session is None:
            return
        try:
            for key, value in attributes.items():
                if value is None:
                    continue
                safe = self.session.policy.value(value)
                scalar_array = (
                    isinstance(safe, list)
                    and (not safe or type(safe[0]) in (str, int, float, bool))
                    and all(type(item) is type(safe[0]) for item in safe)
                )
                if not isinstance(safe, (str, int, float, bool)) and not scalar_array:
                    safe = self.session.policy.json(safe)
                self.span.set_attribute(key, safe)
        except Exception:
            _logger.warning("Trace attributes could not be recorded")

    def content(self, key: str, value: Any) -> None:
        if self.session is not None and self.session.policy.capture_content:
            self.attrs({key: self.session.policy.json(value)})

    def output(self, value: Any) -> None:
        self.content("traceloop.entity.output", value)

    def finish(
        self,
        exit_code: int = 0,
        *,
        reason: str | None = None,
        error_type: str = "DaydreamError",
        message: str | None = None,
    ) -> None:
        self.attrs(
            {"daydream.exit_code": exit_code, "daydream.outcome": reason or ("success" if not exit_code else "error")}
        )
        if self.span is not None:
            try:
                if exit_code and self.session is not None:
                    _record_failure(
                        self.span, self.session, reason=_reason_code(reason), error_type=error_type, message=message
                    )
                else:
                    self.span.set_status(Status(StatusCode.OK))
            except Exception:
                _logger.warning("Trace status could not be recorded")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        try:
            if exc is not None:
                try:
                    message = str(exc)
                except Exception:
                    message = "[UNSERIALIZABLE_ERROR]"
                self.finish(
                    1,
                    reason="cancelled" if not isinstance(exc, Exception) else "error",
                    error_type=type(exc).__name__,
                    message=message,
                )
                if self.session is not None and self.session.policy.capture_content:
                    self.content("daydream.error.message", message)
            if self._context is not None:
                self._context.__exit__(None, None, None)
            if self.span is not None:
                self.span.end()
        except Exception:
            _logger.warning("Trace span finalization failed; execution continues")
        finally:
            if self._token is not None:
                _scope_attributes.reset(self._token)


def step_scope(
    name: str,
    *,
    iteration: Any = None,
    phase: str | None = None,
    stack: str | None = None,
) -> SpanScope:
    return SpanScope(
        current_session(),
        f"daydream.step.{name}",
        "step",
        {
            "daydream.step": name,
            "daydream.phase": phase or name,
            "daydream.iteration": iteration,
            "daydream.stack": stack,
        },
    )


def agent_scope(phase: str, *, backend: str, model: str | None = None) -> SpanScope:
    return SpanScope(
        current_session(),
        f"daydream.agent.{phase}",
        "agent",
        {
            "daydream.phase": phase,
            "daydream.backend": backend,
            "gen_ai.request.model": model,
        },
    )


_USAGE_ATTRIBUTES = {
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
    "cached_tokens": "gen_ai.usage.cache_read.input_tokens",
    "cache_creation_tokens": "gen_ai.usage.cache_creation.input_tokens",
    "reasoning_tokens": "gen_ai.usage.reasoning.output_tokens",
    "cost_usd": "gen_ai.usage.cost",
}


class AttemptObserver:
    """Reconcile one backend invocation without inventing provider request spans."""

    def __init__(self, scope: SpanScope) -> None:
        self.scope = scope
        self.request: dict[str, Any] | None = None
        self.messages: list[dict[str, Any]] = []
        self.reasoning: list[str] = []
        self.text: list[str] = []
        self.structured: Any = None
        self.tools: dict[str, trace.Span] = {}
        self.tool_names: dict[str, str] = {}
        self.message_usage: dict[str, dict[str, int | float]] = {}
        self.usage_metadata: dict[str, dict[str, Any]] = {}
        self.invocation_usage: dict[str, int | float] = {}
        self.final_usage: dict[str, int | float] = {}
        self.models: list[str] = []
        self.providers: list[str] = []
        self.reason: str | None = None

    @property
    def capture(self) -> bool:
        return self.scope.session is not None and self.scope.session.policy.capture_content

    def _identity(self, event: Any) -> None:
        for field, collection, attrs in (
            ("model_name", self.models, ("gen_ai.response.model",)),
            ("provider_name", self.providers, ("gen_ai.provider.name", "gen_ai.system")),
        ):
            value = getattr(event, field, None)
            if field == "model_name" and isinstance(event, RequestEvent):
                continue
            if value is not None:
                if value not in collection:
                    collection.append(value)
                self.scope.attrs(dict.fromkeys(attrs, value))
        self.scope.attrs(
            {
                "gen_ai.conversation.id": getattr(event, "session_id", None),
                "gen_ai.response.finish_reasons": (
                    [event.finish_reason] if getattr(event, "finish_reason", None) is not None else None
                ),
            }
        )
        if isinstance(event, ResultEvent):
            self.scope.attrs(
                {
                    "daydream.duration_ms": event.duration_ms,
                    "daydream.duration_api_ms": event.duration_api_ms,
                }
            )

    def observe(self, event: AgentEvent) -> None:
        if self.scope.session is None:
            return
        try:
            self._observe(event)
        except Exception:
            _logger.warning("Trace event could not be recorded; execution continues")

    def _observe(self, event: AgentEvent) -> None:
        self._identity(event)
        policy = self.scope.session.policy if self.scope.session else None
        if isinstance(event, RequestEvent):
            self.scope.attrs(
                {
                    "gen_ai.request.model": event.model_name,
                    "gen_ai.request.reasoning_effort": event.reasoning_effort,
                    "daydream.request.timestamp": event.timestamp,
                }
            )
            if self.capture and policy is not None:
                self.request = policy.value(
                    {
                        key: value
                        for key, value in {
                            "prompt": event.prompt,
                            "system_prompt": event.system_prompt,
                            "output_schema": event.output_schema,
                        }.items()
                        if value is not None
                    }
                )
        elif isinstance(event, TextEvent) and self.capture and policy is not None:
            safe_text = policy.text(event.text)
            self.text.append(safe_text)
            if (
                self.messages
                and self.messages[-1]["role"] == "assistant"
                and self.messages[-1]["parts"][-1]["type"] == "text"
            ):
                self.messages[-1]["parts"][-1]["content"] += safe_text
            else:
                self.messages.append({"role": "assistant", "parts": [{"type": "text", "content": safe_text}]})
        elif isinstance(event, ThinkingEvent) and self.capture and policy is not None:
            self.reasoning.append(policy.text(event.text))
        elif isinstance(event, ToolStartEvent):
            self._start_tool(event)
        elif isinstance(event, ToolResultEvent):
            self._end_tool(event)
        elif isinstance(event, MetricsEvent):
            values = self._usage(event)
            if event.usage_scope == "invocation":
                self.invocation_usage.update(values)
                self.scope.attrs({"daydream.started_at": event.started_at})
            else:
                key = event.message_id or f"anonymous:{len(self.message_usage)}"
                self.message_usage[key] = values
                self.usage_metadata[key] = {
                    "message_id": event.message_id,
                    "model": event.model_name,
                    "provider": event.provider_name,
                    "started_at": event.started_at,
                    "timestamp": event.timestamp,
                    "duration_ms": event.duration_ms,
                    "usage": values,
                }
        elif isinstance(event, CostEvent):
            self.final_usage.update(self._usage(event))
            if event.model_usage is not None:
                self.scope.attrs(
                    {"daydream.model_usage": {key: asdict(value) for key, value in event.model_usage.items()}}
                )
        elif isinstance(event, ResultEvent) and self.capture and policy is not None:
            self.structured = policy.value(event.structured_output) if event.structured_output is not None else None

    @staticmethod
    def _usage(event: MetricsEvent | CostEvent) -> dict[str, int | float]:
        values: dict[str, int | float] = {}
        for name in _USAGE_ATTRIBUTES:
            source = {"input_tokens": "prompt_tokens", "output_tokens": "completion_tokens"}.get(name, name)
            value = getattr(event, source if isinstance(event, MetricsEvent) else name, None)
            if value is not None:
                values[name] = value
        return values

    def _start_tool(self, event: ToolStartEvent) -> None:
        session = self.scope.session
        if session is None:
            return
        # Tool spans are direct attempt children even when executions overlap.
        if event.id in self.tools:
            self._close_tool(event.id, reason="interrupted")
        span = session.tracer.start_span(
            session.policy.text(event.name),
            context=trace.set_span_in_context(self.scope.span) if self.scope.span else None,
        )
        self.tools[event.id] = span
        self.tool_names[event.id] = event.name
        for key, value in {
            **_scope_attributes.get(),
            "daydream.span.kind": "tool",
            "traceloop.span.kind": "tool",
            "traceloop.entity.name": event.name,
            "gen_ai.tool.name": event.name,
            "gen_ai.tool.call.id": event.id,
            "daydream.tool.started_at": event.timestamp,
        }.items():
            if value is not None:
                span.set_attribute(key, session.policy.value(value))
        if self.capture:
            args = session.policy.value(event.input)
            span.set_attribute("traceloop.entity.input", session.policy.json(args))
            self.messages.append(
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool_call",
                            "id": session.policy.text(event.id),
                            "name": session.policy.text(event.name),
                            "arguments": args,
                        }
                    ],
                }
            )

    def _end_tool(self, event: ToolResultEvent) -> None:
        session = self.scope.session
        if session is None:
            return
        span = self.tools.get(event.id)
        if span is not None:
            values: dict[str, Any] = {
                "daydream.tool.error": event.is_error,
                "daydream.tool.status": event.status,
                "daydream.tool.exit_code": event.exit_code,
                "daydream.tool.duration_ms": event.duration_ms,
                "daydream.tool.cancelled": event.cancelled,
                "daydream.tool.truncated": event.truncated,
                "daydream.tool.completed_at": event.timestamp,
            }
            for key, value in values.items():
                if value is not None:
                    span.set_attribute(key, session.policy.value(value))
            if self.capture:
                span.set_attribute("traceloop.entity.output", session.policy.json(event.output))
            self._close_tool(
                event.id,
                reason="cancelled" if event.cancelled else "error" if event.is_error else "success",
                message=event.output if event.is_error else None,
            )
        if self.capture:
            self.messages.append(
                {
                    "role": "tool",
                    "parts": [
                        {
                            "type": "tool_call_response",
                            "id": session.policy.text(event.id),
                            "response": session.policy.value(event.output),
                        }
                    ],
                }
            )

    def _close_tool(self, tool_id: str, *, reason: str, message: str | None = None) -> None:
        span = self.tools.pop(tool_id)
        span.set_attribute("daydream.outcome", reason)
        if reason == "success":
            span.set_status(Status(StatusCode.OK))
        elif self.scope.session is not None:
            _record_failure(span, self.scope.session, reason=reason, error_type="ToolError", message=message)
        span.end()

    def abort(self, reason: str) -> None:
        self.reason = "tool_vetoed" if reason.startswith("tool_vetoed:") else reason
        self.scope.finish(1, reason=self.reason)

    def finish(self, exc: BaseException | None) -> None:
        try:
            for tool_id in list(self.tools):
                self._close_tool(
                    tool_id,
                    reason=self.reason
                    or ("cancelled" if exc is not None and not isinstance(exc, Exception) else "interrupted"),
                )
            usage: dict[str, int | float] = {}
            for values in self.message_usage.values():
                for name, value in values.items():
                    usage[name] = usage.get(name, 0) + value
            usage.update(self.invocation_usage)
            usage.update(self.final_usage)
            attrs = {_USAGE_ATTRIBUTES[name]: value for name, value in usage.items()}
            if "input_tokens" in usage and "output_tokens" in usage:
                attrs["gen_ai.usage.total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
            self.scope.attrs(attrs)
            if self.usage_metadata:
                self.scope.attrs({"daydream.message_usage": list(self.usage_metadata.values())})
            self.scope.attrs({"daydream.models": self.models or None, "daydream.providers": self.providers or None})
            if self.capture:
                if self.request is not None:
                    self.scope.content("traceloop.entity.input", self.request)
                    messages = []
                    if "system_prompt" in self.request:
                        messages.append(
                            {"role": "system", "parts": [{"type": "text", "content": self.request["system_prompt"]}]}
                        )
                    messages.append({"role": "user", "parts": [{"type": "text", "content": self.request["prompt"]}]})
                    self.scope.content("gen_ai.input.messages", messages)
                self.scope.content("gen_ai.output.messages", self.messages)
                if self.reasoning:
                    self.scope.content("daydream.reasoning", "".join(self.reasoning))
                self.scope.output(self.structured if self.structured is not None else "".join(self.text))
            if exc is None and self.reason is None:
                self.scope.finish()
        except Exception:
            _logger.warning("Trace attempt finalization failed; execution continues")


@asynccontextmanager
async def attempt_scope(number: int) -> AsyncIterator[AttemptObserver]:
    backend = _scope_attributes.get().get("daydream.backend", "backend")
    with SpanScope(
        current_session(),
        f"daydream.attempt.{number} ({backend} invocation aggregate)",
        "attempt",
        {
            "daydream.attempt": number,
            "daydream.invocation.aggregate": True,
            "gen_ai.operation.name": "chat",
        },
    ) as scope:
        observer = AttemptObserver(scope)
        error: BaseException | None = None
        try:
            yield observer
        except BaseException as exc:
            error = exc
            raise
        finally:
            observer.finish(error)
