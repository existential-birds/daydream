"""Per-run extension registry, seeded with builtins before extension registration.

Own phases, flows, prompts, renderers, supervision and trace factories. Never
import runner or phases here; that would create import cycles."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import TypeVar

from daydream.extensions.api import (
    ExtensionError,
    FlowStep,
    LoopGroup,
    ToolSupervisor,
    UnresolvedExtensionError,
)
from daydream.observability.config import (
    LOCAL_TRACE_DESTINATION,
    ObservabilityError,
    TraceExporterFactory,
    validate_destination_name,
)

FlowEntry = str | LoopGroup

_T = TypeVar("_T")

_VALIDATE_HINT = "run 'daydream ext validate' to check the extension registry"


#: Prompt names of the replaced staged reviewer. An override of one would register and never run.
REMOVED_PROMPTS = frozenset({
    "intent", "alternatives", "per-stack", "structural", "generic-fallback", "arbiter", "supervise", "suppression",
    "merge",
})


class Registry:
    """Mutable per-run store for phases, flows, prompts, renderers, supervision, and tracing."""

    def __init__(self) -> None:
        self._phases: dict[str, FlowStep] = {}
        self._flows: dict[str, list[FlowEntry]] = {}
        self._prompts: dict[str, Callable[..., str]] = {}
        self._renderers: dict[str, Callable[..., str]] = {}
        self._tool_supervisor: ToolSupervisor | None = None
        self._trace_exporters: dict[str, TraceExporterFactory] = {}

    def register_phase(self, step: FlowStep, *, replace: bool = False) -> None:
        """Register a phase by unique name; duplicates require ``replace=True``."""
        if step.name in self._phases and not replace:
            raise ExtensionError(f"phase '{step.name}' is already registered; pass replace=True to override it")
        self._phases[step.name] = step

    def phase(self, name: str) -> FlowStep:
        """Return the registered phase, or raise ``UnresolvedExtensionError``."""
        return self._lookup(self._phases, "phase", name)

    def set_flow(self, flow_name: str, entries: Sequence[FlowEntry]) -> None:
        """Store ordered phase names and loop groups. Resolve names during run_flow
        preflight or ext validate, so registration order remains unrestricted."""
        self._flows[flow_name] = list(entries)

    def flow(self, flow_name: str) -> list[FlowEntry]:
        """Return the flow's ordered entry list, or raise ``UnresolvedExtensionError``."""
        return self._lookup(self._flows, "flow", flow_name).copy()

    def phase_names(self) -> tuple[str, ...]:
        """Return every registered phase name in registration order."""
        return tuple(self._phases)

    def flow_names(self) -> tuple[str, ...]:
        """Return every registered flow name in registration order."""
        return tuple(self._flows)

    @staticmethod
    def _lookup(mapping: dict[str, _T], kind: str, name: str) -> _T:
        try:
            return mapping[name]
        except KeyError:
            raise UnresolvedExtensionError(f"{kind} '{name}' is not registered; {_VALIDATE_HINT}") from None

    def override_prompt(self, name: str, builder: Callable[..., str]) -> None:
        """Upsert the prompt builder for a named prompt."""
        if name in REMOVED_PROMPTS:
            raise ExtensionError(f"prompt '{name}' was removed with the staged reviewer and nothing calls it; "
                                 "remove the override and put review guidance in a review profile")
        if not callable(builder):
            raise ExtensionError(f"prompt '{name}' builder must be callable")
        self._prompts[name] = builder

    def prompt(self, name: str) -> Callable[..., str]:
        """Return the named prompt builder, or raise ``UnresolvedExtensionError``."""
        return self._lookup(self._prompts, "prompt", name)

    def prompt_names(self) -> tuple[str, ...]:
        """Return every registered prompt name in registration order."""
        return tuple(self._prompts)

    def override_renderer(self, name: str, fn: Callable[..., str]) -> None:
        """Upsert the comment renderer for a named slot."""
        if not callable(fn):
            raise ExtensionError(f"renderer '{name}' must be callable")
        self._renderers[name] = fn

    def renderer(self, name: str) -> Callable[..., str]:
        """Return the named renderer, or raise ``UnresolvedExtensionError``."""
        return self._lookup(self._renderers, "renderer", name)

    def renderer_names(self) -> tuple[str, ...]:
        """Return every registered renderer name in registration order."""
        return tuple(self._renderers)

    def register_tool_supervisor(self, fn: ToolSupervisor) -> None:
        """Register the single per-run tool supervisor."""
        if not callable(fn):
            raise ExtensionError("tool supervisor must be callable")
        if self._tool_supervisor is not None:
            raise ExtensionError("tool supervisor is already registered")
        if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__call__", None)):
            raise ExtensionError("tool supervisor must be synchronous")
        self._tool_supervisor = fn

    def tool_supervisor_if_registered(self) -> ToolSupervisor | None:
        """Return the registered tool supervisor, or None when absent."""
        return self._tool_supervisor

    def register_trace_exporter(self, name: str, factory: TraceExporterFactory, *, replace: bool = False) -> None:
        """Register a destination factory without creating a transport or reading credentials."""
        try:
            validate_destination_name(name)
        except ObservabilityError as exc:
            raise ExtensionError(str(exc)) from None
        if name == LOCAL_TRACE_DESTINATION:
            raise ExtensionError(f"trace exporter '{name}' is reserved for the built-in local trace file")
        if not callable(factory):
            raise ExtensionError(f"trace exporter '{name}' factory must be callable")
        if inspect.iscoroutinefunction(factory) or inspect.iscoroutinefunction(getattr(factory, "__call__", None)):
            raise ExtensionError(f"trace exporter '{name}' factory must be synchronous")
        if name in self._trace_exporters and not replace:
            raise ExtensionError(f"trace exporter '{name}' is already registered; pass replace=True to override it")
        self._trace_exporters[name] = factory

    def trace_exporter(self, name: str) -> TraceExporterFactory:
        """Look up a selected factory without invoking it."""
        return self._lookup(self._trace_exporters, "trace exporter", name)

    def trace_exporter_names(self) -> tuple[str, ...]:
        """Return destination names in deterministic registration order."""
        return tuple(self._trace_exporters)
