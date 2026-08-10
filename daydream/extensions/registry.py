"""Per-run extension registry.

``Registry`` holds phases + flows, skill slots, named prompts, a tool
supervisor, and fork stack rules. ``register_builtins()`` seeds it with
everything daydream does today; an optional ``daydream_ext`` package mutates
it through the same API.

This module must not import from ``daydream.runner`` or ``daydream.phases``
(import-cycle guard).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence

from daydream.executors.contract import (
    DAYDREAM_SERVICE_V1,
    MIN_SUPPORTED_DAYDREAM_SERVICE_V1,
    ExecutorError,
    require_capabilities,
)
from daydream.executors.protocol import ReviewExecutor, is_review_executor
from daydream.extensions.api import (
    ExtensionError,
    FlowStep,
    LoopGroup,
    StackRule,
    ToolSupervisor,
    UnresolvedExtensionError,
)

FlowEntry = str | LoopGroup

_VALIDATE_HINT = "run 'daydream ext validate' to check the extension registry"


class Registry:
    """Mutable per-run store for phases, flows, skills, prompts, supervision, and stack rules."""

    def __init__(self) -> None:
        self._phases: dict[str, FlowStep] = {}
        self._flows: dict[str, list[FlowEntry]] = {}
        self._skills: dict[str, str] = {}
        self._prompts: dict[str, Callable[..., str]] = {}
        self._stack_rules: dict[str, StackRule] = {}
        self._tool_supervisor: ToolSupervisor | None = None
        self._executors: dict[str, ReviewExecutor] = {}
        self._publishers: dict[str, object] = {}

    # -- phases -----------------------------------------------------------

    def register_phase(self, step: FlowStep, *, replace: bool = False) -> None:
        """Register a phase by unique name; duplicates require ``replace=True``."""
        if step.name in self._phases and not replace:
            raise ExtensionError(f"phase '{step.name}' is already registered; pass replace=True to override it")
        self._phases[step.name] = step

    def phase(self, name: str) -> FlowStep:
        """Return the registered phase, or raise ``UnresolvedExtensionError``."""
        try:
            return self._phases[name]
        except KeyError:
            raise UnresolvedExtensionError(f"phase '{name}' is not registered; {_VALIDATE_HINT}") from None

    # -- flows ------------------------------------------------------------

    def set_flow(self, flow_name: str, entries: Sequence[FlowEntry]) -> None:
        """Define a flow as an ordered list of phase names and loop groups.

        Entry names are resolved against registered phases by ``run_flow``'s
        pre-flight pass (and ``daydream ext validate``), not at definition time,
        so registration order between phases and flows does not matter.
        """
        self._flows[flow_name] = list(entries)

    def flow(self, flow_name: str) -> list[FlowEntry]:
        """Return the flow's ordered entry list, or raise ``UnresolvedExtensionError``."""
        return list(self._entries(flow_name))

    def phase_names(self) -> tuple[str, ...]:
        """Return every registered phase name in registration order."""
        return tuple(self._phases)

    def flow_names(self) -> tuple[str, ...]:
        """Return every registered flow name in registration order."""
        return tuple(self._flows)

    def insert_before(self, flow_name: str, *, anchor: str, step: FlowEntry) -> None:
        """Insert ``step`` immediately before ``anchor`` in the named flow."""
        self._insert(flow_name, anchor=anchor, step=step, offset=0)

    def insert_after(self, flow_name: str, *, anchor: str, step: FlowEntry) -> None:
        """Insert ``step`` immediately after ``anchor`` in the named flow."""
        self._insert(flow_name, anchor=anchor, step=step, offset=1)

    def remove(self, flow_name: str, step: str) -> None:
        """Remove the entry named ``step`` from the named flow."""
        entries = self._entries(flow_name)
        del entries[self._index_of(flow_name, entries, step)]

    def _insert(self, flow_name: str, *, anchor: str, step: FlowEntry, offset: int) -> None:
        entries = self._entries(flow_name)
        entries.insert(self._index_of(flow_name, entries, anchor) + offset, step)

    def _entries(self, flow_name: str) -> list[FlowEntry]:
        try:
            return self._flows[flow_name]
        except KeyError:
            raise UnresolvedExtensionError(f"flow '{flow_name}' is not registered; {_VALIDATE_HINT}") from None

    @staticmethod
    def _entry_name(entry: FlowEntry) -> str:
        return entry if isinstance(entry, str) else entry.name

    def _index_of(self, flow_name: str, entries: list[FlowEntry], name: str) -> int:
        for index, entry in enumerate(entries):
            if self._entry_name(entry) == name:
                return index
        # Check whether the name exists inside a LoopGroup body so the error
        # message names the containing group instead of implying the step is absent.
        for entry in entries:
            if isinstance(entry, LoopGroup) and name in entry.steps:
                raise UnresolvedExtensionError(
                    f"flow '{flow_name}' step '{name}' is inside loop group '{entry.name}'"
                    f" and cannot be addressed directly; {_VALIDATE_HINT}"
                )
        raise UnresolvedExtensionError(f"flow '{flow_name}' has no step '{name}'; {_VALIDATE_HINT}")

    # -- skill slots ------------------------------------------------------

    def override_skill(self, slot: str, skill: str) -> None:
        """Upsert the skill invocation string for a named slot."""
        self._skills[slot] = skill

    def skill(self, slot: str) -> str:
        """Return the slot's skill string, or raise ``UnresolvedExtensionError``."""
        try:
            return self._skills[slot]
        except KeyError:
            raise UnresolvedExtensionError(f"skill slot '{slot}' is not registered; {_VALIDATE_HINT}") from None

    def skill_if_registered(self, slot: str) -> str | None:
        """Return the slot's skill string, or None; never raises."""
        return self._skills.get(slot)

    def skill_slots(self) -> dict[str, str]:
        """Return a copy of the slot-to-skill-invocation mapping."""
        return dict(self._skills)

    def stack_keys(self) -> set[str]:
        """Return the stack keys of every registered ``stack:<key>`` skill slot."""
        return {slot.removeprefix("stack:") for slot in self._skills if slot.startswith("stack:")}

    # -- prompts ----------------------------------------------------------

    def override_prompt(self, name: str, builder: Callable[..., str]) -> None:
        """Upsert the prompt builder for a named prompt."""
        self._prompts[name] = builder

    def prompt(self, name: str) -> Callable[..., str]:
        """Return the named prompt builder, or raise ``UnresolvedExtensionError``."""
        try:
            return self._prompts[name]
        except KeyError:
            raise UnresolvedExtensionError(f"prompt '{name}' is not registered; {_VALIDATE_HINT}") from None

    def prompt_names(self) -> tuple[str, ...]:
        """Return every registered prompt name in registration order."""
        return tuple(self._prompts)

    # -- tool supervision -------------------------------------------------

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

    # -- stack rules ------------------------------------------------------

    def add_stack(self, rule: StackRule) -> None:
        """Upsert a fork stack rule, keyed by ``stack_name``."""
        self._stack_rules[rule.stack_name] = rule

    def stack_rules(self) -> tuple[StackRule, ...]:
        """Return all fork stack rules in registration order."""
        return tuple(self._stack_rules.values())

    # -- executors / publishers (DAYDREAM_SERVICE_V1 seam) -----------------

    def register_executor(self, name: str, executor: ReviewExecutor, *, service_api: int = DAYDREAM_SERVICE_V1) -> None:
        """Register a ``ReviewExecutor`` by name (capability admission at registration).

        Rejects a duplicate name without ``replace`` and rejects any object
        that is not a conformant ``ReviewExecutor`` or whose declared
        capabilities miss a required one — capability admission is a contract
        STOP condition, enforceable as early as the registry seam. The service
        contract version is asserted (additive extension contract, so this does
        not change ``EXTENSION_API_VERSION``).
        """
        if not is_review_executor(executor):
            raise ExtensionError(f"executor '{name}' is not a conformant ReviewExecutor (DAYDREAM_SERVICE_V1)")
        if not (MIN_SUPPORTED_DAYDREAM_SERVICE_V1 <= service_api <= DAYDREAM_SERVICE_V1):
            raise ExtensionError(
                f"executor '{name}' declares DAYDREAM_SERVICE_V1 = {service_api!r}; "
                f"supported {MIN_SUPPORTED_DAYDREAM_SERVICE_V1}..{DAYDREAM_SERVICE_V1}"
            )
        if name in self._executors:
            raise ExtensionError(f"executor '{name}' is already registered")
        try:
            declared = set(getattr(executor, "capabilities", frozenset()))
            require_capabilities(declared, kind=getattr(executor, "kind", name))
        except ExecutorError as exc:
            raise ExtensionError(f"executor '{name}' cannot be admitted: {exc}") from exc
        self._executors[name] = executor

    def executor(self, name: str) -> ReviewExecutor:
        """Return the registered executor, or raise ``UnresolvedExtensionError``."""
        try:
            return self._executors[name]
        except KeyError:
            raise UnresolvedExtensionError(f"executor '{name}' is not registered; {_VALIDATE_HINT}") from None

    def executor_if_registered(self, name: str) -> ReviewExecutor | None:
        """Return the registered executor, or None; never raises."""
        return self._executors.get(name)

    def executor_names(self) -> tuple[str, ...]:
        """Return every registered executor name in registration order."""
        return tuple(self._executors)

    def register_publisher(self, name: str, publisher: object) -> None:
        """Register a publisher object by name (the trusted publication seam).

        Unlike executors, publishers are not capability-admitted here — the
        trusted GitHub publisher carries its own credential-safety contract
        (implemented by the publisher leaf); this registry only names it so
        service policy can resolve it without a vendor import.
        """
        if name in self._publishers:
            raise ExtensionError(f"publisher '{name}' is already registered")
        self._publishers[name] = publisher

    def publisher(self, name: str) -> object:
        """Return the registered publisher, or raise ``UnresolvedExtensionError``."""
        try:
            return self._publishers[name]
        except KeyError:
            raise UnresolvedExtensionError(f"publisher '{name}' is not registered; {_VALIDATE_HINT}") from None

    def publisher_if_registered(self, name: str) -> object | None:
        """Return the registered publisher, or None; never raises."""
        return self._publishers.get(name)

    def publisher_names(self) -> tuple[str, ...]:
        """Return every registered publisher name in registration order."""
        return tuple(self._publishers)
