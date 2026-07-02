"""Per-run extension registry.

``Registry`` holds three namespaces — phases + flows, skill slots, and named
prompts — plus fork stack rules. ``register_builtins()`` seeds it with
everything daydream does today; an optional ``daydream_ext`` package mutates
it through the same API.

This module must not import from ``daydream.runner`` or ``daydream.phases``
(import-cycle guard).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from daydream.extensions.api import (
    ExtensionError,
    FlowStep,
    LoopGroup,
    StackRule,
    UnresolvedExtensionError,
)

FlowEntry = str | LoopGroup

_VALIDATE_HINT = "run 'daydream ext validate' to check the extension registry"


class Registry:
    """Mutable per-run store for phases, flows, skill slots, prompts, and stack rules."""

    def __init__(self) -> None:
        self._phases: dict[str, FlowStep] = {}
        self._flows: dict[str, list[FlowEntry]] = {}
        self._skills: dict[str, str] = {}
        self._prompts: dict[str, Callable[..., str]] = {}
        self._stack_rules: dict[str, StackRule] = {}

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

    # -- stack rules ------------------------------------------------------

    def add_stack(self, rule: StackRule) -> None:
        """Upsert a fork stack rule, keyed by ``stack_name``."""
        self._stack_rules[rule.stack_name] = rule

    def stack_rules(self) -> tuple[StackRule, ...]:
        """Return all fork stack rules in registration order."""
        return tuple(self._stack_rules.values())
