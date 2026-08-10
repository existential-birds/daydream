"""Durable neutral controller state machine (Plan 008 Step 3).

The review service controller is a pure, deterministic state machine. All of
its transitions are named with neutral terms — ``queued``, ``starting``,
``running``, ``collecting``, ``evaluated``, ``publishing``, and the terminal
``passed``/``failed``/``infra_error``/``cancelled``/``released`` — and never
reference an execution adapter (Sprites, Coder, Kubernetes, local), a model
provider, or a worker-asserted infrastructure identity.

Design rules:

- ``ServiceEvent`` are the only things that can move a job. A valid forward
  edge advances the state; re-delivering an event whose effect is already
  present is an idempotent no-op (a controller restart must be safe); any
  reordered, stale, or superseded event raises ``InvalidTransition``.
- ``INFRA`` and ``CANCEL`` are legal from every non-terminal (active) state:
  the service fails closed on missing coverage and cannot be wedged open.
- ``RELEASE`` is the final transition from any terminal state to ``released``
  and is itself idempotent. Once released, no event except a duplicate
  ``RELEASE`` may move the job again.

This module is pure: it performs no I/O and depends only on the standard
library and its own types, so the property tests in ``test_service_state_machine``
exercise the exact contract the durable controller relies on.
"""

from __future__ import annotations

from enum import Enum


class ServiceState(Enum):
    """The neutral lifecycle states a review job may occupy."""

    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    COLLECTING = "collecting"
    EVALUATED = "evaluated"
    PUBLISHING = "publishing"
    PASSED = "passed"
    FAILED = "failed"
    INFRA_ERROR = "infra_error"
    CANCELLED = "cancelled"
    RELEASED = "released"


# States from which an in-flight execution can still be interrupted. These can
# transition to a terminal via INFRA/CANCEL but are not themselves verdicts.
_ACTIVE_STATES = frozenset(
    {
        ServiceState.QUEUED,
        ServiceState.STARTING,
        ServiceState.RUNNING,
        ServiceState.COLLECTING,
        ServiceState.EVALUATED,
        ServiceState.PUBLISHING,
    }
)

_TERMINAL_STATES = frozenset(
    {
        ServiceState.PASSED,
        ServiceState.FAILED,
        ServiceState.INFRA_ERROR,
        ServiceState.CANCELLED,
    }
)


class ServiceEvent(Enum):
    """The events that drive a review job through its lifecycle."""

    DISPATCH = "dispatch"  # queued -> starting
    STARTED = "started"  # starting -> running
    COLLECT = "collect"  # running -> collecting
    COLLECTED = "collected"  # collecting -> evaluated
    PASS = "pass"  # evaluated -> publishing
    FAIL = "fail"  # evaluated -> failed
    PUBLISHED = "published"  # publishing -> passed
    INFRA = "infra"  # _ACTIVE -> infra_error
    CANCEL = "cancel"  # _ACTIVE -> cancelled
    RELEASE = "release"  # _TERMINAL -> released


# Forward edges, expressed per event. The target of each edge also carries the
# idempotency rule: an event is a no-op when the job already occupies the
# state the event would produce.
_EDGES: dict[ServiceEvent, dict[ServiceState, ServiceState]] = {
    ServiceEvent.DISPATCH: {ServiceState.QUEUED: ServiceState.STARTING},
    ServiceEvent.STARTED: {ServiceState.STARTING: ServiceState.RUNNING},
    ServiceEvent.COLLECT: {ServiceState.RUNNING: ServiceState.COLLECTING},
    ServiceEvent.COLLECTED: {ServiceState.COLLECTING: ServiceState.EVALUATED},
    ServiceEvent.PASS: {ServiceState.EVALUATED: ServiceState.PUBLISHING},
    ServiceEvent.FAIL: {ServiceState.EVALUATED: ServiceState.FAILED},
    ServiceEvent.PUBLISHED: {ServiceState.PUBLISHING: ServiceState.PASSED},
    ServiceEvent.INFRA: {state: ServiceState.INFRA_ERROR for state in _ACTIVE_STATES},
    ServiceEvent.CANCEL: {state: ServiceState.CANCELLED for state in _ACTIVE_STATES},
    ServiceEvent.RELEASE: {state: ServiceState.RELEASED for state in _TERMINAL_STATES},
}

# Reverse index: for each state, which event produced it. Used to decide whether
# a delivered event is a duplicate (its effect is already present) versus an
# invalid reorder. Terminal infra/cancel/release states can be produced by
# several events; any surviving producer identity preserves the no-op contract,
# so the first producer encountered is sufficient. RELEASED has no forward edge
# into it, so it is explicitly bound to RELEASE.
_EVENT_OF_STATE: dict[ServiceState, ServiceEvent] = {
    ServiceState.RELEASED: ServiceEvent.RELEASE,
}
for _event, _table in _EDGES.items():
    for _to in _table.values():
        _EVENT_OF_STATE.setdefault(_to, _event)


class InvalidTransition(Exception):
    """An event was applied that the current state does not legally accept.

    Raised for reordered events, stale/superseded events, or any event that
    would re-open or rewind a job. The controller must not silently absorb
    these — a job may only move forward through the declared neutral path.
    """


def apply(state: ServiceState, event: ServiceEvent) -> ServiceState:
    """Transition ``state`` under ``event``, raising ``InvalidTransition`` on illegal input.

    A forward edge advances to the target. An event whose target the job
    already occupies is an idempotent no-op (returns ``state`` unchanged),
    which is what makes controller restarts and duplicate delivery safe.
    """
    table = _EDGES[event]
    if state in table:
        return table[state]
    # Idempotent duplicate: the event's effect is already present.
    if _EVENT_OF_STATE.get(state) is event:
        return state
    raise InvalidTransition(f"{state.value} cannot accept {event.value}")


def is_terminal(state: ServiceState) -> bool:
    """Return True when ``state`` is a terminal verdict (not ``RELEASED``)."""
    return state in _TERMINAL_STATES
