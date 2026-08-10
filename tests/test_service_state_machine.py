"""Property tests for the neutral controller state machine (Plan 008 Step 3).

The state machine is the pure, deterministic transition core of the review
service controller. These tests pin its contract for exactly the event classes
the durable controller must tolerate or reject:

- duplicate  — re-delivering an already-applied event is an idempotent no-op
- reordered  — delivering a forward event out of sequence raises
- restarted  — a state restored from the storage port continues identically
- cancelled  — cancellation from any active state lands in a terminal, then releases
- stale      — an event that no longer applies (superseded / already passed) raises

All transitions are neutral to the execution provider: only ``ServiceState``
and ``ServiceEvent`` values cross the boundary, never adapter handles.
"""

from __future__ import annotations

import pytest

from daydream.service.states import (
    InvalidTransition,
    ServiceEvent,
    ServiceState,
    apply,
    is_terminal,
)

S = ServiceState
E = ServiceEvent

# The canonical, fully successful single-attempt path.
CANONICAL_PATH = [
    (S.QUEUED, E.DISPATCH, S.STARTING),
    (S.STARTING, E.STARTED, S.RUNNING),
    (S.RUNNING, E.COLLECT, S.COLLECTING),
    (S.COLLECTING, E.COLLECTED, S.EVALUATED),
    (S.EVALUATED, E.PASS, S.PUBLISHING),
    (S.PUBLISHING, E.PUBLISHED, S.PASSED),
    (S.PASSED, E.RELEASE, S.RELEASED),
]

# Every non-terminal state that an execution can be interrupted from.
ACTIVE_STATES = [
    S.QUEUED,
    S.STARTING,
    S.RUNNING,
    S.COLLECTING,
    S.EVALUATED,
    S.PUBLISHING,
]


def test_canonical_success_path() -> None:
    """The documented neutral progression renders exactly, in order."""
    state = S.QUEUED
    observed: list[tuple[ServiceState, ServiceEvent, ServiceState]] = []
    for pre, event, post in CANONICAL_PATH:
        assert state == pre
        state = apply(state, event)
        observed.append((pre, event, state))
        assert state == post
    assert state is S.RELEASED


@pytest.mark.parametrize(
    ("start", "terminal"),
    [
        (S.QUEUED, S.CANCELLED),
        (S.STARTING, S.CANCELLED),
        (S.RUNNING, S.CANCELLED),
        (S.COLLECTING, S.CANCELLED),
        (S.EVALUATED, S.CANCELLED),
        (S.PUBLISHING, S.CANCELLED),
    ],
)
def test_cancel_from_every_active_state(start: ServiceState, terminal: ServiceState) -> None:
    """Cancellation is legal from every active state and releases cleanly."""
    assert apply(start, E.CANCEL) is terminal
    assert is_terminal(terminal)
    assert apply(terminal, E.RELEASE) is S.RELEASED


@pytest.mark.parametrize(
    ("start", "terminal"),
    [
        (S.QUEUED, S.INFRA_ERROR),
        (S.STARTING, S.INFRA_ERROR),
        (S.RUNNING, S.INFRA_ERROR),
        (S.COLLECTING, S.INFRA_ERROR),
        (S.EVALUATED, S.INFRA_ERROR),
        (S.PUBLISHING, S.INFRA_ERROR),
    ],
)
def test_infra_error_from_every_active_state(start: ServiceState, terminal: ServiceState) -> None:
    """Infrastructure failure is a fail-closed terminal from any active state."""
    assert apply(start, E.INFRA) is terminal
    assert is_terminal(terminal)
    assert apply(terminal, E.RELEASE) is S.RELEASED


def failure_path() -> list[tuple[ServiceState, ServiceEvent, ServiceState]]:
    """Canonical path up through evaluation, then a findings FAIL verdict."""
    return [
        *CANONICAL_PATH[:4],  # ... -> EVALUATED
        (S.EVALUATED, E.FAIL, S.FAILED),
        (S.FAILED, E.RELEASE, S.RELEASED),
    ]


def test_findings_verdict_lands_in_failed_then_release() -> None:
    """A FAIL verdict terminates before publication and still releases."""
    state = S.QUEUED
    for pre, event, post in failure_path():
        assert state == pre
        state = apply(state, event)
        assert state == post
    assert state is S.RELEASED


# --- Duplicate events (idempotent no-op) -----------------------------------

@pytest.mark.parametrize(
    ("post", "event"),
    [
        (S.STARTING, E.DISPATCH),
        (S.RUNNING, E.STARTED),
        (S.COLLECTING, E.COLLECT),
        (S.EVALUATED, E.COLLECTED),
        (S.PUBLISHING, E.PASS),
        (S.PASSED, E.PUBLISHED),
        (S.FAILED, E.FAIL),
        (S.RELEASED, E.RELEASE),
        (S.INFRA_ERROR, E.INFRA),
        (S.CANCELLED, E.CANCEL),
    ],
)
def test_duplicate_event_is_an_idempotent_noop(post: ServiceState, event: ServiceEvent) -> None:
    """Re-delivering an event whose effect is already present changes nothing."""
    # First application (single active state whose only path leads to `post`).
    pre = _sole_precursor(post, event)
    first = apply(pre, event)
    assert first is post
    # Duplicate delivery leaves the state untouched.
    second = apply(first, event)
    assert second is first


def test_repeated_duplicate_deliveries_are_stable() -> None:
    """An event delivered many times never advances beyond its effect."""
    state = S.QUEUED
    state = apply(state, E.DISPATCH)
    for _ in range(5):
        assert apply(state, E.DISPATCH) is state  # no-op every time


# --- Reordered events (invalid) -------------------------------------------

@pytest.mark.parametrize(
    "event",
    [
        E.STARTED,
        E.COLLECT,
        E.COLLECTED,
        E.PASS,
        E.PUBLISHED,
        E.RELEASE,
    ],
)
def test_reordered_forward_event_raises(event: ServiceEvent) -> None:
    """A forward event delivered before its predecessor is rejected, not queued."""
    with pytest.raises(InvalidTransition):
        apply(S.QUEUED, event)


def test_infra_then_forward_event_raises() -> None:
    """After an infra terminal, a stale forward event can never re-open the job."""
    state = apply(S.RUNNING, E.INFRA)
    assert is_terminal(state)
    with pytest.raises(InvalidTransition):
        apply(state, E.COLLECT)


# --- Stale / superseded events --------------------------------------------

def test_stale_event_after_path_is_rejected() -> None:
    """An event from an already-passed stage cannot rewind or re-fire a job."""
    state = S.QUEUED
    for pre, event, post in CANONICAL_PATH:
        assert state == pre
        state = apply(state, event)
    # STARTED was consumed back at STARTING -> RUNNING; it is stale now.
    with pytest.raises(InvalidTransition):
        apply(state, E.STARTED)


def test_superseded_attempt_stale_event_is_rejected() -> None:
    """A cancelled job refuses the stale forward events of its cancelled attempt."""
    state = apply(apply(S.RUNNING, E.CANCEL), E.RELEASE)
    assert state is S.RELEASED
    for event in (E.COLLECT, E.COLLECTED, E.PASS, E.PUBLISHED):
        with pytest.raises(InvalidTransition):
            apply(state, event)


# --- Restart (state restored from storage continues identically) -----------

def test_restore_from_every_state_continues_identically() -> None:
    """A restored state applies the same remaining transitions as an uninterrupted run.

    This is the controller-restart property: the storage port can hand back any
    neutral state and the machine picks up exactly where it left off — there is
    no hidden in-memory context to lose.
    """
    for index, (pre, event, post) in enumerate(CANONICAL_PATH):
        restored = pre  # what a fresh controller loads from storage
        # The event that advanced us to `post` must still be applicable.
        assert apply(restored, event) is post
        # And a strictly later forward event (a skip) is rejected — the machine
        # cannot be jumped past an unconsumed step.
        for _later_index in range(index + 1, len(CANONICAL_PATH)):
            later_event = CANONICAL_PATH[_later_index][1]
            with pytest.raises(InvalidTransition):
                apply(restored, later_event)


def test_release_is_idempotent_after_restart() -> None:
    """A controller restarting in a terminal state re-releases without error."""
    for terminal in (S.PASSED, S.FAILED, S.INFRA_ERROR, S.CANCELLED):
        assert apply(terminal, E.RELEASE) is S.RELEASED
        assert apply(S.RELEASED, E.RELEASE) is S.RELEASED  # already released


# --- Whole-machine invariants ---------------------------------------------

def test_every_terminal_releases() -> None:
    """Every terminal state can be released, and release is final.

    Once released, no *other* event can move the job again — only a duplicate
    RELEASE (a controller retrying a release it may already have recorded) is
    tolerated as an idempotent no-op.
    """
    for terminal in (S.PASSED, S.FAILED, S.INFRA_ERROR, S.CANCELLED):
        assert is_terminal(terminal)
        assert apply(terminal, E.RELEASE) is S.RELEASED
    # After release, every non-release event is rejected.
    for event in ServiceEvent:
        if event is E.RELEASE:
            continue
        with pytest.raises(InvalidTransition):
            apply(S.RELEASED, event)


def test_active_states_are_never_terminal() -> None:
    """No active state is misreported as a terminal verdict."""
    for state in ACTIVE_STATES:
        assert not is_terminal(state)


# --- Helpers ---------------------------------------------------------------


def _sole_precursor(post: ServiceState, event: ServiceEvent) -> ServiceState:
    """Return the single legal precursor state for a given (event, post) edge.

    Kept local so tests read against the machine's own contract rather than a
    duplicated transition table.
    """
    # The canonical/failure paths enumerate every reachable edge; find the one
    # whose post matches. This is the machine's observable behavior, not an
    # independent re-implementation.
    for pre, ev, out in [*CANONICAL_PATH, *failure_path()]:
        if ev is event and out is post:
            return pre
    for state in ACTIVE_STATES:
        if apply(state, event) is post and state is not post:
            return state
    raise AssertionError(f"no precursor found for {post} via {event}")
