"""One run-scoped outage circuit that gate-keeps retry decisions.

When a backend is genuinely down, every concurrent invocation otherwise burns
its own retry ladder against the same dead endpoint. The circuit collapses those
ladders into one coordinated decision: after a threshold of consecutive
retryable failures it opens, and retries (never first attempts) are suppressed
until a single half-open probe is admitted.

State is pure and caller-clocked: every method takes an explicit ``now`` so the
circuit never reads a clock of its own and tests can drive it deterministically.
A single :class:`threading.RLock` guards the fields, mirroring
``run_context.py``'s run-scoped lock discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

#: The three observable circuit states; ``state()`` never returns anything else.
CIRCUIT_CLOSED = "closed"
CIRCUIT_OPEN = "open"
CIRCUIT_HALF_OPEN = "half_open"


@dataclass(frozen=True)
class CircuitAdmission:
    """The retry decision for one failure, plus the state it was made in."""

    allowed: bool
    state: str


class OutageCircuit:
    """Run-scoped consecutive-failure breaker for retry ladders.

    ``closed`` admits every retry; ``open`` suppresses retries until the probe
    interval has elapsed; ``half_open`` admits exactly one probe at a time. The
    circuit only ever decides whether a *retry* may dispatch -- first attempts
    are never consulted, so stale open state cannot block a healthy call.
    """

    def __init__(self, *, failure_threshold: int, probe_interval_s: float) -> None:
        self._failure_threshold = failure_threshold
        self._probe_interval_s = probe_interval_s
        self._lock = RLock()
        self._state = CIRCUIT_CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._probe_outstanding = False

    def record_failure(self, now: float) -> None:
        """Count one consecutive retryable failure, opening at the threshold."""
        with self._lock:
            self._consecutive_failures += 1
            if self._state == CIRCUIT_HALF_OPEN:
                # A failed probe re-opens the circuit for a full interval.
                self._state = CIRCUIT_OPEN
                self._opened_at = now
                self._probe_outstanding = False
            elif self._state == CIRCUIT_CLOSED and (
                self._consecutive_failures >= self._failure_threshold
            ):
                self._state = CIRCUIT_OPEN
                self._opened_at = now
                self._probe_outstanding = False

    def record_success(self) -> None:
        """Close the circuit and clear the consecutive-failure count."""
        with self._lock:
            self._state = CIRCUIT_CLOSED
            self._consecutive_failures = 0
            self._opened_at = None
            self._probe_outstanding = False

    def admit_retry(self, now: float) -> CircuitAdmission:
        """Decide whether a retry may dispatch, transitioning as needed."""
        with self._lock:
            if self._state == CIRCUIT_CLOSED:
                return CircuitAdmission(True, CIRCUIT_CLOSED)
            if self._state == CIRCUIT_OPEN:
                assert self._opened_at is not None
                if now >= self._opened_at + self._probe_interval_s:
                    self._state = CIRCUIT_HALF_OPEN
                    self._probe_outstanding = True
                    return CircuitAdmission(True, CIRCUIT_HALF_OPEN)
                return CircuitAdmission(False, CIRCUIT_OPEN)
            if self._probe_outstanding:
                return CircuitAdmission(False, CIRCUIT_HALF_OPEN)
            self._probe_outstanding = True
            return CircuitAdmission(True, CIRCUIT_HALF_OPEN)

    def state(self, now: float) -> str:
        """Return the observed state at *now* without transitioning it."""
        with self._lock:
            return self._state
