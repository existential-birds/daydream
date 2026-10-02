"""Coordinate retries across a run after consecutive retryable failures.

First attempts remain unrestricted. The circuit opens at its threshold and
later admits one half-open probe. Callers supply the clock; an RLock guards
shared state across concurrent invocations."""

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
    """Closed admits retries; open waits for the probe interval; half_open admits
    one probe. Only retries consult this circuit, so stale state cannot block
    a healthy first attempt."""

    def __init__(self, *, failure_threshold: int, probe_interval_s: float) -> None:
        self._failure_threshold = failure_threshold
        self._probe_interval_s = probe_interval_s
        self._lock = RLock()
        self._state = CIRCUIT_CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    def record_failure(self, now: float) -> bool:
        """Count a retryable failure. Return True when it opens or reopens the circuit,
        allowing that caller's already-admitted retry to finish."""
        with self._lock:
            self._consecutive_failures += 1
            # A failed probe re-opens the circuit for a full interval.
            if self._state == CIRCUIT_HALF_OPEN or (
                self._state == CIRCUIT_CLOSED
                and self._consecutive_failures >= self._failure_threshold
            ):
                self._state = CIRCUIT_OPEN
                self._opened_at = now
                return True
            return False

    def record_success(self) -> None:
        """Close the circuit and clear the consecutive-failure count."""
        with self._lock:
            self._state = CIRCUIT_CLOSED
            self._consecutive_failures = 0
            self._opened_at = None

    def admit_retry(self, now: float) -> CircuitAdmission:
        """Decide whether a retry may dispatch, transitioning as needed."""
        with self._lock:
            if self._state == CIRCUIT_CLOSED:
                return CircuitAdmission(True, CIRCUIT_CLOSED)
            if self._state == CIRCUIT_OPEN:
                assert self._opened_at is not None
                if now >= self._opened_at + self._probe_interval_s:
                    self._state = CIRCUIT_HALF_OPEN
                    return CircuitAdmission(True, CIRCUIT_HALF_OPEN)
                return CircuitAdmission(False, CIRCUIT_OPEN)
            return CircuitAdmission(False, CIRCUIT_HALF_OPEN)

    def state(self) -> str:
        """Return the observed state without transitioning it."""
        with self._lock:
            return self._state
