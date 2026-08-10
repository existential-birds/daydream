"""Transactional controller store: port, models, and error taxonomy.

The controller (Plan 008, leaf-B) drives a durable state machine across
``queued -> starting -> running -> collecting -> evaluated -> publishing ->
passed|failed|infra_error|cancelled -> released``. That machine must sit on a
transactional, crash-safe store that gives it compare-and-set claims, leases and
heartbeats, append-only attempt history, idempotent event handling, and restart
recovery.

This module is the *storage port* the controller programs against plus the
neutral models both sides share. It deliberately contains no implementation and
no vendor/SDK/executor fields: the controller binds an opaque ``ExecutionRef``
to an attempt and the store persists and returns it verbatim, never parsing the
handle. Implementations live in ``store_memory.py`` (an in-memory conformance
double for hermetic tests) and ``store_sqlite.py`` (the production store).

State-machine *legality* (which transitions are permitted) is the controller's
concern. The store is a compare-and-set persistence layer: it guarantees that a
claim only lands when the job is already in an expected state and (for an active
lease) is owned by the claimant, that leases expire on their deadline, that
attempt history is append-only, and that idempotent/duplicate events are safe.

Hermetic: none of this touches GitHub, a provider, an executor, or the network.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping


def _utcnow() -> datetime:
    """Timezone-aware UTC now used as the durable default timestamp."""
    return datetime.now(timezone.utc)


class ServiceState(str, Enum):
    """Neutral durable lifecycle states shared by controller and store.

    Values are stable lowercase strings; they are what the store persists.
    ``passed``, ``failed``, ``infra_error`` and ``cancelled`` are the four
    terminal verdicts; ``released`` is the final disposition after bounded
    artifacts are externalized.
    """

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


#: States that are neither an in-flight attempt nor a released disposition.
#: Recovery scans everything outside the terminal set + released.
TERMINAL_STATES: frozenset[ServiceState] = frozenset(
    {ServiceState.PASSED, ServiceState.FAILED, ServiceState.INFRA_ERROR, ServiceState.CANCELLED}
)
NON_RECOVERABLE_STATES: frozenset[ServiceState] = TERMINAL_STATES | {ServiceState.RELEASED}


class ClaimStatus(str, Enum):
    """Outcome of a compare-and-set claim against a job.

    - ``OK``: the claim landed (or was an idempotent replay of the same claim).
    - ``CONFLICT``: the job is not in the expected state, so the transition is
      rejected. A consumer must re-read and re-retry (or treat the job as
      superseded).
    - ``LEASED``: the job is in the expected state but holds a live lease owned
      by a different claimant. Retry only after the lease expires.
    """

    OK = "ok"
    CONFLICT = "conflict"
    LEASED = "leased"


class StoreError(Exception):
    """Base error for the transactional store."""


class JobNotFoundError(StoreError):
    """A job/attempt the caller referenced does not exist."""


class IdempotencyError(StoreError):
    """A create_job violated the idempotency key contract."""


class StateConflictError(StoreError):
    """An owner-preserving update_state hit a divergent state/owner/attempt."""


@dataclass(frozen=True)
class JobRecord:
    """Mutable durable job row (immutable snapshot when read).

    Attributes:
        job_id: Stable job identity (the immutable review job id).
        idempotency_key: Logical identity used to deduplicate enqueue events.
        target_key: Opaque identity of the review target (candidate). Compared
            by the controller for supersession, never parsed by the store.
        round: Logical round this job belongs to.
        state: Current durable state.
        version: Optimistic-concurrency counter; bumped on every CAS transition.
        current_attempt_id: The attempt currently holding the job's lease.
        owner: Identity of the current lease holder (worker/attempt claimant).
        lease_expires_at: When the current lease lapses; ``None`` when idle.
        created_at / updated_at: Durable timestamps.
    """

    job_id: str
    idempotency_key: str
    target_key: str
    round: int
    state: ServiceState
    version: int = 1
    current_attempt_id: str | None = None
    owner: str | None = None
    lease_expires_at: datetime | None = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class AttemptRecord:
    """One append-only attempt row.

    The ``execution_ref`` is an opaque handle returned verbatim so the
    controller can ask the executor to :meth:`inspect` it on restart. It is
    never parsed by the store. ``artifact_refs`` are bounded name -> digest/uri
    pointers (non-secret) stored before release.
    """

    job_id: str
    attempt_id: str
    owner: str
    state: ServiceState
    execution_ref: str
    artifact_refs: tuple[tuple[str, str], ...] = ()
    externalized: bool = False
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class RecoverableAttempt:
    """A non-terminal job the controller must reconcile after a restart.

    The controller passes ``execution_ref`` back to the registered executor's
    ``inspect`` — never parses it here.
    """

    job_id: str
    attempt_id: str | None
    state: ServiceState
    execution_ref: str | None
    owner: str | None
    lease_expired: bool


class ServiceStore(ABC):
    """Transactional storage port for the review-service controller.

    Every mutating operation is atomic and, where documented, compare-and-set.
    Implementations are required to be safe under concurrent claimers (exactly
    one wins a CAS race). The SQLite production implementation additionally
    persists across process restarts; the in-memory implementation is the
    hermetic conformance double used by the shared test suite.
    """

    # -- jobs ----------------------------------------------------------------
    @abstractmethod
    def create_job(self, job: JobRecord) -> JobRecord:
        """Persist ``job`` idempotently by ``idempotency_key``.

        A re-create with the same idempotency key returns the canonical existing
        job. A re-create whose idempotency key maps to a *different* job id
        raises :class:`IdempotencyError`.
        """

    @abstractmethod
    def get_job(self, job_id: str) -> JobRecord | None:
        """Return the current job row, or ``None`` if absent."""

    # -- claims / state transitions -----------------------------------------
    @abstractmethod
    def claim(
        self,
        job_id: str,
        attempt_id: str,
        *,
        expected: frozenset[ServiceState],
        new_state: ServiceState,
        owner: str,
        execution_ref: str,
        now: datetime,
        ttl_seconds: float,
    ) -> ClaimStatus:
        """Compare-and-set transition that also (re)acquires a lease.

        Lands only when the job's current state is in ``expected``. If the job
        holds a live lease owned by another claimant, returns :data:`LEASED`.
        If it is already in ``new_state`` for the same attempt/owner (an
        idempotent replay of a claim whose response was lost), returns :data:`OK`
        without perturbation. On success binds ``execution_ref`` to the attempt.
        """

    @abstractmethod
    def update_state(
        self,
        job_id: str,
        from_state: ServiceState,
        to_state: ServiceState,
        *,
        attempt_id: str,
        owner: str,
        now: datetime,
    ) -> None:
        """Owned, owner-preserving CAS advance.

        Requires the job to be in ``from_state`` and its current attempt/owner
        to match. Raises :class:`StateConflictError` otherwise so the controller
        fails loud on a stale/divergent view instead of silently overwriting.
        """

    # -- leases / heartbeats -------------------------------------------------
    @abstractmethod
    def heartbeat(
        self,
        job_id: str,
        attempt_id: str,
        *,
        owner: str,
        now: datetime,
        ttl_seconds: float,
    ) -> bool:
        """Renew the lease for ``owner``; False if not owner or lease already lapsed."""

    # -- artifacts -----------------------------------------------------------
    @abstractmethod
    def bind_artifacts(
        self,
        job_id: str,
        attempt_id: str,
        *,
        owner: str,
        artifact_refs: Mapping[str, str],
    ) -> None:
        """Persist bounded (name -> digest/uri) artifact pointers for an attempt.

        Must run before the attempt's job is released. ``artifact_refs`` are
        bounded, non-secret references — never raw artifacts, secrets, or
        executor handles.
        """

    @abstractmethod
    def externalize(self, job_id: str, attempt_id: str, *, owner: str) -> None:
        """Mark the attempt's bounded artifacts as externalized (pre-release)."""

    # -- recovery / reads ----------------------------------------------------
    @abstractmethod
    def execution_ref(self, job_id: str, attempt_id: str) -> str | None:
        """Return the opaque execution ref for an attempt, verbatim."""

    @abstractmethod
    def attempt_history(self, job_id: str) -> list[AttemptRecord]:
        """Append-only attempt rows for a job, oldest first."""

    @abstractmethod
    def recoverable(self, *, now: datetime) -> list[RecoverableAttempt]:
        """All non-terminal jobs needing post-restart reconciliation.

        Includes queued jobs, in-flight attempts (live or expired leases), and
        paused states. ``lease_expired`` is True when a live lease lapsed.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any held resources (connections, file handles)."""
