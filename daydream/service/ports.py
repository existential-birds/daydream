"""Storage and executor ports for the durable review controller (Plan 008 Step 3).

This leaf owns the *contract* of these ports, not their implementations. The
transactional store (compare-and-set/leases/heartbeats/attempt-history/
recovery) is a separate leaf (leaf-C); the executor registry/conformance suite
is its own step (Step 4). The controller depends only on these Protocols, so
either can be swapped without changing the controller state machine.

Rules:

- ``ControllerStorage`` methods may raise ``StoreConflict`` to mean "the
  claimed/expected row moved under you — retry the read". They never raise for
  a normal not-found; ``load`` returns ``None``.
- Every method is async: the store and executor are out-of-process concerns, and
  the service is fully ``anyio``-driven like the rest of Daydream.
- No vendor field, SDK object, or worker-asserted infrastructure identity can
  cross either port surface.
"""

from __future__ import annotations

from typing import Protocol

from daydream.service.models import ControllerRecord, ExecutionRef, ExecutionSnapshot, JobSpec


class StoreConflict(Exception):
    """A compare-and-set claim or expected-precondition failed.

    Callers retry by re-reading the row. A real store maps this onto its CAS /
    optimistic-lock primitive; the conformance/leaf-C implementation decides the
    exact mechanics. The controller only needs this one signal.
    """


class JobAlreadyExists(Exception):
    """An ``insert`` was called for a ``job_id`` that already exists (idempotency violation)."""


class ControllerStorage(Protocol):
    """Persistently stores ``ControllerRecord`` rows for the controller.

    This is the port leaf-C implements with a transactional, durable store. The
    controller uses exactly the operations below and nothing else, so the store
    can offer compare-and-set claims, heartbeats, attempt history, and recovery
    without the controller knowing how.
    """

    async def insert(self, record: ControllerRecord) -> None:
        """Insert a new record; raise ``JobAlreadyExists`` if ``job_id`` is present.

        Idempotency: a caller that lost the race re-reads and finds its own row.
        """
        ...

    async def load(self, job_id: str) -> ControllerRecord | None:
        """Return the record for ``job_id`` or ``None`` when absent."""
        ...

    async def transition(
        self,
        job_id: str,
        *,
        expected_state: object,
        new_state: object,
    ) -> ControllerRecord:
        """Atomically move ``job_id`` from ``expected_state`` to ``new_state``.

        Raise ``StoreConflict`` when the current state differs from
        ``expected_state`` (a concurrent writer, lease, or stale CAS). Return
        the updated record otherwise.
        """
        ...

    async def bind_execution(
        self,
        job_id: str,
        *,
        expected_state: object,
        ref: ExecutionRef,
    ) -> ControllerRecord:
        """Persist ``ref`` for ``job_id`` under a CAS guard on ``expected_state``."""
        ...

    async def bind_artifacts(
        self,
        job_id: str,
        *,
        expected_state: object,
        hashes: tuple[str, ...],
        worker_verdict: str,
        blocked: bool = False,
        completed_lenses: frozenset[str] = frozenset(),
    ) -> ControllerRecord:
        """Persist collected artifact hashes + verdict, guarded by ``expected_state``.

        Also records whether a blocking finding is present and which lenses the
        artifact actually covered, so evaluation can fail closed on incomplete
        coverage or a blocking finding regardless of process exit code.
        """
        ...

    async def mark_superseded(self, job_id: str, *, by_invalidation: int) -> ControllerRecord:
        """Permanently mark ``job_id`` as superseded by a newer candidate."""
        ...

    async def bump_retry(self, job_id: str) -> ControllerRecord:
        """Increment the retry counter on ``job_id`` (new attempt claims one retry)."""
        ...

    async def reset_retry(
        self,
        job_id: str,
        *,
        expected_state: object,
        record: ControllerRecord,
    ) -> ControllerRecord:
        """Replace the durable row with ``record`` (a fresh queued attempt) under a CAS guard.

        Persists the full retry reset — cleared execution ref and worker
        verdict, bumped attempt number and retry counter — so ``load`` agrees
        with the record the controller returns instead of a row still carrying
        the failed attempt.
        """
        ...


class ReviewExecutor(Protocol):
    """Neutral compute/workspace adapter for a single review execution (DAYDREAM_SERVICE_V1).

    Methods are exactly ``start`` / ``inspect`` / ``cancel`` / ``collect`` /
    ``release``. The executor must prove the registered capabilities; admission
    rejects any executor that cannot. The controller treats ``ExecutionRef`` as
    opaque and never parses its handle.
    """

    async def start(self, spec: JobSpec) -> ExecutionRef:
        """Begin an execution for ``spec`` and return its opaque ``ExecutionRef``."""
        ...

    async def inspect(self, ref: ExecutionRef) -> ExecutionSnapshot:
        """Report the live state of a stored opaque ref (restart reconciliation)."""
        ...

    async def cancel(self, ref: ExecutionRef) -> None:
        """Strongly cancel the execution identified by ``ref``."""
        ...

    async def collect(self, ref: ExecutionRef) -> object:
        """Retrieve the passive artifact envelope for ``ref``."""
        ...

    async def release(self, ref: ExecutionRef, disposition: str) -> None:
        """Deterministically release the execution with a disposition string."""
        ...
