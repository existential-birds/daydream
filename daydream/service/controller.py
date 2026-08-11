"""Durable neutral controller for the review service (Plan 008 Step 3).

The controller drives each job through the ``ServiceState`` machine, persisting
every transition through the ``ControllerStorage`` port and dispatching to a
registered ``ReviewExecutor``. It is deliberately *neutral*:

- It never parses an ``ExecutionRef`` handle — on restart it hands the stored
  opaque ref back to the executor via ``inspect``.
- It binds opaque execution references **separately** from collected artifact
  hashes, and rejects late/stale artifacts for a superseded or cancelled job
  without disturbing the live execution reference.
- It delegates the durable, transactional store to the storage port (leaf-C);
  this module implements the *policy* the store and executor implement.

Return values are the updated ``ControllerRecord`` so tests and callers read
ground truth from the store, never from in-memory assumption.
"""

from __future__ import annotations

import asyncio

from daydream.service.admission import AdmissionController
from daydream.service.models import (
    VERDICT_CLEAN,
    VERDICT_INFRA_ERROR,
    ArtifactEnvelope,
    ControllerRecord,
    JobSpec,
)
from daydream.service.ports import ControllerStorage, ReviewExecutor, StoreConflict
from daydream.service.states import InvalidTransition, ServiceEvent, ServiceState, apply

# :router-to-operator marker: when a job exhausts retries or cannot be admitted,
# the controller routes it onward instead of thrashing the worker profile.
ROUTE_TO_OPERATOR = "route-to-operator"


class ControllerError(Exception):
    """Base error for controller operations."""


class Superseded(ControllerError):
    """The job was superseded by a newer candidate and must not proceed."""


class LateArtifact(ControllerError):
    """An artifact arrived for a job that is already cancelled/superseded/released."""


class AdmissionBackoff:
    """A job could not be admitted and should be retried later (respect backoff)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason


class ServiceController:
    """Single-process controller over the neutral state machine.

    Args:
        storage: The durable storage port (implementation owned by leaf-C).
        executor: The registered review executor.
        admission: Global admission + retry budgets.
    """

    def __init__(
        self,
        storage: ControllerStorage,
        executor: ReviewExecutor,
        admission: AdmissionController | None = None,
    ) -> None:
        self._storage = storage
        self._executor = executor
        self._admission = admission or AdmissionController()
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, job_id: str) -> asyncio.Lock:
        return self._locks.setdefault(job_id, asyncio.Lock())

    # -- public entry points ----------------------------------------------

    async def enqueue(self, record: ControllerRecord) -> ControllerRecord:
        """Insert a new queued job, or return the existing one (idempotent)."""
        existing = await self._storage.load(record.job_id)
        if existing is not None:
            return existing
        await self._storage.insert(record)
        return record

    async def dispatch(self, job_id: str) -> ControllerRecord | AdmissionBackoff:
        """Move a queued job toward ``running`` by starting an execution.

        Admission is checked first: if any budget is saturated the job is left
        queued and an ``AdmissionBackoff`` is returned so the caller can retry —
        never a hard failure and never an unbounded Pi fan-out. A superseded job
        is cancelled rather than started.
        """
        async with self._lock(job_id):
            record = await self._require(job_id)
            if record.superseded_by is not None:
                await self._cancel(record)
                raise Superseded(f"job {job_id} superseded by invalidation {record.superseded_by}")
            if record.state in (
                ServiceState.STARTING,
                ServiceState.RUNNING,
                ServiceState.COLLECTING,
                ServiceState.EVALUATED,
            ):
                # Already dispatched (duplicate delivery) — idempotent no-op.
                # STARTING is included: a crash between the DISPATCH transition
                # and STARTED must not start the execution a second time.
                return record

            reason = self._admission.can_start(record.spec)
            if reason is not None:
                return AdmissionBackoff(reason)
            # Consume the admission slot atomically with the capacity check (no
            # await between check and act), so concurrent dispatches cannot
            # transiently exceed the cap and the failure path below releases
            # exactly what this dispatch claimed.
            self._admission.start(record.spec)

            # queued -> starting (claim the slot).
            record = await self._transition(record, ServiceEvent.DISPATCH)
            try:
                ref = await self._executor.start(record.spec)
            except Exception:
                # Executor could not begin; fail closed to infra so retry_infra
                # decides routing instead of this dispatch pass.
                self._admission.release(record.spec)
                return await self._transition(record, ServiceEvent.INFRA)

            record = await self._storage.bind_execution(
                record.job_id, expected_state=record.state, ref=ref
            )
            # starting -> running only after the ref is durably bound.
            return await self._transition(record, ServiceEvent.STARTED)

    async def collect(self, job_id: str) -> ControllerRecord:
        """Gather and persist the passive artifact, then move to ``evaluated``.

        Rejects late artifacts: if the job is superseded or settled, the
        artifact is refused even though the execution reference is left
        untouched. The admission slot for the execution is released here —
        once the artifact is collected the Pi/execution work is done, even
        fictionally before the cheap publisher runs.
        """
        async with self._lock(job_id):
            record = await self._require(job_id)
            if self._settled(record):
                raise LateArtifact(f"artifact for {job_id} rejected: {record.state.value}")
            if record.state is ServiceState.EVALUATED and record.worker_verdict is not None:
                # Already collected once (duplicate delivery) — idempotent.
                return record
            if record.execution_ref is None:
                raise ControllerError(f"job {job_id} has no execution ref to collect")

            envelope = self._coerce_envelope(await self._executor.collect(record.execution_ref))
            record = await self._transition(record, ServiceEvent.COLLECT)
            record = await self._storage.bind_artifacts(
                record.job_id,
                expected_state=record.state,
                hashes=envelope.artifact_hashes,
                worker_verdict=envelope.worker_verdict,
                blocked=envelope.blocked,
                completed_lenses=envelope.completed_lenses,
            )
            record = await self._transition(record, ServiceEvent.COLLECTED)
            self._admission.release(record.spec)
            return record

    async def evaluate(self, job_id: str) -> ControllerRecord:
        """Decide the terminal outcome from the collected verdict.

        A ``clean`` verdict with no blocking finding and every required lens
        covered moves to ``publishing`` via ``pass``; anything else fails
        closed to ``failed``. Infra verdicts are left ``infra_error`` for
        explicit retry routing — never silently converted to clean.
        """
        async with self._lock(job_id):
            record = await self._require(job_id)
            if record.worker_verdict is None:
                raise ControllerError(f"job {job_id} cannot be evaluated before collection")

            if record.worker_verdict == VERDICT_INFRA_ERROR:
                # Move the collected infra verdict to the terminal infra_error
                # state: EVALUATED is active, not terminal, so a job left here
                # can never be retried (retry_infra's CAS expects INFRA_ERROR),
                # its execution is never released, and restart reconciliation
                # wedges it in EVALUATED forever.
                return await self._transition(record, ServiceEvent.INFRA)

            complete = record.spec.required_lenses <= record.completed_lenses
            if record.worker_verdict == VERDICT_CLEAN and not record.blocked and complete:
                return await self._transition(record, ServiceEvent.PASS)
            return await self._transition(record, ServiceEvent.FAIL)

    async def publish(self, job_id: str) -> ControllerRecord:
        """Complete publication to ``passed`` (the only positive terminal)."""
        async with self._lock(job_id):
            record = await self._require(job_id)
            record = await self._transition(record, ServiceEvent.PUBLISHED)
            await self._release_execution(record)
            return record

    async def cancel(self, job_id: str) -> ControllerRecord:
        """Strongly cancel the job, releasing the execution deterministically."""
        async with self._lock(job_id):
            record = await self._require(job_id)
            return await self._cancel(record)

    async def release(self, job_id: str) -> ControllerRecord:
        """Move a settled job to the final ``released`` state and release its execution.

        ``RELEASE`` is the final, idempotent transition from any terminal state
        (``passed``/``failed``/``infra_error``/``cancelled``); the execution is
        deterministically released afterward. Active jobs must reach a terminal
        first (``publish`` for a clean round, ``cancel`` to interrupt), so a
        stray release cannot wedge the machine open.
        """
        async with self._lock(job_id):
            record = await self._require(job_id)
            record = await self._transition(record, ServiceEvent.RELEASE)
            await self._release_execution(record)
            return record

    async def supersede(self, job_id: str, *, by_invalidation: int) -> ControllerRecord:
        """Cancel an in-flight job because a newer candidate head replaced it.

        The old execution is cancelled and any later artifact is rejected; the
        newer candidate is admitted through its own idempotent enqueue.
        """
        async with self._lock(job_id):
            record = await self._require(job_id)
            record = await self._storage.mark_superseded(job_id, by_invalidation=by_invalidation)
            return await self._cancel(record)

    async def reconcile_restart(self, job_id: str) -> ControllerRecord:
        """Reconcile a persisted job against live execution after a restart.

        Never parses the handle: asks the registered executor to inspect the
        stored opaque ref, then aligns the neutral state. Settled jobs are left
        as-is; never-started jobs stay queued (for a future dispatch pick-up).
        """
        async with self._lock(job_id):
            record = await self._require(job_id)
            if self._settled(record):
                return record
            if record.execution_ref is None:
                return record  # not started before restart; still queued/starting

            snapshot = await self._executor.inspect(record.execution_ref)
            if not snapshot.running and not snapshot.terminal:
                # Execution vanished without a terminal — fail closed.
                return await self._cancel(record)
            return record

    async def retry_infra(self, job_id: str) -> ControllerRecord | str:
        """Retry a classified infra failure with a fresh attempt, if budget allows.

        Returns a record reset to ``queued`` when a retry is admitted, or the
        ``ROUTE_TO_OPERATOR`` marker when the retry budget is exhausted.
        """
        async with self._lock(job_id):
            record = await self._require(job_id)
            if not self._admission.infra_retry_available(record.spec):
                return ROUTE_TO_OPERATOR
            self._admission.record_infra_retry(record.spec)
            self._admission.release(record.spec)
            # Deterministically release the failed execution before the durable
            # row forgets it — the retried attempt must never reuse it.
            await self._release_execution(record)
            # Reset the durable row to queued with a fresh attempt spec.
            fresh = ControllerRecord(
                job_id=record.job_id,
                spec=JobSpec(
                    job_id=record.spec.job_id,
                    candidate=record.spec.candidate,
                    service=record.spec.service,
                    backend=record.spec.backend,
                    provider=record.spec.provider,
                    model=record.spec.model,
                    required_lenses=record.spec.required_lenses,
                    attempt_number=record.spec.attempt_number + 1,
                ),
                state=ServiceState.QUEUED,
                execution_ref=None,
                artifact_hashes=(),
                worker_verdict=None,
                retries_used=record.retries_used + 1,
                superseded_by=record.superseded_by,
                trigger_ref=record.trigger_ref,
                store_fields=dict(record.store_fields),
            )
            # Persist the full reset (cleared execution ref/verdict, bumped
            # attempt) via the storage port so load() agrees with the returned
            # record instead of a QUEUED row still carrying the failed attempt.
            return await self._storage.reset_retry(
                record.job_id, expected_state=ServiceState.INFRA_ERROR, record=fresh
            )

    # -- internals -----------------------------------------------------------

    async def _require(self, job_id: str) -> ControllerRecord:
        record = await self._storage.load(job_id)
        if record is None:
            raise ControllerError(f"unknown job {job_id}")
        return record

    async def _transition(self, record: ControllerRecord, event: ServiceEvent) -> ControllerRecord:
        # Duplicate no-op: the state machine already reflects this event.
        if apply(record.state, event) is record.state:
            return record
        try:
            new_state = apply(record.state, event)
        except InvalidTransition:
            raise ControllerError(
                f"illegal transition {record.state.value} -[{event.value}]-> ?"
            ) from None
        try:
            return await self._storage.transition(
                record.job_id,
                expected_state=record.state,
                new_state=new_state,
            )
        except StoreConflict:
            # Lost a CAS to a concurrent/durable writer; re-read and return truth.
            fresh = await self._storage.load(record.job_id)
            if fresh is None:
                raise ControllerError(f"job {record.job_id} vanished mid-transition") from None
            return fresh

    async def _cancel(self, record: ControllerRecord) -> ControllerRecord:
        record = await self._transition(record, ServiceEvent.CANCEL)
        self._admission.release(record.spec)
        await self._release_execution(record)
        return record

    async def _release_execution(self, record: ControllerRecord) -> None:
        if record.execution_ref is not None:
            await self._executor.release(record.execution_ref, disposition=record.state.value)

    @staticmethod
    def _settled(record: ControllerRecord) -> bool:
        return record.state in (
            ServiceState.PASSED,
            ServiceState.FAILED,
            ServiceState.INFRA_ERROR,
            ServiceState.CANCELLED,
            ServiceState.RELEASED,
        )

    @staticmethod
    def _coerce_envelope(envelope: object) -> ArtifactEnvelope:
        if isinstance(envelope, ArtifactEnvelope):
            return envelope
        if isinstance(envelope, dict):
            if "worker_verdict" not in envelope:
                # Fail closed: a missing verdict must never default to clean.
                raise ControllerError(
                    "executor returned a dict envelope without a 'worker_verdict'"
                )
            return ArtifactEnvelope(
                worker_verdict=str(envelope["worker_verdict"]),
                completed_lenses=frozenset(envelope.get("completed_lenses", [])),
                artifact_hashes=tuple(envelope.get("artifact_hashes", [])),
                blocked=bool(envelope.get("blocked", False)),
            )
        raise ControllerError(f"executor returned unknown artifact type {type(envelope)!r}")
