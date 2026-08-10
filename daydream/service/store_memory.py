"""In-memory conformance implementation of :class:`ServiceStore`.

This is the hermetic storage double used by the shared ``test_service_store``
suite and by the controller's own tests. Applications MUST swap it for
:class:`daydream.service.store_sqlite.SqliteServiceStore` in production — it is
process-local, ephemeral, and explicitly NOT durable across restarts.

Correctness invariants (same as the SQLite store):
- exactly one claimer wins a compare-and-set race,
- a live lease is held by exactly one owner,
- attempt history is append-only,
- idempotent claim/create events are safe no-ops.
"""

from __future__ import annotations

from datetime import datetime
from threading import RLock
from typing import Mapping

from daydream.service.store import (
    NON_RECOVERABLE_STATES,
    AttemptRecord,
    ClaimStatus,
    IdempotencyError,
    JobNotFoundError,
    JobRecord,
    RecoverableAttempt,
    ServiceState,
    ServiceStore,
    StateConflictError,
)


class InMemoryServiceStore(ServiceStore):
    """Thread-safe, in-process-only ServiceStore for tests and conformance."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        # Append-only attempt ledger: (job_id, attempt_id) -> AttemptRecord.
        self._attempts: dict[tuple[str, str], AttemptRecord] = {}
        self._lock = RLock()

    # ------------------------------------------------------------------ jobs
    def create_job(self, job: JobRecord) -> JobRecord:
        with self._lock:
            existing = self._jobs.get(job.job_id)
            if existing is not None:
                if existing.idempotency_key != job.idempotency_key:
                    raise IdempotencyError(
                        f"job {job.job_id!r} re-created with a different idempotency key"
                    )
                return existing
            # Idempotency key is a global logical identity: it must not alias two jobs.
            for other in self._jobs.values():
                if other.idempotency_key == job.idempotency_key:
                    raise IdempotencyError(
                        f"idempotency key {job.idempotency_key!r} already bound to job {other.job_id!r}"
                    )
            self._jobs[job.job_id] = job
            return job

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    # ---------------------------------------------------------- claims / SM
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
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)

            status = _derive_claim_status(job, attempt_id, expected, new_state, owner, now)
            if status is not ClaimStatus.OK:
                return status

            new_job = _with(
                job,
                state=new_state,
                current_attempt_id=attempt_id,
                owner=owner,
                lease_expires_at=_after(now, ttl_seconds),
                updated_at=now,
                version=job.version + 1,
            )
            self._jobs[job_id] = new_job
            self._attempts.setdefault(
                (job_id, attempt_id),
                AttemptRecord(
                    job_id=job_id,
                    attempt_id=attempt_id,
                    owner=owner,
                    state=new_state,
                    execution_ref=execution_ref,
                    created_at=now,
                ),
            )
            return ClaimStatus.OK

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
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job.state != from_state or job.current_attempt_id != attempt_id or job.owner != owner:
                raise StateConflictError(
                    f"job {job_id} not in {from_state.value}/{attempt_id}/{owner} "
                    f"(state={job.state.value}, attempt={job.current_attempt_id}, owner={job.owner})"
                )
            self._jobs[job_id] = _with(job, state=to_state, updated_at=now, version=job.version + 1)

    # ------------------------------------------------------------ heartbeats
    def heartbeat(
        self,
        job_id: str,
        attempt_id: str,
        *,
        owner: str,
        now: datetime,
        ttl_seconds: float,
    ) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job.owner != owner or job.current_attempt_id != attempt_id:
                return False
            if job.lease_expires_at is None or job.lease_expires_at <= now:
                return False
            self._jobs[job_id] = _with(
                job, lease_expires_at=_after(now, ttl_seconds), updated_at=now, version=job.version + 1
            )
            return True

    # -------------------------------------------------------------- artifacts
    def bind_artifacts(
        self,
        job_id: str,
        attempt_id: str,
        *,
        owner: str,
        artifact_refs: Mapping[str, str],
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job.owner != owner or job.current_attempt_id != attempt_id:
                raise StateConflictError(f"job {job_id} not owned by {owner}/{attempt_id}")
            key = (job_id, attempt_id)
            current = self._attempts.get(key)
            if current is None:
                raise JobNotFoundError(f"attempt {attempt_id} for job {job_id}")
            self._attempts[key] = _with_attempt(
                current, artifact_refs=tuple(sorted(artifact_refs.items()))
            )

    def externalize(self, job_id: str, attempt_id: str, *, owner: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job.owner != owner or job.current_attempt_id != attempt_id:
                raise StateConflictError(f"job {job_id} not owned by {owner}/{attempt_id}")
            key = (job_id, attempt_id)
            current = self._attempts.get(key)
            if current is None:
                raise JobNotFoundError(f"attempt {attempt_id} for job {job_id}")
            self._attempts[key] = _with_attempt(current, externalized=True)

    # ------------------------------------------------------------- reads
    def execution_ref(self, job_id: str, attempt_id: str) -> str | None:
        with self._lock:
            rec = self._attempts.get((job_id, attempt_id))
            return rec.execution_ref if rec is not None else None

    def attempt_history(self, job_id: str) -> list[AttemptRecord]:
        with self._lock:
            return [rec for (jid, _aid), rec in self._attempts.items() if jid == job_id]

    def recoverable(self, *, now: datetime) -> list[RecoverableAttempt]:
        with self._lock:
            out: list[RecoverableAttempt] = []
            for job in self._jobs.values():
                if job.state in NON_RECOVERABLE_STATES:
                    continue
                ref: str | None = None
                if job.current_attempt_id is not None:
                    rec = self._attempts.get((job.job_id, job.current_attempt_id))
                    ref = rec.execution_ref if rec is not None else None
                lease_expired = job.lease_expires_at is not None and job.lease_expires_at <= now
                out.append(
                    RecoverableAttempt(
                        job_id=job.job_id,
                        attempt_id=job.current_attempt_id,
                        state=job.state,
                        execution_ref=ref,
                        owner=job.owner,
                        lease_expired=lease_expired,
                    )
                )
            return out

    def close(self) -> None:
        pass


def _derive_claim_status(
    job: JobRecord,
    attempt_id: str,
    expected: frozenset[ServiceState],
    new_state: ServiceState,
    owner: str,
    now: datetime,
) -> ClaimStatus:
    """Compute a claim outcome without mutating state.

    Idempotent replay (same attempt, same owner, already in ``new_state``) reads
    as OK regardless of the expected set — a lost controller response must be
    safe to redeliver.
    """
    if job.state == new_state and job.current_attempt_id == attempt_id and job.owner == owner:
        return ClaimStatus.OK
    if job.state not in expected:
        return ClaimStatus.CONFLICT
    # Only a live, mismatched lease blocks; an expired or absent lease is claimable.
    if job.owner is not None and job.owner != owner and job.lease_expires_at is not None:
        if job.lease_expires_at > now:
            return ClaimStatus.LEASED
    return ClaimStatus.OK


def _with(job: JobRecord, **changes: object) -> JobRecord:
    """Return a new JobRecord snapshot with ``changes`` applied."""
    return JobRecord(
        job_id=job.job_id,
        idempotency_key=job.idempotency_key,
        target_key=job.target_key,
        round=job.round,
        state=changes.get("state", job.state),  # type: ignore[arg-type]
        version=changes.get("version", job.version),  # type: ignore[arg-type]
        current_attempt_id=changes.get("current_attempt_id", job.current_attempt_id),  # type: ignore[arg-type]
        owner=changes.get("owner", job.owner),  # type: ignore[arg-type]
        lease_expires_at=changes.get("lease_expires_at", job.lease_expires_at),  # type: ignore[arg-type]
        created_at=job.created_at,
        updated_at=changes.get("updated_at", job.updated_at),  # type: ignore[arg-type]
    )


def _with_attempt(rec: AttemptRecord, **changes: object) -> AttemptRecord:
    """Return a new AttemptRecord snapshot with ``changes`` applied."""
    return AttemptRecord(
        job_id=rec.job_id,
        attempt_id=rec.attempt_id,
        owner=changes.get("owner", rec.owner),  # type: ignore[arg-type]
        state=changes.get("state", rec.state),  # type: ignore[arg-type]
        execution_ref=changes.get("execution_ref", rec.execution_ref),  # type: ignore[arg-type]
        artifact_refs=changes.get("artifact_refs", rec.artifact_refs),  # type: ignore[arg-type]
        externalized=changes.get("externalized", rec.externalized),  # type: ignore[arg-type]
        created_at=rec.created_at,
    )


def _after(now: datetime, ttl_seconds: float) -> datetime:
    from datetime import timedelta

    return now + timedelta(seconds=ttl_seconds)
