"""Hermetic in-memory storage + scripted executor fakes for service tests.

These implement the ``ControllerStorage`` and ``ReviewExecutor`` ports with
plain in-memory dicts, honoring the same CAS semantics the durable store leaf
will provide (via ``StoreConflict`` on optimistic-lock mismatch). They are test
doubles only — the real transactional store is a separate leaf (leaf-C) — and
contain no vendor or provider logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from daydream.service.models import (
    VERDICT_FINDINGS,
    ControllerRecord,
    ExecutionRef,
    ExecutionSnapshot,
    JobSpec,
)
from daydream.service.ports import (
    JobAlreadyExists,
    StoreConflict,
)
from daydream.service.states import ServiceState


class InMemoryStorage:
    """Minimal in-memory ``ControllerStorage`` honoring CAS semantics."""

    def __init__(self) -> None:
        self._rows: dict[str, ControllerRecord] = {}

    def _get(self, job_id: str) -> ControllerRecord | None:
        return self._rows.get(job_id)

    def _cas(self, job_id: str, expected: object, mutate) -> ControllerRecord:
        current = self._rows.get(job_id)
        if current is None:
            raise StoreConflict(f"no row for {job_id}")
        if current.state is not expected:
            raise StoreConflict(f"state mismatch: {current.state} != {expected}")
        self._rows[job_id] = mutate(current)
        return self._rows[job_id]

    async def insert(self, record: ControllerRecord) -> None:
        if record.job_id in self._rows:
            raise JobAlreadyExists(record.job_id)
        self._rows[record.job_id] = record

    async def load(self, job_id: str) -> ControllerRecord | None:
        return self._rows.get(job_id)

    def load_job(self, job_id: str) -> ControllerRecord:
        """Synchronous accessor for assertions; raises KeyError when absent."""
        return self._rows[job_id]

    async def transition(self, job_id: str, *, expected_state, new_state) -> ControllerRecord:
        return self._cas(job_id, expected_state, lambda r: _with_state(r, new_state))

    async def bind_execution(self, job_id: str, *, expected_state, ref: ExecutionRef) -> ControllerRecord:
        return self._cas(job_id, expected_state, lambda r: _with_ref(r, ref))

    async def bind_artifacts(
        self,
        job_id: str,
        *,
        expected_state,
        hashes,
        worker_verdict,
        blocked=False,
        completed_lenses=frozenset(),
    ) -> ControllerRecord:
        def _mutate(r: ControllerRecord) -> ControllerRecord:
            return ControllerRecord(
                job_id=r.job_id,
                spec=r.spec,
                state=r.state,
                execution_ref=r.execution_ref,
                artifact_hashes=hashes,
                worker_verdict=worker_verdict,
                retries_used=r.retries_used,
                superseded_by=r.superseded_by,
                blocked=blocked,
                completed_lenses=completed_lenses,
                trigger_ref=r.trigger_ref,
                store_fields=dict(r.store_fields),
            )

        return self._cas(job_id, expected_state, _mutate)

    async def mark_superseded(self, job_id: str, *, by_invalidation: int) -> ControllerRecord:
        def _mutate(r: ControllerRecord) -> ControllerRecord:
            return ControllerRecord(
                job_id=r.job_id,
                spec=r.spec,
                state=r.state,
                execution_ref=r.execution_ref,
                artifact_hashes=r.artifact_hashes,
                worker_verdict=r.worker_verdict,
                retries_used=r.retries_used,
                superseded_by=by_invalidation,
                blocked=r.blocked,
                completed_lenses=r.completed_lenses,
                trigger_ref=r.trigger_ref,
                store_fields=dict(r.store_fields),
            )

        current = self._rows.get(job_id)
        if current is None:
            raise StoreConflict(f"no row for {job_id}")
        self._rows[job_id] = _mutate(current)
        return self._rows[job_id]

    async def bump_retry(self, job_id: str) -> ControllerRecord:
        def _mutate(r: ControllerRecord) -> ControllerRecord:
            return ControllerRecord(
                job_id=r.job_id,
                spec=r.spec,
                state=r.state,
                execution_ref=r.execution_ref,
                artifact_hashes=r.artifact_hashes,
                worker_verdict=r.worker_verdict,
                retries_used=r.retries_used + 1,
                superseded_by=r.superseded_by,
                blocked=r.blocked,
                completed_lenses=r.completed_lenses,
                trigger_ref=r.trigger_ref,
                store_fields=dict(r.store_fields),
            )

        return self._cas(job_id, ServiceState.INFRA_ERROR, _mutate)


def _with_state(r: ControllerRecord, new_state) -> ControllerRecord:
    return ControllerRecord(
        job_id=r.job_id,
        spec=r.spec,
        state=new_state,
        execution_ref=r.execution_ref,
        artifact_hashes=r.artifact_hashes,
        worker_verdict=r.worker_verdict,
        retries_used=r.retries_used,
        superseded_by=r.superseded_by,
        blocked=r.blocked,
        completed_lenses=r.completed_lenses,
        trigger_ref=r.trigger_ref,
        store_fields=dict(r.store_fields),
    )


def _with_ref(r: ControllerRecord, ref: ExecutionRef) -> ControllerRecord:
    return ControllerRecord(
        job_id=r.job_id,
        spec=r.spec,
        state=r.state,
        execution_ref=ref,
        artifact_hashes=r.artifact_hashes,
        worker_verdict=r.worker_verdict,
        retries_used=r.retries_used,
        superseded_by=r.superseded_by,
        blocked=r.blocked,
        completed_lenses=r.completed_lenses,
        trigger_ref=r.trigger_ref,
        store_fields=dict(r.store_fields),
    )


@dataclass
class ScriptedExecutor:
    """Scripted ``ReviewExecutor`` test double for flow assertions.

    Attributes:
        envelopes: Queue of ``ArtifactEnvelope``-shaped dicts returned by ``collect``.
        start_raises: When set, ``start`` raises this exception.
        inspect_script: Optional list of ``ExecutionSnapshot`` for ``inspect``.
    """

    envelopes: list[object] = field(default_factory=list)
    start_raises: Exception | None = None
    inspect_script: list[ExecutionSnapshot] = field(default_factory=list)
    started: list[JobSpec] = field(default_factory=list)
    cancelled: list[ExecutionRef] = field(default_factory=list)
    released: list[tuple[ExecutionRef, str]] = field(default_factory=list)

    async def start(self, spec: JobSpec) -> ExecutionRef:
        if self.start_raises is not None:
            raise self.start_raises
        self.started.append(spec)
        return ExecutionRef(
            executor_kind="fake",
            adapter_version="1",
            opaque_handle=f"opaque-{len(self.started)}",
            attempt_id=f"attempt-{spec.attempt_number}",
        )

    async def inspect(self, ref: ExecutionRef) -> ExecutionSnapshot:
        if self.inspect_script:
            return self.inspect_script.pop(0)
        return ExecutionSnapshot(running=True, terminal=False)

    async def cancel(self, ref: ExecutionRef) -> None:
        self.cancelled.append(ref)

    async def collect(self, ref: ExecutionRef) -> object:
        if not self.envelopes:
            return {"worker_verdict": VERDICT_FINDINGS, "completed_lenses": [], "blocked": True}
        return self.envelopes.pop(0)

    async def release(self, ref: ExecutionRef, disposition: str) -> None:
        self.released.append((ref, disposition))
