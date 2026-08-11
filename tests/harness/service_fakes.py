"""Hermetic in-memory storage + scripted executor fakes for service tests.

These implement the ``ControllerStorage`` and ``ReviewExecutor`` ports with
plain in-memory dicts, honoring the same CAS semantics the durable store leaf
will provide (via ``StoreConflict`` on optimistic-lock mismatch). They are test
doubles only — the real transactional store is a separate leaf (leaf-C) — and
contain no vendor or provider logic.

``ServiceStoreStorageAdapter`` is the bridge that presents a real
:class:`daydream.service.store.ServiceStore` implementation (the production
SQLite store, or the in-memory conformance store) behind the controller's
``ControllerStorage`` port, so acceptance tests can run the real controller
over the real store.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import cast

from daydream.service.models import (
    VERDICT_FINDINGS,
    ControllerRecord,
    ExecutionRef,
    ExecutionSnapshot,
    JobSpec,
    LensInventory,
    ReviewPolicy,
    ReviewTarget,
    RoundRecord,
    SourceOfTruth,
    TargetKind,
    TerminalOutcome,
)
from daydream.service.ports import (
    JobAlreadyExists,
    StoreConflict,
)
from daydream.service.states import ServiceState
from daydream.service.store import (
    ClaimStatus,
    JobNotFoundError,
    JobRecord,
    ServiceStore,
    StateConflictError,
)
from daydream.service.store import (
    ServiceState as StoreServiceState,
)

# Shared candidate-identity / protected-config digest constants so service
# tests do not re-derive the same magic strings in every file.
CANDIDATE_SHA = "a" * 40
CANDIDATE_TREE = "b" * 40
BASE_SHA = "c" * 40
DIFF_DIGEST = "d" * 64
CONFIG_DIGEST = "e" * 64
CONFIG_SOURCE = SourceOfTruth(ref="refs/heads/main", sha=BASE_SHA, digest=CONFIG_DIGEST)


def make_target(
    *,
    kind: TargetKind = TargetKind.PR_HEAD,
    config_digest: str = CONFIG_DIGEST,
    candidate_sha: str = CANDIDATE_SHA,
) -> ReviewTarget:
    return ReviewTarget(
        repo="acme/widgets",
        kind=kind,
        candidate_sha=candidate_sha,
        candidate_tree=CANDIDATE_TREE,
        base_sha=BASE_SHA,
        pr_number=77 if kind is TargetKind.PR_HEAD else None,
        merge_group_id="mg-1" if kind is TargetKind.MERGE_GROUP else None,
        diff_digest=DIFF_DIGEST,
        config_source=SourceOfTruth(ref="refs/heads/main", sha=BASE_SHA, digest=config_digest),
        invalidation_id="job-1",
    )


def make_policy(
    *,
    rounds: int = 2,
    complete_lens: tuple[str, ...] = ("python", "security"),
    concurrent_rounds: bool = True,
    publisher: str = "github-checks",
    check_name: str = "daydream/review",
    executor: str = "local-fake",
    config_digest: str = CONFIG_DIGEST,
) -> ReviewPolicy:
    return ReviewPolicy(
        backend="pi",
        provider="nous",
        model="deepseek/deepseek-v4-flash-0731",
        required_rounds=rounds,
        complete_lens=LensInventory(required=set(complete_lens)),
        executor=executor,
        concurrent_rounds=concurrent_rounds,
        immutable_reviewer_bundle="sha256:" + "0" * 64,
        deadline_s=1800.0,
        hard_budget_s=3600.0,
        publisher=publisher,
        check_name=check_name,
        source=SourceOfTruth(ref="refs/heads/main", sha=BASE_SHA, digest=config_digest),
    )


def make_round(
    *,
    attempt_id: str = "r1",
    target: ReviewTarget | None = None,
    outcome: TerminalOutcome = TerminalOutcome.CLEAN,
    completed_lenses: tuple[str, ...] = ("python", "security"),
    finding_count: int = 0,
    partial_artifacts: bool = False,
    execution_ref: str = "opaque:exec-1",
) -> RoundRecord:
    target = target or make_target()
    return RoundRecord(
        attempt_id=attempt_id,
        target=target,
        outcome=outcome,
        completed_lenses=set(completed_lenses),
        finding_count=finding_count,
        partial_artifacts=partial_artifacts,
        execution_ref=execution_ref,
    )


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

    async def reset_retry(self, job_id: str, *, expected_state, record: ControllerRecord) -> ControllerRecord:
        """Replace the durable row with the fresh retry record under a CAS guard."""
        return self._cas(job_id, expected_state, lambda _r: record)


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
class FakeScriptedExecutor:
    """Scripted ``ReviewExecutor`` test double for flow assertions.

    Attributes:
        envelopes: Queue of ``ArtifactEnvelope``-shaped dicts returned by ``collect``.
        start_raises: When set, ``start`` raises this exception.
        inspect_script: Optional list of ``ExecutionSnapshot`` for ``inspect``.
        inspect_raises: When set, ``inspect`` raises this exception (a conformant
            executor raises ``UnknownExecutionError`` for a lost execution).
    """

    envelopes: list[object] = field(default_factory=list)
    start_raises: Exception | None = None
    inspect_script: list[ExecutionSnapshot] = field(default_factory=list)
    inspect_raises: Exception | None = None
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
        if self.inspect_raises is not None:
            raise self.inspect_raises
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


class ServiceStoreStorageAdapter:
    """Test-side bridge: a real ``ServiceStore`` behind ``ControllerStorage``.

    The job state machine — ``state``, ``version``, the attempt/owner lease, and
    the append-only attempt ledger — lives in the wrapped store via
    ``create_job`` / ``claim`` / ``update_state``, so the real controller drives
    the production SQLite CAS/lease machinery end to end. The controller-only
    payload — the full ``JobSpec``, the opaque ``ExecutionRef``, the collected
    artifact hashes/verdict/lenses, and the superseded/retry markers — has no
    ``ServiceStore`` slot, so it lives in a job-keyed side table; that is the
    exact ABI gap documented in :mod:`daydream.service.ports` and
    :mod:`daydream.service.store`. The store row remains ground truth for
    ``state`` on every ``load``.
    """

    _OWNER = "controller"
    _TTL_SECONDS = 3600.0

    def __init__(self, store: ServiceStore) -> None:
        self._store = store
        self._records: dict[str, ControllerRecord] = {}

    # -- ControllerStorage ---------------------------------------------------

    async def insert(self, record: ControllerRecord) -> None:
        if self._store.get_job(record.job_id) is not None:
            raise JobAlreadyExists(record.job_id)
        self._store.create_job(
            JobRecord(
                job_id=record.job_id,
                idempotency_key=record.job_id,
                target_key=record.spec.candidate.candidate_sha,
                round=record.spec.attempt_number,
                state=_store_state(record.state),
                version=1,
            )
        )
        self._records[record.job_id] = record

    async def load(self, job_id: str) -> ControllerRecord | None:
        job = self._store.get_job(job_id)
        if job is None:
            return None
        record = self._records.get(job_id)
        if record is None:
            raise StoreConflict(f"job {job_id} exists in the store but has no controller record")
        # The store row is ground truth for state; the side table holds the
        # controller-only payload.
        return replace(record, state=ServiceState(job.state.value))

    async def transition(
        self,
        job_id: str,
        *,
        expected_state: object,
        new_state: object,
    ) -> ControllerRecord:
        record = await self._cas_load(job_id, expected_state)
        job = self._store.get_job(job_id)
        assert job is not None
        try:
            if job.current_attempt_id is None:
                # First advance on a fresh row: the real store acquires the lease
                # and opens the attempt ledger via claim.
                status = self._store.claim(
                    job_id,
                    f"a{record.spec.attempt_number}",
                    expected=frozenset({_store_state(record.state)}),
                    new_state=_store_state(cast(ServiceState, new_state)),
                    owner=self._OWNER,
                    execution_ref="",
                    now=_utcnow(),
                    ttl_seconds=self._TTL_SECONDS,
                )
                if status is not ClaimStatus.OK:
                    raise StoreConflict(f"claim rejected: {status.value}")
            else:
                owner = job.owner
                assert owner is not None
                self._store.update_state(
                    job_id,
                    from_state=_store_state(record.state),
                    to_state=_store_state(cast(ServiceState, new_state)),
                    attempt_id=job.current_attempt_id,
                    owner=owner,
                    now=_utcnow(),
                )
        except (JobNotFoundError, StateConflictError) as exc:
            raise StoreConflict(str(exc)) from exc
        self._records[job_id] = replace(record, state=new_state)  # type: ignore[arg-type]
        return self._records[job_id]

    async def bind_execution(
        self,
        job_id: str,
        *,
        expected_state: object,
        ref: ExecutionRef,
    ) -> ControllerRecord:
        record = await self._cas_load(job_id, expected_state)
        self._records[job_id] = replace(record, execution_ref=ref)
        return self._records[job_id]

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
        record = await self._cas_load(job_id, expected_state)
        self._records[job_id] = replace(
            record,
            artifact_hashes=hashes,
            worker_verdict=worker_verdict,
            blocked=blocked,
            completed_lenses=completed_lenses,
        )
        return self._records[job_id]

    async def mark_superseded(self, job_id: str, *, by_invalidation: int) -> ControllerRecord:
        record = await self.load(job_id)
        if record is None:
            raise StoreConflict(f"no row for {job_id}")
        self._records[job_id] = replace(record, superseded_by=by_invalidation)
        return self._records[job_id]

    async def bump_retry(self, job_id: str) -> ControllerRecord:
        record = await self._cas_load(job_id, ServiceState.INFRA_ERROR)
        self._records[job_id] = replace(record, retries_used=record.retries_used + 1)
        return self._records[job_id]

    async def reset_retry(
        self,
        job_id: str,
        *,
        expected_state: object,
        record: ControllerRecord,
    ) -> ControllerRecord:
        await self._cas_load(job_id, expected_state)
        # Move the durable row back to queued (fresh attempt) through the real
        # store's claim machinery so get_job/recoverable agree with the record
        # the controller returns.
        try:
            status = self._store.claim(
                job_id,
                f"a{record.spec.attempt_number}",
                expected=frozenset({_store_state(cast(ServiceState, expected_state))}),
                new_state=_store_state(record.state),
                owner=self._OWNER,
                execution_ref="",
                now=_utcnow(),
                ttl_seconds=self._TTL_SECONDS,
            )
            if status is not ClaimStatus.OK:
                raise StoreConflict(f"claim rejected: {status.value}")
        except JobNotFoundError as exc:
            raise StoreConflict(str(exc)) from exc
        self._records[job_id] = record
        return record

    # -- internals -----------------------------------------------------------

    async def _cas_load(self, job_id: str, expected_state: object) -> ControllerRecord:
        record = await self.load(job_id)
        if record is None:
            raise StoreConflict(f"no row for {job_id}")
        if record.state is not expected_state:
            raise StoreConflict(f"state mismatch: {record.state} != {expected_state}")
        return record


def _store_state(state: ServiceState) -> StoreServiceState:
    return StoreServiceState(state.value)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
