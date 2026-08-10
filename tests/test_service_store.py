"""Transactional controller store: conformance + state-machine/property tests.

Covers the storage port the controller (leaf-B) calls: compare-and-set claims,
heartbeats/lease expiry, attempt history, idempotency, and restart recovery.
The SAME suite runs against both the in-memory conformance implementation and
the production SQLite implementation so neither can drift from the port.

State-machine legality (which transitions are permitted) is the controller's
concern; the store is a compare-and-set persistence layer and must hold the
CAS, lease, and recovery invariants the controller relies on.

Hermetic only: no GitHub, provider, Sprite, executor, or network.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from daydream.service.store import (
    ClaimStatus,
    IdempotencyError,
    JobNotFoundError,
    JobRecord,
    ServiceState,
    ServiceStore,
    StateConflictError,
    StoreError,
)
from daydream.service.store_memory import InMemoryServiceStore
from daydream.service.store_sqlite import SqliteServiceStore

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _job(job_id: str, *, state: ServiceState = ServiceState.QUEUED) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        idempotency_key=f"idem-{job_id}",
        target_key=f"target-{job_id}",
        round=1,
        state=state,
        version=1,
        current_attempt_id=None,
        owner=None,
        lease_expires_at=None,
        created_at=T0,
        updated_at=T0,
    )


def _claim_job(
    s: ServiceStore,
    job_id: str,
    attempt: str,
    *,
    expected: set[ServiceState],
    new: ServiceState,
    owner: str,
    ref: str,
    now: datetime = T0,
    ttl: float = 30.0,
) -> ClaimStatus:
    return s.claim(
        job_id,
        attempt,
        expected=frozenset(expected),
        new_state=new,
        owner=owner,
        execution_ref=ref,
        now=now,
        ttl_seconds=ttl,
    )


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Generator[Callable[[str], ServiceStore], None, None]:
    kind: str = request.param
    opened: list[ServiceStore] = []

    def _open(name: str) -> ServiceStore:
        s: ServiceStore
        if kind == "memory":
            s = InMemoryServiceStore()
        else:
            s = SqliteServiceStore(path=tmp_path / f"{name}.db")
        opened.append(s)
        return s

    yield _open
    for s in opened:
        s.close()


# --------------------------------------------------------------------------- #
# CAS claims
# --------------------------------------------------------------------------- #


def test_claim_cas_transition(store: Callable[[str], ServiceStore]) -> None:
    s = store("cas")
    s.create_job(_job("j1"))

    status = _claim_job(s, "j1", "a1", expected={ServiceState.QUEUED}, new=ServiceState.STARTING, owner="w1", ref="r1")
    assert status == ClaimStatus.OK
    record = s.get_job("j1")
    assert record is not None
    assert record.state == ServiceState.STARTING
    assert record.owner == "w1"
    assert record.current_attempt_id == "a1"
    assert record.version == 2


def test_claim_conflict_rejected(store: Callable[[str], ServiceStore]) -> None:
    s = store("conflict")
    s.create_job(_job("j1"))
    _claim_job(s, "j1", "a1", expected={ServiceState.QUEUED}, new=ServiceState.STARTING, owner="w1", ref="r1")

    # A stale expected state must not overwrite the already-advanced record.
    res = _claim_job(s, "j1", "a2", expected={ServiceState.QUEUED}, new=ServiceState.RUNNING, owner="w2", ref="r2")
    assert res == ClaimStatus.CONFLICT
    record = s.get_job("j1")
    assert record is not None
    assert record.state == ServiceState.STARTING
    assert record.current_attempt_id == "a1"


def test_claim_leased_by_other(store: Callable[[str], ServiceStore]) -> None:
    s = store("leased")
    s.create_job(_job("j1"))
    _claim_job(s, "j1", "a1", expected={ServiceState.QUEUED}, new=ServiceState.RUNNING, owner="w1", ref="r1")
    # Different owner + still-live lease -> LEASED (not CONFLICT).
    res = _claim_job(s, "j1", "a2", expected={ServiceState.RUNNING}, new=ServiceState.COLLECTING, owner="w2", ref="r2")
    assert res == ClaimStatus.LEASED


def test_claim_after_lease_expiry_reclaims(store: Callable[[str], ServiceStore]) -> None:
    s = store("reclaim")
    s.create_job(_job("j1"))
    _claim_job(s, "j1", "a1", expected={ServiceState.QUEUED}, new=ServiceState.RUNNING, owner="w1", ref="r1")
    later = T0 + timedelta(seconds=31)  # past the 30s ttl
    res = _claim_job(
        s, "j1", "a2", expected={ServiceState.RUNNING}, new=ServiceState.RUNNING,
        owner="w2", ref="r2", now=later,
    )
    assert res == ClaimStatus.OK
    record = s.get_job("j1")
    assert record is not None
    assert record.owner == "w2"


def test_claim_missing_job_raises(store: Callable[[str], ServiceStore]) -> None:
    s = store("missing")
    with pytest.raises(JobNotFoundError):
        _claim_job(s, "nope", "a1", expected={ServiceState.QUEUED}, new=ServiceState.STARTING, owner="w1", ref="r1")


def test_claim_is_idempotent_for_same_attempt(store: Callable[[str], ServiceStore]) -> None:
    """Redelivering the same claim event must be a no-op, not a competing lease."""
    s = store("dup-claim")
    s.create_job(_job("j1"))
    r1 = _claim_job(s, "j1", "a1", expected={ServiceState.QUEUED}, new=ServiceState.RUNNING, owner="w1", ref="r1")
    r2 = _claim_job(s, "j1", "a1", expected={ServiceState.QUEUED}, new=ServiceState.RUNNING, owner="w1", ref="r1")
    assert r1 == ClaimStatus.OK
    assert r2 == ClaimStatus.OK
    record = s.get_job("j1")
    assert record is not None
    assert record.owner == "w1"
    assert s.execution_ref("j1", "a1") == "r1"


def test_cas_race_exactly_one_winner(store: Callable[[str], ServiceStore]) -> None:
    """N concurrent claims for the same transition: exactly one lands the CAS."""
    s = store("race")
    s.create_job(_job("j1"))

    def attempt(i: int) -> ClaimStatus:
        return _claim_job(
            s, "j1", f"a{i}", expected={ServiceState.QUEUED}, new=ServiceState.STARTING,
            owner=f"w{i}", ref=f"r{i}",
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        statuses = list(pool.map(attempt, range(64)))

    assert statuses.count(ClaimStatus.OK) == 1
    record = s.get_job("j1")
    assert record is not None
    assert record.state == ServiceState.STARTING
    winner = statuses.index(ClaimStatus.OK)
    assert record.current_attempt_id == f"a{winner}"
    assert record.owner == f"w{winner}"


# --------------------------------------------------------------------------- #
# Heartbeats & lease expiry
# --------------------------------------------------------------------------- #


def test_heartbeat_renews(store: Callable[[str], ServiceStore]) -> None:
    s = store("hb")
    s.create_job(_job("j1"))
    _claim_job(s, "j1", "a1", expected={ServiceState.QUEUED}, new=ServiceState.RUNNING, owner="w1", ref="r1")
    # Correct owner renews.
    assert s.heartbeat("j1", "a1", owner="w1", now=T0 + timedelta(seconds=10), ttl_seconds=30.0) is True
    # Wrong owner cannot renew a valid lease.
    assert s.heartbeat("j1", "a1", owner="w2", now=T0 + timedelta(seconds=11), ttl_seconds=30.0) is False
    # Expired lease cannot be renewed by its former owner (no live lease).
    assert s.heartbeat("j1", "a1", owner="w1", now=T0 + timedelta(seconds=100), ttl_seconds=30.0) is False


def test_lease_expiry_recovery(store: Callable[[str], ServiceStore]) -> None:
    s = store("recover")
    s.create_job(_job("j1"))
    s.create_job(_job("j2"))
    _claim_job(s, "j1", "a1", expected={ServiceState.QUEUED}, new=ServiceState.RUNNING, owner="w1", ref="sprite://opaque-1")
    _claim_job(s, "j2", "a1", expected={ServiceState.QUEUED}, new=ServiceState.RUNNING, owner="w2", ref="sprite://opaque-2")
    # Renew j2 so it is still live.
    assert s.heartbeat("j2", "a1", owner="w2", now=T0 + timedelta(seconds=20), ttl_seconds=30.0) is True

    expired = T0 + timedelta(seconds=40)
    pending = {r.job_id: r for r in s.recoverable(now=expired)}

    # On restart the controller reconciles every non-terminal job. A live lease
    # (j2, renewed past +40) is still listed but flagged lease_expired=False; an
    # expired lease (j1) is flagged True. Both carry their opaque execution ref.
    j1 = pending.get("j1")
    assert j1 is not None
    assert j1.execution_ref == "sprite://opaque-1"
    assert j1.lease_expired is True
    assert j1.state == ServiceState.RUNNING

    j2 = pending.get("j2")
    assert j2 is not None
    assert j2.lease_expired is False
    assert j2.execution_ref == "sprite://opaque-2"


# --------------------------------------------------------------------------- #
# Attempt history & opaque execution refs
# --------------------------------------------------------------------------- #


def test_execution_ref_persisted_opaque(store: Callable[[str], ServiceStore]) -> None:
    s = store("opaque")
    opaque = "KIND/1/sprite://host/pool/instance"
    s.create_job(_job("j1"))
    _claim_job(s, "j1", "a1", expected={ServiceState.QUEUED}, new=ServiceState.RUNNING, owner="w1", ref=opaque)
    # Returned verbatim for executor.inspect; the store never parses it.
    assert s.execution_ref("j1", "a1") == opaque
    history = s.attempt_history("j1")
    assert len(history) == 1
    assert history[0].attempt_id == "a1"
    assert history[0].execution_ref == opaque


def test_attempt_history_append_only(store: Callable[[str], ServiceStore]) -> None:
    s = store("history")
    s.create_job(_job("j1"))
    # Three attempts on the same job by one controller-owned worker, each claiming
    # the state the prior reached and recording its own opaque execution ref.
    chain = [
        ("a1", ServiceState.QUEUED, ServiceState.STARTING, "r-a1"),
        ("a2", ServiceState.STARTING, ServiceState.RUNNING, "r-a2"),
        ("a3", ServiceState.RUNNING, ServiceState.COLLECTING, "r-a3"),
    ]
    for i, (attempt, pre, post, ref) in enumerate(chain, start=1):
        status = _claim_job(
            s, "j1", attempt, expected={pre}, new=post, owner="w1", ref=ref,
            now=T0 + timedelta(seconds=i),
        )
        assert status == ClaimStatus.OK
    history = s.attempt_history("j1")
    assert [a.attempt_id for a in history] == ["a1", "a2", "a3"]
    assert history[0].execution_ref == "r-a1"
    assert history[2].execution_ref == "r-a3"


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #


def test_idempotent_create(store: Callable[[str], ServiceStore]) -> None:
    s = store("idem")
    first = s.create_job(_job("j1"))
    second = s.create_job(_job("j1"))
    assert first.job_id == second.job_id == "j1"
    assert s.get_job("j1") is not None


def test_create_conflicts_on_distinct_target(store: Callable[[str], ServiceStore]) -> None:
    s = store("idem-conflict")
    # Two jobs SHARE an idempotency key -> the second enqueue must be rejected.
    j1 = _job("j1")
    clash = _job("jX")
    s.create_job(j1)

    with pytest.raises(IdempotencyError):
        s.create_job(replace(clash, idempotency_key=j1.idempotency_key))


# --------------------------------------------------------------------------- #
# Restart recovery
# --------------------------------------------------------------------------- #


def test_restart_mid_transition_reconciliation(store: Callable[[str], ServiceStore]) -> None:
    """Controller crash in every non-terminal state; recovery lists each with its opaque ref."""
    s = store("restart")
    transitions = [
        ("j-queued", ServiceState.QUEUED, ServiceState.QUEUED),
        ("j-starting", ServiceState.QUEUED, ServiceState.STARTING),
        ("j-running", ServiceState.STARTING, ServiceState.RUNNING),
        ("j-collecting", ServiceState.RUNNING, ServiceState.COLLECTING),
        ("j-evaluated", ServiceState.RUNNING, ServiceState.EVALUATED),
        ("j-publishing", ServiceState.EVALUATED, ServiceState.PUBLISHING),
    ]
    for idx, (job_id, pre, post) in enumerate(transitions):
        s.create_job(_job(job_id, state=pre))
        _claim_job(
            s, job_id, "a1", expected={pre}, new=post,
            owner="w", ref=f"r-{job_id}", now=T0 + timedelta(seconds=idx), ttl=300.0,
        )

    # A fully released job is never recoverable.
    s.create_job(_job("j-done", state=ServiceState.PUBLISHING))
    _claim_job(
        s, "j-done", "a1", expected={ServiceState.PUBLISHING}, new=ServiceState.PUBLISHING,
        owner="w", ref="r-done", now=T0, ttl=300.0,
    )
    s.update_state(
        "j-done", ServiceState.PUBLISHING, ServiceState.PASSED,
        attempt_id="a1", owner="w", now=T0 + timedelta(seconds=1),
    )
    s.update_state(
        "j-done", ServiceState.PASSED, ServiceState.RELEASED,
        attempt_id="a1", owner="w", now=T0 + timedelta(seconds=1),
    )

    recoverable = {r.job_id: r for r in s.recoverable(now=T0 + timedelta(seconds=5))}
    for job_id in ("j-queued", "j-starting", "j-running", "j-collecting", "j-evaluated", "j-publishing"):
        assert job_id in recoverable, f"{job_id} should be recoverable after crash"
        assert recoverable[job_id].execution_ref == f"r-{job_id}"
    assert "j-done" not in recoverable


def test_execution_ref_after_restart_for_inspect(store: Callable[[str], ServiceStore]) -> None:
    """After reopening a persisted SQLite store, the opaque ref survives for executor.inspect.

    The in-memory store is explicitly ephemeral, so this persistence guarantee is
    SQLite-only.
    """
    s = store("inspect")
    from daydream.service.store_sqlite import SqliteServiceStore

    if not isinstance(s, SqliteServiceStore):
        pytest.skip("persistence is a SQLite-only guarantee")
    s.create_job(_job("j1"))
    _claim_job(s, "j1", "a1", expected={ServiceState.QUEUED}, new=ServiceState.RUNNING, owner="w1", ref="sprite://opaque-ref")
    s.close()

    s2 = store("inspect")
    assert s2.execution_ref("j1", "a1") == "sprite://opaque-ref"
    rec = s2.recoverable(now=T0 + timedelta(minutes=1))
    assert any(r.job_id == "j1" and r.execution_ref == "sprite://opaque-ref" and r.lease_expired for r in rec)


# --------------------------------------------------------------------------- #
# Error hierarchy
# --------------------------------------------------------------------------- #


def test_error_hierarchy() -> None:
    assert issubclass(JobNotFoundError, StoreError)
    assert issubclass(IdempotencyError, StoreError)
    # Port returns are the enum, not unrelated sentinels.
    assert {*ClaimStatus} == {ClaimStatus.OK, ClaimStatus.CONFLICT, ClaimStatus.LEASED}


def test_released_jobs_not_recoverable(store: Callable[[str], ServiceStore]) -> None:
    s = store("released")
    s.create_job(_job("j1"))
    _claim_job(s, "j1", "a1", expected={ServiceState.QUEUED}, new=ServiceState.RUNNING, owner="w1", ref="r1")
    s.update_state("j1", ServiceState.RUNNING, ServiceState.PASSED, attempt_id="a1", owner="w1", now=T0)
    s.update_state("j1", ServiceState.PASSED, ServiceState.RELEASED, attempt_id="a1", owner="w1", now=T0)
    assert all(r.job_id != "j1" for r in s.recoverable(now=T0 + timedelta(minutes=1)))


# --------------------------------------------------------------------------- #
# Bounded artifacts / externalize-before-release
# --------------------------------------------------------------------------- #


def test_bind_artifacts_then_externalize(store: Callable[[str], ServiceStore]) -> None:
    """Bounded artifact refs bind before release and externalize without loss."""
    s = store("artifacts")
    s.create_job(_job("j1"))
    _claim_job(s, "j1", "a1", expected={ServiceState.QUEUED}, new=ServiceState.RUNNING, owner="w1", ref="r1")
    s.bind_artifacts("j1", "a1", owner="w1", artifact_refs={"findings.json": "sha256:abc", "summary.md": "sha256:def"})
    s.externalize("j1", "a1", owner="w1")

    history = s.attempt_history("j1")
    assert history[0].externalized is True
    assert dict(history[0].artifact_refs) == {"findings.json": "sha256:abc", "summary.md": "sha256:def"}

    s.update_state("j1", ServiceState.RUNNING, ServiceState.PASSED, attempt_id="a1", owner="w1", now=T0)
    s.update_state("j1", ServiceState.PASSED, ServiceState.RELEASED, attempt_id="a1", owner="w1", now=T0)


def test_bind_artifacts_requires_owner(store: Callable[[str], ServiceStore]) -> None:
    s = store("artifact-owner")
    s.create_job(_job("j1"))
    _claim_job(s, "j1", "a1", expected={ServiceState.QUEUED}, new=ServiceState.RUNNING, owner="w1", ref="r1")
    # A different worker cannot bind artifacts to an attempt it does not own.
    with pytest.raises(StateConflictError):
        s.bind_artifacts("j1", "a1", owner="w2", artifact_refs={"x": "1"})


def test_externalize_cannot_steal_lease(store: Callable[[str], ServiceStore]) -> None:
    s = store("artifact-steal")
    s.create_job(_job("j1"))
    _claim_job(s, "j1", "a1", expected={ServiceState.QUEUED}, new=ServiceState.RUNNING, owner="w1", ref="r1")
    with pytest.raises(StateConflictError):
        s.externalize("j1", "a1", owner="w2")
