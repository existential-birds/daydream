"""Common executor conformance suite (DAYDREAM_SERVICE_V1).

Runs the SAME contract assertions against two structurally different hermetic
adapters so the neutrality of the common models is proven, not assumed:

- :class:`LocalExecutor` — real filesystem workspace, time-based asyncio
  lifecycle, durable on-disk state (restart reconciliation against real
  storage).
- :class:`ScriptedExecutor` — in-memory, step-based lifecycle with no I/O and
  no clock dependence (proves the suite never leaks filesystem/store/timing
  assumptions into the contract).

Adapters are exercised only through the ``ReviewExecutor`` port
(``start`` / ``inspect`` / ``cancel`` / ``collect`` / ``release``) and neutral
models. No vendor/SDK/worker-asserted infrastructure field may appear in a
common model, so each test also asserts that snapshots/envelopes carry no
infrastructure identity beyond the opaque handle.

Where the two lifecycles differ (Local needs time to reach a terminal state,
Scripted advances a step per inspect), ``settle`` polls ``inspect`` until the
execution is terminal. The suite has no adapter-typed branching.
"""

from __future__ import annotations

import asyncio
from dataclasses import fields
from typing import Any, Callable

import pytest

from daydream.executors.contract import (
    REQUIRED_CAPABILITIES,
    ExecutionOutcome,
    ExecutionRef,
    ExecutionStatus,
    ExecutorCapability,
    ExecutorError,
    ExecutorJob,
    UnknownExecutionError,
    require_capabilities,
)
from daydream.executors.local import LocalExecutor
from daydream.executors.protocol import ReviewExecutor, is_review_executor
from daydream.executors.scripted import ScriptedExecutor

_TERMINAL = frozenset(
    {
        ExecutionStatus.EVALUATED,
        ExecutionStatus.CANCELLED,
        ExecutionStatus.INFRA_ERROR,
        ExecutionStatus.RELEASED,
    }
)


async def settle(executor: ReviewExecutor, ref: ExecutionRef, *, attempts: int = 200) -> ExecutionStatus:
    """Poll ``inspect`` until the execution is terminal; return the final status.

    Local advances on wall-clock time, Scripted advances one step per inspect;
    polling normalizes both without the suite branching on adapter kind.
    """
    status = ExecutionStatus.QUEUED
    for _ in range(attempts):
        snapshot = await executor.inspect(ref)
        status = snapshot.status
        if status in _TERMINAL:
            return status
        await asyncio.sleep(0.005)
    return status


def _job(*, attempt: str = "attempt-1", key: str = "key-1", **payload: Any) -> ExecutorJob:
    return ExecutorJob(attempt_id=attempt, idempotency_key=key, payload=payload or {"lenses": ["python"]})


@pytest.fixture
def local_factory(tmp_path) -> Callable[[], LocalExecutor]:
    root = tmp_path / "local-root"

    def _make() -> LocalExecutor:
        return LocalExecutor(root, work_seconds=0.01)

    return _make


@pytest.fixture
def scripted_factory() -> Callable[[], ScriptedExecutor]:
    # Share ONE durable store across every instance the factory hands out, so
    # restart reconciliation (a fresh instance over the same backing sees a
    # prior ref) is exercised exactly as Local's shared on-disk root is.
    store: dict[str, dict[str, Any]] = {}

    def _make() -> ScriptedExecutor:
        return ScriptedExecutor(store)

    return _make


@pytest.fixture()
def factory(request: Any) -> Callable[[], ReviewExecutor]:
    """Parametrized over both hermetic adapters (local_factory / scripted_factory)."""
    return request.getfixturevalue(request.param)


PARAM_BOTH = pytest.mark.parametrize("factory", ["local_factory", "scripted_factory"], indirect=True)


# --- capability admission -------------------------------------------------


@PARAM_BOTH
def test_declared_capabilities_cover_required(factory: Callable[[], ReviewExecutor]) -> None:
    executor = factory()
    assert REQUIRED_CAPABILITIES.issubset(executor.capabilities)
    # The frozen contract requires ALL declared capabilities; a single optional
    # capability is not part of the required set, so admission only demands the
    # subset. Here both adapters declare exactly the required set.
    assert executor.capabilities == REQUIRED_CAPABILITIES


def test_capability_admission_rejects_missing_capability() -> None:
    partial = frozenset({ExecutorCapability.EXCLUSIVE_WORKSPACE})
    with pytest.raises(Exception, match="lacks required capabilities"):
        require_capabilities(set(partial), kind="partial")


# --- start / inspect lifecycle --------------------------------------------


@PARAM_BOTH
@pytest.mark.asyncio
async def test_start_returns_opaque_ref(factory: Callable[[], ReviewExecutor]) -> None:
    executor = factory()
    ref = await executor.start(_job())
    assert isinstance(ref.executor_kind, str) and ref.executor_kind
    assert isinstance(ref.adapter_version, int)
    assert isinstance(ref.opaque_handle, str) and ref.opaque_handle
    assert isinstance(ref.attempt_id, str) and ref.attempt_id


@PARAM_BOTH
@pytest.mark.asyncio
async def test_clean_execution_reaches_evaluated_and_collects(factory: Callable[[], ReviewExecutor]) -> None:
    executor = factory()
    ref = await executor.start(_job())
    status = await settle(executor, ref)
    assert status in (ExecutionStatus.EVALUATED, ExecutionStatus.INFRA_ERROR)
    envelope = await executor.collect(ref)
    assert envelope.ref == ref
    assert envelope.outcome in (ExecutionOutcome.CLEAN, ExecutionOutcome.FINDINGS)
    assert envelope.completed_lenses  # the configured lens is reported


@PARAM_BOTH
@pytest.mark.asyncio
async def test_findings_outcome_collected(factory: Callable[[], ReviewExecutor]) -> None:
    executor = factory()
    ref = await executor.start(_job(outcome="findings", lenses=["go"]))
    await settle(executor, ref)
    envelope = await executor.collect(ref)
    assert envelope.outcome == ExecutionOutcome.FINDINGS
    assert "go" in envelope.completed_lenses


@PARAM_BOTH
@pytest.mark.asyncio
async def test_inspect_unknown_ref_raises(factory: Callable[[], ReviewExecutor]) -> None:
    executor = factory()
    ghost = ExecutionRef(executor_kind=executor.kind, adapter_version=1, opaque_handle="ghost", attempt_id="nope")
    with pytest.raises(UnknownExecutionError):
        await executor.inspect(ghost)


def test_review_executor_structural_check() -> None:
    import tempfile
    from pathlib import Path

    assert is_review_executor(LocalExecutor(Path(tempfile.mkdtemp())))
    assert is_review_executor(ScriptedExecutor())
    assert not is_review_executor(object())


# --- idempotency ----------------------------------------------------------


@PARAM_BOTH
@pytest.mark.asyncio
async def test_repeat_start_is_idempotent(factory: Callable[[], ReviewExecutor]) -> None:
    executor = factory()
    job = _job(attempt="attempt-idem", key="idem-key")
    ref1 = await executor.start(job)
    ref2 = await executor.start(job)
    assert ref1.opaque_handle == ref2.opaque_handle
    assert ref1.attempt_id == ref2.attempt_id


@PARAM_BOTH
@pytest.mark.asyncio
async def test_distinct_attempts_are_distinct_executions(factory: Callable[[], ReviewExecutor]) -> None:
    executor = factory()
    a = await executor.start(_job(attempt="a", key="k"))
    b = await executor.start(_job(attempt="b", key="k"))
    assert a.opaque_handle != b.opaque_handle


# --- restart reconciliation ------------------------------------------------


@PARAM_BOTH
@pytest.mark.asyncio
async def test_fresh_instance_reconciles_prior_ref(factory: Callable[[], ReviewExecutor]) -> None:
    """A fresh executor instance (same durable backing) can inspect a prior ref.

    Local backs its store on disk under the shared root; Scripted shares the
    caller-owned store. Both satisfy restart reconciliation. Because the two
    adapters reconcile through different durable backing, the test asserts the
    observable contract (a second instance over the same backing sees the ref)
    without reaching into adapter internals.
    """
    executor1 = factory()
    ref = await executor1.start(_job(attempt="restart", key="r"))
    await settle(executor1, ref)
    second = factory()
    # Same durable backing (shared root/store) => the ref is still visible.
    snapshot = await second.inspect(ref)
    assert snapshot.ref == ref
    assert snapshot.status in _TERMINAL


# --- cancel ---------------------------------------------------------------


@PARAM_BOTH
@pytest.mark.asyncio
async def test_cancel_sets_cancelled(factory: Callable[[], ReviewExecutor]) -> None:
    executor = factory()
    ref = await executor.start(_job(attempt="cancel", key="c"))
    await executor.cancel(ref)
    await settle(executor, ref)
    envelope = await executor.collect(ref)
    assert envelope.outcome == ExecutionOutcome.CANCELLED


@PARAM_BOTH
@pytest.mark.asyncio
async def test_cancel_unknown_ref_raises(factory: Callable[[], ReviewExecutor]) -> None:
    executor = factory()
    ghost = ExecutionRef(executor_kind=executor.kind, adapter_version=1, opaque_handle="nope", attempt_id="x")
    with pytest.raises(UnknownExecutionError):
        await executor.cancel(ghost)


# --- collect / release ----------------------------------------------------


@PARAM_BOTH
@pytest.mark.asyncio
async def test_release_then_collect_does_not_return_artifacts(factory: Callable[[], ReviewExecutor]) -> None:
    """After deterministic release the execution is gone: collect must not
    return an evaluated artifact envelope for a released execution."""
    executor = factory()
    ref = await executor.start(_job(attempt="rel", key="r2"))
    await settle(executor, ref)
    await executor.release(ref, disposition="complete")
    with pytest.raises((UnknownExecutionError, ExecutorError)):
        await executor.collect(ref)


@PARAM_BOTH
@pytest.mark.asyncio
async def test_collect_nonterminal_raises(factory: Callable[[], ReviewExecutor]) -> None:
    executor = factory()
    ref = await executor.start(_job(attempt="nt", key="n2"))
    with pytest.raises(ExecutorError):
        await executor.collect(ref)


# --- vendor-error mapping --------------------------------------------------


@PARAM_BOTH
@pytest.mark.asyncio
async def test_vendor_error_maps_to_neutral_infra(factory: Callable[[], ReviewExecutor]) -> None:
    executor = factory()
    ref = await executor.start(_job(attempt="vendor", key="v", outcome="infra_error_vendor"))
    status = await settle(executor, ref)
    assert status == ExecutionStatus.INFRA_ERROR
    envelope = await executor.collect(ref)
    assert envelope.outcome == ExecutionOutcome.INFRA_ERROR


@PARAM_BOTH
@pytest.mark.asyncio
async def test_common_models_carry_no_vendor_fields(factory: Callable[[], ReviewExecutor]) -> None:
    """No adapter infrastructure identity may leak into snapshots/envelopes/refs.

    This is the contract's vendor-neutrality gate: the only identity fields are
    the neutral ref envelope. Any extra field is a STOP condition (adapter SDK
    object or worker-asserted infra identity entering common schema).
    """
    executor = factory()
    ref = await executor.start(_job(attempt="noinfra", key="n3"))
    await settle(executor, ref)
    snapshot = await executor.inspect(ref)
    ref_fields = {f.name for f in fields(ref)}
    snap_fields = {f.name for f in fields(snapshot)}
    assert not (ref_fields - {"executor_kind", "adapter_version", "opaque_handle", "attempt_id"})
    assert not (snap_fields - {"ref", "status", "started_at_iso", "completed_at_iso"})
    envelope = await executor.collect(ref)
    env_fields = {f.name for f in fields(envelope)}
    assert not (env_fields - {"ref", "outcome", "completed_lenses", "artifact_sha256"})
