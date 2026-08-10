"""Real-path tests for the Local executor (daydream/executors/local.py).

These test the Local adapter's *real* observable behavior — genuine on-disk
workspace creation, release cleanup ordering, durable state across a fresh
instance (restart reconciliation), and vendor-error mapping — entering through
its public async methods with a real filesystem. Hermetic: no network, no
external backend.

The Local executor is dev/test infrastructure and is NOT claimed safe for
merge-authorizing untrusted code; these tests prove its lifecycle contract,
not any sandbox.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from daydream.executors.contract import (
    ExecutionOutcome,
    ExecutionRef,
    ExecutionStatus,
    ExecutorError,
    ExecutorJob,
)
from daydream.executors.local import LocalExecutor

_TERMINAL = frozenset(
    {ExecutionStatus.EVALUATED, ExecutionStatus.CANCELLED, ExecutionStatus.INFRA_ERROR, ExecutionStatus.RELEASED}
)


@pytest.fixture
def executor(tmp_path: Path) -> LocalExecutor:
    return LocalExecutor(tmp_path / "root", work_seconds=0.01)


@pytest.fixture
def job() -> ExecutorJob:
    return ExecutorJob(attempt_id="attempt-1", idempotency_key="key-1", payload={"lenses": ["python", "go"]})


async def _settle(executor: LocalExecutor, ref: ExecutionRef) -> ExecutionStatus:
    status = ExecutionStatus.QUEUED
    for _ in range(200):
        status = (await executor.inspect(ref)).status
        if status in _TERMINAL:
            return status
        await asyncio.sleep(0.005)
    return status


# --- real workspace -------------------------------------------------------


@pytest.mark.asyncio
async def test_start_creates_real_workspace_dir(executor: LocalExecutor, job: ExecutorJob) -> None:
    ref = await executor.start(job)
    assert exists(ref, executor)
    # The execution workspace is a real directory under the shared root.
    assert (executor.root / ref.opaque_handle).is_dir()


@pytest.mark.asyncio
async def test_release_deterministically_removes_workspace(executor: LocalExecutor, job: ExecutorJob) -> None:
    ref = await executor.start(job)
    await _settle(executor, ref)
    await executor.release(ref, disposition="complete")
    # Real cleanup ordering: no artifacts, no state, no workspace directory remain.
    exec_dir = executor.root / ref.opaque_handle
    assert not exec_dir.exists()
    # The shared root and other state survive.
    assert executor.root.is_dir()


# --- idempotency on real storage ------------------------------------------


@pytest.mark.asyncio
async def test_restart_fresh_instance_reconciles_prior_ref(tmp_path: Path) -> None:
    """A brand-new LocalExecutor over the same on-disk root sees a prior ref."""
    root = tmp_path / "root"
    first = LocalExecutor(root, work_seconds=0.01)
    job = ExecutorJob(attempt_id="restart", idempotency_key="r", payload={"lenses": ["python"]})
    ref = await first.start(job)
    await _settle(first, ref)
    second = LocalExecutor(root)  # fresh instance, same durable root
    snapshot = await second.inspect(ref)
    assert snapshot.ref == ref
    assert snapshot.status in _TERMINAL
    envelope = await second.collect(ref)
    assert envelope.outcome in (ExecutionOutcome.CLEAN, ExecutionOutcome.FINDINGS)
    await second.release(ref, disposition="complete")


# --- vendor error mapping --------------------------------------------------


@pytest.mark.asyncio
async def test_vendor_error_is_mapped_not_rethrown(tmp_path: Path) -> None:
    executor = LocalExecutor(tmp_path / "root", work_seconds=0.01)
    ref = await executor.start(
        ExecutorJob(attempt_id="vendor", idempotency_key="v", payload={"outcome": "infra_error_vendor"})
    )
    status = await _settle(executor, ref)
    assert status == ExecutionStatus.INFRA_ERROR
    envelope = await executor.collect(ref)
    assert envelope.outcome == ExecutionOutcome.INFRA_ERROR


@pytest.mark.asyncio
async def test_collect_on_cancelled_yields_cancelled_outcome(executor: LocalExecutor, job: ExecutorJob) -> None:
    ref = await executor.start(job)
    await executor.cancel(ref)
    envelope = await executor.collect(ref)
    assert envelope.outcome == ExecutionOutcome.CANCELLED


@pytest.mark.asyncio
async def test_unknown_ref_raises_uniform_error(executor: LocalExecutor) -> None:
    ghost = ExecutionRef(executor_kind="local", adapter_version=1, opaque_handle="absent", attempt_id="x")
    with pytest.raises(ExecutorError):
        await executor.inspect(ghost)


@pytest.mark.asyncio
async def test_admission_gate_local_declares_required_capabilities(tmp_path: Path) -> None:
    from daydream.executors.contract import REQUIRED_CAPABILITIES

    executor = LocalExecutor(tmp_path / "root")
    assert REQUIRED_CAPABILITIES.issubset(executor.capabilities)
    assert executor.kind == "local"


def exists(ref: ExecutionRef, executor: LocalExecutor) -> bool:
    return (executor.root / ref.opaque_handle / "state.json").is_file()
