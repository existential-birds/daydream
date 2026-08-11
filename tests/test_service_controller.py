"""Controller tests (Plan 008 Step 3, leaf-B).

Each test drives the real ``ServiceController`` against an in-memory storage
fake (honoring CAS semantics) and a scripted executor fake. The tests assert
observable outcomes — persisted state, artifact rejection, cancellation,
retry routing — never that a function "was called". They exercise the exact
duplicate / reordered / restarted / cancelled / stale events the property
suite covers at the pure state-machine level, plus the controller's storage
and executor interaction.
"""

from __future__ import annotations

import pytest

from daydream.executors import UnknownExecutionError
from daydream.service.admission import AdmissionController, Budgets
from daydream.service.controller import (
    ROUTE_TO_OPERATOR,
    AdmissionBackoff,
    LateArtifact,
    ServiceController,
    Superseded,
)
from daydream.service.models import (
    VERDICT_CLEAN,
    CandidateTarget,
    ControllerRecord,
    ExecutionSnapshot,
    JobSpec,
)
from daydream.service.states import ServiceState
from tests.harness.service_fakes import FakeScriptedExecutor, InMemoryStorage

S = ServiceState


def _spec(job_id: str = "j1", *, lens: frozenset[str] | None = None, attempt: int = 1) -> JobSpec:
    return JobSpec(
        job_id=job_id,
        candidate=CandidateTarget(
            target_kind="pr_head",
            repo="owner/repo",
            candidate_sha="c" * 40,
            tree_digest="t" * 40,
            base_sha="b" * 40,
            invalidation_id=1,
        ),
        service="svc",
        backend="pi",
        provider="nous",
        model="deepseek-v4-flash-0731",
        required_lenses=lens or frozenset({"py"}),
        attempt_number=attempt,
    )


def _record(job_id: str = "j1", *, lens: frozenset[str] | None = None) -> ControllerRecord:
    return ControllerRecord(job_id=job_id, spec=_spec(job_id, lens=lens))


def _fresh(
    lens: frozenset[str] | None = None,
    *,
    retries: int | None = None,
) -> tuple[ServiceController, InMemoryStorage, FakeScriptedExecutor]:
    storage = InMemoryStorage()
    executor = FakeScriptedExecutor()
    budgets = Budgets(retries={"svc": retries}) if retries is not None else Budgets()
    admission = AdmissionController(budgets)
    controller = ServiceController(storage, executor, admission=admission)
    return controller, storage, executor


async def _started(controller: ServiceController) -> ControllerRecord:
    result = await controller.dispatch("j1")
    assert isinstance(result, ControllerRecord), f"expected record, got {result!r}"
    return result


# --- happy path: clean round publishes --------------------------------------


async def test_clean_round_publishes() -> None:
    controller, _, executor = _fresh(lens=frozenset({"py"}))
    await controller.enqueue(_record(lens=frozenset({"py"})))
    started = await _started(controller)
    assert started.state is S.RUNNING
    assert started.execution_ref is not None
    assert executor.started[0].attempt_number == 1

    executor.envelopes.append(
        {"worker_verdict": VERDICT_CLEAN, "completed_lenses": ["py"], "artifact_hashes": ("h1",)}
    )
    collected = await controller.collect("j1")
    assert collected.state is S.EVALUATED
    assert collected.artifact_hashes == ("h1",)
    assert collected.completed_lenses == frozenset({"py"})

    evaluated = await controller.evaluate("j1")
    assert evaluated.state is S.PUBLISHING

    published = await controller.publish("j1")
    assert published.state is S.PASSED
    assert executor.released and executor.released[0][1] == "passed"


# --- duplicate events (idempotent) ------------------------------------------


async def test_enqueue_is_idempotent() -> None:
    controller, storage, _ = _fresh()
    await controller.enqueue(_record())
    await controller.enqueue(_record())
    assert storage.load_job("j1").job_id == "j1"


async def test_duplicate_dispatch_does_not_double_start() -> None:
    controller, _, executor = _fresh()
    await controller.enqueue(_record())
    await _started(controller)
    result = await controller.dispatch("j1")
    assert isinstance(result, ControllerRecord) and result.state is S.RUNNING
    assert len(executor.started) == 1  # no second start


async def test_duplicate_collect_is_stable() -> None:
    controller, _, executor = _fresh(lens=frozenset({"py"}))
    await controller.enqueue(_record(lens=frozenset({"py"})))
    await _started(controller)
    executor.envelopes.append(
        {"worker_verdict": VERDICT_CLEAN, "completed_lenses": ["py"], "artifact_hashes": ("h1",)}
    )
    first = await controller.collect("j1")
    assert first.state is S.EVALUATED
    # A duplicated collection is a no-op, not an error (transition is stable).
    again = await controller.collect("j1")
    assert again.state is S.EVALUATED


# --- reordered / stale events ------------------------------------------------


async def test_collect_before_dispatch_raises() -> None:
    controller, _, _ = _fresh()
    await controller.enqueue(_record())
    with pytest.raises(Exception):  # no execution ref -> ControllerError
        await controller.collect("j1")


async def test_evaluate_before_collect_raises() -> None:
    controller, _, _ = _fresh()
    await controller.enqueue(_record())
    with pytest.raises(Exception):  # no worker verdict -> ControllerError
        await controller.evaluate("j1")


async def test_publish_before_evaluate_rejected() -> None:
    controller, _, executor = _fresh(lens=frozenset({"py"}))
    await controller.enqueue(_record(lens=frozenset({"py"})))
    await _started(controller)
    executor.envelopes.append(
        {"worker_verdict": VERDICT_CLEAN, "completed_lenses": ["py"], "artifact_hashes": ()}
    )
    await controller.collect("j1")
    # Still EVALUATED; publish before evaluate is an illegal (reordered) step.
    with pytest.raises(Exception):
        await controller.publish("j1")


# --- cancelled ---------------------------------------------------------------


async def test_cancel_from_running_releases_execution() -> None:
    controller, _, executor = _fresh()
    await controller.enqueue(_record())
    await _started(controller)
    cancelled = await controller.cancel("j1")
    assert cancelled.state is S.CANCELLED
    assert executor.released and executor.released[0][1] == "cancelled"


async def test_cancelled_job_rejects_late_artifact() -> None:
    controller, _, executor = _fresh(lens=frozenset({"py"}))
    await controller.enqueue(_record(lens=frozenset({"py"})))
    await _started(controller)
    await controller.cancel("j1")
    executor.envelopes.append(
        {"worker_verdict": VERDICT_CLEAN, "completed_lenses": ["py"], "artifact_hashes": ()}
    )
    with pytest.raises(LateArtifact):
        await controller.collect("j1")


# --- superseded head ---------------------------------------------------------


async def test_superseding_head_cancels_old_and_rejects_artifact() -> None:
    controller, _, executor = _fresh(lens=frozenset({"py"}))
    await controller.enqueue(_record(lens=frozenset({"py"})))
    await _started(controller)

    superseded = await controller.supersede("j1", by_invalidation=2)
    assert superseded.state is S.CANCELLED
    assert superseded.superseded_by == 2
    executor.envelopes.append(
        {"worker_verdict": VERDICT_CLEAN, "completed_lenses": ["py"], "artifact_hashes": ()}
    )
    with pytest.raises(LateArtifact):
        await controller.collect("j1")
    with pytest.raises(Superseded):
        await controller.dispatch("j1")


# --- restarted (reconcile) ---------------------------------------------------


async def test_reconcile_leaves_running_execution_alone() -> None:
    controller, _, executor = _fresh(lens=frozenset({"py"}))
    await controller.enqueue(_record(lens=frozenset({"py"})))
    await _started(controller)
    executor.inspect_script.append(ExecutionSnapshot(running=True, terminal=False))
    after = await controller.reconcile_restart("j1")
    assert after.state is S.RUNNING


async def test_reconcile_fails_closed_when_execution_vanished() -> None:
    controller, _, executor = _fresh(lens=frozenset({"py"}))
    await controller.enqueue(_record(lens=frozenset({"py"})))
    await _started(controller)
    executor.inspect_script.append(ExecutionSnapshot(running=False, terminal=False))
    after = await controller.reconcile_restart("j1")
    assert after.state is S.CANCELLED


async def test_reconcile_fails_closed_when_executor_forgot_execution() -> None:
    """A conformant executor raises ``UnknownExecutionError`` for a lost execution
    (it never returns a non-running/non-terminal snapshot); the controller must
    fail closed to cancelled instead of letting the exception wedge the job in
    its active state across the restart.
    """
    controller, _, executor = _fresh(lens=frozenset({"py"}))
    await controller.enqueue(_record(lens=frozenset({"py"})))
    await _started(controller)
    executor.inspect_raises = UnknownExecutionError("execution forgotten across restart")
    after = await controller.reconcile_restart("j1")
    assert after.state is S.CANCELLED
    # The deterministic release ran on the lost ref; nothing is left wedged.
    assert executor.released and executor.released[0][1] == "cancelled"


async def test_reconcile_never_inspects_unstarted_job() -> None:
    controller, _, executor = _fresh()
    await controller.enqueue(_record())
    after = await controller.reconcile_restart("j1")
    assert after.state is S.QUEUED
    assert executor.inspect_script == []  # no handle to parse; nothing inspected


# --- infra retry -------------------------------------------------------------


async def test_infra_start_failure_fails_closed_then_retries_by_budget() -> None:
    storage = InMemoryStorage()
    executor = FakeScriptedExecutor(start_raises=RuntimeError("provider 503"))
    admission = AdmissionController(Budgets(retries={"svc": 1}))
    controller = ServiceController(storage, executor, admission=admission)
    await controller.enqueue(_record())
    result = await controller.dispatch("j1")
    assert isinstance(result, ControllerRecord)
    assert result.state is S.INFRA_ERROR  # failed closed, not clean

    retried = await controller.retry_infra("j1")
    assert retried != ROUTE_TO_OPERATOR
    assert isinstance(retried, ControllerRecord)
    assert retried.state is S.QUEUED
    assert retried.spec.attempt_number == 2

    # Second infra: dispatch fails again; budget now exhausted -> route to operator.
    await controller.dispatch("j1")
    outcome = await controller.retry_infra("j1")
    assert outcome == ROUTE_TO_OPERATOR


async def test_findings_verdict_fails_closed_and_is_not_retried() -> None:
    """A findings verdict fails closed; infra retry is never offered for it."""
    controller, _, executor = _fresh(lens=frozenset({"py"}), retries=5)
    await controller.enqueue(_record(lens=frozenset({"py"})))
    await _started(controller)
    executor.envelopes.append(
        {"worker_verdict": "findings", "completed_lenses": ["py"], "blocked": True}
    )
    await controller.collect("j1")
    evaluated = await controller.evaluate("j1")
    assert evaluated.state is S.FAILED
    # Findings never consumed an infra retry.
    assert controller._admission.in_flight().fleet == 0


# --- incomplete coverage fails closed ----------------------------------------


async def test_missing_lens_fails_closed() -> None:
    controller, _, executor = _fresh(lens=frozenset({"py", "go"}))
    await controller.enqueue(_record(lens=frozenset({"py", "go"})))
    await _started(controller)
    executor.envelopes.append(
        {"worker_verdict": VERDICT_CLEAN, "completed_lenses": ["py"], "artifact_hashes": ()}
    )
    await controller.collect("j1")
    evaluated = await controller.evaluate("j1")
    assert evaluated.state is S.FAILED


async def test_blocking_finding_fails_closed_despite_clean_exit() -> None:
    controller, _, executor = _fresh(lens=frozenset({"py"}))
    await controller.enqueue(_record(lens=frozenset({"py"})))
    await _started(controller)
    executor.envelopes.append(
        {"worker_verdict": VERDICT_CLEAN, "completed_lenses": ["py"], "blocked": True}
    )
    await controller.collect("j1")
    evaluated = await controller.evaluate("j1")
    assert evaluated.state is S.FAILED


# --- admission backoff -------------------------------------------------------


async def test_admission_backoff_does_not_start() -> None:
    storage = InMemoryStorage()
    executor = FakeScriptedExecutor()
    admission = AdmissionController(Budgets(fleet=0))
    controller = ServiceController(storage, executor, admission=admission)
    await controller.enqueue(_record())
    result = await controller.dispatch("j1")
    assert isinstance(result, AdmissionBackoff)
    assert "fleet" in result.reason
    assert len(executor.started) == 0
    assert storage.load_job("j1").state is S.QUEUED
