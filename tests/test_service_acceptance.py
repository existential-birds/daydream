"""End-to-end acceptance tests for the integrated review service (Plan 008).

These tests drive the REAL merged pieces together — the durable controller
(leaf-B) through the canonical executor bridge (leaf-D ``ScriptedExecutor``)
feeding the fail-closed ``PolicyEvaluator`` (leaf-E) and a recording publisher
(mocking the GitHub seam). They confirm the tree satisfies the exact-candidate,
complete-lens, mutation-free authorization semantics required by the frozen
contracts:

1. enqueue head A, force-push B while A's configured rounds run; a B round with
   one lens failing can never authorize B;
2. retry B clean — only a COMPLETE configured set for the current candidate may
   publish success;
3. merge-queue M1 -> M2 replacement can never authorize M2 using M1's rounds.

All hermetic: the only external seam (the publisher / ``gh_api``) is a recording
fake inside the test; no network, provider, or executor-process is touched.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from daydream.executors import (
    ExecutionRef as CanonicalExecutionRef,
)
from daydream.executors import ScriptedExecutor, UnknownExecutionError
from daydream.service.controller import ServiceController
from daydream.service.executor_bridge import ExecutionBridge
from daydream.service.models import (
    CandidateTarget,
    ControllerRecord,
    JobSpec,
    LensInventory,
    ReviewPolicy,
    ReviewTarget,
    TargetKind,
)
from daydream.service.policy import PolicyDecision, PolicyEvaluator
from daydream.service.publisher import Publisher, PublishError, PublishReceipt, PublishRequest
from daydream.service.runner import ReviewRunner
from daydream.service.states import ServiceState
from daydream.service.store_sqlite import SqliteServiceStore
from tests.harness.service_fakes import (
    BASE_SHA,
    CANDIDATE_SHA,
    CANDIDATE_TREE,
    CONFIG_SOURCE,
    FakeScriptedExecutor,
    InMemoryStorage,
    ServiceStoreStorageAdapter,
    make_round,
    make_target,
)


def _policy(rounds: int = 2, lenses: tuple[str, ...] = ("py", "sec")) -> ReviewPolicy:
    return ReviewPolicy(
        backend="pi",
        provider="nous",
        model="deepseek/deepseek-v4-flash-0731",
        required_rounds=rounds,
        complete_lens=LensInventory(required=set(lenses)),
        executor="local-fake",
        concurrent_rounds=False,
        immutable_reviewer_bundle="sha256:" + "0" * 64,
        deadline_s=600,
        hard_budget_s=600,
        publisher="github-checks",
        check_name="daydream/review",
        source=CONFIG_SOURCE,
    )


def _job(sha: str = CANDIDATE_SHA, lenses: frozenset[str] = frozenset({"py", "sec"}), attempt: int = 1) -> JobSpec:
    return JobSpec(
        job_id=f"job-{sha[-4:]}-{attempt}",
        candidate=CandidateTarget(
            target_kind="pr_head",
            repo="acme/widgets",
            candidate_sha=sha,
            tree_digest=CANDIDATE_TREE,
            base_sha=BASE_SHA,
            invalidation_id=1,
        ),
        service="svc",
        backend="pi",
        provider="nous",
        model="deepseek/deepseek-v4-flash-0731",
        required_lenses=lenses,
        attempt_number=attempt,
    )


class RecordingPublisher(Publisher):
    """Recording fake publisher: the mocked GitHub/publisher seam."""

    def __init__(self) -> None:
        self.published: list[PublishRequest] = []
        self.stale_sha: str | None = None  # when set, emulate a replaced live identity

    def publish(self, req: PublishRequest) -> PublishReceipt:
        if self.stale_sha is not None and req.conclusion == "success":
            raise PublishError(
                f"live candidate {self.stale_sha!r} no longer matches final SHA {req.target_sha!r}; "
                "refusing to publish success for a replaced candidate"
            )
        self.published.append(req)
        return PublishReceipt(external_id=req.external_id, check_run_id=1)


def _runner(
    rounds: int = 1,
    *,
    storage: InMemoryStorage | ServiceStoreStorageAdapter | None = None,
) -> tuple[ReviewRunner, RecordingPublisher, InMemoryStorage | ServiceStoreStorageAdapter]:
    publisher = RecordingPublisher()
    storage = storage or InMemoryStorage()
    bridge = ExecutionBridge(ScriptedExecutor())
    controller = ServiceController(storage, bridge, admission=None)  # type: ignore[arg-type]
    runner = ReviewRunner(
        controller=controller,
        policy=_policy(rounds=rounds),
        evaluator=PolicyEvaluator(),
        publisher=publisher,
    )
    return runner, publisher, storage


async def _run_clean_round(
    runner: ReviewRunner, *, attempt: int, target: ReviewTarget | None = None
) -> ControllerRecord:
    """Drive one round through the real controller + bridge + scripted executor.

    The bridge maps the spec's lens set into the executor's ``payload.lenses``,
    so the scripted adapter materializes a CLEAN full-lens envelope. The
    controller transitions to EVALUATED with a clean verdict.
    """
    spec = _job(attempt=attempt)
    result = await runner.run_one_round(job_spec=spec, target=target or make_target())
    assert isinstance(result, ControllerRecord)
    assert result.worker_verdict == "clean"
    return result


# ---------------------------------------------------------------------------
# Acceptance 1 & 2: enqueue head A, force-push B; only the COMPLETE configured
# round set for the current candidate may publish success.
# ---------------------------------------------------------------------------


async def test_incomplete_round_set_never_publishes() -> None:
    """One uploaded clean round when two are configured -> fail closed, no publish."""
    runner, publisher, _ = _runner(rounds=2)
    await _run_clean_round(runner, attempt=1, target=make_target())

    decision = await runner.publish(make_target(), external_id="job-b")
    assert decision.outcome is PolicyDecision.FAIL
    assert publisher.published == []  # never reached the publisher


async def test_only_complete_configured_round_set_for_b_publishes() -> None:
    """Retry B clean: success is published exactly once, only once the complete
    configured round set (both rounds) is bound to B's exact candidate."""
    runner, publisher, storage = _runner(rounds=2)
    assert isinstance(storage, InMemoryStorage)  # only the default in-memory storage has load_job
    target_b = make_target(candidate_sha=CANDIDATE_SHA)

    await _run_clean_round(runner, attempt=1, target=target_b)
    partial = await runner.publish(target_b, external_id="job-b")
    assert partial.outcome is PolicyDecision.FAIL
    assert len(publisher.published) == 0

    await _run_clean_round(runner, attempt=2, target=target_b)
    full = await runner.publish(target_b, external_id="job-b")
    assert full.outcome is PolicyDecision.SUCCESS
    assert len(publisher.published) == 1
    assert publisher.published[0].target_sha == CANDIDATE_SHA
    assert publisher.published[0].conclusion == "success"
    # The job lifecycle closes with the publish: both round jobs reach PASSED
    # (and their executions are released) — never parked in PUBLISHING.
    assert storage.load_job(_job(attempt=1).job_id).state is ServiceState.PASSED
    assert storage.load_job(_job(attempt=2).job_id).state is ServiceState.PASSED


async def test_b_round_with_missing_lens_cannot_authorize_b() -> None:
    """A B round that completes only one of its two lenses cannot authorize B."""
    runner, publisher, storage = _runner(rounds=2)
    target_b = make_target(candidate_sha=CANDIDATE_SHA)

    await _run_clean_round(runner, attempt=1, target=target_b)
    # Second round completes only "py" — "sec" is missing (partial-artifact /
    # missing-lens is a fail-closed condition even when the process exited 0).
    partial_lens = make_round(
        attempt_id="round-b-missing",
        target=target_b,
        completed_lenses=("py",),
        execution_ref="ref-b",
    )
    decision = runner.evaluator.evaluate(target_b, runner.policy, runner.rounds() + [partial_lens])

    assert decision.outcome is PolicyDecision.FAIL
    assert publisher.published == []


# ---------------------------------------------------------------------------
# Acceptance 3: merge-queue M1 -> M2 replacement can never authorize M2 with M1
# ---------------------------------------------------------------------------


def test_m1_rounds_can_never_authorize_replaced_m2() -> None:
    """Rounds bound to the old M1 merge-group candidate cannot authorize a
    success for the replaced M2 candidate at the same commit queue."""
    m1 = make_target(candidate_sha="1" * 40, kind=TargetKind.MERGE_GROUP)
    m2 = make_target(candidate_sha="2" * 40, kind=TargetKind.MERGE_GROUP)

    round_on_m1 = make_round(
        attempt_id="m1-round",
        target=m1,
        completed_lenses=("py", "sec"),
        execution_ref="ref-m1",
    )
    decision = PolicyEvaluator().evaluate(m2, _policy(rounds=1), [round_on_m1])

    assert decision.outcome is PolicyDecision.FAIL
    assert decision.reason  # fail-closed reason is always set on FAIL


async def test_stale_live_identity_never_publishes_success() -> None:
    """Force-push B after A's review: the publisher's live-identity revalidation
    refuses a success, so a replaced head can never be authorized by A's rounds."""
    runner, publisher, storage = _runner(rounds=1)
    assert isinstance(storage, InMemoryStorage)  # only the default in-memory storage has load_job
    target_a = make_target(candidate_sha=CANDIDATE_SHA)
    await _run_clean_round(runner, attempt=1, target=target_a)

    # head moved: live identity is no longer the reviewed A candidate
    publisher.stale_sha = "9" * 40
    with pytest.raises(PublishError, match="no longer matches"):
        await runner.publish(target_a, external_id="job-a")
    assert publisher.published == []
    # A refused publish must not wedge the round job in PUBLISHING: the runner
    # cancels it (releasing the execution) before propagating the error.
    assert storage.load_job(_job(attempt=1).job_id).state is ServiceState.CANCELLED


async def test_failed_round_job_reaches_released() -> None:
    """A findings round fails closed and its job is released, not parked."""
    publisher = RecordingPublisher()
    storage = InMemoryStorage()
    executor = FakeScriptedExecutor()
    executor.envelopes.append(
        {
            "worker_verdict": "findings",
            "completed_lenses": (),
            "artifact_hashes": ("h1",),
            "blocked": True,
        }
    )
    controller = ServiceController(storage, executor)
    runner = ReviewRunner(
        controller=controller,
        policy=_policy(rounds=1),
        evaluator=PolicyEvaluator(),
        publisher=publisher,
    )
    spec = _job()
    record = await runner.run_one_round(job_spec=spec, target=make_target())
    assert record.state is ServiceState.FAILED  # the round itself failed closed
    # The runner closes the lifecycle: FAILED -> RELEASED with a deterministic
    # execution release — the job is never left parked in a terminal state.
    assert storage.load_job(spec.job_id).state is ServiceState.RELEASED
    assert len(executor.released) == 1 and executor.released[0][1] == "released"

    decision = await runner.publish(make_target(), external_id="job-a")
    assert decision.outcome is PolicyDecision.FAIL
    assert publisher.published == []


async def test_bridge_release_without_binding_still_reaches_executor() -> None:
    """A fresh bridge (post-restart) must deterministically release a reconciled ref.

    The ``reconcile_restart`` scenario hands opaque refs from the durable store
    back to a brand-new bridge whose in-memory bindings are empty; release must
    still run the executor's deterministic release instead of silently leaking
    the workspace.
    """
    executor = ScriptedExecutor()
    ref = await ExecutionBridge(executor).start(_job())
    # Simulate a controller restart: the new bridge instance shares the same
    # executor but has no in-memory bindings for *ref*.
    fresh = ExecutionBridge(executor)
    await fresh.release(ref, "released")
    # The executor's deterministic release ran: the execution's resources are
    # gone (inspect no longer knows the ref). The canonical adapter inspects a
    # canonical ref — rebuilt from the controller-shaped ref exactly as a fresh
    # bridge's ``_resolve`` would.
    with pytest.raises(UnknownExecutionError):
        await executor.inspect(
            CanonicalExecutionRef(
                executor_kind=ref.executor_kind,
                adapter_version=int(ref.adapter_version),
                opaque_handle=ref.opaque_handle,
                attempt_id=ref.attempt_id,
            )
        )


async def test_real_controller_drives_real_sqlite_store(tmp_path) -> None:
    """The durable controller runs a full lifecycle over the REAL SQLite store.

    ``ControllerStorage`` and ``ServiceStore`` are deliberately separate ABIs
    (see ``daydream.service.ports`` / ``daydream.service.store``); the
    ``ServiceStoreStorageAdapter`` bridges them in tests so the real controller
    exercises the production store's CAS/claim/lease machinery end to end.
    Every transition must land in the SQLite row — never only in adapter
    memory — and a clean publish must leave nothing recoverable behind.
    """
    store = SqliteServiceStore(path=tmp_path / "service.db")
    try:
        runner, publisher, _ = _runner(rounds=2, storage=ServiceStoreStorageAdapter(store))
        target = make_target()

        await _run_clean_round(runner, attempt=1, target=target)
        job1 = _job(attempt=1).job_id
        row = store.get_job(job1)
        assert row is not None
        # run_one_round ends at evaluate, which auto-passes a clean complete
        # round (EVALUATED -> PUBLISHING); the durable row must reflect it.
        assert row.state.value == ServiceState.PUBLISHING.value
        assert len(store.attempt_history(job1)) == 1  # the claim opened the ledger

        await _run_clean_round(runner, attempt=2, target=target)
        decision = await runner.publish(target, external_id="job-a")
        assert decision.outcome is PolicyDecision.SUCCESS
        assert len(publisher.published) == 1

        for attempt in (1, 2):
            job_id = _job(attempt=attempt).job_id
            row = store.get_job(job_id)
            assert row is not None
            assert row.state.value == ServiceState.PASSED.value  # terminal, not wedged
            assert row.version >= 2  # each CAS transition bumps the durable version
        # Nothing remains to reconcile after the clean publish.
        assert store.recoverable(now=datetime.now(timezone.utc)) == []
    finally:
        store.close()
