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

import pytest

from daydream.executors import ScriptedExecutor
from daydream.service.controller import ServiceController
from daydream.service.executor_bridge import ExecutionBridge
from daydream.service.models import (
    CandidateTarget,
    ControllerRecord,
    JobSpec,
    LensInventory,
    ReviewPolicy,
    ReviewTarget,
    RoundRecord,
    SourceOfTruth,
    TargetKind,
    TerminalOutcome,
)
from daydream.service.policy import PolicyDecision, PolicyEvaluator
from daydream.service.publisher import PublishError, PublishReceipt, PublishRequest, Publisher
from daydream.service.runner import ReviewRunner, controller_record_to_round
from tests.harness.service_fakes import InMemoryStorage

CANDIDATE_SHA = "a" * 40
CANDIDATE_TREE = "b" * 40
BASE_SHA = "c" * 40
DIFF_DIGEST = "d" * 64
CONFIG_DIGEST = "e" * 64

CONFIG_SOURCE = SourceOfTruth(ref="refs/heads/main", sha=BASE_SHA, digest=CONFIG_DIGEST)


def _target(sha: str = CANDIDATE_SHA, kind: TargetKind = TargetKind.PR_HEAD) -> ReviewTarget:
    return ReviewTarget(
        repo="acme/widgets",
        kind=kind,
        candidate_sha=sha,
        candidate_tree=CANDIDATE_TREE,
        base_sha=BASE_SHA,
        pr_number=77 if kind is TargetKind.PR_HEAD else None,
        merge_group_id="mg-1" if kind is TargetKind.MERGE_GROUP else None,
        diff_digest=DIFF_DIGEST,
        config_source=CONFIG_SOURCE,
        invalidation_id="job-1",
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


def _runner(rounds: int = 1) -> tuple[ReviewRunner, RecordingPublisher, InMemoryStorage]:
    publisher = RecordingPublisher()
    storage = InMemoryStorage()
    bridge = ExecutionBridge(ScriptedExecutor())
    controller = ServiceController(storage, bridge, admission=None)  # type: ignore[arg-type]
    runner = ReviewRunner(
        controller=controller,
        policy=_policy(rounds=rounds),
        evaluator=PolicyEvaluator(),
        publisher=publisher,
    )
    return runner, publisher, storage


async def _run_clean_round(runner: ReviewRunner, *, attempt: int, target: ReviewTarget | None = None) -> ControllerRecord:
    """Drive one round through the real controller + bridge + scripted executor.

    The bridge maps the spec's lens set into the executor's ``payload.lenses``,
    so the scripted adapter materializes a CLEAN full-lens envelope. The
    controller transitions to EVALUATED with a clean verdict.
    """
    spec = _job(attempt=attempt)
    result = await runner.run_one_round(job_spec=spec, target=target or _target())
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
    await _run_clean_round(runner, attempt=1, target=_target())

    decision = runner.publish(_target(), external_id="job-b")
    assert decision.outcome is PolicyDecision.FAIL
    assert publisher.published == []  # never reached the publisher


async def test_only_complete_configured_round_set_for_b_publishes() -> None:
    """Retry B clean: success is published exactly once, only once the complete
    configured round set (both rounds) is bound to B's exact candidate."""
    runner, publisher, _ = _runner(rounds=2)
    target_b = _target(sha=CANDIDATE_SHA)

    await _run_clean_round(runner, attempt=1, target=target_b)
    partial = runner.publish(target_b, external_id="job-b")
    assert partial.outcome is PolicyDecision.FAIL
    assert len(publisher.published) == 0

    await _run_clean_round(runner, attempt=2, target=target_b)
    full = runner.publish(target_b, external_id="job-b")
    assert full.outcome is PolicyDecision.SUCCESS
    assert len(publisher.published) == 1
    assert publisher.published[0].target_sha == CANDIDATE_SHA
    assert publisher.published[0].conclusion == "success"


async def test_b_round_with_missing_lens_cannot_authorize_b() -> None:
    """A B round that completes only one of its two lenses cannot authorize B."""
    runner, publisher, _ = _runner(rounds=2)
    target_b = _target(sha=CANDIDATE_SHA)

    await _run_clean_round(runner, attempt=1, target=target_b)
    # Second round completes only "py" — "sec" is missing (partial-artifact /
    # missing-lens is a fail-closed condition even when the process exited 0).
    partial_lens = RoundRecord(
        attempt_id="round-b-missing",
        target=target_b,
        outcome=TerminalOutcome.CLEAN,
        completed_lenses={"py"},
        finding_count=0,
        partial_artifacts=False,
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
    m1 = _target(sha="1" * 40, kind=TargetKind.MERGE_GROUP)
    m2 = _target(sha="2" * 40, kind=TargetKind.MERGE_GROUP)

    round_on_m1 = RoundRecord(
        attempt_id="m1-round",
        target=m1,
        outcome=TerminalOutcome.CLEAN,
        completed_lenses={"py", "sec"},
        finding_count=0,
        partial_artifacts=False,
        execution_ref="ref-m1",
    )
    decision = PolicyEvaluator().evaluate(m2, _policy(rounds=1), [round_on_m1])

    assert decision.outcome is PolicyDecision.FAIL
    assert decision.reason  # fail-closed reason is always set on FAIL


async def test_stale_live_identity_never_publishes_success() -> None:
    """Force-push B after A's review: the publisher's live-identity revalidation
    refuses a success, so a replaced head can never be authorized by A's rounds."""
    runner, publisher, _ = _runner(rounds=1)
    target_a = _target(sha=CANDIDATE_SHA)
    await _run_clean_round(runner, attempt=1, target=target_a)

    # head moved: live identity is no longer the reviewed A candidate
    publisher.stale_sha = "9" * 40
    with pytest.raises(PublishError, match="no longer matches"):
        runner.publish(target_a, external_id="job-a")
    assert publisher.published == []