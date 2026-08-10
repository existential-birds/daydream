"""Compose the review-service pieces into an end-to-end merge-authorization flow.

Plan 008 deliberately split the durable review service across leaves; this
module is the thin integration that ties them together at the boundaries the
leaves left open:

- the durable controller state machine (``daydream.service.controller``) drives
  one job per execution;
- the canonical executor (``daydream.executors``) provides compute/workspace
  lifecycle behind an opaque ref, reached through the controller seam via
  :class:`daydream.service.executor_bridge.ExecutionBridge`;
- the protected policy + fail-closed :class:`daydream.service.policy.PolicyEvaluator`
  decides whether a *complete configured set* of rounds may authorize an exact
  candidate;
- the trusted :class:`daydream.github_app.GitHubChecksPublisher` durably records
  the decision, revalidating live identity before success.

``ReviewRunner`` is the only place that binds these together. It does not
re-implement any leaf: it feeds each round through the real ``ServiceController``,
maps the collected controller record to a ``RoundRecord``, aggregates all rounds
for a target, asks ``PolicyEvaluator``, and only publishes the exact decision.
A stale / replaced candidate or an incomplete round set can never publish.

Hermetic: the only external seam is the ``Publisher`` itself; tests substitute a
recording publisher or mock ``git_ops.gh_api`` (the GitHub checks backend is
exercised by the publisher leaf's own suite).
"""

from __future__ import annotations

from dataclasses import dataclass

from daydream.service.controller import ServiceController
from daydream.service.models import (
    VERDICT_CLEAN,
    ControllerRecord,
    JobSpec,
    ReviewPolicy,
    ReviewTarget,
    RoundRecord,
    TerminalOutcome,
)
from daydream.service.policy import Decision, PolicyEvaluator
from daydream.service.publisher import Publisher, PublishRequest


def round_outcome_from_verdict(verdict: str) -> TerminalOutcome:
    """Map a controller worker verdict to the policy terminal outcome."""
    return {
        VERDICT_CLEAN: TerminalOutcome.CLEAN,
        "findings": TerminalOutcome.FINDINGS,
        "infra_error": TerminalOutcome.INFRA_ERROR,
        "cancelled": TerminalOutcome.CANCELLED,
    }[verdict]


def controller_record_to_round(
    record: ControllerRecord,
    *,
    target: ReviewTarget,
    finding_count: int,
) -> RoundRecord:
    """Build a policy ``RoundRecord`` from a collected controller record.

    Args:
        record: The evaluated controller record (state EVALUATED or later).
        target: The :class:`daydream.service.models.ReviewTarget` this job bound.
        finding_count: Blocking-findings count (the worker artifact reports it;
            a controller record carries ``blocked`` but not the raw count).
    """
    outcome = round_outcome_from_verdict(record.worker_verdict or VERDICT_CLEAN)
    attempt_id = str(record.spec.attempt_number)
    if record.execution_ref is not None:
        attempt_id = record.execution_ref.attempt_id
    return RoundRecord(
        attempt_id=attempt_id,
        target=target,
        outcome=outcome,
        completed_lenses=set(record.completed_lenses),
        finding_count=finding_count,
        partial_artifacts=record.worker_verdict == "infra_error",
        execution_ref=getattr(record.execution_ref, "opaque_handle", "") or "",
    )


@dataclass
class ReviewRunner:
    """Verified end-to-end runner: rounds -> policy decision -> publication.

    Attributes:
        controller: The durable ``ServiceController`` (bound to a storage +
            executor port; the executor may be wrapped by ``ExecutionBridge``).
        policy: The protected policy the runner evaluates against.
        evaluator: The fail-closed ``PolicyEvaluator``.
        publisher: The trusted publisher holding checks-write authority.
    """

    controller: ServiceController
    policy: ReviewPolicy
    evaluator: PolicyEvaluator
    publisher: Publisher

    async def run_one_round(
        self,
        *,
        job_spec: JobSpec,
        target: ReviewTarget,
        finding_count: int = 0,
    ) -> ControllerRecord:
        """Run a single configured round through the real controller.

        Drives the ``ServiceController``: enqueue -> dispatch -> collect ->
        evaluate. The evaluated controller record is returned so the caller can
        aggregate it into the round set. An infra or cancelled round is returned
        as-is (the evaluator fails the set closed on it).
        """
        record = ControllerRecord(job_id=job_spec.job_id, spec=job_spec)
        await self.controller.enqueue(record)
        result = await self.controller.dispatch(job_spec.job_id)
        if not isinstance(result, ControllerRecord):
            return record
        await self.controller.collect(job_spec.job_id)
        evaluated = await self.controller.evaluate(job_spec.job_id)
        self._rounds[round(evaluated.spec.attempt_number)] = controller_record_to_round(
            evaluated, target=target, finding_count=finding_count
        )
        return evaluated

    def rounds(self) -> list[RoundRecord]:
        """The rounds collected so far, in attempt order."""
        return [self._rounds[k] for k in sorted(self._rounds)]

    def evaluate(self, target: ReviewTarget) -> Decision:
        """Ask the policy evaluator for *target* given the collected rounds."""
        return self.evaluator.evaluate(target, self.policy, self.rounds())

    def publish(self, target: ReviewTarget, *, external_id: str) -> Decision:
        """Evaluate and, on SUCCESS, durably publish the exact decision.

        Only a complete, clean, exact-candidate round set produces ``SUCCESS``,
        and only then is the trusted publisher called. A stale target (the
        publisher's live-identity revalidation fails) raises before any success
        is written; an incomplete set never reaches the publisher.
        """
        decision = self.evaluate(target)
        if decision.success and self.publisher is not None:
            req = PublishRequest(
                external_id=external_id,
                conclusion="success",
                summary=(
                    f"{self.policy.check_name} clean after "
                    f"{len(self.rounds())} independent complete round(s) "
                    f"on candidate {target.candidate_sha or ''}"
                ),
                repo=target.repo,
                target_sha=target.candidate_sha or "",
                check_name=self.policy.check_name,
                target=target,
            )
            self.publisher.publish(req)
        return decision

    def __post_init__(self) -> None:
        self._rounds: dict[int, RoundRecord] = {}
