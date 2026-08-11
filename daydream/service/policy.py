"""Fail-closed policy evaluator (Plan 008 Step 5).

The ``PolicyEvaluator`` decides whether a set of completed review rounds
authorizes publishing success for an exact candidate. It evaluates EXACTLY the
protected service policy — it never hard-codes a round count, backend, executor,
lens set, or private Check name. Every condition is fail-closed: one clean
round when the policy requires more, a mixed identity, a stale candidate, a
missing lens, findings, partial artifacts, or an untrusted config change can
NEVER produce success.

The evaluator is pure logic over typed records; it performs no network I/O and
holds no credentials. Publication is separate (see ``daydream.service.publisher``
and the trusted ``daydream.github_app`` Checks publisher).
"""

from __future__ import annotations

from dataclasses import dataclass

from daydream.service.models import (
    PolicyDecision,
    ReviewPolicy,
    ReviewTarget,
    RoundRecord,
    TerminalOutcome,
)


@dataclass(frozen=True)
class Decision:
    """A fail-closed policy decision.

    Attributes:
        outcome: ``SUCCESS`` or ``FAIL``.
        reason: Human-readable reason, always set on FAIL (empty on SUCCESS).
        published_attempt_ids: The distinct round attempts that compose success,
            empty on FAIL.
    """

    outcome: PolicyDecision
    reason: str = ""
    published_attempt_ids: frozenset[str] = frozenset()

    @property
    def success(self) -> bool:
        return self.outcome is PolicyDecision.SUCCESS


class PolicyEvaluator:
    """Evaluate completed rounds against the protected service policy."""

    def evaluate(
        self,
        target: ReviewTarget,
        policy: ReviewPolicy,
        rounds: list[RoundRecord],
    ) -> Decision:
        """Return the terminal decision for *rounds* against *policy*.

        Fail-closed order (returns the first failing check):

        1. The policy itself must be trusted: its config digest must match the
           target's protected config source. A mismatch means the policy under
           which rounds ran differs from the target's protected policy, which is
           an untrusted-config condition.
        2. Exactly ``policy.required_rounds`` distinct complete rounds are
           required. Fewer than required — including a single clean round when
           more are configured — fails closed.
        3. Every round must be bound to this exact candidate: same candidate
           SHA, tree, diff digest, and config digest. A round bound to any other
           identity is mixed/stale and never composes.
        4. Every round must be a full pass: complete lens coverage, no partial
           artifacts, CLEAN outcome, and zero findings (findings are findings
           even when the process exited zero).
        5. No round may be a shallow reuse of another (distinct attempt ids).

        Args:
            target: The exact candidate being authorized.
            policy: The protected review policy.
            rounds: Completed round records to evaluate.

        Returns:
            A Decision. Success is returned only when the complete, configured
            set of independent full rounds all bound to this exact candidate are
            clean with complete artifacts and full lens coverage.
        """
        # 1. untrusted config: evaluated policy digest must match target.
        if policy.source.digest != target.config_source.digest:
            return Decision(
                PolicyDecision.FAIL,
                "policy config digest does not match the target's protected config digest; "
                "refusing to evaluate under untrusted config",
            )

        required = policy.required_rounds

        # 2. required round count (never hard-coded).
        if len(rounds) != required:
            return Decision(
                PolicyDecision.FAIL,
                f"expected {required} independent review rounds, got {len(rounds)}; "
                "fail-closed (a single clean round is never sufficient when more are configured)",
            )

        # Duplicate/shallow reuse detection: distinct attempt ids required.
        attempt_ids = [round_.attempt_id for round_ in rounds]
        if len(set(attempt_ids)) != len(attempt_ids):
            return Decision(
                PolicyDecision.FAIL,
                "rounds reuse an attempt id; a review session cannot be reused as a "
                "second independent pass",
            )

        # 3 & 4. each round is a full, complete, clean pass on this exact candidate.
        for round_ in rounds:
            if not self._bound_to_target(round_, target):
                if round_.target.config_source.digest != target.config_source.digest:
                    return Decision(
                        PolicyDecision.FAIL,
                        f"round {round_.attempt_id} ran under a different protected config "
                        f"digest (untrusted config change); a round must bind to the target's "
                        "exact protected policy",
                    )
                return Decision(
                    PolicyDecision.FAIL,
                    f"round {round_.attempt_id} is bound to a different candidate identity "
                    "(stale or mixed); only this exact candidate can be authorized",
                )
            if not policy.complete_lens.is_subset(round_.completed_lenses):
                missing = policy.complete_lens.missing(round_.completed_lenses)
                return Decision(
                    PolicyDecision.FAIL,
                    f"round {round_.attempt_id} is missing required lens(es) "
                    f"{sorted(missing)}; a full review requires every lens",
                )
            if round_.partial_artifacts:
                return Decision(
                    PolicyDecision.FAIL,
                    f"round {round_.attempt_id} produced partial artifacts; "
                    "incomplete output is infrastructure failure, never clean",
                )
            if round_.outcome is not TerminalOutcome.CLEAN:
                return Decision(
                    PolicyDecision.FAIL,
                    f"round {round_.attempt_id} terminated {round_.outcome.value}; "
                    "a non-clean round can never authorize success",
                )
            if round_.finding_count > 0:
                return Decision(
                    PolicyDecision.FAIL,
                    f"round {round_.attempt_id} reported {round_.finding_count} blocking "
                    "finding(s); findings are findings even when the process exits zero",
                )

        return Decision(
            PolicyDecision.SUCCESS,
            published_attempt_ids=frozenset(attempt_ids),
        )

    @staticmethod
    def _bound_to_target(round_: RoundRecord, target: ReviewTarget) -> bool:
        """True when the round reviewed exactly this candidate identity."""
        round_identity = round_.target.identity
        target_identity = target.identity
        # Any unbound identity (e.g. missing candidate SHA) is untrustworthy.
        if "" in round_identity:
            return False
        return round_identity == target_identity
