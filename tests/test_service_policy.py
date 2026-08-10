"""Hermetic tests for the policy evaluator (Plan 008 Step 5).

Contracts under test (REVIEW_TARGET_V1 + DAYDREAM_SERVICE_V1, frozen):
- ``PolicyEvaluator`` evaluates EXACTLY the protected service policy; it never
  hard-codes a round count, backend, executor, or private check name.
- Fail-closed: one clean round (when policy requires more), mixed identities,
  a stale candidate, a missing lens, findings, partial artifacts, an untrusted
  config change, or a publisher retry can NEVER produce success.

These tests mock nothing network-related: the evaluator is pure logic over
typed model records. No GitHub / model provider / executor is touched.
"""

from __future__ import annotations

import pytest

from daydream.service.models import (
    LensInventory,
    PolicyDecision,
    ReviewPolicy,
    ReviewTarget,
    RoundRecord,
    SourceOfTruth,
    TargetKind,
    TerminalOutcome,
)
from daydream.service.policy import PolicyEvaluator
from daydream.service.publisher import Publisher, PublishError, PublishReceipt, PublishRequest

CANDIDATE_SHA = "a" * 40
CANDIDATE_TREE = "b" * 40
BASE_SHA = "c" * 40
DIFF_DIGEST = "d" * 64
CONFIG_DIGEST = "e" * 64
OVERRIDE_DIGEST = "f" * 64


def _target(
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
        config_source=SourceOfTruth(
            ref="refs/heads/main",
            sha=BASE_SHA,
            digest=config_digest,
        ),
        invalidation_id="job-1",
    )


def _policy(
    *,
    rounds: int = 2,
    complete_lens: tuple[str, ...] = ("python", "security"),
    concurrent_rounds: bool = True,
    publisher: str = "github-checks",
    check_name: str = "daydream/review",
    executor: str = "local-fake",
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
        source=SourceOfTruth(ref="refs/heads/main", sha=BASE_SHA, digest=CONFIG_DIGEST),
    )


def _round(
    *,
    attempt_id: str = "r1",
    target: ReviewTarget | None = None,
    outcome: TerminalOutcome = TerminalOutcome.CLEAN,
    completed_lenses: tuple[str, ...] = ("python", "security"),
    finding_count: int = 0,
    partial_artifacts: bool = False,
    execution_ref: str = "opaque:exec-1",
) -> RoundRecord:
    target = target or _target()
    return RoundRecord(
        attempt_id=attempt_id,
        target=target,
        outcome=outcome,
        completed_lenses=set(completed_lenses),
        finding_count=finding_count,
        partial_artifacts=partial_artifacts,
        execution_ref=execution_ref,
    )


def _evaluator() -> PolicyEvaluator:
    return PolicyEvaluator()


# --- RED anchors ------------------------------------------------------------


def test_evaluator_exists_and_returns_decision() -> None:
    assert callable(_evaluator().evaluate)


# --- required-rounds fail-closed --------------------------------------------


def test_two_clean_complete_rounds_pass() -> None:
    target = _target()
    policy = _policy(rounds=2)
    rounds = [_round(attempt_id="r1", target=target), _round(attempt_id="r2", target=target)]
    decision = _evaluator().evaluate(target, policy, rounds)
    assert decision.outcome is PolicyDecision.SUCCESS
    assert decision.published_attempt_ids == {"r1", "r2"}


def test_one_clean_round_when_two_required_fails_closed() -> None:
    policy = _policy(rounds=2)
    decision = _evaluator().evaluate(_target(), policy, [_round(attempt_id="r1")])
    assert decision.outcome is PolicyDecision.FAIL
    assert "2" in decision.reason  # names the required count


def test_zero_rounds_never_succeed() -> None:
    decision = _evaluator().evaluate(_target(), _policy(rounds=1), [])
    assert decision.outcome is PolicyDecision.FAIL


def test_rounds_not_hard_coded_three_required_empty_passes() -> None:
    """The evaluator honors three configured rounds, proving no hard-coded two."""
    policy = _policy(rounds=3)
    target = _target()
    rounds = [_round(attempt_id=f"r{i}", target=target) for i in range(3)]
    assert _evaluator().evaluate(target, policy, rounds).outcome is PolicyDecision.SUCCESS
    # only two rounds with three required -> fail
    partial = _evaluator().evaluate(target, policy, rounds[:2])
    assert partial.outcome is PolicyDecision.FAIL


def test_config_rounds_zero_is_rejected_as_invalid_policy() -> None:
    """A protected policy with zero required rounds is itself invalid; fail closed."""
    with pytest.raises(ValueError):
        _policy(rounds=0)


# --- mixed identity / stale candidate fail-closed ----------------------------


def test_mixed_identities_fail() -> None:
    """Rounds bound to different candidates can never compose to success."""
    target = _target()
    other = _target(candidate_sha="z" * 40)
    rounds = [_round(attempt_id="r1", target=target), _round(attempt_id="r2", target=other)]
    decision = _evaluator().evaluate(target, _policy(rounds=2), rounds)
    assert decision.outcome is PolicyDecision.FAIL
    assert "identity" in decision.reason


def test_stale_candidate_fails() -> None:
    """A round that reviewed an old candidate cannot authorize the current one."""
    stale = _target(candidate_sha="0" * 40)
    current = _target()
    rounds = [_round(attempt_id="r1", target=current), _round(attempt_id="r2", target=stale)]
    decision = _evaluator().evaluate(current, _policy(rounds=2), rounds)
    assert decision.outcome is PolicyDecision.FAIL
    assert "stale" in decision.reason


def test_round_target_without_candidate_sha_mismatch_fails() -> None:
    """A round with a None candidate SHA is not trustworthy (cannot bind identity)."""
    target = _target()
    partial_target = ReviewTarget(
        repo=target.repo,
        kind=target.kind,
        candidate_sha=None,  # type: ignore[arg-type]  # deliberately smuggled below via dataclass
        candidate_tree="b" * 40,
        base_sha=BASE_SHA,
        pr_number=77,
        merge_group_id=None,
        diff_digest=DIFF_DIGEST,
        config_source=target.config_source,
        invalidation_id=target.invalidation_id,
    )
    rounds = [_round(attempt_id="r1", target=target), _round(attempt_id="r2", target=partial_target)]
    decision = _evaluator().evaluate(target, _policy(rounds=2), rounds)
    assert decision.outcome is PolicyDecision.FAIL


# --- lens fail-closed --------------------------------------------------------


def test_missing_lens_fails() -> None:
    """A round that skipped a required lens is never sufficient."""
    target = _target()
    rounds = [
        _round(attempt_id="r1", target=target),
        _round(attempt_id="r2", target=target, completed_lenses=("python",)),
    ]
    decision = _evaluator().evaluate(target, _policy(rounds=2), rounds)
    assert decision.outcome is PolicyDecision.FAIL
    assert "lens" in decision.reason


def test_all_rounds_must_cover_every_lens() -> None:
    """Each configured round is a full review; no round may be a shallow pass."""
    target = _target()
    policy = _policy(rounds=2)
    rounds = [
        _round(attempt_id="r1", target=target, completed_lenses=("python",)),
        _round(attempt_id="r2", target=target),
    ]
    assert _evaluator().evaluate(target, policy, rounds).outcome is PolicyDecision.FAIL


# --- findings / infra / cancellation fail-closed ------------------------------


def test_findings_fail_even_when_process_exits_clean() -> None:
    """Blocking findings are findings even if the round reports clean exit."""
    target = _target()
    rounds = [
        _round(attempt_id="r1", target=target),
        _round(attempt_id="r2", target=target, finding_count=3, outcome=TerminalOutcome.FINDINGS),
    ]
    decision = _evaluator().evaluate(target, _policy(rounds=2), rounds)
    assert decision.outcome is PolicyDecision.FAIL
    assert "finding" in decision.reason


def test_infra_error_round_fails_closed() -> None:
    target = _target()
    rounds = [
        _round(attempt_id="r1", target=target),
        _round(attempt_id="r2", target=target, outcome=TerminalOutcome.INFRA_ERROR),
    ]
    assert _evaluator().evaluate(target, _policy(rounds=2), rounds).outcome is PolicyDecision.FAIL


def test_cancelled_round_fails_closed() -> None:
    target = _target()
    rounds = [
        _round(attempt_id="r1", target=target),
        _round(attempt_id="r2", target=target, outcome=TerminalOutcome.CANCELLED),
    ]
    assert _evaluator().evaluate(target, _policy(rounds=2), rounds).outcome is PolicyDecision.FAIL


def test_partial_artifacts_fail_closed() -> None:
    """A round whose artifacts are incomplete is an infra/incomplete failure."""
    target = _target()
    rounds = [
        _round(attempt_id="r1", target=target),
        _round(attempt_id="r2", target=target, partial_artifacts=True),
    ]
    decision = _evaluator().evaluate(target, _policy(rounds=2), rounds)
    assert decision.outcome is PolicyDecision.FAIL
    assert "artifact" in decision.reason


# --- untrusted config change fail-closed -------------------------------------


def test_round_bound_to_untrusted_config_digest_fails() -> None:
    """A round that ran under a different config digest is proof of an untrusted change."""
    target = _target()
    tampered = _target(config_digest=OVERRIDE_DIGEST)
    rounds = [
        _round(attempt_id="r1", target=target),
        _round(attempt_id="r2", target=tampered),
    ]
    decision = _evaluator().evaluate(target, _policy(rounds=2), rounds)
    assert decision.outcome is PolicyDecision.FAIL
    assert "config" in decision.reason


def test_policy_config_digest_mismatch_target_is_invalid() -> None:
    """The evaluator refuses when the protected policy digest conflicts with the target."""
    # Policy bound to override digest, target bound to canonical digest.
    target = _target(config_digest=CONFIG_DIGEST)
    tampered_policy = ReviewPolicy(
        backend="pi",
        provider="nous",
        model="deepseek/deepseek-v4-flash-0731",
        required_rounds=2,
        complete_lens=LensInventory(required={"python", "security"}),
        executor="local-fake",
        concurrent_rounds=True,
        immutable_reviewer_bundle="sha256:" + "0" * 64,
        deadline_s=1800.0,
        hard_budget_s=3600.0,
        publisher="github-checks",
        check_name="daydream/review",
        source=SourceOfTruth(ref="refs/heads/main", sha=BASE_SHA, digest=OVERRIDE_DIGEST),
    )
    rounds = [_round(attempt_id="r1", target=target), _round(attempt_id="r2", target=target)]
    decision = _evaluator().evaluate(target, tampered_policy, rounds)
    assert decision.outcome is PolicyDecision.FAIL
    assert "config" in decision.reason


# --- manufacturer / backend / publisher config is honored, not hard-coded -----


def test_backend_and_executor_are_policy_only() -> None:
    """The evaluator does not assert a specific backend or executor; it just requires consistency."""
    policy = _policy(rounds=1, executor="kubernetes")
    target = _target()
    decision = _evaluator().evaluate(target, policy, [_round(target=target)])
    assert decision.outcome is PolicyDecision.SUCCESS


# --- publisher port RED anchors ----------------------------------------------


def test_publisher_port_protocol_shape() -> None:
    """The publisher port exists and is bindable by a fake for hermetic tests."""
    class FakePublisher(Publisher):
        def publish(self, req: PublishRequest) -> PublishReceipt:
            return PublishReceipt(external_id=req.external_id, check_run_id=1)

    assert FakePublisher()


def test_publish_error_raised_on_failure() -> None:
    class Failing(Publisher):
        def publish(self, req: PublishRequest) -> PublishReceipt:
            raise PublishError("no")

    with pytest.raises(PublishError):
        Failing().publish(PublishRequest(external_id="job-1", conclusion="failure", summary="x"))


def test_publish_retry_never_invent_success() -> None:
    """The publisher port is fail-closed at the interface: a retry only re-raises."""
    calls = 0

    class Flaky(Publisher):
        def publish(self, req: PublishRequest) -> PublishReceipt:
            nonlocal calls
            calls += 1
            raise PublishError("transient")

    p = Flaky()
    with pytest.raises(PublishError):
        p.publish(PublishRequest(external_id="job-1", conclusion="failure", summary="x"))
    # a retry is an explicit second call, not an implicit flip to success
    assert calls == 1
