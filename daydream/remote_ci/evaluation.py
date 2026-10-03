"""Pure CI policy evaluation and normalized verdict construction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

from daydream.redaction import redact_structured_text
from daydream.remote_ci.evidence import (
    CIObservation,
    PRCIBinding,
    RemoteCILimits,
    RemoteCISnapshot,
    RemoteCIStatus,
    RemoteCITarget,
    RemoteCIVerdict,
    RequiredContext,
    _is_finite_number,
    _is_positive_int,
    _required_text,
)


def _fixed_identity_matches(target: RemoteCITarget, binding: PRCIBinding) -> bool:
    return (
        binding.pr_number == target.pr_number
        and binding.pr_url == target.pr_url
        and binding.base_repository == target.base_repository
        and binding.base_ref == target.base_ref
        and binding.head_repository == target.head_repository
        and binding.head_ref == target.head_ref
    )


def required_context_label(item: RequiredContext) -> str:
    if item.app_id is None:
        return item.context
    return f"{item.context} (app {item.app_id})"


def required_context_matches(
    item: RequiredContext,
    observation: CIObservation,
) -> bool:
    if item.app_id is not None:
        return (
            observation.source == "check_run"
            and observation.context == item.context
            and observation.app_id == item.app_id
        )
    if observation.source == "check_run":
        return observation.context == item.context
    return observation.context.casefold() == item.context.casefold()


@dataclass(frozen=True)
class RequiredEvidence:
    """The policy partition and required-context outcomes for one observation set."""

    required: tuple[CIObservation, ...]
    advisory: tuple[CIObservation, ...]
    failing: tuple[str, ...]
    pending: tuple[str, ...]
    missing: tuple[str, ...]


def partition_required_observations(
    contexts: Sequence[RequiredContext], observations: Sequence[CIObservation]
) -> RequiredEvidence:
    required_observations: list[CIObservation] = []
    matched_ids: set[int] = set()
    failing: list[str] = []
    pending: list[str] = []
    missing: list[str] = []
    for required in contexts:
        matches = [
            item for item in observations if required_context_matches(required, item)
        ]
        if not matches:
            missing.append(required_context_label(required))
            continue
        required_observations.extend(matches)
        matched_ids.update(id(item) for item in matches)
        label = required_context_label(required)
        if any(item.state == "fail" for item in matches):
            failing.append(label)
        elif any(item.state == "pending" for item in matches):
            pending.append(label)
    return RequiredEvidence(
        required=tuple(dict.fromkeys(required_observations)),
        advisory=tuple(item for item in observations if id(item) not in matched_ids),
        failing=tuple(failing),
        pending=tuple(pending),
        missing=tuple(missing),
    )


def _verdict(
    snapshot: RemoteCISnapshot,
    *,
    status: RemoteCIStatus,
    reason: str,
    evidence_sha: str | None,
    required_observations: Sequence[CIObservation] = (),
    advisory_observations: Sequence[CIObservation] = (),
    failing: Sequence[str] = (),
    pending: Sequence[str] = (),
    missing: Sequence[str] = (),
    stable_polls: int,
    elapsed: float,
) -> RemoteCIVerdict:
    all_observations = (*required_observations, *advisory_observations)
    urls = tuple(sorted({item.url for item in all_observations if item.url is not None}))
    return RemoteCIVerdict(
        status=status,
        reason=reason,
        target=snapshot.target,
        binding=snapshot.binding,
        policy=snapshot.policy,
        active_workflow_count=len(snapshot.active_workflows),
        evidence_sha=evidence_sha,
        required_observations=tuple(required_observations),
        advisory_observations=tuple(advisory_observations),
        failing_contexts=tuple(failing),
        pending_contexts=tuple(pending),
        missing_contexts=tuple(missing),
        urls=urls,
        diagnostic=None,
        stable_polls=stable_polls,
        elapsed_seconds=elapsed,
    )


def evaluate_remote_ci(
    snapshot: RemoteCISnapshot,
    *,
    elapsed: float,
    stable_polls: int,
    limits: RemoteCILimits,
) -> RemoteCIVerdict:
    """Evaluate one complete snapshot under the resolved remote-CI policy."""
    if (
        not _is_finite_number(elapsed) or elapsed < 0
    ):
        raise ValueError("elapsed time must be nonnegative")
    if not isinstance(stable_polls, int) or isinstance(stable_polls, bool) or stable_polls < 1:
        raise ValueError("stable poll count must be positive")
    binding = snapshot.binding
    if binding.state != "open" or not _fixed_identity_matches(snapshot.target, binding):
        return _verdict(
            snapshot,
            status="superseded",
            reason="the fixed pull request identity changed or closed",
            evidence_sha=None,
            stable_polls=stable_polls,
            elapsed=elapsed,
        )
    if binding.head_sha != snapshot.target.pushed_sha:
        status: RemoteCIStatus = "missing" if elapsed >= limits.discovery_seconds else "pending"
        return _verdict(
            snapshot,
            status=status,
            reason="the pushed commit is not yet the pull request head",
            evidence_sha=None,
            stable_polls=stable_polls,
            elapsed=elapsed,
        )

    evidence_sha, observations = snapshot.evidence

    evidence = partition_required_observations(snapshot.policy.contexts, observations)

    def make(status: RemoteCIStatus, reason: str) -> RemoteCIVerdict:
        return _verdict(
            snapshot,
            status=status,
            reason=reason,
            evidence_sha=evidence_sha,
            required_observations=evidence.required,
            advisory_observations=evidence.advisory,
            failing=evidence.failing,
            pending=evidence.pending,
            missing=evidence.missing,
            stable_polls=stable_polls,
            elapsed=elapsed,
        )

    if evidence.failing:
        return make("failed", "a required CI producer failed")
    if evidence.pending:
        if elapsed >= limits.completion_seconds:
            return make("timed_out", "required CI remained pending")
        return make("pending", "required CI is pending")
    if evidence.missing:
        if elapsed >= limits.discovery_seconds:
            return make("missing", "required CI was not reported")
        return make("pending", "waiting for required CI")

    if snapshot.policy.contexts:
        if stable_polls < limits.stable_polls:
            return make("pending", "required CI identity is stabilizing")
        return make("passed", "all required CI passed")

    if observations:
        if any(item.state == "pending" for item in observations):
            if elapsed >= limits.completion_seconds:
                return make("timed_out", "reported CI remained pending")
            return make("pending", "reported CI is pending")
        if stable_polls < limits.stable_polls:
            return make("pending", "reported CI identity is stabilizing")
        return make("passed", "all reported CI reached a terminal state")

    if snapshot.active_workflows:
        if elapsed >= limits.discovery_seconds:
            return make("missing", "active workflows reported no CI for the pushed commit")
        return make("pending", "waiting for active workflows")
    if elapsed < limits.discovery_seconds:
        return make("pending", "discovering remote CI")
    if stable_polls < limits.stable_polls:
        return make("pending", "empty CI identity is stabilizing")
    return make("no_ci", "no remote CI is configured")



def _external_verdict(
    target: RemoteCITarget,
    *,
    status: Literal["unavailable", "superseded", "cancelled"],
    reason: str,
    detail: str | None,
    elapsed: float,
    stable_polls: int,
    prior: RemoteCIVerdict | None,
    limit: int,
) -> RemoteCIVerdict:
    diagnostic = None
    if detail:
        diagnostic = redact_structured_text(detail)[:limit]
    if prior is not None:
        return replace(
            prior,
            status=status,
            reason=reason,
            diagnostic=diagnostic,
            elapsed_seconds=elapsed,
            stable_polls=max(stable_polls, 0),
        )
    return _unobserved_verdict(
        status=status,
        reason=reason,
        target=target,
        diagnostic=diagnostic,
        stable_polls=max(stable_polls, 0),
        elapsed_seconds=elapsed,
    )


def _unobserved_verdict(
    *,
    status: RemoteCIStatus,
    reason: str,
    target: RemoteCITarget | None,
    diagnostic: str | None = None,
    stable_polls: int = 0,
    elapsed_seconds: float = 0.0,
) -> RemoteCIVerdict:
    """Construct a verdict without claiming a completed CI observation."""
    return RemoteCIVerdict(
        status=status,
        reason=reason,
        target=target,
        binding=None,
        policy=None,
        active_workflow_count=None,
        evidence_sha=None,
        required_observations=(),
        advisory_observations=(),
        failing_contexts=(),
        pending_contexts=(),
        missing_contexts=(),
        urls=(),
        diagnostic=diagnostic,
        stable_polls=stable_polls,
        elapsed_seconds=elapsed_seconds,
    )


def unavailable_remote_ci_verdict(
    *,
    reason: str,
    diagnostic: str | None = None,
    target: RemoteCITarget | None = None,
    elapsed_seconds: float = 0.0,
    limit: int = 2_000,
) -> RemoteCIVerdict:
    """Create an honest fail-closed verdict before a full PR target exists."""
    if not _is_positive_int(limit):
        raise ValueError("diagnostic limit must be positive")
    if (
        not _is_finite_number(elapsed_seconds) or elapsed_seconds < 0
    ):
        raise ValueError("elapsed time must be a finite nonnegative number")
    safe_reason = redact_structured_text(_required_text(reason, "unavailable reason"))[:limit]
    safe_diagnostic = None
    if diagnostic:
        safe_diagnostic = redact_structured_text(diagnostic)[:limit]
    return _unobserved_verdict(
        status="unavailable",
        reason=safe_reason,
        target=target,
        diagnostic=safe_diagnostic,
        elapsed_seconds=elapsed_seconds,
    )


def pending_remote_ci_verdict(
    target: RemoteCITarget,
    reason: str = "discovering remote CI",
) -> RemoteCIVerdict:
    """Create the current-target marker written before the first CI request."""
    if not isinstance(target, RemoteCITarget):
        raise ValueError("pending remote CI requires a normalized target")
    safe_reason = redact_structured_text(_required_text(reason, "pending reason"))[
        :2_000
    ]
    return _unobserved_verdict(status="pending", reason=safe_reason, target=target)


def _deadline_verdict(
    snapshot: RemoteCISnapshot,
    *,
    elapsed: float,
    stable_polls: int,
    registered: bool,
    limits: RemoteCILimits,
) -> RemoteCIVerdict:
    verdict = evaluate_remote_ci(
        snapshot, elapsed=elapsed, stable_polls=stable_polls, limits=limits
    )
    if verdict.status != "pending":
        return verdict
    if registered:
        return replace(
            verdict,
            status="timed_out",
            reason="remote CI did not stabilize before the completion deadline",
        )
    return replace(
        verdict,
        status="missing",
        reason="remote CI could not be established before the discovery deadline",
    )
