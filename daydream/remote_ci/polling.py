"""Poll CI under fixed discovery/completion deadlines and persist trusted states."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from math import isfinite
from typing import Literal

import anyio

from daydream.git_ops import (
    DeadlineExpired,
    GitError,
    GitHubRequestBudget,
)
from daydream.remote_ci.evaluation import (
    _deadline_verdict,
    _external_verdict,
    evaluate_remote_ci,
)
from daydream.remote_ci.evidence import (
    DEFAULT_LIMITS,
    RemoteCIIdentityMismatch,
    RemoteCILimits,
    RemoteCISnapshot,
    RemoteCITarget,
    RemoteCIVerdict,
    _is_finite_number,
)
from daydream.remote_ci.github import (
    RemoteCIFetcher,
)


def _snapshot_stability_key(snapshot: RemoteCISnapshot) -> tuple[object, ...]:
    workflows = tuple(
        (row["id"], row["name"], row["path"], row["state"])
        for row in snapshot.active_workflows
    )
    observations = tuple(
        (item.source, item.context, item.app_id, item.state, item.raw_state)
        for item in snapshot.evidence[1]
    )
    return (snapshot.binding, snapshot.policy, workflows, observations)


async def wait_for_remote_ci(
    target: RemoteCITarget,
    *,
    fetcher: RemoteCIFetcher,
    limits: RemoteCILimits = DEFAULT_LIMITS,
    monotonic: Callable[[], float] = anyio.current_time,
    monotonic_started_at: float | None = None,
    sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
    on_snapshot: Callable[[RemoteCIVerdict], None],
) -> RemoteCIVerdict:
    """Poll under immutable deadlines and emit only complete trusted states."""
    clock_now = monotonic()
    if not isfinite(clock_now):
        raise ValueError("remote CI monotonic clock must be finite")
    if monotonic_started_at is None:
        started = clock_now
    else:
        if (
            not _is_finite_number(monotonic_started_at) or monotonic_started_at > clock_now
        ):
            raise ValueError("remote CI monotonic start must be finite and not in the future")
        started = float(monotonic_started_at)
    discovery_deadline = started + limits.discovery_seconds
    completion_deadline = started + limits.completion_seconds
    if not isfinite(discovery_deadline) or not isfinite(completion_deadline):
        raise ValueError("remote CI absolute deadlines must be finite")
    last_snapshot: RemoteCISnapshot | None = None
    last_verdict: RemoteCIVerdict | None = None
    last_key: tuple[object, ...] | None = None
    stable_polls = 0
    registered = False
    bound_head = False
    cancelled_error = anyio.get_cancelled_exc_class()

    def publish(verdict: RemoteCIVerdict) -> RemoteCIVerdict:
        on_snapshot(verdict)
        return verdict

    def external(
        status: Literal["unavailable", "superseded", "cancelled"],
        reason: str,
        detail: str | None,
        *,
        now: float,
    ) -> RemoteCIVerdict:
        """A non-evaluated verdict stamped with the loop's invariant bookkeeping."""
        return _external_verdict(
            target,
            status=status,
            reason=reason,
            detail=detail,
            elapsed=max(0.0, now - started),
            stable_polls=stable_polls,
            prior=last_verdict,
            limit=limits.diagnostic_chars,
        )

    def deadline_verdict(now: float, first_reason: str) -> RemoteCIVerdict:
        elapsed = max(0.0, now - started)
        if last_snapshot is not None:
            return _deadline_verdict(
                last_snapshot,
                elapsed=elapsed,
                stable_polls=stable_polls,
                registered=registered,
                limits=limits,
            )
        return external("unavailable", first_reason, None, now=now)

    def interrupted(reason: str) -> None:
        verdict = external("cancelled", reason, None, now=monotonic())
        with anyio.CancelScope(shield=True):
            try:
                on_snapshot(verdict)
            except BaseException:  # noqa: BLE001 - the original cancellation or interrupt must win
                pass

    try:
        now = started
        while now < completion_deadline:
            now = monotonic()
            if not isfinite(now):
                raise ValueError("remote CI monotonic clock must remain finite")
            active_deadline = completion_deadline if registered else discovery_deadline
            if now >= active_deadline:
                return publish(deadline_verdict(now, "remote CI deadline expired before one complete poll"))

            budget = GitHubRequestBudget(
                deadline=active_deadline,
                per_request_seconds=limits.request_seconds,
                monotonic=monotonic,
            )
            try:
                snapshot = await fetcher.fetch(target, budget=budget)
            except DeadlineExpired:
                return publish(deadline_verdict(monotonic(), "remote CI deadline expired during the first poll"))
            except RemoteCIIdentityMismatch as exc:
                verdict = external(
                    "superseded",
                    "the fixed pull request identity changed",
                    str(exc),
                    now=monotonic(),
                )
                return publish(verdict)
            except GitError as exc:
                now = monotonic()
                if last_snapshot is not None and now >= active_deadline:
                    verdict = deadline_verdict(now, "remote CI deadline expired")
                else:
                    verdict = external(
                        "unavailable",
                        "GitHub remote CI evidence is unavailable",
                        str(exc),
                        now=now,
                    )
                return publish(verdict)

            now = monotonic()
            if not isfinite(now):
                raise ValueError("remote CI monotonic clock must remain finite")
            if bound_head and snapshot.binding.head_sha != target.pushed_sha:
                current = evaluate_remote_ci(
                    snapshot,
                    elapsed=max(0.0, now - started),
                    stable_polls=max(stable_polls, 1),
                    limits=limits,
                )
                verdict = replace(
                    current,
                    status="superseded",
                    reason="the pull request head changed after binding to the pushed commit",
                )
                return publish(verdict)
            if snapshot.binding.head_sha == target.pushed_sha:
                bound_head = True

            key = _snapshot_stability_key(snapshot)
            stable_polls = stable_polls + 1 if key == last_key else 1
            last_key = key
            last_snapshot = snapshot
            relevant_observations = snapshot.evidence[1]
            verdict = evaluate_remote_ci(
                snapshot,
                elapsed=max(0.0, now - started),
                stable_polls=stable_polls,
                limits=limits,
            )
            last_verdict = verdict
            if bound_head and snapshot.policy.contexts:
                registered = bool(verdict.required_observations) and not bool(
                    verdict.missing_contexts
                )
            else:
                registered = bound_head and bool(relevant_observations)
            if verdict.status != "pending":
                return publish(verdict)

            active_deadline = completion_deadline if registered else discovery_deadline
            if now >= active_deadline:
                return publish(deadline_verdict(now, "remote CI deadline expired"))
            on_snapshot(verdict)
            delay = min(limits.poll_seconds, active_deadline - now)
            if delay <= 0 or not isfinite(delay):
                continue
            await sleep(delay)
        verdict = external(
            "unavailable",
            "remote CI completion deadline elapsed without a trusted terminal state",
            None,
            now=now,
        )
        return publish(verdict)
    except cancelled_error:
        interrupted("remote CI verification was cancelled")
        raise
    except KeyboardInterrupt:
        interrupted("remote CI verification was interrupted")
        raise
