"""Verify CI against the exact successfully pushed repository, ref, and SHA."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
from rich.markup import escape as escape_markup

from daydream import git_ops
from daydream.agent import console
from daydream.deep.artifacts import DeepArtifact
from daydream.deep.state import DeepState
from daydream.extensions.api import Stop
from daydream.flows.engine import FlowContext
from daydream.git_ops import GitError
from daydream.phases import PushReceipt
from daydream.trajectory import DaydreamPhase, host_phase_scope, now_iso
from daydream.ui import print_error, print_info, print_success, print_warning

if TYPE_CHECKING:
    from daydream.remote_ci import RemoteCITarget, RemoteCIVerdict


def _resolve_remote_ci_target(ctx: FlowContext, receipt: PushReceipt) -> RemoteCITarget:
    """Bind the pushed repository/ref to one configured P04 pull request."""
    from daydream import pr_review
    from daydream.remote_ci import RemoteCITarget

    configured_repo = ctx.config.pr_repo
    configured_pr = ctx.config.pr_number
    if git_ops.split_owner_repo(configured_repo or "") is None:
        raise GitError("remote CI requires a configured GitHub base repository")
    assert configured_repo is not None
    if not isinstance(configured_pr, int) or isinstance(configured_pr, bool) or configured_pr <= 0:
        raise GitError("remote CI requires a configured pull request number")
    if receipt.pushed_repository is None:
        raise GitError("the successful push remote has no GitHub repository identity")

    pr = pr_review.find_pr_by_number(
        ctx.work.repo, configured_pr, auth=ctx.github_execution.auth
    )
    if pr is None:
        raise GitError(f"configured pull request #{configured_pr} was not found")
    base_repository = f"{pr.owner}/{pr.repo}"
    expected = {
        "pull request": (configured_pr, pr.number),
        "configured base repository": (configured_repo.lower(), base_repository.lower()),
        "base ref": (ctx.work.base_branch, pr.base_ref),
        "head repository": (receipt.pushed_repository.lower(), (pr.head_repo or "").lower()),
        "head ref": (receipt.branch, pr.head_ref),
    }
    mismatches = [name for name, (wanted, actual) in expected.items() if wanted != actual]
    if mismatches:
        raise GitError(
            "remote CI identity does not match the pushed target: "
            + ", ".join(mismatches)
        )
    if pr.head_repo is None:
        raise GitError("pull request head repository is unavailable")
    return RemoteCITarget(
        target_dir=ctx.work.repo.resolve(),
        base_repository=base_repository,
        base_ref=pr.base_ref,
        head_repository=pr.head_repo,
        head_ref=pr.head_ref,
        pr_number=pr.number,
        pr_url=pr.url,
        remote=receipt.remote,
        pushed_sha=receipt.sha,
    )


def _print_remote_ci_result(verdict: RemoteCIVerdict) -> None:
    """Render only normalized GitHub evidence and its explicit limitations."""
    target = verdict.target
    if target is not None:
        print_info(
            console,
            escape_markup(
                f"Remote CI target: {target.base_repository} PR #{target.pr_number} "
                f"at {target.pushed_sha}"
            ),
        )
    print_info(
        console,
        escape_markup(f"Remote CI result: {verdict.status} — {verdict.reason}"),
    )
    if verdict.evidence_sha is not None:
        print_info(
            console,
            escape_markup(f"Remote CI evidence SHA: {verdict.evidence_sha}"),
        )
    if verdict.failing_contexts:
        print_warning(console, f"Failing CI: {', '.join(verdict.failing_contexts)}")
    if verdict.pending_contexts:
        print_warning(console, f"Pending CI: {', '.join(verdict.pending_contexts)}")
    if verdict.missing_contexts:
        print_warning(console, f"Missing CI: {', '.join(verdict.missing_contexts)}")
    advisory = [
        item.context
        for item in verdict.advisory_observations
        if item.state in {"fail", "pending"}
    ]
    if advisory:
        print_warning(console, f"Advisory CI not green: {', '.join(advisory)}")
    for url in verdict.urls:
        print_info(console, escape_markup(f"CI details: {url}"))


async def _step_remote_ci(ctx: FlowContext) -> Stop | None:
    """Bind and wait for exact pushed-SHA GitHub CI, failing closed."""
    deep_state = DeepState(ctx.data)
    from daydream.remote_ci import (
        DEFAULT_LIMITS,
        GitHubRemoteCIFetcher,
        pending_remote_ci_verdict,
        unavailable_remote_ci_verdict,
        wait_for_remote_ci,
        write_remote_ci_handoff,
        write_remote_ci_verdict,
    )

    receipt = deep_state.push_receipt
    if not isinstance(receipt, PushReceipt):
        return None
    state = deep_state.fix_cycle_state
    limits = DEFAULT_LIMITS
    started_at = now_iso()
    monotonic_started = anyio.current_time()
    discovery_deadline = monotonic_started + limits.discovery_seconds
    completion_deadline = monotonic_started + limits.completion_seconds
    poll_count = 0
    verdict: RemoteCIVerdict | None = None

    def write_snapshot(snapshot: RemoteCIVerdict) -> None:
        write_remote_ci_verdict(
            DeepArtifact.REMOTE_CI_VERDICT.at(deep_state.dd),
            snapshot,
            session_id=state.session_id,
            poll_count=poll_count,
            started_at=started_at,
            updated_at=now_iso(),
            discovery_deadline=discovery_deadline,
            completion_deadline=completion_deadline,
            limits=limits,
        )

    def persist(snapshot: RemoteCIVerdict) -> None:
        nonlocal poll_count, verdict
        poll_count += 1
        verdict = snapshot
        write_snapshot(snapshot)

    caught: BaseException | None = None
    cancelled_type = anyio.get_cancelled_exc_class()
    try:
        async with host_phase_scope(DaydreamPhase.REMOTE_CI) as phase:
            try:
                target = _resolve_remote_ci_target(ctx, receipt)
            except Exception as exc:
                verdict = unavailable_remote_ci_verdict(
                    reason="remote CI target identity is unavailable",
                    diagnostic=str(exc),
                )
                write_snapshot(verdict)
            else:
                # One monotonic start owns both the durable deadline metadata
                # and the waiter's request budgets.  Resolution above is a
                # separate bounded P04 lookup and does not consume CI polling
                # time; persisting the initial state below does.
                started_at = now_iso()
                monotonic_started = anyio.current_time()
                discovery_deadline = monotonic_started + limits.discovery_seconds
                completion_deadline = monotonic_started + limits.completion_seconds
                write_snapshot(pending_remote_ci_verdict(target))
                # The new target is durable before the previous attempt's
                # guidance is retired. Do this before any CI request so a
                # blocked or abruptly interrupted resume cannot expose it.
                DeepArtifact.REMOTE_CI_HANDOFF.at(deep_state.dd).unlink(missing_ok=True)
                print_info(
                    console,
                    f"Verifying remote CI for {target.base_repository} PR "
                    f"#{target.pr_number} at {target.pushed_sha}",
                )
                try:
                    verdict = await wait_for_remote_ci(
                        target,
                        fetcher=GitHubRemoteCIFetcher(
                            limits=limits, auth=ctx.github_execution.auth
                        ),
                        limits=limits,
                        monotonic_started_at=monotonic_started,
                        on_snapshot=persist,
                    )
                except (cancelled_type, KeyboardInterrupt) as exc:
                    phase.stop_reason = (
                        "cancelled" if isinstance(exc, cancelled_type) else "interrupted"
                    )
                    caught = exc
                else:
                    phase.stop_reason = verdict.status
            if caught is None and verdict is not None:
                phase.stop_reason = verdict.status
    except Exception as exc:
        print_error(console, "Remote CI verification failed", str(exc))
        return Stop(1)
    if caught is not None:
        if verdict is not None:
            try:
                with anyio.CancelScope(shield=True):
                    write_remote_ci_handoff(
                        DeepArtifact.REMOTE_CI_HANDOFF.at(deep_state.dd),
                        verdict,
                        session_id=state.session_id,
                    )
            except Exception as exc:
                print_error(console, "Remote CI handoff persistence failed", str(exc))
        raise caught
    if verdict is None:
        print_error(console, "Remote CI verification failed", "no verdict was produced")
        return Stop(1)

    _print_remote_ci_result(verdict)
    handoff = DeepArtifact.REMOTE_CI_HANDOFF.at(deep_state.dd)
    if verdict.status in {"passed", "no_ci"}:
        try:
            handoff.unlink(missing_ok=True)
        except OSError as exc:
            print_error(console, "Remote CI handoff cleanup failed", str(exc))
            return Stop(1)
        if verdict.status == "passed":
            print_success(console, "Exact pushed-SHA remote CI passed.")
        else:
            print_success(console, "Remote CI was observably not configured.")
        return None
    try:
        write_remote_ci_handoff(handoff, verdict, session_id=state.session_id)
    except Exception as exc:
        print_error(console, "Remote CI handoff persistence failed", str(exc))
    return Stop(1)
