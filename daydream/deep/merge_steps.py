"""Item loading and review publication steps of the deep flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from daydream.agent import console
from daydream.artifact_visibility import review_output_path_for
from daydream.deep.artifacts import (
    DeepArtifact,
)
from daydream.deep.state import DeepState
from daydream.extensions.api import Stop
from daydream.flows.engine import FlowContext
from daydream.review_budget import (
    render_review_warnings,
    review_warnings,
)
from daydream.trajectory import (
    get_current_recorder,
)
from daydream.ui import print_error, print_warning


async def _step_load_items(ctx: FlowContext) -> Stop | None:
    """Host-side merged-items guard + render-only markdown recovery."""
    deep_state = DeepState(ctx.data)
    target_dir = ctx.work.repo
    dd = deep_state.dd

    merged_report = review_output_path_for(
        target_dir,
        session=ctx.artifacts,
        allow_standalone=ctx.allow_standalone_artifacts,
    )

    # merged-items.json is the canonical source of truth; review-output.md is render-only.
    items_file = DeepArtifact.MERGED_ITEMS.at(dd)
    if not items_file.is_file():
        print_error(
            console,
            "Missing Merged Items",
            f"Expected canonical merged items at {items_file}",
        )
        return Stop(1)

    # Best-effort recover the render-only markdown from the deep-dir copy for
    # the exit message when the canonical file is absent. Non-fatal.
    if not merged_report.exists():
        deep_copy = DeepArtifact.MERGED_REPORT.at(dd)
        if deep_copy.exists():
            merged_report.write_text(deep_copy.read_text())


    warning = render_review_warnings(review_warnings(dd))
    if warning:
        for report in (merged_report, DeepArtifact.MERGED_REPORT.at(dd)):
            if report.exists() and warning not in report.read_text():
                report.write_text(warning + "\n\n" + report.read_text())

    deep_state.merged_report = merged_report
    deep_state.items_file = items_file
    return None


async def _step_findings_out(ctx: FlowContext) -> Stop:
    """Stop at the review boundary; recorder-scoped finalization owns the export."""
    return Stop(0)


async def _step_post_review(ctx: FlowContext) -> Stop | None:
    """Offer to post in loop mode; ``--comment`` auto-posts.

    In comment mode posting is the run's deliverable, so a missing PR or a
    failed GitHub submission ends the run with exit code 1 instead of the
    warn-and-continue the default deep flow gets (#8). Report-only review mode
    never resolves a PR or enters the posting helper.
    """
    deep_state = DeepState(ctx.data)
    if ctx.config.findings_out is None:
        from daydream.deep.review_terminal import finalize_review

        if not deep_state.review_coverage.is_finalized:
            finalize_review(ctx, "completed")
    # A review without usable evidence still publishes its truthful notice, then fails the run.
    failed = (ctx.data.get("terminal_review_result") or {}).get("analysis_state") == "failed"
    if deep_state.mode == "review":
        return Stop(1) if failed else None

    from daydream.pr_review import PostStatus, post_review_to_pr_from_report, resolve_review_renderers
    from daydream.pr_run_info import LiveRunInfoSource, render_live_run_info

    recorder = get_current_recorder()
    run_info = render_live_run_info(LiveRunInfoSource(recorder, ctx.artifacts))
    if run_info.diagnostic is not None:
        print_warning(console, run_info.diagnostic)

    items_file: Path = deep_state.items_file
    pr_kwargs: dict[str, Any] = (
        {"pr_number": ctx.config.pr_number}
        if ctx.config.pr_number is not None
        else {}
    )
    warnings = review_warnings(deep_state.dd)
    if warnings:
        pr_kwargs["review_warnings"] = warnings
    from daydream.review_reporting import render_github_review_context, render_review_run_id

    run_markdown = run_info.markdown
    result = ctx.data.get("terminal_review_result")
    if deep_state.mode == "comment":
        from daydream.pr_review import PRInfo

        pinned = ctx.data.get("analyzed_pr")
        if not isinstance(pinned, PRInfo):
            print_warning(console, "Committed PR target was not captured; rerun the review before posting.")
            return Stop(1)
        pr_kwargs.update(captured_pr=pinned, snapshot_diff=ctx.data["snapshot_diff"],
                         publication_dir=ctx.work.source)
    pr_kwargs["review_complete"] = isinstance(result, dict) and result.get("analysis_state") == "complete"
    try:
        pr_kwargs["review_summary"] = render_github_review_context(result, review_warnings=warnings)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print_warning(console, f'Review summary unavailable: {type(exc).__name__}')
        try:
            pr_kwargs["review_summary"] = render_github_review_context(None, review_warnings=warnings)
        except (OSError, ValueError, TypeError, RecursionError):
            pr_kwargs["review_summary"] = '⚠️ **Review completeness unavailable.**'
    try:
        run_id = render_review_run_id(result)
        if run_id:
            run_markdown += '\n\n' + run_id
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        print_warning(console, f'Review run identity unavailable: {type(exc).__name__}')
    outcome = await post_review_to_pr_from_report(
        ctx.work.repo,
        items_file,
        run_info=run_markdown,
        renderers=resolve_review_renderers(ctx.registry),
        console=console,
        post=deep_state.mode == "comment",
        approve_on_clean=ctx.config.approve_on_clean,
        diagram_blocks=(deep_state.diagrams or {}).get("blocks"),
        run_context=ctx.run_context,
        auth=ctx.github_execution.auth,
        **pr_kwargs,
    )
    if failed or deep_state.mode == "comment" and outcome in (PostStatus.NO_PR, PostStatus.FAILED):
        return Stop(1)
    return None
