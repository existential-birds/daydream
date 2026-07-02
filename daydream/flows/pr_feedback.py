"""PR-feedback flow steps, lifted verbatim from ``runner._run_pr_feedback``.

Each step is a verbatim lift of a contiguous block of the old imperative
body (Pattern: Step extraction): locals became ``ctx.data[...]`` entries,
``_resolve_backend(config, "<phase>", cache)`` became
``ctx.backend_for("<phase>")``, and early ``return <int>`` became
``return Stop(<int>)``. ``phase_scope`` wrappers, error handling, and print
calls moved with their blocks unchanged, so the flow is behavior-neutral.

``register_builtins`` registers :data:`STEPS` and the ``pr-feedback`` flow
definition; ``_run_pr_feedback`` keeps the preamble (arg validation, backend
info block, trajectory recorder) and delegates here via ``run_flow``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from daydream.agent import console
from daydream.extensions.api import FlowStep, Stop
from daydream.phases import (
    FixResult,
    phase_commit_push_auto,
    phase_fetch_pr_feedback,
    phase_fix,
    phase_parse_feedback,
    phase_respond_pr_feedback,
)
from daydream.trajectory import DaydreamPhase, phase_scope
from daydream.ui import print_error, print_info, print_success, print_warning

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext


async def _step_fetch_feedback(ctx: FlowContext) -> None:
    """Fetch bot review comments via the fetch-pr-feedback skill."""
    await phase_fetch_pr_feedback(
        ctx.backend_for("pr_feedback"), ctx.work, ctx.data["pr_number"], ctx.data["bot"]
    )


async def _step_parse_feedback(ctx: FlowContext) -> Stop | None:
    """Parse the fetched feedback into actionable items; stop when none."""
    try:
        async with phase_scope(DaydreamPhase.PARSE):
            feedback_items = await phase_parse_feedback(ctx.backend_for("parse"), ctx.work)
    except ValueError:
        print_error(console, "Parse Failed", "Failed to parse PR feedback. Exiting.")
        return Stop(1)

    if not feedback_items:
        print_info(console, "No actionable feedback found in PR comments.")
        return Stop(0)

    ctx.data["feedback_items"] = feedback_items
    return None


async def _step_fix_items(ctx: FlowContext) -> Stop | None:
    """Apply a fix per feedback item; abort before commit when all fail."""
    feedback_items = ctx.data["feedback_items"]
    fix_backend = ctx.backend_for("fix")

    # Fix sequentially to avoid concurrent access to one mutable backend.
    results: list[FixResult] = []
    total_items = len(feedback_items)
    async with phase_scope(DaydreamPhase.FIX):
        for idx, item in enumerate(feedback_items, start=1):
            try:
                await phase_fix(fix_backend, ctx.work, item, idx, total_items)
                results.append((item, True, None))
            except Exception as e:
                results.append((item, False, f"{type(e).__name__}: {e}"))

    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    if not successful:
        print_error(
            console,
            "All Fixes Failed",
            f"All {len(failed)} fix(es) failed. Aborting before commit.",
        )
        return Stop(1)

    ctx.data["results"] = results
    ctx.data["successful"] = successful
    ctx.data["failed"] = failed
    return None


async def _step_commit_push(ctx: FlowContext) -> Stop | None:
    """Commit and push the applied fixes."""
    results: list[FixResult] = ctx.data["results"]
    try:
        await phase_commit_push_auto(
            ctx.backend_for("review"), ctx.work, items=[item for item, _ok, _err in results if _ok],
        )
    except Exception as e:
        print_error(console, "Commit/Push Failed", str(e))
        return Stop(1)
    return None


async def _step_respond_feedback(ctx: FlowContext) -> Stop:
    """Reply on each addressed comment and print the run summary."""
    pr_number = ctx.data["pr_number"]
    try:
        await phase_respond_pr_feedback(
            ctx.backend_for("pr_feedback"), ctx.work, pr_number, ctx.data["bot"], ctx.data["results"]
        )
    except Exception as e:
        print_warning(console, f"Failed to respond to PR comments: {e}")
        print_info(console, "Fixes were already pushed successfully.")

    successful = ctx.data["successful"]
    failed = ctx.data["failed"]
    console.print()
    print_success(
        console,
        f"PR #{pr_number}: {len(successful)} fix(es) applied"
        + (f", {len(failed)} failed" if failed else ""),
    )

    return Stop(0)


STEPS: tuple[FlowStep, ...] = (
    FlowStep(name="fetch-feedback", run=_step_fetch_feedback, config_phase="pr_feedback"),
    FlowStep(name="parse-feedback", run=_step_parse_feedback, config_phase="parse"),
    FlowStep(name="fix-items", run=_step_fix_items, config_phase="fix"),
    # config_phase "review" mirrors the old body's use of the review backend for the commit.
    FlowStep(name="commit-push", run=_step_commit_push, config_phase="review"),
    FlowStep(name="respond-feedback", run=_step_respond_feedback, config_phase="pr_feedback"),
)
