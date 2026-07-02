"""Review/comment flow steps, lifted verbatim from ``runner._run_review_or_comment``.

Each step is a verbatim lift of a contiguous block of the old imperative
body (Pattern: Step extraction): locals became ``ctx.data[...]`` entries,
``_resolve_backend(config, "<phase>", cache)`` became
``ctx.backend_for("<phase>")``, and early ``return <int>`` became
``return Stop(<int>)``. ``phase_scope`` wrappers, error handling, and print
calls moved with their blocks unchanged, so the flow is behavior-neutral.

One ``"review"`` flow serves both modes: ``ctx.data["post_to_pr"]`` carries
``--comment`` vs ``--review``, gating ``emit-findings`` (review mode with
``--findings-out``) and ``post-comments`` (comment mode) via ``enabled``
predicates. ``register_builtins`` registers :data:`STEPS` and the ``review``
flow definition; ``_run_review_or_comment`` keeps the preamble (diff
computation, diff-file write, trajectory recorder, info block) and delegates
here via ``run_flow``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from daydream.agent import console
from daydream.exploration import ExplorationContext, safe_explore
from daydream.exploration_runner import count_changed_files, pre_scan, select_tier
from daydream.extensions.api import FlowStep, Stop
from daydream.phases import phase_alternative_review, phase_understand_intent
from daydream.trajectory import DaydreamPhase, phase_scope
from daydream.ui import phase_subtitle, print_dim, print_phase_hero, print_success

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext


async def _step_exploration(ctx: FlowContext) -> None:
    """Pre-scan exploration and materialise it to disk for phase prompts."""
    from daydream.runner import _compute_diff_ref

    config = ctx.config
    target_dir = ctx.work.repo
    diff = ctx.data["diff"]
    daydream_dir: Path = ctx.data["daydream_dir"]

    # Pre-scan exploration, unless already populated (injected by caller/tests).
    if config.exploration_context is None:
        tier = select_tier(count_changed_files(diff or ""))
        if tier == "skip":
            print_dim(console, "Skipping exploration -- trivial diff")
            config.exploration_context = ExplorationContext()
        else:
            print_phase_hero(console, "EXPLORE", phase_subtitle("EXPLORE"))
            explore_backend = ctx.backend_for("exploration")
            print_dim(console, f"Exploration model: {explore_backend.model}")
            async with phase_scope(DaydreamPhase.EXPLORATION):
                config.exploration_context = await safe_explore(
                    pre_scan,
                    explore_backend,
                    target_dir,
                    diff,
                    config.exploration_depth,
                    diff_ref=_compute_diff_ref(target_dir),
                )

    # Materialise exploration to disk so phase prompts can reference files.
    exploration_dir: Path | None = None
    if config.exploration_context is not None:
        exploration_dir = config.exploration_context.write_to_dir(daydream_dir / "exploration")
    ctx.data["exploration_dir"] = exploration_dir


async def _step_intent(ctx: FlowContext) -> None:
    """Understand the intent of the changes (LISTEN)."""
    async with phase_scope(DaydreamPhase.INTENT):
        ctx.data["intent_summary"] = await phase_understand_intent(
            ctx.backend_for("review"), ctx.work, ctx.data["diff_path"],
            ctx.data["log"], ctx.data["branch"],
            exploration_dir=ctx.data["exploration_dir"],
        )


async def _step_alternatives(ctx: FlowContext) -> None:
    """Evaluate the implementation for concrete issues (WONDER)."""
    async with phase_scope(DaydreamPhase.ALTERNATIVES):
        ctx.data["issues"] = await phase_alternative_review(
            ctx.backend_for("review"), ctx.work, ctx.data["diff_path"],
            ctx.data["intent_summary"],
            exploration_dir=ctx.data["exploration_dir"],
        )


async def _step_emit_findings(ctx: FlowContext) -> Stop | None:
    """Phase A artifact emission (``--findings-out``, review only)."""
    from daydream.runner import _emit_findings_artifact

    rc = _emit_findings_artifact(ctx.work.repo, ctx.config, ctx.data["issues"])
    if rc != 0:
        return Stop(rc)
    return None


async def _step_no_issues_exit(ctx: FlowContext) -> Stop | None:
    """Stop with success when the review found nothing; else clean up exploration."""
    if not ctx.data["issues"]:
        print_success(console, "No issues found — the implementation looks good!")
        return Stop(0)

    # Today's placement: after the zero-issues early return, before posting.
    exploration_cleanup = ctx.work.repo / ".daydream" / "exploration"
    if exploration_cleanup.is_dir():
        shutil.rmtree(exploration_cleanup)
    return None


async def _step_post_comments(ctx: FlowContext) -> None:
    """Post findings as inline PR comments (``--comment`` only)."""
    from daydream.pr_review import post_review_to_pr_from_alt_issues

    await post_review_to_pr_from_alt_issues(
        ctx.work.repo, ctx.data["issues"], console=console,
    )


def _is_review_with_findings_out(ctx: FlowContext) -> bool:
    return not ctx.data["post_to_pr"] and ctx.config.findings_out is not None


def _is_comment_mode(ctx: FlowContext) -> bool:
    return bool(ctx.data["post_to_pr"])


STEPS: tuple[FlowStep, ...] = (
    # Phase names are a single registry namespace and the deep flow owns the
    # plain "exploration"/"intent"/"alternatives" names, so this flow's
    # variants get flow-qualified names (the shallow-exploration convention).
    # config_phase keeps each step's original per-phase config key, so
    # [tool.daydream.phases.*] resolution is unchanged.
    FlowStep(name="review-exploration", run=_step_exploration, config_phase="exploration"),
    FlowStep(name="review-intent", run=_step_intent, config_phase="intent"),
    FlowStep(name="review-alternatives", run=_step_alternatives, config_phase="wonder"),
    FlowStep(name="emit-findings", run=_step_emit_findings, enabled=_is_review_with_findings_out),
    FlowStep(name="no-issues-exit", run=_step_no_issues_exit),
    FlowStep(name="post-comments", run=_step_post_comments, enabled=_is_comment_mode),
)
