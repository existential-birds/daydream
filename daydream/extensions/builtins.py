"""Built-in registry seed.

``register_builtins(registry)`` seeds the registry with Daydream's prompt names
and two flow definitions (deep, improve). Review/comment/shallow are modes of
the deep flow (#330).

Uses only function-local late imports (import-cycle guard): this module must
not import from ``daydream.runner`` or ``daydream.phases`` at module level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from daydream.extensions.registry import FlowEntry, Registry


def register_builtins(registry: Registry) -> None:
    """Seed ``registry`` with Daydream's built-in phases, flows, and prompts."""
    _register_improve_builtins(registry)
    _register_builtin_prompts(registry)
    _register_builtin_renderers(registry)
    _register_builtin_flows(registry)


def _register_improve_builtins(registry: Registry) -> None:
    """Register the native improve named prompts (no audit skill slots)."""
    from daydream.improve import prompts

    registry.override_prompt("audit", prompts.build_audit_prompt)
    registry.override_prompt("vet", prompts.build_vet_prompt)
    registry.override_prompt("plan-writer", prompts.build_plan_writer_prompt)


def _register_builtin_prompts(registry: Registry) -> None:
    """Seed the v1 named-prompt inventory (contract content, see docs/extensions.md).

    Parse/test/commit/setup-investigator/failure-summarizer prompts are
    intentionally NOT registered: they are schema- and control-loop-coupled.
    """
    from daydream import phases
    from daydream.deep import prompts as deep_prompts

    registry.override_prompt("intent", phases.build_intent_prompt)
    registry.override_prompt("alternatives", phases.build_alternative_review_prompt)
    registry.override_prompt("fix", phases._build_fix_prompt)
    registry.override_prompt("per-stack", deep_prompts.build_per_stack_prompt)
    registry.override_prompt("structural", deep_prompts.build_structural_prompt)
    registry.override_prompt("generic-fallback", deep_prompts.build_generic_fallback_prompt)
    registry.override_prompt("arbiter", deep_prompts.build_arbiter_prompt)
    registry.override_prompt("supervise", deep_prompts.build_supervise_prompt)
    registry.override_prompt("suppression", deep_prompts.build_suppression_prompt)
    registry.override_prompt("merge", deep_prompts.build_merge_prompt)
    registry.override_prompt("verify", deep_prompts.build_verification_prompt)
    registry.override_prompt("fix-verify", deep_prompts.build_fix_verify_prompt)
    registry.override_prompt("diagram_sequence", deep_prompts.build_sequence_diagram_prompt)
    registry.override_prompt("diagram_flowchart", deep_prompts.build_flowchart_prompt)


def _register_builtin_renderers(registry: Registry) -> None:
    """Seed the built-in comment renderers (byte-identical to today's markdown)."""
    from daydream import pr_review

    registry.override_renderer("finding", pr_review.default_render_finding)
    registry.override_renderer("summary", pr_review.default_render_summary)


def _register_builtin_flows(registry: Registry) -> None:
    """Seed the built-in flow definitions (deep + improve only, #330)."""
    from daydream.deep import orchestrator as deep
    from daydream.flows.engine import LoopGroup
    from daydream.improve import orchestrator as improve

    for step in deep.STEPS:
        registry.register_phase(step)
    # Issue #744: the fix cycle is a fix -> verify -> re-dispatch loop. The
    # flat ``fix`` step is wrapped together with the post-fix ``fix-verify``
    # step in a LoopGroup (budget 3 rounds); ``fix-verify`` emits BreakLoop when
    # no actionable verdicts remain and the group ends. Every other step stays a
    # plain entry. ``max_iterations`` is a ctx lambda: the round budget is a
    # fixed 3 per the issue.
    entries: list[FlowEntry] = []
    for step in deep.STEPS:
        if step.name == "fix":
            entries.append(
                LoopGroup(
                    name="fix-verify-loop",
                    steps=("fix", "fix-verify"),
                    max_iterations=lambda ctx: 3,
                )
            )
        elif step.name != "fix-verify":
            entries.append(step.name)
    registry.set_flow("deep", entries)

    for step in improve.STEPS:
        registry.register_phase(step)
    registry.set_flow("improve", [step.name for step in improve.STEPS])
