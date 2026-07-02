"""Built-in registry seed.

``register_builtins(registry)`` seeds the registry with everything daydream
does today: built-in skill slots, prompt names, and all four flow definitions
(pr-feedback, review/comment, shallow, deep).

Uses only function-local late imports (import-cycle guard): this module must
not import from ``daydream.runner`` or ``daydream.phases`` at module level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from daydream.extensions.registry import Registry


def register_builtins(registry: Registry) -> None:
    """Seed ``registry`` with daydream's built-in phases, flows, skills, and prompts."""
    from daydream import config

    for stack_key, skill in config.SKILL_MAP.items():
        registry.override_skill(f"stack:{stack_key}", skill)
    registry.override_skill("structural", config.STRUCTURE_SKILL)
    registry.override_skill("pr-feedback-fetch", config.PR_FEEDBACK_FETCH_SKILL)
    registry.override_skill("pr-feedback-respond", config.PR_FEEDBACK_RESPOND_SKILL)

    _register_builtin_prompts(registry)
    _register_builtin_flows(registry)


def _register_builtin_prompts(registry: Registry) -> None:
    """Seed the v1 named-prompt inventory (contract content, see docs/extensions.md).

    Parse/test/commit/setup-investigator/failure-summarizer prompts are
    intentionally NOT registered: they are schema- and control-loop-coupled.
    """
    from daydream import phases
    from daydream.deep import prompts as deep_prompts

    registry.override_prompt("review", phases.build_review_prompt)
    registry.override_prompt("intent", phases.build_intent_prompt)
    registry.override_prompt("alternatives", phases.build_alternative_review_prompt)
    registry.override_prompt("fix", phases._build_fix_prompt)
    registry.override_prompt("per-stack", deep_prompts.build_per_stack_prompt)
    registry.override_prompt("structural", deep_prompts.build_structural_prompt)
    registry.override_prompt("generic-fallback", deep_prompts.build_generic_fallback_prompt)
    registry.override_prompt("arbiter", deep_prompts.build_arbiter_prompt)
    registry.override_prompt("merge", deep_prompts.build_merge_prompt)
    registry.override_prompt("verify", deep_prompts.build_verification_prompt)


def _register_builtin_flows(registry: Registry) -> None:
    """Seed the built-in flow definitions."""
    from daydream.deep import orchestrator as deep
    from daydream.flows import pr_feedback, review, shallow

    for step in pr_feedback.STEPS:
        registry.register_phase(step)
    registry.set_flow("pr-feedback", [step.name for step in pr_feedback.STEPS])

    for step in review.STEPS:
        registry.register_phase(step)
    registry.set_flow("review", [step.name for step in review.STEPS])

    for step in shallow.STEPS:
        registry.register_phase(step)
    registry.set_flow("shallow", list(shallow.FLOW))

    for step in deep.STEPS:
        registry.register_phase(step)
    registry.set_flow("deep", [step.name for step in deep.STEPS])
