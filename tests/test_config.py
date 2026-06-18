"""Tests for daydream.config module."""

from daydream.config import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
    DEFAULT_EXPLORATION_MODEL,
    PHASE_DEFAULT_MODELS,
)

PHASE_NAMES = {
    "review", "per_stack_review", "arbiter", "parse", "fix", "test", "verify",
    "exploration", "intent", "wonder", "envision", "merge", "pr_feedback",
}


def test_phase_default_models_covers_both_backends():
    assert set(PHASE_DEFAULT_MODELS.keys()) == {"claude", "codex"}


def test_phase_default_models_covers_every_phase_for_each_backend():
    for backend_name in ("claude", "codex"):
        assert set(PHASE_DEFAULT_MODELS[backend_name].keys()) == PHASE_NAMES, (
            f"{backend_name} default table missing phase entries"
        )


def test_phase_default_models_claude_tier_assignments():
    claude = PHASE_DEFAULT_MODELS["claude"]
    # PARSE is the cheap tier
    assert claude["parse"] == "claude-haiku-4-5"
    # Expensive tier: REVIEW, WONDER, ENVISION, MERGE, INTENT, PR_FEEDBACK
    for phase in ("review", "wonder", "envision", "merge", "intent", "pr_feedback"):
        assert claude[phase] == "claude-opus-4-8"
    # Mid tier: FIX, TEST, EXPLORATION, PER_STACK_REVIEW
    for phase in ("fix", "test", "exploration", "per_stack_review"):
        assert claude[phase] == "claude-sonnet-4-6"


def test_per_stack_review_and_arbiter_split():
    """#168: per-stack fan-out defaults to Sonnet; the arbiter stays on Opus."""
    claude = PHASE_DEFAULT_MODELS["claude"]
    assert claude["per_stack_review"] == "claude-sonnet-4-6"
    assert claude["arbiter"] == "claude-opus-4-8"
    codex = PHASE_DEFAULT_MODELS["codex"]
    assert codex["per_stack_review"] == "gpt-5.5"
    assert codex["arbiter"] == "gpt-5.5"


def test_phase_default_models_codex_uses_gpt_5_5_for_every_phase():
    codex = PHASE_DEFAULT_MODELS["codex"]
    for phase in PHASE_NAMES:
        assert codex[phase] == "gpt-5.5", (
            f"codex phase {phase} should default to gpt-5.5 in v1"
        )


def test_default_exploration_model_matches_claude_phase_default():
    # EXPLORE precedent: DEFAULT_EXPLORATION_MODEL is the fallback when no flag
    # is set and table lookup misses; keep it consistent with the table for Claude.
    assert DEFAULT_EXPLORATION_MODEL == PHASE_DEFAULT_MODELS["claude"]["exploration"]


def test_default_constants_still_exported():
    # Sanity: existing default constants remain importable for backend creation fallbacks.
    assert isinstance(DEFAULT_CLAUDE_MODEL, str)
    assert isinstance(DEFAULT_CODEX_MODEL, str)


def test_structure_skill_constant_not_user_selectable() -> None:
    """The structural reviewer is a meta-stack: invokable internally, never via CLI."""
    from daydream.config import (
        REVIEW_SKILLS,
        SKILL_MAP,
        STRUCTURE_SKILL,
        STRUCTURE_STACK_NAME,
        ReviewSkillChoice,
    )

    assert STRUCTURE_SKILL == "beagle-core:review-structure"
    assert STRUCTURE_STACK_NAME == "structure"
    assert STRUCTURE_SKILL not in SKILL_MAP.values()
    assert STRUCTURE_STACK_NAME not in SKILL_MAP
    assert STRUCTURE_SKILL not in REVIEW_SKILLS.values()
    assert all(choice.name != "STRUCTURE" for choice in ReviewSkillChoice)
