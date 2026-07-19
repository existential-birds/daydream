"""Tests for daydream.config module."""

from daydream.config import (
    AUDIT_CATEGORIES,
    AUDIT_SKILL_MAP,
    DEFAULT_EXPLORATION_MODEL,
    DEFAULT_PI_MODEL,
    EFFORT_TIERS,
    PHASE_DEFAULT_EFFORT,
    PHASE_DEFAULT_MODELS,
)

PHASE_NAMES = {
    "review", "per_stack_review", "arbiter", "suppression", "supervise", "parse", "fix", "test", "verify",
    "exploration", "intent", "wonder", "merge", "pr_feedback", "recon", "audit", "vet", "plan_write",
}


def test_audit_categories_match_playbook() -> None:
    assert set(AUDIT_CATEGORIES) == {
        "correctness",
        "security",
        "performance",
        "tests",
        "tech-debt",
        "dependencies",
        "dx",
        "docs",
        "direction",
    }


def test_quick_effort_tier_uses_high_confidence_core_categories() -> None:
    assert EFFORT_TIERS["quick"].categories == ("correctness", "security", "tests")


def test_audit_skill_map_values_are_plugin_skill_names() -> None:
    for stack_skills in AUDIT_SKILL_MAP.values():
        for skill in stack_skills.values():
            plugin, separator, skill_name = skill.partition(":")
            assert separator == ":" and plugin and skill_name


def test_phase_default_models_covers_all_backends():
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
    # Expensive tier: REVIEW, WONDER, MERGE, PR_FEEDBACK, VET, PLAN_WRITE
    for phase in ("review", "wonder", "merge", "pr_feedback", "vet", "plan_write"):
        assert claude[phase] == "claude-opus-4-8"
    # Mid tier: FIX, TEST, EXPLORATION, PER_STACK_REVIEW, INTENT, RECON, AUDIT
    for phase in ("fix", "test", "exploration", "per_stack_review", "intent", "recon", "audit"):
        assert claude[phase] == "claude-sonnet-5"


def test_per_stack_review_and_arbiter_split():
    """#168: per-stack fan-out defaults to Sonnet; the arbiter stays on Opus."""
    claude = PHASE_DEFAULT_MODELS["claude"]
    assert claude["per_stack_review"] == "claude-sonnet-5"
    assert claude["arbiter"] == "claude-opus-4-8"
    codex = PHASE_DEFAULT_MODELS["codex"]
    assert codex["per_stack_review"] == "gpt-5.6-terra"
    assert codex["arbiter"] == "gpt-5.6-sol"


def test_suppression_uses_cheap_tier():
    """#232: the precision-mode suppression pass defaults to the cheap mid tier
    (never per-finding Opus). Pi keeps its single backend fallback."""
    assert PHASE_DEFAULT_MODELS["claude"]["suppression"] == "claude-sonnet-5"
    assert PHASE_DEFAULT_MODELS["codex"]["suppression"] == "gpt-5.6-terra"
    assert DEFAULT_PI_MODEL == "glm-5.2"


def test_phase_default_models_codex_tier_assignments():
    """Codex mirrors the Claude cheap/mid/heavy tiering across the GPT-5.6 lineup."""
    codex = PHASE_DEFAULT_MODELS["codex"]
    assert codex["parse"] == "gpt-5.6-luna"
    for phase in (
        "fix", "test", "verify", "exploration", "per_stack_review", "suppression", "supervise", "intent", "recon",
        "audit",
    ):
        assert codex[phase] == "gpt-5.6-terra", f"codex phase {phase} should default to the mid tier"
    for phase in ("review", "arbiter", "wonder", "merge", "pr_feedback", "vet", "plan_write"):
        assert codex[phase] == "gpt-5.6-sol", f"codex phase {phase} should default to the heavy tier"


def test_phase_default_effort_is_codex_only_and_covers_every_phase():
    """Only Codex consumes reasoning effort, so only Codex has a default table."""
    assert set(PHASE_DEFAULT_EFFORT.keys()) == {"codex"}
    assert set(PHASE_DEFAULT_EFFORT["codex"].keys()) == PHASE_NAMES


def test_phase_default_effort_tier_assignments():
    effort = PHASE_DEFAULT_EFFORT["codex"]
    for phase in ("parse", "exploration", "recon"):
        assert effort[phase] == "low", f"{phase} should be latency-tier effort"
    for phase in ("fix", "test", "verify", "suppression", "supervise", "merge", "intent"):
        assert effort[phase] == "medium", f"{phase} should be baseline effort"
    for phase in ("per_stack_review", "review", "wonder", "pr_feedback", "audit", "plan_write"):
        assert effort[phase] == "high", f"{phase} should be high effort"
    # The arbiter and vet phases are scoped quality-first passes over small inputs.
    for phase in ("arbiter", "vet"):
        assert effort[phase] == "xhigh"


def test_pi_model_is_a_backend_fallback_not_a_phase_override():
    """Pi uses one backend fallback instead of phase-specific defaults."""
    assert "pi" not in PHASE_DEFAULT_MODELS
    assert DEFAULT_PI_MODEL == "glm-5.2"


def test_default_pi_model_is_glm_5_2():
    assert DEFAULT_PI_MODEL == "glm-5.2"


def test_default_exploration_model_matches_claude_phase_default():
    # EXPLORE precedent: DEFAULT_EXPLORATION_MODEL is the fallback when no flag
    # is set and table lookup misses; keep it consistent with the table for Claude.
    assert DEFAULT_EXPLORATION_MODEL == PHASE_DEFAULT_MODELS["claude"]["exploration"]


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
