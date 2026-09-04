"""Tests for daydream.config module."""
from typing import Any

import pytest

from daydream.config import (
    AUDIT_CATEGORIES,
    DEEP_PHASE_DEFAULT_EFFORT,
    DEFAULT_DIAGRAM_MIN_BRANCH_POINTS,
    DEFAULT_DIAGRAM_MIN_CODE_FILES,
    DEFAULT_DIAGRAM_MIN_MODULES,
    DEFAULT_EXPLORATION_MODEL,
    DEFAULT_PI_MODEL,
    DIAGRAM_KINDS,
    DIAGRAM_LABEL_CAP_EDGE,
    DIAGRAM_LABEL_CAP_MESSAGE,
    DIAGRAM_LABEL_CAP_NODE,
    DIAGRAM_LABEL_CAP_PARTICIPANT,
    DIAGRAM_MAX_BLOCKS,
    DIAGRAM_MAX_EDGES,
    DIAGRAM_MAX_MESSAGES,
    DIAGRAM_MAX_NODES,
    DIAGRAM_MAX_PARTICIPANTS,
    DIAGRAM_MODES,
    EFFORT_TIERS,
    IMPROVE_PHASE_DEFAULT_EFFORT,
    PHASE_DEFAULT_EFFORT,
    PHASE_DEFAULT_MODELS,
    REASONING_EFFORT_LEVELS,
    STACK_CHOICES,
    STRUCTURE_STACK_NAME,
)

PHASE_NAMES = {
    "review",
    "per_stack_review",
    "arbiter",
    "suppression",
    "supervise",
    "parse",
    "fix",
    "test",
    "verify",
    "exploration",
    "intent",
    "wonder",
    "merge",
    "diagram",
    "recon",
    "audit",
    "vet",
    "plan_write",
}
IMPROVE_PHASE_NAMES = {"recon", "audit", "vet", "plan_write"}


def test_no_pr_feedback_skill_constants() -> None:
    """M7/M8: no PR-feedback skill constants remain in config."""
    from daydream import config

    assert not hasattr(config, "PR_FEEDBACK_FETCH_SKILL")
    assert not hasattr(config, "PR_FEEDBACK_RESPOND_SKILL")


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
    }


def test_stack_choices_are_neutral_stack_names() -> None:
    """M1: STACK_CHOICES names built-in scopes, never skill strings."""
    assert STACK_CHOICES == (
        "python",
        "react",
        "elixir",
        "go",
        "rust",
        "ios",
    )
    for stack in STACK_CHOICES:
        assert "/" not in stack and ":" not in stack


def test_quick_effort_tier_uses_high_confidence_core_categories() -> None:
    assert EFFORT_TIERS["quick"].categories == (
        "correctness",
        "security",
        "tests",
        "tech-debt",
    )


def test_effort_tiers_carry_partition_group_ceilings() -> None:
    assert EFFORT_TIERS["quick"].max_partition_groups is None  # quick never partitions
    assert EFFORT_TIERS["standard"].max_partition_groups == 8
    assert EFFORT_TIERS["deep"].max_partition_groups is None


def test_phase_default_models_covers_all_backends() -> None:
    assert set(PHASE_DEFAULT_MODELS.keys()) == {"claude", "codex"}


def test_phase_default_models_covers_every_phase_for_each_backend() -> None:
    for backend_name in ("claude", "codex"):
        assert set(PHASE_DEFAULT_MODELS[backend_name].keys()) == PHASE_NAMES, (
            f"{backend_name} default table missing phase entries"
        )


def test_phase_default_models_claude_tier_assignments() -> None:
    claude = PHASE_DEFAULT_MODELS["claude"]
    # PARSE is the cheap tier
    assert claude["parse"] == "claude-haiku-4-5"
    # Expensive tier: REVIEW, WONDER, MERGE, VET, PLAN_WRITE
    for phase in ("review", "wonder", "merge", "vet", "plan_write"):
        assert claude[phase] == "claude-opus-5"
    # Mid tier: FIX, TEST, EXPLORATION, PER_STACK_REVIEW, INTENT, DIAGRAM,
    # RECON, AUDIT
    for phase in (
        "fix",
        "test",
        "exploration",
        "per_stack_review",
        "intent",
        "diagram",
        "recon",
        "audit",
    ):
        assert claude[phase] == "claude-sonnet-5"


def test_per_stack_review_and_arbiter_split() -> None:
    """#168: per-stack fan-out defaults to Sonnet; the arbiter stays on Opus."""
    claude = PHASE_DEFAULT_MODELS["claude"]
    assert claude["per_stack_review"] == "claude-sonnet-5"
    assert claude["arbiter"] == "claude-opus-5"
    codex = PHASE_DEFAULT_MODELS["codex"]
    assert codex["per_stack_review"] == "gpt-5.6-terra"
    assert codex["arbiter"] == "gpt-5.6-sol"


def test_suppression_uses_cheap_tier() -> None:
    """#232: the precision-mode suppression pass defaults to the cheap mid tier
    (never per-finding Opus)."""
    assert PHASE_DEFAULT_MODELS["claude"]["suppression"] == "claude-sonnet-5"
    assert PHASE_DEFAULT_MODELS["codex"]["suppression"] == "gpt-5.6-terra"


def test_phase_default_models_codex_tier_assignments() -> None:
    """Codex mirrors the Claude cheap/mid/heavy tiering across the GPT-5.6 lineup."""
    codex = PHASE_DEFAULT_MODELS["codex"]
    assert codex["parse"] == "gpt-5.6-luna"
    for phase in (
        "fix",
        "test",
        "verify",
        "exploration",
        "per_stack_review",
        "suppression",
        "supervise",
        "intent",
        "diagram",
        "recon",
        "audit",
    ):
        assert codex[phase] == "gpt-5.6-terra", f"codex phase {phase} should default to the mid tier"
    for phase in ("review", "arbiter", "wonder", "merge", "vet", "plan_write"):
        assert codex[phase] == "gpt-5.6-sol", f"codex phase {phase} should default to the heavy tier"


def test_deep_effort_table_stays_codex_only() -> None:
    """Improve tiering must not move deep-review behavior for claude/pi.

    Claude and Pi have no deep-phase entry, so those phases resolve to None and
    each driver keeps the ambient default it always had.
    """
    assert set(DEEP_PHASE_DEFAULT_EFFORT.keys()) == {"codex"}
    assert set(DEEP_PHASE_DEFAULT_EFFORT["codex"].keys()) == PHASE_NAMES - IMPROVE_PHASE_NAMES
    for backend in ("claude", "pi"):
        for phase in PHASE_NAMES - IMPROVE_PHASE_NAMES:
            assert phase not in PHASE_DEFAULT_EFFORT[backend], f"{backend}/{phase}"


def test_improve_effort_table_covers_every_backend() -> None:
    """The improve advisor is tiered on all three drivers."""
    assert set(IMPROVE_PHASE_DEFAULT_EFFORT.keys()) == {"claude", "codex", "pi"}
    for backend, table in IMPROVE_PHASE_DEFAULT_EFFORT.items():
        assert set(table.keys()) == IMPROVE_PHASE_NAMES, backend


def test_merged_table_is_the_union_of_its_two_halves() -> None:
    assert PHASE_DEFAULT_EFFORT["codex"] == {
        **DEEP_PHASE_DEFAULT_EFFORT["codex"],
        **IMPROVE_PHASE_DEFAULT_EFFORT["codex"],
    }
    assert PHASE_DEFAULT_EFFORT["claude"] == IMPROVE_PHASE_DEFAULT_EFFORT["claude"]


def test_phase_default_effort_levels_are_valid_for_every_driver() -> None:
    """Only the five levels every driver accepts may appear in the table."""
    assert REASONING_EFFORT_LEVELS == ("low", "medium", "high", "xhigh", "max")
    for backend, table in PHASE_DEFAULT_EFFORT.items():
        for phase, level in table.items():
            assert level in REASONING_EFFORT_LEVELS, f"{backend}/{phase}={level}"


def test_deep_phase_effort_tier_assignments() -> None:
    effort = DEEP_PHASE_DEFAULT_EFFORT["codex"]
    for phase in ("parse", "exploration"):
        assert effort[phase] == "low", f"{phase} should be latency-tier effort"
    for phase in (
        "fix",
        "test",
        "verify",
        "suppression",
        "supervise",
        "merge",
        "intent",
        "diagram",
    ):
        assert effort[phase] == "medium", f"{phase} should be baseline effort"
    for phase in ("per_stack_review", "review", "wonder"):
        assert effort[phase] == "high", f"{phase} should be high effort"
    # The arbiter is a scoped quality-first pass over a small input.
    assert effort["arbiter"] == "xhigh"


@pytest.mark.parametrize("backend", ["claude", "codex", "pi"])
def test_improve_phase_effort_tier_assignments(backend: Any) -> None:
    effort = IMPROVE_PHASE_DEFAULT_EFFORT[backend]
    assert effort["recon"] == "low"
    assert effort["audit"] == "high"
    assert effort["vet"] == "xhigh"


@pytest.mark.parametrize("backend", ["claude", "codex", "pi"])
def test_plan_write_is_pinned_to_max_reasoning_on_every_backend(backend: Any) -> None:
    """Plan authoring and plan repair both ride the plan_write key."""
    assert PHASE_DEFAULT_EFFORT[backend]["plan_write"] == "max"


@pytest.mark.parametrize("backend", ["claude", "codex"])
def test_plan_write_is_pinned_to_the_top_model_tier(backend: Any) -> None:
    """plan_write shares the top tier with the heaviest review phases."""
    models = PHASE_DEFAULT_MODELS[backend]
    assert models["plan_write"] == models["review"] == models["arbiter"]


def test_default_pi_model_is_nous_deepseek_flash() -> None:
    assert DEFAULT_PI_MODEL == "deepseek/deepseek-v4-flash-0731"


def test_default_exploration_model_matches_claude_phase_default() -> None:
    # EXPLORE precedent: DEFAULT_EXPLORATION_MODEL is the fallback when no flag
    # is set and table lookup misses; keep it consistent with the table for Claude.
    assert DEFAULT_EXPLORATION_MODEL == PHASE_DEFAULT_MODELS["claude"]["exploration"]


def test_structure_constant_is_scope_metadata_not_a_skill() -> None:
    """M2: the structural meta-stack is a scope name, never a skill string."""
    assert STRUCTURE_STACK_NAME == "structure"
    assert "/" not in STRUCTURE_STACK_NAME and ":" not in STRUCTURE_STACK_NAME


def test_diagram_kinds_are_the_two_supported_kinds_in_render_order() -> None:
    """#1113: sequence renders before flowchart when both are eligible."""
    assert DIAGRAM_KINDS == ("sequence", "flowchart")


def test_diagram_modes_cover_auto_each_kind_both_and_off() -> None:
    """#1113: the mode vocabulary is the union of auto, each kind, both, off."""
    assert DIAGRAM_MODES == ("auto", "sequence", "flowchart", "both", "off")
    assert set(DIAGRAM_KINDS) < set(DIAGRAM_MODES)


def test_diagram_eligibility_defaults_match_the_documented_thresholds() -> None:
    """#1113: 3 code files / 2 modules / 3 branch points."""
    assert DEFAULT_DIAGRAM_MIN_CODE_FILES == 3
    assert DEFAULT_DIAGRAM_MIN_MODULES == 2
    assert DEFAULT_DIAGRAM_MIN_BRANCH_POINTS == 3


def test_diagram_render_caps_are_positive_and_bound_every_collection() -> None:
    """#1113: every rendered collection has a cap, enforced before the floor."""
    caps = {
        "participants": DIAGRAM_MAX_PARTICIPANTS,
        "messages": DIAGRAM_MAX_MESSAGES,
        "blocks": DIAGRAM_MAX_BLOCKS,
        "nodes": DIAGRAM_MAX_NODES,
        "edges": DIAGRAM_MAX_EDGES,
    }
    assert caps == {
        "participants": 10,
        "messages": 40,
        "blocks": 8,
        "nodes": 25,
        "edges": 40,
    }
    labels = (
        DIAGRAM_LABEL_CAP_PARTICIPANT,
        DIAGRAM_LABEL_CAP_MESSAGE,
        DIAGRAM_LABEL_CAP_NODE,
        DIAGRAM_LABEL_CAP_EDGE,
    )
    assert labels == (40, 80, 60, 30)


def test_diagram_phase_is_mid_tier_on_both_model_backends() -> None:
    """#1113: the diagram author is a mid-tier phase, never the heavy tier."""
    assert PHASE_DEFAULT_MODELS["claude"]["diagram"] == PHASE_DEFAULT_MODELS["claude"]["intent"]
    assert PHASE_DEFAULT_MODELS["codex"]["diagram"] == PHASE_DEFAULT_MODELS["codex"]["intent"]
    assert DEEP_PHASE_DEFAULT_EFFORT["codex"]["diagram"] == "medium"
