"""Configuration constants for daydream.

Provide centralized configuration values used throughout the daydream package.
This module contains constants for skill mappings, file paths, and regex patterns
used by the review and fix loop system.

Exports:
    ReviewSkillChoice: Enum for review skill menu choices.
    REVIEW_SKILLS: dict[ReviewSkillChoice, str] - Mapping of review type identifiers to skill names.
    REVIEW_OUTPUT_FILE: str - Default filename for storing review results.
    UNKNOWN_SKILL_PATTERN: str - Regex pattern for detecting unknown skill errors.
    DEFAULT_CLAUDE_MODEL: str - Default Claude model id when no override is given.
    DEFAULT_CODEX_MODEL: str - Default Codex model id when no override is given.
    DEFAULT_PI_MODEL: str - Default Pi model id when no override is given (z.ai
        coding plan GLM default).
    DEFAULT_EXPLORATION_MODEL: str - Default model for the EXPLORE phase.
    PHASE_DEFAULT_MODELS: dict[str, dict[str, str]] - Per-backend per-phase default
        model mapping. Outer key is backend name ("claude", "codex", or "pi"),
        inner key is the phase name (lowercase, e.g. "review", "parse", "fix"),
        value is the concrete model id.
    STRUCTURE_SKILL: str - Beagle skill name for the structural-maintainability
        meta-stack reviewer. Invoked internally by deep mode; not user-selectable
        (intentionally absent from REVIEW_SKILLS, SKILL_MAP, and ReviewSkillChoice).
    STRUCTURE_STACK_NAME: str - Stack identifier emitted by detect_stacks for the
        structural meta-stack assignment.
"""

from enum import Enum

# Default model ids — single source of truth. Resolved by ``create_backend`` only
# when no explicit override is supplied. Every other layer takes ``model: str``
# as required and does no fallback of its own.
DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"
DEFAULT_CODEX_MODEL = "gpt-5.3-codex"
DEFAULT_PI_MODEL = "glm-5.2"
DEFAULT_EXPLORATION_MODEL = "claude-sonnet-4-6"

# Caps the 1.5–5h time tail from a single unbounded run_agent turn (issue #169).
DEFAULT_WALL_BUDGET_S = 1800.0
DEFAULT_TOOL_CALL_BUDGET = 50

# Per-file-group aggregate budget for the fix phase (issue #201). The
# per-invocation guards above bound each individual run_agent turn; these bound
# the *cumulative* cost of all fix turns targeting a single file group, so one
# runaway file (the #186 pattern: 9 serial fix calls on one file) cannot
# silently dominate a run. Enforced between calls in ``phase_fix_parallel``
# (Approach B — no mid-call abort). Overridable via ``[tool.daydream]``.
#
# Values validated against 139–484 archived runs in ~/.daydream/archive/runs:
#   600s: pi fix calls run p90=623s / max=1731s, so 600s caps the 1837s/5-call
#   pi runaway to 2 fixes and the #186 9-call group to 2, while a legit slow
#   single-call group still rides its own 1800s per-call wall budget.
#   6 items: >6 findings on one file is 3.5% of files, and the dropped tail is
#   the lowest-severity findings (the group is severity-sorted).
DEFAULT_GROUP_MAX_WALL_S = 600.0  # 10 min of wall-clock across one file group
DEFAULT_GROUP_MAX_SERIAL_ITEMS = 6  # max per-finding fix calls in one group

# Per-backend per-phase default model table. The phase resolver in
# ``daydream.runner._resolve_backend`` looks up
# ``PHASE_DEFAULT_MODELS[backend_name][phase_name]`` when no explicit per-phase
# flag is supplied. Phase names are lowercase and match the strings passed by
# every call site (``"review"``, ``"parse"``, ``"fix"``, ``"test"``,
# ``"exploration"``, ``"intent"``, ``"wonder"``, ``"merge"``,
# ``"pr_feedback"``).
#
# Claude tiering:
#   - cheap (haiku):   PARSE
#   - mid   (sonnet):  FIX, TEST, EXPLORATION, PER_STACK_REVIEW, INTENT, SUPPRESSION
#   - heavy (opus):    REVIEW, WONDER, MERGE, PR_FEEDBACK, ARBITER
#
# ``suppression`` (issue #232) is the precision-mode skeptical second opinion over
# borderline uncontested findings; it runs on the cheap mid tier by design (never
# per-finding Opus) -- one batched Sonnet call over all suppression targets.
#
# ``per_stack_review`` and ``arbiter`` split the deep per-stack fan-out off the
# heavy ``review`` tier (issue #168): the N per-stack reviewers run on Sonnet
# while a single Opus arbiter re-reviews only the high-severity/contested
# findings they surface. ``per_stack_review`` is independently overridable from
# ``review``/``wonder``/``merge``.
#
# Codex side defaults to ``gpt-5.5`` across the board in v1; per-phase tiering
# for codex is deferred until concrete model picks across the codex lineup are
# settled.
#
# Pi side defaults to ``glm-5.2`` (z.ai coding plan) across the board; per-phase
# tiering is deferred until the GLM lineup (glm-5.2 / glm-4.5-air / etc.) is
# mapped to tiers.
PHASE_DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "claude": {
        "parse": "claude-haiku-4-5",
        "fix": "claude-sonnet-4-6",
        "test": "claude-sonnet-4-6",
        "verify": "claude-sonnet-4-6",
        "exploration": "claude-sonnet-4-6",
        "per_stack_review": "claude-sonnet-4-6",
        "review": "claude-opus-4-8",
        "arbiter": "claude-opus-4-8",
        "suppression": "claude-sonnet-4-6",
        "wonder": "claude-opus-4-8",
        "merge": "claude-opus-4-8",
        "intent": "claude-sonnet-4-6",
        "pr_feedback": "claude-opus-4-8",
    },
    "codex": {
        "parse": "gpt-5.5",
        "fix": "gpt-5.5",
        "test": "gpt-5.5",
        "verify": "gpt-5.5",
        "exploration": "gpt-5.5",
        "per_stack_review": "gpt-5.5",
        "review": "gpt-5.5",
        "arbiter": "gpt-5.5",
        "suppression": "gpt-5.5",
        "wonder": "gpt-5.5",
        "merge": "gpt-5.5",
        "intent": "gpt-5.5",
        "pr_feedback": "gpt-5.5",
    },
    "pi": {
        "parse": "glm-5.2",
        "fix": "glm-5.2",
        "test": "glm-5.2",
        "verify": "glm-5.2",
        "exploration": "glm-5.2",
        "per_stack_review": "glm-5.2",
        "review": "glm-5.2",
        "arbiter": "glm-5.2",
        "suppression": "glm-5.2",
        "wonder": "glm-5.2",
        "merge": "glm-5.2",
        "intent": "glm-5.2",
        "pr_feedback": "glm-5.2",
    },
}


class ReviewSkillChoice(Enum):
    """Enum for review skill menu choices."""

    PYTHON = "1"
    REACT = "2"
    ELIXIR = "3"
    GO = "4"
    RUST = "5"
    IOS = "6"


# Skill mapping for review types
REVIEW_SKILLS: dict[ReviewSkillChoice, str] = {
    ReviewSkillChoice.PYTHON: "beagle-python:review-python",
    ReviewSkillChoice.REACT: "beagle-react:review-frontend",
    ReviewSkillChoice.ELIXIR: "beagle-elixir:review-elixir",
    ReviewSkillChoice.GO: "beagle-go:review-go",
    ReviewSkillChoice.RUST: "beagle-rust:review-rust",
    ReviewSkillChoice.IOS: "beagle-ios:review-ios",
}

# CLI skill name to full skill path mapping (derived from REVIEW_SKILLS to avoid duplication)
SKILL_MAP: dict[str, str] = {choice.name.lower(): skill for choice, skill in REVIEW_SKILLS.items()}

# Output file for review results
REVIEW_OUTPUT_FILE = ".review-output.md"

# Pattern to detect unknown skill errors
UNKNOWN_SKILL_PATTERN = r"Unknown skill: ([\w:-]+)"

# Structural-maintainability meta-stack. Deep mode appends a synthetic
# ``StackAssignment`` with ``stack_name=STRUCTURE_STACK_NAME`` and
# ``skill_invocation=STRUCTURE_SKILL`` so the structural reviewer always runs
# alongside per-language reviewers. Intentionally NOT added to ``REVIEW_SKILLS``,
# ``SKILL_MAP``, or ``ReviewSkillChoice`` — this skill is a meta-stack invoked by
# the orchestrator, never selected from the CLI.
STRUCTURE_SKILL: str = "beagle-core:review-structure"
STRUCTURE_STACK_NAME: str = "structure"

# PR-feedback skills for the ``daydream feedback <pr#>`` flow, seeded into the
# extension registry as the ``pr-feedback-fetch`` / ``pr-feedback-respond`` slots.
PR_FEEDBACK_FETCH_SKILL: str = "beagle-core:fetch-pr-feedback"
PR_FEEDBACK_RESPOND_SKILL: str = "beagle-core:respond-pr-feedback"

# Self-hosted review-bot setup constants — single source of truth shared by the
# ``daydream setup`` orchestrator, the packaged workflow YAML, and the browser
# guide. Drift between these names and the workflow templates is guarded by
# ``tests/test_templates_packaging.py``.
SETUP_SECRET_NAMES: tuple[str, ...] = (
    "DAYDREAM_APP_ID",
    "DAYDREAM_APP_PRIVATE_KEY",
    "ANTHROPIC_API_KEY",
)
BOT_HANDLE_VAR: str = "DAYDREAM_BOT_HANDLE"
APP_PERMISSIONS: dict[str, str] = {
    "pull_requests": "write",
    "contents": "read",
    "metadata": "read",
    "actions": "write",
}
