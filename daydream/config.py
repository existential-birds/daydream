"""Configuration constants for daydream.

Provide centralized configuration values used throughout the daydream package.
This module contains constants for skill mappings, file paths, and regex patterns
used by the review and fix loop system.

Exports:
    AUDIT_CATEGORIES: tuple[str, ...] - Improve audit categories.
    AUDIT_SKILL_MAP: dict[str, dict[str, str]] - Audit skill names by category
        and detected stack.
    EffortTier: Frozen improve audit effort-tier configuration.
    EFFORT_TIERS: dict[str, EffortTier] - Improve audit effort tiers.
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
        model mapping. Outer key is backend name ("claude" or "codex"),
        inner key is the phase name (lowercase, e.g. "review", "parse", "fix"),
        value is the concrete model id.
    PHASE_DEFAULT_EFFORT: dict[str, dict[str, str]] - Per-backend per-phase default
        reasoning effort, same key shape as PHASE_DEFAULT_MODELS. Only consumed by
        the Codex backend.
    STRUCTURE_SKILL: str - Beagle skill name for the structural-maintainability
        meta-stack reviewer. Invoked internally by deep mode; not user-selectable
        (intentionally absent from REVIEW_SKILLS, SKILL_MAP, and ReviewSkillChoice).
    STRUCTURE_STACK_NAME: str - Stack identifier emitted by detect_stacks for the
        structural meta-stack assignment.
"""

from dataclasses import dataclass
from enum import Enum

# Default model ids — single source of truth. Resolved by ``create_backend`` only
# when no explicit override is supplied. Every other layer takes ``model: str``
# as required and does no fallback of its own.
DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"
DEFAULT_CODEX_MODEL = "gpt-5.6-sol"
DEFAULT_PI_MODEL = "glm-5.2"
DEFAULT_EXPLORATION_MODEL = "claude-sonnet-5"

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
# ``"pr_feedback"``, ``"recon"``, ``"audit"``, ``"vet"``,
# ``"plan_write"``).
#
# Claude tiering:
#   - cheap (haiku):   PARSE
#   - mid   (sonnet):  FIX, TEST, EXPLORATION, PER_STACK_REVIEW, INTENT,
#                      SUPPRESSION, RECON, AUDIT
#   - heavy (opus):    REVIEW, WONDER, MERGE, PR_FEEDBACK, ARBITER, VET,
#                      PLAN_WRITE
#
# Codex tiering mirrors it across the GPT-5.6 lineup:
#   - cheap (gpt-5.6-luna):   PARSE
#   - mid   (gpt-5.6-terra):  FIX, TEST, VERIFY, EXPLORATION, PER_STACK_REVIEW,
#                             INTENT, SUPPRESSION, SUPERVISE, RECON, AUDIT
#   - heavy (gpt-5.6-sol):    REVIEW, WONDER, MERGE, PR_FEEDBACK, ARBITER, VET,
#                             PLAN_WRITE
#
# ``suppression`` (issue #232) is the precision-mode skeptical second opinion over
# borderline uncontested findings; it runs on the cheap mid tier by design (never
# per-finding Opus) -- one batched Sonnet call over all suppression targets.
# ``supervise`` is the batched findings supervisor over canonical merged items;
# it uses the same Sonnet tier by default.
#
# ``per_stack_review`` and ``arbiter`` split the deep per-stack fan-out off the
# heavy ``review`` tier (issue #168): the N per-stack reviewers run on Sonnet
# while a single Opus arbiter re-reviews only the high-severity/contested
# findings they surface. ``per_stack_review`` is independently overridable from
# ``review``/``wonder``/``merge``.
#
# ``PHASE_DEFAULT_EFFORT`` supplies the matching per-phase reasoning-effort
# defaults; see its own docstring below.
#
# Pi's ``DEFAULT_PI_MODEL`` is resolved by ``PiBackend`` after Pi's own settings
# have had a chance to select a model. It is a backend fallback, not a
# per-phase override, so it intentionally does not appear in this table.
PHASE_DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "claude": {
        "parse": "claude-haiku-4-5",
        "fix": "claude-sonnet-5",
        "test": "claude-sonnet-5",
        "verify": "claude-sonnet-5",
        "exploration": "claude-sonnet-5",
        "per_stack_review": "claude-sonnet-5",
        "review": "claude-opus-4-8",
        "arbiter": "claude-opus-4-8",
        "suppression": "claude-sonnet-5",
        "supervise": "claude-sonnet-5",
        "wonder": "claude-opus-4-8",
        "merge": "claude-opus-4-8",
        "intent": "claude-sonnet-5",
        "pr_feedback": "claude-opus-4-8",
        "recon": "claude-sonnet-5",
        "audit": "claude-sonnet-5",
        "vet": "claude-opus-4-8",
        "plan_write": "claude-opus-4-8",
    },
    "codex": {
        "parse": "gpt-5.6-luna",
        "fix": "gpt-5.6-terra",
        "test": "gpt-5.6-terra",
        "verify": "gpt-5.6-terra",
        "exploration": "gpt-5.6-terra",
        "per_stack_review": "gpt-5.6-terra",
        "review": "gpt-5.6-sol",
        "arbiter": "gpt-5.6-sol",
        "suppression": "gpt-5.6-terra",
        "supervise": "gpt-5.6-terra",
        "wonder": "gpt-5.6-sol",
        "merge": "gpt-5.6-sol",
        "intent": "gpt-5.6-terra",
        "pr_feedback": "gpt-5.6-sol",
        "recon": "gpt-5.6-terra",
        "audit": "gpt-5.6-terra",
        "vet": "gpt-5.6-sol",
        "plan_write": "gpt-5.6-sol",
    },
}

# Per-backend per-phase default reasoning effort, resolved by
# ``daydream.runner._resolved_reasoning_effort`` as the lowest precedence tier
# (below ``--reasoning-effort`` and both config-file tiers). A backend absent
# from this table, or a phase absent from its sub-table, resolves to ``None`` —
# the backend then applies its own ambient default.
#
# Only Codex consumes the resolved value. GPT-5.6 accepts ``none``, ``low``,
# ``medium``, ``high``, ``xhigh``, and ``max``, defaulting to ``medium``.
# Tiering follows OpenAI's guidance: ``low`` for latency-sensitive mechanical
# work, ``medium`` as the balanced baseline, ``high``/``xhigh`` where more
# reasoning buys measured quality. ``arbiter`` gets ``xhigh`` because it is the
# scoped quality-first pass over only high-severity/contested findings, so the
# extra reasoning is bounded to a small input.
PHASE_DEFAULT_EFFORT: dict[str, dict[str, str]] = {
    "codex": {
        "parse": "low",
        "fix": "medium",
        "test": "medium",
        "verify": "medium",
        "exploration": "low",
        "per_stack_review": "high",
        "review": "high",
        "arbiter": "xhigh",
        "suppression": "medium",
        "supervise": "medium",
        "wonder": "high",
        "merge": "medium",
        "intent": "medium",
        "pr_feedback": "high",
        "recon": "low",
        "audit": "high",
        "vet": "xhigh",
        "plan_write": "high",
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

AUDIT_CATEGORIES: tuple[str, ...] = (
    "correctness",
    "security",
    "performance",
    "tests",
    "tech-debt",
    "dependencies",
    "dx",
    "docs",
    "direction",
)


@dataclass(frozen=True)
class EffortTier:
    """Configuration for one improve audit effort tier."""

    categories: tuple[str, ...] | None
    max_concurrency: int
    high_confidence_only: bool
    max_findings: int | None
    include_investigate: bool


EFFORT_TIERS: dict[str, EffortTier] = {
    "quick": EffortTier(
        categories=("correctness", "security", "tests"),
        max_concurrency=1,
        high_confidence_only=True,
        max_findings=6,
        include_investigate=False,
    ),
    "standard": EffortTier(
        categories=None,
        max_concurrency=4,
        high_confidence_only=False,
        max_findings=None,
        include_investigate=False,
    ),
    "deep": EffortTier(
        categories=None,
        max_concurrency=4,
        high_confidence_only=False,
        max_findings=None,
        include_investigate=True,
    ),
}

AUDIT_SKILL_MAP: dict[str, dict[str, str]] = {
    "correctness": {
        "python": REVIEW_SKILLS[ReviewSkillChoice.PYTHON],
        "react": REVIEW_SKILLS[ReviewSkillChoice.REACT],
        "elixir": REVIEW_SKILLS[ReviewSkillChoice.ELIXIR],
        "go": REVIEW_SKILLS[ReviewSkillChoice.GO],
        "rust": REVIEW_SKILLS[ReviewSkillChoice.RUST],
        "ios": REVIEW_SKILLS[ReviewSkillChoice.IOS],
    },
    "security": {
        "elixir": "beagle-elixir:elixir-security-review",
    },
    "performance": {
        "elixir": "beagle-elixir:elixir-performance-review",
    },
    "tests": {
        "python": "beagle-python:pytest-code-review",
        "go": "beagle-go:go-testing-code-review",
        "rust": "beagle-rust:rust-testing-code-review",
        "elixir": "beagle-elixir:exunit-code-review",
    },
    "tech-debt": {
        "*": "beagle-core:review-structure",
    },
    "dependencies": {},
    "dx": {},
    "docs": {},
    "direction": {},
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
