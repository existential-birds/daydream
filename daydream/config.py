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
    DEFAULT_EXPLORATION_MODEL: str - Default model for the EXPLORE phase.
    PHASE_DEFAULT_MODELS: dict[str, dict[str, str]] - Per-backend per-phase default
        model mapping. Outer key is backend name ("claude" or "codex"), inner key is
        the phase name (lowercase, e.g. "review", "parse", "fix"), value is the
        concrete model id.
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
DEFAULT_EXPLORATION_MODEL = "claude-sonnet-4-6"

# Caps the 1.5–5h time tail from a single unbounded run_agent turn (issue #169).
DEFAULT_WALL_BUDGET_S = 1800.0
DEFAULT_TOOL_CALL_BUDGET = 50

# Per-backend per-phase default model table. The phase resolver in
# ``daydream.runner._resolve_backend`` looks up
# ``PHASE_DEFAULT_MODELS[backend_name][phase_name]`` when no explicit per-phase
# flag is supplied. Phase names are lowercase and match the strings passed by
# every call site (``"review"``, ``"parse"``, ``"fix"``, ``"test"``,
# ``"exploration"``, ``"intent"``, ``"wonder"``, ``"envision"``, ``"merge"``,
# ``"pr_feedback"``).
#
# Claude tiering:
#   - cheap (haiku):   PARSE
#   - mid   (sonnet):  FIX, TEST, EXPLORATION, PER_STACK_REVIEW
#   - heavy (opus):    REVIEW, WONDER, ENVISION, MERGE, INTENT, PR_FEEDBACK, ARBITER
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
        "wonder": "claude-opus-4-8",
        "envision": "claude-opus-4-8",
        "merge": "claude-opus-4-8",
        "intent": "claude-opus-4-8",
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
        "wonder": "gpt-5.5",
        "envision": "gpt-5.5",
        "merge": "gpt-5.5",
        "intent": "gpt-5.5",
        "pr_feedback": "gpt-5.5",
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
