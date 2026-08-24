"""Configuration constants for daydream.

Provide centralized configuration values used throughout the daydream package.
This module contains constants for stack metadata, file paths, and defaults used
by the review and fix loop system.

Exports:
    AUDIT_CATEGORIES: tuple[str, ...] - Improve audit categories.
    STACK_CHOICES: tuple[str, ...] - Supported built-in stack names (no skills).
    EffortTier: Frozen improve audit effort-tier configuration.
    EFFORT_TIERS: dict[str, EffortTier] - Improve audit effort tiers.
    PLAN_WRITE_MAX_CONCURRENCY: int - Improve plan-writer concurrency ceiling.
    VET_BATCH_MAX_FINDINGS: int - Candidate findings per improve vetting batch.
    REVIEW_OUTPUT_FILE: str - Default filename for storing review results.
    DEFAULT_CLAUDE_MODEL: str - Default Claude model id when no override is given.
    DEFAULT_CODEX_MODEL: str - Default Codex model id when no override is given.
    DEFAULT_PI_MODEL: str - Default Pi model id when no override is given (Nous
        research DeepSeek V4 Flash default).
    DEFAULT_EXPLORATION_MODEL: str - Default model for the EXPLORE phase.
    PHASE_DEFAULT_MODELS: dict[str, dict[str, str]] - Per-backend per-phase default
        model mapping. Outer key is backend name ("claude" or "codex"),
        inner key is the phase name (lowercase, e.g. "review", "parse", "fix"),
        value is the concrete model id.
    PHASE_DEFAULT_EFFORT: dict[str, dict[str, str]] - Per-backend per-phase default
        reasoning effort, same key shape as PHASE_DEFAULT_MODELS. Only consumed by
        the Codex backend.
    STRUCTURE_STACK_NAME: str - Stack identifier emitted by detect_stacks for the
        structural meta-stack assignment.
"""

from dataclasses import dataclass

# Default model ids — single source of truth. Resolved by ``create_backend`` only
# when no explicit override is supplied. Every other layer takes ``model: str``
# as required and does no fallback of its own.
DEFAULT_CLAUDE_MODEL = "claude-opus-5"
DEFAULT_CODEX_MODEL = "gpt-5.6-sol"
DEFAULT_PI_MODEL = "deepseek/deepseek-v4-flash-0731"
DEFAULT_EXPLORATION_MODEL = "claude-sonnet-5"

# Caps the 1.5–5h time tail from a single unbounded run_agent turn (issue #169).
DEFAULT_WALL_BUDGET_S = 1800.0

# Unlimited by default: a tool-call count is a poor proxy for a runaway turn, and
# 50 truncated legitimately exploratory phases (wonder/per-stack review) mid-pass,
# failing the run. The wall budget above is the real bound on the time tail; every
# call site still accepts an explicit ceiling.
DEFAULT_TOOL_CALL_BUDGET: int | None = None

# Wall budget for the test-run turn. Deliberately larger than
# DEFAULT_WALL_BUDGET_S: it bounds the TARGET repo's own test suite, not an LLM
# long tail, and a legitimately slow suite must not be truncated. It still
# bounds a hung turn.
TEST_WALL_BUDGET_S = 3600.0

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

# Issue #315: anti-degradation quality gate. Fail-open: flags and surfaces
# erosion/verbosity growth on files the fix phase edited; never aborts the run
# (the test gate stays the hard gate). Deltas are per-file before/after ratios
# from ``analyze_quality``; a file whose delta exceeds a threshold is flagged in
# ``deep/fix-quality-gate.json`` and the run manifest. The *_ABSOLUTE defaults
# are the yardstick for the undefined-baseline fallback: a BEFORE metric that
# is ``None`` (e.g. a file with no functions pre-fix) has no delta to compare,
# so the AFTER value is checked against the absolute knob, never the delta one
# (#329 / CodeRabbit Finding D). Overridable via ``[tool.daydream]``
# (``quality_gate_enabled`` / ``quality_gate_erosion_delta`` /
# ``quality_gate_verbosity_delta`` / ``quality_gate_erosion_absolute`` /
# ``quality_gate_verbosity_absolute``); resolved in ``_step_fix``.
DEFAULT_QUALITY_GATE_ENABLED = True
DEFAULT_QUALITY_GATE_EROSION_DELTA = 0.05
DEFAULT_QUALITY_GATE_VERBOSITY_DELTA = 0.05
DEFAULT_QUALITY_GATE_EROSION_ABSOLUTE = 0.05
DEFAULT_QUALITY_GATE_VERBOSITY_ABSOLUTE = 0.05

# Plan writers are long, expensive turns and hit Pi's provider rate limit when
# they inherit the standard/deep audit fanout of ten. Keep plan generation at
# the prior stable Pi fanout while audit retains its independent tier ceiling.
PLAN_WRITE_MAX_CONCURRENCY = 2

# Vetting prompts inline their candidate findings as JSON, so the batch size is
# what keeps one vet turn readable at monorepo audit volume.
VET_BATCH_MAX_FINDINGS: int = 20

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
        "review": "claude-opus-5",
        "arbiter": "claude-opus-5",
        "suppression": "claude-sonnet-5",
        "supervise": "claude-sonnet-5",
        "wonder": "claude-opus-5",
        "merge": "claude-opus-5",
        "intent": "claude-sonnet-5",
        "pr_feedback": "claude-opus-5",
        "recon": "claude-sonnet-5",
        "audit": "claude-sonnet-5",
        "vet": "claude-opus-5",
        "plan_write": "claude-opus-5",
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
# All three backends consume the resolved value through their own native knob:
# Claude via ``ClaudeAgentOptions.effort``, Codex via
# ``-c model_reasoning_effort=...``, Pi via ``--thinking``. The five levels
# below are the intersection of the three drivers' vocabularies, so any value
# in this table is valid for any backend.
#
# The table is composed from two independently-owned halves so tuning one flow
# never moves the other. Both are merged into ``PHASE_DEFAULT_EFFORT``, which
# is what the resolver reads.
REASONING_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

# Half one: the review/fix pipeline (deep, shallow, review, pr-feedback).
#
# Codex-only, and deliberately so — this is the historical table and its values
# are tuned against Codex's own ambient default. Claude and Pi have no entry
# here, so those phases resolve to ``None`` and each driver keeps applying the
# default it already had. Adding a backend here changes deep-review behavior
# for every existing user of that backend; do that as its own change, on its
# own evidence, not as a side effect of improve work.
#
# Tiering follows OpenAI's guidance: ``low`` for latency-sensitive mechanical
# work, ``medium`` as the balanced baseline, ``high``/``xhigh`` where more
# reasoning buys measured quality. ``arbiter`` gets ``xhigh`` because it is the
# scoped quality-first pass over only high-severity/contested findings, so the
# extra reasoning is bounded to a small input.
DEEP_PHASE_DEFAULT_EFFORT: dict[str, dict[str, str]] = {
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
    },
}

# Half two: the improve advisor (recon, audit, vet, plan_write).
#
# All three backends, because the improve flow runs unattended on a cadence and
# nothing about its output is reviewed in the moment.
#
# ``plan_write`` is pinned to ``max`` everywhere. It covers plan authoring and
# plan repair; those plans are executed later by much
# weaker agents with no context beyond the plan file, so it is the one place
# where spending the most reasoning available is unconditionally correct — a
# handful of calls per run, and every ambiguity left in a plan is paid for by
# the executor.
IMPROVE_PHASE_DEFAULT_EFFORT: dict[str, dict[str, str]] = {
    backend: {
        "recon": "low",
        "audit": "high",
        "vet": "xhigh",
        "plan_write": "max",
    }
    for backend in ("claude", "codex", "pi")
}

PHASE_DEFAULT_EFFORT: dict[str, dict[str, str]] = {
    backend: {
        **DEEP_PHASE_DEFAULT_EFFORT.get(backend, {}),
        **IMPROVE_PHASE_DEFAULT_EFFORT.get(backend, {}),
    }
    for backend in {*DEEP_PHASE_DEFAULT_EFFORT, *IMPROVE_PHASE_DEFAULT_EFFORT}
}

# Supported built-in stack choices (lowercase stack names). This is the neutral
# CLI selector metadata after the native-profile migration: a stack is a language
# scope, not a skill.
STACK_CHOICES: tuple[str, ...] = (
    "python",
    "react",
    "elixir",
    "go",
    "rust",
    "ios",
)

AUDIT_CATEGORIES: tuple[str, ...] = (
    "correctness",
    "security",
    "performance",
    "tests",
    "tech-debt",
    "dependencies",
    "dx",
    "docs",
)



@dataclass(frozen=True)
class EffortTier:
    """Configuration for one improve audit effort tier."""

    categories: tuple[str, ...] | None
    max_concurrency: int
    high_confidence_only: bool
    max_findings: int | None
    include_investigate: bool
    max_partition_groups: int | None


EFFORT_TIERS: dict[str, EffortTier] = {
    "quick": EffortTier(
        categories=("correctness", "security", "tests", "tech-debt"),
        max_concurrency=1,
        high_confidence_only=True,
        max_findings=6,
        include_investigate=False,
        max_partition_groups=None,  # quick audits the whole repo as one group
    ),
    "standard": EffortTier(
        categories=None,
        max_concurrency=10,
        high_confidence_only=False,
        max_findings=None,
        include_investigate=False,
        max_partition_groups=8,
    ),
    "deep": EffortTier(
        categories=None,
        max_concurrency=10,
        high_confidence_only=False,
        max_findings=None,
        include_investigate=True,
        max_partition_groups=None,
    ),
}

# Output file for review results
REVIEW_OUTPUT_FILE = ".review-output.md"

# Issue #309: uncovered-diff-file sweep. After per-stack reviews + parse, the
# deep flow re-reviews diff files no reviewer read with a cheap second-pass
# agent. `uncovered_sweep` toggles the pass (default True);
# `uncovered_sweep_max_files` caps how many uncovered files are swept in one run
# (the remainder is recorded, not silently dropped);
# `uncovered_sweep_min_hunk_lines` skips files whose hunks are trivially small.
DEFAULT_UNCOVERED_SWEEP_ENABLED: bool = True
DEFAULT_UNCOVERED_SWEEP_MAX_FILES: int = 10
DEFAULT_UNCOVERED_SWEEP_MIN_HUNK_LINES: int = 5

# Issue #731: deep-review sharding + coverage-evidence gated uncovered sweep.
# Splits oversized per-language stacks into bounded, dependency-aware shards
# that ride the existing ``stack_name``-keyed pipeline. Default-OFF until the
# sharding/coversweep benchmark gate passes (spec :42-44) -- forensic mode is
# the default and must stay byte-identical; the benchmark harness lives in
# ``bench/sharding-benchmark.py``.
DEFAULT_DEEP_SHARD_ENABLED: bool = False
DEFAULT_DEEP_SHARD_MAX_FILES: int = 5
DEFAULT_DEEP_SHARD_MAX_BYTES: int = 12288  # == INLINE_DIFF_BUDGET_BYTES
DEFAULT_DEEP_SHARD_FANOUT_CAP: int = 16
DEFAULT_DEEP_SHARD_FRONTIER_MAX: int = 8

# Structural-maintainability meta-stack. Deep mode appends a synthetic
# ``StackAssignment`` with ``stack_name=STRUCTURE_STACK_NAME`` so the structural
# reviewer always runs alongside per-language reviewers. It is a scope metadata
# name, not a skill string, and is never selectable from the CLI.
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
    "issues": "write",
    "contents": "read",
    "metadata": "read",
    "actions": "write",
}
