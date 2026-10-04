"""Shared model/effort defaults, budgets, artifact names, and rendering limits.

PHASE_DEFAULT_MODELS and PHASE_DEFAULT_EFFORT map backend names to phase-keyed
values. Improve categories/effort tiers and grounded-diagram vocabulary/caps
are declared alongside their defaults so consumers share one policy.
"""

from dataclasses import dataclass

# Default model ids — single source of truth. Resolved by ``create_backend`` only
# when no explicit override is supplied. Every other layer takes ``model: str``
# as required and does no fallback of its own.
DEFAULT_CLAUDE_MODEL = "claude-opus-5"
DEFAULT_CODEX_MODEL = "gpt-5.6-sol"
DEFAULT_PI_MODEL = "deepseek/deepseek-v4-flash-0731"

# Caps the 1.5–5h time tail from a single unbounded run_agent turn (issue #169).
DEFAULT_WALL_BUDGET_S = 1800.0

# Outer review ceiling. ReviewLimits supplies earlier investigation/finalization
# bounds; exhaustion produces a partial review. Fix turns retain the default above.
REVIEW_WALL_BUDGET_S = 3600.0

# Use wall time to bound exploratory turns; tool counts can truncate valid
# investigation. Each call site may still supply an explicit ceiling.
DEFAULT_TOOL_CALL_BUDGET: int | None = None

# Allow target test suites more wall time than ordinary turns while bounding hangs.
TEST_WALL_BUDGET_S = 3600.0

# Fixed shielded cleanup grace after deadline expiry. Hanging backend teardown
# must neither extend the budget indefinitely nor discard captured partial results.
BUDGET_CLEANUP_GRACE_S = 10.0

# Open the run-scoped circuit after consecutive retryable failures; admit one
# probe after the interval. Keep this fixed interval below maximum backoff
# so coordinated recovery cannot lag a single retry ladder.
RETRY_CIRCUIT_FAILURE_THRESHOLD = 3
RETRY_CIRCUIT_PROBE_INTERVAL_S = 30.0

# Charge every backoff and retry attempt to a separate recovery allowance,
# activated by the first retryable failure. Exhaustion re-raises without dispatch.
# File/environment overrides are supported; zero disables recovery.
DEFAULT_RETRY_RECOVERY_ALLOWANCE_S = 300.0

# Bound cumulative fix cost/time per file group across all turns and retries.
# Thread the group deadline into each call for mid-call enforcement; configurable
# through [tool.daydream].
DEFAULT_GROUP_MAX_WALL_S = 600.0  # 10 min of wall-clock across one file group
DEFAULT_GROUP_MAX_SERIAL_ITEMS = 6  # max per-finding fix calls in one group

# Bound one repair job (issue #1210): a single execution, the job's cumulative
# total across every execution, and how many executions it may run. These three
# values are the issue's *proposals* adopted as configurable defaults — not
# measured optimal values, and not derived from a measurement of real repairs.
# The job total deliberately exceeds one execution's ceiling, because a bounded
# second attempt is the entire point of a repair job; raising it is therefore an
# explicit, reviewable policy change rather than a tuning detail. All three are
# overridable per repository through [tool.daydream], and an invalid value
# degrades back to the default below.
DEFAULT_REPAIR_EXECUTION_WALL_S = 1800.0  # wall ceiling for one repair execution
DEFAULT_REPAIR_JOB_WALL_S = 7200.0  # cumulative wall ceiling across all executions
DEFAULT_REPAIR_MAX_EXECUTIONS = 4  # bounded repair executions per job

# Report per-file erosion/verbosity regressions without blocking the run.
# Use before/after deltas, or absolute after-values when the baseline is undefined.
# Record flags in fix-quality-gate.json and the manifest. _step_fix resolves
# [tool.daydream] quality_gate_* overrides; the test gate remains mandatory.
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

# Phase model fallback after explicit CLI/file resolution. Define shared phase
# tiers once so backends cannot drift when a phase is added. Pi is absent because
# PiBackend first honors Pi's own settings, then DEFAULT_PI_MODEL.
_MODEL_PHASE_TIERS = (
    ("parse",),
    ("fix", "test", "verify", "exploration", "per_stack_review", "suppression",
     "supervise", "diagram", "intent", "recon", "audit"),
    ("review", "arbiter", "wonder", "merge", "vet", "plan_write"),
)
PHASE_DEFAULT_MODELS: dict[str, dict[str, str]] = {
    backend: {phase: model for phases, model in zip(_MODEL_PHASE_TIERS, models, strict=True) for phase in phases}
    for backend, models in {
        "claude": ("claude-haiku-4-5", "claude-sonnet-5", DEFAULT_CLAUDE_MODEL),
        "codex": ("gpt-5.6-luna", "gpt-5.6-terra", DEFAULT_CODEX_MODEL),
    }.items()
}


# Lowest-precedence reasoning effort; missing backend/phase entries leave the
# driver's ambient default. Each backend forwards through its native effort knob.
# Merge deep and Improve tables independently so tuning one flow cannot move the other.

# Deep review/fix effort remains Codex-only; Claude/Pi retain ambient defaults.
# Mechanical phases use lower effort, while bounded quality-focused arbitration
# gets xhigh. Expanding backend coverage requires separate behavioral evidence.
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
        "diagram": "medium",
        "wonder": "high",
        "merge": "medium",
        "intent": "medium",
    },
}

# Improve tiers cover all backends. Plan authoring/repair uses max because
# later executors depend on the plan alone and cannot recover missing context.
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

# Opt-in dependency-aware review shards retain the stack-keyed pipeline;
# disabled mode preserves one reviewer per stack.
DEFAULT_DEEP_SHARD_ENABLED: bool = False
DEFAULT_DEEP_SHARD_MAX_FILES: int = 5
DEFAULT_DEEP_SHARD_MAX_BYTES: int = 12288  # == INLINE_DIFF_BUDGET_BYTES
DEFAULT_DEEP_SHARD_FANOUT_CAP: int = 16
DEFAULT_DEEP_SHARD_FRONTIER_MAX: int = 8

# Review reuse defaults on. --no-review-cache also disables exploration caching.
# Evict least-recently-used entries when count, bytes, or idle age exceeds a bound.
DEFAULT_REVIEW_CACHE_ENABLED: bool = True
DEFAULT_REVIEW_CACHE_MAX_ENTRIES: int = 1024
DEFAULT_REVIEW_CACHE_MAX_BYTES: int = 1024**3
DEFAULT_REVIEW_CACHE_MAX_AGE_DAYS: int = 30

# Selective verification is the evidence-approved default; latency is reported,
# not gated. Config verify_all=true restores every non-exempt finding, while
# extra_risk_categories may only widen mandatory verification. Neither has a CLI flag.
DEFAULT_VERIFY_ALL: bool = False
DEFAULT_EXTRA_RISK_CATEGORIES: tuple[str, ...] = ()

# Synthetic scope metadata for structural review alongside language stacks;
# never a CLI-selectable skill name.
STRUCTURE_STACK_NAME: str = "structure"

# Bot setup names shared with workflow templates; packaging tests guard drift.
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


# Issue #1113: grounded mermaid diagrams in the Code Review Summary. Two kinds,
# rendered in this order when both are eligible.
DIAGRAM_KINDS: tuple[str, ...] = ("sequence", "flowchart")

# auto uses deterministic eligibility; explicit kinds force eligibility only,
# never skip verification. off suppresses diagrams. File config permits auto/off;
# --diagram permits every mode, and --diagram-only permits all except off.
DIAGRAM_MODES: tuple[str, ...] = ("auto", "sequence", "flowchart", "both", "off")

# Sequence eligibility requires these non-test file/module floors plus a cross-module
# import edge. [tool.daydream.diagram] may override min_code_files/min_modules.
DEFAULT_DIAGRAM_MIN_CODE_FILES: int = 3
DEFAULT_DIAGRAM_MIN_MODULES: int = 2

# Flowchart eligibility floor: some changed function must gain or modify at
# least this many branch points inside head-side changed hunks. Overridable via
# ``[tool.daydream.diagram] min_branch_points``.
DEFAULT_DIAGRAM_MIN_BRANCH_POINTS: int = 3

# Apply caps after grounding/pruning and before omission floors; renderers recheck.
# Count every dropped element in capped and report it in the rendered footer.
DIAGRAM_MAX_PARTICIPANTS: int = 10
DIAGRAM_MAX_MESSAGES: int = 40
DIAGRAM_MAX_BLOCKS: int = 8
DIAGRAM_MAX_NODES: int = 25
DIAGRAM_MAX_EDGES: int = 40

# Label length caps, applied by the renderers' sanitizer after control
# characters and mermaid metacharacters are stripped.
DIAGRAM_LABEL_CAP_PARTICIPANT: int = 40
DIAGRAM_LABEL_CAP_MESSAGE: int = 80
DIAGRAM_LABEL_CAP_NODE: int = 60
DIAGRAM_LABEL_CAP_EDGE: int = 30
