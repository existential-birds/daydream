"""Compose deep and diagram flows, their preambles, and public entry points."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from rich.markup import escape as escape_markup

from daydream.agent import console
from daydream.artifact_visibility import ArtifactVisibilityError, artifact_dir_for, artifact_session_active
from daydream.config import (
    DEFAULT_DEEP_SHARD_ENABLED,
    DEFAULT_DEEP_SHARD_FANOUT_CAP,
    DEFAULT_DEEP_SHARD_FRONTIER_MAX,
    DEFAULT_DEEP_SHARD_MAX_BYTES,
    DEFAULT_DEEP_SHARD_MAX_FILES,
    REVIEW_OUTPUT_FILE,
    STRUCTURE_STACK_NAME,
)
from daydream.config_file import _coerce_non_negative_int
from daydream.deep import review_steps
from daydream.deep.artifacts import alternatives_path as _alternatives_path
from daydream.deep.artifacts import check_deep_artifacts, deep_dir, diff_key, diff_key_path
from daydream.deep.artifacts import intent_path as _intent_path
from daydream.deep.dependency import build_import_graph
from daydream.deep.detection import GENERIC_STACK, StackAssignment, detect_stacks
from daydream.deep.diagram_steps import _diagram_mode_for, _resolved_diagram_mode, _step_diagram, _step_post_diagram
from daydream.deep.diff import _diff_changed_files
from daydream.deep.fix_steps import (
    _perform_cleanup,
    _step_commit,
    _step_fix,
    _step_fix_gate,
    _step_fix_verify,
    _step_remote_ci,
    _step_test,
    _step_verify,
)
from daydream.deep.merge_steps import (
    _step_arbiter,
    _step_cross_stack_merge,
    _step_findings_out,
    _step_load_items,
    _step_post_review,
    _step_single_stack_merge,
    _step_supervise,
    _supervisor_mode,
)
from daydream.deep.prompts import bound_deep_diff
from daydream.deep.render import _PIPELINE_STAGE_NAMES
from daydream.deep.review_steps import (
    _clear_sweep_artifacts,
    _step_exploration,
    _step_intent,
    _step_per_stack_parse,
    _step_uncovered_sweep,
    _step_wonder_and_per_stack,
)
from daydream.deep.settings import _resolve_config_value, fresh_ttt
from daydream.deep.sharding import shard_stacks
from daydream.deep.state import DeepState
from daydream.extensions import get_registry
from daydream.extensions.api import FlowStep
from daydream.flows.engine import BackendFactory, FlowContext, run_flow
from daydream.github_app import GitHubExecutionInput
from daydream.phases import PushReceipt
from daydream.review_profile import Pipeline
from daydream.run_context import RunContext, bind_resolved_run_context, resolve_run_context
from daydream.trajectory import DaydreamRunFlow
from daydream.ui import print_error, print_info, print_preflight_notice, print_warning
from daydream.workspace import WorkContext

if TYPE_CHECKING:
    from daydream.runner import RunConfig, _RunArtifacts


def total_agent_count(stack_count: int) -> int:
    """Return the D-30 agent count formula.

    Formula: 2 (TTT intent + alternative-review) + N per-stack reviews
    + N per-stack parse passes + 1 cross-stack merge + 1 conditional
    arbiter (Opus pass over findings at or above the profile's
    ``Arbitration.min_severity`` — default high — plus contested findings). The
    arbiter fires when qualifying findings exist; the pre-flight estimate
    always includes it so users aren't surprised by the extra Opus call.
    The fix-gate agents are user-gated and excluded from the estimate.

    Args:
        stack_count: Number of detected stack assignments (including the
            generic-fallback bucket when present).
    """
    return 2 + stack_count + stack_count + 1 + 1


# Issue #172 — tiny-diff short-circuit. A diff with at most this many changed
# files collapses the per-language fan-out to a single combined assignment and
# skips the merge agent + arbiter (a tiny diff has nothing to cross-stack-merge
# and nothing contested to arbitrate). A 1-file single-language diff is already
# only 2 stacks (lang + structure), so the collapse is a no-op there and the
# count reduction for that case comes entirely from skipping merge+arbiter
# (lever 2); see ``_single_stack_agent_count``.
DEFAULT_SHALLOW_FANOUT_THRESHOLD = 2


def _single_stack_agent_count(stack_count: int) -> int:
    """Return the agent count for a tiny-diff single-stack run (issue #172).

    Single-stack mode runs 2 TTT + N per-stack reviews + N parse passes but
    skips the merge agent and the arbiter (lever 2). The surviving stack list
    after collapse is at most ``[combined-or-single, structure]`` (≤2).

    Args:
        stack_count: Number of stack assignments AFTER the tiny-diff collapse.
    """
    return 2 + stack_count + stack_count


def _config_pipeline(config: RunConfig) -> Pipeline:
    """Return the resolved profile pipeline for a ``RunConfig`` (pre-context).

    ``FlowContext.pipeline()`` is the in-flow accessor; this mirrors it for the
    pre-flight preamble (printed before the FlowContext is constructed) and for
    any config-only call site. Falls back to the packaged default pipeline when
    no profile was resolved (``review_profile is None``).
    """
    if config.review_profile is not None:
        return config.review_profile.profile.pipeline
    from daydream.review_profile import build_default_profile

    return build_default_profile().pipeline


def _shallow_fanout_threshold(config: RunConfig) -> int:
    """Resolve the tiny-diff short-circuit threshold (issue #172, AC7).

    ``0`` disables the short-circuit.
    """
    return _resolve_config_value(config, "shallow_fanout_threshold", DEFAULT_SHALLOW_FANOUT_THRESHOLD)


def _supervise_enabled(ctx: FlowContext) -> bool:
    """Run supervision on fresh flows, not on a fix-only resume."""
    return _supervisor_mode(ctx.config) in {"rules", "llm"} and ctx.config.start_at != "fix"


def _deep_shard_enabled(config: RunConfig) -> bool:
    """Resolve the deep-review sharding toggle (issue #731).

    Precedence mirrors ``_resolve_config_value``: 1)
    ``RunConfig.deep_shard_enabled`` (CLI tier), 2)
    ``DaydreamFileConfig.deep_shard_enabled`` (file-config scalar), 3) built-in
    default :data:`DEFAULT_DEEP_SHARD_ENABLED` (False, preserving the established
    single-agent-per-stack behavior). Resolved via ``_resolve_config_value``
    (``is not None``, not truthiness) so an explicit set-to-False on the
    RunConfig tier forces the feature off even when the file-config scalar
    enables it -- the CLI > file > default precedence holds for explicit False,
    per the ``RunConfig.deep_shard_enabled`` field contract.
    """
    return _resolve_config_value(config, "deep_shard_enabled", DEFAULT_DEEP_SHARD_ENABLED)


def _deep_shard_int(config: RunConfig, attr: str, default: int) -> int:
    """Resolve an integer sharding bound, coercing-and-degrading (issue #731).

    Integer-only non-negative: ``0`` is preserved where meaningful while a
    negative value, a float, a bool, or any non-int degrades to the named
    default -- mirroring ``config_file._coerce_non_negative_int`` (reused,
    import read-only) so a directly-constructed ``RunConfig`` cannot smuggle an
    invalid bound into the sharder.
    """
    value = _resolve_config_value(config, attr, default)
    coerced = _coerce_non_negative_int(value)
    return coerced if coerced is not None else default


def _deep_shard_max_files(config: RunConfig) -> int:
    """Resolve the per-shard max file-count bound (issue #731)."""
    return _deep_shard_int(config, "deep_shard_max_files", DEFAULT_DEEP_SHARD_MAX_FILES)


def _deep_shard_max_bytes(config: RunConfig) -> int:
    """Resolve the per-shard max changed-byte bound (issue #731)."""
    return _deep_shard_int(config, "deep_shard_max_bytes", DEFAULT_DEEP_SHARD_MAX_BYTES)


def _deep_shard_fanout_cap(config: RunConfig) -> int:
    """Resolve the total shard fan-out cap (issue #731)."""
    return _deep_shard_int(config, "deep_shard_fanout_cap", DEFAULT_DEEP_SHARD_FANOUT_CAP)


def _deep_shard_frontier_max(config: RunConfig) -> int:
    """Resolve the per-shard cross-shard frontier cap (issue #731)."""
    return _deep_shard_int(config, "deep_shard_frontier_max", DEFAULT_DEEP_SHARD_FRONTIER_MAX)


def _uncovered_sweep_enabled(ctx: FlowContext) -> bool:
    """Resolve the uncovered-file sweep toggle from the profile pipeline (issue #309).

    Reads ``ctx.pipeline().uncovered_sweep_enabled`` (already host-clamped);
    resume at ``merge``/``fix`` disables the step outright -- the per-stack
    records are already finalized on disk, so a sweep would re-review stale
    coverage.
    """
    if ctx.config.start_at in ("merge", "fix"):
        return False
    return ctx.pipeline().uncovered_sweep_enabled


def _uncovered_sweep_preflight_note(config: RunConfig, changed_files: list[str]) -> str | None:
    """Sweep additive for the pre-flight agent estimate (issue #309 finding 8).

    The pre-flight total counts only the known phases; the uncovered files the
    sweep will review are not known until after per-stack reviews + parse. Every
    swept file adds one review invocation AND one parse invocation (parse per
    stack file), so an honest estimate appends an upper-bound note: 2 agents per
    file, capped by the pipeline capacity and the number of changed files that
    could possibly be swept. Returns ``None`` when the sweep is disabled or
    nothing could be swept.
    """
    if config.start_at in ("merge", "fix"):
        return None
    pipeline = _config_pipeline(config)
    if not pipeline.uncovered_sweep_enabled:
        return None
    eligible = min(len(changed_files), pipeline.uncovered_sweep_max_files)
    if eligible <= 0:
        return None
    return (
        f"(+ up to {2 * eligible} sweep agents: review + parse per uncovered "
        "file, capped by eligible changed files)"
    )


def _collapse_stacks_for_tiny_diff(
    stacks: list[StackAssignment],
    changed_files: list[str],
    *,
    threshold: int,
) -> tuple[list[StackAssignment], bool]:
    """Collapse the per-language fan-out for a tiny diff (issue #172, Fix A lever 1).

    When ``0 < len(changed_files) <= threshold``:

      - If ≥2 distinct *non-structural* stacks exist, merge them into one
        combined assignment. A code+docs/config diff (exactly one *real*
        language stack plus the ``generic`` bucket) absorbs the generic files
        into the language stack so its scope survives; only ≥2 *real* language
        stacks fall back to ``generic`` (a single agent cannot cover two
        per-language scopes).
      - The ``STRUCTURE_STACK_NAME`` meta-stack stays as its own assignment so
        structural findings remain correctly tagged ``lens="structural"``
        downstream (AC6).
      - If only one non-structural stack exists (the common 1-file case), it is
        preserved unchanged.

    Built-in stacks carry no skill-invocation field (M2): the combined
    assignment is scope metadata only.

    Returns ``(stacks, single_stack_mode)`` where ``single_stack_mode`` reports
    whether the tiny-diff gate is active (caller uses it to skip merge+arbiter).
    When the gate is inactive, ``stacks`` is returned unchanged.

    Args:
        stacks: Stack assignments returned by ``detect_stacks``.
        changed_files: Changed file list used to compute the gate.
        threshold: Resolved threshold from ``_shallow_fanout_threshold``. ``0``
            disables the short-circuit (returns inputs unchanged).

    Returns:
        Tuple of ``(possibly_collapsed_stacks, single_stack_mode)``.
    """
    if threshold <= 0 or not (0 < len(changed_files) <= threshold):
        return stacks, False

    non_structural = [s for s in stacks if s.stack_name != STRUCTURE_STACK_NAME]
    structural = [s for s in stacks if s.stack_name == STRUCTURE_STACK_NAME]

    # When ≥2 distinct non-structural stacks exist, merge them into one combined
    # assignment. The combined scope depends on how many *real* language stacks
    # are present:
    #   - exactly one real language stack + the generic bucket (a code+docs/config
    #     tiny diff, e.g. api.py + README.md): absorb the generic files into the
    #     language stack so its scope survives.
    #   - ≥2 real language stacks (e.g. python + react): a single agent cannot
    #     cover two per-language scopes, so fall back to generic.
    #
    if len(non_structural) >= 2:
        combined_files = sorted({f for s in non_structural for f in s.files})
        real_language = [s for s in non_structural if s.stack_name != GENERIC_STACK]
        if len(real_language) == 1:
            lang = real_language[0]
            return (
                [
                    *structural,
                    StackAssignment(
                        stack_name=lang.stack_name,
                        files=combined_files,
                        is_docs_only=False,
                    ),
                ],
                True,
            )
        # ≥2 real-language stacks: one agent cannot cover two per-language scopes,
        # so the combined assignment uses the native generic-fallback scope.
        combined = StackAssignment(
            stack_name=GENERIC_STACK,
            files=combined_files,
            is_docs_only=False,
        )
        return [*structural, combined], True

    # 0 or 1 non-structural stacks: nothing to collapse (lever 1 is a no-op), but
    # the gate is still active so the caller applies lever 2 (skip merge+arbiter).
    return stacks, True


def _stack_preflight_line(stack: StackAssignment) -> str:
    """Format one detected-stack line for the pre-flight notice (skill-free, M2)."""
    docs_suffix = " (docs-only)" if stack.is_docs_only else ""
    return f"{stack.stack_name}: {len(stack.files)} file(s){docs_suffix}"


def _preflight_stage_names(stacks: list[StackAssignment]) -> list[str]:
    """Return user-facing stages, including the structural review when active."""
    stages = list(_PIPELINE_STAGE_NAMES)
    if any(stack.stack_name == STRUCTURE_STACK_NAME for stack in stacks):
        stages.insert(3, "structural review (parallel with per-stack reviews)")
    return stages




def _has_non_daydream_worktree_changes(status: str) -> bool:
    """Whether porcelain output names a path outside Daydream-owned artifacts."""
    for line in status.splitlines():
        paths = line[3:].split(" -> ")
        if not all(
            path == ".daydream"
            or path.startswith(".daydream/")
            or path == REVIEW_OUTPUT_FILE
            for path in paths
        ):
            return True
    return False


def _remote_ci_enabled(ctx: FlowContext) -> bool:
    """Run remote verification only after this flow recorded a successful push."""
    deep_state = DeepState(ctx.data)
    return isinstance(deep_state.push_receipt, PushReceipt)


def _fresh_ttt(ctx: FlowContext) -> bool:
    # Verbatim resume gate: --start-at per-stack/merge/fix skips the TTT phases.
    return fresh_ttt(ctx.config)


def _before_fix_resume(ctx: FlowContext) -> bool:
    # Verbatim resume gate: --start-at fix skips everything up to the fix gate.
    # For post-review, this also keeps the non-idempotent GitHub write off the
    # resume path (duplicate inline reviews on reruns).
    return ctx.config.start_at != "fix"


def _multi_stack_merge_enabled(ctx: FlowContext) -> bool:
    deep_state = DeepState(ctx.data)
    return ctx.config.start_at != "fix" and not deep_state.single_stack_mode


def _single_stack_merge_enabled(ctx: FlowContext) -> bool:
    deep_state = DeepState(ctx.data)
    return ctx.config.start_at != "fix" and deep_state.single_stack_mode


def _diagram_enabled(ctx: FlowContext) -> bool:
    """Whether the grounded-diagram step runs (issue #1113).

    Off on a ``--start-at fix`` resume (the diff-derived signals and the report
    the blocks land in both belong to the earlier run) and off when the
    resolved mode is ``"off"``. Note this is a WEAKER gate than "some kind is
    eligible": when the step runs it always records its eligibility decision in
    ``diagram.json``, which is what makes "why did this PR get no diagram?"
    answerable. Nothing is eligible costs zero agent calls.

    The same ``FlowStep`` object backs the ``diagram`` flow, where this must
    return True: ``start_at`` defaults to ``"review"`` there (the CLI rejects
    ``--start-at`` with ``--diagram-only``) and diagram mode's resolved mode is
    never ``"off"`` (``off`` is not an accepted ``--diagram-only`` value).
    """
    return _before_fix_resume(ctx) and _resolved_diagram_mode(ctx) != "off"


def _findings_out_enabled(ctx: FlowContext) -> bool:
    # Two-phase findings artifact (Phase A): emit the artifact and STOP —
    # never post to the PR and never apply fixes. Phase B posts later.
    return ctx.config.findings_out is not None


def _resolve_mode(config: RunConfig) -> str:
    """Map a RunConfig onto the single deep-flow mode key (#330).

    ``review`` / ``comment`` replace the review flow (stop after post-review);
    ``shallow`` replaces the shallow flow (single-stack deep); ``diagram``
    (issue #1113) reuses the spine's preamble but runs the two-step ``diagram``
    flow. ``loop`` is the unchanged default.
    """
    if config.flow_name in ("review", "shallow"):
        return config.flow_name
    # Issue #1113: checked before ``shallow`` so ``--diagram-only`` is never
    # reinterpreted as a shallow review by an unrelated flag combination.
    if config.output_mode == "diagram":
        return "diagram"
    if config.output_mode == "review":
        return "review"
    if config.output_mode == "comment":
        return "comment"
    if config.shallow:
        return "shallow"
    return "loop"


def _mode_of(ctx: FlowContext) -> str:
    """The active mode, set by ``run_deep``'s dispatch preamble."""
    deep_state = DeepState(ctx.data)
    return str(deep_state.mode)


def _fix_cycle_enabled(ctx: FlowContext) -> bool:
    """The apply-fix gate + verify/fix/test/commit run in loop + shallow modes."""
    return _mode_of(ctx) in ("loop", "shallow")


def _cleanup_applies(ctx: FlowContext) -> bool:
    """Return whether this mode writes ``.review-output.md``."""
    return _mode_of(ctx) in ("loop", "shallow", "review", "comment")


def _cleanup_should_run(ctx: FlowContext, exit_code: int) -> bool:
    """Whether terminal cleanup runs after a deep flow — success-path only.

    A zero exit is a successful completion (an early ``Stop(0)``, e.g. a
    declined fix gate, still honors ``--cleanup``); any non-zero (failure)
    exit skips cleanup so evidence survives (#335). ``--findings-out`` runs
    stop with exit 0 but must keep the rendered report the run was asked to
    produce, so they are excluded (``findings_out is None``).
    """
    return exit_code == 0 and _cleanup_applies(ctx) and ctx.config.findings_out is None


def _flow_kind_for_mode(mode: str) -> DaydreamRunFlow:
    """Recorder run-flow label per mode (preserves the pre-collapse mapping).

    ``diagram`` gets its own label rather than borrowing ``TTT`` (issue #1113):
    ``archive._flow_runs_merge`` returns True for TTT, so a diagram run would
    otherwise inherit a previous deep review's ``merged-items.json`` as its own
    pipeline state -- and diagram runs deliberately leave those artifacts on
    disk.
    """
    if mode == "shallow":
        return DaydreamRunFlow.NORMAL
    if mode == "diagram":
        return DaydreamRunFlow.DIAGRAM
    if mode in ("review", "comment"):
        return DaydreamRunFlow.TTT
    return DaydreamRunFlow.DEEP


def _flow_name_for_mode(mode: str) -> str:
    """The registered flow name a mode runs (issue #1113).

    Every PR-process mode runs the ``deep`` flow; only ``diagram`` has its own.
    ``loop`` / ``comment`` / ``review`` / ``shallow`` are modes, not registered
    flow names, so the raw mode string must never be passed to ``run_flow``.
    """
    return "diagram" if mode == "diagram" else "deep"


# The deep pipeline as a registered flow (D-07):
#
#     exploration pre-scan -> TTT intent -> TTT alternative-review ->
#     per-stack reviews -> per-stack parse + dedup -> uncovered-file sweep (#309)
#     -> arbiter -> cross-stack merge (or the tiny-diff single-stack bypass) ->
#     supervise -> findings-out stop / post-review -> fix gate -> verify -> fix ->
#     test -> commit -> exact-SHA remote CI.
#
# ``register_builtins`` registers :data:`STEPS` and the ``deep`` flow
# definition; ``run_deep`` keeps the preamble and delegates here via
# ``run_flow``. The old imperative body's tier / single_stack_mode /
# ``start_at`` / ``findings_out`` conditions are the ``enabled`` predicates
# above (whole-block gates) or stay inside step bodies (resume branches). The
# mode gates replace the review/comment/shallow flows (#330): review/comment
# modes stop after ``post-review`` (the fix cycle is gated off).
#
# Terminal cleanup (#330) is NOT a step: it is a success-path helper invoked by
# ``_run_review_spine`` after ``run_flow`` returns. Tying it to the run's exit
# code (rather than the end of this tuple) means an early successful ``Stop(0)``
# -- the fix gate declining -- still honors ``--cleanup``, while any non-zero
# (failure) exit skips it to keep evidence (#335).
STEPS: tuple[FlowStep, ...] = (
    FlowStep(name="exploration", run=_step_exploration),
    FlowStep(name="intent", run=_step_intent, enabled=_fresh_ttt),
    FlowStep(
        name="per-stack-reviews",
        run=_step_wonder_and_per_stack,
        config_phase="per_stack_review",
    ),
    FlowStep(name="per-stack-parse", run=_step_per_stack_parse, config_phase="parse", enabled=_before_fix_resume),
    FlowStep(name="uncovered-sweep", run=_step_uncovered_sweep, enabled=_uncovered_sweep_enabled, config_phase="parse"),
    FlowStep(name="arbiter", run=_step_arbiter, enabled=_multi_stack_merge_enabled),
    FlowStep(
        name="cross-stack-merge", run=_step_cross_stack_merge, config_phase="merge", enabled=_multi_stack_merge_enabled
    ),
    FlowStep(name="single-stack-merge", run=_step_single_stack_merge, enabled=_single_stack_merge_enabled),
    FlowStep(name="load-items", run=_step_load_items),
    FlowStep(name="supervise", run=_step_supervise, config_phase="supervise", enabled=_supervise_enabled),
    FlowStep(name="diagram", run=_step_diagram, config_phase="diagram", enabled=_diagram_enabled),
    FlowStep(name="findings-out", run=_step_findings_out, enabled=_findings_out_enabled),
    FlowStep(name="post-review", run=_step_post_review, enabled=_before_fix_resume),
    # Fix cycle: loop + shallow modes only (review/comment stop after post-review).
    FlowStep(name="fix-gate", run=_step_fix_gate, enabled=_fix_cycle_enabled),
    FlowStep(name="verify", run=_step_verify, enabled=_fix_cycle_enabled),
    FlowStep(name="fix", run=_step_fix, enabled=_fix_cycle_enabled),
    FlowStep(name="fix-verify", run=_step_fix_verify, enabled=_fix_cycle_enabled),
    FlowStep(name="test", run=_step_test, enabled=_fix_cycle_enabled),
    # config_phase "fix" mirrors the old body's use of the fix backend for the commit.
    FlowStep(name="commit", run=_step_commit, config_phase="fix", enabled=_fix_cycle_enabled),
    FlowStep(name="remote-ci", run=_step_remote_ci, enabled=_remote_ci_enabled),
)


# The diagram-only flow's own steps (issue #1113). Deliberately NOT appended to
# :data:`STEPS`: ``builtins._register_builtin_flows`` derives the ``deep`` flow
# definition FROM ``STEPS``, so appending ``post-diagram`` there would splice a
# GitHub write into every deep review. The ``diagram`` flow is
# ``exploration -> diagram -> post-diagram``; the first two steps are the same
# objects the deep flow registers.
DIAGRAM_STEPS: tuple[FlowStep, ...] = (
    FlowStep(name="post-diagram", run=_step_post_diagram),
)


@bind_resolved_run_context
async def run_deep(
    config: RunConfig,
    work: WorkContext,
    *,
    run_artifacts: _RunArtifacts | None = None,
    run_context: RunContext | None = None,
    github_execution: GitHubExecutionInput | None = None,
    backend_factory: BackendFactory | None = None,
    allow_standalone: bool = False,
) -> int:
    """Execute the deep-review pipeline (D-07) across every PR-process mode.

    The single ``deep`` flow (#330) handles ``review`` and ``comment`` through
    the review spine, ``shallow`` through single-stack mode, and the unchanged
    default ``loop`` mode.

    Runs the preamble (diff computation, stack detection, tiny-diff collapse,
    trajectory recorder, pre-flight notice) and delegates the pipeline to the
    registered ``deep`` flow (:data:`STEPS`) via ``run_flow``. Supports
    stage-granular resume via
    ``config.start_at in ("ttt", "per-stack", "merge", "fix")``.

    Args:
        config: Run configuration; ``config.shallow`` / ``config.output_mode``
            select the mode. ``config.identity`` carries the GitHub identity
            set by :func:`daydream.runner.run`.
        work: Resolved working environment for the run.
        run_artifacts: The composition root's artifact session and its
            pre-registered output routes, or ``None`` for a standalone caller.
        allow_standalone: Intentional direct callers without ``run_artifacts``
            must pass ``True`` and have no active artifact session.
            Runner-managed calls always keep this false.

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    run_context = resolve_run_context(run_context)
    if run_artifacts is None and not allow_standalone:
        raise ArtifactVisibilityError("standalone deep flow requires allow_standalone=True")
    if run_artifacts is None and artifact_session_active():
        raise ArtifactVisibilityError("standalone deep flow cannot run inside an active artifact session")
    execution = github_execution or GitHubExecutionInput()
    return await _run_review_spine(
        config,
        work,
        _resolve_mode(config),
        run_artifacts=run_artifacts,
        run_context=run_context,
        github_execution=execution,
        backend_factory=backend_factory,
        allow_standalone=allow_standalone,
    )


def _collapse_stacks_for_shallow(
    stacks: list[StackAssignment],
    changed_files: list[str],
    config: RunConfig,
) -> tuple[list[StackAssignment], bool]:
    """Force the single-stack assignment for shallow mode (#330).

    Collapses every non-structural stack into one combined assignment and keeps
    the structural meta-stack separate, so structural findings stay correctly
    tagged ``lens="structural"`` downstream. Returns ``(stacks, True)``.

    The combined assignment's stack, in precedence order:

    - an explicit ``--stack`` (CLI) wins — the combined stack is named by it;
    - otherwise a *sole* detected non-structural stack preserves its name,
      absorbing any generic/docs files — so ``daydream --shallow <repo>``
      without ``--stack`` uses the language reviewer instead of the native
      generic fallback (#6);
    - otherwise (multiple real-language stacks — one agent cannot review two
      per-language scopes — or no real language at all) the combined assignment
      uses the native generic-fallback scope.

    """
    structural = [s for s in stacks if s.stack_name == STRUCTURE_STACK_NAME]
    combined_files = sorted({f for s in stacks for f in s.files}) or changed_files

    non_structural = [s for s in stacks if s.stack_name != STRUCTURE_STACK_NAME]
    real_language = [s for s in non_structural if s.stack_name != GENERIC_STACK]

    if config.stack is not None:
        combined = StackAssignment(
            stack_name=config.stack,
            files=combined_files,
            is_docs_only=False,
        )
    elif len(real_language) == 1:
        # Scope preservation: a sole real-language stack survives unchanged,
        # absorbing any generic/docs files.
        lang = real_language[0]
        combined = StackAssignment(
            stack_name=lang.stack_name,
            files=combined_files,
            is_docs_only=False,
        )
    else:
        # Multiple real-language stacks (one agent cannot cover two per-language
        # scopes) or no real language at all: the combined assignment uses the
        # native generic-fallback scope.
        combined = StackAssignment(
            stack_name=GENERIC_STACK,
            files=combined_files,
            is_docs_only=False,
        )
    return [*structural, combined], True


@bind_resolved_run_context
async def _run_review_spine(
    config: RunConfig,
    work: WorkContext,
    mode: str,
    *,
    run_artifacts: _RunArtifacts | None,
    run_context: RunContext | None = None,
    github_execution: GitHubExecutionInput,
    backend_factory: BackendFactory | None = None,
    allow_standalone: bool = False,
) -> int:
    """Review-spine preamble for the deep pipeline (the former ``run_deep`` body)."""
    artifact_session = None if run_artifacts is None else run_artifacts.session
    run_context = resolve_run_context(run_context)
    # Late imports to avoid circular dependency with runner.
    from daydream import git_ops
    from daydream.backends import Backend
    from daydream.git_ops import GitError, GitTimeoutError
    from daydream.hunk_index import write_hunk_index
    from daydream.phases import _git_branch, _git_log
    from daydream.runner import _default_backend_name, _open_recorder, _resolve_review_profile

    # Cache one Backend instance per (backend_name, resolved_model, resolved_effort)
    # so phases that resolve to the same model/effort share an instance and
    # differing ones stay isolated.
    backend_cache: dict[
        tuple[str, str | None, str | None, Path | None], Backend
    ] = {}

    target_dir = work.repo

    # Preamble (mirrors runner._run_loop_shallow).
    try:
        diff = git_ops.diff(work.repo, work.base_branch, exclude=config.ignore_paths)
    except GitTimeoutError as exc:
        # Transient host-load timeout that survived git_ops' bounded retries.
        # Report it accurately instead of the misleading "Unable to determine
        # base branch" message a genuine ref error would produce (issue #120).
        print_error(console, "Git Timeout", f"git timed out under load: {exc}")
        return 1
    except GitError:
        diff = None
    log = _git_log(target_dir)
    branch = work.head_branch or _git_branch(target_dir)

    if diff is None:
        print_error(console, "Git Error", "Unable to determine base branch for diff")
        return 1
    if not diff.strip():
        subject = "diagram" if mode == "diagram" else "review"
        print_warning(console, f"No diff found -- nothing to {subject}")
        return 0

    daydream_dir = artifact_dir_for(
        target_dir,
        session=artifact_session,
        allow_standalone=allow_standalone,
    )
    daydream_dir.mkdir(exist_ok=True)
    diff_path = daydream_dir / "diff.patch"
    diff_path.write_text(diff)
    # Persist the hunk index immediately after the diff bytes, so the run-time
    # authority (changed file/line ranges) is available to every later step
    # (reviews, arbiter, merge, uncovered sweep) and never predates the patch.
    write_hunk_index(daydream_dir, diff)
    # Diff is immutable from here on; compute the tiering verdict once and reuse
    # it at both the exploration step's gate and the alternatives step's gate.
    tier = review_steps.select_tier(review_steps.count_changed_files(diff))
    dd = deep_dir(
        target_dir,
        session=artifact_session,
        allow_standalone=allow_standalone,
    )
    current_diff_sha = diff_key(diff)
    # Issue #1113: a diagram-only run must NEVER clear ``.daydream/deep/``. It
    # produces none of the artifacts ``diff-key`` attests, and wiping the
    # directory would destroy a previous deep review's intent, alternatives,
    # per-stack records and merged items -- breaking any later ``--start-at
    # merge``/``fix``. Consequence, accepted: diagram mode writes no
    # ``diff-key``, which is correct for a run that produces nothing it could
    # attest. ``deep_dir`` already created ``dd``, so it stays a valid path.
    if mode != "diagram" and config.start_at not in ("per-stack", "merge", "fix"):
        # Fresh run only: a resume must NOT rewrite the key it is checked
        # against, or the staleness gate would self-heal and pass every time.
        shutil.rmtree(dd, ignore_errors=True)
        dd.mkdir(parents=True, exist_ok=True)
        diff_key_path(dd).write_text(current_diff_sha, encoding="utf-8")

    async with _open_recorder(
        config=config, target_dir=target_dir, work=work, flow_kind=_flow_kind_for_mode(mode),
        run_artifacts=run_artifacts,
        allow_standalone=allow_standalone,
    ):
        # Composition-root re-entry: resolution already happened in
        # ``_run_loop_deep`` before this recorder existed; this no-op resolve
        # records the profile onto the active recorder (R12).
        _resolve_review_profile(config)
        console.print()
        print_info(console, f"Target directory: {target_dir}")
        print_info(console, f"Branch: {branch}")
        print_info(console, f"Default backend: {_default_backend_name(config)}")
        # Bot logins look like ``my-app[bot]``; escape so Rich doesn't eat the brackets.
        print_info(console, f"GitHub identity: {escape_markup(config.identity)}")
        console.print()

        # Resume gate (D-34, D-36, D-37) + diff-freshness gate.
        if config.start_at in ("per-stack", "merge", "fix"):
            try:
                check_deep_artifacts(config.start_at, dd, current_diff_sha=current_diff_sha)
                if _has_non_daydream_worktree_changes(git_ops.status_porcelain(target_dir)):
                    raise FileNotFoundError(
                        f"Cannot resume at stage '{config.start_at}' -- the worktree has changed "
                        "since the review artifacts were generated.\n\n"
                        "Resuming would review stale findings against changed code.\n"
                        "Re-run without --start-at to regenerate them."
                    )
            except FileNotFoundError as exc:
                print_error(console, "Unusable Deep Artifacts", str(exc))
                return 1
            # Issue #309: a per-stack resume re-runs the sweep, so the prior
            # run's sweep artifacts are about to be superseded. Clear them now
            # (before new per-stack work) so a rerun whose sweep is disabled,
            # finds nothing, or produces no output cannot leave stale records
            # that a later merge resume would reload. Merge/fix resumes keep
            # them (the sweep is a no-op there and the records must survive).
            # Fail-CLOSED: an artifact that cannot be removed stops the resume
            # (stale records reloaded as current findings would be worse than
            # no resume); this call is at the resume boundary, OUTSIDE the
            # sweep step's fail-open wrapper, so the raise cannot be swallowed.
            if config.start_at == "per-stack":
                try:
                    _clear_sweep_artifacts(dd)
                except OSError as exc:
                    print_error(
                        console, "Unusable Deep Artifacts", f"{exc}\n\nRe-run without --start-at to regenerate them."
                    )
                    return 1

        # Stack detection (from diff file list). Built-in detection is
        # registry-independent (M1); fork stack rules still resolve via the
        # registry inside detect_stacks.
        changed_files = _diff_changed_files(diff)
        stacks = detect_stacks(changed_files)
        # Structural gating (M8): ``detect_stacks`` still emits the structural
        # meta-stack; a profile that disables ``structural_enabled`` removes only
        # that assignment/call here, before collapse/sharding publish the list.
        # This pre-context call reads the same resolved pipeline that
        # ``FlowContext.pipeline()`` resolves in-flow.
        if not _config_pipeline(config).structural_enabled:
            stacks = [s for s in stacks if s.stack_name != STRUCTURE_STACK_NAME]
        # Issue #172 — tiny-diff short-circuit. When the diff is small enough
        # (≤ SHALLOW_FANOUT_THRESHOLD files), collapse the per-language fan-out
        # to a single combined assignment and skip merge+arbiter downstream.
        # ``single_stack_mode`` is recomputed here (top of run_deep) so a
        # ``--start-at merge``/``--start-at fix`` resume on a tiny diff re-enters
        # the same bypass branch rather than routing to the absent merge agent.
        stacks, single_stack_mode = _collapse_stacks_for_tiny_diff(
            stacks, changed_files, threshold=_shallow_fanout_threshold(config)
        )
        # Issue #330 — ``--shallow`` forces the single-stack assignment regardless
        # of diff size, so no arbiter / cross-stack merge runs.
        if mode == "shallow":
            stacks, single_stack_mode = _collapse_stacks_for_shallow(stacks, changed_files, config)

        # Issue #731: deep-review sharding. Runs AFTER the tiny-diff/shallow
        # collapse passes (which must stay byte-identical) and BEFORE
        # ``ctx.data["stacks"]`` is published below. Skipped whenever
        # ``single_stack_mode`` is True (tiny-diff or shallow collapse already
        # folded everything into one stack) and off by default (forensic mode
        # passes the stack list through untouched). ``build_import_graph`` is
        # fail-open (never raises; returns ``{}`` on any failure); byte sizing
        # uses the FULL on-disk ``diff``, not the bounded in-memory value.
        import_graph: dict[str, set[str]] = {}
        sharding_enabled = _deep_shard_enabled(config)
        if sharding_enabled and not single_stack_mode:
            try:
                import_graph = build_import_graph(changed_files, target_dir)
            except Exception:
                import_graph = {}
            stacks = shard_stacks(
                stacks,
                diff,
                max_files=_deep_shard_max_files(config),
                max_bytes=_deep_shard_max_bytes(config),
                fanout_cap=_deep_shard_fanout_cap(config),
                frontier_max=_deep_shard_frontier_max(config),
                graph=import_graph,
            )

        # Issue #1113: the sequence diagram's cross-module rule needs the
        # changed-file import graph, but the sharding branch above builds it
        # only when sharding is enabled (off by default) AND the run is not in
        # single_stack_mode -- so in practice essentially never. Build it here
        # when the diagram step can run, and publish it on ctx.data. The bare
        # ``except Exception`` is required, not defensive: ``build_import_graph``
        # documents itself as never raising, but its ``get_parser`` call reaches
        # ``assert_tree_sitter_safe()``, which raises ``TreeSitterBadVersionError``.
        if (
            not import_graph
            and config.start_at != "fix"
            and _diagram_mode_for(config, mode) != "off"
        ):
            try:
                import_graph = build_import_graph(changed_files, target_dir)
            except Exception:
                import_graph = {}

        # Pre-flight notice (D-30). Agent count reflects the tiny-diff collapse
        # when single_stack_mode is active (issue #172): merge+arbiter are
        # skipped, so the estimate uses ``_single_stack_agent_count``.
        stack_lines = [
            _stack_preflight_line(stack)
            for stack in stacks
            if stack.stack_name != STRUCTURE_STACK_NAME
        ]
        notice_agent_count = (
            _single_stack_agent_count(len(stacks))
            if single_stack_mode
            else total_agent_count(len(stacks))
        )
        # Issue #1113: the notice hardcodes "Deep-review pipeline pre-flight",
        # the five deep pipeline stages and a 2+2N+2 agent estimate. A two-step
        # diagram flow executes none of that, so printing it would be a lie
        # about what the run is doing.
        if mode != "diagram":
            print_preflight_notice(
                console,
                stages=_preflight_stage_names(stacks),
                stack_lines=stack_lines,
                agent_count=notice_agent_count,
                exploration_available=review_steps.EXPLORATION_AVAILABLE,
                sweep_note=_uncovered_sweep_preflight_note(config, changed_files),
            )

        # Flow context (steps communicate through ctx.data); ctx shares
        # run_deep's backend cache so instance-sharing semantics are unchanged.
        # Issue #644 — the in-memory diff is bounded at gather time to
        # ``INLINE_DIFF_BUDGET_BYTES`` via whole-block retention (``diff.patch``
        # above stays FULL on disk for the archival/coverage/eval/training
        # consumers; tiering / ``diff_key`` / ``changed_files`` above already
        # ran on the full ``diff``; the exploration pre-scan and the uncovered
        # sweep read the FULL on-disk patch at step time, never this bounded
        # value). ``bound_deep_diff`` is infallible and runs
        # after the full diff is persisted, so the disk copy is never bounded.
        bounded_diff, bound_info = bound_deep_diff(diff)
        if bound_info.truncated:
            dropped = (
                f"; dropped blocks: {', '.join(bound_info.dropped_paths)}"
                if bound_info.dropped_paths
                else ""
            )
            oversize = (
                f"; oversized block kept whole: {', '.join(bound_info.oversize_paths)}"
                if bound_info.oversize_paths
                else ""
            )
            print_warning(
                console,
                f"Deep diff truncated: {bound_info.original_bytes} -> "
                f"{bound_info.retained_bytes} bytes "
                f"({bound_info.retained_blocks}/{bound_info.total_blocks} blocks retained"
                f"{dropped}{oversize})",
            )
        ctx = FlowContext(
            config=config,
            work=work,
            registry=get_registry(),
            review_profile=config.review_profile,
            private_workspace_owner=None if run_artifacts is None else run_artifacts.owner,
            artifacts=None if run_artifacts is None else run_artifacts.session,
            run_context=run_context,
            github_execution=github_execution,
            _backend_factory=backend_factory,
            data={
                "mode": mode,
                "diff": bounded_diff,
                # Issue #336 — fix-loop scope bound. The reviewed diff's file
                # set threads through ctx.data so the fix gate can partition
                # out-of-scope findings (Task 3) and the fix step can both
                # forward it into the fix prompt (Task 2) and run a post-fix
                # residual check (Task 4). Recomputable from "diff" via
                # ``_diff_changed_files``; a missing key never crashes.
                "changed_files": set(changed_files),
                "diff_path": diff_path,
                "diff_truncated": bound_info.truncated,
                "diff_truncation": bound_info,
                "tier": tier,
                "dd": dd,
                "stacks": stacks,
                # Issue #1113: the changed-file import graph, published so the
                # diagram step's cross-module rule can read it. ``{}`` simply
                # denies that rule; it never fails the run.
                "import_graph": import_graph,
                "single_stack_mode": single_stack_mode,
                "intent_path": _intent_path(dd),
                "alts_path": _alternatives_path(dd),
                "log": log,
                "branch": branch,
                "failed_stacks": {},
            },
            _backend_cache=backend_cache,
            allow_standalone_artifacts=allow_standalone,
        )

        # Nothing is torn down after the flow. .daydream/exploration/ is a
        # content-keyed cache (see ``exploration_cache_key``) the next run reuses
        # on an exact head+diff+tier match and rewrites on a miss, and
        # .daydream/deep/ is preserved per RESEARCH.md Open Question 1 so
        # subsequent --start-at resumes can find the artifacts they need.
        #
        # Cleanup is success-path only (#335); a non-zero exit returns before the guard so evidence survives.
        exit_code = await run_flow(ctx.registry, _flow_name_for_mode(mode), ctx)
        if _cleanup_should_run(ctx, exit_code):
            await _perform_cleanup(ctx)
        return exit_code
