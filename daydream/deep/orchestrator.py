"""Deep-review mode orchestrator (``run_deep``).

Composes existing phase primitives plus deep-mode-specific phases into the
pipeline described by D-07:

    exploration pre-scan -> TTT intent -> TTT alternative-review ->
    per-stack reviews -> per-stack parse + dedup -> cross-stack merge ->
    optional fix gate.

All per-plan logic lives in plans 05-01..05-08; this module is the wiring
layer that stitches them together. No signature changes to any existing
phase primitive (D-39).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable

import anyio
from rich.markup import escape as escape_markup

from daydream.agent import console, get_assume, get_non_interactive, resolve_or_prompt, run_agent
from daydream.backends import effective_fanout_concurrency
from daydream.config import (
    DEFAULT_GROUP_MAX_SERIAL_ITEMS,
    DEFAULT_GROUP_MAX_WALL_S,
    DEFAULT_QUALITY_GATE_ENABLED,
    DEFAULT_QUALITY_GATE_EROSION_ABSOLUTE,
    DEFAULT_QUALITY_GATE_EROSION_DELTA,
    DEFAULT_QUALITY_GATE_VERBOSITY_ABSOLUTE,
    DEFAULT_QUALITY_GATE_VERBOSITY_DELTA,
    DEFAULT_TOOL_CALL_BUDGET,
    DEFAULT_UNCOVERED_SWEEP_ENABLED,
    DEFAULT_UNCOVERED_SWEEP_MAX_FILES,
    DEFAULT_UNCOVERED_SWEEP_MIN_HUNK_LINES,
    DEFAULT_WALL_BUDGET_S,
    REVIEW_OUTPUT_FILE,
    STRUCTURE_STACK_NAME,
)
from daydream.config_file import _coerce_non_negative_int, _coerce_quality_threshold
from daydream.deep.arbiter import select_arbiter_targets, select_suppression_targets
from daydream.deep.artifacts import (
    adjudication_complete_path,
    check_deep_artifacts,
    dedup_candidates_path,
    deep_dir,
    diff_key,
    diff_key_path,
    fix_failures_path,
    fix_leftover_untracked_path,
    fix_quality_gate_path,
    generated_file_violations_path,
    merged_items_path,
    merged_report_path,
    per_stack_failures_path,
    per_stack_records_path,
    test_verdict_path,
)
from daydream.deep.artifacts import (
    alternatives_path as _alternatives_path,
)
from daydream.deep.artifacts import (
    intent_path as _intent_path,
)
from daydream.deep.coverage import (
    build_uncovered_sweep_prompt,
    compute_uncovered_files,
    diff_block_for_file,
    filter_sweepable_files,
)
from daydream.deep.dedup import (
    CandidatePair,
    RecordDuplicatePair,
    build_dedup_candidates,
    build_record_dedup_candidates,
)
from daydream.deep.detection import GENERIC_STACK, StackAssignment, detect_stacks
from daydream.deep.render import render_held_section, render_report
from daydream.deep.scope_issues import (
    _file_out_of_scope_issue,
    _resolve_changed_files,
    _revert_out_of_scope_edits,
)
from daydream.extensions import get_registry
from daydream.extensions.api import FlowStep, Stop
from daydream.flows.engine import FlowContext, run_flow
from daydream.generated_files import (
    _changed_untracked_generated_files,
    _restore_untracked_generated_file,
    _snapshot_untracked_generated_files,
    is_generated_file,
    related_manifest_paths,
)
from daydream.phases import (
    PER_STACK_RECORD_SCHEMA,
    CrossStackMergeError,
    FixResult,
    _resolve_finding_file_ref,
    _write_single_stack_merged_items,
    phase_alternative_review,
    phase_arbiter_review,
    phase_commit_push,
    phase_commit_push_auto,
    phase_cross_stack_merge,
    phase_fetch_pr_feedback,
    phase_fix,
    phase_fix_parallel,
    phase_parse_feedback,
    phase_per_stack_reviews,
    phase_respond_pr_feedback,
    phase_supervise_review,
    phase_suppression_review,
    phase_test_and_heal,
    phase_understand_intent,
    phase_verify_recommendations,
    severity_sorted,
)
from daydream.supervision import (
    RuleBasedSupervisor,
    apply_findings_verdicts,
    revise_finding_fields,
)
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    get_current_recorder,
    maybe_fork,
    phase_scope,
)
from daydream.ui import (
    format_verdict_join,
    phase_subtitle,
    print_dim,
    print_error,
    print_info,
    print_phase_hero,
    print_preflight_notice,
    print_stage_progress,
    print_success,
    print_verification_summary,
    print_warning,
    render_exploration_summary,
)
from daydream.workspace import WorkContext

if TYPE_CHECKING:
    from daydream.runner import RunConfig

# Exploration infrastructure import guard. When Phases 1-4 are not yet
# installed, deep mode still runs -- just without grounding context.
try:
    from daydream.exploration import ExplorationContext, safe_explore
    from daydream.exploration_runner import count_changed_files, pre_scan, select_tier

    EXPLORATION_AVAILABLE = True
except ImportError:  # pragma: no cover -- only hit when Phases 1-4 absent
    EXPLORATION_AVAILABLE = False

# Per-file diff block splitter + the shared per-block path resolver live in
# ``daydream.deep.prompts`` (canonical home of the diff-text -> prompt
# primitives). Imported here because ``_diff_changed_files`` shares them with
# ``prompts._diff_blocks_for_files`` (issue #172 Fix B).
from daydream.deep.prompts import (
    _DIFF_BLOCK_SPLIT,
    _diff_block_path,
)

# User-visible pipeline stages (exploration is a pre-stage banner, not counted).
_PIPELINE_STAGE_NAMES: list[str] = [
    "TTT intent",
    "TTT alternative-review",
    "per-stack reviews",
    "cross-stack merge",
    "optional fix gate",
]


def total_agent_count(stack_count: int) -> int:
    """Return the D-30 agent count formula.

    Formula: 2 (TTT intent + alternative-review) + N per-stack reviews
    + N per-stack parse passes + 1 cross-stack merge + 1 conditional
    arbiter (Opus pass over high-severity / contested findings). The
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


def _resolve_config_value[T: (int, float, bool)](config: RunConfig, attr: str, default: T) -> T:
    """Resolve a scalar setting: ``RunConfig`` attr (when present) > file config > default.

    Precedence mirrors ``_resolve_backend`` / ``_resolved_model`` at
    ``runner.py:295-326``. Uses ``is not None`` checks rather than truthiness
    so a configured value of ``0`` / ``0.0`` is honored.
    """
    value = getattr(config, attr, None)
    if value is not None:
        return value
    file_config = config.file_config
    if file_config is not None:
        value = getattr(file_config, attr, None)
        if value is not None:
            return value
    return default


def _shallow_fanout_threshold(config: RunConfig) -> int:
    """Resolve the tiny-diff short-circuit threshold (issue #172, AC7).

    ``0`` disables the short-circuit.
    """
    return _resolve_config_value(config, "shallow_fanout_threshold", DEFAULT_SHALLOW_FANOUT_THRESHOLD)


def _precision_mode(config: RunConfig) -> bool:
    """Resolve the precision-mode opt-in (issue #232).

    Precedence (highest first), mirroring ``_shallow_fanout_threshold`` above
    and ``_resolve_backend`` / ``_resolved_model`` at ``runner.py:295-326``:

      1. ``RunConfig.precision_mode`` (CLI tier / direct construction).
      2. ``DaydreamFileConfig.precision_mode`` (file-config scalar).
      3. Built-in default ``False`` (byte-identical behavior: the suppression
         predicate is never called and arbiter output is unchanged).

    Uses truthiness rather than ``is not None``: ``False`` is the meaningful
    "off" value, so a set-to-False file-config entry just falls through to the
    default rather than acting as a distinct sentinel.
    """
    if config.precision_mode:
        return True
    file_config = config.file_config
    if file_config is not None and file_config.precision_mode:
        return True
    return False


def _approve_on_clean(config: RunConfig) -> bool:
    """Resolve the approve-on-clean opt-in (issue #343).

    Precedence mirrors ``_precision_mode``: 1) ``RunConfig.approve_on_clean``
    (CLI tier), 2) ``DaydreamFileConfig.approve_on_clean`` (file-config
    scalar), 3) built-in default ``False`` (byte-identical behavior: the
    event stays COMMENT unless a repo explicitly opts in).
    """
    if config.approve_on_clean:
        return True
    file_config = config.file_config
    if file_config is not None and file_config.approve_on_clean:
        return True
    return False


def _supervisor_mode(config: RunConfig) -> str:
    """Resolve the file-config-only findings supervisor mode."""
    file_config = config.file_config
    mode = file_config.supervisor if file_config is not None else None
    return mode if mode in {"off", "rules", "llm"} else "off"


def _supervise_enabled(ctx: FlowContext) -> bool:
    """Run supervision on fresh flows, not on a fix-only resume."""
    return _supervisor_mode(ctx.config) in {"rules", "llm"} and ctx.config.start_at != "fix"


def _uncovered_sweep_max_files(config: RunConfig) -> int:
    """Resolve the per-run uncovered-file sweep capacity cap (issue #309).

    Integer-only non-negative: an explicit ``0`` disables the sweep (nothing is
    swept) while a negative value, a float, a bool, or any non-int degrades to
    the named default -- the same integer-only predicate as the file-config
    coercion ``config_file._coerce_non_negative_int`` (reused, import read-only)
    so a directly-constructed ``RunConfig`` cannot smuggle an invalid capacity
    in. ``filter_sweepable_files`` slices with ``max_files``, so a float here
    would raise TypeError and the fail-open wrapper would discard the ENTIRE
    sweep -- type validation lives at the resolver.
    """
    value = _resolve_config_value(
        config, "uncovered_sweep_max_files", DEFAULT_UNCOVERED_SWEEP_MAX_FILES
    )
    coerced = _coerce_non_negative_int(value)
    return coerced if coerced is not None else DEFAULT_UNCOVERED_SWEEP_MAX_FILES


def _uncovered_sweep_min_hunk_lines(config: RunConfig) -> int:
    """Resolve the minimum hunk size for a file to be swept (issue #309).

    Integer-only non-negative: ``0`` removes the hunk-size floor (every
    uncovered file is eligible) while a negative value, a float, a bool, or any
    non-int degrades to the named default -- mirroring
    ``config_file._coerce_non_negative_int`` so a negative or malformed floor
    can never make zero-change/trivial blocks eligible.
    """
    value = _resolve_config_value(
        config, "uncovered_sweep_min_hunk_lines", DEFAULT_UNCOVERED_SWEEP_MIN_HUNK_LINES
    )
    coerced = _coerce_non_negative_int(value)
    return coerced if coerced is not None else DEFAULT_UNCOVERED_SWEEP_MIN_HUNK_LINES


def _uncovered_sweep_enabled(ctx: FlowContext) -> bool:
    """Resolve the uncovered-file sweep toggle (issue #309).

    Precedence mirrors ``_precision_mode``: ``RunConfig`` field > file-config
    scalar > built-in default (:data:`DEFAULT_UNCOVERED_SWEEP_ENABLED`, the
    single source of configuration defaults). Resume at ``merge``/``fix``
    disables the step outright -- the per-stack records are already finalized
    on disk, so a sweep would re-review stale coverage.
    """
    if ctx.config.start_at in ("merge", "fix"):
        return False
    return _resolve_config_value(ctx.config, "uncovered_sweep", DEFAULT_UNCOVERED_SWEEP_ENABLED)


def _uncovered_sweep_preflight_note(config: RunConfig, changed_files: list[str]) -> str | None:
    """Sweep additive for the pre-flight agent estimate (issue #309 finding 8).

    The pre-flight total counts only the known phases; the uncovered files the
    sweep will review are not known until after per-stack reviews + parse. Every
    swept file adds one review invocation AND one parse invocation (parse per
    stack file), so an honest estimate appends an upper-bound note: 2 agents per
    file, capped by the configured capacity and the number of changed files that
    could possibly be swept. Returns ``None`` when the sweep is disabled or
    nothing could be swept.
    """
    if config.start_at in ("merge", "fix"):
        return None
    if not _resolve_config_value(config, "uncovered_sweep", DEFAULT_UNCOVERED_SWEEP_ENABLED):
        return None
    max_files = _uncovered_sweep_max_files(config)
    eligible = min(len(changed_files), max_files)
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
        into the language stack so its per-language Beagle skill survives; only
        ≥2 *real* language stacks fall back to ``generic`` (a single agent
        cannot invoke two per-language Beagle skills).
      - The ``STRUCTURE_STACK_NAME`` meta-stack stays as its own assignment so
        structural findings remain correctly tagged ``lens="structural"``
        downstream (AC6).
      - If only one non-structural stack exists (the common 1-file case), it is
        preserved unchanged — the per-language skill survives.

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
    # assignment. The combined skill depends on how many *real* language stacks
    # are present:
    #   - exactly one real language stack + the generic bucket (a code+docs/config
    #     tiny diff, e.g. api.py + README.md): absorb the generic files into the
    #     language stack so its per-language Beagle skill survives (the
    #     skill-preservation goal stated in this docstring).
    #   - ≥2 real language stacks (e.g. python + react): a single agent cannot
    #     invoke two per-language Beagle skills, so fall back to generic.
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
                        skill_invocation=lang.skill_invocation,
                        files=combined_files,
                        is_docs_only=False,
                    ),
                ],
                True,
            )
        # ≥2 real-language stacks: one agent cannot invoke two per-language
        # Beagle skills, so the combined assignment uses the generic-fallback
        # skill (skill_invocation=None). is_docs_only is False by construction:
        # ≥2 non-structural stacks means at least one is a real language stack
        # (docs-only diff → single generic stack).
        combined = StackAssignment(
            stack_name=GENERIC_STACK,
            skill_invocation=None,
            files=combined_files,
            is_docs_only=False,
        )
        return [*structural, combined], True

    # 0 or 1 non-structural stacks: nothing to collapse (lever 1 is a no-op), but
    # the gate is still active so the caller applies lever 2 (skip merge+arbiter).
    return stacks, True


def get_installed_skills() -> set[str] | None:
    """Detect which Beagle review-skill plugins are installed.

    Reads the Claude Code plugin registry at
    ``$CLAUDE_CONFIG_DIR/plugins/installed_plugins.json`` (default
    ``~/.claude``) and maps installed plugin names back to stack keys via
    the extension registry's ``stack:<key>`` skill slots, so a remapped
    stack checks the remapped plugin prefix. A stack is considered
    "installed" iff its skill's plugin is present.

    Returns:
        Set of installed stack keys (subset of the registry's stack keys), or
        ``None`` if the registry cannot be read (missing file, bad JSON).
        ``None`` signals "unknown" so callers can fall back to optimistic
        availability without forcing every stack through generic.
    """
    config_dir = Path(os.environ.get("CLAUDE_CONFIG_DIR") or (Path.home() / ".claude"))
    registry = config_dir / "plugins" / "installed_plugins.json"
    try:
        data = json.loads(registry.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    # Structurally invalid payloads (non-dict root, non-dict `plugins`) also
    # signal "unknown" so callers fall back to optimistic availability
    # instead of aborting deep mode on an AttributeError / TypeError.
    if not isinstance(data, dict):
        return None
    plugins = data.get("plugins")
    if not isinstance(plugins, dict):
        return None
    # Keys in the registry look like "<plugin-name>@<marketplace>".
    installed_plugins = {key.split("@", 1)[0] for key in plugins}
    skill_registry = get_registry()
    installed: set[str] = set()
    for stack_key in skill_registry.stack_keys():
        # Slot values are "<plugin-name>:<skill-name>".
        plugin_prefix = skill_registry.skill(f"stack:{stack_key}").split(":", 1)[0]
        if plugin_prefix in installed_plugins:
            installed.add(stack_key)
    return installed


def _diff_changed_files(diff: str) -> list[str]:
    """Extract changed files from a unified diff.

    Parses one file per ``diff --git`` block and contributes a single path
    for each. Prefers the post-state path (``+++ b/<path>``) so renames
    produce only the destination. Falls back to the pre-state path for
    deletions (``+++ /dev/null``) and to the ``diff --git`` header for
    binary / mode-only diffs that lack ``---``/``+++`` lines.

    The per-block path resolution is delegated to ``_diff_block_path`` so the
    unified-diff parsing contract is shared with ``_diff_blocks_for_files``
    rather than duplicated here.

    Returns:
        Unique, insertion-ordered list of changed file paths (excluding
        ``/dev/null`` sentinels).
    """
    files: list[str] = []
    for block in _DIFF_BLOCK_SPLIT.split(diff):
        path = _diff_block_path(block)
        if path and path not in files:
            files.append(path)
    return files


def _stack_preflight_line(stack: StackAssignment) -> str:
    """Format one detected-stack line for the pre-flight notice."""
    skill = stack.skill_invocation or "generic fallback"
    docs_suffix = " (docs-only)" if stack.is_docs_only else ""
    return f"{stack.stack_name}: {skill} -- {len(stack.files)} file(s){docs_suffix}"


def _attach_verdicts(items: list[dict[str, Any]], payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach verifier verdicts to feedback items by matching `id` to `issue_id`.

    `phase_fix` reads the `verifier_verdict` / `evidence` / `unverified_assumptions`
    keys (advisory) and augments its prompt when present; items without a matching
    verdict are left untouched. Correctness rests on `normalize_items` having made
    item ids unique, so structural and per-stack findings can no longer collide on
    the same id.

    Args:
        items: Canonical feedback items, each with an integer `id`.
        payload: Verifier output; `payload["verdicts"]` is a list of entries each
            carrying `issue_id`, `verdict`, `evidence`, `unverified_assumptions`.

    Returns:
        The same `items` list (mutated in place) with verdict keys attached to any
        item whose `id` matched a verdict's `issue_id`.
    """
    payload = payload if isinstance(payload, dict) else {"verdicts": []}
    verdict_lookup: dict[int, dict[str, Any]] = {}
    for entry in payload.get("verdicts", []) or []:
        if not isinstance(entry, dict):
            continue
        issue_id = entry.get("issue_id")
        if not isinstance(issue_id, int):
            continue
        assumptions = entry.get("unverified_assumptions")
        verdict_lookup[issue_id] = {
            "verdict": entry.get("verdict", ""),
            "evidence": entry.get("evidence", ""),
            "unverified_assumptions": assumptions if isinstance(assumptions, list) else [],
        }
    for item in items:
        item_id = item.get("id")
        if not isinstance(item_id, int):
            continue
        match = verdict_lookup.get(item_id)
        if match is not None:
            item["verifier_verdict"] = match["verdict"]
            item["evidence"] = match["evidence"]
            item["unverified_assumptions"] = match["unverified_assumptions"]
    return items


def _candidate_pair_to_json(pair: CandidatePair | RecordDuplicatePair) -> dict[str, Any]:
    """Serialize a CandidatePair dataclass into a JSON-compatible dict."""
    data = asdict(pair)
    # alt_files is a tuple -> convert to list for stable JSON.
    if isinstance(data.get("alt_files"), tuple):
        data["alt_files"] = list(data["alt_files"])
    return data


def _apply_adjudication_verdicts(
    records: list[dict[str, Any]],
    sources: list[str],
    targets: list[int],
    verdicts: dict[int, dict[str, Any]],
    *,
    pass_name: str,
    id_field: str,
    fail_closed: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fold arbiter / suppression verdicts back into the per-stack record set.

    The scoped arbiter (``#168``) and the precision-mode suppression pass
    (``#232``) share an identical positional-rebuild shape; they differ ONLY in
    the fail polarity of two branches -- a missing verdict and an ``id_field``
    mismatch -- which is why this is one parameterised helper rather than two
    ~80-line near-clones (that duplication is what let the stale-index bug at the
    call site hide). Each selected record (``targets[k]`` for 1-based
    ``id = k + 1``) is either revised in place (severity/confidence/description/
    rationale/evidence taken from the verdict) or dropped. ``file``/``line`` are
    never changed -- adjudication revises, it does not re-target findings.

    Fail polarity (the sole axis on which the two passes diverge):

    - ``fail_closed=False`` (arbiter): a record reaches arbitration because it is
      high-severity or contested, so a missing verdict or an ``id_field`` mismatch
      must NOT delete it. The original record is retained unchanged with a warning
      (fail OPEN); only an explicit ``keep:false`` drops it.
    - ``fail_closed=True`` (suppression): a record reaches suppression precisely
      because it is borderline (neither high-severity nor contested), so an
      unconfirmable verdict -- missing, mismatched, or ``keep:false`` -- drops it
      (fail CLOSED), the inverse polarity, safe because nothing important reaches
      this pass.

    Non-selected records always pass through untouched.

    Args:
        records: Per-stack records positionally aligned with ``sources``.
        sources: Per-record originating stack name.
        targets: Indices into ``records`` selected for this pass; ``targets[k]``
            carries 1-based ``id = k + 1`` echoed back in the verdict's
            ``id_field``.
        verdicts: ``id -> verdict`` mapping from the adjudication agent.
        pass_name: Human-readable pass name (``"arbiter"`` / ``"suppression"``)
            used only in warning text.
        id_field: Verdict key carrying the echoed positional id (``"arb_id"`` for
            the arbiter, ``"sup_id"`` for suppression).
        fail_closed: Fail polarity for the missing-verdict / id-mismatch branches
            (see above).

    Returns:
        New ``(records, sources)`` with dropped records removed and surviving
        selected records carrying the verdict's fields. Positional alignment
        between the two lists is preserved.
    """
    import warnings

    polarity_action = (
        "dropping the unconfirmed record"
        if fail_closed
        else "retaining the original record unchanged"
    )
    dropped: set[int] = set()
    for offset, record_index in enumerate(targets):
        verdict_id = offset + 1
        verdict = verdicts.get(verdict_id)
        if verdict is None:
            # No verdict returned for this id -- fail per the caller's polarity.
            warnings.warn(
                f"{pass_name.capitalize()} returned no verdict for {id_field}={verdict_id} "
                f"(record_index={record_index}); {polarity_action}.",
                stacklevel=2,
            )
            if fail_closed:
                dropped.add(record_index)
            continue
        if verdict.get(id_field) != verdict_id:
            # Secondary key guard: the id field in the verdict must match the key
            # we looked it up by. A mismatch would silently bind the verdict to the
            # wrong record -- fail per the caller's polarity rather than mis-apply.
            warnings.warn(
                f"{pass_name.capitalize()} verdict {id_field} mismatch: "
                f"expected {id_field}={verdict_id} "
                f"but verdict contains {id_field}={verdict.get(id_field)!r} "
                f"(record_index={record_index}); {polarity_action}.",
                stacklevel=2,
            )
            if fail_closed:
                dropped.add(record_index)
            continue
        if not verdict.get("keep", False):
            dropped.add(record_index)
            continue
        # Revise IN PLACE so the record keeps its object identity across this
        # compaction. The suppression call site keys its arbiter-exclusion set by
        # ``id(record)`` (#232); a revised record that got a fresh dict here would
        # escape that set and be wrongly re-judged by the suppression pass.
        revise_finding_fields(records[record_index], verdict)

    new_records: list[dict[str, Any]] = []
    new_sources: list[str] = []
    for i, (record, source) in enumerate(zip(records, sources, strict=True)):
        if i in dropped:
            continue
        new_records.append(record)
        new_sources.append(source)
    return new_records, new_sources


def _rewrite_stack_records(
    deep_dir_path: Path,
    stack_record_paths: list[Path],
    records: list[dict[str, Any]],
    sources: list[str],
) -> None:
    """Persist arbiter-revised records back to each per-stack records file (#168).

    The cross-stack merge reads per-stack records by path, so arbitration must
    be reflected on disk, not just in memory. Every language stack file is
    rewritten with its surviving records (an emptied stack becomes ``[]`` rather
    than retaining stale pre-arbitration content).
    """
    by_stack: dict[Path, list[dict[str, Any]]] = {path: [] for path in stack_record_paths}
    for record, source in zip(records, sources, strict=True):
        # `source` may be a bare stack name ("python") on a fresh run, or a
        # filename ("stack-python-records.json") on resume.  Normalise to a
        # Path so both formats resolve to the same key already in `by_stack`.
        if source.endswith("-records.json"):
            dest = deep_dir_path / source
        else:
            dest = per_stack_records_path(deep_dir_path, source)
        if dest in by_stack:
            by_stack[dest].append(record)
    for dest_path, stack_records in by_stack.items():
        dest_path.write_text(json.dumps(stack_records, indent=2))


def _protect_tree_after_fix_failures(
    work: WorkContext,
    target_dir: Path,
    fix_failures: dict[str, str],
    *,
    snapshot: str | None,
    snapshot_captured: bool,
    pre_untracked: set[str],
) -> None:
    """Roll each failed fix group's file back to its pre-fix content.

    For every dropped file-group (keyed by repo-relative path), the partial-fix
    content is FIRST saved to ``.daydream/partial-fixes/<slug>.patch`` (a
    ``git diff`` against the pre-fix snapshot) so no agent work is destroyed,
    THEN the path is restored to exactly its pre-fix state. Only the failed
    paths are touched -- successful groups and unrelated paths are never
    reverted. A failed group's newly-created untracked file (absent from
    *pre_untracked*) has its raw content preserved and is then removed; untracked
    files we cannot attribute to the failed group are left in place.

    Args:
        work: The run's workspace (``work.repo`` is the git working dir).
        target_dir: Resolved target dir (``== work.repo``); root for the
            ``.daydream/partial-fixes`` recovery directory.
        fix_failures: ``{file_group: reason}`` for groups that failed.
        snapshot: ``git stash create`` SHA captured before fixes, or ``None``
            when the pre-fix tracked tree equalled ``HEAD``.
        pre_untracked: Untracked paths present before the fix pass.
    """
    from daydream import git_ops
    from daydream.git_ops import GitError

    if not snapshot_captured:
        # HEAD is not a safe substitute when capturing the pre-fix state
        # failed: it may discard edits that were present before this pass.
        return

    repo = work.repo
    ref = snapshot or "HEAD"
    recovery_dir = target_dir / ".daydream" / "partial-fixes"
    recovery_dir.mkdir(parents=True, exist_ok=True)

    for fkey in sorted(fix_failures):
        slug = fkey.replace("/", "-").replace("\\", "-")
        file_path = repo / fkey
        # 1. Save the partial-fix content first -- non-negotiable, before revert.
        try:
            patch = git_ops.diff_worktree_against(repo, ref, [fkey])
        except GitError as exc:
            patch = ""
            print_warning(console, f"Could not diff partial fix for '{fkey}': {exc}")
        if patch:
            (recovery_dir / f"{slug}.patch").write_text(patch, encoding="utf-8")
        elif file_path.is_file() and fkey not in pre_untracked:
            # Newly-created untracked file (no diff vs ref): preserve raw content.
            try:
                (recovery_dir / f"{slug}.orphan").write_text(
                    file_path.read_text(encoding="utf-8", errors="replace"),
                    encoding="utf-8",
                )
            except OSError:
                pass
        # 2. Restore the path to its pre-fix content.
        try:
            git_ops.restore_paths_from_ref(repo, ref, [fkey])
        except GitError:
            # Path absent at ref => the failed group newly created it. Remove the
            # orphan only when it was not already present pre-fix (attributable).
            if file_path.is_file() and fkey not in pre_untracked:
                try:
                    file_path.unlink()
                except OSError:
                    pass


def _reject_generated_file_edits(
    work: WorkContext,
    target_dir: Path,
    *,
    snapshot: str | None,
    snapshot_captured: bool,
    pre_untracked: set[str],
    pre_untracked_contents: dict[str, bytes] | None = None,
) -> list[str] | None:
    """Restore changed generated files, returning ``None`` if restoration fails."""
    from daydream import git_ops
    from daydream.git_ops import GitError

    if not snapshot_captured:
        # A failed snapshot has no trustworthy pre-fix baseline.  In
        # particular, falling back to HEAD could erase a user's existing edit.
        return []

    repo = work.repo
    ref = snapshot or "HEAD"
    changed = git_ops.changed_files_against(
        repo, ref, preexisting_untracked=pre_untracked
    )

    tracked_violations: list[str] = []
    patches: dict[str, str] = {}
    recovery_dir = target_dir / ".daydream" / "partial-fixes"
    for path in changed:
        try:
            baseline = git_ops.show(repo, ref, path)
        except GitError:
            # Newly-created generated files (notably new migrations) are
            # deliberately allowed.
            continue
        if not is_generated_file(path, baseline):
            continue
        try:
            patch = git_ops.diff_worktree_against(repo, ref, [path])
        except GitError as exc:
            patch = ""
            print_warning(console, f"Could not save forbidden generated-file edit for '{path}': {exc}")
        if not patch:
            # New generated files (notably new migrations) have no baseline
            # diff and are deliberately allowed.
            continue
        tracked_violations.append(path)
        patches[path] = patch

    untracked_baselines = pre_untracked_contents or {}
    untracked_violations = _changed_untracked_generated_files(repo, untracked_baselines)
    direct_violations = [*tracked_violations, *untracked_violations]
    paths_to_restore = list(tracked_violations)
    for path in direct_violations:
        for manifest_path in related_manifest_paths(path):
            if manifest_path in paths_to_restore:
                continue
            try:
                manifest_patch = git_ops.diff_worktree_against(repo, ref, [manifest_path])
                if not manifest_patch:
                    continue
                git_ops.show(repo, ref, manifest_path)
            except GitError:
                continue
            paths_to_restore.append(manifest_path)
            patches[manifest_path] = manifest_patch

    for path, patch in patches.items():
        slug = path.replace("/", "-").replace("\\", "-")
        digest = hashlib.sha256(path.encode("utf-8", errors="surrogateescape")).hexdigest()[:12]
        try:
            recovery_dir.mkdir(parents=True, exist_ok=True)
            (recovery_dir / f"{slug}-{digest}.patch").write_text(patch, encoding="utf-8")
        except OSError as exc:
            print_warning(console, f"Could not write recovery patch for '{path}': {exc}")

    for path in untracked_violations:
        file_path = repo / path
        if not file_path.is_file():
            continue
        slug = path.replace("/", "-").replace("\\", "-")
        digest = hashlib.sha256(path.encode("utf-8", errors="surrogateescape")).hexdigest()[:12]
        try:
            recovery_dir.mkdir(parents=True, exist_ok=True)
            (recovery_dir / f"{slug}-{digest}.orphan").write_bytes(file_path.read_bytes())
        except OSError as exc:
            print_warning(console, f"Could not save forbidden generated-file edit for '{path}': {exc}")

    restoration_failed = False
    if paths_to_restore:
        try:
            git_ops.restore_paths_from_ref(repo, ref, paths_to_restore)
        except GitError as exc:
            print_warning(console, f"Could not restore generated files: {exc}")
            restoration_failed = True

    for path in untracked_violations:
        try:
            _restore_untracked_generated_file(repo, path, untracked_baselines[path])
        except OSError as exc:
            print_warning(console, f"Could not restore generated file '{path}': {exc}")
            restoration_failed = True

    if direct_violations:
        artifact = generated_file_violations_path(deep_dir(target_dir))
        try:
            artifact.write_text(
                json.dumps({"violations": direct_violations, "ref": ref}, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            print_warning(console, f"Could not record generated-file violations: {exc}")
        if not restoration_failed:
            print_warning(
                console,
                f"Reverted forbidden edits to existing generated files: {', '.join(direct_violations)}. "
                "Add a new migration file instead.",
            )
    return None if restoration_failed else direct_violations


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


async def _step_exploration(ctx: FlowContext) -> None:
    """Exploration pre-scan (D-43), reused on an exact key match."""
    from daydream.exploration import cache_key_path, exploration_cache_key, read_cache_key
    from daydream.runner import _compute_diff_ref

    config = ctx.config
    target_dir = ctx.work.repo
    daydream_dir = target_dir / ".daydream"
    diff = ctx.data["diff"]
    tier = ctx.data["tier"]
    exploration_path = daydream_dir / "exploration"

    exploration_dir: Path | None = None
    if not EXPLORATION_AVAILABLE:
        print_warning(
            console,
            "Exploration infrastructure not installed; running deep pipeline "
            "without pre-scan grounding",
        )
    elif config.exploration_context is None:
        # The in-process context short-circuits first; the disk cache is only
        # consulted when there is no in-memory context to reuse.
        cache_key = exploration_cache_key(
            ctx.work.head_sha or "", diff, tier, config.exploration_depth
        )
        if (
            exploration_path.is_dir()
            and read_cache_key(exploration_path) == cache_key
        ):
            # Early return BEFORE the pre_scan/write_to_dir block below: routing
            # a hit through it with an empty in-memory context would overwrite
            # the cached files with "No data collected" stubs.
            print_dim(console, f"Reusing exploration pre-scan from {exploration_path}")
            ctx.data["exploration_dir"] = exploration_path
            return

        # Miss: drop any stale directory so a partial previous result cannot be
        # read as this run's grounding.
        if exploration_path.is_dir():
            shutil.rmtree(exploration_path, ignore_errors=True)

        if tier == "skip":
            print_dim(console, "Skipping exploration -- trivial diff")
            config.exploration_context = ExplorationContext()
        else:
            print_phase_hero(console, "EXPLORE", phase_subtitle("EXPLORE"))
            explore_backend = ctx.backend_for("exploration")
            print_dim(console, f"Exploration model: {explore_backend.model}")
            async with phase_scope(DaydreamPhase.EXPLORATION):
                config.exploration_context = await safe_explore(
                    pre_scan,
                    explore_backend,
                    target_dir,
                    diff,
                    config.exploration_depth,
                    diff_ref=_compute_diff_ref(target_dir),
                )
            console.print(render_exploration_summary(config.exploration_context))
        if config.exploration_context is not None:
            exploration_dir = config.exploration_context.write_to_dir(exploration_path)
            if config.exploration_context.completed:
                cache_key_path(exploration_path).write_text(cache_key, encoding="utf-8")
            ctx.data["exploration_dir"] = exploration_dir
            return
    if EXPLORATION_AVAILABLE and config.exploration_context is not None:
        exploration_dir = config.exploration_context.write_to_dir(exploration_path)
    ctx.data["exploration_dir"] = exploration_dir


async def _step_intent(ctx: FlowContext) -> None:
    """TTT intent analysis, grounded by the PR description when it is fresh.

    On exit, ``ctx.data["intent_authoritative"]`` is set to True when a fresh,
    head-matched PR description with non-whitespace content grounded the intent
    phase (issue #279). Downstream reviewers read this key to determine whether
    to include the authoritative-intent precedence rule in their prompts.
    """
    from daydream import git_ops

    config = ctx.config
    work = ctx.work
    target_dir = work.repo

    print_stage_progress(console, 1, 5, _PIPELINE_STAGE_NAMES[0])
    pr_description: str | None = None
    if config.pr_number is not None:
        pr_view = git_ops.gh_pr_view(target_dir, config.pr_number)
        if pr_view is not None:
            pr_state = pr_view.get("state", "")
            pr_head_oid = pr_view.get("headRefOid", "")
            local_head = work.head_sha
            if pr_state and pr_state.upper() != "OPEN":
                print_warning(
                    console,
                    f"PR #{config.pr_number} state is {pr_state!r} (not OPEN); "
                    "skipping PR description to avoid trusting a stale body",
                )
            elif pr_head_oid and local_head and pr_head_oid != local_head:
                print_warning(
                    console,
                    f"PR #{config.pr_number} head SHA ({pr_head_oid[:12]}) "
                    f"does not match local HEAD ({local_head[:12]}); "
                    "skipping PR description to avoid trusting a mismatched body",
                )
            else:
                pr_description = pr_view.get("body") or None
    # Issue #279: publish whether a fresh, head-matched PR description grounded
    # the intent phase, so downstream reviewers can include the precedence rule.
    # Match build_intent_prompt: whitespace-only bodies are ignored after strip.
    ctx.data["intent_authoritative"] = bool(pr_description and pr_description.strip())
    async with phase_scope(DaydreamPhase.INTENT):
        ctx.data["intent_summary"] = await phase_understand_intent(
            ctx.backend_for("intent"),
            work,
            ctx.data["diff_path"],
            ctx.data["log"],
            ctx.data["branch"],
            exploration_dir=ctx.data["exploration_dir"],
            pr_description=pr_description,
            diff_text=ctx.data["diff"],
        )
    # Each TTT step persists its own half, so a later step's failure cannot
    # discard an artifact this one already produced.
    intent_p = _intent_path(ctx.data["dd"])
    intent_p.write_text(ctx.data["intent_summary"])
    ctx.data["intent_path"] = intent_p


async def _wonder(ctx: FlowContext) -> None:
    """TTT alternative-review (tier-gated) + its artifact write."""
    intent_summary = ctx.data["intent_summary"]

    print_stage_progress(console, 2, 5, _PIPELINE_STAGE_NAMES[1])
    if ctx.data["tier"] == "skip":
        alt_issues: list[dict[str, Any]] = []
        print_dim(console, "Skipping alternatives -- trivial diff")
    else:
        async with phase_scope(DaydreamPhase.ALTERNATIVES):
            alt_issues = await phase_alternative_review(
                ctx.backend_for("wonder"),
                ctx.work,
                ctx.data["diff_path"],
                intent_summary,
                exploration_dir=ctx.data["exploration_dir"],
                diff_text=ctx.data["diff"],
            )

    alts_p = _alternatives_path(ctx.data["dd"])
    alts_p.write_text(json.dumps(alt_issues, indent=2))
    ctx.data["alts_path"] = alts_p


async def _step_wonder_and_per_stack(ctx: FlowContext) -> None:
    """Wonder (TTT alternative-review) alongside the per-stack review fan-out.

    On a fresh multi-stack run the two are siblings in one task group: wonder
    only feeds the merge agent and the dedup pre-filter, so the reviewers do not
    need to wait for it. Their prompts drop the ``alternatives.json`` pointer,
    since the file does not exist yet.

    Single-stack mode and every ``--start-at`` resume keep today's serial order
    and the pointer — in single-stack mode there is no merge agent, so the
    reviewer pointer is the ONLY path wonder findings take into the report.
    """
    # A resume (--start-at per-stack/merge/fix) skips wonder entirely — its
    # artifact is already on disk, which is also why the pointer stays on.
    run_wonder = _fresh_ttt(ctx)
    concurrent = run_wonder and not ctx.data["single_stack_mode"]
    holder: dict[str, BaseException | None] = {"exc": None}

    async def _wonder_guarded() -> None:
        # Held, not degraded: a wonder failure must fail the run, but only after
        # the fan-out's outputs are on disk for a later resume.
        try:
            await _wonder(ctx)
        except Exception as exc:  # noqa: BLE001 -- re-raised after the join
            holder["exc"] = exc

    if run_wonder and not concurrent:
        await _wonder(ctx)

    async with anyio.create_task_group() as tg:
        if concurrent:
            tg.start_soon(_wonder_guarded)
        await _per_stack_body(ctx, include_alternatives=not concurrent)

    if holder["exc"] is not None:
        raise holder["exc"]


async def _per_stack_body(ctx: FlowContext, *, include_alternatives: bool) -> None:
    """Per-stack review fan-out, with failure persistence and resume reconstruction."""
    config = ctx.config
    dd = ctx.data["dd"]
    stacks = ctx.data["stacks"]

    failed_stacks: dict[str, str] = ctx.data["failed_stacks"]
    if config.start_at not in ("merge", "fix"):
        print_stage_progress(console, 3, 5, _PIPELINE_STAGE_NAMES[2])
        async with phase_scope(DaydreamPhase.DEEP, stage="review"):
            per_stack_outputs, failed_stacks = await phase_per_stack_reviews(
                ctx.backend_for("per_stack_review"),
                ctx.work,
                stacks,
                diff_path=ctx.data["diff_path"],
                intent_path=ctx.data["intent_path"],
                alternatives_path=ctx.data["alts_path"],
                exploration_dir=ctx.data["exploration_dir"],
                diff_text=ctx.data["diff"],
                intent_authoritative=ctx.data.get("intent_authoritative", False),
                include_alternatives=include_alternatives,
            )
        # Persist so a later `--start-at merge` resume can still surface
        # uncovered stacks (the in-memory failure map otherwise dies here).
        failures_p = per_stack_failures_path(dd)
        if failed_stacks:
            failures_p.write_text(json.dumps(failed_stacks, indent=2, sort_keys=True))
        elif failures_p.exists():
            # Fresh successful run supersedes any stale failures record.
            failures_p.unlink()
    else:
        # Resume: resurrect any prior failure summary before reconstructing
        # outputs, so failed stacks never re-enter the parse pipeline.
        from daydream.deep.artifacts import per_stack_review_path

        failures_p = per_stack_failures_path(dd)
        loaded = _load_failures(failures_p)
        # Surface a prior cross-stack synthesis failure (issue #361): the
        # structured ``MERGE_FAILURE_KEY`` entry is deliberately excluded from
        # ``failed_stacks`` (below) so it can't be misread as a failed stack /
        # garbled "Uncovered stacks" line, but resuming into a *partial* review
        # must not look clean -- say so explicitly so a ``--start-at fix``
        # relaunch doesn't fix + commit partial findings as if the cross-stack
        # merge had succeeded.
        _warn_prior_merge_failure(loaded)
        # Legacy entries are ``{stack_name: reason}`` str->str. Skip the
        # structured merge-failure entry (``MERGE_FAILURE_KEY``, a dict)
        # so it is never misread as a failed stack that would surface as
        # a garbled "Uncovered stacks" line on a resume (issue #361).
        failed_stacks = {
            str(k): str(v) for k, v in loaded.items() if isinstance(v, str)
        }
        per_stack_outputs = {
            stack.stack_name: per_stack_review_path(dd, stack.stack_name)
            for stack in stacks
            if stack.stack_name not in failed_stacks
        }
    ctx.data["per_stack_outputs"] = per_stack_outputs
    ctx.data["failed_stacks"] = failed_stacks


async def _step_per_stack_parse(ctx: FlowContext) -> Stop | None:
    """Pre-merge parse pass (D-21) + structural partitioning; loads records on a merge resume."""
    config = ctx.config
    dd = ctx.data["dd"]
    stacks = ctx.data["stacks"]
    failed_stacks: dict[str, str] = ctx.data["failed_stacks"]
    per_stack_outputs: dict[str, Path] = ctx.data["per_stack_outputs"]

    print_stage_progress(console, 4, 5, _PIPELINE_STAGE_NAMES[3])

    per_stack_records_paths: list[Path] = []
    all_records: list[dict[str, Any]] = []
    record_sources: list[str] = []
    if config.start_at == "merge":
        # Resume: require a records file per detected stack (except ones in
        # `failed_stacks`). A bare glob would silently drop a stack whose
        # records file is absent, yielding a merged report missing a bucket.
        expected_paths: list[Path] = []
        missing_stacks: list[str] = []
        for stack in stacks:
            if stack.stack_name in failed_stacks:
                continue
            records_path = per_stack_records_path(dd, stack.stack_name)
            if records_path.is_file():
                expected_paths.append(records_path)
            else:
                missing_stacks.append(stack.stack_name)
        if missing_stacks:
            print_error(
                console,
                "Missing Per-Stack Records",
                "Missing parsed records for: "
                + ", ".join(sorted(missing_stacks)),
            )
            return Stop(1)
        # Issue #309: a prior run's uncovered-file sweep records are
        # per-stack-style findings already finalized on disk (the sweep itself
        # is a no-op on resume). Load them so a merge resume keeps the sweep's
        # findings instead of silently dropping them.
        sweep_path = per_stack_records_path(dd, "uncovered")
        if sweep_path.is_file():
            expected_paths.append(sweep_path)
        for records_path in sorted(expected_paths):
            records = json.loads(records_path.read_text())
            per_stack_records_paths.append(records_path)
            source_name = records_path.name
            all_records.extend(records)
            record_sources.extend(source_name for _ in records)
    else:
        # Pre-merge parse pass (D-21). The N parse calls run concurrently;
        # results are consumed in stack_name order below so merge input order
        # is independent of task completion order, keeping the merge prompt
        # and global issue numbering reproducible.
        parse_backend = ctx.backend_for("parse")
        limiter = anyio.CapacityLimiter(effective_fanout_concurrency(10, parse_backend))
        recorder = get_current_recorder()
        parse_results: dict[str, list[dict[str, Any]]] = {}
        parse_failures: dict[str, BaseException] = {}

        async with phase_scope(DaydreamPhase.PARSE):
            async with anyio.create_task_group() as tg:
                for stack_name, output_path in sorted(per_stack_outputs.items()):
                    # Every stack -- language or the structural meta-stack --
                    # parses with the severity-bearing PER_STACK_RECORD_SCHEMA.
                    # Issue #314: the structural reviewer calibrates anti-slop
                    # findings to medium/low, so its parse must carry severity
                    # or that calibration is silently upgraded to high at merge
                    # (the anti-slop rubric's primary home). The structural
                    # meta-stack is still partitioned out of arbitration/dedup
                    # below (unchanged); _append_structural_and_write_merged
                    # preserves the reported severity, falling back to high only
                    # for unlabeled records.
                    record_schema = PER_STACK_RECORD_SCHEMA

                    # Default-arg capture -- prevents late-binding closure bug (Pitfall 2).
                    async def _parse_one(
                        stack_name: str = stack_name,
                        input_path: Path = output_path,
                        schema: dict[str, Any] = record_schema,
                    ) -> None:
                        async with limiter:
                            async with maybe_fork(recorder, f"parse-{stack_name}"):
                                try:
                                    records = await phase_parse_feedback(
                                        parse_backend,
                                        ctx.work,
                                        input_path=input_path,
                                        output_schema=schema,
                                    )
                                except Exception as exc:  # noqa: BLE001 -- captured so one failure cannot cancel siblings mid-write
                                    parse_failures[stack_name] = exc
                                    return
                                # Written per task, not after the join, so a
                                # sibling's failure cannot discard records that
                                # already succeeded (they survive for --start-at).
                                per_stack_records_path(dd, stack_name).write_text(
                                    json.dumps(records, indent=2)
                                )
                                parse_results[stack_name] = records

                    tg.start_soon(_parse_one)

        if recorder is not None:
            recorder.create_dispatch_step(phase=DaydreamPhase.PARSE)
        # Fail the run on the first failure by stack name — same semantics and
        # exception type as the serial loop, just deferred past the join so
        # sibling records that completed are already on disk.
        if parse_failures:
            raise parse_failures[sorted(parse_failures)[0]]

        for stack_name in sorted(parse_results):
            records = parse_results[stack_name]
            per_stack_records_paths.append(per_stack_records_path(dd, stack_name))
            all_records.extend(records)
            record_sources.extend(stack_name for _ in records)

    # Partition structural meta-stack records out before dedup: its lens
    # (file-size budgets, layering, canonical-helper gaps) differs from the
    # language stacks and collapsing it into their dedup pool would demote
    # those findings. Filter both record_sources forms (resume=filename,
    # fresh-run=stack_name) together to preserve the index invariant.
    structural_path_candidate = per_stack_records_path(dd, STRUCTURE_STACK_NAME)
    if structural_path_candidate in per_stack_records_paths:
        structural_records_path: Path | None = structural_path_candidate
        structural_filename = structural_path_candidate.name
        per_stack_records_paths = [
            p for p in per_stack_records_paths if p != structural_path_candidate
        ]
        kept_pairs = [
            (rec, src)
            for rec, src in zip(all_records, record_sources, strict=True)
            if src != STRUCTURE_STACK_NAME and src != structural_filename
        ]
        all_records = [rec for rec, _ in kept_pairs]
        record_sources = [src for _, src in kept_pairs]
    else:
        structural_records_path = None

    ctx.data["records_paths"] = per_stack_records_paths
    ctx.data["records"] = all_records
    ctx.data["record_sources"] = record_sources
    ctx.data["structural_records_path"] = structural_records_path
    return None


def _clear_sweep_artifacts(dd: Path) -> None:
    """Delete the uncovered-file sweep's owned artifacts from ``dd``.

    ``coverage-stats.json``, ``stack-uncovered-records.json``, and every
    ``uncovered-*-review.md`` are the sweep's outputs. A ``--start-at per-stack``
    resume re-runs the sweep, so any artifact left from the prior run must be
    removed BEFORE new per-stack work: otherwise a rerun whose sweep is
    disabled / finds nothing / produces no output would leave stale records
    behind, and a later merge resume would reload them as this run's coverage.
    Merge/fix resumes keep the artifacts (the sweep is a no-op there and the
    records must survive).

    Fail-CLOSED: this cleanup is resume-safety-critical, not best-effort
    diagnostics. When a targeted artifact cannot be removed (a ``OSError`` from
    ``unlink()``, or an artifact that survives the loop), the function raises an
    ``OSError`` with an actionable message and the per-stack resume stops --
    it must never continue with a stale ``stack-uncovered-records.json`` in
    place that a later merge resume would reload as current findings.
    """
    patterns = (
        "coverage-stats.json",
        "stack-uncovered-records.json",
        "uncovered-*-review.md",
    )
    for pattern in patterns:
        for path in dd.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass
    remaining = [p for pattern in patterns for p in dd.glob(pattern)]
    if remaining:
        names = ", ".join(sorted(p.name for p in remaining))
        raise OSError(
            f"stale sweep artifact(s) could not be removed: {names}; "
            "refusing to resume per-stack"
        )


async def _step_uncovered_sweep(ctx: FlowContext) -> None:
    """Issue #309: fail-open second-pass sweep over diff files no reviewer read.

    After per-stack reviews + parse, computes which diff files no ``deep-``
    reviewer read (via ``analyze_coverage``), budget-filters them (hunk size +
    capacity cap), dispatches one cheap reviewer per surviving file, parses the
    findings into ordinary ``PER_STACK_RECORD_SCHEMA`` records, and appends
    them to ``ctx.data`` so the arbiter/merge consume them exactly like any
    per-stack stack's records. Coverage stats land in ``deep/coverage-stats.json``.

    Fail-open: any exception here is caught, logged as a warning, and the step
    returns normally -- the sweep must NEVER fail the run.
    """
    if ctx.config.start_at in ("merge", "fix"):
        # Resume: records are already finalized on disk; a sweep would
        # re-review stale coverage against a diff that already ran.
        return None
    try:
        await _run_uncovered_sweep(ctx)
    except Exception as exc:  # noqa: BLE001 -- fail-open: never fail the run
        print_warning(
            console,
            f"Uncovered-file sweep failed (fail-open): {type(exc).__name__}: {exc}",
        )


async def _run_uncovered_sweep(ctx: FlowContext) -> None:
    """Run the uncovered-file sweep body (issue #309)."""
    config = ctx.config
    dd = ctx.data["dd"]
    recorder = get_current_recorder()
    session_id = recorder.session_id if recorder is not None else None

    uncovered_files, coverage_stats = compute_uncovered_files(dd.parent, session_id)

    swept_files, skipped_small_files, skipped_capacity_files = filter_sweepable_files(
        uncovered_files,
        ctx.data["diff"],
        min_hunk_lines=_uncovered_sweep_min_hunk_lines(config),
        max_files=_uncovered_sweep_max_files(config),
    )

    stats: dict[str, Any] = {
        "pre_sweep": {
            "files_in_diff": coverage_stats["files_in_diff"],
            "files_read_by_reviewers": coverage_stats["files_read_by_reviewers"],
            "coverage_ratio": coverage_stats["coverage_ratio"],
            "uncovered_files": uncovered_files,
        },
        "attempted_files": swept_files,
        "completed_files": [],
        # Issue #309 finding 6: ``covered_files`` is filled from the POST-sweep
        # recompute (verified completed reads of the swept files) and may be a
        # strict subset of ``completed_files`` -- a review written without a
        # Read of the file is an attempt, never coverage. Until the recompute
        # runs it starts empty (fail-open: unverifiable means not claimed).
        "covered_files": [],
        # POST-sweep ratio is recomputed below after the sweep forks land; this
        # pre-sweep snapshot is the fallback when the sweep produces no reads.
        "post_sweep": {
            "files_read_by_reviewers": coverage_stats["files_read_by_reviewers"],
            "coverage_ratio": coverage_stats["coverage_ratio"],
        },
        "sweep_finding_count": 0,
        # The integer skip counts are derived from the filename lists so the
        # two views cannot diverge (issue #309 finding 10).
        "sweep_skipped_capacity": len(skipped_capacity_files),
        "sweep_skipped_small_hunks": len(skipped_small_files),
        "sweep_skipped_capacity_files": skipped_capacity_files,
        "sweep_skipped_small_hunks_files": skipped_small_files,
    }
    stats_p = dd / "coverage-stats.json"

    if not swept_files:
        # A re-run that finds nothing to sweep must still refresh the records
        # artifact to a current empty list so a stale prior file (from the run
        # being resumed) cannot linger and be reloaded by a later merge resume.
        if config.start_at == "per-stack":
            per_stack_records_path(dd, "uncovered").write_text(json.dumps([]))
        stats_p.write_text(json.dumps(stats, indent=2))
        return

    # Cheap-tier dispatch (parse tier), parallel, in diff order. Each sweep
    # fork is `deep-uncovered-<n>` so post-run analyze_coverage counts its
    # reads (the coverage-ratio-improves acceptance criterion).
    parse_backend = ctx.backend_for("parse")
    limiter = anyio.CapacityLimiter(effective_fanout_concurrency(10, parse_backend))
    review_outputs: dict[str, Path] = {}
    sweep_failures: dict[str, str] = {}

    async with anyio.create_task_group() as tg:
        for n, file in enumerate(swept_files):
            output_path = dd / f"uncovered-{n}-review.md"
            prompt = build_uncovered_sweep_prompt(
                file=file,
                hunks=diff_block_for_file(ctx.data["diff"], file) or "",
                intent_path=ctx.data["intent_path"],
                cwd=ctx.work.repo,
                output_path=output_path,
                exploration_dir=ctx.data["exploration_dir"],
            )

            async def _sweep_one(
                file: str = file,
                task_prompt: str = prompt,
                task_output: Path = output_path,
                n: int = n,
            ) -> None:
                async with limiter:
                    async with maybe_fork(recorder, f"deep-uncovered-{n}"):
                        try:
                            _, _, budget_reason = await run_agent(
                                parse_backend,
                                ctx.work.repo,
                                task_prompt,
                                phase=DaydreamPhase.DEEP,
                                tool_call_budget=DEFAULT_TOOL_CALL_BUDGET,
                                wall_budget_s=DEFAULT_WALL_BUDGET_S,
                            )
                            if budget_reason:
                                sweep_failures[file] = f"budget exhausted: {budget_reason}"
                            elif task_output.is_file():
                                # A backend can return normally without writing
                                # its output; only an actual review file counts
                                # as coverage, so the parse loop and
                                # `completed_files` never see phantom outputs.
                                review_outputs[file] = task_output
                            else:
                                sweep_failures[file] = "no review output written"
                        except Exception as exc:  # noqa: BLE001 -- parallel isolation; fail-open
                            sweep_failures[file] = f"{type(exc).__name__}: {exc}"

            tg.start_soon(_sweep_one)

    if recorder is not None:
        recorder.create_dispatch_step(phase=DaydreamPhase.DEEP)

    # Parse each sweep review into PER_STACK_RECORD_SCHEMA records (fail-open:
    # unparseable or failed parses are dropped, never fatal). Parse failures are
    # tracked BY FILENAME into `sweep_failures`, not just the `parse_dropped`
    # count, so coverage-stats.json names exactly which file's parse failed.
    parse_results: dict[str, list[dict[str, Any]]] = {}
    parse_dropped = 0
    parse_failures: dict[str, str] = {}

    async def _parse_all() -> None:
        nonlocal parse_dropped
        async with anyio.create_task_group() as tg:
            for i, (file, output_path) in enumerate(sorted(review_outputs.items())):
                async def _parse_one(
                    file: str = file,
                    input_path: Path = output_path,
                    i: int = i,
                ) -> None:
                    nonlocal parse_dropped
                    async with limiter:
                        async with maybe_fork(recorder, f"parse-uncovered-{i}"):
                            try:
                                records = await phase_parse_feedback(
                                    parse_backend,
                                    ctx.work,
                                    input_path=input_path,
                                    output_schema=PER_STACK_RECORD_SCHEMA,
                                )
                                parse_results[file] = records
                            except Exception as exc:  # noqa: BLE001 -- fail-open drop
                                parse_dropped += 1
                                parse_failures[file] = (
                                    f"parse failed: {type(exc).__name__}: {exc}"
                                )

                tg.start_soon(_parse_one)

    await _parse_all()

    # Recompute coverage AFTER the sweep so the report shows the ratio the
    # sweep actually achieved (the ``deep-uncovered-*`` forks' completed reads
    # now count), never the pre-sweep snapshot. The recompute is fail-open: a
    # failure here falls back to the pre-sweep numbers already stored. The same
    # recompute drives ``covered_files`` (issue #309 finding 6): a swept file is
    # covered only when the post-sweep uncovered list no longer contains it --
    # i.e. a verified completed read of the file happened. A successful review
    # output WITHOUT a read leaves the file in the uncovered list, so it is
    # never claimed as covered.
    post_uncovered: list[str] | None = None
    try:
        post_uncovered, post_coverage = compute_uncovered_files(dd.parent, session_id)
        stats["post_sweep"] = {
            "files_read_by_reviewers": post_coverage["files_read_by_reviewers"],
            "coverage_ratio": post_coverage["coverage_ratio"],
        }
        stats["covered_files"] = sorted(f for f in review_outputs if f not in post_uncovered)
    except Exception:  # noqa: BLE001 -- fail-open: keep the pre-sweep fallback
        pass

    # Merge the sweep records into the per-stack record set exactly like the
    # per-stack parse loop does, so arbiter/merge consume them as ordinary
    # per-stack records (no separate score path). The records file is written
    # whenever at least one sweep review produced output -- an emptied sweep
    # stack is ``[]``, mirroring ``_rewrite_stack_records`` semantics. When NO
    # review produced output, a current empty records artifact is still written
    # so a stale file from a prior run cannot linger.
    sweep_records: list[dict[str, Any]] = []
    for file in sorted(parse_results):
        sweep_records.extend(parse_results[file])
    stats["sweep_parse_dropped"] = parse_dropped
    stats["sweep_failures"] = {**sweep_failures, **parse_failures}
    stats["completed_files"] = sorted(review_outputs)
    # Issue #309 finding 6: per-file attempt status. A completed review output
    # is a completed ATTEMPT; only files with a verified post-sweep completed
    # read are "read". Anything else is "reviewed (hunks only)" and must not
    # move files_read_by_reviewers / coverage_ratio.
    covered_set = set(stats.get("covered_files") or [])
    stats["sweep_attempt_status"] = {
        file: ("read" if file in covered_set else "reviewed (hunks only)")
        for file in sorted(review_outputs)
    }
    if review_outputs:
        records_path = per_stack_records_path(dd, "uncovered")
        records_path.write_text(json.dumps(sweep_records, indent=2))
        ctx.data["records_paths"].append(records_path)
        ctx.data["records"].extend(sweep_records)
        ctx.data["record_sources"].extend("uncovered" for _ in sweep_records)
        stats["sweep_finding_count"] = len(sweep_records)
    else:
        # No review produced output: on a re-run, write a current empty records
        # artifact so a stale file from the resumed run cannot linger.
        if config.start_at == "per-stack":
            per_stack_records_path(dd, "uncovered").write_text(json.dumps([]))
    stats_p.write_text(json.dumps(stats, indent=2))


async def _step_arbiter(ctx: FlowContext) -> None:
    """Scoped arbiter over high-severity/contested findings (#168)."""
    config = ctx.config
    dd = ctx.data["dd"]
    all_records: list[dict[str, Any]] = ctx.data["records"]
    record_sources: list[str] = ctx.data["record_sources"]

    # Scoped Opus arbiter (#168). Sonnet ran the per-stack reviews;
    # a single heavyweight arbiter now re-reviews ONLY the
    # high-severity / contested findings and writes its verdicts back
    # into the per-stack records before merge. A `--start-at merge`
    # resume re-runs arbitration from the on-disk records UNLESS the
    # completion marker proves a prior run already finalised them
    # (#175): a crash between the parse write and the rewrite would
    # otherwise let unarbitrated high-severity findings reach merge.
    #
    # The marker covers the WHOLE adjudication block (arbiter +, in
    # precision mode, suppression): it is written once after BOTH passes
    # have rewritten the per-stack records, so its presence proves the
    # records are fully adjudicated -- not just arbitrated. Renamed from
    # `arbiter_complete_path` so resume reasoning cannot under-read it as
    # arbiter-only (#232 review).
    adjudication_marker = adjudication_complete_path(dd)
    if config.start_at != "merge" or not adjudication_marker.is_file():
        arbiter_targets = select_arbiter_targets(all_records, record_sources)
        # Capture the identities of records the arbiter will see, before
        # `_apply_adjudication_verdicts` compacts the list (#232). `arbiter_targets`
        # are indices into this pre-apply list; once records are dropped the
        # indices shift, so suppression exclusion must be keyed by per-record
        # object identity -- not by the stale positional indices, and not by
        # `(file, line)`: two findings can share one location while only one is
        # arbitrated (a HIGH sibling arbitrated, a LOW sibling not), and a
        # `(file, line)` key would wrongly exclude BOTH, silently skipping the
        # LOW sibling from suppression. `_apply_adjudication_verdicts` revises
        # records in place, so kept AND revised arbiter records keep their
        # identity here; only dropped records fall out.
        arbitrated_ids = {id(all_records[i]) for i in arbiter_targets}
        if arbiter_targets:
            async with phase_scope(DaydreamPhase.DEEP, stage="arbiter"):
                arbiter_backend = ctx.backend_for("arbiter")
                verdicts, arbiter_continuation = await phase_arbiter_review(
                    arbiter_backend,
                    ctx.work,
                    selected_records=[all_records[i] for i in arbiter_targets],
                    diff_path=ctx.data["diff_path"],
                    intent_path=ctx.data["intent_path"],
                    alternatives_path=ctx.data["alts_path"],
                    exploration_dir=ctx.data["exploration_dir"],
                    intent_authoritative=ctx.data.get("intent_authoritative", False),
                )
                # Identity gate: only resume when merge runs on the very same
                # backend instance. A per-phase override that resolves a
                # different backend gets the cold path.
                if arbiter_continuation is not None and arbiter_backend is ctx.backend_for("merge"):
                    ctx.data["arbiter_continuation"] = arbiter_continuation
            all_records, record_sources = _apply_adjudication_verdicts(
                all_records, record_sources, arbiter_targets, verdicts,
                pass_name="arbiter",
                id_field="arb_id",
                fail_closed=False,
            )
            _rewrite_stack_records(
                dd, ctx.data["records_paths"], all_records, record_sources
            )

        # Precision-mode suppression pass (#232). OPT-IN: when precision_mode is
        # off (product default) this block never runs, `select_suppression_targets`
        # is never called, and arbiter output is byte-identical. When on, it gives
        # the borderline (LOW-confidence / low-severity uncontested) findings the
        # arbiter never sees a skeptical second opinion, dropping any it cannot
        # confirm (fail-CLOSED, the inverse of the arbiter). The arbiter target set
        # is the exclusion set so nothing high-severity / contested is re-judged
        # here. One batched agent call, resolved via the cheaper `suppression`
        # phase key (Sonnet default) -- never per-finding Opus.
        if _precision_mode(config):
            suppression_exclude = [
                i for i, r in enumerate(all_records) if id(r) in arbitrated_ids
            ]
            suppression_targets = select_suppression_targets(
                all_records, record_sources, suppression_exclude
            )
            if suppression_targets:
                async with phase_scope(DaydreamPhase.DEEP, stage="suppression"):
                    sup_verdicts = await phase_suppression_review(
                        ctx.backend_for("suppression"),
                        ctx.work,
                        selected_records=[all_records[i] for i in suppression_targets],
                        diff_path=ctx.data["diff_path"],
                        intent_path=ctx.data["intent_path"],
                        alternatives_path=ctx.data["alts_path"],
                        exploration_dir=ctx.data["exploration_dir"],
                    )
                all_records, record_sources = _apply_adjudication_verdicts(
                    all_records, record_sources, suppression_targets, sup_verdicts,
                    pass_name="suppression",
                    id_field="sup_id",
                    fail_closed=True,
                )
                _rewrite_stack_records(
                    dd, ctx.data["records_paths"], all_records, record_sources
                )
        adjudication_marker.write_text("")
    ctx.data["records"] = all_records
    ctx.data["record_sources"] = record_sources


# Structured merge-failure entry reserved in ``per-stack-failures.json`` (issue #361).
# Distinct from per-stack entries (``{stack_name: reason}`` str->str) so the resume
# loader can skip it rather than misread it as a failed stack.
MERGE_FAILURE_KEY = "__merge__"


def _load_failures(path: Path) -> dict[str, Any]:
    """Load a ``per-stack-failures.json`` into a dict, defaulting to ``{}`` on absent/malformed.

    Shared defensive loader for the "load existing per-stack-failures.json"
    pattern (resume loader in ``_per_stack_body`` and merge-failure salvage in
    ``_salvage_merge_failure``). Content is returned verbatim -- including the
    structured ``MERGE_FAILURE_KEY`` entry, which each caller filters or
    handles per its own contract. Only a missing file, malformed JSON, or a
    non-dict root degrades to the ``{}`` "no prior failures" default.
    """
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _warn_prior_merge_failure(loaded: dict[str, Any]) -> None:
    """Warn on resume that a prior cross-stack synthesis failed (issue #361).

    The structured ``MERGE_FAILURE_KEY`` entry is deliberately excluded from
    ``failed_stacks`` (see caller) so it can't be misread as a failed stack /
    garbled "Uncovered stacks" line, but resuming into a *partial* review must
    not look clean -- say so explicitly so a ``--start-at fix`` relaunch doesn't
    fix + commit partial findings as if the cross-stack merge had succeeded.
    """
    merge_entry = loaded.get(MERGE_FAILURE_KEY)
    if merge_entry is not None:
        merge_message = (
            merge_entry.get("message")
            if isinstance(merge_entry, dict)
            else str(merge_entry)
        )
        print_warning(
            console,
            "Prior cross-stack synthesis failed; merged results are PARTIAL. "
            f"{merge_message} (issue #361) -- this resume fixes/verifies the "
            "partial per-stack findings as-is.",
        )


def _clear_merge_failure(dd: Path) -> None:
    """Clear a stale ``__merge__`` salvage record after a successful re-merge.

    ``_salvage_merge_failure`` is the only writer of ``MERGE_FAILURE_KEY``; a
    later successful cross-stack merge (or a fix resume that commits the partial
    findings) must supersede it so a subsequent resume doesn't emit a misleading
    'merged results are PARTIAL' warning for a merge that actually succeeded.
    """
    failures_p = per_stack_failures_path(dd)
    loaded = _load_failures(failures_p)
    if MERGE_FAILURE_KEY not in loaded:
        return
    loaded.pop(MERGE_FAILURE_KEY, None)
    if loaded:
        failures_p.write_text(json.dumps(loaded, indent=2, sort_keys=True))
    elif failures_p.exists():
        failures_p.unlink()


def _drop_cross_stack_duplicates(dd: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the D-27 dedup pre-filter to a host-written partial merge (issue #361).

    In a full merge the merge agent adjudicates ``record_duplicate_pairs``
    (cross-stack records describing the same concern). A salvage writes the
    partial list with no merge agent, so it must apply the pre-filter's computed
    cross-stack duplicate pairs itself -- otherwise the partial
    ``merged-items.json`` carries duplicates into the resume verifier and fix
    gate. Keeps the ``record_a`` side of each pair (deterministic sort order)
    and drops the ``record_b`` side, matched on ``(id, file)`` because per-stack
    record ids are not globally unique.
    """
    dedup_p = dedup_candidates_path(dd)
    if not dedup_p.is_file():
        return records
    try:
        dedup = json.loads(dedup_p.read_text())
    except json.JSONDecodeError:
        return records
    dropped_keys: set[tuple[str, str]] = set()
    for pair in dedup.get("record_duplicate_pairs", []) or []:
        if not isinstance(pair, dict):
            continue
        b_id = pair.get("record_b_id")
        b_file = pair.get("record_b_file")
        if b_id is not None and b_file is not None:
            dropped_keys.add((str(b_id), str(b_file)))
    if not dropped_keys:
        return records
    kept = [
        r
        for r in records
        if (str(r.get("id", "")), str(r.get("file", ""))) not in dropped_keys
    ]
    if len(kept) != len(records):
        print_info(
            console,
            f"Cross-stack merge salvage: dropped {len(records) - len(kept)} "
            "duplicate per-stack record(s) via the D-27 dedup pre-filter",
        )
    return kept


async def _step_cross_stack_merge(ctx: FlowContext) -> Stop | None:
    """Dedup pre-filter (D-27) + cross-stack merge (D-23..D-26).

    A genuinely unparseable merge response (issue #361) is salvaged rather than
    aborting the run: the completed stacks' verdicts are consolidated into a
    partial ``merged-items.json`` + failure record, and the run stops resumably
    (``Stop(1)``) so a relaunch picks up without re-reviewing completed stacks.
    """
    dd = ctx.data["dd"]
    alts_p: Path = ctx.data["alts_path"]
    all_records: list[dict[str, Any]] = ctx.data["records"]
    failed_stacks: dict[str, str] = ctx.data["failed_stacks"]

    # Dedup pre-filter (D-27).
    alt_issues_for_dedup: list[dict[str, Any]] = (
        json.loads(alts_p.read_text()) if alts_p.exists() else []
    )
    pairs = build_dedup_candidates(all_records, alt_issues_for_dedup)
    record_pairs = build_record_dedup_candidates(all_records, sources=ctx.data["record_sources"])
    dedup_p = dedup_candidates_path(dd)
    dedup_p.write_text(
        json.dumps(
            {
                "record_alt_pairs": [_candidate_pair_to_json(p) for p in pairs],
                "record_duplicate_pairs": [_candidate_pair_to_json(p) for p in record_pairs],
            },
            indent=2,
        )
    )

    # Cross-stack merge (D-23..D-26).
    try:
        await phase_cross_stack_merge(
            ctx.backend_for("merge"),
            ctx.work,
            per_stack_records_paths=ctx.data["records_paths"],
            intent_path=ctx.data["intent_path"],
            alternatives_path=alts_p,
            dedup_candidates_path=dedup_p,
            exploration_dir=ctx.data["exploration_dir"],
            failed_stacks=failed_stacks or None,
            structural_records_path=ctx.data["structural_records_path"],
            intent_authoritative=ctx.data.get("intent_authoritative", False),
            continuation=ctx.data.get("arbiter_continuation"),
        )
    except CrossStackMergeError as exc:
        _salvage_merge_failure(ctx, exc)
        return Stop(1)
    # Issue #361: a successful re-merge supersedes any stale salvage record, so
    # the structured ``MERGE_FAILURE_KEY`` entry is cleared here -- otherwise a
    # later ``--start-at merge``/``fix`` resume still warns 'merged results are
    # PARTIAL' even though the cross-stack merge has since succeeded.
    _clear_merge_failure(dd)
    return None


def _salvage_merge_failure(ctx: FlowContext, exc: CrossStackMergeError) -> None:
    """Persist a salvageable cross-stack merge failure (issue #361).

    The merge agent returned a response containing no parseable item list (e.g.
    bare ``str`` prose/refusal/truncated JSON). Instead of aborting the run with
    the completed stacks' verdicts stranded on disk, consolidate the surviving
    per-stack records into a *partial* ``merged-items.json`` + ``review-output.md``
    and record the failure as a structured entry under the reserved
    ``MERGE_FAILURE_KEY`` in ``per-stack-failures.json``. The run then stops
    resumably so a relaunch picks up without re-review.

    Both fallible writes propagate through the project error type -- a genuinely
    unwritable salvage must surface, not silently degrade. The only tolerance is
    loading a missing/malformed existing ``per-stack-failures.json`` as ``{}``
    (the "no prior failures" default); existing per-stack entries are preserved.
    """
    dd = ctx.data["dd"]
    print_error(
        console,
        "Cross-stack merge failed",
        f"{exc}; consolidating surviving per-stack records into a partial report. "
        "Relaunch with --start-at fix to resume.",
    )

    # Build the partial canonical merged-items.json + review-output.md from the
    # surviving per-stack records (reusing the single-stack write helper's shared
    # structural-tagging + render epilogue). Recoverability comes from the
    # structured ``__merge__`` failure record + resumable stop, not a root
    # ``partial`` flag in merged-items.json (no consumer reads it -- issue #361
    # follow-up). Apply the D-27 dedup pre-filter (issue #361): with no merge
    # agent to adjudicate, drop the duplicate side of cross-stack record pairs so
    # the partial list doesn't carry duplicates into the resume verifier/fix gate.
    records = _drop_cross_stack_duplicates(dd, ctx.data["records"])
    _write_single_stack_merged_items(
        ctx.work.repo,
        dd,
        records,
        ctx.data["structural_records_path"],
        failed_stacks=ctx.data.get("failed_stacks") or None,
    )

    # Record the failure for resume. Never drop existing per-stack entries.
    failures_p = per_stack_failures_path(dd)
    failures = _load_failures(failures_p)
    failures[MERGE_FAILURE_KEY] = {
        "response_shape": exc.response_shape,
        "stack_context": exc.stack_context,
        "message": str(exc),
    }
    failures_p.write_text(json.dumps(failures, indent=2, sort_keys=True))
    print_info(console, f"Wrote partial merged items and merge-failure record to {dd}")


async def _step_single_stack_merge(ctx: FlowContext) -> None:
    """Tiny-diff single-stack bypass (#172): host-side merged-items write."""
    failed_stacks: dict[str, str] = ctx.data["failed_stacks"]

    # Issue #172 — tiny-diff single-stack bypass. A ≤2-file diff
    # has nothing to cross-stack-merge and nothing contested to
    # arbitrate, so the host writes ``merged-items.json`` directly
    # via ``normalize_items`` + the exact structural-tagging logic
    # from ``phase_cross_stack_merge``. No arbiter, no dedup, no
    # merge agent. Downstream consumers (fix gate, verifier, PR
    # posting) read the canonical JSON unchanged (AC6).
    _write_single_stack_merged_items(
        ctx.work.repo, ctx.data["dd"], ctx.data["records"], ctx.data["structural_records_path"],
        failed_stacks=failed_stacks or None,
    )


async def _step_load_items(ctx: FlowContext) -> Stop | None:
    """Host-side merged-items guard + render-only markdown recovery."""
    target_dir = ctx.work.repo
    dd = ctx.data["dd"]

    print_stage_progress(console, 5, 5, _PIPELINE_STAGE_NAMES[4])
    merged_report = target_dir / REVIEW_OUTPUT_FILE

    # merged-items.json is the canonical source of truth; review-output.md is
    # render-only. The missing-input guard keys on the JSON so a --start-at fix
    # resume with surviving JSON but absent markdown proceeds rather than bailing.
    items_file = merged_items_path(dd)
    if not items_file.is_file():
        print_error(
            console,
            "Missing Merged Items",
            f"Expected canonical merged items at {items_file}",
        )
        return Stop(1)

    # Best-effort recover the render-only markdown from the deep-dir copy for
    # the exit message when the canonical file is absent (e.g. a --start-at fix
    # resume where the copy to the canonical path never ran). Non-fatal.
    if not merged_report.exists():
        from daydream.deep.artifacts import merged_report_path as _deep_report_path

        deep_copy = _deep_report_path(dd)
        if deep_copy.exists():
            merged_report.write_text(deep_copy.read_text())

    # Issue #309: surface the uncovered-sweep coverage stats on the rendered
    # report. The sweep runs BEFORE the merge writes review-output.md, so the
    # section is appended here, once the report exists, to both the canonical
    # report and its deep-dir copy.
    _append_coverage_section(dd, merged_report, merged_report_path(dd))

    ctx.data["merged_report"] = merged_report
    ctx.data["items_file"] = items_file
    return None


def _append_coverage_section(dd: Path, report: Path, deep_copy: Path) -> None:
    """Append a short ``## Coverage`` section when the sweep produced stats.

    Reads ``deep/coverage-stats.json`` and appends files_in_diff / files read /
    ratio / swept files to both the canonical report and its deep-dir copy. The
    ratio rendered is the POST-sweep value (recomputed after the sweep's forks
    landed); only files whose sweep review produced completed output are labeled
    covered. Failed sweep attempts are surfaced as failures, not claimed as
    coverage. A missing or malformed stats file is a silent no-op -- coverage
    surfacing is advisory, never a gate. ANY failure here (read error, invalid
    JSON, structurally-malformed root, a non-dict shape) warns and returns; it
    can never fail the merge step.
    """
    stats_p = dd / "coverage-stats.json"
    if not stats_p.is_file():
        return
    try:
        stats = json.loads(stats_p.read_text())
        if not isinstance(stats, dict):
            print_warning(
                console,
                "Ignoring malformed coverage stats (expected a JSON object): "
                f"{stats_p}",
            )
            return
        pre_sweep = stats.get("pre_sweep")
        if not isinstance(pre_sweep, dict):
            return
        files_in_diff = pre_sweep.get("files_in_diff")
        if not isinstance(files_in_diff, int):
            return
        lines = [
            "## Coverage",
            f"- Files in diff: {files_in_diff}",
        ]
        # Prefer the POST-sweep numbers (the ratio the sweep actually achieved);
        # fall back to the pre-sweep snapshot when the sweep did not recompute.
        post_sweep = stats.get("post_sweep")
        read_source = post_sweep if isinstance(post_sweep, dict) else pre_sweep
        files_read = read_source.get("files_read_by_reviewers")
        if isinstance(files_read, int):
            lines.append(f"- Files read by reviewers: {files_read}")
        ratio = read_source.get("coverage_ratio")
        if isinstance(ratio, (int, float)):
            lines.append(f"- Coverage ratio: {ratio}")
        # Issue #309 finding 6: only files with a verified completed read are
        # labeled covered. A completed review output WITHOUT a read is a
        # completed attempt -- rendered as "reviewed (hunks only)" -- and never
        # appears on the covered line nor moves the ratio above.
        covered = stats.get("covered_files")
        if isinstance(covered, list) and covered:
            lines.append(f"- Second-pass sweep covered: {', '.join(str(f) for f in covered)}")
        completed = stats.get("completed_files")
        if isinstance(completed, list):
            hunks_only = [
                str(f) for f in completed if not (isinstance(covered, list) and f in covered)
            ]
            if hunks_only:
                lines.append(
                    f"- Second-pass sweep reviewed (hunks only): {', '.join(hunks_only)}"
                )
        failures = stats.get("sweep_failures")
        if isinstance(failures, dict) and failures:
            lines.append(f"- Best-effort sweep failures: {', '.join(sorted(str(f) for f in failures))}")
        skipped = stats.get("sweep_skipped_capacity")
        if isinstance(skipped, int) and skipped:
            lines.append(f"- Sweep capacity-skipped files: {skipped}")
        section = "\n".join(lines) + "\n"
        for target in (report, deep_copy):
            if target.is_file():
                text = target.read_text(encoding="utf-8")
                if "## Coverage" not in text:
                    target.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 -- advisory decoration: never fail the step
        print_warning(
            console,
            "Skipping coverage stats render (advisory; run continues): "
            f"{type(exc).__name__}: {exc}",
        )


async def _step_findings_out(ctx: FlowContext) -> Stop:
    """Two-phase findings artifact (Phase A): emit the strict-schema artifact and STOP."""
    from daydream.runner import _emit_findings_from_items

    items_file: Path = ctx.data["items_file"]
    findings_items: list[dict[str, Any]] = json.loads(items_file.read_text())["items"]
    return Stop(_emit_findings_from_items(ctx.work.repo, ctx.config, findings_items))


async def _step_supervise(ctx: FlowContext) -> None:
    """Apply the configured findings supervisor to canonical merged items."""
    mode = _supervisor_mode(ctx.config)
    file_config = ctx.config.file_config
    items_file: Path = ctx.data["items_file"]
    items = json.loads(items_file.read_text())["items"]
    if mode == "rules":
        assert file_config is not None, "rules mode requires file_config (guaranteed by _supervisor_mode)"
        deny_globs = file_config.supervisor_deny_globs
        verdicts = RuleBasedSupervisor(deny_globs=deny_globs).review_findings(items)
    else:
        async with phase_scope(DaydreamPhase.DEEP, stage="supervise"):
            verdicts = await phase_supervise_review(
                ctx.backend_for("supervise"),
                ctx.work,
                items=items,
                diff_path=ctx.data["diff_path"],
                intent_path=ctx.data["intent_path"],
                alternatives_path=ctx.data["alts_path"],
                exploration_dir=ctx.data["exploration_dir"],
            )
    kept, held, events = apply_findings_verdicts(items, verdicts)
    items_file.write_text(json.dumps({"items": kept, "held": held}, indent=2))

    report = render_report(kept)
    held_section = render_held_section(held)
    if held_section:
        report = report.rstrip() + "\n\n" + held_section + "\n"
    deep_report = merged_report_path(ctx.data["dd"])
    deep_report.write_text(report)
    ctx.data["merged_report"].write_text(report)

    recorder = get_current_recorder()
    if recorder is not None:
        for finding_id, action, reason in events:
            recorder.emit_supervisor_verdict(finding_id, action, reason)
    return None


async def _step_post_review(ctx: FlowContext) -> Stop | None:
    """Offer to post findings as inline PR review comments; ``--comment`` auto-posts.

    In comment mode posting is the run's deliverable, so a missing PR or a
    failed GitHub submission ends the run with exit code 1 instead of the
    warn-and-continue the default deep flow gets (#8).
    """
    from daydream.pr_review import PostStatus, post_review_to_pr_from_report

    items_file: Path = ctx.data["items_file"]
    outcome = await post_review_to_pr_from_report(
        ctx.work.repo,
        items_file,
        console=console,
        post=_mode_of(ctx) == "comment",
        approve_on_clean=_approve_on_clean(ctx.config),
    )
    if _mode_of(ctx) == "comment" and outcome in (PostStatus.NO_PR, PostStatus.FAILED):
        return Stop(1)
    return None


async def _step_fix_gate(ctx: FlowContext) -> Stop | None:
    """Fix-apply gate; on accept, load and severity-sort the canonical items."""
    # Fix-apply gate across the two interaction axes. ``--yes`` auto-applies;
    # an unattended run with no assumption declines (safe_default=False) so a
    # piped/CI run never mutates without intent; otherwise prompt.
    decision = resolve_or_prompt(
        assume=get_assume(),
        interactive=not get_non_interactive(),
        safe_default=False,
        question="Apply fixes now? [y/N]",
        default="n",
    )
    if not decision:
        print_success(console, f"Report written to {ctx.data['merged_report']}. Exiting.")
        return Stop(0)

    # Read canonical merged items directly (validated above). Replaces an LLM
    # re-parse of the markdown, which silently dropped structural findings; here
    # they are ordinary tagged items that reach phase_fix like any other.
    items_file: Path = ctx.data["items_file"]
    items: list[dict[str, Any]] = json.loads(items_file.read_text())["items"]
    if not items:
        print_success(console, "No actionable items -- done.")
        return Stop(0)

    # Issue #336 — pre-fix scope partition. Findings on files OUTSIDE the
    # reviewed diff are filed as GitHub issues (best-effort) and excluded from
    # auto-fix: the loop must not expand the PR's scope. The reviewed-diff file
    # set is resolved via _resolve_changed_files (shared with _step_fix) so the
    # gate and the post-fix residual net agree on the allowed set (a divergence
    # left the residual net strictly weaker than the gate on the resume path).
    changed_files = _resolve_changed_files(ctx)
    if changed_files is not None:
        in_scope: list[dict[str, Any]] = []
        out_of_scope: list[dict[str, Any]] = []
        for item in items:
            (in_scope if (item.get("file") or "") in changed_files else out_of_scope).append(item)
        for item in out_of_scope:
            _file_out_of_scope_issue(ctx, item)
        if out_of_scope:
            print_warning(
                console,
                f"{len(out_of_scope)} finding(s) outside the reviewed diff filed as "
                "issue(s), not fixed.",
            )
        items = in_scope
        # Issue #336 — every finding routed to issues leaves nothing to
        # auto-fix, so short-circuit before a no-op fix pass, a full target
        # test-suite run, and a commit-agent turn. Matches the pre-partition
        # "no actionable items" Stop(0) above. Keying on identity (``is not
        # None``) distinguishes an empty reviewed diff — every file is out of
        # scope, so all findings are filed and the run ends — from ``None``,
        # which skips the partition entirely because scope cannot be judged.
        if not items:
            print_success(
                console,
                "All findings outside the reviewed diff -- filed as issues, nothing to fix.",
            )
            return Stop(0)

    # Severity-ordered (high before medium before low), stable within a
    # tier so equal-severity items keep their canonical merge order.
    ctx.data["items"] = severity_sorted(items)
    return None


async def _step_verify(ctx: FlowContext) -> None:
    """Recommendation verification (#83) + verdict join rendering."""
    dd = ctx.data["dd"]
    items: list[dict[str, Any]] = ctx.data["items"]

    # Recommendation verification (#83). Runs ONLY after the apply-fixes
    # gate accepts, so a declined run (non-interactive / EOF / explicit "N")
    # skips both the verify pass and the recommendation-verdicts.json
    # artifact. A --start-at fix resume still produces verdicts whenever
    # fixes are applied (the gate still runs on resume; accept => verify runs).
    async with phase_scope(DaydreamPhase.VERIFY):
        verdicts_file, verdicts_payload = await phase_verify_recommendations(
            ctx.backend_for("verify"),
            ctx.work,
            merged_items_path=ctx.data["items_file"],
            deep_dir=dd,
        )
    print_verification_summary(console, verdicts_file)

    # Attach verifier verdicts to items by `id` (advisory; phase_fix reads them).
    items = _attach_verdicts(items, verdicts_payload)
    ctx.data["items"] = items
    matched_ids = [i["id"] for i in items if i.get("verifier_verdict") is not None]
    unmatched_ids = [
        i["id"]
        for i in items
        if isinstance(i.get("id"), int)
        and i.get("verifier_verdict") is None
        and i.get("lens") != "structural"
    ]
    # Structural findings are verdict-exempt (in neither matched nor unmatched)
    # but still fixed; itemize them so the "X/Y matched" ratio isn't read as a
    # total that under-counts the items the fix loop iterates.
    structural_ids = [i.get("id") for i in items if i.get("lens") == "structural"]
    # Leftovers (no verdict, non-structural, missing/non-int id) so the
    # buckets always reconcile to len(items); surfaced only when present.
    other_ids = [
        i.get("id")
        for i in items
        if i.get("verifier_verdict") is None
        and i.get("lens") != "structural"
        and not isinstance(i.get("id"), int)
    ]
    console.print(
        format_verdict_join(
            matched=matched_ids,
            unmatched=unmatched_ids,
            structural=structural_ids,
            other=other_ids,
            total=len(items),
        )
    )


async def _capture_quality_before(daydream_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Best-effort pre-fix quality snapshot; ``(None, reason)`` when unavailable.

    ``analyze_quality`` is pure and deterministic (no backend, no network), but
    any failure here degrades to "gate unavailable" rather than failing the run
    -- the anti-degradation gate is fail-open by design (#315). The failure
    reason is returned so the gate can persist an auditable unavailable entry
    instead of leaving a silent blank (#329). The sync tree-walk runs off the
    event loop so parallel fix fan-out is never blocked by the analyzer
    (#329 / CodeRabbit Finding D).
    """
    try:
        from daydream.eval.analyzer import analyze_quality

        return await anyio.to_thread.run_sync(analyze_quality, daydream_dir), None
    except Exception as exc:  # noqa: BLE001 -- fail-open: never fail the run
        return None, f"{type(exc).__name__}: {exc}"


def _quality_delta(before: float | None, after: float | None) -> float | None:
    """Rounded per-file metric delta; ``None`` when either side is undefined."""
    if before is None or after is None:
        return None
    return round(after - before, 4)


def _quality_flagged(
    *,
    erosion_before: float | None,
    erosion_after: float | None,
    erosion_delta: float | None,
    verbosity_before: float | None,
    verbosity_after: float | None,
    verbosity_delta: float | None,
    erosion_threshold: float,
    verbosity_threshold: float,
    erosion_absolute_threshold: float,
    verbosity_absolute_threshold: float,
) -> bool:
    """Whether a fixed file regressed past a threshold (#315).

    A file is flagged when its delta exceeds the threshold (both sides
    defined), or on its absolute AFTER value vs the absolute threshold when the
    BEFORE metric is undefined -- no functions / no non-blank lines pre-fix --
    but the AFTER metric is numeric. The absolute yardstick is a SEPARATE knob
    from the delta one (#329 / CodeRabbit Finding D): an undefined baseline has
    no delta, so a delta threshold is the wrong ruler for the absolute
    comparison. The absolute fallback fires on ANY file with an undefined
    baseline, not just new ones, so an EXISTING file that gains a CC>10
    function (erosion ``None`` pre-fix) is still caught. A metric with both
    sides undefined never flags.
    """
    if erosion_delta is not None and erosion_delta > erosion_threshold:
        return True
    if verbosity_delta is not None and verbosity_delta > verbosity_threshold:
        return True
    if erosion_before is None and erosion_after is not None and erosion_after > erosion_absolute_threshold:
        return True
    if verbosity_before is None and verbosity_after is not None and verbosity_after > verbosity_absolute_threshold:
        return True
    return False


def _quality_gate_threshold(config: RunConfig, attr: str, default: float) -> float:
    """Resolve a quality-gate threshold (delta or absolute), degrading invalid values to *default*.

    Mirrors ``_uncovered_sweep_max_files``: ``RunConfig`` field > file-config
    scalar > default, then the same finite non-negative guard the file-config
    parser applies (#329 / Finding 7). A negative threshold flags every
    unchanged file (a zero delta exceeds it); a NaN/infinite one disables the
    metric (every comparison against it is False) and writes a non-standard
    ``NaN`` to JSON. Both degrade to the named default so a directly-constructed
    ``RunConfig`` / ``DaydreamFileConfig`` cannot smuggle an invalid floor past
    the parser.
    """
    value = _resolve_config_value(config, attr, default)
    coerced = _coerce_quality_threshold(value)
    return coerced if coerced is not None else default


def _current_session_id() -> str | None:
    """Session id binding the quality-gate artifact to the current run."""
    recorder = get_current_recorder()
    return recorder.session_id if recorder is not None else None


def _load_quality_gate_rounds(gate_p: Path, session_id: str | None) -> list[dict[str, Any]]:
    """Load prior rounds for the CURRENT session, or start fresh when rebound.

    Rounds are carried forward only when the artifact's stored ``session_id``
    matches the current run's session (issue #329 / Finding 5): a ``--start-at
    fix`` resume of the SAME session appends rather than clobbers, while an
    artifact left by a DIFFERENT run is discarded with a warning so the first
    run's verdicts can never be archived as the new run's (corrupting the
    manifest / SQLite audit history). An artifact with no stored session is
    treated as belonging to another session.

    Never raises, and a malformed artifact is never treated as authoritative
    prior rounds (issue #329 / Finding 6): invalid JSON, a non-object payload
    (``[]``, ``42``, ...), a missing ``rounds`` list, or non-object round
    entries all degrade to an empty round list WITH a warning, so the current
    run repairs the artifact instead of silently losing the gate.
    """
    try:
        raw = gate_p.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        existing = json.loads(raw)
    except (json.JSONDecodeError, AttributeError, TypeError):
        existing = None
    if not isinstance(existing, dict):
        print_warning(console, f"Quality gate artifact {gate_p} is malformed; starting rounds fresh")
        return []
    if existing.get("session_id") != session_id:
        print_warning(
            console,
            f"Quality gate artifact {gate_p} belongs to another run's session; "
            "its rounds were not carried forward",
        )
        return []
    existing_rounds = existing.get("rounds")
    if not isinstance(existing_rounds, list):
        print_warning(console, f"Quality gate artifact {gate_p} has no rounds list; starting rounds fresh")
        return []
    valid = [r for r in existing_rounds if isinstance(r, dict)]
    if len(valid) != len(existing_rounds):
        print_warning(console, f"Quality gate artifact {gate_p} has non-object round entries; dropping them")
    return valid


def _persist_quality_gate_unavailable(
    *,
    gate_p: Path,
    rounds: list[dict[str, Any]],
    round_no: int,
    stage: str,
    reason: str,
    session_id: str | None,
    erosion_delta_threshold: float,
    verbosity_delta_threshold: float,
    erosion_absolute_threshold: float,
    verbosity_absolute_threshold: float,
) -> None:
    """Persist an auditable ``unavailable`` round entry for THIS round.

    Any existing entry with the same round number is dropped first, so an
    unavailable verdict supersedes a stale successful one from the same round
    (e.g. a ``--start-at fix`` resume). Never raises.
    """
    rounds = [r for r in rounds if r.get("round") != round_no]
    rounds.append({"round": round_no, "unavailable": {"stage": stage, "reason": reason}})
    gate_p.write_text(
        json.dumps(
            {
                "enabled": True,
                "erosion_delta_threshold": erosion_delta_threshold,
                "verbosity_delta_threshold": verbosity_delta_threshold,
                "erosion_absolute_threshold": erosion_absolute_threshold,
                "verbosity_absolute_threshold": verbosity_absolute_threshold,
                "session_id": session_id,
                "rounds": rounds,
            },
            indent=2,
        )
    )
    print_warning(
        console,
        f"Quality gate unavailable (round {round_no}, {stage}): {reason}",
    )


async def _evaluate_quality_gate(
    *,
    enabled: bool,
    erosion_delta_threshold: float,
    verbosity_delta_threshold: float,
    erosion_absolute_threshold: float,
    verbosity_absolute_threshold: float,
    daydream_dir: Path,
    dd: Path,
    candidates: set[str] | None,
    before: dict[str, Any] | None,
    before_unavailable_reason: str | None,
    iteration: int | None,
) -> None:
    """Compute and persist the fix-phase quality-gate verdict (issue #315).

    Fail-open by design: never raises, never stops the run. Every payload is
    bound to the current run's ``session_id`` so a later archive step cannot
    attribute another run's verdict to this one (#329). When disabled, writes
    ``{"enabled": false}``. When the gate would run but cannot be evaluated --
    a pre-fix capture failure, a post-fix capture failure, a changed-file
    enumeration failure (``candidates`` is ``None``, #329 / Finding 6), or a
    persist failure -- an explicit ``unavailable`` round entry is persisted for
    THIS round with the failed stage and reason, superseding any stale verdict
    for that round and keeping the failure auditable instead of reading as a
    clean gate. Each ``_step_fix`` invocation appends one ``rounds`` entry
    (keyed by the flow's loop iteration when present, else the next sequence
    number), so a resume or loop preserves the per-round trend. Flagged files
    are surfaced as warnings with their before/after numbers. The post-fix
    sync tree-walk runs off the event loop so parallel fix fan-out is never
    blocked by the analyzer (#329 / CodeRabbit Finding D).
    """
    session_id = _current_session_id()
    try:
        gate_p = fix_quality_gate_path(dd)
        if not enabled:
            gate_p.write_text(
                json.dumps({"enabled": False, "session_id": session_id}, indent=2)
            )
            return
        rounds = _load_quality_gate_rounds(gate_p, session_id)
        round_no = iteration if iteration is not None else len(rounds) + 1
        if candidates is None:
            _persist_quality_gate_unavailable(
                gate_p=gate_p,
                rounds=rounds,
                round_no=round_no,
                stage="candidates",
                reason="could not enumerate files changed by the fix pass against the pre-fix snapshot",
                session_id=session_id,
                erosion_delta_threshold=erosion_delta_threshold,
                verbosity_delta_threshold=verbosity_delta_threshold,
                erosion_absolute_threshold=erosion_absolute_threshold,
                verbosity_absolute_threshold=verbosity_absolute_threshold,
            )
            return
        if before is None:
            _persist_quality_gate_unavailable(
                gate_p=gate_p,
                rounds=rounds,
                round_no=round_no,
                stage="before",
                reason=before_unavailable_reason or "pre-fix quality snapshot unavailable",
                session_id=session_id,
                erosion_delta_threshold=erosion_delta_threshold,
                verbosity_delta_threshold=verbosity_delta_threshold,
                erosion_absolute_threshold=erosion_absolute_threshold,
                verbosity_absolute_threshold=verbosity_absolute_threshold,
            )
            return
        try:
            from daydream.eval.analyzer import analyze_quality

            after = await anyio.to_thread.run_sync(analyze_quality, daydream_dir)
        except Exception as exc:  # noqa: BLE001 -- fail-open: the gate must never fail the run
            _persist_quality_gate_unavailable(
                gate_p=gate_p,
                rounds=rounds,
                round_no=round_no,
                stage="after",
                reason=f"{type(exc).__name__}: {exc}",
                session_id=session_id,
                erosion_delta_threshold=erosion_delta_threshold,
                verbosity_delta_threshold=verbosity_delta_threshold,
                erosion_absolute_threshold=erosion_absolute_threshold,
                verbosity_absolute_threshold=verbosity_absolute_threshold,
            )
            return
        before_per_file: dict[str, Any] = before.get("per_file") or {}
        after_per_file: dict[str, Any] = after.get("per_file") or {}

        per_file: dict[str, dict[str, Any]] = {}
        for rel in sorted(candidates):
            before_entry = before_per_file.get(rel)
            after_entry = after_per_file.get(rel)
            if before_entry is None and after_entry is None:
                continue
            # Issue #329 / Finding 5: a candidate that parsed pre-fix but is
            # MISSING from the post-fix analyzer output is unparseable after the
            # fix (analyze_quality omits malformed files). Recording null
            # after-metrics with ``flagged=false`` would read as a clean
            # verdict, so mark it explicitly unavailable and flagged -- a fix
            # that breaks a file is a regression, never a pass. Still fail-open:
            # never raises, never stops the run.
            if before_entry is not None and after_entry is None:
                per_file[rel] = {
                    "erosion_before": before_entry.get("erosion"),
                    "erosion_after": None,
                    "erosion_delta": None,
                    "verbosity_before": before_entry.get("verbosity"),
                    "verbosity_after": None,
                    "verbosity_delta": None,
                    "unparseable": True,
                    "flagged": True,
                    "reason": "file missing from post-fix analyzer output (unparseable?)",
                }
                continue
            erosion_before = before_entry.get("erosion") if before_entry is not None else None
            erosion_after = after_entry.get("erosion") if after_entry is not None else None
            verbosity_before = before_entry.get("verbosity") if before_entry is not None else None
            verbosity_after = after_entry.get("verbosity") if after_entry is not None else None
            erosion_delta = _quality_delta(erosion_before, erosion_after)
            verbosity_delta = _quality_delta(verbosity_before, verbosity_after)
            per_file[rel] = {
                "erosion_before": erosion_before,
                "erosion_after": erosion_after,
                "erosion_delta": erosion_delta,
                "verbosity_before": verbosity_before,
                "verbosity_after": verbosity_after,
                "verbosity_delta": verbosity_delta,
                "flagged": _quality_flagged(
                    erosion_before=erosion_before,
                    erosion_after=erosion_after,
                    erosion_delta=erosion_delta,
                    verbosity_before=verbosity_before,
                    verbosity_after=verbosity_after,
                    verbosity_delta=verbosity_delta,
                    erosion_threshold=erosion_delta_threshold,
                    verbosity_threshold=verbosity_delta_threshold,
                    erosion_absolute_threshold=erosion_absolute_threshold,
                    verbosity_absolute_threshold=verbosity_absolute_threshold,
                ),
            }
        rounds = [r for r in rounds if r.get("round") != round_no]
        rounds.append({"round": round_no, "per_file": per_file})
        payload = {
            "enabled": True,
            "erosion_delta_threshold": erosion_delta_threshold,
            "verbosity_delta_threshold": verbosity_delta_threshold,
            "erosion_absolute_threshold": erosion_absolute_threshold,
            "verbosity_absolute_threshold": verbosity_absolute_threshold,
            "session_id": session_id,
            "rounds": rounds,
        }
        gate_p.write_text(json.dumps(payload, indent=2))
        flagged = [rel for rel, entry in per_file.items() if entry["flagged"]]
        if flagged:
            lines = []
            for rel in flagged:
                entry = per_file[rel]
                if entry.get("unparseable"):
                    lines.append(f"  - {rel}: {entry['reason']}")
                else:
                    lines.append(
                        f"  - {rel}: erosion {entry['erosion_before']} -> "
                        f"{entry['erosion_after']}, verbosity "
                        f"{entry['verbosity_before']} -> {entry['verbosity_after']}"
                    )
            print_warning(
                console,
                f"Quality gate flagged {len(flagged)} file(s) after fixes:\n" + "\n".join(lines),
            )
    except Exception as exc:  # noqa: BLE001 - fail-open: the gate must never fail the run
        # Stage "persist": the payload itself could not be written. Record the
        # failure as an unavailable round when the write path still works, and
        # ALWAYS warn -- a gate failure that surfaces nothing reads as a clean
        # pass, which is the exact hazard #329 describes.
        try:
            gate_p = fix_quality_gate_path(dd)
            rounds = _load_quality_gate_rounds(gate_p, session_id)
            round_no = iteration if iteration is not None else len(rounds) + 1
            _persist_quality_gate_unavailable(
                gate_p=gate_p,
                rounds=rounds,
                round_no=round_no,
                stage="persist",
                reason=f"{type(exc).__name__}: {exc}",
                session_id=session_id,
                erosion_delta_threshold=erosion_delta_threshold,
                verbosity_delta_threshold=verbosity_delta_threshold,
                erosion_absolute_threshold=erosion_absolute_threshold,
                verbosity_absolute_threshold=verbosity_absolute_threshold,
            )
        except Exception as inner:  # noqa: BLE001 - nothing left to persist; stay fail-open
            print_warning(
                console,
                f"Quality gate unavailable (round {iteration if iteration is not None else '?'}, "
                f"persist): {type(exc).__name__}: {exc}; could not persist unavailable verdict "
                f"({type(inner).__name__}: {inner})",
            )


def _first_unconfined_finding(
    repo: Path, items: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return the first item whose ``file`` ref the confinement gate rejects.

    Mirrors ``_preflight_finding_file_refs``'s item order (items[0] first,
    then the rest) and reuses the phase's own predicate
    (``_resolve_finding_file_ref``) so there is no grammar drift between the
    phase's preflight raise and this guard's offender identification.
    Returns ``None`` when every item is confined.
    """
    for item in items:
        try:
            _resolve_finding_file_ref(repo, item.get("file"))
        except ValueError:
            return item
    return None


async def _step_fix(ctx: FlowContext) -> Stop | None:
    """Parallel fix pass: pre-fix snapshot capture, phase_fix_parallel, failure protection."""
    from daydream import git_ops

    config = ctx.config
    work = ctx.work
    target_dir = work.repo
    daydream_dir = target_dir / ".daydream"
    dd = ctx.data["dd"]
    items: list[dict[str, Any]] = ctx.data["items"]
    intent_p: Path = ctx.data["intent_path"]

    # Only forward confirmed intent when we ran the intent phase in
    # this invocation.  When resuming via --start-at fix/merge/per-stack
    # the intent phase was skipped, so intent_p may hold a stale
    # artifact from a prior run; injecting it as authoritative would
    # contradict the current diff's context.
    intent_grounded_this_run = config.start_at not in ("per-stack", "merge", "fix")
    # Snapshot the tracked tree + untracked set BEFORE fixes so a failed
    # group's partial, possibly non-compiling edits can be captured and
    # rolled back to exactly their pre-fix content (#203 follow-up).
    try:
        pre_fix_snapshot = git_ops.stash_create(work.repo)
        pre_fix_untracked = set(git_ops.list_untracked(work.repo))
        pre_fix_untracked_contents = _snapshot_untracked_generated_files(work.repo, pre_fix_untracked)
    except (git_ops.GitError, OSError) as exc:
        print_warning(console, f"Could not snapshot tree before fixes: {exc}")
        pre_fix_snapshot = None
        pre_fix_untracked = set()
        pre_fix_untracked_contents = {}
        pre_fix_snapshot_captured = False
    else:
        pre_fix_snapshot_captured = True
    # Issue #543: thread the pre-fix untracked snapshot into the commit steps so
    # _do_commit can exclude user scratch files from the daydream commit instead
    # of sweeping them in via the commit agent's ``git add --all``.
    ctx.data["pre_fix_untracked"] = pre_fix_untracked
    # Pre-fix HEAD is the recommended-patch base only when the tree was
    # clean (stash_create returns None then) -- otherwise the snapshot is
    # the base and HEAD is unused, so skip the rev-parse. Captured now
    # because the commit phase below advances HEAD past the fix.
    if pre_fix_snapshot is None:
        try:
            pre_fix_head = git_ops.head_sha(work.repo)
        except git_ops.GitError:
            pre_fix_head = None
    else:
        pre_fix_head = None
    # Resolve per-file-group fix budgets (#201): file-config override wins, else
    # the config.py default. A runaway file group cannot silently dominate a run.
    group_wall_s = _resolve_config_value(config, "group_max_wall_s", DEFAULT_GROUP_MAX_WALL_S)
    group_serial = _resolve_config_value(config, "group_max_serial_items", DEFAULT_GROUP_MAX_SERIAL_ITEMS)
    # Issue #315: anti-degradation quality gate. Capture the pre-fix quality
    # snapshot before any fix runs; the post-fix capture + verdict happen after
    # failure protection below. Fail-open: a snapshot failure means the gate is
    # unavailable for this run, never a run failure.
    quality_gate_enabled = _resolve_config_value(config, "quality_gate_enabled", DEFAULT_QUALITY_GATE_ENABLED)
    quality_gate_erosion_delta = _quality_gate_threshold(
        config, "quality_gate_erosion_delta", DEFAULT_QUALITY_GATE_EROSION_DELTA
    )
    quality_gate_verbosity_delta = _quality_gate_threshold(
        config, "quality_gate_verbosity_delta", DEFAULT_QUALITY_GATE_VERBOSITY_DELTA
    )
    quality_gate_erosion_absolute = _quality_gate_threshold(
        config, "quality_gate_erosion_absolute", DEFAULT_QUALITY_GATE_EROSION_ABSOLUTE
    )
    quality_gate_verbosity_absolute = _quality_gate_threshold(
        config, "quality_gate_verbosity_absolute", DEFAULT_QUALITY_GATE_VERBOSITY_ABSOLUTE
    )
    if quality_gate_enabled:
        quality_before, quality_before_unavailable = await _capture_quality_before(daydream_dir)
    else:
        quality_before, quality_before_unavailable = None, None
    # Issue #336 — fix-loop scope bound. Thread the reviewed diff's file set
    # into every fix prompt as an explicit "Allowed files" clause. ``None``
    # (no diff context, e.g. a resume that lost ctx.data["diff"]) leaves the
    # prompt unchanged; the prose scope boundary still applies. Resolved via
    # _resolve_changed_files (shared with the gate) so a missing key never
    # crashes and the gate and post-fix residual net agree on the allowed set.
    changed_files = _resolve_changed_files(ctx)
    async with phase_scope(DaydreamPhase.FIX):
        try:
            fix_failures = await phase_fix_parallel(
                ctx.backend_for("fix"),
                work,
                items,
                intent_path=intent_p if (intent_grounded_this_run and intent_p.exists()) else None,
                group_max_wall_s=group_wall_s,
                group_max_serial_items=group_serial,
                changed_files=changed_files,
            )
        except ValueError as exc:
            # Issue #574: the phase's preflight confinement gate raises on an
            # unconfined/missing/non-string finding ``file`` ref. Route it
            # through the same exception-failure recovery below (patch capture,
            # fix_failures persistence, tree restore, generated-file reject,
            # Stop(1)) instead of letting it escape to cli.py's generic
            # handler. The failure entry is keyed by the finding's STABLE id --
            # never the unconfined path -- so the unsafe ref cannot become a
            # grouping key or restore argument.
            offender = _first_unconfined_finding(work.repo, items)
            offender_id = offender.get("id") if offender else None
            offender_file = offender.get("file") if offender else None
            fix_failures = {
                str(offender_id) if offender_id is not None else "<unconfined-finding>": (
                    f"{type(exc).__name__}: {exc} (finding {offender_id}, "
                    f"file {offender_file!r})"
                )
            }
            print_warning(
                console,
                f"Fix preflight rejected an unconfined finding "
                f"(finding {offender_id}, file {offender_file!r}); no fixes applied.",
            )
    # Capture daydream's proposed diff (pre-fix tree → post-fix worktree)
    # NOW, before the fix-failure and test-failure early returns below, so
    # a run that generated a recommendation always archives it — even when
    # tests fail or a fix group is reverted. Best-effort; never raises.
    git_ops.capture_recommended_patch_with_base(
        work.repo,
        pre_fix_snapshot,
        pre_fix_head,
        daydream_dir / "recommended.patch",
        preexisting_untracked=pre_fix_untracked,
    )
    fix_failures_p = fix_failures_path(dd)
    # Partition failures: budget-exceeded groups have their already-applied fixes
    # intact (only remaining findings were skipped) and must NOT be reverted.
    # Exception-failed groups may hold broken partial edits and must be reverted.
    _BUDGET_PREFIX = "file_group_budget_exceeded:"
    exception_failures = {k: v for k, v in fix_failures.items() if not v.startswith(_BUDGET_PREFIX)}
    budget_skips = {k: v for k, v in fix_failures.items() if v.startswith(_BUDGET_PREFIX)}

    all_non_success = {**exception_failures, **budget_skips}
    if all_non_success:
        # Persist so the archive marks the run "partial" instead of
        # "complete" -- the tree holds skipped or reverted groups.
        fix_failures_p.write_text(json.dumps(all_non_success, indent=2, sort_keys=True))
    elif fix_failures_p.exists():
        fix_failures_p.unlink()

    if exception_failures:
        # Only revert and abort for exception-failed groups; budget-exceeded
        # groups' applied fixes are preserved and the run continues.
        _protect_tree_after_fix_failures(
            work,
            target_dir,
            exception_failures,
            snapshot=pre_fix_snapshot,
            snapshot_captured=pre_fix_snapshot_captured,
            pre_untracked=pre_fix_untracked,
        )
        _reject_generated_file_edits(
            work,
            target_dir,
            snapshot=pre_fix_snapshot,
            snapshot_captured=pre_fix_snapshot_captured,
            pre_untracked=pre_fix_untracked,
            pre_untracked_contents=pre_fix_untracked_contents,
        )
        # Enumerate every untracked path that appeared during the fix
        # pass and survived protection. Attribution to a specific group
        # is impossible (shared tree, parallel groups), so we never
        # delete these -- we record them so the partial state is fully
        # auditable instead of silently leaving stray files unaccounted.
        try:
            leftover = sorted(set(git_ops.list_untracked(work.repo)) - pre_fix_untracked)
        except git_ops.GitError:
            leftover = []
        leftover_p = fix_leftover_untracked_path(dd)
        if leftover:
            leftover_p.write_text(json.dumps(leftover, indent=2))
        elif leftover_p.exists():
            leftover_p.unlink()
        print_warning(
            console,
            f"{len(exception_failures)} fix group(s) failed: {sorted(exception_failures)}; "
            "partial edits reverted (patches saved under .daydream/partial-fixes/).",
        )
        return Stop(1)

    # No exception failures: budget-skipped groups (if any) already warned via
    # _record_budget_stop; proceed to test/commit with the applied fixes intact.
    stale_leftover_p = fix_leftover_untracked_path(dd)
    if stale_leftover_p.exists():
        stale_leftover_p.unlink()
    generated_guard_result = _reject_generated_file_edits(
        work,
        target_dir,
        snapshot=pre_fix_snapshot,
        snapshot_captured=pre_fix_snapshot_captured,
        pre_untracked=pre_fix_untracked,
        pre_untracked_contents=pre_fix_untracked_contents,
    )
    if generated_guard_result is None:
        return Stop(1)
    # Issue #336 (Task 4) — post-fix residual scope check: revert any edit the
    # fix pass made outside the reviewed diff and file an issue per residual, so
    # the commit step below can only land in-scope changes. Runs AFTER the
    # generated-file guard (which already reverted generated-file edits, so no
    # double-revert) and BEFORE the quality gate (which then measures the
    # now-scoped edited set).
    pre_fix_ref = pre_fix_snapshot or "HEAD"
    residual_guard_result = _revert_out_of_scope_edits(
        work,
        pre_fix_ref=pre_fix_ref,
        snapshot_captured=pre_fix_snapshot_captured,
        pre_fix_untracked=pre_fix_untracked,
        changed_files=changed_files,
        finding_files={item["file"] for item in ctx.data["items"] if item.get("file")},
    )
    if residual_guard_result is None:
        # Fail-close to match the generated-file guard: an unreverted
        # out-of-scope edit must never reach the commit step.
        return Stop(1)
    # Issue #315: post-fix anti-degradation gate over the files the fix phase
    # edited. Tree is post-fix here (every applied or budget-preserved fix is
    # intact); a regression is flagged and surfaced, never fatal.
    #
    # Issue #329 / Finding 6: gate every python file the fix pass changed, not
    # just the finding-group targets. The fix agent is explicitly allowed to
    # touch files outside a group's named file, so a regression in such a
    # secondary file would otherwise bypass the gate, the artifact, the
    # manifest, and SQLite. The candidate set is derived from the PRE-FIX git
    # snapshot (clean tree -> HEAD; dirty tree -> the stash snapshot) -- the
    # same base the generated-file guard and recommended-patch capture use --
    # scoped to ``*.py``, then unioned with the finding-target files so finding
    # targets stay covered even when their on-disk content did not change.
    # Fail-open: if enumeration raises, candidates stay ``None`` and the gate
    # persists an explicit ``unavailable`` verdict rather than gating on a
    # partial candidate set.
    quality_candidates: set[str] | None
    if quality_gate_enabled:
        try:
            changed_after_fix = git_ops.changed_files_against(
                work.repo, pre_fix_ref, preexisting_untracked=pre_fix_untracked
            )
            quality_candidates = {path for path in changed_after_fix if path.endswith(".py")}
            quality_candidates |= {item["file"] for item in ctx.data["items"] if item.get("file")}
        except git_ops.GitError:
            quality_candidates = None
    else:
        quality_candidates = set()
    await _evaluate_quality_gate(
        enabled=quality_gate_enabled,
        erosion_delta_threshold=quality_gate_erosion_delta,
        verbosity_delta_threshold=quality_gate_verbosity_delta,
        erosion_absolute_threshold=quality_gate_erosion_absolute,
        verbosity_absolute_threshold=quality_gate_verbosity_absolute,
        daydream_dir=daydream_dir,
        dd=dd,
        candidates=quality_candidates,
        before=quality_before,
        before_unavailable_reason=quality_before_unavailable,
        iteration=ctx.data.get("iteration"),
    )
    return None


async def _step_test(ctx: FlowContext) -> Stop | None:
    """Post-fix test validation."""
    async with phase_scope(DaydreamPhase.TEST):
        passed, retries, proceed = await phase_test_and_heal(
            ctx.backend_for("test"), ctx.work, feedback_items=ctx.data["items"]
        )
    # Persisted before the failure early-return so both outcomes leave a verdict.
    # ``passed`` is always the suite's own result: an operator who continues past
    # a red suite records the override in ``ignored``, never as a green verdict.
    test_verdict_path(ctx.data["dd"]).write_text(
        json.dumps({"passed": passed, "retries": retries, "ignored": proceed and not passed}, indent=2)
    )
    if not proceed:
        print_warning(console, "Tests failed after fix attempt.")
        return Stop(1)
    return None


async def _commit_push_or_stop(coro: Awaitable[None]) -> Stop | None:
    """Await a commit/push phase, mapping failure to a clean Stop(1).

    Shared by _step_commit and _step_commit_push so the try/except ->
    print_error("Commit/Push Failed") -> Stop(1) guard lives in one place.
    The phase coroutine is created by the caller but only awaited here, so a
    synchronous GitError from staging still surfaces inside the guard.
    """
    try:
        await coro
    except Exception as e:
        print_error(console, "Commit/Push Failed", str(e))
        return Stop(1)
    return None


async def _step_commit(ctx: FlowContext) -> Stop | None:
    """Commit-and-push the applied fixes."""
    # phase_commit_push runs as part of the fix/commit cycle — reuse
    # the fix backend (no separate "commit" phase identifier).
    # stage_paths can raise GitError synchronously before the agent turn;
    # _commit_push_or_stop surfaces that as a clean Stop(1) instead of
    # an unhandled traceback terminating the deep run.
    return await _commit_push_or_stop(
        phase_commit_push(
            ctx.backend_for("fix"), ctx.work,
            preexisting_untracked=ctx.data.get("pre_fix_untracked"),
        )
    )


async def _perform_cleanup(ctx: FlowContext) -> None:
    """Terminal cleanup: remove the review output when enabled (#330).

    Restores the shallow ``commit-gate`` semantics the single-flow collapse
    dropped: ``--cleanup`` removes ``.review-output.md`` after a successful
    run, ``--no-cleanup`` keeps it, and an unspecified flag falls back to the
    old preamble gate (``--yes`` cleans up, unattended runs keep the artifact
    via ``safe_default=False``, interactive runs prompt). Not a flow step: it
    is invoked by ``_run_review_spine`` on any successful exit, so an early
    ``Stop(0)`` (e.g. a declined fix gate) still honors ``--cleanup`` while
    every failure path (a non-zero exit) skips it to keep evidence.
    """
    config = ctx.config
    target_dir = ctx.work.repo

    if config.cleanup is True:
        enabled = True
    elif config.cleanup is False:
        enabled = False
    else:
        enabled = resolve_or_prompt(
            assume=get_assume(),
            interactive=not get_non_interactive(),
            safe_default=False,
            question="Cleanup review output after completion? [y/N]",
            default="n",
        )

    if not enabled:
        return
    review_output_path = target_dir / REVIEW_OUTPUT_FILE
    if review_output_path.exists():
        review_output_path.unlink()
        print_success(console, f"Cleaned up {REVIEW_OUTPUT_FILE}")


async def _step_fetch_feedback(ctx: FlowContext) -> None:
    """Fetch bot review comments via the fetch-pr-feedback skill (feedback mode)."""
    await phase_fetch_pr_feedback(
        ctx.backend_for("pr_feedback"), ctx.work, ctx.data["pr_number"], ctx.data["bot"]
    )


async def _step_parse_feedback(ctx: FlowContext) -> Stop | None:
    """Parse the fetched feedback into actionable items; stop when none."""
    try:
        async with phase_scope(DaydreamPhase.PARSE):
            feedback_items = await phase_parse_feedback(ctx.backend_for("parse"), ctx.work)
    except ValueError:
        print_error(console, "Parse Failed", "Failed to parse PR feedback. Exiting.")
        return Stop(1)

    if not feedback_items:
        print_info(console, "No actionable feedback found in PR comments.")
        return Stop(0)

    ctx.data["feedback_items"] = feedback_items
    return None


async def _step_fix_items(ctx: FlowContext) -> Stop | None:
    """Apply a fix per feedback item; abort before commit when all fail."""
    from daydream import git_ops

    feedback_items = ctx.data["feedback_items"]
    fix_backend = ctx.backend_for("fix")

    # Issue #543: snapshot the pre-fix untracked set so the feedback commit step
    # can exclude user scratch files from the daydream commit. list_untracked
    # soft-fails to [] on GitError, so set(...) is already the fail-open empty
    # snapshot (deterministic stage = all untracked), mirroring _step_fix.
    pre_fix_untracked = set(git_ops.list_untracked(ctx.work.repo))
    ctx.data["pre_fix_untracked"] = pre_fix_untracked

    # Fix sequentially to avoid concurrent access to one mutable backend.
    results: list[FixResult] = []
    total_items = len(feedback_items)
    async with phase_scope(DaydreamPhase.FIX):
        for idx, item in enumerate(feedback_items, start=1):
            try:
                await phase_fix(fix_backend, ctx.work, item, idx, total_items)
                results.append((item, True, None))
            except Exception as e:
                results.append((item, False, f"{type(e).__name__}: {e}"))

    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    if not successful:
        print_error(
            console,
            "All Fixes Failed",
            f"All {len(failed)} fix(es) failed. Aborting before commit.",
        )
        return Stop(1)

    ctx.data["results"] = results
    ctx.data["successful"] = successful
    ctx.data["failed"] = failed
    return None


async def _step_commit_push(ctx: FlowContext) -> Stop | None:
    """Commit and push the applied fixes (feedback mode)."""
    results: list[FixResult] = ctx.data["results"]
    return await _commit_push_or_stop(
        phase_commit_push_auto(
            ctx.backend_for("review"), ctx.work,
            items=[item for item, _ok, _err in results if _ok],
            preexisting_untracked=ctx.data.get("pre_fix_untracked"),
        )
    )


async def _step_respond_feedback(ctx: FlowContext) -> Stop:
    """Reply on each addressed comment and print the run summary."""
    pr_number = ctx.data["pr_number"]
    try:
        await phase_respond_pr_feedback(
            ctx.backend_for("pr_feedback"), ctx.work, pr_number, ctx.data["bot"], ctx.data["results"]
        )
    except Exception as e:
        print_warning(console, f"Failed to respond to PR comments: {e}")
        print_info(console, "Fixes were already pushed successfully.")

    successful = ctx.data["successful"]
    failed = ctx.data["failed"]
    console.print()
    print_success(
        console,
        f"PR #{pr_number}: {len(successful)} fix(es) applied"
        + (f", {len(failed)} failed" if failed else ""),
    )

    return Stop(0)


def _fresh_ttt(ctx: FlowContext) -> bool:
    # Verbatim resume gate: --start-at per-stack/merge/fix skips the TTT phases.
    return ctx.config.start_at not in ("per-stack", "merge", "fix")


def _before_fix_resume(ctx: FlowContext) -> bool:
    # Verbatim resume gate: --start-at fix skips everything up to the fix gate.
    # For post-review, this also keeps the non-idempotent GitHub write off the
    # resume path (duplicate inline reviews on reruns).
    return ctx.config.start_at != "fix"


def _multi_stack_merge_enabled(ctx: FlowContext) -> bool:
    return ctx.config.start_at != "fix" and not ctx.data["single_stack_mode"]


def _single_stack_merge_enabled(ctx: FlowContext) -> bool:
    return ctx.config.start_at != "fix" and ctx.data["single_stack_mode"]


def _findings_out_enabled(ctx: FlowContext) -> bool:
    # Two-phase findings artifact (Phase A): emit the artifact and STOP —
    # never post to the PR and never apply fixes. Phase B posts later.
    return ctx.config.findings_out is not None


def _resolve_mode(config: RunConfig) -> str:
    """Map a RunConfig onto the single deep-flow mode key (#330).

    ``feedback`` (``daydream feedback <pr#>``) replaces the pr-feedback flow;
    ``review`` / ``comment`` replace the review flow (stop after post-review);
    ``shallow`` replaces the shallow flow (single-stack deep). ``loop`` is the
    unchanged default.
    """
    if config.bot is not None:
        return "feedback"
    if config.flow_name in ("review", "shallow"):
        return config.flow_name
    if config.output_mode == "review":
        return "review"
    if config.output_mode == "comment":
        return "comment"
    if config.shallow:
        return "shallow"
    return "loop"


def _mode_of(ctx: FlowContext) -> str:
    """The active mode, set by ``run_deep``'s dispatch preamble."""
    return ctx.data.get("mode", "loop")


def _feedback_mode(ctx: FlowContext) -> bool:
    """Feedback mode runs the fetch->parse->fix->commit->respond prefix only."""
    return _mode_of(ctx) == "feedback"


def _review_only_mode(ctx: FlowContext) -> bool:
    """Review/comment modes stop after ``post-review``; no fix cycle runs."""
    return _mode_of(ctx) in ("review", "comment")


def _fix_cycle_enabled(ctx: FlowContext) -> bool:
    """The apply-fix gate + verify/fix/test/commit run in loop + shallow modes."""
    return _mode_of(ctx) in ("loop", "shallow")


def _cleanup_applies(ctx: FlowContext) -> bool:
    """The terminal cleanup runs in every mode that writes ``.review-output.md``.

    Loop/shallow/review/comment all render the report in ``load-items``;
    feedback mode runs only the comment-fetch prefix and writes no report, so
    it is gated off (a leftover report there is the user's own file).
    """
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


def _spine_enabled(ctx: FlowContext) -> bool:
    """The review spine is skipped entirely in feedback mode."""
    return not _feedback_mode(ctx)


def _flow_kind_for_mode(mode: str) -> DaydreamRunFlow:
    """Recorder run-flow label per mode (preserves the pre-collapse mapping)."""
    if mode == "shallow":
        return DaydreamRunFlow.NORMAL
    if mode in ("review", "comment"):
        return DaydreamRunFlow.TTT
    return DaydreamRunFlow.DEEP


def _spine_fresh_ttt(ctx: FlowContext) -> bool:
    """Fresh-run TTT gate AND not feedback mode."""
    return _spine_enabled(ctx) and _fresh_ttt(ctx)


def _spine_before_fix(ctx: FlowContext) -> bool:
    """``--start-at fix`` resume gate AND not feedback mode."""
    return _spine_enabled(ctx) and _before_fix_resume(ctx)


def _spine_multi_merge(ctx: FlowContext) -> bool:
    return _spine_enabled(ctx) and _multi_stack_merge_enabled(ctx)


def _spine_single_merge(ctx: FlowContext) -> bool:
    return _spine_enabled(ctx) and _single_stack_merge_enabled(ctx)


def _spine_supervise(ctx: FlowContext) -> bool:
    return _spine_enabled(ctx) and _supervise_enabled(ctx)


def _spine_uncovered(ctx: FlowContext) -> bool:
    return _spine_enabled(ctx) and _uncovered_sweep_enabled(ctx)


def _spine_findings_out(ctx: FlowContext) -> bool:
    return _spine_enabled(ctx) and _findings_out_enabled(ctx)


# The deep pipeline as a registered flow (D-07):
#
#     [feedback prefix: fetch -> parse -> fix-items -> commit-push -> respond]
#     exploration pre-scan -> TTT intent -> TTT alternative-review ->
#     per-stack reviews -> per-stack parse + dedup -> uncovered-file sweep (#309)
#     -> arbiter -> cross-stack merge (or the tiny-diff single-stack bypass) ->
#     supervise -> findings-out stop / post-review -> fix gate -> verify -> fix ->
#     test -> commit.
#
# ``register_builtins`` registers :data:`STEPS` and the ``deep`` flow
# definition; ``run_deep`` keeps the preamble and delegates here via
# ``run_flow``. The old imperative body's tier / single_stack_mode /
# ``start_at`` / ``findings_out`` conditions are the ``enabled`` predicates
# above (whole-block gates) or stay inside step bodies (resume branches). The
# mode gates replace the review/comment/shallow/pr-feedback flows (#330):
# feedback mode runs only the prefix (the review spine is gated off and
# ``respond-feedback`` ends the flow), and review/comment modes stop after
# ``post-review`` (the fix cycle is gated off).
#
# Terminal cleanup (#330) is NOT a step: it is a success-path helper invoked by
# ``_run_review_spine`` after ``run_flow`` returns. Tying it to the run's exit
# code (rather than the end of this tuple) means an early successful ``Stop(0)``
# -- the fix gate declining -- still honors ``--cleanup``, while any non-zero
# (failure) exit skips it to keep evidence (#335).
STEPS: tuple[FlowStep, ...] = (
    # Feedback-mode prefix (was the ``pr-feedback`` flow): runs first and
    # ``respond-feedback`` ends the flow, so the review spine below never runs.
    FlowStep(name="fetch-feedback", run=_step_fetch_feedback, config_phase="pr_feedback", enabled=_feedback_mode),
    FlowStep(name="parse-feedback", run=_step_parse_feedback, config_phase="parse", enabled=_feedback_mode),
    FlowStep(name="fix-items", run=_step_fix_items, config_phase="fix", enabled=_feedback_mode),
    # config_phase "review" mirrors the old body's use of the review backend for the commit.
    FlowStep(name="commit-push", run=_step_commit_push, config_phase="review", enabled=_feedback_mode),
    FlowStep(name="respond-feedback", run=_step_respond_feedback, config_phase="pr_feedback", enabled=_feedback_mode),
    # Review spine (the deep pipeline).
    FlowStep(name="exploration", run=_step_exploration, enabled=_spine_enabled),
    FlowStep(name="intent", run=_step_intent, enabled=_spine_fresh_ttt),
    FlowStep(
        name="per-stack-reviews",
        run=_step_wonder_and_per_stack,
        config_phase="per_stack_review",
        enabled=_spine_enabled,
    ),
    FlowStep(name="per-stack-parse", run=_step_per_stack_parse, config_phase="parse", enabled=_spine_before_fix),
    FlowStep(name="uncovered-sweep", run=_step_uncovered_sweep, enabled=_spine_uncovered, config_phase="parse"),
    FlowStep(name="arbiter", run=_step_arbiter, enabled=_spine_multi_merge),
    FlowStep(
        name="cross-stack-merge", run=_step_cross_stack_merge, config_phase="merge", enabled=_spine_multi_merge
    ),
    FlowStep(name="single-stack-merge", run=_step_single_stack_merge, enabled=_spine_single_merge),
    FlowStep(name="load-items", run=_step_load_items, enabled=_spine_enabled),
    FlowStep(name="supervise", run=_step_supervise, config_phase="supervise", enabled=_spine_supervise),
    FlowStep(name="findings-out", run=_step_findings_out, enabled=_spine_findings_out),
    FlowStep(name="post-review", run=_step_post_review, enabled=_spine_before_fix),
    # Fix cycle: loop + shallow modes only (review/comment stop after post-review).
    FlowStep(name="fix-gate", run=_step_fix_gate, enabled=_fix_cycle_enabled),
    FlowStep(name="verify", run=_step_verify, enabled=_fix_cycle_enabled),
    FlowStep(name="fix", run=_step_fix, enabled=_fix_cycle_enabled),
    FlowStep(name="test", run=_step_test, enabled=_fix_cycle_enabled),
    # config_phase "fix" mirrors the old body's use of the fix backend for the commit.
    FlowStep(name="commit", run=_step_commit, config_phase="fix", enabled=_fix_cycle_enabled),
)


async def run_deep(config: RunConfig, work: WorkContext) -> int:
    """Execute the deep-review pipeline (D-07) across every PR-process mode.

    Collapsed the review/comment/shallow/pr-feedback flows into modes of the
    single ``deep`` flow (#330): ``review`` and ``comment`` run the review
    spine and stop after ``post-review``, ``shallow`` forces single-stack mode,
    and ``feedback`` (``daydream feedback <pr#>``) runs only the
    fetch -> parse -> fix -> commit -> respond prefix. The default ``loop``
    mode is unchanged.

    Runs the preamble (diff computation, stack detection, tiny-diff collapse,
    trajectory recorder, pre-flight notice) and delegates the pipeline to the
    registered ``deep`` flow (:data:`STEPS`) via ``run_flow``. Supports
    stage-granular resume via
    ``config.start_at in ("ttt", "per-stack", "merge", "fix")``.

    Args:
        config: Run configuration; ``config.shallow`` / ``config.output_mode``
            / ``config.bot`` select the mode. ``config.identity`` carries the
            GitHub identity set by :func:`daydream.runner.run`.
        work: Resolved working environment for the run.

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    mode = _resolve_mode(config)
    if mode == "feedback":
        return await _run_feedback_flow(config, work)
    return await _run_review_spine(config, work, mode)


async def _run_feedback_flow(config: RunConfig, work: WorkContext) -> int:
    """Feedback-mode preamble (was ``runner._run_pr_feedback``).

    Validates args, opens the trajectory recorder, prints the info block,
    then delegates to the registered ``deep`` flow's feedback prefix.
    """
    from daydream.runner import _open_recorder

    if config.pr_number is None or config.bot is None:
        print_error(
            console,
            "Invalid PR config",
            "PR number and --bot are required (use: daydream feedback <pr#> --bot <name>).",
        )
        return 1

    pr_number = config.pr_number
    bot = config.bot
    target_dir = work.repo

    async with _open_recorder(
        config=config, target_dir=target_dir, work=work, flow_kind=DaydreamRunFlow.PR,
    ):
        ctx = FlowContext(config=config, work=work, registry=get_registry())
        ctx.data["mode"] = "feedback"
        ctx.data["pr_number"] = pr_number
        ctx.data["bot"] = bot

        console.print()
        print_info(console, f"PR feedback mode: PR #{pr_number}")
        print_info(console, f"Bot: {bot}")
        print_info(console, f"Target directory: {target_dir}")
        print_info(console, f"Model: {ctx.backend_for('review').model}")
        # Bot logins look like ``my-app[bot]``; escape so Rich doesn't eat the brackets.
        print_info(console, f"GitHub identity: {escape_markup(config.identity)}")
        console.print()

        return await run_flow(ctx.registry, "deep", ctx)


def _shallow_skill_invocation(config: RunConfig) -> str | None:
    """Resolve the ``--skill`` stack slot for shallow mode (#330).

    Mirrors the old shallow preamble's precedence: ``stack:<skill>`` slot, then
    a skill value that is itself a registered slot value. ``None`` (no skill or
    unresolvable skill) falls back to the generic-fallback review.
    """
    if config.skill is None:
        return None
    registry = get_registry()
    resolved = registry.skill_if_registered(f"stack:{config.skill}")
    if resolved is not None:
        return resolved
    if config.skill in registry.skill_slots().values():
        return config.skill
    return None


def _collapse_stacks_for_shallow(
    stacks: list[StackAssignment],
    changed_files: list[str],
    config: RunConfig,
) -> tuple[list[StackAssignment], bool]:
    """Force the single-stack assignment for shallow mode (#330).

    Collapses every non-structural stack into one combined assignment and keeps
    the structural meta-stack separate, so structural findings stay correctly
    tagged ``lens="structural"`` downstream. Returns ``(stacks, True)``.

    The combined assignment's skill, in precedence order:

    - an explicit ``--skill`` (CLI) wins — the combined stack is named by it;
    - otherwise a *sole* detected non-structural stack preserves its
      per-language Beagle skill (e.g. ``beagle-python:review-python`` for a
      Python diff), absorbing any generic/docs files — so ``daydream --shallow
      <repo>`` without ``--skill`` uses the language reviewer instead of the
      generic fallback (#6);
    - otherwise (multiple real-language stacks — one agent cannot invoke two
      per-language skills — or no real language at all) the combined assignment
      uses the generic-fallback skill.
    """
    structural = [s for s in stacks if s.stack_name == STRUCTURE_STACK_NAME]
    combined_files = sorted({f for s in stacks for f in s.files}) or changed_files

    non_structural = [s for s in stacks if s.stack_name != STRUCTURE_STACK_NAME]
    real_language = [s for s in non_structural if s.stack_name != GENERIC_STACK]

    if config.skill is not None:
        combined = StackAssignment(
            stack_name=config.skill,
            skill_invocation=_shallow_skill_invocation(config),
            files=combined_files,
            is_docs_only=False,
        )
    elif len(real_language) == 1 and real_language[0].skill_invocation is not None:
        # Skill-preservation: the sole real-language stack survives unchanged
        # (mirrors ``_collapse_stacks_for_tiny_diff`` for code+docs diffs).
        lang = real_language[0]
        combined = StackAssignment(
            stack_name=lang.stack_name,
            skill_invocation=lang.skill_invocation,
            files=combined_files,
            is_docs_only=False,
        )
    else:
        combined = StackAssignment(
            stack_name=GENERIC_STACK,
            skill_invocation=None,
            files=combined_files,
            is_docs_only=False,
        )
    return [*structural, combined], True


async def _run_review_spine(config: RunConfig, work: WorkContext, mode: str) -> int:
    """Review-spine preamble for the deep pipeline (the former ``run_deep`` body)."""
    # Late imports to avoid circular dependency with runner.
    from daydream import git_ops
    from daydream.backends import Backend
    from daydream.git_ops import GitError, GitTimeoutError
    from daydream.phases import _git_branch, _git_log
    from daydream.runner import (
        _open_recorder,
        _resolved_backend_name,
    )

    # Cache one Backend instance per (backend_name, resolved_model, resolved_effort)
    # so phases that resolve to the same model/effort share an instance and
    # differing ones stay isolated.
    backend_cache: dict[tuple[str, str | None, str | None], Backend] = {}

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
        print_warning(console, "No diff found -- nothing to review")
        return 0

    daydream_dir = target_dir / ".daydream"
    daydream_dir.mkdir(exist_ok=True)
    diff_path = daydream_dir / "diff.patch"
    diff_path.write_text(diff)
    # Diff is immutable from here on; compute the tiering verdict once and reuse
    # it at both the exploration step's gate and the alternatives step's gate.
    tier = select_tier(count_changed_files(diff))
    dd = deep_dir(target_dir)
    current_diff_sha = diff_key(diff)
    if config.start_at not in ("per-stack", "merge", "fix"):
        # Fresh run only: a resume must NOT rewrite the key it is checked
        # against, or the staleness gate would self-heal and pass every time.
        shutil.rmtree(dd, ignore_errors=True)
        dd.mkdir(parents=True, exist_ok=True)
        diff_key_path(dd).write_text(current_diff_sha, encoding="utf-8")

    async with _open_recorder(
        config=config, target_dir=target_dir, work=work, flow_kind=_flow_kind_for_mode(mode),
    ):
        console.print()
        print_info(console, f"Target directory: {target_dir}")
        print_info(console, f"Branch: {branch}")
        print_info(console, f"Default backend: {_resolved_backend_name(config, 'review')}")
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

        # Stack detection (from diff file list). Availability is resolved once in
        # runner.run and threaded via config; None flows through to detect_stacks'
        # optimistic default.
        changed_files = _diff_changed_files(diff)
        stacks = detect_stacks(changed_files, skill_availability=config.skill_availability)
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

        # Pre-flight notice (D-30). Agent count reflects the tiny-diff collapse
        # when single_stack_mode is active (issue #172): merge+arbiter are
        # skipped, so the estimate uses ``_single_stack_agent_count``.
        stack_lines = [_stack_preflight_line(s) for s in stacks]
        notice_agent_count = (
            _single_stack_agent_count(len(stacks))
            if single_stack_mode
            else total_agent_count(len(stacks))
        )
        print_preflight_notice(
            console,
            stages=_PIPELINE_STAGE_NAMES,
            stack_lines=stack_lines,
            agent_count=notice_agent_count,
            exploration_available=EXPLORATION_AVAILABLE,
            sweep_note=_uncovered_sweep_preflight_note(config, changed_files),
        )

        # Flow context (steps communicate through ctx.data); ctx shares
        # run_deep's backend cache so instance-sharing semantics are unchanged.
        ctx = FlowContext(
            config=config,
            work=work,
            registry=get_registry(),
            data={
                "mode": mode,
                "diff": diff,
                # Issue #336 — fix-loop scope bound. The reviewed diff's file
                # set threads through ctx.data so the fix gate can partition
                # out-of-scope findings (Task 3) and the fix step can both
                # forward it into the fix prompt (Task 2) and run a post-fix
                # residual check (Task 4). Recomputable from "diff" via
                # ``_diff_changed_files``; a missing key never crashes.
                "changed_files": set(changed_files),
                "diff_path": diff_path,
                "tier": tier,
                "dd": dd,
                "stacks": stacks,
                "single_stack_mode": single_stack_mode,
                "intent_path": _intent_path(dd),
                "alts_path": _alternatives_path(dd),
                "log": log,
                "branch": branch,
                "failed_stacks": {},
            },
            _backend_cache=backend_cache,
        )

        # Nothing is torn down after the flow. .daydream/exploration/ is a
        # content-keyed cache (see ``exploration_cache_key``) the next run reuses
        # on an exact head+diff+tier+depth match and rewrites on a miss, and
        # .daydream/deep/ is preserved per RESEARCH.md Open Question 1 so
        # subsequent --start-at resumes can find the artifacts they need.
        #
        # Cleanup is success-path only (#335); a non-zero exit returns before the guard so evidence survives.
        exit_code = await run_flow(ctx.registry, "deep", ctx)
        if _cleanup_should_run(ctx, exit_code):
            await _perform_cleanup(ctx)
        return exit_code
