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

import json
import os
import shutil
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.markup import escape as escape_markup

from daydream.agent import console, get_assume, get_non_interactive, resolve_or_prompt
from daydream.config import REVIEW_OUTPUT_FILE, STRUCTURE_STACK_NAME
from daydream.deep.arbiter import select_arbiter_targets
from daydream.deep.artifacts import (
    alternatives_path as _alternatives_path,
)
from daydream.deep.artifacts import (
    arbiter_complete_path,
    check_deep_artifacts,
    dedup_candidates_path,
    deep_dir,
    fix_failures_path,
    fix_leftover_untracked_path,
    merged_items_path,
    per_stack_failures_path,
    per_stack_records_path,
)
from daydream.deep.artifacts import (
    intent_path as _intent_path,
)
from daydream.deep.dedup import build_dedup_candidates, build_record_dedup_candidates
from daydream.deep.detection import GENERIC_STACK, StackAssignment, detect_stacks
from daydream.extensions import get_registry
from daydream.extensions.api import Stop
from daydream.flows.engine import FlowContext
from daydream.phases import (
    FEEDBACK_SCHEMA,
    PER_STACK_RECORD_SCHEMA,
    _write_single_stack_merged_items,
    phase_alternative_review,
    phase_arbiter_review,
    phase_commit_push,
    phase_cross_stack_merge,
    phase_fix_parallel,
    phase_parse_feedback,
    phase_per_stack_reviews,
    phase_test_and_heal,
    phase_understand_intent,
    phase_verify_recommendations,
    severity_sorted,
)
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    TrajectoryRecorder,
    default_trajectory_path,
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

    Returns:
        Total agent invocation count surfaced in the pre-flight notice.
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

    Returns:
        Total agent invocation count for the single-stack-mode run.
    """
    return 2 + stack_count + stack_count


def _shallow_fanout_threshold(config: RunConfig) -> int:
    """Resolve the tiny-diff short-circuit threshold (issue #172, AC7).

    Precedence (highest first), mirroring ``_resolve_backend`` /
    ``_resolved_model`` at ``runner.py:295-326``:

      1. ``RunConfig.shallow_fanout_threshold`` (CLI tier).
      2. ``DaydreamFileConfig.shallow_fanout_threshold`` (file-config scalar).
      3. ``DEFAULT_SHALLOW_FANOUT_THRESHOLD`` (built-in default).

    Uses ``is not None`` checks rather than truthiness so a configured value
    of ``0`` (explicitly disable the short-circuit) is honored.

    Args:
        config: Run configuration carrying the CLI/file-config sources.

    Returns:
        The resolved integer threshold. ``0`` disables the short-circuit.
    """
    if config.shallow_fanout_threshold is not None:
        return config.shallow_fanout_threshold
    file_config = config.file_config
    if file_config is not None and file_config.shallow_fanout_threshold is not None:
        return file_config.shallow_fanout_threshold
    return DEFAULT_SHALLOW_FANOUT_THRESHOLD


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

    Args:
        diff: Unified diff text.

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


def _write_ttt_artifacts(
    deep_dir_path: Path, *, intent_summary: str, alt_issues: list[dict[str, Any]]
) -> tuple[Path, Path]:
    """Persist TTT intent + alternatives to the deep artifact directory.

    Args:
        deep_dir_path: Path to ``.daydream/deep/``.
        intent_summary: Confirmed intent summary from phase_understand_intent.
        alt_issues: Issues returned by phase_alternative_review.

    Returns:
        Tuple of (intent_path, alternatives_path).
    """
    intent_p = _intent_path(deep_dir_path)
    alts_p = _alternatives_path(deep_dir_path)
    intent_p.write_text(intent_summary)
    alts_p.write_text(json.dumps(alt_issues, indent=2))
    return intent_p, alts_p


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


def _candidate_pair_to_json(pair: Any) -> dict[str, Any]:
    """Serialize a CandidatePair dataclass into a JSON-compatible dict."""
    if is_dataclass(pair) and not isinstance(pair, type):
        data = asdict(pair)
    else:
        data = dict(pair)
    # alt_files is a tuple -> convert to list for stable JSON.
    if isinstance(data.get("alt_files"), tuple):
        data["alt_files"] = list(data["alt_files"])
    return data


def _apply_arbiter_verdicts(
    records: list[dict[str, Any]],
    sources: list[str],
    targets: list[int],
    verdicts: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fold arbiter verdicts back into the per-stack record set (#168).

    Each selected record (``targets[k]`` for 1-based ``arb_id = k + 1``) is
    revised in place (severity/confidence/description/rationale updated) when the
    arbiter keeps it, and dropped only on an explicit ``keep:false`` verdict. A
    missing verdict or an ``arb_id`` mismatch fails open: the original record is
    retained unchanged (with a warning), because a record reaches arbitration
    precisely because it is high-severity or contested, so a truncated/lazy
    arbiter response must not silently delete it. Non-selected records pass
    through untouched. ``file``/``line`` are never changed -- the arbiter
    adjudicates, it does not re-target findings.

    Returns:
        New ``(records, sources)`` with explicitly rejected records removed and
        surviving selected records carrying the arbiter's fields. Positional
        alignment between the two lists is preserved.
    """
    import warnings

    revised: dict[int, dict[str, Any]] = {}
    dropped: set[int] = set()
    for offset, record_index in enumerate(targets):
        arb_id = offset + 1
        verdict = verdicts.get(arb_id)
        if verdict is None:
            # Arbiter returned no verdict for this arb_id -- fail open: retain the
            # original record unchanged so a truncated/lazy arbiter response cannot
            # silently delete a high-severity or contested finding. Only an explicit
            # keep=False below drops a record.
            warnings.warn(
                f"Arbiter returned no verdict for arb_id={arb_id} "
                f"(record_index={record_index}); retaining original record unchanged.",
                stacklevel=2,
            )
            continue
        if verdict.get("arb_id") != arb_id:
            # Secondary key guard: the arb_id field in the verdict must match the
            # key we looked it up by.  A mismatch means the arbiter emitted a
            # finding with a wrong arb_id, which would silently bind the verdict
            # to the wrong record.  Warn and fail open (retain the original) rather
            # than mis-apply or drop.
            warnings.warn(
                f"Arbiter verdict arb_id mismatch: expected arb_id={arb_id} "
                f"but verdict contains arb_id={verdict.get('arb_id')!r} "
                f"(record_index={record_index}); retaining original record unchanged.",
                stacklevel=2,
            )
            continue
        if not verdict.get("keep", False):
            dropped.add(record_index)
            continue
        revised[record_index] = {
            **records[record_index],
            "severity": verdict.get("severity", records[record_index].get("severity")),
            "confidence": verdict.get("confidence", records[record_index].get("confidence")),
            "description": verdict.get("description", records[record_index].get("description")),
            "rationale": verdict.get("rationale", records[record_index].get("rationale")),
            "evidence": verdict.get("evidence", records[record_index].get("evidence")),
        }

    new_records: list[dict[str, Any]] = []
    new_sources: list[str] = []
    for i, (record, source) in enumerate(zip(records, sources, strict=True)):
        if i in dropped:
            continue
        new_records.append(revised.get(i, record))
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


async def _step_exploration(ctx: FlowContext) -> None:
    """Exploration pre-scan (D-43)."""
    from daydream.runner import _compute_diff_ref

    config = ctx.config
    target_dir = ctx.work.repo
    daydream_dir = target_dir / ".daydream"
    diff = ctx.data["diff"]
    tier = ctx.data["tier"]

    exploration_dir: Path | None = None
    if not EXPLORATION_AVAILABLE:
        print_warning(
            console,
            "Exploration infrastructure not installed; running deep pipeline "
            "without pre-scan grounding",
        )
    elif config.exploration_context is None:
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
    if EXPLORATION_AVAILABLE and config.exploration_context is not None:
        exploration_dir = config.exploration_context.write_to_dir(
            daydream_dir / "exploration"
        )
    ctx.data["exploration_dir"] = exploration_dir


async def _step_intent(ctx: FlowContext) -> None:
    """TTT intent analysis, grounded by the PR description when it is fresh."""
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
    async with phase_scope(DaydreamPhase.INTENT):
        ctx.data["intent_summary"] = await phase_understand_intent(
            ctx.backend_for("intent"),
            work,
            ctx.data["diff_path"],
            ctx.data["log"],
            ctx.data["branch"],
            exploration_dir=ctx.data["exploration_dir"],
            pr_description=pr_description,
        )


async def _step_alternatives(ctx: FlowContext) -> None:
    """TTT alternative-review (tier-gated) + TTT artifact writes."""
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
            )

    ctx.data["intent_path"], ctx.data["alts_path"] = _write_ttt_artifacts(
        ctx.data["dd"], intent_summary=intent_summary, alt_issues=alt_issues
    )


async def _step_per_stack_reviews(ctx: FlowContext) -> None:
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
        # Resume: reconstruct the expected per-stack output paths on disk.
        from daydream.deep.artifacts import per_stack_review_path

        per_stack_outputs = {
            stack.stack_name: per_stack_review_path(dd, stack.stack_name)
            for stack in stacks
        }
        # Resume also resurrects any prior failure summary so the merge
        # prompt can still note uncovered stacks.
        failures_p = per_stack_failures_path(dd)
        if failures_p.is_file():
            try:
                loaded = json.loads(failures_p.read_text())
                if isinstance(loaded, dict):
                    failed_stacks = {str(k): str(v) for k, v in loaded.items()}
            except json.JSONDecodeError:
                failed_stacks = {}
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
            records_path = per_stack_records_path(dd, stack.stack_name)
            if records_path.is_file():
                expected_paths.append(records_path)
            elif stack.stack_name not in failed_stacks:
                missing_stacks.append(stack.stack_name)
        if missing_stacks:
            print_error(
                console,
                "Missing Per-Stack Records",
                "Missing parsed records for: "
                + ", ".join(sorted(missing_stacks)),
            )
            return Stop(1)
        for records_path in sorted(expected_paths):
            records = json.loads(records_path.read_text())
            per_stack_records_paths.append(records_path)
            source_name = records_path.name
            all_records.extend(records)
            record_sources.extend(source_name for _ in records)
    else:
        # Pre-merge parse pass (D-21). Sort by stack_name so merge input
        # order is independent of parallel-task completion order, keeping
        # the merge prompt and global issue numbering reproducible.
        for stack_name, output_path in sorted(per_stack_outputs.items()):
            # Language stacks carry severity so the scoped arbiter can
            # select high/contested findings (#168). The structural
            # meta-stack keeps the severity-free FEEDBACK_SCHEMA: it is
            # high-conviction by construction and is partitioned out of
            # arbitration/dedup below, defaulting to high at merge.
            record_schema = (
                FEEDBACK_SCHEMA if stack_name == STRUCTURE_STACK_NAME else PER_STACK_RECORD_SCHEMA
            )
            async with phase_scope(DaydreamPhase.PARSE):
                records = await phase_parse_feedback(
                    ctx.backend_for("parse"),
                    ctx.work,
                    input_path=output_path,
                    output_schema=record_schema,
                )
            records_path = per_stack_records_path(dd, stack_name)
            records_path.write_text(json.dumps(records, indent=2))
            per_stack_records_paths.append(records_path)
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
    arbiter_marker = arbiter_complete_path(dd)
    if config.start_at != "merge" or not arbiter_marker.is_file():
        arbiter_targets = select_arbiter_targets(all_records, record_sources)
        if arbiter_targets:
            async with phase_scope(DaydreamPhase.DEEP, stage="arbiter"):
                verdicts = await phase_arbiter_review(
                    ctx.backend_for("arbiter"),
                    ctx.work,
                    selected_records=[all_records[i] for i in arbiter_targets],
                    diff_path=ctx.data["diff_path"],
                    intent_path=ctx.data["intent_path"],
                    alternatives_path=ctx.data["alts_path"],
                    exploration_dir=ctx.data["exploration_dir"],
                )
            all_records, record_sources = _apply_arbiter_verdicts(
                all_records, record_sources, arbiter_targets, verdicts
            )
            _rewrite_stack_records(
                dd, ctx.data["records_paths"], all_records, record_sources
            )
        arbiter_marker.write_text("")
    ctx.data["records"] = all_records
    ctx.data["record_sources"] = record_sources


async def _step_cross_stack_merge(ctx: FlowContext) -> None:
    """Dedup pre-filter (D-27) + cross-stack merge (D-23..D-26)."""
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
    )


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
        from daydream.deep.artifacts import merged_report_path

        deep_copy = merged_report_path(dd)
        if deep_copy.exists():
            merged_report.write_text(deep_copy.read_text())

    ctx.data["merged_report"] = merged_report
    ctx.data["items_file"] = items_file
    return None


async def _step_findings_out(ctx: FlowContext) -> Stop:
    """Two-phase findings artifact (Phase A): emit the strict-schema artifact and STOP."""
    from daydream.runner import _emit_findings_from_items

    items_file: Path = ctx.data["items_file"]
    findings_items: list[dict[str, Any]] = json.loads(items_file.read_text())["items"]
    return Stop(_emit_findings_from_items(ctx.work.repo, ctx.config, findings_items))


async def _step_post_review(ctx: FlowContext) -> None:
    """Offer to post findings as inline PR review comments."""
    from daydream.pr_review import post_review_to_pr_from_report

    await post_review_to_pr_from_report(
        ctx.work.repo, merged_items_path(ctx.data["dd"]), console=console
    )


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
            merged_items_path=merged_items_path(dd),
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
    except git_ops.GitError as exc:
        print_warning(console, f"Could not snapshot tree before fixes: {exc}")
        pre_fix_snapshot = None
        pre_fix_untracked = set()
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
    async with phase_scope(DaydreamPhase.FIX):
        fix_failures = await phase_fix_parallel(
            ctx.backend_for("fix"),
            work,
            items,
            intent_path=intent_p if (intent_grounded_this_run and intent_p.exists()) else None,
        )
    # Capture daydream's proposed diff (pre-fix tree → post-fix worktree)
    # NOW, before the fix-failure and test-failure early returns below, so
    # a run that generated a recommendation always archives it — even when
    # tests fail or a fix group is reverted. Best-effort; never raises.
    git_ops.capture_recommended_patch_with_base(
        work.repo, pre_fix_snapshot, pre_fix_head, daydream_dir / "recommended.patch"
    )
    fix_failures_p = fix_failures_path(dd)
    if fix_failures:
        # Persist so the archive marks the run "partial" instead of
        # "complete" -- the tree is no longer a clean success.
        fix_failures_p.write_text(json.dumps(fix_failures, indent=2, sort_keys=True))
        _protect_tree_after_fix_failures(
            work,
            target_dir,
            fix_failures,
            snapshot=pre_fix_snapshot,
            pre_untracked=pre_fix_untracked,
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
            f"{len(fix_failures)} fix group(s) failed: {sorted(fix_failures)}; "
            "partial edits reverted (patches saved under .daydream/partial-fixes/).",
        )
        return Stop(1)
    # Fresh successful fix supersedes any stale failures record.
    if fix_failures_p.exists():
        fix_failures_p.unlink()
    stale_leftover_p = fix_leftover_untracked_path(dd)
    if stale_leftover_p.exists():
        stale_leftover_p.unlink()
    return None


async def _step_test(ctx: FlowContext) -> Stop | None:
    """Post-fix test validation."""
    async with phase_scope(DaydreamPhase.TEST):
        passed, _retries = await phase_test_and_heal(
            ctx.backend_for("test"), ctx.work, feedback_items=ctx.data["items"]
        )
    if not passed:
        print_warning(console, "Tests failed after fix attempt.")
        return Stop(1)
    return None


async def _step_commit(ctx: FlowContext) -> None:
    """Commit-and-push the applied fixes."""
    # phase_commit_push runs as part of the fix/commit cycle — reuse
    # the fix backend (no separate "commit" phase identifier).
    await phase_commit_push(ctx.backend_for("fix"), ctx.work)


async def run_deep(config: RunConfig, work: WorkContext) -> int:
    """Execute the deep-review pipeline (D-07).

    Composes exploration pre-scan, TTT, per-stack fan-out, per-stack parse,
    dedup pre-filter, cross-stack merge, and the optional fix gate into a
    single async flow. Supports stage-granular resume via
    ``config.start_at in ("ttt", "per-stack", "merge", "fix")``.

    Args:
        config: Run configuration. ``config.shallow`` must be False (deep is
            the default); ``config.start_at`` drives resume behavior.
            ``config.identity`` carries the GitHub identity set by
            :func:`daydream.runner.run`.
        work: Resolved working environment for the run.

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    # Late imports to avoid circular dependency with runner.
    from daydream import git_ops
    from daydream.backends import Backend
    from daydream.git_ops import GitError, GitTimeoutError
    from daydream.phases import _git_branch, _git_log
    from daydream.runner import (
        _make_archive_callback,
        _resolved_backend_name,
    )

    # Cache one Backend instance per (backend_name, resolved_model) so phases that
    # resolve to the same model share an instance and differing models stay isolated.
    backend_cache: dict[tuple[str, str | None], Backend] = {}

    target_dir = work.repo

    # Preamble (mirrors run_trust).
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
    # it at both the exploration gate and the alternatives gate below.
    tier = select_tier(count_changed_files(diff))
    dd = deep_dir(target_dir)

    session_id = str(uuid.uuid4())
    trajectory_path = config.trajectory_path or default_trajectory_path(target_dir, session_id)
    async with TrajectoryRecorder(
        path=trajectory_path,
        run_flow=DaydreamRunFlow.DEEP,
        target_dir=target_dir,
        agent_model_name="",
        session_id=session_id,
        explicit_path=config.trajectory_path is not None,
        pr_number=config.pr_number,
        pr_repo=config.pr_repo,
        on_write=_make_archive_callback(config, target_dir, work),
    ):
        console.print()
        print_info(console, f"Target directory: {target_dir}")
        print_info(console, f"Branch: {branch}")
        print_info(console, f"Default backend: {_resolved_backend_name(config, 'review')}")
        # Bot logins look like ``my-app[bot]``; escape so Rich doesn't eat the brackets.
        print_info(console, f"GitHub identity: {escape_markup(config.identity)}")
        console.print()

        # Resume gate (D-34, D-36, D-37).
        if config.start_at in ("per-stack", "merge", "fix"):
            try:
                check_deep_artifacts(config.start_at, dd)
            except FileNotFoundError as exc:
                print_error(console, "Missing Deep Artifact", str(exc))
                return 1

        # Stack detection (from diff file list).
        changed_files = _diff_changed_files(diff)
        installed = get_installed_skills()
        # Optimistic fallback when detection fails: SDK-level MissingSkillError is
        # still caught downstream in phase_per_stack_reviews, so preserving the
        # pre-D-16 behavior is safer than routing everything to generic.
        skill_availability = installed if installed is not None else get_registry().stack_keys()
        stacks = detect_stacks(changed_files, skill_availability=skill_availability)
        # Issue #172 — tiny-diff short-circuit. When the diff is small enough
        # (≤ SHALLOW_FANOUT_THRESHOLD files), collapse the per-language fan-out
        # to a single combined assignment and skip merge+arbiter downstream.
        # ``single_stack_mode`` is recomputed here (top of run_deep) so a
        # ``--start-at merge``/``--start-at fix`` resume on a tiny diff re-enters
        # the same bypass branch rather than routing to the absent merge agent.
        threshold = _shallow_fanout_threshold(config)
        if threshold > 0 and 0 < len(changed_files) <= threshold:
            stacks, _ = _collapse_stacks_for_tiny_diff(
                stacks, changed_files, threshold=threshold
            )
            single_stack_mode = True
        else:
            single_stack_mode = False

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
        )

        # Flow context (steps communicate through ctx.data). The imperative
        # step calls below become registered flow steps run via the engine in
        # a later task; ctx shares run_deep's backend cache so instance-sharing
        # semantics are unchanged during the migration.
        ctx = FlowContext(
            config=config,
            work=work,
            registry=get_registry(),
            data={
                "diff": diff,
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

        # Exploration pre-scan (D-43).
        await _step_exploration(ctx)

        try:
            if config.start_at not in ("per-stack", "merge", "fix"):
                await _step_intent(ctx)
                await _step_alternatives(ctx)

            await _step_per_stack_reviews(ctx)

            if config.start_at != "fix":
                signal = await _step_per_stack_parse(ctx)
                if isinstance(signal, Stop):
                    return signal.exit_code

                if not single_stack_mode:
                    await _step_arbiter(ctx)
                    await _step_cross_stack_merge(ctx)
                else:
                    await _step_single_stack_merge(ctx)

            signal = await _step_load_items(ctx)
            if isinstance(signal, Stop):
                return signal.exit_code

            # Two-phase findings artifact (Phase A). When --findings-out is set,
            # convert the canonical merged items into the strict-schema artifact and
            # STOP: never post to the PR and never apply fixes. Phase B posts later.
            if config.findings_out is not None:
                return (await _step_findings_out(ctx)).exit_code

            # `post_review_to_pr_from_report` is a non-idempotent GitHub write, so
            # `--start-at fix` (resume after the merged report) must skip it to
            # avoid duplicate inline reviews on reruns.
            if config.start_at != "fix":
                await _step_post_review(ctx)

            signal = await _step_fix_gate(ctx)
            if isinstance(signal, Stop):
                return signal.exit_code

            await _step_verify(ctx)

            signal = await _step_fix(ctx)
            if isinstance(signal, Stop):
                return signal.exit_code

            signal = await _step_test(ctx)
            if isinstance(signal, Stop):
                return signal.exit_code

            await _step_commit(ctx)
            return 0

        finally:
            exploration_cleanup = target_dir / ".daydream" / "exploration"
            if exploration_cleanup.is_dir():
                # Best-effort: a raised rmtree here would escape the finally and
                # replace the run's real exit code with a cleanup exception.
                try:
                    shutil.rmtree(exploration_cleanup)
                except OSError as exc:
                    print_warning(
                        console,
                        f"Failed to clean up exploration artifacts at "
                        f"{exploration_cleanup}: {exc}",
                    )
            # .daydream/deep/ is preserved per RESEARCH.md Open Question 1 so
            # subsequent --start-at resumes can find the artifacts they need.
