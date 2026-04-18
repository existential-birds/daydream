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
import re
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.agent import console
from daydream.config import REVIEW_OUTPUT_FILE, SKILL_MAP
from daydream.deep.artifacts import (
    alternatives_path as _alternatives_path,
)
from daydream.deep.artifacts import (
    check_deep_artifacts,
    dedup_candidates_path,
    deep_dir,
    per_stack_records_path,
)
from daydream.deep.artifacts import (
    intent_path as _intent_path,
)
from daydream.deep.dedup import build_dedup_candidates
from daydream.deep.detection import StackAssignment, detect_stacks
from daydream.phases import (
    phase_alternative_review,
    phase_commit_push,
    phase_cross_stack_merge,
    phase_fix,
    phase_parse_feedback,
    phase_per_stack_reviews,
    phase_test_and_heal,
    phase_understand_intent,
)
from daydream.ui import (
    print_error,
    print_info,
    print_preflight_notice,
    print_stage_progress,
    print_success,
    print_warning,
    prompt_user,
)

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

# Codex backend class used for isinstance() in the pre-flight notice
# (D-31 cost_usd=None caveat). Imported here so tests can monkeypatch it.
from daydream.backends.codex import CodexBackend  # noqa: E402

# Match git diff file headers to extract the list of changed files.
_DIFF_FILE_HEADER = re.compile(r"^(?:diff --git a/(\S+)|\+\+\+ b/(\S+))", re.MULTILINE)

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
    + N per-stack parse passes + 1 cross-stack merge. The fix-gate agents
    are user-gated and excluded from the pre-flight estimate.

    Args:
        stack_count: Number of detected stack assignments (including the
            generic-fallback bucket when present).

    Returns:
        Total agent invocation count surfaced in the pre-flight notice.
    """
    return 2 + stack_count + stack_count + 1


def _diff_changed_files(diff: str) -> list[str]:
    """Extract changed files from a unified diff.

    Prefers ``+++ b/<path>`` lines over ``diff --git a/<path>`` because
    the former survives renames where the a-side path no longer exists.

    Args:
        diff: Unified diff text.

    Returns:
        Unique, insertion-ordered list of changed file paths (excluding
        ``/dev/null`` sentinels).
    """
    files: list[str] = []
    for match in _DIFF_FILE_HEADER.finditer(diff):
        path = match.group(1) or match.group(2)
        if path and path != "/dev/null" and path not in files:
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


async def run_deep(config: RunConfig, target_dir: Path) -> int:
    """Execute the deep-review pipeline (D-07).

    Composes exploration pre-scan, TTT, per-stack fan-out, per-stack parse,
    dedup pre-filter, cross-stack merge, and the optional fix gate into a
    single async flow. Supports stage-granular resume via
    ``config.start_at in ("ttt", "per-stack", "merge", "fix")``.

    Args:
        config: Run configuration. ``config.deep`` must be True;
            ``config.start_at`` drives resume behavior.
        target_dir: Resolved target directory path (repo root).

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    # Late imports to avoid circular dependency with runner.
    from daydream.phases import _git_branch, _git_diff, _git_log
    from daydream.runner import _compute_diff_ref, _resolve_backend
    from daydream.ui import phase_subtitle, print_dim, print_phase_hero

    backend = _resolve_backend(config, "review")

    # ------ Preamble (mirrors run_trust) ------
    diff = _git_diff(target_dir, exclude=config.ignore_paths)
    log = _git_log(target_dir)
    branch = _git_branch(target_dir)

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
    dd = deep_dir(target_dir)

    console.print()
    print_info(console, f"Target directory: {target_dir}")
    print_info(console, f"Branch: {branch}")
    print_info(console, f"Model: {config.model or '<backend-default>'}")
    console.print()

    # ------ Resume gate (D-34, D-36, D-37) ------
    if config.start_at in ("per-stack", "merge", "fix"):
        try:
            check_deep_artifacts(config.start_at, dd)
        except FileNotFoundError as exc:
            print_error(console, "Missing Deep Artifact", str(exc))
            return 1

    # ------ Stack detection (from diff file list) ------
    changed_files = _diff_changed_files(diff)
    stacks = detect_stacks(changed_files, skill_availability=set(SKILL_MAP.keys()))

    # ------ Pre-flight notice (D-30, D-31) ------
    stack_lines = [_stack_preflight_line(s) for s in stacks]
    print_preflight_notice(
        console,
        stages=_PIPELINE_STAGE_NAMES,
        stack_lines=stack_lines,
        agent_count=total_agent_count(len(stacks)),
        codex_in_use=isinstance(backend, CodexBackend),
        exploration_available=EXPLORATION_AVAILABLE,
    )

    # ------ Exploration pre-scan (D-43) ------
    exploration_dir: Path | None = None
    if not EXPLORATION_AVAILABLE:
        print_warning(
            console,
            "Exploration infrastructure not installed; running deep pipeline "
            "without pre-scan grounding",
        )
    elif config.exploration_context is None:
        tier = select_tier(count_changed_files(diff))
        if tier == "skip":
            print_dim(console, "Skipping exploration -- trivial diff")
            config.exploration_context = ExplorationContext()
        else:
            print_phase_hero(console, "EXPLORE", phase_subtitle("EXPLORE"))
            config.exploration_context = await safe_explore(
                pre_scan,
                backend,
                target_dir,
                diff,
                config.exploration_depth,
                diff_ref=_compute_diff_ref(target_dir),
            )
    if EXPLORATION_AVAILABLE and config.exploration_context is not None:
        exploration_dir = config.exploration_context.write_to_dir(
            daydream_dir / "exploration"
        )

    try:
        intent_p = _intent_path(dd)
        alts_p = _alternatives_path(dd)

        # ------ Stage 1 + 2: TTT ------
        if config.start_at not in ("per-stack", "merge", "fix"):
            print_stage_progress(console, 1, 5, _PIPELINE_STAGE_NAMES[0])
            intent_summary = await phase_understand_intent(
                backend,
                target_dir,
                diff_path,
                log,
                branch,
                exploration_dir=exploration_dir,
            )

            print_stage_progress(console, 2, 5, _PIPELINE_STAGE_NAMES[1])
            alt_issues = await phase_alternative_review(
                backend,
                target_dir,
                diff_path,
                intent_summary,
                exploration_dir=exploration_dir,
            )

            intent_p, alts_p = _write_ttt_artifacts(
                dd, intent_summary=intent_summary, alt_issues=alt_issues
            )

        # ------ Stage 3: per-stack fan-out ------
        if config.start_at not in ("merge", "fix"):
            print_stage_progress(console, 3, 5, _PIPELINE_STAGE_NAMES[2])
            per_stack_outputs = await phase_per_stack_reviews(
                backend,
                target_dir,
                stacks,
                diff_path=diff_path,
                intent_path=intent_p,
                alternatives_path=alts_p,
                exploration_dir=exploration_dir,
            )
        else:
            # Resume: reconstruct the expected per-stack output paths on disk.
            from daydream.deep.artifacts import per_stack_review_path

            per_stack_outputs = {
                stack.stack_name: per_stack_review_path(dd, stack.stack_name)
                for stack in stacks
            }

        # ------ Stage 4: pre-merge parse + dedup + cross-stack merge ------
        if config.start_at != "fix":
            print_stage_progress(console, 4, 5, _PIPELINE_STAGE_NAMES[3])

            # Pre-merge parse pass (D-21).
            per_stack_records_paths: list[Path] = []
            all_records: list[dict[str, Any]] = []
            for stack_name, output_path in per_stack_outputs.items():
                records = await phase_parse_feedback(
                    backend, target_dir, input_path=output_path
                )
                records_path = per_stack_records_path(dd, stack_name)
                records_path.write_text(json.dumps(records, indent=2))
                per_stack_records_paths.append(records_path)
                all_records.extend(records)

            # Dedup pre-filter (D-27).
            alt_issues_for_dedup: list[dict[str, Any]] = (
                json.loads(alts_p.read_text()) if alts_p.exists() else []
            )
            pairs = build_dedup_candidates(all_records, alt_issues_for_dedup)
            dedup_p = dedup_candidates_path(dd)
            dedup_p.write_text(
                json.dumps([_candidate_pair_to_json(p) for p in pairs], indent=2)
            )

            # Cross-stack merge (D-23..D-26).
            await phase_cross_stack_merge(
                backend,
                target_dir,
                per_stack_records_paths=per_stack_records_paths,
                intent_path=intent_p,
                alternatives_path=alts_p,
                dedup_candidates_path=dedup_p,
                exploration_dir=exploration_dir,
            )

        # ------ Stage 5: optional fix gate (D-28, D-29) ------
        print_stage_progress(console, 5, 5, _PIPELINE_STAGE_NAMES[4])
        merged_report = target_dir / REVIEW_OUTPUT_FILE
        if not merged_report.exists():
            print_error(
                console,
                "Missing Merged Report",
                f"Expected merged report at {merged_report}",
            )
            return 1

        answer = prompt_user(console, "Apply fixes now? [y/N]", "n")
        if answer.strip().lower() not in ("y", "yes"):
            print_success(console, f"Report written to {merged_report}. Exiting.")
            return 0

        items = await phase_parse_feedback(backend, target_dir)
        if not items:
            print_success(console, "No actionable items after parse -- done.")
            return 0

        for idx, item in enumerate(items, start=1):
            await phase_fix(backend, target_dir, item, idx, len(items))

        passed, _retries = await phase_test_and_heal(backend, target_dir)
        if not passed:
            print_warning(console, "Tests failed after fix attempt.")
            return 1

        await phase_commit_push(backend, target_dir)
        return 0

    finally:
        exploration_cleanup = target_dir / ".daydream" / "exploration"
        if exploration_cleanup.is_dir():
            shutil.rmtree(exploration_cleanup)
        # .daydream/deep/ is preserved per RESEARCH.md Open Question 1 so
        # subsequent --start-at resumes can find the artifacts they need.
