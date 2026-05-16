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
import re
import shutil
import uuid
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
    per_stack_failures_path,
    per_stack_records_path,
)
from daydream.deep.artifacts import (
    intent_path as _intent_path,
)
from daydream.deep.dedup import build_dedup_candidates, build_record_dedup_candidates
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
    phase_verify_recommendations,
)
from daydream.trajectory import DaydreamRunFlow, TrajectoryRecorder, default_trajectory_path
from daydream.ui import (
    print_error,
    print_info,
    print_preflight_notice,
    print_stage_progress,
    print_success,
    print_verification_summary,
    print_warning,
    prompt_user,
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

# Codex backend class used for isinstance() in the pre-flight notice
# (D-31 cost_usd=None caveat). Imported here so tests can monkeypatch it.
from daydream.backends.codex import CodexBackend  # noqa: E402

# Per-file block splitter (splits the unified diff at each `diff --git` header).
_DIFF_BLOCK_SPLIT = re.compile(r"^(?=diff --git )", re.MULTILINE)
# `+++ ` and `--- ` file headers inside a single block.
_DIFF_PLUS_HEADER = re.compile(r"^\+\+\+ (\S+)", re.MULTILINE)
_DIFF_MINUS_HEADER = re.compile(r"^--- (\S+)", re.MULTILINE)
# Fallback header for binary / mode-only diffs that lack `--- / +++`.
_DIFF_GIT_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)")

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


def get_installed_skills() -> set[str] | None:
    """Detect which Beagle review-skill plugins are installed.

    Reads the Claude Code plugin registry at
    ``$CLAUDE_CONFIG_DIR/plugins/installed_plugins.json`` (default
    ``~/.claude``) and maps installed plugin names back to ``SKILL_MAP``
    stack keys. Each stack's Beagle skill lives in a plugin named
    ``beagle-<stack>``; a stack is considered "installed" iff that plugin
    is present.

    Returns:
        Set of installed stack keys (subset of ``SKILL_MAP.keys()``), or
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
    installed: set[str] = set()
    for stack_key, skill_invocation in SKILL_MAP.items():
        # SKILL_MAP values are "<plugin-name>:<skill-name>".
        plugin_prefix = skill_invocation.split(":", 1)[0]
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

    Args:
        diff: Unified diff text.

    Returns:
        Unique, insertion-ordered list of changed file paths (excluding
        ``/dev/null`` sentinels).
    """

    def _strip_prefix(path: str, prefix: str) -> str:
        return path[len(prefix) :] if path.startswith(prefix) else path

    files: list[str] = []
    for block in _DIFF_BLOCK_SPLIT.split(diff):
        if not block.startswith("diff --git "):
            continue
        path: str | None = None
        plus = _DIFF_PLUS_HEADER.search(block)
        if plus and plus.group(1) != "/dev/null":
            path = _strip_prefix(plus.group(1), "b/")
        if path is None:
            minus = _DIFF_MINUS_HEADER.search(block)
            if minus and minus.group(1) != "/dev/null":
                path = _strip_prefix(minus.group(1), "a/")
        if path is None:
            git = _DIFF_GIT_HEADER.match(block)
            if git:
                path = git.group(2)
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


async def run_deep(config: RunConfig, work: WorkContext) -> int:
    """Execute the deep-review pipeline (D-07).

    Composes exploration pre-scan, TTT, per-stack fan-out, per-stack parse,
    dedup pre-filter, cross-stack merge, and the optional fix gate into a
    single async flow. Supports stage-granular resume via
    ``config.start_at in ("ttt", "per-stack", "merge", "fix")``.

    Args:
        config: Run configuration. ``config.shallow`` must be False (deep is
            the default); ``config.start_at`` drives resume behavior.
        work: Resolved working environment for the run.

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    # Late imports to avoid circular dependency with runner.
    from daydream import git_ops
    from daydream.backends import Backend
    from daydream.git_ops import GitError
    from daydream.phases import _git_branch, _git_log
    from daydream.runner import _compute_diff_ref, _make_archive_callback, _resolve_backend
    from daydream.ui import phase_subtitle, print_dim, print_phase_hero

    # Per-phase backends are resolved on demand via `_resolve_backend(config,
    # phase, backend_cache)`. The cache reuses one Backend instance per
    # (backend_name, resolved_model) tuple so identical phases share an
    # instance while phases that resolve to different models stay isolated.
    backend_cache: dict[tuple[str, str | None], Backend] = {}

    target_dir = work.repo

    # ------ Preamble (mirrors run_trust) ------
    try:
        diff = git_ops.diff(work.repo, work.base_branch, exclude=config.ignore_paths)
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
        on_write=_make_archive_callback(config, target_dir),
    ):
        console.print()
        print_info(console, f"Target directory: {target_dir}")
        print_info(console, f"Branch: {branch}")
        print_info(console, f"Default backend: {config.backend}")
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
        installed = get_installed_skills()
        # Optimistic fallback when detection fails: SDK-level MissingSkillError is
        # still caught downstream in phase_per_stack_reviews, so preserving the
        # pre-D-16 behavior is safer than routing everything to generic.
        skill_availability = installed if installed is not None else set(SKILL_MAP.keys())
        stacks = detect_stacks(changed_files, skill_availability=skill_availability)

        # ------ Pre-flight notice (D-30, D-31) ------
        stack_lines = [_stack_preflight_line(s) for s in stacks]
        # Review-centric preflight: the cost_usd=None caveat fires when the
        # review-phase backend is Codex. Other per-phase backends may differ
        # but the notice is anchored on the review stage.
        review_backend_for_preflight = _resolve_backend(config, "review", backend_cache)
        print_preflight_notice(
            console,
            stages=_PIPELINE_STAGE_NAMES,
            stack_lines=stack_lines,
            agent_count=total_agent_count(len(stacks)),
            codex_in_use=isinstance(review_backend_for_preflight, CodexBackend),
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
                explore_backend = _resolve_backend(config, "exploration", backend_cache)
                print_dim(console, f"Exploration model: {explore_backend.model}")
                config.exploration_context = await safe_explore(
                    pre_scan,
                    explore_backend,
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
                    _resolve_backend(config, "intent", backend_cache),
                    work,
                    diff_path,
                    log,
                    branch,
                    exploration_dir=exploration_dir,
                )

                print_stage_progress(console, 2, 5, _PIPELINE_STAGE_NAMES[1])
                alt_issues = await phase_alternative_review(
                    _resolve_backend(config, "wonder", backend_cache),
                    work,
                    diff_path,
                    intent_summary,
                    exploration_dir=exploration_dir,
                )

                intent_p, alts_p = _write_ttt_artifacts(
                    dd, intent_summary=intent_summary, alt_issues=alt_issues
                )

            # ------ Stage 3: per-stack fan-out ------
            failed_stacks: dict[str, str] = {}
            if config.start_at not in ("merge", "fix"):
                print_stage_progress(console, 3, 5, _PIPELINE_STAGE_NAMES[2])
                per_stack_outputs, failed_stacks = await phase_per_stack_reviews(
                    _resolve_backend(config, "review", backend_cache),
                    work,
                    stacks,
                    diff_path=diff_path,
                    intent_path=intent_p,
                    alternatives_path=alts_p,
                    exploration_dir=exploration_dir,
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

            # ------ Stage 4: pre-merge parse + dedup + cross-stack merge ------
            if config.start_at != "fix":
                print_stage_progress(console, 4, 5, _PIPELINE_STAGE_NAMES[3])

                per_stack_records_paths: list[Path] = []
                all_records: list[dict[str, Any]] = []
                record_sources: list[str] = []
                if config.start_at == "merge":
                    # Resume: validate a records file exists for every detected
                    # stack (except ones explicitly in `failed_stacks`). A bare
                    # `dd.glob` could silently drop a current stack whose prior
                    # records file is absent, producing an authoritative-looking
                    # merged report that is missing an entire language bucket.
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
                        return 1
                    for records_path in sorted(expected_paths):
                        records = json.loads(records_path.read_text())
                        per_stack_records_paths.append(records_path)
                        # Derive stack name from the records filename
                        # (e.g. "stack-python-records.json" -> "stack-python-records.json").
                        source_name = records_path.name
                        all_records.extend(records)
                        record_sources.extend(source_name for _ in records)
                else:
                    # Pre-merge parse pass (D-21).
                    # Sort by stack_name so merge input order doesn't depend on the
                    # completion order of the parallel per-stack tasks that
                    # populated `per_stack_outputs` -- keeps the merge prompt and
                    # global issue numbering reproducible across runs.
                    for stack_name, output_path in sorted(per_stack_outputs.items()):
                        records = await phase_parse_feedback(
                            _resolve_backend(config, "parse", backend_cache),
                            work,
                            input_path=output_path,
                        )
                        records_path = per_stack_records_path(dd, stack_name)
                        records_path.write_text(json.dumps(records, indent=2))
                        per_stack_records_paths.append(records_path)
                        all_records.extend(records)
                        record_sources.extend(stack_name for _ in records)

                # Dedup pre-filter (D-27).
                alt_issues_for_dedup: list[dict[str, Any]] = (
                    json.loads(alts_p.read_text()) if alts_p.exists() else []
                )
                pairs = build_dedup_candidates(all_records, alt_issues_for_dedup)
                record_pairs = build_record_dedup_candidates(all_records, sources=record_sources)
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
                    _resolve_backend(config, "merge", backend_cache),
                    work,
                    per_stack_records_paths=per_stack_records_paths,
                    intent_path=intent_p,
                    alternatives_path=alts_p,
                    dedup_candidates_path=dedup_p,
                    exploration_dir=exploration_dir,
                    failed_stacks=failed_stacks or None,
                )

            # ------ Stage 5: optional fix gate (D-28, D-29) ------
            print_stage_progress(console, 5, 5, _PIPELINE_STAGE_NAMES[4])
            merged_report = target_dir / REVIEW_OUTPUT_FILE

            # Recover from deep-dir artifact when the canonical file is absent
            # (e.g. agent wrote to .daydream/deep/ but Python copy didn't run
            # during a --start-at fix resume).
            if not merged_report.exists():
                from daydream.deep.artifacts import merged_report_path

                deep_copy = merged_report_path(dd)
                if deep_copy.exists():
                    merged_report.write_text(deep_copy.read_text())

            if not merged_report.exists():
                print_error(
                    console,
                    "Missing Merged Report",
                    f"Expected merged report at {merged_report}",
                )
                return 1

            # Recommendation verification (issue #83). Runs unconditionally as
            # a sub-step of the fix gate, so a `--start-at fix` resume still
            # produces verdicts. The verifier is read-only and idempotent;
            # writes `recommendation-verdicts.json` inside `dd`. Must precede
            # both `post_review_to_pr_from_report` and the y/N gate so verdicts
            # are available regardless of resume entry point.
            verdicts_file = await phase_verify_recommendations(
                _resolve_backend(config, "verify", backend_cache),
                work,
                merged_report_path=merged_report,
                deep_dir=dd,
            )
            print_verification_summary(console, verdicts_file)

            # Offer to post findings as inline PR review comments.
            # `post_review_to_pr_from_report` is a non-idempotent GitHub write, so
            # `--start-at fix` (resume after the merged report) must skip it to
            # avoid duplicate inline reviews on reruns.
            if config.start_at != "fix":
                from daydream.pr_review import post_review_to_pr_from_report

                await post_review_to_pr_from_report(
                    target_dir, merged_report, console=console
                )

            answer = prompt_user(console, "Apply fixes now? [y/N]", "n")
            if answer.strip().lower() not in ("y", "yes"):
                print_success(console, f"Report written to {merged_report}. Exiting.")
                return 0

            items = await phase_parse_feedback(
                _resolve_backend(config, "parse", backend_cache), work
            )
            if not items:
                print_success(console, "No actionable items after parse -- done.")
                return 0

            # Attach verifier verdicts to feedback items by `id`. `phase_fix`
            # already reads `verifier_verdict` / `evidence` keys (advisory) and
            # augments its prompt when present; items without a matching
            # verdict are left untouched.
            try:
                verdicts_payload = json.loads(verdicts_file.read_text())
            except (OSError, json.JSONDecodeError):
                verdicts_payload = {"verdicts": []}
            verdict_lookup: dict[int, dict[str, str]] = {}
            for entry in verdicts_payload.get("verdicts", []) or []:
                if not isinstance(entry, dict):
                    continue
                issue_id = entry.get("issue_id")
                if not isinstance(issue_id, int):
                    continue
                verdict_lookup[issue_id] = {
                    "verdict": entry.get("verdict", ""),
                    "evidence": entry.get("evidence", ""),
                }
            for item in items:
                item_id = item.get("id")
                if not isinstance(item_id, int):
                    continue
                match = verdict_lookup.get(item_id)
                if match is not None:
                    item["verifier_verdict"] = match["verdict"]
                    item["evidence"] = match["evidence"]

            for idx, item in enumerate(items, start=1):
                await phase_fix(
                    _resolve_backend(config, "fix", backend_cache),
                    work,
                    item,
                    idx,
                    len(items),
                )

            passed, _retries = await phase_test_and_heal(
                _resolve_backend(config, "test", backend_cache), work
            )
            if not passed:
                print_warning(console, "Tests failed after fix attempt.")
                return 1

            # phase_commit_push runs as part of the fix/commit cycle — reuse
            # the fix backend (no separate "commit" phase identifier).
            await phase_commit_push(
                _resolve_backend(config, "fix", backend_cache), work
            )
            return 0

        finally:
            exploration_cleanup = target_dir / ".daydream" / "exploration"
            if exploration_cleanup.is_dir():
                # Best-effort cleanup: an rmtree failure here would otherwise
                # escape the finally and replace the run's real exit code with a
                # cleanup exception, hiding the actual outcome from the caller.
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
