"""Exploration orchestrator.

Provides two exploration entries:

- ``pre_scan`` is diff-scoped. Modest default changes use static mapping and
  bounded guideline capture; other changes add tiered model specialists.
- ``repo_scan`` is repo-scoped and diff-less. It samples tracked files and runs
  only the repo-survey specialist to discover repository conventions.
"""

from __future__ import annotations

import re
import stat
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeAlias

from daydream import git_ops
from daydream import review_profile as _rp
from daydream.backends import effective_fanout_concurrency
from daydream.config import DEFAULT_TOOL_CALL_BUDGET, DEFAULT_WALL_BUDGET_S
from daydream.exploration import (
    Convention,
    Dependency,
    ExplorationContext,
    FileInfo,
    merge_contexts,
)
from daydream.prompt_budget import truncate_utf8_to_budget
from daydream.prompts.exploration_subagents import (
    DEPENDENCY_TRACER_SCHEMA,
    PATTERN_SCANNER_SCHEMA,
    TEST_MAPPER_SCHEMA,
    build_dependency_tracer_prompt,
    build_pattern_scanner_prompt,
    build_repo_survey_prompt,
    build_test_mapper_prompt,
    mapping_source_files,
)
from daydream.review_budget import ReviewLimits
from daydream.review_evidence import FinalizationContext
from daydream.run_context import RunContext, bind_resolved_run_context, resolve_run_context
from daydream.trajectory import (
    DaydreamPhase,
    DispatchHandle,
    LifecycleReasonCode,
    LifecycleStatus,
    dispatch_scope,
    finish_partial_or_failed,
    get_current_recorder,
    maybe_fork,
)
from daydream.tree_sitter_index import _parse_diff_name_status, detect_affected_files

if TYPE_CHECKING:
    from pathlib import Path

    from daydream.backends import Backend


Tier: TypeAlias = Literal["skip", "single", "parallel"]

# This regex parses git's own diff header output (not source code), so a
# regex is the right tool here per D-04 (no tree-sitter for non-source text).
_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+) b/", re.MULTILINE)

_SPECIALIST_TIMEOUT_SECONDS = 300  # 5 minutes

# Pi's dependency mapping can still be productive after two minutes. Allow
# synthesis time as well as investigation, with outer grace for stream cleanup.
_PRE_SCAN_REVIEW_LIMITS = ReviewLimits(300, 120, 16)
_PRE_SCAN_TIMEOUT_SECONDS = (
    _PRE_SCAN_REVIEW_LIMITS.investigation_s + _PRE_SCAN_REVIEW_LIMITS.finalization_s + 30
)

# Cap subagents at 50 turns: on large repos they otherwise exhaust their
# context window and lose track of the task (D-06 graceful degradation).
EXPLORATION_MAX_TURNS = 50

_PRE_SCAN_STRATEGIES = (
    "exploration.pattern_scan", "exploration.dependency_trace", "exploration.test_mapping",
)
_STATIC_DIFF_MAX_BYTES = 65_536
_STATIC_GUIDANCE_MAX_BYTES = 8192


def _static_guidance(repo_root: Path, modified_files: list[FileInfo]) -> str:
    """Capture bounded local guidance without executing model discovery."""
    root = repo_root.resolve()
    candidates = [root / name for name in ("AGENTS.md", "CLAUDE.md", ".editorconfig")]
    for source in modified_files:
        path = (repo_root / source.path).resolve()
        if not path.is_relative_to(root):
            continue
        for parent in reversed(path.relative_to(root).parents):
            if str(parent) != ".":
                candidates.extend(root / parent / name for name in ("AGENTS.md", "CLAUDE.md", ".editorconfig"))
    candidates.append(root / "docs/conventions.md")
    notes = (
        "Deterministic pre-scan: static affected-file memberships and bounded guideline excerpts only; "
        "not correctness review or coverage evidence. No model mapping pass was needed. "
        "Missing or truncated guidance does not establish a convention.\n"
    )
    tail_marker = "\n[Further guideline excerpts omitted by the aggregate byte limit.]"
    remaining = _STATIC_GUIDANCE_MAX_BYTES - len((notes + tail_marker).encode("utf-8"))
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            if resolved in seen or not resolved.is_relative_to(root) or not stat.S_ISREG(resolved.stat().st_mode):
                continue
            seen.add(resolved)
            label = f"\nRepository guideline {candidate.relative_to(root)} (untrusted data):\n"
            allowance = min(2048, remaining - len(label.encode("utf-8")))
            if allowance < 128:
                return notes + tail_marker
            with resolved.open("rb") as stream:
                raw = stream.read(allowance + 1)
            excerpt = truncate_utf8_to_budget(raw.decode("utf-8", errors="replace"), allowance, "\n[truncated]")
            block = label + excerpt
            notes += block
            remaining -= len(block.encode("utf-8"))
        except (OSError, ValueError):
            continue
    return notes


def count_changed_files(diff_text: str) -> int:
    """Count unique file paths in a unified-diff string.

    Returns:
        Number of unique ``a/<path>`` entries in ``diff --git`` headers.
    """
    if not diff_text:
        return 0
    return len({m.group(1) for m in _DIFF_HEADER_RE.finditer(diff_text)})


def select_tier(file_count: int) -> Tier:
    """Pick an exploration tier based on the number of changed files.

    - 0 or 1 files -> ``"skip"`` (no exploration)
    - 2 or 3 files -> ``"single"`` (dependency-tracer only)
    - 4+ files     -> ``"parallel"`` (all three specialists)
    """
    if file_count <= 1:
        return "skip"
    if file_count <= 3:
        return "single"
    return "parallel"


def _coerce_records(
    entries: Any,
    factory: Callable[..., Any],
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
    fixed: dict[str, Any] | None = None,
) -> list[Any]:
    out: list[Any] = []
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            kwargs: dict[str, Any] = {field: str(entry[field]) for field in required}
            kwargs.update({field: str(entry.get(field, "")) for field in optional})
            if fixed:
                kwargs.update(fixed)
            out.append(factory(**kwargs))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _coerce_file_infos(entries: Any) -> list[FileInfo]:
    return _coerce_records(
        entries, FileInfo, ("path", "role"), ("summary", "source_file"),
        {"provenance": "llm"},
    )


def _coerce_conventions(entries: Any) -> list[Convention]:
    return _coerce_records(entries, Convention, ("name", "description"), ("source",))


def _coerce_dependencies(entries: Any) -> list[Dependency]:
    return _coerce_records(entries, Dependency, ("source", "target", "relationship"))


def _coerce_guidelines(entries: Any) -> list[str]:
    if not isinstance(entries, list):
        return []
    return [str(g) for g in entries if isinstance(g, (str, int, float))]


def _parse_envelope(envelope: dict[str, Any]) -> ExplorationContext:
    """Convert a structured envelope dict into an ``ExplorationContext``.

    Missing top-level keys (single-tier case) are tolerated. Malformed
    sub-entries are skipped silently rather than raising.
    """
    files: list[FileInfo] = []
    conventions: list[Convention] = []
    dependencies: list[Dependency] = []
    guidelines: list[str] = []

    pattern = envelope.get("pattern_scanner")
    if isinstance(pattern, dict):
        conventions.extend(_coerce_conventions(pattern.get("conventions")))
        guidelines.extend(_coerce_guidelines(pattern.get("guidelines")))

    dep = envelope.get("dependency_tracer")
    if isinstance(dep, dict):
        files.extend(_coerce_file_infos(dep.get("affected_files")))
        dependencies.extend(_coerce_dependencies(dep.get("dependencies")))

    test = envelope.get("test_mapper")
    if isinstance(test, dict):
        files.extend(_coerce_file_infos(test.get("affected_files")))

    return ExplorationContext(
        affected_files=files,
        conventions=conventions,
        dependencies=dependencies,
        guidelines=guidelines,
    )


@bind_resolved_run_context
async def pre_scan(
    backend: Backend,
    repo_root: Path,
    diff_text: str,
    diff_ref: str = "HEAD",
    strategies: dict[str, str] | None = None,
    *,
    run_context: RunContext | None = None,
) -> ExplorationContext:
    """Run the pre-scan exploration pipeline for a diff.

    Steps:
        1. Build a static affected-files list from ``detect_affected_files``.
        2. Count unique files in the diff and pick a tier.
        3. For ``"skip"`` -- return the static context, no backend call.
        4. Modest default changes return static mapping plus guideline excerpts.
        5. Otherwise, for ``"single"`` / ``"parallel"`` -- launch parallel
           ``backend.execute()`` calls (one per specialist), parse results,
           merge with the static context.

    Args:
        repo_root: Repository root used by ``detect_affected_files``.
        diff_text: Raw git diff string used for static mapping and bounded
            specialist context (complete when small, advisory excerpts otherwise).
        diff_ref: Git ref or range (e.g. ``"main...HEAD"``) passed to specialist
            prompts so they can run ``git diff <ref> -- <file>`` per file.
        strategies: Optional mapping of the four exploration strategy contents
            (``exploration.pattern_scan`` / ``exploration.dependency_trace`` /
            ``exploration.test_mapping`` / ``exploration.repository_survey``).
            When ``None`` (the default), the packaged default-profile contents
            are used, so non-profile callers stay operable.
    """
    run_context = resolve_run_context(run_context)
    defaults = _rp.build_default_profile().strategies
    if strategies is None:
        strategies = {
            "exploration.pattern_scan": defaults["exploration.pattern_scan"].content,
            "exploration.dependency_trace": defaults["exploration.dependency_trace"].content,
            "exploration.test_mapping": defaults["exploration.test_mapping"].content,
            "exploration.repository_survey": defaults["exploration.repository_survey"].content,
        }
    import anyio

    from daydream.agent import run_agent

    static_files: list[FileInfo] = []
    try:
        static_files = detect_affected_files(diff_text, repo_root)
    except Exception:  # noqa: BLE001 - best-effort path; exploration degrades silently per D-08
        pass

    if not static_files:
        # Static resolution failed outright; still seed specialists with the
        # changed files so they have a starting point. Reuse the rename-aware
        # name/status parser so a renamed file seeds its new path, not the old.
        changed = sorted({e.path for e in _parse_diff_name_status(diff_text)})
        static_files = [FileInfo(path=p, role="modified") for p in changed]

    static_context = ExplorationContext(affected_files=static_files)

    tier = select_tier(count_changed_files(diff_text))

    if tier == "skip":
        return static_context

    source_files = mapping_source_files(static_files, repo_root)
    if (len({file.path for file in source_files}) <= 3
            and len(diff_text.encode("utf-8")) <= _STATIC_DIFF_MAX_BYTES
            and all(strategies[name] == defaults[name].content for name in _PRE_SCAN_STRATEGIES)):
        static_context.raw_notes = _static_guidance(
            repo_root, [file for file in static_files if file.role == "modified"],
        )
        return static_context

    results: dict[str, Any] = {}
    specialist_failed = False

    specialist_max_turns = EXPLORATION_MAX_TURNS
    recorder = get_current_recorder()
    limiter = anyio.CapacityLimiter(
        effective_fanout_concurrency(10, backend)
    )
    has_mapping_targets = bool(source_files) or strategies["exploration.test_mapping"] != defaults[
        "exploration.test_mapping"
    ].content
    descriptors = (
        ("explore-dependency_tracer",)
        if tier == "single"
        else (
            "explore-pattern_scanner",
            "explore-dependency_tracer",
            *(("explore-test_mapper",) if has_mapping_targets else ()),
        )
    )
    async def _run_specialist(
        name: str,
        prompt: str,
        schema: dict[str, Any],
        dispatch: DispatchHandle | None,
    ) -> None:
        nonlocal specialist_failed
        async with limiter, maybe_fork(
            recorder, f"explore-{name}", dispatch=dispatch,
        ):
            try:
                structured, _, budget_reason = await run_agent(
                    backend, repo_root, prompt, output_schema=schema, max_turns=specialist_max_turns,
                    phase=DaydreamPhase.EXPLORATION,
                    read_only=True,
                    review_limits=_PRE_SCAN_REVIEW_LIMITS,
                    finalization_context=FinalizationContext(
                        task=f"Finalize exploration mapping: {name}",
                        assigned_files=tuple(f.path for f in static_files),
                        output_semantics="Return only the requested conventions, dependency edges, or test mappings "
                        "in the schema. Do not review defects. Omit unconfirmed mappings; empty arrays are valid.",
                        supplied_context=(("change diff", diff_text),),
                    ),
                    wall_budget_s=DEFAULT_WALL_BUDGET_S,
                    tool_call_budget=DEFAULT_TOOL_CALL_BUDGET,
                    run_context=run_context,
                )
                if budget_reason:
                    specialist_failed = True
                if isinstance(structured, dict):
                    results[name] = structured
                else:
                    specialist_failed = True
            except Exception:  # noqa: BLE001 - best-effort path; exploration degrades silently per D-08
                specialist_failed = True
                pass

    # Builders split this bounded static map into changed targets and optional
    # context for each specialist, so imported files never become new targets.
    #
    # Paths are cwd-absolute (rooted at repo_root, the actual worktree). In a
    # linked worktree the agent must not re-root a bare relative path via git
    # topology, which points at the sibling main worktree.
    static_files_abs = [
        FileInfo(
            path=str(repo_root / f.path),
            role=f.role,
            summary=f.summary,
            provenance=f.provenance,
            source_file=f.source_file,
        )
        for f in static_files
    ]

    async with dispatch_scope(
        recorder,
        phase=DaydreamPhase.EXPLORATION,
        descriptors=descriptors,
    ) as dispatch:
        with anyio.move_on_after(_PRE_SCAN_TIMEOUT_SECONDS) as timeout_scope:
            async with anyio.create_task_group() as tg:
                if tier == "single":
                    dep_prompt = build_dependency_tracer_prompt(
                        static_files_abs,
                        diff_ref,
                        cwd=repo_root,
                        strategy=strategies["exploration.dependency_trace"],
                        inline_diff=diff_text,
                    )
                    tg.start_soon(
                        _run_specialist,
                        "dependency_tracer",
                        dep_prompt,
                        DEPENDENCY_TRACER_SCHEMA,
                        dispatch,
                    )
                else:  # parallel
                    tg.start_soon(
                        _run_specialist, "pattern_scanner",
                        build_pattern_scanner_prompt(
                            static_files_abs,
                            diff_ref,
                            cwd=repo_root,
                            strategy=strategies["exploration.pattern_scan"],
                            inline_diff=diff_text,
                        ), PATTERN_SCANNER_SCHEMA, dispatch,
                    )
                    tg.start_soon(
                        _run_specialist, "dependency_tracer",
                        build_dependency_tracer_prompt(
                            static_files_abs,
                            diff_ref,
                            cwd=repo_root,
                            strategy=strategies["exploration.dependency_trace"],
                            inline_diff=diff_text,
                        ),
                        DEPENDENCY_TRACER_SCHEMA,
                        dispatch,
                    )
                    if has_mapping_targets:
                        tg.start_soon(
                            _run_specialist, "test_mapper",
                            build_test_mapper_prompt(
                                static_files_abs,
                                diff_ref,
                                cwd=repo_root,
                                strategy=strategies["exploration.test_mapping"],
                                inline_diff=diff_text,
                                source_only=strategies["exploration.test_mapping"] == defaults[
                                    "exploration.test_mapping"
                                ].content,
                            ), TEST_MAPPER_SCHEMA, dispatch,
                        )
        if dispatch is not None:
            if timeout_scope.cancel_called:
                dispatch.finish(
                    LifecycleStatus.TIMED_OUT, LifecycleReasonCode.TIMED_OUT,
                )
            elif specialist_failed:
                finish_partial_or_failed(dispatch, results)

    if not results:
        static_context.completed = not (specialist_failed or timeout_scope.cancel_called)
        return static_context

    subagent_context = _parse_envelope(results)
    context = merge_contexts(static_context, subagent_context)
    context.completed = not (specialist_failed or timeout_scope.cancel_called)
    return context


def _sample_paths(paths: list[str], limit: int) -> list[str]:
    """Take up to ``limit`` paths spread evenly across a sorted path list.

    Head-truncating `git ls-files` on a large repo yields only the alphabetical
    head -- dotfile directories such as `.agents/` and `.claude/` -- so the
    survey never sees the source tree. An even stride keeps every top-level area
    represented while staying deterministic.
    """
    if limit <= 0 or not paths:
        return []
    if len(paths) <= limit:
        return list(paths)
    stride = len(paths) / limit
    return [paths[int(i * stride)] for i in range(limit)]


@bind_resolved_run_context
async def repo_scan(
    backend: Backend,
    repo_root: Path,
    *,
    max_files: int = 500,
    strategies: dict[str, str] | None = None,
    run_context: RunContext | None = None,
) -> ExplorationContext:
    """Discover repository conventions from a bounded tracked-file sample.

    Returns conventions and guidelines only. The tracked-file sample seeds the
    survey prompt but is not returned: a repo-scoped run has no affected files,
    and emitting one would mislabel the whole repository as change-relevant.

    Args:
        strategies: Optional mapping containing the
            ``exploration.repository_survey`` strategy content. When ``None``
            (the default), the packaged default-profile content is used.
    """
    run_context = resolve_run_context(run_context)
    if strategies is None:
        strategies = {
            "exploration.repository_survey": _rp.build_default_profile().strategies[
                "exploration.repository_survey"
            ].content,
        }
    import anyio

    from daydream.agent import run_agent

    paths: list[str] = []
    try:
        paths = git_ops.ls_files(repo_root)
    except Exception:  # noqa: BLE001 - best-effort path; exploration degrades silently per D-08
        pass

    sample = _sample_paths(paths, max(0, max_files))
    survey: dict[str, Any] = {}
    recorder = get_current_recorder()
    specialist_failed = False
    async def _run_specialist(dispatch: DispatchHandle | None) -> None:
        nonlocal specialist_failed
        async with maybe_fork(
            recorder, "explore-repo_survey", dispatch=dispatch,
        ):
            try:
                structured, _, _ = await run_agent(
                    backend,
                    repo_root,
                    build_repo_survey_prompt(
                        sample,
                        len(paths),
                        cwd=repo_root,
                        strategy=strategies["exploration.repository_survey"],
                    ),
                    output_schema=PATTERN_SCANNER_SCHEMA,
                    max_turns=EXPLORATION_MAX_TURNS,
                    phase=DaydreamPhase.EXPLORATION,
                    read_only=True,
                    wall_budget_s=DEFAULT_WALL_BUDGET_S,
                    tool_call_budget=DEFAULT_TOOL_CALL_BUDGET,
                    run_context=run_context,
                )
                if isinstance(structured, dict):
                    survey.update(structured)
                else:
                    specialist_failed = True
            except Exception:  # noqa: BLE001 - best-effort path; exploration degrades silently per D-08
                specialist_failed = True
                pass

    async with dispatch_scope(
        recorder,
        phase=DaydreamPhase.EXPLORATION,
        descriptors=("explore-repo_survey",),
    ) as dispatch:
        with anyio.move_on_after(_SPECIALIST_TIMEOUT_SECONDS) as timeout_scope:
            await _run_specialist(dispatch)
        if dispatch is not None:
            if timeout_scope.cancel_called:
                dispatch.finish(
                    LifecycleStatus.TIMED_OUT, LifecycleReasonCode.TIMED_OUT,
                )
            elif specialist_failed:
                dispatch.finish(
                    LifecycleStatus.FAILED,
                    LifecycleReasonCode.ALL_CHILDREN_FAILED,
                )

    return ExplorationContext(
        conventions=_coerce_conventions(survey.get("conventions")),
        guidelines=_coerce_guidelines(survey.get("guidelines")),
    )


__all__ = [
    "Tier",
    "count_changed_files",
    "pre_scan",
    "repo_scan",
    "select_tier",
]
