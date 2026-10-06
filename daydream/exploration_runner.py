"""Diff-scoped pre-scan and repository-wide convention discovery.

Modest default changes use static mapping and bounded guidelines; other
diffs add tiered specialists. Repository scans sample tracked files and
run only the survey specialist."""

from __future__ import annotations

import stat
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, TypeAlias

import anyio

from daydream import git_ops, review_profile as _rp
from daydream.agent import run_agent
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
from daydream.review_budget import ReviewLimits, review_scale_for_scope
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
from daydream.tree_sitter_index import detect_affected_files
from daydream.tree_sitter_index.imports import _parse_diff_name_status

if TYPE_CHECKING:
    from pathlib import Path

    from daydream.backends import Backend


Tier: TypeAlias = Literal["skip", "single", "parallel"]

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
_SURVEY_STRATEGY = "exploration.repository_survey"
_STATIC_DIFF_MAX_BYTES = 65_536
_STATIC_GUIDANCE_MAX_BYTES = 8192


def _packaged_strategies(names: Iterable[str]) -> dict[str, str]:
    """Packaged strategy text for `names`, so callers need no resolved profile."""
    defaults = _rp.build_default_profile().strategies
    return {name: defaults[name].content for name in names}


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
    """Count unique file paths in a unified-diff string."""
    return len({entry.path for entry in _parse_diff_name_status(diff_text)})


def select_tier(file_count: int) -> Tier:
    """Skip 0–1 changed files; trace dependencies for 2–3; run all specialists for 4+."""
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


def _finish_exploration_dispatch(
    dispatch: DispatchHandle | None,
    timeout_scope: Any,
    specialist_failed: bool,
    results: object,
) -> None:
    """Close a specialist dispatch with the shared timeout/failure policy.

    A cancelled scope is TIMED_OUT; otherwise a failed specialist closes the
    dispatch PARTIAL when ``results`` holds a success, else ALL_CHILDREN_FAILED.
    """
    if dispatch is None:
        return
    if timeout_scope.cancel_called:
        dispatch.finish(
            LifecycleStatus.TIMED_OUT, LifecycleReasonCode.TIMED_OUT,
        )
    elif specialist_failed:
        finish_partial_or_failed(dispatch, results)


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
    """Merge static affected-file mapping with optional tiered specialist results.

    Skip-tier and modest default changes avoid model discovery. Specialists get
    complete small diffs or bounded advisory excerpts plus diff_ref for per-file
    Git reads. strategies overrides the four exploration strategies; None uses
    packaged defaults so callers need no resolved profile."""
    run_context = resolve_run_context(run_context)
    defaults = _rp.build_default_profile().strategies
    if strategies is None:
        strategies = _packaged_strategies((*_PRE_SCAN_STRATEGIES, _SURVEY_STRATEGY))
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
                    wall_budget_s=DEFAULT_WALL_BUDGET_S * review_scale_for_scope(),
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

    # Builders split this bounded static map into changed targets and optional
    # context for each specialist, so imported files never become new targets.
    #
    # Paths are cwd-absolute (rooted at repo_root, the actual worktree). In a
    # linked worktree the agent must not re-root a bare relative path via git
    # topology, which points at the sibling main worktree.
    static_files_abs = [replace(f, path=str(repo_root / f.path)) for f in static_files]

    # One ordered roster drives both the dispatch descriptors and the launches,
    # so a specialist cannot be added, dropped, or reordered in one place only.
    # Prompts are built here, before the dispatch scope opens, so the
    # recorder-visible ordering is unchanged.
    plan: list[tuple[str, str, dict[str, Any]]] = []
    if tier != "single":
        plan.append((
            "pattern_scanner",
            build_pattern_scanner_prompt(
                static_files_abs, diff_ref, cwd=repo_root,
                strategy=strategies["exploration.pattern_scan"], inline_diff=diff_text,
            ),
            PATTERN_SCANNER_SCHEMA,
        ))
    plan.append((
        "dependency_tracer",
        build_dependency_tracer_prompt(
            static_files_abs, diff_ref, cwd=repo_root,
            strategy=strategies["exploration.dependency_trace"], inline_diff=diff_text,
        ),
        DEPENDENCY_TRACER_SCHEMA,
    ))
    if tier != "single" and has_mapping_targets:
        plan.append((
            "test_mapper",
            build_test_mapper_prompt(
                static_files_abs, diff_ref, cwd=repo_root,
                strategy=strategies["exploration.test_mapping"], inline_diff=diff_text,
                source_only=strategies["exploration.test_mapping"] == defaults[
                    "exploration.test_mapping"
                ].content,
            ),
            TEST_MAPPER_SCHEMA,
        ))
    descriptors = tuple(f"explore-{name}" for name, _, _ in plan)

    async with dispatch_scope(
        recorder,
        phase=DaydreamPhase.EXPLORATION,
        descriptors=descriptors,
    ) as dispatch:
        with anyio.move_on_after(_PRE_SCAN_TIMEOUT_SECONDS * review_scale_for_scope()) as timeout_scope:
            async with anyio.create_task_group() as tg:
                for name, prompt, schema in plan:
                    tg.start_soon(_run_specialist, name, prompt, schema, dispatch)
        _finish_exploration_dispatch(dispatch, timeout_scope, specialist_failed, results)

    if not results:
        static_context.completed = not (specialist_failed or timeout_scope.cancel_called)
        return static_context

    subagent_context = _parse_envelope(results)
    context = merge_contexts(static_context, subagent_context)
    context.completed = not (specialist_failed or timeout_scope.cancel_called)
    return context


def _sample_paths(paths: list[str], limit: int) -> list[str]:
    """Sample an even stride through sorted paths to include source trees that
    alphabetical head truncation would hide behind dotfile directories."""
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
    """Discover conventions and guidelines from a bounded tracked-file sample.

    Do not return sampled files as affected files: a repository scan has no diff.
    strategies may override repository_survey; None uses its packaged default."""
    run_context = resolve_run_context(run_context)
    if strategies is None:
        strategies = _packaged_strategies((_SURVEY_STRATEGY,))
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
                        strategy=strategies[_SURVEY_STRATEGY],
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

    async with dispatch_scope(
        recorder,
        phase=DaydreamPhase.EXPLORATION,
        descriptors=("explore-repo_survey",),
    ) as dispatch:
        with anyio.move_on_after(_SPECIALIST_TIMEOUT_SECONDS) as timeout_scope:
            await _run_specialist(dispatch)
        _finish_exploration_dispatch(dispatch, timeout_scope, specialist_failed, survey)

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
