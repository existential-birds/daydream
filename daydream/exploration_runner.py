"""Pre-scan orchestrator.

Counts changed files in a diff, selects an exploration tier (skip / single /
parallel), launches specialist ``backend.execute()`` calls in parallel (one
per specialist), parses each specialist's structured-JSON result into a partial
``ExplorationContext``, and merges everything (including the static tree-sitter
file map from ``detect_affected_files``) into a single context.

The orchestrator is intentionally tier-driven so a trivial diff produces zero
backend calls and a multi-file diff fans out to three parallel specialists.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from daydream.exploration import (
    Convention,
    Dependency,
    ExplorationContext,
    FileInfo,
    merge_contexts,
)
from daydream.prompts.exploration_subagents import (
    DEPENDENCY_TRACER_SCHEMA,
    PATTERN_SCANNER_SCHEMA,
    TEST_MAPPER_SCHEMA,
    build_dependency_tracer_prompt,
    build_pattern_scanner_prompt,
    build_test_mapper_prompt,
)
from daydream.tree_sitter_index import detect_affected_files

if TYPE_CHECKING:
    from pathlib import Path

    from daydream.backends import Backend


Tier: TypeAlias = Literal["skip", "single", "parallel"]


# This regex parses git's own diff header output (not source code), so a
# regex is the right tool here per D-04 (no tree-sitter for non-source text).
_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+) b/", re.MULTILINE)


EXPLORATION_ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern_scanner": PATTERN_SCANNER_SCHEMA,
        "dependency_tracer": DEPENDENCY_TRACER_SCHEMA,
        "test_mapper": TEST_MAPPER_SCHEMA,
    },
    # Any subset is allowed -- single tier only emits one key.
    "required": [],
    "additionalProperties": False,
}


def count_changed_files(diff_text: str) -> int:
    """Count unique file paths in a unified-diff string.

    Args:
        diff_text: Raw ``git diff`` output.

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


def _coerce_file_infos(entries: Any) -> list[FileInfo]:
    out: list[FileInfo] = []
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(
                FileInfo(
                    path=str(entry["path"]),
                    role=str(entry["role"]),
                    summary=str(entry.get("summary", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _coerce_conventions(entries: Any) -> list[Convention]:
    out: list[Convention] = []
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(
                Convention(
                    name=str(entry["name"]),
                    description=str(entry["description"]),
                    source=str(entry.get("source", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _coerce_dependencies(entries: Any) -> list[Dependency]:
    out: list[Dependency] = []
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(
                Dependency(
                    source=str(entry["source"]),
                    target=str(entry["target"]),
                    relationship=str(entry["relationship"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


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


async def pre_scan(
    backend: Backend,
    repo_root: Path,
    diff_text: str,
    depth: int = 1,
) -> ExplorationContext:
    """Run the pre-scan exploration pipeline for a diff.

    Steps:
        1. Build a static affected-files list from ``detect_affected_files``.
        2. Count unique files in the diff and pick a tier.
        3. For ``"skip"`` -- return the static context, no backend call.
        4. For ``"single"`` / ``"parallel"`` -- launch parallel
           ``backend.execute()`` calls (one per specialist), parse results,
           merge with the static context.

    Args:
        backend: Backend to invoke.
        repo_root: Repository root used by ``detect_affected_files``.
        diff_text: Raw git diff string.
        depth: Static-resolution depth (forwarded to ``detect_affected_files``).

    Returns:
        Merged ``ExplorationContext``.
    """
    import anyio

    from daydream.agent import _log_debug, run_agent

    static_files: list[FileInfo] = []
    try:
        static_files = detect_affected_files(diff_text, repo_root, depth)
    except Exception as exc:  # pragma: no cover - defensive
        _log_debug(f"[PRE_SCAN] detect_affected_files failed: {exc}\n")

    static_context = ExplorationContext(affected_files=static_files)

    file_count = count_changed_files(diff_text)
    tier = select_tier(file_count)

    if tier == "skip":
        return static_context

    results: dict[str, Any] = {}

    async def _run_specialist(name: str, prompt: str, schema: dict) -> None:
        try:
            structured, _ = await run_agent(backend, repo_root, prompt, output_schema=schema)
            if isinstance(structured, dict):
                results[name] = structured
        except Exception as exc:
            _log_debug(f"[PRE_SCAN] specialist {name} failed: {type(exc).__name__}: {exc}\n")

    file_paths = [f.path for f in static_files]

    async with anyio.create_task_group() as tg:
        if tier == "single":
            dep_prompt = build_dependency_tracer_prompt(diff_text, static_files)
            tg.start_soon(_run_specialist, "dependency_tracer", dep_prompt, DEPENDENCY_TRACER_SCHEMA)
        else:  # parallel
            tg.start_soon(
                _run_specialist, "pattern_scanner",
                build_pattern_scanner_prompt(diff_text, file_paths), PATTERN_SCANNER_SCHEMA,
            )
            tg.start_soon(
                _run_specialist, "dependency_tracer",
                build_dependency_tracer_prompt(diff_text, static_files), DEPENDENCY_TRACER_SCHEMA,
            )
            tg.start_soon(
                _run_specialist, "test_mapper",
                build_test_mapper_prompt(diff_text, file_paths), TEST_MAPPER_SCHEMA,
            )

    if not results:
        _log_debug("[PRE_SCAN] no specialist results collected\n")
        return static_context

    subagent_context = _parse_envelope(results)
    return merge_contexts(static_context, subagent_context)


__all__ = [
    "EXPLORATION_ENVELOPE_SCHEMA",
    "Tier",
    "count_changed_files",
    "pre_scan",
    "select_tier",
]
