"""Pre-scan orchestrator.

Counts changed files in a diff, selects an exploration tier (skip / single /
parallel), launches specialist subagents via ``Backend.execute(agents=...)``,
parses each specialist's structured-JSON envelope into a partial
``ExplorationContext``, and merges everything (including the static tree-sitter
file map from ``detect_affected_files``) into a single context.

The orchestrator is intentionally tier-driven so a trivial diff produces zero
backend calls and a multi-file diff fans out to three specialists in one call.
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
    EXPLORATION_AGENTS,
    PATTERN_SCANNER_SCHEMA,
    TEST_MAPPER_SCHEMA,
    build_dependency_tracer_prompt,
    build_pattern_scanner_prompt,
    build_test_mapper_prompt,
)
from daydream.tree_sitter_index import detect_affected_files

if TYPE_CHECKING:
    from pathlib import Path

    from claude_agent_sdk.types import AgentDefinition

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


def _build_lead_prompt(
    tier: Tier,
    diff_text: str,
    static_files: list[FileInfo],
) -> str:
    """Construct the lead-agent prompt that delegates to specialists."""
    file_paths = [f.path for f in static_files]
    pattern_block = build_pattern_scanner_prompt(diff_text, file_paths)
    dependency_block = build_dependency_tracer_prompt(diff_text, static_files)
    test_block = build_test_mapper_prompt(diff_text, file_paths)

    if tier == "single":
        return f"""You are the lead exploration agent. You have ONE specialist available:
- `dependency-tracer`: extends the import graph by grepping call sites.

Delegate to the dependency-tracer specialist, then emit a SINGLE JSON object
matching the envelope schema with the `dependency_tracer` key populated from
the specialist's response. Other keys may be omitted.

<dependency-tracer-instructions>
{dependency_block}
</dependency-tracer-instructions>

Return ONLY a JSON object of the shape:
{{
  "dependency_tracer": {{ ...DEPENDENCY_TRACER_SCHEMA fields... }}
}}
"""

    return f"""You are the lead exploration agent. You have THREE specialists available:
- `pattern-scanner`: detects conventions and reads guideline files.
- `dependency-tracer`: extends the import graph by grepping call sites.
- `test-mapper`: locates test files for each modified source file.

Delegate to ALL THREE specialists IN PARALLEL, then emit a SINGLE JSON object
matching the envelope schema with `pattern_scanner`, `dependency_tracer`, and
`test_mapper` keys populated from their respective responses.

<pattern-scanner-instructions>
{pattern_block}
</pattern-scanner-instructions>

<dependency-tracer-instructions>
{dependency_block}
</dependency-tracer-instructions>

<test-mapper-instructions>
{test_block}
</test-mapper-instructions>

Return ONLY a JSON object of the shape:
{{
  "pattern_scanner": {{ ...PATTERN_SCANNER_SCHEMA fields... }},
  "dependency_tracer": {{ ...DEPENDENCY_TRACER_SCHEMA fields... }},
  "test_mapper": {{ ...TEST_MAPPER_SCHEMA fields... }}
}}
"""


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
    *,
    live_panel: "Any | None" = None,
) -> ExplorationContext:
    """Run the pre-scan exploration pipeline for a diff.

    Steps:
        1. Build a static affected-files list from ``detect_affected_files``.
        2. Count unique files in the diff and pick a tier.
        3. For ``"skip"`` -- return the static context, no backend call.
        4. For ``"single"`` / ``"parallel"`` -- launch one ``Backend.execute()``
           call with the appropriate ``agents=`` mapping, parse the envelope,
           merge with the static context.

    Subagent failure (parse miss, exception inside the lead) downgrades to the
    static context only -- this is enforced via the outer ``safe_explore``
    wrapper at the call site, but ``pre_scan`` itself never raises on a
    structured-output miss.

    Args:
        backend: Backend to invoke.
        repo_root: Repository root used by ``detect_affected_files``.
        diff_text: Raw git diff string.
        depth: Static-resolution depth (forwarded to ``detect_affected_files``).
        live_panel: Optional UI panel; specialist start/done callbacks are
            issued through it as the run progresses.

    Returns:
        Merged ``ExplorationContext``.
    """
    # Lazy imports to avoid pulling agent.py into module-load time and to
    # match the project convention used elsewhere (see daydream/exploration.py).
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

    if tier == "single":
        agents: dict[str, AgentDefinition] = {
            "dependency-tracer": EXPLORATION_AGENTS["dependency-tracer"],
        }
    else:
        agents = dict(EXPLORATION_AGENTS)

    if live_panel is not None:
        for name in agents:
            live_panel.mark_start(name)

    prompt = _build_lead_prompt(tier, diff_text, static_files)

    try:
        structured, _ = await run_agent(
            backend,
            repo_root,
            prompt,
            output_schema=EXPLORATION_ENVELOPE_SCHEMA,
            agents=agents,
        )
    except Exception as exc:
        _log_debug(f"[PRE_SCAN] run_agent raised: {type(exc).__name__}: {exc}\n")
        if live_panel is not None:
            for name in agents:
                live_panel.mark_failed(name, str(exc))
        return static_context

    if not isinstance(structured, dict):
        _log_debug(
            f"[PRE_SCAN] envelope parse miss: structured={type(structured).__name__}\n"
        )
        if live_panel is not None:
            for name in agents:
                live_panel.mark_failed(name, "no structured output")
        return static_context

    subagent_context = _parse_envelope(structured)

    if live_panel is not None:
        for name in agents:
            live_panel.mark_done(name)

    return merge_contexts(static_context, subagent_context)


__all__ = [
    "EXPLORATION_ENVELOPE_SCHEMA",
    "Tier",
    "count_changed_files",
    "pre_scan",
    "select_tier",
]
