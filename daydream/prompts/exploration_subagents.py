"""Exploration subagent prompts and output schemas.

Three specialist subagents power pre-scan exploration:

- **pattern-scanner**: detects codebase conventions and reads guideline files
  (CLAUDE.md, ruff.toml, etc.) -- satisfies EXPL-04.
- **dependency-tracer**: extends the static-resolved import graph by grepping
  call sites and emits Dependency edges.
- **test-mapper**: locates test files for each modified source file via
  conventional path mapping.

The orchestrator (daydream.exploration_runner) merges their partial
ExplorationContext results with merge_contexts() in daydream.exploration.

A fourth prompt, **repo-survey**, serves the diff-less repo-scoped scan used by
``daydream improve``. It shares the pattern-scanner output shape but must never
share its diff framing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.output_schema import strict_object
from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES, fits_inline_diff_budget
from daydream.prompts.grounding import (
    CWD_GROUNDING_INSTRUCTION,
    UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY,
)
from daydream.prompts.schema_block import schema_block
from daydream.repository_paths import is_test_path

if TYPE_CHECKING:
    from daydream.exploration import FileInfo


# JSON Schemas (mirror style of FEEDBACK_SCHEMA in daydream/phases.py)
PATTERN_SCANNER_SCHEMA: dict[str, Any] = strict_object({
    "conventions": {
        "type": "array",
        "items": strict_object({
            "name": {"type": "string"},
            "description": {"type": "string"},
            "source": {"type": "string"},
        }),
    },
    "guidelines": {
        "type": "array",
        "items": {"type": "string"},
    },
})

DEPENDENCY_TRACER_SCHEMA: dict[str, Any] = strict_object({
    "affected_files": {
        "type": "array",
        "items": strict_object({
            "path": {"type": "string"},
            "role": {
                "type": "string",
                "enum": ["modified", "imported_by", "imports", "test"],
            },
            "summary": {"type": "string"},
        }),
    },
    "dependencies": {
        "type": "array",
        "items": strict_object({
            "source": {"type": "string"},
            "target": {"type": "string"},
            "relationship": {
                "type": "string",
                "enum": ["imports", "calls", "extends", "tests"],
            },
        }),
    },
})

TEST_MAPPER_SCHEMA: dict[str, Any] = strict_object({
    "affected_files": {
        "type": "array",
        "items": strict_object({
            "path": {"type": "string"},
            "role": {"type": "string", "enum": ["test"]},
            "summary": {"type": "string"},
            "source_file": {"type": "string"},
        }),
    },
})


def _files_block(affected_files: list[FileInfo]) -> str:
    """Render the shared ``<affected_files>`` body: one ``- path (role)`` per entry."""
    return "\n".join(f"- {f.path} ({f.role})" for f in affected_files) or "- (none yet)"


def mapping_source_files(affected_files: list[FileInfo], cwd: Path) -> list[FileInfo]:
    """Select changed source targets without excluding unsupported languages.

    Docs, manifests and existing tests remain evidence in the mapper prompt;
    they do not warrant independent searches for test coverage.
    """
    excluded_suffixes = {
        ".md", ".mdx", ".rst", ".txt", ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".lock",
    }
    manifests = {"makefile", "gnumakefile", "dockerfile", "containerfile"}
    targets: list[FileInfo] = []
    for file in affected_files:
        path = Path(file.path)
        relative = str(path.relative_to(cwd)) if path.is_relative_to(cwd) else file.path
        if (file.role == "modified" and not is_test_path(relative)
                and path.suffix.lower() not in excluded_suffixes and path.name.lower() not in manifests):
            targets.append(file)
    return targets


def _change_overview(diff: str) -> str:
    """Retain bounded changed-line excerpts from each file for advisory mapping.

    Unlike review diff input, this projection can omit both context and changed
    lines. Whole lines and their file/hunk headers stay together, and omissions
    are explicit. Equal per-file shares prevent a large addition from hiding
    all subsequent small changes.
    """
    from daydream.deep.prompts import _DIFF_BLOCK_SPLIT, _diff_block_path

    blocks = [block for block in _DIFF_BLOCK_SPLIT.split(diff) if _diff_block_path(block)]
    opening = (
        "<change_overview>\nAdvisory changed-line excerpts; unchanged context is omitted. "
        "This is not source-read or review-coverage evidence, nor a complete patch. "
        "Use it to scope mapping; read source only for unresolved direct relationships.\n"
    )
    closing = "</change_overview>"
    omitted_notice = "[Some file excerpts omitted: insufficient space for their headers.]\n"
    available = INLINE_DIFF_BUDGET_BYTES - len((opening + closing + omitted_notice).encode("utf-8"))
    share = available // max(1, len(blocks))
    excerpts: list[str] = []
    omitted_files = False
    for block in blocks:
        lines = [line + "\n" for line in block.splitlines() if not line.startswith(" ")]
        marker = "[Remaining diff lines omitted from this file.]\n"
        used = 0
        kept: list[str] = []
        for line in lines:
            size = len(line.encode("utf-8"))
            if used + size + len(marker.encode("utf-8")) > share:
                break
            kept.append(line)
            used += size
        # A file excerpt without its complete header cannot orient the mapper.
        header_end = next((i for i, line in enumerate(lines) if line.startswith("@@")), len(lines) - 1)
        if len(kept) <= header_end:
            omitted_files = True
            continue
        excerpts.append("".join(kept) + (marker if len(kept) < len(lines) else ""))
    return opening + "".join(excerpts) + (omitted_notice if omitted_files else "") + closing


def _inspect_changes_block(diff_ref: str, inline_diff: str | None = None) -> str:
    """Render the shared ``To inspect changes`` instruction block."""
    if inline_diff and fits_inline_diff_budget(inline_diff):
        return (
            "The complete change diff is supplied below as repository evidence. "
            "Use it to identify the changed behavior; do not fetch it again.\n"
            f"<change_diff>\n{inline_diff}\n</change_diff>"
        )
    if inline_diff:
        return _change_overview(inline_diff)
    return f"""To inspect changes, Read or Grep any file listed in <affected_files> directly.
If a Bash tool is available, you may also run `git diff {diff_ref} -- <file>`
to see exactly what changed. Do NOT dump the full diff — work file-by-file so
your context stays small."""


def _specialist_prompt(
    strategy: str,
    files_block: str,
    cwd: Path,
    diff_ref: str,
    schema: dict[str, Any],
    *,
    instructions: str = "",
    known_context: list[FileInfo] | None = None,
    inline_diff: str | None = None,
) -> str:
    instructions_block = f"{instructions}\n\n" if instructions else ""
    context_block = (
        "These paths are already identified by the host's static analysis. They are "
        "context candidates, not additional mapping targets or a required reading list. "
        "Do not rediscover their membership or recursively explore their dependencies.\n"
        f"<known_affected_context>\n{_files_block(known_context)}\n</known_affected_context>\n"
        if known_context else ""
    )
    return f"""{strategy}

{UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY}

Complete only the requested repository mapping. Do not conduct another correctness
review or search for defects. Stop when the requested conventions, dependencies,
or test mappings are established; unresolved mappings may be omitted truthfully.

{instructions_block}{CWD_GROUNDING_INSTRUCTION.format(cwd=cwd)}

<affected_files>
{files_block}
</affected_files>
Only the modified files above are mapping targets. The list is not a requirement
to read every file; use the minimum evidence needed for your specialist task.
{context_block}

{_inspect_changes_block(diff_ref, inline_diff)}

{schema_block(schema)}
"""


# Dynamic prompt builders (per-run prompts injecting diff + affected files)
def build_pattern_scanner_prompt(
    affected_files: list[FileInfo], diff_ref: str, *, cwd: Path, strategy: str,
    inline_diff: str | None = None,
) -> str:
    """Build the per-run pattern-scanner prompt.

    The prompt separates modified targets from static context. Small complete
    diffs can be supplied inline; oversized diffs get bounded advisory excerpts.

    Args:
        affected_files: FileInfo entries for files reachable from the diff.
        diff_ref: Git ref (e.g. base branch or SHA) the specialist can diff against.
        cwd: Absolute working directory the agent runs in (grounds path resolution).
        strategy: The profile-owned ``exploration.pattern_scan`` strategy content.
        inline_diff: Complete diff, inlined only when it fits the shared byte budget.
    """
    files_block = _files_block([f for f in affected_files if f.role == "modified"])
    return _specialist_prompt(
        strategy,
        files_block,
        cwd,
        diff_ref,
        PATTERN_SCANNER_SCHEMA,
        inline_diff=inline_diff,
        instructions="""Instructions:
- Read root CLAUDE.md / AGENTS.md if present, as evidence of applicable conventions.
- Read only house-style config relevant to languages or directories changed here.
- Stop after the applicable guidelines and configuration establish conventions.
  Where they are silent, inspect at most one representative changed source per language.
- Do not inventory unrelated stacks or read every changed file to infer more conventions.""",
    )


def build_repo_survey_prompt(
    sample_paths: list[str], total_tracked: int, *, cwd: Path, strategy: str
) -> str:
    """Build the repo-scoped survey prompt used by ``repo_scan``.

    There is no diff in a repo-scoped run, so this prompt must not borrow the
    diff framing of ``build_pattern_scanner_prompt``: no change-set, no
    ``git diff``, and the file list is declared as the partial sample it is.

    Args:
        sample_paths: Repo-relative tracked paths, sampled across the tree.
        total_tracked: Total tracked-file count, so the sample is honest about coverage.
        cwd: Absolute working directory the agent runs in (grounds path resolution).
        strategy: The profile-owned ``exploration.repository_survey`` strategy content.
    """
    sample_block = "\n".join(f"- {path}" for path in sample_paths) or "- (no tracked files)"
    coverage = (
        f"{len(sample_paths)} of {total_tracked} tracked files, sampled across the tree"
        if len(sample_paths) < total_tracked
        else f"all {total_tracked} tracked files"
    )
    return f"""{strategy}

{UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY}

Instructions:
- Read CLAUDE.md / AGENTS.md at the repo root if they exist.
- Read any other house-style config files you find (ruff.toml, .editorconfig, tsconfig.json, go.mod, Cargo.toml).
- Infer conventions from the code itself where config files are silent.
- Cover the repository's real source directories, not just the sample below.

{CWD_GROUNDING_INSTRUCTION.format(cwd=cwd)}

<tracked_file_sample>
{sample_block}
</tracked_file_sample>

The sample above is {coverage} — it is a starting point, NOT the repository's
contents. Run `git ls-files` or Glob to see the full tree, and Read/Grep the
files you need. Work file-by-file so your context stays small.

{schema_block(PATTERN_SCANNER_SCHEMA)}
"""


def build_dependency_tracer_prompt(
    affected_files: list[FileInfo], diff_ref: str, *, cwd: Path, strategy: str,
    inline_diff: str | None = None,
) -> str:
    """Build the per-run dependency-tracer prompt.

    The prompt separates modified targets from static context. Small complete
    diffs can be supplied inline; oversized diffs get bounded advisory excerpts.

    Args:
        affected_files: FileInfo entries for files reachable from the diff, each carrying a `path` and `role`.
        diff_ref: Git ref the specialist can diff against when probing call sites.
        cwd: Absolute working directory the agent runs in (grounds path resolution).
        strategy: The profile-owned ``exploration.dependency_trace`` strategy content.
        inline_diff: Complete diff, inlined only when it fits the shared byte budget.
    """
    files_block = _files_block([f for f in affected_files if f.role == "modified"])
    return _specialist_prompt(
        strategy, files_block, cwd, diff_ref, DEPENDENCY_TRACER_SCHEMA,
        known_context=[f for f in affected_files if f.role != "modified"],
        inline_diff=inline_diff,
        instructions="""Trace only direct dependencies or callers relevant to changed behavior.
The host already found the context paths below; confirm an edge only if it adds
useful relationship evidence. Do not build a transitive dependency closure, trace
unchanged imports of context files, or inspect external package implementations.
Stop once the direct changed boundaries are mapped; omit unresolved edges.""",
    )


def build_test_mapper_prompt(
    affected_files: list[FileInfo], diff_ref: str, *, cwd: Path, strategy: str,
    inline_diff: str | None = None,
    source_only: bool = True,
) -> str:
    """Build the per-run test-mapper prompt.

    The prompt separates modified targets from static context. Small complete
    diffs can be supplied inline; oversized diffs get bounded advisory excerpts.

    Args:
        affected_files: FileInfo entries for files reachable from the diff.
        diff_ref: Git ref the specialist can diff against when locating test files.
        cwd: Absolute working directory the agent runs in (grounds path resolution).
        strategy: The profile-owned ``exploration.test_mapping`` strategy content.
        inline_diff: Complete diff, inlined only when it fits the shared byte budget.
        source_only: Apply the default source-only mapping scope. Custom profile
            strategies retain all modified targets when this is false.
    """
    targets = mapping_source_files(affected_files, cwd) if source_only else [
        file for file in affected_files if file.role == "modified"
    ]
    files_block = _files_block(targets)
    return _specialist_prompt(
        strategy, files_block, cwd, diff_ref, TEST_MAPPER_SCHEMA,
        known_context=[f for f in affected_files if f not in targets],
        inline_diff=inline_diff,
        instructions="""Map tests only for modified source files in <affected_files>.
Use known context test candidates and conventional paths first. Imports and callers
in the context are not new source targets. Changed tests can supply mapping evidence;
do not search for tests of tests. Do not hunt for tests of documentation or manifests
unless an explicit test relationship is already visible in the supplied evidence.
Read only enough of a candidate to confirm what it covers, then return the mapping;
do not inspect assertion quality, execute tests, or investigate missing coverage.
One confirmed covering test per source completes that source's mapping. Include
additional relationships only when already apparent in supplied evidence, without
extra searches. If known candidates and one conventional-path lookup find no test,
omit the unresolved mapping. Once all listed source targets are mapped or unresolved,
return immediately; do not start mapping the manifests, docs, or tests in context.""" if source_only else (
            "Map the modified targets according to the supplied custom strategy. "
            "Use known context candidates as evidence and return once the requested mappings are established."
        ),
    )


__all__ = [
    "DEPENDENCY_TRACER_SCHEMA",
    "PATTERN_SCANNER_SCHEMA",
    "TEST_MAPPER_SCHEMA",
    "build_dependency_tracer_prompt",
    "build_pattern_scanner_prompt",
    "build_repo_survey_prompt",
    "build_test_mapper_prompt",
    "mapping_source_files",
]
