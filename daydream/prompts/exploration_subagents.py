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
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION

if TYPE_CHECKING:
    from pathlib import Path

    from daydream.exploration import FileInfo


# JSON Schemas (mirror style of FEEDBACK_SCHEMA in daydream/phases.py)
PATTERN_SCANNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "conventions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["name", "description", "source"],
                "additionalProperties": False,
            },
        },
        "guidelines": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["conventions", "guidelines"],
    "additionalProperties": False,
}

DEPENDENCY_TRACER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "affected_files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": ["modified", "imported_by", "imports", "test"],
                    },
                    "summary": {"type": "string"},
                },
                "required": ["path", "role", "summary"],
                "additionalProperties": False,
            },
        },
        "dependencies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relationship": {
                        "type": "string",
                        "enum": ["imports", "calls", "extends", "tests"],
                    },
                },
                "required": ["source", "target", "relationship"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["affected_files", "dependencies"],
    "additionalProperties": False,
}

TEST_MAPPER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "affected_files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "role": {"type": "string", "enum": ["test"]},
                    "summary": {"type": "string"},
                },
                "required": ["path", "role", "summary"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["affected_files"],
    "additionalProperties": False,
}


def _schema_block(schema: dict[str, Any]) -> str:
    return "Return ONLY a JSON object matching this schema:\n```json\n" + json.dumps(schema, indent=2) + "\n```"


# Dynamic prompt builders (per-run prompts injecting diff + affected files)
def build_pattern_scanner_prompt(affected_files: list[str], diff_ref: str, *, cwd: Path) -> str:
    """Build the per-run pattern-scanner prompt.

    The prompt passes the affected file list and a diff ref. The specialist
    fetches diff content per-file on demand via its own tools rather than
    receiving the full diff inline, which keeps context small for large diffs.

    Args:
        affected_files: Paths of files touched by the diff under review.
        diff_ref: Git ref (e.g. base branch or SHA) the specialist can diff against.
        cwd: Absolute working directory the agent runs in (grounds path resolution).
    """
    files_block = "\n".join(f"- {p}" for p in affected_files) or "- (none yet)"
    return f"""You are the **pattern-scanner** specialist. Detect codebase conventions
and read guideline files relevant to the changes below.

Instructions:
- Read CLAUDE.md at the repo root if it exists.
- Read any other house-style config files you find (ruff.toml, .editorconfig, tsconfig.json, go.mod, Cargo.toml).
- Infer conventions from the code itself where config files are silent.

{CWD_GROUNDING_INSTRUCTION.format(cwd=cwd)}

<affected_files>
{files_block}
</affected_files>

To inspect changes, run `git diff {diff_ref} -- <file>` for any file listed in
<affected_files>, or Read/Grep the file directly. Do NOT dump the full diff —
work file-by-file so your context stays small.

{_schema_block(PATTERN_SCANNER_SCHEMA)}
"""


def build_dependency_tracer_prompt(affected_files: list[FileInfo], diff_ref: str, *, cwd: Path) -> str:
    """Build the per-run dependency-tracer prompt.

    The prompt passes the affected file list and a diff ref. The specialist
    fetches diff content per-file on demand via its own tools rather than
    receiving the full diff inline.

    Args:
        affected_files: FileInfo entries for files reachable from the diff, each carrying a `path` and `role`.
        diff_ref: Git ref the specialist can diff against when probing call sites.
        cwd: Absolute working directory the agent runs in (grounds path resolution).
    """
    files_block = "\n".join(f"- {f.path} ({f.role})" for f in affected_files) or "- (none yet)"
    return f"""You are the **dependency-tracer** specialist. Extend the affected-files
list beyond the static-resolved imports by grepping for call sites and
reading the implementations. For every import or call edge you confirm,
emit a Dependency record.

{CWD_GROUNDING_INSTRUCTION.format(cwd=cwd)}

<affected_files>
{files_block}
</affected_files>

To inspect changes, run `git diff {diff_ref} -- <file>` for any file listed in
<affected_files>, or Read/Grep the file directly. Do NOT dump the full diff —
work file-by-file so your context stays small.

{_schema_block(DEPENDENCY_TRACER_SCHEMA)}
"""


def build_test_mapper_prompt(affected_files: list[str], diff_ref: str, *, cwd: Path) -> str:
    """Build the per-run test-mapper prompt.

    The prompt passes the affected file list and a diff ref. The specialist
    fetches diff content per-file on demand via its own tools rather than
    receiving the full diff inline.

    Args:
        affected_files: Paths of files touched by the diff under review.
        diff_ref: Git ref the specialist can diff against when locating test files.
        cwd: Absolute working directory the agent runs in (grounds path resolution).
    """
    files_block = "\n".join(f"- {p}" for p in affected_files) or "- (none yet)"
    return f"""You are the **test-mapper** specialist. Locate test files for each modified
source file using conventional path mapping (tests/test_X.py, *.test.ts,
*_test.go, tests/<crate>_test.rs). Emit a FileInfo with role="test" for
each test file you find.

{CWD_GROUNDING_INSTRUCTION.format(cwd=cwd)}

<affected_files>
{files_block}
</affected_files>

To inspect changes, run `git diff {diff_ref} -- <file>` for any file listed in
<affected_files>, or Read/Grep the file directly. Do NOT dump the full diff —
work file-by-file so your context stays small.

{_schema_block(TEST_MAPPER_SCHEMA)}
"""


__all__ = [
    "DEPENDENCY_TRACER_SCHEMA",
    "PATTERN_SCANNER_SCHEMA",
    "TEST_MAPPER_SCHEMA",
    "build_dependency_tracer_prompt",
    "build_pattern_scanner_prompt",
    "build_test_mapper_prompt",
]
