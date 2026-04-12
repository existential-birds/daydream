"""Exploration subagent prompts, output schemas, and AgentDefinition registry.

Three specialist subagents power pre-scan exploration:

- **pattern-scanner**: detects codebase conventions and reads guideline files
  (CLAUDE.md, .coderabbit.yaml, ruff.toml, etc.) -- satisfies EXPL-04.
- **dependency-tracer**: extends the static-resolved import graph by grepping
  call sites and emits Dependency edges.
- **test-mapper**: locates test files for each modified source file via
  conventional path mapping.

The orchestrator (Plan 04) wires these into Backend.execute() via the
EXPLORATION_AGENTS registry and merges their partial ExplorationContext
results with merge_contexts() in daydream.exploration.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from claude_agent_sdk.types import AgentDefinition

if TYPE_CHECKING:
    from daydream.exploration import FileInfo


# ---------------------------------------------------------------------------
# JSON Schemas (mirror style of FEEDBACK_SCHEMA in daydream/phases.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# System prompts (static role description used by AgentDefinition.prompt)
# ---------------------------------------------------------------------------

PATTERN_SCANNER_SYSTEM_PROMPT = """You are the **pattern-scanner** specialist. Your job is to detect the
conventions and house style of a codebase so the review agent does not
recommend changes that contradict existing patterns.

Instructions:
- Read CLAUDE.md at the repo root if it exists.
- Read .coderabbit.yaml at the repo root if it exists.
- Read any other house-style config files you find (ruff.toml, .editorconfig, tsconfig.json, go.mod, Cargo.toml).
- Infer conventions from the code itself where config files are silent.

""" + _schema_block(PATTERN_SCANNER_SCHEMA)


DEPENDENCY_TRACER_SYSTEM_PROMPT = """You are the **dependency-tracer** specialist. You extend the
statically-resolved import graph by grepping for call sites and reading
implementation files. Emit a Dependency edge for every import or call you
confirm, and add any newly-discovered files to affected_files.

""" + _schema_block(DEPENDENCY_TRACER_SCHEMA)


TEST_MAPPER_SYSTEM_PROMPT = """You are the **test-mapper** specialist. For every modified source file,
locate its tests using conventional path mapping:

- `tests/test_X.py` for `daydream/X.py`
- `*.test.ts` sibling for TypeScript modules
- `*_test.go` sibling for Go modules
- `tests/<crate>_test.rs` for Rust crates

Emit a FileInfo entry with role="test" for each test file you find.

""" + _schema_block(TEST_MAPPER_SCHEMA)


# ---------------------------------------------------------------------------
# Dynamic prompt builders (per-run prompts injecting diff + affected files)
# ---------------------------------------------------------------------------


def build_pattern_scanner_prompt(diff_text: str, affected_files: list[str]) -> str:
    """Build the per-run pattern-scanner prompt with diff and known files injected."""
    files_block = "\n".join(f"- {p}" for p in affected_files) or "- (none yet)"
    return f"""You are the **pattern-scanner** specialist. Detect codebase conventions
and read guideline files relevant to the changes below.

Instructions:
- Read CLAUDE.md at the repo root if it exists.
- Read .coderabbit.yaml at the repo root if it exists.
- Read any other house-style config files you find (ruff.toml, .editorconfig, tsconfig.json, go.mod, Cargo.toml).
- Infer conventions from the code itself where config files are silent.

<affected_files>
{files_block}
</affected_files>

<diff>
{diff_text}
</diff>

{_schema_block(PATTERN_SCANNER_SCHEMA)}
"""


def build_dependency_tracer_prompt(diff_text: str, affected_files: list[FileInfo]) -> str:
    """Build the per-run dependency-tracer prompt."""
    files_block = "\n".join(f"- {f.path} ({f.role})" for f in affected_files) or "- (none yet)"
    return f"""You are the **dependency-tracer** specialist. Extend the affected-files
list beyond the static-resolved imports by grepping for call sites and
reading the implementations. For every import or call edge you confirm,
emit a Dependency record.

<affected_files>
{files_block}
</affected_files>

<diff>
{diff_text}
</diff>

{_schema_block(DEPENDENCY_TRACER_SCHEMA)}
"""


def build_test_mapper_prompt(diff_text: str, affected_files: list[str]) -> str:
    """Build the per-run test-mapper prompt."""
    files_block = "\n".join(f"- {p}" for p in affected_files) or "- (none yet)"
    return f"""You are the **test-mapper** specialist. Locate test files for each modified
source file using conventional path mapping (tests/test_X.py, *.test.ts,
*_test.go, tests/<crate>_test.rs). Emit a FileInfo with role="test" for
each test file you find.

<affected_files>
{files_block}
</affected_files>

<diff>
{diff_text}
</diff>

{_schema_block(TEST_MAPPER_SCHEMA)}
"""


# ---------------------------------------------------------------------------
# AgentDefinition registry consumed by the orchestrator (Plan 04)
# ---------------------------------------------------------------------------

EXPLORATION_AGENTS: dict[str, AgentDefinition] = {
    "pattern-scanner": AgentDefinition(
        description="Detects codebase conventions and reads guideline files (CLAUDE.md, .coderabbit.yaml, etc).",
        prompt=PATTERN_SCANNER_SYSTEM_PROMPT,
        tools=["Read", "Glob", "Grep"],
        model="inherit",
    ),
    "dependency-tracer": AgentDefinition(
        description="Extends the import graph by grepping call sites and emits Dependency edges.",
        prompt=DEPENDENCY_TRACER_SYSTEM_PROMPT,
        tools=["Read", "Glob", "Grep"],
        model="inherit",
    ),
    "test-mapper": AgentDefinition(
        description="Locates test files for modified source files via conventional path mapping.",
        prompt=TEST_MAPPER_SYSTEM_PROMPT,
        tools=["Read", "Glob", "Grep"],
        model="inherit",
    ),
}


__all__ = [
    "DEPENDENCY_TRACER_SCHEMA",
    "DEPENDENCY_TRACER_SYSTEM_PROMPT",
    "EXPLORATION_AGENTS",
    "PATTERN_SCANNER_SCHEMA",
    "PATTERN_SCANNER_SYSTEM_PROMPT",
    "TEST_MAPPER_SCHEMA",
    "TEST_MAPPER_SYSTEM_PROMPT",
    "build_dependency_tracer_prompt",
    "build_pattern_scanner_prompt",
    "build_test_mapper_prompt",
]
