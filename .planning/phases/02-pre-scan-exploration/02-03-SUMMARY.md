---
phase: 02-pre-scan-exploration
plan: 03
subsystem: exploration
tags: [subagents, prompts, schemas, merge]
requires: [01-01, 01-02]
provides:
  - daydream.prompts.exploration_subagents.EXPLORATION_AGENTS
  - daydream.prompts.exploration_subagents.build_pattern_scanner_prompt
  - daydream.prompts.exploration_subagents.build_dependency_tracer_prompt
  - daydream.prompts.exploration_subagents.build_test_mapper_prompt
  - daydream.exploration.merge_contexts
affects:
  - daydream/exploration.py
  - tests/test_exploration.py
  - tests/test_exploration_runner.py
tech_stack:
  added: []
  patterns:
    - "JSON Schema dict constants mirroring FEEDBACK_SCHEMA style in phases.py"
    - "AgentDefinition registry consumed by Backend.execute(agents=...)"
key_files:
  created:
    - daydream/prompts/exploration_subagents.py
    - tests/test_exploration_runner.py
  modified:
    - daydream/exploration.py
    - tests/test_exploration.py
decisions:
  - "System prompt + dynamic builder split: registry holds static role+schema, builders inject diff/files at call time"
  - "FileInfo dedup keeps the longest summary; Convention dedup keeps first occurrence"
metrics:
  duration: ~10min
  completed: 2026-04-07
requirements: [EXPL-03, EXPL-04]
---

# Phase 2 Plan 3: Subagent Prompts, Schemas, and merge_contexts Summary

Three specialist exploration subagents (pattern-scanner, dependency-tracer, test-mapper) defined as `AgentDefinition` registry entries with mirrored JSON output schemas, plus a pure `merge_contexts()` helper that folds partial `ExplorationContext` results with field-level de-duplication.

## What Was Built

### `daydream/prompts/exploration_subagents.py` (new, 267 lines)
- **Schemas**: `PATTERN_SCANNER_SCHEMA`, `DEPENDENCY_TRACER_SCHEMA`, `TEST_MAPPER_SCHEMA` — JSON Schema dicts mirroring `FEEDBACK_SCHEMA` style.
- **System prompts**: `PATTERN_SCANNER_SYSTEM_PROMPT`, `DEPENDENCY_TRACER_SYSTEM_PROMPT`, `TEST_MAPPER_SYSTEM_PROMPT` — static role descriptions stored in the registry.
- **Dynamic builders**: `build_pattern_scanner_prompt(diff_text, affected_files)`, `build_dependency_tracer_prompt(diff_text, affected_files)`, `build_test_mapper_prompt(diff_text, affected_files)` inject diff and file context at call time.
- **Registry**: `EXPLORATION_AGENTS: dict[str, AgentDefinition]` with three entries, all using `model="inherit"` and `tools=["Read", "Glob", "Grep"]`.
- pattern-scanner explicitly names `CLAUDE.md` and `.coderabbit.yaml` (EXPL-04).

### `daydream/exploration.py` (extended)
- `merge_contexts(*contexts: ExplorationContext) -> ExplorationContext` with documented de-duplication rules:
  - FileInfo dedup on `(path, role)`; entry with longer summary wins.
  - Convention dedup on `name`; first occurrence wins.
  - Dependency dedup on `(source, target, relationship)`.
  - Guidelines string-identity dedup.
  - `raw_notes` joined with `"\n\n"` (skipping empties).
- Always returns a fresh instance with fresh list fields.

### Tests
- `tests/test_exploration_runner.py` (new, 4 tests) — uses `pytest.importorskip` for the orchestrator module so the file stays valid before Plan 04.
- `tests/test_exploration.py` (extended, 7 new tests) — covers empty merge, fresh-list invariant, FileInfo/Dependency/Convention/Guidelines de-dup, and raw_notes joining.

## Verification

- `uv run pytest`: **160 passed** (was 149 before this plan).
- `uv run mypy daydream`: clean.
- `uv run ruff check daydream`: clean.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `tests/test_exploration_runner.py` did not exist**
- **Found during:** Task 1 (read_first step)
- **Issue:** The plan assumes Plan 02-01 (Wave 0) already created `tests/test_exploration_runner.py` with placeholder tests guarded by a module-level `importorskip`. Phase 2 Plan 01 has not been executed yet (no commits exist for it), so the file is absent.
- **Fix:** Created `tests/test_exploration_runner.py` from scratch with the four prompt tests this plan needs. Rather than restructuring an `importorskip` guard that does not exist, each test that touches the future orchestrator uses `pytest.importorskip("daydream.exploration_runner")` inline. The pattern-scanner test uses `pytest.importorskip("daydream.prompts.exploration_subagents")` per the plan's intent.
- **Files modified:** `tests/test_exploration_runner.py`
- **Commit:** `91c7086`

**2. [Rule 1 - Plan inaccuracy] No `xfail` marker existed on `test_merge_pattern_scanner_result`**
- **Found during:** Task 2
- **Issue:** Plan instructs "remove the xfail marker", but no such test (or marker) is present in `tests/test_exploration.py` (it would have been added by Plan 02-01).
- **Fix:** Added the test fresh under the same name with the body the plan specifies. No xfail to remove.
- **Commit:** `785f572`

These deviations did not change the surface area or behavior of the plan — they just adapted to the actual repo state where the upstream Wave-0 plan has not run yet. Plan 02-01 can still execute later without conflict because none of its planned files are touched here, and the importorskip-based test layout is the same shape Plan 02-01 would have produced.

## Commits

- `91c7086` test(02-03): add failing tests for exploration subagent prompts
- `189b444` feat(02-03): add exploration subagent prompts, schemas, and AgentDefinition registry
- `785f572` test(02-03): add failing tests for merge_contexts
- `289737f` feat(02-03): add merge_contexts() with de-duplication

## Requirements Satisfied

- **EXPL-03** (convention detection): pattern-scanner schema exposes `conventions[]`, dependency-tracer schema exposes `affected_files[]` and `dependencies[]`.
- **EXPL-04** (guideline reading): pattern-scanner system prompt and dynamic prompt both contain `CLAUDE.md` and `.coderabbit.yaml` literals; verified by `test_pattern_scanner_prompt_includes_guideline_files`.

## Self-Check: PASSED

- daydream/prompts/exploration_subagents.py: FOUND
- daydream/exploration.py merge_contexts: FOUND
- tests/test_exploration_runner.py: FOUND
- Commits 91c7086, 189b444, 785f572, 289737f: all FOUND in git log
- Full suite: 160 passed
