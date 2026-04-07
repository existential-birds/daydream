---
phase: 02-pre-scan-exploration
plan: 01
subsystem: backends + exploration scaffolding
tags: [backend-protocol, tree-sitter, test-scaffolding, tdd]
requires:
  - Phase 1 Backend protocol with agents kwarg
provides:
  - Backend protocol with dict[str, AgentDefinition] agents shape
  - tree-sitter + 4 grammar pins in pyproject.toml/uv.lock
  - Wave 0 test scaffolds for tree_sitter_index and exploration_runner
  - Five unified-diff fixtures for Waves 1-2
affects:
  - daydream/backends/__init__.py
  - daydream/backends/claude.py
  - daydream/backends/codex.py
  - daydream/agent.py
tech-stack:
  added:
    - tree-sitter==0.25.2
    - tree-sitter-python==0.25.0
    - tree-sitter-typescript==0.23.2
    - tree-sitter-go==0.25.0
    - tree-sitter-rust==0.24.2
  patterns:
    - pytest.importorskip for scaffolding tests against not-yet-existing modules
    - xfail(strict=True) placeholders unmasked by downstream waves
key-files:
  created:
    - tests/test_tree_sitter_index.py
    - tests/test_exploration_runner.py
    - tests/fixtures/diffs/python_multifile.diff
    - tests/fixtures/diffs/typescript_multifile.diff
    - tests/fixtures/diffs/go_multifile.diff
    - tests/fixtures/diffs/rust_multifile.diff
    - tests/fixtures/diffs/trivial_single.diff
  modified:
    - daydream/backends/__init__.py
    - daydream/backends/claude.py
    - daydream/backends/codex.py
    - daydream/agent.py
    - tests/test_backend_claude.py
    - tests/test_backend_codex.py
    - tests/test_exploration.py
    - pyproject.toml
    - uv.lock
decisions:
  - Backend.execute(agents=...) uses dict[str, AgentDefinition] so specialist names (pattern-scanner, dependency-tracer, test-mapper) survive into ClaudeAgentOptions verbatim
  - ClaudeBackend forwards the dict unchanged -- no explorer-{i} key rewriting
  - Wave 0 scaffolding uses pytest.importorskip + xfail(strict=True) so collection stays green today and Wave 1 gets free signal when implementations come online
metrics:
  duration: ~15min
  completed: 2026-04-06
---

# Phase 02 Plan 01: Wave 0 Foundation Summary

One-liner: Fix Backend protocol to use dict[str, AgentDefinition], pin tree-sitter grammars, and land the Wave 0 test scaffolds + diff fixtures every downstream Wave 1/2 task will hang its automated verification off.

## What Shipped

### Task 1: Backend protocol agents shape (TDD)

Phase 1 wired `agents` as `list[AgentDefinition]` and ClaudeBackend rewrote the keys to `explorer-{i}`, throwing away the specialist names the Agent SDK uses to dispatch subagents. Phase 2's orchestrator needs to pass `{"pattern-scanner": ..., "dependency-tracer": ..., "test-mapper": ...}` and have those exact names land in `ClaudeAgentOptions`.

Fixed end-to-end:

- `daydream/backends/__init__.py` Backend protocol: `agents: dict[str, AgentDefinition] | None = None`
- `daydream/backends/claude.py` ClaudeBackend.execute: same shape; `options.agents = agents` (no rewriting); docstring updated
- `daydream/backends/codex.py` CodexBackend.execute: signature updated to `dict[str, Any]` for protocol compatibility
- `daydream/agent.py` run_agent: mirrors the dict shape
- `tests/test_backend_claude.py`: replaced the old list-shape test with three new tests (TDD RED committed first, then GREEN):
  - `test_execute_passes_agents_dict_to_options` -- asserts exact key preservation for `pattern-scanner` and `dependency-tracer`
  - `test_execute_passes_none_when_no_agents` -- verifies absent/None case
  - `test_backend_protocol_agents_param_is_dict_typed` -- introspects the Protocol annotation

TDD discipline: RED commit (`ab19f60`) lands the failing tests, GREEN commit (`98696de`) lands the implementation.

### Task 2: tree-sitter deps + Wave 0 scaffolds

Pinned and smoke-imported all four grammars against tree-sitter 0.25.2 -- no ABI mismatches:

```
import tree_sitter_python, tree_sitter_typescript, tree_sitter_go, tree_sitter_rust
Language(tree_sitter_python.language())  # OK
Language(tree_sitter_typescript.language_typescript())  # OK
Language(tree_sitter_typescript.language_tsx())  # OK
Language(tree_sitter_go.language())  # OK
Language(tree_sitter_rust.language())  # OK
```

Five unified-diff fixtures under `tests/fixtures/diffs/`:

- `python_multifile.diff` -- 2 files, import edge `api.py -> models.py` (456 B)
- `typescript_multifile.diff` -- 2 files, import edge `api.ts -> models.ts` (399 B)
- `go_multifile.diff` -- 2 files, import edge `api.go -> models/user.go` (392 B)
- `rust_multifile.diff` -- 2 files, `use` edge `api.rs -> models.rs` (369 B)
- `trivial_single.diff` -- 1 file, README change -- drives the "skip tier" case (263 B)

Two new test modules, both `pytest.importorskip`-guarded so pytest collection is green today and Wave 1 only has to remove the guard + xfail markers:

- `tests/test_tree_sitter_index.py`: 7 xfail placeholders covering python/typescript/go/rust impact surfaces, default depth, unsupported language graceful degradation, and deleted-file no-raise (Pitfall 4)
- `tests/test_exploration_runner.py`: 4 xfail placeholders for skip/single/parallel tiers and the pattern-scanner prompt guideline file inclusion. Includes a `RecordingBackend` that captures the agents dict argument -- the exact shape Task 1 just fixed

Extended `tests/test_exploration.py` with `test_merge_pattern_scanner_result` xfail placeholder for Wave 1's `merge_contexts()`.

## Verification

```
uv run pytest                 # 150 passed, 2 skipped, 1 xfailed
uv run mypy daydream          # Success: no issues found in 14 source files
uv run ruff check daydream tests  # All checks passed
```

The 2 "skipped" are the importorskip-guarded scaffolding modules (expected). The 1 "xfailed" is `test_merge_pattern_scanner_result` (expected, strict).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Pulled Phase 1 baseline into the worktree**

- **Found during:** initial orientation
- **Issue:** This executor runs in a worktree branched from `main` (90b1d55), but the plan (and `.planning/` tree) lives on `anderskev/init-gsd`. The worktree lacked `daydream/exploration.py`, the Phase 1 additions to `backends/`, and `.planning/` entirely.
- **Fix:** Checked out `.planning/`, `daydream/`, `tests/`, `pyproject.toml`, and `uv.lock` from `anderskev/init-gsd` and committed the baseline separately (`af805c6`) before starting Task 1.
- **Files modified:** everything in the baseline commit
- **Commit:** `af805c6`

**2. [Rule 2 - Correctness] Updated CodexBackend and run_agent signatures**

- **Found during:** Task 1 GREEN
- **Issue:** The Backend protocol change is load-bearing -- every implementer has to match or mypy breaks. CodexBackend still had `list[Any]` and `daydream/agent.py:run_agent` still had `list[AgentDefinition]`.
- **Fix:** Updated both to `dict[str, ...]` and fixed the Codex test that was passing a list.
- **Files modified:** `daydream/backends/codex.py`, `daydream/agent.py`, `tests/test_backend_codex.py`
- **Commit:** `98696de`

**3. [Rule 3 - Blocking] Cleaned up three pre-existing ruff errors**

- **Found during:** final verification
- **Issue:** `make check` was a stated success criterion. Three pre-existing lint errors (unused `subprocess` in `test_loop.py`, unused `AsyncMock` in `test_exploration.py`, unsorted imports in `test_backends_init.py`) would block `make check` from going green. Plan scope technically excludes these, but the success criterion explicitly requires `make check` to pass.
- **Fix:** Removed unused imports and reordered the one import block. No behavior changes.
- **Commit:** `ec270f8`

### Authentication Gates

None.

## Commits

| # | Hash | Subject |
|---|------|---------|
| 0 | af805c6 | chore(02-01): sync Phase 1 baseline from init-gsd |
| 1 | ab19f60 | test(02-01): add failing tests for dict[str, AgentDefinition] agents shape (RED) |
| 2 | 98696de | feat(02-01): Backend protocol uses dict[str, AgentDefinition] for agents (GREEN) |
| 3 | 30c8dd1 | feat(02-01): pin tree-sitter deps and add Wave 0 test scaffolds + diff fixtures |
| 4 | ec270f8 | chore(02-01): clean up pre-existing lint errors to pass make check |

## Known Stubs

None. The `importorskip`-guarded test modules are intentional Wave 0 scaffolds, explicitly called out by the plan and tracked by strict xfail markers. They will light up automatically when Waves 1-2 land `daydream/tree_sitter_index.py` and `daydream/exploration_runner.py`.

## What This Unlocks

Every Wave 1 and Wave 2 task in Phase 2 now has:

- A real `<automated>` command to run (pytest against an already-collecting test file)
- A corresponding fixture for each supported language
- A Backend protocol shape that accepts named specialist dicts without rewriting

The orchestrator work in Wave 1 can assume `pattern-scanner`, `dependency-tracer`, and `test-mapper` keys survive the round-trip to `ClaudeAgentOptions`.

## Self-Check: PASSED

All 9 referenced files exist on disk. All 5 commit hashes (af805c6, ab19f60, 98696de, 30c8dd1, ec270f8) are reachable from HEAD.
