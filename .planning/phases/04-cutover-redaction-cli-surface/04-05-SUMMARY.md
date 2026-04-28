---
phase: 04-cutover-redaction-cli-surface
plan: 05
subsystem: testing
tags: [ast, regression-guard, pitfall-13, parametrize]

# Dependency graph
requires:
  - phase: 04-04
    provides: Hard removal of _log_debug machinery — preconditions for the AST sweep to pass
provides:
  - "tests/test_cutover_ast.py: AST-level regression guard against re-introduction of legacy debug logging"
  - "Self-excluding parametrized test that walks every .py file under daydream/ and tests/"
  - "Defense-in-depth across four AST node types: Name, Attribute, ImportFrom, Constant"
affects: [05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "AST sweep regression guard via stdlib ast + pytest.parametrize — catches Pitfall 13 (lazy imports inside function bodies) that grep alone misses"

key-files:
  created:
    - tests/test_cutover_ast.py
  modified: []

key-decisions:
  - "Step 0 docstring/comment sweep returned zero hits — Plans 02/03/04 already removed all legitimate references to bracketed log prefixes. No reword commits needed."
  - "Self-exclusion via `Path(__file__).resolve()` comparison rather than a string allowlist or manifest file. Single point of truth, no risk of allowlist drift."
  - "ast.Constant node check covers f-string parts too — Python AST decomposes `f\"[REVERT] failed: {e}\"` into JoinedStr containing a Constant `[REVERT] failed: `. The substring check fires correctly."

patterns-established:
  - "Pattern 1: AST regression guard for symbol removal — when a hard removal must be permanent, an AST sweep at file:line granularity is the right CI gate. Grep misses lazy/nested imports."

requirements-completed:
  - CUT-08

# Metrics
duration: ~6 min
completed: 2026-04-28
---

# Phase 04 Plan 05: AST Cutover Guard Summary

**`tests/test_cutover_ast.py` walks every `.py` file under `daydream/` and `tests/` (73 files total) via `pytest.parametrize`, asserting zero forbidden Name references, zero forbidden Attribute accesses, zero forbidden ImportFrom aliases (catches Pitfall 13 lazy imports), and zero forbidden string-literal log prefixes. Self-excludes via `Path(__file__).resolve()` comparison. Passes on first run; sanity check confirmed it fires with file:line on regression.**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-04-28
- **Completed:** 2026-04-28
- **Tasks:** 1
- **Files modified:** 0 (Step 0 sweep returned zero hits)
- **Files created:** 1

## Step 0 Sweep Results

The pre-write grep for legacy log prefixes inside docstrings/comments returned **zero hits** across `daydream/` and `tests/` (excluding `tests/test_cutover_ast.py`, which did not exist yet). Plans 02, 03, and 04 left no docstring or comment references to the bracketed prefixes, so no reword pass was required.

```
$ grep -rn "\[TEXT\]\|\[PROMPT\]\|\[TOOL_USE\]\|\[TOOL_RESULT\]\|\[TOOL_RESULT_PANEL\]\|\[COST\]\|\[TOKENS\]\|\[REVERT\]\|\[STAGE\]\|\[PARSE_FAIL\]\|\[PARSE_FALLBACK\]\|\[CODEX_RAW\]\|\[CODEX_WARN\]\|\[CODEX_UNHANDLED\]\|\[TTT_REVIEW\]\|\[TTT_PLAN\]\|\[PRE_SCAN\]\|\[PHASE2_ERROR\]\|\[EXECUTE_INIT_ERROR\]\|\[EXECUTE_ERROR\]\|\[SCHEMA\]\|\[SCHEMA_OK\]\|\[SCHEMA_MISS\]\|\[SCHEMA_FALLBACK\]\|\[STRUCTURED_OUTPUT\]\|\[THINKING\]\|\[UI_HEADER\]" daydream/ tests/ --include="*.py" | grep -v "test_cutover_ast.py"
(zero hits)
```

## Final Forbidden Sets

```python
FORBIDDEN_NAMES: set[str] = {
    "_log_debug",
    "_raw_log",
    "_ui_debug",
    "set_debug_log",
    "get_debug_log",
}

FORBIDDEN_ATTRS: set[str] = {
    "debug_log",
}

FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "[REVERT]", "[PARSE_FAIL]", "[STAGE]", "[TTT_REVIEW]", "[TTT_PLAN]",
    "[PRE_SCAN]", "[PROMPT]", "[TEXT]", "[TOOL_USE]", "[TOOL_RESULT]",
    "[TOOL_RESULT_PANEL]", "[COST]", "[TOKENS]", "[CODEX_RAW]",
    "[CODEX_WARN]", "[CODEX_UNHANDLED]", "[SCHEMA]", "[SCHEMA_OK]",
    "[SCHEMA_MISS]", "[SCHEMA_FALLBACK]", "[STRUCTURED_OUTPUT]",
    "[EXECUTE_INIT_ERROR]", "[EXECUTE_ERROR]", "[PHASE2_ERROR]",
    "[PARSE_FALLBACK]", "[THINKING]", "[UI_HEADER]",
)
```

## Coverage

- **Files scanned via parametrize:** 73 (matches `find daydream tests -name '*.py' -not -path '*/__pycache__/*' | wc -l` exactly)
- **First-run result:** 73 passed in 0.46s
- **Full suite after add:** 544 passed (471 prior + 73 new parametrized cases) in 25.79s

## Sanity Check (Regression Detection)

Temporarily appended `_log_debug = lambda x: None` to `daydream/__init__.py` and re-ran the test:

```
FAILED tests/test_cutover_ast.py::test_no_legacy_debug_logging_references[daydream/__init__.py]
Failed: daydream/__init__.py:22: forbidden Name reference '_log_debug'
```

Exact file:line surfaced. Reverted the file; full test suite green again.

## Task Commits

1. **Task 1: Add AST cutover guard against legacy debug-logging revival** — `d7f6082` (test)

## Files Created

- `tests/test_cutover_ast.py` — 136 lines. Stdlib-only (`ast`, `pathlib`, `pytest`). Single parametrized test function `test_no_legacy_debug_logging_references` over `_all_py_files()`. Each file's AST is walked once; Name, Attribute, ImportFrom, and Constant nodes are checked against the forbidden sets.

## Decisions Made

- **Self-exclusion via `Path(__file__).resolve()` comparison.** The plan called for this technique explicitly. No string allowlist, no manifest file, no per-test pragmas — single point of truth that automatically follows the file if it ever moves.
- **Substring match (`prefix in node.value`) for forbidden prefixes.** Catches f-string fragments, multi-line strings, embedded mentions. The cost — possible false positive on a docstring that legitimately mentions a bracketed prefix — is paid one-time at Step 0 and ongoing via the same convention.

## Deviations from Plan

**No deviations.** Step 0 sweep returned zero hits, so no reword was required, and the test passed cleanly on first run.

---

**Total deviations:** 0
**Impact on plan:** N/A.

## Issues Encountered

None.

## Phase 4 Roadmap Success Criterion

> "An AST-level sweep across daydream/ and tests/ finds zero references to _log_debug, debug_log, set_debug_log, get_debug_log..."

**SATISFIED.** `tests/test_cutover_ast.py` is now the executable form of this criterion. CI invocation: `uv run pytest tests/test_cutover_ast.py -x -q`.

## Next Phase Readiness

- Phase 4 is structurally complete: the legacy debug-logging system is gone (Plans 02-04), the CLI surface has been swapped to `--trajectory` with fail-loud semantics (Plan 03), redaction has landed (Plan 01), and the AST sweep guards against regression (this plan).
- All 544 tests pass; ruff and mypy clean across 38 source files plus the new test file.
- Phase 5 (test hardening + documentation) can pick up next.

---
*Phase: 04-cutover-redaction-cli-surface*
*Completed: 2026-04-28*
