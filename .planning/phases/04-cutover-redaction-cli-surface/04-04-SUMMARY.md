---
phase: 04-cutover-redaction-cli-surface
plan: 04
subsystem: refactor
tags: [cutover, dead-code-removal, ast-pitfall, lazy-import, pitfall-13]

# Dependency graph
requires:
  - phase: 04-02
    provides: phases.py / exploration_runner.py already off _log_debug
  - phase: 04-03
    provides: cli.py / runner.py already off _log_debug; RunConfig.debug + debug-init block deleted
provides:
  - "_log_debug function removed from daydream/agent.py (CUT-01)"
  - "AgentState.debug_log field + set_debug_log + get_debug_log removed (CUT-02)"
  - "_raw_log proxy + lazy `from daydream.agent import _log_debug` removed from daydream/backends/codex.py (CUT-05, CUT-06) — Pitfall 13 canary case eliminated"
  - "_ui_debug proxy + 8 call sites removed from daydream/ui.py"
  - "[EXECUTE_INIT_ERROR] / [EXECUTE_ERROR] in run_agent promoted to print_error (D-08)"
affects: [04-05, 05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pre-deletion grep gate: before removing a public symbol, confirm zero callers across the entire repo (excluding the defining file). Documented in plan and re-runnable by anyone."

key-files:
  created: []
  modified:
    - daydream/agent.py
    - daydream/ui.py
    - daydream/backends/codex.py

key-decisions:
  - "Reordered tasks: ran Tasks 2 + 3 (delete _ui_debug + _raw_log proxies) BEFORE Task 1 (delete _log_debug definition). The plan's pre-deletion grep gate would have fired if Task 1 ran first because _ui_debug and _raw_log internally call _log_debug. Reordering preserves the gate's intent (no orphan callers when symbol is deleted) without modifying the plan."
  - "Replaced the empty `elif event_type not in ('turn.started',):` body in codex.py with `pass` rather than deleting the elif. Per plan instructions and to preserve the structural anchor in case future debug logging is reintroduced."

patterns-established:
  - "Pattern 1: Reorder-when-recursive — when a deletion plan's gate would trigger on the plan's own scope (proxy fns that call the symbol being deleted), reorder tasks so the proxies are gone before the symbol is gone."

requirements-completed:
  - CUT-01
  - CUT-02
  - CUT-05
  - CUT-06

# Metrics
duration: ~12 min (inline execution)
completed: 2026-04-28
---

# Phase 04 Plan 04: Hard Removal of `_log_debug` Machinery Summary

**`_log_debug` definition + `AgentState.debug_log` field + getters/setters gone from `agent.py`; `_ui_debug` proxy + 8 call sites gone from `ui.py`; `_raw_log` proxy + lazy import + 5 call sites gone from `backends/codex.py`; `[EXECUTE_INIT_ERROR]`/`[EXECUTE_ERROR]` promoted to `print_error("Backend Init Error", …)` / `print_error("Backend Execution Error", …)`. Phase-wide grep returns zero hits.**

## Performance

- **Duration:** ~12 min (inline execution path)
- **Started:** 2026-04-28
- **Completed:** 2026-04-28
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- `daydream/agent.py` shrank by 73 lines net — `_log_debug` definition, `set_debug_log`, `get_debug_log`, the `debug_log: TextIO | None` field, the `TextIO` import, and ~16 in-flight `_log_debug` call sites in `run_agent` all gone. The remaining `[EXECUTE_INIT_ERROR]` and `[EXECUTE_ERROR]` sites surface via `print_error("Backend Init Error", …)` and `print_error("Backend Execution Error", …)`.
- `daydream/ui.py` shrank by 50 lines net — the `_ui_debug` proxy and all 8 call sites (Bash header, Bash result, panel render, panel set_result, panel finish, registry create, registry finalize-not-in-active, registry finalize-with-panel). The lazy `from daydream.agent import _log_debug` inside `_ui_debug` is the second Pitfall-13 canary; gone.
- `daydream/backends/codex.py` shrank by 13 lines net — the `_raw_log` proxy, its lazy `from daydream.agent import _log_debug`, and all 5 call sites (`[CODEX_RAW]` unparseable, `[CODEX_RAW]`, `[CODEX_WARN]` x2, `[CODEX_UNHANDLED]`). The `turn.started` no-op `elif` keeps a `pass` body for syntactic validity.
- Pre-deletion grep gate ran at `2026-04-28T13:41:53Z` AFTER Tasks 2 + 3 deleted the proxies, and returned zero callers across all four checks.
- Phase-wide verification grep `grep -rn "_log_debug\|_raw_log\|_ui_debug\|debug_log\|set_debug_log\|get_debug_log" daydream/` returns zero matches.
- All 471 existing tests pass; ruff and mypy clean across 38 source files.

## Pre-Deletion Grep Gate (Step 0)

The plan's Step 0 gate is documented as a hard halt mechanism. It fired when initially run because both `_ui_debug` (in `ui.py`) and `_raw_log` (in `codex.py`) internally do `from daydream.agent import _log_debug` followed by `_log_debug(message)`. Both proxies were scheduled for removal by Tasks 2 and 3 of THIS plan. Decision: reorder execution so Tasks 2 + 3 ran first, then re-run the gate, then proceed with Task 1.

After reordering, all four gate checks returned zero output:

```
$ grep -rn "_log_debug" daydream/ tests/ | grep -v "^daydream/agent.py"
(zero)
$ grep -rn "set_debug_log" daydream/ tests/ | grep -v "^daydream/agent.py"
(zero)
$ grep -rn "get_debug_log" daydream/ tests/ | grep -v "^daydream/agent.py"
(zero)
$ grep -rn "debug_log" daydream/ tests/ | grep -v "^daydream/agent.py" | grep -v "set_debug_log\|get_debug_log"
(zero)

Timestamp: 2026-04-28T13:41:53Z
```

## Task Commits

Each task was committed atomically (note reordered execution):

1. **Task 2: Drop _ui_debug proxy and 8 call sites from ui.py** — `9d3f6dc` (refactor)
2. **Task 3: Drop _raw_log proxy, lazy import, and 5 call sites from codex.py** — `919f610` (refactor)
3. **Task 1: Strip _log_debug machinery + promote EXECUTE_* errors in agent.py** — `9d4bb81` (refactor)

## Files Created/Modified

- `daydream/agent.py` — `_log_debug` def removed, `AgentState.debug_log` field removed, `set_debug_log`/`get_debug_log` removed, `TextIO` import removed, `print_error` added to `daydream.ui` import, ~16 silent `_log_debug` calls in `run_agent` removed, two error sites promoted to `print_error("Backend Init Error", ...)` / `print_error("Backend Execution Error", ...)`. Singleton docstring stripped of `set_debug_log` mention.
- `daydream/ui.py` — `_ui_debug` proxy fn removed; 8 call sites deleted (5 single-line, 3 multi-line); zero `from daydream.agent` imports remain.
- `daydream/backends/codex.py` — `_raw_log` proxy fn removed (including its lazy `from daydream.agent import _log_debug`); 5 call sites deleted; the `turn.started` no-op `elif` body becomes `pass`.

## EXECUTE_* Promotion Strings

- `print_error(console, "Backend Init Error", f"{type(exc).__name__}: {exc}")` — at the `event_iter = backend.execute(...)` try/except (was `[EXECUTE_INIT_ERROR]`).
- `print_error(console, "Backend Execution Error", f"{type(exc).__name__}: {exc}")` — at the outer `try:` ... `except Exception as exc:` (was `[EXECUTE_ERROR]`).

## Decisions Made

- **Task reorder (2 → 3 → 1) instead of deleting `_log_debug` definition first.** The plan's pre-deletion grep gate forbids deletion until callers are zero. The proxy functions in `ui.py:_ui_debug` and `codex.py:_raw_log` are themselves callers of `_log_debug` (via lazy imports). Running Task 1 first would have made the gate fire. Reordering keeps the gate's invariant intact and the deletion atomicity unchanged.
- **`pass` rather than full elif removal in codex.py:378.** The `elif event_type not in ("turn.started",):` branch had `_raw_log(...)` as its only body. Replaced with `pass` per plan; preserves the structural anchor.

## Deviations from Plan

**1. Task ordering: Tasks 2 and 3 ran before Task 1.**
- **Found during:** Initial run of Step 0 pre-deletion grep gate.
- **Issue:** The gate found `_log_debug` callers in `daydream/ui.py:30,32` and `daydream/backends/codex.py:38,40` — both inside the `_ui_debug` and `_raw_log` proxies that Tasks 2 and 3 are supposed to delete. The plan's gate logic assumed those proxies were already gone.
- **Fix:** Ran Task 2 (delete `_ui_debug`), then Task 3 (delete `_raw_log`), then re-ran the gate (zero callers), then ran Task 1 (delete the `_log_debug` definition itself).
- **Files modified:** None additional.
- **Verification:** Grep gate timestamp recorded at `2026-04-28T13:41:53Z`; all four checks zero. Phase-wide grep also zero post-Task-1.
- **Committed in:** atomic across `9d3f6dc`, `919f610`, `9d4bb81`.

**No code-level deviations.** Same files, same surgery, same gates.

---

**Total deviations:** 1 (execution-order-only, no code-level deviation)
**Impact on plan:** Zero scope change.

## Issues Encountered

- **Ruff I001 import-block ordering** caught a stylistic issue in `ui.py` after `_ui_debug` removal (a stray blank line). Auto-fixed via `uv run ruff check --fix daydream/ui.py`. Recorded as part of the Task 2 commit.
- **No test fixture updates needed** — confirmed via grep `_log_debug\|debug_log\|set_debug_log\|get_debug_log` against `tests/` returns zero. No test asserted on debug-log content (CUT-03 in Plan 03 already removed the writer; this plan removed the readers).
- **No empty `if`/`else`/`try` blocks needing `pass`** in `ui.py` — every deletion left at least one statement in the surrounding block. Only one site in `codex.py` (the `turn.started` elif) needed `pass`.

## Phase-Wide Verification Grep

```
$ grep -rn "_log_debug\|_raw_log\|_ui_debug\|debug_log\|set_debug_log\|get_debug_log" daydream/
(zero matches)
$ uv run mypy daydream/
Success: no issues found in 38 source files
```

## Next Phase Readiness

- 04-05 (AST-level cutover guard test) can run against a fully-cleared codebase and is expected to detect zero hits — both grep-blind cases (lazy imports inside function bodies) have been verifiably eliminated. The AST sweep test will serve as the regression guard going forward.

---
*Phase: 04-cutover-redaction-cli-surface*
*Completed: 2026-04-28*
