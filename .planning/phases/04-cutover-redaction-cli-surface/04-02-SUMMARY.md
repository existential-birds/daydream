---
phase: 04-cutover-redaction-cli-surface
plan: 02
subsystem: logging
tags: [debug-logging, print_warning, get_quiet_mode, cutover, D-08, D-09]

requires:
  - phase: 02-recorder-core-event-enrichment-mapping
    provides: TrajectoryRecorder lifecycle, get_current_recorder ContextVar, redact_step API
  - phase: 03-subagent-wiring-parallel-continuation
    provides: maybe_fork helper used in phase_per_stack_reviews
provides:
  - phases.py free of _log_debug / get_debug_log references
  - exploration_runner.py free of _log_debug / [PRE_SCAN] references
  - Three operational warnings in phases.py promoted to print_warning + quiet-wrapped per D-09
  - Lazy from-import in exploration_runner.py reduced to run_agent only
affects:
  - 04-03 (agent.py removal of _log_debug definition + AgentState.debug_log)
  - 04-04 (runner.py [PHASE2_ERROR] cutover + debug-init block deletion)
  - 04-05 (CUT-08 AST sweep — must still pass after this plan)

tech-stack:
  added: []
  patterns:
    - "D-09 quiet-wrap idiom: `if not get_quiet_mode(): print_warning(console, msg)` for newly-promoted warning sites"
    - "D-08 silent-removal idiom: bare `except Exception: pass` with `# noqa: BLE001 - best-effort path; exploration degrades silently per D-08`"

key-files:
  created: []
  modified:
    - daydream/phases.py
    - daydream/exploration_runner.py

key-decisions:
  - "Followed plan literally for all 8 cutover steps in phases.py and 3 in exploration_runner.py"
  - "Auto-fix Rule 3: dropped unused `clean_result` assignment after deleting its only consumer (the [REVERT] git-clean log block) to keep ruff F841 clean"

patterns-established:
  - "Quiet-mode wrap: every newly-promoted print_warning site that previously fell through to a debug log gets wrapped in `if not get_quiet_mode():` per D-09"
  - "Silent removal of best-effort failure logs: bare-except annotated with D-08 rationale instead of degraded debug write"

requirements-completed: [CUT-04, CUT-07]

duration: ~5min
completed: 2026-04-28
---

# Phase 04 Plan 02: phases.py + exploration_runner.py Cutover Summary

**Cut `daydream/phases.py` (3 promotions to quiet-wrapped `print_warning`, 4 silent removals) and `daydream/exploration_runner.py` (3 [PRE_SCAN] silent removals) free of legacy `_log_debug` / `get_debug_log` per D-08/D-09.**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-04-28T07:50Z (approx)
- **Completed:** 2026-04-28T07:55:56Z
- **Tasks:** 2 / 2
- **Files modified:** 2

## Accomplishments

- `daydream/phases.py`:
  - Imports cleaned: dropped `_log_debug`, `get_debug_log`; added `get_quiet_mode`.
  - 3 operational sites promoted to `print_warning(console, ...)` AND wrapped with `if not get_quiet_mode():` per D-09:
    - `[REVERT]` failed exception path → `Revert failed: ...`
    - `[TTT_REVIEW]` unexpected result type → `TTT review returned unexpected result type: ...`
    - `[TTT_PLAN]` unexpected result type → consolidated into single quiet-wrapped `Failed to generate structured plan; agent returned <type>`. The pre-existing unwrapped `print_warning(console, "Failed to generate structured plan")` was absorbed into the consolidated call (the new promoted text supersedes it; this is a NEW promoted site under D-09, so wrapping applies).
  - 4 redundant agent-event-mirroring sites silently removed:
    - `[REVERT]` git-clean success informational log
    - `[PARSE_FAIL]` diagnostic block (next-line `print_warning` + raised `ValueError` carry the signal)
    - `[PARSE_FALLBACK]` empty-result log
    - `[STAGE]` per-stack `get_debug_log()` failure block (the `print_warning` immediately after stays AS-IS — out of scope per plan)
- `daydream/exploration_runner.py`:
  - Lazy import reduced from `from daydream.agent import _log_debug, run_agent` to `from daydream.agent import run_agent`.
  - 3 `[PRE_SCAN]` `_log_debug` lines silently removed (best-effort path per D-08); bare `except Exception:` blocks now annotated `# noqa: BLE001 - best-effort path; exploration degrades silently per D-08`.

## Task Commits

1. **Task 1: Promote and silently remove _log_debug sites in daydream/phases.py** — `96e6cd7` (refactor)
2. **Task 2: Remove [PRE_SCAN] _log_debug sites from daydream/exploration_runner.py** — `27d844e` (refactor)

## Files Created/Modified

- `daydream/phases.py` — Cleaned phase functions: 3 quiet-wrapped print_warning promotions, 4 silent removals, imports updated to add `get_quiet_mode`.
- `daydream/exploration_runner.py` — Lazy import trimmed; 3 `[PRE_SCAN]` log lines silently removed; bare-except blocks gain D-08 rationale comment.

## D-09 Quiet-Wrap Confirmation

All 3 newly-promoted `print_warning` sites in phases.py are wrapped with `if not get_quiet_mode():`:

| Site | File:Line | Wrapped |
|------|-----------|---------|
| Revert failed | `daydream/phases.py:494-497` | yes |
| TTT review unexpected result type | `daydream/phases.py:1229-1231` | yes |
| Failed to generate structured plan (consolidated) | `daydream/phases.py:1364-1369` | yes |

Verified by visual read of each site (the multi-line consolidated TTT_PLAN warning is wrapped as a unit; `grep -B 1` only sees the `print_warning(` opening line, but the `if not get_quiet_mode():` is the parent block's gating condition).

## Lines-Changed Totals

```
 daydream/exploration_runner.py | 11 +++++------
 daydream/phases.py             | 30 +++++++++++-------------------
 2 files changed, 16 insertions(+), 25 deletions(-)
```

Net deletion of 9 lines. Roughly:
- phases.py: 11 promoted/quiet-wrap insertions, 19 removals (8 `_log_debug` lines + the 4-line `[STAGE]` `get_debug_log` block + 4 import-list / dead-assignment cleanups + the `clean_result` assignment).
- exploration_runner.py: 5 insertions (`pass` + updated noqa rationale on 2 except blocks + cleaned lazy import line + dropped `as exc` clauses), 6 deletions (3 `_log_debug` calls + adjusted exception clauses).

## Decisions Made

- Plan was followed literally — no scope expansion.
- Auto-fix Rule 3 application: the `[REVERT]` git-clean success log was the only consumer of `clean_result.stdout`. Deleting the consumer turned the `clean_result = subprocess.run(...)` assignment into an unused variable that ruff F841 would flag. Minimal fix: drop the `clean_result =` assignment AND the `text=True` flag (no longer needed since stdout is unread). The `subprocess.run(...)` itself stays as-is for its side effects + `check=True` semantics.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Removed unused `clean_result` assignment in `_revert_repository`**
- **Found during:** Task 1 (phases.py cutover, Step B)
- **Issue:** The plan instructed deletion of the `if clean_result.stdout.strip(): _log_debug(...)` block. After deletion, the `clean_result = subprocess.run(...)` assignment had no consumer, which ruff would flag as F841 (unused variable). Plan acceptance criterion required `uv run ruff check daydream/phases.py` to pass.
- **Fix:** Replaced `clean_result = subprocess.run(... text=True ...)` with `subprocess.run(... )` (dropped the assignment and the now-unnecessary `text=True` flag). Side-effect semantics (`check=True` raising on non-zero exit, captured stderr/stdout swallowed) are unchanged.
- **Files modified:** `daydream/phases.py` (lines 486-493 area)
- **Verification:** `uv run ruff check daydream/phases.py` exits 0; `uv run pytest tests/test_phases.py tests/test_loop.py -x -q` exits 0.
- **Committed in:** `96e6cd7` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 3 - blocking lint cleanup)
**Impact on plan:** Trivial — preserves ruff acceptance criterion without altering observable behavior.

## Issues Encountered

None — all acceptance criteria for both tasks pass on first run; full `uv run pytest tests/` (448 tests) passes.

## User Setup Required

None — pure refactor, no external configuration touched.

## Next Phase Readiness

- Plans 04-01 (Redactor regex), 04-03 (agent.py `_log_debug` removal), 04-04 (runner.py + CLI surface) can proceed independently. This plan does not touch `agent.py`, `runner.py`, or `cli.py`.
- The CUT-08 AST sweep test (Plan 04-05) will be cleaner now: `phases.py` and `exploration_runner.py` no longer reference `_log_debug` / `get_debug_log` so they will not require self-exclusion when that test lands.
- The 5 remaining `_log_debug`/`get_debug_log` references in this milestone live in `agent.py` (definition + 20 call sites + `AgentState.debug_log`), `runner.py` (init block + `[PHASE2_ERROR]`), `backends/codex.py` (lazy import + `_raw_log`), and `ui.py` (`_ui_debug` proxy) — owned by other plans in this phase.

## Self-Check: PASSED

- `daydream/phases.py` exists and contains zero `_log_debug` / `get_debug_log` references — verified.
- `daydream/exploration_runner.py` exists and contains zero `_log_debug` / `[PRE_SCAN]` references — verified.
- Commit `96e6cd7` (Task 1) and `27d844e` (Task 2) exist on the worktree branch — verified via `git log`.
- `uv run pytest tests/ -x -q` exits 0 (448 passed) — verified.
- `uv run ruff check daydream/phases.py daydream/exploration_runner.py` exits 0 — verified per-file in Tasks 1 and 2.

---
*Phase: 04-cutover-redaction-cli-surface*
*Completed: 2026-04-28*
