---
phase: 04-cutover-redaction-cli-surface
plan: 03
subsystem: cli
tags: [argparse, signal-handling, trajectory, atif, contextvar, fail-loud]

# Dependency graph
requires:
  - phase: 04-01
    provides: TrajectoryRecorder.write_partial (synchronous, signal-safe flush) + Redactor for the redaction layer
  - phase: 04-02
    provides: phases.py / exploration_runner.py already migrated off _log_debug
provides:
  - "argparse rejects --debug as unrecognized argument (D-05 hard reject)"
  - "argparse accepts --trajectory <path>, stores into RunConfig.trajectory_path"
  - "_signal_handler reads recorder via ContextVar and flushes <path>.partial before raising KeyboardInterrupt (D-07)"
  - "RunConfig.debug field removed; .review-debug-{ts}.log initialization gone; contextlib + datetime imports dropped"
  - "[PHASE2_ERROR] _log_debug sites in runner.py promoted to print_error (D-08)"
  - "TrajectoryRecorder.__aexit__ branches on explicit_path: print_error + raise SystemExit(2) when --trajectory was passed; print_warning and continue otherwise (D-06)"
affects: [04-04, 04-05, 05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ContextVar reads from synchronous signal handlers (Python signal docs: handlers run on main thread, ContextVar reads are atomic)"
    - "Two-branch fail-loud: SystemExit(2) for explicit user-provided paths, degrade-with-warning for implicit defaults"

key-files:
  created: []
  modified:
    - daydream/cli.py
    - daydream/runner.py
    - daydream/trajectory.py
    - daydream/deep/orchestrator.py

key-decisions:
  - "Inline raise SystemExit(2) from exc inside the except clause (rather than capture-and-raise-after-finally) — finally block still runs first per Python semantics, code reads cleaner, and the literal `from exc` chain is preserved."
  - "explicit_path is sourced from `config.trajectory_path is not None` at every TrajectoryRecorder instantiation site (4 sites: PR, TTT, NORMAL, DEEP). No alternative heuristic."
  - "import contextlib and from datetime import datetime were both dropped from runner.py — both were used only by the deleted debug-init block."

patterns-established:
  - "Pattern 1: Per-run ContextVar with synchronous signal-handler access — write_partial is sync precisely so it works from a signal handler."
  - "Pattern 2: User-intent-driven fail-loud — explicit user request (passed --trajectory) gets fail-loud; defaulted behavior degrades."

requirements-completed:
  - CLI-01
  - CLI-02
  - CLI-03
  - CLI-04
  - CLI-05
  - CUT-03

# Metrics
duration: ~25 min (inline execution after 2 failed worktree subagent attempts)
completed: 2026-04-28
---

# Phase 04 Plan 03: CLI Cutover Summary

**`--debug` removed and replaced by `--trajectory <path>`; SIGINT flushes partial trajectory; RunConfig.debug + debug-file init block deleted; `[PHASE2_ERROR]` promoted to `print_error`; D-06 explicit-path fail-loud branch wired through `TrajectoryRecorder.__aexit__`.**

## Performance

- **Duration:** ~25 min (inline execution path)
- **Started:** 2026-04-28
- **Completed:** 2026-04-28
- **Tasks:** 4
- **Files modified:** 4

## Accomplishments

- `--debug` is hard-rejected by argparse — `daydream --debug ...` exits non-zero with `unrecognized arguments: --debug` (D-05)
- `--trajectory <path>` is the only run-time observability flag; `daydream --help` shows `<target>/.daydream/trajectory.json` as the default-path note (CLI-05)
- `_signal_handler` reads the active recorder via `get_current_recorder()` and calls `write_partial()` before raising `KeyboardInterrupt` (D-07, CLI-03)
- `RunConfig.debug`, the `.review-debug-{ts}.log` initialization, the `contextlib.ExitStack` scaffolding, and the `contextlib` + `datetime` imports were all removed from `runner.py` (CUT-03)
- Both `[PHASE2_ERROR]` `_log_debug` sites in `runner.py` were promoted to `print_error(console, "Phase 2 Error", str(exc))` so users see the underlying parse exception (D-08)
- `TrajectoryRecorder` gained an `explicit_path: bool = False` field; `__aexit__` write failures now branch: explicit-path → `print_error` + `raise SystemExit(2) from exc`; implicit-default → `print_warning` and continue (D-06, CORE-09)
- Every `TrajectoryRecorder(...)` instantiation propagates `explicit_path=config.trajectory_path is not None` — 3 sites in `runner.py` (PR, TTT, NORMAL) + 1 in `daydream/deep/orchestrator.py` (DEEP)

## Task Commits

Each task was committed atomically:

1. **Task 1: Swap --debug for --trajectory in cli.py argparse** — `dea3e43` (feat)
2. **Task 2: Wire SIGINT to TrajectoryRecorder.write_partial in cli.py** — `ec7d7c8` (feat)
3. **Task 3: Drop RunConfig.debug, debug-init block, promote PHASE2_ERROR** — `cfe61eb` (refactor)
4. **Task 4: D-06 explicit-path fail-loud in TrajectoryRecorder.__aexit__** — `8b2ce2e` (feat)

## Files Created/Modified

- `daydream/cli.py` — argparse drops `--debug`, gains `--trajectory PATH`; `_signal_handler` flushes partial trajectory before `KeyboardInterrupt`; new imports: `from pathlib import Path`, `from daydream.trajectory import get_current_recorder`
- `daydream/runner.py` — `RunConfig.debug` field removed; debug-init block + `contextlib.ExitStack` scaffolding deleted; `import contextlib` and `from datetime import datetime` dropped; `_log_debug`/`set_debug_log` removed from `daydream.agent` import; `[PHASE2_ERROR]` promoted to `print_error("Phase 2 Error", str(exc))` at both loop and single-pass branches; `explicit_path=config.trajectory_path is not None` propagated to PR, TTT, NORMAL `TrajectoryRecorder(...)` sites
- `daydream/trajectory.py` — `explicit_path: bool = False` dataclass field added; `__aexit__` write-failure handler now branches on `self.explicit_path` (explicit → `print_error` + `raise SystemExit(2) from exc`; implicit → `print_warning`); `print_error` added to the `daydream.ui` import
- `daydream/deep/orchestrator.py` — `explicit_path=config.trajectory_path is not None` propagated to the `TrajectoryRecorder(...)` site in `run_deep()`

## Decisions Made

- **Inline `raise SystemExit(2) from exc` inside the except clause.** Originally captured the exception to a sentinel and raised after the `finally`, but the inline form is cleaner and Python guarantees `finally` still runs before the SystemExit propagates. The literal `from exc` chain is preserved, satisfying the plan's grep acceptance criterion.
- **Sole source of truth for `explicit_path` is `config.trajectory_path is not None`.** No alternative heuristic. Plan called this out explicitly.
- **Dropped both `import contextlib` and `from datetime import datetime` from runner.py.** Both were used only by the deleted debug-init block; verified via grep before deletion.
- **Removed the `# Set up debug logging if enabled` comment line and the empty `with contextlib.ExitStack()` wrapper in one motion.** Per PATTERNS Option 1 (cleaner). Body of the wrapper (lines that were originally inside `with`) was dedented by 4 spaces — verified by ruff + mypy + full test suite (471 passed).

## Deviations from Plan

**1. Execution path: inline rather than subagent-in-worktree.**
- **Found during:** Wave 2 dispatch.
- **Issue:** Two consecutive `gsd-executor` subagent attempts in worktree isolation hit a hard `Edit/Write` permission denial at the runtime tool layer despite pre-approval. The agent surfaced exhaustive enumeration of every required change after the second attempt.
- **Fix:** User chose "Inline execution (no subagent)" via `AskUserQuestion`. The orchestrator (this assistant) executed all four tasks directly in the main working tree, mirroring `--interactive` mode behavior. Tests, lint, and mypy all pass.
- **Files modified:** none additional vs. plan.
- **Verification:** Same gates as the plan specified — full pytest suite, ruff, mypy, manual D-06 verification script.
- **Committed in:** `dea3e43`, `ec7d7c8`, `cfe61eb`, `8b2ce2e` (atomic per task).

**No code-level deviations.** Every change follows the plan as written; only the execution mechanism changed.

---

**Total deviations:** 1 (execution-path-only, no code-level deviation)
**Impact on plan:** Zero scope change; same files, same patterns, same gates.

## Issues Encountered

- **Two failed worktree subagent attempts** before falling back to inline execution. Both attempts hit `Permission to use Edit/Write has been denied` on `daydream/cli.py` at the very first edit, even with explicit pre-approval text in the agent prompt. Earlier waves (04-01, 04-02) succeeded with the same tool surface — root cause unknown but localized to this run/session/worktree configuration. Worktrees from both failed attempts were torn down cleanly (`git worktree unlock` + `git worktree remove --force` + `git branch -D`).

## Manual Validation

D-06 fail-loud verification script (run inline during execution):

```python
import asyncio
from pathlib import Path
from daydream.trajectory import TrajectoryRecorder, DaydreamRunFlow

async def main_explicit():
    rec = TrajectoryRecorder(
        path=Path("/dev/null/forbidden.json"),
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=Path("/tmp"),
        agent_model_name="opus",
        explicit_path=True,
    )
    rec._write = lambda: (_ for _ in ()).throw(OSError("simulated disk full"))
    try:
        async with rec:
            pass
    except SystemExit as exc:
        print(f"explicit:  OK SystemExit({exc.code}) chained from {type(exc.__cause__).__name__}: {exc.__cause__}")
        return
    print("explicit:  FAIL no SystemExit")

# Same with explicit_path=False — must NOT raise SystemExit.
```

Output:
```
explicit:  OK SystemExit(2) chained from OSError: simulated disk full
implicit:  OK degraded with warning, no SystemExit
```

## Help Output Verification

```
$ uv run python -m daydream.cli --help
...
                [--trajectory PATH] [--cleanup | --no-cleanup] [--review-only]
...
  --trajectory PATH     Write ATIF v1.6 trajectory JSON to this path (default:
                        <target>/.daydream/trajectory.json)
```

## Lines-Changed Totals

| File | Insertions | Deletions |
|------|-----------:|----------:|
| daydream/cli.py | 23 | 7 |
| daydream/runner.py | 147 | 458 |
| daydream/trajectory.py | 15 | 4 |
| daydream/deep/orchestrator.py | 1 | 0 |

(Per `git diff --stat 8aedd5e..HEAD`: 478 insertions / 329 deletions across the four files. The runner.py "deletions" count is inflated by the 4-space dedent of ~315 lines after removing the `with contextlib.ExitStack()` wrapper — net structural change is the deleted debug-init block + the import cleanups + the two PHASE2_ERROR promotions.)

## Next Phase Readiness

- 04-04 (hard removal of `_log_debug` machinery from `agent.py`, `ui.py`, `backends/codex.py`) can run with no remaining call sites in `cli.py`/`runner.py`/`phases.py`/`exploration_runner.py`.
- 04-05 (AST-level cutover guard) will catch any latent imports — no expected hits in the migrated files.
- All 471 tests pass; ruff and mypy clean.

---
*Phase: 04-cutover-redaction-cli-surface*
*Completed: 2026-04-28*
