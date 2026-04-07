---
phase: 02-pre-scan-exploration
plan: 04
subsystem: exploration-runner
tags: [orchestrator, subagents, ui, codex-guard]
requires:
  - 02-01
  - 02-02
  - 02-03
provides:
  - daydream.exploration_runner.pre_scan
  - RunConfig.exploration_context
  - RunConfig.exploration_depth
  - daydream.ui.ExplorationLivePanel
  - CodexBackend agents= guard
affects:
  - daydream/runner.py
  - daydream/phases.py
  - daydream/ui.py
  - daydream/backends/codex.py
tech-stack:
  added: []
  patterns:
    - tier-driven subagent orchestration (skip/single/parallel)
    - envelope schema for combined specialist output
    - Live context-manager UI panel with throbber/done/failed states
key-files:
  created:
    - daydream/exploration_runner.py
    - tests/test_runner.py
    - tests/test_ui.py
  modified:
    - daydream/runner.py
    - daydream/phases.py
    - daydream/ui.py
    - daydream/backends/codex.py
    - tests/test_exploration_runner.py
    - tests/test_integration.py
    - tests/test_backend_codex.py
decisions:
  - "Inline pre_scan call sites in run()/run_trust() rather than helper -- keeps acceptance grep counts visible and removes one indirection"
  - "Single envelope schema with all three sub-keys optional -- single tier reuses the same schema with only dependency_tracer populated"
  - "ExplorationLivePanel flips leftover pending rows to failed on __exit__ so a crashed subagent is visible to the user"
  - "_parse_envelope coerces with try/except per entry rather than schema validation -- malformed entries are skipped, never raise"
metrics:
  duration: 22min
  completed: 2026-04-06
---

# Phase 2 Plan 4: Pre-Scan Orchestrator + Wiring Summary

Wires Plans 02-01/02/03 together: `daydream.exploration_runner.pre_scan`
counts diff files, picks a tier (skip/single/parallel), launches the right
specialist subagents via `Backend.execute(agents=...)`, parses the combined
envelope into an `ExplorationContext`, and merges with the static
tree-sitter file map. Both `run()` and `run_trust()` now populate
`RunConfig.exploration_context` before their first phase call. The four
review phase functions accept `exploration_context=` and prepend
`to_prompt_section()` to their prompts. CodexBackend explicitly refuses
`agents=` with `NotImplementedError`. UX: a per-subagent
`ExplorationLivePanel` (D-15) renders one row per active specialist
during exploration.

## Tasks Completed

| # | Name | Commit |
|---|------|--------|
| 1 | Build daydream/exploration_runner.py with tier dispatch | 4b83b7a |
| 2 | Wire pre_scan into runner.py + UX panel + Codex guard | 02c42fe |

## Key Changes

- `daydream/exploration_runner.py` (new, ~290 LOC): `count_changed_files`,
  `select_tier`, `EXPLORATION_ENVELOPE_SCHEMA`, `_build_lead_prompt`,
  `_parse_envelope`, `pre_scan` async orchestrator. `pre_scan` always merges
  the static `detect_affected_files` result with the subagent envelope so
  EXPL-01/02 hold even on a subagent failure.
- `daydream/runner.py`: `RunConfig` gains `exploration_context` and
  `exploration_depth=1`. Both `run()` (when `start_at == "review"`) and
  `run_trust()` resolve a tier and either print a dim "skipping" notice or
  enter an `ExplorationLivePanel` and call
  `safe_explore(pre_scan, backend, target_dir, diff, depth, live_panel=...)`.
  All four review phase calls receive `exploration_context=config.exploration_context`.
- `daydream/phases.py`: `phase_review`, `phase_understand_intent`,
  `phase_alternative_review`, `phase_generate_plan` accept a keyword-only
  `exploration_context: ExplorationContext | None = None` and prepend
  `exploration_context.to_prompt_section()` to their prompt when set.
- `daydream/ui.py`: new `ExplorationLivePanel` context manager built on
  Rich `Live` with one row per specialist, throbber/done/failed states,
  and `__exit__` that flips leftover `pending` rows to `failed`.
- `daydream/backends/codex.py`: `execute()` raises `NotImplementedError`
  when `agents is not None`, with a clear "use --backend claude" message.

## Tests

- `tests/test_exploration_runner.py`: tier dispatch (`skip`/`single`/`parallel`),
  envelope parse, missing-key tolerance, count/tier helpers. 11 tests.
- `tests/test_runner.py` (new): `RunConfig.exploration_depth` default and
  override; `exploration_context` defaults to None.
- `tests/test_ui.py` (new): `ExplorationLivePanel` row keys, single-tier,
  pending-to-failed-on-exit.
- `tests/test_integration.py`: `test_run_populates_exploration_context`
  monkeypatches `_git_diff` and the four phase functions, asserts
  `config.exploration_context` is populated and forwarded to `phase_review`.
  `test_codex_backend_raises_on_agents` exercises the new guard.
- `tests/test_backend_codex.py`: `test_execute_ignores_agents` replaced by
  `test_execute_raises_on_agents` to match the new contract.

Full suite: **182 passed**, 1 warning. `make check` green.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated stale Codex test**
- **Found during:** Task 2 (`make check`)
- **Issue:** `tests/test_backend_codex.py::test_execute_ignores_agents`
  asserted Codex silently ignores `agents=`. Plan 02-04 explicitly inverts
  that contract.
- **Fix:** Renamed to `test_execute_raises_on_agents` and rewrote to assert
  `NotImplementedError` per the new guard.
- **Files modified:** `tests/test_backend_codex.py`
- **Commit:** 02c42fe

### Minor Plan Reinterpretations

- The plan asked to "extend the existing MockBackend in tests/test_phases.py".
  The existing `MockBackend` in `tests/test_integration.py` already accepts
  `agents=...` and ignores it. To avoid touching shared mocks, a dedicated
  `_AgentsRecordingMockBackend` lives in `tests/test_exploration_runner.py`
  and is imported by the integration test.
- Plan 02-04 referenced `tests/test_runner.py` and `tests/test_ui.py` as
  files to modify, but neither existed. They were created from scratch with
  the required tests.
- The plan asked for `_run_exploration` extracted as a helper. Instead the
  exploration block is inlined in both `run()` and `run_trust()` so the
  acceptance grep counts (`safe_explore` >= 2, `pre_scan` >= 2,
  `ExplorationLivePanel` >= 2) hold without trickery and the call site is
  obvious at each phase boundary.

## Verification

| Requirement | Where Verified |
|-------------|----------------|
| AGNT-01 (parallel tier launches three named specialists) | `test_parallel_tier_launches_three_agents` |
| EXPL-01/02 (static file map flows into exploration context) | `pre_scan` always merges `detect_affected_files` with envelope |
| EXPL-03/04 (pattern-scanner reads guideline files) | `PATTERN_SCANNER_SYSTEM_PROMPT` covered by existing prompt test |
| Both flows populate `RunConfig.exploration_context` | `test_run_populates_exploration_context` + run_trust inline block |
| Codex refuses exploration | `test_execute_raises_on_agents`, `test_codex_backend_raises_on_agents` |
| D-15 per-subagent live panel | `test_exploration_live_panel_*` (3 tests) |
| `make check` green | 182 passed |

## Self-Check: PASSED

- daydream/exploration_runner.py: FOUND
- tests/test_runner.py: FOUND
- tests/test_ui.py: FOUND
- 4b83b7a: FOUND
- 02c42fe: FOUND
- mypy + ruff clean (full daydream package)
- pytest 182 passed
