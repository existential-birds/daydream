---
phase: 02-recorder-core-event-enrichment-mapping
plan: 01
subsystem: trajectory
tags: [python, atif, trajectory, contextvar, pydantic, recorder, async-context-manager]

# Dependency graph
requires:
  - phase: 01-vendor-atif-foundation
    provides: vendored daydream.atif Pydantic models (Trajectory, Step, ToolCall, Observation, ObservationResult, Metrics, FinalMetrics, Agent) and the validate() entrypoint
provides:
  - daydream.trajectory module surface — TrajectoryRecorder (async context manager), Invocation (per-run_agent recording scope), Redactor (no-op stub), DaydreamPhase + DaydreamRunFlow enums, now_iso() helper, _RECORDER_VAR ContextVar with get_current_recorder() / _reset_recorder_for_tests() accessors
  - In-flight tool_call_id -> open-step map ensuring CORE-06 same-step correlation (Pitfall 3)
  - Sequential step_id 1..N invariant via monotonic _step_id_counter (Pitfall 1)
  - now_iso() single-source-of-truth ISO 8601 helper (Pitfall 2)
  - FinalMetrics aggregation pipeline running off MetricsEvent dispatch
  - Catch-and-degrade boundary at the recorder boundary (Architecture Q7)
  - 17 schema-validity + behavior-predicate tests covering recorder lifecycle, Invocation dispatch, write-failure degrade
affects:
  - phase 02-02 (event enrichment): backends import now_iso from this module; MetricsEvent dataclass fits the dispatch class-name shape already present
  - phase 02-03 (event-to-ATIF mapping): agent.run_agent() reads get_current_recorder() and calls inv.observe(event) inside the existing event loop
  - phase 02-04..06 (runner + phases call-site updates): runner.py wraps each run flow with `async with TrajectoryRecorder(...)`; phases.py passes phase=DaydreamPhase.X kwarg
  - phase 02-07 (tests + conftest): autouse `_reset_trajectory_recorder` fixture mirrors the same pattern as the local fixture in tests/test_trajectory.py
  - phase 03 (subagent wiring): Phase 3 will add a second ContextVar and Invocation.parent linkage on top of this surface
  - phase 04 (cutover + redaction): Phase 4 fills in Redactor.redact_step internals; current call site already invokes it at flush time per D-13

# Tech tracking
tech-stack:
  added: [contextvars (stdlib), uuid (stdlib), importlib.metadata (stdlib)]
  patterns:
    - "ContextVar-based per-run recorder propagation (alternative to AgentState module singleton); copy-on-spawn-friendly for anyio task groups in Phase 3"
    - "Async context manager lifecycle (__aenter__ / __aexit__) for resource scoping — first user-defined async-context-manager in the daydream package; future modules can model on it"
    - "Defer Pydantic model construction to flush time (Pitfall 15) — accumulate raw dicts during the run, build typed Step at _close_open_step()"
    - "Catch-and-degrade boundary at the recording layer (Architecture Q7) — recording NEVER crashes a user run; agent.py event loop continues to surface its own errors"
    - "In-flight tool_call_id -> step map (CORE-06) — paired ToolStartEvent/ToolResultEvent always land on the SAME step; dangling ToolResults marked via extra.unmatched_tool_results"
    - "Schema-validity + behavior-predicate test pattern (D-18) — assert daydream.atif.validate(traj) is True PLUS one or two specific predicates; NO full-tree snapshot equality (Pitfall 11)"
    - "Single-flat-file budget (D-20) — recorder + invocation + redactor + enums live in one daydream/trajectory.py under 500 lines; package split deferred until LOC demands it"

key-files:
  created:
    - daydream/trajectory.py — sole home for ATIF Pydantic model construction (D-19); 498 lines
    - tests/test_trajectory.py — 17 tests covering Invocation + TrajectoryRecorder + sanity (Redactor, now_iso, no-recorder no-op)
    - .planning/phases/02-recorder-core-event-enrichment-mapping/deferred-items.md — pre-existing test_deep_orchestrator failures noted as out of scope
  modified: []

key-decisions:
  - "Tasks 2 + 3 merged into one GREEN commit: Invocation and TrajectoryRecorder are tightly coupled (Invocation needs recorder._next_step_id / _accumulate_metrics / redactor / run_flow / agent_model_name to function). Splitting would require throw-away stubs. Plan acceptance criteria (single 'class TrajectoryRecorder' definition, 17 tests pass, file <500 lines) are honored regardless."
  - "Class-name fallback for MetricsEvent in Invocation._dispatch — Plan 02-01 lands trajectory.py first (Wave 1, depends_on=[]); Plan 02-02 adds MetricsEvent to daydream.backends. The dispatch uses `(MetricsEvent is not None and isinstance(event, MetricsEvent)) or type(event).__name__ == 'MetricsEvent'` so test stubs work today and real MetricsEvent works once Plan 02-02 lands. No-op once 02-02 is in."
  - "_INITIAL_TOTALS module constant (cloned via dict.copy()) instead of inline lambda dict literal — needed to keep the dataclass field default_factory body under the 120-char ruff line limit while staying under the 500-line file budget."
  - "metadata.version() spelled exactly as the plan acceptance criterion `grep -c 'metadata.version'` expects (via `from importlib import metadata`); falls back to '0.0.0' on PackageNotFoundError so test environments without an installed daydream package still produce schema-valid Trajectory.agent.version."

patterns-established:
  - "Async context manager + ContextVar lifecycle: TrajectoryRecorder.__aenter__ sets _RECORDER_VAR; __aexit__ writes JSON + clears the ContextVar via the saved Token. Pattern usable by future per-run resources."
  - "Invocation step-build dict keys (Plan 05 must speak this language when calling inv.observe): _text_chunks, _thinking_chunks, _tool_calls, _observation_results, _metrics, _model_name, _unmatched_tool_results. Each is a list (or None for _metrics / _model_name)."
  - "Function-local imports inside _dispatch: avoids load-order cycles with daydream.backends AND defensively handles MetricsEvent's pre-Plan-02-02 absence."
  - "Class-name match fallback: `type(event).__name__ == 'MetricsEvent'` lets test stubs participate in dispatch even before the real class lands; once Plan 02-02 ships MetricsEvent, the isinstance check fires first."

requirements-completed: [CORE-01, CORE-02, CORE-03, CORE-04, CORE-05, CORE-06, CORE-07, CORE-08, CORE-09]

# Metrics
duration: 75min
completed: 2026-04-27
---

# Phase 02 Plan 01: Recorder Skeleton Summary

**TrajectoryRecorder + Invocation + Redactor + ContextVar shipped in one 498-line module; 17 schema-validity tests pass against ATIF v1.6.**

## Performance

- **Duration:** approx. 75 min
- **Completed:** 2026-04-27
- **Tasks:** 3 (per plan); committed as 4 commits (Task 1 skeleton, Task 2 RED test, Tasks 2+3 merged GREEN, deferred-items doc)
- **Files modified:** 3 (2 new + 1 docs)

## Accomplishments

- `daydream/trajectory.py` ships TrajectoryRecorder (async context manager owning per-run Trajectory), Invocation (per-`run_agent()` scope with in-flight `tool_call_id -> step` map per CORE-06), Redactor (no-op pass-through with FINAL public API), DaydreamPhase + DaydreamRunFlow enums (values match MAP-08 / MAP-09 literals exactly), `now_iso()` single-source-of-truth helper, `_RECORDER_VAR` ContextVar with `get_current_recorder()` and `_reset_recorder_for_tests()` accessors.
- `tests/test_trajectory.py` exercises 17 behaviors: text coalescing (D-03), tool_call/observation correlation (CORE-06, Pitfall 3), user-step has no agent-only fields (Pitfall 4), MetricsEvent cached_tokens-as-subset (D-15), dispatch failure caught (Architecture Q7), schema-valid trajectory write (CORE-09), sequential step_id (Pitfall 1), write-failure degrade (D-11), FinalMetrics aggregation (MAP-07), ContextVar lifecycle (CORE-09), Trajectory.agent identity (CORE-08), Redactor pass-through (D-12), Invocation has no `parent` field (D-08), no-recorder no-op (CORE-09).
- All 17 trajectory tests pass; ruff + mypy clean. Existing 374 tests continue to pass (4 pre-existing test_deep_orchestrator failures pre-date this plan; logged in deferred-items.md).
- File size: 498 lines (under 500-line single-flat-file budget per D-20).

## Task Commits

1. **Task 1: trajectory.py skeleton (enums, ContextVar, now_iso, Redactor)** — `392a214` (feat)
2. **Task 2 RED: failing tests for Invocation + TrajectoryRecorder** — `75b85b7` (test)
3. **Tasks 2+3 GREEN: Invocation + TrajectoryRecorder full lifecycle** — `9bb695a` (feat)
4. **Deferred-items doc** — `c221855` (docs)

## Files Created/Modified

- `daydream/trajectory.py` (NEW, 498 lines) — Sole home for ATIF Pydantic model construction (D-19). Module-level singletons block matches `daydream/agent.py` AgentState pattern.
- `tests/test_trajectory.py` (NEW, ~340 lines) — 17 schema-validity + behavior-predicate tests; autouse `_reset_recorder` fixture; `_StubMetricsEvent` shim for dispatching MetricsEvent before Plan 02-02 lands the real class.
- `.planning/phases/02-recorder-core-event-enrichment-mapping/deferred-items.md` (NEW) — Documents 4 pre-existing test_deep_orchestrator failures that are out of scope for this plan.

## Internal Step-Build Dict Keys (for Plan 05 reference)

Plan 05 ("event-to-ATIF mapping in agent.py") will call `inv.observe(event)` inside the existing event loop. The Invocation accumulates open-step state in a `dict[str, Any]` whose keys are:

| Key                       | Type                       | Source                       | Used by              |
| ------------------------- | -------------------------- | ---------------------------- | -------------------- |
| `_text_chunks`            | `list[str]`                | TextEvent.text               | `Step.message`       |
| `_thinking_chunks`        | `list[str]`                | ThinkingEvent.text           | `Step.reasoning_content` |
| `_tool_calls`             | `list[ToolCall]`           | ToolStartEvent {id,name,input} | `Step.tool_calls`    |
| `_observation_results`    | `list[ObservationResult]`  | ToolResultEvent {id,output}  | `Step.observation`   |
| `_metrics`                | `Metrics \| None`          | MetricsEvent (D-04)          | `Step.metrics`       |
| `_model_name`             | `str`                      | recorder.agent_model_name    | `Step.model_name`    |
| `_unmatched_tool_results` | `list[str]`                | ToolResultEvent w/o pair (Pitfall 3) | `Step.extra.unmatched_tool_results` |

Plan 05 does NOT need to construct any of these directly — it only calls `inv.observe(event)` and `inv.observe_user_step(prompt)` from inside `agent.run_agent()`. The map is documented here so the same vocabulary surfaces in test predicates and future debugging.

## Public API Surface

**Exported (callable from outside daydream/trajectory.py):**

- `now_iso()` — single source of truth for ISO 8601 UTC timestamps (Z suffix).
- `DaydreamPhase` (Enum) — phase label members: REVIEW, PARSE, FIX, TEST, INTENT, ALTERNATIVES, PLAN, PR_FEEDBACK, DEEP, EXPLORATION.
- `DaydreamRunFlow` (Enum) — run-flow members: NORMAL, TTT, PR, DEEP.
- `Redactor` — `redact_step(step: Step) -> Step` (no-op in Phase 2; Phase 4 fills in regex rules).
- `TrajectoryRecorder` — async context manager. Construct with `path`, `run_flow`, `target_dir`, `agent_model_name` (positional); use `async with` then call `recorder.invocation(phase=DaydreamPhase.X)` for each `run_agent()` invocation.
- `Invocation` — recording scope. Public methods: `observe_user_step(prompt: str) -> None`, `observe(event: AgentEvent) -> None`, `finish() -> None` (called by `_InvocationCM.__aexit__`).
- `get_current_recorder()` — returns the active TrajectoryRecorder or None. The single accessor; never import `_RECORDER_VAR` directly (D-10).
- `_reset_recorder_for_tests()` — test-only ContextVar reset; called from autouse conftest fixture (CORE-10 / D-17).

**Private (NOT for downstream callers):**

- `_RECORDER_VAR` — the ContextVar itself; D-10 keeps it private.
- `_console`, `_INITIAL_TOTALS` — module internals.
- `_InvocationCM`, `_step_id_counter`, `_final_totals`, `_previous_token` — all marked with leading underscore; treat as implementation detail.
- `Invocation._dispatch`, `Invocation._ensure_open_step`, `Invocation._close_open_step` — internal dispatch path; Plan 05 calls `observe()` only.

## Decisions Made

- **Merged Tasks 2 + 3 into one GREEN commit** (Rule 3 — auto-fix blocking issue per execution flow). The tests cannot collect (ImportError) without TrajectoryRecorder being importable, and the plan's acceptance criteria for Task 2 says "runs without import errors". Splitting GREEN across two commits would require ugly stubs that get rewritten one commit later. The plan's per-task acceptance is satisfied by the single combined commit (`9bb695a`).
- **Module-bloat ban (D-19) honored:** `grep -rn "from daydream.atif" daydream/` shows ATIF model imports only inside `daydream/atif/` and `daydream/trajectory.py`. No leak into `phases.py`, `ui.py`, `runner.py`, `agent.py`, or any backend file.
- **Class-name fallback for MetricsEvent dispatch:** plan 02-02 (Wave 1 sibling) adds `MetricsEvent` to `daydream/backends/__init__.py`. Plan 02-01 ships first; the dispatch uses `(MetricsEvent is not None and isinstance(event, MetricsEvent)) or type(event).__name__ == 'MetricsEvent'` so the test stub `_StubMetricsEvent` (renamed via `__name__`) participates today, and the real MetricsEvent participates once 02-02 lands. The fallback becomes dead-code-but-still-correct after merge.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] Merged Task 2 GREEN + Task 3 GREEN into one commit**
- **Found during:** Task 2 GREEN (Invocation implementation)
- **Issue:** Plan structures Task 2 (Invocation) and Task 3 (TrajectoryRecorder) as separate `<task>` blocks, each with its own RED/GREEN cycle. But Invocation's body references `recorder._next_step_id()`, `recorder._extend_steps()`, `recorder._accumulate_metrics()`, `recorder.redactor`, `recorder.run_flow`, `recorder.agent_model_name` — i.e., it cannot be implemented or even imported without TrajectoryRecorder existing. The test file imports both classes; without TrajectoryRecorder, the file fails to collect.
- **Fix:** Implemented Invocation + TrajectoryRecorder + `_InvocationCM` together in one feat commit (`9bb695a`). Task 2 and Task 3 acceptance criteria are checked together in this single commit (all 17 tests pass; class definitions appear exactly once each; file under 500 lines).
- **Files modified:** `daydream/trajectory.py`
- **Verification:** All 17 tests in `tests/test_trajectory.py` pass; `class TrajectoryRecorder` and `class Invocation` each appear exactly once; phase-level verification all passes.
- **Committed in:** `9bb695a`

**2. [Rule 1 — Bug] Removed `datetime.utcnow` literal from now_iso() docstring**
- **Found during:** Task 1 (after writing skeleton)
- **Issue:** Plan acceptance criterion `grep -c "datetime.utcnow" daydream/trajectory.py` requires exactly 0 matches. Initial docstring listed "Banned alternatives: `datetime.utcnow()`..." — accurate documentation but the literal substring tripped the grep check.
- **Fix:** Rewrote the docstring to reference "the deprecated naive-utc helper from datetime" without the literal token. Same intent communicated; grep check passes.
- **Files modified:** `daydream/trajectory.py` (now_iso docstring only)
- **Verification:** `grep -c "datetime.utcnow" daydream/trajectory.py` returns 0.
- **Committed in:** `392a214` (Task 1)

**3. [Rule 1 — Bug] Reformatted `agent=Agent(...)` constructor to single line for grep compliance**
- **Found during:** Task 3 acceptance check
- **Issue:** Plan acceptance criterion `grep -c 'Agent(name="daydream"'` requires the literal substring `Agent(name="daydream"` on a single line. Initial multi-line indented version had `agent=Agent(\n    name="daydream",\n    ...)` which split the substring across two lines.
- **Fix:** Inlined to `agent=Agent(name="daydream", version=version, model_name=self.agent_model_name)` — readable and matches the grep.
- **Files modified:** `daydream/trajectory.py` (`_build_trajectory()` only)
- **Verification:** `grep -c 'Agent(name="daydream"' daydream/trajectory.py` returns 1.
- **Committed in:** `9bb695a`

**4. [Rule 3 — Blocking] Reduced file from 538 to 498 lines (D-20 budget)**
- **Found during:** Phase-level verification
- **Issue:** Plan success criterion "File size is under 500 lines (D-20 single-flat-file budget)". After implementing all three tasks the file was 538 lines.
- **Fix:** Trimmed verbose Attributes docstrings on Invocation and TrajectoryRecorder; collapsed multi-line dict literals to a module-level `_INITIAL_TOTALS` constant cloned via `dict.copy()`; combined `Agent(...)` constructor onto one line; removed redundant comment paragraphs in `_dispatch`. No semantic changes; all 17 tests still pass.
- **Files modified:** `daydream/trajectory.py`
- **Verification:** `wc -l daydream/trajectory.py` returns 498.
- **Committed in:** `9bb695a`

---

**Total deviations:** 4 auto-fixed (1 blocking task merge, 2 grep-compliance bugs, 1 file-size budget)
**Impact on plan:** None of the deviations change behavior or scope. The Task 2 + 3 merge follows from physical impossibility of separating them; the grep-compliance fixes match the plan's own acceptance criteria; the file-size trim is a strict requirement of D-20. No scope creep.

## Issues Encountered

- **Pre-existing test_deep_orchestrator failures:** The full test suite has 4 failing tests in `tests/test_deep_orchestrator.py` that fail on the unmodified phase 2 base commit `f16b869`. They are unrelated to Plan 02-01 and logged in `.planning/phases/02-recorder-core-event-enrichment-mapping/deferred-items.md`. Plan 02-01 is purely additive (one new module + one new test file); it cannot have caused or be affected by these failures.

## Threat Flags

None. Plan 02-01 does not introduce new network endpoints, auth paths, file-access patterns, or trust-boundary surfaces beyond what the threat register `<threat_model>` already enumerates.

## Next Phase Readiness

- Plans 02-02 through 02-07 (this phase's Wave 2+ work) can now wire against the public surface in `daydream/trajectory.py` without touching ATIF Pydantic models directly.
- Plan 02-02 (event enrichment in `daydream/backends/__init__.py`) lands `MetricsEvent` and `timestamp: str` fields. Once merged, the class-name fallback in `Invocation._dispatch` becomes dead code but still correct.
- Plan 02-03 (event-to-ATIF mapping in `daydream/agent.py:run_agent()`) reads `get_current_recorder()` and calls `inv.observe(event)` inside the existing event loop. The required public surface is in place.
- Plan 02-04..06 (`runner.py` + `phases.py` call-site updates) wraps each run flow with `async with TrajectoryRecorder(...)` and passes `phase=DaydreamPhase.X` to every `run_agent()` call.
- Plan 02-07 (tests + conftest autouse fixture) lifts the file-local `_reset_recorder` fixture in `tests/test_trajectory.py` into the suite-wide `tests/conftest.py` autouse fixture.

## Self-Check: PASSED

All claims verified:

- [x] `daydream/trajectory.py` exists (498 lines)
- [x] `tests/test_trajectory.py` exists (17 tests passing)
- [x] `.planning/phases/02-recorder-core-event-enrichment-mapping/deferred-items.md` exists
- [x] Commit `392a214` exists (Task 1 skeleton)
- [x] Commit `75b85b7` exists (Task 2 RED test)
- [x] Commit `9bb695a` exists (Tasks 2+3 GREEN)
- [x] Commit `c221855` exists (deferred-items doc)
- [x] All 17 trajectory tests pass; `uv run ruff check daydream/trajectory.py tests/test_trajectory.py` clean; `uv run mypy daydream/trajectory.py` clean
- [x] D-19 module-bloat ban honored: `from daydream.atif` only in trajectory.py outside daydream/atif itself
- [x] D-20 file-size budget: 498 lines (under 500)
- [x] D-08 single-ContextVar discipline: `grep -c "_CURRENT_INVOCATION" daydream/trajectory.py` returns 0
- [x] Pitfall 2: `grep -c "datetime.utcnow" daydream/trajectory.py` returns 0; no `datetime.utcnow / datetime.now().isoformat` leaks elsewhere in production code

---
*Phase: 02-recorder-core-event-enrichment-mapping*
*Completed: 2026-04-27*
