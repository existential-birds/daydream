---
phase: 02-recorder-core-event-enrichment-mapping
plan: 05
subsystem: agent
tags: [python, agent, integration, atif, trajectory, recorder]

# Dependency graph
requires:
  - phase: 02-recorder-core-event-enrichment-mapping
    plan: 01
    provides: TrajectoryRecorder + Invocation public surface, get_current_recorder, DaydreamPhase / DaydreamRunFlow enums, Invocation._dispatch already speaks EVNT-02 attribute names
  - phase: 02-recorder-core-event-enrichment-mapping
    plan: 02
    provides: MetricsEvent dataclass and inclusion in AgentEvent union; timestamp fields on every event
  - phase: 02-recorder-core-event-enrichment-mapping
    plan: 03
    provides: Claude backend emits MetricsEvent per AssistantMessage (verified by tests/test_backend_claude_metrics.py)
  - phase: 02-recorder-core-event-enrichment-mapping
    plan: 04
    provides: Codex backend emits MetricsEvent at turn.completed (verified by tests/test_backend_codex_metrics.py)
provides:
  - daydream.agent.run_agent now opens an Invocation lifecycle around the existing event loop when get_current_recorder() returns a recorder; user prompt becomes a user Step (MAP-01); all 6 existing event types plus the new MetricsEvent branch route into inv.observe(event) (MAP-02..06)
  - Required keyword-only phase: DaydreamPhase argument on run_agent (D-05) — transitional sentinel default raises TypeError on un-updated call sites for cleaner failure mode mid-wave
  - tests/test_agent_recorder_integration.py — 12 schema-validity + behavior-predicate tests against a deterministic MockBackend
  - End-to-end MAP-07 verification: FinalMetrics totals equal sum of per-step Metrics across two run_agent calls inside the same recorder
affects:
  - phase 02-06 (call-site updates): MUST land next to restore the full 343-test suite to green; Plan 06 updates every run_agent call site to pass phase=DaydreamPhase.X (16 in phases.py + 1 in exploration_runner.py + 4 TrajectoryRecorder constructions in run flows)
  - phase 02-07 (conftest autouse fixture): the file-local _reset_recorder fixture in test_agent_recorder_integration.py is replaced by the suite-wide autouse fixture in tests/conftest.py
  - phase 04 (cutover + redaction): Phase 4 wires the Redactor rule list; the recorder boundary is already in place, no change to agent.py needed
  - phase 07 (Plan 07 sentinel re-tightening): replace _PHASE_REQUIRED sentinel with no default at all (D-05 strict enforcement)

# Tech tracking
tech-stack:
  added: [contextlib.nullcontext]
  patterns:
    - "Invocation lifecycle wrap pattern: `recorder.invocation(phase=phase) if recorder is not None else nullcontext(None)` keeps the with-shape uniform whether or not a recorder is active (CORE-09 no-op)"
    - "Transitional sentinel default for required keyword args: `_PHASE_REQUIRED: Any = object()` with explicit `if x is _SENTINEL: raise TypeError(...)` in the function body. Lets a contract change land in one wave and call-site updates land in the next without touching the suite mid-wave with mysterious errors"
    - "Schema-validity + behavior-predicate test pattern (D-18) extended to integration tests: every test that produces a trajectory asserts daydream.atif.validate(traj) is True PLUS one or two specific behavioral predicates"
    - "MockBackend dataclass implements Backend protocol structurally (no inheritance); execute() returns an async generator that replays a canned event list deterministically"
    - "isinstance order matters when subclasses share a check: MetricsEvent branch inserted BEFORE CostEvent (the planner-correct order); both are independent dataclasses so the order is purely a readability concern"

key-files:
  created:
    - tests/test_agent_recorder_integration.py — 12 tests against MockBackend covering MAP-01..09, D-05, D-15, CORE-09, Pitfall 4
  modified:
    - daydream/agent.py — added imports for nullcontext, MetricsEvent, DaydreamPhase, get_current_recorder; added `*, phase: DaydreamPhase = _PHASE_REQUIRED` to run_agent signature; wrapped event loop with `async with invocation_cm as inv:`; inserted `inv.observe(event)` at end of each isinstance branch; new isinstance(event, MetricsEvent) branch before CostEvent

key-decisions:
  - "Transitional sentinel pattern WAS used as planned. Reasoning: the alternative (committing only the test file in this plan, then signature change + GREEN in Plan 06) collapses Plans 05 and 06 into one wave with mixed concerns. The sentinel keeps the contract change isolated to this plan AND keeps the failure mode explicit (TypeError with a clear message that names the missing argument and points at Plans 06/07). Net cost: ~6 lines of code (the sentinel object + the runtime check) that Plan 07 deletes. Worth it for the clean wave boundary and readable diagnostic."
  - "MockBackend dataclass exposes `events` as a public field (instead of `_events` private). Reason: tests benefit from being able to inspect/extend the canned event list — and the dataclass form keeps construction trivially compatible with the Backend Protocol's structural typing. No semantic difference; ergonomics-only choice."
  - "MockBackend.execute signature mirrors Backend.execute exactly (cwd, prompt, output_schema, continuation, agents, max_turns) so any future positional-args change in the protocol surfaces as a test failure. Currently every kwarg is unused by MockBackend itself."
  - "Cleanup-of-UI-state moved INSIDE the `async with invocation_cm` block (the `if not use_callback:` agent_renderer.finish() / tool_registry.finish_all() / console.print() three-line block at the end of the for-loop). Originally this lived AFTER the for-loop and OUTSIDE the inner try/except. Moving it inside the async-with keeps the recorder finalize as the very last action before the outer finally — important so any exception during cleanup degrades cleanly via the try/except wrapping. The semantic ordering of UI cleanup before recorder finish is preserved."
  - "The existing `try/except Exception as exc` around the event-loop dispatch (originally lines 333-441) STAYS at the same level — the `async with invocation_cm` lives INSIDE this try/except, so disk-write failures during recorder.__aexit__ go through TrajectoryRecorder's own catch-and-degrade (D-11) rather than this outer handler."

patterns-established:
  - "agent.py + trajectory.py recorder boundary: agent.py only ever calls `inv.observe(event)` and `inv.observe_user_step(prompt)` — never touches Step/ToolCall/Trajectory directly. D-19 module-bloat ban enforced by grep at acceptance time."
  - "Plan-pair sequencing pattern (single contract change spanning two waves): Plan A introduces the contract + transitional sentinel; Plan B updates all call sites; Plan A's tests validate in isolation by using the new contract directly. The sentinel keeps the suite recoverable with a clear diagnostic in the gap."
  - "Test file structure for run_agent integration: file-local _reset_recorder autouse fixture (until Plan 07 lifts to conftest), MockBackend dataclass replaying canned events, _make_recorder helper, _run_with_recorder helper for single-invocation cases plus inline `async with recorder` for multi-invocation tests."

requirements-completed: [MAP-01, MAP-02, MAP-03, MAP-04, MAP-05, MAP-06, MAP-07]

# Metrics
duration: 30min
completed: 2026-04-26
---

# Phase 02 Plan 05: agent.py Recorder Integration Summary

**run_agent now wraps its event loop with an Invocation lifecycle when a TrajectoryRecorder is active; the user prompt becomes a user Step and every AgentEvent (including the new MetricsEvent) routes into inv.observe; required keyword-only phase argument added with transitional sentinel default; D-19 module-bloat ban honored.**

## Sequencing Status

**Suite is intentionally red after this plan; Plan 02-06 restores green.**

After this plan, `uv run pytest` reports `67 failed, 356 passed`. All 67 failures are `TypeError: run_agent() requires keyword-only argument 'phase'` from call sites in `daydream/phases.py`, `daydream/runner.py`, `daydream/exploration_runner.py`, `daydream/deep/orchestrator.py`, and a few test fixture sites that wrap them. This is exactly the contained red state predicted by the plan: Plan 06 (next wave, depends_on this plan) updates every call site to pass `phase=DaydreamPhase.X` and restores the full suite to green. Plans 05 and 06 form one atomic Wave-4 contract change with no checkpoint between them.

The transitional sentinel default (`_PHASE_REQUIRED`) raises a clear TypeError with a message that names the missing argument and references Plans 06/07. There are no silent failures.

## Performance

- **Duration:** approx. 30 min
- **Completed:** 2026-04-26
- **Tasks:** 1 (planned), committed as 2 commits (RED test + GREEN implementation)
- **Files modified:** 2 (1 new test file + 1 source edit)

## Accomplishments

- `daydream/agent.py:run_agent` accepts a keyword-only `phase: DaydreamPhase` argument with the documented transitional sentinel default. When called without `phase`, raises `TypeError("run_agent() requires keyword-only argument 'phase' (use DaydreamPhase.X). Plan 06 updates the call sites; Plan 07 will tighten this back to a hard signature requirement.")`.
- The event loop is wrapped in `async with invocation_cm as inv:` where `invocation_cm = recorder.invocation(phase=phase)` if a recorder is active, else `nullcontext(None)`. The for-loop and existing UI dispatch (`_log_debug`, `agent_renderer.append`, `tool_registry.create`, etc.) are preserved verbatim.
- Each isinstance branch ends with `if inv is not None: inv.observe(event)` — exactly 7 such calls (one per event type: TextEvent, ThinkingEvent, ToolStartEvent, ToolResultEvent, MetricsEvent, CostEvent, ResultEvent). MetricsEvent is a NEW branch inserted before CostEvent.
- `inv.observe_user_step(prompt=prompt)` is called once at invocation entry, just inside the `async with`.
- `tests/test_agent_recorder_integration.py` has 12 tests covering: user prompt → user Step (MAP-01 / Pitfall 4), TextEvent → agent Step (MAP-02), tool call/observation same-step pairing (CORE-06 / MAP-04 / MAP-05 / Pitfall 3), MetricsEvent → Step.metrics with cached subset semantics (MAP-06 / D-15), FinalMetrics totals = sum of per-step (MAP-07), no-recorder no-op (CORE-09), per-Step extra labels (MAP-08 / MAP-09), per-call phase + per-recorder run_flow, ThinkingEvent → reasoning_content (MAP-03), CostEvent does not break recording, signature is keyword-only, missing phase raises TypeError.
- All 12 new integration tests pass (`uv run pytest tests/test_agent_recorder_integration.py -v`).
- All 49 Phase 2 tests pass together (`tests/test_agent_recorder_integration.py + test_trajectory.py + test_backends_events.py + test_backend_claude_metrics.py + test_backend_codex_metrics.py`).
- Ruff and mypy clean on `daydream/agent.py` and the new test file.

## Task Commits

1. **Task 1 RED: failing tests for run_agent + TrajectoryRecorder integration** — `1f7197a` (test)
2. **Task 1 GREEN: wire TrajectoryRecorder into run_agent event loop** — `2a3f1cc` (feat)

Two commits because the plan's task is `tdd="true"` and the RED/GREEN cycle is required for D-18 compliance.

## Files Created/Modified

- `tests/test_agent_recorder_integration.py` (NEW, 378 lines) — 12 schema-validity + behavior-predicate tests; file-local `_reset_recorder` autouse fixture (Plan 07 lifts to conftest); `MockBackend` dataclass replaying canned events; `_make_recorder` and `_run_with_recorder` helpers.
- `daydream/agent.py` (MODIFIED) — added 4 imports (`contextlib.nullcontext`, `daydream.backends.MetricsEvent`, `daydream.trajectory.{DaydreamPhase, get_current_recorder}`); added `_PHASE_REQUIRED` sentinel constant; added `*, phase: DaydreamPhase = _PHASE_REQUIRED` to `run_agent` signature; wrapped event loop with `async with invocation_cm as inv:`; inserted `inv.observe(event)` at end of each isinstance branch; new `isinstance(event, MetricsEvent)` branch inserted before the CostEvent branch.

## Was the Transitional Sentinel Pattern Needed?

**Yes — used as planned, and it earned its place.**

The plan's `<action>` block in Plan 02-05 documented two options:
1. Tighten the signature to `*, phase: DaydreamPhase` with no default — full-suite goes RED until every call site is updated by Plan 06.
2. Add a transitional sentinel — full-suite still goes RED but with a clear, predictable diagnostic at every un-updated call site.

I chose option 2. Reasoning:

- **Failure mode quality.** Without the sentinel, mypy + Python's positional/keyword machinery would report the error, but at runtime a missing keyword-only argument with no default raises `TypeError: run_agent() missing 1 required keyword-only argument: 'phase'`. Functional, but doesn't tell future-me (or a recovery-from-incomplete-wave executor) what's going on. The sentinel's TypeError message names Plans 06/07 and points at the next concrete remediation step.
- **Cost is trivial.** ~6 lines: the sentinel constant, the type-ignore comment on the default, and the `if phase is _PHASE_REQUIRED: raise TypeError(...)` block. Plan 07 deletes them all in one micro-edit.
- **Plan-pair atomicity preserved.** Plan 05's tests validate this plan's contract change in isolation (12/12 pass). Plan 06's job is purely call-site updates — no concurrent debugging of Plan 05 internals.

The pattern is documented in `patterns-established` above for re-use. Future contract changes that span waves should consider it.

## MockBackend Shape (for Plan 07 / future tests to build on)

```python
@dataclass
class MockBackend:
    events: list[AgentEvent]

    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
    ) -> AsyncIterator[AgentEvent]:
        events = self.events  # capture for closure
        async def _gen() -> AsyncIterator[AgentEvent]:
            for event in events:
                yield event
        return _gen()

    async def cancel(self) -> None:
        return None

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"
```

Key features:
- **Dataclass, no inheritance.** Backend is a Protocol; structural typing means MockBackend doesn't import Backend at all (only uses it for type-hint inspection in tests).
- **execute() returns an async generator factory.** Backend.execute() is `def` (not `async def`); it returns an `AsyncIterator[AgentEvent]`. The closure captures `self.events` so test mutation between invocations is safe.
- **Events list is public (no leading underscore).** Tests benefit from being able to extend or inspect the canned list.
- **Signature mirrors Backend.execute exactly.** Any future protocol change surfaces here as a test break.
- **format_skill_invocation returns Claude-style "/skill" form.** Codex tests can override in a subclass if needed.

## UI Boundary Concerns (panel registry, agent renderer)

The existing `agent_renderer = AgentTextRenderer(console)` and `tool_registry = LiveToolPanelRegistry(console, _state.quiet_mode)` are constructed at the start of the function (when `not use_callback`) and persist across the event loop. Two interactions worth noting:

1. **Cleanup placement.** The original `if not use_callback: agent_renderer.finish() / tool_registry.finish_all() / console.print()` cleanup block at lines 432-435 had to move INSIDE the `async with invocation_cm:` block. Moving it outside would have it run AFTER `inv.finish()` (which flushes the trajectory) — that's wrong because UI cleanup is a higher-priority operation than recorder finalization (a stalled progress bar visible after the agent returns is worse than a missing trajectory). The current ordering is: UI cleanup → exit `async with` (trajectory flush + clear ContextVar) → exit outer try/finally (remove backend from `_state.current_backends`).

2. **Renderer/registry are recorder-agnostic.** Neither `AgentTextRenderer` nor `LiveToolPanelRegistry` knows about the recorder; they only read events to drive Rich panels. This isolation is good — D-19 is about ATIF model construction, but the spirit applies more broadly: UI logic in `daydream/ui.py` (and consumed by `daydream/agent.py`) stays orthogonal to the trajectory pipeline. If Plan 07 or Phase 4 ever needs UI to react to recorder state, that's a new integration point and should go through a new public method on TrajectoryRecorder, not through importing the recorder into the UI module.

## Decisions Made

- **Transitional sentinel pattern used.** See dedicated section above.
- **MockBackend dataclass with public `events` field.** Ergonomics > strict private/public discipline for a test helper.
- **UI cleanup moved INSIDE the async with block.** Preserves UI-cleanup-before-recorder-flush ordering.
- **MetricsEvent branch inserted BEFORE CostEvent branch.** Both are independent dataclasses (no subclass relationship), so the isinstance order is a readability choice. The plan's "Phase 2 insertion shape" code block puts MetricsEvent before CostEvent; followed that.

## Deviations from Plan

None — plan executed exactly as written.

The `<action>` block specified the transitional sentinel pattern and gave the exact code shape; the `<acceptance_criteria>` block listed precise grep counts and test invocations; the `<verification>` block was satisfied as written. No Rule 1/2/3 fixes needed.

The only minor adjustment was that the cleanup block moved INSIDE the `async with` (the plan's code sample showed it inside as well, but the textual narrative above the sample was ambiguous). This is documented in the Decisions section.

## Issues Encountered

- **`uv run` requires sandbox disabled.** The `~/.cache/uv` directory is not writable inside the agent sandbox, so all `uv run pytest`, `uv run ruff`, `uv run mypy` invocations were run with `dangerouslyDisableSandbox: true`. Same constraint applies to Plans 02-01 through 02-04; non-blocking.

## Threat Flags

None. Plan 02-05's edits are within the existing trust boundaries enumerated in the plan's `<threat_model>` (process → trajectory file via TrajectoryRecorder.__aexit__; process → ContextVar via get_current_recorder). No new endpoints, auth paths, file access patterns, or schema changes at trust boundaries.

The threat register's `T-02-12` (prompt content captured raw) is honored: `inv.observe_user_step(prompt=prompt)` passes the prompt through unredacted because the recorder's flush-time Redactor is the redaction site (Phase 4 fills in the rule list). The wiring is exercised by Phase 2 — Phase 4 changes only `Redactor.redact_step`'s body without touching this call site.

## Next Phase Readiness

- **Plan 02-06 (Wave 5):** Updates the 16 `run_agent` call sites in `daydream/phases.py`, the 1 in `daydream/exploration_runner.py`, and adds 4 `TrajectoryRecorder(...)` constructions (3 in `daydream/runner.py` + 1 in `daydream/deep/orchestrator.py`). Also adds `RunConfig.trajectory_path: Path | None = None`. After Plan 06: full 343-test suite returns to green.
- **Plan 02-07 (Wave 6):** Lifts the file-local `_reset_recorder` autouse fixture in `test_trajectory.py` and `test_agent_recorder_integration.py` into a suite-wide autouse fixture in `tests/conftest.py` (CORE-10 / D-17). Also re-tightens the `_PHASE_REQUIRED` sentinel to `phase: DaydreamPhase` with no default and removes the runtime check (Plan 07 task).
- **Phase 4 (cutover):** No change needed in `agent.py` — the recorder boundary is in place; Phase 4 fills in `Redactor.redact_step` internals.

## Self-Check: PASSED

All claims verified:

- [x] `daydream/agent.py` modified — `git log -1 --format=%H -- daydream/agent.py` returns `2a3f1cc`
- [x] `tests/test_agent_recorder_integration.py` exists (378 lines) — `wc -l tests/test_agent_recorder_integration.py` returns 378
- [x] Commit `1f7197a` exists (Task 1 RED test)
- [x] Commit `2a3f1cc` exists (Task 1 GREEN implementation)
- [x] All 12 new integration tests pass: `uv run pytest tests/test_agent_recorder_integration.py -v` exits 0
- [x] All 49 Phase 2 tests pass together: `uv run pytest tests/test_agent_recorder_integration.py tests/test_trajectory.py tests/test_backends_events.py tests/test_backend_claude_metrics.py tests/test_backend_codex_metrics.py -v` exits 0
- [x] `uv run mypy daydream/agent.py` exits 0
- [x] `uv run ruff check daydream/agent.py tests/test_agent_recorder_integration.py` exits 0
- [x] `grep -c "phase: DaydreamPhase" daydream/agent.py` returns 1 (>= 1)
- [x] `grep -c "from daydream.trajectory import" daydream/agent.py` returns 1 (exactly 1)
- [x] `grep -c "get_current_recorder" daydream/agent.py` returns 2 (>= 1; 1 import, 1 call)
- [x] `grep -c "inv.observe" daydream/agent.py` returns 10 (>= 7; 7 isinstance branches + 1 observe_user_step + 2 in comments)
- [x] `grep -c "isinstance(event, MetricsEvent)" daydream/agent.py` returns 1 (exactly 1)
- [x] D-19 module-bloat ban: `grep -nE "^[^#]*\b(Step|ToolCall|Trajectory|Observation|ObservationResult|Metrics|FinalMetrics)\(" daydream/agent.py` returns 0 matches
- [x] `grep -nE "from daydream.atif" daydream/agent.py` returns 0 matches
- [x] EVNT-02 names in trajectory.py: `grep -nE "event\.prompt_tokens|event\.completion_tokens" daydream/trajectory.py` returns 4 (>= 2)
- [x] Full suite: `67 failed, 356 passed` — exactly the contained red state predicted; all 67 failures are `TypeError: run_agent() requires keyword-only argument 'phase'` (Plan 06 fixes)

---
*Phase: 02-recorder-core-event-enrichment-mapping*
*Completed: 2026-04-26*
