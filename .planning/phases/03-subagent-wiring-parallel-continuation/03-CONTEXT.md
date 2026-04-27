# Phase 3: Subagent Wiring (Parallel + Continuation) - Context

**Gathered:** 2026-04-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Daydream's three parallel-fan-out flows (`phase_fix_parallel`, `daydream/deep/orchestrator.run_deep` via `phase_per_stack_reviews`, `exploration_runner.pre_scan`) emit one sibling trajectory file per parallel invocation, linked from the parent via `ObservationResult.subagent_trajectory_ref`. Continuation flows (`phase_test_and_heal` with continuation tokens) stay in the same trajectory as multi-turn conversations. The sequential phase chain (`phase_review` → `phase_parse_feedback` → `phase_fix` → `phase_test_and_heal`) continues emitting as continuous steps in one root trajectory file (unchanged from Phase 2).

**Out of phase scope (deferred):**
- Redaction of sibling trajectory content (Phase 4: REDA-01..06)
- `_log_debug` removal and CLI surface changes (Phase 4: CUT-* + CLI-*)
- Test hardening for subagent shapes (Phase 5: TEST-07)

</domain>

<decisions>
## Implementation Decisions

### Parallel Fork Surface (SUBA-02, SUBA-03, SUBA-04, SUBA-07)

- **D-01: Explicit fork wrapper — phases call `async with recorder.fork(descriptor):`.** Three call sites change: `phase_fix_parallel` in `daydream/phases.py`, `phase_per_stack_reviews` in `daydream/phases.py`, and `_run_specialist` closure in `daydream/exploration_runner.py`. Each parallel task's closure captures its child recorder. The fork is auditable and matches Phase 2 D-08's "one coherent unit" principle.
- **D-02: `recorder.fork(descriptor)` returns a full `TrajectoryRecorder`.** The child has its own `step_id` counter (starts at 1), its own `steps` list, and its own output path. It inherits `session_id`, `run_flow`, `agent_model_name`, and `target_dir` from the parent. It carries a `parent: TrajectoryRecorder` backref for observation patching. All existing Invocation/observe machinery works unchanged against the child recorder.
- **D-03: Auto-patch parent in child's `__aexit__`.** When a child recorder exits, it writes its sibling trajectory file and calls `self.parent._register_sibling(self.path, self.descriptor)`. The parent accumulates sibling registrations. Zero bookkeeping required in phase code beyond the `async with recorder.fork(descriptor):` wrapper.
- **D-04: Single ContextVar (`_RECORDER_VAR`) — no second ContextVar needed.** Phase 2 D-08 anticipated `_CURRENT_INVOCATION` but the explicit fork approach eliminates the need. Each child `TrajectoryRecorder.__aenter__` sets `_RECORDER_VAR` for its task scope; `run_agent()` picks it up via `get_current_recorder()`. On child `__aexit__`, `_RECORDER_VAR` resets to the parent. One ContextVar is simpler and sufficient.

### Sibling File Naming (SUBA-06)

- **D-05: Semantic descriptor names.** Each fan-out site uses a domain-meaningful descriptor:
  - `phase_fix_parallel`: `fix-0`, `fix-1`, `fix-2`, ... (index matches the feedback item order)
  - `phase_per_stack_reviews` (deep mode): `deep-python`, `deep-typescript`, `deep-react`, ... (stack name from `StackAssignment.stack_name`)
  - `pre_scan` (exploration): `explore-pattern-scanner`, `explore-dependency-tracer`, `explore-test-mapper`
- **D-06: Descriptors are slugified to filesystem-safe characters.** `_safe_descriptor(raw: str) -> str` applies `re.sub(r'[^a-z0-9-]', '-', raw.lower()).strip('-')`. Stack names today are simple lowercase but future or user-defined names could contain special characters.
- **D-07: Session ID in filename is first 8 hex chars of the UUID4.** Full path: `<target>/.daydream/trajectories/<first-8-hex>.<descriptor>.json`. 4 billion possible values — collision-free in practice. Keeps filenames scannable (e.g., `a1b2c3d4.fix-0.json` at 24 chars vs 51 chars with full UUID).

### Continuation Semantics (SUBA-01, SUBA-05)

- **D-08: Each `run_agent()` call creates a new `Invocation`, continuation or not.** Phase 2 D-09 ("one Invocation per `run_agent()` call") holds unchanged. Continuation calls append new user+agent steps to the same trajectory with incrementing `step_id`. The trajectory reads as a multi-turn conversation: prompt → response → prompt → response. No special-casing in the recorder.
- **D-09: No continuation linking marker.** Agent identity is preserved via same `Agent` metadata, `session_id`, and trajectory file. ATIF has no native continuation concept — adding a custom `extra.continuation_of_step_id` is noise. Consumers see a multi-turn conversation; that's sufficient.
- **D-10: Same continuation behavior for both backends.** Whether the continuation token is meaningful (Codex thread resume) or a no-op (Claude), the trajectory shape is identical: new Invocation, append steps. Backend-specific token mechanics are invisible to the recorder.

### Parent Step Shape for Fan-Outs (SUBA-02, SUBA-03, SUBA-04)

- **D-11: Synthetic dispatch step created after the task group completes.** The parent recorder's `create_dispatch_step(phase=DaydreamPhase.X)` creates an agent step with `message="Dispatching N parallel [fix|deep|exploration] tasks"` and one `ObservationResult.subagent_trajectory_ref` per successfully-registered sibling. Created AFTER `async with tg:` exits — only includes siblings that wrote their file. No orphan refs to failed tasks.
- **D-12: `subagent_trajectory_ref` uses relative paths from root trajectory directory.** Value is `trajectories/<first-8-hex>.<descriptor>.json` (relative to the directory containing the root `trajectory.json`). Portable — moving `.daydream/` doesn't break refs. Matches Harbor's convention.
- **D-13: Phase code calling `recorder.create_dispatch_step()` is acceptable under the module-bloat ban.** Phase 2 D-19 bans `Step()`, `ToolCall()`, `Trajectory()` construction inside `phases.py`. Calling a recorder method is the same pattern as passing `phase=DaydreamPhase.FIX` to `run_agent()` — no ATIF model imports needed in `phases.py` beyond the existing enum.

### Claude's Discretion

- Internal structure of `_register_sibling()` (list vs dict, thread-safety for concurrent registration from anyio tasks)
- Whether `create_dispatch_step()` accepts additional kwargs for custom message text or uses a fixed template per phase
- Whether `fork()` accepts optional kwargs for overriding agent_model_name (unlikely needed but possible for mixed-backend deep mode)
- Exact error handling when a child recorder fails to write (degradation vs propagation)
- Whether `_safe_descriptor()` lives as a module-level helper or a staticmethod on `TrajectoryRecorder`

</decisions>

<specifics>
## Specific Ideas

- Phase 2 D-08 explicitly stated Phase 3 "owns the upgrade as one coherent unit" — all fork/sibling/patch mechanics land together. The user confirmed this by selecting the explicit fork wrapper, which keeps the 3 phase-code changes and the recorder.fork() API as one atomic package.
- The user rejected the two-ContextVar architecture (D-04), simplifying the design. This means Phase 2's deferred item "Two-ContextVar architecture (`_RECORDER_VAR` + `_CURRENT_INVOCATION`)" is NOT implemented — single ContextVar is the final design.
- Parent FinalMetrics (SUBA-09) falls out naturally from D-02: each recorder has its own step list and aggregates only its own steps. No explicit exclusion logic needed.
- Step ID isolation (SUBA-08) also falls out from D-02: each child starts its counter at 1 independently.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### ATIF Specification
- `docs/reference/atif_format.md` — Authoritative ATIF spec. Section on `ObservationResult.subagent_trajectory_ref` defines the linking mechanism for sibling trajectories.

### Phase 2 Output (Recorder Core — already landed)
- `daydream/trajectory.py` — Current recorder implementation. Phase 3 adds `fork()`, `_register_sibling()`, `create_dispatch_step()`, `_safe_descriptor()`, and the `parent` field to `TrajectoryRecorder`. Read the full file to understand the existing `__aenter__`/`__aexit__`, `Invocation`, and `_RECORDER_VAR` lifecycle.
- `daydream/agent.py` — `run_agent()` with `phase: DaydreamPhase` kwarg and `get_current_recorder()` guard. No changes expected in Phase 3 — the existing `if recorder is not None:` path works for both parent and child recorders.
- `.planning/phases/02-recorder-core-event-enrichment-mapping/02-CONTEXT.md` — Phase 2 decisions D-01 through D-20. Especially D-08 (minimum surface, Phase 3 owns upgrade), D-09 (one Invocation per run_agent), D-10 (get_current_recorder helper), D-19 (module-bloat ban), D-20 (flat file).

### Parallel Fan-Out Sites (Phase 3 integration points)
- `daydream/phases.py` — `phase_fix_parallel()` (lines ~962–1021) and `phase_per_stack_reviews()` (lines ~1388–1487). Both use `anyio.create_task_group()` + `anyio.CapacityLimiter(4)`. Each gains `recorder.fork(descriptor)` wrapper per D-01.
- `daydream/exploration_runner.py` — `pre_scan()` (lines ~190–280) with `_run_specialist()` closure. Three specialists: `pattern_scanner`, `dependency_tracer`, `test_mapper`. Each gains fork wrapper per D-01.
- `daydream/deep/orchestrator.py` — `run_deep()` calls `phase_per_stack_reviews()`. No changes to the orchestrator itself — the fork happens inside the phase function.

### Continuation Site
- `daydream/phases.py` — `phase_test_and_heal()` (lines ~831–905). Loops with `continuation` token. No Phase 3 changes needed — continuation calls create new Invocations per D-08, appending to the same recorder.

### Project Planning
- `.planning/PROJECT.md` — Key Decisions table (trajectory granularity, ContextVar placement, module-bloat ban). Risk Watch table (hierarchical subagent shape HIGH risk → resolved by sibling-file approach).
- `.planning/REQUIREMENTS.md` — SUBA-01..09 are this phase's 9 requirements verbatim.
- `.planning/ROADMAP.md` — Phase 3 success criteria (5 must-be-true items).

### Codebase Maps
- `.planning/codebase/ARCHITECTURE.md` — Data flow diagrams for all 4 run flows (normal, deep, PR, TTT). Shows where parallel fan-outs occur.
- `.planning/codebase/INTEGRATIONS.md` — Backend protocol shape, CapacityLimiter usage, anyio task group patterns.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`TrajectoryRecorder` (Phase 2)** — Full `__aenter__`/`__aexit__` lifecycle, ContextVar management, step accumulation, trajectory JSON write. Phase 3 extends this class with `fork()`, `create_dispatch_step()`, `_register_sibling()`, and a `parent` field. No architectural changes — additive methods only.
- **`Invocation` (Phase 2)** — Per-`run_agent()` scope with step buffer, in-flight tool_call map, observe() dispatch. Works unchanged against child recorders — no modifications needed.
- **`get_current_recorder()` helper** — Already used by `run_agent()` in `agent.py`. Phase 3 also calls it from 3 phase sites (to get the recorder before forking). Same API, no changes.
- **`DaydreamPhase` enum** — Already imported in `phases.py` and `exploration_runner.py`. `create_dispatch_step()` takes a `phase` arg of this type.
- **`anyio.CapacityLimiter(4)` pattern** — Used in both `phase_fix_parallel` and `phase_per_stack_reviews`. The fork wrapper nests inside the existing limiter/task-group structure.

### Established Patterns
- **Default-arg closure for loop variable capture** — Used in all three fan-out sites (`async def _task(rec=child, ...)`). Fork wrapper follows the same pattern.
- **Exception isolation in task closures** — `phase_fix_parallel` catches exceptions per task and records success/failure. Child recorder `__aexit__` handles the write regardless of task outcome.
- **`if recorder is not None:` guard** — Established in `agent.py`. Phase 3 uses the same guard in phases before calling `recorder.fork()`.

### Integration Points
- **`phases.py` gains 2 new recorder call sites**: `recorder.fork(descriptor)` in the task closure setup and `recorder.create_dispatch_step(phase=...)` after the task group exits. Both wrapped in `if recorder is not None:` guards.
- **`exploration_runner.py` gains 1 fork call site**: inside `_run_specialist()` closure, wrapping the `run_agent()` call.
- **`trajectory.py` gains ~80-120 LOC**: `fork()`, `_register_sibling()`, `create_dispatch_step()`, `_safe_descriptor()`, `_sibling_path()`, and the `parent` field + `_registered_siblings` accumulator. Phase 2 estimated ~400 LOC; Phase 3 pushes to ~500-520 LOC, still under the single-file threshold.
- **`tests/conftest.py`** — No changes. The existing `_reset_trajectory_recorder` autouse fixture resets `_RECORDER_VAR` which covers both parent and child recorders.
- **No changes to `agent.py`** — `run_agent()` is recorder-agnostic; it calls `get_current_recorder()` and works with whatever recorder (parent or child) is active in the ContextVar.

</code_context>

<deferred>
## Deferred Ideas

- **`_CURRENT_INVOCATION` ContextVar** — Originally anticipated by Phase 2 D-08 but eliminated by the explicit fork approach (D-04). Not implemented. If a future feature needs to inspect the active invocation from outside `run_agent()`, revisit then.
- **Mixed-backend deep mode** — `fork()` inherits `agent_model_name` from parent. If deep mode ever uses different backends per stack, the fork API would need a model override. Not a current requirement.
- **Sibling trajectory streaming** — Children write their full trajectory on `__aexit__`. If long-running parallel tasks need mid-run observability, streaming writes could be added. Deferred per PROJECT.md (PERF-01).
- **Cross-sibling deduplication in FinalMetrics** — Parent aggregates only its own steps (SUBA-09). A future viewer might want aggregate totals across root + all siblings. That's a consumer concern, not a recorder concern.

</deferred>

---

*Phase: 03-subagent-wiring-parallel-continuation*
*Context gathered: 2026-04-27*
