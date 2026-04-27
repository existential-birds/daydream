# Phase 2: Recorder Core + Event Enrichment + Mapping - Context

**Gathered:** 2026-04-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Greenfield `daydream/trajectory.py` module ships a `TrajectoryRecorder` that turns the existing `AgentEvent` stream into a single valid ATIF v1.6 trajectory file for one daydream run. Scope:

1. **Recorder + Invocation + Redactor classes** in `daydream/trajectory.py`, propagated via a single `ContextVar` (`_RECORDER_VAR`). One Invocation per `run_agent()` call; sequential phases share one root `Trajectory`.
2. **Event enrichment** — every `AgentEvent` dataclass in `daydream/backends/__init__.py` gains an ISO 8601 UTC `timestamp`; new `MetricsEvent` dataclass; `CostEvent` extended with `cached_tokens`; Claude backend (`backends/claude.py:120-128`) populates `input_tokens` / `output_tokens` / `cached_tokens` from `ResultMessage.usage` (currently dropped); Claude emits `MetricsEvent` per `AssistantMessage.message_id` from `AssistantMessage.usage`; Codex emits `MetricsEvent` from `turn.completed.usage`.
3. **Event-to-ATIF mapping** in `daydream/agent.py:run_agent()` — `[PROMPT]` → user Step; `TextEvent` → agent Step `message`; `ThinkingEvent` → `reasoning_content`; `ToolStartEvent` → `ToolCall`; `ToolResultEvent` → `ObservationResult` correlated via in-flight `tool_call_id → Step` map; `MetricsEvent` → per-step `Metrics`; `ResultEvent` → trajectory `FinalMetrics`. Each Step carries `extra.daydream_phase` and `extra.daydream_run_flow`.

**Out of phase scope (deferred):**
- Sibling trajectory files for parallel `anyio` task groups (Phase 3: SUBA-01..09 — `phase_fix_parallel`, deep, exploration)
- Continuation flows appending to the same trajectory (Phase 3: SUBA-05)
- Redaction patterns and call surfaces (Phase 4: REDA-01..06) — Phase 2 ships only the no-op pass-through `Redactor` API surface
- `_log_debug` removal and `--debug` → `--trajectory <path>` CLI swap (Phase 4: CUT-* + CLI-*)

</domain>

<decisions>
## Implementation Decisions

### Step Coalescing (CORE-05, MAP-01..09)

- **D-01: One ATIF Step per `AssistantMessage`.** Each model turn = one agent Step containing accumulated `message`, `reasoning_content`, all `tool_calls` from that turn, and the matching `ObservationResult`s. Matches ATIF v1.6's natural shape; `tool_call_id` intra-step scope (CORE-06, MAP-05) works cleanly; minimum step count.
- **D-02: Step flush on next `AssistantMessage` or `ResultMessage`.** Step stays "open" as events stream in. Closes when (a) the next `AssistantMessage` starts a new step, or (b) `ResultMessage` signals end of `run_agent()`. `ToolResult` events always land on whichever step holds the matching `ToolCall` via the in-flight `tool_call_id → Step` map per CORE-06.
- **D-03: TextEvent chunks concatenate into a single `message` string.** ATIF v1.6 `Step.message` is `str` (multimodal `ContentPart`/`ImageSource` is out of scope per PROJECT.md). Streaming chunks append to the open Step's message buffer.
- **D-04: `MetricsEvent` lands on the Step opened by its `AssistantMessage.message_id`.** Recorder maps `message_id → Step` at AssistantMessage start. The matching `MetricsEvent` (per MAP-06) attaches to that exact Step. Clean 1:1 — each agent Step has its own `Metrics`.

### Phase Label Propagation (MAP-08, MAP-09)

- **D-05: Phase label is a required keyword-only arg on `run_agent()`.** Signature gains `*, phase: DaydreamPhase` (no default — required). Every call site in `phases.py` (~18) passes a literal enum member. No backwards-compat shim — update all callers and test mocks. Type-checkable; mypy catches missed call sites; enforces MAP-08 at the type level. Explicit signature is the idiomatic Python choice over a second ContextVar.
- **D-06: `DaydreamPhase` and `DaydreamRunFlow` are enums, both defined in `daydream/trajectory.py`.** Members:
  - `DaydreamPhase`: `REVIEW`, `PARSE`, `FIX`, `TEST`, `INTENT`, `ALTERNATIVES`, `PLAN`, `PR_FEEDBACK`, `DEEP`, `EXPLORATION` (from MAP-08).
  - `DaydreamRunFlow`: `NORMAL`, `TTT`, `PR`, `DEEP` (from MAP-09).
  - Enum value strings match the ATIF `extra` field literals exactly (e.g., `DaydreamPhase.REVIEW.value == "review"`).
- **D-07: `run_flow` is a per-trajectory invariant, set once at recorder init from `RunConfig`.** `runner.py` knows the active flow at recorder construction (`run()` / `run_pr_feedback()` / `run_trust()` / `run_deep()`). Recorder stores `run_flow: DaydreamRunFlow` and stamps every Step automatically. Does not change inside a single trajectory — not call-site data.

### Recorder API Shape (CORE-01..10)

- **D-08: Phase 2 ships the *minimum* recorder surface — no anticipatory architecture for Phase 3.** Single `_RECORDER_VAR` ContextVar. Single root `Trajectory` per recorder. `Invocation` exists as a per-`run_agent()` scope (clean home for the step buffer + in-flight `tool_call_id → Step` map per CORE-05/06) but has **no `parent` field** and no second ContextVar. Sequential `run_agent()` calls all flush their steps to `Trajectory.steps`. Phase 3 owns the upgrade as one coherent unit: it adds `_CURRENT_INVOCATION` ContextVar, `Invocation(parent=...)`, sibling-file write, and parent observation patching together. (See `feedback_phase_split_coherence.md` in user memory: the user explicitly rejected building "API surface in Phase 2, wiring in Phase 3" half-built architecture.)
- **D-09: One `Invocation` per `run_agent()` call.** Each invocation opens its own scope, observes events, and on exit flushes its accumulated Steps to the root `Trajectory.steps`. Owns the local in-flight `tool_call_id → Step` map (CORE-06). Continuation handling (SUBA-05) is Phase 3's concern.
- **D-10: `daydream/trajectory.py` exposes a module-level `get_current_recorder() -> TrajectoryRecorder | None` helper.** `agent.py` imports the helper, not the ContextVar. ContextVar (`_RECORDER_VAR`) stays private. Phase 3 will add `get_current_invocation()` alongside without breaking callers. Recorder presence check stays a clean `if recorder is not None:` guard.
- **D-11: Recorder failure mode (CORE-09) — Phase 2 ships only the implicit-write degrade-with-warning path.** Trajectory write failure (disk full, permission) emits `print_warning(...)` and continues; the run does not fail. The explicit `--trajectory <path>` → fail-loud branch is part of Phase 4 (CLI-02) and doesn't exist in Phase 2's surface yet. RunConfig carries the path; when Phase 4 adds the CLI flag, the failure-mode branch lights up.

### Redactor Surface (CORE-01)

- **D-12: Phase 2 ships `Redactor` as a no-op pass-through with the *final* API surface.** Single method: `redact_step(step: Step) -> Step` (returns input unchanged in Phase 2). Recorder calls it on every Step at flush time. Phase 4 fills in regex pattern lists internally — zero changes to the recorder call site, zero changes to public API. This is the clean version of the "Phase N API surface" pattern: the recorder genuinely *uses* the redactor in Phase 2 (calls it on every Step), so wiring is exercised; Phase 4 is purely additive (rule list).
- **D-13: Redaction runs at per-Step flush time, not on serialization at `__aexit__`.** Recorder calls `redactor.redact_step(step)` immediately before adding the finalized Step to `Trajectory.steps`. Phase 4 redactor sees content before validation; partial-write paths (SIGINT → `.partial.json`, Phase 4) inherit the same redaction posture. This eliminates Pitfall 8's "partial trajectories may have less consistent redaction" concern.

### Token Accounting (EVNT-04..07, MAP-06, MAP-07)

- **D-14: Trust per-call semantics for `claude-agent-sdk==0.1.52` `AssistantMessage.usage` and `ResultMessage.usage`.** `MetricsEvent` carries the SDK's reported tokens directly to per-step `Metrics`; no "subtract last_seen_cumulative" defensive logic in Phase 2. PROJECT.md Pitfall 5 risk is gated by an empirical multi-turn fixture test in Phase 5 (TEST-06) — if SDK 0.1.52 turns out to report cumulative, the fix lands then. Implementation: the Claude backend extracts `input_tokens`, `output_tokens`, `cache_read_input_tokens` from `AssistantMessage.usage` for `MetricsEvent` (per-step) and from `ResultMessage.usage` for `CostEvent` (end-of-call signal feeding `FinalMetrics`).
- **D-15: `cached_tokens` is a *subset* of `prompt_tokens`, not additive.** Per the ATIF spec and PITFALLS Pitfall 5, the cached portion of input is reported alongside, not added. Recorder passes `cache_read_input_tokens` directly into `Metrics.cached_tokens` and does NOT add it to `prompt_tokens`.
- **D-16: Codex `MetricsEvent` parity gap is acceptable.** Codex emits `MetricsEvent` from `turn.completed.usage` with `input_tokens` + `output_tokens` only; `cost_usd` and `cached_tokens` set to `None` (per EVNT-07). ATIF Metrics fields are all optional. Do NOT synthesize cost from a token-price table (PITFALLS technical-debt warning).

### Test Pattern (Phase 2 sets the precedent for Phase 5)

- **D-17: Test isolation via autouse `_reset_trajectory_recorder` fixture in `tests/conftest.py` (CORE-10).** Mirrors the existing `reset_state()` pattern for `AgentState`. Resets `_RECORDER_VAR.set(None)` before and after every test so cross-test bleed cannot occur.
- **D-18: New Phase 2 tests use schema-validity + behavior-predicate patterns, NOT full-tree snapshot equality** (per PROJECT.md Constraint, PITFALLS Pitfall 11). Recorder tests assert: (a) the produced `Trajectory` passes `daydream.atif.validate()`, (b) one or two specific behavioral predicates per test (e.g., "this trajectory has at least one agent Step with a Bash tool call whose command contains `pytest`"). Sets the precedent that Phase 5's `tests/test_trajectory.py` extends.

### Module Bloat Ban (PROJECT.md Constraint, PITFALLS Pitfall 14)

- **D-19: Zero ATIF model construction (`Step()`, `ToolCall()`, `Trajectory()`) inside `phases.py` or `ui.py`.** All construction lives in `daydream/trajectory.py`. `phases.py` only passes `phase=DaydreamPhase.X` to `run_agent()`; `ui.py` is untouched in Phase 2. `agent.py` calls `inv.observe(event)` once per event — no construction logic.
- **D-20: Keep `daydream/trajectory.py` as a single flat file.** Estimated ~400 LOC for Phase 2 (recorder + invocation + redactor stub + ContextVar + helpers + ATIF mapping). Phase 3 + Phase 4 may push it past 500 LOC; if so, split into a `daydream/trajectory/` package then. PITFALLS Pitfall 14 explicitly endorses this flat-then-split rule.

### Claude's Discretion

- Exact internal layout of `Invocation` (whether step buffer is a list vs a deque, naming of the in-flight map field, etc.) — implementation detail, not user-facing.
- Whether the no-op `Redactor` exposes any helper methods alongside `redact_step()` (e.g., a private `_redact_text(s: str) -> str` that Phase 4 fills in). Either is fine; the public `redact_step` is the contract.
- Specific format of `Step.timestamp` — must be `now_iso()` style ISO 8601 UTC ending in `Z` per PITFALLS Pitfall 2 (single helper, ban `datetime.utcnow()`); exact helper name is Claude's call.
- Which ATIF model fields show up explicitly versus get defaulted via Pydantic. As long as the produced `Trajectory` passes `daydream.atif.validate()`, the constructor style is open.
- Whether `MetricsEvent` carries `cost_usd` per-step or only on `ResultMessage`-derived `CostEvent`. EVNT-02 lists `cost_usd: float | None` on `MetricsEvent`; AssistantMessage may or may not provide it — if not, `None` is fine.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### ATIF Specification
- `docs/reference/atif_format.md` — Authoritative ATIF spec (format, Pydantic model usage, validator behavior, OpenHands accumulated-to-delta example). Schema version table at lines 396–402.
- ATIF RFC: https://github.com/laude-institute/harbor/blob/main/docs/rfcs/0001-trajectory-format.md — Source RFC referenced by the spec.

### Phase 1 Output (Vendored ATIF — already landed)
- `daydream/atif/__init__.py` — Public re-export shim. Phase 2 imports `Trajectory`, `Agent`, `Step`, `ToolCall`, `Observation`, `ObservationResult`, `Metrics`, `FinalMetrics`, `validate` from here (per Phase 1 D-06).
- `daydream/atif/models/` — Vendored Pydantic models (mechanical-only edits per Phase 1 D-03 / D-05).
- `daydream/atif/validator.py` — Vendored `TrajectoryValidator`; tests use `from daydream.atif import validate` (passthrough wrapper) or `from daydream.atif.validator import TrajectoryValidator` for detailed errors.
- `tests/fixtures/atif_golden/{terminus2,openhands}/` — Golden round-trip fixtures. Phase 2 tests can replay these to confirm the recorder produces compatible output.
- `tests/fixtures/atif_golden/_invalid/` — Negative-path fixture for validator-catches-broken-trajectory tests.
- `.planning/phases/01-vendor-atif-foundation/01-CONTEXT.md` — Phase 1 decisions D-01..D-13, especially D-06 (public API surface) and D-08 (`validate()` returns Harbor's existing surface).

### Project Planning
- `.planning/PROJECT.md` — Active requirements (especially the Trajectory recording + Event enrichment + Mapping bullets), Out of Scope decisions, full Key Decisions table (Recorder placement, MetricsEvent keying, Beagle-skill = `source="user"`, ATIF v1.6 emission pin, Module-bloat ban, ContextVar-not-AgentState).
- `.planning/REQUIREMENTS.md` — CORE-01..10, EVNT-01..07, MAP-01..09 are this phase's 26 requirements verbatim. Traceability section confirms scope.
- `.planning/ROADMAP.md` — Phase 2 success criteria (5 must-be-true items, especially: per-call `input_tokens` empirical confirmation, Beagle prompt = `source="user"`, ContextVar-not-AgentState).
- `.planning/STATE.md` — Risk Watch table maps Phase 2's HIGH-severity risks (step_id ordering, timestamp format, dangling source_call_id, agent-only fields on user steps, token accounting) to verification strategies.

### Research Artifacts
- `.planning/research/ARCHITECTURE.md` — Full ContextVar + Invocation analysis (Q1–Q9). Phase 2 implements Q1 (per-run recorder via ContextVar in `daydream/trajectory.py`), Q2 (record inside the existing `run_agent` event loop, before UI rendering), Q5 (phases as flat sibling steps in one trajectory + `extra.daydream_phase` label), Q6 Stage 1+2+3 (greenfield trajectory.py, backend enrichment, run_agent integration), Q7 (catch-and-degrade at recording boundaries), Q9 (no trajectory-related state on `AgentState`). Defers Q3 (subagent wiring) and Q4 (continuation) to Phase 3.
- `.planning/research/PITFALLS.md` — Pitfalls 1 (step_id sequencing), 2 (ISO 8601 timestamp format), 3 (dangling `source_call_id`), 4 (agent-only fields on user steps), 5 (Claude SDK token accounting), 6 (Codex parity gap), 11 (test brittleness), 14 (`phases.py`/`ui.py` bloat), 15 (Pydantic perf — defer construction to finalize). Pitfalls 7 (subagent shape), 8 (redaction patterns), 9 (SIGINT partial flush), 12 (`--debug` removal), 13 (`_log_debug` orphans), 16 (reasoning leaks) are Phase 3 / Phase 4 concerns.

### Codebase Maps (existing patterns to honor)
- `.planning/codebase/CONVENTIONS.md` — `snake_case` modules, dataclass for config + events, `Backend`-first parameter convention, `from __future__ import annotations` where needed, no silent swallowing.
- `.planning/codebase/STRUCTURE.md` — `Where to Put New Code` rules; `daydream/trajectory.py` as a new top-level module is precedented (`daydream/exploration_runner.py`, `daydream/pr_review.py`).
- `.planning/codebase/CONCERNS.md` — Module size watchlist (`phases.py` 1552, `ui.py` 3470, `runner.py` 781). Phase 2 must not push these — explicit module-bloat ban in PROJECT.md Constraint.
- `.planning/codebase/TESTING.md` — Test patterns (autouse fixtures, mock backend convention, deterministic timestamps via `monkeypatch`).
- `.planning/codebase/ARCHITECTURE.md` — Backend protocol shape, `AgentEvent` union, current `run_agent()` consumption loop.

### Source Files (Phase 2 integration points)
- `daydream/agent.py:283` — `run_agent()` entry; phase 2 adds the keyword-only `phase: DaydreamPhase` arg + recorder Invocation lifecycle around the existing `async for event in event_iter:` loop. Existing `_log_debug` calls stay (Phase 4 cutover removes them).
- `daydream/agent.py:339-430` — Event-loop dispatch on `isinstance(event, ...)`; Phase 2 inserts `inv.observe(event)` inside this loop per Architecture research Q2.
- `daydream/backends/__init__.py` — `AgentEvent` dataclasses gain `timestamp: str`; new `MetricsEvent` dataclass added; `CostEvent` extended with `cached_tokens`. Update `AgentEvent` TypeAlias union.
- `daydream/backends/claude.py:120-128` — Currently drops `input_tokens`, `output_tokens`, `cached_tokens` (always `None`). Phase 2 extracts from `ResultMessage.usage["input_tokens"|"output_tokens"|"cache_read_input_tokens"]` per EVNT-04/05.
- `daydream/backends/claude.py:95` — `AssistantMessage` handling; Phase 2 emits `MetricsEvent` per `AssistantMessage.message_id` from `AssistantMessage.usage` per EVNT-06.
- `daydream/backends/codex.py` — Codex `turn.completed.usage` extraction; Phase 2 emits `MetricsEvent` per EVNT-07. (Existing `[CODEX_RAW]` / `[CODEX_WARN]` log lines stay; Phase 4 cutover removes.)
- `daydream/runner.py` — `run()`, `run_pr_feedback()`, `run_trust()`, `run_deep()` each construct one `TrajectoryRecorder` per call with the active `DaydreamRunFlow`. RunConfig gains `trajectory_path: Path | None` field (default-resolved to `<target>/.daydream/trajectory.json`). Existing `set_debug_log()` plumbing at lines 461–472 stays in Phase 2.
- `daydream/phases.py` — Every `run_agent()` call site (~18) updated to pass `phase=DaydreamPhase.X` (e.g., `phase_review` passes `DaydreamPhase.REVIEW`). No ATIF model construction.
- `tests/conftest.py` — Add autouse `_reset_trajectory_recorder` fixture mirroring `reset_state()`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`daydream/atif/` (Phase 1 output)** — Typed Pydantic models + validator are ready. Phase 2 only constructs models from inside `daydream/trajectory.py`; downstream tests use `daydream.atif.validate()` for schema-validity assertions.
- **`tests/fixtures/atif_golden/`** — Golden round-trip fixtures already in place. Phase 2 tests can use them as a parametrized replay corpus to confirm recorder output is compatible.
- **`AgentState` autouse fixture pattern** — `tests/conftest.py` already has the `reset_state()` autouse fixture pattern. Phase 2's `_reset_trajectory_recorder` fixture follows the same shape (CORE-10).
- **`run_agent()` event-loop dispatch** at `daydream/agent.py:339-430` is the single chokepoint for backend events. One insertion point handles every event type.
- **`RunConfig` dataclass** in `daydream/runner.py` is the natural carrier for `trajectory_path`. Existing dataclass-passed-through-phases convention applies cleanly.

### Established Patterns
- **`Backend` first parameter convention** in `phases.py` — Phase 2 adds `phase: DaydreamPhase` as a *keyword-only* argument, preserving the convention.
- **`AgentEvent` dataclass union pattern** — Phase 2's `MetricsEvent` follows the same shape; updating the `AgentEvent` TypeAlias to include it is a one-line change.
- **Module-level singleton + getter/setter pair** (`set_debug_log`/`get_debug_log` for `AgentState`) — Phase 2 introduces a similar pair for the recorder ContextVar (`get_current_recorder()` only; setter is implicit via `__aenter__`).
- **`anyio.run(run, config)` event loop entry** — `TrajectoryRecorder.__aenter__` / `__aexit__` is the natural async-context shape; nests cleanly inside the existing `runner.run()` flow.

### Integration Points
- **`runner.py` constructs the recorder once per run.** Phase 2 wraps the existing `run()` body with `async with TrajectoryRecorder(...) as recorder:`. Existing debug-log setup at lines 461–472 is left in place — Phase 4 removes it as part of the cutover.
- **`agent.py:run_agent()` reads recorder via `get_current_recorder()`.** Inserts an `Invocation` lifecycle around the existing event loop. Existing `_log_debug` calls stay (Phase 4 removes them).
- **Event timestamps stamped at the *backend yield* site** per PITFALLS Pitfall 2 ("stamp the timestamp at the *event consumption boundary* in `agent.py:run_agent()`") — but PROJECT.md Active says "All event dataclasses…carry an ISO 8601 UTC `timestamp` field". Reconciliation: the dataclass *holds* the timestamp; the recorder reads it from there. The backend can populate it at yield time (Claude/Codex both have access to a clock); using a single `now_iso()` helper imported by both backends gives one source of truth.
- **`tests/conftest.py` autouse fixture** for ContextVar reset. Mirrors `reset_state()` pattern.

</code_context>

<specifics>
## Specific Ideas

- **PROJECT.md Constraints lock the recorder placement:** "TrajectoryRecorder lives in new `daydream/trajectory.py` module, propagated via `ContextVar` (not `AgentState`)." Phase 2 implements that single-file placement; the `daydream/trajectory/` package split is deferred until LOC count demands it (Pitfall 14).
- **PROJECT.md Constraints lock the test isolation pattern:** "Test isolation via autouse `_reset_trajectory_recorder` fixture in `conftest.py` mirroring the existing `reset_state()` pattern." Phase 2 ships exactly that — same shape, same name pattern.
- **PROJECT.md Constraints lock the module-bloat ban:** "No `Step()`, `ToolCall()`, or `Trajectory()` construction inside `phases.py` or `ui.py` — all ATIF model construction stays in `daydream/trajectory.py`." Phase 2 enforces by construction (no `from daydream.atif import Step` in phases.py; only enum imports).
- **PITFALLS Pitfall 5 → TEST-06 (Phase 5):** Token accounting for SDK 0.1.52 — the empirical multi-turn fixture test in Phase 5 confirms `ResultMessage.usage["input_tokens"]` is per-call. Phase 2 trusts that until proven otherwise. If TEST-06 fails, Phase 2's token extraction needs a `last_seen_cumulative` subtract step.
- **The user explicitly rejected "Phase N API surface, Phase N+1 wiring" half-built architecture** during this discussion (see Area 3 Q1 walk-back). The Redactor pattern (D-12) is acceptable because Phase 2 *exercises* the API (recorder calls `redact_step` on every Step) — the API isn't dormant infrastructure. Future phases must apply the same lens.

</specifics>

<deferred>
## Deferred Ideas

- **Two-ContextVar architecture (`_RECORDER_VAR` + `_CURRENT_INVOCATION`)** — Phase 3. Adding the second ContextVar in Phase 2 would create the half-built parent-linkage state the user explicitly rejected. Phase 3 owns the upgrade as one coherent unit.
- **`Invocation(parent=Invocation | None)` parent linkage** — Phase 3. Same coherence reasoning.
- **Sibling trajectory file write (`<target>/.daydream/trajectories/<id>.json`)** — Phase 3 (SUBA-02..04, SUBA-06).
- **Continuation appending to existing trajectory** — Phase 3 (SUBA-05). Phase 2's `Invocation` model is per-call; continuation handling that *finds* a prior `Invocation` is Phase 3's API addition.
- **Redaction regex patterns + test corpus** — Phase 4 (REDA-01..06). Phase 2 ships only the no-op pass-through; Phase 4 fills the rule list inside `Redactor.redact_step()` without changing the API.
- **`--trajectory <path>` CLI flag + `--debug` removal** — Phase 4 (CLI-01..05, CUT-01..08). Phase 2's `RunConfig` carries `trajectory_path` but the CLI doesn't expose it yet.
- **SIGINT partial-flush to `.partial.json`** — Phase 4 (CLI-03). Phase 2's recorder writes only on clean `__aexit__`.
- **AST-based `_log_debug` orphan sweep** — Phase 4 (CUT-08). Phase 2 leaves `_log_debug` calls untouched.
- **`Trajectory.to_json_dict()` / serialization perf benchmarks** — Phase 5 (Pitfall 15). Phase 2 defers Pydantic model construction to recorder finalize, but doesn't ship a benchmark.

</deferred>

---

*Phase: 02-recorder-core-event-enrichment-mapping*
*Context gathered: 2026-04-26*
