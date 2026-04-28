---
gsd_state_version: 1.0
milestone: v1.6
milestone_name: milestone
status: ready_to_plan
last_updated: "2026-04-28T07:49:49.365Z"
progress:
  total_phases: 5
  completed_phases: 4
  total_plans: 18
  completed_plans: 13
  percent: 80
---

# Project State: Daydream — ATIF Migration

**Last updated:** 2026-04-26 (initialized by roadmapper)

## Project Reference

**Core Value:** Every daydream run produces a valid, replayable ATIF v1.6 trajectory that captures the full agent interaction history, tool I/O, and token/cost metrics.

**Current Focus:** Phase 04 — cutover-redaction-cli-surface

**Milestone:** ATIF v1.6 trajectory recording replaces the prefix-tagged `.review-debug-{ts}.log` system. Hard cutover (no dual-write); 5 phases; 72 v1 requirements.

## Current Position

Phase: 04 (cutover-redaction-cli-surface) — EXECUTING
Plan: 1 of 5
**Phase:** 5
**Plan:** Not started
**Status:** Ready to plan

**Progress (overall):** ░░░░░░░░░░ 0/5 phases complete

| Phase | Status |
|-------|--------|
| 1. Vendor ATIF Foundation | ⏳ Not started (current) |
| 2. Recorder Core + Event Enrichment + Mapping | ⏳ Not started |
| 3. Subagent Wiring (Parallel + Continuation) | ⏳ Not started |
| 4. Cutover + Redaction + CLI Surface | ⏳ Not started |
| 5. Test Hardening + Documentation | ⏳ Not started |

## Performance Metrics

| Metric | Baseline | Current | Target |
|--------|----------|---------|--------|
| Existing tests passing | 343/343 | 343/343 | 343/343 (zero regressions) |
| `_log_debug` call sites | 15+ across `agent.py`, `phases.py`, `runner.py`, `exploration_runner.py`, `backends/codex.py` (incl. lazy import) | 15+ (unchanged) | 0 (verified by AST sweep, Phase 4) |
| Trajectories per run | 0 (only `.review-debug-{ts}.log` produced) | 0 | 1 root + N siblings (parallel flows) |
| Module sizes — `phases.py` | 1552 lines | 1552 lines | ≤ 1602 lines (no trajectory `Step()` construction inside; module-bloat ban) |
| Module sizes — `ui.py` | 3470 lines | 3470 lines | ≤ 3520 lines (no trajectory rendering inside; module-bloat ban) |
| Pydantic dep | Transitive via `claude-agent-sdk` | Transitive | Explicit `pydantic>=2.11.7` (Phase 1) |

## Accumulated Context

### Key Decisions (carried from PROJECT.md)

| Decision | Rationale | Phase |
|----------|-----------|-------|
| Recorder placement: `ContextVar` in `daydream/trajectory.py`, not `AgentState` | Per-run lifecycle, not process-singleton; copy-on-spawn handles anyio parallelism; clean test isolation | Phase 2 |
| Trajectory granularity: root file + sibling files for parallel anyio task groups, linked via `ObservationResult.subagent_trajectory_ref` | Matches ATIF v1.4+ subagent semantics; preserves correct execution graph; FinalMetrics aggregation stays clean | Phase 3 |
| Validation library: vendor `harbor.models.trajectories` + `harbor.utils.trajectory_validator` (~700 LOC pure Pydantic + stdlib, Apache-2.0) | Avoids Harbor's 21+ transitive deps + `litellm` supply-chain quarantine; preserves automatic Pydantic validation | Phase 1 |
| Hard cutover, no dual-write phase | Less churn / faster cutover; existing 343-test suite is the regression gate | Phase 4 |
| Redaction must land with cutover, not deferred | Always-on trajectories + bypass-permissions tool surface = secrets exposure risk; trajectories intended for sharing | Phase 4 |
| ATIF schema version pinned to v1.6 (emission) | Harbor's current default; matches all golden fixtures (Terminus-2 v1.6, OpenHands v1.5); v1.6's `ContentPart`/`ImageSource` additions are backward-compatible (str form remains valid) | Phase 1 (vendoring), Phase 2 (emission) |
| New `MetricsEvent` keyed by `AssistantMessage.message_id`; CostEvent extended with `cached_tokens`; Claude token-extraction bug fixed in same phase | Single end-of-call CostEvent too coarse for ATIF per-step Metrics; data already on `ResultMessage.usage` and `AssistantMessage.usage`; fix lives where data is read | Phase 2 |
| Beagle skill prompt = `source="user"` (not `"system"`) | ATIF reserves `"system"` for system-prompt preambles; agent-only fields on user/system steps is a hard validator failure | Phase 2 |
| `--debug` removed; `--trajectory <path>` added; trajectory always written | Trajectory recording is low-cost / high-value; always-on enables training-data collection from normal runs | Phase 4 |

### Open Questions

- (none at roadmap creation time)

### Todos / Blockers

- (none at roadmap creation time)

### Risk Watch (from PITFALLS.md)

| Risk | Severity | Phase Mitigated | Verification Strategy |
|------|----------|-----------------|-----------------------|
| Non-sequential `step_id` from concurrent `run_agent()` calls | HIGH | Phase 2 (per-trajectory monotonic counter) + Phase 3 (per-sibling counter isolation) | Concurrent-mock test in Phase 3 |
| ISO 8601 timestamp mistakes (`datetime.utcnow()`, naive datetimes) | HIGH | Phase 2 | Single `now_iso()` helper + ban on bare `datetime.now()` |
| Dangling `source_call_id` from Codex synthetic UUIDs | HIGH | Phase 2 (in-flight map) | Codex JSONL fixture replay |
| Agent-only fields on user/system steps | HIGH | Phase 2 (use vendored typed `Step` model directly) | Minimal-user-step golden assertion |
| Token accounting — running totals vs deltas (claude-agent-sdk #112) | HIGH | Phase 2 (EVNT-04/05/06 in same phase as MAP-06/07) | Empirical multi-turn fixture (TEST-06) |
| Privacy / secret leaking in tool args + observation content + reasoning | HIGH | Phase 4 (REDA-01..06 with CUT) | Seeded-secret redaction test (REDA-06) |
| Hierarchical subagent shape (flatten vs nest decision) | HIGH | Phase 3 (sibling files via `subagent_trajectory_ref`, NOT nested) | Deep-mode trajectory inspection (TEST-07) |
| Crash → no trajectory; SIGINT loses partial | MEDIUM | Phase 4 (CLI-03 partial flush) | Kill-then-inspect test |
| Out-of-order Codex events / errored tools | MEDIUM | Phase 2 (defensive merge) | Codex edge-case fixtures |
| Test brittleness from over-asserting trajectory structure | MEDIUM | Phase 5 (schema-validity + behavior-predicate per TEST-05) | Code-review checklist |
| `--debug` removal breaks user CLI invocations | MEDIUM | Phase 4 | Manual test + README update in same PR |
| Orphan `_log_debug` callers (lazy-import gotcha at `codex.py:37`) | MEDIUM | Phase 4 (CUT-06 + CUT-08) | AST sweep, not grep |
| `phases.py` / `ui.py` bloat | MEDIUM | Phase 2 (module-bloat ban: no `Step()`/`ToolCall()` construction outside `daydream/trajectory.py`) | `wc -l` check at end of Phase 4 |
| Pydantic perf for hundreds of events per run | LOW | Phase 2 (defer model construction to `recorder.finalize()`) | 1000-step benchmark in Phase 5 |
| Reasoning content leaks user/system context | MEDIUM | Phase 4 (REDA-04 covers `Step.reasoning_content`) | Seeded-secret redaction (REDA-06) |
| Codex token parity gap (no `cost_usd`, no `cached_tokens`) | MEDIUM | Phase 2 (EVNT-07: leave `None`, don't synthesize) | Cross-backend parity test |

## Session Continuity

**Roadmap defined:** 2026-04-26
**Initialized by:** roadmapper agent
**Configured workflow** (per `.planning/config.json`): research enabled, plan-check enabled, verifier enabled, code-review enabled (standard depth), security enforcement at ASVS-1 (block on `high`), worktrees enabled, branching strategy `none`

**Resume instructions:**

- To plan the current phase: `/gsd-plan-phase 1`
- To inspect current state: read this file plus `.planning/ROADMAP.md`
- To review requirements coverage: `.planning/REQUIREMENTS.md` Traceability section
- To revisit research findings: `.planning/research/{ARCHITECTURE,PITFALLS}.md` (note: `STACK.md` and `FEATURES.md` were referenced in the planning prompt but not loaded by the roadmapper; the architecture and pitfalls files contained sufficient context for phase derivation)

---
*State initialized: 2026-04-26 after roadmap creation*
