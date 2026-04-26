# Roadmap: Daydream — ATIF Migration

**Defined:** 2026-04-26
**Granularity:** Coarse-leaning Standard (5 phases)
**Coverage:** 72/72 v1 requirements mapped
**Core Value:** Every daydream run produces a valid, replayable ATIF v1.6 trajectory that captures the full agent interaction history, tool I/O, and token/cost metrics.

## Phases

- [ ] **Phase 1: Vendor ATIF Foundation** - Vendor Harbor's trajectory models + validator + golden fixtures so the rest of the migration has a typed, validated substrate without the supply-chain risk of a runtime Harbor dep
- [ ] **Phase 2: Recorder Core + Event Enrichment + Mapping** - Greenfield `daydream/trajectory.py` recorder with ContextVar propagation, plus `AgentEvent` enrichment (timestamp, MetricsEvent, Claude token extraction) and event-to-ATIF mapping wired into `run_agent()`
- [ ] **Phase 3: Subagent Wiring (Parallel + Continuation)** - Parallel `anyio` task groups (`phase_fix_parallel`, deep mode, exploration pre-scan) emit sibling trajectory files linked via `subagent_trajectory_ref`; continuations append to the same trajectory
- [ ] **Phase 4: Cutover + Redaction + CLI Surface** - Hard removal of `_log_debug` and all 15+ call sites (AST-verified, including the lazy-import gotcha in `codex.py:37`), redaction policy implemented and applied to all trajectory content surfaces, `--debug` removed and `--trajectory <path>` added
- [ ] **Phase 5: Test Hardening + Documentation** - All 343 existing tests verified passing post-migration, new trajectory/redaction/golden-fixture/subagent test suites land, README/CHANGELOG/CLAUDE.md/NOTICE document the new format and breaking CLI change

## Phase Details

### Phase 1: Vendor ATIF Foundation
**Goal**: A self-contained `daydream/atif/` package gives the rest of the migration typed Pydantic models and a working validator without pulling Harbor's 21+ transitive deps (including the litellm supply-chain quarantine).
**Depends on**: Nothing (first phase; greenfield code under `daydream/atif/`)
**Requirements**: VEND-01, VEND-02, VEND-03, VEND-04, VEND-05
**Success Criteria** (what must be TRUE):
  1. A developer can `from daydream.atif.models import Trajectory, Step, ToolCall, ObservationResult, Metrics, FinalMetrics, Agent` with no remaining `from harbor import …` references anywhere in `daydream/` or `tests/`
  2. `daydream/atif/validator.py` accepts every Harbor golden fixture under `tests/fixtures/atif_golden/` (Terminus-2 + OpenHands) when invoked programmatically
  3. `pyproject.toml` declares `pydantic>=2.11.7` as an explicit `[project.dependencies]` entry (not just transitive via `claude-agent-sdk`); `uv sync` resolves cleanly
  4. `daydream/atif/NOTICE` and `daydream/atif/LICENSE` document Apache-2.0 attribution to Harbor for the vendored ~700 LOC
  5. The existing 343-test suite still passes (vendoring is purely additive code; no production module imports from `daydream/atif/` yet)
**Plans**: 4 plans
- [x] 01-01-PLAN.md — Vendor Harbor v0.5.0 source tree (models + validator) + LICENSE + NOTICE + golden fixtures (mechanical-only edits per D-03)
- [x] 01-02-PLAN.md — Add explicit pydantic>=2.11.7 dependency and ruff per-file-ignores stanza for daydream/atif/** to pyproject.toml
- [x] 01-03-PLAN.md — Author daydream/atif/__init__.py public re-export shim, hand-authored negative fixture, and smoke test (VEND-01, VEND-02, VEND-05)
- [x] 01-04-PLAN.md — Phase-gate verification: zero Harbor imports, ruff/mypy clean, 348 pytest passes (VEND-04)

### Phase 2: Recorder Core + Event Enrichment + Mapping
**Goal**: A normal sequential daydream run (review → parse → fix → test) produces a single valid ATIF v1.6 trajectory file with non-empty `Metrics` blocks, correct timestamps, and proper user/agent step segmentation. This phase fixes the dropped-token bug in `backends/claude.py:120-128` in the same patch as the recorder lands so downstream tests can validate against goldens.
**Depends on**: Phase 1 (recorder imports vendored ATIF models; tests assert against vendored validator)
**Requirements**: CORE-01, CORE-02, CORE-03, CORE-04, CORE-05, CORE-06, CORE-07, CORE-08, CORE-09, CORE-10, EVNT-01, EVNT-02, EVNT-03, EVNT-04, EVNT-05, EVNT-06, EVNT-07, MAP-01, MAP-02, MAP-03, MAP-04, MAP-05, MAP-06, MAP-07, MAP-08, MAP-09
**Success Criteria** (what must be TRUE):
  1. Running `daydream <target>` against a small fixture project produces a `<target>/.daydream/trajectory.json` whose `Metrics.prompt_tokens` and `Metrics.completion_tokens` are populated (not `None`) on every Claude agent step, and whose `Step.source` is `"user"` for the Beagle skill prompt and `"agent"` for the response
  2. Each trajectory step carries an ISO 8601 UTC `timestamp`, a `extra.daydream_phase` label (one of `"review" | "parse" | "fix" | "test" | "intent" | "alternatives" | "plan" | "pr_feedback" | "deep" | "exploration"`), and `extra.daydream_run_flow` (one of `"normal" | "ttt" | "pr" | "deep"`)
  3. Every `ToolCall(tool_call_id=…)` has a paired `ObservationResult(source_call_id=…)` in the **same step**, validated by the vendored validator (intra-step tool_call_id scope per ATIF v1.6)
  4. The trajectory's `FinalMetrics` totals equal the sum of per-step `Metrics` (no running-totals leak from `claude-agent-sdk` `ResultMessage.usage`); a multi-turn empirical test confirms `input_tokens` is per-call (SDK issue #112 risk gate)
  5. The recorder is propagated via a `ContextVar` defined in `daydream/trajectory.py` (NOT on `AgentState`); `tests/conftest.py` has an autouse `_reset_trajectory_recorder` fixture mirroring `reset_state()`, and direct test invocation of `run_agent()` without an active recorder is a clean no-op
**Plans**: TBD

### Phase 3: Subagent Wiring (Parallel + Continuation)
**Goal**: Daydream's three parallel-fan-out flows (`phase_fix_parallel`, `daydream/deep/orchestrator.run_deep`, `exploration_runner.pre_scan`) emit one sibling trajectory file per parallel invocation, linked from the parent via `ObservationResult.subagent_trajectory_ref`. Continuations stay in the same trajectory.
**Depends on**: Phase 2 (needs `ContextVar` machinery from CORE-02 + recorder lifecycle from CORE-03 already in place; subagent wiring is the layer above it)
**Requirements**: SUBA-01, SUBA-02, SUBA-03, SUBA-04, SUBA-05, SUBA-06, SUBA-07, SUBA-08, SUBA-09
**Success Criteria** (what must be TRUE):
  1. Running `daydream --deep <target>` produces one root `<target>/.daydream/trajectory.json` plus one sibling per stack under `<target>/.daydream/trajectories/<session_id>.<descriptor>.json`; each sibling's `Trajectory.session_id` matches the root's
  2. The root trajectory's parent step carries an `ObservationResult.subagent_trajectory_ref` pointing at each sibling file path; the vendored validator accepts both root and siblings
  3. `phase_fix_parallel` with N parallel fixes produces N sibling trajectories (one per fix), and `exploration_runner.pre_scan` produces one sibling per specialist (pattern_scanner, dependency_tracer, test_mapper)
  4. `step_id` counters are isolated per trajectory file — no collisions across siblings — and parent `FinalMetrics` aggregates ONLY parent steps (sibling totals stay in their own file with no double-counting)
  5. A `run_agent_with_continuation` continuation call appends to the existing trajectory's step list (preserves agent identity) and does NOT spawn a sibling; the sequential phase chain (`phase_review` → `phase_parse_feedback` → `phase_fix` → `phase_test_and_heal`) emits as continuous steps in one root file
**Plans**: TBD

### Phase 4: Cutover + Redaction + CLI Surface
**Goal**: The legacy `_log_debug` system is gone — all 15+ call sites in `agent.py`, all `[REVERT]/[PARSE_FAIL]/[STAGE]/[TTT_*]` lines in `phases.py`, all `[CODEX_*]` lines in `backends/codex.py`, the `[PRE_SCAN]` lines in `exploration_runner.py`, the `--debug` flag, the `AgentState.debug_log` field, and the `.review-debug-{ts}.log` initialization in `runner.py`. Redaction lands in the same release so always-on trajectories never ship raw secrets. The lazy-import `from daydream.agent import _log_debug` inside `daydream/backends/codex.py:37` is verified-removed via AST sweep, not just grep. `--trajectory <path>` replaces `--debug` and SIGINT flushes a partial trajectory.
**Depends on**: Phase 3 (cutover removes the legacy system *after* the new system — including subagent flows — is proven working; redaction must apply to root + sibling content uniformly so SUBA must already work)
**Requirements**: REDA-01, REDA-02, REDA-03, REDA-04, REDA-05, REDA-06, CUT-01, CUT-02, CUT-03, CUT-04, CUT-05, CUT-06, CUT-07, CUT-08, CLI-01, CLI-02, CLI-03, CLI-04, CLI-05
**Success Criteria** (what must be TRUE):
  1. An AST-level sweep across `daydream/` and `tests/` finds zero references to `_log_debug`, `debug_log`, `set_debug_log`, `get_debug_log`, the `[CODEX_RAW]/[CODEX_WARN]/[CODEX_UNHANDLED]/[REVERT]/[PARSE_FAIL]/[STAGE]/[TTT_*]/[PRE_SCAN]/[PROMPT]/[TEXT]/[TOOL_USE]/[COST]` log prefixes, including the lazy import inside `daydream/backends/codex.py:_raw_log` (which previously imported `_log_debug` from inside a function body)
  2. `daydream --debug` is rejected by argparse (or shows a one-release deprecation pointer at `--trajectory`); `daydream --trajectory <custom-path> <target>` writes the trajectory to `<custom-path>` and ignores the default; `--ttt`, `--pr`, `--deep`, and `--review-only` flags continue to work and produce trajectories
  3. A trajectory generated with seeded inputs containing `sk-test-12345`, `ghp_test123`, `xoxb-test456`, `AKIA0000TESTKEY00000`, a JWT (`eyJ…`), `/Users/ka/foo`, `/home/alice/bar`, and a `.env`-style `OPENAI_API_KEY=sk-real-key` line produces output where none of those literals appear in `ToolCall.arguments`, `ObservationResult.content`, `Step.message`, or `Step.reasoning_content` — redaction is redact-or-omit, never raw-pass-through
  4. Sending SIGINT (Ctrl-C) or SIGTERM mid-run flushes the in-progress trajectory to `<path>.partial` with `extra.partial=true`; the file passes the vendored validator (partial trajectories with no `final_metrics` are valid per ATIF v1.6)
  5. Help text (`daydream --help`) describes the trajectory output, the `--trajectory` flag semantics, and that the redactor is on by default; `make lint` and `make typecheck` pass cleanly with the legacy code removed
**Plans**: TBD

### Phase 5: Test Hardening + Documentation
**Goal**: Migration-complete signal — all 343 existing tests pass, new test suites cover the recorder, redaction, golden-fixture round-trip, and subagent file shapes (using schema-validity + behavior-predicate patterns, not full-tree snapshot equality). README, CHANGELOG, CLAUDE.md, and `daydream/atif/NOTICE` document the new format, the breaking CLI change, and consumer integration paths.
**Depends on**: Phase 4 (final test pass and consumer documentation can only be authored once the full migrated system — recorder, subagents, redaction, CLI surface — is in place)
**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04, TEST-05, TEST-06, TEST-07, DOCS-01, DOCS-02, DOCS-03, DOCS-04, DOCS-05, DOCS-06
**Success Criteria** (what must be TRUE):
  1. `make test` reports all 343 pre-existing tests passing plus new tests in `tests/test_trajectory.py` (recorder lifecycle, step coalescing, tool-call correlation, validator round-trip), `tests/test_redaction.py` (per-pattern positive/negative cases), `tests/test_atif_models.py` (parametrized golden-fixture acceptance), and the subagent test exercising `phase_fix_parallel` + deep + exploration sibling shapes
  2. The new test suite follows schema-validity + behavior-predicate patterns (no `assert trajectory == expected_dict` full-tree comparisons); a deliberately-broken trajectory is caught by the validator path, and the empirical multi-turn fixture confirms `claude-agent-sdk==0.1.52` `ResultMessage.usage["input_tokens"]` is per-call (SDK issue #112 gate)
  3. `README.md` documents the trajectory output format, the default path (`<target>/.daydream/trajectory.json`), the `--trajectory` flag, the redaction policy (and what users should NOT expect to see scrubbed), and points consumers at Harbor / replay viewers / SFT-RL training pipelines
  4. `CHANGELOG.md` calls out the breaking CLI change (`--debug` removed, `--trajectory <path>` added) under a versioned heading; `CLAUDE.md` mentions the trajectory format and the `daydream/trajectory.py` location; `daydream/atif/NOTICE` carries Apache-2.0 attribution
  5. A reader can run `daydream <target>`, open the resulting `<target>/.daydream/trajectory.json`, and follow the README's links to validate it with Harbor's external validator and replay it in a viewer — the migration is consumer-ready
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Vendor ATIF Foundation | 0/4 | Planned | - |
| 2. Recorder Core + Event Enrichment + Mapping | 0/0 | Not started | - |
| 3. Subagent Wiring (Parallel + Continuation) | 0/0 | Not started | - |
| 4. Cutover + Redaction + CLI Surface | 0/0 | Not started | - |
| 5. Test Hardening + Documentation | 0/0 | Not started | - |

## Coverage Summary

**v1 requirements:** 72 total
**Mapped:** 72 (100%) ✓
**Unmapped:** 0

| Category | Count | Phase |
|----------|-------|-------|
| VEND (Vendoring) | 5 | Phase 1 |
| CORE (Recorder Core) | 10 | Phase 2 |
| EVNT (Event Enrichment) | 7 | Phase 2 |
| MAP (Event-to-ATIF Mapping) | 9 | Phase 2 |
| SUBA (Subagent Wiring) | 9 | Phase 3 |
| REDA (Privacy & Redaction) | 6 | Phase 4 |
| CUT (Legacy Removal & Cutover) | 8 | Phase 4 |
| CLI (CLI Surface) | 5 | Phase 4 |
| TEST (Test Suite) | 7 | Phase 5 |
| DOCS (Documentation) | 6 | Phase 5 |

## Dependency Notes

- **Phase 1 → Phase 2**: Recorder (Phase 2) imports vendored Pydantic models (Phase 1); tests assert against vendored validator
- **Phase 2 internal coupling (intentional)**: EVNT-04/05/06 (Claude token extraction from `ResultMessage.usage`) MUST land with CORE-* / MAP-* — splitting them produces empty `Metrics` blocks and breaks downstream golden-fixture tests
- **Phase 2 → Phase 3**: SUBA-* depends on CORE-02 (`ContextVar` machinery) being in place so anyio task-group copy-on-spawn correctly establishes parent → child recorder relationships without threading recorder through every phase signature
- **Phase 3 → Phase 4**: Cutover removes legacy `_log_debug` only after parallel/continuation flows produce valid trajectories; REDA-* must land with CUT-* per pitfall #8 (always-on trajectories + bypass-permissions tool surface = secrets must be scrubbed before any release ships)
- **Phase 4 → Phase 5**: Final test hardening and consumer-facing docs author once the full migrated surface (recorder + subagents + redaction + CLI) is stable
- **AST sweep, not grep**: CUT-06 covers the lazy import `from daydream.agent import _log_debug` *inside a function body* in `daydream/backends/codex.py:37` — Phase 4 must use AST-based verification (CUT-08) so this gotcha doesn't survive a naive grep

---
*Roadmap defined: 2026-04-26 from 72 requirements + 4 research artifacts*
