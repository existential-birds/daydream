# Requirements: Daydream — ATIF Migration

**Defined:** 2026-04-26
**Core Value:** Every daydream run produces a valid, replayable ATIF v1.6 trajectory that captures the full agent interaction history, tool I/O, and token/cost metrics.

## v1 Requirements

### Vendoring (VEND)

- [ ] **VEND-01**: `harbor.models.trajectories.*` source vendored into `daydream/atif/models/` (Apache-2.0 LICENSE + NOTICE included)
- [ ] **VEND-02**: `harbor.utils.trajectory_validator` source vendored into `daydream/atif/validator.py` with no external Harbor imports
- [ ] **VEND-03**: `pydantic>=2.11.7` promoted from transitive (via `claude-agent-sdk`) to explicit `[project.dependencies]` entry in `pyproject.toml`
- [ ] **VEND-04**: Vendored modules import only from stdlib + `pydantic`; no remaining `from harbor import …` references anywhere in the daydream source tree
- [ ] **VEND-05**: Harbor's golden trajectory fixtures (Terminus-2 + OpenHands `*.trajectory.json`) vendored into `tests/fixtures/atif_golden/`

### Recorder Core (CORE)

- [ ] **CORE-01**: New `daydream/trajectory.py` module exposes `TrajectoryRecorder`, `Invocation`, and `Redactor` classes; no ATIF model construction lives outside this module
- [ ] **CORE-02**: `TrajectoryRecorder` instance is propagated via a `ContextVar` defined in `daydream/trajectory.py` (NOT on `AgentState`)
- [ ] **CORE-03**: Recorder has `__aenter__` / `__aexit__` lifecycle that writes the trajectory JSON file on clean exit
- [ ] **CORE-04**: Recorder maintains a monotonic `step_id` counter starting at 1 (per ATIF schema requirement)
- [ ] **CORE-05**: Recorder buffers `TextEvent` chunks into a single `Step.message`, flushing on tool call / thinking / result boundaries
- [ ] **CORE-06**: Recorder maintains an in-flight `tool_call_id → Step` map so `ObservationResult.source_call_id` always lands in the same Step as its `ToolCall`
- [ ] **CORE-07**: Recorder generates per-run `session_id` (UUID4) at run start; same value used by root and all sibling trajectories
- [ ] **CORE-08**: Recorder captures `Agent(name="daydream", version=<package version>, model_name=<per-backend>)` once at trajectory init
- [ ] **CORE-09**: Recorder failure (e.g., disk write error) does NOT crash the user's review/fix run when the recorder is implicit; explicit `--trajectory <path>` elevates write failure to error+exit
- [ ] **CORE-10**: Test isolation: autouse `_reset_trajectory_recorder` fixture in `tests/conftest.py` mirrors the existing `reset_state()` pattern

### Event Enrichment (EVNT)

- [ ] **EVNT-01**: All `AgentEvent` dataclasses in `daydream/backends/__init__.py` carry an ISO 8601 UTC `timestamp: str` field generated via `datetime.now(timezone.utc).isoformat()`
- [ ] **EVNT-02**: New `MetricsEvent` dataclass added to `daydream/backends/__init__.py` (fields: `message_id: str`, `prompt_tokens: int`, `completion_tokens: int`, `cached_tokens: int | None`, `cost_usd: float | None`, `timestamp: str`)
- [ ] **EVNT-03**: `CostEvent` extended with `cached_tokens: int | None`; remains end-of-call signal for `FinalMetrics` aggregation
- [ ] **EVNT-04**: Claude backend (`backends/claude.py:120-128`) populates `input_tokens` / `output_tokens` from `ResultMessage.usage["input_tokens"]` and `["output_tokens"]` (currently `None`; data is already on `msg.usage`)
- [ ] **EVNT-05**: Claude backend populates `cached_tokens` from `ResultMessage.usage["cache_read_input_tokens"]` (cached_tokens = subset of prompt_tokens, per ATIF spec)
- [ ] **EVNT-06**: Claude backend emits `MetricsEvent` per `AssistantMessage.message_id` from `AssistantMessage.usage`
- [ ] **EVNT-07**: Codex backend emits `MetricsEvent` from `turn.completed.usage` (`input_tokens` and `output_tokens` only; `cost_usd` and `cached_tokens` set to `None` — acceptable, ATIF Metrics fields are optional)

### Event-to-ATIF Mapping (MAP)

- [ ] **MAP-01**: Each `run_agent()` invocation begins with a `Step(source="user", message=<prompt>)` (Beagle skill prompt has `source="user"` per ATIF; `"system"` reserved for system-prompt preambles)
- [ ] **MAP-02**: `TextEvent` → `Step(source="agent", message=<accumulated text>)`; consecutive chunks coalesce
- [ ] **MAP-03**: `ThinkingEvent` → `Step(source="agent", reasoning_content=<text>)` on the same step as accompanying text/tool calls
- [ ] **MAP-04**: `ToolStartEvent` → `ToolCall(tool_call_id=<sdk id>, function_name=<name>, arguments=<input>)` attached to the active agent step
- [ ] **MAP-05**: `ToolResultEvent` → `ObservationResult(source_call_id=<id>, content=<output>)` correlated via tool ID and emitted on the SAME step as its matching `ToolCall` (validator scopes `tool_call_id` intra-step)
- [ ] **MAP-06**: `MetricsEvent` → per-step `Metrics(prompt_tokens, completion_tokens, cached_tokens, cost_usd)` attached to the agent step matching its `message_id`
- [ ] **MAP-07**: `ResultEvent` → trajectory-level `FinalMetrics(total_prompt_tokens, total_completion_tokens, total_cached_tokens, total_cost_usd, total_steps)` aggregated from per-step values
- [ ] **MAP-08**: Each `Step` carries an `extra.daydream_phase` label (one of: `"review"`, `"parse"`, `"fix"`, `"test"`, `"intent"`, `"alternatives"`, `"plan"`, `"pr_feedback"`, `"deep"`, `"exploration"`)
- [ ] **MAP-09**: Each `Step` carries `extra.daydream_run_flow` label (one of: `"normal"`, `"ttt"`, `"pr"`, `"deep"`)

### Subagent Wiring (SUBA)

- [ ] **SUBA-01**: Sequential phase chain (`phase_review` → `phase_parse_feedback` → `phase_fix` → `phase_test_and_heal`) emits as continuous steps in ONE root trajectory file
- [ ] **SUBA-02**: `phase_fix_parallel` (parallel `anyio` task group) emits one sibling trajectory file per parallel fix invocation; root step's `ObservationResult.subagent_trajectory_ref` points to each sibling
- [ ] **SUBA-03**: `daydream/deep/orchestrator.run_deep()` per-stack fan-out emits one sibling trajectory file per stack; root step linked via `subagent_trajectory_ref`
- [ ] **SUBA-04**: `daydream/exploration_runner.pre_scan()` specialist subagents emit sibling trajectory files; parent's pre-scan step linked via `subagent_trajectory_ref`
- [ ] **SUBA-05**: `run_agent_with_continuation` continuation calls APPEND to the same trajectory (preserving agent identity); they do NOT spawn a sibling
- [ ] **SUBA-06**: Sibling trajectory files inherit the parent `session_id` and are written to `<root_dir>/.daydream/trajectories/<session_id>.<descriptor>.json`
- [ ] **SUBA-07**: ContextVar copy-on-spawn (anyio task-local propagation) correctly establishes parent → child recorder relationship without any explicit threading through phase signatures
- [ ] **SUBA-08**: Parallel-task `step_id` counters are isolated per trajectory file (no collisions across siblings)
- [ ] **SUBA-09**: Parent `FinalMetrics` aggregates ONLY parent-trajectory steps; sibling totals stay in their own file (no double-counting)

### Privacy & Redaction (REDA)

- [ ] **REDA-01**: `Redactor` policy implemented in `daydream/trajectory.py` covering: API key patterns (`sk-*`, `ghp_*`, `xoxb-*`, `AKIA*`), JWT tokens, git remote URLs with embedded credentials
- [ ] **REDA-02**: Redactor scrubs file paths containing usernames (`/Users/<name>/`, `/home/<name>/`, `C:\\Users\\<name>\\`)
- [ ] **REDA-03**: Redactor scrubs `.env`-style key=value lines from tool I/O content
- [ ] **REDA-04**: Redaction applied to `ToolCall.arguments`, `ObservationResult.content`, `Step.message`, AND `Step.reasoning_content`
- [ ] **REDA-05**: Redaction failure mode: redact-or-omit; never raw-pass-through
- [ ] **REDA-06**: Redaction unit tests cover each pattern category with realistic positive and negative examples

### Legacy Removal & Cutover (CUT)

- [ ] **CUT-01**: `_log_debug()` and all 15+ call sites in `daydream/agent.py` removed
- [ ] **CUT-02**: `AgentState.debug_log` field, `set_debug_log()`, `get_debug_log()` removed
- [ ] **CUT-03**: Debug file initialization in `daydream/runner.py:460-472` (`.review-debug-{ts}.log`) removed
- [ ] **CUT-04**: All phase-level prefix-tagged log lines removed from `daydream/phases.py` (`[REVERT]`, `[PARSE_FAIL]`, `[STAGE]`, `[TTT_REVIEW]`, `[TTT_PLAN]`, etc.)
- [ ] **CUT-05**: All Codex-specific prefix-tagged log lines removed from `daydream/backends/codex.py` (`[CODEX_RAW]`, `[CODEX_WARN]`, `[CODEX_UNHANDLED]`)
- [ ] **CUT-06**: Lazy import `from daydream.agent import _log_debug` inside a function in `daydream/backends/codex.py:37` removed (AST-level sweep, not just grep)
- [ ] **CUT-07**: `[PRE_SCAN]` exploration logging removed from `daydream/exploration_runner.py:189-278`
- [ ] **CUT-08**: AST sweep verifies no remaining references to `_log_debug`, `debug_log`, `set_debug_log`, `get_debug_log` anywhere in `daydream/` or `tests/`

### CLI Surface (CLI)

- [ ] **CLI-01**: `--debug` CLI flag removed from `daydream/cli.py`
- [ ] **CLI-02**: `--trajectory <path>` CLI flag added; default = `<target_dir>/.daydream/trajectory.json`
- [ ] **CLI-03**: SIGINT (Ctrl-C) and SIGTERM handlers flush partial trajectory to `<path>.partial` before exit
- [ ] **CLI-04**: `--ttt`, `--pr`, `--deep`, and `--review-only` flags continue to work and produce trajectories
- [ ] **CLI-05**: Help text updated to describe trajectory output and flag semantics

### Test Suite (TEST)

- [ ] **TEST-01**: All 343 existing tests pass post-migration (no regressions)
- [ ] **TEST-02**: New `tests/test_trajectory.py` covers `TrajectoryRecorder` lifecycle, step coalescing, tool-call correlation, and validator round-trip
- [ ] **TEST-03**: New `tests/test_redaction.py` covers each `Redactor` pattern category with positive and negative cases
- [ ] **TEST-04**: New `tests/test_atif_models.py` parametrizes over Harbor's vendored golden fixtures and confirms our vendored validator accepts them all
- [ ] **TEST-05**: Trajectory tests follow schema-validity + behavior-predicate pattern, NOT full-tree snapshot equality (avoids brittle locks on schema details)
- [ ] **TEST-06**: Empirical multi-turn fixture test verifies `ResultMessage.usage["input_tokens"]` is per-call (not cumulative) for `claude-agent-sdk==0.1.52` (SDK issue #112 risk gate)
- [ ] **TEST-07**: Subagent test covers `phase_fix_parallel`, deep mode, and exploration pre-scan producing valid root + sibling trajectory file sets

### Documentation (DOCS)

- [ ] **DOCS-01**: README documents trajectory output format, default path, and `--trajectory` flag semantics
- [ ] **DOCS-02**: README explains redaction policy and what users should NOT expect to see scrubbed
- [ ] **DOCS-03**: README points consumers at Harbor / replay viewers / SFT-RL integration paths
- [ ] **DOCS-04**: CHANGELOG entry covers the breaking CLI change (`--debug` removed, `--trajectory <path>` added)
- [ ] **DOCS-05**: `daydream/atif/NOTICE` documents Apache-2.0 attribution for vendored Harbor code
- [ ] **DOCS-06**: `CLAUDE.md` updated to mention the trajectory format and `daydream/trajectory.py` location

## v2 Requirements

Deferred to a future milestone. Tracked but not in current roadmap.

### Streaming & Performance

- **PERF-01**: Mid-run streaming trajectory writes (currently batch at run completion)
- **PERF-02**: Memory ceiling guard for very long runs (graceful flush + truncate if buffer exceeds N MB)

### Tooling

- **TOOL-01**: `daydream replay <trajectory.json>` subcommand for trajectory playback
- **TOOL-02**: `daydream stats <trajectory.json>` subcommand for cost/token analytics

### Multimodal

- **MM-01**: Support for ATIF v1.6 `ContentPart` / `ImageSource` if daydream agents start emitting screenshots / images

## Out of Scope

| Feature | Reason |
|---------|--------|
| Backward-compatibility shim for `.review-debug-*.log` | Hard cutover; no known external consumer; preserving the prefix-tagged format dilutes the migration's value |
| Dual-write phase (both old log and new trajectory) | Less churn / faster cutover; existing 343-test suite is the regression gate |
| Mid-run streaming writes | ATIF expects a single coherent document; mid-run writes complicate atomic correctness for marginal observability gain |
| Trajectory upload / external delivery | Daydream writes locally only; shipping to Harbor / S3 / viewers is a consumer concern |
| ATIF schema versions other than v1.6 (emission) | Validator accepts v1.0–v1.6 by Literal; emission pinned to v1.6 only |
| Repurposing `--debug` for UI verbosity | CLI minimalism; users wanting verbose output read the trajectory |
| Recording subprocess calls (`gh`, `git`, tree-sitter parses) | ATIF models LLM-agent interactions; subprocess noise belongs elsewhere |
| `harbor` as a runtime dependency | 21+ transitive deps + `litellm` supply-chain quarantine; vendoring the ~700 LOC submodule is cleaner |
| Multimodal `ContentPart` emission | Daydream emits text only; v1.6's `str` form of `Step.message` is valid for our use case |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| VEND-01 | Phase 1 | Pending |
| VEND-02 | Phase 1 | Pending |
| VEND-03 | Phase 1 | Pending |
| VEND-04 | Phase 1 | Pending |
| VEND-05 | Phase 1 | Pending |
| CORE-01 | Phase 2 | Pending |
| CORE-02 | Phase 2 | Pending |
| CORE-03 | Phase 2 | Pending |
| CORE-04 | Phase 2 | Pending |
| CORE-05 | Phase 2 | Pending |
| CORE-06 | Phase 2 | Pending |
| CORE-07 | Phase 2 | Pending |
| CORE-08 | Phase 2 | Pending |
| CORE-09 | Phase 2 | Pending |
| CORE-10 | Phase 2 | Pending |
| EVNT-01 | Phase 2 | Pending |
| EVNT-02 | Phase 2 | Pending |
| EVNT-03 | Phase 2 | Pending |
| EVNT-04 | Phase 2 | Pending |
| EVNT-05 | Phase 2 | Pending |
| EVNT-06 | Phase 2 | Pending |
| EVNT-07 | Phase 2 | Pending |
| MAP-01 | Phase 2 | Pending |
| MAP-02 | Phase 2 | Pending |
| MAP-03 | Phase 2 | Pending |
| MAP-04 | Phase 2 | Pending |
| MAP-05 | Phase 2 | Pending |
| MAP-06 | Phase 2 | Pending |
| MAP-07 | Phase 2 | Pending |
| MAP-08 | Phase 2 | Pending |
| MAP-09 | Phase 2 | Pending |
| SUBA-01 | Phase 3 | Pending |
| SUBA-02 | Phase 3 | Pending |
| SUBA-03 | Phase 3 | Pending |
| SUBA-04 | Phase 3 | Pending |
| SUBA-05 | Phase 3 | Pending |
| SUBA-06 | Phase 3 | Pending |
| SUBA-07 | Phase 3 | Pending |
| SUBA-08 | Phase 3 | Pending |
| SUBA-09 | Phase 3 | Pending |
| REDA-01 | Phase 4 | Pending |
| REDA-02 | Phase 4 | Pending |
| REDA-03 | Phase 4 | Pending |
| REDA-04 | Phase 4 | Pending |
| REDA-05 | Phase 4 | Pending |
| REDA-06 | Phase 4 | Pending |
| CUT-01 | Phase 4 | Pending |
| CUT-02 | Phase 4 | Pending |
| CUT-03 | Phase 4 | Pending |
| CUT-04 | Phase 4 | Pending |
| CUT-05 | Phase 4 | Pending |
| CUT-06 | Phase 4 | Pending |
| CUT-07 | Phase 4 | Pending |
| CUT-08 | Phase 4 | Pending |
| CLI-01 | Phase 4 | Pending |
| CLI-02 | Phase 4 | Pending |
| CLI-03 | Phase 4 | Pending |
| CLI-04 | Phase 4 | Pending |
| CLI-05 | Phase 4 | Pending |
| TEST-01 | Phase 5 | Pending |
| TEST-02 | Phase 5 | Pending |
| TEST-03 | Phase 5 | Pending |
| TEST-04 | Phase 5 | Pending |
| TEST-05 | Phase 5 | Pending |
| TEST-06 | Phase 5 | Pending |
| TEST-07 | Phase 5 | Pending |
| DOCS-01 | Phase 5 | Pending |
| DOCS-02 | Phase 5 | Pending |
| DOCS-03 | Phase 5 | Pending |
| DOCS-04 | Phase 5 | Pending |
| DOCS-05 | Phase 5 | Pending |
| DOCS-06 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 72 total (5 VEND + 10 CORE + 7 EVNT + 9 MAP + 9 SUBA + 6 REDA + 8 CUT + 5 CLI + 7 TEST + 6 DOCS)
- Mapped to phases: 72 (100%) ✓
- Unmapped: 0

**Phase distribution:**
- Phase 1 (Vendor ATIF Foundation): 5 reqs (VEND)
- Phase 2 (Recorder Core + Event Enrichment + Mapping): 26 reqs (CORE + EVNT + MAP)
- Phase 3 (Subagent Wiring): 9 reqs (SUBA)
- Phase 4 (Cutover + Redaction + CLI Surface): 19 reqs (REDA + CUT + CLI)
- Phase 5 (Test Hardening + Documentation): 13 reqs (TEST + DOCS)

---
*Requirements defined: 2026-04-26*
*Traceability mapped: 2026-04-26 by roadmapper agent*
