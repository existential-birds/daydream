# Daydream

## What This Is

Daydream is a Python CLI that automates code review and fix loops using the Claude Agent SDK. It launches review agents equipped with Beagle skills to review code, parse actionable feedback, apply fixes automatically, and validate by running tests — with secondary flows for stack-agnostic PR review (`--ttt`), GitHub PR feedback ingestion (`--pr`), and multi-stack deep review (`--deep`). Today it logs each run as unstructured prefix-tagged lines (`[TEXT]`, `[TOOL_USE]`, `[COST]`, …) into a flat `.review-debug-{ts}.log`. The current milestone replaces that custom debug logging with the **Agent Trajectory Interchange Format (ATIF v1.6)** so daydream's runs become first-class, machine-parseable trajectories interoperable with the Harbor ecosystem.

## Core Value

Reviews and recommendations must be grounded in actual codebase understanding — not guesses based on the diff alone. (Unchanged.) For this milestone specifically: **every daydream run produces a valid, replayable ATIF v1.6 trajectory that captures the full agent interaction history, tool I/O, and token/cost metrics.**

## Requirements

### Validated

<!-- Inferred from existing codebase as of 2026-04-26 -->

- ✓ **Review skill invocation** — `phase_review()` invokes Beagle review skills (`beagle-python:review-python`, `beagle-react:review-frontend`, `beagle-elixir:review-elixir`) and writes to `.review-output.md` — existing
- ✓ **Structured feedback parsing** — `phase_parse_feedback()` extracts actionable issues as JSON via `FEEDBACK_SCHEMA` — existing
- ✓ **Sequential and parallel fix application** — `phase_fix()` and `phase_fix_parallel()` apply fixes one-by-one or via `anyio` task groups — existing
- ✓ **Test-and-heal loop** — `phase_test_and_heal()` runs tests, retries failures, reverts uncommitted work on failure — existing
- ✓ **Backend abstraction** — `Backend` Protocol with `ClaudeBackend` (claude-agent-sdk) and `CodexBackend` (codex CLI) implementations — existing
- ✓ **Unified `AgentEvent` stream** — `TextEvent`, `ThinkingEvent`, `ToolStartEvent`, `ToolResultEvent`, `CostEvent`, `ResultEvent` consumed by `run_agent()` — existing
- ✓ **Trust-the-technology PR review** — `run_trust()` understands intent, evaluates alternatives, generates implementation plans — existing
- ✓ **GitHub PR feedback ingestion** — `run_pr_feedback()` fetches PR review comments and applies fixes — existing
- ✓ **Deep multi-stack review** — `daydream/deep/orchestrator.py` runs parallel per-stack reviews and merges them — existing
- ✓ **Tree-sitter import resolution** — `daydream/tree_sitter_index.py` resolves imports for Python/TypeScript/Go/Rust — existing
- ✓ **Rich terminal UI** — Live panels, throbbers, Dracula theme via `daydream/ui.py` — existing
- ✓ **Test coverage** — 343 tests covering CLI, runner, phases, backends, deep mode, tree-sitter, exploration — existing

### Active

<!-- Current scope: ATIF migration. -->

**Trajectory recording (root + sibling files):**

- [x] Every daydream run produces a root `.trajectory.json` capturing sequential phases (review → parse → fix → test) as continuous steps in one trajectory
- [x] Parallel `anyio` task groups emit **sibling trajectory files** linked from the parent via `ObservationResult.subagent_trajectory_ref`. Patterns covered: `phase_fix_parallel`, `daydream/deep/orchestrator.run_deep`, `exploration_runner.pre_scan`
- [x] Continuation flows (`run_agent_with_continuation`) stay in the same trajectory as continuous steps (preserves agent identity across continuation tokens)
- [x] Trajectory is always written (no `--debug` opt-in); root output path controllable via `--trajectory <path>`; sibling files land beside it under `.daydream/trajectories/<id>.json`

**Event-to-ATIF mapping:**

- [x] `[PROMPT]` log → first user step per `run_agent()` invocation (Beagle skill prompt has `source="user"`, not `"system"` — ATIF reserves `"system"` for system-prompt preambles)
- [x] `TextEvent` → agent step `message`; consecutive chunks accumulate into one step (flush on tool call, thinking, or result)
- [x] `ThinkingEvent` → agent step `reasoning_content`
- [x] `ToolStartEvent` → `ToolCall(tool_call_id, function_name, arguments)` attached to parent agent step
- [x] `ToolResultEvent` → `ObservationResult(source_call_id, content)` correlated via tool ID — must land in the **same step** as its `ToolCall` (validator scopes `tool_call_id` intra-step)
- [x] New `MetricsEvent(message_id, prompt_tokens, completion_tokens, cached_tokens, cost_usd)` → per-step `Metrics`, fired per `AssistantMessage.message_id`
- [x] `CostEvent` extended with `cached_tokens: int | None`; remains end-of-call signal for trajectory-level `FinalMetrics`
- [x] `ResultEvent` → final agent step + trajectory-level `FinalMetrics(total_prompt_tokens, total_completion_tokens, total_cached_tokens, total_cost_usd, total_steps)`
- [x] Each daydream phase labelled in `Step.extra.daydream_phase` (`"review"`, `"parse"`, `"fix"`, `"test"`, `"intent"`, `"alternatives"`, `"plan"`, `"pr_feedback"`, `"deep"`, `"exploration"`)

**Event enrichment:**

- [x] All event dataclasses in `daydream/backends/__init__.py` carry an ISO 8601 UTC `timestamp` field (`datetime.now(timezone.utc).isoformat()`)
- [x] Claude backend (`backends/claude.py:120-128`) populates `input_tokens` / `output_tokens` / `cached_tokens` from `ResultMessage.usage` (data is already on `msg.usage` dict; currently dropped on the floor)
- [x] Claude backend emits `MetricsEvent` per `AssistantMessage.message_id` from `AssistantMessage.usage`
- [x] Codex backend emits `MetricsEvent` from `turn.completed.usage` (`input_tokens` and `output_tokens` only; `cost_usd` and `cached_tokens` set to `None` — acceptable, ATIF Metrics fields are optional)
- [x] Per-run `session_id` (UUID4) generated at run start; propagated through all `run_agent()` calls and inherited by sibling trajectories
- [x] Agent identity captured: `Agent(name="daydream", version=<package version>, model_name=<per-backend>)`

**Schema validation (vendored, not Harbor as runtime dep):**

- [x] `harbor.models.trajectories.*` and `harbor.utils.trajectory_validator` vendored into `daydream/atif/` (Apache-2.0; ~700 LOC pure Pydantic + stdlib; LICENSE + NOTICE included) — Validated in Phase 1: vendor-atif-foundation
- [x] `pydantic>=2.11.7` promoted from transitive (via claude-agent-sdk) to explicit dep in `pyproject.toml` — Validated in Phase 1: vendor-atif-foundation
- [x] Trajectories built using vendored Pydantic models; schema validation automatic at construction time
- [x] Test suite validates produced trajectories against the ATIF v1.6 schema
- [x] Harbor's golden trajectory fixtures (Terminus-2 + OpenHands) vendored into `tests/fixtures/atif_golden/` and parametrized round-trip test confirms our validator accepts them — Validated in Phase 1: vendor-atif-foundation (smoke test in `tests/test_atif_vendor_smoke.py`)

**Privacy / redaction (must land with cutover, not deferred):**

- [x] `Redactor` policy implemented in `daydream/trajectory.py` covering: API key patterns (`sk-*`, `ghp_*`, `xoxb-*`, `AKIA*`), JWT tokens, git remote URLs with credentials, file paths containing usernames (`/Users/<name>/`, `/home/<name>/`), and `.env`-style key=value lines
- [x] Redaction applied to `ToolCall.arguments`, `ObservationResult.content`, `Step.message`, and `Step.reasoning_content`
- [x] Redaction failure mode: redact-or-omit, never raw-pass-through

**Legacy removal (hard cutover, no dual-write):**

- [x] `_log_debug()` and all 15+ call sites in `agent.py` removed
- [x] `AgentState.debug_log` field, `set_debug_log()`, `get_debug_log()` removed
- [x] Debug file initialization in `runner.py` (`.review-debug-{ts}.log`) removed
- [x] All phase-level `[REVERT]`, `[PARSE_FAIL]`, `[STAGE]`, `[TTT_*]` log lines removed from `phases.py`
- [x] All Codex-specific `[CODEX_RAW]`, `[CODEX_WARN]`, `[CODEX_UNHANDLED]` log lines removed from `backends/codex.py`
- [x] **Sneaky lazy-import**: `from daydream.agent import _log_debug` *inside a function* in `daydream/backends/codex.py:37` removed (grep would miss this; AST sweep required)
- [x] `[PRE_SCAN]` exploration logging removed from `exploration_runner.py`
- [x] `--debug` CLI flag removed; SIGINT handler flushes partial trajectory to `.trajectory.partial.json`
- [x] All existing 343 tests pass post-migration

**Documentation:**

- [x] README documents trajectory format, output path, redaction policy, and consumer integration (Harbor, viewers, training pipelines)
- [x] CHANGELOG entry covers the breaking CLI change (`--debug` removed, `--trajectory <path>` added)

### Out of Scope

- **Backward-compatibility shim for `.review-debug-*.log`** — hard cutover; prefix-tagged log format is gone in one release. Rationale: no known external consumer; preserving it adds churn and dilutes the migration's value.
- **Streaming trajectory writes (mid-run)** — trajectory is built in memory and written at run completion. Rationale: ATIF expects a single coherent document; mid-run writes complicate atomic correctness for marginal observability gain. Revisit only if a long-running flow needs it.
- **Dual-write phase** — Phase 1 will not keep `_log_debug` running alongside the new recorder. Rationale: less churn, fewer commits, faster path to value; existing test suite catches regressions.
- **Trajectory upload / external delivery** — daydream writes `.trajectory.json` locally only. Rationale: shipping to Harbor / S3 / viewers is the consumer's job, not daydream's.
- **ATIF schema versions other than v1.6** — pinned to v1.6 (Harbor's current default; matches all golden fixtures). Rationale: supporting older versions is unnecessary complexity; the validator accepts v1.0–v1.6 by `Literal` field but emission is v1.6 only.
- **Harbor as a runtime dependency** — vendored only. Rationale: Harbor 0.5.0 pulls 21+ transitive packages (~150-250 MB) including `litellm` (which had a March 2026 PyPI quarantine for malicious `.pth` exfiltrating SSH/cloud tokens — Harbor's `litellm>=1.80.8` bound was never tightened). The trajectory submodule itself is ~700 LOC pure Pydantic + stdlib, Apache-2.0 — vendoring is cleaner than living with the supply-chain exposure.
- **Multimodal content (`ContentPart`/`ImageSource`) added in ATIF v1.6** — daydream emits text only, no image steps. Rationale: no current use case; emitting `Step.message` as `str` is valid in v1.6.
- **Repurposing `--debug` for UI verbosity** — the flag is removed, not reused. Rationale: keeps the CLI surface minimal; users who want verbose console output can read the trajectory file.
- **Trajectories from MCP/non-agent subprocess calls** (`gh`, `git`, tree-sitter parses) — only `run_agent()` invocations are recorded. Rationale: ATIF models LLM-agent interactions; subprocess noise belongs elsewhere.

## Context

**Existing project state:**
- Brownfield Python 3.12 CLI; ~10 modules in `daydream/` package, 343 tests in `tests/`, ruff + mypy clean, CI green on push
- 3 distinct run flows in `runner.py`: `run()` (normal review→fix→test), `run_pr_feedback()` (`--pr`), `run_trust()` (`--ttt`); plus `daydream/deep/orchestrator.py:run_deep()` for multi-stack review
- All agent interactions go through `daydream/agent.py:run_agent()` which consumes the unified `AgentEvent` stream from a `Backend` Protocol implementation
- Module-level singleton `AgentState` in `agent.py` carries cross-cutting state (`debug_log`, `quiet_mode`, `model`, `shutdown_requested`, `current_backends`)
- `daydream/ui.py` is 3470 lines (single-module bottleneck for any UI change); `daydream/phases.py` is 1552 lines

**Why ATIF, why now:**
- Current `[PREFIX] line` log format is unparseable without regex, has no per-event timestamps, no session correlation, and `CostEvent.input_tokens`/`output_tokens` are always `None` from the Claude backend (data is available in `ResultMessage`, just not extracted)
- ATIF v1.4 is the standard format used by Claude Code, OpenHands, Codex, Gemini CLI, Mini-SWE-Agent, and Terminus-2 — adopting it makes daydream trajectories consumable by Harbor's validator, replay tooling, SFT/RL training pipelines, and any cross-tool trajectory viewer
- Harbor publishes Pydantic models (`harbor.models.trajectories`) that give us type-safe construction and automatic validation, removing the need to vendor a JSON Schema or hand-roll dataclasses

**Reference docs:**
- ATIF spec: `docs/reference/atif_format.md` (this repo)
- Source RFC: https://github.com/laude-institute/harbor/blob/main/docs/rfcs/0001-trajectory-format.md
- Codebase map: `.planning/codebase/` (ARCHITECTURE.md, STACK.md, STRUCTURE.md, CONCERNS.md, TESTING.md, INTEGRATIONS.md, CONVENTIONS.md)

## Constraints

- **SDK**: Must continue using `claude-agent-sdk` for agent capabilities — no custom orchestration framework. ATIF is layered on top of the existing `Backend` / `AgentEvent` abstraction.
- **Backends**: Both Claude and Codex backends must produce ATIF trajectories. Codex parity is partial by design (no `cost_usd`, no `cached_tokens` from `turn.completed.usage`) — ATIF Metrics fields are all optional, so this is acceptable.
- **Existing tests**: All 343 tests must pass post-migration. New tests added for trajectory recording, redaction, and Harbor-golden round-trip validation.
- **CLI**: `--debug` is removed; `--trajectory <path>` is added. The `--ttt`, `--pr`, and `--deep` flags continue to work and produce trajectories.
- **Dependencies**: No `harbor` runtime dep. Vendored ATIF code under `daydream/atif/` (Apache-2.0). `pydantic>=2.11.7` promoted to explicit dep (already transitive via `claude-agent-sdk`).
- **Schema version**: Pinned to ATIF-v1.6 (emission). No multi-version emission support.
- **Recorder placement**: `TrajectoryRecorder` lives in new `daydream/trajectory.py` module, propagated via `ContextVar` (not `AgentState`). Test isolation via autouse `_reset_trajectory_recorder` fixture in `conftest.py` mirroring the existing `reset_state()` pattern.
- **Module-bloat ban**: No `Step()`, `ToolCall()`, or `Trajectory()` construction inside `phases.py` or `ui.py` — all ATIF model construction stays in `daydream/trajectory.py`.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| **Trajectory granularity**: root file for sequential phases + sibling files for parallel `anyio` task groups, linked via `ObservationResult.subagent_trajectory_ref` | Matches ATIF v1.4+ subagent semantics (separate files, not nested `Trajectory`); preserves correct execution-graph representation across `phase_fix_parallel` / deep / exploration; FinalMetrics aggregation stays clean per-trajectory | — Pending |
| **Validation library**: vendor `harbor.models.trajectories` + `harbor.utils.trajectory_validator` into `daydream/atif/` (Apache-2.0, ~700 LOC pure Pydantic) | Avoids Harbor's 21+ transitive deps including the litellm supply-chain quarantine; preserves automatic Pydantic validation; promotes `pydantic>=2.11.7` to explicit dep | — Pending |
| **`--debug` flag removed; trajectory always written**; output path via `--trajectory <path>` | Trajectory recording is low cost / high value; always-on enables training-data collection from normal runs; removes a duplicated concern from the CLI | — Pending |
| **Hard cutover, no dual-write phase** | Less churn, fewer commits; existing 343-test suite is sufficient regression coverage; no known external consumer of `.review-debug-*.log` | — Pending |
| **ATIF schema version pinned to v1.6** (was v1.4 in original PRD) | Harbor's current default; all golden fixtures (Terminus-2 v1.6, OpenHands v1.5) are at v1.5+; pinning to v1.4 would make "validate against goldens" parametrized test meaningless. v1.6's multimodal `ContentPart`/`ImageSource` additions are backward-compatible (str form remains valid) | — Pending |
| **Recorder placement: ContextVar in `daydream/trajectory.py`, not `AgentState`** | Per-run lifecycle, not process-singleton; `ContextVar` copy-on-spawn handles `anyio` parallel task groups automatically; clean test isolation; mirrors anyio's task-local propagation | — Pending |
| **New `MetricsEvent` keyed by `AssistantMessage.message_id`** (replaces single end-of-call CostEvent for per-step Metrics); CostEvent extended with `cached_tokens`; Claude backend token-extraction bug fixed in same phase | Single end-of-call CostEvent is too coarse for ATIF's per-step Metrics; data is already available in `ResultMessage.usage` and `AssistantMessage.usage`; fix lives where the data is read | — Pending |
| **Beagle skill prompt = `source="user"` (not `"system"`)** | ATIF reserves `"system"` for system-prompt preambles; setting agent-only fields on user/system steps is a hard validator failure | — Pending |
| **Redaction must land with cutover, not deferred** | Always-on trajectories + bypass-permissions tool surface means tool args / file reads / reasoning content can carry API keys, credentials, user paths; current debug log is local-only and opt-in but trajectories are intended for sharing with Harbor / training pipelines | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-29 after milestone completion (all 5 phases complete, 72/72 requirements verified). All Active requirements checked off.*
