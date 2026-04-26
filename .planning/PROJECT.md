# Daydream

## What This Is

Daydream is a Python CLI that automates code review and fix loops using the Claude Agent SDK. It launches review agents equipped with Beagle skills to review code, parse actionable feedback, apply fixes automatically, and validate by running tests — with secondary flows for stack-agnostic PR review (`--ttt`), GitHub PR feedback ingestion (`--pr`), and multi-stack deep review (`--deep`). Today it logs each run as unstructured prefix-tagged lines (`[TEXT]`, `[TOOL_USE]`, `[COST]`, …) into a flat `.review-debug-{ts}.log`. The current milestone replaces that custom debug logging with the **Agent Trajectory Interchange Format (ATIF)** so daydream's runs become first-class, machine-parseable trajectories interoperable with the Harbor ecosystem.

## Core Value

Reviews and recommendations must be grounded in actual codebase understanding — not guesses based on the diff alone. (Unchanged.) For this milestone specifically: **every daydream run produces a valid, replayable ATIF v1.4 trajectory that captures the full agent interaction history, tool I/O, and token/cost metrics.**

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

**Trajectory recording:**

- [ ] Every daydream run produces a single ATIF v1.4-compliant `.trajectory.json` capturing the entire orchestration (review → parse → fix → test, plus deep/PR/TTT flows)
- [ ] `run_agent()` invocations within a run are nested as hierarchical steps using ATIF's subagent delegation model
- [ ] Trajectory is always written (no `--debug` opt-in); output path controllable via `--trajectory <path>`

**Event-to-ATIF mapping:**

- [ ] `[PROMPT]` log → first user step per `run_agent()` call
- [ ] `TextEvent` → agent step `message`; consecutive chunks accumulate into one step
- [ ] `ThinkingEvent` → agent step `reasoning_content`
- [ ] `ToolStartEvent` → `ToolCall(tool_call_id, function_name, arguments)` attached to parent agent step
- [ ] `ToolResultEvent` → `ObservationResult(source_call_id, content)` correlated via tool ID
- [ ] `CostEvent` → per-step `Metrics(prompt_tokens, completion_tokens, cached_tokens, cost_usd)`
- [ ] `ResultEvent` → final agent step + trajectory-level `FinalMetrics`

**Event enrichment:**

- [ ] All event dataclasses in `daydream/backends/__init__.py` carry an ISO 8601 `timestamp` field
- [ ] Claude backend extracts `prompt_tokens` / `completion_tokens` / `cached_tokens` from SDK `ResultMessage` (currently always `None`)
- [ ] Codex backend produces equivalent token coverage from its CLI stream
- [ ] Per-run `session_id` (UUID) generated at run start and propagated through all `run_agent()` calls
- [ ] Agent identity captured: `name="daydream"`, `version` from package, `model_name` per backend

**Schema validation:**

- [ ] Trajectories built using Harbor's Pydantic models (`harbor.models.trajectories.{Trajectory,Agent,Step,ToolCall,Observation,ObservationResult,Metrics,FinalMetrics}`)
- [ ] Test suite validates produced trajectories against the ATIF schema
- [ ] Multi-turn / continuation flows produce coherent trajectories

**Legacy removal (hard cutover, no dual-write):**

- [ ] `_log_debug()` and all 15+ call sites in `agent.py` removed
- [ ] `AgentState.debug_log`, `set_debug_log()`, `get_debug_log()` removed
- [ ] Debug file initialization in `runner.py` (`.review-debug-{ts}.log`) removed
- [ ] All phase-level `[REVERT]`, `[PARSE_FAIL]`, `[STAGE]`, `[TTT_*]` log lines removed
- [ ] All Codex-specific `[CODEX_RAW]`, `[CODEX_WARN]`, `[CODEX_UNHANDLED]` log lines removed
- [ ] `[PRE_SCAN]` exploration logging removed from `exploration_runner.py`
- [ ] `--debug` CLI flag removed
- [ ] All existing tests pass (no regressions in 343-test suite)

**Documentation:**

- [ ] README documents trajectory format, output path, and consumer integration (Harbor, viewers, training pipelines)

### Out of Scope

- **Backward-compatibility shim for `.review-debug-*.log`** — hard cutover; prefix-tagged log format is gone in one release. Rationale: no known external consumer; preserving it adds churn and dilutes the migration's value.
- **Streaming trajectory writes (mid-run)** — trajectory is built in memory and written at run completion. Rationale: ATIF expects a single coherent document; mid-run writes complicate atomic correctness for marginal observability gain. Revisit only if a long-running flow needs it.
- **Dual-write phase** — Phase 1 will not keep `_log_debug` running alongside the new recorder. Rationale: less churn, fewer commits, faster path to value; existing test suite catches regressions.
- **Trajectory upload / external delivery** — daydream writes `.trajectory.json` locally only. Rationale: shipping to Harbor / S3 / viewers is the consumer's job, not daydream's.
- **ATIF schema versions other than v1.4** — pinned to current. Rationale: every supported agent in the Harbor ecosystem is on v1.4; supporting older versions is unnecessary complexity.
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
- **Backends**: Both Claude and Codex backends must produce equivalent ATIF trajectories — the `Backend` Protocol already normalizes events; trajectory recording lives in `agent.py`, not in any single backend.
- **Existing tests**: All 343 tests must pass post-migration. New tests added for trajectory recording and validation.
- **CLI**: `--debug` is removed; `--trajectory <path>` is added. The `--ttt` and `--pr` flags continue to work and produce trajectories.
- **Dependency footprint**: `harbor` (or its `harbor.models.trajectories` submodule) added to `pyproject.toml`. Install footprint must be vetted before adoption — if Harbor pulls in heavy transitive deps, fall back to vendoring the JSON Schema and writing our own Pydantic models (already a transitive dep via `claude-agent-sdk`).
- **Schema version**: Pinned to ATIF-v1.4. No multi-version support.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Trajectory granularity = one per daydream run, with subagents nested via ATIF's subagent delegation | Best for replay and SFT/RL pipelines; matches how Harbor consumers expect trajectories; daydream's orchestration is conceptually one session even though `run_agent()` is called many times | — Pending |
| Validation library = take Harbor as a dep, build trajectories with its Pydantic models | Automatic validation, no schema drift, matches what every other Harbor-integrated agent does. Risk: install footprint must be vetted. | — Pending |
| `--debug` flag removed; trajectory always written; output path via `--trajectory <path>` | Trajectory recording is low cost / high value; always-on means trajectories from normal runs become available for learning/training; removes a duplicate concern | — Pending |
| Hard cutover, no dual-write phase | Less churn, fewer commits, faster to value; existing 343-test suite is sufficient regression coverage; no known external consumer of `.review-debug-*.log` | — Pending |
| Pin to ATIF-v1.4 | Current spec version supported by every Harbor-integrated agent; no need for multi-version support | — Pending |

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
*Last updated: 2026-04-26 after initialization*
