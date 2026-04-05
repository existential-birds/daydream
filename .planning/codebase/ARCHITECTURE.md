# Architecture

**Analysis Date:** 2026-04-05

## Pattern Overview

**Overall:** Phased pipeline with a pluggable backend abstraction

**Key Characteristics:**
- Linear phase pipeline (review → parse → fix → test → commit) orchestrated by a single async `run()` function
- Backend protocol abstraction: all agent calls go through a `Backend` instance, never the SDK directly
- Unified event-stream model: both backends emit identical `AgentEvent` dataclass instances, consumed by `agent.py` which drives the Rich UI
- Module-level singleton state (`AgentState`) for cross-cutting concerns (debug log, quiet mode, model, shutdown flag)
- Three distinct run flows dispatched by `runner.py`: normal, PR feedback (`--pr`), and trust-the-technology (`--ttt`)

## Layers

**CLI Layer:**
- Purpose: Argument parsing, signal handling, process lifecycle
- Location: `daydream/cli.py`
- Contains: `main()`, `_parse_args()`, `_signal_handler()`, `_auto_detect_pr_number()`
- Depends on: `runner.py`, `agent.py` (console/state setters), `ui.py`
- Used by: Console entry point (`pyproject.toml` scripts), `daydream/__main__.py`

**Orchestration Layer:**
- Purpose: Run flow selection, backend instantiation, phase sequencing, loop control
- Location: `daydream/runner.py`
- Contains: `RunConfig` dataclass, `run()`, `run_pr_feedback()`, `run_trust()`, `_resolve_backend()`
- Depends on: `phases.py`, `backends/`, `agent.py`, `ui.py`, `config.py`
- Used by: `cli.py` (via `anyio.run(run, config)`)

**Phase Layer:**
- Purpose: Stateless async functions implementing each discrete workflow step
- Location: `daydream/phases.py`
- Contains: `phase_review()`, `phase_parse_feedback()`, `phase_fix()`, `phase_test_and_heal()`, `phase_commit_push()`, `phase_fetch_pr_feedback()`, `phase_fix_parallel()`, `phase_understand_intent()`, `phase_alternative_review()`, `phase_generate_plan()`, `phase_commit_iteration()`
- Depends on: `agent.run_agent()`, `backends.Backend`, `ui.py`, `config.py`
- Used by: `runner.py` exclusively

**Agent Layer:**
- Purpose: Backend execution wrapper; drives the Rich UI from the event stream; owns global state
- Location: `daydream/agent.py`
- Contains: `run_agent()`, `AgentState`, state getters/setters, `detect_test_success()`, `MissingSkillError`
- Depends on: `backends/`, `ui.py`
- Used by: `phases.py` (all agent invocations go through `run_agent()`)

**Backend Layer:**
- Purpose: SDK/CLI adapters that translate external APIs into a unified `AgentEvent` stream
- Location: `daydream/backends/`
- Contains: `Backend` protocol, `ClaudeBackend`, `CodexBackend`, all `AgentEvent` dataclasses, `create_backend()` factory
- Depends on: `claude-agent-sdk` (Claude), external `codex` CLI (Codex)
- Used by: `agent.py` (consumes events), `runner.py` (instantiates via `create_backend`)

**UI Layer:**
- Purpose: All terminal rendering — panels, phase heroes, tables, prompts, cost display
- Location: `daydream/ui.py` (3295 lines)
- Contains: Rich-based live panels (`LiveToolPanelRegistry`, `ParallelFixPanel`, `ShutdownPanel`), print helpers, `SummaryData`, `AgentTextRenderer`
- Depends on: `rich` only
- Used by: `cli.py`, `runner.py`, `phases.py`, `agent.py`

**Config Layer:**
- Purpose: Centralized constants — skill mappings, output file path, regex patterns
- Location: `daydream/config.py`
- Contains: `ReviewSkillChoice` enum, `REVIEW_SKILLS`, `SKILL_MAP`, `REVIEW_OUTPUT_FILE`, `UNKNOWN_SKILL_PATTERN`
- Depends on: nothing
- Used by: `runner.py`, `phases.py`, `agent.py`

**Prompts Layer:**
- Purpose: System prompt builders for agent interactions
- Location: `daydream/prompts/review_system_prompt.py`
- Contains: `CodebaseMetadata` dataclass, `build_review_system_prompt()`
- Depends on: nothing
- Used by: Not currently wired into main flow (exploratory/unused module)

## Data Flow

**Normal Review-Fix-Test Flow:**

1. `cli.main()` parses args → builds `RunConfig` → calls `anyio.run(run, config)`
2. `runner.run()` resolves `target_dir`, `skill`, creates `Backend` instances via `create_backend()`
3. `runner.run()` calls `phase_review(review_backend, target_dir, skill)` in `phases.py`
4. `phase_review()` builds a skill invocation prompt (e.g., `/beagle-python:review-python`) and calls `run_agent(backend, cwd, prompt)`
5. `run_agent()` calls `backend.execute(cwd, prompt)` and consumes the `AsyncIterator[AgentEvent]`
6. Each `AgentEvent` drives a Rich UI update (`TextEvent` → `AgentTextRenderer`, `ToolStartEvent` → `LiveToolPanelRegistry`, `CostEvent` → `print_cost`)
7. `run_agent()` returns `(output_text, continuation_token)` to the phase function
8. The review skill writes `.review-output.md` to the target directory (side effect inside the agent's tool calls)
9. `phase_parse_feedback()` re-runs `run_agent()` with a structured prompt and `output_schema=FEEDBACK_SCHEMA`; returns `list[dict]`
10. `phase_fix()` is called once per issue item (sequential by default)
11. `phase_test_and_heal()` runs `run_agent()` for test execution, parses output with `detect_test_success()`, and loops interactively on failure
12. `phase_commit_push()` prompts user then invokes the `beagle-core:commit-push` skill

**PR Feedback Flow:**

1. `runner.run_pr_feedback()` is dispatched when `config.pr_number` is set
2. `phase_fetch_pr_feedback()` invokes `beagle-core:fetch-pr-feedback --pr N --bot NAME` → writes `.review-output.md`
3. Same `phase_parse_feedback()` and `phase_fix()` pipeline as normal flow
4. `phase_commit_push_auto()` commits and pushes without prompt
5. `phase_respond_pr_feedback()` invokes `beagle-core:respond-pr-feedback`

**Trust-the-Technology Flow (`--ttt`):**

1. `runner.run_trust()` gathers `git diff`, `git log`, branch name
2. Writes diff to `.daydream/diff.patch`
3. `phase_understand_intent()` → iterative confirmation loop with user
4. `phase_alternative_review()` → structured output via `ALTERNATIVE_REVIEW_SCHEMA`
5. `phase_generate_plan()` → writes `.daydream/plan-{timestamp}.md`

**Loop Mode (`--loop`):**

1. Validates clean working tree before starting
2. Calls `_run_loop_iteration()` up to `max_iterations` times
3. On success: `phase_commit_iteration()` commits before next review pass; records pre-commit SHA as `diff_base` so next review only covers new changes
4. On test failure: `revert_uncommitted_changes()` rolls back via `git checkout . && git clean -fd`

**Backend Event Stream:**

```
backend.execute(cwd, prompt) → AsyncIterator[AgentEvent]
    TextEvent       → AgentTextRenderer.append()
    ThinkingEvent   → print_thinking()
    ToolStartEvent  → LiveToolPanelRegistry.create()
    ToolResultEvent → panel.set_result(); panel.finish()
    CostEvent       → print_cost()
    ResultEvent     → extracts structured_output, continuation token
```

**State Management:**
- `AgentState` singleton in `agent.py` holds: `debug_log`, `quiet_mode`, `model`, `shutdown_requested`, `current_backends`
- Modified only through named setters (`set_quiet_mode()`, `set_model()`, etc.); reset via `reset_state()` in tests
- `RunConfig` dataclass in `runner.py` carries per-run configuration; immutable after construction

## Key Abstractions

**Backend Protocol (`daydream/backends/__init__.py`):**
- Purpose: Decouples phase logic from specific AI providers
- Pattern: Structural typing (`Protocol`) with three methods: `execute()`, `cancel()`, `format_skill_invocation()`
- `format_skill_invocation()` returns `/{key}` for Claude, `${name}` for Codex
- Implementations: `ClaudeBackend` (`daydream/backends/claude.py`), `CodexBackend` (`daydream/backends/codex.py`)

**AgentEvent Union (`daydream/backends/__init__.py`):**
- Purpose: Backend-agnostic event vocabulary consumed by `run_agent()`
- Types: `TextEvent`, `ThinkingEvent`, `ToolStartEvent`, `ToolResultEvent`, `CostEvent`, `ResultEvent`
- All are `@dataclass` instances; no base class, just a `TypeAlias` union

**RunConfig (`daydream/runner.py`):**
- Purpose: Single configuration object carrying all CLI settings into `run()`
- Pattern: `@dataclass` with defaults; constructed entirely in `_parse_args()`
- Per-phase backend overrides: `review_backend`, `fix_backend`, `test_backend`

**FixResult (`daydream/phases.py`):**
- Purpose: Typed return from fix operations
- Type: `tuple[dict[str, Any], bool, str | None]` — (item, success, error_message)

**Structured Output Schemas (`daydream/phases.py`):**
- `FEEDBACK_SCHEMA`: JSON Schema for `phase_parse_feedback()` — extracts `{issues: [{id, description, file, line}]}`
- `ALTERNATIVE_REVIEW_SCHEMA`: For `--ttt` alternative review — includes severity, recommendation
- `PLAN_SCHEMA`: For `phase_generate_plan()` — file-level change plan

## Entry Points

**CLI Entry:**
- Location: `daydream/cli.py:main()`
- Triggers: `daydream` command (pyproject.toml scripts), `python -m daydream`
- Responsibilities: Signal handling (SIGINT/SIGTERM), arg parsing, `anyio.run(run, config)`, exit codes

**Module Entry:**
- Location: `daydream/__main__.py`
- Triggers: `python -m daydream`
- Responsibilities: Delegates to `cli.main()`

**Programmatic Entry:**
- Location: `daydream/runner.py:run()`
- Triggers: `anyio.run(run, config)` from `cli.main()`, direct call in tests
- Responsibilities: Full workflow orchestration; accepts `RunConfig | None`

## Error Handling

**Strategy:** Raise-and-catch at orchestration boundary; phases propagate exceptions up to `runner.run()`

**Patterns:**
- `MissingSkillError` raised in `run_agent()` when `UNKNOWN_SKILL_PATTERN` matches text; caught in `runner.py` to print install instructions
- `ValueError` raised in `phase_parse_feedback()` on bad structured output; caught in `runner.py`, prints "Parse Failed"
- `KeyboardInterrupt` from SIGINT handler propagates out of `anyio.run()`; caught in `cli.main()` for graceful shutdown display
- All unhandled exceptions caught in `cli.main()` → `sys.exit(1)`
- Test failures in loop mode trigger `revert_uncommitted_changes()` before returning `False`

## Cross-Cutting Concerns

**Logging:** Optional debug file at `{target_dir}/.review-debug-{timestamp}.log`; written via `_log_debug()` in `agent.py`; controlled by `AgentState.debug_log`

**Validation:** CLI validation in `_parse_args()` (mutual exclusions, required combinations); target directory checked as valid path in `runner.run()`; review file existence checked before skip-to-phase

**Authentication:** Handled entirely by `claude-agent-sdk` and external `codex` CLI; daydream has no auth code

**Concurrency:** `anyio` task groups used in `phase_fix_parallel()` with `CapacityLimiter(4)`; normal fix flow is sequential to avoid concurrent writes to the same backend instance

---

*Architecture analysis: 2026-04-05*
