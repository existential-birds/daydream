<!-- refreshed: 2026-04-26 -->
# Architecture

**Analysis Date:** 2026-04-26

## System Overview

```text
┌────────────────────────────────────────────────────────────────────┐
│                      CLI Entry Points                               │
│  `daydream/cli.py:main()`   `daydream/__main__.py`                 │
└───────────────────────────────┬────────────────────────────────────┘
                                │ anyio.run(run, config)
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                              │
│  `daydream/runner.py`                                              │
│  run()  run_pr_feedback()  run_trust()                             │
│  ──────────────────────────────────────────────────────────        │
│  dispatches to deep mode:  `daydream/deep/orchestrator.py`         │
└───┬───────────────┬───────────────┬─────────────────┬─────────────┘
    │ normal flow   │ --pr flow     │ --ttt flow      │ --deep flow
    ▼               ▼               ▼                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                       Phase Functions                               │
│  `daydream/phases.py`                                              │
│  phase_review()  phase_parse_feedback()  phase_fix()               │
│  phase_test_and_heal()  phase_commit_push()                        │
│  phase_understand_intent()  phase_alternative_review()             │
│  phase_generate_plan()  phase_per_stack_reviews()                  │
│  phase_cross_stack_merge()  phase_fetch_pr_feedback()              │
└───────────────────────────────┬────────────────────────────────────┘
                                │ run_agent()
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Agent Execution Layer                            │
│  `daydream/agent.py` — run_agent(), AgentState, detect_test_success│
└───────────────────────────────┬────────────────────────────────────┘
                                │ Backend.execute()
                    ┌───────────┴───────────┐
                    ▼                       ▼
        ┌─────────────────┐     ┌─────────────────────┐
        │  ClaudeBackend  │     │    CodexBackend      │
        │  `backends/     │     │    `backends/        │
        │   claude.py`    │     │     codex.py`        │
        └────────┬────────┘     └──────────┬──────────┘
                 │                         │
                 ▼                         ▼
        claude-agent-sdk              codex CLI (external)
                                          │
┌────────────────────────────────────────────────────────────────────┐
│  Unified Event Stream: `daydream/backends/__init__.py`             │
│  TextEvent  ThinkingEvent  ToolStartEvent  ToolResultEvent         │
│  CostEvent  ResultEvent  ContinuationToken                         │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                        Terminal UI Layer                            │
│  `daydream/ui.py`                                                  │
│  Rich panels, live updates, phase heroes, cost display             │
└────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| CLI | Arg parsing, signal handling, process lifecycle | `daydream/cli.py` |
| Runner | Flow selection, backend creation, phase sequencing | `daydream/runner.py` |
| Phases | Stateless async workflow steps | `daydream/phases.py` |
| Agent | Backend wrapper, event stream → UI, global state | `daydream/agent.py` |
| ClaudeBackend | Translates claude-agent-sdk messages to AgentEvents | `daydream/backends/claude.py` |
| CodexBackend | Translates codex CLI JSONL output to AgentEvents | `daydream/backends/codex.py` |
| UI | All terminal output — Rich panels, tables, prompts | `daydream/ui.py` |
| Config | Centralized constants, skill mappings | `daydream/config.py` |
| Deep Orchestrator | deep-review pipeline wiring | `daydream/deep/orchestrator.py` |
| Exploration | Pre-scan context types and subagent coordination | `daydream/exploration.py`, `daydream/exploration_runner.py` |
| Tree-Sitter Index | Static import resolution for exploration | `daydream/tree_sitter_index.py` |
| PR Review | Post review findings as inline GitHub PR comments | `daydream/pr_review.py` |

## Pattern Overview

**Overall:** Linear phase pipeline with backend protocol abstraction and unified event-stream model.

**Key Characteristics:**
- All AI calls go through `run_agent()` in `daydream/agent.py` — never the SDK directly
- The `Backend` protocol decouples phase logic from specific AI providers (Claude SDK or Codex CLI)
- Both backends emit identical `AgentEvent` dataclass instances consumed by `run_agent()`
- Three distinct run flows dispatched by `runner.py`: `run()` (normal), `run_pr_feedback()` (PR mode), `run_trust()` (TTT mode), and `run_deep()` (deep mode in `deep/orchestrator.py`)
- Module-level singleton state (`AgentState`) manages cross-cutting concerns

## Layers

**CLI Layer:**
- Purpose: Entry point, argument parsing, signal handling
- Location: `daydream/cli.py`
- Contains: `main()`, `_parse_args()`, `_signal_handler()`, `_auto_detect_pr_number()`
- Depends on: `runner.py`, `agent.py`, `ui.py`
- Used by: Console entry point (`pyproject.toml`), `daydream/__main__.py`

**Orchestration Layer:**
- Purpose: Flow selection, backend instantiation, phase sequencing, loop control
- Location: `daydream/runner.py`
- Contains: `RunConfig` dataclass, `run()`, `run_pr_feedback()`, `run_trust()`, `_resolve_backend()`
- Depends on: `phases.py`, `backends/`, `agent.py`, `ui.py`, `config.py`, `exploration.py`
- Used by: `cli.py` via `anyio.run(run, config)`

**Phase Layer:**
- Purpose: Stateless async functions implementing each discrete workflow step
- Location: `daydream/phases.py`
- Contains: All `phase_*()` functions and prompt builders (`build_review_prompt()`, `build_intent_prompt()`, etc.)
- Depends on: `agent.run_agent()`, `backends.Backend`, `ui.py`, `config.py`
- Used by: `runner.py` exclusively

**Agent Layer:**
- Purpose: Backend execution wrapper; drives Rich UI from event stream; owns global state
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
- Location: `daydream/ui.py` (3470 lines)
- Contains: `LiveToolPanelRegistry`, `ParallelFixPanel`, `ShutdownPanel`, `SummaryData`, `AgentTextRenderer`, print helpers
- Depends on: `rich` only
- Used by: `cli.py`, `runner.py`, `phases.py`, `agent.py`

**Config Layer:**
- Purpose: Centralized constants — skill mappings, output file path, regex patterns
- Location: `daydream/config.py`
- Contains: `ReviewSkillChoice` enum, `REVIEW_SKILLS`, `SKILL_MAP`, `REVIEW_OUTPUT_FILE`, `UNKNOWN_SKILL_PATTERN`
- Depends on: nothing
- Used by: `runner.py`, `phases.py`, `agent.py`, `deep/`

**Exploration Layer:**
- Purpose: Pre-scan codebase context to ground review agents in real imports and conventions
- Location: `daydream/exploration.py`, `daydream/exploration_runner.py`, `daydream/tree_sitter_index.py`
- Contains: `ExplorationContext`, `FileInfo`, `Convention`, `Dependency` types; `pre_scan()`, `select_tier()`, `count_changed_files()`; tree-sitter static import resolution
- Depends on: `backends/`, `prompts/exploration_subagents.py`, `tree_sitter_*` libraries
- Used by: `runner.py` (injects into `RunConfig.exploration_context` before phases), `deep/orchestrator.py`

**Deep Review Layer:**
- Purpose: Multi-stack review pipeline orchestration (TTT + per-stack + cross-stack merge)
- Location: `daydream/deep/`
- Contains: `run_deep()` orchestrator, `detect_stacks()` file router, artifact path helpers, dedup pre-filter, per-stack/merge prompt builders
- Depends on: `phases.py`, `runner.py`, `backends/`, `ui.py`, `config.py`, `exploration.py`
- Used by: `runner.py` (`run()` delegates to `run_deep()` when `config.deep=True`)

## Data Flow

### Normal Flow (review → parse → fix → test)

1. `cli.main()` parses args into `RunConfig` (`daydream/cli.py`)
2. `anyio.run(run, config)` calls `runner.run()` (`daydream/runner.py`)
3. `runner.run()` optionally runs `pre_scan()` exploration (`daydream/exploration_runner.py`)
4. `phase_review()` invokes Beagle skill via backend, writes `.review-output.md` (`daydream/phases.py:645`)
5. `phase_parse_feedback()` extracts JSON issues via structured output (`daydream/phases.py:729`)
6. `phase_fix()` applies each issue fix one-by-one (`daydream/phases.py:794`)
7. `phase_test_and_heal()` runs tests, retry loop if failures (`daydream/phases.py:831`)
8. `phase_commit_push()` commits and pushes changes (`daydream/phases.py:905`)

### Deep Review Flow (`--deep`)

1. `runner.run()` dispatches to `run_deep()` (`daydream/deep/orchestrator.py:235`)
2. Exploration pre-scan populates `ExplorationContext`
3. TTT intent phase: `phase_understand_intent()` writes `intent.md`
4. TTT alternative review: `phase_alternative_review()` writes `alternatives.json`
5. Stack detection: `detect_stacks()` routes changed files to stack assignments (`daydream/deep/detection.py`)
6. Per-stack fan-out: `phase_per_stack_reviews()` runs one agent per stack concurrently with `anyio.CapacityLimiter(4)` (`daydream/phases.py:1383`)
7. Per-stack parse: `phase_parse_feedback()` called per stack
8. Dedup pre-filter: `build_dedup_candidates()` finds same-concern candidates across stacks (`daydream/deep/dedup.py`)
9. Cross-stack merge: `phase_cross_stack_merge()` merges all per-stack records into final `.review-output.md` (`daydream/phases.py:1487`)
10. Optional fix gate: user-gated fix loop

### PR Feedback Flow (`--pr`)

1. `run_pr_feedback()` fetches comments via `phase_fetch_pr_feedback()` (`daydream/runner.py:195`)
2. Parse → fix → `phase_commit_push_auto()` → `phase_respond_pr_feedback()`

### Trust-the-Technology Flow (`--ttt`)

1. `run_trust()` builds diff, optionally runs exploration (`daydream/runner.py:288`)
2. `phase_understand_intent()` → `phase_alternative_review()` → `phase_generate_plan()`
3. Optionally posts findings via `daydream/pr_review.py`

**State Management:**
- `AgentState` singleton in `daydream/agent.py` holds: `quiet_mode`, `model`, `shutdown_requested`, `current_backends`
- Modified only through named setters; reset via `reset_state()` in tests
- `RunConfig` dataclass in `daydream/runner.py` carries per-run configuration
- `ExplorationContext` populated by `pre_scan()` and stored on `RunConfig.exploration_context`

## Key Abstractions

**Backend Protocol:**
- Purpose: Decouples phase logic from specific AI providers
- Pattern: Structural typing (`Protocol`) with three methods: `execute()`, `cancel()`, `format_skill_invocation()`
- `format_skill_invocation()` returns `/{key}` for Claude, `${name}` for Codex
- Factory: `create_backend(name, model)` in `daydream/backends/__init__.py`
- Implementations: `ClaudeBackend` (`daydream/backends/claude.py`), `CodexBackend` (`daydream/backends/codex.py`)

**AgentEvent Union Type:**
- Purpose: Backend-agnostic event vocabulary consumed by `run_agent()`
- Types: `TextEvent`, `ThinkingEvent`, `ToolStartEvent`, `ToolResultEvent`, `CostEvent`, `ResultEvent`
- All are `@dataclass` instances; no base class, just a `TypeAlias` union defined in `daydream/backends/__init__.py`

**RunConfig Dataclass:**
- Purpose: Single configuration object carrying all CLI settings into `run()`
- Pattern: `@dataclass` with defaults; constructed entirely in `_parse_args()`
- Per-phase backend overrides: `review_backend`, `fix_backend`, `test_backend`
- Located in `daydream/runner.py`

**ExplorationContext:**
- Purpose: Aggregated pre-scan results (affected files, conventions, dependencies) for review prompt injection
- Pattern: `@dataclass` with `to_prompt_section()` and `write_to_dir()` methods
- Located in `daydream/exploration.py`; populated by `pre_scan()` in `daydream/exploration_runner.py`

**FixResult TypeAlias:**
- Type: `tuple[dict[str, Any], bool, str | None]` — (item, success, error_message)
- Used in `run_pr_feedback()` to collect per-fix outcomes before commit/respond

**JSON Schemas:**
- `FEEDBACK_SCHEMA`: For `phase_parse_feedback()` — extracts `{issues: [{id, description, file, line, confidence, rationale}]}`
- `ALTERNATIVE_REVIEW_SCHEMA`: For `--ttt` alternative review — includes severity, recommendation, files
- `PLAN_SCHEMA`: For `phase_generate_plan()` — file-level change plan
- All defined inline in `daydream/phases.py`

**StackAssignment:**
- Purpose: Routes changed files to per-stack review agents in deep mode
- Pattern: `@dataclass` with `stack_name`, `files`, `skill_invocation`, `is_docs_only`
- Located in `daydream/deep/detection.py`; produced by `detect_stacks()`

## Entry Points

**Console Script:**
- Location: `daydream/cli.py:main()`
- Triggers: `daydream` command (pyproject.toml scripts), `python -m daydream`
- Responsibilities: Signal handling (SIGINT/SIGTERM), arg parsing, `anyio.run(run, config)`, exit codes

**Module Entry:**
- Location: `daydream/__main__.py`
- Triggers: `python -m daydream`
- Responsibilities: Delegates to `cli.main()`

**Primary Run Function:**
- Location: `daydream/runner.py:run()`
- Triggers: `anyio.run(run, config)` from `cli.main()`, direct call in tests
- Responsibilities: Full workflow orchestration; accepts `RunConfig | None`

**Deep Mode Entry:**
- Location: `daydream/deep/orchestrator.py:run_deep()`
- Triggers: `runner.run()` when `config.deep=True`
- Responsibilities: Full deep-review pipeline (exploration → TTT → per-stack → merge → fix gate)

## Architectural Constraints

- **Async runtime:** `anyio.run()` is the event loop entry point; all phase functions are `async`. Parallel fan-out uses `anyio.create_task_group()` with `anyio.CapacityLimiter(4)` (see `phase_per_stack_reviews()` and `phase_fix_parallel()` in `daydream/phases.py`)
- **Global state:** `_state = AgentState()` at module level in `daydream/agent.py`; accessed only through getter/setter functions; `reset_state()` restores defaults for test isolation
- **No circular imports:** `daydream/deep/orchestrator.py` uses late imports (`from daydream.phases import ...` inside `run_deep()`) to avoid circular dependency with `runner.py`
- **Backend is first param:** All phase functions take `backend: Backend` as their first parameter
- **Single SDK mode:** `ClaudeAgentOptions.permission_mode="bypassPermissions"` and `setting_sources=["user"]` — reads `~/.claude/settings.json` for API keys
- **No direct print():** All user output goes through `daydream/ui.py` functions (`print_info`, `print_error`, etc.) using the Rich `Console` singleton

## Anti-Patterns

### Calling the SDK directly from phases

**What happens:** Importing `ClaudeSDKClient` or `ClaudeAgentOptions` in `daydream/phases.py` directly.
**Why it's wrong:** Bypasses the `Backend` protocol, breaks Codex backend compatibility, and can't be mocked in tests.
**Do this instead:** Call `run_agent(backend, cwd, prompt)` — all SDK interaction must flow through `daydream/agent.py:run_agent()`.

### Accessing `_state` directly

**What happens:** Importing `_state` from `daydream/agent.py` and reading/writing fields directly.
**Why it's wrong:** Bypasses the reset contract; tests that call `reset_state()` won't clear modifications made via direct field access.
**Do this instead:** Use the getter/setter functions (`set_quiet_mode()`, `get_model()`, etc.) or call `reset_state()` in test teardown.

### Adding phase logic to `runner.py`

**What happens:** Putting backend call logic directly inside `run()`, `run_pr_feedback()`, or `run_trust()`.
**Why it's wrong:** `runner.py` is the wiring layer, not the implementation. Phase logic in runner cannot be reused by deep mode or tested in isolation.
**Do this instead:** Add a new `phase_*()` function in `daydream/phases.py` and call it from `runner.py`.

## Error Handling

**Strategy:** Raise typed exceptions for domain errors; catch at appropriate layer.

**Patterns:**
- `MissingSkillError` raised in `run_agent()` when `UNKNOWN_SKILL_PATTERN` matches text; caught in `runner.py` to print install instructions
- `ValueError` raised in `phase_parse_feedback()` on bad structured output; caught in `runner.py`, prints "Parse Failed"
- `KeyboardInterrupt` from SIGINT handler propagates out of `anyio.run()`; caught in `cli.main()` for graceful shutdown display
- `FileNotFoundError` from `check_review_file_exists()` and `check_deep_artifacts()` caught in `runner.py` and `deep/orchestrator.py`
- Subprocess errors caught by type: `except (subprocess.SubprocessError, OSError):`
- JSON/external errors caught narrowly: `except (json.JSONDecodeError, FileNotFoundError):`
- All unhandled exceptions caught in `cli.main()` → `sys.exit(1)`

## Cross-Cutting Concerns

**Validation:** JSON Schema validation of structured agent output in `phase_parse_feedback()` and `_validate_issue()`; schema defined inline in `daydream/phases.py`
**Authentication:** No custom auth — reads Claude API keys from `~/.claude/settings.json` via `setting_sources=["user"]` in `ClaudeAgentOptions`
**Artifact persistence:** Review artifacts written under `target/.daydream/` for deep mode and `target/.review-output.md` for normal/TTT mode

---

*Architecture analysis: 2026-04-26*
