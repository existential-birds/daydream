# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Daydream is an automated code review and fix loop using the Claude Agent SDK. It launches review agents equipped with Beagle skills (specialized knowledge modules) to review code, parse actionable feedback, apply fixes automatically, and validate changes by running tests.

## Commands

```bash
# Install dependencies and git hooks
make install
make hooks

# Run the CLI
daydream [TARGET] [OPTIONS]

# Run as module
python -m daydream

# Examples
daydream /path/to/project --python      # Python/FastAPI review
daydream /path/to/project --typescript  # React/TypeScript review
daydream --review-only /path/to/project # Review only, skip fixes
daydream --debug /path/to/project       # Enable debug logging

# Development
make lint       # Run ruff linter
make typecheck  # Run mypy type checker
make test       # Run pytest
make check      # Run all CI checks locally
```

## Architecture

The package follows a phased execution model:

```
cli.py → runner.py → phases.py → agent.py
                  ↘ ui.py (terminal output)
```

### Module Responsibilities

- **cli.py**: Entry point, argument parsing, signal handlers (SIGINT/SIGTERM)
- **runner.py**: Main orchestration via `run()` async function, `RunConfig` dataclass
- **phases.py**: Four workflow phases:
  1. `phase_review()` - Invoke Beagle review skill, write to `.review-output.md`
  2. `phase_parse_feedback()` - Extract actionable issues as JSON
  3. `phase_fix()` - Apply fixes one-by-one
  4. `phase_test_and_heal()` - Run tests, interactive retry/fix loop
- **agent.py**: Claude SDK client wrapper, `run_agent()` streams responses, `AgentState` dataclass for consolidated state, `MissingSkillError` exception
- **ui.py**: Rich-based terminal UI with Dracula theme, live-updating panels
- **config.py**: Skill mappings, constants

### Key Patterns

- All agent interactions use `ClaudeSDKClient` from `claude-agent-sdk` with `bypassPermissions` mode
- Streaming responses are processed via async iterator over message types (AssistantMessage, UserMessage, ResultMessage)
- Tool call panels use Rich's `Live` for animated throbbers during execution
- Global state consolidated in `AgentState` dataclass (debug_log, quiet_mode, model, shutdown_requested) with `_current_client` for SDK instance

## Dependencies

- `claude-agent-sdk` - Claude Code SDK for agent interactions
- `anyio` - Async I/O abstraction (used for `anyio.run()`)
- `rich` - Terminal UI components
- `pyfiglet` - ASCII art header

## Prerequisites

Requires the Beagle plugin for Claude Code to be installed. The review skills (`beagle-python:review-python`, `beagle-react:review-frontend`, `beagle-elixir:review-elixir`) are provided by Beagle.

<!-- GSD:project-start source:PROJECT.md -->
## Project

**Daydream**

An automated code review and fix loop using the Claude Agent SDK. Daydream launches review agents equipped with Beagle skills to review code, parse actionable feedback, apply fixes automatically, and validate changes. It also has a "trust the technology" mode (`--ttt`) that does stack-agnostic PR review: understand intent, evaluate alternatives, and generate implementation plans.

**Core Value:** Reviews and recommendations must be grounded in actual codebase understanding — not guesses based on the diff alone.

### Constraints

- **SDK**: Must use `claude-agent-sdk` for subagent capabilities — no custom orchestration framework
- **Backends**: Exploration must work through the `Backend` protocol (or extend it cleanly)
- **Existing tests**: 50+ tests must continue passing — don't break existing flows
- **CLI interface**: `--ttt` flag and normal flow must both benefit from exploration
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.12 - All source code in `daydream/` package
## Runtime
- Python 3.12 (requires `>=3.12` per `pyproject.toml`)
- uv (all commands via `uv run`, `uv sync`)
- Lockfile: `uv.lock` present and committed
## Frameworks
- anyio 4.0+ - Async I/O abstraction; `anyio.run()` is the entry point in `daydream/cli.py`, `anyio.create_task_group()` used for parallel fixes in `daydream/phases.py`
- rich 13.0+ - All terminal output: Live panels, themed console, progress indicators; configured in `daydream/ui.py`
- pyfiglet 1.0+ - ASCII art phase headers rendered via `daydream/ui.py`
- hatchling - Build backend declared in `pyproject.toml` `[build-system]`
## Key Dependencies
- `claude-agent-sdk` 0.1.27+ - Core Claude integration; `ClaudeSDKClient`, `ClaudeAgentOptions`, and all message types imported in `daydream/backends/claude.py`
- `anyio` 4.0+ - Required for all async execution; used in `daydream/cli.py` (entry), `daydream/phases.py` (task groups, capacity limiting)
- `rich` 13.0+ - Terminal UI; `Console`, `Live`, panels, Dracula-inspired theme in `daydream/ui.py`
- `pyfiglet` 1.0+ - Phase name banners in `daydream/ui.py`
## Configuration
- No `.env` file; no environment variable loading library used
- The Claude backend reads model/permission config via `ClaudeAgentOptions` in `daydream/backends/claude.py`
- `setting_sources=["user", "project", "local"]` means Claude reads `~/.claude/settings.json` for API keys and plugin config
- `pyproject.toml` - Single source of truth for project metadata, dependencies, tool config
- `[tool.mypy]` - `python_version = "3.12"`, `ignore_missing_imports = true`
- `[tool.ruff]` - `line-length = 120`, `target-version = "py312"`, rules: `E`, `F`, `I`, `W`
- `[tool.pytest.ini_options]` - `asyncio_mode = "auto"`, `asyncio_default_fixture_loop_scope = "function"`
## Development Tooling
- ruff 0.9+ - Run via `uv run ruff check daydream` (`make lint`)
- mypy 1.14+ - Run via `uv run mypy daydream` (`make typecheck`)
- pytest 8.0+ with pytest-asyncio 0.24+ - Run via `uv run pytest -v` (`make test`)
- No CI pipeline file detected (no `.github/workflows/`, no `Makefile` CI target beyond local `make check`)
- Pre-push hook at `scripts/hooks/pre-push` runs lint + typecheck + full test suite before every push
## CLI Entry Point
- Console script: `daydream = "daydream.cli:main"` declared in `pyproject.toml`
- Can also run as module: `python -m daydream`
## Platform Requirements
- Python 3.12+, uv package manager
- Requires Beagle plugin installed in Claude Code (`~/.claude/settings.json`)
- `codex` CLI must be on PATH when using the Codex backend
- No server deployment; purely a CLI tool executed locally
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- `snake_case.py` for all modules: `cli.py`, `runner.py`, `phases.py`, `agent.py`
- Private helpers prefixed with underscore: `_parse_args`, `_resolve_backend`, `_log_debug`
- Test files prefixed with `test_`: `test_cli.py`, `test_integration.py`, `test_phases.py`
- `snake_case` throughout: `run_agent`, `phase_review`, `detect_test_success`
- Private helpers start with `_`: `_signal_handler`, `_build_fix_prompt`, `_get_head_sha`
- Phase functions follow `phase_<name>` convention: `phase_review`, `phase_fix`, `phase_parse_feedback`, `phase_test_and_heal`
- Setter/getter pairs for module-level state: `set_debug_log`/`get_debug_log`, `set_quiet_mode`/`get_quiet_mode`
- `snake_case` throughout
- Boolean flags use descriptive names: `cleanup_enabled`, `shutdown_requested`, `trust_the_technology`
- Type-annotated at declaration where possible
- `PascalCase` for all classes: `RunConfig`, `AgentState`, `MockBackend`, `ClaudeBackend`
- Event dataclasses use `<Name>Event` pattern: `TextEvent`, `ToolStartEvent`, `CostEvent`, `ResultEvent`
- Enums use `PascalCase` class name, `UPPER_SNAKE` members: `ReviewSkillChoice.PYTHON`
- Exception classes use `Error` suffix: `MissingSkillError`, `CodexError`
- Protocol classes named by role: `Backend`
## Code Style
- Ruff formatter (target: Python 3.12, line length: 120)
- Config in `pyproject.toml` under `[tool.ruff]`
- Ruff with rule sets `E`, `F`, `I`, `W` (pycodestyle errors, pyflakes, isort, warnings)
- `# noqa: S603, S607` suppression for `subprocess.run` calls with hardcoded args — always accompanied by an inline comment explaining why it's safe
- `# noqa: E402` for necessary late imports in `__init__.py`
- mypy with `ignore_missing_imports = true` (for external SDKs without stubs)
- Python 3.12 union syntax: `str | None`, `list[Backend]`, `dict[str, Any]`
- `from __future__ import annotations` used in `backends/__init__.py` for forward references
- `TYPE_CHECKING` guard for `AsyncIterator` import: `if TYPE_CHECKING: from collections.abc import AsyncIterator`
- Return types annotated on all public functions
## Import Organization
- Grouped imports from same module on separate lines, collected into single `from X import (...)` block
- No star imports
- No `__all__` except in `backends/__init__.py` which explicitly re-exports public API
- None. All imports use full `daydream.<module>` paths.
## Dataclasses
- Prefer `@dataclass` for config and event types: `RunConfig`, `AgentState`, all event types
- Mutable defaults use `field(default_factory=...)`: `current_backends: list[Backend] = field(default_factory=list)`
- Docstrings on dataclasses list all attributes under `Attributes:` section
## Module-Level State
- Singleton state via `_state = AgentState()` at module level in `agent.py`
- State accessed only through getter/setter functions — never by importing `_state` directly
- `reset_state()` function provided for test isolation
## Error Handling
- Raise typed exceptions for domain errors: `MissingSkillError`, `CodexError`, `ValueError`
- Caller catches at appropriate layer: `run()` in `runner.py` catches `MissingSkillError` and `ValueError` from phases
- Subprocess errors caught by type: `except (subprocess.SubprocessError, OSError):`
- JSON/external errors caught narrowly: `except (json.JSONDecodeError, FileNotFoundError):`
- `except Exception as e:` only at top-level CLI boundary in `cli.py`
- No silent swallowing — always log or re-raise
- Always called with `capture_output=True`, `text=True`, `timeout=N`, `shell=False`
- `# noqa: S603` comment on every call explaining args are not user-controlled
## Logging
- Prefixed log entries: `[PROMPT]`, `[TEXT]`, `[TOOL_USE]`, `[TOOL_RESULT]`, `[COST]`, `[SCHEMA_OK]`, `[WARN]`
- No `print()` statements in library code; all user-facing output via `daydream.ui` functions: `print_info`, `print_error`, `print_success`, `print_warning`
- Rich `Console` object (`console = create_console()`) is module-level singleton in `agent.py`
## Comments
- Long comment blocks at module level explaining architectural decisions (e.g., singleton pattern explanation in `agent.py`)
- Inline `# noqa:` always followed by a reason comment
- Algorithm steps in complex flows labeled (e.g., `# Phase 1: Review`, `# Phase 2: Parse feedback`)
- No obvious/redundant comments
- All public functions have Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections
- Private helpers (`_log_debug`) have single-line docstrings
- Dataclass docstrings include `Attributes:` section listing all fields
- Module-level docstring in every file; `config.py` includes an `Exports:` section
## Function Design
- Config passed as a dataclass (`RunConfig`) rather than many keyword args
- Backend always first parameter in phase functions: `phase_fix(backend: Backend, target_dir: Path, ...)`
- Phase functions return meaningful types: `phase_test_and_heal` returns `tuple[bool, int]` (passed, retries)
- `run()` returns `int` exit code always
## Protocol Pattern (Backend)
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Linear phase pipeline (review → parse → fix → test → commit) orchestrated by a single async `run()` function
- Backend protocol abstraction: all agent calls go through a `Backend` instance, never the SDK directly
- Unified event-stream model: both backends emit identical `AgentEvent` dataclass instances, consumed by `agent.py` which drives the Rich UI
- Module-level singleton state (`AgentState`) for cross-cutting concerns (debug log, quiet mode, model, shutdown flag)
- Three distinct run flows dispatched by `runner.py`: normal, PR feedback (`--pr`), and trust-the-technology (`--ttt`)
## Layers
- Purpose: Argument parsing, signal handling, process lifecycle
- Location: `daydream/cli.py`
- Contains: `main()`, `_parse_args()`, `_signal_handler()`, `_auto_detect_pr_number()`
- Depends on: `runner.py`, `agent.py` (console/state setters), `ui.py`
- Used by: Console entry point (`pyproject.toml` scripts), `daydream/__main__.py`
- Purpose: Run flow selection, backend instantiation, phase sequencing, loop control
- Location: `daydream/runner.py`
- Contains: `RunConfig` dataclass, `run()`, `run_pr_feedback()`, `run_trust()`, `_resolve_backend()`
- Depends on: `phases.py`, `backends/`, `agent.py`, `ui.py`, `config.py`
- Used by: `cli.py` (via `anyio.run(run, config)`)
- Purpose: Stateless async functions implementing each discrete workflow step
- Location: `daydream/phases.py`
- Contains: `phase_review()`, `phase_parse_feedback()`, `phase_fix()`, `phase_test_and_heal()`, `phase_commit_push()`, `phase_fetch_pr_feedback()`, `phase_fix_parallel()`, `phase_understand_intent()`, `phase_alternative_review()`, `phase_generate_plan()`, `phase_commit_iteration()`
- Depends on: `agent.run_agent()`, `backends.Backend`, `ui.py`, `config.py`
- Used by: `runner.py` exclusively
- Purpose: Backend execution wrapper; drives the Rich UI from the event stream; owns global state
- Location: `daydream/agent.py`
- Contains: `run_agent()`, `AgentState`, state getters/setters, `detect_test_success()`, `MissingSkillError`
- Depends on: `backends/`, `ui.py`
- Used by: `phases.py` (all agent invocations go through `run_agent()`)
- Purpose: SDK/CLI adapters that translate external APIs into a unified `AgentEvent` stream
- Location: `daydream/backends/`
- Contains: `Backend` protocol, `ClaudeBackend`, `CodexBackend`, all `AgentEvent` dataclasses, `create_backend()` factory
- Depends on: `claude-agent-sdk` (Claude), external `codex` CLI (Codex)
- Used by: `agent.py` (consumes events), `runner.py` (instantiates via `create_backend`)
- Purpose: All terminal rendering — panels, phase heroes, tables, prompts, cost display
- Location: `daydream/ui.py` (3295 lines)
- Contains: Rich-based live panels (`LiveToolPanelRegistry`, `ParallelFixPanel`, `ShutdownPanel`), print helpers, `SummaryData`, `AgentTextRenderer`
- Depends on: `rich` only
- Used by: `cli.py`, `runner.py`, `phases.py`, `agent.py`
- Purpose: Centralized constants — skill mappings, output file path, regex patterns
- Location: `daydream/config.py`
- Contains: `ReviewSkillChoice` enum, `REVIEW_SKILLS`, `SKILL_MAP`, `REVIEW_OUTPUT_FILE`, `UNKNOWN_SKILL_PATTERN`
- Depends on: nothing
- Used by: `runner.py`, `phases.py`, `agent.py`
- Purpose: System prompt builders for agent interactions
- Location: `daydream/prompts/review_system_prompt.py`
- Contains: `CodebaseMetadata` dataclass, `build_review_system_prompt()`
- Depends on: nothing
- Used by: Not currently wired into main flow (exploratory/unused module)
## Data Flow
```
```
- `AgentState` singleton in `agent.py` holds: `debug_log`, `quiet_mode`, `model`, `shutdown_requested`, `current_backends`
- Modified only through named setters (`set_quiet_mode()`, `set_model()`, etc.); reset via `reset_state()` in tests
- `RunConfig` dataclass in `runner.py` carries per-run configuration; immutable after construction
## Key Abstractions
- Purpose: Decouples phase logic from specific AI providers
- Pattern: Structural typing (`Protocol`) with three methods: `execute()`, `cancel()`, `format_skill_invocation()`
- `format_skill_invocation()` returns `/{key}` for Claude, `${name}` for Codex
- Implementations: `ClaudeBackend` (`daydream/backends/claude.py`), `CodexBackend` (`daydream/backends/codex.py`)
- Purpose: Backend-agnostic event vocabulary consumed by `run_agent()`
- Types: `TextEvent`, `ThinkingEvent`, `ToolStartEvent`, `ToolResultEvent`, `CostEvent`, `ResultEvent`
- All are `@dataclass` instances; no base class, just a `TypeAlias` union
- Purpose: Single configuration object carrying all CLI settings into `run()`
- Pattern: `@dataclass` with defaults; constructed entirely in `_parse_args()`
- Per-phase backend overrides: `review_backend`, `fix_backend`, `test_backend`
- Purpose: Typed return from fix operations
- Type: `tuple[dict[str, Any], bool, str | None]` — (item, success, error_message)
- `FEEDBACK_SCHEMA`: JSON Schema for `phase_parse_feedback()` — extracts `{issues: [{id, description, file, line}]}`
- `ALTERNATIVE_REVIEW_SCHEMA`: For `--ttt` alternative review — includes severity, recommendation
- `PLAN_SCHEMA`: For `phase_generate_plan()` — file-level change plan
## Entry Points
- Location: `daydream/cli.py:main()`
- Triggers: `daydream` command (pyproject.toml scripts), `python -m daydream`
- Responsibilities: Signal handling (SIGINT/SIGTERM), arg parsing, `anyio.run(run, config)`, exit codes
- Location: `daydream/__main__.py`
- Triggers: `python -m daydream`
- Responsibilities: Delegates to `cli.main()`
- Location: `daydream/runner.py:run()`
- Triggers: `anyio.run(run, config)` from `cli.main()`, direct call in tests
- Responsibilities: Full workflow orchestration; accepts `RunConfig | None`
## Error Handling
- `MissingSkillError` raised in `run_agent()` when `UNKNOWN_SKILL_PATTERN` matches text; caught in `runner.py` to print install instructions
- `ValueError` raised in `phase_parse_feedback()` on bad structured output; caught in `runner.py`, prints "Parse Failed"
- `KeyboardInterrupt` from SIGINT handler propagates out of `anyio.run()`; caught in `cli.main()` for graceful shutdown display
- All unhandled exceptions caught in `cli.main()` → `sys.exit(1)`
- Test failures in loop mode trigger `revert_uncommitted_changes()` before returning `False`
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
