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

# Golden paths (near-zero-flag; `daydream /path` == `daydream review /path`)
daydream /path/to/project                          # review → fix → test (deep multi-stack)
daydream --comment /path/to/project                # review → post inline PR comments, then exit

# Other verbs / flags
daydream --shallow -s python /path/to/project      # Shallow Python review-fix-test loop
daydream --review /path/to/project                 # Review only, skip fixes
daydream --yes /path/to/project                    # Auto-apply fixes without prompting
daydream --loop 3 /path/to/project                 # Repeat review-fix-test up to 3 rounds
daydream feedback 42 --bot "<bot-login>[bot]" /path/to/project  # Bot PR comments
daydream --non-interactive /path/to/project        # Unattended/harness run: take safe defaults, no prompts
daydream --help-all                                # Full advanced flag surface (--start-at, --trajectory, ...)
daydream --review --findings-out findings.json --pr-number 7 /path  # Phase A: emit strict-schema findings artifact
daydream post-findings findings.json --pr 7 --head-sha <sha> --repo owner/repo  # Phase B: validate + post (unattended)

# Data pipeline moved under the `corpus` namespace
daydream corpus harvest                            # Annotate archived runs
daydream corpus build --out out.jsonl              # Project to JSONL training corpus
daydream corpus label <session_id> --outcome accepted

# Per-phase model/backend overrides are config-file-only (no CLI flags):
#   set [tool.daydream] / [tool.daydream.phases.<phase>] in pyproject.toml or .daydream.toml.
#   Precedence: CLI (--model/--backend) > config file > built-in default.

# Development
make lint       # Run ruff linter
make typecheck  # Run mypy type checker
make test       # Run pytest
make check      # Run all CI checks locally
```

## Testing Standard (mandatory)

Every user-visible behavior must have at least one **real-path test**: a test that
enters from the production entrypoint (`runner.run` / the CLI) with real
dependencies — real temp git worktree, real filesystem, real event loop — mocking only
the external network/API backend (via the `Backend` protocol / `create_backend` seam).
Tests must assert observable outcomes (exit code, files written, fixes applied or
declined, transcript state), never that a function was merely called. Unit tests are
supplementary, not a substitute. When shipping a feature, the real-path test that drives
its production path through the prompt/decision it changes is part of the deliverable —
not an optional smoke step. Reference exemplar: the non-interactive/EOF gate tests in
`tests/test_deep_orchestrator.py` drive `runner.run` to the real `prompt_user` gate and
fail if the feature is reverted.

**No caveats — all work is completed.** Do not close out a task with deferred items,
"optional" follow-ups, "out of scope for now", unverified references, or smoke-tests
substituted for real coverage. If a behavior is in scope, its real-path test is written,
run, and green before sign-off — not described, deferred, or left as a note. A reference
you haven't verified against live code is not a caveat to flag; verify it, then state it
as fact. Either the work is done and proven, or it is explicitly still in progress —
never "done, with caveats."

## Non-Negotiable: Fix Bugs at the Root — Never Bypass, Paper Over, or Conceal

This is the highest-priority directive in this file. Violating it makes the agent
dangerous and unusable. It overrides any instinct to complete the literal task quickly.

1. **A bug in a safety or verification mechanism is fixed at its root cause —
   never bypassed.** If a pre-push hook, CI gate, test, type check, lint, or guard
   fails or misbehaves, the only acceptable response is to find and fix the
   underlying defect. The following are FORBIDDEN as "solutions": `git push
   --no-verify`, skipping or disabling tests, commenting out a check, deleting a
   guard, lowering an assertion, or otherwise routing around the mechanism. The
   mechanism is not an obstacle between you and the goal — making it pass *honestly*
   IS the goal. If you ever catch yourself reaching for a bypass flag, stop: that is
   the signal that you have not found the real bug yet.

2. **Own your own bugs in plain language.** If code you wrote is broken, say "I
   wrote a bug" and fix it. Do NOT describe your own defect as the tool, hook,
   environment, or framework being "destructive," "buggy," "hazardous," or "an
   obstacle." Externalizing your own mistake to avoid fixing it is a form of
   dishonesty and is banned.

3. **Fix the bug where it lives, not at a convenient layer.** Do not paper over a
   defect by adding hacks to an unrelated or downstream component (e.g. special-case
   workarounds in a standard tool/hook to compensate for broken application or test
   code). Diagnose the actual source and fix it there. Keep standard infrastructure
   standard.

4. **A dangerous bug outranks the assigned task.** If you discover a defect that
   corrupts state, loses data, mutates the wrong target, or compromises a safety
   mechanism, stop and fix it before continuing — and state it immediately, plainly,
   and without minimizing. Never proceed with the original task on top of a known
   dangerous bug.

5. **Never claim success that isn't verified-working.** "Committed," "pushed," or
   "done" is not "working." Do not report a task complete, fine, or successful unless
   you have verified the actual behavior works (see Testing Standard). Premature or
   false completion claims — declaring victory to end the interaction while a known
   problem remains — are forbidden.

6. **When the user pushes back, the workaround is wrong — find the simple fix.**
   If the user questions your approach on the same point more than once ("why are you
   skipping it?", "just fix it"), STOP defending, working around, or re-explaining.
   Treat their objection as correct and do the direct, root-cause fix they are
   pointing at. Do not make the user extract the truth or the obvious fix through
   repeated confrontation. The obvious, simple fix is almost always the right one;
   reach for it first, not last.

## Architecture

The package follows a phased execution model:

```
cli.py → runner.py → phases.py → agent.py
                  ↘ ui/ (terminal output)
```

### Module Responsibilities

- **cli.py**: Entry point, argument parsing, signal handlers (SIGINT/SIGTERM)
- **runner.py**: Main orchestration via `run()` async function, `RunConfig` dataclass
- **phases.py**: Stateless `phase_*()` workflow steps. Shallow loop: `phase_review` → `phase_parse_feedback` → `phase_fix` → `phase_test_and_heal`. Deep pipeline: `phase_understand_intent`, `phase_alternative_review`, `phase_per_stack_reviews`, `phase_arbiter_review`, `phase_cross_stack_merge`. PR feedback: `phase_fetch_pr_feedback`, `phase_respond_pr_feedback`
- **agent.py**: Backend wrapper, `run_agent()` streams responses, `AgentState` dataclass for consolidated state, `MissingSkillError` exception
- **trajectory.py**: ATIF v1.6 trajectory recorder, `TrajectoryRecorder` with `ContextVar` propagation, `Invocation` per-`run_agent` scope, `Redactor` for secret scrubbing, `DaydreamPhase`/`DaydreamRunFlow` enums
- **ui/**: Rich-based terminal UI package (Dracula theme, live-updating panels); split into `console`, `panels`, `messages`, `tools`, `agent_text`, `summary`, `theme`, `colorize`
- **config.py**: Skill mappings, constants

### Key Patterns

- All agent interactions use `ClaudeSDKClient` from `claude-agent-sdk` with `bypassPermissions` mode
- Streaming responses are processed via async iterator over message types (AssistantMessage, UserMessage, ResultMessage)
- Tool call panels use Rich's `Live` for animated throbbers during execution
- Global state consolidated in `AgentState` dataclass (quiet_mode, model, shutdown_requested) with `_current_client` for SDK instance
- Trajectory recording via `TrajectoryRecorder` propagated through `ContextVar`; parallel fan-outs use `recorder.fork()` for sibling trajectory files

## Dependencies

- `claude-agent-sdk` - Claude Code SDK for agent interactions
- `anyio` - Async I/O abstraction (used for `anyio.run()`)
- `rich` - Terminal UI components
- `pyfiglet` - ASCII art header

## Prerequisites

Requires the Beagle plugin for Claude Code to be installed. The review skills (`beagle-python:review-python`, `beagle-react:review-frontend`, `beagle-elixir:review-elixir`) are provided by Beagle.

## Project

**Daydream**

Daydream is a Python CLI that automates code review and fix loops using the Claude Agent SDK. It launches review agents equipped with Beagle skills to review code, parse actionable feedback, apply fixes automatically, and validate by running tests. The default flow is a deep multi-stack pipeline; `--shallow` opts into a single-skill loop; `--comment` and `--review` turn the run into a review-only flow that posts to a PR or writes a report. The `daydream feedback <pr#>` subcommand ingests bot review comments. Each run is recorded as an **Agent Trajectory Interchange Format (ATIF v1.6)** trajectory, interoperable with the Harbor ecosystem.

**Core Value:** Reviews and recommendations must be grounded in actual codebase understanding — not guesses based on the diff alone. Every daydream run produces a valid, replayable ATIF v1.6 trajectory capturing the full agent interaction history, tool I/O, and token/cost metrics.

### Constraints

- **SDK**: Agent capabilities go through `claude-agent-sdk` (Claude backend); ATIF is layered on top of the `Backend` / `AgentEvent` abstraction, which keeps backends pluggable (Claude / Codex / Pi).
- **Backends**: All backends produce ATIF trajectories. Codex/Pi parity is partial by design (e.g. no `cost_usd`, no `cached_tokens` from usage) — ATIF Metrics fields are all optional, so this is acceptable.
- **Dependencies**: No `harbor` runtime dep. Vendored ATIF code under `daydream/atif/` (Apache-2.0, from Harbor v0.5.0). `pydantic>=2.11.7` is an explicit dep.
- **Schema version**: Pinned to ATIF-v1.6 (emission). No multi-version emission support.
- **Recorder placement**: `TrajectoryRecorder` lives in `daydream/trajectory.py`, propagated via `ContextVar` (not `AgentState`). Test isolation via autouse `_reset_trajectory_recorder` fixture in `conftest.py`, mirroring the `reset_state()` pattern.
- **Module-bloat ban**: No ATIF model construction (`Step`, `ToolCall`, `Trajectory`) inside `phases.py` or `ui/` — all ATIF model construction stays in `daydream/trajectory.py`.

## Technology Stack

## Languages
- Python 3.12 - All source code in `daydream/` package; `requires-python = ">=3.12"` declared in `pyproject.toml`
- None. This is a pure Python CLI tool.
## Runtime
- Python 3.12 (minimum required per `pyproject.toml`)
- No `.python-version` file present; version pinned solely via `pyproject.toml`
- uv — all commands run via `uv run`, `uv sync`
- Lockfile: `uv.lock` present and committed (revision 3)
## Frameworks
- anyio 4.12.1 — `anyio.run()` is the process entry point in `daydream/cli.py`; `anyio.create_task_group()` and `anyio.CapacityLimiter` used for parallel fix invocations in `daydream/phases.py`
- rich 14.3.1 — All terminal output; `Live` panels, themed `Console`, progress indicators, Dracula-inspired theme configured in `daydream/ui/`
- pyfiglet 1.0.4 — Phase name banners rendered via `daydream/ui/`
- tree-sitter 0.25.2 — Core parsing engine used in `daydream/tree_sitter_index.py` for import resolution
- tree-sitter-python 0.25.0 — Python language grammar
- tree-sitter-typescript 0.23.2 — TypeScript and TSX grammars
- tree-sitter-go 0.25.0 — Go language grammar
- tree-sitter-rust 0.24.2 — Rust language grammar
- hatchling — Build backend declared in `pyproject.toml` `[build-system]`
- pytest 9.0.3 with pytest-asyncio 1.3.0 — Run via `uv run pytest -v` (`make test`)
- `asyncio_mode = "auto"` configured in `pyproject.toml` `[tool.pytest.ini_options]`
## Key Dependencies
- `claude-agent-sdk` 0.1.52 — Core Claude integration; `ClaudeSDKClient`, `ClaudeAgentOptions`, and all message types imported in `daydream/backends/claude.py`. Pinned to exact version in `pyproject.toml` (`==0.1.52`).
- `mcp` 1.26.0 — Pulled in transitively by `claude-agent-sdk`; MCP tool call events handled in `daydream/backends/codex.py`
- `anyio` 4.12.1 — Required for all async execution; entry point and task orchestration
- `rich` 14.3.1 — All user-facing terminal output; the `daydream/ui/` package depends entirely on it
- `httpx` 0.28.1 — HTTP client pulled in transitively by `claude-agent-sdk`/`mcp`
- `pydantic` — Pulled in transitively; used internally by `claude-agent-sdk` and `mcp`
- `python-dotenv` — Transitively included; not explicitly used in daydream source code
- `jsonschema` — Transitively included; JSON Schema validation used in claude-agent-sdk internals
## Configuration
- No `.env` file; no direct environment variable loading in daydream source code
- Claude backend reads API keys from `~/.claude/settings.json` via `setting_sources=["user"]` in `ClaudeAgentOptions` (`daydream/backends/claude.py`, line 78)
- `CLAUDE_CONFIG_DIR` environment variable can override the default `~/.claude` directory (used in `daydream/deep/orchestrator.py`)
- Codex backend requires `codex` CLI on `$PATH`; credentials managed by the Codex CLI itself
- Pi backend (`--backend pi`) requires the `pi` CLI (`@earendil-works/pi-coding-agent`) on `$PATH`; z.ai coding plan (GLM models) is registered via a pi extension at `~/.pi/extensions/zai-provider/` (installed once via `pi install`). `--provider` defaults to `zai` (matching `DEFAULT_PI_MODEL = "glm-5.2"`); optional `PI_PROVIDER` / `PI_API_KEY` / `PI_THINKING` env overrides forward as `pi` CLI flags.
- `pyproject.toml` — Single source of truth: project metadata, dependencies, tool configuration
- `[tool.mypy]` — `python_version = "3.12"`, `ignore_missing_imports = true`
- `[tool.ruff]` — `line-length = 120`, `target-version = "py312"`, rules: `E`, `F`, `I`, `W`
- `[tool.pytest.ini_options]` — `asyncio_mode = "auto"`, `asyncio_default_fixture_loop_scope = "function"`
## Platform Requirements
- Python 3.12+, uv package manager
- Beagle plugin installed in Claude Code (`~/.claude/settings.json`) — required for `beagle-*:review-*` skills
- `gh` (GitHub CLI) on `$PATH` — required for PR feedback and PR posting features
- `git` on `$PATH` — required for all diff and commit operations
- `codex` CLI on `$PATH` — required only when using `--backend codex`
- `pi` CLI (`@earendil-works/pi-coding-agent`) on `$PATH` — required only when using `--backend pi`
- Pre-push hook at `scripts/hooks/pre-push` runs lint + typecheck + full test suite before every push
- No server deployment; purely a CLI tool executed locally
- Console script entrypoint: `daydream = "daydream.cli:main"` declared in `pyproject.toml`
- Can also run as module: `python -m daydream` via `daydream/__main__.py`

## Conventions

## Naming Patterns
- `snake_case.py` for all modules: `cli.py`, `runner.py`, `phases.py`, `agent.py`, `tree_sitter_index.py`
- Test files prefixed with `test_`: `test_cli.py`, `test_phases.py`, `test_backend_claude.py`
- Sub-packages as directories with `__init__.py`: `daydream/backends/`, `daydream/deep/`, `daydream/prompts/`
- `snake_case` throughout: `run_agent`, `phase_review`, `detect_test_success`, `_git_diff`
- Private helpers prefixed with underscore: `_signal_handler`, `_build_fix_prompt`, `_parse_args`
- Phase functions follow `phase_<name>` convention: `phase_review`, `phase_fix`, `phase_parse_feedback`, `phase_test_and_heal`, `phase_understand_intent`, `phase_generate_plan`
- Prompt builder functions follow `build_<thing>_prompt`: `build_review_prompt`, `build_intent_prompt`, `build_plan_prompt`
- Setter/getter pairs for module-level state: `set_quiet_mode`/`get_quiet_mode`
- `snake_case` throughout
- Boolean flags use descriptive names: `cleanup_enabled`, `shutdown_requested`, `force_worktree`, `shallow`
- Type-annotated at declaration where possible
- `PascalCase` for all classes: `RunConfig`, `AgentState`, `MockBackend`, `ClaudeBackend`, `CodexBackend`, `PiBackend`
- Event dataclasses use `<Name>Event` pattern: `TextEvent`, `ToolStartEvent`, `CostEvent`, `ResultEvent`, `ThinkingEvent`, `ToolResultEvent`
- Enums use `PascalCase` class name, `UPPER_SNAKE` members: `ReviewSkillChoice.PYTHON`, `ReviewSkillChoice.RUST`
- Exception classes use `Error` suffix: `MissingSkillError`, `CodexError`
- Protocol classes named by role: `Backend`
- Constants are `UPPER_SNAKE_CASE`: `REVIEW_OUTPUT_FILE`, `TEST_OUTPUT_TAIL_LINES`, `UNKNOWN_SKILL_PATTERN`, `SKILL_MAP`
## Code Style
- Ruff formatter (target: Python 3.12, line length: 120)
- Config in `pyproject.toml` under `[tool.ruff]`
- Ruff with rule sets `E`, `F`, `I`, `W` (pycodestyle errors, pyflakes, isort, warnings)
- mypy with `ignore_missing_imports = true` (for external SDKs without stubs)
- `# noqa: S603` comment on every `subprocess.run()` call explaining args are not user-controlled
- `# noqa: S607` on hardcoded command name lists
- `# noqa: E402` for necessary late imports (e.g., in `backends/__init__.py`)
- `# noqa: BLE001` for intentionally broad exception catches in parallel isolation contexts
- Python 3.12 union syntax: `str | None`, `list[Backend]`, `dict[str, Any]`
- `from __future__ import annotations` used in modules that need forward references (`backends/__init__.py`, `runner.py`)
- `TYPE_CHECKING` guard for expensive or circular imports: `if TYPE_CHECKING: from collections.abc import AsyncIterator`
- Return types annotated on all public functions
## Import Organization
- Multiple names from the same module collected into a single `from X import (...)` block
- No star imports anywhere
- No `__all__` except in `daydream/backends/__init__.py`, which explicitly re-exports the public API
- None. All local imports use full `daydream.<module>` paths (e.g., `from daydream.backends import Backend`)
## Dataclasses
- Prefer `@dataclass` for config and event types: `RunConfig` in `daydream/runner.py`, `AgentState` in `daydream/agent.py`, all event types in `daydream/backends/__init__.py`
- Mutable defaults use `field(default_factory=...)`: `current_backends: list[Backend] = field(default_factory=list)`
- Docstrings on dataclasses list all attributes under an `Attributes:` section
## Module-Level State
- Singleton state via `_state = AgentState()` at module level in `daydream/agent.py`
- State accessed only through getter/setter functions — never by importing `_state` directly
- `reset_state()` function provided for test isolation
- Module-level `console = create_console()` also treated as a singleton in `daydream/agent.py`
## Error Handling
- Raise typed exceptions for domain errors: `MissingSkillError`, `CodexError`, `ValueError`
- Caller catches at appropriate layer: `run()` in `daydream/runner.py` catches `MissingSkillError` and `ValueError` from phases
- Subprocess errors caught by type: `except (subprocess.SubprocessError, OSError):`
- JSON/external errors caught narrowly: `except (json.JSONDecodeError, FileNotFoundError):`
- `except Exception as e:` only at the top-level CLI boundary in `daydream/cli.py`
- No silent swallowing — always log or re-raise
## Logging
- No `print()` statements in library code; all user-facing output via `daydream.ui` functions: `print_info`, `print_error`, `print_success`, `print_warning`, `print_dim`
- Rich `Console` object (`console = create_console()`) is module-level singleton in `daydream/agent.py`
- Observability via ATIF v1.6 trajectory files written by `TrajectoryRecorder` in `daydream/trajectory.py`; trajectory recorder propagated via `ContextVar` (separate from `AgentState` singleton)
## Comments
- Long comment blocks at module level explaining architectural decisions (e.g., singleton pattern explanation in `daydream/agent.py`)
- Inline `# noqa:` always followed by a reason comment (e.g., `# noqa: S603 - arguments are not user-controlled`)
- Algorithm phases labeled inline (e.g., `# Phase 1: Review`, `# Phase 2: Parse feedback`)
- No obvious or redundant comments
- All public functions have Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections where applicable
- Private helpers (`_signal_handler`, `_build_fix_prompt`) have single-line docstrings
- Dataclass docstrings include `Attributes:` section listing all fields
- Module-level docstring in every file; `daydream/config.py` includes an `Exports:` section listing what the module provides
## Function Design
- Config passed as a dataclass (`RunConfig`) rather than many keyword args
- Backend always first parameter in phase functions: `phase_fix(backend: Backend, target_dir: Path, ...)`
- Phase functions return meaningful types: `phase_test_and_heal` returns `tuple[bool, int]` (passed, retries)
- `run()` in `daydream/runner.py` always returns `int` exit code
- Keyword-only arguments enforced via `*` in signatures for safety-critical params (e.g., `phase_parse_feedback(backend, cwd, *, input_path=None)`)
## Protocol Pattern (Backend)
- Three required methods: `execute()`, `cancel()`, `format_skill_invocation()`
- `execute()` is an `AsyncIterator[AgentEvent]` (not `async def`)
- Implementations: `ClaudeBackend` (`daydream/backends/claude.py`), `CodexBackend` (`daydream/backends/codex.py`)
- PiBackend: subprocess + JSONL (mirrors Codex) — `daydream/backends/pi.py`; `format_skill_invocation()` uses path-reference injection (Pi has no skill registry); `message_id=""` (no per-message id, like Codex); `--session-id <uuid>` (persistent) for fresh runs, `--session-id <id>` for resume (never `--no-session`, which is ephemeral and non-resumable); `--tools read,find,ls,grep` for `read_only=True`
- Mock backends in tests implement the same three-method interface inline (no base class required)
## Async Patterns
- `anyio.run(run, config)` is the top-level async entry point in `daydream/cli.py`
- `anyio.create_task_group()` used for parallel operations in `daydream/phases.py`
- `asyncio_mode = "auto"` in pytest means `async def test_*` functions run automatically without explicit `@pytest.mark.asyncio` in most files (though explicit marks are also used)
- Async generators used for backend `execute()` — callers use `async for event in backend.execute(...)`

## Architecture Reference

## Component Responsibilities
| Component | Responsibility | File |
|-----------|----------------|------|
| CLI | Arg parsing, signal handling, process lifecycle | `daydream/cli.py` |
| Runner | Flow selection, backend creation, phase sequencing | `daydream/runner.py` |
| Phases | Stateless async workflow steps | `daydream/phases.py` |
| Agent | Backend wrapper, event stream → UI, global state | `daydream/agent.py` |
| Trajectory | ATIF v1.6 recorder, redaction, ContextVar propagation | `daydream/trajectory.py` |
| ClaudeBackend | Translates claude-agent-sdk messages to AgentEvents | `daydream/backends/claude.py` |
| CodexBackend | Translates codex CLI JSONL output to AgentEvents | `daydream/backends/codex.py` |
| PiBackend | Translates pi CLI JSONL output to AgentEvents | `daydream/backends/pi.py` |
| UI | All terminal output — Rich panels, tables, prompts | `daydream/ui/` |
| Config | Centralized constants, skill mappings | `daydream/config.py` |
| Deep Orchestrator | deep-review pipeline wiring | `daydream/deep/orchestrator.py` |
| Exploration | Pre-scan context types and subagent coordination | `daydream/exploration.py`, `daydream/exploration_runner.py` |
| Tree-Sitter Index | Static import resolution for exploration | `daydream/tree_sitter_index.py` |
| PR Review | Post review findings as inline GitHub PR comments | `daydream/pr_review.py` |
## Pattern Overview
- All AI calls go through `run_agent()` in `daydream/agent.py` — never the SDK directly
- The `Backend` protocol decouples phase logic from specific AI providers (Claude SDK or Codex CLI)
- Both backends emit identical `AgentEvent` dataclass instances consumed by `run_agent()`
- Five distinct run flows dispatched by `_dispatch()` in `runner.py`: `_run_pr_feedback()` (PR feedback), `_run_comment()` (comment mode), `_run_review()` (review mode), `_run_loop_shallow()` (single-stack), and `_run_loop_deep()` (deep multi-stack, default)
- Module-level singleton state (`AgentState`) manages cross-cutting concerns
## Layers
- Purpose: Entry point, argument parsing, signal handling
- Location: `daydream/cli.py`
- Contains: `main()`, `_parse_args()`, `_signal_handler()`, `_auto_detect_pr_number()`
- Depends on: `runner.py`, `agent.py`, `ui/`
- Used by: Console entry point (`pyproject.toml`), `daydream/__main__.py`
- Purpose: Flow selection, backend instantiation, phase sequencing, loop control
- Location: `daydream/runner.py`
- Contains: `RunConfig` dataclass, `run()`, `run_feedback()`, `_dispatch()`, `_resolve_backend()`
- Depends on: `phases.py`, `backends/`, `agent.py`, `ui/`, `config.py`, `exploration.py`
- Used by: `cli.py` via `anyio.run(run, config)`
- Purpose: Stateless async functions implementing each discrete workflow step
- Location: `daydream/phases.py`
- Contains: All `phase_*()` functions and prompt builders (`build_review_prompt()`, `build_intent_prompt()`, etc.)
- Depends on: `agent.run_agent()`, `backends.Backend`, `ui/`, `config.py`
- Used by: `runner.py` exclusively
- Purpose: Backend execution wrapper; drives Rich UI from event stream; owns global state
- Location: `daydream/agent.py`
- Contains: `run_agent()`, `AgentState`, state getters/setters, `detect_test_success()`, `MissingSkillError`
- Depends on: `backends/`, `ui/`
- Used by: `phases.py` (all agent invocations go through `run_agent()`)
- Purpose: SDK/CLI adapters that translate external APIs into a unified `AgentEvent` stream
- Location: `daydream/backends/`
- Contains: `Backend` protocol, `ClaudeBackend`, `CodexBackend`, `PiBackend`, all `AgentEvent` dataclasses, `create_backend()` factory
- Depends on: `claude-agent-sdk` (Claude), external `codex` CLI (Codex), external `pi` CLI (Pi)
- Used by: `agent.py` (consumes events), `runner.py` (instantiates via `create_backend`)
- Purpose: All terminal rendering — panels, phase heroes, tables, prompts, cost display
- Location: `daydream/ui/` package (`console`, `panels`, `messages`, `tools`, `agent_text`, `summary`, `theme`, `colorize`)
- Contains: `LiveToolPanelRegistry`, `ParallelFixPanel`, `ShutdownPanel`, `SummaryData`, `AgentTextRenderer`, print helpers
- Depends on: `rich` only
- Used by: `cli.py`, `runner.py`, `phases.py`, `agent.py`
- Purpose: Centralized constants — skill mappings, output file path, regex patterns
- Location: `daydream/config.py`
- Contains: `ReviewSkillChoice` enum, `REVIEW_SKILLS`, `SKILL_MAP`, `REVIEW_OUTPUT_FILE`, `UNKNOWN_SKILL_PATTERN`
- Depends on: nothing
- Used by: `runner.py`, `phases.py`, `agent.py`, `deep/`
- Purpose: Pre-scan codebase context to ground review agents in real imports and conventions
- Location: `daydream/exploration.py`, `daydream/exploration_runner.py`, `daydream/tree_sitter_index.py`
- Contains: `ExplorationContext`, `FileInfo`, `Convention`, `Dependency` types; `pre_scan()`, `select_tier()`, `count_changed_files()`; tree-sitter static import resolution
- Depends on: `backends/`, `prompts/exploration_subagents.py`, `tree_sitter_*` libraries
- Used by: `runner.py` (injects into `RunConfig.exploration_context` before phases), `deep/orchestrator.py`
- Purpose: Multi-stack review pipeline orchestration (TTT + per-stack + cross-stack merge)
- Location: `daydream/deep/`
- Contains: `run_deep()` orchestrator, `detect_stacks()` file router, artifact path helpers, dedup pre-filter, per-stack/merge prompt builders
- Depends on: `phases.py`, `runner.py`, `backends/`, `ui/`, `config.py`, `exploration.py`
- Used by: `runner.py` (`run()` delegates to `run_deep()` by default; `--shallow` opts out)
## Data Flow
### Shallow Loop (`--shallow`): review → parse → fix → test
### Deep Review (default): exploration → intent → alternatives → per-stack → merge
### PR Feedback (`daydream feedback <pr#> --bot <name>`)
### Comment / Review (`--comment` / `--review`)
- `AgentState` singleton in `daydream/agent.py` holds: `quiet_mode`, `model`, `shutdown_requested`, `current_backends`
- Modified only through named setters; reset via `reset_state()` in tests
- `RunConfig` dataclass in `daydream/runner.py` carries per-run configuration
- `ExplorationContext` populated by `pre_scan()` and stored on `RunConfig.exploration_context`
## Key Abstractions
- Purpose: Decouples phase logic from specific AI providers
- Pattern: Structural typing (`Protocol`) with three methods: `execute()`, `cancel()`, `format_skill_invocation()`
- `format_skill_invocation()` returns `/{key}` for Claude, `${name}` for Codex
- Factory: `create_backend(name, model)` in `daydream/backends/__init__.py`
- Implementations: `ClaudeBackend` (`daydream/backends/claude.py`), `CodexBackend` (`daydream/backends/codex.py`)
- PiBackend: subprocess + JSONL (mirrors Codex) — `daydream/backends/pi.py`; `format_skill_invocation()` uses path-reference injection (Pi has no skill registry); `message_id=""` (no per-message id, like Codex); `--session-id <uuid>` (persistent) for fresh runs, `--session-id <id>` for resume (never `--no-session`, which is ephemeral and non-resumable); `--tools read,find,ls,grep` for `read_only=True`
- Purpose: Backend-agnostic event vocabulary consumed by `run_agent()`
- Types: `TextEvent`, `ThinkingEvent`, `ToolStartEvent`, `ToolResultEvent`, `CostEvent`, `MetricsEvent`, `TurnEndEvent`, `ResultEvent`
- All are `@dataclass` instances; no base class, just a `TypeAlias` union defined in `daydream/backends/__init__.py`
- Purpose: Single configuration object carrying all CLI settings into `run()`
- Pattern: `@dataclass` with defaults; constructed entirely in `_parse_args()`
- Dispatch fields: `pr_number` (PR feedback flow), `output_mode` (`"loop"` | `"comment"` | `"review"`), `shallow` (single-stack vs deep)
- Per-phase backend overrides: `review_backend`, `fix_backend`, `test_backend`
- Located in `daydream/runner.py`
- Note: The archive manifest (`daydream/archive/manifest.py`) emits `review_only` and `deep` keys derived from `output_mode` and `shallow` for index schema compatibility; these are not RunConfig fields
- Purpose: Aggregated pre-scan results (affected files, conventions, dependencies) for review prompt injection
- Pattern: `@dataclass` with `to_prompt_section()` and `write_to_dir()` methods
- Located in `daydream/exploration.py`; populated by `pre_scan()` in `daydream/exploration_runner.py`
- Type: `tuple[dict[str, Any], bool, str | None]` — (item, success, error_message)
- Used in `run_pr_feedback()` to collect per-fix outcomes before commit/respond
- `FEEDBACK_SCHEMA`: For `phase_parse_feedback()` — extracts `{issues: [{id, description, file, line, confidence, rationale}]}`
- `ALTERNATIVE_REVIEW_SCHEMA`: For `--comment`/`--review` alternative review — includes severity, recommendation, files
- `PLAN_SCHEMA`: For `phase_generate_plan()` — file-level change plan
- All defined inline in `daydream/phases.py`
- Purpose: Routes changed files to per-stack review agents in deep mode
- Pattern: `@dataclass` with `stack_name`, `files`, `skill_invocation`, `is_docs_only`
- Located in `daydream/deep/detection.py`; produced by `detect_stacks()`
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
- Location: `daydream/deep/orchestrator.py:run_deep()`
- Triggers: `runner.run()` by default (when `config.shallow=False` and no PR/comment/review override)
- Responsibilities: Full deep-review pipeline (exploration → TTT → per-stack → merge → fix gate)
## Architectural Constraints
- **Async runtime:** `anyio.run()` is the event loop entry point; all phase functions are `async`. Parallel fan-out uses `anyio.create_task_group()` with `anyio.CapacityLimiter(4)` (see `phase_per_stack_reviews()` and `phase_fix_parallel()` in `daydream/phases.py`)
- **Global state:** `_state = AgentState()` at module level in `daydream/agent.py`; accessed only through getter/setter functions; `reset_state()` restores defaults for test isolation
- **No circular imports:** `daydream/deep/orchestrator.py` uses late imports (`from daydream.phases import ...` inside `run_deep()`) to avoid circular dependency with `runner.py`
- **Backend is first param:** All phase functions take `backend: Backend` as their first parameter
- **Single SDK mode:** `ClaudeAgentOptions.permission_mode="bypassPermissions"` and `setting_sources=["user"]` — reads `~/.claude/settings.json` for API keys
- **No direct print():** All user output goes through `daydream/ui/` functions (`print_info`, `print_error`, etc.) using the Rich `Console` singleton
## Anti-Patterns
- Calling the SDK directly from phases — always go through `run_agent()`
- Accessing `_state` directly — use the getters/setters
- Adding phase logic to `runner.py` — phases live in `phases.py`
## Error Handling
- `MissingSkillError` raised in `run_agent()` when `UNKNOWN_SKILL_PATTERN` matches text; caught in `runner.py` to print install instructions
- `ValueError` raised in `phase_parse_feedback()` on bad structured output; caught in `runner.py`, prints "Parse Failed"
- `KeyboardInterrupt` from SIGINT handler propagates out of `anyio.run()`; caught in `cli.main()` for graceful shutdown display
- `FileNotFoundError` from `check_review_file_exists()` and `check_deep_artifacts()` caught in `runner.py` and `deep/orchestrator.py`
- Subprocess errors caught by type: `except (subprocess.SubprocessError, OSError):`
- JSON/external errors caught narrowly: `except (json.JSONDecodeError, FileNotFoundError):`
- All unhandled exceptions caught in `cli.main()` → `sys.exit(1)`
