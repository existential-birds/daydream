# Codebase Structure

**Analysis Date:** 2026-04-05

## Directory Layout

```
daydream/                          # Project root
├── daydream/                      # Main package
│   ├── __init__.py                # Version declaration (__version__ = "0.10.0")
│   ├── __main__.py                # python -m daydream entry point
│   ├── cli.py                     # Argument parsing, signal handling, entry point
│   ├── runner.py                  # Orchestration: RunConfig, run(), run_pr_feedback(), run_trust()
│   ├── phases.py                  # Phase functions: review, parse, fix, test, commit
│   ├── agent.py                   # run_agent(), AgentState singleton, MissingSkillError
│   ├── ui.py                      # All Rich terminal output (3295 lines)
│   ├── config.py                  # Skill mappings, REVIEW_OUTPUT_FILE, regex patterns
│   ├── backends/
│   │   ├── __init__.py            # Backend protocol, AgentEvent types, create_backend() factory
│   │   ├── claude.py              # ClaudeBackend (claude-agent-sdk wrapper)
│   │   └── codex.py               # CodexBackend (external codex CLI wrapper)
│   ├── prompts/
│   │   ├── __init__.py            # Empty
│   │   └── review_system_prompt.py # CodebaseMetadata, build_review_system_prompt() (unused in main flow)
│   └── rlm/                       # Empty (placeholder, only __pycache__)
├── tests/
│   ├── test_cli.py                # CLI argument parsing and validation tests
│   ├── test_integration.py        # End-to-end run() flow tests with MockBackend
│   ├── test_loop.py               # Loop mode and iteration tests
│   ├── test_phases.py             # Unit tests for phase functions
│   ├── test_backend_claude.py     # ClaudeBackend event translation tests
│   ├── test_backend_codex.py      # CodexBackend JSONL parsing tests
│   ├── test_backends_init.py      # create_backend() factory tests
│   ├── fixtures/
│   │   └── codex_jsonl/           # JSONL fixture files for Codex backend tests
│   │       ├── simple_text.jsonl
│   │       ├── tool_use.jsonl
│   │       ├── structured_output.jsonl
│   │       └── ...
│   └── rlm/                       # Empty (placeholder)
├── scripts/
│   ├── _demo_common.py            # Shared helpers for demo scripts
│   ├── demo_review.py             # Demo: run review phase standalone
│   ├── demo_parse.py              # Demo: run parse phase standalone
│   ├── demo_fix.py                # Demo: run fix phase standalone
│   ├── demo_test.py               # Demo: run test phase standalone
│   ├── demo_edit_surgery.py       # Demo: targeted edit operations
│   ├── run_demo_python.py         # Full demo pipeline for Python projects
│   └── hooks/
│       └── pre-push               # Git hook: runs lint + typecheck + tests before push
├── .claude/
│   ├── commands/                  # Claude slash commands (gen-release-notes.md, release.md, etc.)
│   └── skills/                    # Local skill reference docs
│       ├── claude-agent-sdk/      # SDK reference documentation
│       └── rich-library/          # Rich library reference documentation
├── docs/
│   └── plans/                     # Stored implementation plans
├── logs/                          # Runtime log output (gitignored)
├── .planning/
│   └── codebase/                  # GSD codebase map documents (this file's directory)
├── .review-output.md              # Review output written by agents to project root (gitignored)
├── pyproject.toml                 # Package metadata, dependencies, tool config
├── Makefile                       # Developer shortcuts: install, hooks, lint, typecheck, test, check
├── CLAUDE.md                      # Project instructions for Claude Code
└── CHANGELOG.md                   # Version history
```

## Directory Purposes

**`daydream/` (package root):**
- Purpose: All production code lives here; no src/ layout
- Key files: `cli.py` (entry), `runner.py` (orchestration), `phases.py` (workflow steps), `agent.py` (SDK wrapper), `ui.py` (terminal output)

**`daydream/backends/`:**
- Purpose: AI provider adapters behind the `Backend` protocol
- Contains: Protocol definition, event dataclasses, `ClaudeBackend`, `CodexBackend`, `create_backend()` factory
- Key files: `__init__.py` (protocol + factory), `claude.py`, `codex.py`
- Adding a new backend: implement `execute()`, `cancel()`, `format_skill_invocation()` and register in `create_backend()`

**`daydream/prompts/`:**
- Purpose: Reusable prompt builders
- Contains: `review_system_prompt.py` — currently exploratory, not called from main flow
- Key files: `review_system_prompt.py`

**`daydream/rlm/`:**
- Purpose: Placeholder directory (empty, no production code)
- Generated: No
- Committed: Yes (empty with `__pycache__`)

**`tests/`:**
- Purpose: All test files, co-located with fixture data
- Contains: Unit tests, integration tests, backend-specific tests
- Key files: `test_integration.py` (full flow), `test_phases.py` (phase functions), `test_loop.py` (loop mode)

**`tests/fixtures/codex_jsonl/`:**
- Purpose: JSONL replay fixtures for Codex backend unit tests
- Generated: No (hand-crafted)
- Committed: Yes

**`scripts/`:**
- Purpose: Developer demo scripts and git hooks; not part of the installed package
- Contains: Per-phase demos that can be run standalone without a full review loop

**`.claude/`:**
- Purpose: Claude Code project configuration — slash commands and local skill documentation
- Generated: No
- Committed: Yes

**`logs/`:**
- Purpose: Runtime debug logs from `--debug` flag; gitignored
- Generated: Yes (at runtime)
- Committed: No

## Key File Locations

**Entry Points:**
- `daydream/cli.py`: `main()` — CLI entry point registered in pyproject.toml
- `daydream/__main__.py`: `python -m daydream` entry
- `daydream/runner.py`: `run(config)` — programmatic/async entry

**Configuration:**
- `daydream/config.py`: Skill names, output file path, error patterns
- `pyproject.toml`: Package metadata, dependencies, ruff/mypy/pytest config
- `Makefile`: Dev workflow targets

**Core Logic:**
- `daydream/phases.py`: All phase functions (1062 lines) — where workflow behavior lives
- `daydream/agent.py`: `run_agent()` and `AgentState` — the adapter between phases and backends
- `daydream/runner.py`: `RunConfig` and flow dispatch (`run`, `run_pr_feedback`, `run_trust`)

**Backend Abstraction:**
- `daydream/backends/__init__.py`: `Backend` protocol, `AgentEvent` union, `create_backend()` factory
- `daydream/backends/claude.py`: Claude SDK adapter
- `daydream/backends/codex.py`: Codex CLI adapter

**Testing:**
- `tests/test_integration.py`: End-to-end tests patching `create_backend` with `MockBackend`
- `tests/test_phases.py`: Isolated phase function tests
- `tests/fixtures/codex_jsonl/`: JSONL fixtures for Codex backend replay tests

## Naming Conventions

**Files:**
- `snake_case.py` throughout — no exceptions
- Phase functions prefixed `phase_`: `phase_review`, `phase_parse_feedback`, `phase_fix`
- Backend files named after provider: `claude.py`, `codex.py`
- Test files prefixed `test_`: `test_cli.py`, `test_phases.py`, `test_integration.py`
- Demo scripts prefixed `demo_`: `demo_review.py`, `demo_fix.py`

**Functions:**
- Public async phase functions: `async def phase_<name>(backend, cwd, ...) -> ...`
- Private helpers: prefixed `_`: `_parse_args()`, `_resolve_backend()`, `_log_debug()`
- State accessors: `get_<field>()` / `set_<field>()` pattern in `agent.py`

**Classes:**
- `PascalCase`: `RunConfig`, `AgentState`, `ClaudeBackend`, `CodexBackend`, `MissingSkillError`
- UI panels: `<Name>Panel` — `ParallelFixPanel`, `ShutdownPanel`, `LiveToolPanelRegistry`
- Event types: `<Name>Event` — `TextEvent`, `ToolStartEvent`, `CostEvent`

**Constants:**
- `UPPER_SNAKE_CASE` in `config.py`: `REVIEW_OUTPUT_FILE`, `REVIEW_SKILLS`, `SKILL_MAP`, `UNKNOWN_SKILL_PATTERN`
- JSON schemas: `<PURPOSE>_SCHEMA` — `FEEDBACK_SCHEMA`, `ALTERNATIVE_REVIEW_SCHEMA`, `PLAN_SCHEMA`

## Where to Add New Code

**New phase function:**
- Implementation: `daydream/phases.py` — add `async def phase_<name>(backend: Backend, cwd: Path, ...) -> ...`
- Wire-up: `daydream/runner.py` — import and call at appropriate point in `run()` or the relevant flow function
- Tests: `tests/test_phases.py` — use `MockBackend` or `MockBackendWithEvents` pattern

**New review skill:**
- Register in `daydream/config.py`: add to `ReviewSkillChoice` enum, `REVIEW_SKILLS` dict, and `SKILL_MAP`
- Add CLI flag in `daydream/cli.py`: add `add_argument` to the `skill_group`

**New backend:**
- Implementation: `daydream/backends/<name>.py` — implement `Backend` protocol (`execute`, `cancel`, `format_skill_invocation`)
- Register: `daydream/backends/__init__.py` — add branch in `create_backend()`
- Tests: `tests/test_backend_<name>.py` — follow pattern in `test_backend_claude.py` or `test_backend_codex.py`

**New CLI flag:**
- Add `parser.add_argument(...)` in `daydream/cli.py:_parse_args()`
- Add field to `RunConfig` dataclass in `daydream/runner.py`
- Pass field value in `RunConfig(...)` at the end of `_parse_args()`
- Consume in `runner.run()` or the relevant flow function

**New UI component:**
- Implementation: `daydream/ui.py` — add class or function following existing Rich patterns
- Export via the module's implicit namespace (no `__all__` in `ui.py`)

**Shared utilities / git helpers:**
- Git helpers: `daydream/phases.py` — add private function prefixed `_git_<action>()`
- Agent utilities: `daydream/agent.py` — add helper alongside `detect_test_success()`

## Special Directories

**`.planning/codebase/`:**
- Purpose: GSD codebase map documents consumed by `/gsd:plan-phase` and `/gsd:execute-phase`
- Generated: By GSD map-codebase agent
- Committed: Yes

**`.claude/`:**
- Purpose: Claude Code project-level configuration and local skill docs
- Generated: No
- Committed: Yes

**`.daydream/` (written at runtime in target project, not in this repo):**
- Purpose: Temporary workspace for `--ttt` mode — stores `diff.patch`, `plan-{timestamp}.md`
- Generated: Yes, by `run_trust()` in `runner.py`
- Committed: No (lives in target project, not daydream itself)

**`logs/`:**
- Purpose: Runtime output, not version-controlled
- Generated: Yes
- Committed: No

**`.venv/`:**
- Purpose: Python virtual environment managed by `uv`
- Generated: Yes
- Committed: No

---

*Structure analysis: 2026-04-05*
