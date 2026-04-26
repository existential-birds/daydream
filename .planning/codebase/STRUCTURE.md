# Project Structure

**Analysis Date:** 2026-04-26

## Top-Level Layout

```text
daydream/                           # Repo root
├── daydream/                       # Main Python package
│   ├── __init__.py                 # Package version + public surface
│   ├── __main__.py                 # `python -m daydream` entry → cli.main()
│   ├── cli.py                      # Arg parsing, signal handlers, anyio.run()
│   ├── runner.py                   # run() / run_pr_feedback() / run_trust() flows
│   ├── phases.py                   # All phase_*() and build_*_prompt() helpers
│   ├── agent.py                    # run_agent(), AgentState, run_agent_with_continuation
│   ├── ui.py                       # Rich UI (panels, live, themes) — 3470 lines
│   ├── config.py                   # Constants, ReviewSkillChoice enum, SKILL_MAP
│   ├── exploration.py              # ExplorationContext + dataclasses
│   ├── exploration_runner.py       # pre_scan(), tier selection, fan-out
│   ├── tree_sitter_index.py        # Static import resolution (Py/TS/Go/Rust)
│   ├── pr_review.py                # GitHub PR inline-comment posting
│   ├── backends/
│   │   ├── __init__.py             # Backend protocol, AgentEvent union, factory
│   │   ├── claude.py               # ClaudeBackend (claude-agent-sdk)
│   │   └── codex.py                # CodexBackend (codex CLI subprocess)
│   ├── deep/
│   │   ├── __init__.py             # Public exports
│   │   ├── orchestrator.py         # run_deep() pipeline wiring
│   │   ├── detection.py            # detect_stacks() routing
│   │   ├── dedup.py                # build_dedup_candidates()
│   │   ├── artifacts.py            # Path helpers for `.daydream/deep/`
│   │   └── prompts.py              # Per-stack and merge prompt builders
│   └── prompts/
│       ├── __init__.py
│       ├── review_system_prompt.py # build_review_system_prompt()
│       └── exploration_subagents.py # Sub-agent prompts for pre-scan
├── tests/                          # Pytest suite (343 tests, 25 files)
│   ├── conftest.py                 # exploration_context_fixture, multi_stack_target
│   ├── test_*.py                   # One file per module under test
│   └── fixtures/
│       ├── codex_jsonl/            # Recorded codex CLI streams (.jsonl)
│       ├── deep/                   # Deep-mode artifact samples
│       └── diffs/                  # Multi-language diff fixtures
├── scripts/
│   ├── _demo_common.py             # Shared demo runner harness
│   ├── demo_review.py              # Review-only demo
│   ├── demo_fix.py / demo_parse.py / demo_test.py / demo_edit_surgery.py
│   ├── run_demo_python.py          # End-to-end demo on a Python target
│   ├── redrive_post.py             # Re-post PR comments from artifacts
│   └── hooks/
│       └── pre-push                # Lint + typecheck + pytest before push
├── docs/
│   ├── plans/                      # Design docs (e.g. 2026-02-21-go-review-design.md)
│   └── reference/                  # Format references (e.g. atif_format.md)
├── .github/
│   ├── workflows/                  # CI: ruff + mypy + pytest on push/PR
│   └── ISSUE_TEMPLATE/
├── .planning/                      # GSD planning artifacts (this directory)
├── .daydream/                      # Runtime output (gitignored) — deep-mode artifacts
├── pyproject.toml                  # Single source of truth for deps + tooling
├── uv.lock                         # uv lockfile (committed)
├── Makefile                        # install / lint / typecheck / test / check / hooks
├── CLAUDE.md                       # Project instructions for Claude Code
└── README.md
```

## Key Locations

**Entry points:**
- Console script: `daydream/cli.py:main()` (registered in `pyproject.toml` `[project.scripts]`)
- Module entry: `daydream/__main__.py` → delegates to `cli.main()`
- Top-level async: `daydream/runner.py:run()` invoked via `anyio.run(run, config)`
- Deep-mode entry: `daydream/deep/orchestrator.py:run_deep()` (called when `RunConfig.deep=True`)
- TTT entry: `daydream/runner.py:run_trust()` (called when `RunConfig.trust_the_technology=True`)
- PR feedback entry: `daydream/runner.py:run_pr_feedback()` (called when `RunConfig.pr` is set)

**Configuration:**
- Project metadata + dependencies + tool config: `pyproject.toml`
- mypy config: `[tool.mypy]` in `pyproject.toml`
- ruff config: `[tool.ruff]` and `[tool.ruff.lint]` in `pyproject.toml`
- pytest config: `[tool.pytest.ini_options]` in `pyproject.toml`
- Lockfile: `uv.lock` at repo root

**Core logic:**
- All phases: `daydream/phases.py` — single 1552-line module; one `phase_<name>()` per workflow step
- Backend abstractions: `daydream/backends/__init__.py` (protocol + events + factory)
- Backend implementations: `daydream/backends/claude.py`, `daydream/backends/codex.py`
- Constants: `daydream/config.py` (`ReviewSkillChoice`, `SKILL_MAP`, `REVIEW_OUTPUT_FILE`, `UNKNOWN_SKILL_PATTERN`)

**Deep review pipeline:**
- Orchestrator: `daydream/deep/orchestrator.py:run_deep()`
- Stack detection: `daydream/deep/detection.py:detect_stacks()`
- Cross-stack dedup: `daydream/deep/dedup.py:build_dedup_candidates()`
- Artifact path helpers: `daydream/deep/artifacts.py`
- Per-stack and merge prompt builders: `daydream/deep/prompts.py`

**Exploration / pre-scan:**
- Public types: `daydream/exploration.py` (`ExplorationContext`, `FileInfo`, `Convention`, `Dependency`)
- Runner / tier selection: `daydream/exploration_runner.py:pre_scan()` + `select_tier()`
- Static import index: `daydream/tree_sitter_index.py` (Python, TypeScript, Go, Rust)
- Sub-agent prompts: `daydream/prompts/exploration_subagents.py`

**UI:**
- All Rich rendering: `daydream/ui.py` (3470 lines)
- Print helpers: `print_info`, `print_error`, `print_success`, `print_warning`, `print_dim`
- Live panel registries: `LiveToolPanelRegistry`, `ParallelFixPanel`, `ShutdownPanel`
- Theme: `NEON_THEME` (Dracula-inspired)

**Tests:**
- Test files: `tests/test_<module>.py` (mirrors source layout)
- Shared fixtures: `tests/conftest.py`
- Recorded codex output: `tests/fixtures/codex_jsonl/*.jsonl`
- Multi-language diff samples: `tests/fixtures/diffs/*.diff`

**Hooks:**
- Pre-push: `scripts/hooks/pre-push` (install via `make hooks`)

**Demos:**
- Demo scripts: `scripts/demo_*.py` and `scripts/run_demo_python.py`
- Shared demo harness: `scripts/_demo_common.py`

## Naming Conventions

**Modules:**
- `snake_case.py` for all source modules: `cli.py`, `runner.py`, `phases.py`, `tree_sitter_index.py`
- Sub-packages are directories with `__init__.py`: `daydream/backends/`, `daydream/deep/`, `daydream/prompts/`
- Test modules mirror the unit under test: `tests/test_phases.py` covers `daydream/phases.py`

**Files:**
- Demo scripts use the `demo_` prefix: `scripts/demo_review.py`, `scripts/demo_fix.py`
- Plan documents in `docs/plans/` use `YYYY-MM-DD-<topic>.md` form

**Functions:**
- Phase functions: `phase_<name>()` — `phase_review`, `phase_parse_feedback`, `phase_fix`, `phase_test_and_heal`, `phase_commit_push`, `phase_per_stack_reviews`, `phase_cross_stack_merge`
- Prompt builders: `build_<thing>_prompt()` — `build_review_prompt`, `build_intent_prompt`, `build_plan_prompt`
- Setter/getter pairs for module-level state: `set_debug_log`/`get_debug_log`, `set_quiet_mode`/`get_quiet_mode`, `set_model`/`get_model`
- Private helpers: leading underscore (`_signal_handler`, `_log_debug`, `_resolve_backend`, `_parse_args`)

**Classes:**
- `PascalCase` throughout: `RunConfig`, `AgentState`, `ClaudeBackend`, `CodexBackend`
- Event dataclasses use `<Name>Event` suffix: `TextEvent`, `ThinkingEvent`, `ToolStartEvent`, `ToolResultEvent`, `CostEvent`, `ResultEvent`
- Exceptions use `Error` suffix: `MissingSkillError`, `CodexError`
- Protocol classes named by role: `Backend`

**Constants:**
- `UPPER_SNAKE_CASE`: `REVIEW_OUTPUT_FILE`, `SKILL_MAP`, `UNKNOWN_SKILL_PATTERN`, `TEST_OUTPUT_TAIL_LINES`

## Where to Put New Code

**New phase function:**
- Location: `daydream/phases.py`
- Naming: `async def phase_<name>(backend: Backend, target_dir: Path, ...)`
- Signature: `backend` is always the first parameter; use keyword-only args (`*`) for optional flags
- Wire it into a flow in `daydream/runner.py` — never call it from `daydream/cli.py`
- Add a matching test in `tests/test_phases.py`

**New backend:**
- Location: `daydream/backends/<name>.py`
- Implement the `Backend` protocol from `daydream/backends/__init__.py` (`execute`, `cancel`, `format_skill_invocation`)
- Register the backend in `create_backend()` (`daydream/backends/__init__.py`)
- Add tests in `tests/test_backend_<name>.py` mirroring `test_backend_claude.py` / `test_backend_codex.py`

**New CLI flag:**
- Location: `daydream/cli.py:_parse_args()` for the `argparse` declaration
- Location: `daydream/runner.py:RunConfig` for the dataclass field
- Wire flag → field in `_parse_args()`; phases read from `config.<field>`
- Add CLI tests in `tests/test_cli.py`

**New skill mapping:**
- Location: `daydream/config.py`
- Add to `SKILL_MAP` and (if applicable) `REVIEW_SKILLS` and `ReviewSkillChoice`
- Update `tests/test_*.py` that exercise skill resolution

**New deep-review stage:**
- Location: `daydream/deep/orchestrator.py` (wiring) + a new helper in `daydream/deep/<stage>.py` if non-trivial
- Reuse phase functions from `daydream/phases.py` where possible — do not duplicate phase logic

**New prompt builder:**
- Location: `daydream/phases.py` for inline phase-specific prompts (`build_<x>_prompt`)
- Location: `daydream/prompts/` for reusable system prompts shared across phases or sub-agents

**New UI component:**
- Location: `daydream/ui.py`
- Use the existing `console = create_console()` singleton; never instantiate a separate `Console`
- Honor `get_quiet_mode()` from `daydream/agent.py` before printing

**New test:**
- Location: `tests/test_<module>.py` (one file per source module)
- Pattern: `async def test_<behavior>(tmp_path, monkeypatch)`
- Stub UI calls via `monkeypatch.setattr("daydream.<module>.print_info", lambda *a, **kw: None)` (and the other `print_*` helpers)
- Mock backends inline as small classes implementing `execute` / `cancel` / `format_skill_invocation`

**New demo:**
- Location: `scripts/demo_<name>.py`
- Reuse the harness in `scripts/_demo_common.py`

**New design / planning doc:**
- Location: `docs/plans/<YYYY-MM-DD>-<topic>.md` for design proposals
- Location: `docs/reference/<topic>.md` for stable reference material

## Special Directories

**`.planning/`:**
- GSD workflow state — codebase maps, ADRs, phase plans
- Generated and managed by GSD commands; not application source

**`.daydream/`:**
- Runtime artifact output (gitignored) — deep-mode review artifacts, intermediate JSON
- Created by `daydream/deep/orchestrator.py` and `daydream/deep/artifacts.py`

**`scripts/hooks/`:**
- Git hooks installed via `make hooks` (symlinks them into `.git/hooks/`)
- Currently: `pre-push` matches CI checks (ruff + mypy + pytest)

**`tests/fixtures/`:**
- `codex_jsonl/` — recorded `codex` CLI streams replayed in `test_backend_codex.py`
- `diffs/` — sample diffs per language used by exploration / detection tests
- `deep/` — deep-mode artifact samples (currently a `README.md`)

---

*Structure analysis: 2026-04-26*
