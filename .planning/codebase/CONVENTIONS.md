# Coding Conventions

**Analysis Date:** 2026-04-26

## Naming Patterns

**Files:**
- `snake_case.py` for all modules: `cli.py`, `runner.py`, `phases.py`, `agent.py`, `tree_sitter_index.py`
- Test files prefixed with `test_`: `test_cli.py`, `test_phases.py`, `test_backend_claude.py`
- Sub-packages as directories with `__init__.py`: `daydream/backends/`, `daydream/deep/`, `daydream/prompts/`

**Functions:**
- `snake_case` throughout: `run_agent`, `phase_review`, `detect_test_success`, `_git_diff`
- Private helpers prefixed with underscore: `_signal_handler`, `_build_fix_prompt`, `_parse_args`, `_log_debug`
- Phase functions follow `phase_<name>` convention: `phase_review`, `phase_fix`, `phase_parse_feedback`, `phase_test_and_heal`, `phase_understand_intent`, `phase_generate_plan`
- Prompt builder functions follow `build_<thing>_prompt`: `build_review_prompt`, `build_intent_prompt`, `build_plan_prompt`
- Setter/getter pairs for module-level state: `set_debug_log`/`get_debug_log`, `set_quiet_mode`/`get_quiet_mode`, `set_model`/`get_model`

**Variables:**
- `snake_case` throughout
- Boolean flags use descriptive names: `cleanup_enabled`, `shutdown_requested`, `trust_the_technology`, `review_only`
- Type-annotated at declaration where possible

**Types/Classes:**
- `PascalCase` for all classes: `RunConfig`, `AgentState`, `MockBackend`, `ClaudeBackend`, `CodexBackend`
- Event dataclasses use `<Name>Event` pattern: `TextEvent`, `ToolStartEvent`, `CostEvent`, `ResultEvent`, `ThinkingEvent`, `ToolResultEvent`
- Enums use `PascalCase` class name, `UPPER_SNAKE` members: `ReviewSkillChoice.PYTHON`, `ReviewSkillChoice.RUST`
- Exception classes use `Error` suffix: `MissingSkillError`, `CodexError`
- Protocol classes named by role: `Backend`
- Constants are `UPPER_SNAKE_CASE`: `REVIEW_OUTPUT_FILE`, `TEST_OUTPUT_TAIL_LINES`, `UNKNOWN_SKILL_PATTERN`, `SKILL_MAP`

## Code Style

**Formatting:**
- Ruff formatter (target: Python 3.12, line length: 120)
- Config in `pyproject.toml` under `[tool.ruff]`

**Linting:**
- Ruff with rule sets `E`, `F`, `I`, `W` (pycodestyle errors, pyflakes, isort, warnings)
- mypy with `ignore_missing_imports = true` (for external SDKs without stubs)
- `# noqa: S603` comment on every `subprocess.run()` call explaining args are not user-controlled
- `# noqa: S607` on hardcoded command name lists
- `# noqa: E402` for necessary late imports (e.g., in `backends/__init__.py`)
- `# noqa: BLE001` for intentionally broad exception catches in parallel isolation contexts

**Type Annotations:**
- Python 3.12 union syntax: `str | None`, `list[Backend]`, `dict[str, Any]`
- `from __future__ import annotations` used in modules that need forward references (`backends/__init__.py`, `runner.py`)
- `TYPE_CHECKING` guard for expensive or circular imports: `if TYPE_CHECKING: from collections.abc import AsyncIterator`
- Return types annotated on all public functions

## Import Organization

**Order (enforced by ruff isort):**
1. Standard library imports
2. Third-party imports
3. Local `daydream.*` imports

**Grouping:**
- Multiple names from the same module collected into a single `from X import (...)` block
- No star imports anywhere
- No `__all__` except in `daydream/backends/__init__.py`, which explicitly re-exports the public API

**Path Aliases:**
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

**subprocess.run() pattern:**
```python
result = subprocess.run(  # noqa: S603 - arguments are not user-controlled
    ["git", "diff", "--stat"],
    cwd=target_dir,
    capture_output=True,
    text=True,
    timeout=30,
    check=False,
)
```
Always called with `capture_output=True`, `text=True`, `timeout=N`, `shell=False`.

## Logging

- Debug log entries prefixed: `[PROMPT]`, `[TEXT]`, `[TOOL_USE]`, `[TOOL_RESULT]`, `[COST]`, `[SCHEMA_OK]`, `[WARN]`
- No `print()` statements in library code; all user-facing output via `daydream.ui` functions: `print_info`, `print_error`, `print_success`, `print_warning`, `print_dim`
- Rich `Console` object (`console = create_console()`) is module-level singleton in `daydream/agent.py`
- `_log_debug()` private helper in `daydream/agent.py` writes structured entries to the debug log file

## Comments

- Long comment blocks at module level explaining architectural decisions (e.g., singleton pattern explanation in `daydream/agent.py`)
- Inline `# noqa:` always followed by a reason comment (e.g., `# noqa: S603 - arguments are not user-controlled`)
- Algorithm phases labeled inline (e.g., `# Phase 1: Review`, `# Phase 2: Parse feedback`)
- No obvious or redundant comments
- All public functions have Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections where applicable
- Private helpers (`_log_debug`, `_signal_handler`) have single-line docstrings
- Dataclass docstrings include `Attributes:` section listing all fields
- Module-level docstring in every file; `daydream/config.py` includes an `Exports:` section listing what the module provides

## Function Design

- Config passed as a dataclass (`RunConfig`) rather than many keyword args
- Backend always first parameter in phase functions: `phase_fix(backend: Backend, target_dir: Path, ...)`
- Phase functions return meaningful types: `phase_test_and_heal` returns `tuple[bool, int]` (passed, retries)
- `run()` in `daydream/runner.py` always returns `int` exit code
- Keyword-only arguments enforced via `*` in signatures for safety-critical params (e.g., `phase_parse_feedback(backend, cwd, *, input_path=None)`)

## Protocol Pattern (Backend)

The `Backend` protocol in `daydream/backends/__init__.py` uses structural typing (`Protocol`):
- Three required methods: `execute()`, `cancel()`, `format_skill_invocation()`
- `execute()` is an `AsyncIterator[AgentEvent]` (not `async def`)
- Implementations: `ClaudeBackend` (`daydream/backends/claude.py`), `CodexBackend` (`daydream/backends/codex.py`)
- Mock backends in tests implement the same three-method interface inline (no base class required)

## Async Patterns

- `anyio.run(run, config)` is the top-level async entry point in `daydream/cli.py`
- `anyio.create_task_group()` used for parallel operations in `daydream/phases.py`
- `asyncio_mode = "auto"` in pytest means `async def test_*` functions run automatically without explicit `@pytest.mark.asyncio` in most files (though explicit marks are also used)
- Async generators used for backend `execute()` — callers use `async for event in backend.execute(...)`

---

*Convention analysis: 2026-04-26*
