# Coding Conventions

**Analysis Date:** 2026-04-05

## Naming Patterns

**Files:**
- `snake_case.py` for all modules: `cli.py`, `runner.py`, `phases.py`, `agent.py`
- Private helpers prefixed with underscore: `_parse_args`, `_resolve_backend`, `_log_debug`
- Test files prefixed with `test_`: `test_cli.py`, `test_integration.py`, `test_phases.py`

**Functions:**
- `snake_case` throughout: `run_agent`, `phase_review`, `detect_test_success`
- Private helpers start with `_`: `_signal_handler`, `_build_fix_prompt`, `_get_head_sha`
- Phase functions follow `phase_<name>` convention: `phase_review`, `phase_fix`, `phase_parse_feedback`, `phase_test_and_heal`
- Setter/getter pairs for module-level state: `set_debug_log`/`get_debug_log`, `set_quiet_mode`/`get_quiet_mode`

**Variables:**
- `snake_case` throughout
- Boolean flags use descriptive names: `cleanup_enabled`, `shutdown_requested`, `trust_the_technology`
- Type-annotated at declaration where possible

**Types / Classes:**
- `PascalCase` for all classes: `RunConfig`, `AgentState`, `MockBackend`, `ClaudeBackend`
- Event dataclasses use `<Name>Event` pattern: `TextEvent`, `ToolStartEvent`, `CostEvent`, `ResultEvent`
- Enums use `PascalCase` class name, `UPPER_SNAKE` members: `ReviewSkillChoice.PYTHON`
- Exception classes use `Error` suffix: `MissingSkillError`, `CodexError`
- Protocol classes named by role: `Backend`

## Code Style

**Formatting:**
- Ruff formatter (target: Python 3.12, line length: 120)
- Config in `pyproject.toml` under `[tool.ruff]`

**Linting:**
- Ruff with rule sets `E`, `F`, `I`, `W` (pycodestyle errors, pyflakes, isort, warnings)
- `# noqa: S603, S607` suppression for `subprocess.run` calls with hardcoded args — always accompanied by an inline comment explaining why it's safe
- `# noqa: E402` for necessary late imports in `__init__.py`

**Type Checking:**
- mypy with `ignore_missing_imports = true` (for external SDKs without stubs)
- Python 3.12 union syntax: `str | None`, `list[Backend]`, `dict[str, Any]`
- `from __future__ import annotations` used in `backends/__init__.py` for forward references
- `TYPE_CHECKING` guard for `AsyncIterator` import: `if TYPE_CHECKING: from collections.abc import AsyncIterator`
- Return types annotated on all public functions

## Import Organization

**Order (enforced by ruff/isort):**
1. Standard library (`import json`, `from pathlib import Path`)
2. Third-party (`import anyio`, `from rich.console import Console`)
3. Local package (`from daydream.agent import ...`, `from daydream.backends import ...`)

**Style:**
- Grouped imports from same module on separate lines, collected into single `from X import (...)` block
- No star imports
- No `__all__` except in `backends/__init__.py` which explicitly re-exports public API

**Path Aliases:**
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

**Patterns:**
- Raise typed exceptions for domain errors: `MissingSkillError`, `CodexError`, `ValueError`
- Caller catches at appropriate layer: `run()` in `runner.py` catches `MissingSkillError` and `ValueError` from phases
- Subprocess errors caught by type: `except (subprocess.SubprocessError, OSError):`
- JSON/external errors caught narrowly: `except (json.JSONDecodeError, FileNotFoundError):`
- `except Exception as e:` only at top-level CLI boundary in `cli.py`
- No silent swallowing — always log or re-raise

**subprocess.run:**
- Always called with `capture_output=True`, `text=True`, `timeout=N`, `shell=False`
- `# noqa: S603` comment on every call explaining args are not user-controlled

## Logging

**Framework:** Custom `_log_debug()` helper in `agent.py` — writes to a file handle when debug mode is on, no-ops otherwise.

**Patterns:**
- Prefixed log entries: `[PROMPT]`, `[TEXT]`, `[TOOL_USE]`, `[TOOL_RESULT]`, `[COST]`, `[SCHEMA_OK]`, `[WARN]`
- No `print()` statements in library code; all user-facing output via `daydream.ui` functions: `print_info`, `print_error`, `print_success`, `print_warning`
- Rich `Console` object (`console = create_console()`) is module-level singleton in `agent.py`

## Comments

**When to Comment:**
- Long comment blocks at module level explaining architectural decisions (e.g., singleton pattern explanation in `agent.py`)
- Inline `# noqa:` always followed by a reason comment
- Algorithm steps in complex flows labeled (e.g., `# Phase 1: Review`, `# Phase 2: Parse feedback`)
- No obvious/redundant comments

**Docstrings:**
- All public functions have Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections
- Private helpers (`_log_debug`) have single-line docstrings
- Dataclass docstrings include `Attributes:` section listing all fields
- Module-level docstring in every file; `config.py` includes an `Exports:` section

## Function Design

**Size:** Phase functions are large (50-150 lines) because they orchestrate I/O — internal helpers extracted when logic is reusable (e.g., `_build_fix_prompt`, `_parse_issue_selection`)

**Async:** All agent-calling functions are `async def`. Pure utility functions are sync.

**Parameters:**
- Config passed as a dataclass (`RunConfig`) rather than many keyword args
- Backend always first parameter in phase functions: `phase_fix(backend: Backend, target_dir: Path, ...)`

**Return Values:**
- Phase functions return meaningful types: `phase_test_and_heal` returns `tuple[bool, int]` (passed, retries)
- `run()` returns `int` exit code always

## Protocol Pattern (Backend)

```python
class Backend(Protocol):
    def execute(self, cwd, prompt, output_schema=None, continuation=None) -> AsyncIterator[AgentEvent]: ...
    async def cancel(self) -> None: ...
    def format_skill_invocation(self, skill_key: str, args: str = "") -> str: ...
```

Concrete implementations: `daydream/backends/claude.py` (`ClaudeBackend`), `daydream/backends/codex.py` (`CodexBackend`). Factory: `create_backend(name, model)` in `daydream/backends/__init__.py`.

---

*Convention analysis: 2026-04-05*
