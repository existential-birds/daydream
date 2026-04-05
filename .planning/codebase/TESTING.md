# Testing Patterns

**Analysis Date:** 2026-04-05

## Test Framework

**Runner:**
- pytest 8.x
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`
- `asyncio_mode = "auto"` — all async tests run automatically without explicit `@pytest.mark.asyncio` decorator (though it is still applied in some test files for clarity)
- `asyncio_default_fixture_loop_scope = "function"` — each test gets a fresh event loop

**Assertion Library:**
- pytest built-in assertions only (no separate assertion library)

**Run Commands:**
```bash
make test                    # Run all tests (uv run pytest -v)
uv run pytest -v             # Verbose output
uv run pytest tests/test_cli.py  # Run single file
make check                   # lint + typecheck + test
```

## Test File Organization

**Location:** Separate `tests/` directory at project root, not co-located with source.

**Naming:**
- `test_<module>.py` mirrors the module being tested: `test_cli.py` tests `cli.py`, `test_phases.py` tests `phases.py`
- `test_backends_init.py` tests `backends/__init__.py`
- `test_backend_claude.py`, `test_backend_codex.py` test concrete backend implementations
- `test_integration.py` for end-to-end flows
- `test_loop.py` for loop mode feature

**Structure:**
```
tests/
├── fixtures/
│   └── codex_jsonl/           # JSONL replay files for Codex backend
│       ├── simple_text.jsonl
│       ├── tool_use.jsonl
│       ├── structured_output.jsonl
│       └── ...
├── rlm/                       # (additional test resources)
├── test_backend_claude.py
├── test_backend_codex.py
├── test_backends_init.py
├── test_cli.py
├── test_integration.py
├── test_loop.py
└── test_phases.py
```

**Total tests:** 129 (as of analysis date)

## Test Structure

**Suite Organization:**
```python
# Async tests: decorated or auto-detected
@pytest.mark.asyncio
async def test_phase_test_and_heal_fix_uses_fresh_context(tmp_path, monkeypatch):
    """Test that fix-and-retry starts fresh (no continuation) with enriched prompt."""
    ...

# Sync test classes for related unit tests
class TestBuildFixPrompt:
    """Tests for _build_fix_prompt helper."""

    def test_short_output_included_fully(self):
        from daydream.phases import _build_fix_prompt
        ...
```

**Patterns:**
- Docstrings on every test function explaining the scenario being tested
- Setup via `monkeypatch` to silence UI functions; teardown implicit (monkeypatch auto-resets)
- Imports of tested functions done inside test body (not at module level) to keep patches applied before the module initializes
- `tmp_path` pytest fixture used extensively for filesystem tests

## Mocking

**Framework:** `monkeypatch` (pytest built-in) for most patching. `unittest.mock` (`AsyncMock`, `MagicMock`, `patch`) for Codex backend subprocess mocking.

**UI Silencing Pattern (used in nearly every test):**
```python
monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
monkeypatch.setattr("daydream.phases.print_success", lambda *a, **kw: None)
monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())
```

**Backend Injection Pattern (integration tests):**
```python
monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: MockBackend())
```

**User Prompt Mocking:**
```python
# Single response
monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")

# Sequence of responses
responses = iter(["No, it's a login page with OAuth", "y"])
monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: next(responses))

# Prompt trap (assert should not be called)
def runner_prompt_trap(*args, **kwargs):
    raise AssertionError("Should not prompt for skill selection in --ttt mode")
monkeypatch.setattr("daydream.runner.prompt_user", runner_prompt_trap)
```

**What to Mock:**
- All `daydream.ui` print functions in tests that don't exercise UI
- `daydream.runner.create_backend` to inject mock backends
- `daydream.phases.prompt_user` and `daydream.runner.prompt_user` for interactive flows
- `daydream.ui.time.sleep` to skip animation delays
- `asyncio.create_subprocess_exec` for Codex backend process tests

**What NOT to Mock:**
- `tmp_path` filesystem operations — tests write real files to pytest's temp dir
- Git subprocess calls in `_git_diff`, `_git_log`, `_git_branch` tests — these initialize real git repos
- Backend `execute()` method itself — use protocol-conformant mock classes instead

## Mock Backend Pattern

Tests define inline mock classes that implement the `Backend` protocol:

```python
class SomePurposeBackend:
    async def execute(self, cwd, prompt, output_schema=None, continuation=None):
        yield TextEvent(text="Expected response text")
        yield ResultEvent(structured_output={"issues": [...]}, continuation=None)

    async def cancel(self):
        pass

    def format_skill_invocation(self, skill_key, args=""):
        return f"/{skill_key}"
```

For multi-turn flows, `nonlocal call_count` tracks invocations:
```python
call_count = 0

class StatefulBackend:
    async def execute(self, cwd, prompt, output_schema=None, continuation=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield TextEvent(text="First response")
            yield ResultEvent(structured_output=None, continuation=token)
        else:
            yield TextEvent(text="Second response")
            yield ResultEvent(structured_output=None, continuation=None)
    ...
```

`MockBackend` (in `test_integration.py`) implements prompt-content-based routing for full-flow tests. `MockBackendWithEvents` accepts a pre-built event list for precise tool-panel rendering tests.

## Fixtures and Factories

**pytest Fixtures (in `test_integration.py`):**
```python
@pytest.fixture
def mock_backend(monkeypatch):
    """Patch create_backend to return MockBackend."""
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: MockBackend())
    return MockBackend

@pytest.fixture
def mock_ui(monkeypatch):
    """Patch UI functions that require user input."""
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *args, **kwargs: "n")
    monkeypatch.setattr("daydream.runner.prompt_user", lambda *args, **kwargs: "n")

@pytest.fixture
def target_project(tmp_path: Path) -> Path:
    """Create a minimal project structure for testing."""
    project = tmp_path / "test_project"
    project.mkdir()
    (project / "main.py").write_text("def hello():\n    return 'world'\n")
    (project / ".review-output.md").write_text("# Code Review\n...")
    return project
```

**JSONL Fixture Files (for Codex backend, in `tests/fixtures/codex_jsonl/`):**
- Each file contains newline-delimited JSON events matching the Codex CLI output format
- Helper `_make_mock_process(fixture_name)` loads the file and creates an `asyncio.Process` mock
- Files: `simple_text.jsonl`, `tool_use.jsonl`, `structured_output.jsonl`, `turn_failed.jsonl`, etc.

**Test Data:**
- Inline dictionaries matching the feedback item schema: `{"id": 1, "description": "...", "file": "...", "line": N}`
- Git repos initialized via `subprocess.run(["git", ...])` calls within test body using `tmp_path`

## Coverage

**Requirements:** Not formally enforced (no `--cov` in pytest config or Makefile)

**View Coverage:**
```bash
uv run pytest --cov=daydream --cov-report=term-missing
```

## Test Types

**Unit Tests:**
- Pure function tests: `TestBuildFixPrompt`, `test_parse_issue_selection_*`, event dataclass tests in `test_backends_init.py`
- CLI arg parsing: all tests in `test_cli.py` — set `sys.argv` via monkeypatch, call `_parse_args()` directly

**Integration Tests:**
- End-to-end `run()` calls in `test_integration.py` and `test_loop.py` with mock backends injected
- Phase function tests in `test_phases.py` with inline protocol-conformant backends
- Full `run_trust()` flow: `test_run_trust_full_flow`, `test_run_trust_does_not_prompt_for_skill`

**Backend Tests:**
- Claude SDK translation layer: `test_backend_claude.py` — mocks the SDK client via `monkeypatch.setattr`
- Codex JSONL parsing: `test_backend_codex.py` — replays fixture files through mock process

**UI Rendering Tests:**
- Tool panel lifecycle tests in `test_integration.py` using `StringIO` console capture
- ANSI stripping helper `strip_ansi()` defined in both `test_integration.py` and `test_loop.py` for assertion comparisons

## Common Patterns

**Async Testing:**
```python
@pytest.mark.asyncio
async def test_something(tmp_path, monkeypatch):
    result = await some_phase_function(backend, tmp_path)
    assert result == expected
```

**Error/Exit Testing:**
```python
with pytest.raises(SystemExit):
    _parse_args()  # should call parser.error()

with pytest.raises(ValueError, match="Unknown backend"):
    create_backend("invalid")
```

**Console Output Capture:**
```python
from io import StringIO
from rich.console import Console
from daydream.ui import NEON_THEME

output = StringIO()
test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
monkeypatch.setattr("daydream.agent.console", test_console)

await run_agent(backend, Path("/tmp"), "prompt")

plain_text = strip_ansi(output.getvalue())
assert "expected text" in plain_text
```

**Git Repo Setup (for filesystem-dependent tests):**
```python
env = {**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
       "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"}
subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
subprocess.run(["git", "checkout", "-b", "main"], cwd=tmp_path, capture_output=True)
(tmp_path / "file.txt").write_text("content")
subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, env=env)
```

## Pre-Push Hook

`scripts/hooks/pre-push` runs `make lint && make typecheck && make test` before every push. Install with `make hooks`.

---

*Testing analysis: 2026-04-05*
