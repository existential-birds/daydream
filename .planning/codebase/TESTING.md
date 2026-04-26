# Testing

**Analysis Date:** 2026-04-26

## Framework

**Test Runner:**
- pytest 9.0.3 — declared in `pyproject.toml` under `[project.optional-dependencies]` dev section
- pytest-asyncio 1.3.0 — async test support, configured via `[tool.pytest.ini_options]`

**Configuration (`pyproject.toml`):**

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```

- `asyncio_mode = "auto"` means `async def test_*` functions run automatically without an explicit `@pytest.mark.asyncio` decorator (though many tests still mark explicitly for clarity)
- `asyncio_default_fixture_loop_scope = "function"` ensures each test gets a fresh event loop, preventing cross-test contamination
- No `pytest.ini`, `setup.cfg`, or `conftest.py`-level marker registrations

**Run Commands:**
- `make test` → `uv run pytest -v`
- Direct: `uv run pytest tests/test_phases.py::test_phase_test_and_heal_fix_uses_fresh_context -v`

## Test Layout

**Location:** `tests/` at repo root — flat structure; no nested test packages.

**File-per-module convention:** One test file per source module under test:

| Source module | Test file |
|---|---|
| `daydream/cli.py` | `tests/test_cli.py` |
| `daydream/agent.py` | `tests/test_agent_detect_success.py` |
| `daydream/runner.py` | `tests/test_runner.py` |
| `daydream/phases.py` | `tests/test_phases.py`, `tests/test_phase_parse_input_path.py` |
| `daydream/backends/__init__.py` | `tests/test_backends_init.py` |
| `daydream/backends/claude.py` | `tests/test_backend_claude.py` |
| `daydream/backends/codex.py` | `tests/test_backend_codex.py` |
| `daydream/exploration.py` | `tests/test_exploration.py` |
| `daydream/exploration_runner.py` | `tests/test_exploration_runner.py` |
| `daydream/tree_sitter_index.py` | `tests/test_tree_sitter_index.py` |
| `daydream/pr_review.py` | `tests/test_pr_review.py` |
| `daydream/ui.py` | `tests/test_ui.py` |
| `daydream/deep/artifacts.py` | `tests/test_deep_artifacts.py` |
| `daydream/deep/dedup.py` | `tests/test_deep_dedup.py` |
| `daydream/deep/detection.py` | `tests/test_deep_detection.py` |
| `daydream/deep/orchestrator.py` | `tests/test_deep_orchestrator.py`, `tests/test_deep_fanout.py`, `tests/test_deep_integration.py`, `tests/test_deep_merge.py` |
| `daydream/deep/prompts.py` | `tests/test_deep_prompts.py` |
| (cross-cutting) | `tests/test_integration.py`, `tests/test_loop.py` |

**Test Counts:**
- ~343 test functions across 25 test files (per `grep -E '^(async )?def test_' tests/*.py`)
- Largest test modules: `tests/test_deep_orchestrator.py` (936 lines), `tests/test_integration.py` (817 lines), `tests/test_pr_review.py` (719 lines), `tests/test_loop.py` (500 lines)

## Test Function Patterns

**Naming:**
- `def test_<behavior>(...)` — `snake_case`, descriptive of the asserted behavior, not the function under test
- Example: `test_phase_test_and_heal_fix_uses_fresh_context`, `test_detect_stacks_routes_python_to_python_review`

**Async tests:**
- `async def test_<name>(tmp_path, monkeypatch)` is the dominant form
- Some files use the explicit `@pytest.mark.asyncio` decorator (`tests/test_phases.py`); both work because `asyncio_mode = "auto"`

**Standard fixtures used:**
- `tmp_path` (pytest built-in) — temp directory for git repos, output files, fake targets
- `monkeypatch` (pytest built-in) — used in 90%+ of tests to stub UI helpers, console, prompt input, and module attributes
- `capsys` / `capfd` — occasional use for asserting on stdout/stderr where `monkeypatch`-based UI silencing would lose information

## Shared Fixtures

Defined in `tests/conftest.py`:

**`exploration_context_fixture`:**
- Returns a populated `ExplorationContext` with one modified file, one convention, one dependency edge
- Used by Phase 03 review-integration tests to verify prompt-injection behavior

**`exploration_dir_fixture(tmp_path, exploration_context_fixture)`:**
- Writes the exploration context to a temp dir using `ExplorationContext.write_to_dir()`
- Used when phase prompt builders need a real exploration directory on disk

**`multi_stack_target(tmp_path)`:**
- Builds a fully-initialized git repo with two commits (`init` on `main`, `change` on `feature`)
- Repo contains a Python file (`api.py`), a TSX file (`App.tsx`), and a Markdown file (`README.md`) — exercises the multi-stack routing path
- Used by deep-mode orchestrator and integration tests

## Mocking Patterns

**`monkeypatch` is the dominant mocking mechanism.** No `unittest.mock`, no `pytest-mock` — direct attribute replacement on imported modules.

**Inline mock backend pattern** — small classes defined inside the test function implementing the `Backend` protocol (`execute`, `cancel`, `format_skill_invocation`):

```python
class FreshContextBackend:
    async def execute(self, cwd, prompt, output_schema=None, continuation=None, agents=None, max_turns=None):
        nonlocal call_count
        call_count += 1
        captured_prompts.append(prompt)
        captured_continuations.append(continuation)
        if call_count == 1:
            yield TextEvent(text="1 failed, 0 passed")
            yield ResultEvent(structured_output=None, continuation=token)
        # ...

    async def cancel(self):
        pass

    def format_skill_invocation(self, skill_key, args=""):
        return f"/{skill_key}"
```

This pattern appears in `tests/test_phases.py`, `tests/test_integration.py`, `tests/test_deep_orchestrator.py`, `tests/test_deep_fanout.py`, `tests/test_deep_merge.py`, `tests/test_phase_parse_input_path.py`, `tests/test_pr_review.py`, and others.

**UI silencing via `monkeypatch.setattr`** — every test that invokes a phase stubs the print helpers and the console so test output stays clean and assertions don't need to scrape Rich rendering:

```python
monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
monkeypatch.setattr("daydream.phases.print_success", lambda *a, **kw: None)
monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)
monkeypatch.setattr("daydream.phases.print_menu", lambda *a, **kw: None)
monkeypatch.setattr("daydream.phases.print_error", lambda *a, **kw: None)
monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())
```

**Console capture pattern** — when output content matters, the test redirects to an in-memory `StringIO` Console:

```python
output = StringIO()
test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
monkeypatch.setattr("daydream.agent.console", test_console)
```

Used in `tests/test_integration.py` to verify Rich rendering of cost panels, summaries, etc.

**Prompt input mocking:**

```python
choices = iter(["2"])
monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: next(choices, "3"))
```

Drives the interactive retry/fix loop in `phase_test_and_heal` from a deterministic sequence.

**Real git repos in `tmp_path`:**
- Tests build genuine git repos via `subprocess.run(["git", "init", ...], ...)` rather than mocking `subprocess`
- See `multi_stack_target` fixture and `tests/test_integration.py` for the canonical setup
- All `subprocess.run` calls use `# noqa: S603 # noqa: S607` because args are hardcoded and not user-supplied

## Recorded Output Fixtures

**JSONL fixtures for `CodexBackend`** — `tests/fixtures/codex_jsonl/`:
- `simple_text.jsonl` — basic text streaming
- `output_text_blocks.jsonl` — multi-block text output
- `streamed_structured_output.jsonl` — structured-output streaming
- `structured_output.jsonl` — structured-output completion
- `tool_use.jsonl` — tool-call message replay
- `toplevel_text.jsonl` — top-level text events
- `turn_completed_result.jsonl` — turn-completed result envelope
- `turn_failed.jsonl` — turn-failed error path

These let `tests/test_backend_codex.py` exercise the codex CLI parsing without spawning the actual `codex` subprocess.

**Diff fixtures** — `tests/fixtures/diffs/`:
- `python_multifile.diff`, `typescript_multifile.diff`, `go_multifile.diff`, `rust_multifile.diff`, `trivial_single.diff`
- Used by exploration and stack-detection tests to assert routing without recreating diffs in code

**Deep-mode fixtures** — `tests/fixtures/deep/`:
- Currently only contains a `README.md`; no replay artifacts yet

## Coverage Approach

- **No coverage tool configured.** No `pytest-cov` in dependencies; no `.coveragerc`; no coverage gate in CI
- Coverage is enforced implicitly via the file-per-module convention plus the pre-push hook running the full suite

## CI / Pre-Push Hooks

**Pre-push hook (`scripts/hooks/pre-push`)** runs the full check suite locally before every push:

```bash
uv run ruff check daydream    # Lint
uv run mypy daydream          # Type check
uv run pytest -v              # Full test suite
```

Install with `make hooks` (symlinks `scripts/hooks/pre-push` into `.git/hooks/`).

**GitHub Actions (`.github/workflows/`)** mirror the pre-push hook on every push and PR to `main`:
- Ubuntu runner, Python 3.12, uv with cache enabled
- Steps: `uv sync` → `uv run ruff check daydream` → `uv run mypy daydream` → `uv run pytest -v`

## Test Isolation

- `daydream/agent.py` exposes `reset_state()` to restore the singleton `_state = AgentState()` defaults; called explicitly in tests that mutate global state
- `asyncio_default_fixture_loop_scope = "function"` ensures async fixtures are scoped per-test (no event-loop leakage)
- `tmp_path` ensures each test gets a unique directory; tests never write to the repo root

## Anti-Patterns to Avoid

**Importing `_state` from `daydream/agent.py` directly:**
- The module-level singleton must only be touched via setter/getter functions (`set_quiet_mode`, `get_model`, etc.)
- Direct field writes bypass the `reset_state()` contract and leak across tests

**Calling `ClaudeSDKClient` or the real `codex` CLI from tests:**
- All backend interaction in tests goes through inline mock classes that implement the `Backend` protocol
- Real `claude-agent-sdk` calls are never made during the test suite

**Asserting on Rich-rendered output:**
- Prefer `monkeypatch`-stubbed print helpers and assert on captured arguments
- When real rendering matters, redirect the `daydream.agent.console` singleton to a `StringIO` and assert on the buffer string

---

*Testing analysis: 2026-04-26*
