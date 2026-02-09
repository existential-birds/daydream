# Continuous Loop Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `--loop` flag that repeats the review-parse-fix-test cycle until zero issues or max iterations reached.

**Architecture:** Extend `RunConfig` with `loop` and `max_iterations` fields. When `loop=True`, the runner wraps the existing phase calls in a `while` loop that accumulates stats across iterations. A new `print_iteration_divider()` in ui.py renders the iteration separator, and `SummaryData` gains two fields for loop reporting.

**Tech Stack:** Python, pytest, Rich (terminal UI)

---

### Task 1: Add `loop` and `max_iterations` to RunConfig

**Files:**
- Modify: `daydream/runner.py:65-78` (RunConfig dataclass fields)

**Step 1: Write the failing test**

Create `tests/test_loop.py`:

```python
"""Tests for continuous loop mode."""

import pytest

from daydream.runner import RunConfig


def test_runconfig_loop_defaults():
    config = RunConfig()
    assert config.loop is False
    assert config.max_iterations == 5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_loop.py::test_runconfig_loop_defaults -v`
Expected: FAIL with `AttributeError: ... has no attribute 'loop'`

**Step 3: Write minimal implementation**

In `daydream/runner.py`, add two fields to `RunConfig` after line 78 (`test_backend: str | None = None`):

```python
    loop: bool = False
    max_iterations: int = 5
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_loop.py::test_runconfig_loop_defaults -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_loop.py daydream/runner.py
git commit -m "feat(loop): add loop and max_iterations to RunConfig"
```

---

### Task 2: Add CLI flags `--loop` and `--max-iterations`

**Files:**
- Modify: `daydream/cli.py:198-250` (argument parsing and RunConfig construction)
- Test: `tests/test_cli.py`

**Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_loop_flag_default_off(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python"])
    config = _parse_args()
    assert config.loop is False
    assert config.max_iterations == 5


def test_loop_flag_enabled(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python", "--loop"])
    config = _parse_args()
    assert config.loop is True


def test_max_iterations_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python", "--loop", "--max-iterations", "10",
    ])
    config = _parse_args()
    assert config.max_iterations == 10
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::test_loop_flag_default_off tests/test_cli.py::test_loop_flag_enabled tests/test_cli.py::test_max_iterations_flag -v`
Expected: FAIL — `_parse_args()` doesn't recognize `--loop`

**Step 3: Write minimal implementation**

In `daydream/cli.py`, add two new arguments after the `--model` argument block (after line 203):

```python
    parser.add_argument(
        "--loop",
        action="store_true",
        default=False,
        help="Repeat review-fix-test cycle until zero issues or max iterations",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        metavar="N",
        dest="max_iterations",
        help="Maximum loop iterations (default: 5, only meaningful with --loop)",
    )
```

In the `return RunConfig(...)` block (lines 235-250), add the new fields:

```python
        loop=args.loop,
        max_iterations=args.max_iterations,
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add daydream/cli.py tests/test_cli.py
git commit -m "feat(loop): add --loop and --max-iterations CLI flags"
```

---

### Task 3: Add CLI validation for `--loop` conflicts

**Files:**
- Modify: `daydream/cli.py:207-234` (validation block)
- Test: `tests/test_cli.py`

**Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_loop_review_only_conflict(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python", "--loop", "--review-only",
    ])
    with pytest.raises(SystemExit):
        _parse_args()


def test_loop_start_at_conflict(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python", "--loop", "--start-at", "fix",
    ])
    with pytest.raises(SystemExit):
        _parse_args()


def test_max_iterations_without_loop_accepted(monkeypatch, capsys):
    """--max-iterations without --loop is accepted but prints a warning."""
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python", "--max-iterations", "3",
    ])
    config = _parse_args()
    assert config.max_iterations == 3
    assert config.loop is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::test_loop_review_only_conflict tests/test_cli.py::test_loop_start_at_conflict tests/test_cli.py::test_max_iterations_without_loop_accepted -v`
Expected: First two FAIL (no SystemExit raised), third should PASS already

**Step 3: Write minimal implementation**

In `daydream/cli.py`, add validation after the existing `--start-at` / `--review-only` check (after line 209):

```python
    # Validate --loop mutual exclusions
    if args.loop:
        if args.review_only:
            parser.error("--loop and --review-only are mutually exclusive")
        if args.start_at != "review":
            parser.error("--loop requires starting at review phase (incompatible with --start-at)")
```

And before the `return RunConfig(...)`, add the warning:

```python
    # Warn if --max-iterations without --loop
    if args.max_iterations != 5 and not args.loop:
        import warnings
        warnings.warn("--max-iterations has no effect without --loop", stacklevel=1)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add daydream/cli.py tests/test_cli.py
git commit -m "feat(loop): validate --loop flag conflicts"
```

---

### Task 4: Add `print_iteration_divider()` to ui.py

**Files:**
- Modify: `daydream/ui.py` (add new function near the phase hero section)
- Test: `tests/test_loop.py`

**Step 1: Write the failing test**

Append to `tests/test_loop.py`:

```python
import re
from io import StringIO

from rich.console import Console

from daydream.ui import NEON_THEME, print_iteration_divider

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def test_print_iteration_divider():
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80, theme=NEON_THEME)
    print_iteration_divider(test_console, 2, 5)
    plain = strip_ansi(output.getvalue())
    assert "Iteration 2 of 5" in plain
    assert "━" in plain
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_loop.py::test_print_iteration_divider -v`
Expected: FAIL with `ImportError: cannot import name 'print_iteration_divider'`

**Step 3: Write minimal implementation**

Add to `daydream/ui.py`, right before the Summary Component section (before line 1996):

```python
def print_iteration_divider(console: Console, iteration: int, max_iterations: int) -> None:
    """Print an iteration divider for loop mode.

    Args:
        console: Rich Console instance for output.
        iteration: Current iteration number (1-based).
        max_iterations: Maximum iterations allowed.

    """
    console.print()
    label = f" Iteration {iteration} of {max_iterations} "
    console.print(Rule(label, style=STYLE_PURPLE))
    console.print()
```

Also make sure `Rule` is imported from `rich.rule` at the top of ui.py. Check if it's already imported — if not, add:

```python
from rich.rule import Rule
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_loop.py::test_print_iteration_divider -v`
Expected: PASS

**Step 5: Commit**

```bash
git add daydream/ui.py tests/test_loop.py
git commit -m "feat(loop): add print_iteration_divider to ui"
```

---

### Task 5: Add loop fields to SummaryData and update `print_summary()`

**Files:**
- Modify: `daydream/ui.py:2001-2068` (SummaryData and print_summary)
- Test: `tests/test_loop.py`

**Step 1: Write the failing tests**

Append to `tests/test_loop.py`:

```python
from daydream.ui import SummaryData, print_summary


def test_summary_data_loop_fields_default():
    data = SummaryData(
        skill="python", target="/tmp", feedback_count=3,
        fixes_applied=3, test_retries=0, tests_passed=True,
    )
    assert data.loop_mode is False
    assert data.iterations_used == 1


def test_summary_loop_mode_shows_iterations():
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80, theme=NEON_THEME)
    data = SummaryData(
        skill="python", target="/tmp", feedback_count=8,
        fixes_applied=8, test_retries=1, tests_passed=True,
        loop_mode=True, iterations_used=3,
    )
    print_summary(test_console, data)
    plain = strip_ansi(output.getvalue())
    assert "Iterations" in plain
    assert "3" in plain
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_loop.py::test_summary_data_loop_fields_default tests/test_loop.py::test_summary_loop_mode_shows_iterations -v`
Expected: FAIL — SummaryData doesn't have `loop_mode` or `iterations_used`

**Step 3: Write minimal implementation**

In `daydream/ui.py`, add two fields to the `SummaryData` dataclass (after `review_only: bool = False` on line 2022):

```python
    loop_mode: bool = False
    iterations_used: int = 1
```

In `print_summary()`, add iteration row when `loop_mode=True`. In the `else` branch (full mode, line 2056), add this before the "Fixes Applied" row:

```python
        if data.loop_mode:
            table.add_row("Iterations", str(data.iterations_used))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_loop.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add daydream/ui.py tests/test_loop.py
git commit -m "feat(loop): add loop_mode and iterations_used to SummaryData"
```

---

### Task 6: Implement the loop logic in runner.py (loop=True path)

**Files:**
- Modify: `daydream/runner.py:344-419` (the phase execution block inside `run()`)
- Modify: `daydream/runner.py` imports (add `print_iteration_divider`)
- Test: `tests/test_loop.py`

This is the core task. The existing single-pass flow (lines 344-419) gets wrapped in a loop when `config.loop=True`.

**Step 1: Write the failing tests**

Append to `tests/test_loop.py`:

```python
from pathlib import Path
from typing import Any

from daydream.backends import ResultEvent, TextEvent
from daydream.runner import RunConfig, run


class LoopMockBackend:
    """Mock backend that returns different results on successive calls.

    Tracks call count and uses prompt content to determine responses.
    The `review_results` list controls what phase_parse_feedback returns
    on each iteration (list of issue lists, one per iteration).
    """

    def __init__(self, review_results: list[list[dict[str, Any]]], tests_pass: bool = True):
        self._review_results = review_results
        self._tests_pass = tests_pass
        self._parse_call = 0
        self.call_log: list[str] = []

    async def execute(self, cwd, prompt, output_schema=None, continuation=None):
        prompt_lower = prompt.lower()
        self.call_log.append(prompt_lower[:80])

        if "beagle-" in prompt_lower and "review" in prompt_lower:
            yield TextEvent(text="Review complete.")
            yield ResultEvent(structured_output=None, continuation=None)
        elif "extract" in prompt_lower and "json" in prompt_lower:
            issues = (
                self._review_results[self._parse_call]
                if self._parse_call < len(self._review_results)
                else []
            )
            self._parse_call += 1
            yield TextEvent(text="Parsed.")
            yield ResultEvent(structured_output={"issues": issues}, continuation=None)
        elif "fix this issue" in prompt_lower:
            yield TextEvent(text="Fixed.")
            yield ResultEvent(structured_output=None, continuation=None)
        elif "test suite" in prompt_lower or "run the project" in prompt_lower:
            if self._tests_pass:
                yield TextEvent(text="All 1 tests passed. 0 failed.")
            else:
                yield TextEvent(text="1 test failed.")
            yield ResultEvent(structured_output=None, continuation=None)
        elif "commit-push" in prompt_lower:
            yield TextEvent(text="Committed.")
            yield ResultEvent(structured_output=None, continuation=None)
        else:
            yield TextEvent(text="OK")
            yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self):
        pass

    def format_skill_invocation(self, skill_key, args=""):
        return f"/{skill_key}" + (f" {args}" if args else "")


@pytest.fixture
def loop_target(tmp_path: Path) -> Path:
    project = tmp_path / "loop_project"
    project.mkdir()
    (project / "main.py").write_text("def hello():\n    return 'world'\n")
    (project / ".review-output.md").write_text("# Review\n\n1. Issue in main.py:1\n")
    return project


@pytest.fixture
def mock_ui_loop(monkeypatch):
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "n")
    monkeypatch.setattr("daydream.runner.prompt_user", lambda *a, **kw: "n")


@pytest.mark.asyncio
async def test_loop_exits_on_zero_issues(loop_target, mock_ui_loop, monkeypatch):
    """Issues on iteration 1, zero on iteration 2 -> exits 0."""
    issue = {"id": 1, "description": "Add type hints", "file": "main.py", "line": 1}
    backend = LoopMockBackend(review_results=[[issue], []])

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=5,
    )
    exit_code = await run(config)

    assert exit_code == 0
    assert backend._parse_call == 2  # called parse twice


@pytest.mark.asyncio
async def test_loop_respects_max_iterations(loop_target, mock_ui_loop, monkeypatch):
    """Always returns issues -> stops at max_iterations, exits 1."""
    issue = {"id": 1, "description": "Persistent issue", "file": "main.py", "line": 1}
    # 3 iterations, all return the same issue
    backend = LoopMockBackend(review_results=[[issue], [issue], [issue]])

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=3,
    )
    exit_code = await run(config)

    assert exit_code == 1
    assert backend._parse_call == 3


@pytest.mark.asyncio
async def test_loop_stops_on_test_failure(loop_target, mock_ui_loop, monkeypatch):
    """Tests fail mid-loop -> stops immediately, exits 1."""
    issue = {"id": 1, "description": "Issue", "file": "main.py", "line": 1}
    backend = LoopMockBackend(review_results=[[issue], [issue]], tests_pass=False)

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=5,
    )
    exit_code = await run(config)

    assert exit_code == 1
    assert backend._parse_call == 1  # stopped after first iteration


@pytest.mark.asyncio
async def test_loop_accumulates_stats(loop_target, mock_ui_loop, monkeypatch):
    """Stats accumulate across iterations."""
    issue1 = {"id": 1, "description": "Issue A", "file": "main.py", "line": 1}
    issue2 = {"id": 2, "description": "Issue B", "file": "main.py", "line": 2}
    # Iteration 1: 2 issues, Iteration 2: 1 issue, Iteration 3: 0 issues
    backend = LoopMockBackend(review_results=[[issue1, issue2], [issue1], []])

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

    # Capture summary data
    captured_summary = {}

    original_print_summary = None
    import daydream.runner as runner_mod
    original_print_summary = runner_mod.print_summary

    def capture_summary(console, data):
        captured_summary["feedback_count"] = data.feedback_count
        captured_summary["fixes_applied"] = data.fixes_applied
        captured_summary["iterations_used"] = data.iterations_used
        captured_summary["loop_mode"] = data.loop_mode
        original_print_summary(console, data)

    monkeypatch.setattr("daydream.runner.print_summary", capture_summary)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=5,
    )
    exit_code = await run(config)

    assert exit_code == 0
    assert captured_summary["feedback_count"] == 3  # 2 + 1
    assert captured_summary["fixes_applied"] == 3  # 2 + 1
    assert captured_summary["iterations_used"] == 3
    assert captured_summary["loop_mode"] is True


@pytest.mark.asyncio
async def test_loop_false_single_pass(loop_target, mock_ui_loop, monkeypatch):
    """loop=False behaves identically to existing single-pass flow."""
    issue = {"id": 1, "description": "Issue", "file": "main.py", "line": 1}
    backend = LoopMockBackend(review_results=[[issue]])

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=False,
    )
    exit_code = await run(config)

    assert exit_code == 0
    assert backend._parse_call == 1  # single pass only
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_loop.py::test_loop_exits_on_zero_issues -v`
Expected: FAIL — runner.py doesn't have loop logic yet

**Step 3: Write minimal implementation**

In `daydream/runner.py`, add `print_iteration_divider` to the imports from `daydream.ui`:

```python
from daydream.ui import (
    SummaryData,
    phase_subtitle,
    print_dim,
    print_error,
    print_info,
    print_iteration_divider,
    print_menu,
    print_phase_hero,
    print_skipped_phases,
    print_success,
    print_summary,
    print_warning,
    prompt_user,
)
```

Then replace the phase execution block in `run()`. The current code at lines 344-419 becomes:

```python
        feedback_items: list[dict[str, Any]] = []
        fixes_applied = 0
        test_retries = 0
        tests_passed = True
        iteration = 0

        if config.loop:
            # --- Loop mode: repeat review-parse-fix-test ---
            while iteration < config.max_iterations:
                iteration += 1

                if iteration > 1:
                    print_iteration_divider(console, iteration, config.max_iterations)

                # Phase 1: Review
                assert skill is not None, "skill must be set when starting at review phase"
                try:
                    await phase_review(review_backend, target_dir, skill)
                except MissingSkillError as e:
                    _print_missing_skill_error(e.skill_name)
                    return 1

                # Phase 2: Parse feedback
                try:
                    items = await phase_parse_feedback(review_backend, target_dir)
                except ValueError as exc:
                    _log_debug(f"[PHASE2_ERROR] {exc}\n")
                    print_error(console, "Parse Failed", "Failed to parse feedback. Exiting.")
                    return 1

                if not items:
                    print_info(console, f"Clean review on iteration {iteration}")
                    break

                feedback_items.extend(items)

                # Phase 3: Fix
                print_phase_hero(console, "HEAL", phase_subtitle("HEAL"))
                for i, item in enumerate(items, 1):
                    await phase_fix(fix_backend, target_dir, item, i, len(items))
                    fixes_applied += 1

                # Phase 4: Test
                tests_passed, retries = await phase_test_and_heal(test_backend, target_dir)
                test_retries += retries

                if not tests_passed:
                    print_warning(console, f"Tests failed on iteration {iteration}")
                    break

            else:
                # while loop exhausted without break — max iterations reached
                if feedback_items:
                    print_warning(
                        console,
                        f"Reached max iterations ({config.max_iterations}), "
                        f"{len(feedback_items)} issues found across all iterations",
                    )

        else:
            # --- Single-pass mode (existing behavior) ---

            # Phase 1: Review
            if config.start_at == "review":
                assert skill is not None, "skill must be set when starting at review phase"
                try:
                    await phase_review(review_backend, target_dir, skill)
                except MissingSkillError as e:
                    _print_missing_skill_error(e.skill_name)
                    return 1

            # Phase 2: Parse feedback
            if config.start_at in ("review", "parse", "fix"):
                try:
                    feedback_items = await phase_parse_feedback(review_backend, target_dir)
                except ValueError as exc:
                    _log_debug(f"[PHASE2_ERROR] {exc}\n")
                    print_error(console, "Parse Failed", "Failed to parse feedback. Exiting.")
                    return 1

            # Review-only exit
            if config.review_only:
                print_summary(
                    console,
                    SummaryData(
                        skill=skill or "N/A",
                        target=str(target_dir),
                        feedback_count=len(feedback_items),
                        fixes_applied=0,
                        test_retries=0,
                        tests_passed=True,
                        review_only=True,
                    ),
                )
                return 0

            # Phase 3: Fix
            if config.start_at in ("review", "parse", "fix"):
                if feedback_items:
                    print_phase_hero(console, "HEAL", phase_subtitle("HEAL"))
                    for i, item in enumerate(feedback_items, 1):
                        await phase_fix(fix_backend, target_dir, item, i, len(feedback_items))
                        fixes_applied += 1
                else:
                    print_info(console, "No feedback items found, skipping fix phase")

            # Phase 4: Test
            tests_passed, test_retries = await phase_test_and_heal(test_backend, target_dir)
            iteration = 1

        # Print summary
        print_summary(
            console,
            SummaryData(
                skill=skill or "N/A",
                target=str(target_dir),
                feedback_count=len(feedback_items),
                fixes_applied=fixes_applied,
                test_retries=test_retries,
                tests_passed=tests_passed,
                loop_mode=config.loop,
                iterations_used=iteration if config.loop else 1,
            ),
        )

        # Commit if tests passed
        if tests_passed:
            await phase_commit_push(review_backend, target_dir)

            if cleanup_enabled:
                review_output_path = target_dir / REVIEW_OUTPUT_FILE
                if review_output_path.exists():
                    review_output_path.unlink()
                    print_success(console, f"Cleaned up {REVIEW_OUTPUT_FILE}")

            return 0
        else:
            return 1
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_loop.py -v`
Expected: ALL PASS

**Step 5: Run all existing tests to verify no regressions**

Run: `pytest -v`
Expected: ALL PASS (single-pass behavior unchanged, existing integration tests still pass)

**Step 6: Commit**

```bash
git add daydream/runner.py tests/test_loop.py
git commit -m "feat(loop): implement continuous loop mode in runner"
```

---

### Task 7: Run full CI checks

**Files:** None (verification only)

**Step 1: Run linter**

Run: `make lint`
Expected: PASS — no new lint errors

**Step 2: Run type checker**

Run: `make typecheck`
Expected: PASS — all types consistent

**Step 3: Run full test suite**

Run: `make test`
Expected: ALL PASS

**Step 4: Commit any fixes if needed**

If any lint/type issues found, fix and commit:

```bash
git add -u
git commit -m "fix: lint and type fixes for loop mode"
```
