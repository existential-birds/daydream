---
phase: 05-test-hardening-documentation
reviewed: 2026-04-28T22:15:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - tests/test_multi_turn_tokens.py
  - tests/test_subagent_shapes.py
  - tests/test_trajectory.py
  - tests/test_redaction.py
  - CHANGELOG.md
  - CLAUDE.md
  - README.md
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: issues_found
---

# Phase 5: Code Review Report

**Reviewed:** 2026-04-28T22:15:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Phase 5 adds four test files covering multi-turn token semantics, subagent trajectory shapes, core trajectory recorder behavior, and redaction rules, plus documentation updates to CHANGELOG.md, CLAUDE.md, and README.md. Test quality is generally high: `test_multi_turn_tokens.py` exercises the real `run_agent()` production path end-to-end, `test_trajectory.py` and `test_subagent_shapes.py` drive the real `TrajectoryRecorder` and `Invocation` code with mock backends (not over-mocked), and `test_redaction.py` validates the `Redactor` against ATIF model types. Tests verify behavioral predicates rather than snapshot equality. Documentation updates in CLAUDE.md reflect the new trajectory system accurately.

The README architecture diagram is stale -- it was not updated to reflect `trajectory.py`, `atif/`, `deep/`, `exploration.py`, or `pr_review.py`. Two test findings relate to a fragile global monkeypatch and a test gap around signal-handler visibility for forked recorders.

## Warnings

### WR-01: README architecture tree is stale -- missing 6+ modules added since v0.5.0

**File:** `README.md:209-222`
**Issue:** The architecture tree diagram lists only the original 7 modules (`cli.py`, `runner.py`, `phases.py`, `agent.py`, `ui.py`, `config.py`, `prompts/`, `backends/`). It is missing `trajectory.py`, `atif/`, `deep/`, `exploration.py`, `exploration_runner.py`, `tree_sitter_index.py`, and `pr_review.py` -- all of which are documented in CLAUDE.md's Component Responsibilities table and are core to the current codebase. The rest of the README (features list, output files table, trajectory section) was updated correctly, making this diagram the single inconsistency. Users reading the README get an incomplete picture of the package structure.

**Fix:**
```text
daydream/
├── cli.py                # Entry point, argument parsing, signal handling
├── runner.py             # Main orchestration (standard + PR feedback flows)
├── phases.py             # Core phases (review, parse, fix, test) + PR feedback helpers
├── agent.py              # Agent event consumer and helper functions
├── trajectory.py         # ATIF v1.6 trajectory recorder, redaction, ContextVar propagation
├── ui.py                 # Neon terminal UI components (Rich-based)
├── config.py             # Configuration constants
├── exploration.py        # Pre-scan codebase context types
├── exploration_runner.py # Exploration orchestration and specialist fan-out
├── tree_sitter_index.py  # Static import resolution for exploration
├── pr_review.py          # Post review findings as inline GitHub PR comments
├── prompts/              # Review system prompt templates
├── atif/                 # Vendored ATIF v1.6 models and validator (Apache-2.0)
├── deep/                 # Multi-stack deep review pipeline
└── backends/             # Backend abstraction layer
    ├── __init__.py       # Backend protocol, event types, create_backend() factory
    ├── claude.py         # Claude SDK backend
    └── codex.py          # OpenAI Codex CLI backend (JSONL event stream)
```

### WR-02: Global `Path.write_text` monkeypatch in `test_write_failure_degrades_with_warning` also poisons the recorder's own `__aexit__` write

**File:** `tests/test_trajectory.py:307-311`
**Issue:** The monkeypatch of `Path.write_text` is applied before `async with recorder:` is entered. When the `async with` block exits, `__aexit__` calls `_write()` which also calls `self.path.write_text(...)`. Because the monkeypatch is still active, the main trajectory write also fails with `PermissionError`. The `__aexit__` handler catches this and emits a "Trajectory write failed" warning, but the test only asserts the presence of that warning message -- it does not verify that the warning was emitted by `_write()` specifically (as intended) rather than by some other code path. The test passes today because `explicit_path` defaults to False, but it is testing an unintended double-failure rather than a clean single-failure scenario.

**Fix:** Apply the monkeypatch inside the `async with` block, after the invocation completes but before `__aexit__`:
```python
async def test_write_failure_degrades_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behavior C: PermissionError on write emits warning; run does not raise."""
    recorder = _make_recorder(tmp_path)
    warnings_emitted: list[str] = []

    def fake_print_warning(_console: Any, message: str) -> None:
        warnings_emitted.append(message)

    monkeypatch.setattr("daydream.trajectory.print_warning", fake_print_warning)

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="hi"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        # Patch AFTER steps are accumulated but BEFORE __aexit__ writes
        monkeypatch.setattr(
            Path,
            "write_text",
            lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")),
        )

    assert any("Trajectory write failed" in m for m in warnings_emitted)
```

### WR-03: No test coverage for `get_signal_recorder()` visibility of forked child recorders

**File:** `tests/test_trajectory.py:1029-1044`
**Issue:** `test_get_signal_recorder_returns_innermost_for_nested_recorders` tests nested `TrajectoryRecorder` instances (both go through `__aenter__` which pushes to `_ACTIVE_RECORDERS`). However, `_ForkCM.__aenter__` (line 804-817 of `trajectory.py`) does NOT push the child to `_ACTIVE_RECORDERS` -- it only sets the `ContextVar`. This means `get_signal_recorder()` called during a SIGINT inside a forked child's scope returns the parent, not the child. If this is intentional (partial flush cascades from parent), it should be documented and tested. If not, it is a production bug in `trajectory.py`. Either way, there is a test gap: no test verifies `get_signal_recorder()` behavior inside a `recorder.fork()` scope.

**Fix:** Add a test that documents the expected behavior:
```python
async def test_get_signal_recorder_inside_fork_returns_parent(tmp_path: Path) -> None:
    """Fork children are NOT pushed to _ACTIVE_RECORDERS; signal handler sees parent."""
    from daydream.trajectory import get_signal_recorder

    recorder = _make_recorder(tmp_path)
    async with recorder:
        assert get_signal_recorder() is recorder
        async with recorder.fork("fix-0") as child:
            # ContextVar sees child, but signal handler sees parent
            assert get_current_recorder() is child
            assert get_signal_recorder() is recorder  # NOT child
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                _observe_text_and_result(inv)
        assert get_signal_recorder() is recorder
```

## Info

### IN-01: Duplicate `MockBackend` + `_make_recorder` + `_read_trajectory` + `_observe_text_and_result` helpers across test files

**File:** `tests/test_multi_turn_tokens.py:48-105`, `tests/test_subagent_shapes.py:37-90`, `tests/test_trajectory.py:37-49,490-494`
**Issue:** Three test files define nearly identical `MockBackend` dataclasses, `_make_recorder()`, `_read_trajectory()`, and `_observe_text_and_result()` helpers. While test file independence is sometimes preferred over shared fixtures, these are byte-for-byte identical and would benefit from a shared `tests/helpers.py` or `conftest.py` fixture to reduce maintenance burden when the `Backend` protocol signature changes.
**Fix:** Extract shared test helpers into `tests/conftest.py` or a `tests/trajectory_helpers.py` module.

### IN-02: Substantial test overlap between `test_trajectory.py` and `test_subagent_shapes.py`

**File:** `tests/test_subagent_shapes.py:210-312`, `tests/test_trajectory.py:560-858`
**Issue:** Both files test step_id isolation across siblings (SUBA-08), parent metrics excluding children (SUBA-09), continuation appending to same file (SUBA-05), session_id inheritance (SUBA-06), and dispatch step refs. `test_subagent_shapes.py` was designed for scenario-level fork validation while `test_trajectory.py` has unit-level versions of the same. The overlap means changing fork behavior requires updating both files. Not a defect, but maintenance cost.
**Fix:** Consider having `test_subagent_shapes.py` focus exclusively on multi-child fan-out scenarios (3+ children, mixed descriptors) and removing the single-child SUBA-05/08/09 cases that are already thoroughly covered in `test_trajectory.py`.

---

_Reviewed: 2026-04-28T22:15:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
