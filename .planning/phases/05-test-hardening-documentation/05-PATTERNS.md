# Phase 5: Test Hardening + Documentation - Pattern Map

**Mapped:** 2026-04-28
**Files analyzed:** 10 (4 test files to audit/extend, 2 potential new test files, 4 doc files to update)
**Analogs found:** 8 / 10

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `tests/test_trajectory.py` | test | CRUD (recorder lifecycle) | self (audit-in-place) | exact |
| `tests/test_redaction.py` | test | transform (regex scrub) | self (audit-in-place) | exact |
| `tests/test_atif_vendor_smoke.py` | test | request-response (validate) | self (audit-in-place) | exact |
| `tests/test_trajectory_fixture.py` | test | event-driven (ContextVar leak) | self (audit-in-place) | exact |
| `tests/test_multi_turn_tokens.py` (new, TEST-06) | test | request-response (mock backend) | `tests/test_agent_recorder_integration.py` | exact |
| `tests/test_subagent_shapes.py` (new, TEST-07) | test | event-driven (fork/sibling) | `tests/test_trajectory.py` lines 460-810 | exact |
| `README.md` | doc | N/A | self (existing structure) | exact |
| `CHANGELOG.md` | doc | N/A | self (existing format) | exact |
| `CLAUDE.md` | doc | N/A | self (existing structure) | exact |
| `daydream/atif/NOTICE` | doc | N/A | self (existing content) | exact |

## Pattern Assignments

### `tests/test_trajectory.py` (test, audit-in-place)

**Role:** Audit existing 1019 lines against TEST-02 requirements. Fill gaps only.

**Module docstring pattern** (lines 1-6):
```python
"""Tests for daydream/trajectory.py -- TrajectoryRecorder + Invocation + Redactor.

Per D-18, tests follow schema-validity + behavior-predicate patterns. Full-tree
snapshot equality is banned (Pitfall 11). Most assertions go through
``daydream.atif.validate()`` plus one or two specific behavioral predicates.
"""
```

**Imports pattern** (lines 8-34):
```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream.atif import validate as atif_validate
from daydream.backends import (
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    Invocation,
    Redactor,
    TrajectoryRecorder,
    _safe_descriptor,
    get_current_recorder,
    now_iso,
)
```

**Test helper pattern** (lines 37-50):
```python
def _make_recorder(tmp_path: Path, *, agent_model_name: str = "opus") -> TrajectoryRecorder:
    """Construct a TrajectoryRecorder rooted in tmp_path (test helper)."""
    return TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name=agent_model_name,
    )


def _read_trajectory(path: Path) -> dict[str, Any]:
    """Load the produced trajectory JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))
```

**Core schema-validity + behavior-predicate pattern** (lines 57-69):
```python
async def test_text_event_then_result_produces_one_agent_step(tmp_path: Path) -> None:
    """Behavior 1: One agent Step from a single TextEvent + ResultEvent."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="Hello world"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    traj = _read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
    assert agent_steps[0]["message"] == "Hello world"
```

**Observe helper for fork tests** (lines 465-468):
```python
def _observe_text_and_result(inv: Any, text: str = "output") -> None:
    """Helper: observe a TextEvent + ResultEvent to produce a minimal agent step."""
    inv.observe(TextEvent(text=text))
    inv.observe(ResultEvent(structured_output=None, continuation=None))
```

**Monkeypatch-based failure injection** (lines 271-294):
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
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="hi"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    assert any("Trajectory write failed" in m for m in warnings_emitted)
```

---

### `tests/test_redaction.py` (test, audit-in-place)

**Role:** Audit existing 418 lines against TEST-03 requirements. Fill gaps only.

**Test helper pattern** (lines 17-44):
```python
def _user_step(message: str) -> Step:
    """Construct a minimal user Step with *message* (test helper)."""
    return Step(
        step_id=1,
        timestamp=now_iso(),
        source="user",
        message=message,
        extra={"daydream_phase": "review", "daydream_run_flow": "normal"},
    )


def _agent_step(
    message: str = "ok",
    reasoning_content: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    observation: Observation | None = None,
) -> Step:
    """Construct a minimal agent Step (test helper)."""
    return Step(
        step_id=2,
        timestamp=now_iso(),
        source="agent",
        model_name="opus",
        message=message,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        observation=observation,
        extra={"daydream_phase": "review", "daydream_run_flow": "normal"},
    )
```

**Positive + negative case pair pattern** (lines 51-185):
```python
# Positive case: assert secret is gone, replacement token is present
def test_redactor_scrubs_openai_api_key() -> None:
    """REDA-01: sk-* tokens replaced with [REDACTED_API_KEY]."""
    out = Redactor().redact_step(_user_step("token=sk-test-12345abcdef done"))
    assert isinstance(out.message, str)
    assert "sk-test-12345abcdef" not in out.message
    assert "[REDACTED_API_KEY]" in out.message

# Negative case: assert clean input passes through unredacted
def test_redactor_preserves_clean_paths() -> None:
    """Negative: relative paths without /Users//home/ prefix pass through."""
    out = Redactor().redact_step(_user_step("./src/app.py"))
    assert isinstance(out.message, str)
    assert out.message == "./src/app.py"
```

**Parametrized regression test pattern** (lines 322-369):
```python
@pytest.mark.parametrize(
    "non_secret",
    [
        "MONKEY_PATCH=enabled",
        "KEYBOARD_LAYOUT=qwerty",
        "AUTHOR=alice",
        "TOKENIZED=foo",
        "KEYSTORE=path/to/store",
    ],
)
def test_env_var_pattern_does_not_match_substring_lookalikes(non_secret: str) -> None:
    """WR-03 regression: env-var redaction is segment-aware, not substring-based."""
    out = Redactor().redact_step(_user_step(non_secret))
    assert isinstance(out.message, str)
    assert "[REDACTED_ENV_VAR]" not in out.message
    name, _, value = non_secret.partition("=")
    assert value in out.message, f"Expected {value!r} preserved in {out.message!r}"
```

---

### `tests/test_atif_vendor_smoke.py` (test, audit-in-place)

**Role:** Audit existing 67 lines against TEST-04. Already parametrizes over golden fixtures.

**Parametrized golden fixture pattern** (lines 41-48):
```python
def _golden_paths() -> list[Path]:
    return sorted(p for p in GOLDEN_DIR.rglob("*.json") if "_invalid" not in p.parts)


@pytest.mark.parametrize("golden_path", _golden_paths(), ids=lambda p: p.name)
def test_golden_fixtures_validate(golden_path: Path) -> None:
    """VEND-05 + D-09: every Terminus-2 (v1.6) and OpenHands (v1.5) golden validates."""
    assert validate(golden_path) is True
```

**Negative fixture pattern** (lines 51-57):
```python
def test_invalid_fixture_rejected() -> None:
    """D-13: deliberately-broken fixture fails validation."""
    invalid_path = GOLDEN_DIR / "_invalid" / "non-sequential-step-id.json"
    validator = TrajectoryValidator()
    assert validator.validate(invalid_path) is False
    assert any("step_id" in err.lower() for err in validator.errors), validator.errors
```

---

### `tests/test_trajectory_fixture.py` (test, audit-in-place)

**Role:** Audit existing 52 lines against TEST-04/TEST-05. Validates autouse fixture isolation.

**ContextVar leak/detect pattern** (lines 32-52):
```python
def test_a_leak_recorder_var() -> None:
    """Set the ContextVar without cleanup -- exercises the leak path."""
    sentinel: Any = _SentinelRecorder()
    _RECORDER_VAR.set(sentinel)
    assert get_current_recorder() is sentinel


def test_b_recorder_var_starts_clean() -> None:
    """Assert the ContextVar is None at the START of the test."""
    assert get_current_recorder() is None
```

---

### `tests/test_multi_turn_tokens.py` (new, TEST-06)

**Analog:** `tests/test_agent_recorder_integration.py`

**MockBackend dataclass pattern** (lines 60-95):
```python
@dataclass
class MockBackend:
    """Minimal Backend implementation that replays a canned event list."""

    events: list[AgentEvent]

    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
    ) -> AsyncIterator[AgentEvent]:
        events = self.events

        async def _gen() -> AsyncIterator[AgentEvent]:
            for event in events:
                yield event

        return _gen()

    async def cancel(self) -> None:
        return None

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"
```

**Multi-invocation recorder test pattern** (lines 209-255):
```python
async def test_final_metrics_equal_sum_of_per_step_metrics(tmp_path: Path) -> None:
    """MAP-07 / Roadmap success criterion 4 -- FinalMetrics totals match per-step sum."""
    recorder = _make_recorder(tmp_path)
    target_path = recorder.path
    backend1 = MockBackend([
        TextEvent(text="first"),
        MetricsEvent(
            message_id="msg_01",
            prompt_tokens=100,
            completion_tokens=20,
            cached_tokens=5,
            cost_usd=0.001,
        ),
        ResultEvent(structured_output=None, continuation=None),
    ])
    backend2 = MockBackend([
        TextEvent(text="second"),
        MetricsEvent(
            message_id="msg_02",
            prompt_tokens=200,
            completion_tokens=40,
            cached_tokens=15,
            cost_usd=0.002,
        ),
        ResultEvent(structured_output=None, continuation=None),
    ])
    async with recorder:
        await run_agent(backend1, tmp_path, "first prompt", phase=DaydreamPhase.REVIEW)
        await run_agent(backend2, tmp_path, "second prompt", phase=DaydreamPhase.FIX)

    assert target_path.exists()
    traj = json.loads(target_path.read_text())
    assert atif_validate(traj) is True

    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    sum_prompt = sum(s["metrics"]["prompt_tokens"] for s in agent_steps if s.get("metrics"))
    # ... assert per-step values match individual MetricsEvent values (NOT cumulative)
```

TEST-06 extends this pattern: 3 sequential `run_agent()` calls with known token values. Assert each step's `Metrics.prompt_tokens` matches the per-call value, not cumulative. The test is a gate: it passes or fails.

---

### `tests/test_subagent_shapes.py` (new, TEST-07)

**Analog:** `tests/test_trajectory.py` lines 460-810 (fork/sibling tests)

**Fork + validate both parent and child pattern** (lines 816-833):
```python
async def test_fork_validator_accepts_both(tmp_path: Path) -> None:
    """Both parent and child trajectories pass atif_validate."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.fork("fix-0") as child:
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                _observe_text_and_result(inv)
        recorder.create_dispatch_step(phase=DaydreamPhase.FIX)
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            _observe_text_and_result(inv)

    parent_traj = _read_trajectory(recorder.path)
    child_traj = _read_trajectory(child.path)

    assert atif_validate(parent_traj, validate_images=False) is True
    assert atif_validate(child_traj, validate_images=False) is True
```

**Multiple forks pattern** (lines 786-809):
```python
async def test_multiple_forks_all_registered(tmp_path: Path) -> None:
    """Three sequential forks all register with parent; dispatch step has 3 refs."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        for i in range(3):
            async with recorder.fork(f"fix-{i}") as child:
                async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                    _observe_text_and_result(inv, f"child-{i}")
        recorder.create_dispatch_step(phase=DaydreamPhase.FIX)
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            _observe_text_and_result(inv)

    parent_traj = _read_trajectory(recorder.path)
    assert atif_validate(parent_traj, validate_images=False) is True

    dispatch_steps = [
        s for s in parent_traj["steps"]
        if s["source"] == "agent" and "Dispatching" in s.get("message", "")
    ]
    assert len(dispatch_steps) == 1
    results = dispatch_steps[0]["observation"]["results"]
    assert len(results) == 3
```

TEST-07 extends this pattern with MockBackend driving `phase_fix_parallel` / deep / exploration through the recorder's `fork()` path. Uses real recorder code with fake backends.

---

### `README.md` (doc, update)

**Existing structure** (221 lines):
- Lines 1-8: Title + badge + description
- Lines 9-19: Features list
- Lines 21-38: Prerequisites + Beagle install
- Lines 40-94: Installation + Usage + More Examples + CLI Options table
- Lines 124-186: How It Works (Standard Review, TTT, PR Feedback)
- Lines 187-194: Output Files table
- Lines 195-210: Architecture tree
- Lines 212-221: Dependencies + License

New "Trajectory Output" section (DOCS-01..03) inserts after line 194 (Output Files), before Architecture. D-05 says 3-4 sentences max with link to `docs/reference/atif_format.md`.

**Output Files table pattern** (lines 187-194):
```markdown
## Output Files

| File | Description |
|------|-------------|
| `.review-output.md` | Review results (removed with `--cleanup`, required for `--start-at parse/fix`) |
| `.review-debug-{timestamp}.log` | Debug log (created when `--debug` is enabled) |
| `.daydream/plan-{timestamp}.md` | Implementation plan (created by `--ttt` mode) |
```

The debug log row will be replaced and a trajectory row added.

---

### `CHANGELOG.md` (doc, update)

**Existing entry format** (lines 10-61, `[0.13.1]` entry):
```markdown
## [0.13.1] - 2026-04-26

### Changed

- **pr-review:** Restyle inline GitHub PR reviews with severity emoji prefixes...

### Fixed

- **deep:** Stop target-repo `.claude/settings.json` from blocking agent file writes...

### Added

- **scripts:** Add `scripts/redrive_post.py` for re-driving PR comment posts...
```

New `[0.14.0]` entry per D-09: Single heading with `### Breaking`, `### Added`, `### Removed` subsections. Insert above `[Unreleased]` (line 9) or replace it.

**Link format** (lines 359-377):
```markdown
[unreleased]: https://github.com/existential-birds/daydream/compare/v0.13.1...HEAD
[0.13.1]: https://github.com/existential-birds/daydream/compare/v0.13.0...v0.13.1
```

---

### `CLAUDE.md` (doc, update)

**Module Responsibilities section** (lines 44-55):
```markdown
### Module Responsibilities

- **cli.py**: Entry point, argument parsing, signal handlers (SIGINT/SIGTERM)
- **runner.py**: Main orchestration via `run()` async function, `RunConfig` dataclass
- **phases.py**: Four workflow phases: ...
- **agent.py**: Claude SDK client wrapper, `run_agent()` streams responses...
- **ui.py**: Rich-based terminal UI with Dracula theme, live-updating panels
- **config.py**: Skill mappings, constants
```

Add `trajectory.py` entry here. Also update `--debug` reference in Commands section (line 30) and Module Responsibilities to reflect cutover.

---

### `daydream/atif/NOTICE` (doc, verify)

Already complete at 40 lines. D-05 says verify completeness -- it documents vendored source, commit hash, Apache-2.0 license, and attribution. No update needed unless the vendored version changed (it has not since Phase 1).

---

## Shared Patterns

### Schema-Validity + Behavior-Predicate (D-18)
**Source:** `tests/test_trajectory.py` lines 57-69
**Apply to:** All test files (audit criterion for TEST-05)
```python
# Step 1: Run the code under test, produce a trajectory
traj = _read_trajectory(recorder.path)
# Step 2: Assert schema validity
assert atif_validate(traj, validate_images=False) is True
# Step 3: Assert 1-2 specific behavioral predicates
agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
assert len(agent_steps) == 1
assert agent_steps[0]["message"] == "Hello world"
```
**Anti-pattern (banned):** `assert trajectory == expected_dict` (full-tree snapshot equality)

### Autouse Fixture Reset (D-17 / CORE-10)
**Source:** `tests/conftest.py` lines 125-141
**Apply to:** All trajectory test files (inherited automatically)
```python
@pytest.fixture(autouse=True)
def _reset_trajectory_recorder():
    """Clear the trajectory ContextVar before AND after every test."""
    from daydream.trajectory import _reset_recorder_for_tests

    _reset_recorder_for_tests()
    yield
    _reset_recorder_for_tests()
```

### MockBackend (inline, dataclass variant)
**Source:** `tests/test_agent_recorder_integration.py` lines 61-95
**Apply to:** TEST-06 (multi-turn tokens), TEST-07 (subagent shapes)
```python
@dataclass
class MockBackend:
    events: list[AgentEvent]

    def execute(
        self, cwd: Path, prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
    ) -> AsyncIterator[AgentEvent]:
        events = self.events
        async def _gen() -> AsyncIterator[AgentEvent]:
            for event in events:
                yield event
        return _gen()

    async def cancel(self) -> None:
        return None

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"
```

### MockBackend (inline class variant, for phase-level tests)
**Source:** `tests/test_phases.py` lines 34-56
**Apply to:** TEST-07 if driving `phase_fix_parallel` directly
```python
class FreshContextBackend:
    async def execute(self, cwd, prompt, output_schema=None, continuation=None, agents=None, max_turns=None):
        nonlocal call_count
        call_count += 1
        captured_prompts.append(prompt)
        if call_count == 1:
            yield TextEvent(text="output text")
            yield ResultEvent(structured_output=None, continuation=token)
        # ...

    async def cancel(self):
        pass

    def format_skill_invocation(self, skill_key, args=""):
        return f"/{skill_key}"
```

### UI Silencing
**Source:** `tests/test_phases.py` lines 21-27
**Apply to:** TEST-07 if invoking phase functions
```python
monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
monkeypatch.setattr("daydream.phases.print_success", lambda *a, **kw: None)
monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)
monkeypatch.setattr("daydream.phases.print_menu", lambda *a, **kw: None)
monkeypatch.setattr("daydream.phases.print_error", lambda *a, **kw: None)
monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())
```

### Recorder Helper Pair
**Source:** `tests/test_trajectory.py` lines 37-49
**Apply to:** TEST-06, TEST-07 (reuse verbatim)
```python
def _make_recorder(tmp_path: Path, *, agent_model_name: str = "opus") -> TrajectoryRecorder:
    return TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name=agent_model_name,
    )

def _read_trajectory(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
```

### CHANGELOG Entry Format
**Source:** `CHANGELOG.md` lines 10-61
**Apply to:** New `[0.14.0]` entry
```markdown
## [0.14.0] - YYYY-MM-DD

### Breaking

- **cli:** Remove `--debug` flag...

### Added

- **trajectory:** Add ATIF v1.6 trajectory output...
- **cli:** Add `--trajectory <path>` flag...

### Removed

- **agent:** Remove `_log_debug()` system and `.review-debug-*.log` files...
```

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (none) | | | All files have strong analogs in the existing codebase |

Every file in this phase either already exists (audit-in-place) or has an exact-match analog in the existing test suite. Phase 5 is purely test-hardening + docs, so all patterns are established.

## Metadata

**Analog search scope:** `tests/`, `README.md`, `CHANGELOG.md`, `CLAUDE.md`, `daydream/atif/NOTICE`, `tests/conftest.py`
**Files scanned:** 10 target files + 5 analog references
**Pattern extraction date:** 2026-04-28
