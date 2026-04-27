# Phase 3: Subagent Wiring (Parallel + Continuation) - Pattern Map

**Mapped:** 2026-04-27
**Files analyzed:** 4 (modified)
**Analogs found:** 4 / 4

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `daydream/trajectory.py` | service | event-driven | self (Phase 2 `TrajectoryRecorder.__aenter__/__aexit__` + `_InvocationCM`) | exact |
| `daydream/phases.py` | controller | parallel fan-out | self (`phase_fix_parallel` lines 962-1036 + `phase_per_stack_reviews` lines 1388-1489) | exact |
| `daydream/exploration_runner.py` | controller | parallel fan-out | `daydream/phases.py:phase_fix_parallel` (same task-group + closure pattern) | role-match |
| `tests/test_trajectory.py` | test | unit | self (Phase 2 tests: `_make_recorder` helper + schema-validity assertions) | exact |

## Pattern Assignments

### `daydream/trajectory.py` (service, event-driven) — MODIFY

**Analog:** Self — the Phase 2 `TrajectoryRecorder` and `_InvocationCM` classes in the same file.

**Imports pattern** (lines 1-46):
```python
from __future__ import annotations

import json
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.atif import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from daydream.ui import create_console, print_warning

if TYPE_CHECKING:
    from daydream.backends import AgentEvent
```
Phase 3 adds `SubagentTrajectoryRef` to the `daydream.atif` import block and `import re` at the top for `_safe_descriptor()`.

**Async context manager pattern** — `_InvocationCM` (lines 483-498):
```python
class _InvocationCM:
    """Async context manager wrapping an Invocation (internal helper)."""

    def __init__(self, recorder: TrajectoryRecorder, phase: DaydreamPhase) -> None:
        self._recorder = recorder
        self._phase = phase
        self._invocation: Invocation | None = None

    async def __aenter__(self) -> Invocation:
        self._invocation = Invocation(recorder=self._recorder, phase=self._phase)
        return self._invocation

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._invocation is not None:
            self._invocation.finish()
            self._invocation = None
```
The new `_ForkCM` class follows this exact pattern: private helper class, `__init__` stores parent state, `__aenter__` creates the child object, `__aexit__` cleans up. The key difference is `_ForkCM.__aexit__` also writes the sibling file and registers with the parent.

**ContextVar set/reset pattern** — `TrajectoryRecorder.__aenter__/__aexit__` (lines 396-411):
```python
async def __aenter__(self) -> "TrajectoryRecorder":
    self._previous_token = _RECORDER_VAR.set(self)
    return self

async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    try:
        self._write()
    except Exception as exc:  # noqa: BLE001 - implicit write degrade-with-warning per D-11
        print_warning(
            _console,
            f"Trajectory write failed: {type(exc).__name__}: {exc}",
        )
    finally:
        if self._previous_token is not None:
            _RECORDER_VAR.reset(self._previous_token)
            self._previous_token = None
```
`_ForkCM.__aenter__` must call `_RECORDER_VAR.set(child)` and store the token on the child. `_ForkCM.__aexit__` must write then reset in `finally`, identical ordering to this pattern.

**Step construction pattern** — `Invocation._close_open_step` (lines 320-354):
```python
agent_step = Step(
    step_id=self.recorder._next_step_id(),
    timestamp=now_iso(),
    source="agent",
    message=message_text,
    model_name=d["_model_name"],
    reasoning_content=reasoning,
    tool_calls=tool_calls,
    observation=observation,
    metrics=d["_metrics"],
    extra=extra,
)
self.steps.append(self.recorder.redactor.redact_step(agent_step))
```
`create_dispatch_step()` constructs a `Step` following this exact pattern: `step_id` from `self._next_step_id()`, `timestamp` from `now_iso()`, `source="agent"`, `extra` dict with `daydream_phase` and `daydream_run_flow`. The new field is `observation` with `ObservationResult.subagent_trajectory_ref`.

**Error handling pattern** — degrade-with-warning (lines 400-411):
```python
try:
    self._write()
except Exception as exc:  # noqa: BLE001
    print_warning(
        _console,
        f"Trajectory write failed: {type(exc).__name__}: {exc}",
    )
```
Child recorder write failures in `_ForkCM.__aexit__` use the same catch-and-warn pattern. Recording must never crash a run (Architecture Q7).

**`_write()` empty-guard pattern** (lines 473-480):
```python
def _write(self) -> None:
    if not self.steps:
        return
    trajectory = self._build_trajectory()
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.path.write_text(json.dumps(trajectory.to_json_dict(), indent=2), encoding="utf-8")
```
Child recorders reuse this exact method via inheritance — if the child has 0 steps (failed before any `run_agent()`), `_write()` returns without creating a file. The parent's `_register_sibling()` checks `child.path.exists()` before registering.

---

### `daydream/phases.py` (controller, parallel fan-out) — MODIFY

**Analog:** Self — the existing `phase_fix_parallel` (lines 962-1036) and `phase_per_stack_reviews` (lines 1388-1489).

**Imports pattern** (lines 1-37):
```python
from daydream.trajectory import DaydreamPhase
```
Phase 3 adds `get_current_recorder` to this import line: `from daydream.trajectory import DaydreamPhase, get_current_recorder`.

**Task closure with default-arg capture** — `phase_fix_parallel` (lines 1004-1021):
```python
async def _fix_task(
    task_index: int = index,
    task_item: dict[str, Any] = item,
    task_prompt: str = prompt,
) -> None:
    def callback(message: str, i: int = task_index) -> None:
        panel.update_row(i, message)

    try:
        async with limiter:
            await run_agent(backend, cwd, task_prompt, progress_callback=callback, phase=DaydreamPhase.FIX)
        panel.complete_row(task_index)
        results.append((task_item, True, None))
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        panel.fail_row(task_index, error_msg)
        results.append((task_item, False, error_msg))

tg.start_soon(_fix_task)
```
The fork wrapper nests as the outermost context manager inside the closure, BEFORE the `async with limiter:`. The `if recorder is not None:` guard wraps the fork. The non-recorder path duplicates the existing body unchanged.

**Task closure with default-arg capture** — `phase_per_stack_reviews` (lines 1461-1487):
```python
async def _task(
    stack_name: str = stack.stack_name,
    task_prompt: str = prompt,
    task_output: Path = output_path,
) -> None:
    try:
        async with limiter:
            await run_agent(backend, cwd, task_prompt, phase=DaydreamPhase.DEEP)
        results[stack_name] = task_output
    except Exception as e:  # noqa: BLE001 -- intentionally broad for parallel isolation
        reason = f"{type(e).__name__}: {e}"
        failures[stack_name] = reason
        # ...

tg.start_soon(_task)
```
Same fork-wrapping pattern: `async with recorder.fork(f"deep-{stack_name}"):` around the try/except body.

**Post-task-group hook point** — `phase_fix_parallel` (line 1024):
```python
    panel.finish()  # line 1024, immediately after `async with anyio.create_task_group() as tg:` exits
```
`recorder.create_dispatch_step(phase=DaydreamPhase.FIX)` is called after the task group exits, before `panel.finish()` or the results summary.

**Post-task-group hook point** — `phase_per_stack_reviews` (line 1489):
```python
    return results, failures  # line 1489, immediately after task group exits
```
`recorder.create_dispatch_step(phase=DaydreamPhase.DEEP)` is called after the task group exits, before the return.

---

### `daydream/exploration_runner.py` (controller, parallel fan-out) — MODIFY

**Analog:** `daydream/phases.py:phase_fix_parallel` (lines 962-1036) — same `anyio.create_task_group()` + inner closure pattern.

**Imports pattern** (lines 1-34):
```python
from daydream.trajectory import DaydreamPhase
```
Phase 3 adds `get_current_recorder` to this import line: `from daydream.trajectory import DaydreamPhase, get_current_recorder`.

**Task closure pattern** — `_run_specialist` (lines 244-253):
```python
async def _run_specialist(name: str, prompt: str, schema: dict) -> None:
    try:
        structured, _ = await run_agent(
            backend, repo_root, prompt, output_schema=schema, max_turns=specialist_max_turns,
            phase=DaydreamPhase.EXPLORATION,
        )
        if isinstance(structured, dict):
            results[name] = structured
    except Exception as exc:
        _log_debug(f"[PRE_SCAN] specialist {name} failed: {type(exc).__name__}: {exc}\n")
```
Unlike `phase_fix_parallel`, `_run_specialist` takes `name` as a positional arg (not default-arg capture from a loop variable). The fork wrapper goes inside this function body: `recorder = get_current_recorder()` before the task group, then inside `_run_specialist`: `if recorder is not None: async with recorder.fork(f"explore-{name}"):` wrapping the try/except.

**Task group pattern** (lines 257-273):
```python
async with anyio.create_task_group() as tg:
    if tier == "single":
        tg.start_soon(_run_specialist, "dependency_tracer", dep_prompt, DEPENDENCY_TRACER_SCHEMA)
    else:  # parallel
        tg.start_soon(
            _run_specialist, "pattern_scanner",
            build_pattern_scanner_prompt(file_paths, diff_ref), PATTERN_SCANNER_SCHEMA,
        )
        # ... two more start_soon calls
```
`recorder.create_dispatch_step(phase=DaydreamPhase.EXPLORATION)` is called after this task group exits (line 274), before the `if not results:` check.

---

### `tests/test_trajectory.py` (test, unit) — MODIFY

**Analog:** Self — the existing Phase 2 tests in the same file.

**Test helper pattern** — `_make_recorder` (lines 71-78):
```python
def _make_recorder(tmp_path: Path, *, agent_model_name: str = "opus") -> TrajectoryRecorder:
    """Construct a TrajectoryRecorder rooted in tmp_path (test helper)."""
    return TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name=agent_model_name,
    )
```
Fork tests reuse `_make_recorder` for the parent recorder, then call `recorder.fork(descriptor)` to get child recorders.

**Test helper pattern** — `_read_trajectory` (lines 81-83):
```python
def _read_trajectory(path: Path) -> dict[str, Any]:
    """Load the produced trajectory JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))
```
Fork tests use `_read_trajectory` for both parent and sibling trajectory files.

**Schema-validity assertion pattern** (lines 99-103):
```python
traj = _read_trajectory(recorder.path)
assert atif_validate(traj, validate_images=False) is True
agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
assert len(agent_steps) == 1
assert agent_steps[0]["message"] == "Hello world"
```
All Phase 3 tests follow this: load JSON, validate via `atif_validate`, then assert specific behavioral predicates. No full-tree snapshot equality (D-18 ban).

**Autouse fixture pattern** (lines 63-68):
```python
@pytest.fixture(autouse=True)
def _reset_recorder() -> Any:
    """Reset _RECORDER_VAR before and after every test (mirrors D-17)."""
    _reset_recorder_for_tests()
    yield
    _reset_recorder_for_tests()
```
Already covers fork/sibling scenarios — `_RECORDER_VAR.set(None)` clears any leftover child or parent recorder. No changes needed to conftest.py.

**Behavioral test naming** (examples):
```
test_text_event_then_result_produces_one_agent_step
test_step_ids_sequential_across_two_invocations
test_context_var_set_inside_and_cleared_after
```
Phase 3 tests follow the same `test_<what_it_verifies>` pattern. Per RESEARCH.md test map:
- `test_fork_contextvar_isolation` (SUBA-07)
- `test_sibling_inherits_session_id` (SUBA-06)
- `test_step_id_isolation_across_siblings` (SUBA-08)
- `test_parent_metrics_exclude_children` (SUBA-09)
- etc.

**Import block** (lines 1-35):
```python
from daydream.atif import validate as atif_validate
from daydream.backends import (
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    Invocation,
    Redactor,
    TrajectoryRecorder,
    _reset_recorder_for_tests,
    get_current_recorder,
    now_iso,
)
```
Phase 3 tests may not need additional imports beyond what exists — `get_current_recorder` is already imported. Fork tests will call `recorder.fork()` (method on `TrajectoryRecorder`).

---

## Shared Patterns

### ContextVar Set/Reset Lifecycle
**Source:** `daydream/trajectory.py` lines 396-411 (`TrajectoryRecorder.__aenter__/__aexit__`)
**Apply to:** `_ForkCM.__aenter__/__aexit__` — child recorder sets `_RECORDER_VAR` on enter, resets to parent on exit.
```python
# __aenter__: set ContextVar to self, store token
self._previous_token = _RECORDER_VAR.set(self)

# __aexit__: write first, then reset ContextVar in finally
try:
    self._write()
except Exception as exc:  # noqa: BLE001
    print_warning(_console, f"...: {type(exc).__name__}: {exc}")
finally:
    if self._previous_token is not None:
        _RECORDER_VAR.reset(self._previous_token)
        self._previous_token = None
```

### Degrade-With-Warning Error Handling (Architecture Q7)
**Source:** `daydream/trajectory.py` lines 220-224 (`Invocation.observe`) and lines 400-407 (`TrajectoryRecorder.__aexit__`)
**Apply to:** All new recorder methods (`fork()`, `_register_sibling()`, `create_dispatch_step()`, child `__aexit__`)
```python
except Exception as exc:  # noqa: BLE001 - recording must never crash a run
    print_warning(_console, f"Trajectory recording: {type(exc).__name__}: {exc}")
```

### `if recorder is not None:` Guard
**Source:** `daydream/agent.py` line 28 — already established in `run_agent()` via `get_current_recorder()`
**Apply to:** All three integration sites in `phases.py` and `exploration_runner.py` before calling `recorder.fork()` or `recorder.create_dispatch_step()`
```python
recorder = get_current_recorder()
# ... task group ...
if recorder is not None:
    recorder.create_dispatch_step(phase=DaydreamPhase.FIX)
```

### Task Closure Default-Arg Capture
**Source:** `daydream/phases.py` lines 1004-1008 and 1461-1465
**Apply to:** Fork wrapping inside closures — the fork must capture the recorder reference (or descriptor) by value, not by late-binding closure.

### Step extra Dict Convention
**Source:** `daydream/trajectory.py` lines 335-340 (`Invocation._close_open_step`)
**Apply to:** `create_dispatch_step()` — must include `daydream_phase` and `daydream_run_flow` keys.
```python
extra={
    "daydream_phase": self.phase.value,
    "daydream_run_flow": self.recorder.run_flow.value,
}
```
For `create_dispatch_step()`, the pattern is the same but sourced from `self` (the recorder) rather than `self.recorder` (since the method is on `TrajectoryRecorder` directly):
```python
extra={
    "daydream_phase": phase.value,
    "daydream_run_flow": self.run_flow.value,
}
```

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (none) | — | — | All 4 files have exact or role-match analogs within the existing codebase |

## ATIF Model Surface for Dispatch Steps

The following vendored models are used by `create_dispatch_step()`. No new ATIF models are needed.

| Model | File | Key Fields |
|-------|------|------------|
| `SubagentTrajectoryRef` | `daydream/atif/models/subagent_trajectory_ref.py` | `session_id: str`, `trajectory_path: str \| None`, `extra: dict \| None` |
| `ObservationResult` | `daydream/atif/models/observation_result.py` | `source_call_id: str \| None`, `content: str \| ... \| None`, `subagent_trajectory_ref: list[SubagentTrajectoryRef] \| None` |
| `Observation` | (imported via `daydream/atif`) | `results: list[ObservationResult]` |
| `Step` | (imported via `daydream/atif`) | `step_id`, `timestamp`, `source`, `message`, `observation`, `extra`, etc. |

## Metadata

**Analog search scope:** `daydream/`, `tests/`
**Files scanned:** 8 (trajectory.py, phases.py, exploration_runner.py, test_trajectory.py, agent.py, atif/models/subagent_trajectory_ref.py, atif/models/observation_result.py, atif/models/__init__.py)
**Pattern extraction date:** 2026-04-27
