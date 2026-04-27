---
phase: 03-subagent-wiring-parallel-continuation
reviewed: 2026-04-27T18:45:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - daydream/exploration_runner.py
  - daydream/phases.py
  - daydream/trajectory.py
  - tests/test_trajectory.py
findings:
  critical: 0
  warning: 3
  info: 3
  total: 6
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-04-27T18:45:00Z
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Re-reviewed the Phase 3 implementation after the prior review's fixes were applied (commits bf20aa4, e09c357, 0712b00, bf5389bf). All four prior findings (CR-01, WR-01, WR-02, WR-03) have been correctly addressed. The core fork/ContextVar/sibling-registration lifecycle is sound: ContextVar copy-on-task-create semantics ensure child recorders are isolated to their task, `_maybe_fork` cleanly eliminates the recorder/no-recorder branch duplication, `_safe_descriptor` now rejects degenerate inputs, `create_dispatch_step` catches `ValueError` from `Path.relative_to`, and `_ForkCM.__aexit__` tracks write success via boolean flag.

Three new warnings identified: dead MetricsEvent import/dispatch code, `_current_message_id` never being set (dead message-id correlation infrastructure), and test stubs that bypass the production MetricsEvent class. Three info items for stale comments, redundant fixtures, and private-API naming.

## Warnings

### WR-01: `_current_message_id` is never set -- MetricsEvent message_id correlation is dead

**File:** `daydream/trajectory.py:211`
**Issue:** `Invocation._current_message_id` is initialized to `""` and never assigned any other value anywhere in the codebase. The condition at line 341 (`if self._current_message_id:`) is therefore always False, so `_message_id_to_step` is never populated. The MetricsEvent handler at line 303 does `self._message_id_to_step.get(event.message_id, self._open_step_dict)` which always falls back to `self._open_step_dict`. This means MetricsEvents are always attached to whatever step happens to be open, not correlated by `message_id` to the correct step. In multi-turn invocations where multiple agent steps are produced, a MetricsEvent with a specific `message_id` will land on the wrong step if the step that produced that message has already been closed.
**Fix:** If message-id correlation is deferred to a later phase, add an explicit TODO comment and remove the dead `_message_id_to_step` dict and `_current_message_id` field to avoid misleading readers into thinking correlation works. If it should work now, wire `_current_message_id` from the backend event stream (the Claude backend's `AssistantMessage.message_id` would need to be surfaced as a new event or field on `TextEvent`).

### WR-02: `_dispatch` defensive MetricsEvent import and class-name fallback are dead code

**File:** `daydream/trajectory.py:262-265,297-298`
**Issue:** `MetricsEvent` is now a production class in `daydream.backends.__init__` (line 110), making the `try/except ImportError` defensive import at lines 262-265 unreachable. The `type(event).__name__ == "MetricsEvent"` fallback at line 298 is also dead code in production -- it only fires for objects that match the class name but are not instances of the real class. This fallback allows any object with `__name__ == "MetricsEvent"` and the right attributes to be processed, bypassing type safety. The stale comment at lines 252-253 still says "defensively handle MetricsEvent's absence in Plan 02-01 (it lands in Plan 02-02)" when Plan 02-02 has already landed.
**Fix:** Import `MetricsEvent` alongside the other event types at line 254 and remove the `try/except` block. Replace the compound condition at line 297-298 with `elif isinstance(event, MetricsEvent):`. Update the comment.
```python
from daydream.backends import (
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
# ...
elif isinstance(event, MetricsEvent):
```

### WR-03: Test file uses stub MetricsEvent instead of the real production class

**File:** `tests/test_trajectory.py:43-62`
**Issue:** `_StubMetricsEvent` duplicates the field signature of the real `MetricsEvent` and sets `__name__ = "MetricsEvent"`. Since the real `MetricsEvent` exists in `daydream.backends`, these tests exercise only the class-name fallback path (WR-02), not the `isinstance` path. If `MetricsEvent` gains additional required fields, changes attribute names, or adds validation, the tests will not catch the regression because they never construct or dispatch the real class. The stub's `prompt_tokens` and `completion_tokens` are typed `int | None` but the real class types them as `int` (required) -- this type mismatch is invisible to tests.
**Fix:** Replace `_StubMetricsEvent` with the real `MetricsEvent`:
```python
from daydream.backends import MetricsEvent

# In tests:
metrics_event = MetricsEvent(
    message_id="msg-1",
    prompt_tokens=500,
    completion_tokens=80,
    cached_tokens=100,
    cost_usd=0.01,
)
```

## Info

### IN-01: Stale Plan 02-02 comments in trajectory.py

**File:** `daydream/trajectory.py:252-253,300`
**Issue:** Comments at lines 252-253 reference "Plan 02-02" as future work and line 300 says "the test stub before Plan 02-02's real MetricsEvent lands." Plan 02-02 has landed -- `MetricsEvent` is a production class at `daydream/backends/__init__.py:110`.
**Fix:** Update comments to reflect current state: MetricsEvent is a production class, the defensive import is no longer needed.

### IN-02: Redundant autouse fixture in test_trajectory.py

**File:** `tests/test_trajectory.py:65-70`
**Issue:** The `_reset_recorder` autouse fixture calls `_reset_recorder_for_tests()` before and after every test. The suite-wide `_reset_trajectory_recorder` autouse fixture in `tests/conftest.py:125-141` already does the same thing for all tests. The per-file fixture is redundant.
**Fix:** Remove the `_reset_recorder` fixture from `test_trajectory.py`.

### IN-03: `_maybe_fork` uses underscore-prefix but is imported across module boundaries

**File:** `daydream/trajectory.py:68`
**Issue:** `_maybe_fork` is prefixed with underscore (private by project convention: "Private helpers prefixed with underscore") but is explicitly imported in `daydream/phases.py:18` and `daydream/exploration_runner.py:33`. This sends mixed signals about whether it is internal or part of the public API.
**Fix:** Either rename to `maybe_fork` (public) or add an `__all__` to `trajectory.py` that includes it, documenting the intentional cross-module usage.

---

_Reviewed: 2026-04-27T18:45:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
