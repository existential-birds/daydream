---
phase: 02-recorder-core-event-enrichment-mapping
reviewed: 2026-04-26T00:00:00Z
depth: standard
files_reviewed: 19
files_reviewed_list:
  - daydream/agent.py
  - daydream/backends/__init__.py
  - daydream/backends/claude.py
  - daydream/backends/codex.py
  - daydream/deep/orchestrator.py
  - daydream/exploration_runner.py
  - daydream/phases.py
  - daydream/runner.py
  - daydream/trajectory.py
  - tests/conftest.py
  - tests/fixtures/codex_jsonl/turn_completed_partial_usage.jsonl
  - tests/fixtures/codex_jsonl/turn_completed_with_usage.jsonl
  - tests/test_agent_recorder_integration.py
  - tests/test_backend_claude_metrics.py
  - tests/test_backend_codex_metrics.py
  - tests/test_backends_events.py
  - tests/test_integration.py
  - tests/test_phase_2_integration.py
  - tests/test_trajectory.py
  - tests/test_trajectory_fixture.py
findings:
  critical: 0
  warning: 6
  info: 5
  total: 11
status: issues_found
---

# Phase 2: Code Review Report

**Reviewed:** 2026-04-26
**Depth:** standard
**Files Reviewed:** 19 (10 source + 9 test/fixture)
**Status:** issues_found

## Summary

Phase 2 wires the ATIF v1.6 trajectory recorder into the existing daydream
runtime. The implementation is largely sound: the `TrajectoryRecorder` /
`Invocation` lifecycle is well-isolated, ATIF model construction stays
inside `daydream/trajectory.py` (D-19), the `ContextVar` propagation
correctly resets via the autouse conftest fixture, and the
`MetricsEvent` boundary correctly renames `input_tokens`/`output_tokens`
to ATIF-shaped `prompt_tokens`/`completion_tokens` per EVNT-02 and D-15.

Key concerns surfaced during review:

1. **Dead message-id-to-step routing.** `Invocation._current_message_id`
   is declared but never assigned, so the documented per-message metrics
   routing (`_message_id_to_step`) silently falls back to "current open
   step" on every dispatch. The intended D-04 mapping is not actually
   implemented even though the data structures exist.
2. **Stale `ToolResultEvent`s after a Step closes are silently dropped**
   without populating `extra.unmatched_tool_results` — they land on a
   detached, frozen-in-place dict and never reach disk.
3. **`record_sources` semantics differ between the merge-resume and
   normal merge branches** in `deep/orchestrator.py`, producing
   different dedup-candidate JSON depending on the resume path.

No critical (data-loss / security) issues were found. All findings below
are correctness or quality concerns that should be addressed before this
phase is considered complete.

## Warnings

### WR-01: `_current_message_id` is never assigned — message-id-keyed metrics routing is dead code

**File:** `daydream/trajectory.py:187,279,317-318`
**Issue:** `Invocation._current_message_id` is declared with `default=""` and
referenced in two places (`_message_id_to_step.get(event.message_id, ...)`
on line 279 and `if self._current_message_id: ... = self._open_step_dict`
on line 317-318), but it is never assigned anywhere in the codebase
(`grep '_current_message_id' daydream/trajectory.py` shows only the
declaration and two reads). Result: `_message_id_to_step` is always
empty, so the lookup at line 279 always returns the default
(`self._open_step_dict`). The MAP-06 / D-04 design — "MetricsEvent
attaches to the agent step that owns its `message_id`" — is documented
in the dataclass docstring (`_message_id_to_step: AssistantMessage.message_id
-> open-step (D-04)`) but not implemented. Today's behavior happens to
work because Claude emits MetricsEvent inside the same AssistantMessage
that produced the text, so the open step is correct by accident.
**Fix:** Either remove the dead state and lookup (and the misleading
docstring) or wire up the assignment in `_dispatch` when a TextEvent /
ToolStartEvent first opens a step for a known `message_id`. If the
intent is "MetricsEvent always attaches to the currently open step,"
delete `_current_message_id`, `_message_id_to_step`, and simplify
`_dispatch` to:
```python
target = self._open_step_dict
if target is None:
    self._ensure_open_step()
    target = self._open_step_dict
```

### WR-02: Stale `ToolResultEvent` after a Step closes is silently dropped

**File:** `daydream/trajectory.py:260-272,320-334`
**Issue:** `_in_flight_tools` maps `tool_call_id -> open-step-dict` at
ToolStartEvent time. When `_close_open_step()` runs (on ResultEvent or
the next user step), it copies `d["_observation_results"]` via `list(...)`
into a Pydantic Step but does NOT remove entries from `_in_flight_tools`.
If a `ToolResultEvent` for that id arrives later, `_in_flight_tools.pop`
returns the (now detached, already-frozen) dict, and the result is
appended to `host["_observation_results"]` — a list that is no longer
referenced by any Step. The data is silently lost AND the
`unmatched_tool_results` extra is not populated either, so the
trajectory has no record this happened. Only Codex's pending-id miss
produces the unmatched-results signal.
**Fix:** Either drain `_in_flight_tools` of any entries pointing at the
closing dict in `_close_open_step()`, or detect the "host is detached"
condition in the ToolResultEvent branch and route to
`_unmatched_tool_results` instead:
```python
elif isinstance(event, ToolResultEvent):
    host = self._in_flight_tools.pop(event.id, None)
    if host is None or host is not self._open_step_dict:
        self._ensure_open_step()
        assert self._open_step_dict is not None
        self._open_step_dict["_unmatched_tool_results"].append(event.id)
        return
    host["_observation_results"].append(...)
```

### WR-03: `record_sources` differs between resume and non-resume merge branches

**File:** `daydream/deep/orchestrator.py:444-451 vs 458-466`
**Issue:** In the `start_at == "merge"` resume branch (line 449), each
record's source is set to `records_path.name` (e.g.
`"python-records.json"`). In the non-resume branch (line 466), the
source is the bare `stack_name` (e.g. `"python"`). These values are
passed straight into `build_record_dedup_candidates(..., sources=record_sources)`,
which means the dedup-candidates JSON written to disk has different
"source" labels depending on whether the user resumed at `--start-at
merge` or ran the pipeline through. Reproducibility / round-trip
parity (a Phase 2 success criterion) is broken on the resume path.
The inline comment on line 447-448 (`"Derive stack name from the
records filename (e.g. 'stack-python-records.json' -> 'stack-python-records.json')"`)
even acknowledges the value is the filename but apparently intended it
to be the stack name.
**Fix:** Derive the stack name from the records filename using the
inverse of `per_stack_records_path` (or carry the `stack_name` through
`expected_paths` as `(stack_name, records_path)` tuples). Concrete
patch:
```python
for stack in stacks:
    records_path = per_stack_records_path(dd, stack.stack_name)
    if records_path.is_file():
        expected_paths.append((stack.stack_name, records_path))
    elif stack.stack_name not in failed_stacks:
        missing_stacks.append(stack.stack_name)
...
for stack_name, records_path in sorted(expected_paths, key=lambda p: p[0]):
    records = json.loads(records_path.read_text())
    per_stack_records_paths.append(records_path)
    all_records.extend(records)
    record_sources.extend(stack_name for _ in records)
```

### WR-04: `Backend` Protocol typing for `agents` is wider than the SDK declares

**File:** `daydream/backends/codex.py:88` vs `daydream/backends/__init__.py:194`
**Issue:** The `Backend` Protocol declares
`agents: dict[str, AgentDefinition] | None = None`, but `CodexBackend.execute`
declares `agents: dict[str, Any] | None = None`. Structural conformance
holds (the Codex implementation accepts a wider input), but this
weakens the type contract — any caller who passes a typed
`AgentDefinition` mapping has no static guarantee Codex would even see
the SDK type because the implementation never imports `AgentDefinition`.
Compounding this, Codex's runtime check `if agents:` raises
NotImplementedError on a non-empty dict regardless of value type. A
downstream caller passing a non-`AgentDefinition` dict by mistake would
never be caught at typecheck time.
**Fix:** Match the Protocol signature exactly. Import `AgentDefinition`
from `claude_agent_sdk.types` under `TYPE_CHECKING` and type the
parameter as `dict[str, AgentDefinition] | None`.

### WR-05: `phase_understand_intent` while-True loop has no exit guard against runaway corrections

**File:** `daydream/phases.py:1155-1185`
**Issue:** The intent-confirmation loop reruns indefinitely as long as
the user keeps providing corrections. Every iteration spawns a new
`run_agent` call (and now a new `Invocation` scope on the recorder). A
user accidentally typing the same correction repeatedly burns API
quota and grows the trajectory step count without bound. There is no
iteration cap, no signal handler integration, and the only exit is
`y/yes`. While not introduced by this phase, the recorder change makes
this materially more expensive — every iteration emits a full
Invocation worth of Steps to the trajectory.
**Fix:** Add a max-iteration guard (e.g. 5) that breaks with a clear
warning and returns the latest `intent_text`, mirroring the
`max_iterations` pattern already used in `runner.py`'s normal flow.

### WR-06: `_dispatch` MetricsEvent branch can construct invalid Metrics if `event.prompt_tokens` is None

**File:** `daydream/trajectory.py:284-289`
**Issue:** `MetricsEvent` declares `prompt_tokens: int` and
`completion_tokens: int` as required (non-Optional), and both backends
guard their emission with `is not None` checks. However, the test stub
`_StubMetricsEvent` in `tests/test_trajectory.py:43-55` declares both
fields as `int | None`, and code in `_dispatch` (line 285-288)
passes them straight through without re-validating. If a future
backend emits a malformed MetricsEvent (or a test passes None
deliberately), `Metrics(prompt_tokens=None, ...)` constructs cleanly
(ATIF Metrics fields are all Optional) but `_accumulate_metrics`
on line 437-447 in trajectory.py special-cases None correctly. The
only real risk is the silent type contract drift between the stub and
the production class. The exception will be caught by `observe()`'s
broad `except`, but the symptom (a `print_warning`) is hard to debug.
**Fix:** Either tighten `_StubMetricsEvent`'s field types to match the
production class (`int`, not `int | None`), or add an explicit
`prompt_tokens=event.prompt_tokens or 0` defensive guard in `_dispatch`
with a comment explaining why. The simpler fix is the test stub:
```python
@dataclass
class _StubMetricsEvent:
    message_id: str
    prompt_tokens: int           # NOT int | None
    completion_tokens: int       # NOT int | None
    cached_tokens: int | None
    cost_usd: float | None
```

## Info

### IN-01: Misleading inline comment on `record_sources` derivation

**File:** `daydream/deep/orchestrator.py:447-449`
**Issue:** The comment reads
`"Derive stack name from the records filename (e.g. 'stack-python-records.json' -> 'stack-python-records.json')"`.
The before-and-after example is identical — the comment was apparently
intended to show a transform like `"stack-python-records.json" ->
"python"` but the right-hand side was never updated. This makes the
intent of the line below (`source_name = records_path.name`) unclear
to a reader; combined with WR-03 above, this is part of the same root
cause.
**Fix:** Update the comment to either reflect the actual current
behavior ("source is the records filename") or, after applying WR-03,
to reflect the intended transform.

### IN-02: `_log_debug` lazy import in `daydream/backends/codex.py` is documented but creates a load-order trap

**File:** `daydream/backends/codex.py:35-40`
**Issue:** The module-private `_raw_log` helper imports
`daydream.agent._log_debug` lazily inside the function body to avoid
a circular import. This works, but every JSONL line goes through this
extra import lookup. More importantly, if `daydream.agent` is mocked
or partially initialized during a test, `_log_debug` may not exist
yet, raising AttributeError that bubbles out of `_raw_log` into the
backend's main loop. The pattern is widely used in the codebase, but
the comment "Lazy import to avoid circular dependency at module load
time" doesn't acknowledge the runtime risk.
**Fix:** Optional. Either accept the existing pattern (it's idiomatic
here) or wrap the import in a try/except and degrade silently when
`daydream.agent` isn't fully loaded.

### IN-03: `nullcontext(None)` async-compatibility relies on Python 3.10+ behavior

**File:** `daydream/agent.py:8,361-362`
**Issue:** The shape `invocation_cm: Any = recorder.invocation(phase=phase) if recorder is not None else nullcontext(None)` followed by `async with invocation_cm` works because Python 3.10+ extended `contextlib.nullcontext` to support both sync and async context manager protocols. The project pins `requires-python = ">=3.12"` (per CLAUDE.md) so this is safe today, but a comment marking the dependency would prevent regressions if anyone refactors the type to a pure-sync `contextmanager` later.
**Fix:** Add a one-line comment near the `from contextlib import nullcontext` import:
```python
# nullcontext supports async-with from Python 3.10+ — required for the
# "no recorder" no-op path in run_agent below.
```

### IN-04: `Invocation` dataclass uses underscore-prefixed fields as part of its public-ish surface

**File:** `daydream/trajectory.py:184-187`
**Issue:** Fields like `_open_step_dict`, `_in_flight_tools`,
`_message_id_to_step`, `_current_message_id` are declared as
underscore-prefixed in a dataclass. dataclass field names beginning
with underscore are still accepted but are an unusual idiom (they are
considered private but `__init__` still generates them as `__init__`
positional / keyword args). Tests are using `inv._dispatch` for
monkey-patching (test_trajectory.py:230) which is fine for tests but
indicates the boundary is permeable.
**Fix:** Optional. Either keep the convention (test-only access is
clearly documented) or rename to non-private names and add a comment
that they are not part of the public API.

### IN-05: `_INITIAL_TOTALS` mutable default uses `dict.copy()` — could be `frozenset` of items + dict comprehension

**File:** `daydream/trajectory.py:48,393`
**Issue:** `_INITIAL_TOTALS` is a module-level dict copied via
`.copy()` in the `default_factory`. The `# noqa: E501` is there but
the line is short; the noqa is for "ruff E501 - line too long" but the
line at 48 is well under 120 chars. This noqa is unnecessary noise.
**Fix:** Drop the `# noqa: E501` comment if running `ruff check` on
the file confirms no warning fires today.

---

_Reviewed: 2026-04-26_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
