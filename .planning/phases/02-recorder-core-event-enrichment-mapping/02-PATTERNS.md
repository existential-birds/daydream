# Phase 2: Recorder Core + Event Enrichment + Mapping — Pattern Map

**Mapped:** 2026-04-26
**Files analyzed:** 8 (1 NEW, 7 MODIFY)
**Analogs found:** 8 / 8

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `daydream/trajectory.py` (NEW) | service (recorder) + module-singleton + ContextVar | event-driven (consumes `AgentEvent` stream → constructs ATIF Pydantic models on flush) | `daydream/agent.py` (singleton state + getter/setter), `daydream/exploration_runner.py` (top-level orchestrator module precedent) | role-match (split: state-singleton from `agent.py`, top-level-module shape from `exploration_runner.py`) |
| `daydream/agent.py` (MODIFY) | service (run_agent event loop) | event-driven (async iterator dispatch on `isinstance`) | itself, lines 339–430 (existing event loop is the insertion point) | exact (in-place edit) |
| `daydream/backends/__init__.py` (MODIFY) | model (event dataclasses + Backend protocol) | type-only (dataclass union) | itself (existing `AgentEvent` dataclass family) | exact (additive edit — new field, new dataclass, extend union) |
| `daydream/backends/claude.py` (MODIFY) | service (SDK adapter, async generator) | streaming (yields `AgentEvent` per SDK message) | itself, lines 95, 120–128 | exact (in-place edit at the patch site) |
| `daydream/backends/codex.py` (MODIFY) | service (subprocess JSONL adapter, async generator) | streaming (yields `AgentEvent` per JSONL event) | itself, line 308 (`turn.completed` branch) | exact (in-place edit at the patch site) |
| `daydream/runner.py` (MODIFY) | controller (RunConfig + run flow orchestrator) | request-response (async run flow) | itself (existing `run()`/`run_pr_feedback()`/`run_trust()`/`run_deep()`) | exact (in-place edit) |
| `daydream/phases.py` (MODIFY) | service (~18 `run_agent()` call sites) | request-response | itself (existing call signatures) | exact (in-place call-site update — pass `phase=DaydreamPhase.X`) |
| `tests/conftest.py` (MODIFY) | test infrastructure (autouse fixture) | test setup/teardown | `tests/test_phase_parse_input_path.py:32` (autouse pattern) + `daydream.agent.reset_state()` (function-level reset analog) | role-match (compose: `autouse=True` shape + `reset_state()` semantics) |

---

## Pattern Assignments

### `daydream/trajectory.py` (NEW — recorder + invocation + redactor + ContextVar + enums)

**Two analogs apply, splitting concerns:**

1. **Module placement / top-level-flat-file shape** — model on `daydream/exploration_runner.py` (and `daydream/pr_review.py`).
2. **Singleton + getter/setter accessor pattern** — model on `daydream/agent.py` `AgentState` block.
3. **ATIF model construction** — Pydantic models from `daydream/atif/models/*.py` (Phase 1 vendored output).

#### Analog 1: Top-level orchestrator module shape (`daydream/exploration_runner.py:1–46`)

**Module docstring + future-annotations + TYPE_CHECKING imports** — copy this file-header shape:

```python
"""Pre-scan orchestrator.

Counts changed files in a diff, selects an exploration tier (skip / single /
parallel), launches specialist ``backend.execute()`` calls in parallel...
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from daydream.exploration import (
    Convention,
    Dependency,
    ExplorationContext,
    ...
)

if TYPE_CHECKING:
    from pathlib import Path

    from daydream.backends import Backend
```

Apply: `daydream/trajectory.py` opens with `from __future__ import annotations`, uses `TYPE_CHECKING` to push `daydream.atif.*` imports out of import-time when possible (Pydantic class defs are heavy), `from typing import TYPE_CHECKING, Any` at the top.

#### Analog 2: Module-singleton + getter/setter pair (`daydream/agent.py:48–123`)

**Singleton dataclass declaration** (lines 48–66):

```python
@dataclass
class AgentState:
    """Consolidated state for agent module.

    Attributes:
        debug_log: File handle for debug logging, or None to disable.
        quiet_mode: True to hide tool calls and results, False to show them.
        ...
    """

    debug_log: TextIO | None = None
    quiet_mode: bool = False
    model: str = "opus"
    shutdown_requested: bool = False
    current_backends: list[Backend] = field(default_factory=list)


# Module-level Singletons
# =======================
# This module uses a singleton pattern for global state management. The module
# is imported once, creating these instances which persist for the process lifetime.
# Access and modify state through the getter/setter functions below (get_state,
# set_debug_log, set_quiet_mode, etc.) rather than accessing _state directly.
# Use reset_state() to restore defaults between test runs or CLI invocations.

_state = AgentState()
```

**Getter/setter pair** (lines 103–123):

```python
def set_debug_log(log_file: TextIO | None) -> None:
    """Set the debug log file handle.

    Args:
        log_file: File handle for debug logging, or None to disable.

    Returns:
        None
    """
    _state.debug_log = log_file


def get_debug_log() -> TextIO | None:
    """Get the current debug log file handle.

    Returns:
        The current debug log file handle, or None if not set.
    """
    return _state.debug_log
```

**Reset function** (lines 90–100) — Phase 2 needs an analog `_RECORDER_VAR.set(None)` reset path for the autouse fixture:

```python
def reset_state() -> None:
    """Reset the global agent state to defaults.

    Creates a new AgentState instance with default values.

    Returns:
        None
    """
    global _state
    _state = AgentState()
```

**Apply to `daydream/trajectory.py`:**

- Replace module-level `_state = AgentState()` with **`_RECORDER_VAR: ContextVar[TrajectoryRecorder | None] = ContextVar("_RECORDER_VAR", default=None)`** — a ContextVar instead of a module-level dataclass instance. Per PROJECT.md Constraints "propagated via `ContextVar` (not `AgentState`)".
- Provide a single public accessor `get_current_recorder() -> TrajectoryRecorder | None` (per D-10) that returns `_RECORDER_VAR.get()`. NO `set_*` companion — setter is implicit via `TrajectoryRecorder.__aenter__` (`_RECORDER_VAR.set(self)`).
- Mirror the `# Module-level Singletons` block-comment header so the architectural intent is documented inline.

#### Analog 3: ATIF Pydantic model construction (Phase 1 vendored output)

**Source files:**
- `daydream/atif/models/step.py` — Step constructor signature (lines 14–77).
- `daydream/atif/models/tool_call.py` — ToolCall constructor (lines 8–24).
- `daydream/atif/models/observation.py` + `observation_result.py` — Observation/ObservationResult.
- `daydream/atif/models/metrics.py` — Metrics (lines 8–44).
- `daydream/atif/models/final_metrics.py` — FinalMetrics (lines 8–40).
- `daydream/atif/models/trajectory.py` — Trajectory (lines 12–57) + `to_json_dict()` (lines 59–68).
- `daydream/atif/models/agent.py` — Agent (lines 8–32).

**Public API import (use the re-export shim, NOT submodule paths)** — from `daydream/atif/__init__.py:13–25`:

```python
from daydream.atif import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
    validate,
)
```

**Step construction (agent step from `AssistantMessage`)** — required fields: `step_id` (≥ 1, sequential), `source` (`"system" | "user" | "agent"`), `message` (str). Agent-only fields constrained: per `Step.validate_agent_only_fields` (`models/step.py:92–109`), `model_name`, `reasoning_effort`, `reasoning_content`, `tool_calls`, `metrics` MUST be `None` if `source != "agent"`. Phase 2 user steps (from `[PROMPT]`) MUST omit those fields:

```python
# User step from prompt
Step(
    step_id=next_step_id,            # 1-indexed sequential
    timestamp=now_iso(),              # ISO 8601 UTC ending in "Z"
    source="user",
    message=prompt_text,
    extra={
        "daydream_phase": phase.value,        # e.g. "review"
        "daydream_run_flow": run_flow.value,  # e.g. "normal"
    },
)

# Agent step from AssistantMessage (built incrementally as events stream)
Step(
    step_id=next_step_id,
    timestamp=now_iso(),
    source="agent",
    model_name=...,                                # from MetricsEvent or AgentState.model
    message="".join(text_chunks),                  # concatenated TextEvent.text per D-03
    reasoning_content="\n".join(thinking_chunks),  # concatenated ThinkingEvent.text or None
    tool_calls=[ToolCall(...), ...] or None,
    observation=Observation(results=[ObservationResult(...), ...]) or None,
    metrics=Metrics(...) or None,                  # from MetricsEvent (D-04)
    extra={"daydream_phase": phase.value, "daydream_run_flow": run_flow.value},
)
```

**ToolCall construction** (every field required):

```python
ToolCall(
    tool_call_id=event.id,            # from ToolStartEvent.id (Claude block.id or Codex UUID)
    function_name=event.name,         # ToolStartEvent.name
    arguments=event.input,            # ToolStartEvent.input dict (must be dict, never None)
)
```

**ObservationResult — `source_call_id` MUST point at a ToolCall in the SAME step** (per `Trajectory.validate_tool_call_references`, `models/trajectory.py:82–103`). Hence the in-flight `tool_call_id → Step` map per CORE-06: a ToolResultEvent always lands on whichever step holds the matching ToolStartEvent.

```python
ObservationResult(
    source_call_id=event.id,          # ToolResultEvent.id; MUST match a ToolCall in this Step
    content=event.output,             # str (or list[ContentPart] in v1.6 multimodal — out of scope)
)
```

**Metrics — `cached_tokens` is a SUBSET of `prompt_tokens`** per D-15:

```python
Metrics(
    prompt_tokens=usage.get("input_tokens"),       # NOT input + cached
    completion_tokens=usage.get("output_tokens"),
    cached_tokens=usage.get("cache_read_input_tokens"),  # subset, not added
    cost_usd=event.cost_usd,                       # may be None for Codex
)
```

**Trajectory finalize** (recorder `__aexit__`):

```python
trajectory = Trajectory(
    schema_version="ATIF-v1.6",       # pinned per PROJECT.md
    session_id=self._session_id,      # uuid.uuid4() at recorder init
    agent=Agent(name="daydream", version="0.1.0", model_name=...),
    steps=self._all_steps,            # list[Step], step_id 1..N sequential
    final_metrics=FinalMetrics(
        total_prompt_tokens=...,
        total_completion_tokens=...,
        total_cached_tokens=...,
        total_cost_usd=...,
        total_steps=len(self._all_steps),
    ),
)
# Use to_json_dict() — exclude_none=True drops absent fields per ATIF custom (default).
path.write_text(json.dumps(trajectory.to_json_dict(), indent=2))
```

**Validation invariants the recorder MUST honor (so `daydream.atif.validate()` returns True):**

1. `step_id` MUST be sequential from 1 (`models/trajectory.py:70–80`).
2. `source` ∈ `{"system", "user", "agent"}` (`models/step.py:26`).
3. `message` is required (str or list[ContentPart]; Phase 2 = str only per PROJECT.md "no multimodal").
4. Agent-only fields MUST be None on non-agent steps (`models/step.py:92–109`).
5. `ObservationResult.source_call_id` MUST match a `ToolCall.tool_call_id` in the SAME step (`models/trajectory.py:82–103`).
6. Timestamps MUST be ISO 8601 (parseable via `datetime.fromisoformat(v.replace("Z", "+00:00"))` per `models/step.py:81–90`).
7. `model_config = {"extra": "forbid"}` on every model — extra fields land in the `extra` dict, NOT as new top-level keys.

#### `now_iso()` helper — NEW utility (no analog in codebase)

`grep` found no existing ISO 8601 timestamp helper. The codebase uses `datetime.now().strftime(...)` for filename-only timestamps (`daydream/runner.py:463`, `daydream/phases.py:1260`, `phases.py:1369`). Per PITFALLS Pitfall 2, Phase 2 introduces a single source-of-truth helper:

```python
from datetime import datetime, timezone

def now_iso() -> str:
    """Return current UTC time as ISO 8601 with trailing 'Z'.

    Returns:
        Timestamp string parseable by Step.validate_timestamp.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
```

**Where it lives:** in `daydream/trajectory.py` (Phase 2 sole consumer); imported by `daydream/backends/claude.py` and `daydream/backends/codex.py` for event-yield-time stamping per CONTEXT.md `<code_context>` paragraph on "single source of truth".

**Ban:** `datetime.utcnow()` (deprecated; lacks tz) — already absent from the codebase, keep it absent.

---

### `daydream/agent.py` (MODIFY — `run_agent` event loop)

**In-place edit; itself is the analog.**

#### Imports & signature (lines 1–33, 283–292)

**Current `run_agent` signature** (line 283, no `phase` arg):

```python
async def run_agent(
    backend: Backend,
    cwd: Path,
    prompt: str,
    output_schema: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    continuation: ContinuationToken | None = None,
    agents: dict[str, AgentDefinition] | None = None,
    max_turns: int | None = None,
) -> tuple[str | Any, ContinuationToken | None]:
```

**Phase 2 signature** (per D-05 — keyword-only required):

```python
async def run_agent(
    backend: Backend,
    cwd: Path,
    prompt: str,
    *,
    phase: DaydreamPhase,                          # NEW: required keyword-only (D-05)
    output_schema: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    continuation: ContinuationToken | None = None,
    agents: dict[str, AgentDefinition] | None = None,
    max_turns: int | None = None,
) -> tuple[str | Any, ContinuationToken | None]:
```

Add import: `from daydream.trajectory import DaydreamPhase, get_current_recorder`.

Note: pre-existing kwargs (`output_schema`, `progress_callback`, …) are currently positional-or-keyword. Adding `*` BEFORE `phase` makes everything after `prompt` keyword-only. Since every existing call site already passes them by name (verified by reading `phases.py:720, 774, 827, 861, 887, 922, 950, 1011, 1060, 1078, 1113, 1156, 1221, 1357, 1463, 1544`), this is non-breaking — but consciously document the bar shift in the docstring.

#### Existing event-loop dispatch (lines 339–430) — the insertion point

The single chokepoint per Architecture Q2:

```python
async for event in event_iter:
    if isinstance(event, TextEvent):
        output_parts.append(event.text)
        _log_debug(f"[TEXT] {event.text}\n")
        # ... existing UI dispatch ...

    elif isinstance(event, ThinkingEvent):
        _log_debug(f"[THINKING] {event.text}\n")
        # ... existing UI dispatch ...

    elif isinstance(event, ToolStartEvent):
        _log_debug(f"[TOOL_USE] {event.name}({event.input}) id={event.id} ...")
        # ... existing UI dispatch ...

    elif isinstance(event, ToolResultEvent):
        _log_debug(f"[TOOL_RESULT...] id={event.id} ...")
        # ... existing UI dispatch ...

    elif isinstance(event, CostEvent):
        if event.cost_usd:
            _log_debug(f"[COST] ${event.cost_usd:.4f}\n")
            # ... existing UI dispatch ...
        elif event.input_tokens is not None:
            _log_debug(f"[TOKENS] in={event.input_tokens} out={event.output_tokens}\n")

    elif isinstance(event, ResultEvent):
        if event.structured_output is not None:
            structured_result = event.structured_output
            # ... existing structured-output handling ...
        result_continuation = event.continuation
```

#### Phase 2 insertion shape

**Wrap the event loop with the `Invocation` lifecycle** (per D-09 — one `Invocation` per `run_agent()` call). Insertion is *before* the per-event dispatch, NOT inside any branch. The new `MetricsEvent` branch is added ONCE alongside the existing `isinstance` chain:

```python
recorder = get_current_recorder()

# Open invocation scope for this run_agent call
async with (recorder.invocation(phase=phase) if recorder is not None else nullcontext()) as inv:
    if inv is not None:
        inv.observe_user_step(prompt=prompt)        # one user Step from [PROMPT]

    async for event in event_iter:
        if isinstance(event, TextEvent):
            # ... existing UI dispatch (unchanged) ...
            if inv is not None:
                inv.observe(event)                   # NEW: append to current agent step

        elif isinstance(event, ThinkingEvent):
            # ... existing dispatch ...
            if inv is not None:
                inv.observe(event)

        elif isinstance(event, ToolStartEvent):
            # ... existing dispatch ...
            if inv is not None:
                inv.observe(event)                   # registers tool_call_id → current step

        elif isinstance(event, ToolResultEvent):
            # ... existing dispatch ...
            if inv is not None:
                inv.observe(event)                   # looks up step via in-flight map

        elif isinstance(event, MetricsEvent):        # NEW BRANCH (D-04)
            if inv is not None:
                inv.observe(event)                   # attaches Metrics to step keyed by message_id

        elif isinstance(event, CostEvent):
            # ... existing dispatch ...
            if inv is not None:
                inv.observe(event)                   # contributes to FinalMetrics aggregate

        elif isinstance(event, ResultEvent):
            # ... existing dispatch (unchanged) ...
            if inv is not None:
                inv.observe(event)                   # closes any open agent step
```

**Constraints:**
- Existing `_log_debug` calls STAY (Phase 4 cutover removes them).
- `nullcontext()` from `contextlib` keeps the no-recorder branch shape identical (test code without conftest fixture continues to pass).
- D-19 (module-bloat ban): NO `Step()` / `ToolCall()` / `Trajectory()` construction in `agent.py`. The single allowed call is `inv.observe(event)`.
- Catch-and-degrade per Architecture Q7 — wrap `inv.observe()` in a `try/except Exception` only at the recorder boundary inside `daydream/trajectory.py`, NOT in `agent.py`. Agent code never silently swallows.

---

### `daydream/backends/__init__.py` (MODIFY — event dataclass enrichment)

**In-place edit; itself is the analog.**

#### Existing dataclass shape (lines 21–78)

```python
@dataclass
class TextEvent:
    """Agent text output."""
    text: str


@dataclass
class ThinkingEvent:
    """Extended thinking / reasoning."""
    text: str


@dataclass
class ToolStartEvent:
    """Tool invocation started."""
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultEvent:
    """Tool invocation completed."""
    id: str
    output: str
    is_error: bool


@dataclass
class CostEvent:
    """Cost and usage information."""
    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None


@dataclass
class ResultEvent:
    """Final event in the stream. Carries structured output and continuation token."""
    structured_output: Any | None
    continuation: ContinuationToken | None


AgentEvent = TextEvent | ThinkingEvent | ToolStartEvent | ToolResultEvent | CostEvent | ResultEvent
```

#### Phase 2 changes

**1. `timestamp: str` field added to EVERY event dataclass.** Per CONTEXT.md `<code_context>` reconciliation paragraph: the dataclass holds the timestamp, the backend yield site populates it, the recorder reads from it.

```python
@dataclass
class TextEvent:
    """Agent text output."""
    text: str
    timestamp: str = field(default_factory=now_iso)   # NEW; ISO 8601 UTC ending in "Z"
```

Apply identically to `ThinkingEvent`, `ToolStartEvent`, `ToolResultEvent`, `CostEvent`, `ResultEvent`.

Add imports: `from dataclasses import dataclass, field` (already imports `dataclass`; adding `field`); `from daydream.trajectory import now_iso`.

**Note on circular imports:** `daydream/trajectory.py` imports from `daydream.atif` (heavy) but does NOT import from `daydream.backends`. `daydream.backends.__init__` would import only `now_iso` from trajectory — keep that one symbol cheap (no top-level Pydantic imports in `trajectory.py`; lift them to TYPE_CHECKING or function-local where possible).

**2. NEW `MetricsEvent` dataclass** (per EVNT-02, EVNT-06, EVNT-07):

```python
@dataclass
class MetricsEvent:
    """Per-step LLM token/cost usage.

    Emitted once per AssistantMessage by the Claude backend (keyed via
    AssistantMessage.message_id), and once per turn.completed by the Codex
    backend. Lands on the agent Step opened by the matching message_id (D-04).

    Attributes:
        message_id: Identifier matching the AssistantMessage that owns this
            metric. Empty string for backends that do not surface a per-message
            id (Codex). The recorder uses message_id to attach Metrics to the
            correct Step.
        input_tokens: Prompt tokens for this turn (None when unavailable).
        output_tokens: Completion tokens for this turn (None when unavailable).
        cached_tokens: Subset of input_tokens served from cache (None when
            unavailable; Codex always emits None per D-16).
        cost_usd: Per-turn cost in USD (None when unavailable; Codex always
            emits None — see EVNT-07 / D-16).
        timestamp: ISO 8601 UTC, defaults to now_iso() at yield time.
    """

    message_id: str
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    cost_usd: float | None
    timestamp: str = field(default_factory=now_iso)
```

**3. Extend `CostEvent` with `cached_tokens`** (per EVNT-04):

```python
@dataclass
class CostEvent:
    """Cost and usage information."""
    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None     # NEW (per EVNT-04, D-15)
    timestamp: str = field(default_factory=now_iso)
```

**4. Update the union TypeAlias** (line 78):

```python
AgentEvent = (
    TextEvent
    | ThinkingEvent
    | ToolStartEvent
    | ToolResultEvent
    | CostEvent
    | MetricsEvent      # NEW
    | ResultEvent
)
```

**5. Update `__all__`** (lines 128–140) to add `"MetricsEvent"`.

---

### `daydream/backends/claude.py` (MODIFY — populate usage + emit MetricsEvent)

**In-place edit; itself is the analog. Patch sites: lines 95 (AssistantMessage handling) and 120–128 (ResultMessage handling).**

#### Existing `AssistantMessage` branch (line 95)

Currently emits `TextEvent`/`ThinkingEvent`/`ToolStartEvent` only. Phase 2 emits `MetricsEvent` after processing all blocks:

```python
if isinstance(msg, AssistantMessage):
    for block in msg.content:
        if isinstance(block, TextBlock) and block.text:
            yield TextEvent(text=block.text)
        elif isinstance(block, ThinkingBlock) and block.thinking:
            yield ThinkingEvent(text=block.thinking)
        elif isinstance(block, ToolUseBlock):
            if block.name == "StructuredOutput":
                continue
            yield ToolStartEvent(
                id=block.id,
                name=block.name,
                input=block.input or {},
            )
    # Phase 2 (EVNT-06): emit MetricsEvent keyed by message_id (D-04)
    if msg.usage is not None:
        yield MetricsEvent(
            message_id=getattr(msg, "message_id", "") or "",
            input_tokens=msg.usage.get("input_tokens"),
            output_tokens=msg.usage.get("output_tokens"),
            cached_tokens=msg.usage.get("cache_read_input_tokens"),
            cost_usd=None,                         # not available per-message; D-Discretion
        )
```

#### Existing `ResultMessage` branch (lines 120–128) — **the patch site that drops tokens today**

```python
# CURRENT (Phase 1)
elif isinstance(msg, ResultMessage):
    if msg.structured_output is not None:
        structured_result = msg.structured_output
    if msg.total_cost_usd is not None:
        yield CostEvent(
            cost_usd=msg.total_cost_usd,
            input_tokens=None,                    # ← DROPPED
            output_tokens=None,                   # ← DROPPED
        )
```

```python
# PHASE 2 (EVNT-04, EVNT-05)
elif isinstance(msg, ResultMessage):
    if msg.structured_output is not None:
        structured_result = msg.structured_output
    if msg.total_cost_usd is not None or msg.usage is not None:
        usage = msg.usage or {}
        yield CostEvent(
            cost_usd=msg.total_cost_usd,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cached_tokens=usage.get("cache_read_input_tokens"),  # NEW field on CostEvent
        )
```

Per D-14: trust per-call semantics; no defensive "subtract last_seen_cumulative" subtraction in Phase 2 — Phase 5 TEST-06 confirms via empirical multi-turn fixture.

#### Add imports

```python
from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    CostEvent,
    MetricsEvent,           # NEW
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
```

---

### `daydream/backends/codex.py` (MODIFY — emit MetricsEvent at turn.completed)

**In-place edit; itself is the analog. Patch site: line 308 (`turn.completed` branch).**

#### Existing branch (lines 308–315)

```python
elif event_type == "turn.completed":
    usage = event.get("usage", {})
    yield CostEvent(
        cost_usd=None,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )
```

#### Phase 2 shape

```python
elif event_type == "turn.completed":
    usage = event.get("usage", {})
    # Phase 2 (EVNT-07): emit MetricsEvent for the turn (D-16 parity gap acceptable)
    yield MetricsEvent(
        message_id="",                            # Codex has no AssistantMessage.message_id
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cached_tokens=None,                       # Codex parity gap (D-16)
        cost_usd=None,                            # Codex parity gap (D-16) — DO NOT synthesize
    )
    yield CostEvent(
        cost_usd=None,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cached_tokens=None,                       # NEW field on CostEvent
    )
```

**Constraint per PITFALLS technical-debt warning:** do NOT synthesize `cost_usd` from a token-price table. ATIF Metrics fields are all optional; `None` is the right answer.

#### Add import

```python
from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    CostEvent,
    MetricsEvent,           # NEW
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
```

---

### `daydream/runner.py` (MODIFY — RunConfig field + recorder construction in 4 run flows)

**In-place edit; itself is the analog.**

#### Existing `RunConfig` shape (lines 61–108)

```python
@dataclass
class RunConfig:
    """Configuration for a daydream run.

    Attributes:
        target: Target directory path for the review. ...
        ...
    """

    target: str | None = None
    skill: str | None = None
    model: str | None = None
    debug: bool = False
    cleanup: bool | None = None
    quiet: bool = True
    review_only: bool = False
    start_at: str = "review"
    pr_number: int | None = None
    bot: str | None = None
    backend: str = "claude"
    review_backend: str | None = None
    fix_backend: str | None = None
    test_backend: str | None = None
    loop: bool = False
    max_iterations: int = 5
    trust_the_technology: bool = False
    deep: bool = False
    exploration_context: ExplorationContext | None = None
    exploration_depth: int = 1
    ignore_paths: list[str] = field(default_factory=list)
```

#### Phase 2 addition

```python
    trajectory_path: Path | None = None    # NEW (Phase 2). Default-resolved to
                                           # <target>/.daydream/trajectory.json by
                                           # the run flows. Phase 4 wires --trajectory CLI flag.
```

Update Attributes docstring with the same prose.

#### Existing run flow constructions

**Four entry points each construct one recorder per call** (per D-07: `run_flow` = per-trajectory invariant):

| Flow | Function | DaydreamRunFlow | Existing wrap point |
|------|----------|-----------------|---------------------|
| Normal | `run()` (line 385) | `DaydreamRunFlow.NORMAL` | `with contextlib.ExitStack() as stack:` (line 466) — wrap inside the ExitStack |
| PR Feedback | `run_pr_feedback()` (line 195) | `DaydreamRunFlow.PR` | top of body, before `phase_fetch_pr_feedback` (line 227) |
| Trust-the-Technology | `run_trust()` (line 288) | `DaydreamRunFlow.TTT` | top of body, before `phase_understand_intent` (line 349) |
| Deep | `run_deep()` (in `daydream/deep/orchestrator.py`) | `DaydreamRunFlow.DEEP` | wrap pipeline body |

**Pattern (existing `set_debug_log` block at lines 461–472 — the Phase 2 analog):**

```python
# CURRENT (debug log setup — leave in place per CONTEXT.md "Existing debug-log
# setup at lines 461–472 is left in place — Phase 4 removes it")
debug_log_path: Path | None = None
if config.debug:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_log_path = target_dir / f".review-debug-{timestamp}.log"

with contextlib.ExitStack() as stack:
    if debug_log_path is not None:
        debug_log_file = stack.enter_context(open(debug_log_path, "w", encoding="utf-8"))
        set_debug_log(debug_log_file)
        stack.callback(set_debug_log, None)
        stack.callback(print_info, console, f"Debug log saved: {debug_log_path}")
        print_info(console, f"Debug log: {debug_log_path}")
```

**Phase 2 wrap shape** — `TrajectoryRecorder` is an `async with` context manager (CORE-09). The existing `with contextlib.ExitStack() as stack:` is sync; Phase 2 adds an `async with` *around or inside* the ExitStack. Cleanest shape: `async with` immediately after the ExitStack opens (the ExitStack stays for sync resources like the debug-log file handle):

```python
trajectory_path = config.trajectory_path or (target_dir / ".daydream" / "trajectory.json")

with contextlib.ExitStack() as stack:
    # ... existing debug-log setup stays unchanged ...

    async with TrajectoryRecorder(
        path=trajectory_path,
        run_flow=DaydreamRunFlow.NORMAL,            # PR / TTT / DEEP for the other flows
        target_dir=target_dir,
        model=config.model or "opus",
    ) as _recorder:
        # ... existing run body ...
```

`TrajectoryRecorder.__aenter__` calls `_RECORDER_VAR.set(self)`; `__aexit__` clears it and writes the trajectory. Implicit-write degrade-with-warning per D-11; explicit `--trajectory <path>` fail-loud branch is Phase 4.

**Imports added to `daydream/runner.py`:**

```python
from daydream.trajectory import DaydreamRunFlow, TrajectoryRecorder
```

---

### `daydream/phases.py` (MODIFY — pass `phase=DaydreamPhase.X` at every `run_agent` call site)

**In-place edit; itself is the analog. ZERO ATIF model construction (D-19, PROJECT.md ban).**

#### Existing call signature pattern

Phase functions follow the **Backend-first-parameter convention** (see CLAUDE.md Function Design):

```python
async def phase_review(
    backend: Backend,
    cwd: Path,
    skill: str,
    *,
    diff_base: str | None = None,
    exploration_dir: Path | None = None,
    exclude: list[str] | None = None,
) -> None:
    ...
    await run_agent(backend, cwd, prompt)
```

#### Phase 2 update — keyword-only `phase=` argument at all 17 call sites

Each `run_agent(...)` call gains `phase=DaydreamPhase.X`. Mapping (per D-06 enum members and the host phase functions):

| Line | Phase Function (`grep` host) | Pass |
|------|-------------------------------|------|
| 720 | `phase_review` | `phase=DaydreamPhase.REVIEW` |
| 774 | `phase_parse_feedback` | `phase=DaydreamPhase.PARSE` |
| 827 | `phase_fix` | `phase=DaydreamPhase.FIX` |
| 861 | `phase_test_and_heal` (test invocation) | `phase=DaydreamPhase.TEST` |
| 887 | `phase_test_and_heal` (heal sub-invocation) | `phase=DaydreamPhase.FIX` |
| 922 | `phase_commit_push` | `phase=DaydreamPhase.FIX` (commit work; PROJECT.md MAP-08 enum has no COMMIT — bucket under FIX or expand enum if discretion permits) — flag for planner |
| 950 | `phase_fetch_pr_feedback` | `phase=DaydreamPhase.PR_FEEDBACK` |
| 1011 | `phase_fix_parallel` (in-flight per parallel task — Phase 3 sibling rules apply, but Phase 2 still passes the label) | `phase=DaydreamPhase.FIX` |
| 1060 | `phase_commit_iteration` | `phase=DaydreamPhase.FIX` |
| 1078 | `phase_commit_push_auto` | `phase=DaydreamPhase.FIX` |
| 1113 | `phase_respond_pr_feedback` | `phase=DaydreamPhase.PR_FEEDBACK` |
| 1156 | `phase_understand_intent` | `phase=DaydreamPhase.INTENT` |
| 1221 | `phase_alternative_review` | `phase=DaydreamPhase.ALTERNATIVES` |
| 1357 | `phase_generate_plan` | `phase=DaydreamPhase.PLAN` |
| 1463 | `phase_per_stack_reviews` (per-stack inner) | `phase=DaydreamPhase.DEEP` |
| 1544 | `phase_cross_stack_merge` | `phase=DaydreamPhase.DEEP` |

**Plus:** `daydream/exploration_runner.py` `pre_scan()` (called from `runner.py` and `deep/orchestrator.py`) — exploration sub-invocations need `phase=DaydreamPhase.EXPLORATION`. Enumerate during planning to confirm the exploration call site landscape.

**Concrete update example** (line 720):

```python
# CURRENT
await run_agent(backend, cwd, prompt)

# PHASE 2
await run_agent(backend, cwd, prompt, phase=DaydreamPhase.REVIEW)
```

**Add import to `daydream/phases.py`:**

```python
from daydream.trajectory import DaydreamPhase
```

NO `from daydream.atif import Step, ToolCall, Trajectory` — D-19 explicit ban.

---

### `tests/conftest.py` (MODIFY — autouse `_reset_trajectory_recorder` fixture)

**Two analogs compose:**

1. **Autouse fixture pattern** — `tests/test_phase_parse_input_path.py:32–40` (file-local autouse).
2. **Reset semantics** — `daydream.agent.reset_state()` (function-level analog at `agent.py:90–100`).

#### Analog 1: File-local autouse pattern (`tests/test_phase_parse_input_path.py:32–40`)

```python
@pytest.fixture(autouse=True)
def _silence_ui(monkeypatch):
    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr(
        "daydream.phases.console",
        type("C", (), {"print": lambda *a, **kw: None})(),
    )
```

This is *file-local* — Phase 2 lifts the `autouse=True` shape into `tests/conftest.py` so it applies suite-wide.

#### Analog 2: Reset-function semantics (`daydream/agent.py:90–100`)

```python
def reset_state() -> None:
    """Reset the global agent state to defaults."""
    global _state
    _state = AgentState()
```

Phase 2 needs the equivalent for the ContextVar:

```python
# In daydream/trajectory.py (sibling of get_current_recorder)
def _reset_recorder_for_tests() -> None:
    """Test-only: clear the recorder ContextVar.

    Use only from the autouse conftest fixture to ensure cross-test isolation.
    Production code MUST go through TrajectoryRecorder.__aenter__/__aexit__.
    """
    _RECORDER_VAR.set(None)
```

#### Phase 2 conftest.py addition

```python
@pytest.fixture(autouse=True)
def _reset_trajectory_recorder():
    """Clear the trajectory ContextVar before and after every test.

    Mirrors daydream.agent.reset_state() for AgentState (CORE-10 / D-17).
    Prevents cross-test bleed when a test forgets to wrap recorder usage in
    ``async with TrajectoryRecorder(...)``.
    """
    from daydream.trajectory import _reset_recorder_for_tests

    _reset_recorder_for_tests()
    yield
    _reset_recorder_for_tests()
```

**Constraint:** the lazy import keeps `tests/conftest.py` from importing `daydream.trajectory` at collect time — matches the existing lazy-import idiom in `daydream/backends/codex.py:37` (`from daydream.agent import _log_debug` inside `_raw_log`).

---

## Shared Patterns

### Backend-first-parameter convention

**Source:** `daydream/phases.py:794` (and every other `phase_*` function).
**Apply to:** every `run_agent` call site in `phases.py` AND the new `phase=DaydreamPhase.X` keyword-only argument (D-05).

```python
async def phase_fix(backend: Backend, cwd: Path, item: dict[str, Any], item_num: int, total: int) -> None:
    ...
```

The **keyword-only** `phase=` arg is added with a literal `*,` separator in `run_agent`. Phase 2 does NOT add `phase=` to the `phase_*` functions — they already know their phase identity at the call site, and threading it through every signature would bloat the API.

### Module docstring + future-annotations + TYPE_CHECKING imports

**Source:** `daydream/exploration_runner.py:1–46`, `daydream/pr_review.py:1–32`.
**Apply to:** `daydream/trajectory.py`.

```python
"""<one-line summary>.

<paragraph describing the responsibility, life cycle, integration points>.
"""

from __future__ import annotations

import <stdlib>
from typing import TYPE_CHECKING, ...

if TYPE_CHECKING:
    from pathlib import Path
    from daydream.backends import AgentEvent
```

Reason: per CLAUDE.md "Code Style — Python 3.12 union syntax: `str | None`, `list[Backend]`" + "`from __future__ import annotations` used in modules that need forward references". Trajectory.py needs forward refs to `Invocation`, `Step`, etc., to avoid load-order issues with the heavy `daydream.atif` namespace.

### Error-handling: catch-and-degrade at the recording boundary (Architecture Q7)

**Source:** Architecture research Q7 ("catch-and-degrade at recording boundaries").
**Apply to:** `daydream/trajectory.py` recorder methods.
**NOT applied to:** `daydream/agent.py` event loop (no silent swallow per CLAUDE.md "No silent swallowing — always log or re-raise").

```python
# In daydream/trajectory.py — Invocation.observe()
def observe(self, event: AgentEvent) -> None:
    try:
        # ATIF model construction / step buffering
        self._dispatch(event)
    except Exception as exc:           # noqa: BLE001 — recording must never crash a run
        # Best-effort: emit a print_warning, NEVER swallow without trace.
        print_warning(console, f"Trajectory recording: {type(exc).__name__}: {exc}")
```

**`# noqa: BLE001` reason comment is REQUIRED** per CLAUDE.md Code Style ("`# noqa: BLE001` for intentionally broad exception catches in parallel isolation contexts").

The boundary is symmetric: `__aexit__` write-failure also catches and warns (D-11) for Phase 2's implicit-write path. Phase 4 (CLI-02) lights up the explicit fail-loud branch.

### `now_iso()` single source of truth (PITFALLS Pitfall 2)

**Source:** NEW utility in `daydream/trajectory.py` (no existing analog).
**Apply to:** `daydream/backends/__init__.py` `field(default_factory=now_iso)` on every event dataclass; `daydream/trajectory.py` Step construction; future Phase 4 partial-write paths.
**Ban:** `datetime.utcnow()` (deprecated, lacks tz); ad-hoc `datetime.now().isoformat()` (no `Z` suffix → may fail Pydantic validator on parse).

```python
from datetime import datetime, timezone

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
```

Validator round-trip check: `datetime.fromisoformat(now_iso().replace("Z", "+00:00"))` must succeed — see `daydream/atif/models/step.py:81–90`.

### Test pattern — schema validity + behavioral predicates (D-18)

**Source:** `tests/test_atif_vendor_smoke.py` (existing).
**Apply to:** Phase 5 `tests/test_trajectory.py` (Phase 5 owns), but the precedent is set in Phase 2.

```python
# Schema-validity assertion (existing pattern, lines 46–48)
@pytest.mark.parametrize("golden_path", _golden_paths(), ids=lambda p: p.name)
def test_golden_fixtures_validate(golden_path: Path) -> None:
    assert validate(golden_path) is True
```

Phase 2 tests added in plans for this phase MUST follow this shape: assert `daydream.atif.validate(produced_trajectory) is True` PLUS one or two specific behavioral predicates ("trajectory has at least one agent Step with a Bash tool call"). NO `assert produced == fixture_dict` deep-equality (PITFALLS Pitfall 11).

### Lazy-import idiom for circular avoidance

**Source:** `daydream/backends/codex.py:37` (`from daydream.agent import _log_debug` inside `_raw_log`).
**Apply to:** `tests/conftest.py` `_reset_trajectory_recorder` fixture (lazy `from daydream.trajectory import _reset_recorder_for_tests`); any path where `daydream/trajectory.py` would be imported by a test that also patches Pydantic-heavy paths.

```python
def _raw_log(message: str) -> None:
    """Log raw event to the agent debug log if available."""
    # Lazy import to avoid circular dependency at module load time
    from daydream.agent import _log_debug
    _log_debug(message)
```

### `# noqa:` reason-comment convention

**Source:** `daydream/runner.py:180` (`subprocess.run(...)  # noqa: S603 - arguments are not user-controlled`).
**Apply to:** `BLE001` in trajectory.py recorder boundary; any new `subprocess` calls Phase 2 introduces (none expected, but if added, follow the convention).

Per CLAUDE.md Code Style: "Inline `# noqa:` always followed by a reason comment".

---

## No Analog Found

| File / Construct | Role | Reason |
|-------|------|--------|
| `now_iso()` helper | utility | No ISO 8601 timestamp helper exists in the codebase today. Closest is `datetime.now().strftime("%Y%m%d_%H%M%S")` for filenames. Phase 2 introduces this as a NEW utility per PITFALLS Pitfall 2. |
| `ContextVar` propagation | state | No existing module uses `ContextVar`. The closest pattern is module-level singletons (`AgentState`, `console`). Phase 2 introduces ContextVar usage; future modules can model on it. |
| `async with` recorder lifecycle | resource | No existing `async with` user-defined context manager exists in the daydream package; the only `async with` usages are SDK clients (`ClaudeSDKClient`). The recorder's `__aenter__`/`__aexit__` shape is new — model on standard Python idiom (similar shape as `ClaudeSDKClient` usage at `daydream/backends/claude.py:90`). |
| Autouse global-test-isolation fixture | test infrastructure | `tests/conftest.py` does NOT currently have an autouse fixture (verified via `grep autouse tests/conftest.py` → no match). The only autouse fixture in the suite is `_silence_ui` at `tests/test_phase_parse_input_path.py:32` and per-file `_init_repo_with_exclude_fixture` at `tests/test_phases.py:288`. Phase 2 sets the precedent with `_reset_trajectory_recorder`. |

---

## Metadata

**Analog search scope:** `daydream/`, `daydream/atif/models/`, `daydream/backends/`, `daydream/deep/`, `tests/`, `tests/fixtures/atif_golden/`.

**Files scanned:** `daydream/agent.py`, `daydream/runner.py`, `daydream/phases.py`, `daydream/backends/__init__.py`, `daydream/backends/claude.py`, `daydream/backends/codex.py`, `daydream/exploration_runner.py`, `daydream/pr_review.py`, `daydream/deep/orchestrator.py`, `daydream/atif/__init__.py`, `daydream/atif/models/{step,tool_call,observation,observation_result,metrics,final_metrics,trajectory,agent}.py`, `daydream/atif/validator.py`, `tests/conftest.py`, `tests/test_atif_vendor_smoke.py`, `tests/test_phase_parse_input_path.py`, `tests/test_phases.py`, `tests/test_runner.py`, `tests/fixtures/atif_golden/terminus2/hello-world-invalid-json.trajectory.json`.

**Pattern extraction date:** 2026-04-26.
