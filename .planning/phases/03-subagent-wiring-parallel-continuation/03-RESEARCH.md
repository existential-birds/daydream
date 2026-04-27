# Phase 3: Subagent Wiring (Parallel + Continuation) - Research

**Researched:** 2026-04-27
**Domain:** anyio ContextVar propagation, ATIF subagent linking, parallel trajectory recording
**Confidence:** HIGH

## Summary

Phase 3 extends the Phase 2 `TrajectoryRecorder` (498 LOC in `daydream/trajectory.py`) with `fork()`, `_register_sibling()`, `create_dispatch_step()`, and `_safe_descriptor()` to support three parallel fan-out sites and verify that continuation flows already work correctly. The core mechanism is anyio's ContextVar copy-on-spawn behavior: each child task in a `create_task_group()` inherits the parent's ContextVar value but mutations in the child do not propagate back. This means `fork().__aenter__` can replace the ContextVar with a child recorder, all `run_agent()` calls inside that task pick up the child, and `fork().__aexit__` writes the sibling file and registers it with the parent.

The vendored ATIF models already support the linking mechanism: `ObservationResult.subagent_trajectory_ref` accepts a `list[SubagentTrajectoryRef]` where each ref carries a `session_id` and `trajectory_path`. A dispatch step with no `tool_calls` and no `source_call_id` passes the vendored validator. The `Trajectory.validate_step_ids` validator requires step_ids sequential from 1 per trajectory file, which naturally isolates child counters. Three integration sites change: `phase_fix_parallel` (unused in production but defined), `phase_per_stack_reviews` (deep mode), and `pre_scan._run_specialist` (exploration). No changes to `agent.py` or `runner.py`.

**Primary recommendation:** Implement `fork()` as an async context manager on `TrajectoryRecorder` that creates a child recorder, sets the ContextVar, writes sibling JSON on exit, and registers with parent. Integration sites wrap their per-task closures with `async with recorder.fork(descriptor):`. Add `create_dispatch_step()` called after each task group exits.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Explicit fork wrapper -- phases call `async with recorder.fork(descriptor):`
- **D-02:** `recorder.fork(descriptor)` returns a full `TrajectoryRecorder` with own step_id counter, steps list, output path. Inherits session_id, run_flow, agent_model_name, target_dir from parent. Carries parent backref.
- **D-03:** Auto-patch parent in child's `__aexit__` via `_register_sibling()`
- **D-04:** Single ContextVar (`_RECORDER_VAR`) -- no second ContextVar needed
- **D-05:** Semantic descriptor names (fix-0, deep-python, explore-pattern-scanner, etc.)
- **D-06:** Descriptors slugified via `_safe_descriptor()` using `re.sub(r'[^a-z0-9-]', '-', raw.lower()).strip('-')`
- **D-07:** Session ID first 8 hex chars in filename: `<target>/.daydream/trajectories/<first-8-hex>.<descriptor>.json`
- **D-08:** Each `run_agent()` creates new Invocation, continuation or not (unchanged from Phase 2)
- **D-09:** No continuation linking marker
- **D-10:** Same continuation behavior for both backends
- **D-11:** Synthetic dispatch step created after task group completes via `create_dispatch_step()`
- **D-12:** `subagent_trajectory_ref` uses relative paths from root trajectory directory
- **D-13:** Phase code calling `recorder.create_dispatch_step()` is acceptable under module-bloat ban

### Claude's Discretion
- Internal structure of `_register_sibling()` (list vs dict, thread-safety for concurrent registration)
- Whether `create_dispatch_step()` accepts additional kwargs for custom message text or uses a fixed template per phase
- Whether `fork()` accepts optional kwargs for overriding agent_model_name
- Exact error handling when a child recorder fails to write (degradation vs propagation)
- Whether `_safe_descriptor()` lives as a module-level helper or a staticmethod

### Deferred Ideas (OUT OF SCOPE)
- `_CURRENT_INVOCATION` ContextVar (eliminated by explicit fork approach)
- Mixed-backend deep mode (fork inherits agent_model_name from parent)
- Sibling trajectory streaming (children write full trajectory on __aexit__)
- Cross-sibling deduplication in FinalMetrics
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SUBA-01 | Sequential phase chain emits as continuous steps in ONE root trajectory file | Already works from Phase 2 -- continuation calls create new Invocations appending to same recorder. Verified via existing tests. |
| SUBA-02 | `phase_fix_parallel` emits one sibling per parallel fix invocation; root step's ObservationResult.subagent_trajectory_ref points to each | `fork()` wraps each `_fix_task` closure; `create_dispatch_step()` after task group exit. Function is currently unused in production but still needs wiring. |
| SUBA-03 | `run_deep()` per-stack fan-out emits one sibling per stack | `fork()` wraps each `_task` closure in `phase_per_stack_reviews()`; `create_dispatch_step()` after task group exit. |
| SUBA-04 | `pre_scan()` specialist subagents emit sibling trajectories | `fork()` wraps `_run_specialist()` closure; `create_dispatch_step()` after task group exit. |
| SUBA-05 | Continuation calls APPEND to same trajectory, do NOT spawn sibling | Already works -- `phase_test_and_heal` passes `continuation` to `run_agent()` which creates new Invocation per D-08. No Phase 3 changes needed beyond verification. |
| SUBA-06 | Sibling files inherit parent session_id, written to `<root>/.daydream/trajectories/<session_id>.<descriptor>.json` | `fork()` inherits `session_id` from parent. Path constructed via `_sibling_path()`. |
| SUBA-07 | ContextVar copy-on-spawn establishes parent-child relationship without explicit threading | Verified via empirical test: anyio 4.12.1 copies ContextVar values to child tasks; mutations isolated per task. |
| SUBA-08 | step_id counters isolated per trajectory file | Each child TrajectoryRecorder has its own `_step_id_counter` starting at 0. |
| SUBA-09 | Parent FinalMetrics aggregates ONLY parent steps; no double-counting | Each recorder has its own `_final_totals` dict. Child metrics stay in child file. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Fork lifecycle (create child recorder, set ContextVar, write on exit) | `daydream/trajectory.py` | -- | All ATIF model construction stays in trajectory.py per D-19 |
| Sibling registration (parent accumulates child paths) | `daydream/trajectory.py` | -- | Internal recorder bookkeeping |
| Dispatch step creation (synthetic step with subagent_trajectory_ref) | `daydream/trajectory.py` | `daydream/phases.py` (caller) | trajectory.py constructs the Step; phases.py calls `recorder.create_dispatch_step()` |
| Fork call sites (wrap task closures) | `daydream/phases.py` | `daydream/exploration_runner.py` | 2 sites in phases.py, 1 in exploration_runner.py |
| ContextVar propagation | Python runtime (anyio) | -- | anyio task groups copy ContextVar automatically |
| Continuation behavior | `daydream/agent.py` | -- | Already works; no changes needed |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| anyio | 4.12.1 | Task groups + ContextVar copy-on-spawn | Already the project's async runtime [VERIFIED: importlib.metadata] |
| pydantic | >=2.11.7 | ATIF model construction (SubagentTrajectoryRef, ObservationResult, Step) | Already vendored in daydream/atif/ [VERIFIED: pyproject.toml] |

### Supporting
No new dependencies needed. Phase 3 is purely additive methods on `TrajectoryRecorder` + integration site changes.

**Installation:** None required. All dependencies already present.

## Architecture Patterns

### System Architecture Diagram

```
                    runner.py / deep/orchestrator.py
                    opens TrajectoryRecorder (parent)
                    sets _RECORDER_VAR = parent
                              |
                    +---------+---------+
                    |                   |
               Sequential          Parallel fan-out
               phases              (anyio task group)
                    |                   |
              run_agent()        +------+------+------+
              picks up parent    |      |      |      |
              from ContextVar  fork() fork() fork() fork()
              appends steps    sets   sets   sets   sets
              to parent        CV=    CV=    CV=    CV=
                               child0 child1 child2 child3
                                 |      |      |      |
                             run_agent picks up childN
                             appends steps to childN
                                 |      |      |      |
                             __aexit__: write sibling JSON
                             + _register_sibling(parent)
                                 |      |      |      |
                              +--+------+------+------+
                              |
                    create_dispatch_step()
                    creates synthetic Step in parent
                    with ObservationResult.subagent_trajectory_ref
                    pointing at each sibling file
                              |
                    Remaining sequential phases
                    continue appending to parent
                              |
                    parent __aexit__: write root trajectory.json
```

### Recommended Project Structure (no changes)
```
daydream/
├── trajectory.py       # +80-120 LOC: fork(), _register_sibling(), create_dispatch_step(), _safe_descriptor()
├── agent.py            # NO CHANGES
├── phases.py           # +12-18 LOC: fork wrappers + create_dispatch_step calls at 2 sites
├── exploration_runner.py  # +6-8 LOC: fork wrapper at 1 site
├── backends/           # NO CHANGES
├── atif/               # NO CHANGES (SubagentTrajectoryRef already vendored)
└── deep/
    └── orchestrator.py # NO CHANGES (fork happens inside phase_per_stack_reviews)
```

### Pattern 1: Fork Wrapper in Task Closure
**What:** Each parallel task wraps its `run_agent()` call with `async with recorder.fork(descriptor):` so the child recorder is active for the duration of the task.
**When to use:** Every parallel fan-out that should produce a sibling trajectory.
**Example:**
```python
# Source: D-01 from 03-CONTEXT.md + existing phase_fix_parallel pattern
recorder = get_current_recorder()

async with anyio.create_task_group() as tg:
    for index, item in enumerate(feedback_items):
        async def _fix_task(
            task_index: int = index,
            task_item: dict[str, Any] = item,
            task_prompt: str = prompt,
        ) -> None:
            # Fork creates child recorder, sets ContextVar to child
            if recorder is not None:
                async with recorder.fork(f"fix-{task_index}"):
                    try:
                        async with limiter:
                            await run_agent(backend, cwd, task_prompt, ...)
                        # ...
                    except Exception as e:
                        # ...
            else:
                # No recorder active; run without fork
                try:
                    async with limiter:
                        await run_agent(backend, cwd, task_prompt, ...)
                except Exception as e:
                    # ...

        tg.start_soon(_fix_task)

# After task group exits, create dispatch step
if recorder is not None:
    recorder.create_dispatch_step(phase=DaydreamPhase.FIX)
```

### Pattern 2: Dispatch Step After Task Group
**What:** After `async with tg:` exits, the parent recorder creates a synthetic step that links to all successfully-registered siblings.
**When to use:** Immediately after every task group that used fork().
**Example:**
```python
# Source: D-11 from 03-CONTEXT.md
# create_dispatch_step() reads _registered_siblings and creates:
# Step(
#     source="agent",
#     message="Dispatching N parallel fix tasks",
#     observation=Observation(results=[
#         ObservationResult(
#             subagent_trajectory_ref=[
#                 SubagentTrajectoryRef(session_id=self.session_id, trajectory_path=rel_path)
#             ]
#         )
#         for each sibling
#     ]),
# )
```

### Pattern 3: Child Recorder __aexit__ Auto-Patch
**What:** When a child recorder's `__aexit__` runs, it writes its sibling JSON file and calls `self.parent._register_sibling(self.path, self.descriptor)` to add itself to the parent's sibling list.
**When to use:** Automatic -- no manual bookkeeping in phase code.
**Example:**
```python
# Source: D-03 from 03-CONTEXT.md
# Inside TrajectoryRecorder.__aexit__ (child path):
async def __aexit__(self, exc_type, exc_val, exc_tb):
    try:
        self._write()
    except Exception as exc:
        print_warning(_console, f"Sibling trajectory write failed: ...")
    finally:
        if self._previous_token is not None:
            _RECORDER_VAR.reset(self._previous_token)
            self._previous_token = None
    # Register with parent AFTER write (only includes successful writes)
    if self.parent is not None and self.path.exists():
        self.parent._register_sibling(self.path, self.descriptor)
```

### Anti-Patterns to Avoid
- **Passing recorder through function signatures:** Use ContextVar copy-on-spawn. `get_current_recorder()` inside `run_agent()` picks up whichever recorder is active. No signature changes needed.
- **Creating dispatch step inside task closures:** Dispatch step must be created AFTER the task group exits, not inside individual tasks. Only the parent recorder should create it.
- **Importing ATIF models in phases.py:** D-19 bans `Step()`, `ToolCall()`, etc. in phases.py. Use `recorder.create_dispatch_step()` which encapsulates construction.
- **Nested fork() calls:** Not needed and not supported. Each fan-out site is at most one level deep (parent -> child). If a child task itself contains parallel fan-out, that's a separate design problem for a future phase.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ContextVar propagation to child tasks | Manual parameter threading | anyio copy-on-spawn | Already works; anyio 4.12.1 copies ContextVar to task group children automatically [VERIFIED: empirical test] |
| ATIF subagent linking | Custom JSON linking format | `SubagentTrajectoryRef` Pydantic model | Already vendored in `daydream/atif/models/subagent_trajectory_ref.py` [VERIFIED: codebase] |
| Trajectory validation | Custom validation | `daydream.atif.validate()` | Validates step_id sequencing, tool_call references, subagent refs [VERIFIED: empirical test] |
| File path slugification | Custom per-site sanitization | Single `_safe_descriptor()` helper | Centralizes the regex; D-06 specifies exact pattern |

**Key insight:** Phase 3 is primarily integration work -- the infrastructure (ContextVar, ATIF models, validator, recorder) already exists from Phase 1+2. The new code is additive methods on `TrajectoryRecorder` plus thin wrappers at 3 call sites.

## Common Pitfalls

### Pitfall 1: ContextVar Reset Token Ordering in fork().__aexit__
**What goes wrong:** If `_RECORDER_VAR.reset(token)` is called before `_write()`, and `_write()` fails, the ContextVar is already restored to the parent but the sibling file is missing. Subsequent code in the same task would write to the parent instead of the (failed) child.
**Why it happens:** Premature token reset.
**How to avoid:** Reset the token in a `finally` block AFTER the write attempt. The write should come first; the ContextVar reset should happen regardless of write success.
**Warning signs:** Missing sibling files but no error messages.

### Pitfall 2: fork() Nesting with CapacityLimiter
**What goes wrong:** The fork() `async with` must nest INSIDE the task closure but can nest either inside or outside the `async with limiter:` block. If fork() is outside the limiter, the child recorder is active even while waiting for the limiter, which is semantically correct but the ContextVar is set before the task "starts work."
**Why it happens:** Ambiguous nesting order.
**How to avoid:** Put fork() as the outermost wrapper in the task closure (before limiter). This ensures the child recorder is active for the entire task lifetime including the limiter wait. The child recorder doesn't accumulate any steps while waiting, so this is safe.
**Warning signs:** Step timestamps that predate the actual work.

### Pitfall 3: _register_sibling Concurrency
**What goes wrong:** Multiple child tasks calling `parent._register_sibling()` simultaneously could corrupt the parent's sibling list if the method is not atomic.
**Why it happens:** anyio tasks can suspend at any `await` point; if `_register_sibling` contains an await, interleaving can occur.
**How to avoid:** `_register_sibling()` must be a synchronous method (no `await`). Since anyio uses asyncio (single-threaded cooperative), a synchronous `list.append()` is atomic -- no interleaving possible within a single synchronous operation. [VERIFIED: asyncio is single-threaded; anyio uses asyncio backend in this project]
**Warning signs:** Duplicate or missing sibling registrations.

### Pitfall 4: phase_fix_parallel Is Currently Dead Code
**What goes wrong:** Implementing fork() in `phase_fix_parallel` but never testing it in an integration flow because no production code path calls it.
**Why it happens:** `phase_fix_parallel` is defined in `phases.py` but not imported or called from `runner.py` or any orchestrator. The normal flow uses sequential `phase_fix` (runner.py line 624). [VERIFIED: grep confirms no callers]
**How to avoid:** The Phase 3 tests must directly invoke `phase_fix_parallel` with a mock backend and verify the sibling output. Do not rely on end-to-end flow testing for this function.
**Warning signs:** Untested fork integration in a function that appears to be actively used.

### Pitfall 5: Dispatch Step step_id Sequencing
**What goes wrong:** `create_dispatch_step()` must use the parent's `_next_step_id()` counter. If it's accidentally called on a child recorder, or if steps from the child leak into the parent, step_id sequencing breaks and Pydantic's `validate_step_ids` rejects the trajectory.
**Why it happens:** `create_dispatch_step()` is called after the task group exits, when the ContextVar is back to the parent. But if there's a bug in reset ordering, the "current recorder" might not be the parent.
**How to avoid:** `create_dispatch_step()` is called explicitly on the recorder instance (not via `get_current_recorder()`), so there's no ContextVar ambiguity. The method uses `self._next_step_id()` which is always the parent's counter.
**Warning signs:** Trajectory validation error "expected step_id N, got M".

### Pitfall 6: Empty Trajectory Handling for Failed Children
**What goes wrong:** A child task fails before any `run_agent()` call, producing a child recorder with 0 steps. `Trajectory.steps` has `min_length=1`, so Pydantic rejects it.
**Why it happens:** Exception in task setup before the agent call.
**How to avoid:** The existing `TrajectoryRecorder._write()` already has a guard: `if not self.steps: return`. The child simply won't produce a file. Since `_register_sibling()` is only called if `self.path.exists()` (per D-03), the parent won't reference it.
**Warning signs:** Orphan sibling refs pointing to non-existent files.

### Pitfall 7: _safe_descriptor Double-Hyphens
**What goes wrong:** The regex `re.sub(r'[^a-z0-9-]', '-', raw.lower()).strip('-')` produces double-hyphens for inputs like `Fix_Issue (3)` -> `fix-issue--3`.
**Why it happens:** Multiple consecutive non-alphanumeric characters each get replaced with a hyphen.
**How to avoid:** Add a `re.sub(r'-{2,}', '-', ...)` collapse step, or document that double-hyphens are acceptable. For the actual descriptors used (D-05: `fix-0`, `deep-python`, `explore-pattern-scanner`), this never occurs -- they're all clean lowercase-hyphenated strings. [VERIFIED: empirical test]
**Warning signs:** Ugly filenames but no functional issue.

## Code Examples

### fork() Implementation Skeleton
```python
# Source: D-01, D-02, D-03, D-04, D-07 from 03-CONTEXT.md
# In daydream/trajectory.py

def _safe_descriptor(raw: str) -> str:
    """Slugify a descriptor to filesystem-safe characters (D-06)."""
    return re.sub(r'[^a-z0-9-]', '-', raw.lower()).strip('-')

def _sibling_path(self) -> Path:
    """Compute sibling trajectory file path (D-07)."""
    short_id = self.session_id[:8]  # first 8 hex chars
    return (
        self.target_dir / ".daydream" / "trajectories"
        / f"{short_id}.{_safe_descriptor(self.descriptor)}.json"
    )

# On TrajectoryRecorder:

def fork(self, descriptor: str) -> "_ForkCM":
    """Create a child recorder for a parallel task (D-01, D-02).

    Usage: ``async with recorder.fork("fix-0") as child:``

    The child inherits session_id, run_flow, agent_model_name,
    target_dir, and redactor from the parent. It has its own step_id
    counter (starts at 0), steps list, and output path.
    """
    return _ForkCM(parent=self, descriptor=descriptor)
```

### _ForkCM Implementation Skeleton
```python
# Source: D-02, D-03, D-04 from 03-CONTEXT.md

class _ForkCM:
    def __init__(self, parent: TrajectoryRecorder, descriptor: str) -> None:
        self._parent = parent
        self._descriptor = descriptor
        self._child: TrajectoryRecorder | None = None

    async def __aenter__(self) -> TrajectoryRecorder:
        child = TrajectoryRecorder(
            path=self._parent._sibling_path_for(self._descriptor),
            run_flow=self._parent.run_flow,
            target_dir=self._parent.target_dir,
            agent_model_name=self._parent.agent_model_name,
            redactor=self._parent.redactor,
            session_id=self._parent.session_id,
        )
        child.parent = self._parent
        child.descriptor = self._descriptor
        child._previous_token = _RECORDER_VAR.set(child)
        self._child = child
        return child

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        child = self._child
        if child is None:
            return
        try:
            child._write()
        except Exception as exc:
            print_warning(_console, f"Sibling trajectory write failed: {exc}")
        finally:
            if child._previous_token is not None:
                _RECORDER_VAR.reset(child._previous_token)
                child._previous_token = None
        # Register with parent only if file was written
        if child.path.exists():
            self._parent._register_sibling(child.path, self._descriptor)
```

### create_dispatch_step() Skeleton
```python
# Source: D-11, D-12, D-13 from 03-CONTEXT.md

def create_dispatch_step(self, *, phase: DaydreamPhase) -> None:
    """Create a synthetic dispatch step linking to registered siblings (D-11).

    Called from phase code after ``async with tg:`` exits. Creates one
    Step with ObservationResult entries pointing at each sibling file.
    Only includes siblings that wrote their file (no orphan refs).
    """
    if not self._registered_siblings:
        return

    results = []
    for sibling_path, descriptor in self._registered_siblings:
        rel_path = sibling_path.relative_to(self.target_dir / ".daydream")
        results.append(
            ObservationResult(
                content=f"Dispatched to {descriptor}",
                subagent_trajectory_ref=[
                    SubagentTrajectoryRef(
                        session_id=self.session_id,
                        trajectory_path=str(rel_path),
                    )
                ],
            )
        )

    count = len(self._registered_siblings)
    step = Step(
        step_id=self._next_step_id(),
        timestamp=now_iso(),
        source="agent",
        model_name=self.agent_model_name,
        message=f"Dispatching {count} parallel {phase.value} tasks",
        observation=Observation(results=results),
        extra={
            "daydream_phase": phase.value,
            "daydream_run_flow": self.run_flow.value,
        },
    )
    self.steps.append(self.redactor.redact_step(step))
    # Clear siblings for potential reuse (unlikely but clean)
    self._registered_siblings.clear()
```

### Integration Site: phase_per_stack_reviews
```python
# Source: D-01, D-05 from 03-CONTEXT.md + existing code at phases.py:1435-1487
# Shows the delta to the existing function

recorder = get_current_recorder()

async with anyio.create_task_group() as tg:
    for stack in stacks:
        # ... existing prompt building ...

        async def _task(
            stack_name: str = stack.stack_name,
            task_prompt: str = prompt,
            task_output: Path = output_path,
        ) -> None:
            try:
                if recorder is not None:
                    async with recorder.fork(f"deep-{stack_name}"):
                        async with limiter:
                            await run_agent(backend, cwd, task_prompt, phase=DaydreamPhase.DEEP)
                else:
                    async with limiter:
                        await run_agent(backend, cwd, task_prompt, phase=DaydreamPhase.DEEP)
                results[stack_name] = task_output
            except Exception as e:
                # ... existing error handling ...

        tg.start_soon(_task)

# After task group exits:
if recorder is not None:
    recorder.create_dispatch_step(phase=DaydreamPhase.DEEP)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Phase 2 ContextVar with no fork | Phase 3 adds fork() for parallel task groups | This phase | Sibling trajectory files for parallel fan-outs |
| Flat step list for all run_agent calls | Same flat list for sequential; dispatch step for parallel | This phase | Root trajectory has ObservationResult refs to siblings |
| No subagent linking | SubagentTrajectoryRef in dispatch steps | This phase | Machine-parseable execution graph |

**Deprecated/outdated:**
- Phase 2 docstring mentions "Phase 3 adds the second ContextVar" -- this was rejected by D-04. Single ContextVar is the final design. The docstring at trajectory.py line 15 needs updating.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `_register_sibling()` using synchronous `list.append()` is safe under anyio cooperative multitasking (no mutex needed) | Common Pitfalls / Pitfall 3 | Race condition causing missing sibling refs. Mitigation: asyncio is single-threaded; `list.append` is GIL-atomic. Risk is effectively zero. |
| A2 | `phase_fix_parallel` will eventually be called from production code (currently dead code) | Common Pitfalls / Pitfall 4 | Wiring fork() into a function nobody calls. Mitigation: Wire it anyway -- the function is part of the codebase and requirements specify it (SUBA-02). Tests exercise it directly. |
| A3 | trajectory.py at ~580-620 LOC after Phase 3 does not need a package split | Architecture Patterns | File becomes unwieldy. Mitigation: Phase 2 CONTEXT.md D-20 endorsed flat-then-split; 620 LOC is still manageable for a single-concern module. |

## Open Questions

1. **Double-hyphen in _safe_descriptor for edge-case inputs**
   - What we know: The regex produces `fix-issue--3` for `Fix_Issue (3)`. Current descriptors from D-05 never hit this.
   - What's unclear: Whether to add a collapse step (`re.sub(r'-{2,}', '-', ...)`) now or defer.
   - Recommendation: Add the collapse step -- it's one line and makes filenames cleaner. Low risk.

2. **trajectory.py docstring mentions "Phase 3 adds the second ContextVar" (line 15)**
   - What we know: D-04 eliminated the second ContextVar. The docstring is stale.
   - What's unclear: Nothing -- it should be updated.
   - Recommendation: Fix the docstring as part of the first plan that touches trajectory.py.

3. **Should create_dispatch_step accept a custom message parameter?**
   - What we know: D-11 says "Dispatching N parallel [fix|deep|exploration] tasks". The phase label from DaydreamPhase determines the word.
   - What's unclear: Whether a fixed template per phase is sufficient or callers need custom text.
   - Recommendation: Use a fixed template that derives the label from the DaydreamPhase value. If callers need custom text later, add an optional `message` kwarg.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 + pytest-asyncio 1.3.0 |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/test_trajectory.py -x -v` |
| Full suite command | `uv run pytest -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SUBA-01 | Sequential phases produce single root file with continuous step_ids | unit | `uv run pytest tests/test_trajectory.py::test_sequential_phases_single_file -x` | Wave 0 |
| SUBA-02 | phase_fix_parallel produces N sibling files with subagent_trajectory_ref | unit | `uv run pytest tests/test_trajectory.py::test_fix_parallel_siblings -x` | Wave 0 |
| SUBA-03 | deep mode per-stack fan-out produces N siblings | unit | `uv run pytest tests/test_trajectory.py::test_deep_per_stack_siblings -x` | Wave 0 |
| SUBA-04 | exploration pre_scan produces specialist siblings | unit | `uv run pytest tests/test_trajectory.py::test_exploration_siblings -x` | Wave 0 |
| SUBA-05 | Continuation appends to same trajectory, no sibling | unit | `uv run pytest tests/test_trajectory.py::test_continuation_appends_no_sibling -x` | Wave 0 |
| SUBA-06 | Sibling files inherit session_id, correct path | unit | `uv run pytest tests/test_trajectory.py::test_sibling_inherits_session_id -x` | Wave 0 |
| SUBA-07 | ContextVar copy-on-spawn establishes parent-child | unit | `uv run pytest tests/test_trajectory.py::test_fork_contextvar_isolation -x` | Wave 0 |
| SUBA-08 | step_id counters isolated per file | unit | `uv run pytest tests/test_trajectory.py::test_step_id_isolation_across_siblings -x` | Wave 0 |
| SUBA-09 | Parent FinalMetrics excludes child steps | unit | `uv run pytest tests/test_trajectory.py::test_parent_metrics_exclude_children -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_trajectory.py -x -v`
- **Per wave merge:** `uv run pytest -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] Fork-related tests in `tests/test_trajectory.py` -- covers SUBA-02, SUBA-03, SUBA-04, SUBA-06, SUBA-07, SUBA-08, SUBA-09
- [ ] Integration tests for phase_fix_parallel / phase_per_stack_reviews / pre_scan with mock backend -- covers SUBA-02, SUBA-03, SUBA-04
- [ ] Continuation verification test -- covers SUBA-01, SUBA-05

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | -- |
| V3 Session Management | no | -- |
| V4 Access Control | no | -- |
| V5 Input Validation | yes | `_safe_descriptor()` slugifies user-derived strings before using as filenames |
| V6 Cryptography | no | -- |

### Known Threat Patterns for this phase

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via malicious descriptor | Tampering | `_safe_descriptor()` strips all non-alphanumeric characters except hyphens; output is always `<target>/.daydream/trajectories/<hex>.<slug>.json` -- no directory separators possible |
| Sibling file content leaking secrets | Information Disclosure | Deferred to Phase 4 (REDA-01..06). Phase 3 creates the files; Phase 4 applies redaction before the cutover makes trajectories always-on. |

## Sources

### Primary (HIGH confidence)
- `daydream/trajectory.py` (498 LOC) -- Current Phase 2 recorder implementation, fully read [VERIFIED: codebase]
- `daydream/atif/models/observation_result.py` -- ObservationResult.subagent_trajectory_ref field definition [VERIFIED: codebase]
- `daydream/atif/models/subagent_trajectory_ref.py` -- SubagentTrajectoryRef(session_id, trajectory_path, extra) [VERIFIED: codebase]
- `daydream/atif/models/trajectory.py` -- Trajectory.validate_step_ids (sequential from 1) and validate_tool_call_references [VERIFIED: codebase]
- anyio 4.12.1 ContextVar copy-on-spawn behavior [VERIFIED: empirical test in current environment]
- `daydream/phases.py:962-1036` -- phase_fix_parallel function [VERIFIED: codebase, confirmed NOT called from any production path]
- `daydream/phases.py:1388-1489` -- phase_per_stack_reviews function [VERIFIED: codebase, called from deep/orchestrator.py]
- `daydream/exploration_runner.py:190-280` -- pre_scan with _run_specialist closure [VERIFIED: codebase]
- Dispatch step with subagent_trajectory_ref and no tool_calls passes vendored validator [VERIFIED: empirical test]

### Secondary (MEDIUM confidence)
- `.planning/phases/03-subagent-wiring-parallel-continuation/03-CONTEXT.md` -- All 13 decisions (D-01..D-13) [VERIFIED: read in full]
- `.planning/phases/02-recorder-core-event-enrichment-mapping/02-CONTEXT.md` -- Phase 2 decisions D-08, D-09, D-19, D-20 [VERIFIED: read in full]

### Tertiary (LOW confidence)
- None. All findings verified against codebase or empirical tests.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies; all infrastructure verified in codebase
- Architecture: HIGH -- ContextVar copy-on-spawn verified empirically; ATIF models verified; all integration sites read and understood
- Pitfalls: HIGH -- identified 7 pitfalls from codebase analysis; empirically verified the critical ones (ContextVar isolation, validator acceptance of dispatch steps, asyncio single-threaded safety)

**Research date:** 2026-04-27
**Valid until:** Stable -- no external dependencies or fast-moving APIs. Valid for the duration of this milestone.
