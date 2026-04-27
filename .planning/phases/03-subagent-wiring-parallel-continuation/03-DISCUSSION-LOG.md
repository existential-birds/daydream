# Phase 3: Subagent Wiring (Parallel + Continuation) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-27
**Phase:** 03-subagent-wiring-parallel-continuation
**Areas discussed:** Parallel fork surface, Sibling file naming, Continuation semantics, Parent step shape

---

## Parallel Fork Surface

### Q1: How should parallel tasks acquire their own child recorder?

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit wrapper | Phases call `async with recorder.fork(descriptor):`. 3 call sites change. Clear, auditable. | ✓ |
| Transparent auto-fork | Invocation.__aenter__ detects parallel context via ContextVar. Zero phase changes but implicit magic. | |
| ContextVar swap at task boundary | Helper `with_child_recorder(descriptor)` called inside each task closure. Middle ground. | |

**User's choice:** Explicit wrapper
**Notes:** None

### Q2: What does recorder.fork() return?

| Option | Description | Selected |
|--------|-------------|----------|
| Full TrajectoryRecorder | Own step_id counter, steps list, output path. Inherits session_id, run_flow, agent_model_name. parent=self backref. | ✓ |
| Lightweight SiblingRecorder | New class with minimal interface. Duplicates logic already in TrajectoryRecorder. | |

**User's choice:** Full TrajectoryRecorder
**Notes:** None

### Q3: Should child auto-patch parent trajectory on exit?

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-patch in __aexit__ | Child calls self.parent._register_sibling(). Zero phase-side bookkeeping. | ✓ |
| Explicit phase registration | Phase code calls parent_recorder.register_sibling() after task group. More boilerplate. | |

**User's choice:** Auto-patch in __aexit__
**Notes:** None

### Q4: Is the second ContextVar (_CURRENT_INVOCATION) still needed?

| Option | Description | Selected |
|--------|-------------|----------|
| Not needed | fork() + existing Invocation lifecycle covers all cases. One ContextVar is simpler. | ✓ |
| Still needed | Add _CURRENT_INVOCATION for tracking active invocation across async boundaries. | |
| You decide | Claude evaluates during planning. | |

**User's choice:** Not needed
**Notes:** Eliminates the two-ContextVar architecture deferred from Phase 2 D-08.

---

## Sibling File Naming

### Q1: What naming pattern for sibling trajectory descriptors?

| Option | Description | Selected |
|--------|-------------|----------|
| Semantic names | fix-0, deep-python, explore-pattern-scanner. Human-readable, self-describing. | ✓ |
| Flat index names | sub-0, sub-1. Simpler but no domain context. | |
| Phase-prefixed index | fix-0, deep-0, explore-0. Source identified but not task identity. | |

**User's choice:** Semantic names
**Notes:** None

### Q2: Should stack names be sanitized for file paths?

| Option | Description | Selected |
|--------|-------------|----------|
| Slugify to safe chars | re.sub(r'[^a-z0-9-]', '-', name.lower()).strip('-'). Future-proofs against special chars. | ✓ |
| Trust as-is | Stack names are controlled code, currently all lowercase alpha. | |

**User's choice:** Slugify to safe chars
**Notes:** None

### Q3: Full UUID4 or truncated session_id in filenames?

| Option | Description | Selected |
|--------|-------------|----------|
| First 8 chars | 4B possible values, collision-free in practice, scannable filenames. | ✓ |
| Full UUID4 | No ambiguity, correlates directly. But 50+ char filenames. | |
| You decide | Claude picks. | |

**User's choice:** First 8 hex chars
**Notes:** None

---

## Continuation Semantics

### Q1: How should continuation calls appear in the trajectory?

| Option | Description | Selected |
|--------|-------------|----------|
| New Invocation, same recorder | Each run_agent() creates new Invocation. Steps append as multi-turn conversation. No special-casing. | ✓ |
| Extend existing Invocation | Continuation detects prior Invocation and appends to it. Requires reopen logic. | |

**User's choice:** New Invocation, same recorder
**Notes:** Phase 2 D-09 holds unchanged.

### Q2: Should continuation steps carry a linking marker?

| Option | Description | Selected |
|--------|-------------|----------|
| No marker needed | Agent identity preserved via same metadata, session_id, trajectory file. ATIF has no native continuation concept. | ✓ |
| Add extra.continuation_of_step_id | Links back to previous result step_id. Custom ATIF extra field. | |

**User's choice:** No marker needed
**Notes:** None

### Q3: Same continuation behavior for both backends?

| Option | Description | Selected |
|--------|-------------|----------|
| Same behavior, both backends | Backend-agnostic trajectory shape. ContinuationToken internals invisible to recorder. | ✓ |
| You decide | Claude evaluates during implementation. | |

**User's choice:** Same behavior, both backends
**Notes:** None

---

## Parent Step Shape

### Q1: How should the parent trajectory represent a parallel fan-out?

| Option | Description | Selected |
|--------|-------------|----------|
| Synthetic dispatch step | Agent step with "Dispatching N parallel tasks" message + one ObservationResult.subagent_trajectory_ref per sibling. | ✓ |
| Attach to last real step | Last step before fan-out gets refs appended. Mixes real output with dispatch metadata. | |
| One dispatch step per child | One synthetic step per sibling. Granular but verbose. | |

**User's choice:** Synthetic dispatch step
**Notes:** None

### Q2: When is the synthetic dispatch step created?

| Option | Description | Selected |
|--------|-------------|----------|
| After task group completes | Only includes siblings that successfully wrote their file. No orphan refs. | ✓ |
| Before task group, update after | Placeholder then update. More complex. Position in timeline is accurate. | |

**User's choice:** After task group completes
**Notes:** None

### Q3: Relative or absolute path in subagent_trajectory_ref?

| Option | Description | Selected |
|--------|-------------|----------|
| Relative to root trajectory | trajectories/a1b2c3d4.fix-0.json. Portable, matches Harbor convention. | ✓ |
| Absolute path | Full filesystem path. Breaks on move. | |

**User's choice:** Relative to root trajectory
**Notes:** None

### Q4: Does phase code calling recorder.create_dispatch_step() violate the module-bloat ban?

| Option | Description | Selected |
|--------|-------------|----------|
| Acceptable — recorder method, not model construction | Same pattern as passing phase= to run_agent(). No ATIF imports needed beyond enum. | ✓ |
| Move dispatch to recorder.__aexit__ | Avoid trajectory calls in phases.py. Recorder infers phase labels. | |

**User's choice:** Acceptable
**Notes:** D-19 bans Step()/ToolCall()/Trajectory() construction, not recorder method calls.

---

## Claude's Discretion

- Internal structure of `_register_sibling()` (list vs dict, thread-safety)
- Whether `create_dispatch_step()` accepts additional kwargs or uses fixed template
- Whether `fork()` accepts optional overrides for agent_model_name
- Error handling when child recorder fails to write
- Whether `_safe_descriptor()` is module-level helper or staticmethod

## Deferred Ideas

- `_CURRENT_INVOCATION` ContextVar — eliminated by explicit fork approach, not implemented
- Mixed-backend deep mode fork override
- Sibling trajectory streaming (PERF-01)
- Cross-sibling FinalMetrics aggregation — consumer concern
