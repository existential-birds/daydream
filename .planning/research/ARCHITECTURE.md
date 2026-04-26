# Architecture Research — ATIF Trajectory Recording for Daydream

**Domain:** Brownfield Python CLI; replacing custom `_log_debug` text logging with ATIF v1.4 trajectory recording layered onto the existing `Backend` / `AgentEvent` / `run_agent()` pipeline.
**Researched:** 2026-04-26
**Confidence:** HIGH for daydream-internal placement decisions (verified against source). MEDIUM for ATIF subagent semantics (Harbor RFC URL returned 404; relying on the ATIF doc in this repo + Harbor framework docs page + secondary search results that all consistently describe `subagent_trajectory_ref` as the linkage mechanism).

## System Overview — Recorder Inserted Into the Existing Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│ cli.py:main()                                                    │
│   anyio.run(run, config)                                         │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ runner.py:run() / run_pr_feedback() / run_trust() / run_deep()   │
│   ── creates ONE TrajectoryRecorder per run                      │
│   ── activates it via async context manager (writes on exit)     │
└────────────────────────────┬─────────────────────────────────────┘
                             │ (recorder pushed onto a ContextVar)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ phases.py — phase_review / phase_parse_feedback / phase_fix /    │
│             phase_test_and_heal / phase_understand_intent / …    │
│   ── unchanged signatures (backend, cwd, …)                      │
│   ── each phase calls run_agent() N times                        │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ agent.py:run_agent()                                             │
│   ── reads recorder from ContextVar                              │
│   ── opens an Invocation scope (== one ATIF agent-step group)    │
│   ── feeds every AgentEvent into recorder.observe(event)         │
│   ── on exit, finalizes the invocation                           │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ backends/  Backend.execute() → AgentEvent stream                 │
│   ── Claude / Codex                                              │
│   ── enriched: every event carries an ISO 8601 `timestamp`       │
│   ── ResultEvent carries prompt/completion/cached token counts   │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ trajectory.py (NEW)                                              │
│   TrajectoryRecorder       — owns Trajectory + step counter      │
│   Invocation                — per-run_agent scope                │
│   _RECORDER_VAR (ContextVar) — implicit propagation              │
│   harbor.models.trajectories.* — typed construction              │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
                  .daydream/trajectory.json (root)
                  .daydream/trajectories/<id>.json (subagents)
```

## Component Responsibilities

| Component | Owns | Does NOT Own |
|-----------|------|--------------|
| `daydream/trajectory.py` (NEW) | `TrajectoryRecorder`, `Invocation`, `_RECORDER_VAR` ContextVar, mapping `AgentEvent` → ATIF `Step` / `ToolCall` / `ObservationResult` / `Metrics`, on-disk write at run end, Harbor `Trajectory` Pydantic construction | UI rendering; subprocess logging; phase orchestration |
| `daydream/agent.py:run_agent()` | Opening a recorder `Invocation`, pushing events into the recorder, raising `MissingSkillError` (unchanged) | Storing/persisting the trajectory; building Harbor models |
| `daydream/backends/__init__.py` | `AgentEvent` dataclasses with new `timestamp: str` (ISO 8601) field; `CostEvent` already has `input_tokens`/`output_tokens`; `ResultEvent` extended to carry `cached_tokens` | Trajectory construction; recorder calls |
| `daydream/backends/claude.py` | Populating `timestamp` on every yield; extracting `ResultMessage.usage.input_tokens`, `output_tokens`, `cache_read_input_tokens` into `ResultEvent` | Recorder; Harbor models |
| `daydream/backends/codex.py` | Same enrichment from codex JSONL stream | Same |
| `daydream/runner.py` | Constructing one `TrajectoryRecorder` per run, opening it as an async context manager, choosing the output path (`config.trajectory_path` or default), removing all `set_debug_log` / `--debug` plumbing | Recorder internals |
| `daydream/cli.py` | `--trajectory <path>` flag; removal of `--debug` flag | Recorder; trajectory path resolution policy |

## Architectural Decisions (Direct Answers to the 9 Questions)

### 1. Where does `TrajectoryRecorder` live? — **Per-run, owned by `runner.py`, propagated via `ContextVar`. Not on `AgentState`.**

**Recommendation:** Construct one `TrajectoryRecorder` per call to `run()` / `run_pr_feedback()` / `run_trust()` / `run_deep()`. Push it onto a module-level `ContextVar` in `daydream/trajectory.py`. `run_agent()` reads it via `get_current_recorder()`. The `RunConfig` carries the desired output path (`trajectory_path: Path | None`), not the recorder itself.

**Why not `AgentState`:**

- `AgentState`'s `debug_log` is a process-lifetime `TextIO` handle; that pattern is fine for a flat append-only log but wrong for a per-run object that must be opened, finalized, and written atomically.
- `AgentState` is module-level singleton state. Tests share the process. A run-scoped recorder on a singleton creates cross-test bleed unless every test calls `reset_state()` — which `tests/test_phases.py` and others don't reliably do.
- Multiple concurrent runs (rare but possible in tests using `anyio.create_task_group()` against the same module) would race on the singleton.
- The "no silent swallowing" / "always log or re-raise" convention in `CONVENTIONS.md` argues for explicit lifecycle: `async with recorder.open(path) as r:` is reviewable; `set_debug_log(...)` + `stack.callback(set_debug_log, None)` (current pattern in `runner.py:466-470`) is awkward and harder to reason about across nested flows.

**Why not pass through every `phase_*()` parameter:**

- 18 `run_agent()` call sites in `phases.py` (verified via grep). Threading a `recorder` parameter through every phase signature is a 30+ file diff that touches every mock backend in tests. `Backend` is already first-param; adding `recorder` second-param breaks the convention and every test fixture.
- A `ContextVar` has the propagation properties we want (parent-task → child-task automatic copy when a task group is spawned, per `contextvars` semantics in CPython 3.12) without changing call signatures.

**Why ContextVar over passing on `RunConfig`:**

- `RunConfig` is already on every phase via existing wiring, but `run_agent()` in `agent.py` does not receive `RunConfig` — and shouldn't, because that would couple the agent layer to runner-specific configuration. ContextVar is the smallest possible interface.
- ContextVar is also how the official `anyio` docs recommend propagating per-task ambient state (matches the project's existing `anyio.run(run, config)` and `anyio.create_task_group()` patterns).

**Concrete shape:**

```python
# daydream/trajectory.py
from contextvars import ContextVar

_RECORDER_VAR: ContextVar["TrajectoryRecorder | None"] = ContextVar(
    "daydream_trajectory_recorder", default=None
)

def get_current_recorder() -> "TrajectoryRecorder | None":
    return _RECORDER_VAR.get()

class TrajectoryRecorder:
    def __init__(self, path: Path, agent_name: str, agent_version: str, model: str): ...
    async def __aenter__(self) -> "TrajectoryRecorder":
        self._token = _RECORDER_VAR.set(self)
        return self
    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            self._write_to_disk()  # always write, even on exception
        finally:
            _RECORDER_VAR.reset(self._token)

    def begin_invocation(self, *, parent: "Invocation | None", prompt: str, cwd: Path) -> "Invocation": ...
    def end_invocation(self, inv: "Invocation") -> None: ...
```

### 2. How does `run_agent()` push events into the recorder? — **Inside the existing `async for event in event_iter:` loop, alongside the `_log_debug(...)` calls it replaces.**

**Recommended call site:** In `daydream/agent.py:run_agent()`, the existing event loop (currently lines 339-430) already does an `isinstance` dispatch on every event. Recording happens **at the top of each branch, before UI rendering**, so that even if the UI raises (e.g., terminal disconnect), the trajectory is still captured in memory.

**Why not in the backend:** The `Backend` protocol is intentionally thin — it normalizes external APIs into `AgentEvent`. Putting trajectory logic in each backend duplicates code (Claude + Codex + future backends), couples the SDK adapter to ATIF concepts, and makes mock backends in tests need to know about trajectories. The phases.py `run_agent` callers are the single chokepoint; record there.

**Why not in a wrapper around the iterator:** A wrapper (`async for event in record(event_iter, recorder):`) is tempting but loses access to the `MissingSkillError` raise, which must happen synchronously inside the consumer loop (current line 350) so that the renderer/registry can be `finish()`ed correctly.

**Concrete shape:**

```python
# daydream/agent.py — inside run_agent()
recorder = get_current_recorder()
parent = get_current_invocation()  # also a ContextVar, see Q3
inv: Invocation | None = None
if recorder is not None:
    inv = recorder.begin_invocation(parent=parent, prompt=prompt, cwd=cwd)

token: Token | None = None
if inv is not None:
    token = _CURRENT_INVOCATION.set(inv)

try:
    async for event in event_iter:
        if inv is not None:
            inv.observe(event)  # <-- the one new line per event branch (or one upstream)
        # ... existing UI rendering branches unchanged ...
finally:
    if inv is not None:
        recorder.end_invocation(inv)
    if token is not None:
        _CURRENT_INVOCATION.reset(token)
```

Place `inv.observe(event)` **before** the existing `_log_debug` removal. This is a single insertion that the hard cutover then prunes.

### 3. Subagent / parent-child wiring — **`ContextVar[Invocation]` for the current invocation; `Invocation.subagent_trajectory_ref` populated when a child closes; ATIF v1.4 spec calls for separate trajectory files referenced by `subagent_trajectory_ref` in the parent's `ObservationResult`.**

This is the highest-stakes call. Two findings:

**Finding A — ATIF v1.4 spec semantics (MEDIUM confidence — Harbor RFC URL returned 404 when fetched directly; verified against the ATIF reference doc shipped in `docs/reference/atif_format.md` plus harborframework.com plus three independent search summaries):**

ATIF v1.4 represents subagent delegation as **separate trajectory documents linked by reference**, NOT as nested objects in the parent trajectory. The mechanism:

> "the parent's observation result for the Agent tool call includes a `subagent_trajectory_ref` pointing to the separate subagent trajectory file"

There is **no `parent_step_id` field** on `Step`. The hierarchy is encoded in:

- The parent's `Step.tool_calls[*]` containing a tool call whose semantics are "delegate to subagent"
- The parent's `Step.observation.results[*].subagent_trajectory_ref` field pointing to the child trajectory file path/URI
- The child trajectory is its own valid `Trajectory` document with its own `session_id`, `agent`, `steps`, and `final_metrics`

This means daydream produces **N+1 files per run** when subagents are present: one root trajectory + one per subagent invocation.

**Finding B — Daydream's three subagent flavors map cleanly to this:**

1. **Per-stack reviews** (`phases.py:phase_per_stack_reviews`) — fan-out via `anyio.create_task_group()` with `CapacityLimiter(4)`. Each per-stack `run_agent()` call becomes a child trajectory referenced from the orchestrator's "spawn per-stack reviews" step. The default-arg-capture pattern at `phases.py:1455-1463` already isolates each task; ContextVar copy-on-spawn handles the parent linkage automatically.
2. **Exploration specialists** (`exploration_runner.py:pre_scan` — pattern_scanner, dependency_tracer, test_mapper) — same model: parent step is "run exploration", children are one trajectory per specialist.
3. **Sequential phases inside a run** (review → parse → fix → test) — these are NOT subagents; they're sibling top-level steps in the same root trajectory (see Q5).

**Concrete proposal (parent-child via ContextVar + observation linkage):**

```python
# daydream/trajectory.py
_CURRENT_INVOCATION: ContextVar["Invocation | None"] = ContextVar(
    "daydream_current_invocation", default=None
)

@dataclass
class Invocation:
    """One run_agent() scope; one ATIF subagent trajectory if it has a parent."""
    invocation_id: str  # uuid4
    parent: "Invocation | None"
    prompt: str
    steps: list[Step]
    started_at: str
    cwd: Path
    # When this invocation finalizes, the recorder writes a child trajectory to
    # .daydream/trajectories/<invocation_id>.json and patches the parent's
    # most-recent step's observation with a subagent_trajectory_ref.
```

When `run_agent()` enters and `get_current_invocation()` returns a non-None value, the new invocation is a **child**. On `end_invocation`:

- Root invocation (parent is None): steps are appended to the root `Trajectory.steps` directly.
- Child invocation: serialized to `.daydream/trajectories/<invocation_id>.json` as its own `Trajectory`; the parent's last agent-step's `Observation.results` gets a new `ObservationResult(source_call_id="<delegate_id>", content="<short summary>", subagent_trajectory_ref="trajectories/<invocation_id>.json")`.

For parallel children (`anyio.create_task_group()` spawning multiple `run_agent()` tasks), `ContextVar.set()` inside each task is isolated — this is anyio/asyncio standard behavior. Each task closes its own child trajectory; the parent's observation accumulates one `ObservationResult` per child.

**Why not `parent_step_id` field on Step:** ATIF v1.4 doesn't have one. Adding daydream-specific fields under `extra` would work but breaks interoperability with Harbor's validator and any downstream replay tooling. Stick with `subagent_trajectory_ref`.

**Why not nested inline in the parent's Step:** Same reason. Harbor's Pydantic models don't accept a `Step` inside a `Step`. The validator would reject.

### 4. Continuation flows — **Same trajectory, additional steps. No new subagent.**

`run_agent_with_continuation` (and the manual continuation pattern in `phase_test_and_heal` at `phases.py:861`) represents multi-turn within a single agent invocation. ATIF treats this as **additional steps appended to the same trajectory**:

- A `ContinuationToken` round-trip is logically: user re-prompts the same agent → agent responds → maybe more tool calls → final result. That's exactly what ATIF's `Step` sequence already models, alternating `source: "user"` and `source: "agent"`.
- A new subagent (= new trajectory file) is the wrong abstraction here because the agent identity (model, system prompt, conversation context) is preserved across the continuation. ATIF subagents are for *delegation to a different agent*.

**Recommendation:** When `run_agent()` is called with a `continuation` token, **do not start a new `Invocation`**. Append a new `user`-source `Step` to the existing invocation's step list (representing the re-prompt) and continue accumulating. The recorder needs an API like `recorder.continue_invocation(prev_inv, new_prompt) -> Invocation` that finds the prior invocation by token and resumes its step list.

**Practical concern:** today's `run_agent()` doesn't keep `Invocation` references between calls — the caller passes an opaque `ContinuationToken`. The recorder needs a `dict[str, Invocation]` mapping `ContinuationToken.data["session_id"]` → `Invocation`. This is a small addition; the alternative (treat continuations as new sibling invocations) loses the conversational continuity that consumers want for SFT/RL.

### 5. Phase boundaries — **(a) Sibling top-level steps in one flat trajectory. NOT subagents. NOT just `extra` labels.**

The four phases (`phase_review`, `phase_parse_feedback`, `phase_fix`, `phase_test_and_heal`) — and the deep-mode equivalents (`phase_understand_intent`, `phase_alternative_review`, `phase_per_stack_reviews`, `phase_cross_stack_merge`) — all run as the **same logical agent** (often even the same backend model, definitely the same `name="daydream"`, `version="<pkg version>"` in the ATIF `Agent` block).

Each `run_agent()` call within these phases is a fresh prompt → response → tools → result cycle. In ATIF terms that's a `user` step (the prompt) followed by one or more `agent` steps. The phases themselves are **organizational boundaries** within the daydream run, not separate agents.

**Recommendation:**

- One root `Trajectory` per daydream run.
- Each top-level `run_agent()` call (i.e., `Invocation` with `parent=None`) appends a `user` step + the agent's response steps to `Trajectory.steps`.
- The phase name is recorded under each step's `extra: {"daydream_phase": "review", ...}` so consumers can group/filter steps by phase without inventing new ATIF fields.
- Per-stack reviews and exploration specialists ARE subagents (Q3) because they conceptually delegate to a parallel worker. Phases are sequential and same-agent — they're not.

**Why not (b) parent steps with subagent children:** Would produce 4-8 trajectory files per run for what is conceptually one conversation. Inflates file count, breaks the "one trajectory per session" mental model that SFT/RL consumers expect, and obscures the phase ordering when reading the parent file.

**Why not (c) nothing — phases are just labels:** Labels in `extra` are good (and we recommend them), but the question is whether to *also* group the steps. The answer is: don't group via subagents (option b), do label via `extra` (this option). They're complementary, not exclusive.

### 6. Build order — **5-stage DAG, minimum coupling**

```
┌────────────────────────────────────────────────────────────────┐
│ Stage 0: Foundation (no daydream code changes)                 │
│   ├─ Add `harbor` to pyproject.toml; vet install footprint     │
│   └─ If footprint blocks: vendor harbor.models.trajectories.*  │
└────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 1: trajectory.py module (greenfield, no edits to others) │
│   ├─ TrajectoryRecorder + Invocation + ContextVars             │
│   ├─ AgentEvent → Step/ToolCall/Observation mapping            │
│   ├─ Subagent file-write + parent-observation patching         │
│   └─ Tests with hand-rolled AgentEvent fixtures                │
└────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 2: Backend enrichment (additive, doesn't break anything) │
│   ├─ Add timestamp: str field to AgentEvent dataclasses        │
│   ├─ ClaudeBackend: populate timestamp on every yield;         │
│   │   extract input_tokens/output_tokens/cached_tokens         │
│   │   from ResultMessage.usage into ResultEvent                │
│   ├─ CodexBackend: same enrichment from JSONL stream           │
│   └─ Existing tests must pass (additive fields, backward-OK)   │
└────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 3: run_agent() integration (the cutover point)           │
│   ├─ Wrap event loop in Invocation; call inv.observe(event)    │
│   ├─ Wire ContextVar push/pop                                  │
│   ├─ runner.py: construct recorder; async with recorder; pass  │
│   │   --trajectory path through RunConfig.trajectory_path      │
│   └─ DELETE _log_debug, set_debug_log/get_debug_log,           │
│       AgentState.debug_log, --debug flag, debug_log_path       │
│       initialization, all [PROMPT]/[TEXT]/[TOOL_USE]/[COST]/   │
│       [REVERT]/[PARSE_FAIL]/[STAGE]/[CODEX_*]/[PRE_SCAN] sites │
└────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 4: Subagent wiring (per-stack, exploration)              │
│   ├─ Per-stack reviews already use anyio task groups —         │
│   │   ContextVar copy-on-spawn handles parent linkage          │
│   ├─ Exploration specialists same                              │
│   ├─ Verify parent observation gets one subagent_trajectory_ref│
│   │   per child                                                │
│   └─ Validate produced trajectories with Harbor's validator    │
└────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 5: Continuation flows + edge cases                       │
│   ├─ phase_test_and_heal continuation → append-to-invocation   │
│   ├─ Run all 343 existing tests; fix regressions               │
│   ├─ Add ATIF-specific test suite (validation, replay)         │
│   └─ Documentation: README + docs/reference                    │
└────────────────────────────────────────────────────────────────┘
```

**Critical dependency edges:**

- Stage 1 must precede Stage 3 (`run_agent` needs the recorder API to call into).
- Stage 2 must precede Stage 3 in *the same commit/PR* if you want the trajectory's `Metrics` block to be non-empty (otherwise tokens are `None`). Splitting Stage 2 first as a no-op refactor (timestamps + token extraction with no consumers) is fine — produces ATIF-correct data once consumers exist.
- Stage 4 depends on Stage 3 because subagent wiring needs the ContextVar machinery to already be in place.
- Stage 5's continuation logic depends on Stage 1's `continue_invocation()` API existing.

**Note on Stage 0:** If `harbor` brings heavy transitive deps (the project's existing risk note in `PROJECT.md` flagged this), the fallback is vendoring `harbor.models.trajectories.*` (Pydantic models, ~500 lines) plus the JSON Schema for validator round-trip. This decision should be made before Stage 1 begins, not during.

### 7. Failure modes — **Catch and degrade at trajectory-recording boundaries; never let recording crash a run.**

Daydream's existing convention (`CONVENTIONS.md`): "No silent swallowing — always log or re-raise." But that's for domain logic. Trajectory recording is *observability* — its failure modes must not break the user's review/fix workflow.

**Recommendation:**

| Failure | Behavior | Where |
|---------|----------|-------|
| `harbor` import fails (Stage 0 vetting bypassed somehow) | Crash at `daydream/trajectory.py` import time. Fix: re-vet deps. | Module top |
| Pydantic validation fails when building `Step` | Catch in `Invocation.observe()`; emit `print_warning(console, ...)`; continue recording subsequent events; mark trajectory as `extra: {"validation_errors": [...]}` | `trajectory.py` |
| File write at run end fails (disk full, permission) | Catch in `TrajectoryRecorder.__aexit__`; emit `print_warning(...)`; preserve in-memory trajectory in `AgentState.last_trajectory` for inspection; do NOT fail the run | `trajectory.py` |
| Subagent child trajectory write fails | Catch; emit `print_warning(...)`; record `extra: {"subagent_write_error": "..."}` on the parent's observation result; do NOT fail the parent invocation | `trajectory.py` |
| Recorder is None when `run_agent` runs (e.g., direct test invocation) | No-op — guard every recorder call with `if recorder is not None:` | `agent.py` |

**Justification:** The existing pattern for `_log_debug` is exactly this — it's a no-op if `_state.debug_log is None`. The recorder is the conceptual replacement; same posture. Domain errors (`MissingSkillError`, `ValueError` from parse) keep their existing behavior (raise to runner, print user-friendly message).

**One exception** to "degrade silently": if the user passed `--trajectory <path>` explicitly, treat write failure as **error-level** (`print_error`, exit 1). If trajectory is opt-out-only (always written to default path), warn-and-continue. The CLI flag's presence signals user intent.

### 8. Test fixtures — **`tests/fixtures/trajectories/` for golden ATIF samples; complements existing fixture layout.**

Existing layout:
- `tests/fixtures/codex_jsonl/` — recorded codex CLI streams replayed in `test_backend_codex.py`
- `tests/fixtures/deep/` — deep-mode artifact samples
- `tests/fixtures/diffs/` — multi-language diff samples

**Recommendation:** Add `tests/fixtures/trajectories/` containing:

```
tests/fixtures/trajectories/
├── README.md                              # explains regen procedure
├── minimal_review_only.json               # one run_agent call, no tools
├── review_with_tools.json                 # one run_agent, multiple ToolCall/Observation
├── multi_phase_normal_run.json            # review→parse→fix→test, one trajectory
├── deep_run_root.json                     # multi-stack run root
├── deep_run_subagents/
│   ├── per_stack_python.json              # child trajectory referenced from root
│   ├── per_stack_typescript.json
│   └── exploration_dependency_tracer.json
└── continuation_test_heal.json            # multi-turn continuation in one trajectory
```

**Relationship to existing fixtures:**

- `codex_jsonl/` is *input* to backend tests (replay raw CLI output). `trajectories/` is *output* of recorder tests (golden ATIF documents that produced trajectories must match, modulo timestamps).
- `deep/` currently only has a README; `deep_run_subagents/` here is the deep-mode equivalent for trajectory testing — these complement, not replace.
- New `tests/test_trajectory.py` consumes these. Existing `test_backend_*.py` may add a small "round-trip a fake stream into a recorder, validate output" test that uses `codex_jsonl/` as input and `trajectories/` as expected output (closing the loop on backend-emits-events, recorder-builds-trajectory).

**Golden-file regeneration:** Provide a `make regen-trajectory-fixtures` target that runs daydream against a stable target dir (e.g., `tests/fixtures/sample_target/`) and writes the produced `.trajectory.json` to fixtures, with timestamps normalized to a fixed value for diff stability.

### 9. AgentState lifecycle — **Recorder lives on `RunConfig` (path) and a ContextVar (instance), not on `AgentState`. `AgentState.debug_log` is removed entirely.**

After cutover, `AgentState` shrinks:

```python
@dataclass
class AgentState:
    # debug_log: TextIO | None    ← REMOVED
    quiet_mode: bool = False
    model: str = "opus"
    shutdown_requested: bool = False
    current_backends: list[Backend] = field(default_factory=list)
```

**Why nothing trajectory-related on `AgentState`:**

- Singleton lifetime (process) ≠ recorder lifetime (one run). Putting it there re-creates the cross-test bleed risk.
- `reset_state()` would need to know how to dispose of an in-flight recorder, which is a layering inversion (state module → trajectory module dependency).
- The ContextVar lives in `daydream/trajectory.py`, owned by that module. Test isolation is handled by entering the recorder context manager in tests that need it; tests that don't will see `_RECORDER_VAR.get()` return `None` cleanly.

**One small caveat — ContextVar test isolation:** ContextVars don't auto-reset between pytest tests if a test forgets to close its context manager. Add a fixture to `conftest.py`:

```python
@pytest.fixture(autouse=True)
def _reset_trajectory_recorder():
    from daydream.trajectory import _RECORDER_VAR, _CURRENT_INVOCATION
    _RECORDER_VAR.set(None)
    _CURRENT_INVOCATION.set(None)
    yield
    _RECORDER_VAR.set(None)
    _CURRENT_INVOCATION.set(None)
```

This mirrors `reset_state()`'s role for `AgentState` and keeps the convention consistent.

## Anti-Patterns to Avoid During Migration

### Anti-Pattern 1: Putting recorder calls in each backend

**What people do:** Add `recorder.observe(...)` inside `ClaudeBackend.execute()` and `CodexBackend.execute()` directly.
**Why it's wrong:** Duplicates logic, couples backends to ATIF, breaks every mock backend in tests.
**Do this instead:** Backends only emit `AgentEvent`s (with timestamps + tokens). Recording happens in `run_agent()` — single chokepoint.

### Anti-Pattern 2: Using `extra: {"parent_step_id": ...}` for hierarchy

**What people do:** Invent a `parent_step_id` field under `extra` because the spec doesn't have one.
**Why it's wrong:** Breaks Harbor validator round-trip semantics; downstream consumers won't honor it; defeats the purpose of adopting a standard.
**Do this instead:** Use `subagent_trajectory_ref` on `ObservationResult` per ATIF v1.4 spec. Separate child files. One ATIF concept, one daydream implementation.

### Anti-Pattern 3: Threading recorder through every phase signature

**What people do:** `phase_review(backend, recorder, target_dir, ...)`, `phase_fix(backend, recorder, ...)`, etc.
**Why it's wrong:** 18+ call sites, every test fixture changes, breaks "Backend is first param" convention.
**Do this instead:** ContextVar in `trajectory.py`. Phases stay unchanged.

### Anti-Pattern 4: Streaming trajectory writes during the run

**What people do:** Append to `.trajectory.json` after every event so partial runs are observable.
**Why it's wrong:** Already in PROJECT.md "Out of Scope" — ATIF expects a single coherent document; partial files won't validate. Mid-run writes complicate atomic correctness.
**Do this instead:** Build in-memory; write atomically on `__aexit__`. If long-running observability becomes a need later, address it then.

### Anti-Pattern 5: Mixing UI rendering and recording in the same loop branch

**What people do:** Conditionally skip recording when `quiet_mode` is on, or call `print_cost` from inside the recorder.
**Why it's wrong:** Recording is independent of UI presentation. They have different lifecycles, different failure modes, different audiences.
**Do this instead:** Record every event always (subject to recorder presence). UI rendering stays governed by `get_quiet_mode()` separately. They share a loop but not state.

## Data Flow

### Single-run normal flow

```
cli.main()
  ↓ argparse → RunConfig(trajectory_path=Path(".daydream/trajectory.json"))
runner.run(config)
  ↓ async with TrajectoryRecorder(path=config.trajectory_path, ...) as r:
  ↓   _RECORDER_VAR.set(r)
  ↓   phase_review(review_backend, target_dir, skill, ...)
  ↓     run_agent(backend, cwd, prompt)
  ↓       inv = r.begin_invocation(parent=None, prompt=prompt, cwd=cwd)
  ↓       async for event in backend.execute(...):
  ↓         inv.observe(event)   ← maps to Step/ToolCall/Observation/Metrics
  ↓         <existing UI rendering>
  ↓       r.end_invocation(inv)  ← appends inv.steps to root Trajectory.steps
  ↓   phase_parse_feedback(...)  ← same shape
  ↓   phase_fix(...)              ← same shape
  ↓   phase_test_and_heal(...)    ← uses continuation; same Invocation extended
  ↓ __aexit__ writes Trajectory.to_json_dict() → trajectory_path
```

### Subagent flow (deep mode per-stack)

```
runner.run() → run_deep() → phase_per_stack_reviews()
  ↓ async with anyio.create_task_group() as tg:
  ↓   tg.start_soon(_task, "python")    # ContextVar copies into new task
  ↓   tg.start_soon(_task, "typescript")
  ↓
  ↓ Each task:
  ↓   run_agent(backend, cwd, per_stack_prompt)
  ↓     parent = get_current_invocation()  ← orchestrator's "fan-out" inv
  ↓     child_inv = r.begin_invocation(parent=parent, ...)
  ↓     <events into child_inv>
  ↓     r.end_invocation(child_inv)
  ↓       ↳ writes .daydream/trajectories/<child_id>.json
  ↓       ↳ patches parent_inv's last step's Observation with
  ↓          ObservationResult(source_call_id="<delegate_id>",
  ↓                            content="per-stack python complete",
  ↓                            subagent_trajectory_ref="trajectories/<child_id>.json")
```

## Scaling Considerations (per-run, not user count)

| Run shape | Trajectory file count | Approx size |
|-----------|------------------------|-------------|
| Normal (review→parse→fix×N→test) | 1 | 50KB-500KB depending on tool I/O |
| Loop mode (5 iterations) | 1 (steps accumulate) | ~5x normal |
| Deep mode (3 stacks + exploration) | 1 root + ~6 children | ~10x normal across files |
| TTT (intent+alternatives+plan) | 1 | ~normal |

**No backend pressure.** This is local file I/O, written once per run. Even a deep run stays comfortably under 10MB total.

## Integration Points

### External Services
| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Harbor | Pydantic models from `harbor.models.trajectories` | First-class library dep; vet install footprint per Stage 0 |
| Harbor validator | Optional, used in tests via `harbor.utils.trajectory_validator` | Round-trip test: build → write → validate |
| ATIF spec | v1.4, `schema_version="ATIF-v1.4"` literal | Pinned; no multi-version support per PROJECT.md |

### Internal Boundaries
| Boundary | Communication | Notes |
|----------|---------------|-------|
| `runner.py` ↔ `trajectory.py` | Construct + `async with`, pass path | One ownership point per run |
| `agent.py` ↔ `trajectory.py` | ContextVar read; `Invocation.observe()` calls | No direct import of internals |
| `backends/` ↔ `trajectory.py` | None (one-way: events flow up, recorder reads upstream) | Backends remain trajectory-unaware |
| `tests/conftest.py` ↔ `trajectory.py` | autouse ContextVar reset fixture | Mirrors `reset_state()` pattern for AgentState |

## Sources

- [ATIF v1.4 reference (in-repo)](file:///Users/ka/github/existential-birds/daydream/docs/reference/atif_format.md) — HIGH confidence; this is the canonical spec for the project
- [Harbor framework docs — ATIF page](https://www.harborframework.com/docs/agents/trajectory-format) — HIGH confidence on the high-level model; thin on subagent specifics
- [Harbor RFC 0001 (laude-institute mirror)](https://github.com/laude-institute/harbor/blob/main/docs/rfcs/0001-trajectory-format.md) — direct fetch returned 404 at research time; spec text accessed via secondary indexers
- [WebSearch synthesis on subagent_trajectory_ref](https://harborframework.com) — MEDIUM confidence on exact field shape; cross-confirmed by three independent search summaries describing the same `ObservationResult.subagent_trajectory_ref` mechanism
- daydream/agent.py:283-460 — current `run_agent` event loop (verified)
- daydream/runner.py:385-781 — current `run()` lifecycle including `set_debug_log` plumbing (verified)
- daydream/phases.py:1380-1480 — per-stack parallel pattern (verified)
- daydream/exploration_runner.py — pre-scan parallel specialists (verified)
- .planning/codebase/CONVENTIONS.md — singleton/state conventions (verified)
- .planning/codebase/CONCERNS.md — module sizes, BLE001 sites, singleton risk (verified)

---
*Architecture research: 2026-04-26*
