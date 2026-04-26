# Feature Research

**Domain:** ATIF v1.4 trajectory emission for daydream (brownfield CLI agent)
**Researched:** 2026-04-26
**Confidence:** HIGH (Pydantic models verified directly from Harbor source; one ecosystem inconsistency flagged)

## Executive Summary

Daydream's milestone is to replace prefix-tagged debug logs with valid ATIF v1.4 trajectories produced via Harbor's Pydantic models. The schema is small, strict (`extra: "forbid"` on every model), and surprisingly opinionated about a handful of details that drive architecture:

1. **Subagent delegation is by reference, not nesting.** ATIF represents subagents as **separate trajectory files** linked from the parent's `ObservationResult.subagent_trajectory_ref` (a `list[SubagentTrajectoryRef]`). There is no `parent_step_id`, no nested `Trajectory`, and no `subagent_steps` list. This forces daydream to either (a) emit one root trajectory plus N sibling files, or (b) emit a single flat trajectory and accept that subagent fan-out is collapsed. **This decision drives the milestone's data model.**
2. **`tool_call_id` is per-step scoped, but Harbor enforces stronger.** The validator confirms each `ObservationResult.source_call_id` matches a `tool_call_id` *in the same step's `tool_calls` array*. So IDs only need to be unique within one step — but reusing IDs across steps is allowed (and harmless for the validator). For replay clarity, daydream should make them globally unique within a trajectory anyway (Claude SDK's `block.id` already is).
3. **Per-step metrics are per-LLM-call values, not deltas.** Harbor's golden Terminus-2 trajectory shows per-step `prompt_tokens` growing across turns (682, 750, 820...) because conversation context grows; `total_prompt_tokens` is the *sum* of those per-call values, not the last value. This means daydream's mapping ("CostEvent → Metrics") is correct as long as it captures what the LLM was billed for *that turn* — which is exactly what `claude-agent-sdk`'s `ResultMessage.usage` provides.
4. **`cached_tokens` is a SUBSET of `prompt_tokens`, not a separate bucket.** Harbor's docstring is explicit: `cached_tokens` = "Subset of prompt_tokens that were cache hits." Mapping `claude-agent-sdk`'s `cache_read_input_tokens` → `cached_tokens` and `input_tokens + cache_read + cache_creation` → `prompt_tokens` matches Claude Code's own Harbor adapter.
5. **The Beagle skill prompt is `source="user"`, NOT `"system"`.** ATIF's `system` source is reserved for system-prompt-style messages. The prompt daydream sends to `run_agent()` (e.g., `/beagle-python:review-python`) is the user's instruction to the agent — `source="user"`. Confirmed by Claude Code's adapter (`role="user"` → `source="user"`) and OpenHands' adapter.

## Required vs Optional Fields (ATIF v1.4 — Harbor Pydantic models)

Sourced directly from `harbor.models.trajectories.*`. **Every model has `model_config = {"extra": "forbid"}`** — unknown fields fail validation. Fields with `default=...` are required; fields with `default=None` are optional. Below, **R** = required, **O** = optional.

### `Trajectory` (root)

| Field | Type | Req | Notes |
|-------|------|-----|-------|
| `schema_version` | `Literal["ATIF-v1.0"..."ATIF-v1.6"]` | R | Pin to `"ATIF-v1.4"` per milestone scope. |
| `session_id` | `str` | R | Unique identifier for the entire run. UUID4 is conventional. |
| `agent` | `Agent` | R | See `Agent` model. |
| `steps` | `list[Step]` | R | `min_length=1`. Step IDs validated sequential from 1. |
| `notes` | `str \| None` | O | Free-form documentation; required if `final_metrics.total_steps != len(steps)`. |
| `final_metrics` | `FinalMetrics \| None` | O | Recommended; OpenHands populates `AgentContext` from these. |
| `continued_trajectory_ref` | `str \| None` | O | Path to next trajectory if this one is continued. **Do not set** in v1.4 daydream. |
| `extra` | `dict[str, Any] \| None` | O | Custom root-level metadata. **Differentiator opportunity.** |

**Validators:** `validate_step_ids()` (sequential from 1), `validate_tool_call_references()` (each `source_call_id` matches a `tool_call_id` in the same step).

### `Agent`

| Field | Type | Req | Notes |
|-------|------|-----|-------|
| `name` | `str` | R | E.g., `"daydream"`. |
| `version` | `str` | R | Daydream package version. |
| `model_name` | `str \| None` | O | Default model for the run (e.g., `"claude-opus-4"`). |
| `tool_definitions` | `list[dict] \| None` | O | OpenAI function-calling schema format. Optional but valuable for replay. |
| `extra` | `dict \| None` | O | Custom config — good place for backend name. |

### `Step`

| Field | Type | Req | Notes |
|-------|------|-----|-------|
| `step_id` | `int` | R | `ge=1`; sequential from 1 — enforced by trajectory validator. |
| `timestamp` | `str \| None` | O | ISO 8601; validated. **Recommended for daydream** (PROJECT.md already requires it on events). |
| `source` | `Literal["system", "user", "agent"]` | R | See semantics below. |
| `model_name` | `str \| None` | O | Per-turn model override; **agent-only**. |
| `reasoning_effort` | `str \| float \| None` | O | **agent-only**. Not relevant for daydream. |
| `message` | `str \| list[ContentPart]` | R | **In v1.4: string only** (multimodal `ContentPart` is v1.6). Empty string `""` is valid. |
| `reasoning_content` | `str \| None` | O | **agent-only**. Maps from `ThinkingEvent`. |
| `tool_calls` | `list[ToolCall] \| None` | O | **agent-only**. |
| `observation` | `Observation \| None` | O | Allowed on agent and system steps (v1.2+). |
| `metrics` | `Metrics \| None` | O | **agent-only**. |
| `extra` | `dict \| None` | O | Custom step-level metadata. |

**Validator:** `validate_agent_only_fields()` raises if `model_name`, `reasoning_effort`, `reasoning_content`, `tool_calls`, or `metrics` is set on a non-agent step. This means **the user step that opens each `run_agent()` invocation cannot carry the prompt's token cost** — costs only attach to agent steps.

### `ToolCall`

| Field | Type | Req | Notes |
|-------|------|-----|-------|
| `tool_call_id` | `str` | R | Maps from `ToolStartEvent.id` / `ToolResultEvent.id`. |
| `function_name` | `str` | R | Maps from `ToolStartEvent.name`. |
| `arguments` | `dict[str, Any]` | R | Empty dict `{}` allowed. Maps from `ToolStartEvent.input`. |

### `Observation`

| Field | Type | Req | Notes |
|-------|------|-----|-------|
| `results` | `list[ObservationResult]` | R | Array, even for single tool result. |

### `ObservationResult`

| Field | Type | Req | Notes |
|-------|------|-----|-------|
| `source_call_id` | `str \| None` | O | References a `tool_call_id` in the **same step**. Null for non-tool feedback. |
| `content` | `str \| list[ContentPart] \| None` | O | **String only in v1.4.** From `ToolResultEvent.output`. |
| `subagent_trajectory_ref` | `list[SubagentTrajectoryRef] \| None` | O | **The only mechanism for subagent delegation.** |

### `SubagentTrajectoryRef`

| Field | Type | Req | Notes |
|-------|------|-----|-------|
| `session_id` | `str` | R | The subagent trajectory's `session_id`. |
| `trajectory_path` | `str \| None` | O | Path to the sibling subagent trajectory file. |
| `extra` | `dict \| None` | O | Custom subagent metadata. |

### `Metrics` (per-step, agent-only)

All fields **optional** (default=None). Daydream should populate the first four; the token-IDs/logprobs are RL-training-only and unavailable from claude-agent-sdk.

| Field | Type | Notes |
|-------|------|-------|
| `prompt_tokens` | `int \| None` | "Total input tokens including cached and non-cached." |
| `completion_tokens` | `int \| None` | "Total tokens generated by the LLM response." |
| `cached_tokens` | `int \| None` | **"Subset of prompt_tokens that were cache hits."** |
| `cost_usd` | `float \| None` | Per-call cost. |
| `prompt_token_ids` | `list[int] \| None` | RL training. Skip — not exposed by claude-agent-sdk. |
| `completion_token_ids` | `list[int] \| None` | RL training. Skip. |
| `logprobs` | `list[float] \| None` | Skip — not exposed. |
| `extra` | `dict \| None` | Custom metrics — e.g., per-call duration_ms. |

### `FinalMetrics` (trajectory-level, all optional)

| Field | Type | Notes |
|-------|------|-------|
| `total_prompt_tokens` | `int \| None` | "Sum of all prompt tokens across all steps, including cached tokens." |
| `total_completion_tokens` | `int \| None` | Sum across all steps. |
| `total_cached_tokens` | `int \| None` | Sum across all steps. |
| `total_cost_usd` | `float \| None` | "Total real monetary cost ... including cost for subagents, if any." |
| `total_steps` | `int \| None` | `ge=0`. If `!= len(steps)` (e.g., subagent steps not inlined), document why in `Trajectory.notes`. |
| `extra` | `dict \| None` | Custom aggregate metrics. |

## Step-Source Semantics

| Source | When | Daydream Mapping |
|--------|------|-------------------|
| `"user"` | User-initiated input or instruction to the agent | **The prompt sent to `run_agent()`** (Beagle skill invocation, intent prompt, etc.). Becomes step 1 of each `run_agent()` segment. |
| `"agent"` | LLM response — text, reasoning, tool calls, observations of those tool calls | All `TextEvent` / `ThinkingEvent` / `ToolStartEvent` / `ToolResultEvent` / `CostEvent` from the backend. |
| `"system"` | System-prompt style preamble or system-initiated operations | **Optional in daydream.** Could host the always-on guidance daydream injects (e.g., the global `ClaudeAgentOptions.allowed_tools` constraint, schema-output framing). Skipping it is valid. |

**Key validator constraint:** `model_name`, `reasoning_effort`, `reasoning_content`, `tool_calls`, `metrics` are **agent-only** — setting them on user/system steps is a hard validation failure. So token usage for "the agent's response to a user prompt" attaches to the **agent step that follows**, not the user step that triggered it.

**Daydream-specific:** The Beagle skill invocation (e.g., the literal string `/beagle-python:review-python`) is the *user prompt* even though it triggers system-side skill loading — the agent receives it as their instruction. This is `source="user"`, matching how Claude Code's Harbor adapter handles it (`role="user"` → `source="user"`).

## Subagent Delegation (the architecturally critical decision)

ATIF v1.4 represents subagent delegation through **`ObservationResult.subagent_trajectory_ref`** — a list of `SubagentTrajectoryRef` objects, each pointing to a separate trajectory file by `session_id` and optional `trajectory_path`.

### What this means for daydream

There is **no** `parent_step_id`, **no** nested `Trajectory` field, **no** `subagent_steps` array. The shape is:

```
parent.trajectory.json          ← root daydream run
└── steps[N].observation.results[*].subagent_trajectory_ref[*]
    └── { session_id: "<uuid>", trajectory_path: "subagent_<name>_<uuid>.json" }

subagent_<name>_<uuid>.json     ← sibling file, fully self-contained Trajectory
```

This is confirmed by:
- The Pydantic model in `src/harbor/models/trajectories/subagent_trajectory_ref.py`.
- The agent-lens convention: "subagent_\<name\>_\<id\>.json" sibling files, parent's `subagent_trajectory_ref` points to them, "Subagent messages are filtered from the parent trajectory to keep it clean".
- Harbor's `FinalMetrics.total_cost_usd` doc: "Total real monetary cost for the entire trajectory, including cost for subagents, if any" — i.e., **the parent rolls up subagent cost**, but step-by-step traffic lives in the child files.

### Daydream call-sites that emit subagent trajectories

| Call site | Today | Treatment |
|-----------|-------|-----------|
| `phase_review` → `run_agent()` | Direct call from main run | **Same trajectory** as parent. Inline as steps. |
| `phase_parse_feedback` → `run_agent()` | Direct call (structured output) | Inline as steps. |
| `phase_fix` (sequential) → `run_agent()` × N | Direct calls in a loop | Inline; one user/agent step segment per fix. |
| `phase_test_and_heal` → `run_agent()` | Direct call | Inline. |
| **`exploration_runner.pre_scan` → `run_agent()` × {1,3} parallel** | Subagents via `anyio.create_task_group` | **Sibling trajectory file per specialist** (`subagent_dependency_tracer_<uuid>.json` etc.); parent step's observation gets a `SubagentTrajectoryRef` per specialist. |
| **`deep/orchestrator` → per-stack `run_agent()` × N parallel** | Capacity-limited fan-out | **Sibling trajectory file per stack**; parent's stack-fan-out step's observation gets one `SubagentTrajectoryRef` per stack. |
| **`phase_fix_parallel` → `run_agent()` × N parallel** | Concurrent fix subagents | **Sibling trajectory file per parallel fix**. |

### Why this works

- **Replay tools** can walk the `subagent_trajectory_ref` graph to reconstruct the full execution.
- **SFT/RL pipelines** can train on subagent trajectories independently (each is a complete `Trajectory`).
- **Step-id sequentiality stays intact** in every file (1, 2, 3, ... in each).
- **Cost rollup is well-defined**: parent's `total_cost_usd` includes subagent cost; per-step metrics in the parent only cover what the parent's LLM was billed.

### Why the alternative (single flat trajectory) is worse

- Parallel subagents would collide on `step_id` ordering. Inlining 3 specialist agents' steps as steps 4, 5, 6, 7, 8, 9 hides parallelism.
- Token rollup becomes ambiguous — was step 7 a parent or subagent step?
- Loses interop with Harbor consumers that expect `SubagentTrajectoryRef`.

**Recommendation: emit sibling files for parallel `anyio` task-group invocations; inline sequential `run_agent()` calls as continuous steps in the parent.**

## tool_call_id Correlation Contract

**Schema constraint (Harbor validator):** Each `ObservationResult.source_call_id` must reference a `tool_call_id` that exists **in the same step's `tool_calls` array**.

```python
# From harbor.models.trajectories.trajectory.validate_tool_call_references:
for step in self.steps:
    if step.observation is None: continue
    tool_call_ids = {tc.tool_call_id for tc in (step.tool_calls or [])}
    for result in step.observation.results:
        if result.source_call_id is not None:
            if result.source_call_id not in tool_call_ids:
                raise ValueError(...)  # validation failure
```

**Implications:**

1. **A step with tool calls must include the matching observation in the same step.** You cannot put `tool_calls` on step 5 and the matching `observation` on step 6.
2. **`tool_call_id` is therefore intra-step scoped by the validator.** But for clarity, daydream should use globally-unique IDs (claude-agent-sdk's `block.id` is already a UUID-style string, e.g., `toolu_01ABC...`).
3. **Out-of-order observations within a step are fine** — the constraint is set membership, not array index alignment.
4. **Daydream's claude.py already emits `ToolStartEvent.id == ToolUseBlock.id` and `ToolResultEvent.id == ToolResultBlock.tool_use_id`.** These match by construction → use them directly.

**Pattern for the recorder:** group consecutive `ToolStartEvent` + `ToolResultEvent` pairs into a single ATIF agent step with both `tool_calls` and `observation.results` populated. When the agent emits text *between* tool calls, that's a step boundary — start a new agent step.

## Per-Step Metrics: Deltas vs Running Totals (RESOLVED)

**Verdict: per-step metrics are *per-LLM-call* values, not deltas, not cumulative running totals.**

Evidence:
- Harbor's `Metrics.prompt_tokens` doc: "Total input tokens including cached and non-cached" — i.e., **what was sent in this call**, including the conversation history that the LLM saw.
- Harbor's `FinalMetrics.total_prompt_tokens` doc: "**Sum** of all prompt tokens across all steps."
- Golden file `terminus_2/hello-world-context-summarization.trajectory.json`: per-step `prompt_tokens` = 682, 750, 820, ... (each is what was billed that turn — and grows because the conversation context grew). Final total = sum of those per-step values (~7802).
- OpenHands adapter explicitly **subtracts cumulative source values** to recover this per-call accounting: `delta_prompt = curr - prev`. This is a *correction* for OpenHands' source format, not a different ATIF semantic.
- Claude Code adapter uses values directly because `claude-agent-sdk`'s `ResultMessage.usage` is already per-call.

**Daydream mapping:**
- `claude-agent-sdk` `ResultMessage` arrives once per turn with `usage.input_tokens`, `usage.output_tokens`, `usage.cache_read_input_tokens`, `usage.cache_creation_input_tokens`, and `total_cost_usd`.
- Map directly to per-step `Metrics`:
  - `prompt_tokens = input_tokens + cache_read_input_tokens + cache_creation_input_tokens`
  - `completion_tokens = output_tokens`
  - `cached_tokens = cache_read_input_tokens` (the read-from-cache subset)
  - `cost_usd = total_cost_usd` of *this* `ResultMessage` (per-turn, not cumulative)

**Caveat (data-collection bug today):** `daydream/backends/claude.py:120-128` always sets `input_tokens=None, output_tokens=None` on the emitted `CostEvent`. This is a bug PROJECT.md already calls out. The recorder needs the data — fixing the backend extraction is a prerequisite milestone task.

**Caveat (claude-agent-sdk `ResultMessage` semantics):** GitHub issue [anthropics/claude-code-sdk-python#112](https://github.com/anthropics/claude-code-sdk-python/issues/112) reports `ResultMessage.usage["input_tokens"]` is sometimes "not correct" — historically it has been the *delta from the previous turn* rather than the per-call total. **MEDIUM confidence** that this is fully resolved in current `claude-agent-sdk`. Daydream should validate against a known multi-turn run before assuming straight pass-through.

## `cached_tokens` Semantics

`Metrics.cached_tokens` = "Subset of prompt_tokens that were cache hits."

- It is **part of `prompt_tokens`**, not a separate addend. Sum invariant: `prompt_tokens >= cached_tokens`.
- For Claude SDK: `cached_tokens = usage["cache_read_input_tokens"]`.
- `cache_creation_input_tokens` is **not** `cached_tokens` — that's tokens written *to* cache (charged at premium rate). Sum it into `prompt_tokens` but not `cached_tokens`.
- This matches Claude Code's Harbor adapter exactly.

## `extra` Field Conventions

ATIF reserves `extra: dict[str, Any] | None` at every level for custom metadata. **No central registry of conventions** — each agent integration uses its own keys. From the spec example, Agent shows `extra={"agent_class": "CodeActAgent"}`.

**Daydream conventions to adopt (Differentiators):**

| Level | Suggested keys | Why |
|-------|----------------|-----|
| `Trajectory.extra` | `flow: "review" \| "pr" \| "ttt" \| "deep"`, `target_dir: str`, `cli_args: dict`, `git_head_sha: str` | Replay context; lets viewers segment by run flow. |
| `Agent.extra` | `backend: "claude" \| "codex"`, `daydream_phase_overrides: dict`, `beagle_plugin_version: str` | Distinguishes Claude vs Codex backend in trajectory analysis. |
| `Step.extra` | `phase: "review" \| "parse" \| "fix" \| "test" \| "intent" \| "alternative" \| "plan"`, `phase_iteration: int`, `is_test_retry: bool` | Critical for replay — lets training pipelines weight phases differently. |
| `ObservationResult.extra` | `truncated: bool`, `original_size_bytes: int` | If daydream ever truncates large tool outputs. |
| `Metrics.extra` | `duration_ms: int`, `cache_creation_tokens: int` (not `cached_tokens`) | Preserves cache_creation_input_tokens distinct from cache reads. |
| `SubagentTrajectoryRef.extra` | `kind: "exploration_specialist" \| "deep_per_stack" \| "parallel_fix"`, `name: "dependency_tracer"`, `parent_step_id: int` | **Recovers parent linkage** since v1.4 has no top-level field for it. |
| `FinalMetrics.extra` | `subagent_count: int`, `subagent_total_cost_usd: float`, `phase_breakdown: {phase: cost}` | Lets dashboards show cost-by-phase without parsing every step. |

## Feature Landscape

### Table Stakes (Required for valid v1.4 trajectory or expected by all consumers)

Features users (Harbor validator, replay tools, training pipelines) assume exist. Missing these = invalid trajectory or fails downstream consumption.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Pin `schema_version="ATIF-v1.4"` | Required field; out-of-scope versions break Harbor validators that pin to v1.4 | LOW | Hardcode constant. |
| Generate `session_id` per run (UUID4) | Required field; identifies the trajectory for cross-tool correlation | LOW | `uuid.uuid4().hex` at run start in `RunConfig`. |
| Populate `Agent(name, version)` | Required fields | LOW | `name="daydream"`, `version=importlib.metadata.version("daydream")`. |
| `Agent.model_name` per backend | Optional but expected by every consumer | LOW | E.g., `"claude-opus-4"`, read from `RunConfig.model`. |
| Sequential `step_id` from 1 | **Hard validator constraint** — `validate_step_ids()` raises | LOW | Increment counter; reset per trajectory file (so sibling subagent files restart at 1). |
| Each `run_agent()` call opens with one user step carrying the prompt | Required by ATIF semantics — agent steps must be reactions to a user input | LOW | Add `Step(source="user", message=prompt)` at start of each `run_agent()` segment. |
| ISO 8601 `timestamp` on every step | Optional but Harbor validator validates format if present; PROJECT.md commits to it | LOW | `datetime.now(UTC).isoformat()` per event. |
| Map `TextEvent` → agent `Step.message` | Core event type; consecutive chunks accumulate into one step until a tool call or thinking break | MEDIUM | Coalescer state machine; flush step on tool call / thinking / next user prompt. |
| Map `ThinkingEvent` → agent `Step.reasoning_content` | Core event type | LOW | Concatenate consecutive thinking chunks. |
| Map `ToolStartEvent` → `Step.tool_calls[*]` | Core event type | LOW | Direct field copy: `id`/`name`/`input` → `tool_call_id`/`function_name`/`arguments`. |
| Map `ToolResultEvent` → `Step.observation.results[*]` | Core event type; **must be in same step as the matching tool call** | MEDIUM | Buffer until result arrives or until next agent text begins. |
| Map `CostEvent` → `Step.metrics` per step | Core event type; required for cost rollup | LOW (after backend fix) | Attach to the agent step it terminates. |
| Trajectory-final `FinalMetrics` summing all steps' metrics | Optional but expected; OpenHands populates `AgentContext` from these | LOW | Pass over `steps` and sum. |
| Validate trajectory against Harbor's Pydantic models in CI | Catches regressions; prevents producing invalid trajectories | LOW | Construct via `Trajectory(...)` → Pydantic validates eagerly. |
| Output path via `--trajectory <path>` CLI flag | PROJECT.md commits to it | LOW | Already in scope. |
| Subagent trajectories as sibling files with `subagent_trajectory_ref` linkage | Required for any parallel `anyio.create_task_group` flow to roundtrip through Harbor | HIGH | See Subagent Delegation section. |
| Forbid unknown fields (Pydantic `extra: forbid`) on construction | Pydantic enforces; means daydream cannot stash arbitrary data outside `extra` | LOW | Document for contributors. |
| Tool result content as `str` (v1.4 — not multimodal) | v1.4 spec | LOW | `ToolResultEvent.output` is already a string. |
| Backend token extraction fix (`prompt_tokens`, `completion_tokens`, `cached_tokens` populated) | Currently always None — without this, every `Metrics` is empty and `FinalMetrics` is meaningless | LOW | Fix `daydream/backends/claude.py:120-128` to read from `msg.usage`. |

### Differentiators (Boost daydream's trajectories beyond minimum viable)

Features not required for validity but make daydream's trajectories materially more useful for replay, debugging, and training.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| `Trajectory.extra.flow` (`"review" \| "pr" \| "ttt" \| "deep"`) | Lets dashboards segment runs by flow; lets training pipelines weight by flow type | LOW | Single field write at run start. |
| `Step.extra.phase` (`"review" \| "parse" \| "fix" \| ...`) | Replay tools can show phase boundaries without inferring from prompt content; SFT can train per-phase | LOW | Threaded through `run_agent()` via a new `phase` argument. |
| `SubagentTrajectoryRef.extra.parent_step_id` | Recovers parent linkage that v1.4 lacks at the top level — viewers can render hierarchies | LOW | Single integer in `extra`. |
| `SubagentTrajectoryRef.extra.kind` (`"exploration_specialist"`, `"deep_per_stack"`, `"parallel_fix"`) | Subagent role classification — exploration is research, fix is mutation — different replay semantics | LOW | Set at task-group dispatch site. |
| `Agent.tool_definitions` | Lets replay tools render tool signatures; required input for some training pipelines | MEDIUM | Reflect `ClaudeAgentOptions.allowed_tools` + Beagle skill manifest. |
| `Agent.extra.backend` (`"claude" \| "codex"`) | Cross-backend analysis — compare token efficiency between Claude and Codex on the same task | LOW | Single string. |
| `Metrics.extra.duration_ms` per step | Latency analysis; complements token cost with wall-time | LOW | Stopwatch around each LLM turn. |
| `FinalMetrics.extra.phase_breakdown` (cost per phase) | Lets ops see "review consumed 60% of cost, fix 30%, test 10%" without scanning steps | MEDIUM | Aggregate from `Step.extra.phase`. |
| `Trajectory.extra.git_head_sha` | Reproducibility — pin which commit was reviewed | LOW | Already collected in `phases.py:_get_head_sha`. |
| `Trajectory.extra.cli_args` | Full reproducibility — replay exact invocation | LOW | Serialize `RunConfig`. |
| `Trajectory.notes` populated when `total_steps != len(steps)` | Required by `FinalMetrics.total_steps` doc when subagent steps are excluded | LOW | Auto-add note when subagents emitted. |
| `FinalMetrics.total_cost_usd` includes subagent rollup | Spec says it should; Harbor consumers expect it | LOW | Sum parent + each subagent's `total_cost_usd`. |
| Validate every produced trajectory in tests via `TrajectoryValidator` | Schema-drift protection beyond Pydantic-only validation (catches step-id sequentiality, tool-call ref correctness) | LOW | Add fixture; one test per flow. |
| Append-only writer that flushes valid trajectory on `KeyboardInterrupt` | Today the debug log writes incrementally and survives Ctrl-C; users will expect the same | MEDIUM | Catch in `cli.main()`'s shutdown handler; flush partial trajectory. |

### Anti-Features (Don't build)

Features that look attractive but create problems for daydream's ATIF integration.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Streaming/incremental writes mid-run | "Mirror what `_log_debug` does today; survive crashes" | Already in PROJECT.md Out of Scope; ATIF is a coherent document; partial writes risk invalid JSON; complicates Pydantic construction | Write at run completion; on shutdown flush a best-effort `Trajectory` if at least one step exists |
| Inlining subagent steps in the parent trajectory | "Single file is easier to read" | Breaks step-id sequentiality (parallel subagents can't interleave cleanly); parent step metrics conflate parent + subagent costs; loses the `SubagentTrajectoryRef` interop pattern Harbor consumers expect | Sibling files via `subagent_trajectory_ref` |
| Embedding raw `subprocess.run` outputs (`gh`, `git`, tree-sitter) as steps | "Capture everything for replay" | PROJECT.md Out of Scope; ATIF models LLM-agent interactions, not arbitrary subprocess noise; bloats trajectories; risks env-leak via tool output | Skip non-`run_agent()` calls. If captured at all, store as `Trajectory.extra.subprocess_log` (truncated). |
| Storing `prompt_token_ids`, `completion_token_ids`, `logprobs` | "ATIF supports them; collect for future RL" | claude-agent-sdk does not expose these; fabricating them defeats their RL purpose; adds storage with no signal | Leave `None`. Revisit only if SDK exposes them. |
| Recording every `[CODEX_RAW]` line as an event | "Don't lose Codex debug detail" | These are raw stream noise, not semantic agent events; bloats trajectory; not consumable by Harbor | Codex backend should *parse* CLI output into the existing `AgentEvent` types; raw lines belong in `extra` only if a parse failure occurs |
| Custom `schema_version="daydream-v1"` | "We have unique fields" | Breaks compatibility with every Harbor consumer; defeats the point of adopting ATIF | Pin to `"ATIF-v1.4"`; put custom data in `extra` |
| Allowing trajectories to omit `final_metrics` silently | "Saves a few lines" | Breaks downstream consumers (OpenHands populates `AgentContext` from `final_metrics`; null = lost data) | Always populate, even if computing requires summing zeroed `Metrics` |
| Logging `~/.claude/settings.json`, env vars, or full repo paths in `extra` | "More context = better replay" | Settings file holds API keys; full paths leak username/machine info | Strip secrets explicitly; use repo-relative paths; whitelist what goes in `extra` |
| Setting `Step.message=""` for empty agent turns | "Schema requires it" | Required field accepts empty string but it's noise; it's better to suppress the step entirely | Only emit a step when there's actual content (text, thinking, or tool call) |
| Adding `parent_step_id` as a top-level `Step` field | "Subagents need parent linkage" | v1.4 doesn't define this field; `extra: forbid` means it would fail validation | Use `SubagentTrajectoryRef.extra.parent_step_id` instead |
| Auto-uploading trajectories to Harbor / S3 / etc. | "Make them useful out of the box" | PROJECT.md Out of Scope; daydream is a CLI tool, not an ingestion pipeline | Document trajectory location in README; let consumers wire their own upload |

## Feature Dependencies

```
[Backend token extraction fix]                       (claude.py CostEvent populates input_tokens/output_tokens)
    └──required-by──> [Per-step Metrics population]
                          └──required-by──> [FinalMetrics aggregation]
                                                └──required-by──> [Differentiator: phase_breakdown]

[ISO 8601 timestamp on AgentEvent]
    └──required-by──> [Step.timestamp population]
                          └──enhances──> [Differentiator: Metrics.extra.duration_ms]

[session_id generated per run]
    └──required-by──> [Trajectory.session_id]
                          └──required-by──> [Subagent trajectories as siblings]
                                                └──required-by──> [SubagentTrajectoryRef linkage]

[Coalescer for TextEvent → Step.message]
    └──required-by──> [Tool-call grouping (tool_calls + observation in same Step)]
                          └──required-by──> [Validator passes (validate_tool_call_references)]

[Phase parameter threaded through run_agent()]
    └──required-by──> [Differentiator: Step.extra.phase]
                          └──enhances──> [Differentiator: FinalMetrics.extra.phase_breakdown]

[Sibling trajectory file emission]
    └──required-by──> [Exploration parallel subagents producing valid trajectories]
    └──required-by──> [Deep mode per-stack subagents producing valid trajectories]
    └──required-by──> [Parallel fix subagents producing valid trajectories]
```

### Dependency Notes

- **Backend token extraction fix is on the critical path:** every metric, every aggregate, every cost dashboard depends on it. PROJECT.md flags this; the milestone cannot ship valid `FinalMetrics` without it.
- **Coalescer (TextEvent grouping) is the trickiest piece:** AgentEvent is event-by-event but ATIF Steps are turn-by-turn. The recorder must buffer `TextEvent` chunks and flush a step when (a) a `ToolStartEvent` arrives — or (b) a `ThinkingEvent` arrives between text — or (c) `ResultEvent` arrives.
- **Sibling-file subagent emission** must be wired in `exploration_runner.pre_scan`, `deep/orchestrator.run_deep`, and `phase_fix_parallel`. Sequential `run_agent()` calls do *not* need this — they're inlined.
- **Phase context propagation** is a small, isolated change (one new optional argument on `run_agent()`) but unlocks several differentiators (`Step.extra.phase`, `FinalMetrics.extra.phase_breakdown`).

## MVP Definition

### Launch With (this milestone — minimum viable ATIF)

The set required for "every daydream run produces a valid ATIF v1.4 trajectory" per PROJECT.md Active requirements.

- [ ] **Backend token extraction fix** in `daydream/backends/claude.py` (CostEvent populates `input_tokens`, `output_tokens`, plus new `cached_tokens` field) — prerequisite for everything else
- [ ] **AgentEvent timestamp field** on every dataclass in `daydream/backends/__init__.py` (ISO 8601, set at emit time)
- [ ] **`session_id` generation** at run start, propagated through `run_agent()` calls
- [ ] **TrajectoryRecorder class** that consumes `AgentEvent` stream and produces a `Trajectory` (Pydantic model). Lives next to `agent.py`; called from each `run_agent()` invocation.
- [ ] **Step coalescer** — TextEvent buffer, ToolStart/ToolResult pairing within a step, ResultEvent → step closure
- [ ] **User-prompt step injection** — every `run_agent()` opens with a `Step(source="user", message=prompt)` segment
- [ ] **All required Pydantic fields populated** — `schema_version="ATIF-v1.4"`, `session_id`, `Agent(name, version)`, `steps[]`
- [ ] **FinalMetrics computed at trajectory close** by summing per-step `Metrics`
- [ ] **Sibling trajectory emission for parallel subagents** — `exploration_runner.pre_scan`, `deep/orchestrator`, `phase_fix_parallel` each emit one `Trajectory` per task and link via `SubagentTrajectoryRef`
- [ ] **`--trajectory <path>` CLI flag** with sensible default (`<target>/.daydream/trajectory.json`)
- [ ] **Atomic write at run completion** — full `Trajectory` validated by Pydantic, then written
- [ ] **Test fixture validates produced trajectories** via `TrajectoryValidator` from Harbor (one test per flow: normal, PR, TTT, deep)
- [ ] **Hard cutover** — remove `_log_debug`, `--debug`, all prefix-tagged logging per PROJECT.md
- [ ] **README updates** documenting trajectory location and consumer integration

### Add After Validation (next milestone)

Differentiators that materially improve trajectory utility once the foundation works.

- [ ] **`Step.extra.phase` annotation** — thread `phase` through `run_agent()`; phases.py call sites pass their phase name
- [ ] **`Trajectory.extra` flow + git_head_sha + cli_args** — captured at run start
- [ ] **`Agent.extra.backend`** — distinguishes Claude/Codex
- [ ] **`SubagentTrajectoryRef.extra.parent_step_id` + `kind`** — recovers parent linkage
- [ ] **`Metrics.extra.duration_ms`** — per-call latency
- [ ] **Best-effort flush on `KeyboardInterrupt`** — graceful shutdown writes whatever steps exist

### Future Consideration (later)

- [ ] **`Agent.tool_definitions` from Beagle manifest** — defer until Beagle exposes a stable schema
- [ ] **`FinalMetrics.extra.phase_breakdown`** — depends on `Step.extra.phase` landing first
- [ ] **`prompt_token_ids` / `completion_token_ids`** — defer until claude-agent-sdk exposes them; out of scope today
- [ ] **Multi-version support (v1.5, v1.6 with multimodal `ContentPart`)** — PROJECT.md pins v1.4; revisit only if a downstream consumer requires v1.6

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Backend token extraction fix | HIGH (unblocks all metrics) | LOW | P1 |
| TrajectoryRecorder + step coalescer | HIGH (core feature) | MEDIUM | P1 |
| User-prompt step injection | HIGH (required for valid trajectory) | LOW | P1 |
| FinalMetrics aggregation | HIGH (consumer expectation) | LOW | P1 |
| Sibling subagent trajectory files | HIGH (correctness for parallel flows) | HIGH | P1 |
| Pydantic validation in tests | HIGH (regression safety) | LOW | P1 |
| `--trajectory <path>` flag | HIGH (PROJECT.md commits) | LOW | P1 |
| Hard cutover (remove `_log_debug`) | HIGH (PROJECT.md commits) | LOW | P1 |
| `Step.extra.phase` annotation | MEDIUM (replay quality) | LOW | P2 |
| `Trajectory.extra.flow` / `git_head_sha` | MEDIUM (replay quality) | LOW | P2 |
| `Metrics.extra.duration_ms` | MEDIUM (perf analysis) | LOW | P2 |
| `SubagentTrajectoryRef.extra.parent_step_id` | MEDIUM (hierarchy reconstruction) | LOW | P2 |
| Graceful flush on Ctrl-C | MEDIUM (parity with `_log_debug`) | MEDIUM | P2 |
| `Agent.tool_definitions` | LOW (replay completeness) | MEDIUM | P3 |
| `FinalMetrics.extra.phase_breakdown` | LOW (depends on P2 first) | LOW | P3 |
| `prompt_token_ids` / RL fields | LOW (no SDK source) | HIGH | P3 (defer) |

**Priority key:** P1 = must have for milestone, P2 = should have / next milestone, P3 = future.

## Competitor / Reference Agent Feature Analysis

How other ATIF-emitting agents handle the same questions:

| Feature | Claude Code (Harbor adapter) | OpenHands (Harbor adapter) | Terminus-2 (golden) | Daydream Plan |
|---------|------------------------------|----------------------------|---------------------|---------------|
| `source` for skill/agent prompt | `"user"` (`role="user"` → `source="user"`) | `"user"` for instruction; `"system"` for system prompt | `"user"` step opens trajectory | `"user"` for Beagle skill prompt; skip system steps |
| Per-step metrics | Direct from claude-agent-sdk per-call usage | Compute deltas via subtraction (source has cumulative) | Per-call (growing prompt_tokens by turn) | Direct from `claude-agent-sdk.ResultMessage.usage` |
| Subagent representation | `subagent_trajectory_ref=None` (no Task-tool capture) | Not implemented in adapter | Sibling files (3 subagents in golden) | **Sibling files** for `anyio` task groups |
| `tool_call_id` source | `block.id` from Claude SDK, preserved verbatim | Generated from event ID | UUID-style | `ToolStartEvent.id` (= `block.id` already) |
| Step coalescing | Per-message; consecutive blocks → one step | Per-event with role grouping | Tool calls + observation in same step | Coalesce TextEvents until tool call or thinking break |
| `Trajectory.extra` usage | Not extensively populated | Adds `agent_class` to `Agent.extra` | Includes session metadata in `extra` | **Use heavily** (flow, git sha, cli args) |
| Cache token handling | `prompt = input + cache_read + cache_creation`; `cached = cache_read` | N/A | N/A | Match Claude Code |

## Confidence and Validation

**HIGH confidence:**
- All Pydantic field definitions (read directly from `harbor.models.trajectories.*` source on GitHub)
- Trajectory validator constraints (`validate_step_ids`, `validate_tool_call_references`, `validate_agent_only_fields`)
- Subagent representation (`SubagentTrajectoryRef` exists; `ObservationResult.subagent_trajectory_ref` is the linkage)
- `cached_tokens` semantics (subset of `prompt_tokens`)
- `extra: forbid` everywhere (model_config explicit)

**MEDIUM confidence:**
- Per-step metrics being "per-LLM-call" rather than "running totals." Verified against Terminus-2 golden trajectory but not against an explicit RFC statement (the upstream RFC URL returned 404 during research; `harborframework.com` docs pages don't elaborate). The Pydantic docstrings + golden file evidence is consistent and sufficient.
- `claude-agent-sdk.ResultMessage.usage` semantics. Issue #112 in claude-code-sdk-python notes historical inaccuracies. **Daydream should validate empirically** with a multi-turn test before assuming straight pass-through.

**LOW confidence:**
- `Step.extra` conventions used in the field — no central registry; suggested keys are daydream-internal proposals, not ecosystem standards. Safe because `extra` is by design a free-form bag.

## Sources

- ATIF spec doc (this repo): `docs/reference/atif_format.md`
- Project context: `.planning/PROJECT.md`
- Daydream architecture: `.planning/codebase/ARCHITECTURE.md`
- Harbor Pydantic models (verified verbatim):
  - [`Trajectory`](https://github.com/harbor-framework/harbor/blob/main/src/harbor/models/trajectories/trajectory.py)
  - [`Step`](https://github.com/harbor-framework/harbor/blob/main/src/harbor/models/trajectories/step.py)
  - [`Agent`](https://github.com/harbor-framework/harbor/blob/main/src/harbor/models/trajectories/agent.py)
  - [`ToolCall`](https://github.com/harbor-framework/harbor/blob/main/src/harbor/models/trajectories/tool_call.py)
  - [`Observation`](https://github.com/harbor-framework/harbor/blob/main/src/harbor/models/trajectories/observation.py)
  - [`ObservationResult`](https://github.com/harbor-framework/harbor/blob/main/src/harbor/models/trajectories/observation_result.py)
  - [`SubagentTrajectoryRef`](https://github.com/harbor-framework/harbor/blob/main/src/harbor/models/trajectories/subagent_trajectory_ref.py)
  - [`Metrics`](https://github.com/harbor-framework/harbor/blob/main/src/harbor/models/trajectories/metrics.py)
  - [`FinalMetrics`](https://github.com/harbor-framework/harbor/blob/main/src/harbor/models/trajectories/final_metrics.py)
  - [`__init__.py`](https://github.com/harbor-framework/harbor/blob/main/src/harbor/models/trajectories/__init__.py)
- Reference adapters:
  - [Claude Code Harbor adapter](https://github.com/harbor-framework/harbor/blob/main/src/harbor/agents/installed/claude_code.py)
  - [OpenHands Harbor adapter](https://github.com/harbor-framework/harbor/blob/main/src/harbor/agents/installed/openhands.py)
- Golden trajectory: [terminus_2/hello-world-context-summarization.trajectory.json](https://github.com/harbor-framework/harbor/blob/main/tests/golden/terminus_2/hello-world-context-summarization.trajectory.json)
- ATIF docs site: [Harbor framework docs](https://www.harborframework.com/docs/agents/trajectory-format)
- Subagent capture pattern: [agent-lens (dreadnode)](https://github.com/dreadnode/agent-lens)
- claude-agent-sdk usage fields: [GitHub issue on Python SDK ResultMessage tokens](https://github.com/anthropics/claude-code-sdk-python/issues/112)

---
*Feature research for: ATIF v1.4 trajectory emission for daydream*
*Researched: 2026-04-26*
