# Research Synthesis: Daydream — ATIF Migration

**Synthesized:** 2026-04-26
**Sources:** STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md (in this directory)

## DECISIONS PENDING USER REVIEW (resolved 2026-04-26)

Four research findings contradicted or refined initial PROJECT.md decisions. All four were surfaced to the user and resolved before requirements scoping. Final outcomes are reflected in PROJECT.md's Key Decisions table; this section preserves the rationale.

### Decision 1: Harbor dependency strategy → VENDOR

**Original PROJECT.md decision:** Take Harbor as a runtime dep.

**Research finding (Stack):** Harbor 0.5.0 has 21 required transitive packages (~150–250 MB) including `litellm`, `fastapi`, `uvicorn`, `datasets`, `supabase`, `tinker`. `litellm>=1.80.8` (Harbor's bound) accepted versions 1.82.7/1.82.8 which were PyPI-quarantined March 2026 for a malicious `.pth` file exfiltrating SSH keys and cloud tokens. Harbor's bound has not been tightened. The trajectory submodule itself is pure Pydantic + stdlib, ~700 LOC, Apache-2.0.

**Resolution:** Vendor `harbor.models.trajectories.*` and `harbor.utils.trajectory_validator` into `daydream/atif/`. Promote `pydantic>=2.11.7` to explicit dep. No new runtime dependencies, no supply-chain exposure.

### Decision 2: ATIF schema version → BUMP TO v1.6

**Original PROJECT.md decision:** Pin to ATIF-v1.4.

**Research finding (Stack):** Harbor's current `Trajectory` model defaults to `"ATIF-v1.6"`. No Harbor golden fixture is at v1.4 — OpenHands goldens are v1.5, Terminus-2 goldens are v1.6. Validator accepts v1.0–v1.6 by `Literal` field. v1.6 adds `ContentPart`/`ImageSource` (multimodal) — daydream emits text only, so the `str` form of `Step.message` remains valid in v1.6.

**Resolution:** Bump emission pin to `"ATIF-v1.6"`. Validator accepts older versions; emission is v1.6 only.

### Decision 3: Subagent representation → SIBLING FILES via subagent_trajectory_ref

**Original PROJECT.md decision:** "`run_agent()` invocations nested as hierarchical steps using ATIF's subagent delegation model" (mechanism unspecified).

**Research finding (Architecture + Features):** ATIF v1.4+ uses `ObservationResult.subagent_trajectory_ref: list[SubagentTrajectoryRef]` pointing to **separate sibling trajectory files**. There is no `parent_step_id`, no nested `Trajectory`, no `subagent_steps`.

**Resolution:** Mapping confirmed.

| Daydream construct | ATIF mapping |
|---|---|
| Sequential phases (review → parse → fix → test) | Inline as continuous steps in one root trajectory |
| `phase_fix_parallel` (parallel `anyio` task group) | Sibling trajectory files via `SubagentTrajectoryRef` |
| `daydream/deep/orchestrator.run_deep` per-stack fan-out | Sibling trajectory files |
| `daydream/exploration_runner.pre_scan` specialists | Sibling trajectory files |
| `run_agent_with_continuation` continuations | Append to same trajectory (preserve agent identity) |

### Decision 4: Per-step Metrics → NEW MetricsEvent + CostEvent.cached_tokens

**Original PROJECT.md decision:** "`CostEvent` → per-step `Metrics`" (one CostEvent per `run_agent()` call).

**Research finding (Stack + Features):** Current `CostEvent` fires once per call (`ResultMessage`). It has no `cached_tokens` field. `daydream/backends/claude.py:120-128` emits `input_tokens=None, output_tokens=None` despite data being present in `ResultMessage.usage`. ATIF wants per-step Metrics ideally keyed to each LLM turn via `AssistantMessage.message_id`.

**Resolution:** Add new `MetricsEvent(message_id, prompt_tokens, completion_tokens, cached_tokens, cost_usd, timestamp)` keyed by `message_id`, fired per `AssistantMessage` during streaming. Extend `CostEvent` with `cached_tokens` and fix Claude backend's data-loss bug in the same phase.

## Executive Summary

Daydream's ATIF migration is a well-scoped brownfield replacement: swap unstructured prefix-tagged debug lines for machine-parseable ATIF v1.6 trajectory documents. The schema is mature (Pydantic-enforced, `extra: "forbid"` on every model), the reference adapters are documented, and the token data is already on `claude-agent-sdk`'s `ResultMessage.usage` and `AssistantMessage.usage` — it's just not being extracted. The highest-confidence path is to vendor Harbor's 700-LOC trajectory submodule (Apache-2.0, pure Pydantic), fix the one-line token extraction bug in the Claude backend, and build a `TrajectoryRecorder` that sits alongside `agent.py`'s event loop propagated via `ContextVar`.

The two structural risks are subagent wiring and redaction. Subagent wiring requires emitting sibling trajectory files for parallel `anyio` task groups and coordinating parent observations with `SubagentTrajectoryRef` entries — getting this wrong produces trajectories that fail Harbor's validator or misrepresent the execution graph. Redaction must land in the same release as the always-on trajectory writer: daydream's bypass-permissions tool surface means bash output, file reads, and reasoning content can contain API keys, git credentials, and user paths. The current debug log is local-only and opt-in; trajectories are always-on and intended for sharing with Harbor and training pipelines.

## Key Findings by Dimension

### Stack
- **Don't take Harbor as runtime dep.** Vendor the ~700-LOC pure-Pydantic submodule + validator under `daydream/atif/` (Apache-2.0).
- **Promote `pydantic>=2.11.7` to explicit dep** in `pyproject.toml` (already transitive via `claude-agent-sdk`).
- **Token extraction is a one-line fix** at `daydream/backends/claude.py:120-128`. `ResultMessage.usage` carries `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`. `AssistantMessage.usage` carries the same keys per-step (with `message_id` for parallel-tool dedup).
- **Codex parity is partial.** `turn.completed.usage` provides `input_tokens` + `output_tokens` only — no cost, no cache. Acceptable since ATIF Metrics fields are optional.
- **Stdlib `uuid.uuid4()` + `datetime.now(timezone.utc).isoformat()` are sufficient.** No `ulid-py`.
- **Test corpus:** vendor Harbor's `tests/golden/` (13 Terminus-2 + 4 OpenHands trajectories) into `tests/fixtures/atif_golden/` and parametrize over them.

### Features
- **Required vs optional fields** documented per ATIF level (`Trajectory`, `Step`, `ToolCall`, `Observation`, `ObservationResult`, `Metrics`, `FinalMetrics`).
- **Step source semantics:** Beagle skill prompt = `source="user"` (NOT `"system"`). ATIF reserves `"system"` for system-prompt preambles. Setting agent-only fields (`model_name`, `reasoning_content`, `tool_calls`, `metrics`) on user/system steps is a hard validator failure.
- **`tool_call_id` is intra-step scoped.** `source_call_id` must reference a `tool_call_id` in the *same step's* `tool_calls`. This forces buffer-until-matched + flush-together pattern.
- **Per-step Metrics are per-LLM-call values, not deltas.** Verified against Terminus-2 golden.
- **`cached_tokens` is a SUBSET of `prompt_tokens`** (per Pydantic docstring). Map: `prompt_tokens = input + cache_read + cache_creation`; `cached_tokens = cache_read`.
- **Subagent delegation = `ObservationResult.subagent_trajectory_ref`** pointing to separate files. Not nested. Not `parent_step_id`.

### Architecture
- **Recorder placement:** per-run `ContextVar` in new `daydream/trajectory.py` module, NOT on `AgentState`. Per-run lifecycle, copy-on-spawn handles anyio task groups automatically, clean test isolation.
- **Recording call site:** inside `run_agent()`'s existing `async for event in event_iter:` loop, before UI rendering. Single chokepoint; backends stay trajectory-unaware.
- **Subagent representation:** sibling trajectory files linked by `subagent_trajectory_ref`. ContextVar copy-on-spawn handles parent → child establishment without threading recorder through every phase signature.
- **Phases are sibling top-level steps** in one flat trajectory with `extra.daydream_phase` label. Not subagents — that would inflate file count and break "one session" mental model.
- **Build order DAG:** Stage 0 vet harbor → Stage 1 greenfield `trajectory.py` → Stage 2 backend enrichment → Stage 3 `run_agent()` cutover → Stage 4 subagent wiring → Stage 5 continuation + tests + docs.
- **Failure mode:** catch and degrade. Recorder failures must not crash the user's review/fix run. Exception: explicit `--trajectory <path>` flag elevates write failure to error+exit.
- **AgentState shrinks:** `debug_log: TextIO | None` removed entirely.

### Pitfalls (16 catalogued; 7 HIGH severity)
- **Pitfall 8 (privacy / secret redaction)** — must-land-with-cutover security item. Tool args carry API keys, file paths, env dumps.
- **Pitfall 13 (lazy-import escape)** — `daydream/backends/codex.py:37` does `from daydream.agent import _log_debug` *inside a function*. AST sweep needed; grep won't catch it.
- **Pitfall 7 (subagent vs flat steps)** — daydream has 3 concurrency patterns (sequential phase chain, `phase_fix_parallel`, deep-mode fan-out, exploration). Mapping resolved (see Decision 3).
- **Pitfall 14 (module bloat)** — explicit ban on adding `Step()` calls inside `phases.py` or `ui.py`. All ATIF model construction stays in `daydream/trajectory.py`.
- **Pitfall 5 (token semantics)** + **Pitfall 3 (dangling source_call_id)** — two highest-risk schema-validation failures.
- **Pitfall 11 (test brittleness)** — schema-validity + behavior-predicate tests, NOT full-tree snapshot equality.

## Refined Build DAG (5 phases)

The architecture's 5-stage internal DAG was promoted to the project's 5 user-visible phases. See ROADMAP.md for the canonical structure. In short:

1. **Vendor ATIF Foundation** — VEND-01..05 (5 reqs)
2. **Recorder Core + Event Enrichment + Mapping** — CORE + EVNT + MAP (26 reqs); intentionally bundled because splitting yields broken intermediate states (empty Metrics blocks)
3. **Subagent Wiring** — SUBA-01..09 (9 reqs)
4. **Cutover + Redaction + CLI** — REDA + CUT + CLI (19 reqs); redaction lands WITH cutover per Pitfall 8
5. **Test Hardening + Documentation** — TEST + DOCS (13 reqs)

## Open Questions Deferred to Phase Research

Surfaced during research; not blocking initialization but flagged for the relevant phase planner:

1. **Sibling trajectory file naming convention** — `SubagentTrajectoryRef.trajectory_path` accepts arbitrary paths. The PROJECT.md/REQUIREMENTS.md convention is `<root_dir>/.daydream/trajectories/<session_id>.<descriptor>.json` but the `<descriptor>` slug pattern (which segment of the run flow it represents) is up to the planner to lock in.
2. **`--ttt` sub-phase trajectory granularity** — `phase_understand_intent`, `phase_alternative_review`, `phase_generate_plan` are sequential calls. PROJECT.md says one trajectory per run, suggesting inline. Confirm during Phase 2 / Phase 3 planning.
3. **`--pr` mode prompt source** — GitHub-fetched comments aren't `run_agent()` output; they become the *prompt* for the next agent step. They should be `source="user"` content but enriched (perhaps via `Step.extra.pr_comment_id`). Worth designing during Phase 3.
4. **Empirical SDK token-drift verification** — `claude-agent-sdk==0.1.52` issue #112 historically reported drift. Phase 2 must include a multi-turn fixture test before shipping (TEST-06 covers this).
5. **Apache-2.0 attribution mechanics** — verify Harbor `LICENSE` and `NOTICE` requirements at vendor time (DOCS-05 covers).
6. **`ResultMessage.model_usage` data** — spec is silent on whether per-model breakdown belongs in `Agent.extra` or per-step `Step.extra`. OpenHands example puts model details on `Agent`.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Harbor source verified directly; SDK dataclass shapes verified against official docs and source; Codex fixtures verified in-repo |
| Features | HIGH | Harbor Pydantic models read verbatim; all validator constraints verified; golden fixtures inspected |
| Architecture | HIGH (daydream-internal) / MEDIUM (ATIF subagent semantics) | Harbor RFC URL returned 404 at research time; subagent model inferred from Pydantic field shape + harborframework.com docs + cross-search confirmation |
| Pitfalls | HIGH (codebase-grounded) / MEDIUM (cross-tool conventions) | 13 of 16 pitfalls grounded in daydream source or ATIF spec; 3 inferred from OpenHands/Claude Code adapter patterns |

**Overall: HIGH** for critical path (vendor → trajectory.py → backend fix → run_agent integration). MEDIUM for subagent wiring specifics and SDK token semantics — both have empirical validation gates in Phase 2 (TEST-06) and Phase 3.
