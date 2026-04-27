# Phase 2: Recorder Core + Event Enrichment + Mapping - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-26
**Phase:** 02-recorder-core-event-enrichment-mapping
**Areas discussed:** Step coalescing rules, Phase label propagation, Phase 2 ↔ Phase 3 API split, Redactor stub shape

---

## Step coalescing rules

| Option | Description | Selected |
|--------|-------------|----------|
| One Step per AssistantMessage | Each model turn = one agent Step containing accumulated text, reasoning, tool_calls, observations | ✓ |
| Split on every tool boundary | Every ToolStart begins a new Step; risks breaking tool_call_id intra-step invariant | |
| Split on text-after-tool | Coalesce text+tools, but text re-emerging after a tool result begins new Step | |

**User's choice:** One Step per AssistantMessage

| Option | Description | Selected |
|--------|-------------|----------|
| On next AssistantMessage or ResultMessage | Step stays open as events stream; closes on next AssistantMessage start or ResultMessage | ✓ |
| Eagerly on each event | Each event mutates and possibly closes step; risks closing before tool result | |
| Only at end of run_agent() | All events buffered, single step-build pass at end | |

**User's choice:** On next AssistantMessage or ResultMessage

| Option | Description | Selected |
|--------|-------------|----------|
| Concatenate into single message string | Append chunks to current Step's message; final message is whole turn's text | ✓ |
| Store as ContentParts list | ATIF v1.6 multimodal alternative; out of scope per PROJECT.md | |
| Final-text-block only | Drop intermediate streaming chunks | |

**User's choice:** Concatenate into single message string

| Option | Description | Selected |
|--------|-------------|----------|
| The Step opened by that AssistantMessage | Recorder maps message_id → Step at AssistantMessage start; MetricsEvent attaches by ID match | ✓ |
| Always the last open agent Step | MetricsEvent attaches to whichever Step is currently open | |

**User's choice:** The Step opened by that AssistantMessage

**Notes:** No further follow-ups; user moved to next area.

---

## Phase label propagation

| Option | Description | Selected |
|--------|-------------|----------|
| Keyword-only arg on run_agent() | `*, phase: str` required arg; ~18 call sites updated | ✓ (revised: enum, required, no backwards-compat) |
| Phase ContextVar | Mirrors recorder ContextVar pattern; zero signature changes | |
| Bundle phase + future labels into a new dataclass arg | `*, ctx: AgentCallContext` for future-proofing | |

**User's choice (free-text):** "required arg on run_agent and it should be an enum, we do not care about backwards compat, just update callers"

**Notes:** First framing recommended ContextVar to avoid signature churn (per Architecture research Anti-Pattern 3). User pushed back: clean idiomatic Python beats hidden ambient state. Reframed: phase label is per-call data, signature is the right home; recorder still uses ContextVar because it's run-scoped infrastructure. Enum gives type-safety; the existing daydream/config.py ReviewSkillChoice enum sets the precedent for str-valued enums.

| Option | Description | Selected |
|--------|-------------|----------|
| Required — no default | Every run_agent() caller must declare its phase; mypy catches misses | ✓ |
| Default to empty string or None | Lower-friction migration; risks silent MAP-08 violation | |

**User's choice:** Required — no default

| Option | Description | Selected |
|--------|-------------|----------|
| Set once at recorder init from RunConfig | Per-trajectory invariant; runner knows the flow at construction | ✓ |
| Also a kwarg on run_agent() | Symmetric with phase but unnecessary — run_flow doesn't change inside a trajectory | |

**User's choice:** Set once at recorder init from RunConfig

| Option | Description | Selected |
|--------|-------------|----------|
| daydream/trajectory.py | Lives next to TrajectoryRecorder; trajectory-domain data | ✓ |
| daydream/config.py | Centralizes with ReviewSkillChoice but couples config to trajectory concepts | |
| daydream/phases.py | Lives with producers but module-bloat ban discourages additions | |

**User's choice:** daydream/trajectory.py

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — DaydreamRunFlow enum | Type-safe; matches MAP-09's closed set | ✓ |
| No — plain str field on RunConfig | Matches existing RunConfig style but loses type safety | |

**User's choice:** Yes — DaydreamRunFlow enum

---

## Phase 2 ↔ Phase 3 API split

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal — single-trajectory only | Phase 2: one ContextVar, one Trajectory, no Invocation parent. Phase 3 adds second ContextVar + parent linkage + sibling write together as one coherent unit | ✓ (after walk-back) |
| Full architecture — ship two ContextVars + Invocation parent linkage now | Phase 2 ships the architectural skeleton; Phase 3 just turns parallel paths on | (initially selected, then walked back) |
| Half-and-half — ship Invocation but not parent linkage | Middle ground; Phase 2 introduces Invocation but no parent field | |

**User's choice (after walk-back):** Minimal — single-trajectory only

**Notes:** User initially selected "Full architecture" but pushed back on the resulting follow-up question (about how to handle `parent != None` in Phase 2 when sibling write isn't wired): "this phase split seems poor since they are dependent" / "its just weird that you say this is ideal, then say that we have to build some broken architecture first only to change it in phase 3." The complaint was correct — anticipating Phase 3 in Phase 2's API surface meant shipping half-built parent linkage that Phase 3 then has to flip on. Reframing in Reading B: "Minimal" creates a coherent Phase 3 that owns the upgrade end-to-end (second ContextVar + Invocation parent + sibling write + parent observation patching as one unit). Saved as user feedback memory: `feedback_phase_split_coherence.md`.

| Option | Description | Selected |
|--------|-------------|----------|
| One Invocation per run_agent() call | Each call opens its own scope; clean home for in-flight tool map and step buffer | ✓ |
| No Invocation — recorder buffers globally | Simpler but mixes scopes when multiple run_agent() calls overlap | |

**User's choice:** One Invocation per run_agent() call

| Option | Description | Selected |
|--------|-------------|----------|
| Module-level get_current_recorder() helper | trajectory.py exposes module-level helper; ContextVar stays private | ✓ |
| Expose _RECORDER_VAR directly | Less indirection but exposes implementation detail | |

**User's choice:** Module-level get_current_recorder() helper

---

## Redactor stub shape

| Option | Description | Selected |
|--------|-------------|----------|
| No-op pass-through with the final API surface | Phase 2 ships redact_step(step) -> Step returning input unchanged; recorder calls it on every Step; Phase 4 fills rule list | ✓ |
| Empty class shell, recorder doesn't call it yet | Phase 4 implements methods AND wires recorder to call them — same half-built pattern just rejected | |
| Reconsider — push CORE-01's Redactor mention to Phase 4 | Argue that Phase 2 shouldn't expose Redactor; would require ROADMAP.md edit | |

**User's choice:** No-op pass-through with the final API surface

**Notes:** Distinction from the half-built pattern rejected in Area 3: this Redactor surface is *exercised* in Phase 2 (recorder calls `redact_step` on every Step at flush time), so Phase 4 is purely additive (add patterns to the rule list). The wiring is real; only the rules are deferred.

| Option | Description | Selected |
|--------|-------------|----------|
| Single redact_step(step) -> Step | One method; recorder doesn't need to know which fields are scrubbed | ✓ |
| Per-surface methods | Finer-grained Phase 4 control but couples recorder to redactor scope | |
| redact_trajectory(traj) -> Trajectory | Single bulk call at finalize; risks running post-validation | |

**User's choice:** Single redact_step(step) -> Step

| Option | Description | Selected |
|--------|-------------|----------|
| Per-step at flush time | Recorder calls redactor before adding finalized Step to Trajectory.steps; partial-write paths inherit | ✓ |
| On serialization at __aexit__ | Latest-possible redaction but partial.json paths emit raw secrets | |

**User's choice:** Per-step at flush time

---

## Claude's Discretion

- Internal `Invocation` layout (deque vs list, in-flight map field naming)
- Whether `Redactor` exposes private helper methods alongside `redact_step()`
- Exact name of the `now_iso()` helper used to stamp event timestamps
- ATIF model construction style (constructor explicitness vs Pydantic defaults) as long as `daydream.atif.validate()` accepts the output
- Whether `MetricsEvent.cost_usd` is populated per-step from `AssistantMessage.usage` (if available) or only from `ResultMessage`-derived `CostEvent`

## Deferred Ideas

- Two-ContextVar architecture (`_RECORDER_VAR` + `_CURRENT_INVOCATION`) — Phase 3
- `Invocation(parent=Invocation | None)` parent linkage — Phase 3
- Sibling trajectory file write — Phase 3 (SUBA-02..04, SUBA-06)
- Continuation appending to existing trajectory — Phase 3 (SUBA-05)
- Redaction regex patterns + test corpus — Phase 4 (REDA-01..06)
- `--trajectory <path>` CLI flag + `--debug` removal — Phase 4 (CLI-01..05, CUT-01..08)
- SIGINT partial-flush to `.partial.json` — Phase 4 (CLI-03)
- AST-based `_log_debug` orphan sweep — Phase 4 (CUT-08)
- `Trajectory.to_json_dict()` perf benchmarks — Phase 5 (Pitfall 15 gate)
