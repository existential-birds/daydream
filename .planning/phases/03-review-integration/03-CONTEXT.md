# Phase 3: Review Integration - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Downstream consumption of `ExplorationContext` produced by Phase 2. This phase makes review and TTT plan output **grounded** in actual codebase context: cross-file dependency tracing surfaces in review output, every recommendation carries a confidence score with verification rationale, recommendations that contradict detected conventions are filtered (and contradictions in the *reviewed code itself* are elevated to first-class issues), and TTT plan steps cite real file paths and symbols. Both `--ttt` and normal flows must produce exploration-enriched output via the unified architecture Phase 2 wired in.

**Strictness mandate:** This is reviewing **AI-generated code**. Reviews must be strict and thorough. Confidence calibration, contradiction detection, and grounding all serve that goal — no soft-pedaling, no filtering away real issues.

</domain>

<decisions>
## Implementation Decisions

### Confidence Scoring (QUAL-02)
- **D-01:** Confidence is **schema-enforced**, not prompt-only. Extend `FEEDBACK_SCHEMA` and `ALTERNATIVE_REVIEW_SCHEMA` so every issue object requires:
  ```
  { id, description, file, line, confidence: "HIGH"|"MEDIUM"|"LOW", rationale: string }
  ```
  Structured output enforces presence; downstream parsing stays trivial. No "forgot to label" failure mode.
- **D-02:** Confidence semantics (in the agent prompt):
  - **HIGH** — Verified directly against `ExplorationContext` (a `Dependency` edge, a `Convention` entry, or a file in `affected_files` confirms it)
  - **MEDIUM** — Consistent with exploration context but not directly verified by a specific entry
  - **LOW** — Diff-only inference; no exploration evidence
- **D-03:** `rationale` must explicitly state **what was verified vs. what was inferred** (per success criterion #2). Empty/vague rationales are a prompt failure — instruct the agent to cite the specific `ExplorationContext` field or note "no exploration evidence" for LOW.

### Convention Handling (QUAL-03)
- **D-04:** Two distinct cases must not be conflated:
  1. **Reviewer recommendations that contradict detected conventions** → filter/drop. The reviewer should not suggest changes that violate house style. This is the QUAL-03 success criterion.
  2. **Reviewed code that contradicts detected conventions** → elevate to **HIGH-confidence issues**. This is exactly what we want to catch in AI-generated code. These are *not* filtered — they are the most valuable findings.
- **D-05:** The agent prompt must distinguish these explicitly: "Before proposing a fix, check it against `ExplorationContext.conventions`. If your fix violates a convention, drop it. If the reviewed code violates a convention, that IS the issue — flag it as HIGH confidence with the convention citation in `rationale`."
- **D-06:** No "soft" flagging tier. Either a recommendation survives convention check or it doesn't. Either reviewed code violates a convention (HIGH issue) or it doesn't. Strictness over nuance — this is AI-code review.

### Plan Grounding (OUTP-01)
- **D-07:** Extend `PLAN_SCHEMA` so each plan step requires:
  ```
  references: [{ file: string, symbol: string }]
  ```
  Populated from `ExplorationContext.affected_files` (file paths) and dependency-tracer output (symbol names).
- **D-08:** Empty `references` array is allowed but signals a LOW-confidence step. The TTT renderer should visually distinguish steps with empty references (e.g., dim text or warning marker) so the user can spot ungrounded steps.
- **D-09:** No fabricated references. Prompt instruction: "Only include references that appear in the provided ExplorationContext. If you cannot ground a step, leave references empty rather than inventing paths."

### Cross-File Dep Surfacing (QUAL-01)
- **D-10:** Both presentation patterns — top "Dependency Impact" section *and* inline per-issue references.
  - **Top section:** Summarizes the call-chain analysis from `ExplorationContext.dependencies` before listing issues. Gives reviewers the big picture.
  - **Per-issue:** When an issue's `rationale` cites a dependency, include the file:symbol reference inline. Lets reviewers trace the evidence for each finding.
- **D-11:** The "Dependency Impact" section is rendered by the prompt instruction, not a separate post-processing pass. The agent is told to begin its review output with the dependency summary section before issues.

### Architecture Unification (OUTP-02)
- **D-12:** Single shared helper for prompt injection: `ExplorationContext.to_prompt_section()` (method on the dataclass). Returns a structured-text rendering of the context suitable for prepending to any phase prompt.
- **D-13:** All four phase functions (`phase_review`, `phase_understand_intent`, `phase_alternative_review`, `phase_generate_plan`) call `to_prompt_section()` and prepend its output to their existing task-specific preamble. One source of truth for how exploration becomes prompt text.
- **D-14:** `to_prompt_section()` lives on the `ExplorationContext` dataclass in `daydream/exploration.py` (Phase 1's home for that type). Co-locates rendering with the data model.

### Claude's Discretion
- Exact prompt language for confidence semantics, convention checks, and plan grounding — Claude writes during planning.
- The structured-text format `to_prompt_section()` produces — markdown sections, fenced blocks, whatever renders cleanly in agent prompts.
- How the TTT plan renderer visually marks LOW-confidence (empty-references) steps — dim text, icon, whatever fits existing UI patterns.
- Whether the "Dependency Impact" section in `.review-output.md` has a fixed header string or is left to the agent — Claude picks based on parser fragility.
- Schema migration approach for `FEEDBACK_SCHEMA`/`ALTERNATIVE_REVIEW_SCHEMA`/`PLAN_SCHEMA` — additive field with prompt-side instruction is fine; no backwards-compat shims needed since this is internal.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 1 + 2 Foundation (MUST read — this phase consumes their output)
- `daydream/exploration.py` — `ExplorationContext`, `FileInfo`, `Convention`, `Dependency` — the data structures this phase reads from and adds `to_prompt_section()` to
- `.planning/phases/01-exploration-infrastructure/01-CONTEXT.md` — Phase 1 decisions on data model
- `.planning/phases/02-pre-scan-exploration/02-CONTEXT.md` — Phase 2 decisions on how `ExplorationContext` gets populated and where it lives on `RunConfig`

### Schemas to Extend
- `daydream/phases.py` — `FEEDBACK_SCHEMA` (D-01: add `confidence` + `rationale`)
- `daydream/phases.py` — `ALTERNATIVE_REVIEW_SCHEMA` (D-01: add `confidence` + `rationale`)
- `daydream/phases.py` — `PLAN_SCHEMA` (D-07: add `references[]` per step)

### Prompt Injection Targets
- `daydream/phases.py` — `phase_review` (normal flow consumer)
- `daydream/phases.py` — `phase_understand_intent` (TTT consumer)
- `daydream/phases.py` — `phase_alternative_review` (TTT consumer)
- `daydream/phases.py` — `phase_generate_plan` (TTT consumer; also where plan grounding lives)

### Output Rendering
- `daydream/ui.py` — TTT plan rendering (where empty-references steps need visual distinction per D-08)
- `daydream/config.py` — `REVIEW_OUTPUT_FILE` constant (`.review-output.md` target)

### Requirements & Roadmap
- `.planning/REQUIREMENTS.md` — QUAL-01, QUAL-02, QUAL-03, OUTP-01, OUTP-02 definitions
- `.planning/ROADMAP.md` — Phase 3 goal + five success criteria

### Codebase Maps
- `.planning/codebase/ARCHITECTURE.md` — Layer responsibilities
- `.planning/codebase/CONVENTIONS.md` — Established patterns to respect

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ExplorationContext` and helper dataclasses from Phase 1 — already typed, ready to grow a `to_prompt_section()` method
- `RunConfig.exploration_context` (Phase 2 D-14) — already in place, populated by pre-scan, available to every phase function
- `FEEDBACK_SCHEMA`, `ALTERNATIVE_REVIEW_SCHEMA`, `PLAN_SCHEMA` in `daydream/phases.py` — schema-extension is the established pattern for structured output
- Existing `_build_*_prompt()` helpers in `phases.py` — the natural seam for prepending `to_prompt_section()` output

### Established Patterns
- Schema-enforced structured output for parsed agent results (no text scraping)
- Prompt assembly via small helper functions per phase
- `RunConfig` carries cross-phase data; no hidden module state
- Google-style docstrings on all public functions
- Strict mypy + ruff — schema dict literals must remain typeable

### Integration Points
- Each of the four consumer phase functions: 1-line addition to prepend `config.exploration_context.to_prompt_section()` if context is non-None
- Each of the three schemas: additive field changes
- TTT plan UI in `ui.py`: branch on `step.references` emptiness for visual distinction
- `.review-output.md` rendering: agent-driven, no parser changes needed if D-11 holds (agent writes the section directly)

</code_context>

<specifics>
## Specific Ideas

- **This phase reviews AI-generated code.** Strictness is the north star. When in doubt between "filter" and "surface", surface. When in doubt between LOW and MEDIUM confidence, prefer the lower bucket so reviewers know it's diff-only.
- **Two distinct convention cases** (D-04) is the most important conceptual point — the planner and researcher must internalize this. Filtering reviewer suggestions is QUAL-03; elevating reviewed-code violations is the entire point of grounded review.
- **No fabricated references** (D-09) — if the agent invents file paths, the whole grounding promise collapses. Prompt language must be explicit and the schema must allow empty arrays so there's no incentive to make things up.
- **Single `to_prompt_section()` helper** (D-12) is the OUTP-02 enforcement mechanism. Reviewer/researcher should not invent per-phase rendering variants.

</specifics>

<deferred>
## Deferred Ideas

- **Confidence-score calibration via test corpus** — Could measure whether HIGH issues are actually verified post-hoc. v2 quality work, not v1.
- **Convention conflict telemetry** — Counting how often the reviewer's own suggestions get filtered for convention violation would be a good prompt-quality signal. Out of scope for v1.
- **Plan reference auto-validation** — Could verify that every `references[]` entry actually exists in the repo before rendering. Would catch fabrication. Defer to v2; for v1 we trust the prompt + schema.
- **Soft-flag tier for minor convention drift** — Explicitly rejected (D-06). Strictness over nuance.
- **On-demand exploration when grounding is insufficient** — That's Phase 4's job.

</deferred>

---

*Phase: 03-review-integration*
*Context gathered: 2026-04-07*
