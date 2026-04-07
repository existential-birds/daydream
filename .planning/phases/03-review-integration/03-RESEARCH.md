# Phase 3: Review Integration - Research

**Researched:** 2026-04-07
**Domain:** Schema-enforced structured output + prompt engineering for grounded AI code review
**Confidence:** HIGH

## Summary

Phase 3 is a **schema + prompt + rendering** phase, not a new-infrastructure phase. Phase 2 already wired
`config.exploration_context` through to all four consumer phase functions and `ExplorationContext.to_prompt_section()`
already exists in `daydream/exploration.py`. The four `_build_*_prompt` helpers in `daydream/phases.py`
already prepend `to_prompt_section()` output when `exploration_context` is non-None (verified at lines
421-422, 843-844, 924-925, 1070-1071).

What remains for Phase 3: (1) extend three existing JSON schemas with required `confidence`, `rationale`,
and `references[]` fields; (2) rewrite the per-phase prompt preambles to teach the agent confidence
semantics, the two-distinct-convention-cases rule, and the no-fabricated-references rule; (3) add a
"Dependency Impact" section instruction to the review prompt; (4) update the TTT plan renderer in
`ui.py` to visually distinguish steps with empty `references[]`.

**Primary recommendation:** Treat this phase as a surgical edit to `daydream/phases.py` schemas + prompt
builders, plus a small visual change in `daydream/ui.py`. No new modules. No new dependencies. The
strictness mandate (this is reviewing AI-generated code) is enforced primarily through prompt language
and schema requirements — there is no separate filter pass.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Confidence Scoring (QUAL-02)**
- **D-01:** Confidence is **schema-enforced**, not prompt-only. Extend `FEEDBACK_SCHEMA` and
  `ALTERNATIVE_REVIEW_SCHEMA` so every issue object requires:
  ```
  { id, description, file, line, confidence: "HIGH"|"MEDIUM"|"LOW", rationale: string }
  ```
  Structured output enforces presence; downstream parsing stays trivial. No "forgot to label" failure mode.
- **D-02:** Confidence semantics (in the agent prompt):
  - **HIGH** — Verified directly against `ExplorationContext` (a `Dependency` edge, a `Convention` entry,
    or a file in `affected_files` confirms it)
  - **MEDIUM** — Consistent with exploration context but not directly verified by a specific entry
  - **LOW** — Diff-only inference; no exploration evidence
- **D-03:** `rationale` must explicitly state **what was verified vs. what was inferred** (per success
  criterion #2). Empty/vague rationales are a prompt failure — instruct the agent to cite the specific
  `ExplorationContext` field or note "no exploration evidence" for LOW.

**Convention Handling (QUAL-03)**
- **D-04:** Two distinct cases must not be conflated:
  1. **Reviewer recommendations that contradict detected conventions** → filter/drop. The reviewer
     should not suggest changes that violate house style. This is the QUAL-03 success criterion.
  2. **Reviewed code that contradicts detected conventions** → elevate to **HIGH-confidence issues**.
     This is exactly what we want to catch in AI-generated code. These are *not* filtered — they are
     the most valuable findings.
- **D-05:** The agent prompt must distinguish these explicitly: "Before proposing a fix, check it
  against `ExplorationContext.conventions`. If your fix violates a convention, drop it. If the
  reviewed code violates a convention, that IS the issue — flag it as HIGH confidence with the
  convention citation in `rationale`."
- **D-06:** No "soft" flagging tier. Either a recommendation survives convention check or it doesn't.
  Either reviewed code violates a convention (HIGH issue) or it doesn't. Strictness over nuance —
  this is AI-code review.

**Plan Grounding (OUTP-01)**
- **D-07:** Extend `PLAN_SCHEMA` so each plan step requires:
  ```
  references: [{ file: string, symbol: string }]
  ```
  Populated from `ExplorationContext.affected_files` (file paths) and dependency-tracer output (symbol names).
- **D-08:** Empty `references` array is allowed but signals a LOW-confidence step. The TTT renderer
  should visually distinguish steps with empty references (e.g., dim text or warning marker) so the
  user can spot ungrounded steps.
- **D-09:** No fabricated references. Prompt instruction: "Only include references that appear in the
  provided ExplorationContext. If you cannot ground a step, leave references empty rather than
  inventing paths."

**Cross-File Dep Surfacing (QUAL-01)**
- **D-10:** Both presentation patterns — top "Dependency Impact" section *and* inline per-issue references.
  - **Top section:** Summarizes the call-chain analysis from `ExplorationContext.dependencies` before
    listing issues. Gives reviewers the big picture.
  - **Per-issue:** When an issue's `rationale` cites a dependency, include the file:symbol reference
    inline. Lets reviewers trace the evidence for each finding.
- **D-11:** The "Dependency Impact" section is rendered by the prompt instruction, not a separate
  post-processing pass. The agent is told to begin its review output with the dependency summary
  section before issues.

**Architecture Unification (OUTP-02)**
- **D-12:** Single shared helper for prompt injection: `ExplorationContext.to_prompt_section()`
  (method on the dataclass). Returns a structured-text rendering of the context suitable for
  prepending to any phase prompt.
- **D-13:** All four phase functions (`phase_review`, `phase_understand_intent`, `phase_alternative_review`,
  `phase_generate_plan`) call `to_prompt_section()` and prepend its output to their existing
  task-specific preamble. One source of truth for how exploration becomes prompt text.
- **D-14:** `to_prompt_section()` lives on the `ExplorationContext` dataclass in `daydream/exploration.py`
  (Phase 1's home for that type). Co-locates rendering with the data model.

### Claude's Discretion
- Exact prompt language for confidence semantics, convention checks, and plan grounding — Claude
  writes during planning.
- The structured-text format `to_prompt_section()` produces — markdown sections, fenced blocks,
  whatever renders cleanly in agent prompts.
- How the TTT plan renderer visually marks LOW-confidence (empty-references) steps — dim text, icon,
  whatever fits existing UI patterns.
- Whether the "Dependency Impact" section in `.review-output.md` has a fixed header string or is left
  to the agent — Claude picks based on parser fragility.
- Schema migration approach for `FEEDBACK_SCHEMA`/`ALTERNATIVE_REVIEW_SCHEMA`/`PLAN_SCHEMA` — additive
  field with prompt-side instruction is fine; no backwards-compat shims needed since this is internal.

### Deferred Ideas (OUT OF SCOPE)
- **Confidence-score calibration via test corpus** — v2 quality work.
- **Convention conflict telemetry** — counting filtered suggestions. Out of scope for v1.
- **Plan reference auto-validation** — verifying `references[]` against the repo. Defer to v2; v1
  trusts prompt + schema.
- **Soft-flag tier for minor convention drift** — explicitly rejected (D-06). Strictness over nuance.
- **On-demand exploration when grounding is insufficient** — that's Phase 4's job.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| QUAL-01 | Cross-file dependency tracing follows call chains to catch breaking changes | `ExplorationContext.dependencies` already populated by Phase 2 dependency-tracer subagent. Surface via D-10/D-11 (top section + inline per-issue refs) — driven by prompt instruction in `phase_review` builder. |
| QUAL-02 | Each recommendation carries confidence score with rationale | D-01: schema-enforced `confidence` enum + required `rationale` string in `FEEDBACK_SCHEMA` and `ALTERNATIVE_REVIEW_SCHEMA`. D-02 defines the three buckets. D-03 requires verified-vs-inferred phrasing. |
| QUAL-03 | Recommendations contradicting conventions are filtered | D-04/D-05: prompt-driven, two-case distinction. Reviewer self-filters proposed fixes against `ExplorationContext.conventions`. Reviewed-code violations are *elevated* as HIGH issues, not filtered. |
| OUTP-01 | TTT plan generation references actual file paths and signatures | D-07: extend `PLAN_SCHEMA` with required `references[]` per step. D-09: prompt forbids fabrication; empty array preferred over invented paths. D-08: empty arrays render with visual warning in `ui.py`. |
| OUTP-02 | Both TTT and normal flows use same exploration architecture | Already TRUE in code: D-12/D-13 — `to_prompt_section()` is the single shared helper, called by all four phase prompt builders. Phase 3 must not introduce per-phase rendering variants. |
</phase_requirements>

## Standard Stack

No new dependencies. Phase 3 uses what is already in `pyproject.toml`:

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `claude-agent-sdk` | >=0.1.52 | Structured output via `output_schema` kwarg on `run_agent()` | Already the project backend; schema enforcement is a first-class SDK feature |
| `rich` | >=13.0 | Live panels, dim text, themed styles for TTT plan renderer | Already used throughout `daydream/ui.py`; `Style(dim=True)` and inline markup are the established pattern for visual de-emphasis |

**Verification:** No package upgrades needed. `claude-agent-sdk>=0.1.52` was bumped in Phase 1
(INFR-01, complete). All structured-output schema work in `daydream/phases.py` already validates
through this SDK version — confirmed by passing tests at the end of Phase 2.

## Architecture Patterns

### Recommended Project Structure (no changes)
```
daydream/
├── exploration.py     # ExplorationContext + to_prompt_section() (already exists)
├── phases.py          # Schemas + _build_*_prompt helpers (edit here)
└── ui.py              # TTT plan renderer (small visual edit)
```

### Pattern 1: Schema-Enforced Structured Output
**What:** JSON Schema dict literal passed as `output_schema` to `run_agent()`. SDK rejects responses
that fail validation, so required fields are guaranteed at the parse site.
**When to use:** Every parsed agent result. Already the established pattern in `daydream/phases.py`.
**Example (additive field, current FEEDBACK_SCHEMA shape):**
```python
# daydream/phases.py — pattern to extend, not replace
FEEDBACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "description": {"type": "string"},
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    # NEW (D-01):
                    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                    "rationale": {"type": "string"},
                },
                "required": ["id", "description", "file", "line", "confidence", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["issues"],
    "additionalProperties": False,
}
```

### Pattern 2: Prompt Builder Prepending
**What:** Each phase function has a `_build_*_prompt(...)` helper. Prompt builders prepend
`exploration_context.to_prompt_section()` when context is non-None, then add task-specific instructions.
**When to use:** All four consumer phases. Already wired (lines 421-422, 843-844, 924-925, 1070-1071
in `daydream/phases.py`).
**Example seam to edit:**
```python
# daydream/phases.py:421 (already exists, just extend with D-02/D-05/D-09 instructions)
if exploration_context is not None:
    section = exploration_context.to_prompt_section()
    if section:
        prompt_parts.append(section)
        prompt_parts.append(_confidence_and_convention_instructions())  # NEW helper
```

### Pattern 3: Co-located Helper Methods on Dataclasses
**What:** Rendering logic for a dataclass lives as a method on the dataclass. `to_prompt_section()`
on `ExplorationContext` is the canonical example.
**When to use:** Whenever a dataclass needs a single canonical text/UI rendering. D-14 mandates this.
**Anti-pattern to avoid:** Free-standing `render_exploration_context(ctx)` functions in `phases.py`
or `ui.py`. They fragment the rendering and break OUTP-02.

### Anti-Patterns to Avoid
- **Per-phase prompt rendering variants for exploration context.** D-12/D-13 forbid this. One helper.
- **Post-processing filter pass over agent output.** Convention filtering (D-04 case 1) happens
  *inside the agent's prompt*, not as a Python pass over `result["issues"]`. The agent is told to
  not propose violating fixes; the schema enforces presence of `confidence`/`rationale`.
- **Optional `confidence`/`rationale` fields with a "default to LOW" parser fallback.** Schema must
  list them in `required`. No silent defaults.
- **Fabricated references in plan steps.** D-09. Empty array > invented paths. Prompt must say so explicitly.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Confidence labeling on parsed issues | A Python post-processor that scans rationales for keywords | `output_schema` enum field with `required` | The SDK guarantees presence; keyword scanning is a fragile retrofit |
| Convention-violation filtering of reviewer suggestions | A second-pass agent or Python filter | A prompt instruction inside `_build_review_prompt` | Adding a pass doubles cost and latency for what is fundamentally a prompt rule |
| Plan-reference validation | A Python validator that walks `references[]` against the file system | Trust the schema + prompt for v1 (deferred to v2) | Out of scope per CONTEXT.md "Deferred Ideas" |
| Exploration → prompt text rendering | A new render function in `phases.py` or `ui.py` | `ExplorationContext.to_prompt_section()` | OUTP-02 / D-12 / D-13 demand a single source of truth |
| Dependency Impact section parsing | A markdown parser scanning `.review-output.md` | Prompt instruction telling the agent to write the section in a known format (D-11) | Parser fragility is exactly what D-11 calls out; agent-driven is robust |

**Key insight:** Every "Don't hand-roll" entry above is a place where the strictness mandate
("this reviews AI-generated code") could tempt us into adding a defensive Python pass. Resist.
Schema + prompt is the load-bearing pair; Python code in this phase is glue, not gatekeeping.

## Common Pitfalls

### Pitfall 1: Optional Schema Fields Defeat Enforcement
**What goes wrong:** Adding `confidence` to `properties` but forgetting to add it to `required`.
The SDK then accepts responses without the field, defeating D-01.
**Why it happens:** JSON Schema's defaults are permissive; `additionalProperties: False` does not
imply "all listed properties are required."
**How to avoid:** For every additive field, edit *both* `properties` and `required` in the same diff.
Verify by stripping the field from a fixture response and confirming the SDK raises.
**Warning signs:** A test passes that should have failed; a `result["issues"][0]["confidence"]` lookup
needs a `.get()` with a default.

### Pitfall 2: Confidence Inflation
**What goes wrong:** The agent labels everything HIGH because the prompt language is too soft.
**Why it happens:** LLMs default to confident phrasing; "verified" is fuzzier than "cited a specific
ExplorationContext entry."
**How to avoid:** D-02/D-03 are explicit. The prompt MUST require the rationale to *quote or name
the specific exploration entry* for HIGH (e.g., "verified by Dependency edge `foo.py imports bar.py`")
and MUST require LOW for any diff-only inference. Provide a worked example in the prompt.
**Warning signs:** Test fixture reviews where 90%+ of issues are HIGH; rationales that say "verified
by exploration" without naming a specific file/symbol/convention.

### Pitfall 3: Conflating the Two Convention Cases
**What goes wrong:** A reviewed-code convention violation gets filtered out (treated as case 1)
instead of elevated to a HIGH issue (case 2). Real bugs in AI-generated code disappear.
**Why it happens:** D-04 is a non-obvious distinction; LLMs default to being polite about
existing code.
**How to avoid:** D-05's prompt language must include a worked example of *each* case side-by-side.
The phrase "If your fix would violate a convention, drop it. If the reviewed code violates a
convention, flag it as HIGH" needs to appear verbatim.
**Warning signs:** Test fixtures where the reviewed code violates a known convention but the agent
produces zero HIGH issues citing that convention.

### Pitfall 4: Fabricated `references[]` in Plan Steps
**What goes wrong:** Agent invents file paths or symbol names that look plausible but don't exist.
**Why it happens:** LLMs hallucinate when the schema requires an array and they have no grounding.
**How to avoid:** D-09 — schema allows `references` to be `[]`, and prompt explicitly says "empty
is preferred over invented." The TTT renderer flags empty arrays (D-08), making LOW-grounded steps
visible to the user rather than punishing the agent for honesty.
**Warning signs:** Plan steps citing files not present in `ExplorationContext.affected_files`;
symbols not present in any dependency edge.

### Pitfall 5: Missing Prompt Update on One of Four Phase Builders
**What goes wrong:** `phase_review` gets the new instructions but `phase_alternative_review` doesn't,
breaking OUTP-02 by producing inconsistent output across flows.
**Why it happens:** Four near-identical edit sites; easy to miss one in review.
**How to avoid:** Plan tasks must enumerate all four `_build_*_prompt` helpers explicitly. Consider
extracting the new prompt language into shared helper functions
(e.g., `_confidence_and_convention_instructions()`, `_plan_grounding_instructions()`) so the four
sites just call them. Then a grep for the helper name confirms all four call sites.
**Warning signs:** Tests pass for `phase_review` but `phase_alternative_review` produces issues
without `confidence` (or vice versa).

## Code Examples

### Extending FEEDBACK_SCHEMA (D-01)
```python
# daydream/phases.py — additive edit, lines 105-125
FEEDBACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "description": {"type": "string"},
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                    "rationale": {"type": "string"},
                },
                "required": ["id", "description", "file", "line", "confidence", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["issues"],
    "additionalProperties": False,
}
```

### Extending PLAN_SCHEMA with references[] (D-07)
```python
# daydream/phases.py — additive edit inside the change-item shape
"changes": {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "file": {"type": "string"},
            "description": {"type": "string"},
            "action": {"type": "string", "enum": ["modify", "create", "delete"]},
            "references": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "symbol": {"type": "string"},
                    },
                    "required": ["file", "symbol"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["file", "description", "action", "references"],
        "additionalProperties": False,
    },
},
```
Note: `references` is required, but the array MAY be empty (D-08/D-09).

### Shared Prompt Instruction Helpers (avoids Pitfall 5)
```python
# daydream/phases.py — new module-level helpers to be called from all four builders

def _confidence_and_convention_instructions() -> str:
    """Prompt language for QUAL-02 confidence labeling and QUAL-03 convention handling.

    Called by _build_review_prompt, _build_understand_intent_prompt,
    _build_alternative_review_prompt, _build_plan_prompt to keep all four phases
    in lockstep on these rules.
    """
    return (
        "## Confidence and Convention Rules\n\n"
        "For every issue you report, you MUST set `confidence` and `rationale`:\n"
        "- HIGH: directly verified by a specific entry in the Exploration Context above. "
        "  Your rationale MUST name the specific Dependency edge, Convention entry, or "
        "  affected file that supports the issue.\n"
        "- MEDIUM: consistent with the Exploration Context but not pinned to a specific entry.\n"
        "- LOW: inferred from the diff alone, no exploration evidence. Your rationale MUST "
        "  state 'no exploration evidence'.\n\n"
        "Convention handling has TWO distinct cases — do not conflate them:\n"
        "1. Before proposing a fix, check it against the Codebase Conventions section. "
        "   If your fix would violate a convention, DROP IT — do not include it.\n"
        "2. If the reviewed code itself violates a convention, that IS the issue. Flag it "
        "   as HIGH confidence and cite the convention by name in `rationale`.\n\n"
        "You are reviewing AI-generated code. Be strict. Prefer LOW over MEDIUM when uncertain."
    )


def _plan_grounding_instructions() -> str:
    """Prompt language for OUTP-01 plan reference grounding."""
    return (
        "## Plan Reference Grounding\n\n"
        "For every change in the plan, populate `references` with `{file, symbol}` entries "
        "drawn ONLY from the Exploration Context above. Do not invent file paths or symbol "
        "names. If you cannot ground a step in the Exploration Context, leave `references` "
        "as an empty array — the renderer will flag ungrounded steps for the user."
    )
```

### TTT Plan Renderer Visual Distinction (D-08)
```python
# daydream/ui.py — illustrative pattern; exact integration depends on existing render code
# Look for the function that renders plan changes; the seam is per-change-item.

def _render_plan_change(change: dict[str, Any], console: Console) -> None:
    is_grounded = bool(change.get("references"))
    style = "" if is_grounded else "dim"
    marker = "" if is_grounded else " [yellow](ungrounded)[/yellow]"
    console.print(f"[{style}]- {change['file']}: {change['description']}{marker}[/{style}]")
    for ref in change.get("references", []):
        console.print(f"    [dim]→ {ref['file']}::{ref['symbol']}[/dim]")
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ with pytest-asyncio 0.24+ |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`, `asyncio_mode = "auto"`) |
| Quick run command | `uv run pytest tests/test_phases.py -x` |
| Full suite command | `uv run pytest -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| QUAL-02 | `FEEDBACK_SCHEMA` requires `confidence` enum and `rationale` string | unit | `uv run pytest tests/test_phases.py::test_feedback_schema_requires_confidence_and_rationale -x` | Wave 0 |
| QUAL-02 | `ALTERNATIVE_REVIEW_SCHEMA` requires `confidence` enum and `rationale` string | unit | `uv run pytest tests/test_phases.py::test_alternative_review_schema_requires_confidence_and_rationale -x` | Wave 0 |
| QUAL-02 | Parsing rejects responses missing `confidence` or `rationale` | unit | `uv run pytest tests/test_phases.py::test_parse_feedback_rejects_unlabeled -x` | Wave 0 |
| QUAL-01 | `phase_review` prompt mentions Dependency Impact and exploration dependencies | unit | `uv run pytest tests/test_phases.py::test_review_prompt_includes_dependency_impact -x` | Wave 0 |
| QUAL-03 | Review prompt distinguishes both convention cases (filter fix vs. flag reviewed code) | unit | `uv run pytest tests/test_phases.py::test_review_prompt_distinguishes_convention_cases -x` | Wave 0 |
| OUTP-01 | `PLAN_SCHEMA` requires `references[]` per change with `{file, symbol}` items | unit | `uv run pytest tests/test_phases.py::test_plan_schema_requires_references -x` | Wave 0 |
| OUTP-01 | Plan prompt instructs against fabricated references | unit | `uv run pytest tests/test_phases.py::test_plan_prompt_forbids_fabrication -x` | Wave 0 |
| OUTP-02 | All four `_build_*_prompt` helpers prepend `to_prompt_section()` when context non-None | unit | `uv run pytest tests/test_phases.py::test_all_phase_builders_inject_exploration -x` | Partially exists (extend) |
| OUTP-02 | All four `_build_*_prompt` helpers include the shared confidence/convention instructions | unit | `uv run pytest tests/test_phases.py::test_all_phase_builders_use_shared_instructions -x` | Wave 0 |
| OUTP-01 | TTT plan renderer visually marks empty-references steps | unit | `uv run pytest tests/test_ui.py::test_plan_renderer_dims_ungrounded_steps -x` | Wave 0 |
| OUTP-02 | End-to-end: MockBackend returns labeled issues for both `--ttt` and normal flows | integration | `uv run pytest tests/test_integration.py::test_exploration_enriched_output_both_flows -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_phases.py -x`
- **Per wave merge:** `uv run pytest -v`
- **Phase gate:** Full suite green + `make check` (lint + typecheck + tests) before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_phases.py` — add schema-shape tests for all three extended schemas (new test cases in the existing file)
- [ ] `tests/test_phases.py` — add prompt-content tests for all four `_build_*_prompt` helpers (assert key phrases present)
- [ ] `tests/test_ui.py` — add or extend test for TTT plan renderer visual distinction on empty `references[]`
- [ ] `tests/test_integration.py` — extend an existing MockBackend-based integration test (or add new) to assert both `--ttt` and normal flow consume the new schema fields end-to-end
- [ ] Test fixture: a small `ExplorationContext` instance with a Dependency, a Convention, and an affected file — reusable across the new tests (consider a pytest fixture in `tests/conftest.py`)

## Sources

### Primary (HIGH confidence)
- `.planning/phases/03-review-integration/03-CONTEXT.md` — User-locked decisions D-01 through D-14
- `.planning/REQUIREMENTS.md` — QUAL-01/02/03, OUTP-01/02 definitions
- `.planning/ROADMAP.md` — Phase 3 success criteria (5 items)
- `daydream/exploration.py` (read in full) — `ExplorationContext`, `to_prompt_section()` already implemented; data model is stable
- `daydream/phases.py` (lines 105-190 read; grep results for all relevant symbols) — `FEEDBACK_SCHEMA`, `ALTERNATIVE_REVIEW_SCHEMA`, `PLAN_SCHEMA` shapes verified; exploration-context injection already wired at lines 421-422, 843-844, 924-925, 1070-1071

### Secondary (MEDIUM confidence)
- Project `CLAUDE.md` (root) — schema/prompt conventions, mypy/ruff strictness, pattern of `_build_*_prompt` helpers
- `.planning/STATE.md` — confirms Phase 1 + 2 complete, exploration data plumbed end-to-end

### Tertiary (LOW confidence)
- None. All findings verified against source files and locked CONTEXT.md decisions.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; existing tooling verified in `pyproject.toml` and Phase 1/2 history
- Architecture: HIGH — every integration point exists in code (verified by direct read + grep of `daydream/phases.py`)
- Pitfalls: HIGH for Pitfalls 1, 3, 4, 5 (mechanically verifiable); MEDIUM for Pitfall 2 (confidence inflation depends on LLM behavior, only mitigatable in prompt language)

**Research date:** 2026-04-07
**Valid until:** 2026-05-07 (30 days — internal schema/prompt work, no fast-moving external surfaces)
