# Phase 2: Pre-scan Exploration - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Parallel subagents explore affected codebase areas before review starts. This phase builds the actual exploration logic on top of Phase 1's infrastructure: `detect_affected_files()` for impact surface detection, three specialized subagents (pattern-scanner, dependency-tracer, test-mapper) that run in parallel via `Backend.execute(agents=...)`, diff-size-based scaling, and wiring into both `--ttt` and normal review flows. Downstream consumption of the resulting `ExplorationContext` (grounded recommendations, confidence scores, convention filtering) is Phase 3's job — Phase 2 delivers the context; Phase 3 uses it.

</domain>

<decisions>
## Implementation Decisions

### Subagent Design
- **D-01:** Three distinct specialist subagents, each with a focused prompt and clear ownership of `ExplorationContext` fields:
  - **pattern-scanner** → `conventions` + `guidelines` (reads CLAUDE.md, .coderabbit.yaml, house-style configs, infers conventions from code)
  - **dependency-tracer** → `dependencies` + contributes to `affected_files` (call-chain and import tracing beyond the initial impact surface)
  - **test-mapper** → contributes to `affected_files` with `role="test"` (maps changed files to their test coverage)
- **D-02:** Subagents read project guidelines directly as part of their own exploration — no pre-extraction step. Parallel launch with no sequential bottleneck beats redundant file-read cost. Pattern-scanner is the primary owner of guideline reading (EXPL-04).
- **D-03:** Each subagent returns structured JSON matching its `ExplorationContext` fields. A merge function in `daydream/exploration.py` combines the three results into a single `ExplorationContext`. Type-safe and testable; avoids fragile text parsing.

### Diff Analysis & File Discovery
- **D-04:** `detect_affected_files()` uses **tree-sitter** for AST-based import/dependency parsing. Regex-based parsing is explicitly rejected as too unreliable (comments, strings, multi-line imports, conditional requires break naive patterns).
- **D-05:** Initial language support: **Python, TypeScript/JavaScript, Go, Rust**. Elixir and Swift are on the near-term roadmap — add grammars when those stacks come online. Do not pull in the full `tree-sitter-languages` bundle.
- **D-06:** For files in unsupported languages, `detect_affected_files()` adds them to `affected_files` with `role="modified"` but performs no dependency tracing. The dependency-tracer subagent investigates these files manually via its own tools (grep, read). Graceful degradation without fragile regex.
- **D-07:** Import tracing depth is **configurable via `RunConfig`, default = 1** (one hop — direct imports and direct importers). Matches EXPL-02's "immediate imports/callers" language while letting power users tune for specific projects.

### Scaling Thresholds
- **D-08:** Tier metric is **changed file count** from the diff. Simple, predictable, matches how humans reason about PR size. Line count is rejected as a primary metric (500-line formatting change ≠ 500-line refactor).
- **D-09:** Tier boundaries:
  - **Skip** (0–1 changed files): No exploration; empty `ExplorationContext`
  - **Single** (2–3 changed files): Run **dependency-tracer only** — the highest-value signal on small diffs is "what else does this touch?"
  - **Parallel** (4+ changed files): Run all three subagents in parallel via `Backend.execute(agents=[...])`
- **D-10:** No global kill-switch flag in this phase. Scaling is automatic based on diff size. If users want to force exploration on/off, that's a future RunConfig addition.

### Flow Integration
- **D-11:** Exploration wires into **both `--ttt` and normal review flows** in this phase (satisfies OUTP-02 from the start rather than deferring). Unified architecture from day one avoids a second integration pass.
- **D-12:** In `run_trust()`, exploration runs **before `phase_understand_intent`**. All three downstream phases (intent, alternative review, plan generation) receive the `ExplorationContext`. Even the intent summary benefits from codebase awareness.
- **D-13:** In `run()` (normal flow), exploration runs **before `phase_review`**. The review agent receives the context via its prompt injection path.
- **D-14:** `ExplorationContext` is stored on **`RunConfig`** as `exploration_context: ExplorationContext | None = None`. Populated by the pre-scan step, read by downstream phase functions. Follows the existing config-passing pattern; avoids hidden state on `AgentState`.

### UX
- **D-15:** Exploration uses a **live panel with per-subagent status** via the existing `LiveToolPanelRegistry` pattern in `daydream/ui.py`. Each subagent (pattern-scanner, dependency-tracer, test-mapper) appears as a live row with throbber → done state. Consistent with how tool calls render today.
- **D-16:** A dedicated **"EXPLORE" phase hero** (via `print_phase_hero()`) announces the exploration step before the live panel activates. Maintains the existing phase-hero visual rhythm.
- **D-17:** When exploration is skipped (single file or zero), print a brief dim-text notice (`"Skipping exploration — trivial diff"`) instead of the hero + panel. No wasted screen real estate.

### Claude's Discretion
- Exact prompt text for each specialist subagent (pattern-scanner, dependency-tracer, test-mapper) — Claude designs during planning to match the structured output contract.
- JSON schemas for each subagent's structured output — Claude derives from the `ExplorationContext` field types during planning.
- Internal structure of the merge function — whatever produces a well-formed `ExplorationContext`.
- Tree-sitter integration details: which Python binding to use (`tree-sitter` + per-language packages vs. `py-tree-sitter-languages` scoped subset), how grammars are loaded, caching of parsers.
- `detect_affected_files()` function signature and return shape beyond "list of files with role + dependency edges" — Claude picks what fits the existing module layout.
- How the live panel's per-subagent rows are labeled and what intermediate status text shows (e.g. "reading CLAUDE.md...", "tracing imports...").

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 1 Foundation (MUST read — this phase builds directly on it)
- `daydream/exploration.py` — `ExplorationContext`, `FileInfo`, `Convention`, `Dependency`, `safe_explore()` — all the data structures and degradation infra this phase populates
- `daydream/backends/__init__.py` — `Backend` protocol with the `agents` kwarg added in Phase 1
- `daydream/backends/claude.py` — `ClaudeBackend.execute()` — how `AgentDefinition` is passed through to `ClaudeSDKClient`
- `.planning/phases/01-exploration-infrastructure/01-CONTEXT.md` — Phase 1 decisions that lock the foundation this phase extends

### Flow Integration Targets
- `daydream/runner.py` — `run()` and `run_trust()` — where exploration gets wired in (D-12, D-13)
- `daydream/phases.py` — `phase_understand_intent`, `phase_alternative_review`, `phase_generate_plan`, `phase_review` — downstream consumers that receive `ExplorationContext` via `RunConfig`
- `daydream/agent.py` — `run_agent()` — the wrapper around `Backend.execute()` that will pass `agents` through

### UI Patterns
- `daydream/ui.py` — `LiveToolPanelRegistry`, `print_phase_hero`, `print_dim` — the rendering primitives the exploration UX reuses (D-15, D-16, D-17)

### Requirements & Roadmap
- `.planning/REQUIREMENTS.md` — EXPL-01, EXPL-02, EXPL-03, EXPL-04, AGNT-01 definitions
- `.planning/ROADMAP.md` — Phase 2 goal + success criteria (the five bullets this phase must satisfy)

### Codebase Maps
- `.planning/codebase/ARCHITECTURE.md` — Layer responsibilities, data flow, existing patterns
- `.planning/codebase/CONVENTIONS.md` — Established code style and patterns to respect
- `.planning/codebase/STACK.md` — Dependency policy, SDK version context

### External (tree-sitter)
- No local specs — researcher should fetch current tree-sitter Python binding documentation and grammar package status for Python/TS-JS/Go/Rust during the research phase

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ExplorationContext`, `FileInfo`, `Convention`, `Dependency` — typed data model ready to be populated (Phase 1)
- `safe_explore()` — graceful-degradation wrapper; wrap the pre-scan step in this (Phase 1, D-06)
- `Backend.execute(agents=...)` — already accepts `AgentDefinition` list (Phase 1, D-01)
- `LiveToolPanelRegistry` in `daydream/ui.py` — live multi-row panel pattern for concurrent tool calls; reuse for per-subagent status rows
- `print_phase_hero()` — existing phase-hero renderer for the "EXPLORE" phase marker
- `RunConfig` dataclass in `daydream/runner.py` — add `exploration_context` field here
- `_git_diff()` helper in `runner.py` — already used by `run_trust` to build diff input; reuse for file-count tier detection and for `detect_affected_files()`

### Established Patterns
- `@dataclass` for all data types — new helper types in `exploration.py` follow this
- `from __future__ import annotations` + `TYPE_CHECKING` guard — use for tree-sitter type imports that may be expensive to load
- Google-style docstrings with `Args:` / `Returns:` / `Raises:` on public functions
- Phase functions take `backend: Backend` as first parameter — any new phase helper follows this
- Structured JSON output via schemas — `FEEDBACK_SCHEMA`, `ALTERNATIVE_REVIEW_SCHEMA`, `PLAN_SCHEMA` already exist in phases; new subagent output schemas live alongside them

### Integration Points
- `RunConfig` in `runner.py` → add `exploration_context: ExplorationContext | None = None`
- `run()` in `runner.py` → call pre-scan before `phase_review()`
- `run_trust()` in `runner.py` → call pre-scan before `phase_understand_intent()`
- `phase_review`, `phase_understand_intent`, `phase_alternative_review`, `phase_generate_plan` → accept and inject `ExplorationContext.to_prompt_section()` into their prompts (actual consumption may partly bleed into Phase 3 depending on how thin Phase 2 keeps it)
- New module or section for `detect_affected_files()` + tier logic + subagent orchestration — likely new file `daydream/exploration_runner.py` (or extended `exploration.py`, Claude's discretion)
- `pyproject.toml` → add tree-sitter + language grammar dependencies

</code_context>

<specifics>
## Specific Ideas

- Phase 2's three subagents (pattern-scanner, dependency-tracer, test-mapper) are named explicitly in the roadmap success criteria — these names should appear in code (as `AgentDefinition` identifiers) and in the UI rows.
- User was explicit about regex unreliability: "I am worried about reliability of regex" — tree-sitter is a hard requirement, not a negotiable.
- Elixir and Swift support is "coming soon" per user — plan tree-sitter integration so adding a grammar is a one-line registration, not a refactor.
- Default import-trace depth = 1 is locked, but the `RunConfig` parameter is required so it can be tuned without code changes.

</specifics>

<deferred>
## Deferred Ideas

- **Global exploration force-on/off flag** — Deferred. Automatic scaling by file count is sufficient for v1. Revisit if users report wanting to override.
- **Two-hop or configurable deeper import tracing as default** — Deferred. Default is 1 hop; users can tune via RunConfig. Raising the default is a v2 consideration.
- **Elixir + Swift tree-sitter grammars** — Noted as "coming soon" but out of scope for this phase. Register them when those stacks are live.
- **Token budgets for subagents** — Deferred. Per D-07 from Phase 1, no artificial timeouts; per this phase, no token caps. Revisit if cost becomes a problem.
- **Exploration result caching** — Already marked as v2 (ADVX-01 in REQUIREMENTS.md). Not revisited.
- **Consumption details (confidence scores, convention filtering, grounded recommendations)** — These are Phase 3's job per the roadmap. Phase 2 produces the context; Phase 3 uses it.

</deferred>

---

*Phase: 02-pre-scan-exploration*
*Context gathered: 2026-04-06*
