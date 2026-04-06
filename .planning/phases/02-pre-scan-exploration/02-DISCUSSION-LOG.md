# Phase 2: Pre-scan Exploration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 02-pre-scan-exploration
**Areas discussed:** Subagent roles & responsibilities, Diff analysis & file discovery, Scaling thresholds, Flow integration & UX

---

## Subagent roles & responsibilities

### Q1: How specialized should each subagent be?

| Option | Description | Selected |
|--------|-------------|----------|
| Distinct specialists | Focused prompts, each fills specific ExplorationContext fields. Pattern-scanner → conventions, dependency-tracer → deps + affected_files, test-mapper → tests. | ✓ |
| General explorers with focus hints | Same base prompt + focus-area parameter. More overlap, catches things specialists miss. | |
| Two agents instead of three | Merge pattern-scanner + test-mapper into one 'context agent', keep dependency-tracer separate. | |

**User's choice:** Distinct specialists (recommended)
**Notes:** Matches the three names already in the roadmap success criteria.

### Q2: Should subagents read project guidelines themselves or via a pre-extract step?

| Option | Description | Selected |
|--------|-------------|----------|
| Subagents read directly | Each subagent reads CLAUDE.md + configs as part of its own exploration. No sequential pre-step, parallel launch immediate. | ✓ |
| Pre-extract and inject | Lightweight pre-step reads guideline files and injects into all subagent prompts. Consistent but adds sequential latency. | |

**User's choice:** Subagents read directly (recommended)
**Notes:** User asked which approach is more performant. Explanation: pre-extract adds wall-clock latency before parallel launch; redundant file I/O cost is trivial vs LLM calls. Direct-read wins on performance.

### Q3: How should subagent results be collected and merged into ExplorationContext?

| Option | Description | Selected |
|--------|-------------|----------|
| Structured output per agent | Each subagent returns JSON matching its ExplorationContext fields; merge function combines into single context. Type-safe, testable. | ✓ |
| Free-text with post-parse | Subagents return markdown; post-processing parses into fields. Fragile. | |
| Direct ExplorationContext population | Pass shared instance, agents append via tool calls. Requires tool infra for mutation. | |

**User's choice:** Structured output per agent (recommended)

---

## Diff analysis & file discovery

### Q1: How should import/dependency tracing work across languages?

| Option | Description | Selected |
|--------|-------------|----------|
| Regex-based, language-agnostic | Regex patterns for imports across Python/JS/TS/Go/Rust/Elixir. Fast, no deps, graceful fallback. | |
| Let the subagent figure it out | detect_affected_files() identifies changed files only; dependency-tracer subagent does tracing via its tools. | |
| AST-based with tree-sitter | Tree-sitter for precise AST parsing. Accurate, handles aliased/dynamic imports. Native dependency added. | ✓ |

**User's choice:** AST-based with tree-sitter
**Notes:** User was explicit: "I am worried about reliability of regex." Tree-sitter is a hard requirement.

### Q2: Which languages should tree-sitter support out of the box?

| Option | Description | Selected |
|--------|-------------|----------|
| Python + TS/JS only | Minimal initial dep footprint. | |
| Python + TS/JS + Go + Rust | Broader coverage, systems languages. | ✓ |
| All tree-sitter-languages bundle | 30+ grammars bundled. Maximum coverage, lots of native code. | |

**User's choice:** Python + TS/JS + Go + Rust
**Notes:** Elixir and Swift are "coming soon" per user; treat as future additions, not part of Phase 2 scope. Tree-sitter integration should make adding a grammar trivial (one-line registration).

### Q3: For files in unsupported languages, how should detect_affected_files() handle them?

User asked a clarifying question first: "When would this happen?" Clarification provided:
- Config/data/docs files (YAML, TOML, JSON, Dockerfile, Markdown) — no imports to trace anyway
- Other languages (Elixir, Ruby, Java, Swift, etc.) — plausible given Beagle's stack coverage

Question reframed with context. User's concern: regex fallback would be unreliable.

| Option | Description | Selected |
|--------|-------------|----------|
| No tracing, hand to subagent | Add to affected_files with role='modified', no dep tracing. Dependency-tracer subagent investigates via grep/read. | ✓ |
| Regex fallback anyway | Accept unreliability, ship regex patterns. | |
| Skip entirely with warning | Log warning, exclude from affected_files. | |

**User's choice:** No tracing, hand to subagent (recommended)
**Notes:** User explicitly rejected regex fallback over reliability concerns.

### Q4: How deep should tree-sitter import tracing go for supported languages?

| Option | Description | Selected |
|--------|-------------|----------|
| One hop | Direct imports + direct importers. Matches EXPL-02. Bounded. | |
| Two hops | Transitive imports. Explodes on hub files. | |
| Configurable, default=1 | RunConfig parameter, default=1, tunable per project. | ✓ |

**User's choice:** Configurable, default=1

---

## Scaling thresholds

### Q1: What metric should define diff size tiers?

| Option | Description | Selected |
|--------|-------------|----------|
| Changed file count | Simple, predictable, human-intuitive. | ✓ |
| Total lines changed | More granular but misleading for formatting changes. | |
| Hybrid (files + lines) | More accurate, more tunable knobs. | |

**User's choice:** Changed file count (recommended)

### Q2: What should the tier thresholds be?

| Option | Description | Selected |
|--------|-------------|----------|
| Skip ≤1, single 2–3, parallel ≥4 | Conservative, errs toward exploration. | ✓ |
| Skip ≤2, single 3–5, parallel ≥6 | More aggressive skipping, saves tokens on small fixes. | |
| Always parallel, never skip | Simplest, most expensive, most consistent. | |

**User's choice:** Skip ≤1, single 2–3, parallel ≥4 (recommended)

### Q3: When the 'single agent' tier fires, which subagent runs?

| Option | Description | Selected |
|--------|-------------|----------|
| Dependency-tracer | Highest-value signal on small diffs: "what else does this touch?" | ✓ |
| Pattern-scanner | Conventions matter most for small fixes. | |
| All three, sequentially | Full context but slower; negates the point of a lighter tier. | |

**User's choice:** Dependency-tracer (recommended)

---

## Flow integration & UX

### Q1: Which flows should run pre-scan exploration in Phase 2?

| Option | Description | Selected |
|--------|-------------|----------|
| Both --ttt and normal | Unified from day one. Satisfies OUTP-02. | ✓ |
| --ttt only for now | Smaller blast radius, extend to normal in Phase 3. | |
| Normal flow only | Primary flow first, defer --ttt. | |

**User's choice:** Both --ttt and normal (recommended)

### Q2: Where in run_trust() should exploration happen?

| Option | Description | Selected |
|--------|-------------|----------|
| Before phase_understand_intent | Maximizes value, even intent benefits from codebase awareness. | ✓ |
| Between intent and alternative_review | Intent stays lean. | |
| Parallel with phase_understand_intent | Minimizes wall-clock, adds coordination complexity. | |

**User's choice:** Before phase_understand_intent (recommended)

### Q3: What should the user see during exploration?

| Option | Description | Selected |
|--------|-------------|----------|
| Live panel with per-agent status | LiveToolPanelRegistry pattern; each subagent as a live row. | ✓ |
| Simple phase hero + spinner | Single spinner, less detail. | |
| Silent with summary at end | Minimal noise, no progress feedback. | |

**User's choice:** Live panel with per-agent status (recommended)

### Q4: How should exploration results be passed to downstream review phases?

| Option | Description | Selected |
|--------|-------------|----------|
| RunConfig field | exploration_context on RunConfig, read by phase functions. Fits existing pattern. | ✓ |
| Separate parameter threaded through phases | Verbose but data flow visible at call sites. | |
| Module-level state via AgentState | Hides data flow; consistent with debug_log/model. | |

**User's choice:** RunConfig field (recommended)

---

## Claude's Discretion

- Exact subagent prompts (pattern-scanner, dependency-tracer, test-mapper)
- JSON output schemas for each subagent
- Internal merge-function structure
- Tree-sitter integration details (binding choice, grammar loading, parser caching)
- `detect_affected_files()` signature beyond the core contract
- Live panel row labels and intermediate status text

## Deferred Ideas

- Global exploration force-on/off flag
- Two-hop or deeper default tracing
- Elixir + Swift tree-sitter grammars
- Token budgets for subagents
- Exploration result caching (already v2: ADVX-01)
- Confidence scores, convention filtering, grounded recommendations (Phase 3)
