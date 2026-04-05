# Project Research Summary

**Project:** Daydream — Subagent Exploration Layer
**Domain:** AI-powered code review with codebase-aware subagent orchestration
**Researched:** 2026-04-05
**Confidence:** HIGH

## Executive Summary

Daydream is adding a subagent-powered codebase exploration layer to an existing Python CLI code review tool. The research confirms a clear and low-risk implementation path: the Claude Agent SDK's native `AgentDefinition` and `agents` parameter on `ClaudeAgentOptions` provide exactly the primitives needed without any new dependencies. The existing backend abstraction requires a single additive parameter (`agents: dict[str, AgentDefinition] | None`), and the async concurrency infrastructure (anyio task groups, capacity limiters) is already in place. This is an additive milestone — no rewrites, no new packages, no new frameworks.

The recommended approach is a phased exploration architecture built in six implementation steps: extend the Backend protocol, build affected-file detection and data structures, create the agent definition factory, implement `phase_explore()`, wire exploration context into downstream review phases, and add CLI flags. Three specialized read-only Sonnet subagents (pattern-scanner, dependency-tracer, test-mapper) running in parallel provide the performance and quality benefits. Exploration results are structured via `output_schema`, synthesized into a compact `ExplorationContext`, and injected into review prompts as grounded context rather than diff-only analysis.

The primary risk is unbounded exploration: subagents following import chains can spiral into reading entire codebases, producing exponential token costs and degraded review quality from context overload. The mitigation is non-negotiable — depth limits, per-subagent file caps, token budget monitoring, and a synthesis step before handing context to the review agent must be built before any exploration logic. The secondary architectural risk is letting Claude-specific subagent details leak through the Backend protocol; a clean `agents` parameter and a `supports_subagents` capability flag keep the Codex backend viable.

## Key Findings

### Recommended Stack

No new dependencies are required. The entire exploration layer is built on `claude-agent-sdk` (minimum version bumped from `>=0.1.27` to `>=0.1.52` for `AgentDefinition` support) and `anyio` (already present). The project should stay with `ClaudeSDKClient`-based streaming rather than migrating to the `query()` API — the session-based approach integrates cleanly with the existing event stream and UI. Third-party multi-agent frameworks (LangGraph, CrewAI, AutoGen) are explicitly ruled out: they add massive dependency bloat for capabilities the SDK provides natively.

**Core technologies:**
- `claude-agent-sdk >= 0.1.52`: `AgentDefinition` + `agents` param on `ClaudeAgentOptions` — native subagent support, version floor ensures `disallowedTools`, `maxTurns`, and deadlock fixes are present
- `anyio.create_task_group()`: parallel subagent fan-out — already used for `phase_fix_parallel()`, same pattern applies
- `anyio.CapacityLimiter`: throttle concurrent subagent count — already in use with limiter of 4, exploration needs 2-4
- Subagent model strategy: Opus for review/fix/plan agents, Sonnet for exploration subagents (5x cheaper, sufficient for read-only file scanning)

### Expected Features

**Must have (table stakes):**
- Impact surface mapping — git diff to affected files to exploration scope; without this, reviews are shallow diff-only analysis
- Diff-adjacent file reading — files touched by the diff plus immediate imports/callers; industry baseline since GitHub Copilot's March 2026 agentic overhaul
- Cross-file dependency tracing — follow call chains to catch breaking changes; Greptile's and CodeRabbit's core value prop
- Convention/pattern detection — discover how the codebase does things before recommending alternatives; top source of false positives when missing
- Repository/project guidelines integration — exploration subagents must read CLAUDE.md and Beagle skill configs before exploring

**Should have (competitive):**
- Parallel pre-scan subagents — same pattern Claude Code itself uses internally; turns 60-second sequential scan into 15-second parallel scan
- On-demand deep-dive spawning — review agent spawns a focused explorer when it hits uncertainty; unique advantage of the Agent SDK approach
- Confidence scoring per recommendation — honest about what was verified vs. guessed; industry true-positive rate is ~45%, transparency builds trust
- Convention contradiction filtering — post-review filter suppressing recommendations that contradict established codebase patterns

**Defer (v2+):**
- Tribal knowledge capture — requires a working feedback loop; build review quality first, then add learning
- Exploration-informed plan generation (TTT) — wire into TTT after exploration is proven in normal review flow
- Full convention contradiction filter — start by injecting conventions into the review prompt; build the post-filter only if false positives remain high
- Resumable subagent sessions — one-shot exploration sufficient for Phase 1; session resumption adds complexity for marginal gain

**Anti-features (explicitly do not build):**
- Full codebase indexing/embedding — the agent has direct file access; vector DBs are server-side infrastructure overkill for a CLI
- Style/formatting nitpicks — linters handle this; prompts should explicitly exclude cosmetic issues
- Comment volume maximization — 3.6 actionable comments per PR is the target, not 20 generic observations
- Custom orchestration framework — SDK handles lifecycle, context isolation, and tool delegation natively

### Architecture Approach

The SDK's `Agent` tool is the orchestration mechanism. Daydream's job is to configure `AgentDefinition` instances, pass them through `Backend.execute()` to `ClaudeAgentOptions`, and write prompts that tell the main agent when and how to spawn explorers. A new `phase_explore()` function runs before `phase_review()`, using three focused read-only Sonnet subagents in parallel. Results are parsed into an `ExplorationContext` dataclass and injected as structured text into downstream phase prompts. Exploration is a non-blocking enrichment: if it fails or times out, review proceeds without it.

**Major components:**
1. `ExplorationConfig` (new dataclass) — holds `AgentDefinition` instances, prompt templates, affected-file detection settings; consumed by `runner.py` and `phases.py`
2. `phase_explore()` (new phase function) — pre-scan phase; runs the main explore agent which spawns three subagents; returns `ExplorationContext`; called before `phase_review()` in both review and TTT flows
3. `ExplorationContext` (new dataclass) — structured output: `file_map`, `conventions`, `dependencies`, `test_files`, `architecture_notes`, `summary`; injected into downstream prompts
4. `detect_affected_files()` (new utility) — git diff (staged + unstaged) to file list; starting point for subagent exploration scope
5. Agent definition factory `create_exploration_agents()` — produces three `AgentDefinition` instances: `pattern-scanner`, `dependency-tracer`, `test-mapper`; all Sonnet, all `["Read", "Grep", "Glob"]` only
6. Backend protocol extension — `agents: dict[str, AgentDefinition] | None` added to `execute()`; Codex backend ignores it or delegates to a temporary Claude backend for exploration

### Critical Pitfalls

1. **Unbounded exploration** — subagents follow import chains infinitely, producing $10-50/review costs and context overload. Prevention: hard cap at 2 dependency levels, 15-20 files per subagent, $0.50/100K token abort threshold, 60-second time box, explicit directory bounds in subagent prompts. Must be implemented before any exploration code.

2. **Context overload in parent agent** — three subagents returning 2,000-5,000 token summaries each, combined with diff and project context, can push the review agent past 100K tokens, triggering the 200K surcharge tier and degrading review quality. Prevention: cap subagent summaries at 1,000 tokens, add a synthesis step (Haiku/Sonnet) that compresses exploration into a 1,500-2,000 token briefing before the review agent receives it.

3. **Backend protocol leak** — writing `isinstance(backend, ClaudeBackend)` checks in `phases.py`. Prevention: `agents` parameter on `execute()` with a `supports_subagents: bool` capability flag; subagent definitions stay in orchestration layer, not inside the backend.

4. **Conflicting exploration findings** — parallel subagents examining overlapping areas return contradictory conclusions (accounts for 36.9% of multi-agent system failures). Prevention: partition by concern, not file area (one for architecture patterns, one for test conventions, one for error handling); include reconciliation prompt in synthesis step.

5. **"Agent" vs "Task" tool name ambiguity** — the SDK renamed the tool from "Task" to "Agent" in Claude Code v2.1.63 but still uses "Task" in `system:init` tool lists. Prevention: check for both names in all tool name comparisons from day one.

## Implications for Roadmap

Based on research, the build order is dictated by hard dependencies. Safeguards must precede exploration logic. Protocol extensions must precede phase implementation. Phase integration must precede CLI flags.

### Phase 1: Exploration Foundation
**Rationale:** Everything depends on this. The budget/safety system must exist before any exploration code runs. The Backend protocol extension is the prerequisite for all subsequent phases. These are pure data structures and utility functions — low-risk, no agent interaction yet.
**Delivers:** Safe exploration infrastructure: token budget system, `Backend.execute(agents=...)` parameter, `detect_affected_files()` utility, `ExplorationContext` dataclass, `EXPLORATION_SCHEMA`, event type additions (`SubagentStartEvent`, `SubagentResultEvent`), "Agent"/"Task" dual-name detection.
**Addresses:** Impact surface mapping, repository/project guidelines integration (reading boundaries)
**Avoids:** Unbounded exploration (Pitfall 1), Backend protocol leak (Pitfall 3), Agent/Task naming bug (Pitfall 5), unstructured exploration output (Pitfall 8)
**Research flag:** Standard patterns — SDK docs are authoritative, implementation is mechanical.

### Phase 2: Exploration Phase Implementation
**Rationale:** With the protocol and data structures in place, the agent definition factory and `phase_explore()` can be built and tested in isolation before touching any existing phases.
**Delivers:** `create_exploration_agents()` factory with three Sonnet subagents (pattern-scanner, dependency-tracer, test-mapper); `phase_explore()` wired into `runner.py` as a pre-scan step; diff-size scaling logic (skip for <3 files, single agent for 3-10, parallel for 10+); synthesis/summarization step before context handoff.
**Addresses:** Parallel pre-scan subagents, diff-adjacent file reading, cross-file dependency tracing, convention/pattern detection
**Avoids:** Context overload (Pitfall 3), conflicting exploration findings (Pitfall 4), exploration latency UX (Pitfall 5)
**Research flag:** Needs real SDK testing — prompt engineering for subagent quality is iterative; mock backends can validate structure but not output quality.

### Phase 3: Review Integration
**Rationale:** With a working `ExplorationContext`, downstream phases can be modified to consume it. These are additive prompt changes — the safest type of modification.
**Delivers:** `phase_review()`, `phase_alternative_review()`, and `phase_generate_plan()` updated to accept optional `ExplorationContext` and inject conventions, dependencies, and summary into prompts; confidence scoring field added to the review output schema; exploration-as-enrichment pattern (graceful degradation if explore phase fails).
**Addresses:** Convention/pattern detection injection, confidence scoring per recommendation, exploration-informed plan generation (TTT)
**Avoids:** Making exploration mandatory (Anti-Pattern 4 from ARCHITECTURE.md), continuation token misuse (Anti-Pattern 2)
**Research flag:** Standard patterns — prompt injection is well-understood; confidence scoring schema extension is mechanical.

### Phase 4: On-Demand Exploration + UX Polish
**Rationale:** Pre-scan covers known unknowns; on-demand covers unknowns discovered mid-review. This is a quality multiplier, but it depends on the pre-scan being solid first. UX work (Rich progress panels for subagent activity) belongs here once the event types from Phase 1 are exercised.
**Delivers:** On-demand deep-dive spawning (capped at 2 per session) for mid-review uncertainty; Rich UI progress for exploration (simplified "Exploring [area]..." panel initially, nested panels deferred); `--no-explore` / `--no-on-demand-explore` CLI flags; cost display per-subagent in UI; `--cost-limit` flag; full `--explore` / `--no-explore` defaults (on for TTT, off for normal review).
**Addresses:** On-demand deep-dive spawning, convention contradiction filtering (via on-demand verification), full CLI flag coverage
**Avoids:** Unpredictable latency (Pitfall 7), nested tool events breaking Rich UI (Pitfall 6), Codex backend divergence (Pitfall 10)
**Research flag:** On-demand exploration mid-review is less-documented than pre-scan; may need iteration on the review prompt to reliably trigger exploration at the right times.

### Phase Ordering Rationale

- Phases 1 and 2 before Phases 3 and 4: `ExplorationContext` must exist before it can be consumed. The data flows in one direction.
- Budget/safety system in Phase 1, not Phase 2: The only way to safely test exploration is with budget limits already enforced. Building exploration without limits first is how the $2,400 overnight incident happens.
- On-demand in Phase 4, not Phase 2: Pre-scan must be solid and demonstrably reducing review errors before adding mid-review spawning. Otherwise it's impossible to tell if on-demand exploration is helping or creating noise.
- Codex backend compatibility addressed in Phase 1 (protocol design) to prevent a Phase 4 rewrite.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** Subagent prompt quality is empirical — the `AgentDefinition` prompts for pattern-scanner, dependency-tracer, and test-mapper will need iteration against real codebases. Plan for 2-3 prompt refinement cycles.
- **Phase 4:** On-demand exploration via the general-purpose subagent (built-in when `Agent` is in `allowedTools`) is less documented than pre-defined custom subagents. How reliably the review agent recognizes its own uncertainty and invokes the Agent tool needs testing.

Phases with standard patterns (skip research-phase):
- **Phase 1:** Backend protocol extension is mechanical — `Optional` parameter, no behavior change. Event type additions follow existing patterns in `AgentEvent` union.
- **Phase 3:** Prompt injection is well-understood. Schema extension (adding `confidence` field) follows existing patterns in `phases.py`.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | SDK subagent API verified from official Anthropic docs. Version requirements verified from GitHub changelog. No speculative dependencies. |
| Features | MEDIUM | Industry benchmarks (Augment Code, CodeAnt, Greptile) converge on the same patterns. Single-vendor sources for some data points. Core feature set is well-validated; confidence scoring specifics are less verified. |
| Architecture | HIGH | Official SDK docs plus practitioner sources align. Three-subagent design is well-reasoned. Data flow and component boundaries are internally consistent. |
| Pitfalls | HIGH | Unbounded exploration and context overload are backed by official Anthropic engineering blog and factory.ai research. Cost figures from production incident reports. Multi-agent failure stats from academic source (lower confidence on exact 36.9% figure). |

**Overall confidence:** HIGH

### Gaps to Address

- **Subagent prompt quality is unknown until tested**: The `AgentDefinition` prompts in ARCHITECTURE.md are well-reasoned starting points, but actual codebase exploration quality requires integration testing with real projects. Plan for iterative prompt refinement in Phase 2.
- **On-demand exploration trigger reliability**: The review agent's ability to recognize its own uncertainty and invoke the Agent tool is a behavior that depends on prompt design. This is not covered by existing documentation and needs empirical validation.
- **Codex backend exploration path**: PITFALLS.md suggests Codex can get exploration via multiple sequential `Backend.execute()` calls rather than SDK subagents. This design is sound but untested; it may require a separate `phase_explore_sequential()` implementation that wraps the same `ExplorationContext` output.
- **Token accounting for subagent synthesis step**: Adding a Haiku/Sonnet synthesis agent introduces another LLM call. The cost/quality tradeoff of synthesis vs. direct injection of capped summaries needs measurement during Phase 2.

## Sources

### Primary (HIGH confidence)
- [Subagents in the SDK - Claude API Docs](https://platform.claude.com/docs/en/agent-sdk/subagents) — subagent API, `AgentDefinition` fields, `Agent` tool behavior
- [Agent SDK reference - Python](https://platform.claude.com/docs/en/agent-sdk/python) — full Python API reference, `ClaudeAgentOptions`, streaming events
- [claude-agent-sdk on PyPI](https://pypi.org/project/claude-agent-sdk/) — v0.1.56 current, version floor verification
- [Releases - claude-agent-sdk-python](https://github.com/anthropics/claude-agent-sdk-python/releases) — changelog: `AgentDefinition` in v0.1.51, `disallowedTools`/`maxTurns` in v0.1.52
- [Anthropic: Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — lost-in-the-middle, context overload mitigations
- [AnyIO 4.13.0 documentation - Tasks](https://anyio.readthedocs.io/en/stable/tasks.html) — task groups, capacity limiters

### Secondary (MEDIUM confidence)
- [Augment Code: How we built a high-quality AI code review agent](https://www.augmentcode.com/blog/how-we-built-high-quality-ai-code-review-agent) — false-positive patterns, convention detection importance, 45% industry true-positive rate
- [GitHub Copilot code review agentic architecture](https://github.blog/changelog/2026-03-05-copilot-code-review-now-runs-on-an-agentic-architecture/) — agentic review as industry baseline
- [Greptile: Graph-based codebase context](https://www.greptile.com/docs/how-greptile-works/graph-based-codebase-context) — cross-file dependency tracing patterns
- [Factory.ai: The Context Window Problem](https://factory.ai/news/context-window-problem) — LLM context degradation at 32K+ tokens
- [RocketEdge: AI Agent Cost Control](https://rocketedge.com/2026/03/15/your-ai-agent-bill-is-30x-higher-than-it-needs-to-be-the-6-tier-fix/) — production cost incident data, $2,400 overnight figure
- [Multi-Agent Reliable Decision-Making](https://multiagents.org/2025_artifacts/reliable_decision_making_for_multi_agent_llm_systems.pdf) — reconciliation strategies for conflicting agent outputs
- [StackToHeap: Claude Code subagents for code review](https://stacktoheap.com/blog/2025/08/10/code-reviews-that-dont-suck-claude-code-subagents/) — practical implementation patterns
- [Qodo: 5 AI Code Review Pattern Predictions 2026](https://www.qodo.ai/blog/5-ai-code-review-pattern-predictions-in-2026/) — industry direction

### Tertiary (LOW confidence)
- [FutureAGI: Why Multi-Agent LLM Systems Fail](https://futureagi.substack.com/p/why-do-multi-agent-llm-systems-fail) — 36.9% inter-agent misalignment stat (single source, needs validation)
- [CodeAnt AI Code Review Benchmark 2026](https://www.codeant.ai/blogs/ai-code-review-benchmark-results-from-200-000-real-pull-requests) — 3.6 actionable comments per PR baseline (vendor benchmark)
- [DEV Community: State of AI Code Review 2026](https://dev.to/rahulxsingh/the-state-of-ai-code-review-in-2026-trends-tools-and-whats-next-2gfh) — industry trends (community article)

---
*Research completed: 2026-04-05*
*Ready for roadmap: yes*
