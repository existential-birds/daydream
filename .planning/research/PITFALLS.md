# Domain Pitfalls

**Domain:** Subagent-powered codebase exploration for code review CLI
**Researched:** 2026-04-05

## Critical Pitfalls

Mistakes that cause rewrites, runaway costs, or fundamental architecture problems.

### Pitfall 1: Unbounded Exploration (The $2,400 Overnight Problem)

**What goes wrong:** Exploration subagents recursively follow imports, type references, and call chains without a termination condition. A single exploration of an `auth/` module spirals into reading the entire codebase because everything eventually connects to everything. Token costs compound: each file read adds to context, making the next decision more expensive. Production incidents have seen agents rack up $2,400 in API charges overnight from infinite exploration loops.

**Why it happens:** Codebases are graphs, not trees. Every module has transitive dependencies. An exploration agent told to "understand the context around this change" has no natural stopping point. The agent reads a file, sees 5 imports, reads those, sees 15 more imports, and so on. Unlike a human who intuitively knows when they have "enough" context, LLMs will keep exploring because each new file contains plausibly relevant information.

**Consequences:** Token costs scale exponentially with exploration depth. A 3-level deep exploration of a mid-size project can easily consume 200K+ tokens per subagent. With multiple subagents running in parallel, a single review could cost $10-50 in API charges. Worse, the exploration results become less useful as they grow -- the "lost-in-the-middle" phenomenon means the agent loses track of findings from early in the exploration.

**Prevention:**
- Hard cap exploration depth at 2 levels (direct dependencies only, no transitive)
- Hard cap total files read per subagent at 15-20
- Hard cap total tokens per exploration subagent (track via CostEvent, abort at threshold)
- Time-box exploration to 60 seconds per subagent
- Use Sonnet for exploration subagents (cheaper, fast enough for file reading)
- Provide explicit exploration boundaries in the subagent prompt: "Read ONLY files in `auth/` and `middleware/`, ignore all other directories"

**Detection:** Monitor CostEvent streams during exploration. If a single subagent exceeds $0.50 or 100K input tokens, something is wrong. Track files-read count per subagent and alert at >20.

**Phase mapping:** Must be addressed in Phase 1 (exploration architecture). Build the budget/limit system before any exploration logic.

### Pitfall 2: Backend Protocol Cannot Express Subagents

**What goes wrong:** The current `Backend` protocol has a single `execute(cwd, prompt, output_schema, continuation)` method that maps to one agent invocation. Claude Agent SDK subagents are defined via `agents` parameter on `ClaudeAgentOptions` and invoked through the `Agent` tool -- a fundamentally different interaction model than prompt-in/events-out. Attempting to shoehorn subagent orchestration through the existing `Backend.execute()` creates either a leaky abstraction (Claude-specific subagent logic bleeding through) or a lowest-common-denominator API that cannot use SDK subagents at all.

**Why it happens:** The Backend protocol was designed for single-agent, single-turn interactions. Subagents require: (1) agent definitions passed at query time, (2) the `Agent` tool in `allowedTools`, (3) detecting `ToolStartEvent` with `name="Agent"` and `parent_tool_use_id` on child messages, and (4) optionally resuming subagents via session IDs. None of this maps to `execute()`.

**Consequences:** Either the Codex backend gets left behind (subagents are Claude-only, breaking the backend abstraction), or you build a parallel orchestration path that duplicates the Backend protocol's responsibilities. Both outcomes create maintenance burden and architectural confusion.

**Prevention:**
- Extend the Backend protocol with an `execute_with_agents()` method or an `agents` parameter on `execute()`
- Accept that Codex backend may not support subagents initially -- use a capability flag (`supports_subagents: bool`) rather than forcing a uniform API
- Keep subagent definitions in the orchestration layer (phases.py / runner.py), not inside the backend -- the backend just needs to know how to pass agent configs through to the SDK
- Add new event types to AgentEvent union: `SubagentStartEvent`, `SubagentResultEvent` for UI tracking

**Detection:** If you find yourself writing `if isinstance(backend, ClaudeBackend)` checks in phases.py, the abstraction has leaked.

**Phase mapping:** Must be addressed in Phase 1. The protocol extension is a prerequisite for all exploration work.

### Pitfall 3: Exploration Results Exceed Parent Context Capacity

**What goes wrong:** Multiple exploration subagents each return 2,000-5,000 token summaries. The parent agent receives all summaries, plus the original diff, plus the review prompt, plus CLAUDE.md project context. This easily pushes the parent past 100K tokens before the actual review begins. At 200K tokens, Anthropic applies a 2x cost surcharge to the entire request (not just tokens above the threshold). The parent agent's review quality degrades because the "lost-in-the-middle" effect means exploration findings from the second and third subagent get ignored.

**Why it happens:** Each subagent returns its full findings as a single text block. The parent accumulates all of them. There is no built-in summarization or prioritization step between exploration and review. Research shows 11 out of 13 LLMs drop below 50% baseline performance at just 32K tokens when tasks require more than surface-level pattern matching.

**Consequences:** Review quality actually decreases compared to the no-exploration baseline, because the agent is drowning in context it cannot effectively attend to. Costs double from the 200K surcharge. The user perceives slower, worse reviews at higher cost -- the opposite of the feature's goal.

**Prevention:**
- Cap each subagent's return summary at 1,000 tokens (enforce via output_schema with max token guidance in prompt)
- Implement a synthesis step: a lightweight Haiku/Sonnet agent that merges all exploration summaries into a single 1,500-2,000 token briefing before passing to the review agent
- Structure exploration results as prioritized bullet points, not prose -- helps the parent agent scan and attend
- Monitor total context size before launching the review phase; if > 80K tokens after exploration, trigger summarization

**Detection:** Track total input tokens on the review phase's first CostEvent. Compare review quality (issue count, specificity) between exploration-enabled and exploration-disabled runs during development. If exploration-enabled reviews find fewer issues, context overload is likely.

**Phase mapping:** Phase 2 (integration with review flow). Build summarization/synthesis before wiring exploration into the review pipeline.

### Pitfall 4: Conflicting Exploration Findings

**What goes wrong:** Parallel exploration subagents examine overlapping code areas and return contradictory conclusions. One subagent reports "this module uses the repository pattern" while another reports "data access is done through direct ORM calls." The review agent receives both, cannot reconcile, and either picks one arbitrarily (producing wrong recommendations) or hedges with vague advice (producing useless recommendations).

**Why it happens:** Each subagent has isolated context. They cannot see each other's findings. Different starting files or different exploration paths through the same code lead to different mental models. Research shows inter-agent misalignment accounts for 36.9% of multi-agent system failures, with communication breakdowns and conflicting outputs as primary causes.

**Consequences:** Review produces contradictory feedback. User loses trust in the tool. In the worst case, the fix phase applies changes based on a wrong mental model, introducing bugs.

**Prevention:**
- Partition exploration by concern, not by file area: one subagent for "architectural patterns," another for "test conventions," another for "error handling patterns" -- this reduces overlap
- Assign non-overlapping file scopes to parallel explorers when partitioning by area
- Include a reconciliation prompt in the synthesis step: "If findings conflict, flag the conflict explicitly rather than choosing one"
- For the TTT flow, the WONDER phase should receive raw exploration summaries with source attribution so it can weigh conflicting signals

**Detection:** Search review output for hedging language ("it appears," "in some files," "inconsistently") which indicates unresolved conflicts. Track cases where the fix phase reverts or contradicts exploration findings.

**Phase mapping:** Phase 2. Design the partitioning strategy before running parallel explorers.

## Moderate Pitfalls

### Pitfall 5: Exploration Latency Kills the UX

**What goes wrong:** Pre-scan exploration adds 30-60 seconds before the review begins. For small diffs (1-3 files), users wait longer for exploration than the review itself takes. The tool feels slower than before the feature was added.

**Prevention:**
- Scale exploration to diff size: skip pre-scan entirely for diffs touching <3 files; use lightweight single-subagent exploration for 3-10 files; use parallel multi-subagent exploration for 10+ files
- Show Rich live UI progress during exploration (file being read, subagent status) so the wait feels productive
- Use Sonnet or Haiku for exploration subagents -- they are 5-10x faster than Opus for file-reading tasks
- Implement a `--no-explore` flag for users who want speed over depth

**Detection:** Time the exploration phase separately from the review phase. If exploration > 50% of total wall-clock time for diffs under 10 files, the threshold tuning is wrong.

**Phase mapping:** Phase 1 (exploration architecture) for the scaling logic; Phase 3 (UX integration) for the progress UI.

### Pitfall 6: Subagent Tool Events Break the Existing UI

**What goes wrong:** The current Rich UI (LiveToolPanelRegistry, AgentTextRenderer) expects a flat stream of ToolStartEvent/ToolResultEvent from a single agent. Subagent invocations produce nested tool events: the parent emits a ToolStartEvent for the Agent tool, then child messages arrive with `parent_tool_use_id` set, containing their own tool events. The UI either crashes on unexpected event shapes, shows garbled nested panels, or silently drops subagent activity.

**Prevention:**
- Add SubagentStartEvent and SubagentResultEvent to the AgentEvent union type
- Update LiveToolPanelRegistry to handle nested tool panels (indented or grouped by subagent)
- During initial development, log subagent events to debug log but render only a single "Exploring [area]..." panel in the UI -- defer the rich nested UI to a later phase
- Test UI rendering with mock subagent event streams before connecting to real SDK

**Detection:** Run the tool with `--debug` and check that all subagent tool calls appear in the debug log. Visually inspect the terminal for garbled or overlapping Rich panels.

**Phase mapping:** Phase 1 for event type additions; Phase 3 for full UI integration.

### Pitfall 7: On-Demand Exploration Creates Unpredictable Latency

**What goes wrong:** The review agent encounters uncertainty mid-review and spawns an exploration subagent. The user sees the review pause for 15-30 seconds with no explanation, then resume. If the agent spawns multiple on-demand explorers, the review takes 3-5x longer than expected, and the user has no way to predict when it will finish.

**Prevention:**
- Cap on-demand explorations to 2 per review session (configurable via RunConfig)
- Show explicit UI state when an on-demand exploration fires: "Investigating [module] for more context..."
- Prefer pre-scan over on-demand: invest more in upfront exploration so mid-review exploration is rarely needed
- Allow the user to disable on-demand exploration with `--no-on-demand-explore`

**Detection:** Count on-demand exploration invocations per review. If average > 1.5, the pre-scan is not covering enough.

**Phase mapping:** Phase 2 (on-demand exploration). This is a later-phase feature that depends on pre-scan being solid.

### Pitfall 8: Exploration Without Structured Output Produces Unusable Results

**What goes wrong:** Exploration subagents return free-form text summaries. The parent agent must parse prose to extract architectural patterns, conventions, and file relationships. Parsing is unreliable, and the parent often misinterprets or ignores unstructured findings.

**Prevention:**
- Use `output_schema` on exploration subagents to enforce structured JSON responses: `{ "patterns": [...], "conventions": [...], "key_files": [...], "dependencies": [...] }`
- The current Backend protocol already supports `output_schema` -- use it
- Test the schema with representative codebases to ensure the exploration agent can consistently fill it

**Detection:** Parse exploration results programmatically before passing to the review agent. If JSON parsing fails > 5% of the time, the schema or prompt needs adjustment.

**Phase mapping:** Phase 1. Define the exploration output schema as part of the exploration architecture.

## Minor Pitfalls

### Pitfall 9: Session/Agent ID Management for Resumable Explorations

**What goes wrong:** The SDK supports resuming subagents via session_id and agent_id, but these IDs must be captured from the message stream, stored, and correctly passed on resume. Losing a session ID means re-running the entire exploration from scratch. The ID extraction requires parsing message content with regex (`agentId:\s*([a-f0-9-]+)`), which is fragile.

**Prevention:**
- Store session/agent IDs in a dataclass alongside exploration results
- Only use resumable subagents when the use case clearly benefits (iterative deepening); for one-shot exploration, skip resumption entirely
- Wrap the ID extraction in a tested utility function, not inline regex

**Phase mapping:** Phase 2 or later. One-shot exploration in Phase 1 does not need resumption.

### Pitfall 10: Codex Backend Left Behind

**What goes wrong:** All exploration work is implemented against the Claude Agent SDK's subagent API. The Codex backend has no equivalent subagent primitive, so it gets zero exploration capability. Over time, the two backends diverge significantly, and the Codex path becomes a second-class citizen.

**Prevention:**
- Design exploration as a layer above the backend, not within it: exploration orchestration lives in phases.py, calling multiple `backend.execute()` invocations rather than relying on SDK-native subagents
- This means Codex gets exploration too (multiple sequential/parallel `execute()` calls), just without SDK-level context isolation
- Use SDK subagents as an optimization for Claude backend, not as the only path

**Detection:** Run the test suite against both backends after every exploration feature is added. If Codex tests are being skipped or mocked away, divergence is happening.

**Phase mapping:** Phase 1 architecture decision. Must be decided before implementation begins.

### Pitfall 11: The "Agent" vs "Task" Tool Name Transition

**What goes wrong:** The Claude Agent SDK renamed the subagent invocation tool from "Task" to "Agent" in Claude Code v2.1.63. Current SDK releases emit "Agent" in tool_use blocks but still use "Task" in system:init tools list and permission_denials. Code that checks for only one name silently misses the other, causing subagent events to be dropped or misrouted.

**Prevention:**
- Check for both `"Agent"` and `"Task"` in all tool name comparisons
- Pin a minimum SDK version that uses "Agent" consistently
- Add a test that verifies subagent detection works with both tool names

**Phase mapping:** Phase 1. Bake this into the event detection code from the start.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Exploration architecture (Phase 1) | Unbounded exploration, Backend protocol mismatch | Budget system first, protocol extension before any exploration code |
| Pre-scan integration (Phase 1) | Over-exploration on small diffs | Scale exploration to diff size, skip for <3 files |
| Review integration (Phase 2) | Context overload in parent agent | Synthesis/summarization step between exploration and review |
| On-demand exploration (Phase 2) | Unpredictable latency, conflicting findings | Cap at 2 per session, partition by concern |
| UI integration (Phase 3) | Nested tool events break Rich panels | New event types, simplified initial UI, defer rich nesting |
| TTT flow integration (Phase 2-3) | Exploration duplicating LISTEN phase work | Reuse LISTEN phase's diff/log data as exploration seed |
| Cost tracking (Phase 1) | Costs invisible to user until bill arrives | Surface per-subagent cost in UI, add `--cost-limit` flag |

## Sources

- [Claude Agent SDK Subagents Documentation](https://platform.claude.com/docs/en/agent-sdk/subagents) - HIGH confidence, official docs
- [Anthropic: Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) - HIGH confidence, official engineering blog
- [Factory.ai: The Context Window Problem](https://factory.ai/news/context-window-problem) - MEDIUM confidence, detailed research on lost-in-the-middle and context degradation
- [OpenReview: Analyzing Token Consumptions in Agentic Coding Tasks](https://openreview.net/forum?id=1bUeVB3fov) - MEDIUM confidence, academic analysis of agent token waste
- [RocketEdge: AI Agent Cost Control](https://rocketedge.com/2026/03/15/your-ai-agent-bill-is-30x-higher-than-it-needs-to-be-the-6-tier-fix/) - MEDIUM confidence, production cost data
- [Multi-Agent Reliable Decision-Making](https://multiagents.org/2025_artifacts/reliable_decision_making_for_multi_agent_llm_systems.pdf) - MEDIUM confidence, reconciliation strategies
- [FutureAGI: Why Multi-Agent LLM Systems Fail](https://futureagi.substack.com/p/why-do-multi-agent-llm-systems-fail) - LOW confidence, single source for the 36.9% misalignment stat

---

*Pitfalls audit: 2026-04-05*
