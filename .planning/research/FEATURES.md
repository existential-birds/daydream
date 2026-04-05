# Feature Landscape: Subagent-Powered Codebase Exploration for Code Review

**Domain:** AI-powered code review with codebase understanding
**Researched:** 2026-04-05

## Table Stakes

Features users expect from any AI code review tool claiming "codebase awareness." Missing these means reviews are shallow diff-only analysis -- the exact problem Daydream is trying to solve.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Diff-adjacent file reading** | Every serious tool reads files touched by the diff plus their immediate imports/callers. Without this, the reviewer is blind to breaking changes. GitHub Copilot's March 2026 agentic overhaul made this the baseline. | Low | Already partially possible via agent tool access; needs systematic triggering before review starts |
| **Cross-file dependency tracing** | A change to a function signature breaks callers. Greptile's core value prop is following call chains across files. CodeRabbit's Codegraph does the same. Users now expect "did you check what calls this?" | Medium | Grep/Glob-based for Daydream's CLI context. No need for full semantic indexing -- the agent has shell access |
| **Convention/pattern detection** | Reviews that suggest patterns contradicting the existing codebase are worse than no review. Augment Code documents this as a top false-positive source. The agent must discover how the codebase already does things before recommending alternatives | Medium | Explore N representative files for the pattern in question. E.g., "how does this codebase handle errors?" before suggesting a different approach |
| **Impact surface mapping** | Before reviewing, identify which parts of the codebase a change could affect. This is the "exploration phase" that Daydream's PROJECT.md calls out as missing | Medium | Pre-scan: diff -> affected files -> transitive dependencies -> scope boundary. This is the subagent's primary job |
| **Repository/project guidelines integration** | Augment, CodeRabbit, and Greptile all support project-level rules (CLAUDE.md, .coderabbit.yaml, AGENTS.md). Reviews must respect documented conventions | Low | Daydream already has CLAUDE.md and Beagle skills. Ensure exploration subagents read project guidelines before reviewing |

## Differentiators

Features that go beyond table stakes and would make Daydream's reviews meaningfully better than "smart diff analysis."

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **On-demand deep-dive spawning** | When the review agent hits uncertainty ("I'm not sure if this breaks the auth flow"), it spawns a focused explorer subagent to investigate that specific question. No other CLI tool does this -- Greptile/CodeRabbit handle it server-side with pre-built indexes. This is Daydream's unique advantage as a Claude Agent SDK tool | High | Core differentiator from PROJECT.md. Requires the review agent to recognize its own uncertainty and request exploration. Claude Agent SDK subagent spawning is the mechanism |
| **Confidence scoring per recommendation** | Each review comment carries a confidence level (HIGH/MEDIUM/LOW) based on how much codebase evidence the agent found. LOW-confidence items get flagged as "needs human verification." Augment Code's benchmark shows 45% true-positive rate industry-wide -- being honest about uncertainty is a competitive advantage | Medium | Structured output schema already exists in Daydream. Add a `confidence` field with rationale. The agent scores based on: did it verify against actual code, or is it guessing from the diff alone? |
| **Parallel pre-scan subagents** | Before review starts, spawn 3-5 exploration subagents in parallel, each investigating a different affected area of the codebase. This is exactly the pattern Claude Code itself uses internally (3 parallel Explore subagents). Turns a 60-second sequential scan into a 15-second parallel one | Medium | Claude Agent SDK supports parallel subagent execution. Each subagent gets: target area, list of questions to answer, file access tools. Results are aggregated into a context document for the review agent |
| **Exploration-informed plan generation** | TTT flow's ENVISION phase generates implementation plans. Plans generated after exploration are executable because they reference actual file paths, actual function signatures, actual patterns. Plans without exploration are aspirational at best | Medium | Wire exploration results into `phase_generate_plan()` prompt. The plan schema already has file-level granularity -- exploration provides the data to populate it accurately |
| **Convention contradiction filtering** | Post-review pass that filters out any recommendation that contradicts a pattern the codebase uses consistently. "You suggest extracting to a utility function, but this codebase deliberately co-locates helpers with their consumers." Augment Code identifies this as the #1 source of annoying false positives | Medium | Two-pass: (1) explore and document codebase conventions, (2) review, (3) filter recommendations against conventions. Or: include conventions in the review prompt so they never get generated |
| **Tribal knowledge capture** | When a human dismisses a review comment with an explanation ("we do it this way because X"), store that as a learned rule for future reviews. Augment Code found that tribal knowledge drives most high-impact review comments | High | Requires persistence layer (file-based is fine for CLI). Store dismissed-with-reason items in `.daydream/learned-rules.yaml`. Feed into future review prompts. Long-term investment |

## Anti-Features

Features to explicitly NOT build. These are traps that waste effort or actively harm review quality.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Full codebase indexing/embedding** | Greptile's approach requires a server-side vector DB, continuous index updates, and infrastructure Daydream doesn't need. The agent has direct file access via tools -- it can grep and read on demand. Building an embedding pipeline is over-engineering for a CLI tool | Use the agent's native tool access (Read, Grep, Glob, Bash). The "index" is the filesystem. Subagents explore on demand |
| **Style/formatting nitpicks** | Industry data shows these are the #1 source of review fatigue and the lowest-value comments. Linters handle this better. Augment Code deliberately filters these out. CodeAnt's benchmark penalizes tools that produce them | Configure review prompts to explicitly exclude formatting, naming style, and cosmetic issues. Focus exploration on correctness, architecture, and patterns |
| **Comment volume maximization** | More comments != better review. Augment Code's benchmark shows the best tools produce ~3.6 actionable comments per PR, not 20 generic observations. High volume destroys developer trust | Optimize for precision over recall. Fewer, high-confidence comments. Use confidence scoring to suppress LOW-confidence items by default |
| **Custom orchestration framework** | PROJECT.md explicitly rules this out. The Claude Agent SDK has native subagent capabilities. Building a custom scheduler, queue, or orchestration layer duplicates SDK functionality and creates maintenance burden | Use `claude-agent-sdk` subagent primitives directly. Let the SDK handle lifecycle, context management, and tool delegation |
| **Real-time / streaming exploration results** | Tempting to show exploration progress in the UI, but it adds complexity without improving review quality. The user cares about the review output, not the exploration process | Show a simple progress indicator ("Exploring 4 areas...") via existing Rich UI. Don't stream intermediate exploration findings to the terminal |
| **Multi-model ensemble review** | Running the same review through multiple models and merging results sounds clever but doubles/triples cost and latency for marginal quality improvement. The industry has moved toward single-model quality optimization | Invest in better prompts and more context for one model rather than running multiple models |

## Feature Dependencies

```
Repository guidelines integration
    |
    v
Impact surface mapping ──> Diff-adjacent file reading
    |                           |
    v                           v
Cross-file dependency tracing   Convention/pattern detection
    |                           |
    v                           v
Parallel pre-scan subagents ──> On-demand deep-dive spawning
    |                           |
    v                           v
Convention contradiction filtering
    |
    v
Confidence scoring per recommendation
    |
    v
Exploration-informed plan generation
    |
    v
Tribal knowledge capture (long-term)
```

**Critical path:** Impact surface mapping -> Pre-scan subagents -> On-demand spawning -> Confidence scoring

**Explanation:**
1. You must map what areas are affected before you can explore them (impact surface mapping)
2. Pre-scan subagents need that map to know where to look
3. On-demand spawning requires the review agent to already have pre-scan context (so it knows what it doesn't know)
4. Confidence scoring requires exploration infrastructure to be in place (confidence = "did I verify this against actual code?")
5. Convention filtering and plan generation consume exploration results
6. Tribal knowledge capture is independent but needs the review feedback loop working first

## MVP Recommendation

**Prioritize (Phase 1 -- Exploration Foundation):**
1. Impact surface mapping -- the subagent that reads the diff and identifies affected areas
2. Diff-adjacent file reading -- explore files directly touched and their imports
3. Convention/pattern detection -- discover how the codebase does things before reviewing

**Prioritize (Phase 2 -- Subagent Integration):**
4. Parallel pre-scan subagents -- the performance multiplier
5. On-demand deep-dive spawning -- the quality multiplier
6. Confidence scoring -- honest about what was verified vs. guessed

**Defer:**
- Tribal knowledge capture: Requires a feedback loop that doesn't exist yet. Build the review quality first, then add learning
- Exploration-informed plan generation: Wire exploration into TTT after the exploration layer is proven in normal review flow
- Convention contradiction filtering: Start by including conventions in the review prompt (cheap). Build the post-filter only if false positives remain high

## Sources

- [Augment Code: How we built a high-quality AI code review agent](https://www.augmentcode.com/blog/how-we-built-high-quality-ai-code-review-agent) -- MEDIUM confidence (detailed technical blog, single source)
- [StackToHeap: Claude Code subagents for code review](https://stacktoheap.com/blog/2025/08/10/code-reviews-that-dont-suck-claude-code-subagents/) -- MEDIUM confidence (practical implementation example)
- [Greptile: Graph-based codebase context](https://www.greptile.com/docs/how-greptile-works/graph-based-codebase-context) -- HIGH confidence (official documentation)
- [GitHub Copilot code review agentic architecture](https://github.blog/changelog/2026-03-05-copilot-code-review-now-runs-on-an-agentic-architecture/) -- HIGH confidence (official announcement)
- [CodeRabbit: 2025 was the year of AI speed, 2026 will be quality](https://www.coderabbit.ai/blog/2025-was-the-year-of-ai-speed-2026-will-be-the-year-of-ai-quality) -- MEDIUM confidence (vendor blog)
- [CodeAnt AI Code Review Benchmark 2026](https://www.codeant.ai/blogs/ai-code-review-benchmark-results-from-200-000-real-pull-requests) -- MEDIUM confidence (third-party benchmark)
- [Greptile AI Code Review Benchmarks 2025](https://www.greptile.com/benchmarks) -- MEDIUM confidence (vendor benchmark)
- [DEV Community: State of AI Code Review in 2026](https://dev.to/rahulxsingh/the-state-of-ai-code-review-in-2026-trends-tools-and-whats-next-2gfh) -- LOW confidence (community article)
