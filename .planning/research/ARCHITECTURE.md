# Architecture Patterns

**Domain:** Subagent-powered codebase exploration for AI code review
**Researched:** 2026-04-05

## Recommended Architecture

### The Core Insight

The Claude Agent SDK already handles subagent orchestration, context isolation, and parallel execution. Daydream does not need to build an orchestration framework. It needs to:

1. Define exploration subagents via `AgentDefinition`
2. Pass them into the existing `Backend.execute()` call path
3. Let the SDK's `Agent` tool handle spawning, parallelism, and result aggregation

The main agent (the one running the review or TTT phase) acts as the orchestrator. It decides when to spawn explorers based on its prompt instructions. Daydream's job is to configure the right subagents and feed the main agent a prompt that tells it when and how to use them.

### Component Architecture

```
                        RunConfig
                            |
                     runner.py (flow selection)
                            |
              +-------------+-------------+
              |                           |
        run() flow                   run_trust() flow
              |                           |
    phase_explore()  <-- NEW    phase_explore()  <-- NEW (shared)
              |                           |
    phase_review()              phase_understand_intent()
              |                           |
    phase_parse_feedback()      phase_alternative_review()
              |                           |
        ...continues...         phase_generate_plan()
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| **ExplorationConfig** (new dataclass) | Holds `AgentDefinition` instances for exploration subagents, prompt templates, affected-file detection settings | `runner.py`, `phases.py` |
| **phase_explore()** (new phase function) | Pre-scan phase: runs the main explore agent which spawns subagents to map affected areas; returns `ExplorationContext` | `runner.py` (called by), `agent.py` (via `run_agent()`) |
| **ExplorationContext** (new dataclass) | Structured output from exploration: file map, patterns detected, conventions found, dependency graph fragment | `phase_explore()` (produces), downstream phases (consume) |
| **Affected-file detector** (new utility) | Determines which files/directories need exploration from git diff, skill type, or PR scope | `phase_explore()`, `runner.py` |
| **Agent definitions** (new module) | Factory functions that produce `AgentDefinition` instances for different explorer types | `phase_explore()` |
| **Backend protocol** (existing, extended) | Needs `agents` parameter added to `execute()` to pass `AgentDefinition` dicts through to the SDK | `ClaudeBackend` (implements), `agent.py` (calls) |

### What Does NOT Change

- `run_agent()` signature stays the same (or gains one optional param for agents)
- `AgentEvent` union stays the same -- subagent invocations emit `ToolStartEvent`/`ToolResultEvent` with `name="Agent"`
- Phase functions remain stateless async functions
- `RunConfig` gains at most an `explore: bool` flag
- UI layer needs no changes -- subagent tool calls render through the existing `LiveToolPanelRegistry`

## Data Flow

### Pre-scan Exploration (New)

```
1. runner.py gathers context:
   - git diff (existing) -> affected files list
   - skill type -> relevant file patterns (e.g., *.py for python skill)
   - target_dir -> codebase root

2. runner.py calls phase_explore(backend, target_dir, affected_files, skill):
   - Builds exploration prompt with file list and instructions
   - Configures AgentDefinition instances:
     * "pattern-scanner": reads project structure, conventions, config files
     * "dependency-tracer": traces imports/dependencies from affected files
     * "test-mapper": finds related test files and test patterns
   - Calls run_agent(backend, target_dir, prompt,
       output_schema=EXPLORATION_SCHEMA, agents=agent_defs)
   - Main agent spawns subagents via Agent tool (SDK handles parallelism)
   - Each subagent explores its assigned area, returns findings as text
   - Main agent aggregates subagent results into structured output
   - Returns ExplorationContext dataclass

3. ExplorationContext flows into downstream phases:
   - phase_review() gets context injected into its prompt
   - phase_alternative_review() gets context injected into its prompt
   - phase_generate_plan() gets context for grounded plan generation
```

### ExplorationContext Structure

```python
@dataclass
class ExplorationContext:
    """Structured output from the exploration phase."""

    # What the codebase looks like around affected areas
    file_map: dict[str, str]       # {path: brief_description}

    # Patterns and conventions discovered
    conventions: list[str]          # e.g., "Uses repository pattern for DB access"

    # Dependencies traced from affected files
    dependencies: dict[str, list[str]]  # {affected_file: [files_it_depends_on]}

    # Test coverage mapping
    test_files: dict[str, list[str]]    # {source_file: [test_files]}

    # Architecture notes
    architecture_notes: list[str]   # e.g., "Event-driven with RabbitMQ"

    # Raw exploration text for injection into prompts
    summary: str                    # Natural language summary
```

### How Context Flows Into Review Prompts

```
# Before (current):
phase_review() prompt = "/{skill} {target_dir}"

# After:
phase_review() prompt = """
## Codebase Context (from exploration)
{exploration_context.summary}

### Key Conventions
{bullet_list(exploration_context.conventions)}

### Affected File Dependencies
{formatted(exploration_context.dependencies)}

## Review Task
/{skill} {target_dir}

IMPORTANT: Your recommendations must respect the conventions listed above.
Do NOT recommend patterns that contradict existing codebase conventions.
"""
```

### On-Demand Exploration (Future Enhancement)

The SDK does not support dynamically spawning new subagent types mid-conversation. All `AgentDefinition` instances must be provided at `query()` time. However, the built-in `general-purpose` subagent is always available when `Agent` is in `allowedTools`. This means:

- Pre-scan covers known areas via specialized subagents
- On-demand exploration during review uses the built-in general-purpose subagent
- The review agent's prompt instructs it: "If you encounter uncertainty about codebase patterns, use the Agent tool to explore before making recommendations"

This is the correct architecture because custom subagent definitions are static, but the general-purpose agent can handle any ad-hoc exploration task.

## Backend Protocol Extension

The `Backend.execute()` method needs an `agents` parameter:

```python
class Backend(Protocol):
    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, AgentDefinition] | None = None,  # NEW
    ) -> AsyncIterator[AgentEvent]: ...
```

### ClaudeBackend Implementation

```python
# In ClaudeBackend.execute():
options = ClaudeAgentOptions(
    cwd=str(cwd),
    permission_mode="bypassPermissions",
    setting_sources=["user", "project", "local"],
    model=self.model,
    output_format=output_format,
    max_buffer_size=10 * 1024 * 1024,
    # NEW: pass agents and ensure Agent tool is allowed
    agents=agents,
    allowed_tools=["Read", "Grep", "Glob", "Bash", "Edit", "Write", "Agent"]
        if agents else None,
)
```

### CodexBackend

Codex does not support the `AgentDefinition` subagent protocol. The exploration phase should be Claude-only. When the default backend is Codex, `phase_explore()` should create a temporary Claude backend for exploration, then hand off to Codex for downstream phases. This aligns with the existing `_resolve_backend()` per-phase override pattern.

## Subagent Definitions

### Explorer Subagents

Three specialized read-only subagents, all using `sonnet` model for cost efficiency:

```python
def create_exploration_agents(
    affected_files: list[str],
    skill_type: str,
) -> dict[str, AgentDefinition]:
    return {
        "pattern-scanner": AgentDefinition(
            description="Scans project structure, config files, and conventions. "
                       "Use to understand how the codebase is organized.",
            prompt=f"""You are a codebase structure analyst. Your job is to understand
how this project is organized and what conventions it follows.

Examine:
- Directory structure and naming conventions
- Config files (pyproject.toml, tsconfig.json, .eslintrc, etc.)
- Common patterns in existing code (error handling, logging, testing)
- Framework-specific conventions

Affected files for context: {', '.join(affected_files[:20])}

Return a concise summary of conventions and patterns found.""",
            tools=["Read", "Grep", "Glob"],
            model="sonnet",
        ),

        "dependency-tracer": AgentDefinition(
            description="Traces import chains and dependencies from specific files. "
                       "Use to understand what code is connected to the changes.",
            prompt=f"""You are a dependency analysis specialist. Trace the import
and dependency chains for these files:

{chr(10).join(f'- {f}' for f in affected_files[:20])}

For each file:
1. What does it import? (direct dependencies)
2. What imports it? (reverse dependencies)
3. What shared types/interfaces does it use?

Return a dependency map showing the connection graph.""",
            tools=["Read", "Grep", "Glob"],
            model="sonnet",
        ),

        "test-mapper": AgentDefinition(
            description="Finds test files and test patterns related to specific source files. "
                       "Use to understand test coverage and testing conventions.",
            prompt=f"""You are a test coverage analyst. Find test files related to:

{chr(10).join(f'- {f}' for f in affected_files[:20])}

Determine:
1. Which test files cover these source files?
2. What testing patterns are used? (pytest fixtures, mocks, factories, etc.)
3. What test commands are configured? (check Makefile, package.json scripts, etc.)

Return a mapping of source files to their test files and a summary of testing patterns.""",
            tools=["Read", "Grep", "Glob"],
            model="sonnet",
        ),
    }
```

### Why Three Subagents, Not One

- **Parallelism**: SDK runs subagents concurrently. Three focused subagents finish faster than one comprehensive explorer.
- **Context isolation**: Each subagent's 200K context window is dedicated to its specific task. A single agent exploring everything risks context pressure.
- **Targeted prompts**: A dependency tracer with a focused prompt produces better results than a general explorer asked to "also trace dependencies."
- **Cost**: Sonnet subagents cost ~1/5 of Opus. Three sonnet subagents cost less than one Opus agent doing the same work.

### Why Not More Subagents

- The SDK spawns subagents via the `Agent` tool. Each invocation is a tool call the parent agent decides to make. More than 3-4 subagents means the parent spends its context on coordination overhead.
- Diminishing returns: pattern scanning, dependency tracing, and test mapping cover the three dimensions that matter for grounded review. Security scanning and performance analysis are the review agent's job, not the explorer's.

## Patterns to Follow

### Pattern 1: Structured Exploration Output

Use `output_schema` on the exploration phase to get typed results, not free-form text.

```python
EXPLORATION_SCHEMA = {
    "type": "object",
    "properties": {
        "file_map": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Map of file paths to brief descriptions"
        },
        "conventions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Coding conventions and patterns discovered"
        },
        "dependencies": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string"}
            },
            "description": "Map of affected files to their dependencies"
        },
        "test_files": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string"}
            },
            "description": "Map of source files to related test files"
        },
        "architecture_notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key architecture observations"
        },
        "summary": {
            "type": "string",
            "description": "Natural language summary of exploration findings"
        }
    },
    "required": ["conventions", "dependencies", "summary"]
}
```

### Pattern 2: Phase-as-Enrichment, Not Phase-as-Gate

Exploration should enrich downstream phases, not block them. If exploration fails or times out, the review should still proceed (just without exploration context). This matches how the existing pipeline works -- phases are sequential but independent.

```python
# In runner.py:
exploration_context = None
try:
    exploration_context = await phase_explore(backend, target_dir, affected_files, skill)
except Exception:
    print_warning(console, "Exploration phase failed -- proceeding without context")

# Pass to review phase regardless
await phase_review(review_backend, target_dir, skill, exploration_context)
```

### Pattern 3: Affected-File Detection

Git diff is the source of truth for what changed. Expand outward from there.

```python
def detect_affected_files(target_dir: Path) -> list[str]:
    """Get files affected by current changes, plus their immediate neighbors."""
    # Direct changes from git diff
    diff_result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        capture_output=True, text=True, cwd=target_dir
    )
    changed = diff_result.stdout.strip().splitlines()

    # Also include staged changes
    staged_result = subprocess.run(
        ["git", "diff", "--name-only", "--cached"],
        capture_output=True, text=True, cwd=target_dir
    )
    changed.extend(staged_result.stdout.strip().splitlines())

    return sorted(set(changed))
```

The exploration subagents expand from this list by tracing imports and finding related tests. Daydream gives them the starting point; they find the blast radius.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Building a Custom Orchestration Layer

**What:** Creating a `SubagentManager`, `ExplorationOrchestrator`, or similar class that manages subagent lifecycle.
**Why bad:** The SDK already does this. Custom orchestration means maintaining parallel state management, handling cancellation, dealing with failures -- all things the SDK handles internally.
**Instead:** Define `AgentDefinition` instances. Pass them to `execute()`. The main agent decides when to spawn them. The SDK handles the rest.

### Anti-Pattern 2: Passing Exploration Results as Continuation Tokens

**What:** Trying to chain exploration into review via `ContinuationToken`.
**Why bad:** Continuation tokens are for multi-turn conversations with the same agent. Exploration and review are different agents with different prompts. Continuation would carry the entire exploration conversation into the review agent's context, wasting tokens.
**Instead:** Extract structured data from exploration. Inject it as text into the review prompt. Clean separation.

### Anti-Pattern 3: Exploring Everything

**What:** Running exploration against the entire codebase regardless of what changed.
**Why bad:** Expensive (3 sonnet subagents reading hundreds of files), slow (minutes of exploration before review starts), and noisy (exploration findings about unrelated code dilute the signal).
**Instead:** Start from affected files. Let subagents expand outward from there. For a 5-file diff, exploration should touch 20-50 files, not 500.

### Anti-Pattern 4: Making Exploration Mandatory

**What:** Requiring exploration for every review run.
**Why bad:** Small changes (typo fix, dependency bump) don't benefit from exploration. Users reviewing a known area don't need the agent to rediscover what they already know.
**Instead:** Exploration is opt-in (default on for TTT, off for normal review with `--explore` flag). Always skippable. Always degradable (review works without it).

### Anti-Pattern 5: Subagents That Write Files

**What:** Giving exploration subagents `Edit`, `Write`, or `Bash` tools.
**Why bad:** Exploration is read-only by definition. Write access creates side effects that interfere with the review phase. A subagent that modifies files before review starts corrupts the diff the review agent sees.
**Instead:** All exploration subagents get `["Read", "Grep", "Glob"]` only. No exceptions.

## Scalability Considerations

| Concern | Small project (<50 files) | Medium project (50-500 files) | Large monorepo (500+ files) |
|---------|--------------------------|-------------------------------|----------------------------|
| Exploration scope | Likely unnecessary; 3 subagents overkill | Sweet spot; subagents explore affected neighborhood efficiently | Must limit to affected directories, not full repo scan |
| Subagent count | 1-2 subagents sufficient | 3 subagents (the default set) | 3 subagents with tighter scope constraints in prompts |
| Context pressure | No concern | No concern | Subagent prompts must include directory bounds to prevent wandering |
| Cost | ~$0.02-0.05 for 3 sonnet subagents | ~$0.05-0.15 | ~$0.10-0.30 (still small vs. review agent cost) |
| Latency | 5-15 seconds | 15-45 seconds | 30-90 seconds (acceptable as pre-scan) |

## Suggested Build Order

Dependencies between components dictate implementation order:

### Phase 1: Backend Protocol Extension

Extend `Backend.execute()` to accept `agents` parameter. Update `ClaudeBackend` to pass agents through to `ClaudeAgentOptions`. Update `CodexBackend` to ignore the parameter. Update `run_agent()` to forward agents. This is the foundation everything else builds on.

**Dependencies:** None (extends existing interfaces)
**Risk:** Low -- additive change, optional parameter with None default

### Phase 2: Affected-File Detection + ExplorationContext

Build `detect_affected_files()` utility. Define `ExplorationContext` dataclass. Define `EXPLORATION_SCHEMA`. These are pure data structures and utility functions with no agent interaction.

**Dependencies:** None
**Risk:** Low -- pure functions and dataclasses

### Phase 3: Agent Definition Factory

Build `create_exploration_agents()` factory function. Requires `AgentDefinition` import from `claude_agent_sdk`. Test with mock backends.

**Dependencies:** Phase 1 (needs agents parameter to exist)
**Risk:** Medium -- prompt engineering for subagent quality is iterative

### Phase 4: phase_explore() Implementation

Wire everything together: call `detect_affected_files()`, create agent definitions, call `run_agent()` with agents and schema, parse into `ExplorationContext`. Add to `runner.py` flow.

**Dependencies:** Phases 1, 2, 3
**Risk:** Medium -- integration testing with real SDK needed

### Phase 5: Downstream Integration

Modify `phase_review()`, `phase_alternative_review()`, `phase_generate_plan()` to accept optional `ExplorationContext` and inject it into prompts.

**Dependencies:** Phase 4 (needs ExplorationContext to exist)
**Risk:** Low -- additive prompt changes

### Phase 6: CLI Flags and Defaults

Add `--explore` / `--no-explore` flags. Default on for TTT, off for normal review. Wire into `RunConfig`.

**Dependencies:** Phase 5
**Risk:** Low -- CLI plumbing

## Sources

- [Subagents in the SDK - Claude API Docs](https://platform.claude.com/docs/en/agent-sdk/subagents) -- HIGH confidence, official documentation
- [Agent SDK reference - Python](https://platform.claude.com/docs/en/agent-sdk/python) -- HIGH confidence, official reference
- [5 AI Code Review Pattern Predictions in 2026 - Qodo](https://www.qodo.ai/blog/5-ai-code-review-pattern-predictions-in-2026/) -- MEDIUM confidence, industry analysis
- [AI Coding Agents in 2026: Coherence Through Orchestration](https://mikemason.ca/writing/ai-coding-agents-jan-2026/) -- MEDIUM confidence, practitioner perspective
- [The Atlas Method: 7 AI Agents Inside Copilot](https://patrickarobinson.com/blog/atlas-method-copilot-agents/) -- MEDIUM confidence, architectural patterns
- [Agent Architecture: Building AI-Powered Development Harnesses](https://blakecrosley.com/guides/agent-architecture) -- MEDIUM confidence, design patterns

---

*Architecture research: 2026-04-05*
