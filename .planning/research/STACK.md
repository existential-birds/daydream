# Technology Stack

**Project:** Daydream — Subagent Exploration Layer
**Researched:** 2026-04-05

## Recommended Stack

This is an **additive milestone** on top of an existing Python CLI. The core stack (Python 3.12, anyio, rich, hatchling) is already locked in. This document covers only the new dependencies and SDK features needed for subagent orchestration.

### Core: Claude Agent SDK Subagent API

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| `claude-agent-sdk` | `>=0.1.52` (currently 0.1.56) | Subagent definitions via `AgentDefinition`, `agents` param on `ClaudeAgentOptions` | Native SDK subagents — no custom orchestration framework needed. `AgentDefinition` with `description`, `prompt`, `tools`, `model` fields landed in v0.1.51; `disallowedTools`, `maxTurns` added in v0.1.52. | HIGH |
| `anyio` | `>=4.0` (currently 4.13) | Parallel subagent execution via `create_task_group()` + `CapacityLimiter` | Already used for parallel fixes. Same pattern applies to running multiple exploration subagents concurrently. No new dependency. | HIGH |

**Key API surface:**

```python
from claude_agent_sdk import ClaudeAgentOptions, AgentDefinition

options = ClaudeAgentOptions(
    agents={
        "codebase-explorer": AgentDefinition(
            description="Explores codebase areas affected by changes",
            prompt="You are a codebase exploration specialist...",
            tools=["Read", "Grep", "Glob"],  # Read-only — no edits
            model="sonnet",  # Cheaper model for exploration
        ),
    },
    allowed_tools=["Read", "Grep", "Glob", "Agent"],  # Agent tool required
    # ...existing options...
)
```

**Critical facts about SDK subagents (verified from official docs):**

1. Subagents are invoked via the `Agent` tool — `"Agent"` must be in `allowed_tools`
2. Each subagent gets a fresh context window — no parent conversation leaks
3. Parent receives only the subagent's final message (not intermediate tool calls)
4. Subagents **cannot spawn other subagents** — no recursive nesting
5. `model` field accepts short names only: `"sonnet"`, `"opus"`, `"haiku"`, `"inherit"`
6. Claude auto-delegates based on `description` field, or you can force with "Use the X agent to..."
7. Subagent transcripts survive main conversation compaction (stored separately)

### Async Orchestration (no new dependencies)

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| `anyio.create_task_group()` | 4.x | Fan-out multiple exploration agents | Already in use for `phase_fix_parallel()`. Structured concurrency ensures all subagents complete or cancel together. | HIGH |
| `anyio.CapacityLimiter` | 4.x | Throttle concurrent subagent count | Already in use (limiter of 4). Exploration likely needs 2-4 concurrent agents. | HIGH |

### No New Dependencies Required

The subagent exploration layer requires **zero new pip dependencies**. Everything needed is already in the stack:

- `claude-agent-sdk` — bump minimum from `>=0.1.27` to `>=0.1.52` for `AgentDefinition` + `agents` param
- `anyio` — already present, version sufficient
- `rich` — already present, for exploration progress UI

### Version Bump Required

```toml
# pyproject.toml change:
# FROM:
"claude-agent-sdk>=0.1.27"
# TO:
"claude-agent-sdk>=0.1.52"
```

**Rationale:** v0.1.51 introduced `AgentDefinition` with `skills`, `memory`, `mcpServers`. v0.1.52 added `disallowedTools`, `maxTurns`, `initialPrompt`, and fixed critical `query()` deadlock bugs. The `>=0.1.52` floor ensures all subagent features are available.

## Two API Approaches: `query()` vs `ClaudeSDKClient`

The SDK offers two execution models. The project currently uses `ClaudeSDKClient`.

| Approach | Current Use | Subagent Fit | Recommendation |
|----------|-------------|--------------|----------------|
| `ClaudeSDKClient` (session-based) | Yes — `ClaudeBackend.execute()` | Supports `agents` in `ClaudeAgentOptions`. Subagent events stream through `receive_response()` as `ToolUseBlock` (name="Agent") and `ToolResultBlock`. | **Keep using this.** Already integrated with the event stream and UI. |
| `query()` (one-shot) | No | Also supports `agents`. Simpler API but no multi-turn within a session. | Don't migrate. Would require rewriting `ClaudeBackend`. |

**Decision: Stay with `ClaudeSDKClient`.** The `agents` parameter goes on `ClaudeAgentOptions`, which `ClaudeBackend` already constructs. Subagent invocations will appear as `ToolUseBlock(name="Agent")` and `ToolResultBlock` events in the existing stream.

## Architecture Integration Points

### Backend Protocol Extension

The `Backend` protocol needs an `agents` parameter on `execute()`:

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

Alternatively, pass agents at `ClaudeBackend.__init__()` time. The init-time approach is simpler if agent definitions are static per run; the execute-time approach is needed if different phases use different subagent configurations (more likely given TTT vs normal flow differences).

### Subagent Event Detection

Subagent invocations surface as existing event types — no new event dataclasses needed:

```python
# In ClaudeBackend.execute():
elif isinstance(block, ToolUseBlock):
    if block.name == "Agent":
        # Subagent invoked — can track in UI
        yield ToolStartEvent(id=block.id, name="Agent", input=block.input or {})
    elif block.name == "StructuredOutput":
        continue
    else:
        yield ToolStartEvent(...)
```

### Model Cost Strategy

| Agent Role | Model | Rationale |
|------------|-------|-----------|
| Main review/fix agent | `opus` | Quality-critical — needs to understand code deeply |
| Exploration subagents | `sonnet` | Read-only scanning — fast and cheap. 5x cheaper than opus. |
| Plan generation (TTT) | `opus` | Needs synthesis quality |

Using `model="sonnet"` on `AgentDefinition` for exploration subagents keeps costs manageable when fanning out 3-5 explorers per review.

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Subagent framework | SDK native `AgentDefinition` | Custom multi-process orchestration (e.g., spawning multiple `ClaudeSDKClient` instances in parallel via anyio task groups) | SDK handles context isolation, tool restriction, and model selection natively. Custom approach = more code, more bugs, no access to `Agent` tool's automatic delegation. |
| Subagent framework | SDK native `AgentDefinition` | LangGraph / CrewAI / AutoGen | Massive dependency bloat for a CLI tool. SDK subagents do exactly what's needed. These frameworks are designed for multi-model orchestration — overkill here. |
| Async runtime | anyio (keep) | asyncio directly | anyio already in use, provides structured concurrency (task groups) that raw asyncio lacks. |
| Exploration parallelism | `anyio.create_task_group()` | `asyncio.gather()` | Task groups provide structured cancellation. If one explorer fails, the rest cancel cleanly. `gather()` requires manual exception handling. |
| SDK API | `ClaudeSDKClient` (keep) | `query()` function | Would require rewriting `ClaudeBackend` and losing multi-turn capability. No benefit for this use case. |

## What NOT to Use

| Technology | Why Avoid |
|------------|-----------|
| LangChain / LangGraph | Massive transitive dependency tree. SDK subagents are purpose-built for this exact pattern. |
| CrewAI / AutoGen | Multi-agent frameworks designed for multi-model, multi-provider setups. Daydream is Claude-native. |
| `multiprocessing` / `concurrent.futures` | The Claude Agent SDK spawns subprocesses internally. Adding another process layer creates complexity for no gain. anyio task groups handle concurrency. |
| Custom agent message passing | SDK's `Agent` tool handles parent-to-subagent communication. Don't reinvent it. |
| Filesystem-based `.claude/agents/` definitions | Programmatic `AgentDefinition` is recommended by Anthropic for SDK apps. Filesystem approach is for interactive Claude Code usage. |

## Installation

```bash
# No new packages — just bump the existing claude-agent-sdk minimum
uv sync  # After updating pyproject.toml
```

## Sources

- [Subagents in the SDK - Claude API Docs](https://platform.claude.com/docs/en/agent-sdk/subagents) — Official subagent documentation (HIGH confidence)
- [Agent SDK reference - Python](https://platform.claude.com/docs/en/agent-sdk/python) — Full Python API reference (HIGH confidence)
- [claude-agent-sdk on PyPI](https://pypi.org/project/claude-agent-sdk/) — v0.1.56 as of 2026-04-04 (HIGH confidence)
- [Releases - claude-agent-sdk-python](https://github.com/anthropics/claude-agent-sdk-python/releases) — Changelog: AgentDefinition in v0.1.51, expanded in v0.1.52 (HIGH confidence)
- [AnyIO 4.13.0 documentation - Tasks](https://anyio.readthedocs.io/en/stable/tasks.html) — Task groups and capacity limiters (HIGH confidence)

---

*Stack research: 2026-04-05*
