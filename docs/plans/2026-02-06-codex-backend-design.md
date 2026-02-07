# Codex CLI Backend Support

**Date:** 2026-02-06
**Status:** Draft
**Author:** brainstorming session

## Problem

Daydream is tightly coupled to `claude-agent-sdk`. Users who prefer OpenAI models or want to mix backends (e.g., Codex for review, Claude for fixes) have no path forward.

## Decision

Introduce a backend abstraction layer so daydream can run against either Claude Code SDK or Codex CLI, chosen per-phase at runtime.

## Context

- Codex CLI exposes `codex exec --experimental-json` which streams JSONL events over stdout — the same pattern the Codex TypeScript SDK uses.
- Beagle skills use `SKILL.md` + `references/` format, which is identical to Codex skills. They're already installed at `~/.agents/skills/` where Codex discovers them automatically.
- Daydream's phases (review → parse → fix → test) are prompt-driven. The prompts are mostly backend-agnostic except for skill invocation syntax.

## Architecture

### Unified Event Stream

Both backends yield a common set of events that the UI layer consumes. This replaces direct SDK type handling in `agent.py`.

```
daydream/
  backends/
    __init__.py        # Backend protocol, event types, factory
    claude.py          # Claude Agent SDK wrapper
    codex.py           # Codex CLI subprocess wrapper
  agent.py             # Simplified: consumes event stream, renders UI
  cli.py               # New --backend and --*-backend flags
  runner.py            # Creates backend per-phase
  phases.py            # Adapts prompts per-backend
  config.py            # Unchanged
  ui.py                # Unchanged
```

### Event Types

```python
@dataclass
class TextEvent:
    """Agent text output."""
    text: str

@dataclass
class ThinkingEvent:
    """Extended thinking / reasoning."""
    text: str

@dataclass
class ToolStartEvent:
    """Tool invocation started."""
    id: str
    name: str
    input: dict[str, Any]

@dataclass
class ToolResultEvent:
    """Tool invocation completed."""
    id: str
    output: str
    is_error: bool

@dataclass
class CommandEvent:
    """Shell command execution (Codex-originated)."""
    command: str
    output: str
    exit_code: int | None
    status: str  # "in_progress" | "completed" | "failed"

@dataclass
class FileChangeEvent:
    """File modification (Codex-originated)."""
    path: str
    kind: str  # "add" | "delete" | "update"

@dataclass
class CostEvent:
    """Cost and usage information."""
    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None

@dataclass
class ResultEvent:
    """Structured output result."""
    structured_output: Any | None

AgentEvent = (
    TextEvent | ThinkingEvent | ToolStartEvent | ToolResultEvent
    | CommandEvent | FileChangeEvent | CostEvent | ResultEvent
)
```

### Backend Protocol

```python
class Backend(Protocol):
    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]: ...

    async def cancel(self) -> None: ...
```

### Claude Backend

Extracts the existing `run_agent()` message loop from `agent.py` into a class that yields unified events:

| Claude SDK Type | Daydream Event |
|---|---|
| `TextBlock` | `TextEvent` |
| `ThinkingBlock` | `ThinkingEvent` |
| `ToolUseBlock` | `ToolStartEvent` |
| `ToolResultBlock` | `ToolResultEvent` |
| `ResultMessage.structured_output` | `ResultEvent` |
| `ResultMessage.total_cost_usd` | `CostEvent` |

Implementation wraps `ClaudeSDKClient` with `bypassPermissions` mode, same as today.

### Codex Backend

Spawns `codex exec --experimental-json` as an async subprocess. Pipes the prompt to stdin, reads JSONL events from stdout.

| Codex JSONL Event | Daydream Event |
|---|---|
| `item.completed` + `agent_message` | `TextEvent` |
| `item.completed` + `reasoning` | `ThinkingEvent` |
| `item.started` + `command_execution` | `ToolStartEvent` (name="shell") |
| `item.completed` + `command_execution` | `CommandEvent` |
| `item.completed` + `file_change` | `FileChangeEvent` |
| `item.started` + `mcp_tool_call` | `ToolStartEvent` |
| `item.completed` + `mcp_tool_call` | `ToolResultEvent` |
| `turn.completed` | `CostEvent` (tokens only, no USD) |
| `turn.failed` | raises `CodexError` |

CLI construction:

```python
args = [
    "codex", "exec", "--experimental-json",
    "--model", self.model,
    "--sandbox", "workspace-write",
    "--cd", str(cwd),
]
if output_schema:
    args.extend(["--output-schema", schema_path])
```

Structured output uses a temp file written with the JSON schema, passed via `--output-schema`. The agent's final `agent_message.text` contains the JSON response.

### Cancellation

- **Claude:** calls `client.terminate()` on the SDK client (existing pattern).
- **Codex:** sends SIGTERM to the subprocess, then SIGKILL after timeout.

## CLI Interface

### New Flags

```
--backend, -b       Default backend: "claude" (default) or "codex"
--review-backend    Override backend for review phase
--fix-backend       Override backend for fix phase
--test-backend      Override backend for test phase
```

### RunConfig Changes

```python
@dataclass
class RunConfig:
    # ...existing fields...
    backend: str = "claude"
    review_backend: str | None = None   # None = use default
    fix_backend: str | None = None
    test_backend: str | None = None
```

### Backend Factory

```python
def create_backend(name: str, model: str) -> Backend:
    if name == "codex":
        return CodexBackend(model=model)
    return ClaudeBackend(model=model)
```

The runner resolves the effective backend per-phase:

```python
review_backend_name = config.review_backend or config.backend
review_backend = create_backend(review_backend_name, config.model)
```

## Skill Invocation

Beagle skills and Codex skills share the identical `SKILL.md` format. Beagle skills installed at `~/.agents/skills/` are already discoverable by Codex.

The only difference is invocation syntax:

| Backend | Review Prompt |
|---|---|
| Claude | `/{skill}\n\nWrite the full review output to {path}.` |
| Codex | `Use the {skill_name} skill to review this code. Write findings to {path}.` |

For Claude, skills use the `beagle-python:review-python` namespace. For Codex, we strip the namespace to get the skill name (`review-python`) since Codex discovers skills by directory name.

Fix and test phase prompts are already natural language and work with both backends unchanged.

## Migration Path

### Phase 1: Extract backend abstraction
- Create `backends/` package with protocol and event types
- Extract `ClaudeBackend` from current `agent.py`
- Update `run_agent()` to consume event stream
- No behavior change — Claude remains the only backend

### Phase 2: Implement Codex backend
- Add `CodexBackend` with subprocess + JSONL parsing
- Add CLI flags for backend selection
- Adapt phase prompts for Codex skill invocation

### Phase 3: Per-phase overrides
- Wire `--review-backend`, `--fix-backend`, `--test-backend` flags
- Runner creates separate backends per-phase

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `--experimental-json` flag is experimental | Monitor Codex releases; flag name may stabilize |
| Codex doesn't expose cost in USD | Show token counts instead; calculate cost from pricing if needed |
| Codex skill discovery may not match Beagle namespaces | Strip namespace prefix; verify skill is discovered via prompt feedback |
| Different error semantics between backends | Unified error handling in `run_agent()` wraps backend-specific exceptions |

## Open Questions

1. Should `--model` auto-select appropriate defaults per backend? (e.g., `opus` for Claude, `o3` for Codex)
2. Should we support Codex's `--full-auto` mode as a convenience flag?
3. Should we add a `codex review` mode that uses the built-in `codex review` subcommand instead of `codex exec` with a review skill?
