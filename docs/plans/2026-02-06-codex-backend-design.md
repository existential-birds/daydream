# Codex CLI Backend Support

**Date:** 2026-02-06
**Updated:** 2026-02-07
**Status:** Approved
**Author:** brainstorming session

## Problem

Daydream is tightly coupled to `claude-agent-sdk`. Users who prefer OpenAI models or want to mix backends (e.g., Codex for review, Claude for fixes) have no path forward.

## Decision

Introduce a backend abstraction layer so daydream can run against either Claude Code SDK or Codex CLI, chosen per-phase at runtime.

## Context

- Codex CLI exposes `codex exec --experimental-json` which streams JSONL events over stdout — the same pattern the Codex TypeScript SDK uses.
- Beagle skills use `SKILL.md` + `references/` format, which is identical to Codex skills. They're already installed at `~/.agents/skills/` where Codex discovers them automatically.
- Daydream's phases (review → parse → fix → test, plus PR feedback phases) are prompt-driven. The prompts are backend-agnostic except for skill invocation syntax.
- The Codex TypeScript SDK writes the prompt to stdin, closes it immediately, then reads JSONL line-by-line from stdout via `readline`. Our Python backend follows the same pattern.

## Architecture

### Unified Event Stream

Both backends yield a common set of events that the UI layer consumes. This replaces direct SDK type handling in `agent.py`. All Codex-specific events (command execution, file changes) are normalized to the `ToolStartEvent`/`ToolResultEvent` pair so the UI doesn't need backend-specific panel types.

```
daydream/
  backends/
    __init__.py        # Backend protocol, event types, factory
    claude.py          # Claude Agent SDK wrapper
    codex.py           # Codex CLI subprocess wrapper
  agent.py             # Simplified: consumes event stream, renders UI
  cli.py               # New --backend and --*-backend flags
  runner.py            # Creates backend per-phase
  phases.py            # Accepts Backend, uses format_skill_invocation()
  config.py            # Unchanged (SKILL_MAP used by backends)
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
class CostEvent:
    """Cost and usage information."""
    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None

@dataclass
class ContinuationToken:
    """Opaque token for multi-turn interactions."""
    backend: str
    data: dict[str, Any]

@dataclass
class ResultEvent:
    """Final event in the stream. Carries structured output and continuation token."""
    structured_output: Any | None
    continuation: ContinuationToken | None

AgentEvent = (
    TextEvent | ThinkingEvent | ToolStartEvent
    | ToolResultEvent | CostEvent | ResultEvent
)
```

No `CommandEvent` or `FileChangeEvent` — Codex command executions and file changes are normalized to `ToolStartEvent`/`ToolResultEvent` pairs. This keeps the UI layer backend-agnostic.

### Backend Protocol

Hybrid stateless/continuation model. Each call to `execute()` is independent by default. Backends can optionally return a `ContinuationToken` in the `ResultEvent` that callers pass back for multi-turn interactions (e.g., the test-and-heal retry loop). Claude ignores the token; Codex uses it for thread resumption.

```python
class Backend(Protocol):
    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
    ) -> AsyncIterator[AgentEvent]: ...

    async def cancel(self) -> None: ...

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str: ...
```

### Claude Backend

Extracts the existing `run_agent()` message loop from `agent.py` into a class that yields unified events:

```python
class ClaudeBackend:
    def __init__(self, model: str = "opus"):
        self.model = model
        self.client: ClaudeSDKClient | None = None
```

| Claude SDK Type | Daydream Event |
|---|---|
| `TextBlock` | `TextEvent` |
| `ThinkingBlock` | `ThinkingEvent` |
| `ToolUseBlock` | `ToolStartEvent` |
| `ToolResultBlock` | `ToolResultEvent` |
| `ResultMessage.structured_output` | `ResultEvent` |
| `ResultMessage.total_cost_usd` | `CostEvent` |

Implementation wraps `ClaudeSDKClient` with `bypassPermissions` mode, same as today. Continuation tokens are ignored — Claude doesn't use thread resumption.

Skill invocation: `/{beagle-python:review-python}` (slash + namespaced name from `SKILL_MAP`).

### Codex Backend

Spawns `codex exec --experimental-json` as an async subprocess. Writes the prompt to stdin all at once, closes stdin immediately, then reads JSONL events line-by-line from stdout.

```python
class CodexBackend:
    def __init__(self, model: str = "gpt-5.3-codex"):
        self.model = model
        self.process: asyncio.subprocess.Process | None = None
        self.thread_id: str | None = None
```

CLI construction:

```python
args = [
    "codex", "exec", "--experimental-json",
    "--model", self.model,
    "--sandbox", "danger-full-access",
    "--cd", str(cwd),
]
if output_schema:
    schema_path = write_temp_schema(output_schema)
    args.extend(["--output-schema", schema_path])
if continuation and continuation.backend == "codex":
    args.extend(["resume", continuation.data["thread_id"]])
```

Structured output uses a temp file written with the JSON schema, passed via `--output-schema`. The agent's final `agent_message.text` contains the JSON response. Temp file is cleaned up after the turn completes.

Sandbox mode is `danger-full-access` to match Claude's `bypassPermissions` behavior — full parity between backends.

Skill invocation: `$review-python` (dollar sign + directory name, stripped from `SKILL_MAP` namespace). Codex discovers skills from `~/.agents/skills/` by directory name.

#### Codex JSONL Event Mapping

| Codex JSONL Event | Daydream Event |
|---|---|
| `thread.started` | Captures `thread_id` for continuation token |
| `item.started` + `agent_message` | `TextEvent(text="")` |
| `item.completed` + `agent_message` | `TextEvent(text=item.text)` |
| `item.started` + `reasoning` | `ThinkingEvent(text="")` |
| `item.completed` + `reasoning` | `ThinkingEvent(text=item.text)` |
| `item.started` + `command_execution` | `ToolStartEvent(id, name="shell", input={"command": item.command})` |
| `item.completed` + `command_execution` | `ToolResultEvent(id, output=aggregated_output, is_error=exit_code!=0)` |
| `item.completed` + `file_change` | Synthetic `ToolStartEvent(name="patch")` + `ToolResultEvent(output=formatted_paths)` |
| `item.started` + `mcp_tool_call` | `ToolStartEvent(id, name=item.tool, input=item.arguments)` |
| `item.completed` + `mcp_tool_call` | `ToolResultEvent(id, output=result.content, is_error=bool(item.error))` |
| `turn.completed` | `CostEvent(cost_usd=None, input_tokens, output_tokens)` then `ResultEvent` with continuation token |
| `turn.failed` | Raises `CodexError(error.message)` |

Notable Codex JSONL behaviors discovered during exploration:

- **`file_change` has no `item.started` event** — only `item.completed` is emitted. The backend emits a synthetic `ToolStartEvent`/`ToolResultEvent` pair together.
- **`command_execution` output is not streamed** — `aggregated_output` is empty at `item.started` and only populated at `item.completed`. No `item.updated` events for output deltas (there's a TODO in the Codex codebase).
- **`command_execution` has a `declined` status** — emitted when the sandbox denies a command.
- **`todo_list` uses `item.updated`** — the only item type that emits update events. Could map to progress tracking in future.

### Cancellation

- **Claude:** calls `client.terminate()` on the SDK client (existing pattern).
- **Codex:** sends SIGTERM to the subprocess, then SIGKILL after timeout. Matches how the Codex TypeScript SDK handles `AbortSignal` — Node's `spawn()` sends SIGTERM when the signal aborts.

## agent.py Changes

`run_agent()` simplifies to an event consumer. It no longer imports Claude SDK types — it only knows about `AgentEvent`. Returns a tuple of `(output, continuation_token)`.

```python
async def run_agent(
    backend: Backend,
    cwd: Path,
    prompt: str,
    output_schema: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    continuation: ContinuationToken | None = None,
) -> tuple[str | Any, ContinuationToken | None]:
```

The `progress_callback` path is preserved for parallel fix agents. When a callback is provided, `TextEvent` and `ToolStartEvent` route to the callback instead of the live UI panels. `ParallelFixPanel` continues to work unchanged.

`AgentState.current_clients` becomes `AgentState.current_backends: list[Backend]`. The signal handler calls `backend.cancel()` instead of `client.terminate()`.

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
def create_backend(name: str, model: str | None = None) -> Backend:
    if name == "codex":
        return CodexBackend(model=model or "gpt-5.3-codex")
    return ClaudeBackend(model=model or "opus")
```

The runner resolves the effective backend per-phase:

```python
review_backend_name = config.review_backend or config.backend
review_backend = create_backend(review_backend_name, config.model or None)
```

### Model Handling

Each backend has its own default model — Claude uses `opus`, Codex uses `gpt-5.3-codex`. The `--model` flag is a raw passthrough: the value is forwarded verbatim to whichever backend is active. If the user doesn't pass `--model`, each backend uses its own default. If switching backends, the user is responsible for passing a valid model ID.

## Skill Invocation

Beagle skills and Codex skills share the identical `SKILL.md` format. Beagle skills installed at `~/.agents/skills/` are already discoverable by Codex.

Each backend owns its invocation syntax via `format_skill_invocation()`:

| Backend | Syntax | Example |
|---|---|---|
| Claude | `/{namespace:skill}` | `/beagle-python:review-python` |
| Codex | `$skill-name` | `$review-python` |

Claude uses the full namespaced name from `SKILL_MAP`. Codex strips the namespace prefix (splits on `:`, takes the last part) since Codex discovers skills by directory name.

Fix and test phase prompts are already natural language and work with both backends unchanged. PR feedback phases (fetch, respond, commit-push) follow the same skill invocation pattern — no special casing.

## Migration Path

### Phase 1: Extract backend abstraction

No behavior change. Claude remains the only backend. The abstraction is proven.

- Create `daydream/backends/__init__.py` — `Backend` protocol, event dataclasses, `ContinuationToken`, `create_backend()` factory
- Create `daydream/backends/claude.py` — Extract `ClaudeBackend` from current `agent.py`
- Simplify `agent.py` — `run_agent()` consumes event stream, renders UI. Remove SDK imports.
- Update `phases.py` — Accept `Backend` parameter, use `backend.format_skill_invocation()`
- Update `runner.py` — Create backend, pass to phases
- Update tests — Mock at the `Backend` level instead of `ClaudeSDKClient`

### Phase 2: Implement Codex backend

- Create `daydream/backends/codex.py` — Subprocess management, JSONL parsing, event mapping
- Wire `--backend` flag in CLI and `RunConfig`
- Add Codex-specific tests with canned JSONL fixtures

### Phase 3: Per-phase overrides and continuation tokens

- Wire `--review-backend`, `--fix-backend`, `--test-backend` flags
- Implement `ContinuationToken` in `CodexBackend` (thread resumption)
- Wire continuation through `phase_test_and_heal` retry loop

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `--experimental-json` flag is experimental | Monitor Codex releases; flag name may stabilize |
| Codex doesn't expose cost in USD | Show token counts instead; calculate cost from pricing if needed |
| Codex skill discovery may not match Beagle namespaces | Strip namespace prefix; verify skill is discovered via prompt feedback |
| Different error semantics between backends | Unified error handling in `run_agent()` wraps backend-specific exceptions |
| `file_change` has no `item.started` event | Emit synthetic start/result pair together in the Codex backend |
| `command_execution` output not streamed | UI shows throbber until completion; no incremental output display |
| `command_execution` can be `declined` by sandbox | Map to `ToolResultEvent(is_error=True)` with descriptive message |

## Resolved Questions

1. **Model defaults per backend:** Each backend has its own default (Claude: `opus`, Codex: `gpt-5.3-codex`). `--model` is a raw passthrough — no abstract tier mapping.
2. **Codex full-auto mode:** Always use `danger-full-access` sandbox to match Claude's `bypassPermissions`. No extra flag needed.
3. **Codex review subcommand:** Always use `codex exec` with Beagle skills. The dedicated `codex review` subcommand bypasses Beagle and isn't used.
