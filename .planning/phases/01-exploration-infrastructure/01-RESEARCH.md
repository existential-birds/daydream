# Phase 1: Exploration Infrastructure - Research

**Researched:** 2026-04-05
**Domain:** Claude Agent SDK subagents, Python dataclass design, Backend protocol extension
**Confidence:** HIGH

## Summary

This phase builds foundation infrastructure: extend the `Backend` protocol with an `agents` parameter, create `ExplorationContext` data structures, bump the SDK, and wire graceful degradation. All decisions are locked in CONTEXT.md -- no open design questions remain.

The existing codebase is well-structured for this work. The `Backend` protocol in `backends/__init__.py` has a clean 4-parameter `execute()` signature that accepts a 5th kwarg additively. `AgentDefinition` already exists in the current SDK (0.1.27) and is unchanged in 0.1.52 -- the version bump is for broader subagent stability, not for a new type. The `ExplorationContext` dataclass follows the exact pattern used by `AgentState`, `RunConfig`, and all event dataclasses throughout the project.

**Primary recommendation:** Implement in three parallel workstreams (SDK bump, Backend protocol extension, ExplorationContext module) then wire degradation handling as a final integration step.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Add optional `agents: list[AgentDefinition] | None = None` kwarg to `Backend.execute()`. Additive change -- backward compatible with all existing call sites.
- **D-02:** `CodexBackend` silently ignores the `agents` parameter. No error, no warning -- exploration simply doesn't happen with Codex. Phases check the exploration result, not which backend ran.
- **D-03:** `ExplorationContext` uses typed fields -- `affected_files: list[FileInfo]`, `conventions: list[Convention]`, `dependencies: list[Dependency]`, `guidelines: list[str]`, `raw_notes: str`. Not a generic dict or text blob.
- **D-04:** Include a `to_prompt_section() -> str` method that renders the context into text suitable for prompt injection into review agents.
- **D-05:** `ExplorationContext` and its supporting types (`FileInfo`, `Convention`, `Dependency`) live in a new `daydream/exploration.py` module, separate from `backends/`. This is a domain concept, not a backend event.
- **D-06:** When exploration fails (SDK error, exception), show a visible warning banner in the Rich UI: "Exploration failed -- proceeding with review only". Review continues with an empty `ExplorationContext`.
- **D-07:** No artificial timeout on exploration. Subagents run to completion. Degradation handles actual errors (SDK failures, exceptions), not slowness.
- **D-08:** Hard pin `claude-agent-sdk == 0.1.52` in `pyproject.toml`. Reproducibility over convenience.
- **D-09:** Use `AgentDefinition` directly from the SDK -- no wrapper dataclass. `ClaudeBackend` passes it through to `ClaudeSDKClient`. We're already coupled to this SDK; adding indirection would be pointless abstraction.

### Claude's Discretion
- Exact field types for `FileInfo`, `Convention`, `Dependency` supporting dataclasses -- Claude picks what makes sense during planning
- Internal structure of `to_prompt_section()` output format -- whatever produces clear context for downstream review agents

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| INFR-01 | `claude-agent-sdk` bumped to `>=0.1.52` for `AgentDefinition` support | SDK 0.1.52 is available on PyPI. `AgentDefinition` already exists in 0.1.27 but 0.1.52 is required for subagent stability. Hard pin per D-08. |
| INFR-02 | Exploration results aggregated into structured `ExplorationContext` for review prompt injection | New `daydream/exploration.py` module with typed dataclasses per D-03/D-04/D-05. Follows existing `@dataclass` patterns throughout the project. |
| INFR-03 | Exploration degrades gracefully (review proceeds if exploration fails) | `print_warning()` exists in `daydream/ui.py` for banner display per D-06. Empty `ExplorationContext` as fallback. |
| AGNT-03 | Backend protocol extended with `agents` parameter for subagent support | `Backend.execute()` Protocol in `backends/__init__.py` takes additive kwarg. `ClaudeBackend` passes `agents` to `ClaudeAgentOptions`. `CodexBackend` ignores per D-02. |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Verification required:** Must run `make lint`, `make typecheck`, `make test` after changes (ruff, mypy, pytest)
- **Phased execution:** No more than 5 files per phase
- **Dataclass pattern:** All data types use `@dataclass`, mutable defaults use `field(default_factory=...)`
- **Google docstrings:** `Args:`, `Returns:`, `Raises:` on all public functions
- **Import style:** `from __future__ import annotations` in all modules, `TYPE_CHECKING` guard for type-only imports
- **Error handling:** Raise typed exceptions, catch narrowly, no silent swallowing
- **UI output:** All user-facing output via `daydream.ui` functions, never raw `print()`
- **Naming:** `snake_case` for modules/functions, `PascalCase` for classes, private helpers prefixed with `_`
- **50+ existing tests must pass:** 129 tests currently collected

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| claude-agent-sdk | 0.1.52 (hard pin) | Subagent definitions via `AgentDefinition`, `ClaudeAgentOptions.agents` | Only SDK for Claude Code programmatic access; `AgentDefinition` is the native subagent type |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dataclasses (stdlib) | Python 3.12 | `ExplorationContext`, `FileInfo`, `Convention`, `Dependency` | All data structure definitions |
| rich | 13.0+ (existing) | Warning banner for degradation | `print_warning()` already exists in `daydream/ui.py` |

No new dependencies are needed beyond the SDK version bump.

**Installation:**
```bash
# Update pyproject.toml to claude-agent-sdk==0.1.52, then:
uv sync
```

**Version verification:** SDK 0.1.52 confirmed available via `uv pip install --dry-run`. `AgentDefinition` fields verified: `description`, `prompt`, `tools`, `model`.

## Architecture Patterns

### New Module Placement
```
daydream/
├── exploration.py       # NEW: ExplorationContext, FileInfo, Convention, Dependency
├── backends/
│   ├── __init__.py      # MODIFY: Backend.execute() adds agents kwarg
│   ├── claude.py        # MODIFY: pass agents to ClaudeAgentOptions
│   └── codex.py         # MODIFY: accept and ignore agents param
├── agent.py             # MODIFY: run_agent() passes agents through
└── ...
```

### Pattern 1: Additive Protocol Extension
**What:** Add optional kwarg to `Backend.execute()` Protocol method with `None` default
**When to use:** When extending a protocol without breaking existing implementations
**Example:**
```python
# Source: daydream/backends/__init__.py (existing pattern, extended)
from claude_agent_sdk.types import AgentDefinition

class Backend(Protocol):
    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: list[AgentDefinition] | None = None,  # NEW
    ) -> AsyncIterator[AgentEvent]: ...
```

### Pattern 2: Domain Dataclass with Render Method
**What:** Dataclass with a `to_prompt_section()` method that produces structured text for prompt injection
**When to use:** When data needs to be serialized into natural language for LLM consumption
**Example:**
```python
# Source: project conventions (dataclass pattern from AgentState, RunConfig)
from dataclasses import dataclass, field

@dataclass
class FileInfo:
    """Information about a file relevant to the review."""
    path: str
    role: str  # "modified", "imported_by", "imports", "test"
    summary: str = ""

@dataclass
class Convention:
    """A codebase convention or pattern."""
    name: str
    description: str
    source: str = ""  # where it was found (e.g., "CLAUDE.md", "inferred from code")

@dataclass
class Dependency:
    """A dependency relationship between files."""
    source: str  # file that depends
    target: str  # file depended upon
    relationship: str  # "imports", "calls", "extends", "tests"

@dataclass
class ExplorationContext:
    """Aggregated exploration results for review prompt injection."""
    affected_files: list[FileInfo] = field(default_factory=list)
    conventions: list[Convention] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)
    guidelines: list[str] = field(default_factory=list)
    raw_notes: str = ""

    def to_prompt_section(self) -> str:
        """Render exploration context as text for prompt injection."""
        ...
```

### Pattern 3: Graceful Degradation via Empty Default
**What:** When an operation fails, return an empty instance of the expected type rather than propagating the error
**When to use:** When a failure in one subsystem should not block the primary workflow
**Example:**
```python
# Degradation pattern for exploration
try:
    context = await run_exploration(backend, cwd, diff)
except Exception:
    print_warning(console, "Exploration failed -- proceeding with review only")
    context = ExplorationContext()  # empty, all defaults
```

### Anti-Patterns to Avoid
- **Wrapping `AgentDefinition`:** D-09 explicitly forbids this. Use the SDK type directly.
- **Raising on Codex `agents`:** D-02 says silently ignore, not raise or warn.
- **Adding timeout:** D-07 says no artificial timeout. Let subagents run.
- **Putting exploration types in `backends/`:** D-05 says `exploration.py` is a domain module, not a backend concern.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Subagent definitions | Custom agent config type | `AgentDefinition` from SDK | Already has `description`, `prompt`, `tools`, `model` fields |
| Warning display | Custom print/logging | `print_warning()` from `daydream.ui` | Existing Rich-styled warning banner with Dracula theme |
| Async iteration | Custom event loop | `async for event in backend.execute(...)` | Existing pattern handles all event types |

## Common Pitfalls

### Pitfall 1: Protocol Structural Typing vs Default Args
**What goes wrong:** Adding a kwarg to the Protocol definition doesn't automatically make it optional in implementations. Each concrete class must also have the default.
**Why it happens:** Python Protocol uses structural typing -- the Protocol defines the interface shape, but each class must independently declare the default value.
**How to avoid:** Add `agents: list[AgentDefinition] | None = None` to ALL three signatures: `Backend.execute()`, `ClaudeBackend.execute()`, `CodexBackend.execute()`.
**Warning signs:** `mypy` will catch this -- a concrete class missing the default will fail structural type checking.

### Pitfall 2: Circular Import with AgentDefinition
**What goes wrong:** Importing `AgentDefinition` from `claude_agent_sdk.types` in `backends/__init__.py` could cause import issues since `__init__.py` is imported by everyone.
**Why it happens:** `backends/__init__.py` is the package init, imported by `claude.py` and `codex.py` which also import from `claude_agent_sdk`.
**How to avoid:** Use `TYPE_CHECKING` guard for `AgentDefinition` import in `backends/__init__.py` since it's only needed for the type annotation. At runtime, the parameter defaults to `None` so the type is never instantiated here.
**Warning signs:** Import errors at module load time.

### Pitfall 3: Breaking Existing Call Sites
**What goes wrong:** If `run_agent()` in `agent.py` doesn't pass `agents` through, or passes it positionally, existing calls break.
**Why it happens:** `backend.execute(cwd, prompt, output_schema, continuation)` is called positionally in `agent.py` line 310.
**How to avoid:** Add `agents` as a keyword-only parameter to `run_agent()` and pass it as a keyword to `backend.execute()`. All 15+ existing call sites in `phases.py` don't use `agents` and won't need changes.
**Warning signs:** Any of the 129 existing tests failing.

### Pitfall 4: SDK Version Constraint Conflict
**What goes wrong:** Changing from `>=0.1.27` to `==0.1.52` might conflict with other packages depending on `claude-agent-sdk`.
**Why it happens:** Hard pins can cause dependency resolution failures.
**How to avoid:** Run `uv sync` after the change and verify no resolution errors. Currently only `daydream` depends on this package.
**Warning signs:** `uv sync` fails or `uv.lock` shows conflicts.

### Pitfall 5: Empty ExplorationContext Serialization
**What goes wrong:** `to_prompt_section()` on an empty `ExplorationContext` could produce noisy/misleading prompt text ("No files found, no conventions, no dependencies...").
**Why it happens:** Not handling the empty case specially.
**How to avoid:** When all fields are empty/default, `to_prompt_section()` should return an empty string so it adds nothing to the review prompt.
**Warning signs:** Review prompts containing placeholder text about missing exploration data.

## Code Examples

Verified patterns from the existing codebase:

### SDK AgentDefinition Usage
```python
# Source: .claude/skills/claude-agent-sdk/REFERENCE.md
# Verified: AgentDefinition exists in SDK 0.1.27, fields confirmed via runtime introspection
from claude_agent_sdk.types import AgentDefinition

agent = AgentDefinition(
    description="Reviews code for bugs and style",
    prompt="You are a code reviewer. Check for bugs, security issues, and style.",
    tools=["Read", "Grep", "Glob"],
    model="sonnet"
)

# Passed to ClaudeAgentOptions
options = ClaudeAgentOptions(
    agents={"code-reviewer": agent},
    # ... other options
)
```

### Existing Backend.execute() Call in agent.py
```python
# Source: daydream/agent.py line 310 (current code)
event_iter = backend.execute(cwd, prompt, output_schema, continuation)
# Must become:
event_iter = backend.execute(cwd, prompt, output_schema, continuation, agents=agents)
```

### ClaudeBackend Passing agents to SDK
```python
# Source: daydream/backends/claude.py (pattern to follow)
options = ClaudeAgentOptions(
    cwd=str(cwd),
    permission_mode="bypassPermissions",
    setting_sources=["user", "project", "local"],
    model=self.model,
    output_format=output_format,
    max_buffer_size=10 * 1024 * 1024,
    agents=self._build_agents_dict(agents),  # NEW: convert list to dict
)
```

### Warning Banner for Degradation
```python
# Source: daydream/ui.py line 1447 (existing function)
from daydream.ui import print_warning, create_console
console = create_console()
print_warning(console, "Exploration failed -- proceeding with review only")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single-agent review | Subagent-based exploration + review | SDK 0.1.x introduced `AgentDefinition` | Enables pre-scan exploration before review |
| `claude-agent-sdk>=0.1.27` (range) | `claude-agent-sdk==0.1.52` (hard pin) | This phase | Reproducible builds, known subagent behavior |

## Open Questions

1. **AgentDefinition `agents` dict key naming**
   - What we know: `ClaudeAgentOptions.agents` takes `dict[str, AgentDefinition]` where keys are agent names
   - What's unclear: When `Backend.execute()` receives `list[AgentDefinition]`, how to convert to the dict format the SDK expects (need a key per agent)
   - Recommendation: Use `description` field or a sequential key like `"explorer-0"`, `"explorer-1"`. The key is only used internally by the SDK for tool routing. Planner should decide the naming scheme.

2. **`to_prompt_section()` output format**
   - What we know: Must produce text suitable for prompt injection
   - What's unclear: Exact formatting (markdown headers? plain text? XML tags?)
   - Recommendation: Use markdown with clear section headers. This is Claude's discretion per CONTEXT.md.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ with pytest-asyncio 0.24+ |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest -x -q` |
| Full suite command | `uv run pytest -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INFR-01 | `AgentDefinition` imports from SDK 0.1.52 | unit | `uv run pytest tests/test_backends_init.py -x` | Wave 0: add import test |
| INFR-02 | `ExplorationContext` instantiation + `to_prompt_section()` | unit | `uv run pytest tests/test_exploration.py -x` | Wave 0: new file |
| INFR-03 | Exploration failure produces empty context, does not raise | unit | `uv run pytest tests/test_exploration.py::test_degradation -x` | Wave 0: new file |
| AGNT-03 | `Backend.execute()` accepts `agents` kwarg, existing calls unbroken | unit + regression | `uv run pytest tests/test_backends_init.py tests/test_backend_claude.py tests/test_backend_codex.py -x` | Existing files, add new tests |

### Sampling Rate
- **Per task commit:** `uv run pytest -x -q`
- **Per wave merge:** `uv run pytest -v && uv run ruff check daydream && uv run mypy daydream`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_exploration.py` -- covers INFR-02, INFR-03 (ExplorationContext, degradation)
- [ ] Add `agents` kwarg tests to `tests/test_backends_init.py` -- covers AGNT-03
- [ ] Add `agents` kwarg tests to `tests/test_backend_claude.py` -- covers AGNT-03 (ClaudeBackend)
- [ ] Add `agents` kwarg tests to `tests/test_backend_codex.py` -- covers AGNT-03 (CodexBackend ignores)
- [ ] Add SDK import verification test -- covers INFR-01

## Sources

### Primary (HIGH confidence)
- `daydream/backends/__init__.py` -- Current Backend protocol, verified 4-param execute() signature
- `daydream/backends/claude.py` -- Current ClaudeBackend, verified ClaudeAgentOptions usage
- `daydream/backends/codex.py` -- Current CodexBackend, verified execute() signature
- `daydream/agent.py` -- Current run_agent(), verified single call site to backend.execute() at line 310
- `.claude/skills/claude-agent-sdk/REFERENCE.md` -- AgentDefinition fields, ClaudeAgentOptions.agents parameter
- Runtime introspection -- AgentDefinition fields confirmed: description, prompt, tools, model
- `uv pip install --dry-run claude-agent-sdk==0.1.52` -- version availability confirmed

### Secondary (MEDIUM confidence)
- `.claude/skills/claude-agent-sdk/SKILL.md` -- SDK overview and usage patterns

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- single SDK dependency, version confirmed available, AgentDefinition verified
- Architecture: HIGH -- all touched files read, patterns extracted from existing code, only 4-5 files touched
- Pitfalls: HIGH -- identified from direct code analysis (Protocol typing, circular imports, call site compatibility)

**Research date:** 2026-04-05
**Valid until:** 2026-05-05 (stable -- SDK is pinned, patterns are internal)
