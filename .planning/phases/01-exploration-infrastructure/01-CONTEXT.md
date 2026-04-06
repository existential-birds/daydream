# Phase 1: Exploration Infrastructure - Context

**Gathered:** 2026-04-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the safe, structured foundation for all exploration work: extend the Backend protocol with subagent support, create the ExplorationContext data structures, bump the SDK version, and ensure graceful degradation when exploration fails. No actual exploration logic — that's Phase 2.

</domain>

<decisions>
## Implementation Decisions

### Backend Protocol Extension
- **D-01:** Add optional `agents: list[AgentDefinition] | None = None` kwarg to `Backend.execute()`. Additive change — backward compatible with all existing call sites.
- **D-02:** `CodexBackend` silently ignores the `agents` parameter. No error, no warning — exploration simply doesn't happen with Codex. Phases check the exploration result, not which backend ran.

### ExplorationContext Data Structures
- **D-03:** `ExplorationContext` uses typed fields — `affected_files: list[FileInfo]`, `conventions: list[Convention]`, `dependencies: list[Dependency]`, `guidelines: list[str]`, `raw_notes: str`. Not a generic dict or text blob.
- **D-04:** Include a `to_prompt_section() -> str` method that renders the context into text suitable for prompt injection into review agents.
- **D-05:** `ExplorationContext` and its supporting types (`FileInfo`, `Convention`, `Dependency`) live in a new `daydream/exploration.py` module, separate from `backends/`. This is a domain concept, not a backend event.

### Degradation Behavior
- **D-06:** When exploration fails (SDK error, exception), show a visible warning banner in the Rich UI: "Exploration failed — proceeding with review only". Review continues with an empty `ExplorationContext`.
- **D-07:** No artificial timeout on exploration. Subagents run to completion. Degradation handles actual errors (SDK failures, exceptions), not slowness.

### SDK Version Strategy
- **D-08:** Hard pin `claude-agent-sdk == 0.1.52` in `pyproject.toml`. Reproducibility over convenience.
- **D-09:** Use `AgentDefinition` directly from the SDK — no wrapper dataclass. `ClaudeBackend` passes it through to `ClaudeSDKClient`. We're already coupled to this SDK; adding indirection would be pointless abstraction.

### Claude's Discretion
- Exact field types for `FileInfo`, `Convention`, `Dependency` supporting dataclasses — Claude picks what makes sense during planning
- Internal structure of `to_prompt_section()` output format — whatever produces clear context for downstream review agents

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Backend Protocol
- `daydream/backends/__init__.py` — Current `Backend` protocol definition, `AgentEvent` union, all event dataclasses
- `daydream/backends/claude.py` — `ClaudeBackend` implementation, SDK imports and usage patterns

### SDK
- `pyproject.toml` — Current dependency versions, build configuration

### Architecture
- `.planning/codebase/ARCHITECTURE.md` — Full architecture analysis, data flow, layer responsibilities
- `.planning/codebase/STACK.md` — Technology stack, SDK version info
- `.planning/REQUIREMENTS.md` — INFR-01, INFR-02, INFR-03, AGNT-03 requirement definitions

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Backend` protocol in `backends/__init__.py` — clean 4-param `execute()` signature to extend with 5th kwarg
- `AgentEvent` dataclass union — pattern for typed event vocabulary (can inform `ExplorationContext` field types)
- `create_backend()` factory — no changes needed, just passes through to backend constructors

### Established Patterns
- `@dataclass` for all data types — `ExplorationContext` follows this convention
- `TYPE_CHECKING` guard for iterator types — same pattern for new type imports
- `from __future__ import annotations` in backends — use in `exploration.py`
- Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections on all public functions

### Integration Points
- `Backend.execute()` signature — add `agents` kwarg here
- `ClaudeBackend.execute()` — pass `agents` through to `ClaudeSDKClient`
- `CodexBackend.execute()` — accept and ignore `agents` param
- `run_agent()` in `agent.py` — caller that invokes `backend.execute()`, will pass agents through
- `pyproject.toml` dependencies — bump `claude-agent-sdk` version

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-exploration-infrastructure*
*Context gathered: 2026-04-05*
