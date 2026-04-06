# Phase 1: Exploration Infrastructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-05
**Phase:** 01-exploration-infrastructure
**Areas discussed:** Protocol extension, ExplorationContext shape, Degradation behavior, SDK integration

---

## Protocol Extension

| Option | Description | Selected |
|--------|-------------|----------|
| Add agents kwarg | Add optional `agents: list[AgentDefinition] \| None = None` to execute(). Minimal change, backward compatible. | ✓ |
| New execute_with_agents() method | Separate method for exploration-aware calls. Keeps execute() untouched but adds protocol surface area. | |
| Config object pattern | Bundle all execution options into an ExecutionConfig dataclass. Future-proof but bigger refactor. | |

**User's choice:** Add agents kwarg (Recommended)
**Notes:** None

### Follow-up: CodexBackend behavior with agents param

| Option | Description | Selected |
|--------|-------------|----------|
| Silently ignore | CodexBackend ignores agents param — exploration just doesn't happen with Codex. | ✓ |
| Log a warning | CodexBackend logs a visible warning that subagents aren't supported, then proceeds. | |
| Raise NotImplementedError | Fail explicitly if someone tries to use agents with Codex. | |

**User's choice:** Silently ignore (Recommended)
**Notes:** None

---

## ExplorationContext Shape

| Option | Description | Selected |
|--------|-------------|----------|
| Typed fields | Named fields for each exploration dimension — affected_files, conventions, dependencies, guidelines. Typed and documented. | ✓ |
| Minimal with render | Just raw exploration text plus a render method. Less structure upfront, easier to iterate. | |
| You decide | Claude picks the right level of structure during planning. | |

**User's choice:** Typed fields (Recommended)
**Notes:** None

### Follow-up: Module location

| Option | Description | Selected |
|--------|-------------|----------|
| Own module (exploration.py) | New daydream/exploration.py module. ExplorationContext is a domain concept, not a backend event. | ✓ |
| In backends/__init__.py | Co-locate with other dataclasses. Everything backend-adjacent stays together. | |
| You decide | Claude picks the best location during planning based on import dependencies. | |

**User's choice:** Own module (Recommended)
**Notes:** None

---

## Degradation Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Warning banner | Show a visible warning in Rich UI when exploration fails. Review continues. | ✓ |
| Silent fallback | Log to debug file only. User never sees a failure unless they check debug logs. | |
| Debug-log + quiet indicator | Full details in debug log, subtle "(no exploration)" tag next to review phase header. | |

**User's choice:** Warning banner (Recommended)
**Notes:** None

### Follow-up: Exploration timeout

| Option | Description | Selected |
|--------|-------------|----------|
| 30 seconds | Generous for parallel subagents to scan a typical project. | |
| 60 seconds | More headroom for large codebases. | |
| You decide | Claude picks a reasonable default and makes it configurable. | |

**User's choice:** (Other) "Why would we time out? Just let them finish."
**Notes:** No artificial timeout. Subagents run to completion. Degradation handles actual errors (SDK failures, exceptions), not slowness.

---

## SDK Integration

| Option | Description | Selected |
|--------|-------------|----------|
| Direct SDK types | Use AgentDefinition directly from the SDK. No wrapper layer. | ✓ |
| Thin wrapper dataclass | Own ExplorerAgent dataclass that maps to AgentDefinition inside ClaudeBackend. | |
| You decide | Claude picks based on what the SDK actually exposes after the version bump. | |

**User's choice:** Direct SDK types (Recommended)
**Notes:** None

### Follow-up: Version pinning strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Floor >= 0.1.52 | Minimum version. Users get bugfixes automatically. Matches existing pattern. | |
| Hard pin == 0.1.52 | Exact version for reproducibility. Prevents surprise breakage. | ✓ |
| Compatible release ~= 0.1.52 | Allows 0.1.x patches but not 0.2.0. | |

**User's choice:** Hard pin == 0.1.52
**Notes:** Reproducibility over convenience.

---

## Claude's Discretion

- Exact field types for FileInfo, Convention, Dependency supporting dataclasses
- Internal structure of `to_prompt_section()` output format

## Deferred Ideas

None — discussion stayed within phase scope
