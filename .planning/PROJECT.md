# Daydream

## What This Is

An automated code review and fix loop using the Claude Agent SDK. Daydream launches review agents equipped with Beagle skills to review code, parse actionable feedback, apply fixes automatically, and validate changes. It also has a "trust the technology" mode (`--ttt`) that does stack-agnostic PR review: understand intent, evaluate alternatives, and generate implementation plans.

## Core Value

Reviews and recommendations must be grounded in actual codebase understanding — not guesses based on the diff alone.

## Requirements

### Validated

- Review loop: review, parse feedback, fix, test, commit (existing)
- Backend abstraction: Claude and Codex backends via `Backend` protocol (existing)
- PR feedback mode: `--pr` flag for GitHub PR comment review (existing)
- Trust-the-technology mode: `--ttt` for stack-agnostic 3-phase review (existing)
- Rich terminal UI with Dracula theme and live-updating panels (existing)
- Structured output schemas for feedback parsing and alternative review (existing)
- Per-phase backend overrides (existing)
- Parallel fix mode with `--parallel-fixes` (existing)

### Active

- [ ] Subagent-powered codebase exploration before review (pre-scan affected areas)
- [ ] On-demand subagent spawning during review when agent encounters uncertainty
- [ ] Review recommendations that reference actual codebase patterns and conventions
- [ ] Elimination of recommendations that contradict existing patterns
- [ ] Generated plans that are executable because they understand the full codebase context
- [ ] Unified exploration architecture shared between TTT and normal review flows
- [ ] Use Claude Agent SDK's native subagent capabilities for orchestration

### Out of Scope

- Building a custom orchestration framework — use SDK subagents instead
- Changing the Beagle skill system — skills are external, review quality comes from exploration
- Real-time collaboration features — this is a CLI tool
- Web UI or dashboard — terminal-first

## Context

The current TTT flow (`run_trust` in `runner.py`) has three phases:
1. **LISTEN** (`phase_understand_intent`) — reads diff + git log, presents intent, user confirms
2. **WONDER** (`phase_alternative_review`) — evaluates implementation, returns structured issues
3. **ENVISION** (`phase_generate_plan`) — generates implementation plan for selected issues

The problem: phases 2 and 3 operate primarily on the diff without deeply exploring the surrounding codebase. This leads to:
- **Missing context**: recommendations that break things or duplicate existing work because the agent didn't explore related code
- **Surface-level analysis**: finds superficial issues but misses real architectural problems
- **Bad plans**: generated plans aren't executable because they don't account for actual codebase patterns

The normal review flow has similar issues — the review agent gets a skill invocation and the target directory but doesn't systematically explore affected areas before reviewing.

The fix: introduce a subagent exploration layer that maps the affected codebase areas before review starts, and spawns additional explorers on demand when the review agent encounters uncertainty. This should use the Claude Agent SDK's native subagent capabilities rather than custom orchestration.

Architecture: `cli.py -> runner.py -> phases.py -> agent.py -> backends/`
- Backend protocol with `ClaudeBackend` and `CodexBackend`
- All agent calls go through `run_agent()` which drives the Rich UI
- Module-level singleton state via `AgentState` dataclass

## Constraints

- **SDK**: Must use `claude-agent-sdk` for subagent capabilities — no custom orchestration framework
- **Backends**: Exploration must work through the `Backend` protocol (or extend it cleanly)
- **Existing tests**: 50+ tests must continue passing — don't break existing flows
- **CLI interface**: `--ttt` flag and normal flow must both benefit from exploration

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use SDK subagents, not custom orchestration | Less code to maintain, leverage SDK improvements | Validated in Phase 1 — Backend.execute() accepts AgentDefinition lists |
| Unified exploration for TTT + normal flows | Avoid duplicating exploration logic across flows | -- Pending |
| Pre-scan + on-demand hybrid | Pre-scan catches known areas, on-demand handles surprises | -- Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-06 after Phase 1 completion — exploration infrastructure in place*
