---
phase: 01-exploration-infrastructure
plan: 01
subsystem: infra
tags: [claude-agent-sdk, backend-protocol, subagent, agentdefinition]

requires: []
provides:
  - "Backend.execute() agents kwarg for subagent support"
  - "claude-agent-sdk pinned at 0.1.52"
  - "run_agent() agents forwarding to backend"
affects: [02-pre-scan-exploration, 03-review-integration, 04-on-demand-exploration]

tech-stack:
  added: ["claude-agent-sdk==0.1.52 (pinned from >=0.1.27)"]
  patterns: ["Additive protocol extension with optional kwarg", "TYPE_CHECKING guard for SDK type imports"]

key-files:
  created: []
  modified:
    - pyproject.toml
    - daydream/backends/__init__.py
    - daydream/backends/claude.py
    - daydream/backends/codex.py
    - daydream/agent.py
    - tests/test_backends_init.py
    - tests/test_backend_claude.py
    - tests/test_backend_codex.py
    - tests/test_integration.py
    - tests/test_phases.py
    - tests/test_loop.py

key-decisions:
  - "Conditional assignment for agents on ClaudeAgentOptions instead of dict unpacking (mypy compatibility)"
  - "Added from __future__ import annotations to agent.py for TYPE_CHECKING guard compatibility"

patterns-established:
  - "Additive protocol extension: add optional kwarg with None default to Backend.execute() and all implementations"
  - "Agent dict keying: list[AgentDefinition] converted to dict with explorer-{i} keys for SDK"

requirements-completed: [INFR-01, AGNT-03]

duration: 6min
completed: 2026-04-06
---

# Phase 01 Plan 01: SDK Bump and Backend Agents Kwarg Summary

**claude-agent-sdk pinned to 0.1.52 with Backend protocol extended to accept AgentDefinition lists for subagent spawning**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-06T04:06:39Z
- **Completed:** 2026-04-06T04:12:51Z
- **Tasks:** 2
- **Files modified:** 11

## Accomplishments
- Pinned claude-agent-sdk to 0.1.52 for stable subagent support
- Extended Backend.execute() protocol with agents: list[AgentDefinition] | None = None kwarg
- ClaudeBackend passes agents dict to ClaudeAgentOptions, CodexBackend silently ignores
- run_agent() accepts and forwards agents to backend.execute()
- All 134 tests pass (129 existing + 5 new), mypy clean, ruff clean

## Task Commits

Each task was committed atomically:

1. **Task 1: SDK version bump + Backend protocol extension** (TDD)
   - `27988ee` (test: add failing tests for Backend agents kwarg support)
   - `48417a5` (feat: SDK bump to 0.1.52 and Backend protocol agents kwarg)
2. **Task 2: Wire agents kwarg through run_agent()** - `e451d81` (feat)

## Files Created/Modified
- `pyproject.toml` - SDK pin changed from >=0.1.27 to ==0.1.52
- `daydream/backends/__init__.py` - Backend protocol extended with agents kwarg, AgentDefinition TYPE_CHECKING import
- `daydream/backends/claude.py` - ClaudeBackend.execute() accepts agents, passes dict to ClaudeAgentOptions
- `daydream/backends/codex.py` - CodexBackend.execute() accepts agents (ignored)
- `daydream/agent.py` - run_agent() signature extended with agents, forwarded to backend.execute()
- `tests/test_backends_init.py` - 2 new tests: agent_definition_importable, backend_execute_accepts_agents_kwarg
- `tests/test_backend_claude.py` - 2 new tests: execute_passes_agents_to_options, execute_without_agents_no_agents_in_options
- `tests/test_backend_codex.py` - 1 new test: execute_ignores_agents
- `tests/test_integration.py` - MockBackend signatures updated for agents kwarg
- `tests/test_phases.py` - MockBackend signatures updated for agents kwarg
- `tests/test_loop.py` - MockBackend signatures updated for agents kwarg

## Decisions Made
- Used conditional assignment (`options.agents = ...`) instead of dict unpacking in ClaudeAgentOptions constructor -- mypy rejects `**{...}` spread with typed dataclass constructors
- Added `from __future__ import annotations` to `agent.py` to support TYPE_CHECKING guard for AgentDefinition without runtime import

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated all MockBackend.execute() signatures in test files**
- **Found during:** Task 2 (wire agents through run_agent)
- **Issue:** run_agent() now passes agents=agents to backend.execute(), but MockBackend classes in test_integration.py, test_phases.py, and test_loop.py didn't accept the agents kwarg
- **Fix:** Added agents=None to all MockBackend.execute() signatures across 3 test files (14 occurrences total)
- **Files modified:** tests/test_integration.py, tests/test_phases.py, tests/test_loop.py
- **Verification:** All 134 tests pass
- **Committed in:** e451d81 (Task 2 commit)

**2. [Rule 1 - Bug] Fixed mypy error with dict unpacking in ClaudeAgentOptions**
- **Found during:** Task 1 GREEN phase
- **Issue:** `**({"agents": agents_dict} if agents_dict else {})` in ClaudeAgentOptions constructor caused 21 mypy arg-type errors
- **Fix:** Changed to conditional assignment: `if agents: options.agents = {f"explorer-{i}": a for i, a in enumerate(agents)}`
- **Files modified:** daydream/backends/claude.py
- **Verification:** mypy passes with 0 errors
- **Committed in:** 48417a5 (Task 1 GREEN commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes necessary for test compatibility and type safety. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations.

## Known Stubs
None -- all changes are fully wired with no placeholder data.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Backend protocol ready to accept AgentDefinition lists from any caller
- Phase 2 (pre-scan exploration) can now pass agents to run_agent() and backend.execute()
- ExplorationContext dataclass and degradation handling (Plan 02) needed before exploration can be invoked

## Self-Check: PASSED

- All 5 modified source files: FOUND
- All 3 commits (27988ee, 48417a5, e451d81): FOUND
- SUMMARY.md: FOUND
- All 134 tests: PASSED
- mypy: 0 errors
- ruff: all checks passed

---
*Phase: 01-exploration-infrastructure*
*Completed: 2026-04-06*
