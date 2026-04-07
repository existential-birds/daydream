# Roadmap: Daydream

## Overview

Daydream's reviews operate on diffs without understanding the surrounding codebase, producing shallow recommendations that miss architectural problems and contradict existing patterns. This milestone adds a subagent exploration layer that maps affected code areas before review starts, spawns on-demand explorers during review, and injects grounded codebase context into every recommendation. The build order is dictated by hard dependencies: safety infrastructure first, then exploration logic, then review consumption, then on-demand spawning.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Exploration Infrastructure** - Backend protocol extension, data structures, and SDK version bump
- [ ] **Phase 2: Pre-scan Exploration** - Parallel subagent exploration of affected codebase areas before review
- [ ] **Phase 3: Review Integration** - Downstream phases consume exploration context for grounded recommendations
- [ ] **Phase 4: On-demand Exploration** - Mid-review subagent spawning when the review agent encounters uncertainty

## Phase Details

### Phase 1: Exploration Infrastructure
**Goal**: Safe, structured foundation exists for all exploration work
**Depends on**: Nothing (first phase)
**Requirements**: INFR-01, INFR-02, INFR-03, AGNT-03
**Success Criteria** (what must be TRUE):
  1. `claude-agent-sdk >= 0.1.52` is installed and `AgentDefinition` imports work
  2. `Backend.execute()` accepts an optional `agents` parameter without breaking existing calls
  3. `ExplorationContext` dataclass can be instantiated and serialized to structured text for prompt injection
  4. Exploration failure (timeout, SDK error) produces a fallback empty context and does not block review
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — SDK version bump + Backend protocol extension with agents kwarg
- [x] 01-02-PLAN.md — ExplorationContext data structures and graceful degradation

### Phase 2: Pre-scan Exploration
**Goal**: Parallel subagents explore affected codebase areas before review starts
**Depends on**: Phase 1
**Requirements**: EXPL-01, EXPL-02, EXPL-03, EXPL-04, AGNT-01
**Success Criteria** (what must be TRUE):
  1. Running `daydream /path/to/project --ttt` on a project with a multi-file diff produces an `ExplorationContext` containing file maps, conventions, and dependency info before the review phase begins
  2. Three parallel subagents (pattern-scanner, dependency-tracer, test-mapper) each complete and return structured results
  3. Exploration subagents read project guidelines (CLAUDE.md, config files) as part of their scan
  4. `detect_affected_files()` correctly identifies changed files and their immediate imports/callers from the git diff
  5. Exploration scales with diff size (skip for trivial diffs, single agent for small, parallel for large)
**Plans**: 4 plans

Plans:
- [x] 02-01-PLAN.md — Wave 0: fix Backend agents-shape, install tree-sitter, create test scaffolds + diff fixtures
- [x] 02-02-PLAN.md — tree_sitter_index: LANGUAGES registry, parser cache, detect_affected_files()
- [x] 02-03-PLAN.md — Subagent prompts/schemas/EXPLORATION_AGENTS + merge_contexts()
- [ ] 02-04-PLAN.md — exploration_runner orchestrator + wire into run()/run_trust() + UX + Codex guard

### Phase 3: Review Integration
**Goal**: Review and plan generation recommendations are grounded in actual codebase context from exploration
**Depends on**: Phase 2
**Requirements**: QUAL-01, QUAL-02, QUAL-03, OUTP-01, OUTP-02
**Success Criteria** (what must be TRUE):
  1. Review output includes cross-file dependency analysis that traces call chains beyond the diff boundary
  2. Each review recommendation carries a confidence score (HIGH/MEDIUM/LOW) with rationale explaining what was verified vs. inferred
  3. Recommendations that contradict detected codebase conventions are filtered or flagged before presentation
  4. TTT plan generation references actual file paths, function signatures, and patterns discovered during exploration
  5. Both TTT (`--ttt`) and normal review flows use the same exploration architecture and produce exploration-enriched output
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

### Phase 4: On-demand Exploration
**Goal**: Review agent can spawn focused explorer subagents mid-review when it encounters uncertainty
**Depends on**: Phase 3
**Requirements**: AGNT-02
**Success Criteria** (what must be TRUE):
  1. Review agent recognizes uncertainty during analysis and spawns an on-demand explorer subagent to investigate
  2. On-demand exploration is capped (max 2 spawns per session) to prevent runaway cost
  3. On-demand exploration results are incorporated into the review output for the specific issue that triggered them
**Plans**: TBD

Plans:
- [ ] 04-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Exploration Infrastructure | 0/2 | Not started | - |
| 2. Pre-scan Exploration | 0/2 | Not started | - |
| 3. Review Integration | 0/2 | Not started | - |
| 4. On-demand Exploration | 0/1 | Not started | - |
