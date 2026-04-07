# Requirements: Daydream

**Defined:** 2026-04-05
**Core Value:** Reviews and recommendations must be grounded in actual codebase understanding

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### Exploration

- [ ] **EXPL-01**: System maps impact surface from diff (affected files + transitive dependencies)
- [ ] **EXPL-02**: System reads diff-adjacent files (touched files + their immediate imports/callers)
- [x] **EXPL-03**: System detects codebase conventions/patterns before review starts
- [x] **EXPL-04**: Exploration subagents read project guidelines (CLAUDE.md, .coderabbit.yaml, etc.)

### Subagents

- [ ] **AGNT-01**: 3-5 parallel pre-scan subagents explore affected areas before review
- [ ] **AGNT-02**: Review agent spawns on-demand explorer subagents when it encounters uncertainty
- [x] **AGNT-03**: Backend protocol extended with `agents` parameter for subagent support

### Review Quality

- [ ] **QUAL-01**: Cross-file dependency tracing follows call chains to catch breaking changes
- [ ] **QUAL-02**: Each recommendation carries confidence score (HIGH/MEDIUM/LOW) with rationale
- [ ] **QUAL-03**: Recommendations contradicting existing codebase conventions are filtered out

### Output Quality

- [ ] **OUTP-01**: TTT plan generation references actual file paths, signatures, and patterns from exploration
- [ ] **OUTP-02**: Both TTT and normal review flows use the same exploration architecture

### Infrastructure

- [x] **INFR-01**: `claude-agent-sdk` bumped to `>=0.1.52` for `AgentDefinition` support
- [x] **INFR-02**: Exploration results aggregated into structured `ExplorationContext` for review prompt injection
- [x] **INFR-03**: Exploration degrades gracefully (review proceeds if exploration fails)

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Memory Layer

- **MEML-01**: SQLite-based memory persistence for tribal knowledge and conventions
- **MEML-02**: Semantic search for fuzzy retrieval of stored rules and patterns
- **MEML-03**: Tribal knowledge capture from dismissed review comments
- **MEML-04**: Convention memory persists across reviews for the same project

### Advanced Exploration

- **ADVX-01**: Exploration result caching to avoid re-exploring unchanged areas
- **ADVX-02**: Convention contradiction filtering as a separate post-review pass

## Out of Scope

| Feature | Reason |
|---------|--------|
| Exploration budget system (AGNT-04) | Unnecessary complexity for v1 |
| mem0 integration | Heavy infrastructure (LLM + embedder + vector store) conflicts with CLI tool ethos |
| Full codebase indexing/embedding | Agent has direct file access via tools; the filesystem is the index |
| Custom orchestration framework | Use SDK subagents -- explicitly ruled out in project goals |
| Style/formatting nitpicks | Linters handle this better; #1 source of review fatigue |
| Comment volume maximization | Optimize for precision over recall; best tools produce ~3.6 comments per PR |
| Real-time streaming exploration results | Simple progress indicator sufficient; complexity without quality gain |
| Multi-model ensemble review | Doubles cost/latency for marginal quality improvement |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| EXPL-01 | Phase 2 | Pending |
| EXPL-02 | Phase 2 | Pending |
| EXPL-03 | Phase 2 | Complete (02-03) |
| EXPL-04 | Phase 2 | Complete (02-03) |
| AGNT-01 | Phase 2 | Pending |
| AGNT-02 | Phase 4 | Pending |
| AGNT-03 | Phase 1 | Complete |
| QUAL-01 | Phase 3 | Pending |
| QUAL-02 | Phase 3 | Pending |
| QUAL-03 | Phase 3 | Pending |
| OUTP-01 | Phase 3 | Pending |
| OUTP-02 | Phase 3 | Pending |
| INFR-01 | Phase 1 | Complete |
| INFR-02 | Phase 1 | Complete |
| INFR-03 | Phase 1 | Complete |

**Coverage:**
- v1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-04-05*
*Last updated: 2026-04-05 after roadmap revision (removed AGNT-04)*
