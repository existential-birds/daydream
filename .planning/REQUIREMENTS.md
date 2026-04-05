# Requirements: Daydream

**Defined:** 2026-04-05
**Core Value:** Reviews and recommendations must be grounded in actual codebase understanding

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### Exploration

- [ ] **EXPL-01**: System maps impact surface from diff (affected files + transitive dependencies)
- [ ] **EXPL-02**: System reads diff-adjacent files (touched files + their immediate imports/callers)
- [ ] **EXPL-03**: System detects codebase conventions/patterns before review starts
- [ ] **EXPL-04**: Exploration subagents read project guidelines (CLAUDE.md, .coderabbit.yaml, etc.)

### Subagents

- [ ] **AGNT-01**: 3-5 parallel pre-scan subagents explore affected areas before review
- [ ] **AGNT-02**: Review agent spawns on-demand explorer subagents when it encounters uncertainty
- [ ] **AGNT-03**: Backend protocol extended with `agents` parameter for subagent support
- [ ] **AGNT-04**: Exploration budget system enforces hard caps (depth, files, tokens, time)

### Review Quality

- [ ] **QUAL-01**: Cross-file dependency tracing follows call chains to catch breaking changes
- [ ] **QUAL-02**: Each recommendation carries confidence score (HIGH/MEDIUM/LOW) with rationale
- [ ] **QUAL-03**: Recommendations contradicting existing codebase conventions are filtered out

### Output Quality

- [ ] **OUTP-01**: TTT plan generation references actual file paths, signatures, and patterns from exploration
- [ ] **OUTP-02**: Both TTT and normal review flows use the same exploration architecture

### Infrastructure

- [ ] **INFR-01**: `claude-agent-sdk` bumped to `>=0.1.52` for `AgentDefinition` support
- [ ] **INFR-02**: Exploration results aggregated into structured `ExplorationContext` for review prompt injection
- [ ] **INFR-03**: Exploration degrades gracefully (review proceeds if exploration fails)

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
| mem0 integration | Heavy infrastructure (LLM + embedder + vector store) conflicts with CLI tool ethos |
| Full codebase indexing/embedding | Agent has direct file access via tools; the filesystem is the index |
| Custom orchestration framework | Use SDK subagents — explicitly ruled out in project goals |
| Style/formatting nitpicks | Linters handle this better; #1 source of review fatigue |
| Comment volume maximization | Optimize for precision over recall; best tools produce ~3.6 comments per PR |
| Real-time streaming exploration results | Simple progress indicator sufficient; complexity without quality gain |
| Multi-model ensemble review | Doubles cost/latency for marginal quality improvement |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| EXPL-01 | TBD | Pending |
| EXPL-02 | TBD | Pending |
| EXPL-03 | TBD | Pending |
| EXPL-04 | TBD | Pending |
| AGNT-01 | TBD | Pending |
| AGNT-02 | TBD | Pending |
| AGNT-03 | TBD | Pending |
| AGNT-04 | TBD | Pending |
| QUAL-01 | TBD | Pending |
| QUAL-02 | TBD | Pending |
| QUAL-03 | TBD | Pending |
| OUTP-01 | TBD | Pending |
| OUTP-02 | TBD | Pending |
| INFR-01 | TBD | Pending |
| INFR-02 | TBD | Pending |
| INFR-03 | TBD | Pending |

**Coverage:**
- v1 requirements: 16 total
- Mapped to phases: 0
- Unmapped: 16

---
*Requirements defined: 2026-04-05*
*Last updated: 2026-04-05 after initial definition*
