# Phase 5: Test Hardening + Documentation - Context

**Gathered:** 2026-04-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Migration-complete signal. All 343 pre-existing tests verified passing post-migration, new trajectory/redaction/golden-fixture/subagent test suites audited and gaps filled (using schema-validity + behavior-predicate patterns, not full-tree snapshot equality). README, CHANGELOG, CLAUDE.md, and `daydream/atif/NOTICE` document the new format and breaking CLI change.

**Out of phase scope (deferred):**
- Streaming trajectory writes (PERF-01, v2)
- Replay/stats subcommands (TOOL-01, TOOL-02, v2)
- Multimodal ContentPart support (MM-01, v2)

</domain>

<decisions>
## Implementation Decisions

### Test Gap Audit Strategy (TEST-01..05, TEST-07)

- **D-01: Audit-first, fill gaps.** Researcher reads existing test files (`test_trajectory.py`, `test_redaction.py`, `test_atif_vendor_smoke.py`, `test_trajectory_fixture.py` — 83 tests, 1556 LOC) and maps each test to TEST-01..07 requirements. Only write new tests for uncovered requirements. Phases 1-4 already landed substantial test coverage; Phase 5 fills gaps rather than duplicating.
- **D-02: Verify TEST-05 compliance.** Researcher scans existing trajectory/redaction tests for `assert trajectory == expected_dict` full-tree snapshot patterns that violate Phase 2 D-18. Fix any violations found — this is an active audit, not a trust assumption.
- **D-03: TEST-07 subagent shapes — mock fork, assert shapes.** Use a MockBackend to drive `phase_fix_parallel` / deep / exploration through the recorder's `fork()` path, then validate the resulting root + sibling trajectory files against the vendored validator. Tests the real recorder code with fake backends. No pre-recorded fixture files.

### SDK #112 Empirical Test (TEST-06)

- **D-04: Multi-turn mock fixture.** Build a MockBackend that emits 3 sequential `run_agent()` calls with known token values (e.g., turn 1: 100 input, turn 2: 150 input, turn 3: 200 input). Assert each step's `Metrics.prompt_tokens` matches the per-call values, NOT cumulative. This is a gate test — it passes or fails. No conditional delta-subtraction logic.

### README Trajectory Documentation (DOCS-01, DOCS-02, DOCS-03)

- **D-05: Concise README section + link to format doc.** README gets a "Trajectory Output" section with: format overview, default path (`<target>/.daydream/trajectory.json`), `--trajectory` flag usage, redaction summary, 3-4 sentences max. Full format spec with JSON examples stays in `docs/reference/atif_format.md` (already exists). README links to it.
- **D-06: Redaction docs list what IS redacted only.** The 4 categories (API keys, JWTs, user paths, env vars) are listed. Users who need more detail read the format doc. No "what is NOT redacted" section in README.
- **D-07: Consumer integration paths are brief pointers with links.** 2-3 bullet list: validate with Harbor, replay in viewer, use for training data. Link to Harbor repo. Don't document Harbor's API.

### CHANGELOG + Versioning (DOCS-04)

- **D-08: Version 0.14.0.** Minor bump — semver 0.x allows breaking changes on minor. Signals new feature set without implying 1.0 stability.
- **D-09: Single entry with grouped subsections.** One `[0.14.0]` heading with `### Breaking`, `### Added`, `### Removed` subsections. Breaking: `--debug` removed. Added: ATIF trajectory output, `--trajectory` flag, automatic redaction. Removed: `_log_debug` system, `.review-debug-*.log` files.

### Claude's Discretion

- Which existing tests to keep, restructure, or extend vs. write fresh tests for gaps
- Exact placement of the "Trajectory Output" section within README's existing structure
- CLAUDE.md update scope — how much to add about `daydream/trajectory.py` beyond what's already there
- Whether `daydream/atif/NOTICE` needs any updates (it already exists from Phase 1)
- Test file organization — whether TEST-06 and TEST-07 go in existing files or new ones

</decisions>

<specifics>
## Specific Ideas

- 564 tests currently passing (up from 343 baseline). Phases 1-4 added 221 tests. The 343 baseline regression gate (TEST-01) is already met — Phase 5 formalizes this.
- `test_trajectory.py` is already 1019 lines — likely covers much of TEST-02 (recorder lifecycle, step coalescing, tool-call correlation, validator round-trip). The audit will confirm.
- `test_redaction.py` is already 418 lines — likely covers TEST-03 (per-pattern categories). The audit will confirm.
- Phase 2 D-14 explicitly said to trust per-call token semantics and gate with TEST-06. The multi-turn mock fixture is the gate.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### ATIF Specification
- `docs/reference/atif_format.md` — Full format spec. README links to this. Tests validate against this spec.

### Existing Test Files (audit targets)
- `tests/test_trajectory.py` — 1019 lines, ~40+ tests. Covers recorder lifecycle, step coalescing, tool-call correlation, validator round-trip. Audit against TEST-02.
- `tests/test_redaction.py` — 418 lines. Covers per-pattern redaction categories. Audit against TEST-03.
- `tests/test_atif_vendor_smoke.py` — 67 lines. Parametrizes over golden fixtures. Audit against TEST-04.
- `tests/test_trajectory_fixture.py` — 52 lines. Fixture validation. Audit against TEST-04/TEST-05.

### Phase 2-4 Context (test design precedents)
- `.planning/phases/02-recorder-core-event-enrichment-mapping/02-CONTEXT.md` — D-17 (autouse fixture pattern), D-18 (schema-validity + behavior-predicate, NOT full-tree snapshots), D-14 (trust per-call tokens, gate with TEST-06).
- `.planning/phases/03-subagent-wiring-parallel-continuation/03-CONTEXT.md` — D-01..D-04 (fork/sibling machinery). TEST-07 tests exercise this.
- `.planning/phases/04-cutover-redaction-cli-surface/04-CONTEXT.md` — D-01..D-04 (redaction tokens), D-05 (hard `--debug` reject), D-07 (SIGINT partial flush).

### Documentation Targets
- `README.md` — 221 lines. Gets new "Trajectory Output" section (DOCS-01, DOCS-02, DOCS-03).
- `CHANGELOG.md` — 377 lines. Gets `[0.14.0]` entry with Breaking/Added/Removed subsections (DOCS-04).
- `CLAUDE.md` — Project instructions. Gets trajectory format mention (DOCS-06).
- `daydream/atif/NOTICE` — Already exists from Phase 1 (DOCS-05). Verify completeness.

### Project Planning
- `.planning/REQUIREMENTS.md` — TEST-01..07, DOCS-01..06 are this phase's 13 requirements.
- `.planning/ROADMAP.md` — Phase 5 success criteria (5 must-be-true items).

### Codebase Maps
- `.planning/codebase/TESTING.md` — Test patterns, fixture conventions, mock backend pattern, autouse fixtures. Phase 5 tests follow these established patterns.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **83 existing trajectory/redaction/atif tests** — Phase 5 audits and extends rather than rewrites. MockBackend pattern from phases 2-4 reused for TEST-06 and TEST-07.
- **`tests/fixtures/atif_golden/`** — Golden fixtures from Phase 1. Already used by `test_atif_vendor_smoke.py`.
- **`daydream/atif/validate()`** — Vendored validator. All new tests use this for schema-validity assertions per D-18 pattern.
- **Inline MockBackend pattern** — Established across the test suite. TEST-06 and TEST-07 follow the same pattern (define a mock implementing `execute`, `cancel`, `format_skill_invocation`).

### Established Patterns
- **`_reset_trajectory_recorder` autouse fixture** — Already in `conftest.py` from Phase 2. All trajectory tests inherit isolation.
- **`reset_state()` autouse fixture** — Existing pattern for `AgentState` isolation.
- **Schema-validity + behavior-predicate** (Phase 2 D-18) — Assert `validate()` passes + 1-2 specific behavioral checks per test. No `assert trajectory == expected_dict`.

### Integration Points
- **README.md** — New "Trajectory Output" section inserted after existing features/usage sections.
- **CHANGELOG.md** — New `[0.14.0]` entry above `[Unreleased]` (or replacing it).
- **CLAUDE.md** — Trajectory mention added to existing Architecture or Module Responsibilities section.
- **No source code changes** — Phase 5 is tests + docs only. No modifications to `daydream/` package code.

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 05-test-hardening-documentation*
*Context gathered: 2026-04-28*
