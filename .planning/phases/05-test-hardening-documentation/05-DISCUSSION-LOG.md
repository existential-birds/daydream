# Phase 5: Test Hardening + Documentation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-28
**Phase:** 05-test-hardening-documentation
**Areas discussed:** Test gap audit strategy, README trajectory docs, CHANGELOG + versioning, SDK #112 empirical test

---

## Test Gap Audit Strategy

### Q1: How to approach the test gap audit?

| Option | Description | Selected |
|--------|-------------|----------|
| Audit-first, fill gaps | Read existing test files, map to TEST-01..07, write only what's missing | ✓ |
| Write new comprehensive suite | Create new test files regardless, don't audit existing coverage | |
| You decide | Claude picks during planning | |

**User's choice:** Audit-first, fill gaps
**Notes:** 83 tests already exist across 4 files from phases 1-4. Avoid duplicate work.

### Q2: TEST-07 subagent shape test approach?

| Option | Description | Selected |
|--------|-------------|----------|
| Mock fork, assert shapes | MockBackend drives fork() path, validate with vendored validator | ✓ |
| Pre-recorded fixtures | Capture real trajectory JSON, commit as fixtures, parametrize tests | |
| You decide | Claude picks during planning | |

**User's choice:** Mock fork, assert shapes
**Notes:** Tests the real recorder code with fake backends.

### Q3: TEST-05 compliance verification?

| Option | Description | Selected |
|--------|-------------|----------|
| Verify compliance | Scan existing tests for full-tree snapshot patterns, fix violations | ✓ |
| Trust prior phases | Assume phases 1-4 followed D-18, skip the audit | |
| You decide | Claude judges during planning | |

**User's choice:** Verify compliance

---

## README Trajectory Docs

### Q1: How much trajectory detail in README?

| Option | Description | Selected |
|--------|-------------|----------|
| README section + format doc | Concise section in README, full spec in docs/reference/atif_format.md | ✓ |
| All in README | Full section with JSON snippet, field descriptions, integration paths | |
| You decide | Claude picks based on style | |

**User's choice:** README section + format doc

### Q2: Consumer integration paths (DOCS-03)?

| Option | Description | Selected |
|--------|-------------|----------|
| Brief pointers with links | 2-3 bullet list, link to Harbor repo | ✓ |
| Worked examples | Concrete commands, viewer links, Python snippets | |
| You decide | Claude picks during planning | |

**User's choice:** Brief pointers with links

### Q3: Redaction policy docs (DOCS-02)?

| Option | Description | Selected |
|--------|-------------|----------|
| Both (what and what-not) | List 4 categories AND what passes through unredacted | |
| Only what IS redacted | List 4 categories only, users read format doc for details | ✓ |
| You decide | Claude picks based on requirement wording | |

**User's choice:** Only what IS redacted

---

## CHANGELOG + Versioning

### Q1: Version number?

| Option | Description | Selected |
|--------|-------------|----------|
| 0.14.0 (minor bump) | Semver 0.x allows breaking on minor, signals new feature set | ✓ |
| 1.0.0 (major) | Signal production-readiness and stable trajectory format | |
| You decide | Claude picks based on versioning pattern | |

**User's choice:** 0.14.0

### Q2: CHANGELOG entry structure?

| Option | Description | Selected |
|--------|-------------|----------|
| Single entry, grouped sections | [0.14.0] with ### Breaking, ### Added, ### Removed subsections | ✓ |
| Minimal entry | Summary paragraph with bullet list, no subsections | |
| You decide | Claude picks during implementation | |

**User's choice:** Single entry, grouped sections

---

## SDK #112 Empirical Test

### Q1: Test approach without real API?

| Option | Description | Selected |
|--------|-------------|----------|
| Multi-turn mock fixture | MockBackend with 3 sequential calls, known token values, assert per-call | ✓ |
| Recorded SDK fixture | Capture real SDK session, commit as fixture, replay through backend | |
| You decide | Claude picks during planning | |

**User's choice:** Multi-turn mock fixture

### Q2: Failure behavior?

**Withdrawn.** User pointed out the question was manufacturing complexity — TEST-06 is a gate test that passes or fails. No conditional paths needed.

---

## Claude's Discretion

- Which existing tests to keep/restructure/extend vs. write fresh for gaps
- Exact README section placement
- CLAUDE.md update scope
- Whether NOTICE needs updates
- Test file organization for TEST-06 and TEST-07

## Deferred Ideas

None — discussion stayed within phase scope.
