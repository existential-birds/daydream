---
phase: 03-review-integration
plan: 01
subsystem: phases
tags: [schema, prompt, qual-02, qual-03, outp-01]
requires: [daydream.phases, daydream.exploration]
provides: [build_review_prompt, build_intent_prompt, build_alternative_review_prompt, build_plan_prompt, _validate_issue]
affects: [daydream/phases.py, tests/test_phases.py, tests/test_integration.py]
tech-stack:
  added: []
  patterns: [schema-enforced-structured-output, shared-prompt-helpers]
key-files:
  modified:
    - daydream/phases.py
    - tests/test_phases.py
    - tests/test_integration.py
decisions:
  - "PLAN_SCHEMA references[] added at the per-change item level inside nested plan.issues.changes shape (preserves _write_plan_markdown contract)"
  - "Module-level build_*_prompt helpers extracted from phase functions so tests can call them directly without backend wiring"
  - "Integration test exercises phase_parse_feedback + phase_alternative_review directly instead of full run()/run_trust() (avoids changing run signatures from int to list)"
metrics:
  duration: ~10min
  tasks: 2
  files: 3
  completed: 2026-04-07
requirements: [QUAL-01, QUAL-02, QUAL-03, OUTP-01, OUTP-02]
---

# Phase 03 Plan 01: Schema + Prompt Enforcement Summary

Schema-enforced confidence/rationale on FEEDBACK_SCHEMA and ALTERNATIVE_REVIEW_SCHEMA, references[] on PLAN_SCHEMA, and four shared prompt builders that inject confidence/convention/dependency-impact instructions across all four phase prompt sites.

## Tasks

1. Extend three schemas + add `_confidence_and_convention_instructions`, `_plan_grounding_instructions`, `_dependency_impact_instructions`, `_validate_issue`, and four module-level `build_*_prompt` helpers; rewire `phase_review`, `phase_understand_intent`, `phase_alternative_review`, `phase_generate_plan` to call them (commit `e53ac0e`).
2. Remove xfail markers from nine Wave 0 tests; navigate nested PLAN_SCHEMA path; tighten integration test to exercise both flows via lower-level phase functions (commit `251d9b4`).

## Verification

- `uv run pytest tests/test_phases.py -x` -> 33 passed
- `uv run pytest tests/test_integration.py::test_exploration_enriched_output_both_flows -x` -> passed
- `make check` -> 192 passed, 1 xfailed (Wave 2 UI renderer test as expected)
- `uv run mypy daydream/phases.py` -> Success: no issues found

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocker] Wave 0 test_plan_schema_requires_references navigated wrong path**
- **Found during:** Task 1 verification
- **Issue:** Wave 0 scaffold expected `PLAN_SCHEMA["properties"]["changes"]` (flat shape). Existing schema is nested under `plan.issues[].changes[]` and `_write_plan_markdown` depends on that shape.
- **Fix:** Updated the test to navigate the nested path. Plan was ambiguous; preserving the existing nested shape avoids touching `_write_plan_markdown` and the markdown writer test.
- **Files modified:** tests/test_phases.py
- **Commit:** 251d9b4

**2. [Rule 3 - Blocker] Integration test referenced run()/run_trust() return types that don't exist**
- **Found during:** Task 2 verification
- **Issue:** Wave 0 scaffold called `await run(cfg)` and asserted it returned issue lists. `run` and `run_trust` return `int` exit codes; changing them is a Rule-4 architectural change.
- **Fix:** Plan permits "ensure assertions check both flows produce parsed output containing confidence and rationale". Tightened the test to exercise `phase_parse_feedback` (normal flow's parse step) and `phase_alternative_review` (TTT flow's review step) directly. Both return parsed issue lists carrying the schema-enforced fields.
- **Files modified:** tests/test_integration.py
- **Commit:** 251d9b4

**3. [Rule 1 - Bug] ResultEvent constructor signature mismatch in integration test**
- **Found during:** Task 2 first run
- **Issue:** Wave 0 scaffold passed `result=, cost_usd=, duration_ms=` kwargs that don't exist on `ResultEvent` (actual fields: `structured_output`, `continuation`).
- **Fix:** Use the real dataclass field names.
- **Commit:** 251d9b4

## Self-Check: PASSED

- FOUND: daydream/phases.py (build_review_prompt, build_intent_prompt, build_alternative_review_prompt, build_plan_prompt, _validate_issue, _confidence_and_convention_instructions, _plan_grounding_instructions, _dependency_impact_instructions)
- FOUND: tests/test_phases.py (no remaining xfail markers in Wave 0 block)
- FOUND: tests/test_integration.py (no xfail on test_exploration_enriched_output_both_flows)
- FOUND commit: e53ac0e
- FOUND commit: 251d9b4
