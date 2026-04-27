# Deferred Items — Phase 02-recorder-core-event-enrichment-mapping

Out-of-scope discoveries logged during execution; not fixed by the current plan.

## Pre-existing test failures (out of scope, present on phase 2 base commit f16b869)

Discovered while running the full test suite during plan 02-01 execution
(2026-04-26). All four failures reproduce on the unmodified base commit
(`git stash` + retry confirms): they are NOT caused by Plan 02-01 changes.

- `tests/test_deep_orchestrator.py::test_fresh_context_per_stage`
- `tests/test_deep_orchestrator.py::test_per_stack_context_isolation`
- `tests/test_deep_orchestrator.py::test_preflight_notice` (assert 7 == 9)
- `tests/test_deep_orchestrator.py::test_failed_per_stack_surfaces_to_merge_prompt_and_persists`

These do not block Plan 02-01 (which is purely additive — new module +
new tests). Existing 343-test gate is honored if these are excluded as
pre-existing flakes; a separate plan should investigate.
