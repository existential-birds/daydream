---
phase: 03-review-integration
plan: 00
subsystem: tests
tags: [tdd, wave-0, scaffolding, xfail-strict]
requires: [daydream.exploration]
provides: [exploration_context_fixture, wave-0-failing-tests]
affects: [tests/conftest.py, tests/test_phases.py, tests/test_ui.py, tests/test_integration.py]
tech-stack:
  added: []
  patterns: [xfail-strict-tdd]
key-files:
  created:
    - tests/conftest.py
  modified:
    - tests/test_phases.py
    - tests/test_ui.py
    - tests/test_integration.py
decisions:
  - "Use xfail(strict=True) so Wave 1/2 implementation flips them green incrementally"
  - "Shared exploration_context_fixture lives in tests/conftest.py for reuse across phase + integration suites"
metrics:
  duration: ~6min
  tasks: 2
  files: 4
  completed: 2026-04-07
---

# Phase 03 Plan 00: Wave 0 Test Scaffolding Summary

Failing-first scaffolds for the Phase 03 review-integration work: 11 xfail-strict tests covering schema enrichment, prompt builder behavior, plan-renderer dimming, and end-to-end exploration enrichment, plus a shared `exploration_context_fixture` for downstream waves.

## Tasks

1. Shared `ExplorationContext` fixture + 9 schema/prompt xfail tests in `tests/test_phases.py` (commit `e7b8e11`)
2. UI plan-renderer + integration both-flows xfail scaffolds (commit `a3bcec5`)

## Verification

- `uv run pytest`: 182 passed, 11 xfailed
- `uv run ruff check tests/...`: clean

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- FOUND: tests/conftest.py
- FOUND: tests/test_phases.py (9 new test defs)
- FOUND: tests/test_ui.py (1 new test def)
- FOUND: tests/test_integration.py (1 new test def)
- FOUND commit: e7b8e11
- FOUND commit: a3bcec5
