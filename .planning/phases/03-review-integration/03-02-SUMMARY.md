---
phase: 03-review-integration
plan: 02
subsystem: ui
tags: [outp-01, ttt, renderer, d-08]
requires: [daydream.ui]
provides: [render_ttt_plan]
affects: [daydream/ui.py, tests/test_ui.py]
tech-stack:
  added: []
  patterns: [rich-console-styled-output]
key-files:
  modified:
    - daydream/ui.py
    - tests/test_ui.py
decisions:
  - "render_ttt_plan accepts both flat ({changes:[...]}) and nested ({plan:{issues:[{changes:[...]}]}}) plan shapes so existing PLAN_SCHEMA writers and the test fixture both work without a migration"
metrics:
  duration: ~5min
  tasks: 1
  files: 2
  completed: 2026-04-07
requirements: [OUTP-01]
---

# Phase 03 Plan 02: TTT Plan Renderer Visual Distinction Summary

Added `render_ttt_plan()` to `daydream/ui.py`. Plan steps with a non-empty `references` list render with default style and inline `→ file::symbol` references beneath; steps with an empty list render dimmed with a `(ungrounded)` marker, closing the OUTP-01 visibility loop for D-08.

## Tasks

1. Implement `render_ttt_plan(console, plan)` in `daydream/ui.py`, branching on `bool(change.get("references"))`. Removed the Wave 0 xfail marker on `test_plan_renderer_dims_ungrounded_steps` and verified the full suite + `make check` (commit `5907d52`).

## Verification

- `uv run pytest tests/test_ui.py::test_plan_renderer_dims_ungrounded_steps -x` -> passed
- `make check` -> 193 passed, 0 xfailed, 1 unrelated warning
- `grep -c "ungrounded" daydream/ui.py` -> 3 (>= 1)
- `grep -c "ungrounded" tests/test_ui.py` -> 2 (>= 1)
- No xfail decorator above `test_plan_renderer_dims_ungrounded_steps`

## Deviations from Plan

None. The plan called for locating an existing TTT plan renderer in `ui.py`; no such renderer existed (the plan output is currently written to markdown via `_write_plan_markdown` in `phases.py`). The test imports `render_ttt_plan` directly from `daydream.ui`, so the natural fit was to add a new top-level function rather than retrofit `_write_plan_markdown`. Function accepts both the flat test-fixture shape and the nested PLAN_SCHEMA shape so future console-rendering callers can pass either.

## Self-Check: PASSED

- FOUND: daydream/ui.py (render_ttt_plan)
- FOUND: tests/test_ui.py (no xfail on test_plan_renderer_dims_ungrounded_steps)
- FOUND commit: 5907d52
