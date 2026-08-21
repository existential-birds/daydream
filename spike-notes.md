# Spike Notes — Per-Phase Terminal Signal Coverage (issue #762)

Re-verified against HEAD (`d31cb99d`). Conclusion: the spec's Key Decision 2 survives
intact — existing phase-event + deep-artifact signals cover every per-phase terminal
state. No spec revision needed.

## Confirmed signals

1. **Merge ran/failed** — via `deep/per-stack-failures.json["__merge__"]`:
   - `MERGE_FAILURE_KEY = "__merge__"` (orchestrator.py:1952).
   - Written **only** on the merge-failure consolidation path (`_salvage_merge_failure`,
     orchestrator.py:2160).
   - `merged-items.json` is written on BOTH the merge-failure path (`_salvage_merge_failure`,
     ~2197) and success paths (`_step_single_stack_merge`, `phase_cross_stack_merge`).
   - **Load-bearing:** merge `succeeded` ⇔ `merged-items.json` present AND `"__merge__"`
     absent from `per-stack-failures.json`. Presence of `merged-items.json` alone is NOT
     sufficient (it is also written on the failure path).
   - `DaydreamPhase` has NO `MERGE` member; `phase_scope(DaydreamPhase...)` call sites cover
     EXPLORATION/INTENT/ALTERNATIVES/DEEP(stage=…)/PARSE/VERIFY/FIX/TEST — no merge. So merge
     terminal state must come from deep artifacts, not phase events.

2. **Fix via `fix-failures.json`:**
   - `fix_failures_p` written only when `all_non_success` (dropped/reverted groups) is
     non-empty; unlinked when clean (orchestrator.py:3248-3262).
   - Present ⇔ fix phase had dropped/reverted groups (partial); absent ⇔ clean or never ran.

3. **Test via `test-verdict.json`:**
   - `test_verdict_path(dd).write_text(...)` occurs BEFORE the failure early-return
     (orchestrator.py:3445), so `{"passed": bool, "retries": int, "ignored": bool}` is written
     for BOTH outcomes.
   - Present ⇔ test phase ran (both outcomes); absent ⇔ test never ran (merge-failure-no-test,
     review-only, improve).

4. **Cancelled via `archive_status == "partial"` && no fix_failures:**
   - `Trajectory.write_partial()` → `on_write(self, "partial")` (trajectory.py:2042), signal flush = cancelled.
   - `_write` → `on_write(self, "complete")` (trajectory.py:1986).
   - `_archive_run_inner` coerces `complete → partial` when `fix_failures` present (archive/__init__.py:152-156).
   - **Confirmed:** `cancelled` ⇔ `archive_status == "partial"` AND no `fix_failures`; `partial` ⇔ `fix_failures` present.

## Flow awareness
`_flow_fix_test_steps(recorder.run_flow, config.flow_name)` (manifest.py:75) returns
`(runs_fix, runs_test)` by runtime mode/registry: `TTT` → (False, False); `PR` → (True, False)
(fix only); NORMAL/DEEP/IMPROVE/CUSTOM classified by registered pipeline. A phase the flow
does not run is `absent` and neutral; only a phase the flow DOES run that is absent forces
`partial` (run stopped early).

These four confirmed signals + the flow-awareness requirement drive `derive_phase_states` /
`derive_pipeline_status` in Task 3 and the `_archive_run_inner` wiring in Task 4.