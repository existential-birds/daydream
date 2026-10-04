---
schema_version: 1
id: improve-20261004-5585d195
title: Separate optional data persistence from protected review output publication
repository: https://github.com/existential-birds/daydream
discovered_at: null
origin:
  kind: agent
  session_ref: null
  human_confirmation: null
status: proposed
observed:
  commit: 53e77423ade32f80b73aa2a6ace4973b269352fd
  working_tree: clean
  snapshot_ref: null
  dirty_paths: []
benchmark:
  commit: null
  snapshot_ref: null
  verified: false
affected_paths:
- daydream/run_artifacts.py
- daydream/runner.py
- daydream/archive/__init__.py
- daydream/archive/hub.py
- daydream/phases/handoff.py
affected_symbols:
- daydream.run_artifacts._finalize_run_artifacts
- daydream.runner._run_workspace
- daydream.archive.finalize_archive_run
- daydream.archive.hub.upload_run_bundle
- daydream.phases.handoff._resolve_handoff_paths
related_findings: []
resolution:
  status: in_progress
  issues:
  - https://github.com/existential-birds/daydream/issues/1466
  prs: []
  fix_commits:
  - 1a5cfeb01e5951de164834922d8488c55f080bd5
  validation:
  - command: uv run pytest tests/test_runner.py -k controlled_custom_flow -q
    result: 'Before the production fix: 1 failed, 2 passed. Injected archive failure changed result 0 to 1 and exposed exception
      text. New regression tests were uncommitted; their exact tree was not preserved.'
    checked_commit: null
  - command: make check
    result: 'Initial implementation run: 1 failed, 9328 passed, 14 skipped. An old archive-capture test still required index
      failure to roll back review outputs. Source evolved during this run; exact checked snapshot unavailable. The obsolete
      assertion was replaced with observable deep-review export checks.'
    checked_commit: null
  - command: uv run pytest tests/test_archive_data_capture.py -q
    result: 'Passed: 29 tests; includes deep-review findings, explicit trajectory and prepared dump surviving index failure.'
    checked_commit: 1a5cfeb01e5951de164834922d8488c55f080bd5
  - command: make check
    result: 'Passed: 9330 tests, 14 skipped; branch coverage 90.38%, above the 86% gate. Lock, lint, dead-code, typecheck,
      workflow and naming checks passed.'
    checked_commit: 1a5cfeb01e5951de164834922d8488c55f080bd5
---

# Separate optional data persistence from protected review output publication

## Evidence and root cause

Issue [#1466](https://github.com/existential-birds/daydream/issues/1466), a prefactoring slice of [#1465](https://github.com/existential-birds/daydream/issues/1465), supplied the requirement before this implementation. At the pre-fix commit `53e77423ade32f80b73aa2a6ace4973b269352fd`, `daydream/run_artifacts.py::_finalize_run_artifacts` froze one runtime snapshot, caught `ArchiveFinalizationError`, selected `ArtifactDisposition.ROLLBACK`, and returned that error. `daydream/runner.py::_run_workspace` then reported artifact finalization failure and returned 1. Optional archive/evaluation/index failures therefore owned the same failure disposition as protected output publication.

The session inspected those symbols and exercised a real `runner.run` regression with an injected archive failure. Before the production change, an otherwise successful run returned 1, restored the prior explicit trajectory, and printed raw exception text. This was a new uncommitted regression test against the original production source, not a frozen discovery benchmark. Reading the historical source during PR preparation independently confirmed the coupling without resetting the checkout.

## Maintenance value

Give optional data collection its own failure policy while retaining one runtime artifact owner, freeze boundary, and publication transaction. A completed review should remain usable when a persistence destination or evaluator is unavailable. Integrity and required output errors must remain closed failures. This is a responsibility separation, rather than an archive format migration or a cosmetic deletion.

## Proposed simplification and preserved contracts

Reuse the existing protected snapshot and `ArtifactSession.finalize_frozen`. Report optional collection failure with a value-free diagnostic, then select the established complete or partial-evidence disposition from the review outcome. Distinguish identity/frozen-tree/trajectory-projection errors and explicit dump publication errors from optional archive persistence.

Preserve findings output projection, session confinement, opt-outs, exact frozen trajectory bytes, failed/interrupted primary errors, and current training/adjudication semantics. Withhold an incomplete requested dump while preserving its baseline; preserve an already prepared dump when indexing or uploading fails. Handoffs must not promise unavailable archive evidence. Do not add a second runtime artifact lifecycle, change the dataset format, or delete historical data in this ticket.

## Discovery trail and human judgment

The user requested `$implement github issue 1466`, then authorized subagents where needed. The agent read the issue and parent, inspected runner/finalization code, wrote the failing regression before production edits, and obtained independent boundary, Standards, and Spec reviews. The user then invoked `$create-improve-pr` after the implementation commit.

No matching curated record or embedded finding snapshot was found for this issue. This document was captured after implementation on 2026-10-04 using the session evidence and historical source. It does not claim a pre-implementation document, an original discovery timestamp, or an unknown external issue-author discovery process. `origin.kind: agent` describes the active session's code-grounded verification. Task/publication authorization is not an explicit human finding-validity verdict, so status remains proposed and human confirmation is null.

## Discovery matching criteria

- Essential: identify the concrete path by which optional persistence errors select runtime rollback or change a completed review's result, and distinguish it from genuine artifact/publication failures.
- Acceptable alternatives: separate typed errors, explicit outcomes, or another small separation at the existing finalization seam that uses the same protected snapshot and preserves output dispositions.
- Insufficient or invalid: swallowing all artifact exceptions; bypassing frozen evidence validation; printing exception payloads; adding a competing artifact lifecycle; guaranteeing nonexistent archive handoff links; or removing dataset evidence without preserving training semantics.
- Scope: archive/evaluation/index/upload symptoms share this one publication-policy root cause. The broader JSONL replacement epic is separate; it is not resolved by this fix.

## Source and benchmark provenance

Implementation began from clean commit `53e77423ade32f80b73aa2a6ace4973b269352fd` on the existing fix branch. The initial session's `git status --short` was empty, and verified history shows `1a5cfeb01e5951de164834922d8488c55f080bd5` has that exact parent. The same pre-fix SHA is the observed source identity; it is not inferred from a newly advanced merge base. Historical `git show` verified the named paths and symbols.

No immutable discovery benchmark was designated. Benchmark commit and snapshot remain null and applicability remains unverified. The original external discovery time, a durable transcript reference, and exact snapshots of intermediate uncommitted regression/failing-check trees are unavailable. The committed pre-fix source is retrievable, but that alone does not make this a benchmark-ready discovery case. `improve_curated/AGENTS.md` is absent; capture validation uses the publishing skill's canonical template and allowed field values.

This finding, its implementation issue, fix diff, PR description, and future history are reference answers. Exclude them from an evaluated agent's isolated frozen target; do not evaluate discovery by handing it this repair specification.

## Resolution and validation

Implementation commit `1a5cfeb01e5951de164834922d8488c55f080bd5` separates optional persistence from runtime publication, adds closed integrity/publication categories, sanitizes upload/collection diagnostics, preserves prepared dumps on optional later failures, and removes unpromised manifest/archive handoff links. Existing dataset format and source/runtime isolation remain intact.

Validation commands and observed failures/passes are retained in frontmatter. The final successful `make check` and archive-capture test file were run against the source/test tree committed as `1a5cfeb01e5951de164834922d8488c55f080bd5`. PR preparation verified a clean worktree and no differences from that implementation tree. Subsequent finding capture/backlinks are documentation-only and do not retroactively change the checked implementation identity. Resolution remains in progress while the draft PR is open.

## Savings

- Estimated at discovery: Not measured.
- Observed implementation change from `53e77423ade32f80b73aa2a6ace4973b269352fd` to `1a5cfeb01e5951de164834922d8488c55f080bd5`: production +82/-61 lines; tests +351/-28; documentation +7/-2. No generated files or lockfiles changed. Finding capture and backlink documentation are excluded from these implementation-only counts.
- Removed responsibility: optional archive errors deciding runtime rollback and review failure. Added typed boundaries and real-path failure-isolation coverage account for the net growth; LOC reduction is not the objective or training reward.

## Amendments

### 2026-10-04 — Post-implementation capture for draft PR publication

Captured the active session's verified opportunity after the completed fix. Original discovery timing and benchmark applicability remain unknown/unverified; no pre-fix capture or human confirmation was invented. The implementation issue link is retained in `resolution.issues`.
