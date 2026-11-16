# Issue #1889 handoff context

Prepared 2026-11-16 by a prepare-implement session. Line numbers are at HEAD a68b1d7;
re-verify them before editing.

This file and `issue-1889-goal.txt` were force-added under the gitignored `.planning/`
directory and committed only to the handoff branch `claude/zen-gauss-iflqep`. Keep them out
of the implementation PR. Read them with:

```bash
git fetch origin claude/zen-gauss-iflqep
git show FETCH_HEAD:.planning/issue-1889-context.md
```

## Source and acceptance criteria

- Issue: https://github.com/existential-birds/daydream/issues/1889 ("fix(archive): a fixing
  run with a real diff and zero findings is archived as partial").
  - The body was rewritten on 2026-11-16 with the corrected cause, fix design, negative
    cases and acceptance-criteria checklist below.
  - The rewritten issue is authoritative. It has no comments and no linked PRs.
- Sibling: #1847 (no-diff case), fixed by commit 2c8c5e1 (#1891). That fix added
  `_sealed_no_diff()` and the `no_diff=` argument to `derive_pipeline_status` in
  `daydream/archive/pipeline.py`, and a no-diff block in the target test.
- Current acceptance criteria:
  1. An archived `--fix` (`output_mode="loop"`) run gets `pipeline_status: succeeded`
     when its sealed committed review, bound to the same session, is `complete` and has
     zero confirmed findings. Today it gets `partial`.
  2. Extend `tests/test_archive.py::test_current_merge_event_controls_pipeline_status`
     with that case, asserting `succeeded`. The assertion must fail on the current code.
     Prep verified that today `derive_pipeline_status("complete", None,
     {merge: ran/succeeded, the others absent}, runs_merge=runs_fix=runs_test=True)`
     returns `"partial"`.
- Scope boundary: archive `pipeline_status` derivation only. Do not change runtime flow
  control, phase events, exit codes, `phase_states` values, or the archive or manifest
  schemas.

## Baseline observed during prep

- Repository: `/home/user/daydream`, remote `origin` =
  https://github.com/existential-birds/daydream.
- Branch `claude/zen-gauss-iflqep` at HEAD `a68b1d7e5481df69c54c90535d3462150c884fea`,
  which equals `origin/claude/zen-gauss-iflqep`. It is the squash-merge
  `fix(artifacts): validate manifest walk order for prefix-named siblings (#1911)`,
  i.e. the tip of main when observed.
- Worktree is clean: no tracked edits and no untracked non-ignored files. Ignored local
  state: this `.planning/` file and a `.venv` that `uv run` built.
- Local `main` and `origin/main` are stale: `f636a1b` (2026-11-12, not fetched during
  prep). HEAD is 40 commits ahead of them and includes #1891.
- Requested base: none given, so use the captured HEAD `a68b1d7`. PR target: `main`.
- After prep, the handoff commit that adds these two `.planning/` files was pushed on top
  of `a68b1d7` to `origin/claude/zen-gauss-iflqep`.
  - Branch the implementation from `a68b1d7`, not from the handoff commit.
  - If the environment requires developing on `claude/zen-gauss-iflqep` itself, first
    `git rm` both `.planning/issue-1889-*` files in their own commit, so that the PR's
    net diff against `main` excludes them.

## Diagnosis (matches the rewritten issue)

- A zero-finding loop run does not reach `_step_fix`. It stops in `_step_fix_gate`
  (`daydream/deep/fix_steps.py:187-189`, `if not items: ... return Stop(0)`), before
  VERIFY, FIX, TEST, commit and remote-ci.
- The early return in `_step_fix` (`fix_steps.py:432-433`) cannot be reached with
  empty items in the built-in flow: round 1 is guarded by the gate, and later rounds
  `BreakLoop` in `_step_fix_verify` (645-661).
- Flow order (`daydream/deep/orchestrator.py:197-212`): review → load-items → diagram →
  findings-out → post-review → fix-gate → verify → fix → fix-verify → test → commit →
  remote-ci.
  - post-review seals the review: `merge_steps.py:66-70` → `finalize_review`.
  - fix and fix-verify run in `LoopGroup("fix-verify-loop")`
    (`daydream/extensions/builtins.py:95-108`).
- Phase states for this run (`derive_phase_states`, `pipeline.py:717-791`):

  | Phase | State | Source |
  |---|---|---|
  | merge | ran, succeeded | `phase_scope(MERGE)` at `daydream/deep/review_terminal.py:165` |
  | fix | absent | FIX `phase_start` is only emitted at `fix_steps.py:457` |
  | test | absent | `_test_state` (`pipeline.py:355-379`) reads only the session-bound `test-verdict.json` |
  | push, remote_ci | absent | they affect status only when ran is True and the state is partial |

- `runs_fix` and `runs_test` are static flow capabilities:
  - set in `capture_manifest_run_identity` (`daydream/run_artifacts.py:294-312`);
  - stored as `RunPhaseCapabilities` (`daydream/run_snapshot.py:25`);
  - read in `_manifest_state` (`pipeline.py:117-189`).
- `derive_pipeline_status` (`pipeline.py:799-861`) returns partial at lines 837-842,
  because required fix and test phases never ran.

## Evidence that proves "complete with zero findings"

`finalize_review` (`review_terminal.py:169-226`) writes two files to `.daydream/deep/`:

- **`review-coverage.json`**: `coverage.to_dict()`. It has no `analysis_state` and no
  findings, so it is insufficient on its own.
- **`committed-review.json`**: written by `write_committed_review_output`
  (`daydream/committed_review_output.py:84-91`), which validates it and requires
  `run_id` to match the artifact session. Keys:
  - `run_id`;
  - `findings`: the sealed, confirmed findings list;
  - `terminal_result`: `run_id`, `required_phases`, `phase_outcomes`, `pipeline_state`,
    `projection_valid`, `analysis_state` and `reason_codes`
    (`daydream/review_result.py:509-516`; `daydream/committed_review.py` ~905-925).
  - It has no `DeepArtifact` member; the path is `state.dd / "committed-review.json"`.

Further facts:

- The no-diff path (`orchestrator.py:506-507`) also calls `finalize_review`, so this
  file exists there too, with `terminal_result.required_phases == ["no_diff"]`.
- Production archives it. `tests/test_archive_data_capture.py:191-194` already asserts
  that the archived `deep/committed-review.json` has `run_id == manifest session_id`.
- Do not key on `merged-items.json`. It is not session-bound, and extension steps may
  rewrite it (delivered vs confirmed counts).
- `docs/runtime.md:26-27`: a finding count alone cannot establish completeness. Require
  `analysis_state == "complete"`.

## Recommended design (archive-side)

1. In `daydream/archive/pipeline.py`, read `<target>/.daydream/deep/committed-review.json`
   once per manifest. Accept it only if all of these hold:
   - it is a dict;
   - `run_id == session_id`, and `session_id` is not None;
   - `terminal_result` is a dict with `analysis_state == "complete"`.

   Missing, malformed or stale files give False and never turn green.
2. Zero findings means `findings == []`: an exact empty list.
3. In `_manifest_state`, pass `runs_fix=runs_fix and not zero_findings` and
   `runs_test=runs_test and not zero_findings` to `derive_pipeline_status` only.
   - Keep `phase_states` truthful: fix and test stay `{"ran": False, "status": "absent"}`.
   - Leave the fix_failures, recommended_capture and quality-gate reads unchanged.
   - Merge stays required. Its `succeeded` state is the positive evidence at
     `pipeline.py:856`.
4. Consolidate where it simplifies. Prefer one sealed-result reader that also derives
   `no_diff` (`terminal_result.required_phases == ["no_diff"]`, complete), replacing
   `_sealed_no_diff`'s `review-coverage.json` read.
   - If you do, move the test's no-diff block to write `committed-review.json`.
   - Do not keep two readers of sealed review state when one suffices.
   - Keep the `derive_pipeline_status` signature unless removing or renaming `no_diff`
     is simpler.
5. Validation strength:
   - The `_sealed_no_diff` precedent checks exact fields without the full validator.
   - `validate_committed_result` (`daydream/committed_review_model.py:210`) would need a
     fully valid sealed result in the unit test. Building one requires a review program
     (see `tests/test_committed_review.py`), which is expensive.
   - Exact-field checks are acceptable.

Rejected: recording a noop FIX/TEST phase on the early return.

- `_fix_state` (`pipeline.py:330-352`) treats any FIX `phase_start` as succeeded,
  whatever its end status.
- TEST has no event path, so a noop TEST would need a fabricated `test-verdict.json`.
- It emits OTel phase events for work that never happened.
- It still needs a completeness guard.

## Must stay partial (negative cases)

- A sealed result from another session (`run_id` "prior").
- A complete review with at least one sealed finding where fix never ran. Examples: an
  interactive decline at `fix_steps.py:143-145`, or an extension that emptied the items.
- A review that is not complete (incomplete, failed or cancelled) with zero findings.
  Such runs can still exit 0.
- `committed-review.json` absent or malformed.

Acceptable: a zero-finding run whose interactive prompt was declined becomes succeeded,
since there was nothing to fix.

## Docs, schema, changelog

- No schema or version bump:
  - the set of `pipeline_status` values is unchanged;
  - the index `SCHEMA_VERSION = 10` (`daydream/archive/_schema.py`) and
    `MANIFEST_SCHEMA_VERSION` "3.0" stay as they are, because bumping the index version
    refuses existing `index.db` files;
  - archived rows are not rewritten.
- Consumers:
  - producer: `pipeline.py`;
  - caller: `daydream/archive/finalize.py:244`;
  - field: `daydream/archive/manifest.py:106`;
  - index: the `pipeline_status` column;
  - no readers in dataset, training, eval or benchmarks.
- The authoritative home for the rule is the `derive_pipeline_status` docstring
  (`pipeline.py:809-824`, rules 4-5). Also update the `_manifest_state` docstring and any
  reader docstring. Add no second home in `docs/` (no doc describes `pipeline_status`).
- Do not edit the historical 0.36.0 entry at `CHANGELOG.md:304-307`.
- CHANGELOG: add a user-impact bullet under `## [Unreleased]` → `### Fixed`
  (`CHANGELOG.md:62`), ending with
  `([#1889](https://github.com/existential-birds/daydream/issues/1889))`.
- SemVer: a compatible, patch-level bug fix with no stored-format change.

## Tests

- Keeper: extend `test_current_merge_event_controls_pipeline_status`
  (`tests/test_archive.py` ~648-680) and add no new test function.
- Helpers:
  - `review_coverage`: `tests/harness/review_result.py:9`
  - `_write_deep`: ~L350
  - `_manifest_write_snapshot(store, *, session_id, ...)`: ~L176
  - `_archive_snapshot`: ~L194. Its default identity is a fixing run: every
    `RunPhaseCapabilities` flag is True (`_manifest_identity` ~L166).
  - `_merge_events(session_id, status)`: ~L640.
- `frozen_extra={"phase_events": _merge_events("current", "succeeded")}` gives
  merge ran/succeeded.
- Rows to add, all on the same `run`:

  | Case | Expected |
  |---|---|
  | current session, complete, `findings: []` | succeeded (fails on HEAD) |
  | `run_id` "prior" | partial |
  | complete with one finding | partial |
  | not complete, `findings: []` | partial |

  Drop any row that only duplicates the existing no-diff `("prior", "partial")`
  session-binding proof.
- Nothing else asserts partial for a zero-finding fixing run.
  `test_archive_retains_malformed_frozen_merge_evidence_as_unknown` (~L690-704) has no
  `committed-review.json`, so it stays partial.
- Other `pipeline_status` asserters to rerun:
  - `tests/test_archive_data_capture.py:176`
  - `tests/deep_orchestrator/test_push_fix_state_and_stabilization.py:13,213,299`
- Do not add an end-to-end archive run. `tests/deep_orchestrator/test_findings_artifacts.py`
  L86-93 already runs a zero-finding loop with `archive=False`, and adding archiving there
  slows the gate.
- For an optional manual probe, which is not committed:
  - `linked_worktree`: `tests/conftest.py:120`
  - `make_config`: `:268`
  - `archive_dir`: `:472`
  - `StubBackend.review_claims = []`: `tests/harness/stub_backend.py:36`
- Run `$test-audit` (`.agents/skills/test-audit/SKILL.md`) on the changed test.

## Commands

Commands come from the Makefile and CONTRIBUTING.md. Prep ran only the first two.

- `uv lock --check` (passed during prep).
- `uv run pytest tests/test_archive.py -k pipeline_status -q` (1 passed on HEAD during
  prep).
- `uv run pytest tests/test_archive.py tests/test_archive_data_capture.py tests/deep_orchestrator/test_push_fix_state_and_stabilization.py -q`
- `make lint`, `make typecheck` and `make deadcode`.
- The gate, `make check`: lockcheck, install, static, test, coverage-report.
  - `fail_under = 90` (`pyproject.toml`) is enforced in `make test`.
  - Never lower it (`docs/coverage.md`).

## Publication and review

- Commit: signed Conventional Commit with explicit `git add <path>` and hooks enabled.
  Example: `fix(archive): archive a complete zero-finding fixing run as succeeded`.
- PR body:
  - follow `.github/PULL_REQUEST_TEMPLATE.md` through `.agents/skills/pr`;
  - include `Closes #1889`;
  - state the SemVer patch impact;
  - complete every checklist item with evidence in the Test Plan.
- The simplification sweep is required only for more than 5 files or more than 500
  lines (CONTRIBUTING.md#pull-request-requirements). This change should touch about 3
  files.
- Daydream review: `daydream . --backend pi --comment --non-interactive --capture-data`.
  Append `--base <verified PR base SHA>` only if the default base differs.
- Then run `fetch-pr-feedback --include-author --include-self` → `receive-feedback` →
  `respond-pr-feedback`, including `greptile-apps`.
