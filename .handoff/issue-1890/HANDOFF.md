# Handoff: existential-birds/daydream#1890

**Issue:** a dirty fixing run reads supporting new-side source references at the committed `head_sha` instead of
the reviewed working-tree bytes.

This handoff was prepared read-only from `/home/user/daydream` on 2026-11-16, and it is the authoritative context
for the `/goal` prompt. The lane reports beside this file hold the file:line evidence:

* `runtime-contracts.md`
* `consumers-architecture.md`
* `tests-validation.md`

Markers used here:

* **[V]** verified by reading or running the code during preparation.
* **[I]** inferred, not verified.

Line numbers are at baseline `1756782b`; recheck them before editing.

## 1. Source and baseline

* **Source:** issue #1890 (bug, no comments, no linked prerequisites). It was found while fixing #1828, which
  landed as `2c8c5e1` (#1891) and was released in **0.36.0**.
* **#1828 added:**
  * the `tracked_worktree_edits` marker on `analyzed_revision`;
  * terminal/coverage version 3;
  * committed result schema 13 (`RESULT_VERSIONS["worktree"]`).

  Schema 13 has already shipped, so do not change it.
* **Baseline [V]:**
  * The prep checkout was on `claude/epic-brahmagupta-d65wn9` at `1756782b3d06ccc98c848e436f164aa82c7e5c5e`, with
    a clean worktree.
  * `git ls-remote origin main` returned `1756782b…` on 2026-11-16.
  * The local `origin/main` ref was stale (`be2e8c13`) and is an ancestor of HEAD.
* **Base for the implementation branch:** `origin/main`. Fetch it and record the actual SHA. Do **not** base the
  branch on `claude/epic-brahmagupta-d65wn9`: that branch carries only this handoff commit.

## 2. The defect [V, reproduced]

* **Dirty fixing path:**
  * `run_deep` (`daydream/deep/orchestrator.py:312-314`) builds the diff as `git diff <base>...HEAD` followed by
    `git diff HEAD` (`git_ops/queries.py:369-384`).
  * It sets `tracked_worktree_edits = git_ops.has_worktree_edits(work.repo)`.
  * The revision has the marker and no `source_id` (`:454-458`).
* **Where the wrong bytes are read:**
  * `StageInputFactory.admit_submission` (`review_stage_inputs.py:182`) builds
    `_SubmissionSources(revision, work.repo, aliases, captured_source=None)`.
  * The read goes through `_supporting_facts` and then `_source_contents` (`review_source_refs.py:70-78`), which
    falls through to `frozen_source(root, head_sha, path)`.
  * This is the **only production byte read**. The `_supporting_facts` fallbacks at `:229` and `:421` run only
    when `_submission is None` (tests only).
* **When the reads happen:** all byte reads take place during review admission (the `review` step, first in
  `STEPS`), before fix-gate, fix and verify run. Seal, `validate_committed_result` and `validate_target_decisions`
  compare `revision` labels only and never re-read bytes.
* **Reproduced:** a supporting `{path: base.txt, side: new, source_range: 2..2}` selector into an uncommitted line
  is refused with `allowed 1..1`. An in-range selector is admitted with HEAD's digest. See
  `tests-validation.md` §1 and §3.

## 3. Decision: option 1, as a host-only frozen tracked-worktree commit

Both investigation lanes recommend this independently.

**The design:**

* When `run_deep` sets `tracked_worktree_edits` (on the live-checkout branch at `orchestrator.py:312-314`), call
  `git_ops.stash_create(work.repo)` immediately after the diff. That function is at `queries.py:977-986`. It
  writes a dangling commit of the tracked index and worktree, and leaves working files, refs and the stash list
  untouched [V].
* The fix gate already calls it on the same checkout later in the run (`deep/fix_steps.py:203`).
* For a marked revision, new-side supporting reads use
  `frozen_source(root, <that commit>, path)` (`git_ops/source.py:67-98`), with the same 8 MiB bound,
  regular-blob policy and path policy.
* Old-side reads stay at `merge_base_sha`.

**Labels stay put.** `source_revision` is unchanged, so new-side refs keep `revision: head_sha`, qualified by the
marker. Nothing is added to `analyzed_revision`. The result is no schema, version, `diff_key` or
`planned_work_id` change. Under SemVer this is a compatible bug fix and goes under `### Fixed`.

**Why this option:**

* It keeps the AGENTS.md invariant "validate … against frozen side/range/fragment identities", the module
  docstring, and `docs/runtime.md` "pinned regular Git blobs".
* It reads the same tracked-only population as the marker. An untracked path gives `PathAbsentError`, which is the
  correct "unavailable" result.
* It uses Git's clean-filtered bytes, as the diff does.
* A proof of concept stayed stable after the live file was overwritten [V].

**Rejected alternatives:**

* **Reusing `CapturedSource` / `source_id`:**
  * `_revision_version` returns `None` when a revision has both `source_id` and the marker.
  * `validate_committed_result` ties `source_id` to schema 12 and the marker to schema 13.
  * The dataset policy ties `source_id` to `captured_working` and untracked files.
  * `WorkContext.captured_source` requires `repo == capture.root`, which would move the fixes into the capture.
  * It also adds untracked files and new refusals.
* **Option 2 (live working-tree reads):** not frozen, needs a new bounded no-follow reader plus a tracked-path
  check, reads raw bytes rather than filtered ones, and contradicts the docs and the reviewer instructions at
  `committed_review.py:938-939, 972-973`.

## 4. Changes, to verify against current code

1. **Carrier.** Add one keyword-only field to `WorkContext` (`daydream/workspace.py:94-97`), for example
   `reviewed_worktree: str | None`. It follows the existing review-only fields `review_pr` and
   `review_git_context`.
   * Set it with `replace(work, ...)` in `run_deep` before `FlowContext` is built (around `orchestrator.py:416`).
   * `StageInputFactory` reads it the way it reads `work.captured_source` (`review_stage_inputs.py:139`).
   * Do not fold it into `captured_source`, which drives full `validate()` in `_revalidate`.
   * **Fallback carrier,** if `WorkContext` validation or ownership makes the field awkward: an explicit keyword
     argument threaded `plan_committed_review` (`deep/review_terminal.py:115-139`) → `plan_review`
     (`committed_review.py:1150-1165`; read the committed-review runtime contract first) → `StageInputFactory`.
2. **Reader.** In `daydream/review_source_refs.py`, `_source_contents` (`:70-78`) must read a marked revision's
   new side from the snapshot. Thread the snapshot through `_SubmissionSources` (`:108-135`) and
   `_supporting_facts`, either as a parameter or as a small reader with `revision_for` and `read`.
   * The `:73` equality must still hold: `old` maps to `merge_base_sha` and `new` maps to `head_sha`.
   * `_SubmissionSources` is constructed in exactly one production place (`review_stage_inputs.py:182`).
3. **Fail closed.** A marked revision with no snapshot must never fall back to `frozen_source(head_sha)`.
   * Raise a **correctable** refusal for new-side supporting selectors, as `FrozenSourceRejected` becomes
     `SourceSelectorRejected` at `:230-234`. The message should read "new-side supporting source unavailable".
   * Do not raise a fatal `input_unavailable`. Supporting refs are optional, and AGENTS.md says supporting
     metadata cannot veto valid primary decisions.
   * Cases that produce no snapshot: `stash_create` raises on intent-to-add or unmerged entries [V for
     intent-to-add], or returns `None` because of a race.
   * Catch that failure in `run_deep` separately, so the review still runs. A `GitError` there today sets
     `diff=None`, which skips review; do not let that happen.
4. **No change to:**
   * `source_revision`, `part_source_mapping`, `AnalyzedRevision` and its schema;
   * `RESULT_VERSIONS`, `_revision_version`, `validate_committed_result`;
   * `findings.py`, `dataset*.py`, `captured_source.py`, `runner._capture_review_workspace`;
   * report-only, comment and `--findings-out` behavior;
   * clean-run digests.
5. **Snapshot timing.** Take the snapshot right after the diff, with no strict text-equality check. A strict check
   refuses a dirty submodule, because `git diff HEAD` lists it and the stash tree does not [V]. The race window is
   the same one `has_worktree_edits` already has.

## 5. Tests (net-neutral: extend, don't add)

* **Harness** (`tests/harness/stub_backend.py`): `_scripted_review` (`:107-154`) can cite only assigned
  `source_options`.
  * Add an optional per-claim `"supporting"` list, appended to `evidence_refs` at about `:151`.
  * Document it in the `review_claims` comment at `:31-35`.
  * It is backwards compatible: none of the other 12 or more `review_claims` users pass the key [V].
* **Journey:** extend `tests/test_missing_test_command_acceptance.py::test_runner_rejects_missing_canonical_before_fix_or_test`
  (`:59-121`).
  * In the dirty iteration only, write `base.txt = "base\nowner dirty bytes\n"`.
  * Set
    `backend.review_claims = [{"title": "Sample issue", "file": "api.py", "line": 1, "supporting": [{"path": "base.txt", "side": "new", "source_range": {"start": 2, "end": 2}, "basis": "source_inspection"}]}]`.
  * After the loop and **before** reading `test-verdict.json`, read `.daydream/deep/committed-review.json` and
    assert that the supporting refs in `findings[*].payload.evidence_refs` equal
    `[("base.txt", "new", {"start": 2, "end": 2}, sha256(working base.txt))]`.
  * A full sketch is in `tests-validation.md` §1 and `sketch/test_extended_sketch.py`.
  * On current code it fails at that assertion with `[] == [(...)]` and nowhere earlier [V].
  * Write the claim as a full literal, because mypy rejects `review_claims[0]` access.
* **Frozen proof.** A live read and a snapshot read both pass the journey. If the harness can mutate `base.txt`
  after planning and before submit without a production hook, assert that the reviewed digest is still
  recorded. Otherwise say in the PR that frozenness is proven by construction.
* **Negative control:**
  `tests/deep_orchestrator/test_review_native_sources.py::test_source_selectors_construct_supporting_evidence_from_the_pinned_git_blob`
  uses an unmarked revision and pins HEAD blob reads. It must stay green unedited, and it fails under an
  over-broad fix [V].
* **Pre-fix control:** run the extended test against unrepaired production code in an isolated checkout
  (test-audit: "Bug regressions"). Use the test-audit skill for the test changes.

## 6. Docs and changelog

* `docs/runtime.md` `#captured-local-source`, in the fixing paragraph at about `:462-467`: add one sentence. It
  should say that new-side supporting refs of a marked run measure the tracked working bytes frozen when the diff
  was taken, keep `revision: head_sha` beside the marker, treat untracked paths as unavailable, and leave the old
  side unchanged.
* `:802-810` "pinned regular Git blobs" stays true; link to it if useful.
* `CHANGELOG.md` `## [Unreleased]` → `### Fixed` (`:62`): add a user-facing entry for #1890.
* No AGENTS.md change is needed.

## 7. Work lanes

The change is small, so one implementer is fine. If you split it:

* **A — owns every shared file:**
  * `review_source_refs.py`
  * `review_stage_inputs.py`
  * `workspace.py`, or the fallback signature changes in `review_terminal.py` and `committed_review.py`
  * `deep/orchestrator.py`
  * the `git_ops` change, if any
* **B — tests:** `tests/harness/stub_backend.py` and `tests/test_missing_test_command_acceptance.py`.
* **C — docs:** `docs/runtime.md` and `CHANGELOG.md`.
* **Review:** an independent read-only review of the integrated diff.

## 8. Validation (commands verified from the Makefile, pyproject and CONTRIBUTING)

**Setup:**

```bash
uv lock --check
```

```bash
make install
```

**Owner test and negative control** (about 5 s):

```bash
uv run pytest tests/test_missing_test_command_acceptance.py tests/deep_orchestrator/test_review_native_sources.py -q
```

**Wider set** (about 72 s serial; add `-n logical` to parallelize):

```bash
uv run pytest tests/test_committed_review.py tests/test_review_result.py tests/test_findings.py tests/test_captured_source.py tests/test_host_tools.py -q -n logical
```

**Harness consumers:** the 25-file list in `stub-consumers.txt` (about 20 s), run with `-n logical`.

**Focused checks:**

```bash
make lint
```

```bash
make typecheck
```

```bash
make deadcode
```

```bash
git diff --check
```

**Gate,** before pushing (also run by the pre-push hook, which needs `make hooks`):

```bash
make check
```

Its budget is under 2 minutes. Never lower the coverage floor (`docs/coverage.md`).

**Run during preparation [V]:**

* `make install` passed (CPython 3.14.6).
* The named test passed (4.0 s).
* The 7 related files passed: 122 tests in 72 s serial.
* The 25 StubBackend consumer files passed: 118 tests in 20 s.
* `make lint`, `make typecheck`, `make deadcode` and `make check-naming` all passed.

**Not run:** `make check`, `make test` (coverage), `make package-check`, `make actionlint`.

## 9. Out of scope (separate concern)

* **Diff coordinate mixing** [V for the layering, I for the consequences]. The fixing diff concatenates two diffs,
  so assigned ranges, `part_source_mapping` labels, `hunk_index` and `head_line` placement mix
  merge-base, HEAD and working coordinates. A file with both committed and uncommitted hunks gets two blocks.
  * This is the issue's "assigned `api.py` ref … range 1..2 vs HEAD's 1 line" symptom.
  * It is **not** fixed by #1890. Do not change the shared `git_ops.diff`, which `runner.py:919` and
    `improve/recon.py:264` also use.
  * Mention it in the PR as a known separate concern.
* **Ignored-path marker:** `has_worktree_edits` ignores `ignore_paths`, so the marker can be set when only
  excluded paths are dirty. Minor and separate.

## 10. Risks and open questions

* **Dangling objects.** `stash create` writes dangling loose objects to the user's object store at review time.
  The fix gate already does this later in the same run, and `git status` already rewrites the index stat cache.
  If a reviewer objects, option 2 is the documented fallback.
* **GC.** The default `gc.pruneExpire` will not remove a fresh dangling commit mid-run [I].
* **Extension steps.** A custom extension step before `review` that mutates the tree is captured correctly by the
  snapshot, but only if the snapshot is taken at diff time.

## Files here

* `HANDOFF.md`: this document.
* The three lane reports listed at the top.
* `sketch/test_extended_sketch.py`: a scratch copy of the journey with the extension, using a `StubBackend`
  subclass instead of the harness edit. Its `.txt` is the pre-fix failure.
* `sketch/probe_supporting_facts.py` and its `.txt`: the unit-level reproduction.
* `stub-consumers.txt`: the 25 harness consumer test files and their baseline result.
