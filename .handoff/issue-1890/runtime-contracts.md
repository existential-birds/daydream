# #1890 — runtime/contracts lane (read-only investigation)

Repo `/home/user/daydream`, HEAD `1756782b`. No tracked file was edited, staged or checked out. Experiments
ran only in throwaway Git repos under the session scratchpad (not in this repo), importing the repo's
`.venv` package. This file is the only file written in the repo (gitignored by `.gitignore:72 .scratch/`).

Legend: **[V]** verified by reading code or running it; **[I]** inferred.

## TL;DR

* **Recommendation: option 1, built from existing owners and kept host-only.** When `run_deep` marks a
  fixing revision with `tracked_worktree_edits`, freeze the tracked working state as a dangling commit with
  the existing `git_ops.stash_create` (`daydream/git_ops/queries.py:977-986`, already called on the same
  checkout by the fix gate at `daydream/deep/fix_steps.py:203`). Read new-side supporting bytes for the
  marked revision with the existing `frozen_source(root, <that commit>, path)` (`daydream/git_ops/source.py:67-98`).
  The commit is **not** written into `analyzed_revision`. `source_revision` is unchanged, so new-side refs
  keep `revision: head_sha`, qualified by the marker. Result 13, terminal v3, `diff_key`, clean-run digests,
  report-only review and old-side reads stay unchanged.
* **[V]** A scratch proof of concept passed a duck-typed view (`revision_for`/`read`) through the existing
  `captured_source` parameter of `canonicalize_source_selectors`. A new-side selector to uncommitted lines
  6..7 was accepted with the working-bytes digest, and the old side still read the base. After the live file
  was overwritten (standing in for a fix), the view still returned the reviewed digest. On current code the
  same selector is refused (`allowed 1..5`) and an in-range selector gets HEAD's digest.
* **Do not** reuse `source_id`/`CapturedSource` for the fixing path:
  * it collides with four exclusivity checks (`_revision_version`, `validate_committed_result`, the dataset
    run-record policy, `findings.py`);
  * it would turn `work.repo` into the capture, which fixes would then write into;
  * it adds refusal modes to fixing runs: skip-worktree, 256 MiB/100k files, Git env overrides;
  * its population includes untracked files, while the marker means tracked edits only.
* Option 2 (live working-tree reads) is workable, because every byte read happens before the first fix step
  [V]. But it breaks the "frozen side/range/fragment identities" invariant (AGENTS.md) and the
  `review_source_refs` module contract. It needs a new bounded no-follow reader and a tracked-path check, and
  two submissions can see different bytes for the same path. It is the fallback if maintainers refuse to
  write Git objects at review time.

---

## 1. How report-only review captures working source

**Where `CapturedSource` is created [V].** Two call sites use `capture_source(...)`
(`daydream/git_ops/captured_source.py:596-719`):

* **The deep flow** (CLI/embedded/bot `review`/`comment`, or any `--findings-out`) captures in
  `runner._capture_review_workspace` (`daydream/runner.py:692-760`). That function is entered at
  `runner.py:576`, wrapping `open_artifact_session` at `:582`.
  * The early return at `runner.py:703-710` skips capture for custom flows, `diagram`, and
    `loop` (fixing) without `--findings-out`. It also prints the #1828 "Fixing reviews the checkout as is"
    notice.
  * Otherwise `capture_source(work.repo, destination, base_sha=base, head_sha=head, local=not committed,
    exclude=config.ignore_paths)` runs at `:742-743`, with
    `destination = operational_worktree_root(owner) / f"review-capture-{uuid}"` (`:741`).
  * The function yields `replace(work, repo=captured.root, captured_source=captured, ..., is_ephemeral=True)`
    (`:751-753`). Reviewers therefore run with cwd at the capture root
    (`committed_review.py:1596 repo = review.inputs.work.repo`).
* **The `run_committed_review` API** captures at `daydream/committed_review.py:1293-1296` and registers
  `lifetime.callback(_remove_capture, captured)` at `:1295`.

**What `source_id` is [V].**

* The revision's `source_id` is a bare 64-hex SHA-256: `hashlib.sha256(json.dumps({"version": 1, "base":
  base_sha, "before": before, "after": after, "population": population, "exclude": scope},
  sort_keys=True))` (`captured_source.py:695-697`). Only `local=True` sets it; a committed capture has
  `source_id=None`.
* The schema pattern is `^[0-9a-f]{64}$` (`daydream/review_result.py:124`).
* The `source:N` / `source:<64-hex>` form is unrelated: it names a **selector handle** for an assigned
  reference (`review_source_refs.py:49` schema; `source_id(reference)` at `:138-141` hashes a reference's
  identity; scope-local `source:N` aliases are assigned at `review_stage_inputs.py:151-155`).

**How it is stored on the revision dict [V].** `AnalyzedRevision(..., source_id=...)`
(`review_result.py:160-183`; `to_dict` drops `None`). The deep flow sets it at
`orchestrator.py:267, 456-458`; the API sets it at `committed_review.py:1347-1350`. It is stored nowhere
else on the revision.

**Lifetime and cleanup [V].**

* `CapturedSource.remove()` (`captured_source.py:459-462`) checks storage identity, then runs `rmtree`.
* In the deep flow the capture is removed in the `finally` of `_capture_review_workspace`
  (`runner.py:754-758`), i.e. after the artifact session and finalization nested inside it. An integrity
  failure keeps the capture and prints a warning.
* In the API it is removed on `AsyncExitStack` exit (`committed_review.py:1295`).
* `ReviewProgram.seal` revalidates the capture at seal time (`committed_review.py:1139-1147`). Every admission
  revalidates it through `_revalidate` (`review_stage_inputs.py:61-69`), which off-loop is `Review.check`
  (`committed_review.py:241-244`).

**Bounds [V].**

* **Refusals before capture:** inherited Git routing/config env overrides (`captured_source.py:605-607`);
  a destination that is not absent or not disjoint (`:609-612`).
* **Population** (`_population`, `:214-240`): tracked index entries plus `git ls-files` untracked
  nonignored files (`queries._list_untracked_z`), minus Daydream runtime artifacts. It refuses omitted
  skip-worktree paths (`:230-233`) and more than 100,000 files (`:236-237`).
* **Size:**
  * whole working population ≤ 256 MiB (`:647-648`);
  * each file ≤ 256 MiB (`:150-153`);
  * blob reads in `read()` use `_blob` with `immutable._MAX_BLOB_BYTES` = 8 MiB (`:372-377`,
    `immutable.py:29`).
* **Other refusals:** tracked generated Daydream artifacts (`:630-631`).
* **Path confinement:** `_selector` rejects absolute paths, `..`, `.git`, backslash and NUL (`:59-63`).
  Leaves are opened `O_NOFOLLOW` under pinned parents (`immutable._parent`, `_regular_leaf`). Each leaf's
  fingerprint is checked before and after reading; instability retries 3× (`:616-713`).
* **Symlinks and submodules** keep their metadata and are never followed (`:131-148`).

**`revision_for(side)` / `read(side, path)` [V].**

* `revision_for` (`:446-449`) returns `base_sha` for `old`, `source_id or head_sha` for `new`, and raises
  `FrozenSourceRejected` for any other side.
* `read` (`:486-509`):
  1. checks storage directory identity and Git metadata (`HEAD`/`config`/index semantics);
  2. looks the path up in the accepted `_before`/`_after` inventory. An absent path raises
     `PathAbsentError`; a directory or non-regular entry raises `FrozenSourceRejected`;
  3. reads the blob under an inert Git context, bounded to 8 MiB;
  4. for the `new` side, also checks that the native file in the capture still has the accepted mode/oid;
  5. returns `(oid, bytes)`.

`_source_contents` (`review_source_refs.py:70-78`) uses the capture when one is given. It requires
`captured_source.revision_for(side) == source_revision(revision, side)` (`:73-74`). A revision that has
`source_id` but no capture raises `GitError('captured working source is unavailable; restart review')`
(`:76-77`). Every other case reads `frozen_source(root, source_revision(revision, side), path)` (`:78`).

## 2. The fixing path: revision dict, diff, and the #1828 marker

**Commit [V].** #1828 landed as `2c8c5e1` "fix: truthful run records for dirty fixing runs, no-diff
archives and publication failures (#1891)" (`git log --grep 1828` and `-S tracked_worktree_edits` both find
only this commit). It is released in **0.36.0 (2026-11-15)** (`CHANGELOG.md:86`, entry at `:282-291`).
Schema 13 is therefore already shipped.

**Workspace [V].** `open_workspace` stays in place for fixing unless `--branch`/`--worktree`
(`workspace.py:160-177`). `work.repo` is then the user's live checkout. With `--worktree` the checkout is
a fresh detached worktree, which is clean, so no marker is set.

**Diff [V].**

* `run_deep` (`daydream/deep/orchestrator.py:272-319`): with no capture and no `--findings-out`, it computes
  `diff = git_ops.diff(work.repo, work.base_branch, exclude=config.ignore_paths)` (`:313`), then
  `tracked_worktree_edits = git_ops.has_worktree_edits(work.repo)` (`:314`).
* `git_ops.diff` (`git_ops/queries.py:369-384`) **concatenates two diffs**: `git diff <base>...HEAD`, then
  `git diff HEAD` (tracked working tree vs HEAD).
* Reproduced in a scratch repo: a file changed both in a commit and uncommitted appears as **two blocks**.
  The first block's hunks are numbered base→HEAD; the second block's are HEAD→working.
* `has_worktree_edits` (`queries.py:620-627`) is `git status --porcelain` minus `??` and Daydream artifacts.
  `git status` itself rewrites the index stat cache (verified in a scratch repo: `.git/index` bytes change).

**Where the marker is set [V].**

* `orchestrator.py:269-270` initialises `tracked_worktree_edits = False`; `:314` sets it, only in the
  live-checkout branch.
* `:454-458` passes `AnalyzedRevision(captured_head, captured_base, current_diff_sha, pr_base_sha,
  source_id=source_id, tracked_worktree_edits=tracked_worktree_edits).to_dict()` to
  `plan_committed_review`.
* `diagram` mode also runs `:313-314`, but it skips planning (`:449`). In practice the marker reaches review
  only for in-place `loop` runs.

**Consumers of the marker [V]:**

* **Terminal/coverage version:** `review_result._revision_version` (`:153-157`) returns 1/2/3, or **`None`
  when a revision has both `source_id` and the marker**. The value is used by
  `validate_terminal_result` (`:318-320` region) and `ReviewCoverage` evidence (`~:469-474`).
  `ANALYZED_REVISION_SCHEMA` (`:122-127`) is strict, and the marker is `{'const': True}`.
* **Committed result:**
  * `RESULT_VERSIONS = {..., "worktree": 13}` (`committed_review_model.py:128`); `WORKTREE_RESULT_SCHEMA`
    (`:205`).
  * `validate_committed_result` (`:224-226`) requires `source_id ⇔ 12` and `marker ⇔ 13`.
  * `Review.seal` picks 13 (`committed_review.py:927-930`).
* **Findings export:** `findings.py:105` refuses `source_id` or the marker in a commit-bound review artifact.
* **Run records / datasets:**
  * `retain_original_task` gets an `AnalyzedRevision` **without** the marker (`orchestrator.py:326-327`).
  * The run-record schema `original_task.analyzed_revision` is strict, with only optional
    `pr_base_sha`/`source_id` (`dataset.py:84-86`, `_object` sets `additionalProperties: false` at `:56-58`).
  * Dirty state is recorded instead as `input_scope: "committed_and_tracked_worktree"` and
    `dirty_tracked: True` (`dataset_capture.py:45-48, 64-67`).
  * `_validate_run_evidence` (`dataset.py:368-379`) requires `source_id in analyzed_revision` ⇔
    `input_scope == "captured_working"` ⇔ `untracked_files_included`. `final_state.source_id` must equal
    it (`dataset_capture.py:199-200`).
  * `findings.terminal_coverage` is a free-form map (`dataset.py:98`), so it carries v3 with the marker.
* **Tests that pin it:** `tests/test_committed_review.py:704-723` (version/marker binding);
  `tests/test_missing_test_command_acceptance.py:86-95` (dirty journey: coverage `(3, True)`);
  `tests/test_review_result.py:72`; `tests/test_findings.py:192`.

## 3. Why fixing doesn't capture, when references are validated, and conflicts

**Why no capture [V, docs + code].**

* `docs/runtime.md:459-468` says: "Fixing-flow migration is outside this contract: fixing reviews the
  checkout as is, so its diff is the commits plus tracked working-tree edits."
* In code, capture replaces `work.repo` with the private capture root (`runner.py:751-753`). Fix,
  verify, test and commit write `work.repo` (e.g. `fix_steps.py:202-207` snapshots `ctx.work.repo`).
  Capturing would point fixes at the private copy.
* Capture also changes the subject: a net base→working diff that includes untracked files. It adds
  refusals (skip-worktree, size, env overrides), a full object-store copy (`_copy_objects`, `:253-267`) and
  full `validate()` passes.

**When source references and selectors are byte-checked [V].**

* **The only byte reads are inside review admission.** The chain is `StageInputFactory.admit_submission`
  (`review_stage_inputs.py:172-189`) → `_SubmissionSources(self.revision, self.work.repo, aliases,
  self.captured_source)` (`:182`) → `canonicalize_source_selectors` → `_supporting_facts` →
  `_source_contents` (`review_source_refs.py:88-97, 228-229, 420-421`).
* `admit_submission` is called only from the `admit` closure (`committed_review.py:740`). That closure
  serves `review_admit`, the final `structured_output`, and the rejected-final-batch salvage run by
  `Review.close(work_id)` (`:762-781`).
* `Review.close` is called only in each work item's `finally` inside the schedule (`committed_review.py:1677`),
  so it runs during `program.dispatch`.
* Dispatch runs only in the `review` step (`deep/review_terminal.py:142-166`, `_step_committed_review`).
  That step is the **first** step of `STEPS` (`orchestrator.py:197-210`); `fix-gate`/`verify`/`fix`/
  `fix-verify`/`test`/`commit` come later.
* **Nothing reads bytes after review:**
  * `Review.seal` (`committed_review.py:843-850`) only calls `close_authority()`, with no salvage.
  * `ReviewProgram.seal` revalidates only a `captured_source` (`:1139-1147`), and that is `None` for fixing.
  * `validate_committed_result` → `check_refs` (`committed_review_model.py:256-266`) compares `revision`
    strings and calls `validate_source_refs` on **assigned** refs only, with no `source_root`.
  * `validate_target_decisions` (`review_result.py:281-284`) compares `revision` strings only.
  * `review_source` lookups serve canonical diff bytes and never call `frozen_source` or
    `_source_contents` (grep).
* **One review per run:** `_run_loop_deep` calls `run_deep` once (`runner.py:1095-1109`). `plan_committed_review`
  has a single caller (`orchestrator.py:454`). "There is no shallow mode and no resume"
  (`docs/runtime.md:1096-1097`); "Every review starts fresh" (`:569-571`).

**Would option 2 read post-fix bytes?** Not in the built-in flow [V]: every read happens before the fix
gate. That holds by ordering, not by structure. A custom extension step placed before `review` that mutates
the tree, or concurrent edits by the user or the reviewer's own `executed_check` commands, would change
what option 2 reads mid-review [I]. Native reviewer tools run with cwd = `work.repo`, the live checkout
(`committed_review.py:1596, 1645-1658`), so option 2 matches what the reviewer can read natively. Option 1
is immune: the PoC overwrote the file after freezing and got the same digest [V].

**Would option 1 conflict with invariants?** It depends on how it carries identity:

* **Putting `source_id` on the marked revision breaks:**
  1. `_revision_version` → `None` (`review_result.py:157`), so coverage evidence and terminal validation
     fail;
  2. `validate_committed_result` `:224-226`, because `source_id` means 12 and the marker means 13;
  3. dataset `_validate_run_evidence` (`dataset.py:372-374`), because `source_id` implies
     `captured_working` and `untracked_files_included`, which the fixing diff lacks;
  4. `source_revision` would send **old**-side reads unchanged but **new and `diff`** sides to `source_id`
     (`review_source_refs.py:66`, and `:363` for the `diff` side), so assigned refs move too.

  And with `source_id` in the revision but no `captured_source` passed, `_source_contents` raises "restart
  review" (`:76-77`).
* **Putting a `CapturedSource` on `work.captured_source` breaks run_deep's capture semantics.** run_deep
  would then take `diff = captured.diff` (net, not layered; `orchestrator.py:307-308`), `changed_files =
  captured.files` (`:386-387`) and `source_id` (`:267`). Every admission would run a full `validate()`
  (`review_stage_inputs.py:61-66`), and seal would revalidate (`committed_review.py:1141-1146`). Its
  `revision_for('new')` would be the 64-hex `source_id`, failing the equality at `review_source_refs.py:73`.
* **Adding any new key to the revision** (e.g. the frozen commit) changes the strict
  `ANALYZED_REVISION_SCHEMA` (`review_result.py:122-127`) that 0.36.0 readers enforce. That means a new
  result/terminal version, or a documented breaking change, and it changes `planned_work_id` for dirty runs.
* **A host-only view does not touch** recovery/resume (there is none), sealed output (no byte reads at
  seal), the committed-result schema, or `source_revision`'s `diff` side.

## 4. Recommendation and the exact changes

### Recommendation: option 1 via a host-only frozen tracked-worktree commit

**Why (grounded in code):**

* **Frozen, as AGENTS.md requires** ("Validate submitted source references atomically against frozen
  side/range/fragment identities"). This also keeps the module contract "references to frozen review
  inputs" (`review_source_refs.py:1`) and runtime.md's "pinned regular Git blobs" (`docs/runtime.md:802-805`).
  Option 2 breaks all three.
* **Reuses two existing owners with identical bounds and errors.** `git_ops.stash_create`
  (`queries.py:977-986`: "Capture tracked worktree and index changes as a dangling stash commit. Leaves
  working files and stash refs untouched") and `frozen_source` (8 MiB limit, regular-blob policy, path
  selector policy, `FrozenSourceRejected`/`PathAbsentError` mapping). Rejection messages at
  `review_source_refs.py:230-244` therefore stay accurate. Option 2 needs a new no-follow reader with an
  8 MiB bound and a tracked-path check.
* **Same population as the marker** (tracked only). [V] the stash tree holds staged-new and modified tracked
  files, and `frozen_source` on an untracked path raises `PathAbsentError`, which is the correct "unavailable"
  rejection.
* **Same bytes as the diff text** (both go through Git's clean/EOL path); option 2 would digest raw bytes.
  [V] `git diff HEAD` == `git diff HEAD <stash>` for modified and staged-new files.
* **Precedent on this checkout in the same run.** The fix gate already creates such a commit
  (`fix_steps.py:203`), `phases/testing.py:586` does too, and `git status` (`has_worktree_edits`) already
  rewrites the index stat cache [V]. `stash create` leaves refs and stash list untouched [V]. It only adds
  dangling loose objects.
* **No schema, version or digest change**, because the commit stays host-only and `source_revision` is
  unchanged.

### Functions to change (smallest cohesive set)

1. **`daydream/deep/orchestrator.py::run_deep`** (`:312-314`): when `tracked_worktree_edits` is true (and
   the mode plans a review), freeze `git_ops.stash_create(work.repo)` next to the diff. Hand it to review
   planning, ideally as a `WorkContext` kw-only field like `captured_source`/`review_pr`
   (`workspace.py:94-97`), set via `replace(work, ...)` before `FlowContext` (`:416`). `StageInputFactory`
   already reads `work.captured_source` (`review_stage_inputs.py:139`), so this avoids signature changes
   through `plan_committed_review`/`plan_review`. The other route is an explicit kwarg through
   `review_terminal.plan_committed_review` (`:115-139`) → `committed_review.plan_review` (`:1150-1165`) →
   `StageInputFactory.__init__`.
2. **`daydream/review_stage_inputs.py::StageInputFactory`** (`:136-189`): pass the view into
   `_SubmissionSources(...)` at `:182`. Do **not** fold it into `self.captured_source`, which drives the
   full `validate()` in `_revalidate`.
3. **`daydream/review_source_refs.py`:**
   * `_source_contents` (`:70-78`) and the parameter types of `_supporting_facts` (`:88-97`),
     `_SubmissionSources` (`:108-135`), `canonicalize_source_selectors` (`:157-163`) and
     `validate_source_refs` (`:371-374`).
   * Either widen `captured_source` to a small reader protocol (`revision_for`, `read`), or add a separate
     `worktree` parameter.
   * The view's `revision_for` must return `merge_base_sha` for `old` and `head_sha` for `new`, so the
     `:73` equality still holds. Its `read` uses `frozen_source(root, base or commit, path)`. The PoC used
     exactly this shape.
   * **Fail closed:** a revision with `tracked_worktree_edits` and no view must not fall through to
     `frozen_source(head_sha)`, which is the bug. Mirror `:76-77`. Decide the error type deliberately (see
     open questions): `GitError` becomes `HostToolFatal input_unavailable` at `:263-267`;
     `FrozenSourceRejected` becomes a correctable `SourceSelectorRejected` at `:230-234`.
4. **No change to** `source_revision` (`:64-67`), `part_source_mapping` (`:311-368`), `AnalyzedRevision`,
   `RESULT_VERSIONS`, `_revision_version`, `validate_committed_result`, `findings.py`, `dataset*.py`,
   `runner._capture_review_workspace` or `captured_source.py`.

**Binding the view to the diff (implementer must choose) [V for the edge cases]:**

* A strict text equality check of `git diff HEAD <commit>` against the diff's worktree layer would refuse
  runs with a **dirty submodule**. `git diff HEAD` lists `sub`, but the stash tree diff does not [V].
* Deriving the worktree layer from the commit would drop that entry from the reviewed diff. It would also
  need a `git_ops.diff` parameter, and that function is shared with `runner._gather_diff_seed` (`:919`) and
  `improve/recon.py:264`.
* Taking the commit right after the diff, with no check, leaves the same TOCTOU window that today's
  `has_worktree_edits` call at `:314` already has.

**`stash create` failure modes [V/I]:**

* Intent-to-add entries make `git stash create` fail ("Cannot save the current worktree state") [V], while
  `git diff HEAD` includes the file. The fix gate's own `stash_create` (`fix_steps.py:203`) fails the same
  way today, after review.
* Unmerged paths: same expected [I].

Pick either a correctable "new-side supporting context unavailable" (no view, marker present) or an early
refusal. The first keeps today's "review then fix-preflight failure" behavior for these runs.

### Invariants and contracts that must not change

* `analyzed_revision` shape:
  * clean runs keep no marker, no `source_id`, terminal v1, result 10;
  * marked runs keep `{head_sha, merge_base_sha, diff_key[, pr_base_sha], tracked_worktree_edits: true}`,
    terminal v3, result 13;
  * no `source_id` on a fixing revision (`review_result.py:122-157`, `committed_review_model.py:128, 205,
    224-226`).
* `diff_key` and the canonical diff bytes for clean runs (`orchestrator.py:313, 362`), and ideally for dirty
  runs too. `planned_work_id` and `source_id(reference)` for clean runs.
* `source_revision` mapping: old → `merge_base_sha`; new/diff → `source_id` or else `head_sha`
  (`review_source_refs.py:64-67`). New-side refs on marked runs keep `revision: head_sha`.
* Report-only/comment/`--findings-out` capture behavior and `source_id` binding (`runner.py:703-758`,
  `captured_source.py` unchanged). This includes the "restart review" guard for a `source_id` revision
  without its capture (`review_source_refs.py:76-77`).
* Old-side reads stay `frozen_source(root, merge_base_sha, path)`.
* Selector bounds: 8 MiB retrieval, whole-file UTF-8, confined path, `source_range ⊆ 1..lines`,
  ≤4 selectors/8 KiB. Error classes stay the same: correctable `SourceSelectorRejected` versus fatal
  `input_unavailable` (`review_source_refs.py:170-272`).
* Callback-local facts: no validation success survives a callback (`_SubmissionSources.close`, `:133-135`;
  `docs/runtime.md:507-515`).
* Run records (`dataset.py:84-93, 368-379`): `input_scope`/`dirty_tracked`/`untracked_files_included`
  unchanged.
* Clean-checkout negative control: `tests/deep_orchestrator/test_review_native_sources.py::
  test_source_selectors_construct_supporting_evidence_from_the_pinned_git_blob` (per the tests lane) must stay
  green, so the change must stay conditional on the marker.

## 5. Docs and changelog to update

* `docs/runtime.md` "Captured local source", fixing carve-out (`:459-468`): add one sentence saying that
  new-side supporting references of a marked fixing run read the tracked working bytes frozen when the diff
  was taken, keep `head_sha` as `revision` beside the marker, and treat untracked paths as unavailable. Say
  that the old side is unchanged.
* `docs/runtime.md:802-810` ("pinned regular Git blobs … 8 MiB"): still true under option 1; a link back to
  the carve-out may be enough. Under option 2 this sentence would need an explicit live-read exception.
* `docs/runtime.md:563-566` (schema list) is unchanged under the recommendation. It would need a new
  version if any revision key were added.
* `daydream/review_result.py:162-166` `AnalyzedRevision` docstring: optional one-line note.
  `review_source_refs.py:1` docstring stays true under option 1.
* `CHANGELOG.md` `## [Unreleased]` → `### Fixed` (`:8`, `### Fixed` at `~:62`): a user-facing entry saying
  that a `--fix` run over tracked uncommitted edits now resolves supporting new-side references against the
  reviewed working bytes instead of HEAD (#1890), with no format change. If the implementer instead bumps a
  schema, it needs a **Breaking (0.y.z)** entry, because 13 shipped in 0.36.0.
* No AGENTS.md change is needed. The `committed_review*` read-first rule
  (AGENTS.md "Runtime invariants") applies only if `committed_review.py` is touched (e.g. a `plan_review`
  kwarg).

## Adjacent findings (out of scope for #1890; flag only)

* **Layered diff provenance [V].** The fixing diff has two layers (`queries.py:369-384`).
  `part_source_mapping` (`review_source_refs.py:318`) labels both layers' sides as `merge_base_sha`/`head_sha`:
  * the worktree layer's **old** side is really HEAD lines;
  * with option 1, supporting **new** side measures final working lines, while the committed layer's
    assigned new-side ranges are HEAD lines.

  Assigned refs are identity-only and never byte-read, so nothing is refused. This is the "assigned
  `api.py` ref at `head_sha` with range 1..2" mismatch the issue mentions; it is not fixed by either option.
* `head_line` (`committed_review_model.py:419-425`) feeds "head line" placement
  (`review_terminal.py:150-153`). For marked runs a line in the worktree layer is a working-file line [I].
* The dataset `original_task.analyzed_revision` deliberately omits the marker (`orchestrator.py:326-327`
  vs `:456-458`). The dirty state lives in `input_scope`/`dirty_tracked`.

## Verified vs inferred

* **[V] by running** (scratch repos, repo `.venv`):
  * `git_ops.diff` layering;
  * `_source_contents` returning HEAD bytes for a marked revision;
  * `canonicalize_source_selectors` refusing `6..7` (`allowed 1..5`) and digesting HEAD for `5..5`;
  * `git stash create` leaving refs/stash list alone, rewriting index stat, producing a tree equal to the
    `git diff HEAD` content for modified and staged-new files, and failing on intent-to-add;
  * a dirty submodule showing in `git diff HEAD` but not in the stash tree diff;
  * `git status` rewriting the index;
  * the PoC view accepting new-side `6..7` with the working digest, keeping old = base, and staying stable
    after a live overwrite.
* **[V] by reading:** every file:line cited above, including that byte reads happen only during
  `_step_committed_review` and that seal, result validation and lookup read no source bytes.
* **[I]:** unmerged-path failure of `stash create`; risks from extension steps or reviewer `executed_check`
  under option 2; GC (default prune age) not removing the recent dangling commit mid-run; the head-line
  placement semantics for marked runs.

## Open questions for the implementer/maintainers

1. Is writing dangling loose objects to the user's object store at review time acceptable? The fix gate
   already does it later in the same run. If not, use option 2 with a docs carve-out.
2. Should the view be bound to the diff (verify the layer, or derive the layer from the commit), given the
   dirty-submodule and intent-to-add edges? Or only snapshot adjacent to the diff?
3. When a marked run has no view (stash failed): correctable "unavailable" rejection of new-side supporting
   selectors, or refuse the run before review?
4. Should a future schema record the frozen commit, so new-side `revision` is self-verifying rather than
   `head_sha` plus marker? That is a breaking change for 0.36.0 readers, so it belongs in a separate decision
   and is not part of #1890.
