# Issue #1890: consumers and architecture lane

Base: HEAD `1756782b` (read-only investigation). Markers: **[V]** means checked in code at the cited
line. **[I]** means inference that was not executed or reproduced.

## 0. Answer up front

- Production code reads supporting-source bytes through one path only
  **[V]**: `StageInputFactory.admit_submission` (`review_stage_inputs.py:182`) builds
  `_SubmissionSources(self.revision, self.work.repo, aliases, self.captured_source)`, and
  `_SubmissionSources._supporting` (`review_source_refs.py:117-123`) calls `_supporting_facts`, which calls
  `_source_contents` (`:70-78`).
  - The `_supporting_facts` fallbacks at `:229` and `:421` (`_submission is None`) run only from tests.
  - Every other consumer checks the `revision` *label* only:
    - `review_result.py:282`
    - `committed_review_model.py:258`
    - `committed_review_model.py:265`, with no `source_root`, so supporting bytes are never re-read.
- On a fixing run (`loop` mode), `runner._capture_review_workspace` passes `work` through without a capture
  **[V]** (`runner.py:703-710`). As a result:
  - `work.captured_source is None`.
  - `StageInputFactory.captured_source` is `None` (`review_stage_inputs.py:139`).
  - `revision` carries `tracked_worktree_edits` and no `source_id` (`deep/orchestrator.py:312-314, 456-458`).
  - `_source_contents` therefore falls through to `frozen_source(root, revision['head_sha'], path)` (`:78`),
    where `root` is the live checkout (`work.repo`).
- Report-only review gets its `CapturedSource` at `runner.py:742-753`, through `capture_source(..., local=True)`.
  The runner then replaces `work.repo` with the capture root.
- **Best seam [I]:**
  - **Producer:** `deep/orchestrator.py:307-314`, the only place the fixing diff and its marker are computed.
  - **Consumer:** `_source_contents`.
  - **Carrier:** `StageInputFactory`, through `plan_committed_review` → `plan_review` → `StageInputFactory`.
  - **Not a valid carrier:** `WorkContext.captured_source`. It enforces `repo == capture.root`
    (`artifacts/ownership.py:215-219`) and would move the fix cycle into the capture.
- **Larger root cause [V]:** the fixing diff is two concatenated diffs, `git diff <base>...HEAD` followed by
  `git diff HEAD` (`git_ops/queries.py:369-384`).
  - A file with both committed and uncommitted hunks gets two blocks.
  - Assigned ranges mix three coordinate systems: merge-base, HEAD, and working-tree.
  - Fixing only the supporting-byte read does not make every assigned range coordinate-correct. See §4.1.

## 1. Caller map

### 1.1 Inside `daydream/review_source_refs.py` [V]

| Symbol | Line | Called from (in file) | Reads bytes? |
|---|---|---|---|
| `source_revision(revision, side)` | 64-67 | `:73` (capture binding check), `:78` (read revision), `:226` (supporting ref label), `:318` (assigned old/new labels), `:363` (`'diff'` side label, which falls through to `source_id`/`head_sha`), `:416` (supporting label check) | no |
| `_source_contents(root, revision, side, path, captured_source)` | 70-78 | only `_supporting_facts` `:90` | **yes**: `captured_source.read` (`:75`) or `frozen_source(root, source_revision(...), path)` (`:78`) |
| `_supporting_facts(root, revision, side, path, captured)` | 88-97 | `_SubmissionSources._supporting` `:122`; fallbacks `:229`, `:421` (only when `_submission is None`, test-only) | yes (via above) |
| `_SubmissionSources` | 108-135 | `canonicalize` → `canonicalize_source_selectors(..., _submission=self)` `:129-131` | via `_supporting` |
| `canonicalize_source_selectors` | 157-272 | `_SubmissionSources.canonicalize` `:129` | supporting selectors only (`:218-249`), then `validate_source_refs` `:257` |
| `validate_source_refs` | 371-433 | `:257`; `committed_review_model.py:265` | supporting refs only when `source_root` is given (`:411-421`) |
| `part_source_mapping` | 311-368 | `review_stage_inputs.py:164` | no (diff-derived ranges and labels) |

### 1.2 External callers, grepping all of `daydream/` and `tests/` [V]

| Symbol | Caller | `revision` from | `root` / `captured` from |
|---|---|---|---|
| `_SubmissionSources(...)` | `review_stage_inputs.py:182` (`StageInputFactory.admit_submission`), the **only production constructor** | `self.revision` (`:140`, constructor argument) | `self.work.repo`; `self.captured_source = work.captured_source` (`:139`) |
| `_SubmissionSources` (type) | `committed_review.py:94` (import), `:253` `Review._refs`, `:269` `_claim_change`, `:559` `prepare(sources)` | as above | as above |
| `Review._refs` → `sources.canonicalize` | `committed_review.py:256`; called for target decisions `:607`, verdicts `:622`, claims `:272` (via `change` `:632`) | as above | as above |
| `admit_submission` | `committed_review.py:740` (inside `admit`, `:550`), reached from the host tools `review_admit` and `structured_output` (`:750-757`); `HostToolBinding(validate=self.check)` (`:753`) runs the full input check first | | |
| `canonicalize_source_selectors` | production: only via `_SubmissionSources`. Tests: `tests/deep_orchestrator/test_review_native_sources.py:94,122,144,165` | test dict | test repo, `captured=None` |
| `validate_source_refs` | `committed_review_model.py:265` (`validate_committed_result.check_refs`: assigned refs only, no `source_root`); test `test_review_native_sources.py:120` | sealed `analyzed_revision` | none |
| `source_revision` | `review_result.py:282` (`validate_target_decisions`, called with `analyzed_revision` at `:340`, `:456`); `committed_review_model.py:258` (`check_refs`) | stored/sealed revision | none (label equality only) |
| `part_source_mapping` | `review_stage_inputs.py:164` via `_mapped`, from `_parts` `:243`, `_segments` `:295,:314,:325` | `self.revision` | none |
| `source_options` / `source_id` | `review_stage_inputs.py:154,157`; `committed_review_model.py:404,415`; `tests/test_committed_review.py:59,131` | n/a | n/a |
| `lf_lines` | `review_stage_inputs.py:259,272,332` | n/a | n/a |
| schemas, `SourceSelectorRejected`, `SOURCE_REFS_MAX_BYTES` | `review_result.py:22,113,277`; `committed_review_model.py:18-24,73,87,182`; `committed_review.py:94,573,605`; `tests/test_host_tools.py:27,91` | n/a | n/a |

Evidence refs are consumed only in `review_source_refs.py`, `review_stage_inputs.py`, `committed_review.py`,
`committed_review_model.py` and `review_result.py` **[V]** (grep for `evidence_refs|source_refs`).
`findings.py`, the report renderers and `reviews/` never re-read source bytes from refs.

### 1.3 Tracing `revision` and `captured` back to the runner

**Entry A: CLI → runner → deep flow (the fixing path) [V]**

1. `runner.run` opens `open_workspace(...)` (`runner.py:565-575`), then `_capture_review_workspace(...)` (`:576`).
2. `_capture_review_workspace` (`runner.py:692-758`):
   - `mode = _resolve_mode(config)`. `loop` is anything that is not diagram/review/comment
     (`deep/orchestrator.py:129-138`).
   - For loop mode without `findings_out` it yields `work` unchanged (`:703-710`), after printing the
     "Fixing reviews the checkout as is …" notice when `git_ops.has_worktree_edits(work.repo)` (`:706-708`).
   - For review/comment it calls `capture_source(work.repo, destination, base_sha, head_sha, local=not committed,
     exclude=config.ignore_paths)` (`:742-743`). It then yields
     `replace(work, repo=captured.root, captured_source=captured, ...)` (`:751-753`).
   - The CLI rejects `--fix` with `--findings-out` (`commands/review.py:246-247`), so a CLI fixing run never
     takes the capture branch.
3. `_dispatch` (`runner.py:831-860`) calls `_run_loop_deep`, which calls `deep/orchestrator.run_deep` (`:224`).
4. `run_deep`:
   - `source_id = None` when the work is uncaptured (`:267`).
   - `captured_head = git_ops.head_sha(target_dir)` (`:278`).
   - `captured_base = resolve_diff_merge_base(target_dir, work.base_branch, captured_head)` (`:279`).
   - The diff is `git_ops.diff(work.repo, work.base_branch, exclude=...)` (`:313`), and
     `tracked_worktree_edits = git_ops.has_worktree_edits(work.repo)` (`:314`).
   - It writes `diff.patch` and the hunk index (`:349-354`).
   - The revision is `AnalyzedRevision(captured_head, captured_base, current_diff_sha, pr_base_sha,
     source_id=source_id, tracked_worktree_edits=...)` (`:456-458`).
5. `review_terminal.plan_committed_review(ctx, revision=..., index_path, files, started)` (`:115-137`) calls
   `committed_review.plan_review(ctx.config, ctx.work, backends, settings, revision=revision,
   diff_path=state.diff_path, ...)`.
6. `plan_review` (`committed_review.py:1150-1182`) builds `StageInputFactory(backends["discovery"], work,
   StackAssignment("committed", files), change=CanonicalChange.capture(diff_path, index_path,
   work.captured_source), revision=revision, read_only=True)` (`:1161-1165`). It then builds
   `Review(inputs, run_id, ...)` (`:1174`).
7. `StageInputFactory.__init__` (`review_stage_inputs.py:136-160`) sets `captured_source = work.captured_source`
   and `revision = revision`. Part mapping labels every assigned ref through `source_revision(revision, side)`.
8. Dispatch: `_Schedule._dispatch` (`committed_review.py:1589-1640`) calls `review.open(...)`. The assignment
   carries `"captured_root": str(repo)` (`:1604`), the live checkout on a fixing run. Host tool calls go through
   `admit` (`:550`), then `self.inputs.admit_submission` (`:740`), then `_SubmissionSources(...)` (`:182`).
9. Sealing: `ReviewProgram.seal` (`:1139-1147`) validates `inputs.captured_source` if present. `Review.seal`
   selects `RESULT_VERSIONS["worktree"]` (13) when the marker is present (`:929-930`).

**Entry B: library `run_committed_review` [V]**

- Path: `committed_review.py:1185-1430`. It always calls `capture_source(..., local=request.local)` (`:1293-1296`)
  or uses an immutable prepared source (`:1298-1305`).
- The revision is built without a marker (`:1347-1350`). Entry B is not affected by #1890.

**Which `root` the refs read from [V]**

- **In place** (default): `open_workspace` sets `repo = source` (`workspace.py:182-183`). The user's dirty
  checkout is `root`.
- **Ephemeral** (`--branch` or `--worktree`): `ownership.add_worktree(worktree_path, resolved_ref)`
  (`workspace.py:176-181`).
  - `copy_files_into_ephemeral` copies only the configured `[tool.daydream.workspace].copy` entries, or by
    default the *ignored* `.env*` files (`workspace.py:338-382, 467-483`).
  - Tracked uncommitted edits are **not** copied. The marker therefore arises in place, or in the edge case
    where a configured `workspace.copy` names a tracked file **[I]**.
- `run_snapshot.py` holds only archive and recorder identity dataclasses. It has no role in source bytes **[V]**.
- In both cases `frozen_source` reads Git objects (`ls-tree`/`show`) through `root`'s Git directory
  (`git_ops/source.py:67-98`). It never reads working files.

## 2. End-to-end dirty fixing run and where the mismatch arises

1. **Diff [V].** `queries.diff` (`git_ops/queries.py:369-384`) concatenates two diffs:
   `git diff <preferred_base>...HEAD -- . :(exclude)…` and `git diff HEAD -- . …` (working tree, tracked paths
   only).
   - Its docstring rationale ("so resume fingerprints see both") is stale: there is no resume
     (`docs/runtime.md:1096-1097`) **[I]**.
   - Untracked files never enter, which matches the journey contract "Untracked files alone leave the reviewed
     subject exactly the commits" (`tests/test_missing_test_command_acceptance.py:67-68`).
2. **Marker [V].** `has_worktree_edits` (`queries.py:620-627`) reads porcelain-tracked lines outside
   `.daydream` and runtime artifacts. It ignores `config.ignore_paths`, so the marker can be set while the
   worktree block of the diff is empty **[I]** (minor, separate).
3. **What the model sees [V].**
   - Backends run with `cwd=self.work.repo` (`flows/engine.py:80-91`).
   - `review_source` serves only frozen diff bytes and tells the model to "Use read/grep for supporting files"
     (`committed_review.py:362-366`).
   - The model therefore reads **live working bytes** and cites them, while the instructions claim supporting
     selectors resolve "against captured Git blobs" (`:938-939`) and "at the captured revision" (`:972-973`).
4. **What the host checks [V].** For a supporting selector, `canonicalize_source_selectors:226-229` calls
   `_SubmissionSources._supporting` → `_supporting_facts` → `_source_contents`, which reads
   `frozen_source(root, head_sha, path)` (`:78`). Line count and digest come from HEAD bytes. This reproduces
   the issue's `api.py` case.
5. **Assigned refs [V].** `part_source_mapping` labels new-side refs with `head_sha` (`:318`). For the
   worktree block, those ranges are working-tree coordinates (see §4.1). Assigned refs are only checked for
   correspondence, never for bytes (`validate_source_refs:389-410`, `committed_review_model.py:256-266`).
6. **Timing [V].** The review runs before the fix gate. `plan_committed_review` runs at `orchestrator.py:454`,
   before `run_flow` (`:510`), and STEPS order review → … → fix-gate (`:197-212`).
   - Supporting bytes are read only during admission callbacks.
   - Sealed validation never re-reads them.
   - Fix-cycle mutations after the review therefore cannot retroactively invalidate admitted refs, and the live
     checkout is not mutated by Daydream during review.
7. **Fix steps [V].** `fix_steps.py:201-207` takes `stable_ref = git_ops.stash_create(ctx.work.repo) or
   stable_head`. This is a dangling commit holding pre-gate tracked edits. Per `fix_state.py:77-78`, it is the
   authority for rollback and attribution. The fixing flow already relies on a dangling stash commit for the
   whole run, which is a precedent for option 1b below.

**Where report-only review gets a capture, and where fixing passes `None` [V]**

| | Report-only (`review`) | Fixing (`loop`) |
|---|---|---|
| Capture | `runner.py:742-753`, `capture_source(local=True)` | none: `runner.py:709` yields `work` unchanged |
| `work.repo` | capture root | live checkout (or a clean ephemeral worktree) |
| `revision` new-side label | `source_id` | `head_sha` (plus marker) |
| `StageInputFactory.captured_source` | `CapturedSource` (`review_stage_inputs.py:139`) | `None` |
| `_source_contents` path | `captured_source.read('new', path)` (`:75`) | `frozen_source(root, head_sha, path)` (`:78`) |

**Seams**

- **Read choke point:** `review_source_refs._source_contents` (`:70-78`) **[V]**. Today it already guards the
  "`source_id` without capture" case (`:76-77`). A mirrored guard for the marker is the natural shape.
- **Carrier:** `StageInputFactory`. It owns `revision`, `work.repo` and `captured_source`, constructs the only
  `_SubmissionSources` (`:182`), and is what `seal()` (`committed_review.py:1141`) and `_revalidate`
  (`review_stage_inputs.py:61-69`) consult.
  - Threading needs one keyword argument across `plan_committed_review` (`review_terminal.py:115-137`) →
    `plan_review` (`committed_review.py:1150-1165`) → `StageInputFactory.__init__`
    (`review_stage_inputs.py:136-140`) → `_SubmissionSources` (`review_source_refs.py:111-113`) →
    `_supporting_facts` / `_source_contents` **[I]**.
- **Producer:** `deep/orchestrator.py:307-314`, the only place the fixing diff and marker are computed. The
  frozen view must be taken here, adjacent to the diff, so both describe the same bytes **[I]**.
- **Do not** set `WorkContext.captured_source` for fixing **[V]**. Doing so trips these invariants:
  - `_validate_captured_workspace` requires `repo == capture.root` under the private operational root
    (`artifacts/ownership.py:215-219`, called from `:194-196` and `bind_code` `:470-472`,
    `artifact_visibility.py:1177`).
  - It also flips these consumers:
    - `runner.py:599`
    - `flows/engine.py:92-96`
    - `deep/orchestrator.py:267-308, 386`
    - `dataset_capture.py:43-50, 164, 199-201`
    - `deep/review_steps.py:65`
    - `deep/review_terminal.py:88`

## 3. Fix options from an architecture view

### Option 1a: full `CapturedSource` via `capture_source(local=True)`

- Population mismatch **[V]**: the capture includes nonignored untracked files (`captured_source.py:234`). The
  fixing diff excludes them, as the journey test asserts (`tests/test_missing_test_command_acceptance.py:67-68`).
- Identity conflict **[V]**:
  - The capture sets `source_id` (`captured_source.py:697`).
  - `_source_contents` requires `captured.revision_for('new') == source_revision(revision, 'new')` (`:73`), so
    the revision would need `source_id`.
  - `_revision_version` returns `None` when both `source_id` and the marker are present
    (`review_result.py:153-157`).
  - `validate_committed_result` ties `source_id` to schema 12 and the marker to schema 13
    (`committed_review_model.py:224-226`).
  - So this route forces a breaking schema redesign.
- Cost **[V]**:
  - The capture copies the whole object store and materializes the population (`captured_source.py:253-268,
    640-668`).
  - Every admission runs `captured.validate()` (`review_stage_inputs.py:61-66`).
  - The docs already describe this cost for report-only review (`docs/runtime.md:517-542`).
  - It also adds new refusals: skip-worktree paths and Git environment overrides (`captured_source.py:230-233,
    607-609`).
- It cannot ride on `WorkContext` (see §2). It would need separate plumbing anyway.
- `docs/runtime.md:462` says "Fixing-flow migration is outside this contract". This option is that migration.

### Option 1b (recommended, [I]): freeze the reviewed tracked working bytes as a Git object at diff time

- Take a Git-object snapshot of tracked working state at `orchestrator.py:312-314`, for example with the
  existing `git_ops.stash_create` (`queries.py:977-986`) or an equivalent tree writer.
- For a marked revision's new side, read `frozen_source(root, <snapshot>, path)`. The same 8 MiB bound,
  regular-blob check and absence typing apply (`git_ops/source.py:67-98`).
- Advantages:
  - Frozen across the review, which matches the AGENTS.md invariant "validate … against frozen
    side/range/fragment identities".
  - Tracked-only, which matches the fixing diff and the journey contract.
  - Cheap: no object copy, no per-admission full validation.
  - Already used by the fix gate (`fix_steps.py:203`).
  - Blobs are clean-filtered and EOL-normalized like `git diff`'s view **[I]**, so digests and line counts
    agree with the diff in filter/LFS checkouts. Option 2 would not.
- Risks to handle **[I]**:
  - `git stash create` can fail on unmerged index entries, and the fix gate already fails on those. Decide
    whether the review degrades to "supporting source unavailable" or fails earlier.
  - The race between the two `git diff` calls and the snapshot. You can bind them, for example by producing the
    worktree block from the snapshot or by rechecking that `git diff --quiet <snapshot>` still holds.
  - Dangling objects survive the default `gc.pruneExpire`, the same exposure `stable_ref` already has.
  - Submodule `-dirty` lines and intent-to-add entries need a check.
- Carry the snapshot host-only through `StageInputFactory` (§2 seams), unless the label decision below puts it in
  the revision.

### Option 2: read new-side refs from the live working tree for a marked revision

- This is the smallest diff **[V]**: `root` is already the live checkout, so only `_source_contents` plus a
  bounded, confined, no-follow regular-file reader change. Reusable primitives:
  - `immutable._parent`
  - `captured_source._regular_leaf`
  - the `immutable._MAX_SOURCE_BYTES` and 8 MiB conventions
- Downsides **[I]**:
  - Not frozen: user edits during the review change digests between admissions. The `_SubmissionSources` cache
    lives for one callback only (`:117-123`).
  - Widens eligibility to ignored and untracked files, unless you restrict reads to index-tracked paths. That
    conflicts with the runtime-invariant guidance on tracked and nonignored file sets.
  - Raw working bytes can differ from `git diff`'s clean-filtered view (CRLF/eol attributes, LFS).
  - Contradicts `docs/runtime.md:802-804` ("pinned regular Git blobs").

### Label decision, which the issue asks to make deliberately [V]/[I]

- **Keep the ref `revision` label as `head_sha` for marked new sides** and change only which bytes are measured.
  - No schema change: `ANALYZED_REVISION_SCHEMA` is strict (`review_result.py:122-127`), and
    `check_refs` / `validate_target_decisions` keep comparing against `source_revision`.
  - The marker already declares that `head_sha` is not the whole subject. The cost is that `artifact_sha256`
    no longer matches `git show head_sha:path` for edited files.
  - Document this in `docs/runtime.md#captured-local-source`.
- **Label a new identity**, for example the snapshot OID in `analyzed_revision` or a new `source_revision` value
  for marked new sides.
  - Changes strict schemas (`review_result.py:122-127`, terminal/coverage; `committed_review_model.py:173-207`).
  - Changes `planned_work_id` input (`committed_review_model.py:145`) and dataset readers (`dataset.py:85-93`).
  - Needs `RESULT_VERSIONS` 14 and terminal version 4. This is restart-only and breaking.

## 4. Other places that read new-side bytes or lines by `head_sha` for the reviewed subject

| # | Site | What it does | Same mismatch? | Scope |
|---|---|---|---|---|
| 4.1 | `queries.diff` `:369-384` → `StageInputFactory._parts` `:237-244` (joins a path's blocks) → `part_source_mapping` `:341-354`; `CanonicalChange._block_paths` `:72-80` (old side from the first block, new side from the last) | Assigned ranges per hunk; one label per side per part | **Yes [V by code reading]**. In the worktree block, old ranges are HEAD coordinates labeled `merge_base_sha`, and new ranges are working coordinates labeled `head_sha`. In the commit block, new ranges are HEAD coordinates. When one file has both kinds of hunk, uncommitted lines above a committed hunk shift it, so the committed hunk's offered range no longer matches what the model reads. | **Adjacent, separate unless the fix also derives a net `merge_base → snapshot` diff [I].** A net diff aligns every coordinate, as local capture does (`docs/runtime.md:469-472`), but changes fixing diff bytes, `diff_key` and hunks. Do not change shared `git_ops.diff`, which `runner.py:919` and `improve/recon.py:264` also use. Recommend a follow-up issue. |
| 4.2 | `hunk_index.parse_hunks` `:73-150` (written at `orchestrator.py:354`) | Merges both blocks' hunks under one path | Mixed new-side coordinates (from 4.1) | Separate (same root as 4.1) |
| 4.3 | `committed_review_model.head_line` `:419-425` / `reference_covers` `:408-416` → `review_terminal.publish_items` `:153-155` (`"line": head_line(...)`) → fix items | Placement for the fix cycle | Lines are "head" in name, but worktree-block lines are working lines and commit-block lines are HEAD lines. Claim-line admission (`committed_review.py:277-278`) uses the same ranges. | Separate (consequence of 4.1). Fixers edit the working tree, so working coordinates are what they need. |
| 4.4 | `source_revision(revision, 'diff')` `:363` | Label for binary or header-only metadata fragments | Label only; the bytes come from the diff | No change needed |
| 4.5 | `pr_review.resolve_line` `:419-431` and placement `:618-640` (`git show pr.head_sha:path`); loop mode reaches it through `_step_post_review` (`deep/merge_steps.py:121-133`, `post=False`, which offers to post) | PR comment placement at the remote PR head | Uncommitted lines cannot exist on a PR head | Separate (publication of working-tree findings) |
| 4.6 | `reviews/diagrams.py:176-200` (`_HeadEvidence.read` at `head_sha`), called from `validate_diagram_payload` at post time (`pr_review.py:1169-1175`, `diagrams.py:430`); in-flow `diagram_steps.py:369` uses `ls_tree_files(repo, "HEAD")` | Diagram grounding and publication | Possible for staged-new files **[I]** | Separate |
| 4.7 | `deep/fix_state.py:170-174`, `phases/testing.py:431,586` | Fix/test baselines at `stable_ref` (already includes tracked edits) | No: they use the stash snapshot | Out of scope |
| 4.8 | `dataset_capture.retain_original_task` (`orchestrator.py:320-329`; `dataset_capture.py:43-50`) | Records `analyzed_revision` **without** the marker and represents dirtiness as `dirty_tracked` / `input_scope` (`dataset.py:85-93`) | Not a byte read | Out of scope (worth noting: the run record does not carry the marker) |
| 4.9 | `training/harvest.py:784-788` | Corpus `git show sha:path` | Not the live review | Out of scope |

## 5. Docs, contracts and SemVer

Owners to update (one authoritative home, others link):

- `docs/runtime.md`, section `#captured-local-source`:
  - `:462-467` is the fixing paragraph that says "Fixing-flow migration is outside this contract". State here
    which bytes marked new-side supporting refs measure.
  - `:550-556` covers the `head_sha` provenance and `source_id` new-side identity rule.
  - `:563-566` covers versions. Change it only if the label or schema changes.
- `docs/runtime.md`:
  - `:802-809`, in `#pi-host-tool-lifetime-and-limits`: "pinned regular Git blobs". Option 2 would contradict
    it; option 1b keeps it true.
  - `:12-13`, in `#terminal-review-findings-contract`: terminal version 3. Changes only with a version bump.
- Reviewer instructions: `committed_review.py:938-939, 972-973` ("captured Git blobs", "at the captured
  revision"). These stay accurate under option 1b and become inaccurate under option 2.
- `docs/extensions.md:155-165`: the generic selector contract. No change is expected.
- `README.md:49-51` describes only report-only capture and PR comments. No change is expected.
- `CHANGELOG.md`:
  - `[Unreleased]` (`:8`) has `### Fixed` (`:62`); add the #1890 entry there.
  - The #1828 entry is `:282-290`, in the released `[0.36.0]`.

SemVer (policy at `CONTRIBUTING.md:150-168`):

- **Keep labels, fix bytes (1b or 2):** no schema or version change. This is a compatible bug fix under
  `### Fixed`.
  - Behavior change: for schema-13 results, supporting refs to edited files now carry working-byte digests and
    working line bounds, and selectors that were wrongly refused are now admitted.
  - The PR should state that consumers verifying `artifact_sha256` against `head_sha` blobs for marked results
    must not do so.
- **Relabel or add an identity:** breaking for 0.y.z (strict `ANALYZED_REVISION_SCHEMA`, `RESULT_VERSIONS`
  14, terminal 4, restart-only). Needs a `**Breaking (0.y.z):**` entry under `### Changed`, plus migration
  notes.
- **Option 1a:** also breaking, because of the marker and `source_id` exclusivity and the untracked population.

## 6. Proposed disjoint lanes (option 1b, keep labels)

The code change is small. Lanes 1 and 2 can be one person. Each shared file has exactly one owner.

| Lane | Owns | Work |
|---|---|---|
| **1. Review-source chain (owner of all shared review files)** | `daydream/review_source_refs.py`, `daydream/review_stage_inputs.py`, `daydream/committed_review.py` (only the `plan_review` signature and passing it on), `daydream/deep/review_terminal.py` (only the `plan_committed_review` signature), `daydream/deep/orchestrator.py` (`:307-314`, `:454-460`) | Create the snapshot next to the fixing diff. Thread it as one keyword to `StageInputFactory` → `_SubmissionSources`. In `_source_contents`, read the marked new side from the snapshot. Add a guard that a marked revision without a snapshot raises `GitError(...restart review)`, mirroring `:76-77`. Leave `source_revision` labels unchanged. |
| **2. Git primitive** | `daydream/git_ops/queries.py` and/or `daydream/git_ops/source.py`, `daydream/git_ops/__init__.py` | Reuse `stash_create` or add a bound snapshot helper. If needed, add a variant that produces the worktree block from the snapshot so the diff and the bytes cannot diverge. Do **not** change `queries.diff` semantics (shared with `runner.py:919` and `improve/recon.py:264`). Agree the interface with lane 1 first. |
| **3. Tests** | `tests/harness/stub_backend.py` (`_scripted_review` `:106-154`: an optional supporting selector per claim), `tests/test_missing_test_command_acceptance.py` (`:59-121`) | Extend the dirty journey as the issue prescribes. Make the dirty `base.txt` edit add lines and cite a supporting `{path: base.txt, side: new}` range that only the working file has. Assert from `.daydream/deep/committed-review.json` (written at `review_terminal.py:226`) that the ref is admitted with `artifact_sha256 == sha256(working bytes)`. This fails today because the range is refused or the digest equals HEAD's. Prefer extending this one journey over a new unit test, following the AGENTS.md testing standard. `tests/deep_orchestrator/test_review_native_sources.py:60-175` is the existing unit owner if a unit-level row is needed. |
| **4. Docs and changelog** | `docs/runtime.md` (`#captured-local-source` `:455-467` and `:550-556`; check `:802-809`), `CHANGELOG.md` `[Unreleased]` `### Fixed` | One authoritative sentence on what marked new-side supporting refs measure and that the label stays `head_sha`. Link elsewhere. |
| Follow-up (not #1890) | `git_ops/queries.py` diff for fixing, hunk index, `head_line` placement | File a separate issue for 4.1 to 4.3 (concatenated-diff coordinate mixing), unless lane 2 deliberately adopts a net `merge_base → snapshot` diff. |

Under option 2 instead: lane 1 shrinks to `review_source_refs.py` plus a working-tree reader in
`git_ops/source.py`, with no plumbing. Lane 4 must then amend `docs/runtime.md:802-804` and the reviewer
instructions at `committed_review.py:938-939, 972-973`.

## 7. Open points and unverified items

- **[I]** Whether `git stash create` handles intent-to-add entries and dirty submodules without error, and
  whether the tree-to-tree diff (`git diff HEAD <snapshot>`) is byte-identical to the working-tree diff for
  `-dirty` submodule lines. I ran no experiments, under the read-only constraint.
- **[I]** Whether maintainers want commit-block new ranges rebased to working coordinates (§4.1) inside #1890.
  The issue text covers only supporting refs and selector checks.
- **[V]** The marker can be set when only paths in `ignore_paths` are dirty: `has_worktree_edits` ignores
  exclusions while the diff applies them. Minor, separate.
