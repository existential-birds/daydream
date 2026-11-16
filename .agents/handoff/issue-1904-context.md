# Issue #1904 implementation context (prepared 2026-11-16)

This is the detailed handoff for the `/goal` prompt for
https://github.com/existential-birds/daydream/issues/1904. It is force-added under
the gitignored `.agents/` tree in one handoff commit on branch
`claude/sharp-faraday-c3i6fe`; that commit's parent is the base `db0368e`. Read it
with `git fetch origin claude/sharp-faraday-c3i6fe` and then
`git show FETCH_HEAD:.agents/handoff/issue-1904-context.md`. Build the fix branch from
the base, not from the handoff branch, so this file never enters the PR. Line numbers
are from base `db0368e`; re-read the code before editing. The preparation was read-only.
The checks listed below were **identified, not run**, except where marked *ran*.

## 1. Source, scope, acceptance

Issue title: *fix(artifacts): manifest walk order disagrees with the manifest sort
check for prefix-named siblings*. The issue has no comments and no amendments.

- `daydream/artifacts/filesystem.py` `_walk` (:281) and `manifest_tree` (:325) emit
  entries depth-first. Each directory's children are sorted by `os.fsencode(child.name)`
  (:301), and the top-level names by `os.fsencode` (:345).
- `_parse_manifest` (:363) then requires
  `[e.path ...] == sorted(paths, key=os.fsencode)`, which is flat byte order (:383).
- The two orders differ when a directory `D/` that has at least one child sits beside
  a sibling named `D` + a byte below `/` (0x2f). Example: `evidence/` beside
  `evidence.otlp.jsonl`.
- *Ran:* a scratch script reproduced the bug at HEAD `db0368e`.
  - The walk gave `['evidence','evidence/a','evidence.otlp.jsonl']`.
  - The flat sort wants `['evidence','evidence.otlp.jsonl','evidence/a']`.
  - `_parse_manifest` raised `ArtifactVisibilityError: artifact manifest entries are not sorted`.
  - Sorting by the component-tuple key `tuple(os.fsencode(p) for p in path.split("/"))`
    reproduces the walk order exactly.

Acceptance (current, from the issue):

1. Walking and validating use one order. Check every consumer of manifest order:
   persisted frozen manifests, `transactions`, `publication`, baselines. Persisted
   manifests either stay readable or are refused consistently, under AGENTS.md's
   no-migration policy.
2. One regression test at the real filesystem boundary freezes and publishes a tree
   that has `name/` beside `name.ext`. It must fail on current code.
3. `make check` passes, and CHANGELOG `[Unreleased]` → `### Fixed` has an entry.

Non-goals:
- Do not rename `daydream/trajectory/layout.py` `SIBLINGS_DIRNAME = "forks"`. It came
  from #1717 / PR #1903, and no code comment ties it to this bug.
- No manifest schema or `_SCHEMA_VERSION` change.
- No change to the on-disk format of manifests.

## 2. Correction to the issue text (verified by reading `finalize_frozen`)

The issue says the failure happens in `finalize_frozen` → `transactions._retire_transaction`.
That is true only when the clashing names reach a **destination ledger record**.

- **Public-only tree** (pair under `.daydream/...`):
  - `finalize_frozen` (artifact_visibility.py:791) succeeds.
  - It writes `publish-manifest.json` (:837) and `canonical-manifest.json` (:932) in
    walk order.
  - The **next session open** fails: `_open_layout` → `publication._validated_canonical_entries`
    (publication.py:41) raises "not sorted". The canonical rereads at
    artifact_visibility.py:1248/1261 would fail the same way.
- **Dump/finalization-merge destination** (`OutputLabel.DUMP_DIRECTORY`) whose late
  tree holds the pair:
  - `ledger._write_destination_records` (av:890) writes
    `destination-NNNN-published-manifest.json` in walk order.
  - `PUBLISH_VERIFIED` is journaled (av:936).
  - Then `_retire_transaction` (av:946 → transactions.py:105 → ledger.py:136
    `_parse_manifest`) raises.
  - Recovery on session close or next open (`_recover_transaction`, transactions.py:373)
    raises again, so the workspace is stuck.
- **Baseline that already holds the pair:** the failure is at the detach retirement
  (av:938) via ledger.py:134, and in `ledger._persist_live_destination_record` (:171).

## 3. Decision: Option A — validate the walk's order (do not change `manifest_tree`)

Change `_parse_manifest`'s order check to the component-tuple key. Define that key once
in `filesystem.py`, for example `_manifest_order_key(path: str) -> tuple[bytes, ...]`.

Preferred: use the same helper for the `_walk` child sort and the `manifest_tree`
top-level sort, so the producer and the validator share one definition. For
single-component names it is equivalent to sorting by `os.fsencode(name)`. Add a
one-line comment or docstring stating the rule: "manifest order is depth-first
component order, the order the walk writes".

Why A rather than B (B = sort `manifest_tree` output into flat byte order):

- **Edit size.** A changes one validator. B changes the in-memory order of every
  snapshot, ledger and canonical manifest at once.
- **Comparisons stay consistent.** Every order-sensitive comparison (§4) compares
  `manifest_tree` output, or an order-preserving filtered subset of it, against a
  manifest that was written from such output. Nothing re-sorts or merges entries.
- **Persisted files.** Every manifest any version wrote is in walk order.
  - Under A, files that were readable before stay readable, because their walk order
    already equals flat order.
  - Under A, files with the clash, which today block publication or recovery (stuck
    `PUBLISH_VERIFIED` transactions, unopenable canonical state), become readable.
    They match the unchanged walk, so recovery completes. Nothing is rewritten, so
    this is a validator fix, not a migration.
  - Under B, those states stay refused permanently.
- **Order-dependent consumers** need only parent-before-child, which both orders give:
  - `_copy_tree` mkdir without `parents`, plus its reversed chmod pass (filesystem.py:406-427)
  - `publication._restore_manifest_noclobber` (:202-218)
  - the directory merge loops (publication.py:470-483)
  - Nothing needs contiguous subtrees: there is no bisect or slicing.
  - Removal uses `_deepest_first_directories`, which sorts by depth.
- **Risk:** the validator becomes stricter in one respect. It now requires walk order.
  Walk order equals flat order for every manifest the old check accepted, so nothing
  that passed before is refused now.

If the implementer finds evidence against A, choose B, record why in the PR, and
make every reader consistent.

## 4. Order contract map (from read-only agents; verify while editing)

- **Only order validator:** `filesystem.py:383`. Transfer intents
  (`transfer.py:140-169`) check `index == position` only. `external-entries.json`
  (external.py:107) is not a manifest.
- **Order-sensitive `==`/`!=` comparisons:**
  - filesystem.py: 426
  - publication.py: 42, 129, 145, 400, 558
  - transactions.py: 303, 385, 390, 392, 398, 406-407, 413, 417, 423, 429, 431, 470
  - transfer.py: 324
  - artifact_visibility.py: 804, 872, 1222, 1249, 1251, 1262
  - archive/bundle.py: 36
- **Order-independent helpers:** `_manifest_is_subset`, `_entries_belong_to`,
  `_entry_at`, `_deepest_first_directories`, the dict lookups in
  `_merge_directory_from_tree` and `_freeze_tree`.

Persisted manifests (all written by `_atomic_json(_manifest_payload(...))`). av =
artifact_visibility.py, tx = transactions.py, pub = publication.py:

| File | Written | Reread |
|---|---|---|
| `state_root/canonical-manifest.json` | av:932, 1112, 1267; tx:395, 425 | pub:41, tx:302, tx:404, av:1248, av:1261 |
| `transaction/manifest.json` | av:1256 | tx:382 |
| `transaction/publish-manifest.json` | av:837 | tx:380 |
| `destination-NNNN-{baseline,published}-manifest.json` | ledger.py:24/45 | ledger.py:134/136 |
| `frozen-manifest.json` | av:729 | never reread |

Unrelated `fsencode` sorts (Git state and captured source), which must be left alone:
`daydream/phases/publish.py:70,126` and `daydream/git_ops/captured_source.py`.
Git tree order is a different contract.

## 5. Regression test plan (one test, count-neutral)

Extend `tests/test_artifact_visibility.py::test_artifact_session_preserves_resume_mtimes_across_publication_and_reopen`
(:510). It already drives a real Git repo and real sessions:

1. The `mtime-first` session registers public plus dump destinations.
2. It writes the run document `RUN_DOCUMENT_NAME` at
   `session.daydream_dir/"runs"/session_id/` (:634), freezes (:635), writes the late
   dump files under `session.finalization_merge_path(dump_route, snapshot=frozen)`
   (:639-641), and publishes with `_publish` (:642).
3. It reopens the `mtime-second` session (:659) and compares `_manifest(...)`.

Add both cases, about two lines each, with asserts on the published bytes:

- **(a) Public, using the issue's real names.** Before `session.freeze`, create
  `session.daydream_dir/"runs"/session_id/"evidence"/<child>` beside the run
  document `evidence.otlp.jsonl`. Assert the child's bytes under `source/.daydream/...`
  after publish. On current code this fails at the reopen (pub:41). `freeze` validates
  only the listed documents' paths and ancestry, so an extra directory in the run dir
  should be accepted; confirm by running the test.
- **(b) Dump destination, matching the issue's stack.** Next to `late/"added.txt"`,
  add `late/"name"/<child>` and `late/"name.ext"`, and assert them under `dump`. On
  current code this fails inside `_publish` (:642) via `_retire_transaction` →
  ledger.py:136.
- Do **not** seed the pair in the public tree before the first session opens (around
  :584). That fails at the open-time rebaseline (av:1261), not at freeze or publish.
- **Prove it fails on pre-fix code.** Commit or stash nothing extra: apply the test
  edit first, run it, and record the "not sorted" failure. Then apply the fix and
  record the pass. Put both runs in the PR Test Plan.
- Leave the test oracle `_walk_entry`/`_manifest` (tests:157-181) as is. It already
  walks depth-first and is compared only with itself.
- Existing refusal tests must stay green:
  - `test_artifact_session_rejects_closed_recovery_manifest_untouched` (:980)
  - `_nonstatic_subtree_authorization_rejects_corrupt_canonical_proof_case` (:1923)
  - No test checks "not sorted" directly. Do not add a unit test for `_parse_manifest`
    unless review shows the extended test misses a branch. The ordering branch is
    covered by (a) and (b).
- Run the test-audit skill (`.agents/skills/test-audit/SKILL.md`) on the changed test
  file and address its findings.

## 6. Commands (from Makefile, CONTRIBUTING.md and pyproject.toml)

- Setup: `make install` (`uv sync --all-extras`) and `make hooks` (symlinks the
  pre-commit and pre-push hooks; they were **not installed** in the prep checkout).
- Focused (*ran by a prep agent at base*: 35 passed, about 8.5 s wall):
  `uv run pytest tests/test_artifact_visibility.py -n logical`
- Single test: `uv run pytest tests/test_artifact_visibility.py -k preserves_resume_mtimes`
- Neighbours: `uv run pytest tests/test_archive.py tests/test_archive_integration.py`
- Gate: `make check`, which runs lockcheck, install, static (lint, deadcode,
  typecheck, package-check, actionlint, check-naming, test-scripts), test and
  coverage-report. Coverage ratchet: `docs/coverage.md`; do not lower `fail_under`.
- The simplification sweep is required only if the PR changes more than 5 files or
  adds more than 500 lines. This fix should touch 3 files: `filesystem.py`,
  `tests/test_artifact_visibility.py` and `CHANGELOG.md`.

## 7. CHANGELOG draft

Add under `## [Unreleased]` → existing `### Fixed` (CHANGELOG.md:62), using the house
style: a wrapped user-facing sentence plus an issue link. Draft:

> - Finalized artifact trees that hold a directory beside a sibling file whose name
>   extends it with a byte that sorts before `/` (for example `evidence/` beside
>   `evidence.otlp.jsonl`) publish and reopen again. Manifest validation now checks the
>   same depth-first order the tree walk writes, so these manifests are no longer refused
>   as unsorted. Previously, that refusal could leave a publication stuck in recovery
>   ([#1904](https://github.com/existential-birds/daydream/issues/1904)). Persisted
>   manifests keep their format.

SemVer: a compatible bug fix (patch). There is no public contract or schema change.
Docs: no file in docs/, AGENTS.md or README states manifest order (checked by grep),
so no doc update is expected beyond the code comment.

## 8. Baseline observed during preparation

- Repo `/home/user/daydream`, remote `origin` = https://github.com/existential-birds/daydream.
- Branch `claude/sharp-faraday-c3i6fe` at `db0368e176d8133d5a1a574eb1457b7ed1b3db05`
  ("fix(review): resolve dirty fixing-run supporting refs ... (#1908)"). This equals
  `origin/claude/sharp-faraday-c3i6fe`.
- The worktree was clean when preparation started. This handoff file was then
  committed on top of `db0368e`; that commit is not part of the base.
- The local `main` and `origin/main` refs are `f636a1b` (2026-11-12). They are stale:
  an ancestor of HEAD, 39 commits behind. They were observed locally and **not
  fetched**. `db0368e` is probably main's tip at session start, but that is unverified.
- **Requested base:** none given, so use the captured HEAD `db0368e`.
- Signing config: `commit.gpgsign=true`, `gpg.format=ssh`,
  `user.signingkey=/home/claude/.ssh/commit_signing_key.pub`. Only the `.pub` file is
  present, so whether signing works is unverified until the first commit.
- Tools seen: `daydream` (/root/.local/bin), `pi` (/opt/node22/bin), `gh`
  (/usr/local/bin). In Claude cloud sessions, GitHub access may be limited to the
  GitHub MCP tools; use them where `gh` is refused.

## 9. Publication and review loop (from prepare-implement / CONTRIBUTING.md)

- **Commits and push.**
  - Stage paths explicitly (`git add <path>`). Make signed Conventional Commits, e.g.
    `fix(artifacts): validate manifest walk order for prefix-named siblings`.
  - Push with hooks enabled, never `--no-verify`. The pre-push hook verifies
    signatures and runs `make check`.
  - If a hook fails, fix the cause on the branch and retry, or report the blocker.
- **Open the PR.**
  - Use `.github/PULL_REQUEST_TEMPLATE.md` through `.agents/skills/pr/SKILL.md`.
    Fill Summary, Motivation, Test Plan (record commands and results, including the
    pre-fix failure) and Checklist (evidence, or `N/A: reason`).
  - Use `Fixes #1904`.
  - Open it ready for review, targeting `main`.
- **Daydream review.**
  - Verify the PR's base/head repos, refs and SHAs.
  - From the checkout at the PR head, run:
    `daydream . --backend pi --comment --non-interactive --capture-data`.
    Do not pass `--model` or `--yes`.
  - If the default comparison base does not resolve to the verified PR base SHA,
    append `--base <verified ref/SHA>`.
  - Record the command, the base/head SHAs, the resolved merge-base and the result.
- **Feedback.**
  - Run `.agents/skills/fetch-pr-feedback` with `--include-author --include-self`,
    then `receive-feedback`, then `respond-pr-feedback`.
  - Cover every reviewer and bot, including `greptile-apps`: inline threads,
    reviews and general comments.
  - Verify each finding against the current code. Fix confirmed issues, rerun
    affected checks, commit and push with hooks, then reply and resolve eligible
    threads.
  - Rerun Daydream after changes so it covers the latest HEAD.
  - Wait for known queued reviewer or bot runs and for required CI.
  - Review failures and pending checks are blockers. Absence of feedback is not
    proof of success.
- **Stop before merging.**

## 10. Unresolved risks and assumptions

- Placement (a) assumes `freeze` accepts an extra directory in the run dir. Confirm it
  by running the test; if it is refused, place the public pair elsewhere under
  `.daydream/` (e.g. `deep/`), keeping the real names if possible.
- The base may have moved on `origin/main`. Fetch, record the actual base SHA, and
  rebase only if needed, keeping history linear.
- Commit signing in the executing environment is unverified.
