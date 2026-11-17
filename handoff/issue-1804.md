# Handoff: issue #1804 (sealed-result corruption messages)

Prepared read-only for a fresh implementation session. This file lives only on
the handoff branch `claude/affectionate-cerf-via9bx`; never commit it to the
implementation branch or PR. Read it with:

```bash
git fetch origin claude/affectionate-cerf-via9bx
git show FETCH_HEAD:handoff/issue-1804.md
```

## Source

- Issue: https://github.com/existential-birds/daydream/issues/1804 (Bug, assignee
  anderskev, no comments, no linked PRs or amendments when prepared on 2026-11-17).
- Its line numbers (`212-214`, `781`, repro commit `78b4cb86`) are stale. At
  `b96338d` the test is `test_review_revision_admission_and_sealing_contracts`
  (`tests/test_committed_review.py:220`, loop `:232-234`) and the helper is
  `_sealed_result_case` (`:890`), with `pytest.raises(..., match={...}.get(corruption, ""))`
  at `:935-939`.

## Baseline observed

- Checkout `/home/user/daydream`, branch `claude/affectionate-cerf-via9bx`,
  HEAD `b96338da7deafff29cfb41e59389a6abdbb605c2`, clean worktree (only ignored caches).
- `git ls-remote origin main` returned `b96338d...` (= HEAD). The local
  `origin/main` ref was stale (`f636a1b`), so fetch `main` before branching and
  record the actual base SHA.
- Clone is shallow (50 commits); history of the corruption rows is not available.

## Verified findings

Validator: `validate_committed_result` and `_validate_work` in
`daydream/committed_review_model.py` (`:210`). I ran the test with a scratch
pytest plugin that wrapped the validator and printed each `ValueError` (no repo
edits). Actual messages at `b96338d`:

| corruption | raised today | guard | verdict |
|---|---|---|---|
| wrong-source | `source reference differs from captured revision/diff` | `check_refs`, `:263` | intended |
| unchallenged-revision | `invalid retained claim identity/provenance` | `:303` | **wrong guard** |
| foreign-plan | `terminal target decisions differ from planned assignment` | `:279` | intended (the case recomputes `planned_work_id` on purpose to pass `:239`) |
| empty-execution | `unfinished obligations/native execution cannot be complete` | `:339` | intended |

The other five already match: self-judged and unrecorded-verdict raise
`invalid retained challenge evidence`, forged-author raises
`invalid retained claim identity/provenance`, overlap raises
`current work responsibilities overlap`, and double-credit raises
`duplicate completion credit for a target decision`.

### unchallenged-revision reaches the wrong guard

The case sets `claims[0].revision = 2` and edits `payload.trigger`. No work
admitted revision 2, so the provenance check (`:292-303`: `{"uid", "revision":
claim["revision"]}` must appear in `work[revised_by]["admitted_claims"]`) rejects
it first. That is the same guard forged-author already pins. The guard named by
the row, `publication/rejection requires current-revision challenge evidence`
(`:317-321`), has **no coverage in the whole suite**: line 321 was missing in a
full-suite coverage run, and no test matches its message.

Verified fix: append `{"uid": claim uid, "revision": 2}` to the `admitted_claims`
of the work named by `claims[0]["revised_by"]` (`work:1`, the discovery work).
The case then raises `publication/rejection requires current-revision challenge
evidence`. Follow the self-judged row's pattern and comment, e.g. "Keep
provenance valid so only the current-revision challenge guard can reject it."
Pinning `identity/provenance` instead would only duplicate forged-author, which
the testing standard counts as overlap.

## Required change (test-only)

1. In `_sealed_result_case`, index the message dict directly, `{...}[corruption]`,
   so a new corruption without a message fails with `KeyError`.
2. Add the four messages. Use distinctive substrings in the existing style, e.g.
   `captured revision/diff`, `current-revision challenge evidence`,
   `terminal target decisions differ`, `native execution cannot be complete`.
   `match` uses `re.search`, and none of these contain regex metacharacters.
3. Apply the unchallenged-revision provenance fix above.
4. The test count does not change. Add no new test; the corruption rows are the keepers.

Prove each row fails for the intended reason (test-audit: "negative controls
that pass at the wrong guard"):

- Temporarily neutralize each of the four guards (`:263`, `:279`, `:321`, `:339`)
  and confirm the test fails, then restore it. This is a local mutation check only.
- On the base, confirm the repro fails with "matching against an empty string
  will *always* pass".

## Validation

```bash
make install && make hooks
uv lock --check
uv run pytest -q -n0 tests/test_committed_review.py::test_review_revision_admission_and_sealing_contracts -W error::pytest.PytestWarning
uv run pytest -q tests/test_committed_review.py
make lint typecheck deadcode
git diff --check
make check
```

The pre-push hook verifies signatures (`%G?` must be `G`) and runs `make check`.

During preparation:

- The focused test passed without `-W` and failed with it, as the issue states.
- A full-suite run in a scratch environment without extras is **not gate
  evidence**: it had 1105 passed and an import error in `tests/test_hub.py` from
  the missing extras.
- `make check` was not run.

## Publication notes

- **Commit:** a signed Conventional Commit with explicit staging, e.g.
  `test(committed-review): pin the guard each sealed-result corruption reaches (#1804)`.
- **SemVer:** no impact (test-only).
- **CHANGELOG:** test-only precedent (#1915, #1902) adds a `### Fixed` entry under
  `[Unreleased]`. Add one contributor-facing line linking #1804, or mark the item
  N/A with a reason.
- **PR template:** `.github/PULL_REQUEST_TEMPLATE.md` has Summary, Motivation, Test
  Plan and Checklist; Changes and Additional Context are optional. Leave
  inapplicable items unchecked and append `N/A: <reason>`. Use `Fixes #1804`.
- **Daydream review:** `daydream . --backend pi --comment --non-interactive --capture-data`,
  plus `--base <verified SHA>` if the default comparison base is wrong. Leave
  `--model` unset and omit `--yes`. The Pi CLI must be available; if it is not,
  record that as a blocker.
- **Feedback:** use `fetch-pr-feedback --include-author --include-self`, then
  `receive-feedback`, then `respond-pr-feedback`.

## Out of scope (note only)

- `tests/test_committed_review.py:878-880` and `tests/test_captured_source.py:614-617`
  pass `match=None` for some cases, which also skips the message check. That is
  a candidate follow-up issue, not part of #1804.
- Other unreached guards in `committed_review_model.py` (missing coverage lines
  215, 220-221, 233, 236, 239, 242, 251, 266, 291, 323, 327) are also out of scope.

## Lanes

One file is in scope. Give a single worker ownership of
`tests/test_committed_review.py` and `CHANGELOG.md`, and have an independent
read-only reviewer check the integrated diff (code-review skill if available).
More fan-out adds no value here.
