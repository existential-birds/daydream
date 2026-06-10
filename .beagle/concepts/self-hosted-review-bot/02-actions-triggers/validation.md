# Task 11 — Sandbox end-to-end acceptance (live validation)

> Run 2026-06-10 against `existential-birds/daydream-bot-sandbox` (public) by the operator
> account `anderskev` (org owner). Daydream under test: branch `feat/actions-triggers`,
> pinned at install time (see Step 1). Every URL below was observed live via `gh`;
> nothing is inferred or reconstructed.

## Outcome summary

| Step | What | Result |
|---|---|---|
| 1 | Install templates + secrets/vars | DONE — commit `0383733` on sandbox `main`; `ANTHROPIC_API_KEY` later replaced with a working key (see PR #6 re-run below) |
| 2 | Auto review on PR opened | **PASS in full** — initial pass (PR #4) proved the trigger chain but had an invalid key; re-run on PR #6 with a working key produced 4 real inline findings authored by `daydream-review[bot]`, every raw body carrying a hidden `<!-- daydream-finding: <64-hex> -->` marker. Both seeded defects found. (PR #4 pass also exposed and fixed a real daydream bug: errored agent run masqueraded as a clean "no issues" review, fix `58da2c0`) |
| 3 | No re-review on push | PASS — no run for pushed head `56ecfb0` (PR #4); re-confirmed on PR #6: pushing `c3c8450` triggered no auto review |
| 4 | `@daydream-review review` command chain | **PASS in full** — 👀 reaction, **App-token mint with `permission-actions: write` SUCCEEDED**, dispatch ran as `daydream-review[bot]`, and the dispatched run's completion **DID fire `workflow_run`** (the spike's NOT-RUN item, verified live). Reconcile semantics verified on PR #6: fixed finding's comment minimized as OUTDATED, new finding posted exactly once, no fingerprint duplicated (4 stale minimized, 0 failed). Exact-match dedup ("matched → untouched") did not fire live — the second agent run re-worded every message (Assumption 5 drift, documented limitation); covered by the Task 7 fake-gh real-path tests |
| 5 | Negative cases | PASS — bot-authored comments gate to skipped (twice: coderabbitai[bot] and daydream-review[bot]); mention without `review` matched=false, no reaction, no dispatch. Non-collaborator comment NOT TESTED (no second account available) |
| — | Failure surfacing (Task 9 should-have) | PASS — `daydream-review[bot]` posted the failure comment on PR #5 for the `pull_request` shape; dispatch shape logs "no PR resolvable" and skips (documented limitation) |

## Step 1 — Install (fresh-eyes pass on `templates/workflows/README.md`)

- Copied the three templates from `templates/workflows/` to the sandbox's
  `.github/workflows/` on `main`, deleting the superseded spike probe workflows
  (`probe-*.yml`) in the same commit:
  https://github.com/existential-birds/daydream-bot-sandbox/commit/0383733
- **Install-time config:** both `daydream-review.yml` and `daydream-post.yml` were edited
  to pin the daydream install to the branch under test:
  `uv tool install git+https://github.com/existential-birds/daydream@feat/actions-triggers`
  (daydream repo branch head at first validation pass: `58da2c0`; the PR #6 re-run
  installed `d8eeafc` — observed in the run 27281321150 install log).
- Secrets set via `gh secret set` (values never echoed): `DAYDREAM_APP_ID` (= 4014446),
  `DAYDREAM_APP_PRIVATE_KEY` (from PEM file via stdin redirect), `ANTHROPIC_API_KEY`
  (from the operator's shell env). Variable via `gh variable set`:
  `DAYDREAM_BOT_HANDLE=daydream-review`.
- The sandbox was found **private** at validation start (spike notes say it was left
  public); flipped back to public before installing.

### README friction notes (fresh eyes)

1. **No version-pinning guidance.** The `uv tool install git+…/daydream` line inside two
   of the three templates installs the repo's default branch implicitly. An operator who
   wants reproducible installs (or to test a branch) must know to edit two files. The
   README's Install section should name that line as install-time config and recommend
   pinning a tag/SHA.
2. **Reusing an existing App needs re-approval.** Install step 1 says "or reuse the one
   from your daydream App setup" — but adding `actions: write` to an already-installed App
   requires the installation owner to approve the permission change before the command
   workflow can mint its dispatch token (exactly what blocked the spike's probe). Worth a
   sentence.
3. Otherwise the install steps were sufficient — three files, three secrets, one variable,
   in order, no missing steps.

## Step 2 — Auto review on opened PR

- PR #4 (seeded TOCTOU in `src/cache.py`, off-by-one in `src/pager.py`):
  https://github.com/existential-birds/daydream-bot-sandbox/pull/4
- `pull_request (opened)` fired **Daydream Review**:
  https://github.com/existential-birds/daydream-bot-sandbox/actions/runs/27278872827
  — same-repo gate passed, PR head checked out, daydream installed from the pinned
  branch, `daydream --review --non-interactive --pr-number 4 --findings-out … --base origin/main .`
  ran, artifact `daydream-findings` uploaded.
- Its completion fired **Daydream Post**:
  https://github.com/existential-birds/daydream-bot-sandbox/actions/runs/27278904262
  — artifact downloaded, target derived from `workflow_run.head_sha` +
  `pull_requests[0].number` (PR 4, head `1dfbe13…`), App token minted with exactly the
  least-privilege trio, and `daydream post-findings` validated the artifact and exited 0:
  log line `No new findings to post (0 already on PR #4)`.
- **Initial pass (PR #4): inline comments NOT testable.** The findings artifact was empty
  because every agent invocation failed with `Invalid API key`. The operator-provided
  `ANTHROPIC_API_KEY` was invalid upstream — verified twice from the workstation,
  independent of CI (`POST /v1/messages` and `GET /v1/models` both return
  `401 authentication_error: invalid x-api-key`).
- **Re-run with a working key (PR #6) — PASS.** The sandbox secret `ANTHROPIC_API_KEY`
  was replaced with a working key (no key material recorded anywhere). PR #6
  (seeded off-by-one in `src/chunker.py`, TOCTOU in `src/config_loader.py`):
  https://github.com/existential-birds/daydream-bot-sandbox/pull/6
  - `pull_request (opened)` fired **Daydream Review** (success):
    https://github.com/existential-birds/daydream-bot-sandbox/actions/runs/27281321150
    (head `94e91ca`)
  - Its completion fired **Daydream Post** (success):
    https://github.com/existential-birds/daydream-bot-sandbox/actions/runs/27281579749
  - Review https://github.com/existential-birds/daydream-bot-sandbox/pull/6#pullrequestreview-4468531822
    posted **4 inline comments authored by `daydream-review[bot]`**, each raw body ending
    in a hidden `<!-- daydream-finding: <64-hex fingerprint> -->` marker (observed via
    `gh api repos/…/pulls/6/comments --jq '.[].body'`). **Both seeded defects found:**
    - Off-by-one in `chunk_records` (high/HIGH), fp `44450931…`:
      https://github.com/existential-birds/daydream-bot-sandbox/pull/6#discussion_r3388919207
    - TOCTOU race in `load_config` (low/HIGH), fp `564a1e32…`:
      https://github.com/existential-birds/daydream-bot-sandbox/pull/6#discussion_r3388919221
    - Plus two minor findings: non-positive `size` validation, fp `35aa8ea4…`
      (…#discussion_r3388919229) and JSON-error context, fp `f11adf95…`
      (…#discussion_r3388919238)

### Bug found and fixed at root (daydream, commit `58da2c0`)

The first run exposed a real defect: every agent call errored with
`Invalid API key · Fix external API key`, yet the run **exited 0**, printed
"No issues found — the implementation looks good", and uploaded a clean empty artifact —
an errored review masquerading as a passing one. Root cause: the Claude SDK reports fatal
run failures as `ResultMessage(is_error=True)`, not as an exception, and
`daydream/backends/claude.py` ignored the flag. Fix: the backend now raises
`ClaudeAgentError` on an error result; the run fails loudly and writes no artifact.
Regression-guarded by a backend test (`test_error_result_raises_instead_of_clean_empty_result`)
and a real-path test entering from `runner.run`
(`test_review_mode_errored_agent_never_writes_clean_artifact`). Verified live: the PR #5
review run below **fails** (exit ≠ 0) instead of reporting green.

## Step 3 — Push a new commit: no auto re-review

- Pushed `56ecfb0` to the PR #4 branch (fixes the TOCTOU finding via EAFP, seeds a new
  defect in `src/token_gen.py`).
- Observed: **no** new Daydream Review run for that head — the run list shows nothing
  between the original PR #4 review (13:14Z) and the next issue_comment event. `opened` /
  `ready_for_review` are the only auto triggers, as specced.

## Step 4 — `@daydream-review review` command chain (incl. the spike's NOT-RUN item)

- Maintainer (org owner `anderskev`) commented `@daydream-review review` on PR #4:
  https://github.com/existential-birds/daydream-bot-sandbox/pull/4#issuecomment-4670735074
- **Daydream Command** run (all steps green):
  https://github.com/existential-birds/daydream-bot-sandbox/actions/runs/27279776473
  - `Match review command` → matched
  - `Acknowledge with eyes reaction` → 👀 landed on the comment (observed via the
    reactions API: `eyes by github-actions[bot]`)
  - `Mint dispatch token` → **`actions/create-github-app-token` with
    `permission-actions: write` SUCCEEDED** — the App's newly granted `actions: write`
    permission works
  - `Dispatch review workflow` → succeeded
- The dispatched **Daydream Review** run executed with
  `actor = triggering_actor = daydream-review[bot]` (proof the dispatch used the App
  token, not `GITHUB_TOKEN`):
  https://github.com/existential-birds/daydream-bot-sandbox/actions/runs/27279786649
  (conclusion: failure — invalid API key, expected post-fix)
- **Spike NOT-RUN item — VERIFIED LIVE:** that run's completion **fired `workflow_run`**;
  **Daydream Post** ran:
  https://github.com/existential-birds/daydream-bot-sandbox/actions/runs/27279820948
  The `GITHUB_TOKEN` downstream-event suppression does NOT apply to App-token-dispatched
  runs. The full `@bot review` → command → dispatch → review → post chain completes.
  (For this failed dispatch-shape run the surface job logged
  `no PR resolvable for the failed analyze run; skipping comment` — the documented
  limitation: a failed dispatch run leaves no artifact and a default-branch `head_sha`.)
- **Reconcile semantics — VERIFIED LIVE on PR #6** (after the key was replaced; commit
  `c3c8450` fixed the commented-on off-by-one and seeded a new inverted-logic defect in
  `src/dedupe.py`):
  - Maintainer `anderskev` commented exactly `@daydream-review review`:
    https://github.com/existential-birds/daydream-bot-sandbox/pull/6#issuecomment-4671067170
  - **Daydream Command** (success, 👀 `eyes by github-actions[bot]` observed via the
    reactions API; mint + dispatch steps green):
    https://github.com/existential-birds/daydream-bot-sandbox/actions/runs/27281807641
  - Dispatched **Daydream Review** (workflow_dispatch, success):
    https://github.com/existential-birds/daydream-bot-sandbox/actions/runs/27281818428
  - **Daydream Post** (workflow_run, success) — log: artifact validated against
    `HEAD_SHA c3c8450…` / PR 6, then `Stale findings minimized: 4 succeeded, 0 failed`,
    then posted the second review:
    https://github.com/existential-birds/daydream-bot-sandbox/actions/runs/27282118309
  - **Fixed finding minimized as OUTDATED:** the off-by-one comment
    (…#discussion_r3388919207, fp `44450931…`) now has `isMinimized: true,
    minimizedReason: outdated` (GraphQL `reviewThreads` query) ✓
  - **New finding posted exactly once:** "Inverted dedupe logic returns duplicates"
    (fp `5f70e8be…`), one comment, not minimized:
    https://github.com/existential-birds/daydream-bot-sandbox/pull/6#discussion_r3388984753
    — second review: https://github.com/existential-birds/daydream-bot-sandbox/pull/6#pullrequestreview-4468612393
  - **No fingerprint duplicated:** comparing markers across both reviews, all six
    fingerprints (`44450931…`, `564a1e32…`, `35aa8ea4…`, `f11adf95…`, `5f70e8be…`,
    `878d2062…`) appear in exactly one comment each.
  - **Documented limitation observed (Assumption 5, message drift):** the second agent
    run re-worded every finding, so no fingerprint matched exactly — the still-valid
    size-validation finding was minimized and re-posted under a new fingerprint
    (`878d2062…`, …#discussion_r3388984759), and the still-present TOCTOU finding was
    minimized without being re-reported (the second review did not re-find it). This is
    the accepted v1 stateless-dedup behavior per the plan's Assumption 5 / templates
    README; the exact-match "matched → left untouched" path is exercised by the Task 7
    real-path fake-gh test (`test_fresh_post_then_idempotent_repost`).

## Step 5 — Negative cases

- **Non-collaborator `@daydream-review review`: NOT TESTED** — no second (non-collaborator)
  human account is available to the operator. The author gate is structure-tested
  (`author_association ∈ {OWNER, MEMBER, COLLABORATOR}` asserted in
  `tests/test_workflow_templates.py`).
- **Bot comments do not trigger the command path — PASS, observed twice:**
  - `coderabbitai[bot]` summary comment on PR #4 → Daydream Command run created and the
    `dispatch` job **skipped** (gate `comment.user.type != 'Bot'`):
    https://github.com/existential-birds/daydream-bot-sandbox/actions/runs/27279139597
  - Our own `daydream-review[bot]` failure comment on PR #5 → Daydream Command run
    **skipped** the same way:
    https://github.com/existential-birds/daydream-bot-sandbox/actions/runs/27279734743
- **Mention without `review` → no dispatch — PASS:** commented `@daydream-review thanks`
  on PR #4:
  https://github.com/existential-birds/daydream-bot-sandbox/pull/4#issuecomment-4670750775
  → Daydream Command ran and concluded success with `Match review command` executed and
  the reaction / mint / dispatch steps all **skipped**:
  https://github.com/existential-birds/daydream-bot-sandbox/actions/runs/27279873695
  Confirmed via the API: 0 reactions on the comment, and no Daydream Review run was
  created after the comment (latest remains the 13:30Z dispatch run).

## Failure surfacing (Task 9 should-have) — validated incidentally

- PR #5 (seeded off-by-one in `src/batcher.py`):
  https://github.com/existential-birds/daydream-bot-sandbox/pull/5
- Its auto review run **failed** (invalid key, post-fix behavior):
  https://github.com/existential-birds/daydream-bot-sandbox/actions/runs/27279698514
- Daydream Post's `surface-analyze-failure` job resolved the PR from the
  `pull_request`-shape `head_sha` and commented as the App bot:
  https://github.com/existential-birds/daydream-bot-sandbox/actions/runs/27279723246
  https://github.com/existential-birds/daydream-bot-sandbox/pull/5#issuecomment-4670728097
  (`daydream review failed — see run …/27279698514`, author `daydream-review[bot]`)

## What remains NOT TESTED (and why)

1. **Non-collaborator author gate**: no second human account available. The author gate
   is structure-tested (`author_association ∈ {OWNER, MEMBER, COLLABORATOR}` asserted in
   `tests/test_workflow_templates.py`).
2. **Exact-fingerprint dedup live** ("matched → prior comment left untouched"): both PR #6
   reviews produced zero overlapping fingerprints because the agent re-worded every
   message between runs (Assumption 5 drift — accepted, documented). Covered by the
   Task 7 real-path fake-gh integration test; all other reconcile branches (stale →
   minimize OUTDATED, new → post once, no duplicate markers) verified live above.

(The previous item 1 — real findings end-to-end with markers, minimize, post-once — is
now TESTED: the invalid `ANTHROPIC_API_KEY` was replaced with a working key and Steps 2
and 4 were re-run live on PR #6. No key material is recorded here or anywhere.)
