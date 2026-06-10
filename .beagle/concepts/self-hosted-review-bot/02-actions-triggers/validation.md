# Task 11 — Sandbox end-to-end acceptance (live validation)

> Run 2026-06-10 against `existential-birds/daydream-bot-sandbox` (public) by the operator
> account `anderskev` (org owner). Daydream under test: branch `feat/actions-triggers`,
> pinned at install time (see Step 1). Every URL below was observed live via `gh`;
> nothing is inferred or reconstructed.

## Outcome summary

| Step | What | Result |
|---|---|---|
| 1 | Install templates + secrets/vars | DONE — commit `0383733` on sandbox `main` |
| 2 | Auto review on PR opened | Trigger chain PASS (Review → Post fired, artifact validated, post-findings ran); **review content NOT TESTED — operator `ANTHROPIC_API_KEY` is invalid (verified 401 twice, independent of CI)**. Exposed and fixed a real daydream bug: errored agent run masqueraded as a clean "no issues" review (fix `58da2c0`) |
| 3 | No re-review on push | PASS — no run for pushed head `56ecfb0` |
| 4 | `@daydream-review review` command chain | PASS mechanically — 👀 reaction, **App-token mint with `permission-actions: write` SUCCEEDED**, dispatch ran as `daydream-review[bot]`, and the dispatched run's completion **DID fire `workflow_run`** (the spike's NOT-RUN item, now verified live). Dedup/minimize semantics NOT TESTED (no findings possible without a valid key) |
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
  (daydream repo branch head at validation end: `58da2c0`).
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
- **NOT TESTED: inline comments with hidden markers.** The findings artifact was empty
  because every agent invocation failed with `Invalid API key`. The operator-provided
  `ANTHROPIC_API_KEY` is invalid upstream — verified twice from the workstation,
  independent of CI (`POST /v1/messages` and `GET /v1/models` both return
  `401 authentication_error: invalid x-api-key`). No valid key was available, so no real
  findings could be produced; comment posting, marker embedding, and dedup could not be
  observed live. (They are covered by the Task 7 real-path fake-gh integration tests and
  the Task 0 live probes of `minimizeComment`, but Task 11's live pass could not exercise
  them.)

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
- **NOT TESTED:** 👀-then-minimize/post dedup semantics (stale finding minimized as
  OUTDATED, new finding posted once, unchanged finding not duplicated) — requires real
  findings from a working Anthropic key (see Step 2).

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

1. **Real findings end-to-end** (inline comments with hidden markers, stale-comment
   minimize-as-OUTDATED, new-finding-posted-once, unchanged-finding-not-duplicated): the
   only Anthropic key available to this validation is invalid upstream (401 verified
   twice from the workstation). Re-running Steps 2 and 4's content assertions requires a
   valid `ANTHROPIC_API_KEY`; the mechanics around them (chain, artifact handoff,
   validation, posting identity, reactions, gates) are all verified above.
2. **Non-collaborator author gate**: no second human account available.
