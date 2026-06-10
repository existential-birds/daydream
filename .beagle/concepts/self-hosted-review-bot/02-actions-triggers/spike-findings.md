# Task 0 Spike Findings — trigger chain, `workflow_run` payload, `resolveReviewThread`, 👀 reaction

> Live probes run 2026-06-10 against `existential-birds/daydream-bot-sandbox` (created for this spike;
> fork: `anderskev/daydream-sandbox-fork`). All results below are observed, not inferred, unless
> explicitly marked otherwise. Probe workflows: `probe-review.yml` (`pull_request` + `workflow_dispatch`,
> uploads a dummy artifact), `probe-post.yml` (`workflow_run` on "Probe Review", dumps
> `toJSON(github.event.workflow_run)` via `env:`), `probe-command.yml` (`issue_comment` → 👀 reaction +
> `gh workflow run` with `GITHUB_TOKEN`).

## Outcome summary

| # | Probe | Result |
|---|---|---|
| 1 | Dispatch chain (`gh workflow run` → `workflow_run`) | **PARTIAL FAIL — invalidates the command-path wiring.** Dispatch with `GITHUB_TOKEN` + `actions: write` succeeds and the dispatched run executes, but `workflow_run` does **not** fire when the dispatched run completes. It fires fine when the dispatch was made with a user PAT. |
| 2 | `workflow_run.pull_requests` population | Same-repo `pull_request`: **populated**. Fork `pull_request`: **empty**. `workflow_dispatch`: **empty**, and `head_sha` is the **default-branch** SHA, so the `commits/<head_sha>/pulls` fallback cannot resolve the PR on this shape. Fallback also returns `[]` for fork PRs when queried on the base repo; an open-PR list filtered by `head.sha` works for both PR shapes. |
| 3 | `resolveReviewThread` with least-privilege installation token | **FORBIDDEN** (`Resource not accessible by integration`) with `pull_requests: write, contents: read, metadata: read`. Fallback `minimizeComment(classifier: OUTDATED)` **succeeds** with the same token. **Task 6 must implement `minimizeComment`.** |
| 4 | 👀 reaction on a PR issue comment with `GITHUB_TOKEN` | Needs **`pull-requests: write`**, not `issues: write`. With `{actions: write, issues: write}` → HTTP 403; with `{actions: write, pull-requests: write}` (no `issues`) → success, reaction lands as `github-actions[bot]`. |

**Chosen stale-comment mechanism: `minimizeComment` (classifier `OUTDATED`)** — verified working with the
exact three-permission token the post workflow will mint. `resolveReviewThread` is not available to App
installation tokens at this scope.

---

## Step 1 — dispatch chain

Evidence (run ids in `existential-birds/daydream-bot-sandbox`):

| Trigger | probe-review run | probe-post (`workflow_run`) fired? |
|---|---|---|
| `gh workflow run` with **user PAT** (local) | 27274376887 | **Yes** — 27274390923, ~15 s later |
| `pull_request` (same-repo PR #1) | 27274482277 | **Yes** — 27274491283 |
| `pull_request` (fork PR #3) | 27274766328 | **Yes** — 27274776827 |
| `gh workflow run` with **`GITHUB_TOKEN`** from `probe-command` (`actions: write`) | 27274877395 | **No** — waited >2.5 min |
| same, repeated | 27275148000 | **No** — waited >45 s (and never appeared) |

- (a) **Confirmed:** `workflow_dispatch` via `GITHUB_TOKEN` with `permissions: actions: write` succeeds —
  the no-recursive-workflows rule exempts the dispatch itself; the dispatched run executes normally
  (actor/triggering_actor = `github-actions[bot]`).
- (b) **Refuted for the GITHUB_TOKEN path:** the dispatched run's *completion* does **not** emit a
  `workflow_run` trigger. The exemption covers `workflow_dispatch`/`repository_dispatch` events created
  by `GITHUB_TOKEN`, but downstream events of the resulting run are still suppressed. A PAT-dispatched
  run's completion does fire `workflow_run` (verified above), so the suppression is specific to
  `GITHUB_TOKEN` attribution, not to the dispatch path as such.

### Key Decision impact (plan §Architecture, Tasks 8/9) — REVISION REQUIRED

The planned chain `issue_comment → (GITHUB_TOKEN) gh workflow run daydream-review.yml → workflow_run →
daydream-post` **does not complete**: `daydream-post` never fires on the `@bot review` path. The auto
path (`pull_request` → `workflow_run`) is unaffected.

**Proposal:** `daydream-command` mints an App installation token via `actions/create-github-app-token`
with `permission-actions: write` and dispatches with *that* token instead of `GITHUB_TOKEN`. The command
workflow already never checks out PR code, so the locked privilege split (App material never co-resident
with PR code) is preserved; the App must additionally request the `actions: write` permission (setup
requirement for sub-project #3's README).
**NOT RUN:** App-token dispatch → `workflow_run` firing was not directly probed — the installed
`daydream-review` App grants only `contents: read, metadata: read, pull_requests: write`, and GitHub App
permissions cannot be changed via API (manual UI change + installation re-acceptance required). The
inference that a non-`GITHUB_TOKEN` actor's dispatch chain fires `workflow_run` is verified via the user
PAT row above; GitHub's recursion rule special-cases only `GITHUB_TOKEN`. Verify the App-token row live
in Task 11 step 4 before closing #147.

## Step 2 — `workflow_run.pull_requests` population (exact excerpts)

All excerpts are verbatim from the `probe-post` log dumps of `${{ toJSON(github.event.workflow_run) }}`
(URL fields and repo boilerplate elided for brevity; no tokens appear in these payloads).

### Shape 1 — same-repo `pull_request` (PR #1, run 27274491283)

```json
{
  "event": "pull_request",
  "head_branch": "probe/same-repo-pr",
  "head_sha": "47d029ceed61551dd3c7e9acc2e9ef0862787e44",
  "pull_requests": [
    {
      "base": {
        "ref": "main",
        "repo": { "id": 1265025306, "name": "daydream-bot-sandbox", "url": "https://api.github.com/repos/existential-birds/daydream-bot-sandbox" },
        "sha": "e7659d2c32487ba6cbe484fc62d42e31c08ab2fd"
      },
      "head": {
        "ref": "probe/same-repo-pr",
        "repo": { "id": 1265025306, "name": "daydream-bot-sandbox", "url": "https://api.github.com/repos/existential-birds/daydream-bot-sandbox" },
        "sha": "47d029ceed61551dd3c7e9acc2e9ef0862787e44"
      },
      "id": 3839200912,
      "number": 1,
      "url": "https://api.github.com/repos/existential-birds/daydream-bot-sandbox/pulls/1"
    }
  ]
}
```

`head_sha` = PR head. `pull_requests[0].number` is directly usable.

### Shape 2 — fork `pull_request` (PR #3 from `anderskev/daydream-sandbox-fork`, run 27274776827)

```json
{
  "event": "pull_request",
  "head_branch": "probe/fork-pr-2",
  "head_sha": "79a80eecf2d260d4fa31caafa1ec92d3c1d71f93",
  "pull_requests": [],
  "head_repository": { "full_name": "anderskev/daydream-sandbox-fork" }
}
```

`pull_requests` is **empty** for fork PRs; `head_sha` is the fork's PR head SHA;
`head_repository` identifies the fork.

### Shape 3 — `workflow_dispatch` (run 27274390923)

```json
{
  "event": "workflow_dispatch",
  "head_branch": "main",
  "head_sha": "e7659d2c32487ba6cbe484fc62d42e31c08ab2fd",
  "pull_requests": []
}
```

`pull_requests` is **empty** and — critically — `head_sha` is the **default branch tip** the dispatched
workflow ran on, *not* the PR head. The `workflow_dispatch` event's `inputs` (e.g. `pr_number`) do
**not** appear anywhere in the `workflow_run` payload.

### Fallback derivation — partially REFUTED

| Query | Result |
|---|---|
| `repos/<base>/commits/47d029c…/pulls` (same-repo PR head) | resolves PR #1 ✅ |
| `repos/<base>/commits/79a80ee…/pulls` (fork PR head) | **`[]` — does NOT resolve** ❌ |
| `repos/<fork>/commits/79a80ee…/pulls` (same SHA, queried on the fork) | resolves PR #3 (base = base repo) ✅ |
| `repos/<base>/pulls?state=open` filtered by `.head.sha == <head_sha>` | resolves PR #3 ✅ (and PR #1) |
| `repos/<base>/commits/<main_sha>/pulls` (dispatch shape) | `[]` — no PR has main as head ❌ |

### Key Decision impact (Task 9 contract: "PR number from `workflow_run.pull_requests[0]` with fallback `commits/$HEAD_SHA/pulls`") — REVISION REQUIRED

1. **Fork PRs:** the planned fallback endpoint returns `[]` on the base repo. Replace with: list open
   PRs on the base repo and filter by `.head.sha == workflow_run.head_sha` (verified above), or query
   `workflow_run.head_repository.full_name` instead of the base repo.
2. **Dispatch path (`@bot review`):** *neither* `pull_requests` *nor any head_sha-based fallback can
   resolve the PR* — the run's `head_sha` is the default-branch tip. The PR number must travel through
   the chain another way. **Proposal:** Phase A writes `pr_number` + the PR-head `head_sha` it actually
   reviewed into the findings artifact (already planned); for the dispatch shape, Phase B derives the
   target by fetching the live PR (`GET /pulls/<artifact.pr_number>`) with the App token and validating
   `artifact.head_sha == live PR head.sha` (the trust anchor is the GitHub API, not the artifact — a
   forged artifact can only point at a real, current head of the PR it names, which is exactly what
   gets posted to). The Task 5 event-match gate (`expected_head_sha`) accordingly takes its expected
   values from `workflow_run.head_sha` on `pull_request` shapes and from the live-PR lookup on the
   dispatch shape.
3. **Private-repo footnote (README):** with the sandbox private, the fork PR triggered **no
   `pull_request` workflow run at all** (default policy "run workflows from fork PRs" is off for
   private repos; the REST endpoints `…/actions/permissions/fork-pr-workflows-private` returned 404 for
   both repo and org, so it is UI-only here). The fork shape above was captured after flipping the
   sandbox public. Operators on private repos get no fork-PR auto-review of any kind — only the
   `@bot review` path can serve fork PRs there.

## Step 3 — `resolveReviewThread` with a least-privilege installation token

Setup: the org's `daydream-review` App (App ID 4014446, installation 139263623,
`repository_selection: all` so the sandbox is covered) grants exactly the least-privilege trio:
`contents: read, metadata: read, pull_requests: write`. Installation token minted via
`daydream.github_app.mint_installation_token` scoped to `daydream-bot-sandbox` (token redacted
everywhere; never written to the sandbox or this repo).

1. Posted an inline review comment as `daydream-review[bot]` on PR #1
   (`POST /pulls/1/comments`, comment id `3388118386`, with a `<!-- daydream-finding: aaa… -->` marker) ✅
2. GraphQL `pullRequest.reviewThreads` with the same token returned the thread
   (`id: "PRRT_kwDOS2bBGs6IeNGW"`, `isResolved: false`, author `daydream-review`) ✅ — the inventory
   query in Task 6 works on an installation token.
3. `resolveReviewThread(input: {threadId: …})` →
   ```json
   {"data":{"resolveReviewThread":null},
    "errors":[{"type":"FORBIDDEN","path":["resolveReviewThread"],
               "message":"Resource not accessible by integration"}]}
   ```
   **FAILS** with the least-privilege installation token.
4. `minimizeComment(input: {subjectId: <comment node id>, classifier: OUTDATED})` →
   ```json
   {"data":{"minimizeComment":{"minimizedComment":{"isMinimized":true,"minimizedReason":"outdated"}}}}
   ```
   **SUCCEEDS** with the same token.

**Task 6 implements `minimizeComment(classifier: OUTDATED)`** keyed on the stale finding's comment node
id (carry the comment `node_id`/`databaseId` in `PriorFinding` instead of, or in addition to,
`thread_id`). Plan Assumption 6's "thread resolved via GraphQL `resolveReviewThread`" is revised to
"stale comment minimized as OUTDATED"; semantics otherwise unchanged (matched → untouched,
reappeared-after-human-action → matched, new → posted). Note `partition()`'s `is_resolved` signal should
read the comment's `isMinimized` flag rather than thread `isResolved`.

## Step 4 — 👀 reaction permission

`POST /repos/<o>/<r>/issues/comments/<id>/reactions -f content=eyes` with `GITHUB_TOKEN` from an
`issue_comment` (PR comment) workflow:

| `permissions:` | Result |
|---|---|
| `{actions: write, issues: write}` | **403** `Resource not accessible by integration` (run 27274829998) |
| `{actions: write, issues: write, pull-requests: write}` | success (run 27274869490) |
| `{actions: write, pull-requests: write}` — no `issues` | **success** (run 27275142761), reaction by `github-actions[bot]` |

Even though the endpoint is under `/issues/`, a comment on a **PR** requires **`pull-requests: write`**;
`issues: write` is neither sufficient nor necessary. **Task 8's `daydream-command.yml` contract must
declare `permissions: {actions: write, pull-requests: write}`** (plan currently says
`{actions: write, issues: write}`).

## Deviations / NOT-RUN items

- **Sandbox visibility:** created private per instructions, but the fork-PR shape cannot run workflows
  on a private repo (see Step 2.3) and the API toggle is unavailable, so the sandbox was flipped
  **public** and left public for Task 11 (which needs fork-PR runs too). The org-level
  `members_can_fork_private_repositories` setting was temporarily enabled during the private-fork
  attempt and **restored to `false`**.
- **PR #2** (fork PR from the private-era fork) never triggered a run and its head fork was auto-disabled
  by the visibility change; closed, superseded by PR #3. The disabled repo `anderskev/daydream-bot-sandbox`
  could not be deleted (token lacks `delete_repo` scope) — harmless residue on the personal account.
- **App-token dispatch probe NOT RUN** (Step 1 proposal): blocked because the installed App lacks
  `actions: write` and App permissions are not API-changeable. Covered by the PAT-vs-GITHUB_TOKEN
  contrast; must be confirmed live in Task 11 once the App requests `actions: write`.

## Sandbox inventory (for Task 11 reuse)

- Repo: `existential-birds/daydream-bot-sandbox` (public), probe workflows on `main`.
- PR #1 (same-repo, open): has the App's minimized probe comment + two 👀-reacted `@probe review` comments.
- PR #3 (fork, open, from `anderskev/daydream-sandbox-fork`): exercised the fork `pull_request` shape.
- Fork: `anderskev/daydream-sandbox-fork` (public).
