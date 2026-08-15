# Daydream review bot — repository dogfood workflows

These three workflows are this repository's own **repository-only Codex dogfood configuration**
— maintained for this repository and intentionally differing from the packaged workflow templates.

| File | Workflow | Role |
|---|---|---|
| `daydream-review.yml` | Daydream Review | Phase A — runs the reviewer over the PR head (unprivileged), uploads a `daydream-findings` artifact |
| `daydream-command.yml` | Daydream Command | Gatekeeper — listens for `@<bot> review` PR comments and dispatches Daydream Review |
| `daydream-post.yml` | Daydream Post | Phase B — fires when Daydream Review completes, validates the artifact, posts findings as your App bot |

## Install

To install these repository workflows in a repository of your own, create the
`OPENAI_API_KEY` repository secret the review job consumes (it authenticates the
Codex CLI before review), then follow the canonical installation guide:
[`daydream/templates/workflows/README.md#install`](../../daydream/templates/workflows/README.md#install).

## Trigger matrix

| Trigger | Path | Notes |
|---|---|---|
| `@<bot> review` PR comment | Command → Review → Post | Comment author must be OWNER / MEMBER / COLLABORATOR; bot comments are ignored; the PR head current at comment time is bound as the approved target |
| New commit after approval | Review rejects head drift | Comment `@<bot> review` again to approve the new head |
| Fork PRs | Same trusted-comment path | The review remains approval-gated and checks out only the approved head |

There is no automatic PR-open trigger. A credential-bearing review runs only
after a trusted comment explicitly approves the current head.

## Security model — the privilege split

No single job ever holds both PR code and the App private key:

- **Phase A (Daydream Review)** checks out and analyzes untrusted PR code,
  so it is unprivileged: `contents: read` GITHUB_TOKEN, `OPENAI_API_KEY` as
  its only model-provider credential, no App material anywhere. It
  authenticates Codex (`codex login --with-api-key`) before running
  `daydream --review --backend codex`. Its output is a passive data artifact
  (`findings.json`), never code.
- **Daydream Command** never checks out code, so it may hold App credentials:
  it mints a short-lived App token with `actions: write` (to dispatch the
  review) plus `pull-requests: write` (to post the 👀 reaction as the bot
  identity, not `github-actions[bot]`). Both writes flow through the App
  token, so the job's default GITHUB_TOKEN is unprivileged (`permissions: {}`).
- **Phase B (Daydream Post)** holds the App key but only ever checks out the
  base repo's default branch (trusted code). It mints a token with exactly
  `pull-requests: write, contents: read, metadata: read`, downloads the
  artifact, validates it against a strict schema and against the live PR
  (declared head SHA must match — a forged artifact cannot redirect the post),
  and posts. Untrusted values reach shells via `env:` only, never `${{ }}`
  interpolation.

The binding security spec is the daydream repo's
`.beagle/concepts/self-hosted-review-bot/roadmap.md` §"Sub-project #2
security design — the privilege split"; these workflows implement it.

## Dedup limitations (v1)

Re-reviews deduplicate against the bot's own prior comments via hidden
fingerprint markers in each comment body:

- **Exact fingerprint match only.** Identity is file + normalized title +
  anchors + normalized description. A finding whose message drifts between
  runs reads as one stale finding plus one new finding — expect an occasional
  duplicate with rephrased wording.
- **Matched findings are left untouched** — no comment editing.
- **Stale findings are minimized as OUTDATED**, not thread-resolved: GitHub
  App installation tokens cannot call `resolveReviewThread` at this permission
  scope, so the prior comment is collapsed via `minimizeComment(classifier:
  OUTDATED)` instead. The thread itself stays unresolved.
- **A finding that reappears after a human minimized/dismissed it is treated
  as matched** — the bot respects the dismissal and does not re-post.
- **Body-only (non-inline) findings have no thread**; when stale they simply
  stop appearing in the next run's review body.

Comment format is unchanged from `daydream --comment` — these workflows add
triggers and posting identity, not a new output format.
