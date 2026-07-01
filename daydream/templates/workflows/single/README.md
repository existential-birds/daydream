# Daydream review bot — single-file workflow (optional)

`daydream.yml` is an **optional** alternative to the three-file split
(`daydream-review.yml` + `daydream-command.yml` + `daydream-post.yml`) in the
parent directory. It does exactly the same thing — auto-review same-repo PRs on
open, review on demand via `@<bot> review`, post findings as your App bot — in a
single workflow file.

Pick this variant if you want a smaller install and an App that does **not** need
the `Actions: read & write` permission.

## Why one file drops `actions: write`

The split setup's `@<bot> review` path lives in `daydream-command.yml`, which
`workflow_dispatch`es `daydream-review.yml`. A `workflow_dispatch` made with the
built-in `GITHUB_TOKEN` never fires the downstream `workflow_run` trigger, so the
split setup dispatches with an App token carrying `actions: write` — that is the
*only* reason the App needs `Actions: read & write`.

This file replaces the cross-workflow dispatch with four jobs in one run, ordered
by `needs:` (`gate → analyze → post`, plus `surface-failure`). Nothing is
dispatched, so no job needs `actions: write`.

The privilege split is preserved at the **job** level — no single job holds both
PR code and the App key:

| Job | Holds App key? | Checks out PR code? |
|---|---|---|
| `gate` (decide + 👀 ack) | yes (ack only) | no |
| `analyze` (run reviewer) | no | yes |
| `post` (post findings) | yes | no |
| `surface-failure` | yes | no |

## Install

This variant is not installed by `daydream setup` (that path lands the three-file
split). Install it by hand:

1. Complete the GitHub App registration, secrets, and bot-handle variable exactly
   as in the [setup guide](../../../../docs/self-hosted-bot-setup.md), with **one
   difference**: the App needs only **Pull requests: Read and write**,
   **Contents: Read-only**, and **Metadata: Read-only** — **omit** *Actions: Read
   and write*.
2. Copy this `daydream.yml` into the target repository's `.github/workflows/`.
3. Do **not** also install the three split files — running both would review and
   post twice. If you are migrating from the split setup, delete
   `daydream-review.yml`, `daydream-command.yml`, and `daydream-post.yml`.

Everything else (trigger matrix, security model, dedup limitations) matches the
[parent `README.md`](../README.md).
