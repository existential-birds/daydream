# Daydream review bot — repository dogfood workflows

These three workflows are this repository's own **repository-only Codex dogfood configuration**
— maintained for this repository and intentionally differing from the packaged workflow templates.

| File | Workflow | Role |
|---|---|---|
| `daydream-review.yml` | Daydream Review | Phase A — runs the reviewer over the PR head (unprivileged), uploads a `daydream-findings` artifact |
| `daydream-command.yml` | Daydream Command | Gatekeeper — listens for `@<bot> review`, `@<bot> add sequence diagram`, and `@<bot> add flowchart` PR comments and dispatches Daydream Review with the matched command |
| `daydream-post.yml` | Daydream Post | Phase B — fires when Daydream Review completes, validates the artifact, posts findings as your App bot |

## Install

To install these repository workflows in a repository of your own, create the
`OPENAI_API_KEY` repository secret the review job consumes (it authenticates the
Codex CLI before review), then follow the canonical installation guide:
[`daydream/templates/workflows/README.md#install`](../../daydream/templates/workflows/README.md#install).

## Trigger matrix, security model, and dedup

The trigger semantics, privilege split, and dedup behavior are identical to the
packaged templates and documented once in the canonical
[`daydream/templates/workflows/README.md`](../../daydream/templates/workflows/README.md).
The only Codex-specific difference: Phase A holds `OPENAI_API_KEY` as its only
model-provider credential, authenticates Codex (`codex login --with-api-key`),
and runs `daydream --review --backend codex`.

## Dependabot dependency updates

`.github/dependabot.yml` keeps dependency updates arriving as small, grouped,
reviewer-friendly PRs across three managed ecosystems:

| Ecosystem | Directory | Notes |
|---|---|---|
| `uv` | `/` | Root workspace dependencies (`pyproject.toml` / `uv.lock`) |
| `uv` | `/rl/daydream_review` | The standalone RL package's own dependencies |
| `npm` | `/.github/workflows` | Deliberately points at this folder's `package.json`, which tracks the single `@openai/codex` dependency used by CI — do not "fix" the directory to `/` |

**Volume bounds.** All three ecosystems run weekly on Mondays (06:00,
`Australia/Brisbane`) with `open-pull-requests-limit: 5` per ecosystem. The
two multi-dependency `uv` blocks each use one minor+patch group, so at most
one grouped PR per uv project; the npm block is ungrouped since it tracks a
single dependency. Commit messages carry the `chore(deps)` prefix.

**Action bumps are automated.** Every third-party action stays pinned to a full
commit SHA — enforced by
`tests/test_workflow_templates.py::test_bot_workflow_action_references_are_pinned_to_commit_shas`
— and the `github-actions` ecosystem raises the bumps.

**Validation gate.** Every dependency PR must pass `make check` before merge —
its lockcheck-first ordering is what catches `uv.lock` / `pyproject.toml`
drift. CI installs codex at the version read from the tracked
`.github/workflows/package.json`, so an npm block bump PR updates CI end-to-end
with no parallel edit; `tests/test_dependabot_config.py` fails if the workflow
reintroduces a hardcoded pin instead of reading the manifest.

**Ownership.** Reviewers are auto-requested via the single-rule `CODEOWNERS`
(`* @existential-birds @anderskev`); no per-path entries were added.
