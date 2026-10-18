# Daydream review bot — repository dogfood workflows

## CI spend policy

`ci.yml` uses standard GitHub-hosted `ubuntu-24.04` runners and retains the
required `check` status on every PR and main/master push. Standard runner usage
is free for this public repository; private repositories use their included
minutes and applicable billing ([GitHub runner documentation](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)). Superseded PR runs are cancelled;
push runs remain independent. `make check` and the Git hooks always run the
complete local gate.

A PR runs lock verification, dependency installation, lint, naming checks, and
workflow lint only when GitHub's effective
active branch rules require both the merge queue and the Actions `check` status.
The originating queue rulesets, including inherited rulesets, must also be active
and explicitly report an empty bypass-actor list. Until those rules are active,
or if their identity or bypass evidence cannot be verified, every PR runs the
complete root gate. GitHub omits bypass actors when the caller lacks write access
to the ruleset; that omitted evidence also keeps the complete gate. This workflow
does not increase token permissions to obtain it.
Documentation and RL-only changes follow the same rule; no path allowlist skips
the queue gate.

`merge_group: checks_requested` runs the complete root suite with branch
coverage and its existing enforced floor on `ubuntu-24.04`, with
four pytest workers. Typechecking and the root dead-code scan also run in full
here. A failing required `check` prevents the queue from merging
the group. Manual dispatches and direct pushes also run the complete gate.
Local Git hooks remain complete regardless of remote queue state.

Only a successful full merge-group job uploads `ci-proof`. Its versioned proof
binds the repository, run, queue head ref and actual checkout SHA/tree to the
captured base ref, commit, and tree. Git verifies the base is an earlier ancestor;
no fixed parent shape is assumed for groups containing multiple PRs. PR basic
checks and legacy PR proofs cannot authorize reuse. A main/master push can
reuse only a successful matching merged-PR push whose entire tree matches that
verified queue checkout and whose previous branch tip belongs to its tested
history. Direct pushes, cross-branch evidence, changed trees, unavailable Git
objects, missing/expired/corrupt proofs and API failures run the complete gate.
The job summary links any reused run. Proof artifacts expire after 14 days.
Each full rerun replaces its coverage report, including failed-suite diagnostics;
coverage retention stays unchanged. A successful full queue rerun replaces the
previous proof artifact only after all gates pass.

Queue activation is a separate repository setting change, performed after this
workflow reaches the default branch. The required check name remains `check`.
Queue rules must have no bypass actors; administrators use the same verified
merge path. This workflow never writes repository settings.

Automatic `rl-check` is paused by maintainer choice because no RL work is active.
Its complete lockcheck, lint, typecheck, dead-code, and Docker test gate remains
available through **Run workflow → run_rl**, or:

```bash
gh workflow run ci.yml --ref main -f run_rl=true
```

Use `--ref YOUR_BRANCH` to check another revision. The root gate also runs in
full on a manual dispatch. `make rl-check` remains the local RL gate. Both
projects and all workflow tool installations use Python 3.14. The standalone
project uses verifiers 0.4.0; Prime-RL training integration is paused because
its Python 3.12 workspace cannot install the migrated package.

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
with no parallel edit.

**Ownership.** Reviewers are auto-requested via the single-rule `CODEOWNERS`
(`* @existential-birds @anderskev`); no per-path entries were added.
