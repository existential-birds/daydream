# Daydream review bot — repository dogfood workflows

## CI spend policy

`ci.yml` keeps Blacksmith runners for speed and retains the required `check`
status on every PR and main/master push. Superseded PR runs are cancelled;
push runs remain independent. `make check` and the Git hooks always run the
complete local gate.

The root CI gate runs in full for code, tests, dependencies, workflow changes,
direct pushes, and manual dispatches. Documentation changes use a lightweight
gate (naming and actionlint). Its allowlist covers the root README, CONTRIBUTING,
CHANGELOG, Markdown under `docs/`, and this workflow README. Product prompts,
templates, and executable files under `docs/` still run the complete gate.
RL-only changes also use the lightweight root gate while RL work is paused.

After a successful PR gate, CI uploads `ci-proof`, recording the actual checkout
commit and tree. A later PR update can use the lightweight gate if only allowed
documentation/RL files differ from that successful checkout. A main/master push
reuses PR CI only when the successful run belongs to the merged PR's final head
and its tested tree exactly matches the pushed tree. Git verifies the recorded
merge commit, tree, and PR parent. Missing/expired/corrupt evidence, API failures,
or a changed base that changes the tree fall back to the complete gate. The job
summary links the reused run. Proof artifacts expire after 14 days; coverage
retention stays unchanged.

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
with no parallel edit; `tests/test_dependabot_config.py` fails if the workflow
reintroduces a hardcoded pin instead of reading the manifest.

**Ownership.** Reviewers are auto-requested via the single-rule `CODEOWNERS`
(`* @existential-birds @anderskev`); no per-path entries were added.
