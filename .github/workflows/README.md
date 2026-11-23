# Repository workflows

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
Documentation-only changes follow the same rule; no path allowlist skips
the queue gate.

`merge_group: checks_requested` runs the complete root suite with branch
coverage and its existing enforced floor on `ubuntu-24.04`, across
eight duration-weighted pytest jobs with four workers each by default. The
planner runs first; static checks (including script unit tests) and pytest then
run independently. A coverage job
validates all checkout/run-bound shard receipts and combines branch data before
enforcing the existing aggregate floor and uploading `coverage-report`. The final
`check` runs with `always()` and explicitly rejects failed, cancelled, missing or
unexpectedly skipped dependencies. Typechecking and the root dead-code scan run
in the static job. Basic/reuse modes never launch full-suite shards. See
[coverage integrity](../../docs/coverage.md). A failing required `check` prevents the queue from merging
the group. Manual dispatches and direct pushes also run the complete gate.
Local Git hooks remain complete regardless of remote queue state.

Manual `test_jobs` accepts integers from 1 through 20. One selects the original
sequential eight-worker benchmark baseline; other counts use four workers per
pytest job. Eight is the selected default. Provisioning stays within each shard.
At twenty shards, concurrent static checks can also consume runner capacity,
so queue delay and observed overlap remain part of the
measurement. The earlier four-job benchmark is preserved as historical evidence.

Only a successful final full merge-group `check` uploads `ci-proof`. Its versioned proof
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

## Release publication

`release.yml` tags and publishes a release when its release PR merges; the rule
lives in [CONTRIBUTING](../../CONTRIBUTING.md#versioning-and-changelog). It runs
only on `main` pushes that change `pyproject.toml`, and on manual dispatch from
`main`. Its token has `contents: write` and nothing else.

## Dependabot dependency updates

`.github/dependabot.yml` keeps dependency updates arriving as small, grouped,
reviewer-friendly PRs across two managed ecosystems:

| Ecosystem | Directory | Notes |
|---|---|---|
| `uv` | `/` | Root workspace dependencies (`pyproject.toml` / `uv.lock`) |
| `github-actions` | `/` | Third-party action pins (see below) |

**Volume bounds.** Both ecosystems run weekly on Mondays (06:00,
`Australia/Brisbane`) with `open-pull-requests-limit: 5` per ecosystem. Each
uses one minor+patch group, so at most one grouped PR each. Commit messages
carry the `chore(deps)` prefix.

**Action bumps are automated.** Every third-party action stays pinned to a full
commit SHA, and the `github-actions` ecosystem raises the bumps.

**Validation gate.** Every dependency PR must pass `make check` before merge —
its lockcheck-first ordering is what catches `uv.lock` / `pyproject.toml`
drift.

**Ownership.** Reviewers are auto-requested via the single-rule `CODEOWNERS`
(`* @existential-birds @anderskev`); no per-path entries were added.
