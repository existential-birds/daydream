# Contributing to Daydream

This guide is for coding agents contributing to Daydream. Read it before
starting work, alongside [AGENTS.md](AGENTS.md), the source of repository guidance.
This document owns the contribution workflow; AGENTS.md owns architecture and
runtime contracts. Update the relevant guidance whenever a change makes it stale.

Start with the [shared agent skill workflow](#shared-agent-skill-workflow):
`prepare-implement` → generate a self-contained `/goal` prompt → execute it.
Use the setup and validation references below during execution, and satisfy the
[pull request requirements](#pull-request-requirements) before requesting review.

## Prerequisites

Verify these prerequisites before running the local gate:

- **Python ≥ 3.14**
- **[`uv`](https://docs.astral.sh/uv/)** — installs and manages the virtualenv and runs every gate tool
- **Claude Code CLI** ([claude.ai/code](https://claude.ai/code)) — the default review backend; the test suite exercises it

Backend CLIs are optional and only needed when you work on that backend:

- **`gh`** (GitHub CLI) — the `--comment` PR-comment mode and `reconcile.py`
- **Codex CLI**, **Pi CLI**, **Osprey CLI** — only for their respective backends in `daydream/backends/`

Docker is optional: `make actionlint` runs the workflow-YAML checks in a
pinned container, but when no Docker daemon is available that target is
skipped with a note (`actionlint skipped: Docker daemon is not available`)
and exits 0 — so `make check` still passes locally. CI always runs
actionlint, so a workflow error a local run skipped will still fail your PR.

## Setup

One-time, from the repo root:

```bash
make install
make hooks
```

`make install` runs `uv sync --all-extras`. All extras are installed on
purpose so the full gate suite runs — the benchmark objective tests need the
`benchmark` extra's harbor package. Note that, like `uv sync`, this does
**not** put a `daydream` command on your `PATH`; for the runnable CLI, see
[Quick start](README.md#quick-start) in the README.

`make hooks` symlinks two git hooks:

- `scripts/hooks/pre-commit` — a fast commit-time gate that runs ruff on the
  staged Python files (scoped to `daydream/`, `tests/`, and
  `rl/daydream_review/`, linted from the index, not the working tree).
- `scripts/hooks/pre-push` — verifies every pushed commit carries a valid
  signature (SSH or GPG), then delegates to `make check`, the full local CI gate.

### SSH signing

The pre-push hook **rejects unsigned commits** and accepts any valid signature
(SSH or GPG). Make sure your commits are signed, e.g. with your SSH key:

1. Configure git to sign with your SSH key, then enable commit signing:

   ```bash
   git config --global gpg.format ssh
   git config --global user.signingkey ~/.ssh/<your-key>.pub
   git config --global commit.gpgsign true
   ```

2. Have your SSH key loaded — `ssh-add -l` must list a key. If it lists
   nothing, run `ssh-add ~/.ssh/<your-key>`.

If the contribution branch contains unsigned commits, re-sign those commits:

```bash
git rebase --exec 'git commit --amend --no-edit -S' HEAD~N
```

where `N` is the number of commits to re-sign.

## Everyday commands and the required gate

The focused targets, all run from the repo root:

| Command | What it does |
|---|---|
| `make lint` | Ruff over `daydream tests scripts/ci_*.py` (120 cols, `E F I W`, py314; `daydream/atif/**` is lint-exempt as vendored code) |
| `make typecheck` | mypy over `daydream tests scripts/ci_*.py` |
| `make test` | `pytest -n auto` with coverage; the branch-coverage floor (`fail_under` in `pyproject.toml`) is enforced here, not in global addopts — a bare or targeted `pytest` run stays plain |
| `make deadcode` | vulture dead-code scan over the root project and the RL package |
| `make coverage-report` | checks that `coverage.xml` exists after `make test`; the measurement + ratchet procedure is in [docs/coverage.md](docs/coverage.md) |
| `make actionlint` | Docker-pinned actionlint over `.github/workflows/*.yml` plus the packaged workflow templates; without `ACTIONLINT_REQUIRE_DOCKER`, a missing daemon skips with a note (exit 0); with it set, a missing daemon fails |
| `make lockcheck` | `uv lock --check` — fails if `uv.lock` is out of sync with `pyproject.toml` |
| `make check-naming` | naming-convention check over the repo (`scripts/check-naming.sh`) |
| `make rl-check` | the RL project's own lockcheck + ruff + mypy + pytest suite, run with a scoped git identity as process env. **This is explicitly not part of `make check`** — it mirrors `ci.yml`, where RL is a manually dispatched job while RL work is paused (its e2e test drives the `claude` CLI that CI's `check` job never installs). Run it by hand whenever you change `rl/daydream_review`. |

The required gate:

```bash
make check
```

which runs, in order:

```text
lockcheck install lint deadcode typecheck test actionlint coverage-report check-naming
```

This is the same set of steps a full `check` job in `.github/workflows/ci.yml` runs —
`rl-check` is the only CI job the gate deliberately omits. One scope difference:
`make deadcode` also runs vulture over `rl/daydream_review/`, which CI's `check`
job does not include — that scan happens in the manual `rl-check` job — so the
local gate is the stricter one. The pre-push hook runs `make check` after
verifying signatures, so a green local `make check` is what keeps your push from
being rejected. CI defaults to eight duration-weighted standard `ubuntu-24.04`
pytest jobs with four workers each. A final `check` joins static checks and validated aggregate branch
coverage; missing or failed shards cannot pass. The script unit tests run in the
static job. See [coverage aggregation](docs/coverage.md) in [the CI shard tooling](scripts/ci_shard.py).

CI runs the full gate before a queued merge and reuses only a verified merge-group tree afterward. PR updates run basic checks only while effective branch rules enforce the queue and required `check`; otherwise they run in full. See the [CI spend policy](.github/workflows/README.md#ci-spend-policy). The local gate always runs in full.

## Testing

Follow the [testing guidance in AGENTS.md](AGENTS.md#testing-standard-mandatory)
when writing, changing, or reviewing tests. Run affected checks during implementation
and the [required gate](#everyday-commands-and-the-required-gate) before publication.
Record commands, results, and any skips in the PR's Test Plan.

The shared pytest harness redirects the home JSONL-store fallback into a per-test
store and replaces the dataset uploader's HF adapter with an offline substitute.
Capture tests should pass an explicit temporary `dataset_store_path`; upload
failure tests should inject an offline adapter. Ambient Hub destinations and
`HF_TOKEN` are removed before each test, so enabling capture in one test cannot
select the operator's dataset or credentials. Production defaults are unchanged.

## Conventions

- **Ruff** — 120 columns, rule set `E F I W`, target Python 3.14.
  `daydream/atif/**` is lint-exempt as vendored Harbor code (see
  [daydream/atif/NOTICE](daydream/atif/NOTICE)); make only mechanical edits
  there.
- **mypy** — configured in `pyproject.toml`, run over `daydream tests scripts/ci_*.py`;
  hand-written stubs live in `mypy_stubs/`.
- **Dependencies** — declared in `pyproject.toml`; keep `uv.lock` in sync (`uv
  lock`) or `make check` fails at its first step (`lockcheck`).
- **Commit messages** — Conventional Commits, e.g. `feat(backends): ...`.
- **History** — keep a linear history; rebase the contribution branch instead of
  creating merge commits.
- **Staging** — stage explicitly (`git add <path>`), never `git add -A`.
- **Documentation** — update it when your change makes it stale, including
  [AGENTS.md](AGENTS.md) when the agent-facing contract changes.
- **Pull requests** — follow the [pull request requirements](#pull-request-requirements).
- **Git hooks** — run every commit and push with hooks enabled. Never bypass
  hooks with `--no-verify`, environment variables, configuration, or any other
  mechanism, even for pre-existing failures. Fix the failure on the branch and
  retry normally, or report the blocker. A commit or push is complete only after
  the hook-enabled command succeeds.
- **No caveats** — work is completed and proven, or explicitly in progress.
  No deferred items, no "optional" follow-ups, no smoke tests substituted for
  real coverage.

## Versioning and changelog

Daydream follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
Assess changes to public contracts and state the compatibility impact in the PR.
For stable releases, incompatible API changes require a major version, compatible
new functionality or deprecations a minor version, and compatible bug fixes a
patch version. Apply SemVer's initial-development rules while the version is
`0.y.z`; still identify breaking changes and migration steps explicitly.
Change version metadata when release/version work is in scope, keeping package
version declarations and `uv.lock` consistent.

[CHANGELOG.md](CHANGELOG.md) follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Add notable changes under
`[Unreleased]`, using the appropriate `Added`, `Changed`, `Deprecated`, `Removed`,
`Fixed`, or `Security` heading. Describe the impact for users and contributors;
link the issue or PR when available. During a release, move those entries into a
versioned section with an ISO date (`YYYY-MM-DD`), newest release first, and
update comparison links. If a PR has no notable change, explain why the changelog
checklist item is N/A.

## Pull request requirements

Every PR must start from [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md),
including PRs created through `gh` or another API. Use the
[pr skill](.agents/skills/pr/SKILL.md) when writing the body; the repository
template takes precedence over illustrative layouts in that skill.

Complete **Summary**, **Motivation**, **Test Plan**, and **Checklist**. Remove
only sections the template explicitly marks optional. Link the source issue
when one exists, using closing keywords only when the PR fully resolves it.

Account for every checklist item before requesting review. Check applicable
items only after verification and put the evidence in **Test Plan**. Keep an
inapplicable conditional item unchecked and append `N/A` with a concrete reason.
An applicable item that is incomplete remains unchecked and blocks readiness for
review; report it as a blocker. Refresh the checklist and evidence after changes
so they describe the latest PR HEAD.

## Where to look deeper

This guide links rather than duplicates; these documents carry the detail:

- [AGENTS.md](AGENTS.md#architecture) — architecture boundaries, runtime invariants,
  and required references for each subsystem.
- [docs/extensions.md](docs/extensions.md) — the versioned extension API for
  forks (`daydream_ext`).
- [docs/coverage.md](docs/coverage.md) — the coverage gate and ratchet
  procedure.

By task:

- Benchmarks → [docs/benchmark.md](docs/benchmark.md)
- Tracing / observability → [docs/observability.md](docs/observability.md)
- Corpus / training launches → [docs/training-launch.md](docs/training-launch.md)
- RL package work → [rl/daydream_review/README.md](rl/daydream_review/README.md)
  (and run `make rl-check` — see the commands table above)

## Shared agent skill workflow

Use this sequence for a GitHub issue or implementation spec:

1. **Prepare:** invoke [prepare-implement](.agents/skills/prepare-implement/SKILL.md).
   Read the complete source and repository guidance, gather bounded context with
   subagents, and capture the exact base and worktree state. Preparation is
   read-only; leave the checkout, refs, and files unchanged.
2. **Generate the goal prompt:** return one self-contained `/goal` prompt with
   the scope, acceptance criteria, verified baseline, implementation lanes,
   validation, publication, and review-loop requirements defined by the skill.
   The prompt must contain enough context for a fresh session to execute it.
3. **Execute the prompt:** run it in a new session. Recheck repository guidance
   and worktree state, create a new branch from the verified requested base or
   captured HEAD, preserve unrelated work, and implement the goal. Validate,
   make signed Conventional Commits with explicit staging, and push with hooks
   enabled. Open a PR following the [PR requirements](#pull-request-requirements),
   then complete the review loop below before reporting merge readiness.

After opening the PR, run the available Daydream reviewer from its checkout:

```bash
daydream . --backend codex --comment --non-interactive
```

Verify the selected PR's base/head first. If the default comparison base differs
from its verified base, append `--base <verified ref/SHA>`; the
[prepare-implement skill](.agents/skills/prepare-implement/SKILL.md) explains how
to resolve and record the exact review target.

Then use [fetch-pr-feedback](.agents/skills/fetch-pr-feedback/SKILL.md),
[receive-feedback](.agents/skills/receive-feedback/SKILL.md), and
[respond-pr-feedback](.agents/skills/respond-pr-feedback/SKILL.md) to collect all
reviewer and bot feedback, verify findings, address valid issues, and reply.
Include other bots such as `greptile-apps`; author/self comments can contain
Daydream findings, so the goal uses `--include-author --include-self`. Follow the
linked skills for the review loop and confirm review and required CI cover the
latest PR HEAD before reporting merge readiness. The workflow stops before merging.

For a curated simplification, use
[publish-improve-finding](.agents/skills/publish-improve-finding/SKILL.md) to capture
the opportunity and provenance, then
[create-improve-pr](.agents/skills/create-improve-pr/SKILL.md) when asked to publish
its implementation. These specialized workflows preserve discovery and benchmark
evidence; they are separate from ordinary issue implementation.

## Where the agent guidance lives

Repository architecture and runtime guidance — invariants, backend protocol,
budgets, and artifact boundaries — lives in [AGENTS.md](AGENTS.md). Reusable agent
workflows live under `.agents/skills/`; the [.gitignore](.gitignore) allowlist
tracks the shared skills linked above. Keep detailed workflow procedures in their
skills rather than duplicating them here. If your PR changes a contract described
in AGENTS.md, update it in the same PR.
