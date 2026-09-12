# Contributing to Daydream

Thanks for contributing! This guide walks you from a fresh clone to a signed,
gate-green pull request: prerequisites, one-time setup, the commands that make
up the local quality gate, and the conventions reviewers will hold your change
to. Daydream is developed with coding agents in the loop, so
[CLAUDE.md](CLAUDE.md) is the agent-facing counterpart of this document — it
carries the architecture and testing rules the agent must follow. This guide
links to it rather than duplicating it: if you are changing behavior that
CLAUDE.md describes, update both.

## Prerequisites

Required for any contribution:

- **Python ≥ 3.12.13**
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
- `scripts/hooks/pre-push` — verifies every pushed commit carries an SSH
  signature, then delegates to `make check`, the full local CI gate.

### SSH signing

The pre-push hook **rejects unsigned commits**. Make sure your commits are
signed with your SSH key:

1. Enable commit signing:

   ```bash
   git config --global commit.gpgsign true
   ```

2. Have your SSH key loaded — `ssh-add -l` must list a key. If it lists
   nothing, run `ssh-add ~/.ssh/<your-key>`.

If you already made commits without signing, re-sign them:

```bash
git rebase --exec 'git commit --amend --no-edit -S' HEAD~N
```

where `N` is the number of commits to re-sign.

## Everyday commands and the required gate

The focused targets, all run from the repo root:

| Command | What it does |
|---|---|
| `make lint` | Ruff over `daydream tests` (120 cols, `E F I W`, py312; `daydream/atif/**` is lint-exempt as vendored code) |
| `make typecheck` | mypy over `daydream tests` |
| `make test` | `pytest -n auto` with coverage; the branch-coverage floor (`fail_under = 86` in `pyproject.toml`) is enforced here, not in global addopts — a bare or targeted `pytest` run stays plain |
| `make deadcode` | vulture dead-code scan over the root project and the RL package |
| `make coverage-report` | checks that `coverage.xml` exists after `make test`; the measurement + ratchet procedure is in [docs/coverage.md](docs/coverage.md) |
| `make actionlint` | Docker-pinned actionlint over `.github/workflows/*.yml` plus the packaged workflow templates; skipped with a note (exit 0) when no Docker daemon is available |
| `make lockcheck` | `uv lock --check` — fails if `uv.lock` is out of sync with `pyproject.toml` |
| `make check-naming` | naming-convention check over the repo (`scripts/check-naming.sh`) |
| `make rl-check` | the RL project's own lockcheck + ruff + mypy + pytest suite, run with a scoped git identity as process env. **This is explicitly not part of `make check`** — it mirrors `ci.yml`, where RL is a separate job (its e2e test drives the `claude` CLI that CI's `check` job never installs). Run it by hand whenever you change `rl/daydream_review`. |

The required gate:

```bash
make check
```

which runs, in order:

```text
lockcheck install lint deadcode typecheck test actionlint coverage-report check-naming
```

This is the same set of steps the `check` job in `.github/workflows/ci.yml` runs —
`rl-check` is the only CI job the gate deliberately omits. The pre-push hook runs
`make check` after verifying signatures, so a green local `make check` is what
keeps your push from being rejected.

## Testing policy

Every user-visible behavior must have at least one **real-path test**: a test
that enters from the production entrypoint (`runner.run` / the CLI) with real
dependencies — a real temp git worktree, a real filesystem, a real event loop —
mocking only the external network/API backend, via the `Backend` protocol /
`create_backend` seam. Tests must assert observable outcomes (exit code, files
written, fixes applied or declined, transcript state), never that a function was
merely called. Unit tests are supplementary, not a substitute.

This is the standard reviewers hold PRs to, and the one the agent-facing
[CLAUDE.md](CLAUDE.md) §Testing standard states as mandatory.

## Exemplars

Read these before writing your first test — they are the reference
implementations of the real-path standard:

- `tests/test_deep_orchestrator.py`:
  `test_apply_fixes_gate_non_interactive_takes_safe_default` and
  `test_apply_fixes_gate_eof_declines_cleanly_no_crash` — the non-interactive /
  EOF gate tests, exercising `runner.run` against a real worktree with only the
  backend stubbed.
- Secondary stub-backend-seam examples: `tests/test_cli.py` and
  `tests/test_integration.py`, which reuse the `_install_stub_backend` and
  `_silence` helpers imported from `tests/test_deep_orchestrator.py`.

## Conventions

- **Ruff** — 120 columns, rule set `E F I W`, target Python 3.12.
  `daydream/atif/**` is lint-exempt as vendored Harbor code (see
  [daydream/atif/NOTICE](daydream/atif/NOTICE)); make only mechanical edits
  there.
- **mypy** — configured in `pyproject.toml`, run over `daydream tests`;
  hand-written stubs live in `mypy_stubs/`.
- **Dependencies** — declared in `pyproject.toml`; keep `uv.lock` in sync (`uv
  lock`) or `make check` fails at its first step (`lockcheck`).
- **Commit messages** — Conventional Commits, e.g. `feat(backends): ...`.
- **Staging** — stage explicitly (`git add <path>`), never `git add -A`.
- **Documentation** — update it when your change makes it stale, including
  [CLAUDE.md](CLAUDE.md) when the agent-facing contract changes.
- **Pull requests** — must include a **Test Plan** section per the
  [PR template](.github/PULL_REQUEST_TEMPLATE.md).
- **Never** bypass the pre-push hook, skip tests, or push with
  `git push --no-verify`.
- **No caveats** — work is completed and proven, or explicitly in progress.
  No deferred items, no "optional" follow-ups, no smoke tests substituted for
  real coverage.

## Where to look deeper

This guide links rather than duplicates; these documents carry the detail:

- [CLAUDE.md](CLAUDE.md) §Architecture — the module responsibility map and
  pipeline walkthrough.
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

## Your first PR

A compact walkthrough; each step links back to the section that explains it:

1. Branch from a clean, up-to-date main:
   `git checkout -b <your-branch>`. (Daydream uses a linear history — rebase
   rather than merge.)
2. `make install && make hooks` — once per clone ([Setup](#setup)), including
   SSH signing so the pre-push hook accepts your commits.
3. Develop with a real-path test per the [testing policy](#testing-policy);
   the [exemplars](#exemplars) show the shape.
4. Run the focused targets for what you touched (`make lint`, `make
   typecheck`, `make test`) — see the [commands table](#everyday-commands-and-the-required-gate).
   If you changed `rl/daydream_review`, also run `make rl-check`.
5. Run the full gate: `make check`.
6. Commit with a Conventional Commits message and explicit staging
   ([Conventions](#conventions)).
7. Push (`git push -u origin <your-branch>`) — the hook re-runs `make check`
   after verifying signatures.
8. Open a PR with a **Test Plan** section per the
   [PR template](.github/PULL_REQUEST_TEMPLATE.md).

## Where the agent guidance lives

Daydream is developed with coding agents in the loop. Everything agent-facing —
architecture invariants, backend protocol, budgets, artifact boundaries — lives
in [CLAUDE.md](CLAUDE.md), which this guide deliberately links instead of
duplicating. If your PR changes any contract described there, update it in the
same PR. Before opening the PR, run through the checklist at the bottom of the
[PR template](.github/PULL_REQUEST_TEMPLATE.md).
