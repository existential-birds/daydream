# Contributing to Daydream

This guide covers setup and contribution checks. [CLAUDE.md](CLAUDE.md) owns the
architecture and behavior contracts; update it when those contracts change.

## Prerequisites

Use Python ≥3.12.13, [uv](https://docs.astral.sh/uv/), and the Claude Code CLI
(the default backend, also exercised by tests). Install `gh` for PR feedback;
Codex, Pi, and Osprey CLIs are needed only for their backends.

Docker runs the pinned actionlint container. A missing daemon skips that local
check with a note; CI always runs it. `ACTIONLINT_REQUIRE_DOCKER=1` makes the
local check require Docker too.

## Setup

```bash
make install # uv sync --all-extras, including Harbor for benchmark tests
make hooks
```

For the runnable CLI, see [Quick start](README.md#quick-start): `uv sync` alone
does not put `daydream` on your PATH.

The pre-commit hook lints staged Python blobs in `daydream/`, `tests/`, and
`rl/daydream_review/`. The pre-push hook verifies every pushed commit's
signature, then runs `make check`. Never bypass either hook.

### SSH signing

The push hook accepts valid SSH or GPG signatures. For SSH:

```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/<your-key>.pub
git config --global commit.gpgsign true
ssh-add ~/.ssh/<your-key>
```

Check the loaded key with `ssh-add -l`. To re-sign unsigned commits:

```bash
git rebase --exec 'git commit --amend --no-edit -S' HEAD~N
```

## Everyday commands and the required gate

Run `make check` before pushing. The [Makefile](Makefile) defines the gate,
which runs, in order:

```text
lockcheck install lint deadcode typecheck test actionlint coverage-report check-naming
```

Each is also a focused `make` target.

- `make test` runs parallel pytest with branch coverage and the configured
  coverage floor. Bare or targeted pytest runs do not measure coverage.
- `make deadcode` scans both the root project and the standalone RL package.
- `make coverage-report` checks the report produced by tests; see
  [coverage policy](docs/coverage.md) for measurement and ratcheting.
- Run `make rl-check` whenever changing `rl/daydream_review`. It has its own
  lock, lint, types, tests, and real-Claude e2e test, and is a separate CI job;
  it is deliberately outside the root gate.

## Testing policy

Every user-visible behavior needs a **real-path test** entering through
`runner.run` or the CLI with a real temporary Git worktree, filesystem, and
event loop. Stub only external network/provider seams (`Backend` /
`create_backend`), and assert observable outcomes: exit status, files,
retained or declined fixes, and transcripts. Unit tests supplement that path.
For [terminal findings](README.md#terminal-review-findings-contract), test real
Git/public outputs and inject filesystem faults at actual operations.

Work is completed and proven, or explicitly in progress. Do not substitute
smoke tests for coverage or defer required checks.

## Exemplars

Start with the non-interactive and EOF fix-gate tests in
[tests/deep_orchestrator/test_fix_gate_cleanup_and_precision.py](tests/deep_orchestrator/test_fix_gate_cleanup_and_precision.py).
[tests/test_cli.py](tests/test_cli.py) and
[tests/test_integration.py](tests/test_integration.py) also use the shared stub
backend seam from `tests/test_deep_orchestrator.py`.

## Conventions

- Ruff: 120 columns, `E F I W`, Python 3.12. Mypy checks `daydream tests`;
  handwritten stubs live in `mypy_stubs/`.
- Vendored `daydream/atif/` is lint-exempt. Follow its NOTICE and re-vendor
  policy; do not introduce local patches.
- Declare dependencies in `pyproject.toml`; update `uv.lock` with `uv lock`.
- Use Conventional Commits, explicit staging (`git add <path>`), and linear
  history (rebase rather than merge). Never use `git add -A`.
- Update stale documentation, including agent contracts in CLAUDE.md.
- PRs need a **Test Plan** and the [template checklist](.github/PULL_REQUEST_TEMPLATE.md).
- Commit and push with hooks enabled. Fix failures on the branch or report
  the blocker; never skip verification or claim unverified success.

## Where to look deeper

Architecture: [CLAUDE.md](CLAUDE.md). Extension API:
[docs/extensions.md](docs/extensions.md). Benchmarks:
[docs/benchmark.md](docs/benchmark.md). Tracing:
[docs/observability.md](docs/observability.md). Training launches:
[docs/training-launch.md](docs/training-launch.md). RL:
[rl/daydream_review/README.md](rl/daydream_review/README.md).

## Your first PR

1. Branch from an up-to-date main; install dependencies and hooks.
2. Implement the change and its real-path tests; run focused checks.
3. Run `make check`, plus `make rl-check` for RL changes.
4. Explicitly stage and create a signed Conventional Commit.
5. Push normally; the hook repeats the full gate.
6. Open a PR with the Test Plan and completed checklist.

## Where the agent guidance lives

[CLAUDE.md](CLAUDE.md) owns architecture, backend, budget, identity, and artifact
contracts. Keep it current in the same PR as any contract change.
