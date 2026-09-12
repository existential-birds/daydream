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
