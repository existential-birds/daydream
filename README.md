# daydream

[![DOI](https://zenodo.org/badge/1147075973.svg)](https://doi.org/10.5281/zenodo.21614348) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/existential-birds/daydream)

```text
Set up https://github.com/existential-birds/daydream for this repository.
Read its README and CLI help, install Daydream with uv and Python 3.14+,
and help me authenticate a supported backend (Claude by default, or Codex,
Pi, Osprey). Run a report-only review with --review against the intended
base branch. Show me the findings and where the trajectory was saved.
```

Daydream reviews code changes, can fix findings and run tests, and records
[ATIF](https://www.harborframework.com/docs/agents/trajectory-format) trajectories
for research toward an open-weight code-review model.

## Quick start

Requires [uv](https://docs.astral.sh/uv/), Python 3.14+, and an authenticated
[Claude Code](https://claude.ai/code) CLI (default backend), or Codex, Pi, or Osprey.
PR comments also require authenticated [gh](https://cli.github.com/).

```bash
git clone https://github.com/existential-birds/daydream.git
cd daydream
uv tool install --editable .
# If daydream is missing from PATH: uv tool update-shell, then restart your shell.
```

## Review code

```bash
daydream --review /path/to/project           # report only
daydream --comment /path/to/project          # post inline PR comments
daydream /path/to/project                    # review, offer fixes, test, commit and push
daydream --review --backend codex --base main /path/to/project
```

Deep multi-stack review is the default. Use `--shallow` for a single pass,
`--help` for common options, and `--help-all` for configuration and advanced flags.
Reports land in `.review-output.md`; trajectories in `.daydream/runs/<id>/`.
Wait for finalization before reading artifacts. Incomplete reviews retain findings
and are labeled incomplete. `daydream setup` installs a self-hosted GitHub bot.

## Research

Add `--capture-data` to retain immutable run records locally. Capture is opt-in;
records contain raw evidence. Upload requires an explicitly selected private
Hugging Face dataset and the `hub` extra. The training pipeline implements
reward construction → SFT → RFT → online RL; production training has not completed.
For agents working on data or experiments: [dataset workflow](docs/runbooks/record-dataset.md),
[training](docs/training-launch.md), [benchmarks](docs/benchmark.md).
