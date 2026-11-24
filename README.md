# daydream

[![DOI](https://zenodo.org/badge/1147075973.svg)](https://doi.org/10.5281/zenodo.21614348) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/existential-birds/daydream)

```text
Set up https://github.com/existential-birds/daydream for this repository.
Read its README and CLI help, install Daydream with uv and Python 3.14+,
and help me authenticate a supported backend (Claude by default, or Codex,
Pi or Antigravity). Run a report-only review with --review against the
intended base branch. Show me the findings and where the run evidence was saved.
```

Daydream reviews code changes, can fix findings and run tests, and records local
[OpenTelemetry](https://opentelemetry.io/) evidence of every agent interaction for
research toward an open-weight code-review model; benchmarks export it as
[ATIF](https://www.harborframework.com/docs/agents/trajectory-format) trajectories.
Its review quality is measured in public on AACR-Bench and ReviewBench in
[code-review-benchmarks](https://github.com/existential-birds/code-review-benchmarks).

## Quick start

Requires [uv](https://docs.astral.sh/uv/), Python 3.14+, and an authenticated
[Claude Code](https://claude.ai/code) CLI (default backend), or Codex, Pi or
[Antigravity](#antigravity-backend). Osprey cannot run a review.
PR comments also require authenticated [gh](https://cli.github.com/).

```bash
git clone https://github.com/existential-birds/daydream.git
cd daydream
uv tool install --editable .
# If daydream is missing from PATH: uv tool update-shell, then restart your shell.
```

## Review code

```bash
daydream /path/to/project                    # report only (same as --review)
daydream --comment /path/to/project          # post inline PR comments
daydream --fix /path/to/project              # review, offer fixes, test, commit and push
daydream --review --backend codex --base main /path/to/project
```

A review ends with findings; only `--fix` [selects fixing](docs/runtime.md#validation-and-repair),
and `--yes` answers its gates. One reviewer covers the whole change and challenges
every finding in a fresh context before reporting it; it needs a backend with native
host tools (Pi, Claude, Codex or Antigravity). Use `--help` for common options and `--help-all` for
configuration and advanced flags.
Reports land in `.review-output.md`; evidence for the [newest runs](docs/runtime.md#artifacts-and-recovery) in
`.daydream/runs/<id>/` (`daydream evidence verify` checks it).
Report-only local review uses the captured net base-to-final-working change,
including nonignored untracked files. PR comments exclude local edits and use a
[pinned committed subject](docs/runtime.md#captured-local-source).
Wait for finalization before reading artifacts. Incomplete reviews retain findings
and are labeled incomplete. `daydream setup` installs a self-hosted GitHub bot.

## Antigravity backend

`--backend antigravity` drives Google's `agy` CLI (1.5.15 or newer, on `PATH` and signed
in) under the [Antigravity contract](docs/runtime.md#antigravity-host-tools). The model
defaults to `agy`'s configured one, cost is unreported, and `agy` always stores the
conversation in its home. Set `DAYDREAM_ANTIGRAVITY_HOME` to run `agy` with that directory
as `HOME`: a dedicated home with none of your settings, agents, hooks or MCP servers, which
needs its own sign-in.

Reports conflict on whether consumer Google AI plans permit launching the official `agy`
binary from another tool. Daydream runs the unmodified binary and never handles its
tokens; you accept that risk when you use a consumer sign-in. An API key or Vertex
credentials avoid it.

## Research

Add `--capture-data` to retain immutable run records locally. Capture is opt-in;
records contain raw evidence. Upload requires an explicitly selected private
Hugging Face dataset and the `hub` extra.
For agents working on data or experiments: [dataset workflow](docs/runbooks/record-dataset.md).
