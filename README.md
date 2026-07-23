# daydream
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/existential-birds/daydream)

Daydream is a code-review agent that produces structured training data from its own runs. It reviews diffs using stack-specific [Beagle](https://github.com/existential-birds/beagle) skills, applies fixes, validates via test suite, and records every agent interaction as an [ATIF v1.7](https://www.harborframework.com/docs/agents/trajectory-format) trajectory. A bitemporal corpus pipeline then scores, labels, and projects those trajectories into JSONL datasets for SFT and RL fine-tuning.

The goal is an open-weight code-review model trained on daydream's own trajectory archive, benchmarked against commercial code-review bots on a held-out PR replay corpus.

![demo](https://github.com/user-attachments/assets/60a80645-36de-410e-afa7-7a96efef3f57)

## Quick Start

Requires Python 3.12.13+, [uv](https://docs.astral.sh/uv/), and [Claude Code](https://claude.ai/code) CLI.

```bash
git clone https://github.com/existential-birds/daydream.git
cd daydream
uv sync
```

Install the [Beagle](https://github.com/existential-birds/beagle) plugin:

```bash
claude plugin marketplace add https://github.com/existential-birds/beagle
claude plugin install beagle
```

Optional: [GitHub CLI](https://cli.github.com/) (`gh`) for PR feedback and `--comment` mode. [Codex CLI](https://openai.com/codex) for `--backend codex`. [Pi CLI](https://pi.dev) for `--backend pi` (z.ai GLM models).

### Golden paths

Two near-zero-flag entry points cover the common cases:

```bash
daydream /path/to/project            # review → fix → test (deep multi-stack)
daydream --comment /path/to/project  # review → post inline PR comments, then exit
```

`daydream /path` is the default verb; `daydream review /path` is identical. The
remaining surface is opt-in:

```bash
daydream --review /path/to/project            # write a report to terminal/markdown, no fixes
daydream --shallow /path/to/project           # single-stack review → parse → fix → test loop
daydream --yes /path/to/project               # auto-apply fixes without prompting
daydream --loop /path/to/project              # repeat review-fix-test until clean (or 5 rounds)
daydream feedback 42 --bot "<bot-login>[bot]" /path/to/project  # fix bot PR comments
```

Run `daydream --help` for the common flags and `daydream --help-all` for the full
advanced surface (`--start-at`, `--ignore-path`, `--worktree`, `--trajectory`, …).

To update: `git pull && uv sync`

## Audit a repository and write implementation plans

`daydream improve` audits the whole repository, verifies candidate findings,
prioritizes them by leverage, and writes self-contained implementation plans.
Every agent call uses a read-only backend profile. Daydream's host code writes
only run artifacts under `.daydream/` and durable advisory artifacts under
`daydream_plans/`; it does not modify tracked source files.

```bash
daydream improve /path/to/project
daydream improve --effort deep --scope "apps/*" /path/to/project
daydream improve --focus security /path/to/project
```

### Effort tiers

| Tier | Audit coverage |
|------|----------------|
| `quick` | Correctness, security, and tests; serial, HIGH-confidence findings only, capped near six |
| `standard` | All nine categories with a concurrency ceiling of ten; the default |
| `deep` | All nine categories with a concurrency ceiling of ten; includes LOW-confidence investigation items |

`--effort` selects audit *breadth* only. It does not change the model or the
reasoning effort — those are per-phase (see [Reasoning Effort](#reasoning-effort)).

On a large repository the audit fans out over *partition groups* — bounded,
stack-homogeneous slices of the tree (services where they exist, directories
elsewhere) — instead of the whole repository at once, so one agent per group per
category each search a bounded surface. `standard` audits at most eight groups
per run, `deep` is unbounded, and `quick` audits the whole repository as one
group. Tune both bounds in `[tool.daydream.improve]`:

```toml
[tool.daydream.improve]
partition_max_files = 400
max_partition_groups = 8
```

Whatever a bound leaves out is named in the report's "What was not audited"
section and in `.daydream/improve/coverage.json` — coverage is never silently
truncated.

### Focus modes

| Focus | Behavior |
|-------|----------|
| `security` | Audit only security |
| `performance` | Audit only performance |
| `tests` | Audit only test coverage |
| `branch` | Audit the merge-base diff and label findings as introduced or inherited |
| `next` | Produce grounded direction proposals as spike plans |

Use `--scope SERVICE_OR_GLOB` to restrict the audit to matching detected
services. The report names detected services that the scope did not cover.

### Plan subcommands

```bash
daydream improve plan "add rate limiting" /path/to/project
```

`plan` runs reconnaissance and writes one plan for the supplied request without
running the category audit.

Each audit writes its report and structured intermediate data under
`.daydream/improve/`. Durable output under `daydream_plans/` consists of
numbered plan files, a `README.md` plan index, and `rejected.json` for findings
that later runs should suppress. In non-interactive mode, Daydream selects the
top five or fewer vetted defect findings by leverage; direction findings are
never selected automatically.

## Architecture

Daydream runs a deep multi-stack review pipeline (exploration, intent analysis, alternative review, per-stack Beagle skill reviews, arbiter pass, cross-stack merge, recommendation verification), with a `--shallow` single-skill mode for simpler projects. Every run is recorded as an ATIF v1.7 trajectory and archived (unless `--no-archive` is passed). A bitemporal corpus pipeline harvests, scores, and projects those trajectories into JSONL datasets for SFT and RL fine-tuning.

Full architectural details about the review pipeline stages, trajectory recording format, corpus pipeline, training roadmap, and benchmarking methodology are documented on the [project page](https://existentialbirds.com/projects/daydream).

## CLI Reference

### Output Modes

| Flag | Behavior |
|------|----------|
| _(default)_ | Deep multi-stack review, fix, test loop |
| `--shallow` | Single-stack review, parse, fix, test |
| `--review` | Write report to terminal/markdown, then exit |
| `--comment` | Post inline PR comments, then exit |

### Additional Commands

```bash
daydream summarize <path>                          # print run-info markdown for a trajectory/run dir
daydream setup /path/to/repo --repo OWNER/REPO    # one-command self-hosted review bot setup
daydream setup /path/to/repo --verify             # read-only install audit (checks secrets, workflows, App)
daydream post-findings findings.json --pr 7 --head-sha <sha> --repo owner/repo  # Phase B: validate + post
daydream --review --findings-out findings.json --pr-number 7 /path  # Phase A: emit findings artifact
daydream feedback 42 --bot "<bot-login>[bot]" /path  # ingest and fix bot PR comments
daydream ext validate                              # resolve-check the daydream_ext extension registry
daydream improve /path/to/project                  # read-only audit and prioritized plans
daydream improve plan "add rate limiting" /path/to/project
```

### Corpus Commands

The data-pipeline verbs live under the `corpus` namespace:

```bash
daydream corpus harvest                              # annotate all archived runs (reward + label)
daydream corpus harvest --dry-run
daydream corpus build --out /path/to/out.jsonl       # project labeled runs to JSONL
daydream corpus build --out out.jsonl --min-reward 0.5 --include-all-labels
daydream corpus build --out out.jsonl --as-of 2026-05-01T00:00:00Z  # pinned snapshot
daydream corpus label <session_id> --outcome accepted  # manual outcome label override
```

### Common Options

```bash
daydream -s python /path/to/project           # force a specific Beagle skill
daydream --backend codex /path/to/project     # override backend (claude, codex, pi)
daydream --model claude-haiku-4-5 /path/to/project  # overrides ALL phases (beats config-file overrides)
daydream --loop 3 /path/to/project            # repeat up to 3 review-fix-test rounds
daydream --yes /path/to/project               # auto-apply fixes without prompting
```

Advanced flags (hidden from `--help`, shown by `--help-all`, all still parse):

```bash
daydream --start-at fix /path/to/project      # resume from a specific phase
daydream --trajectory /tmp/run.json /path/to/project
daydream --ignore-path vendor /path/to/project
daydream --worktree /path/to/project          # force ephemeral worktree
daydream --no-eval /path/to/project           # skip the deterministic evaluation analysis (on by default)
daydream --no-archive /path/to/project        # skip run archival
daydream --non-interactive /path/to/project   # run unattended; take every prompt's safe default
```

`--non-interactive` takes each prompt's safe default: on test failure it writes a `handoff.md` and exits non-zero instead of looping, otherwise it declines fixes and exits 0. It is orthogonal to `--yes`: `--non-interactive` controls *whether* daydream may block on stdin, while `--yes` pre-decides every yes/no gate as "yes". A non-TTY or CI environment (`CI` set) auto-enables non-interactive mode without the flag.

Per-phase model and backend overrides are no longer CLI flags. Set them in the config file (see [Configuration](#configuration)).

## Configuration

Per-phase model/backend selection and global defaults live in a config file, read from the **target repo root** at two sources, merged per-key (the dotfile wins on scalar conflicts):

- `pyproject.toml` under `[tool.daydream]` (lower precedence)
- `.daydream.toml` at the repo root, using bare top-level keys (higher precedence)

```toml
# pyproject.toml  →  [tool.daydream]
[tool.daydream]
model = "claude-opus-4-8"     # global default across phases
backend = "claude"            # global default backend

[tool.daydream.phases.fix]    # per-phase override
backend = "codex"
model = "gpt-5.6-terra"
reasoning_effort = "medium"

[tool.daydream.phases.review]
model = "claude-opus-4-8"
```

Supervisor settings are config-file-only:

| Key | Default | Semantics |
|-----|---------|-----------|
| `supervisor` | `"off"` | Findings supervisor mode: `"off"`, `"rules"`, or `"llm"`. |
| `supervisor_deny_globs` | `[]` | Repository-relative globs shared by findings and tool rules. |
| `tool_supervisor` | `"off"` | Built-in tool policy mode: `"off"` or `"rules"`. |
| `tool_bash_deny` | `[]` | Regular expressions for Bash commands the built-in policy vetoes. |

The LLM supervisor uses one batched call. Configure its model under
`[tool.daydream.phases.supervise]` (or `[phases.supervise]` in `.daydream.toml`).

```toml
# .daydream.toml  (top-level keys; no [tool.daydream] prefix)
model = "claude-opus-4-8"

[phases.fix]
backend = "codex"
```

Phase names are the flow-step config keys (`exploration`, `intent`, `wonder`,
`per_stack_review`, `arbiter`, `merge`, `review`, `parse`, `fix`, `test`, `verify`,
`pr_feedback`, `supervise`, …); any name is accepted, including phases a fork defines through the
[extension seam](docs/extensions.md), which lists the per-flow key tables.
Resolution precedence, highest first:

**CLI > config file (phase, then global) > built-in per-backend default.**

So `--model` beats a `[tool.daydream.phases.*]` override, which beats the
per-backend table in `daydream/config.py`. The same order applies to backend
selection via `--backend` / config / the `claude` fallback. (There is no
environment-variable tier. `DAYDREAM_MODEL`/`DAYDREAM_BACKEND` are not read.)

### Reasoning Effort

`reasoning_effort` is accepted as a global key and per-phase, alongside
`model`/`backend`. All three backends consume it through their own native knob:

| Backend | Knob |
|---------|------|
| `claude` | `ClaudeAgentOptions.effort` → the CLI's `--effort` |
| `codex` | `-c model_reasoning_effort=<level>` |
| `pi` | `--thinking <level>` |

The accepted levels are `low`, `medium`, `high`, `xhigh`, and `max` — the
intersection of the three drivers' vocabularies, so any level is valid for any
backend. (Codex additionally accepts `none`, and Pi `off`/`minimal`; those are
usable via config but have no built-in default.)

Resolution precedence, highest first:

**`--reasoning-effort` > config file (phase, then global) > built-in per-phase default.**

The built-in defaults come from two independently-tuned tables, so changing one
flow never moves the other.

**Review/fix pipeline** (`DEEP_PHASE_DEFAULT_EFFORT`) — Codex only. Claude and
Pi have no entry, so these phases pass no flag and each driver keeps its own
ambient default.

| Effort | Phases |
|--------|--------|
| `low` | `parse`, `exploration` |
| `medium` | `fix`, `test`, `verify`, `suppression`, `supervise`, `merge`, `intent` |
| `high` | `per_stack_review`, `review`, `wonder`, `pr_feedback` |
| `xhigh` | `arbiter` |

**Improve advisor** (`IMPROVE_PHASE_DEFAULT_EFFORT`) — all three backends,
because the flow runs unattended and nothing it produces is reviewed in the
moment.

| Effort | Phases |
|--------|--------|
| `low` | `recon` |
| `high` | `audit` |
| `xhigh` | `vet` |
| `max` | `plan_write` |

`plan_write` is pinned to `max` on every backend. It covers plan authoring
and plan repair — the phases whose output is executed
later by a weaker agent with no context beyond the plan file, so every
ambiguity left in a plan is paid for downstream.

The improve flow runs with **no wall-clock or tool-call budget**. Its turns are
long by design and a budget abort returns partial output that reads as
complete.

When no tier supplies a value the flag is not passed at all, and the backend
applies its own ambient default (for Codex, `model_reasoning_effort` from
`~/.codex/config.toml`; for Pi, the `PI_THINKING` environment variable).

For the `pi` backend, an unset daydream model leaves Pi's own configured
`defaultModel` intact. The built-in `glm-5.2` value is used only when neither
daydream nor Pi has selected a model.

Pi accepts `PI_PROVIDER`, `PI_API_KEY`, and `PI_THINKING` as compatibility
overrides. `PI_THINKING` is Pi's ambient default and is used only when no
resolved `reasoning_effort` applies; a per-phase level always outranks it. For the built-in `zai` provider, daydream passes `PI_API_KEY` to the
child process as `ZAI_API_KEY`; credentials are never placed in process
arguments. Providers without a known native credential environment variable
must be configured through Pi directly instead of `PI_API_KEY`.

Daydream schedules Pi fan-outs with a default concurrency hint of 10 for
standard and deep workflows; quick improve remains serial. Set
`DAYDREAM_PI_FANOUT_CONCURRENCY` to a positive integer to lower or raise the
Pi hint. Each workflow still applies its own ceiling, so this setting is not a
process-global Pi limit and does not affect Claude or Codex's existing hint of
four.

### Cost Pricing

When a backend does not report a USD cost directly (notably Codex), daydream
synthesizes cost from token counts using a price table. Anthropic-backed runs use
the cost the Claude SDK already supplies and typically do not pass through this path.

Per-model cost is resolved in this order, highest first:

**backend-reported `cost_usd` > user `prices.toml` > built-in price table > `-` (with footnote).**

A model present in neither the user file nor the built-in table renders `-` with
the "not in the price table" footnote rather than a fabricated cost.

To override or extend the built-in prices, create `~/.daydream/prices.toml`. The
`DAYDREAM_PRICES_FILE` environment variable overrides that path (a test seam and
power-user escape hatch). The schema is one `[prices."<model>"]` table per model;
each requires `input` and `output` (USD per 1M tokens) and accepts an optional
`cached_input` that defaults to `input`. All prices must be non-negative. Overrides
replace a built-in entry wholesale per model. There is no per-field merge.

```toml
# ~/.daydream/prices.toml: USD per 1M tokens. User entries override built-ins per-model.
[prices."gpt-5.6-sol"]
input = 4.50
cached_input = 0.45
output = 27.00

[prices."my-custom-model"]
input = 2.00
output = 8.00        # cached_input optional → defaults to input
```

## GitHub App Identity

By default, GitHub reads and writes (PR comments, feedback replies) run under
whatever identity the `gh` CLI is authenticated as. To post as a bot you own,
supply GitHub App credentials via environment variables:

```bash
export DAYDREAM_APP_ID=12345                                   # from the App settings page
export DAYDREAM_APP_PRIVATE_KEY="$(cat daydream-bot.private-key.pem)"  # raw PEM content, not a file path
```

When both are set, each run mints a short-lived installation access token
scoped to the target repository and injects it into every `gh` subprocess.
Posts are attributed to `<app-slug>[bot]`, and the active identity (bot or
human) is displayed before any GitHub action.

One-time setup: create a GitHub App (minimum permissions: Pull Requests
read/write, Contents read, Metadata read), generate a private key from the
App settings page, and install the App on the target repository's org or user.

In GitHub Actions:

```yaml
env:
  DAYDREAM_APP_ID: ${{ vars.DAYDREAM_APP_ID }}
  DAYDREAM_APP_PRIVATE_KEY: ${{ secrets.DAYDREAM_APP_PRIVATE_KEY }}
```

Behavior notes:

- Neither var set → ambient `gh` identity, exactly as before (the App identity is opt-in).
- Setting only one of the two vars aborts with an error naming the missing one.
- Posting runs (`--comment`, `--review`, `feedback`) abort if the owner/repo
  cannot be determined or token minting fails. daydream never silently falls
  back to posting under your personal identity. Non-posting runs fall back to
  the ambient identity and continue.
- The private key and minted tokens are redacted from logs and trajectory files.

## Self-hosted Review Bot

Daydream can run as a self-hosted PR review bot in your own repository's GitHub Actions, posting under your own GitHub App identity. The `daydream setup` command automates most of the install (App registration via manifest flow, secret deposit, workflow PR); clicking **Install** on the new App stays manual because GitHub requires it. See the [setup guide](docs/self-hosted-bot-setup.md) for details.

## Output Files

| Path | Description |
|------|-------------|
| `.daydream/runs/<id>/trajectory.json` | ATIF v1.7 trajectory (customize with `--trajectory`) |
| `.daydream/runs/<id>/trajectories/` | Forked sub-trajectories from parallel fan-outs |
| `.daydream/diff.patch` | Unified diff captured at run start |
| `.daydream/deep/` | Deep pipeline artifacts: intent, per-stack reviews, merged report |
| `.review-output.md` | Review findings (removed with `--cleanup`) |
| `~/.daydream/archive/runs/<id>/` | Archived run: manifest, trajectory, review output, evaluation, deep artifacts |
| `~/.daydream/archive/index.db` | SQLite index for cross-project querying |

## Development

```bash
make install
make hooks      # install git hooks
make lint       # ruff linter
make typecheck  # mypy
make test       # pytest
make check      # all CI checks
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
