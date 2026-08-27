# daydream

[![DOI](https://zenodo.org/badge/1147075973.svg)](https://doi.org/10.5281/zenodo.21614348) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/existential-birds/daydream)

Daydream is an automated code-review agent. It reviews a code change, applies fixes, and runs the test suite to validate the result. It records every agent action as a structured trajectory.

The goal of daydream is an open-weight code-review model. Daydream trains this model on the trajectory archive that it collects from its own runs. Training is a staged recipe: reward construction, then SFT cold-start, then RFT (rejection fine-tuning), then online RL. Daydream benchmarks the model against commercial code-review bots on a held-out PR replay corpus.

## Requirements

Daydream requires the following tools:

- Python 3.12.13 or newer
- [uv](https://docs.astral.sh/uv/)
- The [Claude Code](https://claude.ai/code) command line interface

The following tools are optional:

- [GitHub CLI](https://cli.github.com/) (`gh`) for PR feedback and `--comment` mode
- [Codex CLI](https://openai.com/codex) for the `codex` backend
- [Pi CLI](https://pi.dev) for the `pi` backend
- Osprey CLI for the `osprey` backend

## Quick start

Clone the repository and install the dependencies:

```bash
git clone https://github.com/existential-birds/daydream.git
cd daydream
uv sync
```


To update daydream, run the following commands:

```bash
git pull
uv sync
```

## Usage

Run `daydream /path/to/project` to review, fix, and test a project. The command `daydream review /path/to/project` performs the same action.

The default flow is the deep multi-stack pipeline. This pipeline performs the following stages:

1. Pre-scan the repository for imports and conventions.
2. Analyze the author intent.
3. Review each stack with the applicable review profile.
4. Review alternative approaches.
5. Resolve conflicts between findings.
6. Merge the findings across stacks.
7. Verify the findings.
8. Fix the identified issues.
9. Run the test suite to validate the fixes.

Use the common commands for the common tasks:

```bash
daydream /path/to/project                    # review, fix, and test
daydream --comment /path/to/project          # review, then post inline PR comments
daydream --review /path/to/project           # write a report only; no fixes or PR comments
daydream --shallow /path/to/project          # review one stack in one pass
daydream --yes /path/to/project              # apply fixes without prompting
daydream --review-profile review.toml /path/to/project   # explicit review profile
daydream -s python /path/to/project                      # force a specific stack
```

Profile precedence is: explicit `--review-profile <path>` > env `DAYDREAM_REVIEW_PROFILE` > repo-committed `file_config.review_profile` > built-in default. The `--comment` mode posts inline PR comments and exits; `--review` writes a report and exits. Neither runs the fix cycle.

The profile selects analysis settings, but backend, provider, model, reasoning effort, and safety/scoring are host-owned invariants outside the profile — they come from the host, not the profile.

Run `daydream --help` to see the common flags. Run `daydream --help-all` to see the full advanced surface.

## Audit a repository and write implementation plans

The `improve` command audits a whole repository. It verifies each candidate finding, prioritizes the findings by impact, and writes self-contained implementation plans. Every agent call uses a read-only backend profile. Daydream writes only run artifacts under `.daydream/` and advisory plans under `daydream_plans/`. It does not modify tracked source files.

```bash
daydream improve /path/to/project
daydream improve --effort deep --scope "apps/*" /path/to/project
daydream improve --focus security /path/to/project
```

### Effort tiers

`--effort` selects the audit breadth. It does not change the model or the reasoning effort.

| Tier | Audit coverage |
|------|----------------|
| `quick` | Correctness, security, tests, and tech debt. Serial, HIGH-confidence findings only. Cap near six. |
| `standard` | All eight categories. Concurrency ceiling of ten. This is the default. |
| `deep` | All eight categories. Concurrency ceiling of ten. Includes LOW-confidence investigation items. |

On a large repository the audit fans out over partition groups. A partition group is a bounded, stack-homogeneous slice of the tree. Each agent searches one group for one category. The `standard` tier audits at most eight groups per run. The `deep` tier is unbounded. The `quick` tier audits the whole repository as one group.

```toml
[tool.daydream.improve]
partition_max_files = 400
max_partition_groups = 8
```

The report names whatever a bound leaves out. The file `.daydream/improve/coverage.json` also names it. Coverage is never silently truncated.

### Focus modes

| Focus | Behavior |
|-------|----------|
| `security` | Audit only security |
| `performance` | Audit only performance |
| `tests` | Audit only test coverage |
| `branch` | Audit the merge-base diff. Label each finding as introduced or inherited. |

Use `--scope SERVICE_OR_GLOB` to restrict the audit to matching detected services.

### Plan subcommands

The `plan` subcommand runs reconnaissance and writes one plan for the supplied request. It does not run the category audit.

```bash
daydream improve plan "add rate limiting" /path/to/project
```

Each audit writes its report under `.daydream/improve/`. Durable output under `daydream_plans/` contains numbered plan files, an index, a rendered `README.md`, and `rejected.json`. Daydream honors the **Status** cell of a plan in `README.md`. That cell outranks `.index.json`.

### Publish plans as GitHub issues

A repository can opt into unattended issue publication:

```toml
[tool.daydream.improve.github]
publish_issues = true
```

When enabled, improve first writes and validates each plan locally. It then copies the plan into the corresponding GitHub issue. It creates no plan branch, commit, or push. A stable marker in each issue makes reruns idempotent. Daydream reconciles both open and closed issues before it creates any new issue. If reconciliation fails, publication stops.

Only one improve publisher may run against a repository at a time. Use a repository-scoped GitHub Actions concurrency group:

```yaml
concurrency:
  group: daydream-improve-${{ github.repository }}
  cancel-in-progress: false
```

## Training data

This section is for machine learning researchers. Daydream is a data-collection system. It turns every run into a labeled trajectory. You project these trajectories into JSONL datasets. You use these datasets in a staged training recipe.

### Trajectories

Daydream records every agent interaction as an [ATIF v1.7](https://www.harborframework.com/docs/agents/trajectory-format) trajectory. The trajectory records the review pipeline, the model output, the tool calls, the cost, and the result.

Each run writes its trajectory to `<project>/.daydream/runs/<session-id>/trajectory.json`. Parallel fan-outs write sibling trajectories to `trajectories/`. Daydream archives the complete run bundle at `~/.daydream/archive/runs/<session-id>/`. The bundle contains the trajectory, the manifest, the review output, the diff, and the evaluation analysis. An SQLite index at `~/.daydream/archive/index.db` supports cross-project querying.

### Corpus commands

The data-pipeline verbs live under the `corpus` namespace:

```bash
daydream corpus harvest                              # annotate all archived runs
daydream corpus harvest --dry-run
daydream corpus build --out /path/to/out.jsonl       # project labeled runs to JSONL
daydream corpus build --out out.jsonl --min-reward 0.5 --include-all-labels
daydream corpus build --out out.jsonl --as-of 2026-05-01T00:00:00Z  # pinned snapshot
daydream corpus label <session-id> --outcome accepted  # manual outcome override
```

The pipeline has three stages:

1. **Harvest.** Walk the archive. Write one bitemporal annotation per run. Each annotation contains a label, an intrinsic reward, and a valid-at timestamp.
2. **Label.** Override the automated outcome for a run. The manual label beats the automated label.
3. **Build.** Project the annotations into a JSONL training corpus. Add a lineage manifest.

The build stage applies a temporal-leakage guard. It prevents future data from leaking into the past. It applies C5, C8, and C9 filters. It stratifies the corpus by stack.

### Scoring

The harvest stage scores each trajectory. The intrinsic reward is a composite:

- Correctness, weight 0.6
- Grounding, weight 0.4
- A length ramp (a penalty)

The format-valid check dominates. A trajectory that fails the format check receives no reward. Daydream records a posterior-cost axis as a sibling. It is never folded into the intrinsic reward.

### Upload to a private Hugging Face dataset

Upload of trajectories to Hugging Face is opt-in. Only the operator selects the destination. It comes from two sources, highest first:

1. The `--trajectory-hub-repo` CLI flag
2. The `DAYDREAM_TRAJECTORY_HUB_REPO` environment variable

Daydream ignores a `trajectory_hub_repo` key in the target checkout file config. When unset, nothing leaves your machine.

When set, daydream uploads every run's complete archive bundle to the dataset repo as a per-run folder. The upload requires the `huggingface_hub` package and a valid `HF_TOKEN`. If either is missing, or the upload fails, the run is never aborted. Daydream emits a one-line warning and leaves the bundle un-uploaded. On the first upload daydream creates the dataset repo private. It reuses an existing repo with its current visibility.

```sh
export DAYDREAM_TRAJECTORY_HUB_REPO="existentialbirds/daydream-trajectories"
export HF_TOKEN="hf_..."   # required for upload to proceed
```

### Training roadmap

Training follows the staged recipe from the open-weight model epic (issue #86):

| Stage | What it does |
|-------|--------------|
| 0. Reward construction | Train a reward model on the accept/reject labels. Validate it offline against held-out labels before any RL run. |
| 1. SFT cold-start | Supervised fine-tuning as a warm start. bf16 LoRA at rank 64 to 128. |
| 2. RFT | Rejection-sample against the stage-0 reward. Fold the winners back in as dense supervision. |
| 3. Online RL | GRPO against the stage-0 reward. bf16 LoRA at rank 16 to 32. |

SFT is a cold-start stage only. Online RL is the center of gravity of the recipe.

The repository contains an RL recipe and a verifiers environment:

- `rl/train/rl.toml` is a GRPO recipe. It uses prime-rl 0.7.0, LoRA rank 16, and batch size 128. The train and eval sets are two separate corpus directories. Training uses the `pi` backend only.
- `rl/daydream_review_v1/` is a verifiers v1 environment. One rollout is one headless deep run. The reward combines the intrinsic composite and a non-regression metric over the test suite.

The corpus paths and the base model in these files are placeholders. The real training set is tracked as issue #164. Daydream has scaffolded and validated the pipeline, but it has not yet harvested production data. The training pipeline itself is tracked as issue #91.

### Evaluation

The evaluation framework has two arms:

- **Recall.** Compare daydream findings against a human gold baseline. Report inter-annotator agreement (Krippendorff alpha) and PR-level bootstrap confidence intervals.
- **Quality.** Track erosion and verbosity metrics. The fix-phase quality gate uses these metrics to flag degraded fixes.

Private PR benchmarks run on Harbor. See [docs/benchmark.md](docs/benchmark.md) for the benchmark runbook.

## Architecture

Daydream runs a deep multi-stack review pipeline. The pipeline runs exploration, intent analysis, alternative review, per-stack reviews, an arbiter pass, cross-stack merge, and recommendation verification. A `--shallow` mode reviews one stack in a single pass for simpler projects.

Daydream records and archives every run as an ATIF v1.7 trajectory. The `--no-archive` flag skips archival. A bitemporal corpus pipeline harvests, scores, and projects these trajectories into JSONL datasets.

The [project page](https://existentialbirds.com/projects/daydream) documents the full architectural details.

### Backends

Daydream supports four backends. Each implements the same `Backend` protocol and emits the same event stream:

| Backend | Driver |
|---------|--------|
| `claude` | In-process Claude agent SDK. This is the default. |
| `codex` | Codex CLI in a disposable read-only clone |
| `pi` | Pi CLI (Nous DeepSeek models) |
| `osprey` | Osprey CLI |

Select a backend with `--backend`. The selection order, highest first, is:

**CLI `--backend` > config-file phase override > config-file global > built-in default.**

There is no environment-variable tier. `DAYDREAM_MODEL` and `DAYDREAM_BACKEND` are not read.

### Extensions

A fork can extend daydream. A top-level `daydream_ext` package exposes a `register(registry)` function. The function can add phases, reorder flow steps, override prompts, and register stack rules. The extension API is version 6. Verify an extension with `daydream ext validate`. See [docs/extensions.md](docs/extensions.md).

## Configuration

Configuration lives in the target repository root. Daydream reads two sources and merges them per key:

1. `pyproject.toml` under `[tool.daydream]` — lower precedence
2. `.daydream.toml` at the repository root — higher precedence

The dotfile uses bare top-level keys. It wins on scalar conflicts.

```toml
# pyproject.toml  →  [tool.daydream]
[tool.daydream]
model = "claude-opus-5"     # global default across phases
backend = "claude"          # global default backend

[tool.daydream.phases.fix]  # per-phase override
backend = "codex"
model = "gpt-5.6-terra"
reasoning_effort = "medium"
```

```toml
# .daydream.toml  (top-level keys; no [tool.daydream] prefix)
model = "claude-opus-5"

[phases.fix]
backend = "codex"
```

The resolution order, highest first, is:

**CLI > config file (phase, then global) > built-in per-backend default.**

### Per-phase settings

Phase names are the flow-step config keys: `exploration`, `intent`, `wonder`, `per_stack_review`, `arbiter`, `merge`, `review`, `parse`, `fix`, `test`, `verify`, `supervise`, and more. Any name is accepted, including phases a fork defines.

### Reasoning effort

`reasoning_effort` is accepted as a global key and per phase. The accepted levels are `low`, `medium`, `high`, `xhigh`, and `max`. Each backend maps the level to its own knob:

| Backend | Knob |
|---------|------|
| `claude` | `ClaudeAgentOptions.effort` → CLI `--effort` |
| `codex` | `-c model_reasoning_effort=<level>` |
| `pi` | `--thinking <level>` |
| `osprey` | `--effort <level>` |

The resolution order, highest first, is:

**`--reasoning-effort` > config file (phase, then global) > built-in per-phase default.**

### Supervisor settings

Supervisor settings are config-file-only:

| Key | Default | Semantics |
|-----|---------|-----------|
| `supervisor` | `"off"` | Findings supervisor mode: `"off"`, `"rules"`, or `"llm"`. |
| `supervisor_deny_globs` | `[]` | Repository-relative globs shared by findings and tool rules. |
| `tool_supervisor` | `"off"` | Built-in tool policy mode: `"off"` or `"rules"`. |
| `tool_bash_deny` | `[]` | Regular expressions for Bash commands the policy vetoes. |

Configure the LLM supervisor model under `[tool.daydream.phases.supervise]`.

### Uncovered-diff-file sweep

A second-pass reviewer covers diff files that no per-stack reviewer read:

| Key | Default | Semantics |
|-----|---------|-----------|
| `uncovered_sweep` | `true` | Toggle the second pass. |
| `uncovered_sweep_max_files` | `10` | Cap on swept files per run. `0` sweeps nothing. |
| `uncovered_sweep_min_hunk_lines` | `5` | Minimum added/removed hunk lines to be sweepable. `0` removes the floor. |

### Quality gate

The fix-phase anti-degradation quality gate prevents a fix from degrading a file:

| Key | Default | Semantics |
|-----|---------|-----------|
| `quality_gate_enabled` | `true` | Toggle the gate. |
| `quality_gate_erosion_delta` | `0.05` | Per-file erosion-delta threshold. |
| `quality_gate_verbosity_delta` | `0.05` | Per-file verbosity-delta threshold. |
| `quality_gate_erosion_absolute` | `0.05` | Absolute post-fix erosion threshold. |
| `quality_gate_verbosity_absolute` | `0.05` | Absolute post-fix verbosity threshold. |

The gate is fail-open. A flagged file surfaces as a warning plus a manifest record. It never aborts a run. Daydream clamps the thresholds to finite non-negative numbers. An invalid value degrades to the named default.

### Cost pricing

When a backend does not report a USD cost directly, daydream synthesizes the cost from token counts. The resolution order, highest first, is:

**backend-reported cost > user `prices.toml` > built-in price table > `-`.**

To override the built-in prices, create `~/.daydream/prices.toml`:

```toml
# USD per 1M tokens. User entries override built-ins per model.
[prices."gpt-5.6-sol"]
input = 4.50
cached_input = 0.45
output = 27.00
```

The `DAYDREAM_PRICES_FILE` environment variable overrides that path.

## GitHub App identity

By default, GitHub reads and writes run under the identity of the `gh` CLI. To post as a bot, supply GitHub App credentials:

```bash
export DAYDREAM_APP_ID=12345
export DAYDREAM_APP_PRIVATE_KEY="$(cat daydream-bot.private-key.pem)"  # raw PEM content
```

When both variables are set, each run mints a short-lived installation access token. Daydream attributes posts to `<app-slug>[bot]`. It displays the active identity before any GitHub action.

The behavior notes are:

- Neither variable set → ambient `gh` identity.
- Only one set → abort with an error naming the missing one.
- Posting runs abort if daydream cannot determine owner/repo, or if token minting fails.
- Daydream redacts the private key and minted tokens from logs and trajectory files.

## Self-hosted review bot

Daydream can run as a self-hosted PR review bot. It runs in your own repository's GitHub Actions and posts under your own GitHub App identity. The `daydream setup` command automates most of the install: App registration, secret deposit, and a workflow PR. Clicking **Install** on the new App stays manual, because GitHub requires it.

```bash
daydream setup /path/to/repo --repo OWNER/REPO    # one-command bot setup
daydream setup /path/to/repo --verify             # read-only install audit
```

See [docs/self-hosted-bot-setup.md](docs/self-hosted-bot-setup.md) for details.

## Non-interactive mode

`--non-interactive` runs unattended. It takes each prompt's safe default. On a test failure it writes a `handoff.md` and exits non-zero. Otherwise it declines fixes and exits zero. It is orthogonal to `--yes`: `--non-interactive` controls whether daydream may block on stdin, while `--yes` pre-decides every yes/no gate as "yes". A non-TTY or CI environment auto-enables non-interactive mode.

## Output files

| Path | Description |
|------|-------------|
| `.daydream/runs/<id>/trajectory.json` | ATIF v1.7 trajectory |
| `.daydream/runs/<id>/trajectories/` | Forked sub-trajectories from parallel fan-outs |
| `.daydream/diff.patch` | Unified diff captured at run start |
| `.daydream/deep/` | Deep pipeline artifacts |
| `.daydream/exploration/` | Cached pre-scan grounding |
| `.review-output.md` | Review findings (removed with `--cleanup`) |
| `~/.daydream/archive/runs/<id>/` | Archived run: manifest, trajectory, review output, evaluation, deep artifacts |
| `~/.daydream/archive/index.db` | SQLite index for cross-project querying |

The `.daydream/exploration/` cache is reused on an exact key match. The key excludes uncommitted edits. A near-match never counts as a hit, because a stale hit would misground every review prompt. The `--shallow` and `--review` modes delete the directory. Alternating modes degrade to a cache miss, never to stale grounding.

## Development

```bash
make install
make hooks      # install git hooks
make lint       # ruff linter
make typecheck  # mypy
make test       # pytest
make actionlint # workflow YAML checks via Docker
make rl-check   # standalone RL: lockcheck + ruff + mypy + pytest
make check      # all root + workflow + RL CI checks
```

`make check` is the quality-gate portion of the installed pre-push hook (the hook
verifies commit signatures first, then delegates to it). A running Docker daemon
is required for `make actionlint` (the workflow YAML checks run the pinned
container); when no daemon is available that target is skipped with a note and
exits 0, so `make check` still succeeds without a daemon (CI always runs
actionlint).

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
