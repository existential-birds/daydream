# daydream
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/existential-birds/daydream)

Automated code review and fix loop powered by Claude and Codex.

Daydream launches review agents equipped with [Beagle](https://github.com/existential-birds/beagle) skills — specialized knowledge modules for your technology stack (FastAPI, React, Phoenix, and more). It parses actionable feedback from the review, applies fixes automatically, and validates changes by running your test suite.

![demo](https://github.com/user-attachments/assets/60a80645-36de-410e-afa7-7a96efef3f57)
## Features

- **Deep review by default**: Multi-stack parallel pipeline that combines stack-agnostic intent analysis with per-stack Beagle reviews and cross-stack merge
- **Single-stack mode**: Use `--shallow` for a single Beagle skill review-fix-test loop
- **Comment mode**: `--comment` posts inline review findings on the open PR; `--review` writes a report to terminal/markdown
- **Stack-aware reviews**: Beagle skills load framework-specific knowledge (FastAPI patterns, React hooks, Phoenix lifecycle, etc.) as the reviewer encounters relevant code
- **Codebase exploration**: Tree-sitter-powered pre-scan resolves imports and detects conventions to ground reviews in actual codebase context
- **Intelligent parsing**: Extracts actionable issues from review output, skipping positive observations
- **Automated fixes**: Applies fixes one-by-one with minimal changes
- **PR feedback subcommand**: `daydream feedback <pr#>` fetches bot review comments, fixes them sequentially, and responds automatically
- **Multi-backend support**: Claude (default) or OpenAI Codex, with per-phase backend overrides
- **Parallel execution**: Up to 4 concurrent fix agents with live progress tracking
- **Test validation**: Runs your test suite and offers interactive retry/fix options on failure
- **Commit integration**: Optionally commits and pushes changes when complete
- **ATIF v1.6 trajectory recording**: Every run produces a machine-parseable trajectory with automatic secret redaction
- **Run archive**: Automatic archival to `~/.daydream/archive/` with SQLite index for cross-project querying
- **Post-run evaluation**: Deterministic analysis of cost, grounding rate, file coverage, and finding quality
- **Outcome labeling**: Tag archived runs as accepted/rejected/mixed for SFT/RL training datasets

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Claude Code](https://claude.ai/code) CLI
- [Codex CLI](https://openai.com/codex) — required when using `--backend codex`
- [GitHub CLI](https://cli.github.com/) (`gh`) — required for the `feedback` subcommand and `--comment` mode

### Install Beagle

Daydream requires the [Beagle](https://github.com/existential-birds/beagle) plugin for Claude Code.

```bash
claude plugin marketplace add https://github.com/existential-birds/beagle
claude plugin install beagle
```

Verify by running `/beagle:` in Claude Code — you should see the command list.

## Installation

```bash
git clone https://github.com/existential-birds/daydream.git
cd daydream
uv sync
```

## Updating

```bash
cd daydream
git pull
uv sync
```

## Usage

```bash
# Default: deep multi-stack review-fix-test loop
daydream /path/to/project

# Single-stack loop (skip multi-stack auto-detection)
daydream --shallow /path/to/project

# Force a specific Beagle skill instead of auto-detecting
daydream -s python /path/to/project

# Review and post inline PR comments, then exit (no fix, no test)
daydream --comment --branch feat/my-feature /path/to/project

# Review and write a report to terminal/markdown, then exit
daydream --review /path/to/project

# Use Codex backend instead of Claude
daydream --backend codex /path/to/project

# Fix bot review comments on a PR
daydream feedback 42 --bot "coderabbitai[bot]" /path/to/project

# Label an archived run for training datasets
daydream label abc12345 --accepted
```

### More Examples

```bash
# Resume from a specific phase
daydream --shallow --start-at fix /path/to/project

# Resume deep review from the per-stack stage
daydream --start-at per-stack /path/to/project

# Mix backends per phase
daydream --backend codex --fix-backend claude /path/to/project

# Override models per phase (defaults come from the per-backend table)
daydream --review-model claude-opus-4-6 --parse-model claude-haiku-4-5 /path/to/project

# Force ephemeral worktree even without --branch
daydream --worktree /path/to/project

# Generate an implementation plan and embed it in PR comments
daydream --comment --plan --branch feat/my-feature /path/to/project

# Exclude paths from the diff
daydream --ignore-path .planning --ignore-path vendor /path/to/project

# Run with evaluation analysis, skip archival
daydream --eval --no-archive /path/to/project

# Write trajectory to a custom path
daydream --trajectory /tmp/run.json /path/to/project
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `TARGET` | Target directory | Prompt interactively |
| `-s, --skill` | Force review skill: `python`, `react`, `elixir`, `go`, `rust`, `ios` | Auto-detect from changed files |
| `-b, --backend` | Agent backend: `claude` or `codex` | `claude` |
| `--review-backend` | Override backend for the review phase | `--backend` value |
| `--fix-backend` | Override backend for the fix phase | `--backend` value |
| `--test-backend` | Override backend for the test phase | `--backend` value |
| `--review-model` | Override model for the REVIEW phase (default: per-backend table; see README). | Per-backend table (see below) |
| `--parse-model` | Override model for the PARSE phase (default: per-backend table; see README). | Per-backend table (see below) |
| `--fix-model` | Override model for the FIX phase (default: per-backend table; see README). | Per-backend table (see below) |
| `--test-model` | Override model for the TEST phase (default: per-backend table; see README). | Per-backend table (see below) |
| `--exploration-model` | Model for exploration subagents (default: claude-sonnet-4-6). Use a smaller model to save cost. | Per-backend table (see below) |
| `--comment` | Review and post inline PR comments, then exit | |
| `--review` | Review and write a report to terminal/markdown, then exit | |
| `--shallow` | Single-stack review (skip multi-stack auto-detection) | |
| `--branch BRANCH` | Branch to review | cwd's local HEAD |
| `--base BASE` | Base ref to compare against | PR base if any, else `origin/HEAD` |
| `--worktree` | Force ephemeral worktree even when `--branch` is omitted | |
| `--copy PATH` | Extra path to copy into ephemeral worktree (repeatable) | |
| `--plan` | Generate implementation plan and embed in PR comments (with `--comment`) | |
| `--start-at` | Start at phase: `review`, `parse`\*, `fix`, `test`\*, `ttt`†, `per-stack`†, `merge`† | `review` |
| `--loop` | Repeat review-fix-test cycle until zero issues or max iterations | |
| `--max-iterations N` | Maximum loop iterations (only meaningful with `--loop`) | `5` |
| `--trajectory <path>` | Write trajectory to custom path | `<target>/.daydream/runs/<id>/trajectory.json` |
| `--no-archive` | Disable automatic archival to `~/.daydream/archive/` | |
| `--eval` | Run deterministic evaluation analysis and store in archive | |
| `--ignore-path PATH` | Exclude path from diff (repeatable) | |
| `--cleanup` | Remove `.review-output.md` after completion | |
| `--no-cleanup` | Keep `.review-output.md` after completion | |

\* `parse` and `test` are valid only with `--shallow`.
† `ttt`, `per-stack`, `merge` are valid only in deep (default) mode.

### Per-phase model defaults

Each phase resolves its model in this order: explicit per-phase flag → per-backend phase default (below) → backend default. Phases without an override flag (WONDER, ENVISION, MERGE, INTENT, PR_FEEDBACK) still resolve through the per-backend table — they get phase-appropriate models without a knob.

**Claude backend:**

| Phase | Default model |
|-------|---------------|
| `parse` | `claude-haiku-4-5` |
| `fix` | `claude-sonnet-4-6` |
| `test` | `claude-sonnet-4-6` |
| `exploration` | `claude-sonnet-4-6` |
| `review` | `claude-opus-4-6` |
| `wonder` | `claude-opus-4-6` |
| `envision` | `claude-opus-4-6` |
| `merge` | `claude-opus-4-6` |
| `intent` | `claude-opus-4-6` |
| `pr_feedback` | `claude-opus-4-6` |

**Codex backend:**

| Phase | Default model |
|-------|---------------|
| `parse` | `gpt-5.5` |
| `fix` | `gpt-5.5` |
| `test` | `gpt-5.5` |
| `exploration` | `gpt-5.5` |
| `review` | `gpt-5.5` |
| `wonder` | `gpt-5.5` |
| `envision` | `gpt-5.5` |
| `merge` | `gpt-5.5` |
| `intent` | `gpt-5.5` |
| `pr_feedback` | `gpt-5.5` |

The codex table uses `gpt-5.5` for every phase in v1; per-phase tiering for codex is deferred to a future release once concrete model picks across the codex lineup are settled.

### Subcommands

#### `daydream feedback`

Fetch bot review comments from a GitHub PR, fix them, and respond.

```bash
daydream feedback <pr#> --bot <BOT_NAME> [TARGET]
```

| Argument | Description |
|----------|-------------|
| `pr#` | Pull request number to process |
| `--bot BOT_NAME` | Bot username to filter PR comments (e.g. `coderabbitai[bot]`) |
| `TARGET` | Target directory (default: current directory) |

#### `daydream label`

Update outcome labels on an archived run for SFT/RL training datasets.

```bash
daydream label <session_id> --accepted|--rejected|--mixed
```

| Argument | Description |
|----------|-------------|
| `session_id` | Session ID (full UUID or prefix) |
| `--accepted` | Label run as accepted |
| `--rejected` | Label run as rejected |
| `--mixed` | Label run as mixed |

## How It Works

Daydream has four output modes selected by flag combinations:

- **Default loop** (no flag): full review → fix → test cycle
- **`--comment`**: review and post inline PR comments, then exit
- **`--review`**: review and write a report, then exit
- **`daydream feedback <pr#>`**: ingest bot PR comments and fix them

Within the loop mode, `--shallow` opts into a single-stack review-fix-test pipeline; the default is the deep multi-stack pipeline. All modes support multiple backends (Claude and Codex), including per-phase overrides.

### Deep Review (default)

A five-stage pipeline that combines stack-agnostic intent analysis with parallel per-stack Beagle reviews and a cross-stack merge. Use `--start-at` to resume from a specific stage.

1. **Exploration pre-scan**: Tree-sitter import resolution and convention detection (auto-skipped for trivial diffs)
2. **Intent (`ttt`)**: Understands the git diff and commit history to build context
3. **Alternative review**: Identifies potential improvements as numbered issues
4. **Per-stack reviews**: Parallel Beagle skill invocations, one per detected stack (Python, TypeScript, Go, etc.)
5. **Cross-stack merge**: Synthesizes per-stack findings into a unified review with deduplication

After the merge, an optional fix gate offers test/commit/push phases.

### Shallow Review (`--shallow`)

A single-skill four-phase workflow. Use `--start-at` to resume from a specific phase.

#### Phase 1: Review

Invokes the resolved Beagle review skill (e.g., `beagle-python:review-python`). The review output is written to `.review-output.md` in the project root.

#### Phase 2: Parse Feedback

Extracts actionable issues from the review output as structured JSON. Positive observations and summary sections are filtered out.

**Note:** `--start-at parse` requires an existing `.review-output.md` file.

#### Phase 3: Apply Fixes

For each actionable issue:

1. Displays the issue description, file, and line number
2. Prompts the agent to apply the minimal fix needed
3. Shows progress as fixes are applied

**Note:** `--start-at fix` requires an existing `.review-output.md` file.

#### Phase 4: Test and Heal

Runs your project's test suite. On failure, offers interactive options:

- **Retry tests**: Run again without fixes
- **Fix and retry**: Launch the agent to analyze and fix failures
- **Ignore**: Mark as passed and continue
- **Abort**: Exit with failure status

After tests pass, optionally commit and push changes.

**Note:** `--start-at test` skips all other phases and runs tests directly.

### Comment / Review Modes

`--comment` and `--review` run a three-phase conversational review:

1. **Understand intent**: Explores the git diff and commit history to build context
2. **Evaluate alternatives**: Reviews the changes and identifies potential improvements as numbered issues
3. **Generate plan** (optional): With `--plan`, writes an implementation plan to `.daydream/plan-{timestamp}.md`

`--comment` then posts the issues as inline review comments on the open PR for the current branch (use `--branch` to point at a specific branch). `--review` writes the report to terminal/markdown and exits without posting.

### PR Feedback Subcommand

`daydream feedback <pr#> --bot <name>` resolves bot review comments on a PR:

1. **Fetch**: Pulls bot comments from the PR via the `fetch-pr-feedback` Beagle skill
2. **Parse**: Extracts actionable issues (reuses the shallow-mode parser)
3. **Fix**: Applies fixes sequentially, one issue at a time
4. **Commit & Push**: Automatically commits and pushes all changes
5. **Respond**: Posts fix results back on the PR as comment replies

## Output Files

| File | Description |
|------|-------------|
| `.review-output.md` | Review results (removed with `--cleanup`, required for `--start-at parse/fix`) |
| `.daydream/runs/<id>/trajectory.json` | ATIF v1.6 trajectory (customize path with `--trajectory`) |
| `.daydream/trajectories/` | Forked sub-trajectories from parallel fan-outs (fix-parallel, deep, exploration) |
| `.daydream/diff.patch` | Unified git diff captured at run start |
| `.daydream/plan-{timestamp}.md` | Implementation plan (created by `--comment --plan` and `--review`) |
| `.daydream/deep/` | Deep pipeline artifacts: intent, per-stack reviews, merged report (default mode) |

### Archive

Unless `--no-archive` is passed, each run is automatically archived to `~/.daydream/archive/runs/{session_id}/` with:

| File | Description |
|------|-------------|
| `manifest.json` | Run metadata, git context, backend config, token/cost metrics |
| `trajectory.json` | Copy of the run trajectory |
| `review-output.md` | Copy of review findings |
| `evaluation.json` | Deterministic evaluation results (only with `--eval`) |
| `deep/` | Deep artifacts copy (default mode) |
| `diff.patch` | Diff copy |

A SQLite index at `~/.daydream/archive/index.db` enables cross-project querying by repo, backend, cost, grounding rate, and outcome labels.

## Trajectory Output

Every daydream run produces an [ATIF v1.6](https://www.harborframework.com/docs/agents/trajectory-format) trajectory file at `<target>/.daydream/runs/<id>/trajectory.json`. The trajectory captures the full agent interaction history — prompts, responses, tool calls, observations, and per-step token/cost metrics. Use `--trajectory <path>` to write to a custom location.

Sensitive content is automatically redacted before writing: API keys (`sk-*`, `ghp_*`, `xoxb-*`, `AKIA*`), JWT tokens, URL credentials, username segments in file paths, and `.env`-style secret values are replaced with type-specific `[REDACTED_*]` tokens. Interrupted runs (SIGINT/SIGTERM) flush a `<path>.partial` file with `extra.partial=true` so consumers can detect incomplete trajectories.

**Consumer integration:**
- Validate trajectories with [Harbor](https://github.com/laude-institute/harbor)'s trajectory validator
- Replay in any ATIF-compatible viewer
- Use as training data for SFT/RL pipelines (trajectories are machine-parseable by design)
- Label outcomes with `daydream label` for supervised fine-tuning datasets



## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
