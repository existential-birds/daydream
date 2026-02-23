# daydream
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/existential-birds/daydream)

Automated code review and fix loop powered by Claude and Codex.

Daydream launches review agents equipped with [Beagle](https://github.com/existential-birds/beagle) skills — specialized knowledge modules for your technology stack (FastAPI, React, Phoenix, and more). It parses actionable feedback from the review, applies fixes automatically, and validates changes by running your test suite.

![demo](https://github.com/user-attachments/assets/60a80645-36de-410e-afa7-7a96efef3f57)
## Features

- **Trust the technology**: Stack-agnostic review mode (`--ttt`) that understands your PR intent, evaluates alternatives, and generates an implementation plan
- **Stack-aware reviews**: Beagle skills load framework-specific knowledge (FastAPI patterns, React hooks, Phoenix lifecycle, etc.) as the reviewer encounters relevant code
- **Intelligent parsing**: Extracts actionable issues from review output, skipping positive observations
- **Automated fixes**: Applies fixes one-by-one with minimal changes
- **PR feedback mode**: Fetches bot review comments from a PR, fixes in parallel, and responds automatically
- **Multi-backend support**: Claude (default) or OpenAI Codex, with per-phase backend overrides
- **Parallel execution**: Up to 4 concurrent fix agents with live progress tracking
- **Test validation**: Runs your test suite and offers interactive retry/fix options on failure
- **Commit integration**: Optionally commits and pushes changes when complete

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Claude Code](https://claude.ai/code) CLI
- [Codex CLI](https://openai.com/codex) — required when using `--backend codex`
- [GitHub CLI](https://cli.github.com/) (`gh`) — required for PR feedback mode

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

## Usage

```bash
# Review a project with a specific skill
daydream --python /path/to/project
daydream --typescript /path/to/project
daydream --elixir /path/to/project

# Review only, skip fixes
daydream --review-only /path/to/project

# Use Codex backend instead of Claude
daydream --backend codex /path/to/project

# Technology-agnostic review with plan generation
daydream --ttt /path/to/project

# Fix bot review comments on a PR
daydream --pr 42 --bot "coderabbitai[bot]" /path/to/project
```

### More Examples

```bash
# Resume from a specific phase (requires existing .review-output.md)
daydream --start-at fix /path/to/project

# Mix backends per phase
daydream --backend codex --fix-backend claude /path/to/project

# Select model explicitly
daydream --model sonnet /path/to/project

# Auto-detect PR number from current branch
daydream --pr --bot "coderabbitai[bot]" /path/to/project
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `TARGET` | Target directory | Prompt interactively |
| `-s, --skill` | Review skill: `python`, `react`, or `elixir` | Prompt interactively |
| `--python` | Shorthand for `-s python` | |
| `--typescript` | Shorthand for `-s react` | |
| `--elixir` | Shorthand for `-s elixir` | |
| `--go` | Shorthand for `-s go` | |
| `-b, --backend` | Agent backend: `claude` or `codex` | `claude` |
| `--review-backend` | Override backend for the review phase | `--backend` value |
| `--fix-backend` | Override backend for the fix phase | `--backend` value |
| `--test-backend` | Override backend for the test phase | `--backend` value |
| `--model` | Model name (`opus` for Claude, `gpt-5.3-codex` for Codex) | Backend-specific |
| `--trust-the-technology, --ttt` | Technology-agnostic review: understand intent, evaluate alternatives, generate plan | |
| `--review-only` | Skip fixes, only review and parse feedback | |
| `--start-at` | Start at phase: `review`, `parse`, `fix`, or `test` | `review` |
| `--pr [NUMBER]` | PR feedback mode (auto-detects PR number if omitted) | |
| `--bot BOT_NAME` | Bot username to filter PR comments (required with `--pr`) | |
| `--loop` | Repeat review-fix-test cycle until zero issues or max iterations | |
| `--max-iterations N` | Maximum loop iterations (only meaningful with `--loop`) | `5` |
| `--debug` | Save debug log to `.review-debug-{timestamp}.log` | |
| `--cleanup` | Remove `.review-output.md` after completion | |
| `--no-cleanup` | Keep `.review-output.md` after completion | |

## How It Works

Daydream has three modes: **standard review mode** for full codebase reviews, **trust-the-technology mode** for stack-agnostic conversational reviews, and **PR feedback mode** for resolving bot review comments on pull requests. All modes support multiple backends (Claude and Codex), including per-phase overrides.

### Standard Review Mode

Executes a four-phase workflow. Use `--start-at` to resume from a specific phase (phases before the start point are skipped).

#### Phase 1: Review

Invokes the selected Beagle review skill (e.g., `beagle-python:review-python`) against your codebase. The review output is written to `.review-output.md` in the project root.

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

### Trust the Technology Mode

Activated with `--ttt`. A three-phase conversational review that works with any technology stack — no Beagle skills required.

1. **Understand intent**: Explores the git diff and commit history to build context, then presents its understanding for you to confirm or correct
2. **Evaluate alternatives**: Reviews the changes and identifies potential improvements as numbered issues
3. **Generate plan**: For your selected issues, writes an implementation plan to `.daydream/plan-{timestamp}.md`

Trust-the-technology mode is mutually exclusive with `-s/--skill` and skill shorthands (`--python`, `--typescript`, `--elixir`, `--go`), `--review-only`, `--loop`, and `--pr`.

### PR Feedback Mode

Activated with `--pr` and `--bot`. Fetches bot review comments from a GitHub PR and resolves them automatically.

1. **Fetch**: Pulls bot comments from the PR via the `fetch-pr-feedback` Beagle skill
2. **Parse**: Extracts actionable issues from the fetched feedback (reuses standard parser)
3. **Fix**: Applies fixes concurrently with up to 4 parallel agents, each tackling one issue
4. **Commit & Push**: Automatically commits and pushes all changes
5. **Respond**: Posts fix results back on the PR as comment replies

PR feedback mode is mutually exclusive with `--review-only`, `--start-at`, and skill flags (`--python`, `--typescript`, `--elixir`).

## Output Files

| File | Description |
|------|-------------|
| `.review-output.md` | Review results (removed with `--cleanup`, required for `--start-at parse/fix`) |
| `.review-debug-{timestamp}.log` | Debug log (created when `--debug` is enabled) |
| `.daydream/plan-{timestamp}.md` | Implementation plan (created by `--ttt` mode) |

## Architecture

```text
daydream/
├── cli.py       # Entry point, argument parsing, signal handling
├── runner.py    # Main orchestration (standard + PR feedback flows)
├── phases.py    # Core phases (review, parse, fix, test) + PR feedback helpers
├── agent.py     # Agent event consumer and helper functions
├── ui.py        # Neon terminal UI components (Rich-based)
├── config.py    # Configuration constants
├── prompts/     # Review system prompt templates
└── backends/    # Backend abstraction layer
    ├── __init__.py  # Backend protocol, event types, create_backend() factory
    ├── claude.py    # Claude SDK backend
    └── codex.py     # OpenAI Codex CLI backend (JSONL event stream)
```

## Dependencies

- [claude-agent-sdk](https://pypi.org/project/claude-agent-sdk/) - Claude Code SDK for agent interactions
- [anyio](https://anyio.readthedocs.io/) - Async I/O abstraction
- [rich](https://rich.readthedocs.io/) - Terminal formatting and UI components
- [pyfiglet](https://github.com/pwaller/pyfiglet) - ASCII art generation

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
