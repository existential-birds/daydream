# daydream

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/existential-birds/daydream)

Automated code review and fix loop powered by Claude and Codex.

Daydream launches review agents equipped with [Beagle](https://github.com/existential-birds/beagle) skills—specialized knowledge modules that use progressive disclosure to give reviewers precise understanding of your technology stack. The agent parses actionable feedback, applies fixes automatically, and validates changes by running your test suite.

![demo](https://github.com/user-attachments/assets/60a80645-36de-410e-afa7-7a96efef3f57)

## Features

- **Stack-aware reviews**: Beagle skills progressively load framework-specific knowledge (FastAPI patterns, React hooks, SwiftUI lifecycle, etc.) as the reviewer encounters relevant code
- **Intelligent parsing**: Extracts actionable issues from review output, skipping positive observations
- **Automated fixes**: Applies fixes one-by-one with minimal changes
- **PR feedback mode**: Fetch bot review comments from a PR, fix in parallel, and respond automatically
- **Multi-backend support**: Run reviews with Claude (default) or OpenAI Codex, with per-phase backend overrides
- **Parallel execution**: Up to 4 concurrent fix agents with live progress tracking
- **Test validation**: Runs your test suite and offers interactive retry/fix options on failure
- **Commit integration**: Optionally commit and push changes when complete
- **Neon terminal UI**: Retro-styled interface with Dracula theme and animated progress

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Claude Code](https://claude.ai/code) CLI
- [Beagle](https://github.com/existential-birds/beagle) plugin for Claude Code
- [Codex CLI](https://openai.com/codex) — required when using `--backend codex`
- [GitHub CLI](https://cli.github.com/) (`gh`) — required for PR feedback mode

Install Beagle before using daydream:

```bash
claude plugin marketplace add https://github.com/existential-birds/beagle
claude plugin install beagle
```

Verify by running `/beagle:` in Claude Code—you should see the command list.

## Installation

```bash
git clone https://github.com/existential-birds/daydream.git
cd daydream
uv sync
```

## Usage

```bash
# Interactive mode - prompts for target directory and skill
daydream

# Specify target directory
daydream /path/to/project

# Use Python/FastAPI review skill
daydream --python /path/to/project
daydream -s python /path/to/project

# Use React/TypeScript review skill
daydream --typescript /path/to/project
daydream -s react /path/to/project

# Select model (default: backend-specific)
daydream --model sonnet /path/to/project

# Use Codex backend
daydream --backend codex /path/to/project
daydream -b codex --model gpt-5.3-codex /path/to/project

# Mix backends per phase
daydream --backend codex --fix-backend claude /path/to/project

# Review only (no fixes applied)
daydream --review-only /path/to/project

# Resume from a specific phase (requires existing .review-output.md)
daydream --start-at parse /path/to/project   # Skip review, start at parsing
daydream --start-at fix /path/to/project     # Skip review and parse, apply fixes
daydream --start-at test /path/to/project    # Run tests only

# Enable debug logging
daydream --debug /path/to/project

# Control cleanup of review output
daydream --cleanup /path/to/project      # Remove .review-output.md after completion
daydream --no-cleanup /path/to/project   # Keep .review-output.md (useful for CI)

# PR feedback mode - fetch and fix bot review comments
daydream --pr 42 --bot "coderabbitai[bot]" /path/to/project
daydream --pr --bot "coderabbitai[bot]" /path/to/project   # Auto-detect PR from branch
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `TARGET` | Target directory (default: prompt interactively) |
| `-s, --skill` | Review skill: `python`, `react`, or `elixir` |
| `--python` | Shorthand for `-s python` |
| `--typescript` | Shorthand for `-s react` |
| `--elixir` | Shorthand for `-s elixir` |
| `-b, --backend` | Agent backend: `claude` (default) or `codex` |
| `--review-backend` | Override backend for the review phase |
| `--fix-backend` | Override backend for the fix phase |
| `--test-backend` | Override backend for the test phase |
| `--model` | Model name (default: backend-specific — `opus` for Claude, `gpt-5.3-codex` for Codex) |
| `--review-only` | Skip fixes, only review and parse feedback |
| `--start-at` | Start at phase: `review`, `parse`, `fix`, or `test` (default: `review`) |
| `--pr [NUMBER]` | PR feedback mode: fetch and fix bot review comments (auto-detects PR if omitted) |
| `--bot BOT_NAME` | Bot username to filter PR comments (required with `--pr`) |
| `--debug` | Save debug log to `.review-debug-{timestamp}.log` |
| `--cleanup` | Remove `.review-output.md` after completion |
| `--no-cleanup` | Keep `.review-output.md` after completion |

## How It Works

Daydream has two modes: **standard review mode** for full codebase reviews, and **PR feedback mode** for resolving bot review comments on pull requests. Both modes support multiple backends. Use `--backend codex` to run with OpenAI Codex instead of Claude, or mix backends per phase with `--review-backend`, `--fix-backend`, and `--test-backend`.

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
2. Prompts Claude to apply the minimal fix needed
3. Shows progress as fixes are applied

**Note:** `--start-at fix` requires an existing `.review-output.md` file.

#### Phase 4: Test and Heal

Runs your project's test suite. On failure, offers interactive options:

- **Retry tests**: Run again without fixes
- **Fix and retry**: Launch Claude to analyze and fix failures
- **Ignore**: Mark as passed and continue
- **Abort**: Exit with failure status

After tests pass, optionally commit and push changes.

**Note:** `--start-at test` skips all other phases and runs tests directly.

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
