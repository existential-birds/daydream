# daydream

Automated code review and fix loop using the Claude Agent SDK.

Daydream launches review agents equipped with [Beagle](https://github.com/existential-birds/beagle) skills—specialized knowledge modules that use progressive disclosure to give reviewers precise understanding of your technology stack. The agent parses actionable feedback, applies fixes automatically, and validates changes by running your test suite.

## Features

- **Stack-aware reviews**: Beagle skills progressively load framework-specific knowledge (FastAPI patterns, React hooks, SwiftUI lifecycle, etc.) as the reviewer encounters relevant code
- **Intelligent parsing**: Extracts actionable issues from review output, skipping positive observations
- **Automated fixes**: Applies fixes one-by-one with minimal changes
- **Test validation**: Runs your test suite and offers interactive retry/fix options on failure
- **Commit integration**: Optionally commit and push changes when complete
- **Neon terminal UI**: Retro-styled interface with Dracula theme and animated progress

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Claude Code](https://claude.ai/code) CLI
- [Beagle](https://github.com/existential-birds/beagle) plugin for Claude Code

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
daydream -s frontend /path/to/project

# Review only (no fixes applied)
daydream --review-only /path/to/project

# Enable debug logging
daydream --debug /path/to/project

# Auto-cleanup review output after completion
daydream --cleanup /path/to/project
```

### Command Line Options

| Option          | Description                                      |
| --------------- | ------------------------------------------------ |
| `TARGET`        | Target directory (default: prompt interactively) |
| `-s, --skill`   | Review skill: `python` or `frontend`             |
| `--python`      | Shorthand for `-s python`                        |
| `--typescript`  | Shorthand for `-s frontend`                      |
| `--review-only` | Skip fixes, only review and parse feedback       |
| `--debug`       | Save debug log to `.review-debug-{timestamp}.log`|
| `--cleanup`     | Remove `.review-output.md` after completion      |

## How It Works

Daydream executes a four-phase workflow:

### Phase 1: Review

Invokes the selected Beagle review skill (e.g., `beagle:review-python`) against your codebase. The review output is written to `.review-output.md` in the project root.

### Phase 2: Parse Feedback

Extracts actionable issues from the review output as structured JSON. Positive observations and summary sections are filtered out.

### Phase 3: Apply Fixes

For each actionable issue:

1. Displays the issue description, file, and line number
2. Prompts Claude to apply the minimal fix needed
3. Shows progress as fixes are applied

### Phase 4: Test and Heal

Runs your project's test suite. On failure, offers interactive options:

- **Retry tests**: Run again without fixes
- **Fix and retry**: Launch Claude to analyze and fix failures
- **Ignore**: Mark as passed and continue
- **Abort**: Exit with failure status

After tests pass, optionally commit and push changes.

## Output Files

| File                             | Description                                  |
| -------------------------------- | -------------------------------------------- |
| `.review-output.md`              | Review results (cleaned up with `--cleanup`) |
| `.review-debug-{timestamp}.log`  | Debug log (when `--debug` is enabled)        |

## Architecture

```text
daydream/
├── cli.py      # Entry point, argument parsing, signal handling
├── runner.py   # Main orchestration logic and RunConfig
├── phases.py   # Phase functions (review, parse, fix, test)
├── agent.py    # Claude SDK client and helper functions
├── ui.py       # Neon terminal UI components (Rich-based)
└── config.py   # Configuration constants
```

## Dependencies

- [claude-agent-sdk](https://pypi.org/project/claude-agent-sdk/) - Claude Code SDK for agent interactions
- [anyio](https://anyio.readthedocs.io/) - Async I/O abstraction
- [rich](https://rich.readthedocs.io/) - Terminal formatting and UI components
- [pyfiglet](https://github.com/pwaller/pyfiglet) - ASCII art generation

## Contributing

This project is not accepting outside contributions. If you encounter a bug or have a feature request, please [open an issue](https://github.com/existential-birds/daydream/issues).

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
