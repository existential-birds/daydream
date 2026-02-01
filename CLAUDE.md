# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Daydream is an automated code review and fix loop using the Claude Agent SDK. It launches review agents equipped with Beagle skills (specialized knowledge modules) to review code, parse actionable feedback, apply fixes automatically, and validate changes by running tests.

## Commands

```bash
# Install dependencies and git hooks
make install
make hooks

# Run the CLI
daydream [TARGET] [OPTIONS]

# Run as module
python -m daydream

# Examples
daydream /path/to/project --python      # Python/FastAPI review
daydream /path/to/project --typescript  # React/TypeScript review
daydream --review-only /path/to/project # Review only, skip fixes
daydream --debug /path/to/project       # Enable debug logging

# Development
make lint       # Run ruff linter
make typecheck  # Run mypy type checker
make test       # Run pytest
make check      # Run all CI checks locally
```

## Architecture

The package follows a phased execution model:

```
cli.py → runner.py → phases.py → agent.py
                  ↘ ui.py (terminal output)
```

### Module Responsibilities

- **cli.py**: Entry point, argument parsing, signal handlers (SIGINT/SIGTERM)
- **runner.py**: Main orchestration via `run()` async function, `RunConfig` dataclass
- **phases.py**: Four workflow phases:
  1. `phase_review()` - Invoke Beagle review skill, write to `.review-output.md`
  2. `phase_parse_feedback()` - Extract actionable issues as JSON
  3. `phase_fix()` - Apply fixes one-by-one
  4. `phase_test_and_heal()` - Run tests, interactive retry/fix loop
- **agent.py**: Claude SDK client wrapper, `run_agent()` streams responses, `AgentState` dataclass for consolidated state, `MissingSkillError` exception
- **ui.py**: Rich-based terminal UI with Dracula theme, live-updating panels
- **config.py**: Skill mappings, constants

### Key Patterns

- All agent interactions use `ClaudeSDKClient` from `claude-agent-sdk` with `bypassPermissions` mode
- Streaming responses are processed via async iterator over message types (AssistantMessage, UserMessage, ResultMessage)
- Tool call panels use Rich's `Live` for animated throbbers during execution
- Global state consolidated in `AgentState` dataclass (debug_log, quiet_mode, model, shutdown_requested) with `_current_client` for SDK instance

## Dependencies

- `claude-agent-sdk` - Claude Code SDK for agent interactions
- `anyio` - Async I/O abstraction (used for `anyio.run()`)
- `rich` - Terminal UI components
- `pyfiglet` - ASCII art header

## Prerequisites

Requires the Beagle plugin for Claude Code to be installed. The review skills (`beagle:review-python`, `beagle:review-frontend`) are provided by Beagle.
