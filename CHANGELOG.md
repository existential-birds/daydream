# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-02-07

### Added

- **cli:** Add `--elixir` flag and `-s elixir` option for Elixir/Phoenix code reviews ([#7](https://github.com/existential-birds/daydream/pull/7))

  Wires the `beagle-elixir:review-elixir` skill into daydream, covering OTP patterns, LiveView, ExUnit, security, and performance reviews.

### Changed

- **Breaking:** Rename `-s frontend` to `-s react` for the React/TypeScript review skill ([#7](https://github.com/existential-birds/daydream/pull/7))

  **Migration:** Replace `-s frontend` with `-s react` in any scripts or CI configurations. The `--typescript` flag continues to work unchanged.

## [0.2.0] - 2026-02-06

### Added

- **prompts:** Add structured review system prompt module for RLM-based code review ([#5](https://github.com/existential-birds/daydream/pull/5))

  New `daydream.prompts` package with `build_review_system_prompt()`, `build_pr_review_prompt()`, and `get_review_prompt()` for generating context-aware review prompts with codebase metadata, sub-LLM orchestration patterns, and batched analysis strategies.

### Fixed

- **config:** Update Beagle skill names from monolithic format to v2.0 per-plugin format ([#5](https://github.com/existential-birds/daydream/pull/5))

  Skill references now use the correct `beagle-python:review-python`, `beagle-react:review-frontend`, and `beagle-core:commit-push` names. The previous `beagle:review-python` format stopped working after the Beagle v2.0 plugin split.

- **runner:** Fix missing skill detection for new Beagle plugin name format ([#5](https://github.com/existential-birds/daydream/pull/5))

## [0.1.0] - 2026-02-01

Initial release of Daydream - an automated code review and fix loop using the Claude Agent SDK.

### Added

#### Core Package
- Package structure with `pyproject.toml` and `daydream` entry point
- Four-phase workflow: review → parse feedback → fix → test-and-heal
- Claude Agent SDK integration with streaming response support
- Structured output support for feedback parsing

#### CLI
- `daydream` command with target directory argument
- `--python` flag for Python/FastAPI code reviews
- `--typescript` flag for React/TypeScript code reviews
- `--review-only` flag to skip fix and test phases
- `--start-at` option to resume from a specific phase (review, parse, fix, test)
- `--model` option to select Claude model (sonnet, opus, haiku)
- `--no-cleanup` flag for non-interactive runs
- `--debug` flag to enable debug logging
- `--quiet` flag to suppress non-essential output
- Signal handling for graceful shutdown (SIGINT/SIGTERM)

#### UI
- Rich-based terminal UI with Dracula theme
- Live-updating panels with animated throbbers during tool execution
- Formatted tool displays for Glob, Grep, Read, Edit, and TodoWrite
- ASCII art phase hero banners with neon gradient styling
- Quiet mode for minimal output

#### Configuration
- Skill mappings for `beagle:review-python` and `beagle:review-frontend`
- Review output written to `.review-output.md`

#### Development
- GitHub Actions CI workflow with lint, typecheck, and test jobs
- Pre-push hook matching CI checks
- Makefile with `install`, `hooks`, `lint`, `typecheck`, `test`, and `check` targets
- Integration test for full review-fix-test flow
- Demo script for Python/FastAPI reviewer testing

### Fixed

- Force truecolor in Rich Console for consistent CI behavior

### Dependencies

- `claude-agent-sdk` - Claude Code SDK for agent interactions
- `anyio` - Async I/O abstraction
- `rich` - Terminal UI components
- `pyfiglet` - ASCII art header generation

[unreleased]: https://github.com/existential-birds/daydream/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/existential-birds/daydream/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/existential-birds/daydream/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/existential-birds/daydream/releases/tag/v0.1.0
