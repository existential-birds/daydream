# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.3] - 2026-02-14

### Fixed

- **loop:** Capture diff base before commit so next iteration sees the diff ([#21](https://github.com/existential-birds/daydream/pull/21))

  `diff_base` was recorded after `phase_commit_iteration()`, making it equal to HEAD. The next iteration's incremental diff was empty, causing the reviewer to see no changes. The base is now captured before the commit, producing a meaningful pre-fix → post-fix diff.

## [0.6.2] - 2026-02-10

### Fixed

- **loop:** Use conventional commit messages for iteration commits ([#19](https://github.com/existential-birds/daydream/pull/19))

  Iteration commits now describe what was actually changed (e.g., `fix(auth): remove unused import`) instead of the generic `daydream: iteration N fixes` message, making git history self-documenting.

### Security

- **deps:** Update cryptography from 46.0.4 to 46.0.5 ([#18](https://github.com/existential-birds/daydream/pull/18))

## [0.6.1] - 2026-02-10

### Fixed

- **loop:** Use incremental diff after first iteration to enable convergence ([#16](https://github.com/existential-birds/daydream/pull/16))

  After the first loop iteration, subsequent reviews now diff only against the last iteration's commit instead of the full branch diff. This prevents the reviewer from re-flagging already-fixed issues, allowing the loop to converge to zero findings.

## [0.6.0] - 2026-02-09

### Added

- **cli:** Add `--loop` flag for continuous review-fix-test loop mode ([#14](https://github.com/existential-birds/daydream/pull/14))

  Repeats the full review → parse → fix → test cycle until zero issues are found or the iteration cap is reached. Each successful iteration auto-commits changes; failed iterations revert to the last known-good state.

- **cli:** Add `--max-iterations` flag to cap loop iterations (default: 5) ([#14](https://github.com/existential-birds/daydream/pull/14))

- **runner:** Add dirty working tree preflight check in loop mode to prevent data loss ([#14](https://github.com/existential-birds/daydream/pull/14))

- **ui:** Add iteration divider banner between loop cycles ([#14](https://github.com/existential-birds/daydream/pull/14))

### Fixed

- **agent:** Include tool result output in test pass/fail detection ([#14](https://github.com/existential-birds/daydream/pull/14))

  `detect_test_success` now receives actual pytest output from tool results instead of only agent prose, making pass/fail detection reliable.

- **phases:** Enrich test-and-heal fix prompt with truncated test output and changed file list ([#14](https://github.com/existential-birds/daydream/pull/14))

- **runner:** Clean up stale `.review-output.md` between loop iterations to prevent review contamination ([#14](https://github.com/existential-birds/daydream/pull/14))

- **runner:** Exit with code 1 when max iterations reached with unresolved issues ([#14](https://github.com/existential-birds/daydream/pull/14))

- **runner:** Exclude reverted fixes from summary fix count ([#14](https://github.com/existential-birds/daydream/pull/14))

## [0.5.0] - 2026-02-09

### Added

- **backends:** Add backend abstraction layer with `Backend` protocol and unified `AgentEvent` stream ([#12](https://github.com/existential-birds/daydream/pull/12))

  Introduces `daydream/backends/` package with `create_backend()` factory, enabling multiple AI backends behind a common interface. Phase functions now accept a `Backend` parameter and consume unified events instead of SDK-specific message types.

- **backends:** Add `CodexBackend` for OpenAI Codex CLI integration ([#12](https://github.com/existential-birds/daydream/pull/12))

  Spawns `codex exec --experimental-json` as a subprocess and parses the JSONL event stream into unified `AgentEvent` types. Supports structured output, tool use, and continuation tokens for thread resumption.

- **cli:** Add `--backend` / `-b` flag to select AI backend (`claude` or `codex`) ([#12](https://github.com/existential-birds/daydream/pull/12))

- **cli:** Add per-phase backend overrides via `--review-backend`, `--fix-backend`, and `--test-backend` flags ([#12](https://github.com/existential-birds/daydream/pull/12))

### Changed

- **cli:** Change `--model` to accept any string instead of a fixed choice list ([#12](https://github.com/existential-birds/daydream/pull/12))

  Previously restricted to `sonnet`, `opus`, `haiku`. Now accepts arbitrary model identifiers so Codex models (e.g., `gpt-5.3-codex`) work as well.

- **agent:** Simplify `agent.py` to consume unified `AgentEvent` stream, removing all Claude SDK imports ([#12](https://github.com/existential-birds/daydream/pull/12))

- **phases:** Wire continuation tokens through the test-and-heal retry loop for Codex thread resumption ([#12](https://github.com/existential-birds/daydream/pull/12))

## [0.4.0] - 2026-02-07

### Added

- **cli:** Add `--pr` flag for PR feedback mode that fetches and fixes bot review comments ([#9](https://github.com/existential-birds/daydream/pull/9))

  Run `daydream /path --pr 42 --bot coderabbitai[bot]` to fetch bot comments from a PR, apply fixes in parallel, commit, push, and respond. Omit the PR number to auto-detect from the current branch.

- **cli:** Add `--bot` flag to specify which bot's comments to process ([#9](https://github.com/existential-birds/daydream/pull/9))

- **phases:** Add parallel fix execution with up to 4 concurrent agents ([#9](https://github.com/existential-birds/daydream/pull/9))

  Feedback items are fixed concurrently using `anyio` task groups with a capacity limiter, with a live-updating panel showing progress per item.

- **phases:** Add `phase_fetch_pr_feedback()` to pull bot comments via `fetch-pr-feedback` skill ([#9](https://github.com/existential-birds/daydream/pull/9))

- **phases:** Add `phase_respond_pr_feedback()` to post fix results back on the PR ([#9](https://github.com/existential-birds/daydream/pull/9))

- **ui:** Add `ParallelFixPanel` component for tracking concurrent fix progress ([#9](https://github.com/existential-birds/daydream/pull/9))

### Changed

- **runner:** Refactor debug log handling to use `contextlib.ExitStack` for reliable cleanup ([#9](https://github.com/existential-birds/daydream/pull/9))

- **agent:** Update signal handler to support multiple concurrent clients ([#9](https://github.com/existential-birds/daydream/pull/9))

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

[unreleased]: https://github.com/existential-birds/daydream/compare/v0.6.3...HEAD
[0.6.3]: https://github.com/existential-birds/daydream/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/existential-birds/daydream/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/existential-birds/daydream/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/existential-birds/daydream/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/existential-birds/daydream/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/existential-birds/daydream/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/existential-birds/daydream/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/existential-birds/daydream/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/existential-birds/daydream/releases/tag/v0.1.0
