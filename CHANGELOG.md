# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-01

Initial release of Daydream - an automated code review and fix loop using the Claude Agent SDK.

### Added

- **cli:** Add command-line interface with `daydream` entry point and signal handling (SIGINT/SIGTERM)
- **cli:** Add `--python` and `--typescript` flags to select review type
- **cli:** Add `--review-only` flag to skip fix and test phases
- **cli:** Add `--start-at` option to resume from a specific phase
- **cli:** Add `--model` option to select Claude model
- **cli:** Add `--no-cleanup` flag for non-interactive runs
- **cli:** Add `--debug` and `--quiet` flags for output control
- **runner:** Add main orchestration via `run()` async function with `RunConfig` dataclass
- **phases:** Add four-phase workflow: review, parse feedback, fix, and test-and-heal
- **agent:** Add Claude SDK client wrapper with streaming response support
- **agent:** Add structured output support for feedback parsing
- **ui:** Add Rich-based terminal UI with Dracula theme and live-updating panels
- **ui:** Add formatted tool displays for Glob, Grep, Read, Edit, and TodoWrite tools
- **ui:** Add ASCII art phase hero banners with neon gradient styling
- **config:** Add skill mappings for Python/FastAPI and React/TypeScript reviews

[unreleased]: https://github.com/existential-birds/daydream/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/existential-birds/daydream/releases/tag/v0.1.0
