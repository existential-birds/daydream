# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.14.0] - 2026-04-28

### Breaking

- **cli:** Remove `--debug` flag; use `--trajectory <path>` to control trajectory output location. Daydream no longer produces `.review-debug-{timestamp}.log` files.

### Added

- **trajectory:** Every run now produces an [ATIF v1.6](https://www.harborframework.com/docs/agents/trajectory-format) trajectory file at `<target>/.daydream/trajectory.json` capturing the full agent interaction history, tool I/O, and per-step token/cost metrics.
- **cli:** Add `--trajectory <path>` flag to write trajectories to a custom location. Trajectories are always written; the flag only controls the output path.
- **redaction:** Automatic secret redaction applied to all trajectory content — API keys, JWT tokens, file paths with usernames, and `.env`-style secret values are replaced with `[REDACTED_*]` tokens before writing.
- **trajectory:** Parallel fan-out flows (fix-parallel, deep-mode per-stack, exploration specialists) produce sibling trajectory files linked from the root via `subagent_trajectory_ref`.
- **trajectory:** SIGINT/SIGTERM mid-run flushes a partial trajectory to `<path>.partial`.

### Removed

- **agent:** Remove `_log_debug()` debug logging system and all prefix-tagged log lines (`[TEXT]`, `[TOOL_USE]`, `[COST]`, etc.). Trajectory recording replaces all debug observability.
- **agent:** Remove `AgentState.debug_log` field and `set_debug_log()`/`get_debug_log()` accessors.
- **runner:** Remove `.review-debug-{timestamp}.log` file initialization.

## [0.13.1] - 2026-04-26

### Changed

- **pr-review:** Restyle inline GitHub PR reviews with severity emoji prefixes, per-file collapsible `<details>` sections, and 🔮 AI agent prompts under a "Code Review Summary" header ([#52](https://github.com/existential-birds/daydream/pull/52))

  Aligns the posted review with CodeRabbit-style formatting so findings are easier to skim and the consolidated agent prompt is more actionable.

- **pr-review:** Replace the static "here are all findings" AI agent prompt with a fetch-and-fix workflow ([#52](https://github.com/existential-birds/daydream/pull/52))

  The consolidated prompt now instructs the PR author's agent to fetch the latest review comments via `/beagle-core:fetch-pr-feedback` (or `gh api`), verify each against the current code, and only fix valid ones — narrowing to the most recent review instead of all historical comments.

- **deep:** Cap deep-review exploration specialists at 15 turns ([#50](https://github.com/existential-birds/daydream/pull/50))

  Threads a new `max_turns` parameter through the `Backend` protocol, `ClaudeBackend`, `CodexBackend`, and `run_agent()` so callers can bound agent turn count. Prevents context blowup on large repos during the pre-scan stage.

### Fixed

- **deep:** Stop target-repo `.claude/settings.json` from blocking agent file writes ([#52](https://github.com/existential-birds/daydream/pull/52))

  Daydream agents no longer inherit `setting_sources=["project", "local"]` from the target repo, so restrictive permission rules in the project under review can't deny `Write`/`Edit` calls. An explicit `allowed_tools` whitelist is added as a belt-and-suspenders measure. `CLAUDE.md` is still loaded from cwd, so reviews retain project context.

- **deep:** Write the merge report to `.daydream/deep/` to dodge sandbox dotfile blocks ([#51](https://github.com/existential-birds/daydream/pull/51))

  The merge agent now writes `.daydream/deep/review-output.md` (the same directory where per-stack agents already write successfully), and Python copies the result to the canonical `cwd/.review-output.md`. Fix-gate recovery and `--start-at fix` resume both honor the new path. Stale outputs at both locations are cleared before invoking the merge agent, and the run fails with `FileNotFoundError` if the expected report is missing.

- **pr-review:** Snap out-of-hunk inline comment lines to the nearest diff boundary ([#50](https://github.com/existential-birds/daydream/pull/50))

  GitHub was rejecting reviews with `422 "line could not be resolved"` when a finding sat 1–3 lines outside any diff hunk. `classify()` now calls `snap_to_hunk()` (with a centralized `HUNK_TOLERANCE = 3` constant) which returns the original line when inside a hunk, snaps to the nearest boundary when within tolerance, or demotes the comment to the review body. Removes dead `within_hunk` helper.

- **pr-review:** Tolerate bold-wrapped heads and multi-path brackets in the merge agent's review output ([#48](https://github.com/existential-birds/daydream/pull/48))

  When the merge agent drifted to `N. **[FILE:LINE] TITLE**` or stuffed multiple paths into one bracket (`[a.ts:1, b.go:41, c.py:48]`), the parser matched zero issues and the deep run silently skipped the PR post. The head regex now accepts an optional `**`/`__` wrapper via a conditional backref, and multi-path brackets are split on `,` and emitted as one `ParsedIssue` per file. The merge prompt was also tightened to require plain heads and one path per bracket.

- **deep:** Reduce duplicate findings and overconfident refactor recommendations ([#52](https://github.com/existential-birds/daydream/pull/52))

  Adds a record↔record dedup pre-filter so the merge agent consolidates near-identical findings across files into a single entry instead of repeating them per file. Caps refactor/extract-shared-code recommendations at MEDIUM confidence unless the reviewer verified no shared module already exists in the directory. Driven by author feedback on a recent multi-stack review where 12/14 findings were accepted but one was a verbatim duplicate and one an overconfident refactor.

- **deep:** Carry source-stack with cross-stack record dedup pairs ([#52](https://github.com/existential-birds/daydream/pull/52))

  `RecordDuplicatePair` now tracks `source_stack` so per-stack records with the same integer id (assigned independently per stack) don't collide ambiguously when combined. `build_record_dedup_candidates()` requires the `sources` list and validates its length against `records`, raising `ValueError` up front instead of crashing later with `IndexError`.

- **deep:** Stop merge citations from auto-linking to repo issues on GitHub ([#52](https://github.com/existential-birds/daydream/pull/52))

  Source-record citations like `#6` were being parsed by GitHub as links to repo issues/PRs. The merge prompt now instructs the agent to use `item N` notation instead.

- **redrive:** Use composite `(file, id)` keys when tracking consumed records in the redrive script ([#51](https://github.com/existential-birds/daydream/pull/51))

  Per-stack ids are assigned independently, so two stacks can share the same integer id. The previous bare-id key let one finding silently suppress an unrelated finding from a different stack.

### Added

- **scripts:** Add `scripts/redrive_post.py` for re-driving PR comment posts from existing `.daydream/deep/` artifacts ([#51](https://github.com/existential-birds/daydream/pull/51))

  Lets you reattempt the inline-PR-review post step against a prior deep run's artifacts when the original post failed (e.g. transient GitHub API error) without re-running the full pipeline.

## [0.13.0] - 2026-04-19

### Added

- **cli:** Add `--deep` mode for multi-stack code review with inline PR comments ([#45](https://github.com/existential-birds/daydream/pull/45))

  A 5-stage pipeline (exploration → TTT intent → TTT alternatives → per-stack fan-out → cross-stack merge) with an optional fix gate that auto-detects the stacks touched by the diff, fans out per-stack reviews in parallel via the matching Beagle skills, merges findings with dedup, and posts the result as a single atomic inline GitHub PR review. Handles mixed-stack PRs (e.g. Python + React) that existing single-stack modes can't review cleanly. Falls back to generic review when a per-stack skill is unavailable.

- **cli:** Add `--start-at {ttt,per-stack,merge,fix}` for stage-granular resume of `--deep` runs ([#45](https://github.com/existential-birds/daydream/pull/45))

  `.daydream/deep/` artifacts are preserved across runs so an interrupted pipeline can resume from a later stage without re-running earlier work. Each resume target enforces an artifact precondition and fails with an actionable error naming the missing file.

- **pr-review:** Post inline GitHub PR comments from `--ttt` and `--deep` ([#45](https://github.com/existential-birds/daydream/pull/45))

  Anchor-greps each finding to a real head-SHA line, classifies against diff hunks, and posts a single atomic review via the GitHub API. Cross-stack and off-hunk findings fold into the review body with severity (high/medium/low) and confidence (HIGH/MEDIUM/LOW) breakdowns. y/n gated; non-fatal on failure; payload preserved for retry.

### Changed

- **phases:** `phase_parse_feedback` accepts a keyword-only `input_path: Path | None` parameter ([#45](https://github.com/existential-birds/daydream/pull/45))

  Default `None` preserves the existing cwd/`REVIEW_OUTPUT_FILE` behavior for all existing callers. Explicit paths let the per-stack deep-review fan-out parse multiple review files in parallel without colliding.

- **runner:** Derive skill availability from the Claude Code plugin registry at runtime ([#45](https://github.com/existential-birds/daydream/pull/45))

  Reads `$CLAUDE_CONFIG_DIR/plugins/installed_plugins.json` to check whether a `beagle-<stack>` plugin is installed. When absent, deep mode routes that stack to the generic fallback review instead of letting the call silently fail with a swallowed `MissingSkillError`.

## [0.12.0] - 2026-04-17

### Added

- **cli:** Add repeatable `--ignore-path PATH` flag to exclude directories from review ([#42](https://github.com/existential-birds/daydream/pull/42))

  Injects git `:(exclude)` pathspecs into diff collection and instructs review-phase agents to apply the same filter. Useful for excluding `.planning/`, `vendor/`, or generated directories in monorepos so diff noise doesn't drown out real review signal.

### Fixed

- **exploration:** Stop embedding full diff text in specialist subagent prompts ([#42](https://github.com/existential-birds/daydream/pull/42))

  Pattern-scanner, dependency-tracer, and test-mapper subagents now receive affected file paths plus a diff ref and fetch per-file diffs on demand via their existing tools. Fixes "Prompt is too long" failures on monorepo-sized diffs (15k+ lines) by dropping token cost from O(total_diff) to O(per-file lookups).
- **agent:** Correct `detect_test_success()` false negatives on clean-pass outputs ([#42](https://github.com/existential-birds/daydream/pull/42))

  The matcher now extracts structured counts first and falls through to sentinel phrases, handling cases the previous regex missed: "N tests passed" / "0 tests failed" on separate lines, the word "tests" appearing between the count and "failed", and Cargo's native `test result: ok. N passed; 0 failed;` summary. Stops the heal loop from retrying already-passing test runs.

### Security

- **deps:** Bump pyjwt 2.11.0 → 2.12.1 for CVE fix (accepts unknown `crit` header extensions — high) ([#42](https://github.com/existential-birds/daydream/pull/42))
- **deps:** Bump python-multipart 0.0.22 → 0.0.26 for DoS-via-large-preamble CVE fix ([#41](https://github.com/existential-birds/daydream/pull/41), [#42](https://github.com/existential-birds/daydream/pull/42))
- **deps:** Bump pygments 2.19.2 → 2.20.0 for ReDoS CVE in GUID regex ([#42](https://github.com/existential-birds/daydream/pull/42))

## [0.11.1] - 2026-04-13

### Fixed

- **prompts:** Add QUAL-04 error handling semantics guardrail to reduce false positives on intentional log-and-continue patterns ([#38](https://github.com/existential-birds/daydream/pull/38))

  The reviewer now distinguishes critical-path errors (which should be flagged) from best-effort/diagnostic operations (telemetry, debug traces, analytics) that intentionally log a warning and continue. Fix prompts also prevent agents from changing error handling semantics unless the issue specifically explains why the current strategy is wrong.

## [0.11.0] - 2026-04-12

### Added

- **exploration:** Add pre-scan codebase exploration for grounded reviews ([#36](https://github.com/existential-birds/daydream/pull/36))

  Before invoking the review skill, daydream now runs a tiered exploration phase that analyzes the diff, traces dependencies, scans for project conventions, and maps test coverage. The exploration context is injected into the review prompt so findings are grounded in actual codebase structure rather than the diff alone. Trivial diffs skip exploration automatically; multi-file diffs fan out to parallel specialist subagents.

### Security

- **deps:** Update cryptography from 46.0.5 to 46.0.7 ([#34](https://github.com/existential-birds/daydream/pull/34), [#35](https://github.com/existential-birds/daydream/pull/35))

## [0.10.0] - 2026-03-14

### Added

- **cli:** Add `--ios` flag and `-s ios` option for iOS/SwiftUI code review using `beagle-ios:review-ios` ([#32](https://github.com/existential-birds/daydream/pull/32))

## [0.9.0] - 2026-03-14

### Added

- **cli:** Add `--rust` flag and `-s rust` option for Rust code review using `beagle-rust:review-rust` ([#30](https://github.com/existential-birds/daydream/pull/30))
- **cli:** Add Go and Rust entries to the interactive skill selection menu ([#30](https://github.com/existential-birds/daydream/pull/30))

## [0.8.0] - 2026-03-03

### Added

- **cli:** Add `--trust-the-technology` / `--ttt` flag for alternative review mode ([#26](https://github.com/existential-birds/daydream/pull/26))

  Analyzes the git diff of the current branch, presents discovered issues in an interactive table for user selection, then generates a targeted improvement plan. Runs three phases: understand intent, alternative review, and generate plan. Designed for reviewing your own work before opening a PR.

### Fixed

- **ttt:** Distinguish base-branch detection failure from empty diff ([#26](https://github.com/existential-birds/daydream/pull/26))

  `_git_diff` now returns `None` on base-branch detection failure vs empty string for no changes, preventing false "no changes" messages when the base branch cannot be determined.

## [0.7.0] - 2026-02-21

### Added

- **cli:** Add `--go` flag for Go backend code review using `beagle-go:review-go` ([#23](https://github.com/existential-birds/daydream/pull/23))

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

[unreleased]: https://github.com/existential-birds/daydream/compare/v0.14.0...HEAD
[0.14.0]: https://github.com/existential-birds/daydream/compare/v0.13.1...v0.14.0
[0.13.1]: https://github.com/existential-birds/daydream/compare/v0.13.0...v0.13.1
[0.13.0]: https://github.com/existential-birds/daydream/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/existential-birds/daydream/compare/v0.11.1...v0.12.0
[0.11.1]: https://github.com/existential-birds/daydream/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/existential-birds/daydream/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/existential-birds/daydream/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/existential-birds/daydream/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/existential-birds/daydream/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/existential-birds/daydream/compare/v0.6.3...v0.7.0
[0.6.3]: https://github.com/existential-birds/daydream/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/existential-birds/daydream/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/existential-birds/daydream/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/existential-birds/daydream/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/existential-birds/daydream/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/existential-birds/daydream/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/existential-birds/daydream/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/existential-birds/daydream/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/existential-birds/daydream/releases/tag/v0.1.0
