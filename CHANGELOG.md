# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **review:** Embed verification-protocol gates inline instead of a skill-file read in the structural and generic-fallback reviewers

  These two reviewers run with cwd set to the reviewed repo, so the previous `read review-verification-protocol/SKILL.md` instruction resolved against that repo, failed ("skill doesn't exist as a file"), and silently dropped the anchor/evidence/severity gates. The gates (0–3) are now stated inline in `VERIFICATION_PROTOCOL_INSTRUCTION`, mirroring the inline gate-0 embedding already used by `build_verification_prompt`. Both reviewers are language-agnostic, so the protocol's language-specific valid-pattern tables were of little use to them; the gate discipline is what mattered and is now self-contained.

## [0.22.0] - 2026-07-02

### Added

- **extensions:** Versioned extension seam with `daydream_ext` discovery, flow engine, and `daydream ext validate` CLI ([#241](https://github.com/existential-birds/daydream/pull/241))

  Daydream now has a first-class, versioned extension API. An installed `daydream_ext` package is auto-discovered with a version gate and exposed through a `Registry` on a `ContextVar`. The registry routes skill slots (stack, structural, pr-feedback, phase-bound), named prompts (with wholesale override), and stack rules through a single seam, replacing scattered hardcoded lookups. All four flows (deep, shallow, review, pr-feedback) now run through a `run_flow()` engine with enabled gating, `Stop`/`BreakLoop` control, and loop groups. New `daydream ext validate` CLI resolve-checks the entire extension registry.

- **deep:** `--findings-out` drives the deep pipeline and stops before post/fix ([#239](https://github.com/existential-birds/daydream/pull/239))

  The single-file review-bot workflow now points at the default deep pipeline so the bot consumes the validated, arbiter-merged review. When `--findings-out` is set, the orchestrator writes the artifact from merged-items.json and returns before the PR-post block and apply-fixes gate. Also removes the vestigial `--plan`/ENVISION advisory pass and fixes `--review` to stop after findings instead of writing a stray plan file.

- **review:** Structural evidence gate drops speculative/no-evidence findings before merge ([#236](https://github.com/existential-birds/daydream/pull/236))

  Adds a first-class `evidence` field to all finding schemas. A structural gate in the merge path drops any finding without a grounded citation (path:line) before it reaches merged-items.json, review-output.md, the fix stage, PR posting, or the benchmark. Dropped items are logged to a `dropped-speculative.json` audit sidecar. Applies to both the cross-stack merge and the tiny-diff single-stack bypass. The confidence enum collapses to HIGH/MEDIUM (LOW/no-evidence removed).

- **prompts:** Eliminate speculative and breadth-padding generation in review prompts ([#238](https://github.com/existential-birds/daydream/pull/238))

  Removes language that invites evidence-free guesses and scope creep: drops the LOW/no-evidence confidence tier, replaces "Prefer LOW over MEDIUM" with a grounding directive, removes the "Would you have done this differently?" breadth invitation from alternative review, and requires concrete recommendations per issue rather than just the problem.

- **review:** Wire review-verification-protocol into structural and generic-fallback reviewers ([#235](https://github.com/existential-birds/daydream/pull/235))

  The structural and generic-fallback review prompts now load the review-verification-protocol skill (anchor/evidence/severity gates). The verification prompt adds a gate-0 anti-confabulation echo requirement. Product default is unchanged: verdicts stay advisory, zero findings dropped.

- **templates:** Optional single-file review-bot workflow (drops `actions: write`) ([#237](https://github.com/existential-birds/daydream/pull/237))

  Adds `daydream/templates/workflows/single/daydream.yml` as an optional manual-copy alternative to the three-file split. It chains gate, analyze, and post as `needs:`-ordered jobs in one workflow run, replacing the cross-workflow `workflow_dispatch` that required `actions: write`. The privilege split is preserved at the job level: gate holds the App key but no code, analyze holds PR code but no App key, post holds the App key but no PR code. Every action is pinned to a full commit SHA.

- **training:** Eval-on-by-default and recommended-change patch capture ([#234](https://github.com/existential-birds/daydream/pull/234))

  Eval is now on by default (`--eval` becomes `--no-eval` opt-out); `analyze_session` runs on every archive unless explicitly skipped, populating all four eval metrics (grounding_rate, total_findings, coverage_ratio, cost_per_finding_usd). A separate `recommended.patch` (daydream's proposed diff) is captured distinct from the PR-under-review `diff.patch`, with backward-compat fallback for legacy archives. Untracked files created during the fix phase are included in the capture.

## [0.21.0] - 2026-06-30

### Added

- **backends:** Add pi coding-agent backend with ATIF trajectory parity ([#195](https://github.com/existential-birds/daydream/pull/195))

  Daydream now supports z.ai GLM models through a new `PiBackend` that spawns the `pi` CLI as a subprocess and parses its JSONL event stream into unified `AgentEvent` types. Implements the same subprocess+JSONL pattern proven by `CodexBackend`, with full ATIF v1.6 trajectory parity including tool-use events, cost reporting, and structured-output emulation. Per-phase model defaults for z.ai models are configured in `config.py`.

- **bench:** Review-bot comparison harness (daydream vs coderabbit/greptile) ([#208](https://github.com/existential-birds/daydream/pull/208))

  Adds `bench/review-bot-compare/` with a replay-driven harness that runs daydream and SaaS review bots (CodeRabbit, Greptile) against the same held-out PRs, harvests their findings, and judges them on a level playing field. Produces per-PR precision/recall metrics so review quality can be compared head-to-head.

- **bench:** Offline benchmark report — daydream vs the SaaS field ([#224](https://github.com/existential-birds/daydream/pull/224))

  Generates a self-contained HTML benchmark report from comparison-harness results, with per-reviewer scorecards and per-PR breakdowns. Renders entirely offline from cached data — no live API calls needed to produce the report.

- **bench:** Select any reviewer backend (incl. GLM via OpenRouter) with evidence-grade submission parity ([#223](https://github.com/existential-birds/daydream/pull/223))

  The benchmark harness now lets you run any backend as the reviewer — including GLM via OpenRouter — with submission parity guarantees so results are comparable across providers regardless of their native output format.

- **deep:** Parallelize the deep fix loop ([#178](https://github.com/existential-birds/daydream/pull/178))

  Same-file findings are now fixed in parallel via `anyio.CapacityLimiter(4)` fan-out instead of sequentially, significantly cutting wall-clock time for fix phases with multiple independent findings.

- **deep:** Ground review intent in the PR description ([#182](https://github.com/existential-birds/daydream/pull/182))

  The deep orchestrator's intent phase now reads the PR description so review focus aligns with what the author set out to do, rather than reviewing the diff in a vacuum.

- **trajectory:** Record phase timing events ([#205](https://github.com/existential-birds/daydream/pull/205))

  Each phase now emits structured timing events into the ATIF trajectory, enabling per-phase wall-clock analysis from the trajectory file without inferring gaps.

- **trajectory:** Register subtrajectory entries for parallel fix-phase forks ([#220](https://github.com/existential-birds/daydream/pull/220))

  Parallel fix-phase fan-outs now register their sibling trajectories in the parent, so the root trajectory records all forks explicitly rather than leaving them as orphaned files.

- **phases:** Batch same-file findings into single run_agent() call ([#216](https://github.com/existential-birds/daydream/pull/216))

  When multiple findings target the same file, they are now passed to a single agent invocation instead of one per finding, reducing redundant context loading and redundant tool calls.

- **backends/pi:** Retry transient stream-drop errors and mark aborted trajectories partial ([#219](https://github.com/existential-birds/daydream/pull/219))

- **backends/pi:** Tighten fix-phase prompts with concise_fix_prompts flag ([#217](https://github.com/existential-birds/daydream/pull/217))

### Changed

- **agent:** Cap runaway run_agent turns with wall-clock and tool-call budgets ([#185](https://github.com/existential-birds/daydream/pull/185))

  Every `run_agent()` call is now bounded by configurable wall-clock and tool-call limits, preventing a single stuck agent from burning through an entire review budget. Limits are tiered by phase.

- **deep:** Short-circuit tiny-diff fan-out and inline diff hunks in per-stack prompts ([#199](https://github.com/existential-birds/daydream/pull/199))

  Small diffs skip the multi-stack fan-out entirely and inline the raw diff hunks into per-stack review prompts, eliminating unnecessary subagent spawns for trivial PRs.

- **deep:** Make verify conditional, demote intent to Sonnet, gate alternatives by diff size ([#197](https://github.com/existential-birds/daydream/pull/197))

  The verification pass now runs conditionally (skipped when no findings need it), the intent phase moves from Opus to Sonnet, and the alternatives phase is gated by diff size — reducing cost and latency on typical PRs without sacrificing quality on complex ones.

- **backends/codex:** Event-layer cost synthesis for provider parity ([#196](https://github.com/existential-birds/daydream/pull/196))

- **backends/codex:** Surface reasoning_output_tokens for cost attribution ([#193](https://github.com/existential-birds/daydream/pull/193))

- **ui:** Split `ui.py` into a `ui/` package ([#183](https://github.com/existential-birds/daydream/pull/183))

  The 3,500-line `ui.py` monolith is now a proper package with separate modules for console, panels, messages, tools, agent text, theme, colorize, and summary rendering.

### Fixed

- **prompts:** Ground review/exploration agents to cwd in linked worktrees ([#222](https://github.com/existential-birds/daydream/pull/222))

- **agent:** Fix `detect_test_success` false-negative on deselected/skipped pytest output ([#204](https://github.com/existential-birds/daydream/pull/204))

  The test-success detector now handles pytest output containing deselected/skipped counts — previously a clean pass with `N deselected` was classified as a failure, aborting the fix loop.

- **backends/claude:** Eliminate cancel-scope RuntimeError on generator cleanup ([#206](https://github.com/existential-birds/daydream/pull/206))

- **backends/pi:** Retry transient 429 and OOM failures ([#215](https://github.com/existential-birds/daydream/pull/215))

- **backends/pi:** Invoke deep review skills natively ([#213](https://github.com/existential-birds/daydream/pull/213))

- **training:** Drop redundant `--since` filter from `log_shas_since`, raise timeout, surface errors ([#218](https://github.com/existential-birds/daydream/pull/218))

- **backends/codex:** Deterministic tool-id pairing and observable parse failures ([#187](https://github.com/existential-birds/daydream/pull/187))

- **backends/codex:** Real-CLI golden contract test, trajectory parity, and cost-synthesis pinning ([#188](https://github.com/existential-birds/daydream/pull/188), [#189](https://github.com/existential-birds/daydream/pull/189), [#190](https://github.com/existential-birds/daydream/pull/190), [#191](https://github.com/existential-birds/daydream/pull/191))

## [0.20.0] - 2026-06-18

### Added

- **github_app:** Run under operator GitHub App identity with scoped token injection ([#159](https://github.com/existential-birds/daydream/pull/159))

  Daydream can now act as a self-hosted review bot using the operator's own GitHub App. Reviews are authored under the App's identity with a short-lived, repo-scoped installation token injected at runtime, so the maintainer never exposes a personal access token.

- **bot:** Actions trigger surface — @-mention / auto-on-PR review via privilege split ([#160](https://github.com/existential-birds/daydream/pull/160))

  Adds GitHub Actions trigger wiring so a review runs either when the bot is @-mentioned on a PR or automatically on PR open/update. A privilege split keeps the untrusted PR-triggered job read-only while a separate trusted job posts results.

- **bot_setup:** One-command self-hosted review-bot setup and distribution ([#161](https://github.com/existential-birds/daydream/pull/161))

  Bundles the workflow templates and a guided setup flow so an operator can stand up a white-label PR review bot in their own repository with a single command.

- **pricing:** User-overridable model price table ([#165](https://github.com/existential-birds/daydream/pull/165))

  Per-model token prices can now be overridden via config so cost metrics stay accurate as provider pricing changes, without waiting for a daydream release.

### Changed

- **cli:** Verb-first redesign with config-file phase control ([#141](https://github.com/existential-birds/daydream/pull/141))

  **Breaking.** The CLI is now verb-first (`daydream review`, `daydream corpus …`), with a default review shim so `daydream /path` still works. Per-phase `--model`/`--backend` flags are removed in favor of `[tool.daydream]` / `[tool.daydream.phases.<phase>]` config in `pyproject.toml` or `.daydream.toml` (precedence: CLI global `--model`/`--backend` > config file > built-in default). The data pipeline verbs are namespaced under `corpus` (`daydream corpus harvest`/`build`/`label`), `--max-iterations` folds into `--loop [N]`, advanced flags hide behind `--help-all`, and non-interactive mode auto-detects from a non-TTY/CI environment.

- **deep:** Sonnet-first per-stack review with a scoped Opus arbiter ([#175](https://github.com/existential-birds/daydream/pull/175))

  Per-stack reviews now run on Sonnet by default with a scoped Opus arbiter pass, cutting review cost and latency while preserving finding quality.

### Fixed

- **git_ops:** Retry transient git timeouts and distinguish them from genuine errors ([#142](https://github.com/existential-birds/daydream/pull/142))

- **training:** Correct run-level label attribution — footer-not-bot and archive-time PR linkage ([#149](https://github.com/existential-birds/daydream/pull/149))

- **ui:** Render Task-family tools with resolved labels in both render paths ([#150](https://github.com/existential-birds/daydream/pull/150))

- **codex:** Real-path harness infrastructure and Codex backend fixes ([#158](https://github.com/existential-birds/daydream/pull/158))

- **training:** Harden harvest PR-signal labeling; de-cruft and parallelize the test suite ([#174](https://github.com/existential-birds/daydream/pull/174))

### Security

- **deps:** Bump `cryptography`, `python-multipart`, and `starlette` in the uv group ([#162](https://github.com/existential-birds/daydream/pull/162))

## [0.19.0] - 2026-06-04

### Added

- **benchmark:** Add held-out PR replay benchmark harness via `daydream bench` ([#137](https://github.com/existential-birds/daydream/pull/137))

  Replays a corpus of held-out, already-merged PRs through the review pipeline and scores each run's findings against the human reviewers' acted-upon comments, producing precision/recall metrics for the review agent. Lets daydream measure regressions in review quality across changes to skills, prompts, and backends.

- **cli:** Add harness-safe non-interactive mode across all daydream flows ([#132](https://github.com/existential-birds/daydream/pull/132))

  `--non-interactive` makes every flow (deep loop, shallow loop, comment, review, PR feedback) take safe defaults instead of prompting, so daydream can run unattended inside CI or an orchestration harness without blocking on a TTY gate.

- **archive:** Add human-override label precedence, idempotent harvest, and rate-limit handling ([#119](https://github.com/existential-birds/daydream/pull/119))

  Harvest now treats human-applied labels as authoritative over inferred ones, re-running a harvest no longer duplicates entries, and GitHub rate-limit responses are backed off and retried instead of aborting the run.

- **training:** Add a posterior reject-penalty axis to the reward signal ([#115](https://github.com/existential-birds/daydream/pull/115))

  The reward model now penalizes findings that were posted but later rejected by a human reviewer, sharpening the signal that distinguishes useful review comments from noise.

- **pr-comment:** Add per-phase wall-clock latency to metrics ([#109](https://github.com/existential-birds/daydream/pull/109))

### Fixed

- **phases:** Ground the failure handoff in evidence with an enforced read-only summarizer ([#136](https://github.com/existential-birds/daydream/pull/136))

  When a run hands off after a failure, the summary is now produced by a read-only agent constrained to cite actual evidence, preventing the handoff from inventing diagnoses or mutating state.

- **cli:** Detect repo slug and PR number from the target checkout, not the current working directory ([#135](https://github.com/existential-birds/daydream/pull/135))

- **workspace:** Accept a commit-ish `--base` and guard against dash injection ([#134](https://github.com/existential-birds/daydream/pull/134))

- **archive:** Derive `wall_clock_seconds` on every run and unify ISO timestamp parsing ([#133](https://github.com/existential-birds/daydream/pull/133))

- **git_ops:** Diff against the remote default branch instead of the local one ([#113](https://github.com/existential-birds/daydream/pull/113))

- **exploration:** Prevent stuck subagents and unmatched tool results ([#112](https://github.com/existential-birds/daydream/pull/112))

- **runner:** Dispatch on `bot` instead of `pr_number` to prevent false feedback routing ([#111](https://github.com/existential-birds/daydream/pull/111))

### Security

- **deps:** Bump `starlette` from 0.52.1 to 1.0.1 in the uv group ([#138](https://github.com/existential-birds/daydream/pull/138))

## [0.18.0] - 2026-05-26

### Added

- **training:** Corpus pipeline architecture — harvest, reward, bitemporal projection ([#104](https://github.com/existential-birds/daydream/pull/104))

  Introduces the `daydream/training/` package with three pipeline stages: `harvest` collects ATIF trajectory runs from the archive into labeled training corpora, `reward` scores trajectory steps against configurable reward signals (cost efficiency, grounding accuracy, finding acceptance rate), and `bitemporal` projects reward labels back onto trajectory steps using both wall-clock and agent-logical timestamps. Designed for offline RL/RLHF fine-tuning loops over daydream's own review traces.

- **training:** JSONL exporter for ATIF trajectories ([#95](https://github.com/existential-birds/daydream/pull/95))

  Adds `daydream training export` subcommand to convert archived ATIF trajectory files into newline-delimited JSON suitable for ML training pipelines. Each trajectory step becomes one JSONL record with flattened token counts, cost, tool I/O, and phase labels.

- **deep:** Language-agnostic structural review meta-stack ([#101](https://github.com/existential-birds/daydream/pull/101))

  Adds a structural review pass that runs independently of per-language Beagle skills, analyzing cross-cutting concerns (API contract consistency, error propagation patterns, dependency direction violations) using tree-sitter parse trees. Activates automatically in deep mode when multiple stacks are detected.

- **archive:** Capture `source_path` in manifest and add harvest clone cache ([#106](https://github.com/existential-birds/daydream/pull/106))

  The archive manifest now records the absolute `source_path` of the reviewed repository, enabling corpus harvest to trace runs back to their origin repo. Adds a clone cache under `~/.daydream/harvest/clones/` so repeated harvests against the same repo skip redundant clones.

### Changed

- **docs:** Rewrite README for research/ML audience ([#107](https://github.com/existential-birds/daydream/pull/107))

### Fixed

- **tests:** Isolate `DAYDREAM_ARCHIVE_DIR` to stop polluting `~/.daydream/archive` ([#100](https://github.com/existential-birds/daydream/pull/100))

  Tests that exercise archive functionality now use a temporary directory instead of the user's real archive, preventing test runs from creating stale entries in the production archive index.

- **explore:** Remove `.coderabbit.yaml` from pattern-scanner prompts ([#96](https://github.com/existential-birds/daydream/pull/96))

### Security

- **deps:** Bump `idna` in the uv group ([#97](https://github.com/existential-birds/daydream/pull/97))

## [0.17.0] - 2026-05-16

### Breaking

- **cli:** Remove the single `--model` flag in favor of per-phase model overrides ([#82](https://github.com/existential-birds/daydream/pull/82)). Replace `--model <name>` with the per-phase flag(s) you actually want — `--review-model`, `--parse-model`, `--fix-model`, `--test-model`, or `--exploration-model`. Passing `--model` now exits early with a curated error pointing at the new flags (both `--model X` and `--model=X` shapes are caught before argparse runs). Each phase resolves its model in this order: explicit per-phase flag → per-backend phase default → backend default, so phases without an override knob (WONDER, ENVISION, MERGE, INTENT, PR_FEEDBACK) still get phase-appropriate models out of the box. Per-backend defaults are documented in the README's *Per-phase model defaults* section.

### Added

- **deep:** Add a read-only `phase_verify_recommendations` pass between cross-stack merge and the fix gate ([#84](https://github.com/existential-birds/daydream/pull/84)). The verifier audits each merged finding's *recommendation* (not just the finding) against trait/interface contracts and sibling implementations, emits structured `consistent` / `contradicts` / `uncertain` verdicts plus `unverified_assumptions` to `.daydream/deep/recommendation-verdicts.json`, and feeds those verdicts back into `phase_fix` so the fix agent has full verifier context when applying changes. Closes the gap where prior passes only validated *whether findings were real* — never whether the recommended fix itself was correct. Runs unconditionally in stage 5 so `--start-at fix` resumes still produce verdicts. UI prints a one-line summary: `Recommendation verification: N findings · M flagged (X contradicts, Y uncertain)`.

- **cli:** Add per-phase model override flags `--review-model`, `--parse-model`, `--fix-model`, `--test-model` ([#82](https://github.com/existential-birds/daydream/pull/82)). Pairs with the existing `--exploration-model` so callers can tune cost vs. quality per phase — PARSE on haiku, FIX/TEST/EXPLORATION on sonnet, REVIEW/WONDER/ENVISION/MERGE/INTENT/PR_FEEDBACK on opus — without a single-knob compromise.

- **config:** Add `PHASE_DEFAULT_MODELS` table mapping backend → phase → model id ([#82](https://github.com/existential-birds/daydream/pull/82)). `_resolve_backend(config, phase)` consults this table when no explicit per-phase flag is set, and the deep orchestrator resolves per-phase backends for intent / wonder / review / parse / merge / verify / fix / test. UI now prints a dim `Model: <name>` line after every phase hero so the effective per-phase model is visible at a glance.

## [0.16.0] - 2026-05-15

### Added

- **scripts:** Add `scripts/review-historic-pr <PR>` helper for benchmarking daydream against already-merged PRs ([#79](https://github.com/existential-birds/daydream/pull/79))

  Pins the review to the exact code state the PR introduced (PR head against the merge-base on the original target branch) so output is apples-to-apples with whatever was reviewed at PR time (greptile, coderabbit, etc.). Honors `DAYDREAM_ARGS`, `KEEP_BRANCHES`, and optional `ZIP` / `ZIP_OUT` env vars for bundling `.review-output.md` plus the ATIF trajectory directory. Requires `gh`, `git`, `jq`, and `daydream` on `$PATH`.

### Changed

- **trajectory:** Read the daydream version from `daydream.__version__` instead of `importlib.metadata.version("daydream")` when stamping `agent.version` into ATIF trajectories ([#81](https://github.com/existential-birds/daydream/pull/81))

  Brings the trajectory recorder in line with every other version-stamping surface (PR comment renderer, PR review wizard, `Daydream-Version:` git trailer). Eliminates the silent `"0.0.0"` fallback that fired on `PackageNotFoundError` — a broken or partial install will now surface the failure loudly. Also eliminates editable-install lag: `importlib.metadata` reads from the installed package record (which only refreshes on `uv sync`), so trajectories used to stamp the *previous* version after a `__version__` bump until the next sync. Reading the module attribute is always current.

## [0.15.0] - 2026-05-11

### Breaking

- **trajectory:** Change default trajectory path from `<target>/.daydream/trajectory-<ts>-<id>.json` to `<target>/.daydream/runs/<session_id>/trajectory.json` ([#70](https://github.com/existential-birds/daydream/pull/70)). Scripts that glob for `trajectory-*.json` directly under `.daydream/` must update to `runs/*/trajectory.json`. The `--trajectory <path>` flag continues to work for custom locations.

- **cli:** Remove deprecated CLI flags ([#75](https://github.com/existential-birds/daydream/pull/75)). The aliases introduced in #44 alongside the new consolidated surface are gone:

  | Removed | Replacement |
  |---|---|
  | `--ttt`, `--trust-the-technology` | `--comment` |
  | `--review-only` | `--review` |
  | `--deep` | (removed; deep is the default — pass `--shallow` to opt out) |
  | `--pr <n>` (top-level), top-level `--bot` | `daydream feedback <n> --bot <name>` subcommand |
  | `--python`, `--typescript`, `--elixir`, `--go`, `--rust`, `--ios` | `-s <skill>` (or auto-detect from changed files) |

  Removes the `_warn_deprecated` helper and its 5 deprecation warnings from `daydream/cli.py`. Drops the now-dead `RunConfig.review_only`, `trust_the_technology`, `deep`, and `forced_skill` fields and their downstream branches in `runner.py` and `ui.py`. Archive manifest still emits `review_only` / `deep` keys (derived from `output_mode` / `shallow`) so the index schema is unchanged.

### Added

- **workspace:** Unify git operations and add worktree isolation ([#57](https://github.com/existential-birds/daydream/pull/57))

  Every scattered `subprocess.run(["git", ...])` and `gh` callsite is replaced by a single typed `daydream/git_ops.py` wrapper, and a new `daydream/workspace.py` introduces a `WorkContext` + `open_workspace` abstraction so runs can execute against an ephemeral git worktree (or in-place). The headline UX bug from #44 — running daydream from an org/container directory produced four parallel `fatal: not a git repository` errors and the agent confabulated explanations — is now a typed `NotAWorktreeError` raised at the boundary with actionable guidance. The companion silent-failure case (`git diff main...HEAD` while sitting on `main` quietly producing an empty diff) now raises `WrongBranchError` with three recovery paths (check out a feature branch, pass `--branch`, or pass `--worktree`). Adds `--worktree`, `--branch`, `--base`, and repeatable `--copy <path>` CLI flags; `RunConfig` gains 7 fields including `output_mode`. A pure-integer TARGET is now rejected with `did you mean: daydream feedback <pr#>?`.

- **cli:** Add `daydream summarize <path>` subcommand to render the run-info markdown (rollup + per-phase breakdown table + version footer) for a trajectory file or run directory ([#70](https://github.com/existential-birds/daydream/pull/70))

  Pure read + render + print — no GitHub posting. The same call also unifies live and archive trajectory layouts: both now use `<root>/runs/<session_id>/trajectory.json` plus `<root>/runs/<session_id>/trajectories/<descriptor>.json` siblings. The `- **Mode:** <label>` rollup line is removed from `render_run_info_block` and the live PR comment everywhere; the renderer now owns the `<sub>Generated by daydream vX.Y.Z</sub>` footer (previously added by the PR-comment shell).

- **cli:** Skip the ENVISION (plan generation) phase by default in `--comment` mode and add a `--plan` opt-in flag ([#71](https://github.com/existential-birds/daydream/pull/71))

  Eliminates unnecessary token spend and interactive prompts when only posting PR review comments. Pass `--plan` to opt back in: auto-selects all issues, generates a structured implementation plan, and embeds per-issue change instructions (file, action, description, references) in the consolidated agent prompt on the PR comment. `phase_generate_plan` now returns `tuple[Path | None, dict | None]` to surface the raw plan data for downstream consumers.

- **phases:** Add a setup-investigator and a failure-summarizer/handoff for smarter test failure recovery ([#77](https://github.com/existential-birds/daydream/pull/77))

  Option 1 in `phase_test_and_heal`'s recovery menu now runs a read-only setup-investigator subagent (inspects `Makefile`, `pyproject.toml` / `package.json`, CI config, `CLAUDE.md`, `README`) before retrying — if it suggests a different test command, the user confirms before the retry switches. Option 4 ("exit and hand off") now spawns a failure-summarizer subagent that writes `handoff.md` to the archived run directory referencing artifacts by absolute path (no embedded diffs), and explicitly instructs the downstream agent to refuse inline hacks. Adds a soft clipboard wrapper (`daydream/clipboard.py`) using `pbcopy` / `xclip` / `xsel` / `clip.exe` — no new top-level deps. Newly created untracked files now surface in the handoff prompt via `git_ops.changed_files()`. Sub-trajectories are recorded via `TrajectoryRecorder.fork()`.

- **phases:** Tag every daydream commit with `Daydream-Run: <run_id>` and `Daydream-Version: <version>` git trailers ([#73](https://github.com/existential-birds/daydream/pull/73))

  Both `phase_commit_push` and `phase_commit_iteration` inject the trailers into the commit agent prompt and verify them post-commit, amending if the agent omitted them. A new `git_ops.amend_trailers()` helper uses `git interpret-trailers` to append missing trailers, and `git_ops.daydream_commits()` queries the log for commits carrying the `Daydream-Run` trailer.

- **pr-comment:** Enrich the PR summary comment with run metrics, daydream version, and Codex cached tokens ([#66](https://github.com/existential-birds/daydream/pull/66))

  Replaces the single "Mode" line with a structured rollup (model, cost, tokens, cache hit %, steps, tool calls) plus a collapsed per-phase breakdown table and a daydream version footer. Introduces `daydream/pricing.py` with a static `ModelPrice` dataclass and `MODEL_PRICES` table covering `gpt-5.5`, `gpt-5.5-pro`, `gpt-5-codex`, and `gpt-5.3-codex`, so Codex-backed runs can synthesize cost when the backend doesn't report it. `daydream/pr_comment_renderer.py` is a pure renderer — same trajectory in, same markdown out — that aggregates per-phase metrics across one or more sibling trajectories (deep-mode). Also fixes `CodexBackend` to extract `cached_input_tokens` from `turn.completed.usage` (previously hardcoded to `None`).

### Changed

- **agent:** Revert default Claude model from `claude-opus-4-7` to `claude-opus-4-6` ([#68](https://github.com/existential-birds/daydream/pull/68))

  4.7 escalates the SDK's default malware-analysis `<system-reminder>` (injected on every `Read` tool result) into hard refusals on benign user code, citing the reminder verbatim and declining requested edits — a regression vs. 4.6 / 4.5, which treat it as a soft nudge. Trajectory data across three recent deep runs showed 15 reminder-tied refusals out of 135 reads (~11%) and ~2,537 wasted output tokens on "this is benign code, proceeding…" preambles. Reverting to 4.6 until the underlying behavior is addressed.

- **agent:** Make `model` a required `str` end-to-end ([#68](https://github.com/existential-birds/daydream/pull/68)); default lives only in `daydream.config.DEFAULT_CLAUDE_MODEL` / `DEFAULT_CODEX_MODEL`, resolved exactly once in `create_backend`. Removed five duplicated literal fallbacks (`ClaudeBackend.__init__`, `CodexBackend.__init__`, `AgentState.model`, `runner.set_model(... or "...")`, banner display) so a future model bump is one constant change, not a five-file shotgun edit. `set_model` / `get_model` deleted (`AgentState.model` was unused).

- **pr-comment:** Slim the per-phase breakdown columns in the PR summary comment table ([#69](https://github.com/existential-birds/daydream/pull/69))

### Fixed

- **phases:** Guard `_do_commit` trailer amendment against amending non-daydream commits when the pre-commit SHA could not be read ([#73](https://github.com/existential-birds/daydream/pull/73)). When `head_sha()` raised `GitError` before the agent ran, `sha_before` was `None`, so the `sha_after == sha_before` check always passed through, potentially amending a pre-existing user commit with daydream trailers. Now also checks for `sha_before is None` and skips trailer verification in that case.

- **archive:** Migrate the SQLite schema on connect by adding missing `review_backend` / `fix_backend` / `test_backend` columns via `ALTER TABLE` ([#71](https://github.com/existential-birds/daydream/pull/71)), fixing "Run archive failed (non-fatal)" on DBs created before these columns existed.

### Security

- **deps:** Bump `python-multipart` from 0.0.26 to 0.0.27 ([#74](https://github.com/existential-birds/daydream/pull/74)) — adds multipart header limits and safer parse-offset constructors upstream.

## [0.14.0] - 2026-04-29

### Breaking

- **cli:** Remove `--debug` flag; use `--trajectory <path>` to control trajectory output location. Daydream no longer produces `.review-debug-{timestamp}.log` files.

### Added

- **trajectory:** Every run now produces an [ATIF v1.6](https://www.harborframework.com/docs/agents/trajectory-format) trajectory file at `<target>/.daydream/trajectory-<ts>-<id>.json` capturing the full agent interaction history, tool I/O, and per-step token/cost metrics.
- **cli:** Add `--trajectory <path>` flag to write trajectories to a custom location. Trajectories are always written; the flag only controls the output path.
- **redaction:** Automatic secret redaction applied to all trajectory content — API keys, JWT tokens, file paths with usernames, and `.env`-style secret values are replaced with `[REDACTED_*]` tokens before writing.
- **trajectory:** Parallel fan-out flows (fix-parallel, deep-mode per-stack, exploration specialists) produce sibling trajectory files linked from the root via `subagent_trajectory_ref`.
- **trajectory:** SIGINT/SIGTERM mid-run flushes a partial trajectory to `<path>.partial`.

### Archive & Evaluation

- **archive:** Every run is automatically archived to `~/.daydream/archive/runs/{session_id}/` with `manifest.json` metadata, trajectory copy, and review artifacts. SQLite index at `~/.daydream/archive/index.db` enables cross-project querying.
- **cli:** Add `--no-archive` flag to disable automatic archival.
- **eval:** Add `--eval` flag to run deterministic trajectory analysis (cost efficiency, grounding rate, file coverage, finding quality) and store results in the archive.
- **cli:** Add `daydream label <session_id> --accepted|--rejected|--mixed` subcommand to set outcome labels on archived runs.

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

[unreleased]: https://github.com/existential-birds/daydream/compare/v0.22.0...HEAD
[0.22.0]: https://github.com/existential-birds/daydream/compare/v0.21.0...v0.22.0
[0.21.0]: https://github.com/existential-birds/daydream/compare/v0.20.0...v0.21.0
[0.20.0]: https://github.com/existential-birds/daydream/compare/v0.19.0...v0.20.0
[0.19.0]: https://github.com/existential-birds/daydream/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/existential-birds/daydream/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/existential-birds/daydream/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/existential-birds/daydream/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/existential-birds/daydream/compare/v0.14.0...v0.15.0
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
