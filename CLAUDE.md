# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project overview

Automated code review and fix loop: reviews diffs with stack-specific
[Beagle](https://github.com/existential-birds/beagle) skills, applies fixes, validates via test suite, and
records every agent interaction as an
[ATIF v1.7](https://www.harborframework.com/docs/agents/trajectory-format) trajectory. A bitemporal corpus
pipeline scores, labels, and projects those trajectories into JSONL datasets for SFT/RL fine-tuning.

Default flow is the deep multi-stack pipeline; `--shallow` is a single-stack, single pass; `--comment`/`--review`
are review-only; `daydream feedback <pr#>` ingests bot review comments. Four backends — Claude
(in-process SDK), Codex, Pi, and Osprey (subprocess CLIs) — all emit the same `AgentEvent` stream.

Reference docs: `README.md` (user CLI + config), `docs/{extensions,benchmark}.md`.

## Commands

```bash
make install   # uv sync
make hooks     # install pre-push hook
make lint      # ruff check daydream tests bench
make typecheck # mypy daydream tests bench
make test      # pytest -n auto
make check     # lockcheck + lint + typecheck + full pytest (the gate)
```

```bash
# Golden paths (near-zero-flag; `daydream /path` == `daydream review /path`)
daydream /path/to/project                          # review -> fix -> test (deep multi-stack)
daydream --comment /path/to/project                # review -> post inline PR comments, then exit
daydream improve /path/to/project                  # read-only repo audit -> prioritized plans
daydream improve plan "add rate limiting" /path/to/project  # investigate one request -> plan
daydream improve prune-reanchor <NAME> /path/to/project   # remove one executed re-anchor worktree (exit 0 on removal, non-zero when the name is rejected/absent/git-failed)
daydream improve list-reanchor /path/to/project   # list existing -reanchor worktrees

# Other verbs / flags (`--help-all` for the full advanced surface)
daydream --shallow -s python /path/to/project      # shallow Python single-pass review-fix-test
daydream --review /path/to/project                 # review only, skip fixes
daydream --yes /path/to/project                    # auto-apply fixes without prompting
daydream feedback 42 --bot "<bot-login>[bot]" /path/to/project  # bot PR comments
daydream --non-interactive /path/to/project        # unattended/harness run
```

The rest of the surface — `post-findings`, `setup`, `summarize`, `ext validate`, `corpus *` — is in
README "Additional Commands" / "Corpus Commands"; `daydream bench` is in `docs/benchmark.md`.

## Testing standard (mandatory)

Every user-visible behavior must have at least one **real-path test**: a test that enters from the
production entrypoint (`runner.run` / the CLI) with real dependencies (real temp git worktree, real
filesystem, real event loop), mocking only the external network/API backend (via the `Backend` protocol /
`create_backend` seam). Tests must assert observable outcomes (exit code, files written, fixes applied or
declined, transcript state), never that a function was merely called. Unit tests are supplementary, not a
substitute. Reference exemplar: the non-interactive/EOF gate tests in `tests/test_deep_orchestrator.py`.

**No caveats.** All work is completed and proven, or explicitly in progress. No deferred items, no
"optional" follow-ups, no smoke-tests substituted for real coverage.

## Architecture

```text
cli.py -> runner.py -> deep/orchestrator.py -> flows/engine.py (deep FlowSteps)
              |        -> improve/orchestrator.py -> flows/engine.py (improve FlowSteps)
              |        -> flows/engine.py (custom extension flows)
              \-> ui/ (terminal output)
deep FlowSteps -> phases.py -> agent.py -> Backend.execute()
```

- `runner.run()` is the async entry: builds the per-run extension `Registry` onto a `ContextVar`, then
  dispatches PR-process modes (`deep`, `shallow`, `review`, `comment`, or `feedback`) to the single deep
  orchestrator. `improve` and custom extension flows run through `run_flow()` over registered `FlowStep` lists.
- `agent.run_agent()` is the only agent call site. **Never call a backend/SDK directly from phases.**
- Subagent fan-out (exploration, per-stack review, parallel fix) is N parallel `run_agent()` calls under
  `anyio.CapacityLimiter(effective_fanout_concurrency(ceiling, backend))`, **not** SDK `agents=`.
- `TrajectoryRecorder` propagates via `ContextVar`; `recorder.fork()` makes sibling trajectories per fan-out.

### Module responsibilities

| File | Responsibility |
|------|----------------|
| `cli.py` | Args, signals, process lifecycle, subcommand dispatch |
| `runner.py` | Flow preambles (workspace, diff, recorder), backend resolution, registry, dispatch |
| `flows/` | `FlowContext` + `run_flow()` engine: ordering, `enabled` gates, `Stop`/`BreakLoop`, loop groups |
| `extensions/` | `Registry` (phases+flows, skill slots, prompts, stack rules), `daydream_ext` loader |
| `deep/orchestrator.py` | Deep-flow steps: exploration, intent, wonder, per-stack, arbiter, merge, verify, fix |
| `deep/{detection,dedup,artifacts}.py` | `detect_stacks()` router, artifact paths, dedup pre-filter |
| `deep/arbiter.py` | Scoped Opus pass over high-severity/contested findings |
| `improve/` | Read-only recon, category audits, vetting, prioritization, plan artifacts |
| `phases.py` | Stateless async `phase_*()` steps and prompt builders |
| `agent.py` | Backend wrapper, events to UI, global state, budget enforcement |
| `trajectory.py` | ATIF v1.7 recorder, redaction, ContextVar propagation |
| `backends/` | `Backend` protocol, Claude/Codex/Pi/Osprey, `AgentEvent` union, `create_backend()` |
| `ui/` | Rich output (Dracula): `console`, `panels`, `messages`, `tools`, `agent_text`, `summary`, `theme`, `colorize` |
| `config.py`, `config_file.py` | Skill mappings, per-phase model/effort defaults, budgets; `[tool.daydream]` / `.daydream.toml` parser |
| `workspace.py` | `WorkContext`: in-place vs ephemeral detached worktree |
| `git_ops.py` | **Single point of contact for every `git`/`gh` shell-out** |
| `exploration*.py`, `tree_sitter_index.py` | Pre-scan: tree-sitter import resolution, convention detection |
| `supervision.py` | Runtime findings + tool supervision (extension veto seam) |
| `reconcile.py` | Cross-run dedup vs prior bot PR comments (GitHub is the store) |
| `pr_comment_renderer.py` | Pure renderer: trajectory in, markdown out (no I/O) |
| `training/` vs `eval/` | Corpus pipeline (harvest, reward, projection, JSONL) vs deterministic trajectory analysis |
| `prompts/` | Authorial intent, exploration subagents, CWD grounding |

Self-describing modules are not listed: `pr_review.py`, `findings.py`, `pricing.py`, `github_app.py`,
`bot_identity.py`, `bot_setup.py`, `summarize.py`, `archive/`, `benchmark/`.

### Backend protocol

`Backend` (in `backends/__init__.py`) is `model` + `execute()` + `cancel()` + `format_skill_invocation()`.
`execute()` yields the 8-member `AgentEvent` union (`Text`, `Thinking`, `ToolStart`, `ToolResult`, `Cost`,
`Metrics`, `TurnEnd`, `Result`). Adding a backend means producing that stream correctly — phases and the
recorder are backend-agnostic.

### Run-agent budgets

| Bound | Value | Scope |
|-------|-------|-------|
| `DEFAULT_WALL_BUDGET_S` | 1800s | every `run_agent()` turn (improve phases deliberately unbounded) |
| `TEST_WALL_BUDGET_S` | 3600s | the test run — bounds the *target repo's* suite, not an LLM tail |
| `DEFAULT_TOOL_CALL_BUDGET` | `None` | unlimited; per-call `tool_call_budget` still accepted |
| `DEFAULT_GROUP_MAX_WALL_S` / `_SERIAL_ITEMS` | 600s / 6 | cumulative over all fix calls for one file group |
| `EXPLORATION_MAX_TURNS` | 50 | exploration specialists — the only `max_turns` call site |
| `DAYDREAM_STREAM_IDLE_TIMEOUT_S` | 2700s | pi/codex stdout silence before the subprocess is killed |

- Exhaustion emits a `TurnEndEvent` and marks the trajectory partial. **Truncation is never silently
  absorbed**: a truncated wonder or parse raises; a truncated per-stack review goes to `failed_stacks` so
  merge lists it under "Uncovered stacks" instead of recording a clean pass.
- Do not add `max_turns` to fix or verify — it does not fail soft. The turn ends `error_max_turns`, the
  backend raises `MaxTurnsError`, and the fix group lands in `fix-failures.json` and is reverted, throwing
  a real fix away rather than trimming it.
- `run_agent` retries any `retryable` backend error with exponential backoff, **20 attempts**, all backends
  (`DAYDREAM_PI_RETRY_ATTEMPTS` overrides). Never retried: tool-supervisor veto,
  non-transport logic error. A stall fires only on the *absence* of output, never on slow output.

### Config and per-phase model overrides

`config.py` holds `DEFAULT_{CLAUDE,CODEX,PI,EXPLORATION}_MODEL`, `PHASE_DEFAULT_MODELS[backend][phase]`,
`PHASE_DEFAULT_EFFORT` (deep/review half Codex-only; improve half all three backends), budget constants, improve `EFFORT_TIERS`, and skill mappings. Pi
resolves its own configured default before falling back to `DEFAULT_PI_MODEL`.

**Per-phase overrides are config-file-only — there are no per-phase CLI flags.** Set
`[tool.daydream.phases.<phase>]` in `pyproject.toml` or the top-level equivalent in `.daydream.toml`.
Precedence: CLI `--model`/`--backend` > config-file phase > config-file global > backend default; resolved
in `runner._resolve_backend()`. The phase table accepts any registered step's config key, including
fork-defined phases (`docs/extensions.md`).

Improve runtime controls are CLI-derived `RunConfig` fields (`improve_effort`, `improve_focus`,
`improve_scope`, `improve_plan_description`) with **no** env-var equivalents; service discovery and fan-out
bounds are config-file-only (`[tool.daydream.improve]`, keys in README "Configuration"). Fan-out is
`partition-groups × categories`; when `max_partition_groups` binds, the largest groups are kept and every
skipped partition is named in `.daydream/improve/coverage.json` and the report's "What was not audited".

### Deep-review pipeline

```text
exploration pre-scan (cached across runs)
    -> intent analysis (Sonnet)
    -> alternative review (wonder) ∥ per-stack reviews (parallel, Sonnet;
       structural review for cross-cutting concerns)
    -> per-stack parse (parallel)
    -> arbiter review (Opus, scoped to high-severity/contested findings)
    -> cross-stack merge (dedup; resumes the arbiter's session)
    -> recommendation verification (conditional)
    -> fix gate (parallel, batched per-file)
    -> test validation
```

- Wonder ∥ per-stack are siblings in one task group on a fresh multi-stack run (wonder feeds only merge and
  the dedup pre-filter, so reviewer prompts drop the `alternatives.json` pointer; they join before parse).
  Single-stack mode and every `--start-at` resume keep the serial order **and** the pointer — single-stack
  has no merge agent, so that pointer is the only path wonder findings take into the report. This boundary
  is why the extension API is v4.
- The N parse calls run concurrently but are consumed in **stack-name order**, keeping merge input ordering
  and global issue numbering reproducible.
- Intent and wonder prompts inline the diff under `INLINE_DIFF_BUDGET_BYTES` (12 KiB, shared with per-stack),
  else the `diff.patch` pointer. Small diffs skip the fan-out entirely.
- Merge resumes the arbiter's session when both phases resolve to the same backend instance; the resumed
  prompt forces a re-read of the per-stack record files, rewritten after arbitration.
- `.daydream/exploration/` survives the run, reused only on an **exact** key match (format version + head SHA + diff + tier +
  depth, in a sibling `cache-key` file) — a near-match hit would misground every prompt. Uncommitted edits
  are not in the key, so an exact hit on a dirty tree can serve pre-edit exploration. `--shallow`/`--review`
  delete the directory, so alternating modes always miss.
- `--start-at` refuses stale artifacts: a fresh run records its diff in `.daydream/deep/diff-key`; a resume
  whose diff no longer matches (or has no key) exits 1 rather than adjudicating stale findings.

### Extension seam

A fork customizes phases, flows, skills, and prompts from a top-level `daydream_ext` package (found via
`$DAYDREAM_EXT_DIR` → `import daydream_ext`) without editing `daydream/`. It must export
`DAYDREAM_EXT_API` within `MIN_SUPPORTED_EXTENSION_API_VERSION..EXTENSION_API_VERSION` (both 5), may
register one `ToolDecision`-returning tool supervisor, and is resolve-checked by `daydream ext validate`.
Full contract: `docs/extensions.md`.

## Constraints and conventions

- **SDK** `claude-agent-sdk==0.2.116`, must stay ≥ 0.2.111: earlier versions tear down the CLI subprocess
  unshielded on cancellation, so a budget/fan-out cancel mid-stream corrupts anyio's cancel-scope stack.
- **ATIF** vendored from Harbor v0.17.1-9 under `daydream/atif/` (Apache-2.0), pinned to v1.7 emission.
  Re-vendor wholesale on Harbor updates; no local patches. **No `harbor` runtime dep** — ATIF models live in
  `daydream/trajectory.py` only. **Module-bloat ban**: no ATIF construction in `phases.py` or `ui/`.
- Deps live in `pyproject.toml`; keep `uv.lock` in sync via `uv lock` or `make check` fails at step one.
- **`make check`** = `uv lock --check` + ruff/mypy over `daydream tests bench` + pytest;
  `scripts/hooks/pre-push` runs the identical gate.
- Ruff: 120 cols, `E F I W`, py312. `daydream/atif/**` is lint-exempt (vendored, mechanical edits only).
- **Conventional Commits** (`feat(backends): ...`). Stage explicitly (`git add <path>`), never `git add -A`.
- Fix bugs at the root. Never bypass the hook, skip tests, or `git push --no-verify`.
- Own your own bugs in plain language. Never describe your defect as the tool being buggy.
- Never claim success that isn't verified-working.

## Environment variables

| Variable | Scope | Purpose |
|----------|-------|---------|
| `DAYDREAM_APP_ID` / `DAYDREAM_APP_PRIVATE_KEY` | GitHub App | Bot identity (PEM **content**, not a path) |
| `DAYDREAM_BOT_HANDLE` | Actions | Mention handle (no `@`) used by the `@<bot> review` command |
| `DAYDREAM_EXT_DIR` | Extensions | Path to `daydream_ext` (overrides `import daydream_ext`) |
| `DAYDREAM_GH_TIMEOUT_SECONDS` / `_RETRIES` | Git ops | `gh` CLI timeout and retry count |
| `DAYDREAM_TRAJECTORY_HUB_REPO` | Archive | Optional HuggingFace dataset repo to upload each run's bundle to; one of the two operator sources (the other is the CLI `--trajectory-hub-repo` flag). The target checkout's file config is ignored for this |
| `PI_PROVIDER` / `PI_THINKING` | Pi | `--provider` / `--thinking`; `PI_THINKING` loses to a per-phase `reasoning_effort` |
| `PI_API_KEY` | Pi | Copied into the child's provider-native var (e.g. `ZAI_API_KEY`), **never onto argv**; warns and ignores if the provider has no mapped var |
| `DAYDREAM_PI_RETRY_ATTEMPTS` / `_BASE_DELAY_S` / `_MAX_DELAY_S` | Retry | Attempts default 20, all backends |
| `DAYDREAM_FANOUT_CONCURRENCY` | Claude / Codex | Parallel `execute()` hint (default 8; bad value warns). Pi uses `DAYDREAM_PI_FANOUT_CONCURRENCY` (default 10) |
| `DAYDREAM_STREAM_IDLE_TIMEOUT_S` | Pi / Codex | Stdout-silence kill (default 2700; `0` disables) |
| `MARTIAN_API_KEY` / `_BASE_URL` / `_MODEL`, `ANTHROPIC_API_KEY` | Benchmark | Judge endpoint/model (`martian` / `anthropic-direct`) |

Plain path overrides: `DAYDREAM_PRICES_FILE`, `DAYDREAM_ARCHIVE_DIR`, `DAYDREAM_SKILLS_DIR` (Beagle skills,
Pi), `PI_CODING_AGENT_DIR` (`~/.pi/agent`), `CLAUDE_CONFIG_DIR` (`~/.claude`).

## Platform requirements

Python ≥3.12.13 + uv; `git` and `gh` on `$PATH`; Beagle plugin installed in Claude Code (for
`beagle-*:review-*` skills); `codex`/`pi` CLIs only for their backends; pre-push hook via `make hooks`.
