# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project overview

Daydream is an automated code review and fix loop. It reviews diffs using stack-specific [Beagle](https://github.com/existential-birds/beagle) skills, applies fixes, validates via test suite, and records every agent interaction as an [ATIF v1.7](https://www.harborframework.com/docs/agents/trajectory-format) trajectory. A bitemporal corpus pipeline scores, labels, and projects those trajectories into JSONL datasets for SFT and RL fine-tuning.

The default flow is a deep multi-stack pipeline. `--shallow` opts into a single-skill loop. `--comment` and `--review` produce review-only output (PR comments or markdown report). The `daydream feedback <pr#>` subcommand ingests bot review comments.

Three backends are supported: Claude (in-process SDK), Codex (subprocess CLI), and Pi (subprocess CLI for z.ai GLM models). All backends emit the same `AgentEvent` stream, consumed by `run_agent()` and the trajectory recorder.

## Commands

```bash
make install   # install dependencies and git hooks
make hooks     # install pre-push hook
make lint      # ruff
make typecheck # mypy daydream tests
make test      # pytest
make check     # lint + typecheck + full pytest (the gate)
```

```bash
# Golden paths (near-zero-flag; `daydream /path` == `daydream review /path`)
daydream /path/to/project                          # review -> fix -> test (deep multi-stack)
daydream --comment /path/to/project                # review -> post inline PR comments, then exit

# Other verbs / flags
daydream --shallow -s python /path/to/project      # shallow Python review-fix-test loop
daydream --review /path/to/project                 # review only, skip fixes
daydream --yes /path/to/project                    # auto-apply fixes without prompting
daydream --loop 3 /path/to/project                 # repeat review-fix-test up to 3 rounds
daydream feedback 42 --bot "<bot-login>[bot]" /path/to/project  # bot PR comments
daydream --non-interactive /path/to/project        # unattended/harness run
daydream --help-all                                # full advanced flag surface

# Findings artifact (two-phase post)
daydream --review --findings-out findings.json --pr-number 7 /path
daydream post-findings findings.json --pr 7 --head-sha <sha> --repo owner/repo

# Self-hosted review bot
daydream setup /path/to/repo --repo OWNER/REPO     # one-command App + secrets + workflow PR
daydream setup /path/to/repo --verify              # read-only install audit

# Run-info markdown
daydream summarize <path>                          # print trajectory summary for a run dir or file

# Data pipeline (under the `corpus` namespace)
daydream corpus harvest                            # annotate archived runs (reward + label)
daydream corpus build --out out.jsonl              # project labeled runs to JSONL training corpus
daydream corpus label <session_id> --outcome accepted

# Extension seam
daydream ext validate                              # load daydream_ext, resolve-check flows/skills/prompts/stacks

# Benchmark
daydream bench --benchmark-repo ../code-review-benchmark/offline --score

# Per-phase model/backend overrides are config-file-only (no CLI flags):
#   set [tool.daydream] / [tool.daydream.phases.<phase>] in pyproject.toml or .daydream.toml.
#   Precedence: CLI (--model/--backend) > config file > built-in default.
```

## Testing standard (mandatory)

Every user-visible behavior must have at least one **real-path test**: a test that
enters from the production entrypoint (`runner.run` / the CLI) with real
dependencies (real temp git worktree, real filesystem, real event loop), mocking only
the external network/API backend (via the `Backend` protocol / `create_backend` seam).
Tests must assert observable outcomes (exit code, files written, fixes applied or
declined, transcript state), never that a function was merely called. Unit tests are
supplementary, not a substitute. Reference exemplar: the non-interactive/EOF gate tests
in `tests/test_deep_orchestrator.py`.

**No caveats.** All work is completed and proven, or explicitly in progress. No
deferred items, no "optional" follow-ups, no smoke-tests substituted for real coverage.

## Non-negotiable: fix bugs at the root

The highest-priority directive in this file. Violating it makes the agent dangerous
and unusable.

1. A bug in a safety or verification mechanism is fixed at its root cause, never
   bypassed. `git push --no-verify`, skipping tests, commenting out checks, deleting
   guards, lowering assertions are all forbidden. Making the gate pass honestly IS
   the goal.

2. Own your own bugs in plain language. Do not describe your own defect as the tool,
   hook, or framework being "destructive" or "buggy."

3. Fix the bug where it lives, not at a convenient downstream layer.

4. A dangerous bug (state corruption, data loss, compromised safety mechanism)
   outranks the assigned task. Stop and fix it first.

5. Never claim success that isn't verified-working. "Committed" or "pushed" is not
   "working" unless the actual behavior is verified.

6. When the user pushes back more than once on the same point, stop defending and do
   the direct, root-cause fix.

## Architecture

### Execution flow

```text
cli.py -> runner.py -> flows/engine.py (run_flow over registered FlowSteps)
              -> deep/orchestrator.py | flows/{shallow,review,pr_feedback}.py
              -> phases.py -> agent.py -> Backend.execute()
              \-> ui/ (terminal output)
```

- `runner.run()` is the async entry. It builds the per-run extension `Registry`
  (`build_registry()`), sets it on a `ContextVar`, and dispatches one of five
  registered flows — `deep` (default), `shallow`, `review` (`--review`/`--comment`),
  `pr-feedback` (`daydream feedback <pr#>`), or a custom extension flow
  (`--flow NAME`) — each a preamble plus `flows.run_flow()` over the flow's
  ordered `FlowStep` list.
- `agent.run_agent()` is the only agent call site. Never call a backend/SDK directly
  from phases. It wraps `Backend.execute()` and drives the Rich UI + trajectory recorder.
- Subagent fan-out (exploration, per-stack review, parallel fix) is N parallel
  `run_agent()` calls under `anyio.CapacityLimiter(4)`, not SDK `agents=`.
- `TrajectoryRecorder` propagates via `ContextVar`; `recorder.fork()` creates sibling
  trajectories for parallel fan-outs.

### Module responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| CLI | Arg parsing, signal handling, process lifecycle, subcommand dispatch | `cli.py` |
| Runner | Flow preambles (workspace, diff, recorder), backend resolution, registry build, flow dispatch | `runner.py` |
| Flows | `FlowContext` + `run_flow()` engine (ordering, `enabled` gates, `Stop`/`BreakLoop`, loop groups); shallow/review/pr-feedback step functions | `flows/` |
| Extensions | Versioned extension API: `Registry` (phases+flows, skill slots, named prompts, stack rules), `daydream_ext` loader, built-in seeding | `extensions/` |
| Deep orchestrator | Deep-flow step functions (exploration, intent, alternatives, per-stack, arbiter, merge, verify, fix) | `deep/orchestrator.py` |
| Phases | Stateless async `phase_*()` workflow steps and prompt builders | `phases.py` |
| Agent | Backend wrapper, event stream to UI, global state, budget enforcement | `agent.py` |
| Trajectory | ATIF v1.7 recorder, redaction, ContextVar propagation | `trajectory.py` |
| Backends | `Backend` protocol, `ClaudeBackend`, `CodexBackend`, `PiBackend`, `AgentEvent` union, `create_backend()` | `backends/` |
| UI | Rich terminal output (Dracula theme): `console`, `panels`, `messages`, `tools`, `agent_text`, `summary`, `theme`, `colorize` | `ui/` |
| Config | Skill mappings, per-phase model defaults, constants | `config.py` |
| Config file | `[tool.daydream]` / `.daydream.toml` parser for per-phase overrides | `config_file.py` |
| Exploration | Pre-scan codebase context (tree-sitter import resolution, convention detection) | `exploration.py`, `exploration_runner.py`, `tree_sitter_index.py` |
| Deep detection | Stack router (`detect_stacks()`), artifact paths, dedup pre-filter | `deep/detection.py`, `deep/dedup.py`, `deep/artifacts.py` |
| Arbiter | Scoped Opus pass over high-severity/contested findings | `deep/arbiter.py` |
| PR review | Post findings as inline GitHub PR comments | `pr_review.py` |
| PR comment renderer | Pure renderer: trajectory in, markdown out | `pr_comment_renderer.py` |
| Findings | Strict-schema findings artifact builder (two-phase post) | `findings.py` |
| Pricing | Cost synthesis from token counts when backend doesn't report cost | `pricing.py` |
| GitHub App | Scoped installation token minting for bot identity | `github_app.py` |
| Bot setup | One-command App registration, secret deposit, workflow PR | `bot_setup.py` |
| Summarize | Run-info markdown for a trajectory file or run directory | `summarize.py` |
| Archive | Run archival, SQLite index, manifest | `archive/` |
| Training | Corpus pipeline: harvest, reward, bitemporal projection, JSONL export | `training/` |
| Benchmark | `daydream bench` orchestrator, PR acquisition, scoring | `benchmark/` |
| Eval | Deterministic trajectory analysis (cost, grounding, coverage) | `eval/` |
| Prompts | System prompt builder, exploration subagent prompts, CWD grounding instruction | `prompts/` |

### Backend protocol

```python
class Backend(Protocol):
    model: str
    def execute(self, cwd, prompt, output_schema=None, continuation=None,
                agents=None, max_turns=None, read_only=False) -> AsyncIterator[AgentEvent]: ...
    async def cancel(self) -> None: ...
    def format_skill_invocation(self, skill_key: str, args: str = "") -> str: ...
```

Backends yield `AgentEvent` instances (an 8-member union: `TextEvent`,
`ThinkingEvent`, `ToolStartEvent`, `ToolResultEvent`, `CostEvent`, `MetricsEvent`,
`TurnEndEvent`, `ResultEvent`). The `TrajectoryRecorder` consumes this stream and
builds ATIF Steps. Adding a backend means producing this stream correctly; the phases
and recorder are backend-agnostic.

### Run-agent budgets

Every `run_agent()` call is bounded by wall-clock and tool-call limits, tiered by
phase. Budget exhaustion emits a `TurnEndEvent` and marks the trajectory partial.
The default wall budget is 1800s; the default tool-call budget varies by phase.

### Config and per-phase model overrides

`config.py` holds:
- `DEFAULT_CLAUDE_MODEL`, `DEFAULT_CODEX_MODEL`, `DEFAULT_PI_MODEL` constants.
- `PHASE_DEFAULT_MODELS[backend][phase]` Claude/Codex per-phase model tiering;
  Pi resolves its own configured default before falling back to `DEFAULT_PI_MODEL`.
- Skill mappings (`REVIEW_SKILLS`, `SKILL_MAP`).

Per-phase overrides are config-file-only (`[tool.daydream]` /
`[tool.daydream.phases.<phase>]` in `pyproject.toml` or `.daydream.toml`). There are
no per-phase CLI flags. Precedence (highest first): CLI `--model`/`--backend` >
config-file phase override > config-file global > backend default. Resolved in
`runner._resolve_backend()`. `[tool.daydream.phases.<phase>]` accepts any registered
step's config key, including fork-defined phases (per-flow key tables in
`docs/extensions.md`).

### Deep-review pipeline

```text
exploration pre-scan
    -> intent analysis (Sonnet)
    -> alternative review (gated by diff size)
    -> per-stack reviews (parallel, Sonnet; structural review for cross-cutting concerns)
    -> arbiter review (Opus, scoped to high-severity/contested findings)
    -> cross-stack merge (dedup)
    -> recommendation verification (conditional)
    -> fix gate (parallel, batched per-file)
    -> test validation
```

Small diffs short-circuit the fan-out: the multi-stack pipeline is skipped and diff
hunks are inlined directly into a single review prompt.

### Extension seam

A fork customizes phases, skills, and prompts from a top-level `daydream_ext` package
(discovered via `$DAYDREAM_EXT_DIR` → `import daydream_ext`) without editing `daydream/`.
The module exports `DAYDREAM_EXT_API` equal to `EXTENSION_API_VERSION` (currently 1);
`daydream ext validate` resolve-checks the loaded registry. The versioned contract —
name inventories, module shape, bump policy — is `docs/extensions.md`.

## Constraints

- **SDK**: `claude-agent-sdk==0.2.116`. Must stay ≥ 0.2.111: earlier versions tear
  down the CLI subprocess unshielded on the cancellation path, so a budget/fan-out
  cancellation mid-stream corrupts anyio's cancel-scope stack ("Attempted to exit a
  cancel scope that isn't the current tasks's current cancel scope"). Agent capabilities go through the `Backend` /
  `AgentEvent` abstraction; Claude is one of three backends.
- **ATIF**: Vendored from Harbor v0.17.1-9 under `daydream/atif/` (Apache-2.0). Pinned to
  ATIF v1.7 emission. `pydantic>=2.11.7` required.
- **No `harbor` runtime dep.** ATIF models live in `daydream/trajectory.py` only.
- **Module-bloat ban**: No ATIF model construction inside `phases.py` or `ui/`.

## Conventions

- **`make check`** = ruff + `mypy daydream` + pytest. The pre-push hook
  (`scripts/hooks/pre-push`) runs `mypy daydream tests` (includes the tests directory).
  Test-file type errors pass `make check` but fail the pre-push hook.
- Ruff: 120 cols, rules `E F I W`, target py312.
- **Conventional Commits** (`feat(backends): ...`, `fix(agent): ...`). Stage files
  explicitly (`git add <path>`), never `git add -A`.
- **Testing standard**: real-path tests through `runner.run`/CLI, mocking only the
  backend seam. Assert observable outcomes, never "function was called."
- Fix bugs at the root. Never bypass the hook, skip tests, or `git push --no-verify`.

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| `claude-agent-sdk` | 0.2.116 | Claude backend (in-process SDK) |
| `anyio` | >=4.0 | Async runtime, parallel task groups |
| `rich` | >=13.0 | Terminal UI |
| `pyfiglet` | >=1.0 | ASCII art banners |
| `pydantic` | >=2.11.7 | ATIF model validation |
| `PyJWT` | >=2.0 | GitHub App token minting |
| `python-dotenv` | >=1.0 | `.env` loading for `daydream bench` |
| `jsonschema` | >=4.0 | JSON Schema validation |
| `tree-sitter` | 0.25.2 | Static import resolution |
| `tree-sitter-{python,typescript,go,rust}` | various | Language grammars |

## Environment variables

| Variable | Scope | Purpose |
|----------|-------|---------|
| `DAYDREAM_APP_ID` / `DAYDREAM_APP_PRIVATE_KEY` | GitHub App identity | Bot posting under App identity |
| `DAYDREAM_BOT_HANDLE` | GitHub Actions | Bot mention handle (without `@`) |
| `DAYDREAM_AUTO_REVIEW` | GitHub Actions | Set `false` to disable auto-review on PR open |
| `DAYDREAM_PRICES_FILE` | Cost pricing | Override path to `prices.toml` |
| `DAYDREAM_ARCHIVE_DIR` | Archive | Override archive directory |
| `DAYDREAM_SKILLS_DIR` | Pi backend | Override Beagle skill-directory resolution |
| `DAYDREAM_EXT_DIR` | Extensions | Explicit path to the `daydream_ext` package (overrides `import daydream_ext`) |
| `DAYDREAM_GH_TIMEOUT_SECONDS` | Git ops | Override `gh` CLI timeout |
| `DAYDREAM_GH_TIMEOUT_RETRIES` | Git ops | Override `gh` timeout retry count |
| `PI_PROVIDER` / `PI_API_KEY` / `PI_THINKING` | Pi backend | Forwarded as `pi` CLI flags |
| `DAYDREAM_PI_RETRY_ATTEMPTS` / `DAYDREAM_PI_RETRY_BASE_DELAY_S` | Pi backend | Transient retry tuning |
| `CLAUDE_CONFIG_DIR` | Claude backend | Override `~/.claude` directory |
| `MARTIAN_API_KEY` / `MARTIAN_BASE_URL` / `MARTIAN_MODEL` | Benchmark | Judge endpoint and model (`martian` route) |
| `ANTHROPIC_API_KEY` | Benchmark | Direct Anthropic judge (`anthropic-direct` route) |

## Platform requirements

- Python 3.12.13+ (minimum per `pyproject.toml`), uv package manager
- `git` and `gh` on `$PATH`
- Beagle plugin installed in Claude Code (for `beagle-*:review-*` skills)
- `codex` CLI on `$PATH` (only for `--backend codex`)
- `pi` CLI on `$PATH` (only for `--backend pi`)
- Pre-push hook at `scripts/hooks/pre-push`: lint + typecheck + full test suite
- Console script entrypoint: `daydream = "daydream.cli:main"`
