# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project overview

Automated code review and fix loop: reviews diffs, applies fixes, validates via test suite, and
records every agent interaction as an
[ATIF v1.7](https://www.harborframework.com/docs/agents/trajectory-format) trajectory. A bitemporal corpus
pipeline scores, labels, and projects those trajectories into JSONL datasets for SFT/RL fine-tuning.

Default flow is the deep multi-stack pipeline; `--shallow` is a single-stack, single pass; `--comment`/`--review`
are review-only. Four backends — Claude
(in-process SDK), Codex, Pi, and Osprey (subprocess CLIs) — all emit the same `AgentEvent` stream.

Reference docs: `README.md` (user CLI + config), `docs/{extensions,benchmark,observability}.md`.
Review exports follow the [terminal findings contract](README.md#terminal-review-findings-contract); preserve typed coverage, snapshot binding and atomic publication.

## Commands

Use `make install`, then `make hooks`. `make check` runs the required root gate;
the [Makefile](Makefile) defines its focused targets. Run `make rl-check`
separately for changes to the standalone RL package.

```bash
make check # lockcheck + install + lint + deadcode + typecheck + test + actionlint + coverage-report + check-naming (the gate)
```

`daydream /path` (or `daydream review /path`) runs the deep review/fix/test flow.
`--comment` posts reviews; `--review` writes a report; `--shallow` uses one stack.
`daydream improve /path` audits a repository; `improve plan "request" /path`
investigates one plan. See [README.md](README.md) for flags, configuration,
corpus/setup/extensions, and re-anchor commands. Packaged unattended helpers
are documented in [the workflow guide](daydream/templates/workflows/README.md).

## Testing standard (mandatory)

Every user-visible behavior must have at least one **real-path test**: a test that enters from the
production entrypoint (`runner.run` / the CLI) with real dependencies (real temp git worktree, real
filesystem, real event loop), mocking only the external network/API backend (via the `Backend` protocol /
`create_backend` seam). Tests must assert observable outcomes (exit code, files written, fixes applied or
declined, transcript state), never that a function was merely called. Unit tests are supplementary, not a
substitute. Reference exemplar: the non-interactive/EOF gate tests in
`tests/deep_orchestrator/test_fix_gate_cleanup_and_precision.py`.

**No caveats.** All work is completed and proven, or explicitly in progress. No deferred items, no
"optional" follow-ups, no smoke-tests substituted for real coverage.

## Architecture

```text
cli.py -> runner.py -> deep/orchestrator.py -> flows/engine.py (deep FlowSteps)
              |        -> improve/orchestrator.py -> flows/engine.py (improve FlowSteps)
              |        -> flows/engine.py (custom extension flows)
              \-> ui/ (terminal output)
deep FlowSteps -> phases/ -> agent.py -> Backend.execute()
```

- `runner.run()` is the async entry: builds the per-run extension `Registry` onto a `ContextVar`, then
  dispatches PR-process modes (`deep`, `shallow`, `review`, or `comment`) to the single deep
  orchestrator. `improve` and custom extension flows run through `run_flow()` over registered `FlowStep` lists.
- `agent.run_agent()` is the only agent call site. **Never call a backend/SDK directly from phases.**
- Subagent fan-out (exploration, per-stack review, parallel fix) is N parallel `run_agent()` calls under
  `anyio.CapacityLimiter(effective_fanout_concurrency(ceiling, backend))`, **not** SDK `agents=`.
- `TrajectoryRecorder` propagates via `ContextVar`; `recorder.fork()` makes sibling trajectories per fan-out.

### Module responsibilities

| File | Responsibility |
|------|----------------|
| `cli.py`, `commands/` | Process lifecycle and verb dispatch; each command family owns its parser and execution |
| `runner.py` | Workspace lifecycle, registry, and flow dispatch |
| `run_config.py`, `run_artifacts.py` | Run settings and backend precedence; recorder, manifest, and archive lifecycle |
| `run_snapshot.py` | Immutable manifest identity and archive aggregate; runner captures policy before recorder entry |
| `flows/` | `FlowContext` + `run_flow()` engine: ordering, `enabled` gates, `Stop`/`BreakLoop`, loop groups |
| `extensions/` | `Registry` (phases+flows, prompts, stack rules), `daydream_ext` loader |
| `deep/orchestrator.py` | Public deep/diagram entry points, flow assembly, mode gates, diff/stack preamble, and success-only cleanup dispatch |
| `deep/{review,adjudication,merge,diagram,fix}_steps.py` | Review fan-out, record verdicts and arbiter work, publication, grounded diagrams, and authorized fix sequencing |
| `deep/fix_state.py`, `deep/quality_gate.py`, `deep/remote_ci_steps.py` | Fix baselines/capture/confinement, quality checks, and remote CI gates |
| `deep/review_reuse.py` | Reusable-unit identity and shared restore/store bookkeeping |
| `deep/state.py` | Static `DeepData` shape and public extension-input admission for the same `FlowContext.data` dictionary; native producers own internal values |
| `deep/diff.py` | Shared changed-file parsing and full-diff reads, also consumed by scope enforcement and Improve |
| `deep/{detection,dedup,artifacts}.py` | `detect_stacks()` router, artifact paths, dedup pre-filter |
| `deep/records.py` | Host-assigned identity, never content-derived: record `uid` (`stack:ordinal`) at record birth, merged-item `item_uid` (`item:n`) at merge write, and the `source_uids` derivation list |
| `deep/latency.py` | Pure latency-profile vocabulary, risk summary, monotone `route_for`, and the `wonder_decision`/`arbiter_plan` predicates |
| `deep/routing_record.py` | The single writer/reader of `.daydream/deep/latency-routing.json`; write-merged top-level keys, evidence only |
| `deep/arbiter.py` | Scoped Opus pass over high-severity/contested findings |
| `deep/diagram_{types,trigger,schema,render}.py`, `deep/diagram_grounding/` | Grounded diagrams: shared dataclasses, eligibility rules, strict spec schemas, deterministic evidence checking (the sole authority on what may be drawn), pure mermaid emitters |
| `services.py` | The single service-discovery implementation (declared `service_roots` or layout inference), shared by improve and diagram eligibility |
| `improve/orchestrator.py`, `recon.py`, `audit.py` | Flow registration, repository survey/verified commands, and bounded audit/vetting execution |
| `improve/audit_scope.py`, `planning.py` | Partitions, coverage, finding identity, and incremental plan selection |
| `improve/plans.py`, `plan_index.py`, `reanchor.py` | Plan reservation/landing, durable identity/recovery, and re-anchor worktree lifecycle |
| `improve/assemble.py`, `plan_contract.py`, `plan_normalization.py` | Author-content expansion, schema/path/command validation, and safe projection/repair |
| `improve/issue_publication.py`, `reporting.py`, `plan_diagnostics.py` | Validated-plan publication, summaries, and safe diagnostics |
| `phases/` | Review, findings, fixing, testing, and publication operations; `inputs.py` and `schemas.py` own shared preparation/contracts |
| `agent.py`, `agent_retry.py`, `fanout.py`, `ui/agent_stream.py` | Backend budgets and retries, bounded task/recorder lifetimes, and event presentation |
| `trajectory/` | `recorder.py` owns persistence, `scopes.py` owns fork/invocation lifetimes, `invocation.py` handles backend events, and `lifecycle.py` tracks phase/dispatch metadata; timing, layout, and billing stay independent |
| `redaction.py` | Shared credential redaction for backend events, logging, archives, and trajectory text |
| `artifact_visibility.py`, `artifacts/` | Lease acquisition/context binding → live session routing → frozen finalization over owned filesystem, ledger, transfer, publication, and recovery; conflicts preserve competing bytes |
| `observability/` | Operator trace settings, exporter factories, per-run OTel lifecycle, backend-event hydration and export privacy |
| `backends/` | Public protocol/factory; shared `_execution`, `_events`, `_evidence`; provider drivers and Claude transport/guards |
| `ui/` | Rich output (Dracula): `console`, `panels`, `messages`, `tools`, `agent_text`, `summary`, `theme`, `colorize` |
| `config.py`, `config_file.py` | Per-phase model/effort defaults, budgets; `[tool.daydream]` / `.daydream.toml` parser |
| `workspace.py` | `WorkContext`: in-place vs ephemeral detached worktree; private operational worktrees under `~/.daydream/workspaces/` are separate from the runtime artifact root |
| `git_ops/` | **Single Git/GitHub subprocess boundary**: process policy, authentication, queries, mutations, state, and snapshots |
| `exploration*.py`, `tree_sitter_index/` | Pre-scan conventions; safe parser/query runtime, import impact, and statement classification |
| `remote_ci/` | Captured CI evidence, rule parsing, evaluation, GitHub reads, polling, and artifact writes |
| `supervision.py` | Runtime findings + tool supervision (extension veto seam) |
| `reconcile.py` | Cross-run dedup vs prior bot PR comments (GitHub is the store) |
| `pr_review.py`, `reviews/` | Posting orchestration → PR lookup/base capture → classified review submission; shared placement, host-owned identity, rendering, and diagram validation |
| `pr_comment_renderer.py` | Pure renderer: trajectory in, markdown out (no I/O) |
| `training/` vs `eval/` | Corpus pipeline (harvest, reward, projection, JSONL) vs deterministic trajectory analysis; `eval/quality.py` owns source-quality analysis and `eval/latency_report.py` renders the per-profile report |
| `training/harvest.py`, `training/harvest_types.py` | Explicit per-run evidence services and validated immutable inputs; `collect_annotation` shares acquisition with read-only semantic preview, and `build_annotation(row, evidence)` reduces completed evidence without I/O |
| `training/calibration.py` | Fail-closed projection validation, deterministic calibration statistics, `calibration-artifact` emission (`corpus calibrate-reward`) |
| `prompts/` | Authorial intent, exploration subagents, CWD grounding |

Archive hydration separates discovery/contracts, admission checks, and staging in
`archive/hydrate*.py`. GitHub benchmark import separates preflight, transport,
anchors, and evidence in `benchmark/github*.py`. Each facade composes those
operations. Other self-describing modules include findings, pricing, bot setup,
and summaries.

**Latency-profile naming.** The concept is `latency_profile` (config key `latency_profile`, CLI
`--latency-profile`, config-file scalar, `RunConfig.latency_profile`). `forensic` names one profile and
means exactly "today's wonder+arbiter behaviour" (wonder runs at `high`, arbiter at `xhigh`, unsharded);
it is not a synonym for "sharding disabled" or "read-only". Sharding-off is called unsharded/sharding-off.

### Backend protocol

`Backend` (in `backends/__init__.py`) is `model` + `execute()` + `cancel()`.
`execute()` yields the `AgentEvent` union (`Request`, `Text`, `Thinking`, `ToolStart`, `ToolResult`,
`Diagnostic`, `Cost`, `Metrics`, `TurnEnd`, `GenerationStart`, `GenerationEnd`, `Result`). `Request` exposes the effective Daydream-sent request after adapter
transformations. Adding a backend means producing that stream correctly — phases and the
recorder are backend-agnostic. `Diagnostic` is recorder-only parser/transport evidence: the recorder
normalizes and redacts it into JSON-safe `Step.extra.backend_diagnostics`, while observability exports
only scrubbed diagnostic codes and counts on the enclosing attempt span.
Codex transport coverage is necessarily evidence-driven: a recognized public `item.type == "error"`
sentinel produces an incomplete-coverage diagnostic, but if a transport omits a tool and emits no public
marker, Daydream cannot infer the invisible call. Never synthesize a hidden tool pair or emit an
invocation-wide coverage warning without public evidence. The committed generic-tool fixture is a
sanitized public capture with separate raw/sanitized digests; its maintenance script creates an
unpublished public-only candidate that requires exact-run private corroboration and separate review.

Observability is activated only by operator CLI/environment settings. The target's file config never
selects destinations or credentials. A run owns its OTel provider and OpenLLMetry 0.62.3 processors;
never install a global provider or call `Traceloop.init()`. Span ownership is run → executed flow step →
logical agent → backend attempt → tool/generation. Frozen billing ownership assigns usage to either
the structural attempt or its sealed generation children. Native turn boundaries differ by backend,
so do not infer per-provider-call usage from `TurnEndEvent`. Preserve available
terminal metrics before raising a backend error. Exporters register through
`Registry.register_trace_exporter`; registration/validation must not construct exporters.
Full contract and operator recipes: `docs/observability.md` and `docs/extensions.md`.

The Claude backend enforces two always-on `PreToolUse` guards in every phase and profile: the
dangerous-command guard (root-anchored scans, `rm -rf /`) and the background-Bash guard. The host reads a
turn's final text as the phase result and stops consuming the session when the turn ends, at which point
the CLI kills its background tasks — so `Bash(run_in_background=True)` can never report and is denied. The
CLI subprocess env also sets `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` and lifts the Bash timeout ceiling to
`TEST_WALL_BUDGET_S` so a slow test suite has no reason to be backgrounded.

### Run-agent budgets

| Bound | Value | Scope |
|-------|-------|-------|
| `DEFAULT_WALL_BUDGET_S` | 1800s | every `run_agent()` turn (improve phases deliberately unbounded) |
| `TEST_WALL_BUDGET_S` | 3600s | the test run — bounds the *target repo's* suite, not an LLM tail |
| `DEFAULT_TOOL_CALL_BUDGET` | `None` | unlimited; per-call `tool_call_budget` still accepted |
| `DEFAULT_GROUP_MAX_WALL_S` / `_SERIAL_ITEMS` | 600s / 6 | cumulative over all fix calls for one file group |
| `EXPLORATION_MAX_TURNS` | 50 | exploration specialists — the only `max_turns` call site |
| `DAYDREAM_STREAM_IDLE_TIMEOUT_S` | 600s / 2700s | pi response silence / active-tool and codex silence before the subprocess is killed |

- Exhaustion emits a `TurnEndEvent` and marks the trajectory partial. **Truncation is never silently
  absorbed**: a truncated wonder or parse raises; a truncated per-stack review goes to `failed_stacks` so
  merge lists it under "Uncovered stacks" instead of recording a clean pass.
- Do not add `max_turns` to fix or verify — it does not fail soft. The turn ends `error_max_turns`, the
  backend raises `MaxTurnsError`, and the fix group lands in `fix-failures.json` and is reverted, throwing
  a real fix away rather than trimming it.
- `run_agent` retries retryable backend errors with exponential backoff, up to **20 attempts**, all backends
  (`DAYDREAM_PI_RETRY_ATTEMPTS` overrides); a stream stall gets one fresh attempt. Never retried:
  tool-supervisor veto, non-transport logic error. A stall fires only on the *absence* of output, never
  on slow output; pi keeps the long window while a tool is active.

### Config and per-phase model overrides

`config.py` holds `DEFAULT_{CLAUDE,CODEX,PI,EXPLORATION}_MODEL`, `PHASE_DEFAULT_MODELS[backend][phase]`,
`PHASE_DEFAULT_EFFORT` (deep/review half Codex-only; improve half all three backends), budget constants, and improve `EFFORT_TIERS`. Pi
resolves its own configured default before falling back to `DEFAULT_PI_MODEL`. Effort is not a pure table
lookup: the run's latency route sits below the explicit user knobs and above `PHASE_DEFAULT_EFFORT` for
`wonder`/`arbiter`, and each run records the decision in `.daydream/deep/latency-routing.json`.

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
    -> load per-stack structured records
    -> arbiter review (Opus, scoped to high-severity/contested findings)
    -> cross-stack merge (dedup; resumes the arbiter's session)
    -> diagrams (conditional, grounded; sequence and/or flowchart)
    -> recommendation verification (conditional)
    -> fix gate (parallel, batched per-file)
    -> test validation
```

- The fix/verify loop has at most three rounds. Remaining `unresolved` or
  `wrong_target` findings are reported honestly and do not prevent the retained
  patch from proceeding through tests, commit, and push. A `regressed` verdict
  stops publication. Post-test re-verification permits existing unresolved
  findings but stops newly actionable findings; tree identity, scope, tests, and
  repository hooks remain required. Commit messages list only resolved findings.
- Wonder ∥ per-stack are siblings in one task group on a fresh multi-stack run (wonder feeds only merge and
  the dedup pre-filter, so reviewer prompts drop the `alternatives.json` pointer; they join before parse).
  Single-stack mode and every `--start-at` resume keep the serial order **and** the pointer — single-stack
  has no merge agent, so that pointer is the only path wonder findings take into the report. This boundary
  is why the extension API is v4.
- Reviewers return structured records, loaded in **stack-name order** to keep merge input ordering
  and global issue numbering reproducible.
- **Identity is host-assigned.** Per-stack `uid` is `stack:ordinal`, stamped at
  birth and backfilled deterministically for old artifacts. The reviewer's `id`
  restarts per stack; it is not unique. Dedup, arbitration, suppression, and the
  structural fold key on `uid`, never a content fingerprint. Content keys answer
  "same defect?", not "which record?".
- Merge re-emits items. Its `source_uids` provenance is validated against the
  run's pool; unknown values are dropped with a warning. Use `record_uid()` and
  `item_source_uids()`; empty results mean no pre-merge identity.
- **Merged-item identity is `item_uid`.** `normalize_items` assigns dense integer
  `id` values for display on every call; it mints durable `item_uid` once.
  Structural/single-stack items may also retain record `uid`. `source_uids` is
  provenance, never a unique item key. Keep echoed `id`/`issue_id` integer to
  satisfy the strict schemas and finding-number renderer.
- Pi discovery always references the actual live session `diff.patch`, including small diffs. It uses
  ordinary read-only tools and separate structural review, never preloaded finite evidence packets.
  The pointer-only diff has a separate 128 MiB validation ceiling; other sanctioned-input budgets stay
  unchanged. Recovery finalization uses completed investigation evidence and never recaptures raw diff.
  Other backends' intent, wonder and per-stack prompts keep their bounded 12 KiB inline policy and
  transport-specific isolation. Small diffs can still collapse the fan-out.
- Pi sends normal logical prompts plus schema through private temporary `@file` attachments. Dynamic
  review system instructions use Pi's system-prompt file support. Files survive until child teardown
  and are removed on success, failure, cancellation or generator close; tools-disabled calls retain stdin.
- Merge resumes the arbiter's session when both phases resolve to the same backend instance; the resumed
  prompt forces a re-read of the per-stack record files, rewritten after arbitration.
- **Diagrams:** the LLM emits strict evidence-bearing JSON, never Mermaid.
  `deep/diagram_grounding/` alone validates confined paths, existence, line range,
  symbol (±3 snap), tree-sitter kind, and definitions. One repair turn precedes
  dependent pruning, render caps, and omission floors, in that order. Rebuild
  `spec_final` against the strict schema before privileged posting.
- Eligibility re-detects stacks rather than using tiny-diff-collapsed state.
  Write `diagram.json` and `diagram.md`, with no agent call when ineligible.
  Per-kind failure warns and continues ordinary reviews; diagram-only writes
  the artifact and exits 1. Prompts use resolved sanctioned-input transport:
  INLINE includes bounded diff/context with truncation markers and no private
  paths; EXACT_PATHS uses budget-checked references.
- Diagram-only runs exploration → diagram → post-diagram. Preserve prior deep
  artifacts, skip fresh-run cleanup, and write no diff-key. Label the manifest
  `DaydreamRunFlow.DIAGRAM`, never `TTT`, so surviving merged items are not
  inherited as this run's state.
- `.daydream/exploration/` survives the run, reused only on an **exact** key match (format version + head SHA + diff + tier,
  in a sibling `cache-key` file) — a near-match hit would misground every prompt. Uncommitted edits
  are not in the key, so an exact hit on a dirty tree can serve pre-edit exploration. `--shallow`/`--review`
  delete the directory, so alternating modes always miss.
- `--start-at` refuses stale artifacts: a fresh run records its diff in `.daydream/deep/diff-key`; a resume
  whose diff no longer matches (or has no key) exits 1 rather than adjudicating stale findings.

### Extension seam

A fork customizes phases, flows, and prompts from a top-level `daydream_ext` package (found via
`$DAYDREAM_EXT_DIR` → `import daydream_ext`) without editing `daydream/`. It must export
`DAYDREAM_EXT_API` within `MIN_SUPPORTED_EXTENSION_API_VERSION..EXTENSION_API_VERSION` (both 6), may
register one `ToolDecision`-returning tool supervisor, and is resolve-checked by `daydream ext validate`.
Full contract: `docs/extensions.md`.

## Constraints and conventions

- **SDK** is pinned in `pyproject.toml` and must stay ≥ 0.2.111: earlier versions tear down the CLI subprocess
  unshielded on cancellation, so a budget/fan-out cancel mid-stream corrupts anyio's cancel-scope stack.
- **ATIF** vendored from Harbor v0.17.1-9 under `daydream/atif/` (Apache-2.0), pinned to v1.7 emission.
  Re-vendor wholesale on Harbor updates; no local patches. **No `harbor` runtime dep** — ATIF models live in
  `daydream/trajectory/` only. **Module-bloat ban**: no ATIF construction in `phases/` or `ui/`.
- Deps live in `pyproject.toml`; keep `uv.lock` in sync via `uv lock` or `make check` fails at step one.
- **`make check`** is the [Makefile](Makefile) root gate; pre-push verifies
  signatures then runs it. RL uses a separate CI job and `make rl-check`; run
  that gate when changing `rl/daydream_review`.
- Ruff: 120 cols, `E F I W`, py312. `daydream/atif/**` is lint-exempt (vendored, mechanical edits only).
- Root `.editorconfig` declares editor-side defaults (UTF-8/LF/final newline,
  4-space Python, 2-space YAML, 4-space TOML, Makefile tabs, `*.md` trailing-whitespace
  preserved); the `daydream/atif/**` carve-out declares no indent/trim keys
  (a section can add or override inherited keys but never unset them, so
  `[*.py]` 4-space rules still apply to vendored `.py` files).
- **Artifact boundary**: resolve the private owner once from the canonical source; pass that same
  owner and the active artifact session through workspace, recorder, and flow composition. Route
  generated files through the session. Join all writers before freezing immutable evidence;
  archive/evaluate/publish only after that boundary. Never reconstruct public output paths or
  broaden backend read roots.
- **Conventional Commits** (`feat(backends): ...`). Stage explicitly (`git add <path>`), never `git add -A`.
- Fix bugs at the root. Never bypass the hook, skip tests, or `git push --no-verify`.
- Own your own bugs in plain language. Never describe your defect as the tool being buggy.
- Never claim success that isn't verified-working.

## Environment variables

| Variable | Scope | Purpose |
|----------|-------|---------|
| `DAYDREAM_APP_ID` / `DAYDREAM_APP_PRIVATE_KEY` | GitHub App | Bot identity (PEM **content**, not a path) |
| `DAYDREAM_BOT_HANDLE` | Actions | Mention handle (no `@`) used by the three PR-comment commands: `@<bot> review`, `@<bot> add sequence diagram` (alias `@<bot> add sequence`), `@<bot> add flowchart` |
| `DAYDREAM_EXT_DIR` | Extensions | Path to `daydream_ext` (overrides `import daydream_ext`) |
| `DAYDREAM_GH_TIMEOUT_SECONDS` / `_RETRIES` | Git ops | `gh` CLI timeout and retry count |
| `DAYDREAM_GIT_TOKEN` | Harvest | Optional out-of-band auth for sanitized repo clones during `corpus harvest` of private repos (e.g. a GitHub PAT); injected via git config environment variables (`http.extraHeader`), **never embedded in the remote URL** and **never on the command line**. Without it, plain clone via the ambient credential helper |
| `DAYDREAM_TRAJECTORY_HUB_REPO` | Archive | Optional HuggingFace dataset repo to upload each run's bundle to; one of the two operator sources (the other is the CLI `--trajectory-hub-repo` flag). The target checkout's file config is ignored for this |
| `DAYDREAM_TRACE_TO` | Observability | Comma-separated trace destinations; empty/unset disables tracing. `--trace-to` overrides this list; `--no-tracing` disables tracing |
| `DAYDREAM_TRACE_CONTENT` | Observability | Content policy: `full` (default) or `metadata`; overridden by `--trace-content`. Does not enable tracing by itself |
| `PI_PROVIDER` / `PI_THINKING` | Pi | `--provider` / `--thinking`; `PI_THINKING` loses to a per-phase `reasoning_effort` |
| `PI_API_KEY` | Pi | Copied into the child's provider-native var (e.g. `ZAI_API_KEY`), **never onto argv**; warns and ignores if the provider has no mapped var |
| `DAYDREAM_PI_RETRY_ATTEMPTS` / `_BASE_DELAY_S` / `_MAX_DELAY_S` | Retry | Attempts default 20, all backends |
| `DAYDREAM_FANOUT_CONCURRENCY` | Claude / Codex | Parallel `execute()` hint (default 8; bad value warns). Pi uses `DAYDREAM_PI_FANOUT_CONCURRENCY` (default 10) |
| `DAYDREAM_STREAM_IDLE_TIMEOUT_S` | Pi / Codex | Stdout-silence kill (pi response default 600; active-tool/codex default 2700; `0` disables) |
| `DAYDREAM_REVIEW_API_KEY` / `DAYDREAM_JUDGE_API_KEY` | Benchmark | Harbor reviewer/judge OpenRouter keys; read from the launching env, never written into the workspace or task. `daydream/benchmark/harbor/env_policy.py` is the single declaration of which environment names are let through and which are scrubbed. |

Plain path overrides: `DAYDREAM_PRICES_FILE`, `DAYDREAM_ARCHIVE_DIR`, `PI_CODING_AGENT_DIR` (`~/.pi/agent`), `CLAUDE_CONFIG_DIR` (`~/.claude`).

## Platform requirements

Python ≥3.12.13 + uv; `git` and `gh` on `$PATH`; `codex`/`pi` CLIs only for their backends; pre-push hook via `make hooks`.
