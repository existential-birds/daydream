# Pitfalls Research

**Domain:** ATIF v1.4 trajectory emission for an existing Python CLI agent (daydream)
**Researched:** 2026-04-26
**Confidence:** HIGH for items grounded in the ATIF spec (`docs/reference/atif_format.md`) and the daydream codebase; MEDIUM for cross-tool conventions inferred from Harbor's documented validator behavior.

This document is opinionated and daydream-specific. Generic Python advice is excluded. Each pitfall includes warning signs, a concrete prevention strategy, the phase that should address it, and a severity. Severity is calibrated against the milestone's Active requirements in `.planning/PROJECT.md`.

Phase shorthand used throughout (these are research recommendations, not committed phases):
- **P1 — Recorder core**: data model, builder, write path, schema validation harness
- **P2 — Backend wiring**: event-to-step mapping, token extraction, hierarchical subagent recording, Codex parity
- **P3 — Cutover & UX**: removing `_log_debug`, CLI surface change, README, redaction polish

---

## Critical Pitfalls

### Pitfall 1: Non-sequential `step_id` from concurrent / nested `run_agent()` calls

**What goes wrong:**
Harbor's validator rejects trajectories where `step_id` is not `1, 2, 3, …` in order. Daydream calls `run_agent()` many times per run (review → parse → fix → test, plus parallel `phase_fix_parallel` and deep mode's `anyio.CapacityLimiter(4)` fan-out). If each call assigns step IDs from a local counter, IDs collide; if a global counter is updated from concurrent tasks without a lock, IDs go non-sequential or duplicate.

**Why it happens:**
Reviewers reach for "every `run_agent` call is its own trajectory section" thinking, and forget that ATIF requires the *root-level* `steps` array to be sequential from 1. Subagent steps in the hierarchical model are allowed, but daydream's parallel fix loop and deep fan-out aren't true subagents — they're peers in the same trajectory.

**How to avoid:**
- Single `TrajectoryBuilder` per run, owning a monotonic `step_id` counter behind an `anyio.Lock` (or a sync lock since builder mutations are short and CPU-bound).
- Steps are appended in-order at *event consumption time*, not at `run_agent()` invocation time — so even if two `run_agent()` calls run concurrently in `phase_fix_parallel`, the recorder serializes step assignment when their events are merged.
- Decision required upfront: is each `run_agent()` call a flat sequence of steps in the parent, a `subagent_trajectory` field, or a separate trajectory file? Per `.planning/PROJECT.md` "Trajectory granularity = one per daydream run", default to one trajectory with hierarchical subagents for true delegations and flat-append for orchestration calls.

**Warning signs:**
- Validator error: `trajectory.steps.N.step_id: expected M (sequential from 1), got X`
- A unit test that runs two `run_agent` calls concurrently against a mock backend produces a trajectory whose `step_id` values aren't `1..N`.
- Deep mode with `--deep` produces a trajectory but the rendered viewer shows steps out of order.

**Phase to address:** P1 (data model + counter). Verified again in P2 once parallel paths are wired.

**Severity:** HIGH

---

### Pitfall 2: ISO 8601 timestamp format mistakes (naive datetimes, `datetime.utcnow()`, drifting clocks across event types)

**What goes wrong:**
ATIF requires ISO 8601 timestamps. Harbor's validator accepts `2025-01-15T10:30:00Z` and `2025-01-15T10:30:00+00:00`; it rejects naive datetimes serialized without offset, `1737000000` epoch ints, or formats produced by `str(datetime.utcnow())` (no `T`, no offset). Worse, `datetime.utcnow()` returns naive datetimes — using `.isoformat()` on them silently produces a timezone-less string that *looks* valid but isn't.

**Why it happens:**
- `datetime.utcnow()` is the obvious-looking call but is deprecated in Python 3.12 and produces naive datetimes.
- Codex backend events arrive on a JSONL stream from a subprocess; if you stamp them at decode time you may differ from the timestamp the upstream `codex` CLI would have stamped them with.
- Wall-clock timestamps recorded *after* a retry can appear to go backwards if the system clock jumps (NTP correction, suspend/resume on a laptop).

**How to avoid:**
- Single helper: `now_iso() -> str` that returns `datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")`. Use **only** this helper everywhere; ban `datetime.utcnow` and bare `datetime.now()` via a ruff custom rule or a code review checklist item.
- Stamp the timestamp at the *event consumption boundary* in `agent.py:run_agent()`, not inside the backend. This guarantees a single monotonic source.
- Use `time.monotonic()` for *durations* (Phase 7-8 metrics) but never for the trajectory `timestamp` field.
- Fixture the timestamp helper in tests so trajectory tests are deterministic.

**Warning signs:**
- Validator error referencing `timestamp` on any step.
- Trajectory diffs between two test runs differ only in `timestamp` and tests are flaky.
- A step in a long retry loop has a timestamp earlier than the prior step (clock jump).

**Phase to address:** P1.

**Severity:** HIGH

---

### Pitfall 3: Dangling `source_call_id` (observation references a `tool_call_id` that doesn't exist in any prior step)

**What goes wrong:**
ATIF correlates `Observation.results[*].source_call_id` to a `ToolCall.tool_call_id` from the agent step. Harbor's validator explicitly checks "tool call references in observations." Failure modes:
1. A tool errors out *before* emitting a `ToolStartEvent` (Codex `command_execution` declined by sandbox can yield a `ToolResultEvent` whose paired start was never emitted — see `daydream/backends/codex.py:261`).
2. `ToolResultEvent` arrives for an ID we already removed from the in-flight map (`tool_registry.remove()` in `agent.py:396`).
3. Subagent tool calls leak into the parent's `observation` field because the recorder doesn't know which agent context owns the tool ID.
4. Codex generates UUIDs for missing IDs at `item.completed` time when the lookup map miss has already occurred (`backends/codex.py:255-256`, `296-297`) — these synthetic IDs have no matching `tool_call_id`.

**Why it happens:**
The unified `AgentEvent` stream collapses Codex's two-phase ID generation into a single ID, but daydream's recorder must trust whatever ID arrives. When Codex's `pending_item_ids` map misses (the `[CODEX_WARN]` log line), the result event gets a fresh UUID with no preceding start event.

**How to avoid:**
- Recorder maintains an `in_flight: dict[tool_call_id, step_index]` map. On `ToolResultEvent`, look up; if missing, *do not* attach to a random step — instead attach to the most-recent agent step's observation and log an `extra.unmatched_tool_id=true` warning rather than emitting a dangling reference.
- Better: change the Codex backend to emit a synthetic `ToolStartEvent` whenever it generates a UUID at completion time. This eliminates the dangling case at the source rather than papering over it in the recorder.
- For subagent isolation (P2): each `run_agent()` call gets its own `in_flight` map so tool IDs from subagents never get confused with the parent's.
- Validate during construction by passing the trajectory through Harbor's Pydantic models before write — Pydantic should refuse, but Harbor's validator does the cross-reference check; run it in tests.

**Warning signs:**
- Test failure: `ObservationResult.source_call_id 'xxx' has no matching ToolCall in any prior agent step`.
- The trajectory contains a `tool_call_id` that doesn't appear elsewhere in the file.
- `[CODEX_WARN] pending ID lookup miss` is currently logged in the legacy debug log; that warning is the *symptom* of this pitfall.

**Phase to address:** P2 (event-to-step mapping). Mitigation in CodexBackend.

**Severity:** HIGH

---

### Pitfall 4: Agent-only fields on user/system steps (validator hard-fail)

**What goes wrong:**
ATIF separates user, system, and agent steps. The validator "ensures agent-only fields are only present on agent steps" (per `docs/reference/atif_format.md` line 296). User steps must not carry `tool_calls`, `observation`, `metrics`, `model_name`, or `reasoning_content`.

The pitfall: daydream maps `[PROMPT]` to a user step but the obvious refactor is "build a `Step` once and fill in fields as events arrive." If the same builder defaults `tool_calls=[]` instead of `None` on a user step, the resulting model may serialize with an empty list — depending on Pydantic field config, this fails validation.

**Why it happens:**
Convenience: a single `StepBuilder` class for both step types. Empty-list defaults are a Python idiom that backfires when the validator distinguishes "field absent" from "field is empty."

**How to avoid:**
- Use Harbor's typed `Step` model directly with explicit `source="user"|"agent"|"system"`. Pass only the fields applicable to that source. Never construct user steps via the same builder path that fills tool/metrics fields.
- Use `to_json_dict()` (excludes `None` by default per spec) and verify the output for user steps contains exactly `step_id`, `timestamp`, `source`, `message`, plus optional `extra`.
- Snapshot test for a minimal user step verifying the JSON has *exactly* the keys above — catches drift if a future refactor leaks agent-only fields.

**Warning signs:**
- Validator error: `agent-only field 'tool_calls' is not allowed on a user step`.
- Trajectory JSON has user steps with `"tool_calls": []` or `"metrics": null`.

**Phase to address:** P1.

**Severity:** HIGH

---

### Pitfall 5: Token accounting — running totals vs. per-step deltas confusion (claude-agent-sdk specific)

**What goes wrong:**
Claude SDK `ResultMessage.total_cost_usd` is a *running total* (cumulative across the entire request). Per the OpenHands example in the ATIF docs (line 71: "Converts accumulated metrics to per-step deltas"), ATIF expects per-step `Metrics` to be deltas, not running totals. Failure modes:
1. Per-step `cost_usd` is the running total → `FinalMetrics.total_cost_usd ≠ sum(steps.metrics.cost_usd)` → either validator complains or downstream training pipelines double-count.
2. `cached_tokens` are double-counted against `prompt_tokens` (Claude SDK reports them separately, but if the recorder adds them to prompt_tokens it inflates totals).
3. Multi-turn sessions: the second turn's `ResultMessage` reports cumulative-since-session-start, not cumulative-since-last-result.

The current `daydream/backends/claude.py:122-128` extracts only `total_cost_usd` and yields `input_tokens=None, output_tokens=None`. The Active requirements call this out: "Claude backend extracts `prompt_tokens` / `completion_tokens` / `cached_tokens` from SDK `ResultMessage` (currently always `None`)."

**Why it happens:**
The Claude SDK's `ResultMessage.usage` field is structured but its semantics aren't documented as "delta vs. cumulative" in the SDK docs. Engineers infer from one test run and ship.

**How to avoid:**
- Spec test: in P2, write a fixture-based test that sends two prompts in one session, confirms the second `ResultMessage` reports cumulative tokens, and asserts the recorder converts to a delta by subtracting the prior cumulative.
- Track `last_seen_cumulative_tokens` per session in the recorder; emit `Metrics(prompt_tokens=now-last, ...)` on each `ResultEvent`.
- For `cached_tokens`: ATIF treats `cached_tokens` as a *subset* of `prompt_tokens` (cached portion of the prompt), not additive. Pass through whatever Claude reports as `cache_read_input_tokens` (or the SDK equivalent) directly to `Metrics.cached_tokens`; do not add it to `prompt_tokens`.
- Reconcile `FinalMetrics` against `sum(steps[*].metrics)` in a recorder-level `finalize()` method and assert they match; if not, log an `extra.token_reconciliation_drift` field rather than silently rebuilding.

**Warning signs:**
- `FinalMetrics.total_cost_usd > sum(step.metrics.cost_usd)` consistently — running totals leaked.
- `total_prompt_tokens` grows quadratically with turn count — running totals double-counted.
- A trajectory with 10 turns reports the same cost on every step — running total assigned per-step without diff.

**Phase to address:** P2.

**Severity:** HIGH

---

### Pitfall 6: Missing token coverage from CodexBackend (parity gap)

**What goes wrong:**
`CodexBackend` (`daydream/backends/codex.py:308-314`) emits `CostEvent(cost_usd=None, input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))` from `turn.completed.usage`. There is no `cached_tokens` extraction, no per-step cost (Codex CLI does not emit USD cost — only token counts). Resulting Codex trajectories will have `Metrics.cost_usd=None` everywhere and `FinalMetrics.total_cost_usd=0` or `None`.

**Why it happens:**
The two backends have different operational realities: Claude bills by USD and reports it; Codex reports tokens but cost computation is the consumer's job. ATIF's `Metrics.cost_usd` is optional, but if some daydream runs have it and some don't, dashboards comparing runs across backends will be misleading.

**How to avoid:**
- Decision (log in PROJECT.md decisions table): for Codex runs, leave `cost_usd=None` on `Metrics` and `total_cost_usd=None` on `FinalMetrics`. Do **not** synthesize a cost from a token-price table — pricing changes and the trajectory becomes wrong over time.
- Document this in README: "Codex trajectories carry token counts but not USD cost; consumers can compute cost from tokens using current pricing."
- Add a parity test: run the same prompt against both backends (with mocks/recorded fixtures) and verify both produce schema-valid trajectories, even though the cost fields differ.
- For `reasoning_content`: Codex `reasoning` items map cleanly to ATIF's `reasoning_content`. Claude `ThinkingBlock` also maps. Verify both backends produce equivalent `reasoning_content` shapes — this is currently the only field where Codex actually has *more* data than Claude (Codex emits incremental reasoning deltas; Claude emits a single thinking block).

**Warning signs:**
- A run with `--backend codex` produces a trajectory where every `metrics.cost_usd` is missing while a Claude run of the same prompt reports it.
- `FinalMetrics.total_steps` matches but token totals are zero on Codex.

**Phase to address:** P2 (token extraction in both backends).

**Severity:** MEDIUM

---

### Pitfall 7: Subagent / hierarchical trajectory mistakes — flattening vs. nesting

**What goes wrong:**
ATIF supports subagent delegation. Daydream has three places where this distinction matters and they need different treatment:
1. **Phase orchestration** (review → parse → fix): conceptually one session, *not* subagents. Should be flat steps in the root trajectory.
2. **Exploration specialists** (`exploration_runner.py`): currently invoked via `agents=...` parameter to `run_agent()` (the Claude SDK's `AgentDefinition` mechanism). These ARE subagents; should be nested.
3. **Deep mode parallel per-stack reviews** (`anyio.CapacityLimiter(4)`): peer agents running in parallel on different stacks. Conceptually parallel subagents, not nested. ATIF v1.4's hierarchical model requires picking one — and parallel siblings under one parent is the correct shape.

Failure modes:
- Flattening exploration specialists into the parent's step stream → `step_id` chronology lies (the parent agent didn't actually do those tool calls).
- Nesting phase orchestration → trajectory becomes deeply nested for no reason and replay tooling has to walk the tree.
- Race condition: two parallel deep-mode subagents both append to the parent's step list without a lock → step_id collision (Pitfall 1, recurring).
- Parent metrics polluted by subagent costs: if subagent `Metrics` are added to parent steps, the sum doesn't match — `FinalMetrics` should aggregate across the whole tree, but per-step metrics on the parent should *exclude* subagent costs.

**Why it happens:**
ATIF's subagent model is documented in the spec but daydream's existing exploration/deep-mode/phase machinery wasn't built with this distinction in mind. The path of least resistance is "append every event to one flat list."

**How to avoid:**
- Decision matrix, recorded in PROJECT.md decisions:

| Daydream construct | ATIF mapping | Rationale |
|---|---|---|
| Top-level `run_agent()` calls in `runner.py` flow (review → parse → fix → test) | Flat steps in root trajectory | Same logical session, different prompts to the same agent |
| `run_agent(agents=...)` exploration specialists | `subagent_trajectory` field on the dispatching agent step | True delegation; specialists are distinct agents with their own model/identity |
| `phase_fix_parallel` peers | Flat steps with `extra.fix_branch=N` | Same agent, parallelized; not delegation |
| Deep mode per-stack reviews | Sibling subagent trajectories under a parent "deep orchestrator" agent step | Distinct tasks running in parallel; preserves per-stack accounting |

- Per-context isolation: each `run_agent()` invocation gets its own builder context (its own `in_flight` map, its own local step counter that gets reconciled to the global counter when flushed).
- Lock around step append (`anyio.Lock`) — parallel paths must serialize their step writes.
- `FinalMetrics` aggregator walks the tree (root + subagent_trajectory recursively) and sums; per-step metrics stay scoped to their own agent.

**Warning signs:**
- Step count in deep-mode trajectory equals total events across all stacks (suggests flattening).
- `FinalMetrics.total_steps` doesn't equal `len(root.steps)` AND there are no `subagent_trajectory` entries.
- `phase_fix_parallel` produces non-deterministic step ordering across runs (race on step_id).

**Phase to address:** P2 (after P1 builder lands; before deep-mode integration). Add tests in deep-mode test suite.

**Severity:** HIGH

---

### Pitfall 8: Privacy / secret leaking in `tool_calls.arguments` and `observation.content`

**What goes wrong:**
Trajectory writes capture **everything** the agent saw and did. Daydream's bypass-permissions tool surface (`Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`, plus Codex `command_execution`) means tool args and outputs may contain:
- Bash commands with inline `$ANTHROPIC_API_KEY` / `$OPENAI_API_KEY` references *or expanded values* — Codex `aggregated_output` includes stderr, where `env` dumps land.
- File paths with usernames (`/Users/ka/...`) — leaks identity.
- `gh auth token` / `git config --get-all` output if an agent ever runs them.
- Read tool output of `.env`, `~/.claude/settings.json` (which contains the user's Anthropic API key), `~/.netrc`, `~/.ssh/config`.
- Repository content that itself contains secrets (the `.review-debug-{ts}.log` file has no redaction today, but it's local-only; trajectory files become artifacts users may share, post to issues, or upload to Harbor/SFT pipelines).

**Why it happens:**
The current `_log_debug` system is local-debug-only (off by default, written under the target directory). ATIF trajectories are *always-on* per the milestone (`No --debug opt-in; output path controllable via --trajectory`), and the explicit goal is interoperability with Harbor / training pipelines. Always-on + always-shareable + raw bash output = secret exposure waiting to happen.

**How to avoid:**
- Concrete redaction policy implemented in the recorder before serializing:

| Surface | Action |
|---|---|
| `Read` tool args path matching `~/.claude/settings.json`, `.env*`, `~/.netrc`, `~/.ssh/*`, `~/.aws/credentials`, `~/.config/gh/hosts.yml` | Replace `arguments.file_path` with `<REDACTED:secret-file>`, drop `observation.content` |
| `Bash` / `command_execution` containing `gh auth token`, `aws sts get-session-token`, `printenv`, `env` (bare), `cat .env`, `cat ~/.netrc` | Replace `arguments.command` and `observation.content` with `<REDACTED:env-dump>` |
| `observation.content` matching token regex: `sk-[A-Za-z0-9_-]{20,}`, `ghp_[A-Za-z0-9]{36}`, `xoxb-`, `AKIA[0-9A-Z]{16}`, `eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+` (JWT) | Replace match with `<REDACTED:token>` |
| Absolute paths under `/Users/<name>/`, `/home/<name>/`, `C:\Users\<name>\` | Replace username segment with `<USER>` (one-way; not reversible) |
| Email addresses in `observation.content` | Replace with `<REDACTED:email>` |
| Git remote URLs containing `:user:token@` | Strip credential prefix |

- Implementation: a `Redactor` class with a list of `(matcher, replacer)` rules, applied in the recorder's `serialize()` step (not at event-yield time — keeping raw events available for in-process tool panels means UI shows the real path while the trajectory file stores the redacted version).
- Redact before validation, not after — Harbor's validator sees the redacted form.
- Provide `--no-redact` (off by default) escape hatch for local debugging only; document that `--no-redact` trajectories must not be shared.
- Test: a fixture trajectory with seeded API keys/paths must produce a redacted output that the same regex set finds nothing in.
- Document in README: "Trajectories are redacted by default; review the file before sharing."

**Warning signs:**
- A grep over a published trajectory finds `sk-`, `ghp_`, or the user's home directory name.
- An issue report includes a trajectory file with embedded credentials.
- A test trajectory contains `/Users/ka/` (specific to your dev box).

**Phase to address:** P3 (must land before the milestone is complete; same release as the cutover).

**Severity:** HIGH

---

### Pitfall 9: Batch write loses everything on crash / SIGKILL during long runs

**What goes wrong:**
PROJECT.md scopes trajectory recording as batch-write at run completion ("trajectory is built in memory and written at run completion"). Failure modes:
1. Crash mid-run (uncaught exception in `phase_fix`) → no trajectory at all.
2. SIGKILL from OS (OOM killer, user `kill -9`, laptop sleep at the wrong moment) → no trajectory.
3. Long run (deep mode against a large monorepo, 30+ minutes, hundreds of events) builds up memory holding every tool result string. Codex `command_execution` `aggregated_output` can be megabytes; multiplied across 100+ steps and 4 parallel stacks = potential 100s of MB resident.
4. `SIGINT` from user — daydream has signal handlers (`cli.py:_signal_handler`) for graceful shutdown; if those handlers don't trigger trajectory write, Ctrl-C produces nothing.

**Why it happens:**
Batch-write is the documented choice (correct per ATIF: "ATIF expects a single coherent document"). The issue is *partial-coverage*: most failure modes are exception paths, not happy paths.

**How to avoid:**
- Wire the SIGINT/SIGTERM handler in `cli.py` to flush the in-progress trajectory to a `.trajectory.partial.json` file before exit. Use `extra.partial=true` on the trajectory object so consumers know to handle it leniently.
- Wrap `runner.run()` in a try/finally that always writes the trajectory (success → `.trajectory.json`; failure → `.trajectory.partial.json`). Never raise after writing — the original exception still propagates.
- For the very-long-run memory case: cap stored `observation.content` at 256 KB per result (truncate with `<TRUNCATED:N bytes>` suffix); the full content is rarely needed for replay/training, and the `.review-output.md` artifact already captures the agent's full review output separately.
- Stream-write is *out of scope* per PROJECT.md — don't reintroduce it. The truncation cap is the relief valve.
- Validate partial trajectories: schema validation must accept a trajectory with no `final_metrics` (it's optional in ATIF). Test this explicitly.

**Warning signs:**
- A user reports "Ctrl-C'd the run and got no trajectory" — partial write missing.
- Memory profiler shows resident memory growth tracking event count linearly with no plateau on long runs.
- Trajectories from successful runs are 50 MB+ (suggests no truncation cap on tool output).

**Phase to address:** P1 (write path) + P3 (signal handler integration).

**Severity:** MEDIUM

---

### Pitfall 10: Out-of-order events / observations arriving for tool calls that errored out

**What goes wrong:**
- Claude SDK (`backends/claude.py`) yields `ToolUseBlock` then later a `ToolResultBlock` in a `UserMessage`. Order is enforced by the SDK protocol; daydream maps cleanly. Risk = LOW.
- Codex (`backends/codex.py`) emits `item.started → item.updated* → item.completed`. Risk: a `command_execution` with `status="declined"` produces a `ToolResultEvent(is_error=True)` in `agent.py:261` where the corresponding `item.started` may not have been emitted (sandbox-declined commands may go straight to completed).
- Multiple observations for one tool call: ATIF's `Observation.results` is a list, supporting this case. But daydream's current event stream is `one ToolResultEvent per ToolStartEvent`. If Codex ever emits multiple result events for one tool call (e.g., partial output), the recorder must aggregate them — the spec allows it, but the natural recorder shape doesn't.

**Why it happens:**
Codex CLI is a moving target; the JSONL event shapes evolve. Daydream's existing `_log_debug` papers over inconsistencies with `[CODEX_WARN]` and `[CODEX_UNHANDLED]` lines, but ATIF requires a *valid trajectory*, not a tolerant log.

**How to avoid:**
- Recorder treats observations as `dict[tool_call_id, list[ObservationResult]]` internally — supports the multi-result case for free.
- Defensive merge: if a `ToolResultEvent` arrives without a prior `ToolStartEvent` from the same backend session, log to `extra.unmatched_tool_results` on the parent step; do not emit a dangling reference.
- Add Codex JSONL fixtures that exercise these edge cases (sandbox-declined command, missing-id command, completed-without-started) and assert the recorder produces a schema-valid trajectory.
- Tests should include the existing fixtures in `tests/fixtures/codex_jsonl/` (`turn_failed.jsonl` exists; add `command_declined.jsonl` if not present).

**Warning signs:**
- Validator complains about `source_call_id` referenced without matching call (overlap with Pitfall 3, but originating from a different source).
- Recorder emits a step with `observation.results` empty but the agent appeared to call a tool (lost result).

**Phase to address:** P2.

**Severity:** MEDIUM

---

### Pitfall 11: Test brittleness from over-asserting trajectory structure

**What goes wrong:**
Snapshot tests that capture the entire trajectory JSON break on every minor recorder change — adding a new optional `extra` field, renaming an internal helper, or even reordering `extra` keys (since dict ordering is preserved). This creates churn in PRs that have nothing to do with trajectory format.

The bigger trap: tests that assert `trajectory["steps"][2]["tool_calls"][0]["arguments"]["file_path"] == "expected/path"` make the test suite a snapshot of *recorder implementation* rather than *recorder behavior*.

**Why it happens:**
Trajectory is JSON; snapshot testing JSON is the path of least resistance.

**How to avoid:**
- **Two-tier test strategy:**
  - **Schema validity tests**: every test that produces a trajectory runs it through Harbor's `TrajectoryValidator` (or its Pydantic models with strict mode). This is the only kind of "snapshot" — does it pass schema validation. Stable across minor recorder changes.
  - **Behavior tests**: assert on *one* property of the trajectory that the test exists to verify, e.g. "the trajectory has at least one agent step with a `bash` tool call whose command contains `pytest`." Use predicate helpers, not full-tree comparison.
- Avoid `assert trajectory == expected_dict`. Use `assert validator.validate(trajectory)` plus `assert any(step.source == "agent" and step.tool_calls for step in trajectory.steps)`-style predicate checks.
- One golden-file test (in `tests/fixtures/atif/golden_minimal.json`) for "the simplest possible trajectory we can produce" — that one is allowed to be a literal-equality check because it's anchoring the format. Update it in the same PR as a schema bump and call out the bump in the commit message.
- Match the existing daydream pattern (per `.planning/codebase/TESTING.md`): use `monkeypatch` to stub `now_iso` so timestamps are deterministic; use inline mock backends to feed deterministic event streams to `run_agent()`.

**Warning signs:**
- A small refactor to the recorder changes 20+ test files.
- Tests assert on `step_id == 5` rather than "the third agent step has tool calls."
- The trajectory test file in the repo is the largest test fixture by far (>100 KB).

**Phase to address:** P1 (test pattern set during initial recorder build), enforced through P2/P3.

**Severity:** MEDIUM

---

### Pitfall 12: CLI surface migration — removing `--debug` is breaking

**What goes wrong:**
PROJECT.md `Active` requirements: "`--debug` CLI flag removed." This is a breaking change. Failure modes:
1. Existing users' shell aliases / scripts pass `--debug` → daydream errors with `argparse: unrecognized arguments: --debug`. The error is correct but unhelpful.
2. README and docs may still reference `--debug` (currently they do — `README.md:120` and `:192`).
3. Users who relied on the debug log file's content for ad-hoc grep workflows have to relearn.

**Why it happens:**
Hard cutover is the explicit decision (PROJECT.md "Hard cutover, no dual-write phase"). Cutover discipline is fine; *user-facing migration story* is the gap.

**How to avoid:**
- Don't silently delete `--debug`. Keep the argparse argument with `action="store_true"` (no behavior) for one release, but emit a deprecation message: `--debug is removed in this release. Trajectories are now written to .trajectory.json by default. Use --trajectory <path> to control the output location. See: README.md#trajectories`. Then exit 2.
  - *Alternative if cutover must be total*: remove the flag and customize argparse's error output via `argparse.ArgumentParser(epilog=...)` so unrecognized `--debug` triggers a known-unknown-flag handler that prints the same migration message.
- README must change in the same PR as the cutover. Add a "Migration from `--debug`" section pointing at the new `--trajectory` flag and the file format.
- Check `tests/test_cli.py` for any test that passes `--debug` → migrate or delete; ditto any documentation under `docs/`.
- CHANGELOG entry called out as a *breaking change* under `0.x.0` (next minor).

**Warning signs:**
- A user opens an issue: "daydream broke after I upgraded — `--debug: unrecognized argument`."
- Internal scripts in the repo still reference `--debug` after the cutover (run a grep before merging).
- README screenshots/examples include `--debug`.

**Phase to address:** P3.

**Severity:** MEDIUM

---

### Pitfall 13: Migration leaves orphan `_log_debug` callers / dead state

**What goes wrong:**
PROJECT.md lists 15+ call sites in `agent.py` plus call sites in `phases.py` (lines 496, 498, 777, 784, 1226, 1360, 1468), `runner.py` (lines 657, 713), and `exploration_runner.py` (220, 226, 251, 274). Plus the `set_debug_log` / `get_debug_log` getter/setter pair, the `AgentState.debug_log` field, and the `--debug` argparse arg. Failure modes:
1. A call site is missed → import error at runtime when `_log_debug` is removed, or worse, a broken import on a less-traveled code path (e.g., `--ttt --backend codex`) that the test suite doesn't exercise. Reference: `daydream/backends/codex.py:37` does a *lazy import* of `_log_debug` inside `_raw_log` — easy to miss in a dumb grep because the import isn't at module top.
2. `AgentState.debug_log` is removed but tests that monkeypatch `daydream.agent._state.debug_log` directly (anti-pattern, but possible) silently no-op.
3. The `print_info(console, f"Debug log: {debug_log_path}")` line in `runner.py:472` is removed but the surrounding `with stack:` context manager is left dangling.

**Why it happens:**
Hard cutover requires a complete sweep; lazy imports defeat naive grep.

**How to avoid:**
- Use `ast`-based check, not just `grep`. Run `python -m grep_ast '_log_debug|debug_log|set_debug_log|get_debug_log' daydream/` (or equivalent) to find every reference, including those inside string literals and lazy imports.
- Verify after removal: `grep -r '_log_debug\|debug_log\|set_debug_log' daydream/ tests/ docs/ scripts/ README.md` returns *zero* matches before considering the migration complete.
- Also check `pyproject.toml`, `Makefile`, and `.github/workflows/` for `--debug` references in CI.
- Add a one-time test: `test_no_debug_log_artifacts` greps the source tree and asserts no matches. Removed after the cutover lands.
- The `daydream/backends/codex.py:_raw_log` function and its callers (every `[CODEX_RAW]` line) need to be replaced with structured event passthrough into the trajectory's `extra` field, or just deleted. Decision required: are raw Codex event JSON blobs preserved as `extra.codex_raw_event` on the relevant step? Useful for debugging Codex parity issues; pollutes trajectories. Default: **delete**, since Codex events are derivable from the trajectory; preserve only for explicit `--debug-codex` flag (separate, post-milestone).

**Warning signs:**
- After the cutover lands, an unrelated PR introduces an import error in `--ttt` mode — symptom of a lazy-import path that was missed.
- A test that ran with `set_debug_log()` continues to "pass" after the cutover but actually no-ops because the function returns silently.
- Logs from CI contain `[PRE_SCAN]`, `[CODEX_RAW]`, `[REVERT]` prefixes — call sites missed.

**Phase to address:** P3.

**Severity:** MEDIUM

---

### Pitfall 14: `phases.py` (1552 lines) and `ui.py` (3470 lines) grow further during migration

**What goes wrong:**
Per `.planning/codebase/CONCERNS.md`, both modules are existing single-module bottlenecks. The natural place to put trajectory-emission code is *also* the natural place where it grows these modules:
- `phases.py` already holds prompt builders + JSON Schemas; adding ATIF event-to-step mapping ("when phase X starts, append a step with...") here makes the bottleneck worse.
- `ui.py` already renders `print_cost`, panels, and summaries from event data; the temptation will be to add trajectory rendering or a "trajectory summary panel" here.
- `runner.py` (781 lines) is the natural home for trajectory lifecycle (init at run start, finalize at run end), but it'll grow further too.

**Why it happens:**
Path of least resistance: existing modules already have the imports, the test harnesses, and the code patterns. Splitting modules is a separate refactor.

**How to avoid:**
- New module: `daydream/trajectory.py` (single file) for the recorder, builder, redactor, validator harness. Keep it under 500 lines; if it grows, split into `daydream/trajectory/{recorder,builder,redactor,schema}.py` package.
- Trajectory lifecycle hooks live in `runner.py` (`recorder.start()`, `recorder.finalize()`) but the *implementation* lives in the new module. One-line addition per phase.
- Event-to-step mapping lives in `agent.py` (the existing event-consumption boundary), not in `phases.py`. Phase functions never construct trajectory steps directly.
- Code review checklist item: "no `trajectory.` calls inside `phases.py`; no `Step()` construction inside `ui.py`."
- The `daydream/prompts/review_system_prompt.py` precedent (an unused module) shows the project tolerates new top-level modules — do not be shy about creating `daydream/trajectory.py`.

**Warning signs:**
- `phases.py` grows past 1700 lines after the migration.
- A `git log -p phases.py` after merge shows inserts referencing `Trajectory(`, `Step(`, or `recorder.`.
- `ui.py` gains a `print_trajectory_summary()` function (move the panel; let the recorder return a summary dict the UI renders generically).

**Phase to address:** Architectural decision in P1 (where does the recorder live?). Enforced via review in P2/P3.

**Severity:** MEDIUM

---

### Pitfall 15: Performance — Pydantic model construction for hundreds of events per run

**What goes wrong:**
A typical daydream run yields ~50–500 events (text/tool/cost). Deep mode against a large monorepo can hit thousands. Building a Pydantic model per event has measurable cost (Pydantic v2 with model validation is fast, but not free). JSON serialization of the full trajectory at run end re-walks the model graph.

If every event triggers a `Step(...)` Pydantic constructor call, that's allocation + validation per event. With Harbor as a dep, this might also pull in `pydantic-settings` or other transitive baggage.

**Why it happens:**
The "right" pattern (build with typed models) is more expensive than the "wrong" pattern (raw dicts).

**How to avoid:**
- Defer model construction to `recorder.finalize()`. During the run, accumulate plain dataclass-like Python objects (or dicts); construct Pydantic `Trajectory` once at the end. This means validation happens once, not per event.
- Trade-off: errors caught later. Mitigate by adding cheap invariants at append time (check `step_id` is monotonic, `source` is one of the three legal values) without full Pydantic validation.
- Benchmark threshold: a 1000-step trajectory should serialize+validate in <1 second on a developer laptop. Add a perf test in P2 that asserts this; fail the build if it regresses.
- Vet Harbor's transitive deps before merging the dependency. Run `uv sync` in a clean env and `pip-tree`-equivalent; if Harbor pulls in 50+ packages, the fallback documented in `.planning/PROJECT.md` constraints applies (vendor the JSON Schema, write our own Pydantic models — Pydantic is already transitive via `claude-agent-sdk`).
- Profile a real run with `cProfile` once, share the result in the PR description. If trajectory work is >5% of total run time, optimize.

**Warning signs:**
- `make test` runtime increases >20% after the migration lands.
- A single deep-mode run shows trajectory finalize as the slowest single operation in profiler output.
- Memory usage on long runs shows growth tracking event count one-for-one (suggests model objects retained, not just dicts).

**Phase to address:** P1 (architectural choice to defer construction); P2 (benchmark test).

**Severity:** LOW (quantitatively unlikely to bite, but easy to prevent).

---

### Pitfall 16: Reasoning content from ThinkingBlock leaks user/system context

**What goes wrong:**
Claude's `ThinkingBlock` and Codex's `reasoning` items contain the model's internal reasoning, which may include verbatim quotes of:
- User-supplied prompts (already captured in user steps — duplication is fine, but if a prompt contains secrets, they're now in *two* places).
- File contents the agent thought about (potentially sensitive, even though the agent didn't use a tool to read them — Claude's reasoning summarizes context the SDK loaded).
- Model self-reflection that quotes prior tool outputs verbatim (cached_tokens content).

**Why it happens:**
Reasoning content is a relatively new feature; engineers haven't internalized that it's a third leak surface alongside `tool_calls.arguments` and `observation.content`.

**How to avoid:**
- Apply the same `Redactor` from Pitfall 8 to `reasoning_content` on every agent step. Same regex set.
- Decision: do we record reasoning at all, or store a length-only marker (`extra.reasoning_length=1234`)? Default: record it (it's the highest-value field for SFT training, the explicit goal of ATIF adoption). Always-on redaction is the mitigation.
- Document in README that reasoning is recorded and redacted. Users with extra-sensitive workloads can pass `--no-reasoning` to drop the field (post-milestone option, not P1).

**Warning signs:**
- A trajectory's `reasoning_content` contains a string the user redacted from `observation.content` — symptom of reasoning bypassing the redactor.

**Phase to address:** P3 (with the redactor).

**Severity:** MEDIUM

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|---|---|---|---|
| Skip Pydantic, write raw `dict` matching ATIF schema | No new dep, faster build | Schema drift the moment ATIF v1.5 ships; validator errors caught only by external Harbor validator runs | Never — Harbor's models are the authoritative source per PROJECT.md decision |
| Synthesize Codex `cost_usd` from a hardcoded token-price table | Codex/Claude trajectories look uniform | Wrong as soon as pricing changes; trajectories preserved across years become lying historical records | Never |
| Snapshot-test the entire trajectory JSON | One assertion catches everything | Every recorder change is a 20-file PR | Only the one golden-file `tests/fixtures/atif/golden_minimal.json` |
| Inline trajectory code in `phases.py` | No new module, fewer imports | `phases.py` (1552 lines) grows unbounded — already the worst module-size offender | Never — use `daydream/trajectory.py` |
| Skip redaction in v1, "we'll add it later" | Migration ships sooner | First user issue with leaked credentials is a security incident; redaction-after-the-fact requires invalidating already-shared trajectories | Never — redaction is in the same release as cutover |
| Stream-write the trajectory to handle long-run memory | Unbounded run length | Mid-write JSON is invalid; ATIF expects atomic document; PROJECT.md explicitly Out of Scope | Only with explicit revisit decision |
| Keep `_log_debug` alongside ATIF for one release ("dual-write") | Smaller blast radius | Twice the code; dual-write divergences; defeats the migration's "less churn" rationale | PROJECT.md explicitly rules this out — never |
| Grep-only sweep for `_log_debug` removal | Fast | Lazy imports (`daydream/backends/codex.py:37`) missed → broken `--ttt --backend codex` path | Only with AST-based verification on top |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|---|---|---|
| Harbor `harbor.models.trajectories` | Importing from a `harbor` namespace that doesn't exist on PyPI; module path drift | Pin a known-good Harbor version in `pyproject.toml`; verify `from harbor.models.trajectories import Trajectory` works in CI; have a fallback "vendor the JSON Schema" plan documented |
| Harbor `TrajectoryValidator` CLI | Assuming `python -m harbor.utils.trajectory_validator` is on PATH after install | The CLI is part of Harbor's package, not a separate entry point — call it programmatically via `validator.validate(...)` in tests, not via subprocess |
| Claude SDK `ResultMessage.usage` | Treating `usage.cache_read_input_tokens` as additive to `usage.input_tokens` | Cached tokens are a *subset* of input tokens; ATIF treats `cached_tokens` similarly. Pass through, don't add |
| Codex `turn.completed.usage` | Trusting `input_tokens` to be cumulative across turns | Codex per-turn semantics differ from Claude; verify with a recorded JSONL fixture before relying on either |
| Tool ID stability | Assuming Claude `block.id` and Codex `item.id` follow the same format/lifetime | Claude IDs are SDK-assigned UUIDs; Codex sometimes lacks them and daydream synthesizes (see `daydream/backends/codex.py:194-195`). Don't try to parse ID format meaningfully |
| `claude-agent-sdk` `AgentDefinition` subagents | Treating `agents=...` calls as flat steps in the parent | They're true subagent delegations — should land in `subagent_trajectory` per ATIF. Currently used by `exploration_runner.py` |
| PR feedback flow (`run_pr_feedback`) | Recording the `gh` CLI parse as agent tool calls | `gh` calls happen *outside* `run_agent()` and should not appear in the trajectory at all. PROJECT.md explicitly excludes "MCP/non-agent subprocess calls" |
| Deep mode artifact dir | Putting trajectory under `target/.daydream/deep/` | Top-level run trajectory goes to `target/.trajectory.json` (or `--trajectory` path); per-stack subagent trajectories nest inside it. Don't co-locate with deep-mode merge artifacts that have separate lifecycle concerns |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|---|---|---|---|
| Pydantic validation per event | High CPU during a long run; profiler shows `pydantic_core` hot | Defer model construction to `recorder.finalize()`; build dicts during the run | At ~500+ events per run (deep mode against a large monorepo) |
| Unbounded `observation.content` retention | Memory growth tracking event count | Truncate at 256 KB per result with `<TRUNCATED:N bytes>` marker | At ~200+ tool calls with long bash output (`cargo build`, `pytest -v` of a big suite) |
| JSON serialization at finalize | Multi-second pause at run end | Use `to_json_dict()` with `exclude_none=True` (default); accept the cost; profile shows it's a small fraction of run time anyway | Only matters if trajectory exceeds ~50 MB |
| Redactor regex compilation per event | Slow event handling on long runs | Compile regex set once at recorder init; apply per event with no recompile | Trivial to prevent; easy to overlook |
| Synchronous trajectory write blocks `runner.run()` exit | User waits 2s after "Done" before shell returns | Acceptable for batch-write; if it ever exceeds 5s for typical runs, move to a final `asyncio.to_thread(write_path, json_str)` step | At ~10 MB trajectory files |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---|---|---|
| Recording bash command output verbatim including stderr | Env dumps land in stderr; `[ERROR]` lines from cloud CLIs include tokens | Apply token regex set to ALL `observation.content`, including is_error=True cases |
| Relying on `--no-redact` being the default | First share of a trajectory leaks credentials | Default is redaction-on; `--no-redact` requires explicit opt-in and prints a warning |
| Trusting users not to share `.trajectory.partial.json` from a crash | Partial trajectories may have less consistent redaction | Apply redactor to partial-write paths too; add `extra.partial=true` so consumers know |
| Redacting at display time but not write time | UI shows redacted; file has raw secrets | Redact at recorder serialization, not in UI; UI displays whatever the user provided in real time (it's their terminal) |
| Including the user's `~/.claude/settings.json` if an agent reads it for "config inspection" | API key leaked to trajectory | Path-based block list runs before content scanning |
| Dropping git remote URLs without stripping `user:token@` prefix | Git creds embedded in URLs leak | URL-aware redactor that parses git remote strings |

---

## "Looks Done But Isn't" Checklist

- [ ] **Schema validation**: Every test that exercises the recorder calls Harbor's validator and asserts success — verify by adding a deliberately-broken trajectory test that the suite catches it.
- [ ] **Token reconciliation**: `FinalMetrics` totals match `sum(steps[*].metrics.*)` — verify with a multi-turn fixture.
- [ ] **Subagent shape**: Exploration specialists actually appear in `subagent_trajectory`, not flattened — verify by inspecting an exploration-mode trajectory.
- [ ] **Codex parity**: A `--backend codex` run produces a schema-valid trajectory with token counts populated — verify with a recorded JSONL fixture replay test.
- [ ] **Redaction**: Seeding the test trajectory with `sk-test-12345`, `ghp_test123`, `/Users/ka/foo`, and `kevin@example.com` produces a trajectory where none of those literals appear — explicit assertion.
- [ ] **Crash path**: SIGINT during a run produces `.trajectory.partial.json` — verify with a kill-then-inspect test.
- [ ] **Memory cap**: Long-running tool output is truncated at 256 KB — verify by feeding a 10 MB string through the recorder and checking the resulting field length.
- [ ] **`_log_debug` removal complete**: `grep -r '_log_debug\|debug_log' daydream/ tests/ docs/ README.md scripts/` returns zero matches.
- [ ] **`--debug` migration message**: Running `daydream --debug` shows the migration message (or a clear unknown-flag error pointing at `--trajectory`).
- [ ] **README updated**: Migration section, trajectory format reference, `--trajectory` flag documented; old `--debug` references gone (currently `README.md:120` and `:192`).
- [ ] **No new bloat in `phases.py` / `ui.py`**: `wc -l daydream/phases.py daydream/ui.py` after the migration is within 50 lines of pre-migration counts.
- [ ] **Test pattern**: New tests use schema-validity + behavior assertions, not full-tree equality (sample 5 new test files, verify pattern).
- [ ] **Hierarchical step_id correctness**: Deep-mode trajectory with parallel stacks has sequential step_id at each agent level (root, each subagent) — verified by replaying a recorded multi-stack run.
- [ ] **Deprecation warning**: If keeping `--debug` as a one-release shim, the warning text matches the README pointer text exactly.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---|---|---|
| Trajectories produced post-cutover fail Harbor validation | LOW | Add validator to the recorder finalize step; log validator errors to stderr; user can re-run. Do not write invalid trajectories — fail loudly |
| User shares a trajectory with leaked credentials | HIGH | Rotate the leaked credential first; document the leak in CHANGELOG; ship a redactor patch ASAP; consider invalidating previously-uploaded Harbor trajectories |
| `_log_debug` removal misses a call site, breaks `--ttt --backend codex` | MEDIUM | Hot-fix release reverting just that call site to a no-op (`def _log_debug(*a, **kw): pass`) until proper trajectory wire-up; ship within a day |
| Memory blow-up on a long deep-mode run | MEDIUM | Truncation cap is the long-term fix; immediate workaround: tell user to use `--review-only` until patched |
| `step_id` non-sequential due to a race | MEDIUM | Add lock; backfill validator step in tests; reissue trajectories from the affected runs is impossible (data lost) — accept that pre-fix trajectories are unrecoverable |
| Cost double-counting (running totals leaked as per-step) | LOW | Easy fix in the Claude backend's token extraction; trajectories produced before the fix can be recomputed by parsing `step.metrics.cost_usd` cumulatively rather than summing — document the workaround |
| `--debug` deletion breaks user CLI invocations | LOW | Re-add the flag as a no-op shim in a patch release; print the migration message; remove for real in the next minor |
| Lazy-import miss surfaces only in production (`daydream/backends/codex.py:37`) | LOW | Hot-fix; add a `tests/test_lazy_imports.py` that imports every module in every backend combination |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---|---|---|
| 1. Non-sequential `step_id` | P1 | Validator + concurrent-mock test in P2 |
| 2. Timestamp format | P1 | `now_iso()` helper test + ban on `datetime.utcnow` |
| 3. Dangling `source_call_id` | P2 | Codex JSONL fixture replay; recorder `in_flight` invariant test |
| 4. Agent-only fields on user steps | P1 | Pydantic model use; minimal-user-step golden test |
| 5. Token accounting (Claude) | P2 | Multi-turn fixture; `FinalMetrics == sum(steps)` assertion |
| 6. Codex token parity | P2 | Cross-backend parity test |
| 7. Hierarchical subagent shape | P2 | Decision matrix logged in PROJECT.md decisions; deep-mode trajectory inspection test |
| 8. Privacy / secret leaking | P3 | Seeded-secret redaction test |
| 9. Crash → no trajectory | P1 (write path) + P3 (signal handler) | SIGINT integration test |
| 10. Out-of-order events | P2 | Codex edge-case fixtures |
| 11. Test brittleness | P1 (set test pattern) | Code review checklist |
| 12. CLI `--debug` removal | P3 | Manual test + README update in same PR |
| 13. `_log_debug` orphans | P3 | Grep-clean assertion before merge |
| 14. `phases.py` / `ui.py` bloat | P1 (architecture) | `wc -l` check at end of P3 |
| 15. Performance | P1 (defer construction) + P2 (benchmark) | 1000-step benchmark test |
| 16. Reasoning content leaks | P3 | Same redactor test, applied to `reasoning_content` |

---

## Sources

- ATIF v1.4 spec: `docs/reference/atif_format.md` (this repo) — HIGH confidence
- ATIF RFC: https://github.com/laude-institute/harbor/blob/main/docs/rfcs/0001-trajectory-format.md — HIGH (referenced from spec)
- Harbor OpenHands example showing accumulated-to-delta conversion: `docs/reference/atif_format.md` lines 47-66 — HIGH
- daydream codebase: `daydream/agent.py` (call sites), `daydream/backends/codex.py` (event parsing realities), `daydream/backends/claude.py` (token extraction gap), `daydream/runner.py` (debug-log bootstrap), `daydream/phases.py` (additional `_log_debug` callers), `daydream/exploration_runner.py` (lazy import) — HIGH
- `.planning/codebase/CONCERNS.md` (module-size concerns, late imports, broad catches) — HIGH
- `.planning/codebase/TESTING.md` (test patterns, mock backend convention) — HIGH
- `.planning/PROJECT.md` (Active requirements, Out of Scope, Key Decisions) — HIGH
- Cross-tool conventions for trajectory format (running totals vs deltas, redaction practice): inferred from OpenHands and Claude Code documented behaviors plus general SFT/RL pipeline norms — MEDIUM

---
*Pitfalls research for: ATIF integration into daydream*
*Researched: 2026-04-26*
