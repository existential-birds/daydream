# Phase 4: Cutover + Redaction + CLI Surface - Context

**Gathered:** 2026-04-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Hard removal of the legacy `_log_debug` debug logging system — all 30+ call sites across `agent.py`, `phases.py`, `runner.py`, `exploration_runner.py`, `backends/codex.py`, and `ui.py`, plus the `AgentState.debug_log` field and its getters/setters. Redaction policy implemented and applied to all trajectory content surfaces (`ToolCall.arguments`, `ObservationResult.content`, `Step.message`, `Step.reasoning_content`). CLI surface updated: `--debug` removed (hard reject), `--trajectory <path>` added, SIGINT flushes partial trajectory.

**Out of phase scope (deferred):**
- Test suite hardening for trajectory/redaction/golden-fixture (Phase 5: TEST-01..07)
- README/CHANGELOG/CLAUDE.md documentation (Phase 5: DOCS-01..06)

</domain>

<decisions>
## Implementation Decisions

### Redaction Token Style (REDA-01, REDA-02, REDA-03, REDA-04)

- **D-01: Type-specific redaction tokens.** Each pattern category gets its own replacement text: `[REDACTED_API_KEY]` (for `sk-*`, `ghp_*`, `xoxb-*`, `AKIA*` patterns), `[REDACTED_JWT]` (for `eyJ…` tokens), `[REDACTED_USER]` (for username segments in paths), `[REDACTED_ENV_VAR]` (for `.env`-style secret values). Consumers can tell what WAS there without seeing the value.
- **D-02: Path redaction scrubs username segment only.** `/Users/ka/github/project/src/app.py` becomes `/Users/[REDACTED_USER]/github/project/src/app.py`. Preserves project-relative paths for trajectory replay and debugging. Same for `/home/<name>/` and `C:\Users\<name>\` patterns.
- **D-03: .env-style redaction preserves key name, redacts value.** `OPENAI_API_KEY=sk-1234abcd` becomes `OPENAI_API_KEY=[REDACTED_ENV_VAR]`. Only matches lines where the key name contains `KEY`, `SECRET`, `TOKEN`, `PASSWORD`, `CREDENTIAL`, or similar secret-indicating substrings. Non-secret env vars (`DEBUG=true`, `APP_NAME=myproject`) pass through unredacted.
- **D-04: Flat regex on serialized text.** `Redactor._redact_text(s: str) -> str` runs all regex patterns on the raw string. No JSON-aware deep walk of `ToolCall.arguments`. Same method applied uniformly across all 4 ATIF surfaces. Phase 2 D-12's `redact_step(step: Step) -> Step` API stays unchanged — internals get filled with the regex dispatch.

### CLI Surface (CLI-01, CLI-02, CLI-03, CLI-05)

- **D-05: Hard argparse reject for `--debug`.** The flag is removed from `_parse_args()`. Argparse produces an immediate `unrecognized arguments: --debug` error. No deprecation window, no warning-and-continue. CHANGELOG (Phase 5) documents the breaking change.
- **D-06: Explicit `--trajectory <path>` write failure exits with error.** When the user explicitly passes `--trajectory <path>` and the write fails (permission denied, disk full), daydream exits with a non-zero exit code and an error message. This matches the "user asked for it, deliver or fail" contract. Implicit writes (default path) degrade with a warning and continue the run (per Phase 2 D-11, CORE-09).
- **D-07: SIGINT partial flush writes to `<path>.partial` suffix.** Normal completion writes to `<target>/.daydream/trajectory.json`. SIGINT mid-run writes to `<target>/.daydream/trajectory.json.partial` with `extra.partial=true` inside the file. Clean filesystem separation — consumers know a `.partial` file is incomplete without parsing. Explicit `--trajectory /tmp/out.json` + SIGINT produces `/tmp/out.json.partial`.

### Operational Log Line Fate (CUT-01..08)

- **D-08: Promote error/warning log sites to UI; silently remove redundant ones.** The ~15 agent-event-mirroring sites (`[TEXT]`, `[THINKING]`, `[TOOL_USE]`, `[TOOL_RESULT]`, `[COST]`, `[TOKENS]`, `[PROMPT]`, `[SCHEMA_OK]`, `[SCHEMA_FALLBACK]`) are silently removed — fully redundant with trajectory recording. The ~10 operational sites are split:
  - **Promoted to `print_error()`:** `[EXECUTE_ERROR]`, `[EXECUTE_INIT_ERROR]`, `[PHASE2_ERROR]`
  - **Promoted to `print_warning()`:** `[REVERT]` git clean/checkout failures, `[PARSE_FALLBACK]` empty parse result, `[TTT_REVIEW]`/`[TTT_PLAN]` unexpected result types
  - **Silently removed:** `[PRE_SCAN]` specialist failures (exploration is best-effort), `codex.py` and `ui.py` proxy calls
- **D-09: Quiet-mode contract for promoted messages.** `print_error()` calls always display regardless of `--quiet`. `print_warning()` calls check `get_quiet_mode()` and suppress when quiet. Matches daydream's existing quiet-mode contract.

### Claude's Discretion

- AST sweep implementation for CUT-08 (standalone script, pytest parametrized test, or inline assertion — all acceptable as long as it walks the AST of every `.py` file in `daydream/` and `tests/`)
- Exact regex pattern details for each redaction category (anchoring, capture groups, false-positive tuning)
- Ordering and atomicity of legacy removal commits (one big commit vs multiple atomic ones)
- Which tests need mock/assertion updates for `debug_log` removal — audit during implementation
- Whether `_log_debug` definition is removed first or last relative to its call sites
- SIGINT handler integration mechanics (hook placement in `cli.py` vs recorder `__aexit__` vs runner.py)
- Redaction failure mode internals (REDA-05: redact-or-omit, never raw-pass-through — the contract is locked, the implementation is discretionary)

</decisions>

<specifics>
## Specific Ideas

- Phase 2 D-12 designed the Redactor as a no-op pass-through with the final API surface so Phase 4 is purely additive (regex rules). Zero changes to the recorder call site or public API. The `redact_step()` method already runs on every Step at flush time (Phase 2 D-13).
- ROADMAP success criterion SC1 specifies AST-level sweep, not grep — the lazy import `from daydream.agent import _log_debug` inside a function body in `daydream/backends/codex.py:38` is the canonical gotcha that grep would miss.
- Promoted `print_warning`/`print_error` calls should use daydream's existing `ui.print_warning()` and `ui.print_error()` — NOT raw `print()` or `console.print()` — per the "no direct print()" architectural constraint.
- The user explicitly chose hard fail over warn-and-continue for explicit `--trajectory` write errors, overriding the initial lean toward degradation.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### ATIF Specification
- `docs/reference/atif_format.md` — Authoritative ATIF spec. Redaction tokens must not break ATIF validity; `extra.partial` field for partial trajectories.

### Phase 2 Output (Recorder + Redactor API — already landed)
- `daydream/trajectory.py` — Current recorder with no-op `Redactor.redact_step()`. Phase 4 fills in regex patterns inside `_redact_text()`. Also contains `TrajectoryRecorder.__aenter__`/`__aexit__` lifecycle, ContextVar management, `DaydreamPhase`/`DaydreamRunFlow` enums, `now_iso()` helper.
- `.planning/phases/02-recorder-core-event-enrichment-mapping/02-CONTEXT.md` — Phase 2 decisions, especially D-11 (implicit write degrades, explicit fails — now locked per D-06 above), D-12 (Redactor no-op with final API), D-13 (redaction at per-Step flush time).

### Phase 3 Output (Subagent Wiring — already landed)
- `.planning/phases/03-subagent-wiring-parallel-continuation/03-CONTEXT.md` — Phase 3 decisions, especially D-12 (`subagent_trajectory_ref` uses relative paths). Redaction must apply uniformly to root + sibling content.

### Legacy Removal Targets (Phase 4 integration points)
- `daydream/agent.py` — `_log_debug()` definition (~line 208), `AgentState.debug_log` field, `set_debug_log()`/`get_debug_log()` getters/setters, `reset_state()` includes `debug_log` reset, 20+ `_log_debug()` call sites in `run_agent()` event loop.
- `daydream/phases.py` — `[REVERT]` (~line 497-499), `[PARSE_FALLBACK]` (~line 778-785), `[TTT_REVIEW]` (~line 1238), `[TTT_PLAN]` (~line 1372).
- `daydream/runner.py` — `[PHASE2_ERROR]` (~lines 684, 740), debug file initialization (`.review-debug-{ts}.log` setup).
- `daydream/exploration_runner.py` — `[PRE_SCAN]` (~lines 227, 255, 281).
- `daydream/backends/codex.py` — Lazy import `from daydream.agent import _log_debug` inside function body (~line 38), `_raw_log()` proxy (~line 40). **AST sweep target** per CUT-06/CUT-08.
- `daydream/ui.py` — `_log_debug` proxy (~line 32).

### CLI Surface
- `daydream/cli.py` — `_parse_args()` (argparse declarations), `_signal_handler()` (SIGINT/SIGTERM), `--debug` flag to remove, `--trajectory <path>` flag to add.
- `daydream/runner.py` — `RunConfig` dataclass (already has `trajectory_path` from Phase 2), debug file init block to remove.

### Project Planning
- `.planning/PROJECT.md` — Key Decisions table (hard cutover, redaction with cutover, `--debug` removal).
- `.planning/REQUIREMENTS.md` — REDA-01..06, CUT-01..08, CLI-01..05 are this phase's 19 requirements.
- `.planning/ROADMAP.md` — Phase 4 success criteria (5 must-be-true items).
- `.planning/research/PITFALLS.md` — Pitfall 8 (privacy/secret leaking), Pitfall 9 (SIGINT partial flush), Pitfall 12 (`--debug` removal), Pitfall 13 (`_log_debug` orphans via lazy import), Pitfall 16 (reasoning content leaks).

### Codebase Maps
- `.planning/codebase/ARCHITECTURE.md` — Cross-Cutting Concerns section documents current `_log_debug` logging, all data flows showing where debug logging lives.
- `.planning/codebase/CONCERNS.md` — Module size watchlist. Phase 4 removes code — sizes should decrease.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`Redactor` class in `daydream/trajectory.py`** — No-op pass-through with `redact_step(step: Step) -> Step` already called at Step flush time. Phase 4 adds `_redact_text(s: str) -> str` with regex dispatch internally. Zero API changes.
- **`TrajectoryRecorder.__aexit__`** — Already writes trajectory JSON on clean exit. Phase 4 adds the SIGINT partial-flush path alongside.
- **`ui.print_warning()` / `ui.print_error()`** — Existing UI helpers that respect quiet mode. Promoted log lines use these directly.
- **`RunConfig.trajectory_path`** — Already exists from Phase 2. Phase 4 wires it to `_parse_args()` and adds the fail-loud write check.

### Established Patterns
- **`if recorder is not None:` guard** — Used in `agent.py` and `phases.py`. No changes in Phase 4.
- **`reset_state()` in tests** — Phase 4 removes `debug_log` from `AgentState` and removes `set_debug_log`/`get_debug_log`. Tests that reference these must be updated.
- **Quiet-mode contract** — `print_info`/`print_warning` check `get_quiet_mode()`; `print_error` always shows. Promoted log lines follow this existing contract.

### Integration Points
- **`agent.py:run_agent()`** loses ~20 `_log_debug()` calls — the event loop body gets simpler. The trajectory recorder (`inv.observe(event)`) is already the replacement.
- **`cli.py:_parse_args()`** gains `--trajectory` argparse declaration and loses `--debug`.
- **`cli.py:_signal_handler()`** gains partial-trajectory flush logic before raising `KeyboardInterrupt`.
- **`runner.py`** loses the debug file initialization block (~10 lines). `RunConfig.debug` field removed.
- **`AgentState`** dataclass in `agent.py` loses the `debug_log` field and associated getters/setters. `reset_state()` simplified.

</code_context>

<deferred>
## Deferred Ideas

- **`--no-redact` escape hatch** — Mentioned in PITFALLS.md Pitfall 8 mitigation; explicitly deferred per PROJECT.md (privacy-default-on is the decision). Could be added post-milestone if users need raw trajectories for debugging redaction issues.
- **Custom redaction pattern configuration** — Allow users to add their own regex patterns via config. Not a current requirement; redaction categories cover the standard secret patterns.
- **Trajectory streaming writes (PERF-01)** — SIGINT partial-flush addresses the "crash loses everything" concern. Full mid-run streaming deferred to v2 per PROJECT.md.

</deferred>

---

*Phase: 04-cutover-redaction-cli-surface*
*Context gathered: 2026-04-28*
