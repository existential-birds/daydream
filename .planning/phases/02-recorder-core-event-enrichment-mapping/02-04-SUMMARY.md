---
phase: 02-recorder-core-event-enrichment-mapping
plan: 04
subsystem: backends
tags: [python, codex, tokens, metrics, atif]

# Dependency graph
requires:
  - phase: 02-recorder-core-event-enrichment-mapping
    plan: 02
    provides: MetricsEvent dataclass + CostEvent.cached_tokens default-None field on `daydream.backends`
provides:
  - Codex backend MetricsEvent emission at every `turn.completed` event with the documented D-16 parity gap (cost_usd=None, cached_tokens=None, message_id="")
  - Plan 07 (integration tests) can chain off the new `tests/fixtures/codex_jsonl/turn_completed_with_usage.jsonl` and `turn_completed_partial_usage.jsonl` fixtures and reuse the established `_make_mock_process()` JSONL-stream helper
  - Codex CostEvent call site now passes cached_tokens=None explicitly (previously implicit via dataclass default)
affects:
  - phase 02-05 (event-to-ATIF mapping in agent.run_agent): Codex runs now flow through `Invocation._dispatch` via the real `isinstance(event, MetricsEvent)` branch (fast path); no class-name-fallback hop needed
  - phase 02-07 (Harbor-golden round-trip): Codex trajectories now expose per-turn token counts via `Step.metrics.prompt_tokens` / `Step.metrics.completion_tokens` — enables backend-parity assertions that previously could only check Claude

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Boundary rename at emission time (EVNT-02): backend reads SDK keys (`usage['input_tokens']` / `usage['output_tokens']`) and yields MetricsEvent with the ATIF-side names (`prompt_tokens` / `completion_tokens`). Same pattern applies in the parallel Plan 02-03 Claude work."
    - "Optional-data skip pattern: when `EVNT-02` requires int (not Optional), backends MUST skip MetricsEvent emission rather than synthesizing zero or None. CostEvent retains the partial-data signal so FinalMetrics aggregation degrades gracefully."
    - "Parity-gap explicitness: D-16 says Codex `cost_usd` and `cached_tokens` are ALWAYS None. The implementation hardcodes those constants instead of plumbing through optional fields — makes the invariant grep-able (`grep -c 'cost_usd=None' codex.py` is a guardrail, not just documentation)."

key-files:
  created:
    - tests/test_backend_codex_metrics.py — 4 schema-validity + behavior-predicate tests covering EVNT-07, D-16; 102 lines
    - tests/fixtures/codex_jsonl/turn_completed_with_usage.jsonl — 4-line fixture mirroring `simple_text.jsonl` shape with `usage={"input_tokens":200,"output_tokens":100}`
    - tests/fixtures/codex_jsonl/turn_completed_partial_usage.jsonl — same shape, `usage={"input_tokens":200}` only (no output_tokens)
  modified:
    - daydream/backends/codex.py — additive enrichment at the `turn.completed` branch (lines 309-337); +24/-1; final size 457 lines (was 434)

key-decisions:
  - "Reused `_make_mock_process` from `tests/test_backend_codex.py` instead of inlining a per-test variant. The plan called this out as the preferred path; Plan 07 will benefit from the same helper. No new helper-extraction work needed."
  - "MetricsEvent emission gated on BOTH input_tokens AND output_tokens being non-None — partial usage still emits CostEvent (which accepts Optional fields) but NOT MetricsEvent (which doesn't). This matches the EVNT-02 type signature exactly and avoids constructing malformed events."
  - "CostEvent call site updated to pass `cached_tokens=None` explicitly even though Plan 02 made it default. Reasoning: explicit kwargs at the construction site improve diff readability and make the D-16 parity gap visible without cross-file lookup."
  - "Per-turn check `usage.get('input_tokens') is not None` instead of `'input_tokens' in usage` — gracefully handles a turn.completed with `usage={'input_tokens': null}` from a future Codex CLI version. Costs nothing today and avoids a future flake."

patterns-established:
  - "Codex backend's turn.completed branch is now the canonical site for both per-step Metrics (MetricsEvent) and end-of-call FinalMetrics (CostEvent). Each event carries the same numbers — the recorder routes them to different ATIF places (Step.metrics vs FinalMetrics aggregation)."
  - "Codex JSONL fixtures follow a 4-line shape: thread.started + item.started + item.completed + turn.completed. Future tests for new event types should add a fifth fixture line in the same style — minimal valid prefix + the variant being tested."

requirements-completed: [EVNT-07]

# Metrics
duration: 30min
completed: 2026-04-26
---

# Phase 02 Plan 04: Codex MetricsEvent Emission Summary

**Codex backend now emits a per-turn MetricsEvent alongside the legacy CostEvent on every `turn.completed`, with the documented D-16 parity gap (cost_usd=None, cached_tokens=None, message_id="") and EVNT-02 verbatim field names (prompt_tokens / completion_tokens). 4 new tests pass; the 4 pre-existing test_deep_orchestrator failures (already documented in deferred-items.md by Plan 02-01) remain pre-existing.**

## Performance

- **Duration:** approx. 30 min
- **Completed:** 2026-04-26
- **Tasks:** 1 (single TDD task per plan)
- **Commits:** 2 (RED test commit + GREEN implementation commit)
- **Files created:** 3 (1 test module + 2 JSONL fixtures)
- **Files modified:** 1 (`daydream/backends/codex.py`, +24/-1)

## Accomplishments

- **EVNT-07 satisfied:** Codex backend emits a `MetricsEvent` at every `turn.completed` event whose `usage` carries both `input_tokens` and `output_tokens`. The new emission lives inside the existing `elif event_type == "turn.completed":` branch (codex.py:309-331), preserving the legacy CostEvent emission for FinalMetrics aggregation.
- **D-16 parity gap honored explicitly:** `message_id=""` (Codex has no per-message id), `cost_usd=None` (no USD reporting from Codex CLI), `cached_tokens=None` (no cached-token field in `turn.completed.usage`). The constants are hardcoded so the invariant is grep-able (`grep -c "cost_usd=None" daydream/backends/codex.py` is a guardrail).
- **EVNT-02 boundary rename at emission:** the backend reads SDK keys `usage["input_tokens"]` / `usage["output_tokens"]` and yields MetricsEvent with the ATIF-side names `prompt_tokens` / `completion_tokens`. The CostEvent emission keeps the SDK boundary names because `tests/test_backends_init.py::test_cost_event_fields` already pins those exact attributes.
- **Partial-usage degradation:** when `usage` is missing either `input_tokens` or `output_tokens` (e.g., a future Codex CLI quirk), the backend SKIPS MetricsEvent emission — EVNT-02 types both fields as `int`, not `Optional`, so emitting a partial event would be malformed. CostEvent still emits with `output_tokens=None` so FinalMetrics aggregation degrades gracefully.
- **CostEvent call site updated to pass `cached_tokens=None` explicitly** — Plan 02 made the field default to None for back-compat, but explicit kwargs at the construction site make the D-16 parity gap visible without cross-file lookup.
- **4 new tests in `tests/test_backend_codex_metrics.py`** covering each EVNT-07/D-16 predicate: full-usage MetricsEvent emission with EVNT-02 field names, legacy CostEvent preservation, partial-usage MetricsEvent skip path, parity-gap invariants.
- **2 new JSONL fixtures** mirror the existing `simple_text.jsonl` 4-line shape — one full-usage, one partial-usage. Plan 07 (integration tests) can chain off these without redesigning fixtures.

## Task Commits

1. **Task 1 RED — failing test for Codex MetricsEvent emission** — `d469c98` (test)
2. **Task 1 GREEN — emit MetricsEvent in Codex turn.completed branch** — `f2e4d23` (feat)

## Existing Codex JSONL Mock Helper (for Plan 07)

The existing `tests/test_backend_codex.py` defines `_make_mock_process(fixture_name)` (lines 22-47) which builds a `MagicMock` asyncio process whose `stdout.readline` replays lines from `tests/fixtures/codex_jsonl/*.jsonl`. The helper is the single source of truth for the JSONL-stream mock pattern. Plan 02-04's tests reuse it via:

```python
from tests.test_backend_codex import _make_mock_process
```

paired with the established subprocess patch:

```python
with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
    ...
```

Plan 07 should reuse the same import and patch shape; do NOT re-implement the readline iterator.

## JSONL Fixture Content (for Plan 07 chaining)

`tests/fixtures/codex_jsonl/turn_completed_with_usage.jsonl`:
```
{"type":"thread.started","thread_id":"th_test"}
{"type":"item.started","item":{"type":"agent_message","id":"msg_1","content":[]}}
{"type":"item.completed","item":{"type":"agent_message","id":"msg_1","content":[{"type":"text","text":"ok"}]}}
{"type":"turn.completed","usage":{"input_tokens":200,"output_tokens":100}}
```

`tests/fixtures/codex_jsonl/turn_completed_partial_usage.jsonl`:
```
{"type":"thread.started","thread_id":"th_test"}
{"type":"item.started","item":{"type":"agent_message","id":"msg_1","content":[]}}
{"type":"item.completed","item":{"type":"agent_message","id":"msg_1","content":[{"type":"text","text":"ok"}]}}
{"type":"turn.completed","usage":{"input_tokens":200}}
```

Plan 07 can compose these by appending additional events (e.g., a tool_use sequence before turn.completed) to exercise multi-event integration scenarios.

## Final turn.completed Branch Shape (codex.py:309-337)

```python
elif event_type == "turn.completed":
    usage = event.get("usage", {})
    # ... explanatory comment block referencing EVNT-07, D-16, Pitfall 6 ...
    if usage.get("input_tokens") is not None and usage.get("output_tokens") is not None:
        yield MetricsEvent(
            message_id="",
            prompt_tokens=usage["input_tokens"],
            completion_tokens=usage["output_tokens"],
            cached_tokens=None,
            cost_usd=None,
        )
    yield CostEvent(
        cost_usd=None,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cached_tokens=None,
    )
```

Both events emit on the same `turn.completed`. The recorder uses MetricsEvent first (per-step Metrics, D-04); CostEvent feeds FinalMetrics aggregation in the recorder.

## Decisions Made

- **Reused `_make_mock_process` instead of inlining.** The plan called this out as the preferred path. The helper is the single source of truth for the JSONL-stream mock; re-implementing it in the new test file would have created two diverging shapes.
- **Gated MetricsEvent on BOTH usage keys being non-None.** EVNT-02 types `prompt_tokens` and `completion_tokens` as `int`, not `Optional`. Constructing a MetricsEvent with `prompt_tokens=None` would either type-fail or downstream-fail in the recorder. The CostEvent still emits with the partial signal so FinalMetrics aggregation degrades gracefully.
- **Explicit `cached_tokens=None` on the CostEvent call site.** Plan 02 made it default to None for back-compat; the explicit kwarg at the construction site improves diff readability and makes the D-16 parity gap visible without cross-file lookup.

## Plan Acceptance Criteria — Verification

| Criterion | Result |
| --------- | ------ |
| `grep -c "yield MetricsEvent" daydream/backends/codex.py` returns 1 | ✓ 1 |
| `grep -c "MetricsEvent" daydream/backends/codex.py` returns >= 2 | ✓ 4 (import + 1 class-name comment + 1 use + 1 docstring-style hint via comment) |
| `grep -c 'message_id=""' daydream/backends/codex.py` returns exactly 1 | ✓ 1 |
| `grep -c "prompt_tokens=" daydream/backends/codex.py` returns exactly 1 | ✓ 1 |
| `grep -c "completion_tokens=" daydream/backends/codex.py` returns exactly 1 | ✓ 1 |
| `grep -nE "MetricsEvent\(.*input_tokens=" daydream/backends/codex.py` returns no matches | ✓ 0 (MetricsEvent uses EVNT-02 names exclusively) |
| `grep -nE "cost.*=.*\* (input_tokens|output_tokens)" daydream/backends/codex.py` returns no matches | ✓ 0 (no token-price-table synthesis) |
| `tests/test_backend_codex_metrics.py` exists with >= 4 test functions | ✓ 4 |
| `grep -c "from tests.test_backend_codex import _make_mock_process" tests/test_backend_codex_metrics.py` returns 1 | ✓ 1 |
| `grep -c "create_subprocess_exec" tests/test_backend_codex_metrics.py` returns >= 1 | ✓ 4 |
| `tests/fixtures/codex_jsonl/turn_completed_with_usage.jsonl` and `turn_completed_partial_usage.jsonl` exist | ✓ both present |
| `uv run pytest tests/test_backend_codex_metrics.py -v` exits 0 | ✓ 4/4 pass |
| `uv run ruff check daydream/backends/codex.py tests/test_backend_codex_metrics.py` exits 0 | ✓ clean |
| `uv run mypy daydream/backends/codex.py` exits 0 | ✓ clean |
| Both `yield MetricsEvent` and `yield CostEvent` appear in the `turn.completed` branch | ✓ confirmed by inline read |

## Issues Encountered

- **gpg-agent flake on `tests/test_deep_integration.py::test_claude_shape_backend`:** the conftest fixture `multi_stack_target` at `tests/conftest.py:71-100` runs `git init` + `git commit -m 'init'` in a tempdir without setting `commit.gpgsign=false`. The user's global `~/.gitconfig` has `commit.gpgsign = true`, so the fixture commit fails with `error: gpg failed to sign the data: ... No agent running` — non-deterministic depending on whether gpg-agent has been recently exercised in the shell. Workaround: run the suite with `GIT_CONFIG_GLOBAL=/dev/null` (the suite then passes 401, with 4 pre-existing test_deep_orchestrator failures already documented in `deferred-items.md`). The fixture would benefit from a `git config commit.gpgsign false` line — out of scope for this plan; logged in deferred-items.md.

- **Pre-existing test_deep_orchestrator failures (4):** Already documented by Plan 02-01 and Plan 02-02 in `deferred-items.md`. Plan 02-04 is purely additive (new MetricsEvent emission + new tests + new fixtures) and does not touch deep-mode code. The failures cannot have been caused or affected by this plan.

## TDD Gate Compliance

This plan is `tdd="true"` on its single task. The gate sequence is satisfied:

1. **RED gate:** commit `d469c98` (test) — added 4 tests + 2 JSONL fixtures; ran `uv run pytest tests/test_backend_codex_metrics.py` → 1/4 fail (`test_metrics_event_emitted_at_turn_completed` — codex.py:308 only yields CostEvent). RED confirmed.
2. **GREEN gate:** commit `f2e4d23` (feat) — added MetricsEvent emission in the turn.completed branch + import line; ran the same command → 4/4 pass.
3. **REFACTOR gate:** not needed — implementation matched the plan's Action block verbatim, with one inline comment tweak to satisfy `grep -c 'message_id=""'` exactly-1 (the original wording included the literal substring in a comment, producing 2 matches; tweaked to "the message id is the empty string" to keep code-vs-prose distinct). No behavior change.

A `feat` commit follows the `test` commit in `git log`; both are in scope of this plan.

## Threat Flags

None. Plan 02-04 introduces no new network endpoints, auth paths, file-access patterns, or trust-boundary surfaces. The plan's `<threat_model>` already enumerates the only two relevant threats (T-02-10 token-price drift mitigated by D-16 leaving cost_usd=None; T-02-11 token-count info disclosure accepted — token counts are not credentials). The implementation matches both dispositions: no token-price table exists in codex.py, and MetricsEvent fields are not redacted.

## Next Phase Readiness

- **Plan 02-05 (event-to-ATIF mapping in agent.run_agent):** Codex runs now flow through `Invocation._dispatch` via the real `isinstance(event, MetricsEvent)` branch (fast path). The class-name fallback in `Invocation._dispatch` becomes dead-code-but-still-correct for Codex too (same as for Claude after Plan 02-03 lands).
- **Plan 02-07 (Harbor-golden round-trip):** Codex trajectories now expose per-turn token counts via `Step.metrics.prompt_tokens` / `Step.metrics.completion_tokens`. Backend-parity assertions can now check that both Claude and Codex Steps carry per-step Metrics with the same field shape (the only documented difference is Codex `cost_usd=None` vs Claude having a USD value).
- **Sibling Plan 02-03 (Claude backend):** the Claude backend's MetricsEvent emission lives in `daydream/backends/claude.py` (parallel work in a separate worktree). After both worktrees merge, the recorder will see MetricsEvents from both backends with identical shape — modulo the D-16 parity gap (Codex `cost_usd=None`, `cached_tokens=None`, `message_id=""`).

## Self-Check: PASSED

All claims verified:

- [x] `daydream/backends/codex.py` updated (457 lines, +24/-1 vs base)
- [x] `tests/test_backend_codex_metrics.py` created (102 lines, 4 tests)
- [x] `tests/fixtures/codex_jsonl/turn_completed_with_usage.jsonl` created (4 lines)
- [x] `tests/fixtures/codex_jsonl/turn_completed_partial_usage.jsonl` created (4 lines)
- [x] Commit `d469c98` exists (RED test) — `git log --oneline | grep d469c98` returns one match
- [x] Commit `f2e4d23` exists (GREEN feat) — `git log --oneline | grep f2e4d23` returns one match
- [x] All 4 tests in `tests/test_backend_codex_metrics.py` pass; the 17 trajectory + 10 backend-events tests still pass; the 47 existing backend tests still pass
- [x] `uv run ruff check daydream/backends/codex.py tests/test_backend_codex_metrics.py` clean
- [x] `uv run mypy daydream/backends/codex.py` clean
- [x] All grep-based plan acceptance criteria pass (yield MetricsEvent = 1, MetricsEvent >= 2, message_id="" = 1, prompt_tokens= = 1, completion_tokens= = 1, MetricsEvent input_tokens = 0, token-price synthesis = 0)
- [x] Both `yield MetricsEvent` and `yield CostEvent` appear in the same `turn.completed` branch
- [x] No accidental file deletions across the two commits (`git diff --diff-filter=D --name-only HEAD~2 HEAD` empty)
- [x] No untracked files left behind (`git status --short` empty post-commits)
- [x] D-16 parity gap honored: `cost_usd=None`, `cached_tokens=None`, `message_id=""` are hardcoded constants
- [x] EVNT-02 boundary rename: backend reads `usage["input_tokens"]` and yields `MetricsEvent(prompt_tokens=...)` — no leak of SDK boundary keys into MetricsEvent

---
*Phase: 02-recorder-core-event-enrichment-mapping*
*Completed: 2026-04-26*
