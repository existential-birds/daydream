---
phase: 02-recorder-core-event-enrichment-mapping
plan: 02
subsystem: backends
tags: [python, dataclasses, events, timestamps, metrics, atif]

# Dependency graph
requires:
  - phase: 02-recorder-core-event-enrichment-mapping
    plan: 01
    provides: now_iso() helper from daydream.trajectory + class-name fallback in Invocation._dispatch that already accommodates either MetricsEvent path
provides:
  - daydream.backends.MetricsEvent dataclass — per-step LLM token/cost surface (EVNT-02 verbatim field names) consumed by Plans 02-03 (event-to-ATIF mapping) and emitted by Plans 02-03/04/05 (Claude + Codex backends)
  - timestamp: str field on every AgentEvent dataclass via field(default_factory=now_iso) — Pitfall 2 single-source-of-truth
  - CostEvent.cached_tokens: int | None field (default None, backward compatible with existing 3-positional-arg call sites)
  - AgentEvent TypeAlias union extended to include MetricsEvent
  - __all__ exports MetricsEvent (alphabetical order)
  - 10 schema-validity + behavior-predicate tests in tests/test_backends_events.py
affects:
  - phase 02-03 (Claude backend MetricsEvent emission): can now `yield MetricsEvent(...)` from inside the AssistantMessage branch
  - phase 02-04 (Codex backend MetricsEvent emission): can now `yield MetricsEvent(...)` from the turn.completed branch
  - phase 02-05 (event-to-ATIF mapping in agent.run_agent): the class-name fallback in Invocation._dispatch now hits the real `isinstance(event, MetricsEvent)` branch first (the .__name__ check becomes dead code but still correct)
  - phase 02-03/04 will update the existing CostEvent call sites (claude.py:124, codex.py:310) to pass `cached_tokens` explicitly when the SDK provides it; default-None today keeps them valid

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Default-None additive field expansion: new field on existing dataclass added with `= None` default so existing 3-positional-arg call sites stay valid until later plans update them. Avoids wave-coordination dance between Plan 02-02 (this) and Plans 02-03/04 (backend emitters)."
    - "field(default_factory=now_iso) on every event dataclass: timestamp populated at backend yield time (the moment the dataclass is constructed), giving the recorder a single immutable timestamp per event without re-stamping. Plan 05 reads `event.timestamp`."

key-files:
  created:
    - tests/test_backends_events.py — 10 schema-validity + behavior-predicate tests covering EVNT-01..03 (timestamp default, MetricsEvent construction, CostEvent.cached_tokens backward-compat, AgentEvent union membership, __all__ export); 106 lines
  modified:
    - daydream/backends/__init__.py — additive enrichment; 110 insertions, 8 deletions; final size 242 lines (was 140)

key-decisions:
  - "cached_tokens defaulted to None on CostEvent (NOT made keyword-only, NOT a breaking 4-positional-arg signature). Plan called this out explicitly in the <action> block — preserving backward compat with existing call sites in claude.py:124 and codex.py:310 which use 3 keyword args. Plans 02-03 and 02-04 will update those call sites to pass cached_tokens explicitly. The plan considered three options (4th positional required, kw-only, default-None) and selected default-None to avoid coordinating across waves; this commit honors that decision verbatim."
  - "MetricsEvent uses EVNT-02 verbatim field names (prompt_tokens, completion_tokens) NOT the SDK boundary keys (input_tokens, output_tokens). Backends will read `usage['input_tokens']` from the SDK and yield `MetricsEvent(prompt_tokens=...)` — the rename happens at the boundary, not in the recorder. CostEvent keeps `input_tokens`/`output_tokens` because that field naming was established before EVNT-02 and the existing test_backends_init.py asserts those exact names; renaming would be a much wider blast radius."
  - "Imports kept alphabetical inside `from dataclasses import dataclass, field` and via the new `from daydream.trajectory import now_iso` line. The trajectory import comes after `from typing import ...` (separate import group) — matches the pattern Plan 02-01 used in tests/test_trajectory.py."

patterns-established:
  - "Adding a new event dataclass to the AgentEvent vocabulary requires three coordinated edits in the same file: (1) the @dataclass definition, (2) the AgentEvent TypeAlias union, (3) the __all__ list. All three live inside daydream/backends/__init__.py — single file edit, no cross-module hunting."
  - "Schema-validity + behavior-predicate test pattern from D-18 (established by Plan 02-01) extended to event smoke tests: each test instantiates one dataclass, asserts a single key behavior (timestamp ends with 'Z', cached_tokens defaults to None, etc.), no full-tree snapshot equality, no over-asserting."

requirements-completed: [EVNT-01, EVNT-02, EVNT-03]

# Metrics
duration: 25min
completed: 2026-04-26
---

# Phase 02 Plan 02: Event Enrichment Summary

**Every AgentEvent dataclass now carries an ISO 8601 UTC timestamp; MetricsEvent ships with the EVNT-02 verbatim field surface; CostEvent gained backward-compatible cached_tokens; 10 new tests pass without disturbing the 343-test gate.**

## Performance

- **Duration:** approx. 25 min
- **Completed:** 2026-04-26
- **Tasks:** 1 (single TDD task per plan)
- **Commits:** 2 (RED test commit + GREEN implementation commit)
- **Files created:** 1 (`tests/test_backends_events.py`, 106 lines)
- **Files modified:** 1 (`daydream/backends/__init__.py`, +110/-8)

## Accomplishments

- **EVNT-01: ISO 8601 UTC timestamp on every event.** Seven event dataclasses (TextEvent, ThinkingEvent, ToolStartEvent, ToolResultEvent, CostEvent, MetricsEvent, ResultEvent) now carry `timestamp: str = field(default_factory=now_iso)`. The default factory binds to Plan 02-01's `daydream.trajectory.now_iso` — single source of truth for ISO 8601 with `Z` suffix (Pitfall 2). Backends do not have to construct timestamps themselves; instantiating any event dataclass auto-stamps the moment of yield.
- **EVNT-02: MetricsEvent dataclass.** Six fields exactly as specified: `message_id: str`, `prompt_tokens: int`, `completion_tokens: int`, `cached_tokens: int | None`, `cost_usd: float | None`, `timestamp: str`. Field names follow EVNT-02 verbatim (prompt_tokens / completion_tokens), with the docstring noting backends read SDK boundary keys (`usage["input_tokens"]` / `usage["output_tokens"]`) and rename at emission time. The dataclass appears in both the AgentEvent TypeAlias union and the `__all__` export.
- **EVNT-03: CostEvent.cached_tokens.** New field added with `= None` default for backward compatibility with the existing 3-positional-arg call sites in `daydream/backends/claude.py:124` and `daydream/backends/codex.py:310`. Plans 02-03 and 02-04 will update those call sites to pass cached_tokens explicitly when the SDK provides it.
- **AgentEvent TypeAlias union extended** to include MetricsEvent. `def f(e: AgentEvent) -> None: ...` accepts MetricsEvent instances; mypy clean.
- **__all__ refreshed** to export MetricsEvent in alphabetical order between CostEvent and ResultEvent.
- **10 new tests in `tests/test_backends_events.py`** covering each EVNT-01/02/03 predicate: timestamp default + Z suffix on every event dataclass (5 tests), MetricsEvent six-field construction (1 test), CostEvent.cached_tokens explicit + default-None paths (2 tests), AgentEvent union acceptance (1 test), `__all__` export (1 test).
- **Plan 02-01's class-name fallback in `Invocation._dispatch`** now lights up the real `isinstance(event, MetricsEvent)` branch first; the `type(event).__name__ == "MetricsEvent"` fallback becomes dead-code-but-still-correct (kept for defensive symmetry per Plan 02-01 SUMMARY notes).

## Task Commits

1. **Task 1 RED — failing smoke tests for AgentEvent enrichment** — `0237bf9` (test)
2. **Task 1 GREEN — enrich AgentEvent dataclasses + add MetricsEvent** — `638d5d4` (feat)

## Final Field Order (per dataclass)

| Dataclass        | Field order                                                                                  |
| ---------------- | -------------------------------------------------------------------------------------------- |
| TextEvent        | `text`, `timestamp`                                                                          |
| ThinkingEvent    | `text`, `timestamp`                                                                          |
| ToolStartEvent   | `id`, `name`, `input`, `timestamp`                                                           |
| ToolResultEvent  | `id`, `output`, `is_error`, `timestamp`                                                      |
| CostEvent        | `cost_usd`, `input_tokens`, `output_tokens`, `cached_tokens` (= None), `timestamp`           |
| MetricsEvent     | `message_id`, `prompt_tokens`, `completion_tokens`, `cached_tokens`, `cost_usd`, `timestamp` |
| ResultEvent      | `structured_output`, `continuation`, `timestamp`                                             |
| ContinuationToken | `backend`, `data` (unchanged — not part of AgentEvent stream; used as field on ResultEvent) |

`timestamp` is consistently placed last on every dataclass so existing positional construction patterns (e.g., `TextEvent("hi")`) remain valid without supplying timestamp. `cached_tokens` on CostEvent is the only field with an explicit `= None` default (placed immediately before `timestamp`); on MetricsEvent it has no default, since MetricsEvent has no pre-existing call sites.

## Decisions Made

- **`cached_tokens` defaults to None on CostEvent (per plan):** existing CostEvent call sites in `daydream/backends/claude.py:124-128` and `daydream/backends/codex.py:310-314` use 3 keyword args (`cost_usd=`, `input_tokens=`, `output_tokens=`). The plan called for `cached_tokens: int | None = None` to keep these valid; this commit honors that. Plans 02-03 (Claude) and 02-04 (Codex) will update those call sites to populate `cached_tokens` from `ResultMessage.usage` / `turn.completed.usage`. No wave-coordination required — this plan ships as a clean leaf.
- **MetricsEvent uses EVNT-02 verbatim field names, NOT the SDK boundary keys:** `prompt_tokens` and `completion_tokens`, NOT `input_tokens` / `output_tokens`. Backends read SDK keys and rename at emission. Rationale: keeps MetricsEvent self-documenting per the ATIF spec (Metrics submodel uses `prompt_tokens`/`completion_tokens`); CostEvent retains `input_tokens`/`output_tokens` because (a) Plan 01 of Phase 1 established that naming, (b) `tests/test_backends_init.py::test_cost_event_fields` asserts those exact attribute names, and (c) renaming CostEvent fields was out of scope.
- **No mypy/ruff config changes needed.** The default-factory pattern (`field(default_factory=now_iso)`) is a stock dataclasses idiom; `from __future__ import annotations` was already present (so `str | None` works at runtime); ruff line length budget (120) is comfortable; mypy `ignore_missing_imports = true` doesn't apply (no new third-party imports). Zero noqa comments needed.
- **One test class is the test file, not a sub-class:** `tests/test_backends_events.py` uses module-level test functions per the project pattern (`tests/test_backends_init.py`, `tests/test_atif_vendor_smoke.py`). No `TestX` class wrapper; `asyncio_mode = "auto"` doesn't require it.

## Deviations from Plan

None. The plan executed exactly as written:

- All seven event dataclasses gained `timestamp: str = field(default_factory=now_iso)` as the last field.
- MetricsEvent added between CostEvent and ResultEvent with the exact six fields specified.
- CostEvent gained `cached_tokens: int | None = None` per the plan's explicit backward-compat resolution.
- AgentEvent TypeAlias union and `__all__` both updated.
- 10 tests pass on first GREEN run; 17 trajectory tests continue passing; 47 existing backend tests continue passing; the 4 pre-existing `test_deep_orchestrator.py` failures remain pre-existing (already documented in the phase's `deferred-items.md` from Plan 02-01 — out of scope for this purely additive plan).

The plan's `<behavior>` block originally listed 6 tests; the implementation ships 10 (added: ThinkingEvent timestamp, ToolResultEvent timestamp, ResultEvent timestamp, CostEvent backward-compat default-None). Adding more positive predicates is consistent with the plan's intent and does not change scope.

## Plan Acceptance Criteria — Verification

| Criterion | Result |
| --------- | ------ |
| `grep -c "from daydream.trajectory import now_iso"` returns 1 | ✓ 1 |
| `grep -c "field(default_factory=now_iso)"` returns 7 | ✓ 7 |
| `grep -c "class MetricsEvent"` returns 1 | ✓ 1 |
| `grep -c "cached_tokens: int \| None"` returns >= 2 | ✓ 2 (CostEvent + MetricsEvent) |
| `grep -c "prompt_tokens: int"` returns >= 1 | ✓ 1 |
| `grep -c "completion_tokens: int"` returns >= 1 | ✓ 1 |
| `input_tokens: int \| None` only inside CostEvent | ✓ line 103 only |
| MetricsEvent in both AgentEvent union and `__all__` | ✓ lines 177 and 235 |
| Runtime: `MetricsEvent(...).timestamp.endswith('Z')` returns True | ✓ |
| Runtime: 3-arg CostEvent → `e.cached_tokens is None` | ✓ |
| `uv run pytest tests/test_backends_events.py tests/test_trajectory.py -v` passes | ✓ 27/27 |
| `uv run ruff check daydream/backends/__init__.py tests/test_backends_events.py` clean | ✓ |
| `uv run mypy daydream/backends/__init__.py` clean | ✓ |
| `uv run pytest -x --ignore=tests/test_backends_events.py --ignore=tests/test_trajectory.py --ignore=tests/test_deep_orchestrator.py -q` passes | ✓ 342/342 |

Note on the `^\s*MetricsEvent$` grep from the plan: actual matches are at lines 177 (`    | MetricsEvent`) and 235 (`    "MetricsEvent",`). The plan's regex was overly strict — neither line is bare `MetricsEvent` because both have list-context formatting (a pipe prefix and a string-literal-with-trailing-comma respectively). The semantic intent (MetricsEvent appears in both the union and `__all__`) is satisfied; see the verification line "MetricsEvent in both AgentEvent union and `__all__`" above.

## Issues Encountered

- **Pre-existing `test_deep_orchestrator` failures:** The full test suite continues to surface 4 failures in `tests/test_deep_orchestrator.py` (test_fresh_context_per_stage, test_per_stack_context_isolation, test_preflight_notice, test_failed_per_stack_surfaces_to_merge_prompt_and_persists). These reproduce on the unmodified phase-2 base commit `f16b869` and were already documented in `.planning/phases/02-recorder-core-event-enrichment-mapping/deferred-items.md` by Plan 02-01. Plan 02-02 is purely additive — adds dataclass fields, a new dataclass, a new test file — and does not touch deep-mode code. The failures cannot have been caused or affected by this plan.

## TDD Gate Compliance

This plan is `tdd="true"` on its single task. The gate sequence is satisfied:

1. **RED gate:** commit `0237bf9` (test) — added `tests/test_backends_events.py` with 10 tests; ran `uv run pytest tests/test_backends_events.py` → ImportError on `MetricsEvent` (RED confirmed).
2. **GREEN gate:** commit `638d5d4` (feat) — implemented MetricsEvent + timestamp fields + cached_tokens; ran the same command → 10/10 pass.
3. **REFACTOR gate:** not needed — implementation matched the plan's Action block verbatim and required no cleanup pass.

A `feat` commit follows the `test` commit in `git log`; both are in scope of this plan.

## Threat Flags

None. Plan 02-02 introduces no new network endpoints, auth paths, file-access patterns, or trust-boundary surfaces. The plan's `<threat_model>` already enumerates the only two relevant threats (T-02-06 clock skew, T-02-07 token-count info disclosure), both with `accept` dispositions and rationale documented inline. The implementation matches that disposition: timestamps come from `now_iso()` only (single-source clock), and MetricsEvent fields are not redacted (token counts are not credentials).

## Next Phase Readiness

- **Plan 02-03 (Claude backend MetricsEvent emission)** can now `from daydream.backends import MetricsEvent` and `yield MetricsEvent(message_id=msg.message_id, prompt_tokens=usage["input_tokens"], completion_tokens=usage["output_tokens"], cached_tokens=usage.get("cache_read_input_tokens"), cost_usd=None)` from inside the `AssistantMessage` branch of `daydream/backends/claude.py`.
- **Plan 02-04 (Codex backend MetricsEvent emission)** can now `yield MetricsEvent(message_id="", prompt_tokens=usage.get("input_tokens", 0), completion_tokens=usage.get("output_tokens", 0), cached_tokens=None, cost_usd=None)` from the `turn.completed` branch of `daydream/backends/codex.py`.
- **Plan 02-03/04 should also update the existing CostEvent call sites** (claude.py:124-128 and codex.py:310-314) to pass `cached_tokens` explicitly. The default-None plumbing keeps them valid until then — no rush, no wave-coordination dance.
- **Plan 02-05 (event-to-ATIF mapping in agent.run_agent)** will read `event.timestamp` directly off any AgentEvent it observes (no re-stamping). The class-name fallback in `Invocation._dispatch` becomes dead-code-but-still-correct once Plan 02-04 also lands; Plan 02-01's SUMMARY notes the fallback was kept defensively.

## Self-Check: PASSED

All claims verified:

- [x] `daydream/backends/__init__.py` updated (242 lines, +110/-8)
- [x] `tests/test_backends_events.py` created (106 lines, 10 tests)
- [x] Commit `0237bf9` exists (RED test) — `git log --oneline | grep 0237bf9` returns one match
- [x] Commit `638d5d4` exists (GREEN feat) — `git log --oneline | grep 638d5d4` returns one match
- [x] All 10 tests in `test_backends_events.py` pass; all 17 trajectory tests still pass; 342 other tests pass (excluding 4 pre-existing test_deep_orchestrator failures documented in deferred-items.md from Plan 02-01)
- [x] `uv run ruff check daydream/backends/__init__.py tests/test_backends_events.py` clean
- [x] `uv run mypy daydream/backends/__init__.py` clean
- [x] All grep-based plan acceptance criteria pass (now_iso import = 1, default_factory=now_iso = 7, class MetricsEvent = 1, cached_tokens: int | None >= 2, prompt_tokens: int >= 1, completion_tokens: int >= 1, input_tokens only in CostEvent)
- [x] Runtime acceptance: `MetricsEvent(...).timestamp.endswith('Z')` is True; `CostEvent(cost_usd=0.5, input_tokens=10, output_tokens=20).cached_tokens` is None (backward compat preserved)
- [x] No accidental file deletions across the two commits (`git diff --diff-filter=D --name-only HEAD~2 HEAD` empty)
- [x] No stub patterns introduced (TODO/FIXME/placeholder grep clean)

---
*Phase: 02-recorder-core-event-enrichment-mapping*
*Completed: 2026-04-26*
