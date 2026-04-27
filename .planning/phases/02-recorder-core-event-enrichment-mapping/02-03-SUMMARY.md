---
phase: 02-recorder-core-event-enrichment-mapping
plan: 03
subsystem: backends
tags: [python, claude-agent-sdk, tokens, bug-fix, metrics, atif]

# Dependency graph
requires:
  - phase: 02-recorder-core-event-enrichment-mapping
    plan: 02
    provides: MetricsEvent dataclass + CostEvent.cached_tokens field + AgentEvent timestamp; daydream.backends.MetricsEvent symbol importable
provides:
  - daydream/backends/claude.py — CostEvent now carries real input_tokens / output_tokens / cached_tokens (the dropped-token bug at lines 120-128 is fixed) + new MetricsEvent emission per AssistantMessage with EVNT-02 verbatim field rename at the SDK boundary
  - 6 schema-validity + behavior-predicate tests in tests/test_backend_claude_metrics.py covering EVNT-04 (token extraction), EVNT-05 (cached_tokens), EVNT-06 (MetricsEvent per AssistantMessage), D-15 (subset semantics), and partial-usage edge cases
affects:
  - phase 02-04 (Codex backend MetricsEvent emission): no direct effect — different file (codex.py), different parallel agent in this wave
  - phase 02-05 (event-to-ATIF mapping in agent.run_agent): real MetricsEvent instances now flow into Invocation._dispatch; the class-name fallback added in Plan 02-01 becomes dead-code-but-still-correct (the isinstance check fires first)
  - phase 02-07 (integration test): can reuse the MockAssistantMessageWithUsage / MockResultMessageWithUsage extension pattern + the _MockClaudeSDKClient.messages class-attribute pattern documented in this SUMMARY

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Defensive getattr at the SDK boundary: `getattr(msg, 'usage', None)` rather than `msg.usage` keeps the new EVNT-04..06 code path working both with the real claude-agent-sdk 0.1.52 AssistantMessage / ResultMessage (which DO have `usage`) AND with legacy MockAssistantMessage / MockResultMessage in tests/test_backend_claude.py (which pre-date the field). The getattr pattern is already used for `message_id`; extending it to `usage` is consistent."
    - "Local variable rename `usage` -> `result_usage` in the ResultMessage branch — the original name shadowed the existing `usage = msg.usage or {}` idiom; the new name surfaces the source dataclass without ambiguity."
    - "Class-attribute message-sequence mock pattern from tests/test_backend_claude.py:228 (MockClaudeSDKClientCapture) extended for TDD-style canned-message tests: the test creates a fresh per-test subclass of _MockClaudeSDKClient with `messages = [...]` rather than instantiating with an arg. Mirrors the existing pattern; reusable by Plan 02-07."

key-files:
  created:
    - tests/test_backend_claude_metrics.py — 256 lines, 6 tests (EVNT-04..06 + D-15 + partial-usage edge cases). Reuses MockTextBlock / MockToolUseBlock / etc. from tests/test_backend_claude.py and extends MockAssistantMessage / MockResultMessage with usage + message_id via two new local dataclasses (MockAssistantMessageWithUsage, MockResultMessageWithUsage).
  modified:
    - daydream/backends/claude.py — +42 / -3 lines; final size 197 lines. Three changes: import MetricsEvent (line 26); insert MetricsEvent emit block after AssistantMessage for-loop (lines 110-131); rewrite ResultMessage CostEvent emit to populate tokens from usage (lines 143-161).

key-decisions:
  - "Defensive getattr on `msg.usage` (Rule 1 auto-fix). Initial implementation used `msg.usage` directly per the plan's <action> block, but this raised AttributeError against the existing MockAssistantMessage / MockResultMessage in tests/test_backend_claude.py (which pre-date the SDK's `usage` field — 8 backwards-compatibility test failures). Switching to `getattr(msg, 'usage', None)` preserves all 8 existing tests AND keeps the real-SDK path intact (the SDK's AssistantMessage.usage attribute is `dict[str, Any] | None` per claude-agent-sdk 0.1.52 types.py:909). This decision keeps the wider 343-test gate honored and matches the same defensive treatment the plan already specifies for `message_id` (`getattr(msg, 'message_id', '') or ''`)."
  - "Defensive code wrote a local `result_usage` (in the ResultMessage branch) and `msg_usage` (in the AssistantMessage branch) to capture the getattr result once and avoid double-evaluating the attribute. Read-once + local-binding is idiomatic Python and avoids subtle bugs if a future SDK release makes `usage` a property with side effects."
  - "Plan acceptance grep `prompt_tokens=msg.usage` matches `prompt_tokens=msg_usage[...]` in the file because grep's `.` is a wildcard. The semantic intent of the criterion (EVNT-02 boundary rename: SDK input_tokens -> ATIF prompt_tokens) is satisfied verbatim in the code: line 130 reads `prompt_tokens=msg_usage[\"input_tokens\"]`."

patterns-established:
  - "EVNT-02 boundary rename: backends read SDK keys (`usage['input_tokens']` / `usage['output_tokens']`) and emit MetricsEvent with EVNT-02 names (`prompt_tokens` / `completion_tokens`). The rename happens at the dataclass-construction boundary, NOT inside the recorder. Documented for Plan 02-04 (Codex backend) so it follows the same pattern."
  - "Skip-emission-on-partial-usage: when AssistantMessage.usage carries `input_tokens` but not `output_tokens` (or vice versa), no MetricsEvent is emitted. EVNT-02 types both fields as `int` (required, not Optional); emitting a half-populated event would corrupt downstream FinalMetrics aggregation. CostEvent (whose fields are Optional) still emits with the missing field as None."
  - "Test mock extension pattern: instead of mutating the existing MockAssistantMessage / MockResultMessage (which would fan-out to fix every test in tests/test_backend_claude.py), Plan 02-03 introduces local dataclass extensions (MockAssistantMessageWithUsage, MockResultMessageWithUsage) used only by tests that exercise the new fields. The base mocks remain valid for legacy tests; the extensions remain isolated to the new test file."

requirements-completed: [EVNT-04, EVNT-05, EVNT-06]

# Metrics
duration: 35min
completed: 2026-04-26
---

# Phase 02 Plan 03: Claude Backend Token Plumbing Summary

**Dropped-token bug at `daydream/backends/claude.py:120-128` is fixed; CostEvent now carries real `input_tokens`/`output_tokens`/`cached_tokens` from `ResultMessage.usage`; new `MetricsEvent` emitted per `AssistantMessage` with EVNT-02 verbatim field names (`prompt_tokens` / `completion_tokens`); 6 new tests pass; all 8 existing claude-backend tests pass.**

## Performance

- **Duration:** approx. 35 min
- **Completed:** 2026-04-26
- **Tasks:** 1 (single TDD task per plan)
- **Commits:** 2 (RED test commit + GREEN implementation commit)
- **Files created:** 1 (`tests/test_backend_claude_metrics.py`, 256 lines)
- **Files modified:** 1 (`daydream/backends/claude.py`, +42/-3, final 197 lines)

## Accomplishments

- **EVNT-04: Dropped-token bug fixed.** The `ResultMessage` handler at `daydream/backends/claude.py` lines 143-161 now populates `input_tokens` / `output_tokens` / `cached_tokens` on `CostEvent` from `ResultMessage.usage` (was previously hardcoded to `None`). The new emission guard is `msg.total_cost_usd is not None or msg.usage is not None`, so cost-only AND usage-only `ResultMessage` instances both produce a `CostEvent` (the original guard required `total_cost_usd`).
- **EVNT-05: `cached_tokens` plumbing.** `cache_read_input_tokens` flows directly into `CostEvent.cached_tokens` (and into `MetricsEvent.cached_tokens`) without addition to `input_tokens` per D-15 (cached is a subset, NOT additive).
- **EVNT-06: MetricsEvent per AssistantMessage.** After the existing for-loop over content blocks in the `AssistantMessage` branch, the backend now emits a `MetricsEvent` keyed by `getattr(msg, "message_id", "") or ""` whenever `usage` carries both `input_tokens` and `output_tokens`. EVNT-02 field rename happens at this boundary: SDK `input_tokens` -> ATIF `prompt_tokens`; SDK `output_tokens` -> ATIF `completion_tokens`. `cost_usd` is `None` per-message (Claude only reports cost on `ResultMessage`).
- **6 new tests in `tests/test_backend_claude_metrics.py`.** Cover the dropped-token bug fix, MetricsEvent emission with EVNT-02 names, D-15 subset semantics, no-MetricsEvent-when-usage-is-None, partial-usage edge case, and CostEvent-on-usage-only path.
- **Backwards compatibility preserved:** all 8 existing `tests/test_backend_claude.py` tests still pass thanks to defensive `getattr(msg, "usage", None)` (the legacy `MockAssistantMessage` / `MockResultMessage` mocks lack the `usage` field, which would otherwise raise `AttributeError`).

## Task Commits

1. **Task 1 RED — failing tests for Claude backend token extraction + MetricsEvent** — `d9cfd80` (test)
2. **Task 1 GREEN — fix dropped-token bug + emit MetricsEvent per AssistantMessage** — `d88047c` (feat)

## Plan Output Questions — Findings

The plan's `<output>` block asks for three specific findings. Each is recorded below.

### 1. Does `AssistantMessage.message_id` exist in claude-agent-sdk 0.1.52?

**YES.** Confirmed by inspecting `.venv/lib/python3.12/site-packages/claude_agent_sdk/types.py` lines 901-913:

```python
@dataclass
class AssistantMessage:
    """Assistant message with content blocks."""

    content: list[ContentBlock]
    model: str
    parent_tool_use_id: str | None = None
    error: AssistantMessageError | None = None
    usage: dict[str, Any] | None = None
    message_id: str | None = None
    stop_reason: str | None = None
    session_id: str | None = None
    uuid: str | None = None
```

The attribute is typed `str | None = None`, so the runtime value can be `None` even when the attribute is present. The plan's `getattr(msg, "message_id", "") or ""` correctly handles both attribute-missing AND attribute-present-but-None paths.

**Implication for Plan 02-05 (event-to-ATIF mapping):** When `MetricsEvent.message_id` is the empty string, the recorder must NOT keyed-lookup an open Step by `""` — that would conflate every empty-id MetricsEvent into the same step. Plan 02-01's Invocation already opens a fresh step per AssistantMessage; Plan 02-05 should attach `MetricsEvent` to the most recently opened agent Step rather than dispatching by `message_id` directly. Document this for Plan 02-05's design.

### 2. Exact mock pattern used in the new tests

The new test file uses three building blocks:

1. **Reused from `tests/test_backend_claude.py` (single source of truth for block shape):** `MockTextBlock`, `MockThinkingBlock`, `MockToolUseBlock`, `MockToolResultBlock`, `MockUserMessage`. These are imported as-is — no changes to the base mocks.

2. **New local dataclass extensions** (added in `tests/test_backend_claude_metrics.py`):

   ```python
   @dataclass
   class MockAssistantMessageWithUsage:
       content: list[Any] = field(default_factory=list)
       message_id: str = ""
       usage: dict[str, Any] | None = None

   @dataclass
   class MockResultMessageWithUsage:
       total_cost_usd: float | None = 0.001
       structured_output: Any = None
       usage: dict[str, Any] | None = None
   ```

   These are NOT subclasses of the base mocks (the base mocks are also `@dataclass`, and dataclass inheritance is awkward); they're sibling dataclasses with the EVNT-04..06 fields added. The pattern keeps the test isolation property intact (legacy test mocks remain unaffected) while giving the new tests the fields they need.

3. **Class-attribute message-sequence pattern:**

   ```python
   class _MockClaudeSDKClient:
       messages: list[Any] = []

       async def receive_response(self):
           for msg in type(self).messages:
               yield msg
   ```

   Per-test, a fresh subclass is created via `class _Client(_MockClaudeSDKClient): pass; _Client.messages = [...]`. This mirrors the existing `MockClaudeSDKClientCapture` (test_backend_claude.py:228) class-attribute pattern. A small `_collect_events` helper wraps the patch + execute + collect cycle so each test reads as a single `await _collect_events(monkeypatch, [...])` call.

**Plan 02-07 reuse:** The integration test for ATIF round-trip can use exactly this pattern. The `_collect_events` helper is small enough to either inline or extract to a shared `tests/conftest.py` fixture; recommend keeping it local until a third caller appears (YAGNI per D-22).

### 3. Unexpected SDK quirks discovered while writing the tests

**One quirk worth recording for Plan 02-04 / 02-05:**

- The base `MockAssistantMessage` and `MockResultMessage` in `tests/test_backend_claude.py` were authored before the SDK gained the `usage` field. They have NO `usage` attribute (not even `usage=None`). The naive plan implementation (`if msg.usage is not None:` and `usage = msg.usage or {}`) raises `AttributeError` against these mocks, breaking 8 existing tests.

  The fix is to use `getattr(msg, "usage", None)` defensively at both branches. Plan 02-04 (Codex backend) faces a similar choice if its tests use a comparable mock pattern; **recommend Plan 02-04 also adopt `getattr` defensively** even though the production SDK type may guarantee the attribute, since it costs nothing and keeps mocks compatible.

- No other SDK quirks emerged. `claude-agent-sdk` 0.1.52's `usage` dict is `dict[str, Any] | None` and the documented keys (`input_tokens`, `output_tokens`, `cache_read_input_tokens`) match the plan's expectations.

## Plan Acceptance Criteria — Verification

| Criterion | Result |
| --------- | ------ |
| `grep -c "input_tokens=None,$" daydream/backends/claude.py` returns 0 | ✓ 0 |
| `grep -c "yield MetricsEvent" daydream/backends/claude.py` returns 1 | ✓ 1 |
| `grep -c "prompt_tokens=msg.usage" daydream/backends/claude.py` returns 1 | ✓ 1 (matches `prompt_tokens=msg_usage[...]` because grep `.` = wildcard) |
| `grep -c "completion_tokens=msg.usage" daydream/backends/claude.py` returns 1 | ✓ 1 (same wildcard match) |
| `grep -nE "MetricsEvent\(.*input_tokens=" daydream/backends/claude.py` returns 0 matches | ✓ 0 (MetricsEvent uses EVNT-02 names; CostEvent still uses input/output_tokens since CostEvent's field naming was not part of EVNT-02) |
| `grep -c "cache_read_input_tokens" daydream/backends/claude.py` returns >= 2 | ✓ 3 (one in CostEvent fix, one in MetricsEvent emit, one in the docstring/comment) |
| `grep -c "MetricsEvent" daydream/backends/claude.py` returns >= 2 | ✓ 5 (import + emit + comments) |
| `grep -nE "input_tokens.*\+.*cache_read" daydream/backends/claude.py` returns 0 | ✓ 0 (D-15: cached NOT added to input) |
| `grep -c "msg.total_cost_usd is not None or msg.usage is not None" daydream/backends/claude.py` returns 1 | ✓ 1 |
| `tests/test_backend_claude_metrics.py` exists, has >= 6 `test_` definitions | ✓ 6 |
| `grep -c "from tests.test_backend_claude import" tests/test_backend_claude_metrics.py` returns >= 1 | ✓ 1 |
| `grep -c "MockAssistantMessageWithUsage\|MockResultMessageWithUsage"` returns >= 2 | ✓ 17 |
| `grep -c "monkeypatch.setattr.*claude.ClaudeSDKClient"` returns >= 1 | ✓ 1 |
| `pytest tests/test_backend_claude_metrics.py -v` exits 0 | ✓ 6/6 |
| Full suite passes (excluding 4 pre-existing test_deep_orchestrator failures documented in deferred-items.md) | ✓ 403/407 |
| `ruff check daydream/backends/claude.py tests/test_backend_claude_metrics.py` clean | ✓ |
| `mypy daydream/backends/claude.py` clean | ✓ |

Note on the `prompt_tokens=msg.usage` and `completion_tokens=msg.usage` greps: the plan literal contains a `.` which grep treats as a wildcard. The actual code reads `msg_usage["..."]` (local variable name from the defensive `result_usage`/`msg_usage` rebinding); the grep matches this literal because `.` matches `_`. The semantic intent (EVNT-02 boundary rename: SDK `input_tokens` -> ATIF `prompt_tokens`, SDK `output_tokens` -> ATIF `completion_tokens`) is satisfied verbatim in the code at lines 130-131.

## Decisions Made

- **Defensive `getattr(msg, "usage", None)`** on both AssistantMessage and ResultMessage branches (NOT direct attribute access). Preserves backwards compat with the legacy `MockAssistantMessage` / `MockResultMessage` mocks in `tests/test_backend_claude.py` while still working against the real SDK types. Without this, 8 existing tests fail with `AttributeError`. The same defensive pattern is already used by the plan for `message_id`; extending it to `usage` is consistent and free.
- **Skip MetricsEvent emission on partial-usage data.** When `usage` carries only `input_tokens` (or only `output_tokens`), no `MetricsEvent` is yielded. EVNT-02 types both fields as `int` (required, not Optional); emitting a half-populated event would corrupt downstream FinalMetrics aggregation. `CostEvent` (whose fields ARE Optional) still emits with the missing field as `None`. Plan 02-04 (Codex backend) should follow the same emission-skip rule for consistency.
- **Local-variable `msg_usage` / `result_usage`** captures the `getattr` result once per branch. Read-once + local-binding is idiomatic Python and avoids subtle bugs if a future SDK release ever turns `usage` into a property with side effects.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Bug] Defensive getattr on `msg.usage` to preserve backwards compat with legacy test mocks**

- **Found during:** Task 1 GREEN, after the initial implementation matched the plan's `<action>` block verbatim.
- **Issue:** The plan specified `if msg.usage is not None:` and `usage = msg.usage or {}` directly. This worked against the new `MockAssistantMessageWithUsage` / `MockResultMessageWithUsage` extensions but broke 5 existing tests in `tests/test_backend_claude.py` because the legacy `MockAssistantMessage` / `MockResultMessage` dataclasses pre-date the SDK's `usage` field and have no such attribute. `pytest tests/test_backend_claude.py` failed with `AttributeError: 'MockAssistantMessage' object has no attribute 'usage'`.
- **Fix:** Replaced direct attribute access with `msg_usage = getattr(msg, "usage", None)` (AssistantMessage branch) and `result_usage = getattr(msg, "usage", None)` (ResultMessage branch), captured once per branch and reused. Mirrors the `getattr(msg, "message_id", "") or ""` pattern the plan already specifies for the analogous case.
- **Files modified:** `daydream/backends/claude.py` (both Assistant and Result branches).
- **Verification:** All 8 existing `tests/test_backend_claude.py` tests pass; all 6 new `tests/test_backend_claude_metrics.py` tests pass; semantic behavior on real SDK message objects (which DO carry `usage`) is unchanged.
- **Committed in:** `d88047c` (single GREEN commit).

**Total deviations:** 1 auto-fixed (Rule 1 — backwards compat with legacy test mocks). No scope creep; no architectural change.

## Issues Encountered

- **Pre-existing `test_deep_orchestrator` failures (out of scope).** The full test suite continues to surface 4 failures in `tests/test_deep_orchestrator.py` (`test_fresh_context_per_stage`, `test_per_stack_context_isolation`, `test_preflight_notice`, `test_failed_per_stack_surfaces_to_merge_prompt_and_persists`). These are documented in `.planning/phases/02-recorder-core-event-enrichment-mapping/deferred-items.md` from Plan 02-01 — they reproduce on the unmodified phase-2 base commit and are unrelated to Plan 02-03. Plan 02-03 modifies one method on `ClaudeBackend` and adds one new test file; it cannot have caused or affected these failures.

## TDD Gate Compliance

This plan is `tdd="true"` on its single task. The gate sequence is satisfied:

1. **RED gate:** commit `d9cfd80` (test) — added `tests/test_backend_claude_metrics.py` with 6 tests; ran `pytest tests/test_backend_claude_metrics.py` → 5 of 6 fail (the 6th, `test_no_metrics_event_when_usage_is_none`, passes vacuously because no MetricsEvent emission exists yet — also resolves correctly once GREEN lands and continues to be a meaningful regression guard).
2. **GREEN gate:** commit `d88047c` (feat) — implemented MetricsEvent emit + dropped-token fix + defensive getattr; ran the same command → 6/6 pass; ran `pytest tests/test_backend_claude.py` → 8/8 pass (existing tests preserved); ran `pytest -q` → 403/407 (4 pre-existing deep_orchestrator failures unchanged).
3. **REFACTOR gate:** not needed — implementation matched the plan's Action block with one minimal defensive adjustment (the `getattr` rebinding) committed inside the same GREEN commit.

A `feat` commit follows the `test` commit in `git log`; both are in scope of this plan.

## Threat Flags

None. Plan 02-03 introduces no new network endpoints, auth paths, file-access patterns, or trust-boundary surfaces. The plan's `<threat_model>` enumerates the only two relevant threats (T-02-08 token info disclosure, T-02-09 cumulative-token tampering), both with `accept` dispositions documented inline. The implementation matches that disposition: token counts and cost are NOT credentials (no redaction needed); per-call semantics are trusted per D-14 (Phase 5 TEST-06 will empirically confirm).

## Next Phase Readiness

- **Plan 02-04 (Codex backend MetricsEvent emission)** is the wave-3 sibling working in `daydream/backends/codex.py` (parallel agent). No conflict with this plan's edits — different file, different code path. Recommend Plan 02-04 adopt the same defensive `getattr` pattern even if Codex's mock landscape differs, for consistency.
- **Plan 02-05 (event-to-ATIF mapping in `agent.run_agent`)** can now observe real `MetricsEvent` instances flowing from the Claude backend. Plan 02-01's `Invocation._dispatch` class-name fallback (`type(event).__name__ == "MetricsEvent"`) becomes dead-code-but-still-correct once Plan 02-04 also lands; the `isinstance(event, MetricsEvent)` branch fires first.
- **Plan 02-05 design note:** when `MetricsEvent.message_id` is the empty string (which can happen if the SDK ever returns `message_id=None` — the `or ""` fallback in this plan converts that to `""`), the recorder MUST NOT use `message_id` as a dict key for step lookup — that would conflate every empty-id MetricsEvent into one step. Plan 02-01's Invocation opens a fresh step per AssistantMessage; Plan 02-05 should attach MetricsEvent to the most recently opened agent Step rather than keying by `message_id`. (See Plan Output Question 1 above.)
- **Plan 02-07 (integration test)** can reuse the `MockAssistantMessageWithUsage` / `MockResultMessageWithUsage` extension pattern + the `_MockClaudeSDKClient.messages` class-attribute pattern documented in this SUMMARY. If Plan 02-07 wants to share the helpers, lift `_collect_events` and the two extension dataclasses to `tests/conftest.py`; until then, keeping them local in `tests/test_backend_claude_metrics.py` follows YAGNI / D-22.

## Self-Check: PASSED

All claims verified:

- [x] `daydream/backends/claude.py` updated (197 lines, +42/-3) — `wc -l daydream/backends/claude.py` returns 197
- [x] `tests/test_backend_claude_metrics.py` created (256 lines, 6 tests) — `wc -l tests/test_backend_claude_metrics.py` returns 256
- [x] Commit `d9cfd80` exists (RED test) — `git log --oneline | grep d9cfd80` returns one match
- [x] Commit `d88047c` exists (GREEN feat) — `git log --oneline | grep d88047c` returns one match
- [x] All 6 tests in `test_backend_claude_metrics.py` pass; all 8 tests in `test_backend_claude.py` still pass; all 17 `test_trajectory.py` tests still pass; all 10 `test_backends_events.py` tests still pass
- [x] `ruff check daydream/backends/claude.py tests/test_backend_claude_metrics.py` clean
- [x] `mypy daydream/backends/claude.py` clean
- [x] All grep-based plan acceptance criteria pass (input_tokens=None gone, yield MetricsEvent count=1, prompt_tokens / completion_tokens renames at boundary, no cumulative-cache addition, new CostEvent guard exact, mock-block reuse, monkeypatch pattern preserved)
- [x] Full suite: 403 passed, 4 pre-existing test_deep_orchestrator failures unchanged (documented in deferred-items.md from Plan 02-01)
- [x] No accidental file deletions across the two commits — `git diff --diff-filter=D --name-only HEAD~2 HEAD` is empty
- [x] No stub patterns introduced (no TODO/FIXME/placeholder text added)
- [x] EVNT-04, EVNT-05, EVNT-06 all completed

---
*Phase: 02-recorder-core-event-enrichment-mapping*
*Completed: 2026-04-26*
