---
phase: 04-cutover-redaction-cli-surface
plan: 01
subsystem: trajectory
tags: [atif, redaction, sigint, partial, regex, privacy]
dependency_graph:
  requires: [02-01, 02-02, 02-03, 02-04]
  provides: [redactor-regex-dispatch, write-partial, redaction-failed-fallback]
  affects: [daydream/trajectory.py, tests/test_trajectory.py, tests/test_redaction.py]
tech_stack:
  added: []
  patterns: [compiled-regex-module-constants, table-driven-redaction-rules, boundary-catch-redact-or-omit]
key_files:
  created: [tests/test_redaction.py]
  modified: [daydream/trajectory.py, tests/test_trajectory.py]
decisions:
  - "URL-credential rule placed FIRST in _REDACTION_RULES so the captured token isn't double-matched"
  - "Env-var rule placed BEFORE bare API-key rule so OPENAI_API_KEY=sk-* leaks the prefix as [REDACTED_ENV_VAR]"
  - "Pydantic model_copy(update=...) used for all immutable rewrites; passthrough test moved from identity to semantic equality"
  - "REDA-05 redact-or-omit enforced at three layers: per-text, per-arg-value, and outermost where catch falls back to {message: [REDACTION_FAILED]}"
  - "write_partial uses Path.with_suffix(self.path.suffix + .partial) so trajectory.json -> trajectory.json.partial"
metrics:
  tasks_completed: 3
  tasks_total: 3
  tests_added: 22
  tests_total: 56
  files_modified: 2
  files_created: 1
  loc_added: 138
---

# Phase 04 Plan 01: Redactor + write_partial Summary

Filled in the Phase 2 stub Redactor with five regex pattern categories (URL credentials, API keys, JWT, username paths, env-var secrets), wired REDA-05 redact-or-omit fail-safe at three nesting layers, added TrajectoryRecorder.write_partial() for SIGINT flush, shipped 22 new tests.

## Tasks

### Task 1: Redactor regex dispatch (commits 6b8a1d4 test, 3d27f84 impl)

Added to daydream/trajectory.py:

- Five module-level compiled regex constants: _URL_CREDENTIAL_PATTERN, _API_KEY_PATTERN, _JWT_PATTERN, _USERNAME_PATH_PATTERN, _ENV_VAR_PATTERN
- _REDACTION_RULES tuple with order-sensitive dispatch (URL credentials first, env-var second, bare API/JWT/path last)
- Redactor._redact_text flat-regex pipeline iterating _REDACTION_RULES
- Redactor._redact_optional_text for Step.message / Step.reasoning_content with redact-or-omit
- Redactor._redact_arguments for ToolCall.arguments with JSON serialize/redact/parse and per-key fallback
- Redactor._redact_observation for Observation.results[*].content
- Redactor.redact_step walks every text-bearing Step field via model_copy(update=...)

Updated the Phase 2 passthrough test from identity (out is step) to field-by-field semantic equality.

### Task 2: TrajectoryRecorder.write_partial (commits 348c41a test, 96027bf impl)

Synchronous method writing in-flight steps to <path>.partial with extra.partial=true.

- Empty-skip when self.steps is empty (matches _write posture)
- self.path.with_suffix(self.path.suffix + ".partial") so trajectory.json becomes trajectory.json.partial
- Builds via _build_trajectory so per-Step redaction (Phase 2 D-13) is automatically applied
- Sets extra.partial=True on JSON dict, writes JSON
- try/except boundary catches and emits print_warning so flush never crashes shutdown
- Synchronous so signal handler can call without await

Plan 02 wires the CLI signal handler.

### Task 3: tests/test_redaction.py (commit 48aed86)

18 tests organized by REDA category:

| Tests | REDA | Coverage |
|-------|------|----------|
| 4 (sk-, ghp_, xoxb-, AKIA) | REDA-01 | API key positives |
| 1 (eyJ JWT) | REDA-01 | JWT positive |
| 1 (https://oauth2:ghp_...@github.com) | REDA-01 | Git URL credential |
| 3 (/Users/, /home/, C:\\Users\\) | REDA-02 | Username paths |
| 2 (KEY=, PASSWORD=) | REDA-03 | Env-var secrets |
| 3 (DEBUG=true / clean paths / clean URLs) | REDA-03 neg | Negatives |
| 3 (reasoning_content, tool_calls, observation) | REDA-04 | Surface uniformity |
| 1 (monkeypatch _REDACTION_RULES) | REDA-05 | Fail-safe |

Plus 4 write_partial tests added to tests/test_trajectory.py from Task 2.

## Final Regex Patterns

- _URL_CREDENTIAL_PATTERN matches (https?://)([^:@/\s]+):([^@/\s]+)@
- _API_KEY_PATTERN matches sk-, ghp_, xoxb-, AKIA prefixes
- _JWT_PATTERN matches eyJ tripartite JWT format
- _USERNAME_PATH_PATTERN matches /Users/, /home/, C:\\Users\\
- _ENV_VAR_PATTERN matches secret-keyname env-var assignments

_REDACTION_RULES order: URL credential, env var, API key, JWT, username path.

No tuning was required during execution; plan patterns and ordering worked on the first GREEN pass.

## write_partial Method

Builds trajectory through existing _build_trajectory, computes partial path via Path.with_suffix, sets extra.partial=True on the in-memory dict, writes JSON. All wrapped in try/except that emits print_warning. Synchronous, idempotent, empty-skip.

## Test Counts

- tests/test_redaction.py (new): 18 tests
- tests/test_trajectory.py (updated): 38 tests total (was 33 baseline; added 1 RED redaction test, 4 write_partial tests; updated test_redactor_is_passthrough)
- Combined: 56 tests pass cleanly

## Verification

- pytest tests/test_trajectory.py tests/test_redaction.py: 56 passed in 0.48s
- mypy daydream/trajectory.py: Success
- ruff check daydream/trajectory.py tests/test_redaction.py: All checks passed
- grep "def write_partial|def _redact_text|def redact_step" daydream/trajectory.py: exactly 3
- daydream/trajectory.py LOC: 613 -> 751 (+138)

## Deviations from Plan

None. Plan regex patterns, dispatch order, fail-safe layering, and write_partial signature used verbatim.

## Threat Mitigations Applied

- T-04-01: Redactor.redact_step applies _REDACTION_RULES uniformly to all four ATIF surfaces
- T-04-02: _redact_optional_text covers reasoning_content
- T-04-03: _USERNAME_PATH_PATTERN with project-relative tail preservation
- T-04-04: Three-layer try/except produces [REDACTION_FAILED]
- T-04-05: write_partial reuses _build_trajectory; already-redacted steps flow through
- T-04-21: _URL_CREDENTIAL_PATTERN runs first; positive test asserts neither user nor token survives

## Threat Flags

None.

## Known Stubs

None.

## Deferred Issues

Pre-existing failures in the worktree env (NOT caused by this plan):

- tests/test_phases.py and tests/test_integration.py failures: subprocess git commit fixtures lack user.email/user.name in sandbox runtime. Already failing on base commit 8aedd5e.
- tests/test_deep_integration.py and tests/test_loop.py errors same category.
- Pre-existing F401 / I001 ruff errors in tests/test_trajectory.py import block (CostEvent, ThinkingEvent unused). Out of scope per SCOPE BOUNDARY; plan verification targets tests/test_redaction.py only.

## Self-Check: PASSED

- daydream/trajectory.py exists at 751 LOC
- tests/test_redaction.py exists with 18 def test_redactor_* definitions
- tests/test_trajectory.py updated
- All 5 commit hashes present in git log 8aedd5e..HEAD:
  - 6b8a1d4 test(04-01): add failing redaction test + semantic-equality passthrough
  - 3d27f84 feat(04-01): implement Redactor regex dispatch (REDA-01..05)
  - 348c41a test(04-01): add failing TrajectoryRecorder.write_partial tests
  - 96027bf feat(04-01): add TrajectoryRecorder.write_partial for SIGINT flush
  - 48aed86 test(04-01): add tests/test_redaction.py with REDA-01..06 coverage
