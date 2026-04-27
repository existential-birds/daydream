# Deferred Items — Phase 02-recorder-core-event-enrichment-mapping

Out-of-scope discoveries logged during execution; not fixed by the current plan.

## Pre-existing test failures (out of scope, present on phase 2 base commit f16b869)

Discovered while running the full test suite during plan 02-01 execution
(2026-04-26). All four failures reproduce on the unmodified base commit
(`git stash` + retry confirms): they are NOT caused by Plan 02-01 changes.

- `tests/test_deep_orchestrator.py::test_fresh_context_per_stage`
- `tests/test_deep_orchestrator.py::test_per_stack_context_isolation`
- `tests/test_deep_orchestrator.py::test_preflight_notice` (assert 7 == 9)
- `tests/test_deep_orchestrator.py::test_failed_per_stack_surfaces_to_merge_prompt_and_persists`

These do not block Plan 02-01 (which is purely additive — new module +
new tests). Existing 343-test gate is honored if these are excluded as
pre-existing flakes; a separate plan should investigate.

## gpg-agent flake on `tests/test_deep_integration.py` (out of scope, env-dependent)

Discovered while running the full test suite during plan 02-04 execution
(2026-04-26). The conftest fixture `multi_stack_target` at
`tests/conftest.py:71-100` runs `git init` + `git commit -m 'init'` in a
tempdir. If the user's global `~/.gitconfig` has `commit.gpgsign = true`
(default for some developer setups), the fixture commit fails with:

```
error: gpg failed to sign the data
gpg: signing failed: No agent running
```

— non-deterministically, depending on whether gpg-agent has been
recently exercised in the shell. Workaround: run pytest with
`GIT_CONFIG_GLOBAL=/dev/null`. Permanent fix would be a one-line
`git config commit.gpgsign false` inside the fixture's setup block —
out of scope for Plan 02-04 (which only touches Codex backend code).

Suggested follow-up: a tiny conftest hardening plan adds
`commit.gpgsign=false` and `tag.gpgsign=false` to all fixture-created
git repos for deterministic CI behavior.

## Pre-existing lint and typecheck findings (out of scope, present on base 2fa51a7)

Discovered while running `uv run ruff check daydream/ tests/` and
`uv run mypy daydream/` during plan 02-06 execution (2026-04-26). All
findings reproduce on the unmodified base commit `2fa51a7`:

- `tests/test_pr_review.py:111` — E501 line-length violation (128 > 120)
- `tests/test_trajectory.py:19` — F401 unused import `CostEvent`
- `tests/test_trajectory.py:22` — F401 unused import `ThinkingEvent`
- `daydream/trajectory.py:241` — mypy `Cannot assign to a type [misc]`
  in the `MetricsEvent = None  # type: ignore[assignment]` fallback
  inside `Invocation._dispatch`'s function-local import block. The
  existing `# type: ignore[assignment]` comment doesn't cover the
  `[misc]` code mypy flags here.

These are all in code untouched by Plan 02-06 (test fixtures + Plan 01's
`trajectory.py`). A tiny lint-cleanup plan should resolve them; out of
scope for the current plan because the SCOPE BOUNDARY rule limits 02-06
to the call-site update + recorder construction in 4 source files.
