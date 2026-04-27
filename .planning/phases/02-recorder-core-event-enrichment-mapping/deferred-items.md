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
