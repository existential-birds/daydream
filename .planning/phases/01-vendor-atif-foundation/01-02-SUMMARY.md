---
phase: 01-vendor-atif-foundation
plan: 02
subsystem: infra
tags: [pyproject, pydantic, ruff, vendoring, dependencies]

# Dependency graph
requires: []
provides:
  - Explicit pydantic>=2.11.7 declared in [project.dependencies]
  - [tool.ruff.lint.per-file-ignores] sub-table exempting daydream/atif/** from ruff lint rules
affects: [01-01, 01-03, 01-04, 02, 03, 04, 05]

# Tech tracking
tech-stack:
  added: [pydantic (promoted from transitive to explicit)]
  patterns:
    - "Vendored-source ruff exemption via [tool.ruff.lint.per-file-ignores]"

key-files:
  created: []
  modified:
    - pyproject.toml
    - uv.lock

key-decisions:
  - "Inserted pydantic>=2.11.7 between pyfiglet and tree-sitter family (preserves the existing core/tree-sitter grouping in dependencies list)"
  - "Ruff per-file-ignores list mirrors [tool.ruff.lint] select (E, F, I, W) so the stanza stays robust if select is later expanded"

patterns-established:
  - "Vendored Harbor source under daydream/atif/** is exempt from ruff lint rules — D-03 mechanical-only edit policy enforced via [tool.ruff.lint.per-file-ignores]"
  - "Dependencies that flow in transitively but are imported directly by daydream code must be promoted to explicit [project.dependencies] with a floor"

requirements-completed: [VEND-03]

# Metrics
duration: 2min
completed: 2026-04-26
---

# Phase 01 Plan 02: pyproject.toml — explicit pydantic dep + ruff per-file-ignores Summary

**Explicit `pydantic>=2.11.7` floor declared and `daydream/atif/**` exempted from ruff lint via `[tool.ruff.lint.per-file-ignores]`, with zero dependency-resolution churn (pydantic stays at 2.12.5).**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-04-26T17:39:09Z
- **Completed:** 2026-04-26T17:40:25Z
- **Tasks:** 1
- **Files modified:** 2 (pyproject.toml, uv.lock)

## Accomplishments

- `pydantic>=2.11.7` promoted from transitive (via `claude-agent-sdk==0.1.52` → `mcp` → `pydantic`) to explicit `[project.dependencies]` entry — VEND-03 satisfied
- `[tool.ruff.lint.per-file-ignores]` sub-table added with `"daydream/atif/**" = ["E", "F", "I", "W"]` — protects vendored Harbor source from ruff rewrites, preserving D-03 mechanical-only edit policy
- `uv sync` is a clean no-op for resolution: `Resolved 53 packages in 3ms` / `Checked 51 packages in 7ms` — pydantic stays at 2.12.5 (verified via `uv pip list | grep pydantic`)

## Diff Applied

```diff
@@ pyproject.toml @@
 dependencies = [
     "claude-agent-sdk==0.1.52",
     "anyio>=4.0",
     "rich>=13.0",
     "pyfiglet>=1.0",
+    "pydantic>=2.11.7",
     "tree-sitter==0.25.2",

@@ pyproject.toml @@
 [tool.ruff.lint]
 select = ["E", "F", "I", "W"]

+[tool.ruff.lint.per-file-ignores]
+# Vendored from Harbor v0.5.0; see daydream/atif/NOTICE.
+# Mechanical-only edit policy (D-03): no reformatting allowed.
+"daydream/atif/**" = ["E", "F", "I", "W"]
+
 [tool.pytest.ini_options]
```

`git diff pyproject.toml | grep -E '^[-+]' | grep -v '^[-+][-+][-+]' | wc -l` → **6** (matches acceptance criterion: 1 pydantic line + 5 ruff stanza lines including the comments and blank separator).

## uv sync output

```
Resolved 53 packages in 3ms
Checked 51 packages in 7ms
```

No "Updated package" lines; no version changes. `grep -E 'Updated.*pydantic' uv_sync_output` returned no matches (exit 1 — the negative-match acceptance criterion).

`uv pip list | grep -i pydantic`:
```
pydantic                  2.12.5
pydantic-core             2.41.5
pydantic-settings         2.12.0
```

Pydantic resolved to 2.12.5 as expected — well above the >=2.11.7 floor.

## Task Commits

1. **Task 2.1: Add explicit pydantic dependency and ruff per-file-ignores stanza to pyproject.toml** — `43a01d0` (feat)

## Files Created/Modified

- `pyproject.toml` — added `pydantic>=2.11.7` to `[project.dependencies]`; appended `[tool.ruff.lint.per-file-ignores]` sub-table with daydream/atif/** exemption
- `uv.lock` — metadata-only update: pydantic now appears in the project's direct `dependencies` and `requires-dist` blocks (no version churn)

## Decisions Made

- Inserted `pydantic>=2.11.7` between `pyfiglet>=1.0` and `tree-sitter==0.25.2` to preserve the existing visual grouping (core deps then tree-sitter family). The list isn't strictly alphabetized today; this position reads cleanly.
- Ruff per-file-ignores list mirrors the `[tool.ruff.lint] select` list exactly (`["E", "F", "I", "W"]`) so the stanza stays robust if `select` is later expanded. Comments anchor the stanza to its purpose (D-03, Harbor v0.5.0 vendor source).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] Committed uv.lock alongside pyproject.toml**
- **Found during:** Task 2.1 (post-`uv sync` git status check)
- **Issue:** Plan's `<files>` list only declared `pyproject.toml`, but promoting pydantic from transitive to explicit dep causes uv to record it in the project's direct `dependencies` and `requires-dist` blocks of `uv.lock` (a metadata-only update; no version changes). Leaving uv.lock uncommitted would create a drift between pyproject.toml and the lockfile, which the pre-push hook (`make check`) and CI catch.
- **Fix:** Staged and committed `uv.lock` together with `pyproject.toml` in the same task commit. The lockfile diff is 6 added lines across two metadata blocks (`dependencies` and `requires-dist`) — no resolution changes.
- **Files modified:** uv.lock
- **Verification:** `git diff HEAD~1 uv.lock` shows only the two pydantic metadata entries; `uv sync` is a no-op post-commit.
- **Committed in:** `43a01d0` (Task 2.1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 3 blocking — lockfile sync)
**Impact on plan:** Necessary for repository correctness; no scope creep. The plan's verification commands (`uv sync` exits 0, no version changes) remain satisfied.

## Issues Encountered

- Initial `uv run python -c ...` parse-check failed inside the sandbox with `Operation not permitted` on the uv cache (`/Users/ka/.cache/uv/sdists-v9/.git`). Resolved by re-running with sandbox disabled (per the env setup); not a real issue with the edit.

## Verification Evidence

All four greps from the plan's `<verify>` block pass:

```
$ grep -E '"pydantic>=2\.11\.7"' pyproject.toml
    "pydantic>=2.11.7",
$ grep -E '^\[tool\.ruff\.lint\.per-file-ignores\]$' pyproject.toml
[tool.ruff.lint.per-file-ignores]
$ grep -E '^"daydream/atif/\*\*" = \[' pyproject.toml
"daydream/atif/**" = ["E", "F", "I", "W"]
$ grep -E '"daydream/atif/\*\*".*"E".*"F".*"I".*"W"' pyproject.toml
"daydream/atif/**" = ["E", "F", "I", "W"]
```

TOML parses (`tomllib.loads(open('pyproject.toml').read())` → `OK`). `uv sync` exits 0 with no `Updated.*pydantic` line.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- VEND-03 complete: `daydream.atif.models` (vendored in Plan 01-01) can now be imported in downstream phases without a missing-dep error.
- Plan 01-04 (phase verification gate) can rely on `make lint` to honor the per-file-ignores stanza when the vendored ATIF source actually lands.
- Plan 01-03 (vendoring `daydream/atif/__init__.py` + LICENSE/NOTICE) is unblocked. Wave 1 completion (Plans 01-01 and 01-02 in parallel) clears Wave 2 to proceed.

## Self-Check: PASSED

- `pyproject.toml` modified (verified by `git diff` and `grep`)
- `uv.lock` modified (metadata-only update for direct-dep declaration)
- Commit `43a01d0` exists in `git log --oneline`
- All four verification greps return expected matches
- TOML parses successfully via `tomllib`
- `uv sync` is a no-op for resolution; pydantic resolved at 2.12.5

---

*Phase: 01-vendor-atif-foundation*
*Plan: 02*
*Completed: 2026-04-26*
