---
name: daydream-deadcode-sweep
description: "Use when simplifying Daydream code: PR mode for a contribution branch, or a full-repository sweep on request."
---

# Code simplification sweep

Remove verified code while preserving supported behavior and making the remaining
code easier to read. Dead-code removal is one technique; a smaller implementation
of used behavior counts equally.

Any contributor can run this with a coding agent that has read-only subagents, plus
`git`, `uv` and `make`. If your agent has no subagents, run the scouts in sequence.

## Choose the mode

| | PR mode | Full-repository mode |
|---|---|---|
| When | A contribution branch that needs it (see [the PR rule](../../../CONTRIBUTING.md#pull-request-requirements)) | Only when someone asks for a repository-wide sweep |
| Scope | Files in the base-to-head diff | Every tracked file |
| Other files | Read or change them only as needed to finish an investigation or refactor | All in scope |
| Scouts | One per cohesive behavior the diff touches | Ten disjoint slices, with more waves as needed |
| Size rule | Sweep changes need not shrink the whole PR | Insertions must be fewer than deletions |

Use PR mode unless a repository-wide sweep was requested. In PR mode, stay out of
unrelated code: a finding outside the diff is reported only, unless the refactor
needs it.

## What to look for

Reference counts find unused symbols, which are rare here. Most removable code is
live: needless layers, defensive branches for impossible states, parallel
implementations, single-consumer representations, and tests that pin implementation.
Finding it takes reading whole behaviors, so a symbol scan is never the main
technique. Tests also count as references; check production callers separately.

Maintainers decide whether a shipped tool, script, workflow or feature stays. Report
those candidates as decisions and do not remove them.

## Boundaries

- Preserve CLI behavior, supported extension APIs, persisted formats, observable
  results, side effects, authorization boundaries, cancellation and meaningful
  failure handling.
- Prefer fewer concepts and owners. A new helper or dependency must retire more
  complexity than it adds. Count replacement code, imports, callers and tests.
- No line-count gaming through minification, statement packing, file moves or
  deleting useful rationale.
- Never bypass commit or push hooks, signatures, required checks or the coverage floor.

## Procedure

1. Read [AGENTS.md](../../../AGENTS.md) and inspect the branch, diff and status.
   Record the starting commit and any existing changes. Do not reset, stash, stage
   or commit unrelated work.
2. Keep progress outside tracked files, under
   `git rev-parse --path-format=absolute --git-path daydream-deadcode-sweep` in a
   `runs/<run_id>/` directory.
3. Define the scope.
   - **PR mode:** list `git diff --name-only <base>...HEAD` and group the files by
     behavior.
   - **Full-repository mode:** run
     `python3 scripts/sweep/partition.py <repo-root> 10 <run-dir>/inventory.json`
     from this skill directory to partition every tracked file into ten slices.
4. Give each scout the [scan contract](scripts/sweep/prompts/scan.md), its owned paths
   and its findings, progress and coverage paths. Use your own model for scouts unless
   the user chose otherwise. Do not add a time box or tell a scout to favor breadth; if
   a scope is too large to read, make it smaller or add a wave. Scouts are read-only
   and may read any caller or contract outside their paths.
5. Finish discovery before changing anything. Reconcile coverage against the scope.
   An outline, grep or automated scan of a code or test file is not inspection: treat
   that file as `unvisited`, and reassign unvisited, failed or timed-out work. Data
   files, binaries and vendored code need only a recorded reason.
6. Merge and rank findings by net reduction, confidence and verification effort.
   [findings.py](scripts/sweep/findings.py) can deduplicate and batch JSONL findings.
   Record an outcome and reason for each. Treat findings as unverified whatever their
   confidence: recheck callers yourself with an attribute-aware search, including
   `module.name` access and string monkeypatch targets, and run the type checker after
   each change. Follow the [apply contract](scripts/sweep/prompts/apply.md), and finish
   one connected simplification at a time.
7. Run affected tests, including unchanged consumers, then `make check`. Run
   `make rl-check` if `rl/daydream_review` changed. Record real exit codes and skips.
8. Inspect the complete diff. Commit or open a PR only if asked.

## Report

Report actual additions, deletions and net by production, tests and docs; the
simplifications applied; verification and its limits; and each deferred finding.
Count paths by what was actually read, and say how many were read in full. Report
estimates as estimates and findings as leads, not as a to-do list. Name maintainer
decisions separately. In full-repository mode, name the subsystems that need a focused
pass. Distinguish an incomplete scan from a completed sweep. In a PR, put this in
**Test Plan**.

## Maintaining this skill

Keep the prompts in step with this file. After changing the toolkit, run this from
the skill directory:

    python3 -m unittest discover -s scripts/sweep/tests -p 'test_*.py'
