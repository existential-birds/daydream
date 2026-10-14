# Verification Workflow

Verify the selected target tree and real runtime path. Preserve the claim's conditions and source snapshot; the comment may target an earlier commit or stale location. Re-anchor using symbols and surrounding code instead of trusting old line numbers.

## Bind the verification target

For PR feedback, confirm the selected PR number/URL, base repository/ref/SHA,
and head repository/ref/SHA, including fork identity, against current GitHub
metadata. Keep the reviewer's original commit separate from the acquired PR head.
If that head moved since acquisition, refresh metadata and feedback before
verification. Failed acquisition is a blocker, not permission to infer a target.

Use the current checkout as the remote PR verification tree only when its
repository and branch map to that PR's head, HEAD matches the acquired remote
head, and its tracked tree has no local changes. Otherwise inspect that exact
remote head in an isolated snapshot/worktree, including for clean behind or
diverged branches; preserve the user's branch and dirty work. Authorized fixes
belong in a worktree on the selected PR head branch, never an unrelated branch.
Read repository and nested guidance from the tree actually being inspected.

A matching branch may legitimately be ahead of the remote or contain authorized
uncommitted fixes. Verify the remote PR tree and local candidate separately;
record candidate ancestry, its owned committed/uncommitted patch (including
relevant untracked files), and exact checked snapshot. Preserve unrelated edits.
Unclear ancestry or patch ownership blocks fix dispatch/publication. A verified
local fix does not establish that the remote PR is fixed.

Pass the selected base/head repository and branch identities, remote head SHA,
exact target worktree path, and any distinct candidate snapshot to workers,
validation, publication, and response handoffs. For non-PR pasted/file feedback,
use the supplied snapshot or explicitly identify the current checkout as the
verification target; do not invent a PR association.

## Per item

1. Locate the target with `rg`, then read implementation and callers. Consult current repo guidance and relevant architecture/domain docs.
2. Trace reachability, configuration, mode gates, lifecycle, and concurrency. A missing local check is not proof of a bug if an earlier invariant guarantees the input; verify that invariant and callers.
3. Exercise the stated conditions with a meaningful command or test. For behavioral changes, use the production path required by `AGENTS.md`. Redact secrets from outputs and artifacts.
4. Record target worktree path, checked commit or candidate snapshot, current path/line, exact command/result, reproduction conditions, or short quoted output. Distinguish observation from inference.
5. Classify under [EVALUATION.md](EVALUATION.md). Unavailable reproduction or evidence stays uncertain or blocked, never valid by default.

## Claim-specific checks

| Claim | Verification |
|-------|--------------|
| Unused code | Search imports, production/test consumers, dynamic registration, extension exports, and packaged templates; inspect vulture config and use `make deadcode` when relevant |
| Bug/error | Reproduce exact conditions through CLI or `runner.run`; use available `diagnosing-bugs` guidance for difficult failures |
| Import/type error | Read imports and run relevant Ruff/mypy checks through repo commands |
| Style/architecture | Compare current `AGENTS.md`, ADRs/docs, and neighboring implementation; heuristics alone do not establish a defect |
| Performance | Measure a representative baseline and comparison; speculative optimization is not a verified regression |
| Security/privacy | Trace input/credentials through subprocess, events, sanitization, artifact, and publication boundaries; establish actual exposure conditions |
| Tests | Check observable assertions and mandatory real-path coverage; internal call counts alone do not prove user behavior |
| Identity/publication | Trace durable identifiers, snapshot binding, immutable artifact ownership, and atomic publication; preserve source IDs |

For specialized guidance, consult available `python-code-review`,
`pytest-code-review`, or `review-verification-protocol` skills as appropriate.
They supplement the checks above; their absence does not block verification.
Other optional routing and architecture constraints are in
[references/skill-integration.md](references/skill-integration.md).

## Commands

Read the current `Makefile`; it owns the commands. Root development uses Python and `uv`:

```bash
uv lock --check
uv run pytest tests/<relevant_test>.py
make lint
make typecheck
make check
```

`make check` checks the lock before installation or `uv run` can relock, then runs root install, Ruff, vulture, mypy, coverage-bearing pytest, workflow linting, coverage-report verification, and naming checks. Targeted pytest is plain and avoids the full-suite coverage floor. Run `make rl-check` for changes to the separately locked `rl/daydream_review` project; it is not part of `make check`. Dependency edits require consistent lockfiles. Never disable hooks, tests, or configured gates to force success.

Use relevant evidence for triage and required combined-tree gates for implementation/publication. These commands are not a demand to run the whole suite before classifying every item. Record unavailable checks honestly.
