# Verification Workflow

Verify the current code and real runtime path. Preserve the claim's conditions and source snapshot; the comment may target an earlier commit or stale location. Re-anchor using symbols and surrounding code instead of trusting old line numbers.

## Per item

1. Locate the target with `rg`, then read implementation and callers. Consult current repo guidance and relevant architecture/domain docs.
2. Trace reachability, configuration, mode gates, lifecycle, and concurrency. A missing local check is not proof of a bug if an earlier invariant guarantees the input; verify that invariant and callers.
3. Exercise the stated conditions with a meaningful command or test. For behavioral changes, use the production path required by `AGENTS.md`. Redact secrets from outputs and artifacts.
4. Record current path/line, snapshot when relevant, exact command/result, reproduction conditions, or short quoted output. Distinguish observation from inference.
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
