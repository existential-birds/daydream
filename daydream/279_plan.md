# Issue #279 — Authoritative intent propagation

You are the **orchestrator**. Dispatch one sub-agent per task. Do not implement code yourself.

## Context

Repo: `/Users/ka/github/existential-birds/daydream` (Python 3.12, uv, pytest, mypy, ruff).

Work: fix GitHub issue #279 — the deep review flow assembles an "author intent is AUTHORITATIVE" precedence rule for the intent phase, then drops it at the intent → review boundary, so the per-stack, arbiter, and merge reviewers never receive it. A change the author documents as deliberate can still be reported as a high-severity regression.

**The plan is authoritative and lives on disk:**

`/Users/ka/github/existential-birds/daydream/.beagle/plans/deep-authoritative-intent-propagation/plan.md`

Read it in full before dispatching anything. Do not restate it to sub-agents — give each one the absolute path and its task number. The plan contains the Intent block, Assumptions, File Structure, four numbered tasks with exact file anchors, test bodies, behavior contracts, skeletons, enumerated sweeps, and commit messages.

Background reading: `gh issue view 279`.

## Executor contract

The plan's `## Executor Contract` section (near the top of `plan.md`) is the charter for every sub-agent — two attempts per step, shape-not-string failure matching, the baseline-red policy, the stop-and-report template, and the proceed-and-flag rule for ambiguity. It is inlined verbatim in the plan file. Every sub-agent brief must instruct the agent to read that section before starting its task. It is also available as the beagle skill `beagle-core:execution-contract` if this session loads beagle skills.

## Pre-flight (orchestrator does these itself, before Task 1)

1. **Verify commit identity.** Run
   `git -C /Users/ka/github/existential-birds/daydream config user.email`.
   It must be `anthropic@anderskev.com`. This repo's `.git/config` has been polluted by a test fixture identity (`test@example.com`) before; a pre-commit hook blocks it. Fix before any commit.

2. **Create the branch.** All four tasks commit to it; nothing lands on `main`.

   ```bash
   cd /Users/ka/github/existential-birds/daydream
   git checkout main && git pull
   git checkout -b fix/279-authoritative-intent-propagation
   ```

   `main` requires a PR. **Never commit or push to `main`, and never use `--no-verify`.**

3. **Capture the baseline.** Run once and record which tests are already red:

   ```bash
   cd /Users/ka/github/existential-birds/daydream && uv run pytest -n auto
   ```

   Pass the recorded red list to every sub-agent. A test red in the baseline is recorded and skipped — never fixed, never retried, never counted as a regression. Only a test that was green in the baseline and is red now blocks a step.

## Tasks

Dispatch **sequentially** — Task 2 imports the constant Task 1 creates, Task 3 depends on the kwarg Task 2 adds, and Task 4 documents Task 2's and Task 3's surface. Sub-agent type: **general-purpose**.

Each brief must say: read `plan.md`'s `## Executor Contract` and `## Assumptions`, then execute **only** Task N, all six steps in order, including the Step 5 enumerated sweep and the Step 6 commit with the plan's exact commit message. Stage files explicitly with `git add <path>` — never `git add -A`.

| # | Task | Input | Verification command |
|---|------|-------|----------------------|
| 1 | Extract the precedence rule into one shared constant | `plan.md` § Task 1 | `cd /Users/ka/github/existential-birds/daydream && uv run pytest tests/test_phases.py -q` |
| 2 | Gate the rule into the deep prompt builders | `plan.md` § Task 2 | `cd /Users/ka/github/existential-birds/daydream && uv run pytest tests/test_deep_prompts.py -q` |
| 3 | Thread the flag from the intent step to the reviewers | `plan.md` § Task 3 | `cd /Users/ka/github/existential-birds/daydream && uv run pytest tests/test_deep_orchestrator.py -q` |
| 4 | Document the widened prompt contract | `plan.md` § Task 4 | `cd /Users/ka/github/existential-birds/daydream && uv run pytest tests/test_extension_contract_doc.py -q` |

Task 4's Step 4 in the plan calls for `make check`; that is your final integration check below, not the sub-agent's per-task command. The sub-agent runs the guard test above.

Two notes to pass through verbatim:

- **Task 3, Step 2 expects a split result.** `test_pr_body_reaches_intent_prompt` must FAIL (that is the bug). `test_no_pr_body_degrades_cleanly` must already PASS — it is the regression guard proving the gate stays shut. An agent that "fixes" the passing test has misread the step.
- **Scope is closed.** The `supervise`, `suppression`, and `verify` prompts are out of scope per issue #279 and the plan's Intent block. Do not widen into them.

## Reporting contract

Each sub-agent reports back:

1. What it changed (files and the substance of the change).
2. The exact command it ran.
3. The last relevant lines of that command's output.
4. Whether the check passed, plus any `AMBIGUITY: …` lines.

A `BLOCKED` report at the two-attempt budget is a complete, successful hand-back — not a failure to retry into.

## Orchestrator loop

Dispatch tasks in order until each has reported either a passing check or a `BLOCKED` report. A `BLOCKED` task gets **at most one** follow-up sub-agent, briefed with the blocked report verbatim. If that follow-up also returns `BLOCKED`, record the task as blocked, continue with tasks that do not depend on it, and surface it in the final summary. **Never dispatch a third attempt at the same step.** If Task 1 or Task 2 blocks, the downstream tasks depend on it — stop the loop and report rather than dispatching work that cannot compile.

## Final integration check

Once, after the task loop:

```bash
cd /Users/ka/github/existential-birds/daydream && make check
```

That is `uv lock --check` + `ruff check daydream tests bench` + `mypy daydream tests bench` + the full pytest suite — the same gate the pre-push hook runs. Judge it against the recorded baseline: newly-red blocks, baseline red does not.

## Ship

Only if `make check` is green apart from baseline red:

```bash
cd /Users/ka/github/existential-birds/daydream
git push -u origin fix/279-authoritative-intent-propagation
gh pr create \
  --title "fix(deep): propagate authoritative PR-intent framing to the review phases (#279)" \
  --body "<summary + 'Closes #279' + what was verified>"
```

Do not merge the PR. Report its URL.

## Partial-result exit

If tasks remain blocked after the final check, report: what landed and is verified, each `BLOCKED` block verbatim, and what the user must decide. Do not revert completed tasks because a later one stalled. Do not push a branch whose gate is red — report instead. Delivering a verified partial result with a clear blocker list is a **successful** outcome.
