# Test-pruning campaign

Campaign mode audits one subsystem's entire test surface in one coherent change:
for example deep-review publication, backend transport, archive/corpus storage,
or Improve planning. The value bar, retention bar, evidence, and validation in
[SKILL.md](SKILL.md) apply throughout. Advance only when each step's completion
criterion is met. The product-defect procedure in step 7 applies as soon as a
failure is discovered; use it before cutover for baseline failures that block
the lane. Unresolved failures and unavailable required proof remain explicit
blockers.

## 1. Baseline

Pin the base SHA and record checkout state, production/test/support line counts,
collected pytest nodes, and each in-scope file's baseline results. Account for
passed, failed, skipped, and uncollected cases, including parameter rows. Record
the environment and prerequisites for live CLI, GitHub, and Docker cases.

Use an isolated checkout when the working tree has unrelated edits. Preserve a
separate failure list with reproduction commands; baseline failure is not
evidence that a test is stale. If runtime is a goal, collect wall time and
setup/call/teardown durations under fixed workers and coverage settings.

Done when every in-scope file has a baseline result or explicit prerequisite
blocker, and the baseline can be reproduced. Resolve required blockers before
cutover; unavailable live proof stays visibly unverified.

## 2. Lanes and inventory

Partition by production owner, not filename prefix. Deep-review lanes might
cover runner/workspace, review and adjudication, fix authorization, quality and
remote CI gates, findings publication, and shared recording/artifact lifecycle.
Backend lanes distinguish transport parsing, event normalization, invocation
lifecycle, and shared recording instead of treating each adapter as a duplicate.

Include subsystem cases in `tests/contract/`, CLI/integration suites,
`tests/harness/`, fixtures, packaged workflow/sandbox tests, and applicable
live/QA scenarios. Include `rl/daydream_review/` only when that separate project
is part of the authorized subsystem. Give shared files one owner and associate
each scenario within mixed files with its lane.

Done when every owned file and scenario is inventoried, every test declaration
has exactly one lane, and shared support has a designated owner.

## 3. Read-only ledger per lane

Assign bounded read-only agents when delegation is available. Each agent reads
every assigned test and parameter table, fixture dependencies, production
owners, entrypoints, callers, history, and CI routing. Record a written ledger
with the [candidate evidence](SKILL.md#discovery-and-evidence) and one mark per
test declaration. A parametrized declaration may share a mark only when all
rows have the same disposition; otherwise mark rows separately.

| Mark | Required evidence |
| --- | --- |
| `R` — retain | Contract and credible bug caught; record a file move here too. |
| `F` — fix assertion | Contract retained and the vacuous/misdirected assertion to repair. |
| `C` — consolidate | Named keeper that absorbs the unique assertion before deletion. |
| `D` — delete | Remaining proof, or evidence there is no meaningful contract. |

Judge executed paths and assertions, not names. For instance, asserting no
published comment after a failure is vacuous if the fixture never permitted
publication even on success. Verify that the positive control reaches it.

Done when every declaration/row has a mark and evidence; uncertain candidates
remain retained until their evidence is sufficient.

## 4. Layer plan per lane

Review the ledger in a second read-only pass to identify redundant layers.
Name the keeper for each contract and unique cases to move. Prefer production
runner/CLI paths with scripted external boundaries to suites that mock the
orchestrator, registry, recorder, or store whose behavior they claim to prove.
Preserve unit tests with distinct edge cases and backend-specific transport
contracts.

Explicitly map every user-visible behavior to its real-path keeper. Removing an
integration layer in favor of helper tests violates Daydream's testing standard
even if measured coverage stays constant. Identify removable production seams
separately from legitimate extension/external/fault-injection boundaries.

Done when each lane names retired files/layers, a keeper per contract, assertions
to carry over, support/seam changes, and validation commands.

## 5. Cutover

Apply plans lane by lane. Workers own disjoint files, are told they are not alone
in the repository, and must preserve others' edits. Serialize shared conftest,
harness, and support changes through their designated owner. Preserve isolated
Git/artifact/archive roots, environment cleanup, recorder ContextVars, real
SQLite behavior, and cancellation teardown.

Move unique cases first, then retire duplicates and proven-obsolete seams.
Check pytest collection and any explicit inventories/CI selectors for moved
tests, including RL's separate job. Keep branch coverage and other verification
gates intact. Record durable ownership lessons in existing scoped guidance only
when that documentation change is in scope; do not invent line-cap baselines.

Done when every lane plan is applied, its keepers and support consumers pass,
and collection accounts for every intended retained case.

## 6. Preservation review

Have independent read-only reviewers compare the full deleted coverage against
the keepers, divided by boundary group. Supply the pinned baseline, full file
inventory, ledger, layer plans, and final changes rather than a truncated diff
summary. Reviewers look for contracts losing their only proof, fixtures supplying
the promised outcome, unreachable assertions, and hidden mock assumptions.

For each restored contract, deliberately mutate its production owner in an
isolated checkout and confirm the keeper fails for the intended reason. Restore
the exact prior bytes, verify restoration, and rerun the keeper. Apply the same
check to uncertain transferred assertions; a collection/import error is not a
caught behavioral mutation.

Done when every reported gap is restored or rejected with source evidence, each
restored contract catches its mutation, and no mutation remains in the change.

## 7. Product defects

Reproduce retained baseline failures and classify environment, harness, or
product causes. Repair relevant product defects at their owner within authorized
scope, separating them from pruning (and into a separate commit if committing
is authorized). Run the real user flow with a control that removes the fix and
shows the old failure, then the candidate with the same harness and inputs.

Keep unrelated discrepancies outside the campaign's edits; report them as
remaining work or blockers. A retained failure is never a reason to delete its
test, weaken its assertion, disable a hook, or relax coverage.

Done when each required repaired defect has failing control and passing candidate
proof; unresolved required defects keep the campaign in progress.

## 8. Reconcile and hand off

When the base advances, follow Daydream's linear-history policy: rebase onto the
current base on the authorized branch, preserving user edits. Coordinate any
published-history rewrite that is outside existing authorization. If upstream
changes a retired file, inspect every new contract and port it to the keeper
before resolving deletion;
retain the file if the keeper cannot yet preserve its proof.

Rerun the subsystem and shared consumers on the reconciled head, then the
[full Daydream gates](SKILL.md#validation), including RL checks when applicable.
Repeat required external proof where the reconciliation affects it. Hook-enabled
commit/push success is mandatory for any authorized landing.

Hand off the [skill report](SKILL.md#landing-and-handoff), plus baseline/final
test/support and separate production counts, lanes/retired layers/keepers,
preservation gaps and mutation results, defect control/candidate evidence,
coverage results, and comparable runtime measurements if requested. Record
unavailable proof explicitly; completed scope and remaining scope must be clear.

Adapted for Daydream from OpenClaw's
[campaign guide](https://github.com/openclaw/openclaw/blob/main/.agents/skills/test-audit/CAMPAIGN.md).
Upstream copyright and MIT terms are retained in [LICENSE](LICENSE).
