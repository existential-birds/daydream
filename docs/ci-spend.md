# CI spend reduction

The target is at least 50% fewer paid runner minutes while retaining Blacksmith
for speed. Automatic RL checks are paused; the complete RL gate remains available
by manual dispatch. Every queued merge keeps the complete root gate; PR updates run basic checks
only after the enforced queue guarantees full verification before landing. Local Git hooks always run the complete gate.

## Workload baseline

The baseline contains all 79 completed `CI` runs created between
2026-10-07T12:00:00Z and 2026-10-14T12:00:00Z, fetched from the Actions API.
The [records](evidence/ci-spend-baseline.json) include the actual checkout commit
and tree recovered from each job's checkout log. Actions run `head_sha` identifies
the PR head; its tests execute a synthetic merge checkout with its own tree.

| Paid work | Elapsed runner minutes |
|---|---:|
| Root `check` | 685.68 |
| Standalone `rl-check` | 178.63 |
| Total | 864.32 |
| Root work on 15 pushes with an exact successful final-head PR tree | 129.05 |
| Root work on four documentation/RL updates | 33.00 |

All sampled root/RL jobs use the same four-vCPU Blacksmith runner class, so the
minute reduction also reduces their metered compute charge at the same rate.
Elapsed minutes precede provider billing rounding, allowances, and discounts.
This report models the sampled automated CI workload; an invoice and future
workload mix are separate evidence.

The new policy is documented in the
[workflow README](../.github/workflows/README.md#ci-spend-policy). All owned
CI/review jobs remain on Blacksmith. GitHub-managed CodeQL, Dependabot and
dependency-graph services are outside that runner requirement and stay unchanged.
Post-merge reuse requires a successful full merge-group `ci-proof`, an exact
Git tree match, verified source/base history and GitHub's authoritative merged-PR
association. Legacy PR proofs cannot authorize reuse. Missing or expired proof
restores the full gate.

## Worker measurement

The paired measurement is [run 38572385685](https://github.com/existential-birds/daydream/actions/runs/38572385685),
at revision `22611b20c4a03b2b18b14359ba6bd56c02226959`. Its two jobs use identical
Blacksmith four-vCPU runners, dependency locks, Python/Node/Pi versions, complete
test selection, and branch coverage settings. Only the pytest worker count
differs (four versus eight). Both jobs passed 7,111 tests and the 90% floor
(actual branch coverage: 90.66%).

| Python | Workers | Test step | Complete job |
|---|---:|---:|---:|
| 3.12 | 4 | 471 seconds | 539 seconds |
| 3.12 | 8 | 513 seconds | 577 seconds |

Eight workers were 8.9% slower in the test step, so the final workflow retains
an explicit cap of four workers, including when Blacksmith grants extra CPUs.
The temporary matrix has been removed.

## Python 3.14 measurement

The maintainer approved Python 3.14 as the package minimum, CI/tooling runtime,
and sandbox-image runtime. Both locks are regenerated. The standalone RL
project uses verifiers 0.4.0; Prime-RL training integration is explicitly paused
because its workspace still requires Python 3.12. Its manual environment gate
passed 193 tests, including all five Docker tests; only the opt-in billed live
rollout was skipped.

[Coverage.py 7.16.1](https://coverage.readthedocs.io/en/7.16.1/config.html#run-core)
uses the sys.monitoring collector by default on Python 3.14, including branch
coverage. Complete test selection, workers, coverage exclusions, and the floor
remain unchanged. [Run 38580430834](https://github.com/existential-birds/daydream/actions/runs/38580430834)
at `98499566a4ebed1a550d45a1a304bf00d2ad5fea` passed 7,111 tests and five skips
with 90.64% branch coverage on Python 3.14.7. The test step took 427 seconds;
the complete job took 519 seconds. Against the four-worker Python 3.12 job,
this is a 9.3% test-step improvement and a 3.7% complete-job improvement.
Python 3.14 alone does not establish the 50% workload target.

The [focused test consolidation](ci-test-consolidation.md) transfers duplicate
flow assertions into stronger existing keepers and batches Git fixture staging.
[Run 38589585408](https://github.com/existential-birds/daydream/actions/runs/38589585408)
at `1fc0fc8d5374acab4c26f31ca860737d6f4b1891` passed 7,098 tests and five skips,
with 90.63% branch coverage. Its test step took 411 seconds and complete job took
494 seconds, versus 427/519 before consolidation. This is a 3.7% test-step and
4.8% complete-job improvement within Python 3.14.

## Queue workload projection

The previous automatic full-PR policy projected only 40.3% savings. The maintainer
approved moving complete checks to the merge queue. A basic PR job checks locks,
installs dependencies, runs Ruff, naming and actionlint; typechecking, dead-code
analysis, complete tests and branch coverage run before the queue may land a merge.

The sample has 64 PR updates and 15 actual merges/main pushes across 22 PRs.
Reserve one successful full queue job per actual merge, rather than per open or
abandoned PR. Additionally retain **all 23 failed historical root jobs at their
original 201.78-minute cost**, even though 20 involved defects fixed upstream.
The reserve includes failed docs-only check [37690478327](https://github.com/existential-birds/daydream/actions/runs/37690478327/job/113029055305)
(451 seconds), excluded from the earlier full-only failure classification.
No failure repair, cancellation, queue batching or cache savings are credited.

| Projected paid work | Budget | Minutes |
|---|---:|---:|
| 64 basic PR updates | 70 seconds each | 74.67 |
| 15 successful full queue checks | 519 seconds each | 129.75 |
| 15 post-merge push checks with verified reuse | 60 seconds each | 15.00 |
| All 23 historical failed root jobs | Original elapsed time | 201.78 |
| Total | | **421.20** |

```text
64 × 70/60 + 15 × 519/60 + 15 × 60/60 + 201.7833 = 421.20 minutes
1 − (421.20 / 864.3167) = 51.27% reduction
```

The full-job budget uses the slower measured Python 3.14 result (519 seconds),
rather than the consolidated 494-second result. The 70-second basic-job budget
is a component estimate, not a measured standalone basic job: subtracting tests
(411), vulture/mypy (22), Node/Pi setup (9), coverage upload (2) and proof upload
(1) from the observed 494-second job leaves 49 seconds, including checkout,
runtime setup, the planner, basic checks and runner teardown. The extra 21 seconds
reserves variation and the new effective-rules API read. The actual enforced-queue
basic and post-merge paths must be measured after activation; slower jobs, extra
queue rebuilds, direct pushes or additional manual RL runs can reduce savings.
The [model](evidence/ci-queue-spend.json) records these assumptions and source IDs.
This is a workload projection with reserves, not measured achieved savings or
an invoice guarantee. Its 50% headroom is 657.5 seconds: with other terms fixed,
the mean basic job must remain below 80.27 seconds; one extra 519-second queue
attempt fits, while two exceed that headroom.

## Merge enforcement and rollout

This public organization repository can use a merge queue on its existing Free
plan; [GitHub documents availability for public organization repositories](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue).
The server accepted ruleset `25094989`, named `CI merge queue`, with no bypass
actors, required GitHub Actions `check` (integration `15368`), `ALLGREEN`, squash
merges, one entry per build/merge, and a 60-minute check timeout. It is currently
**disabled**, because enabling it before the default-branch workflow handles
`merge_group` would block every merge.

Existing active main ruleset `12346955` still requires `check`; its administrator
bypass was removed and readback reports `current_user_can_bypass: never`.
A failure therefore blocks a merge now. Until queue activation, every PR runs
the full gate. Queue activation requires these ordered checks:

1. Merge PR #1546 through the existing successful full required check.
2. Verify the final reviewed `.github/workflows/ci.yml` and `scripts/ci_plan.py`
   blobs are present on current `main`, and its required `check` succeeded.
3. Read ruleset `25094989` and confirm it still matches the
   [staged settings](evidence/ci-merge-queue-staged.json). Preserve concurrent
   settings changes; re-review any mismatch.
4. Change only that ruleset's enforcement from `disabled` to `active`; read it
   back and verify effective main rules include `merge_queue` and required
   Actions `check`, with no bypass actors. Keep existing ruleset `12346955`.
5. Observe the first queued merge's full check and proof, its post-merge reuse,
   and the next PR basic job; compare their elapsed time with the budgets above.

The workflow has no settings-write permission and does not enqueue or merge PRs.
Disabling the queue restores full PR checks automatically. This PR does not
activate the staged ruleset or merge itself.

## Alternatives considered

| Possibility | Decision |
|---|---|
| Free GitHub-hosted runners for this public repository | Retain Blacksmith as requested for speed; [standard public runners are free](https://docs.github.com/en/billing/concepts/product-billing/github-actions). |
| Pause RL | Approved. Saves 20.7% of sampled elapsed runner time; retains manual execution. |
| Skip every main push | Reuse only exact successful full merge-group trees. Direct pushes, mismatches, and missing evidence run the full gate. |
| Documentation/RL filtering | Superseded by enforced queue scheduling; all queued trees run the full gate regardless of paths. |
| Cancel superseded PR work | Enabled. Keep push runs independent; exclude cancellation savings from the model to avoid overlap or optimistic credit. |
| More workers on the same runner | Paired complete suites showed eight workers were slower. Keep four. |
| Smaller paid runners, ARM, larger runners, or self-hosting | Introduces throughput, architecture, provisioning, or rate tradeoffs. The current machine class stays unchanged. |
| Python 3.14 and its monitoring coverage collector | Approved for every owned project/runtime; verify the complete gate and actual Blacksmith speed before claiming savings. |
| Blacksmith OSS/startup programs | [Provider programs](https://www.blacksmith.sh/pricing) may offer discounts. Eligibility and a negotiated rate are unverified; no savings credited or application submitted. |
| More installation caches | uv caching is already enabled; tests dominate the root job. The latest baseline run spent 491 seconds in tests versus zero seconds installing Python dependencies. |
| Shorter artifact retention or artifact deletion | Existing artifacts total about 130 MB. Detailed October billing showed no net Actions storage charge for this repository; preserve diagnostic coverage artifacts. New tiny proof artifacts expire after 14 days. |
| Remove CodeQL or Dependabot | Preserve security checks and updates; CodeQL uses standard runners. |
| Model review reductions | Already maintainer-dispatched, with no Daydream Review in the last 200 runs at sampling. The clarified target is paid runner minutes. |
| Test subsets or a lower coverage floor | Retain complete root suites and the existing ratchet. Scheduling and verified reuse reduce duplicate work. |
| Draft-only test deferral | Reconstructed draft/ready timelines for all 22 PRs. The only draft full run supplied its PR's sole final proof; a mandatory ready check replaces it. No defensible historical savings. |
| Cached Git seed repositories | Linux microbenchmark projects about eight four-worker seconds; retain ordinary setup because environment/template/metadata cache guards add complexity. |
| Source-guard parsing shortcuts | Small isolated gains, with Unicode/escaped-literal risks. Retain the guards and their current implementation. |
| Mypy cache | Warm isolated typing still rejects a deliberate type error; cache transfer and cross-PR scope limit unmeasured net savings. No savings credited. |

## Reproduction

Fetch runs and each run's jobs with `gh api`, then sum job `completed_at` minus
`started_at`. The benchmark's `Run tests` step timestamps provide its paired
test-time ratio. Individual baseline runs are available at
`https://github.com/existential-birds/daydream/actions/runs/RUN_ID` using the IDs
in the recorded JSON. GitHub's commit/tree APIs verify the captured tree hashes;
the queue model counts 64 PR updates, 15 actual merged events and all 23 root failures.

The queue calculation reserves both the paid PR basic job and post-merge push
job for each merge, plus every historical failure. Cancellation, caches,
provider free tiers and model API changes receive no savings credit.
