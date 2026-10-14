# CI spend reduction

The target is at least 50% fewer paid runner minutes while retaining Blacksmith
for speed. Automatic RL checks are paused; the complete RL gate remains available
by manual dispatch. Code, tests, dependencies, and workflows keep the complete
root gate. Local Git hooks always run the complete gate.

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
[workflow README](../.github/workflows/README.md#ci-spend-policy). Reuse requires
a successful run's `ci-proof`, matching repository/PR/final head and the actual
Git-verified merge parent/tree. Historical runs lacked these new artifacts; the
model assumes that successful PR gates publish them after rollout. Missing or
expired proof restores the full gate.

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
four workers through pytest's automatic selection on the four-vCPU runner.
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
remain unchanged. The Blacksmith runtime measurement and final workload savings
calculation are still in progress.

## Alternatives considered

| Possibility | Decision |
|---|---|
| Free GitHub-hosted runners for this public repository | Retain Blacksmith as requested for speed; [standard public runners are free](https://docs.github.com/en/billing/concepts/product-billing/github-actions). |
| Pause RL | Approved. Saves 20.7% of sampled elapsed runner time; retains manual execution. |
| Skip every main push | Reuse only exact successful PR trees. Direct pushes, mismatches, and missing evidence run the full gate. |
| Documentation/RL filtering | Approved, with a narrow allowlist and Git diffs that include deletions/renames. Product prompts/templates and executable docs still run in full. |
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

## Reproduction

Fetch runs and each run's jobs with `gh api`, then sum job `completed_at` minus
`started_at`. The benchmark's `Run tests` step timestamps provide its paired
test-time ratio. Individual baseline runs are available at
`https://github.com/existential-birds/daydream/actions/runs/RUN_ID` using the IDs
in the recorded JSON. GitHub's commit/tree APIs verify the captured tree hashes;
the baseline routes 15 exact-tree pushes and four lightweight PR updates.

The final calculation will reserve one minute for each reused/lightweight job
and account for the full gate's non-test work. It will not claim savings for
cancelled runs, caches, provider free tiers, or changes to model API usage.
