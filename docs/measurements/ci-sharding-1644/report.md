# Issue 1644: controlled full-CI sharding results

Four pytest jobs reduced median full-gate time from **2,260 seconds to 676 seconds (70.1%)**, exceeding the 25% target. All nine same-commit samples passed exact collection, execution, skip and coverage verification.

Frozen commit [`e395675b66aa0b7e9cf2b882495b352d6cf0e6ce`](https://github.com/existential-birds/daydream/commit/e395675b66aa0b7e9cf2b882495b352d6cf0e6ce); accompanying raw evidence is inventoried in `compact-evidence.json`. [Issue 1644](https://github.com/existential-birds/daydream/issues/1644), [PR 1650](https://github.com/existential-birds/daydream/pull/1650).

## Method and epoch separation

Sequence: four, one, two pytest jobs repeated three times, with each run finishing before the next dispatch. Baseline runs the original sequential gate with eight workers; two/four-job treatments use four workers per pytest job. All use the same committed dependencies, root selection and aggregate branch-coverage floor on free standard ubuntu-24.04 runners. Planner/final-gate and setup/artifact/coverage overhead count toward end-to-end gate time. The baseline adds root-only JUnit output; nested pytest commands do not inherit reporting flags. RL remains separate.

Every sample executed the same 7,860 root test identities with five skips. Treatment selections form an exact disjoint union, and baseline JUnit testcase and skipped identities match that union. All runs use attempt 1; no sample was retried or replaced.

Prior epochs are retained but excluded in full from all new acceptance statistics:

| Source | Classification | Preserved evidence |
|---|---|---|
| `97a9e0fb` | initial validation only; excluded | [excluded source/run evidence](compact-evidence.json) |
| `b5642969` | incomplete failed study; all three runs excluded from this study | [excluded source/run evidence](compact-evidence.json) |

The b564 study retained a [passing four-job run](https://github.com/existential-birds/daydream/actions/runs/39314567927), [passing baseline](https://github.com/existential-birds/daydream/actions/runs/39316349988), and [failed two-job run](https://github.com/existential-birds/daydream/actions/runs/39321780634). The failure exposed shared mutable Registry aliasing in existing test overrides: resetting a ContextVar did not restore the mutated object. The supervision and merge test cases now own fresh registries. One proven identical default-host row was consolidated; independent policies and useful assertions remain. The nine measurements in this report therefore use the repaired source exclusively.

The failed sample retained all 7,861 selected/executed identities, the two failing provider assertions, diagnostic coverage at 90.7478265969%, failed aggregation and failed final check. Gate failure propagation is established; its 777-second gate and 24.85 runner minutes are diagnostic failure evidence, never passing acceptance samples.

## Three-run medians

| Jobs × workers | New verified samples | Median gate seconds | Median runner minutes | Reduction from baseline |
|---|---:|---:|---:|---:|
| 1 × 8 | 3/3 | 2260.0 | 37.5 | 0.0% |
| 2 × 4 | 3/3 | 1282.0 | 31.5 | 43.3% |
| 4 × 4 | 3/3 | 676.0 | 35.2 | 70.1% |

Four jobs reduce median gate time by 70.1% from baseline; the 25% target is met.
Four jobs are 47.3% faster than two by median full-gate time. They use 35.2167 median runner minutes versus 31.4833 for two (11.9% more), and 37.5 for the baseline (6.1% less). Four jobs retain the best measured gate latency; two offer the lowest median runner time. These are sums of non-skipped job durations, not rounded billable minutes.

## Every measured sample

| Round | Jobs | Run/state | Gate seconds | Runner minutes | Initial queue seconds | Peak jobs | Imbalance |
|---|---:|---|---:|---:|---:|---:|---:|
| 1 | 4 | [39327880457](https://github.com/existential-birds/daydream/actions/runs/39327880457) / verified | 737.0 | 40.3 | 4.0 | 5 | 1.180 |
| 1 | 1 | [39329539133](https://github.com/existential-birds/daydream/actions/runs/39329539133) / verified | 1856.0 | 30.8 | 4.0 | 1 | 1.000 |
| 1 | 2 | [39333414151](https://github.com/existential-birds/daydream/actions/runs/39333414151) / verified | 966.0 | 29.2 | 5.0 | 3 | 1.096 |
| 2 | 4 | [39335368946](https://github.com/existential-birds/daydream/actions/runs/39335368946) / verified | 672.0 | 33.9 | 4.0 | 5 | 1.280 |
| 2 | 1 | [39336736635](https://github.com/existential-birds/daydream/actions/runs/39336736635) / verified | 2495.0 | 41.4 | 4.0 | 1 | 1.000 |
| 2 | 2 | [39341306737](https://github.com/existential-birds/daydream/actions/runs/39341306737) / verified | 1311.0 | 41.1 | 4.0 | 3 | 1.061 |
| 3 | 4 | [39343615694](https://github.com/existential-birds/daydream/actions/runs/39343615694) / verified | 676.0 | 35.2 | 3.0 | 5 | 1.255 |
| 3 | 1 | [39344792432](https://github.com/existential-birds/daydream/actions/runs/39344792432) / verified | 2260.0 | 37.5 | 4.0 | 1 | 1.000 |
| 3 | 2 | [39348622800](https://github.com/existential-birds/daydream/actions/runs/39348622800) / verified | 1282.0 | 31.5 | 4.0 | 3 | 1.376 |

### Per-job evidence

| Run/job | Pytest seconds | Pre-test seconds | Plan-to-job wait seconds | Whole job seconds |
|---|---:|---:|---:|---:|
| 39327880457 / test-shard (0) | 570.0 | 23.0 | 3.0 | 597.0 |
| 39327880457 / test-shard (1) | 397.0 | 30.0 | 3.0 | 432.0 |
| 39327880457 / test-shard (2) | 646.0 | 26.0 | 4.0 | 678.0 |
| 39327880457 / test-shard (3) | 576.0 | 28.0 | 3.0 | 609.0 |
| 39329539133 / legacy-check | 1768.0 | 59.0 | 3.0 | 1831.0 |
| 39333414151 / test-shard (0) | 724.0 | 27.0 | 3.0 | 757.0 |
| 39333414151 / test-shard (1) | 877.0 | 28.0 | 3.0 | 911.0 |
| 39335368946 / test-shard (0) | 277.0 | 20.0 | 3.0 | 302.0 |
| 39335368946 / test-shard (1) | 428.0 | 27.0 | 4.0 | 460.0 |
| 39335368946 / test-shard (2) | 538.0 | 30.0 | 4.0 | 574.0 |
| 39335368946 / test-shard (3) | 585.0 | 28.0 | 4.0 | 620.0 |
| 39336736635 / legacy-check | 2401.0 | 67.0 | 3.0 | 2474.0 |
| 39341306737 / test-shard (0) | 1227.0 | 26.0 | 4.0 | 1259.0 |
| 39341306737 / test-shard (1) | 1085.0 | 23.0 | 3.0 | 1112.0 |
| 39343615694 / test-shard (0) | 347.0 | 29.0 | 3.0 | 381.0 |
| 39343615694 / test-shard (1) | 514.0 | 26.0 | 3.0 | 545.0 |
| 39343615694 / test-shard (2) | 438.0 | 23.0 | 3.0 | 465.0 |
| 39343615694 / test-shard (3) | 594.0 | 27.0 | 3.0 | 627.0 |
| 39344792432 / legacy-check | 2167.0 | 66.0 | 3.0 | 2238.0 |
| 39348622800 / test-shard (0) | 545.0 | 23.0 | 2.0 | 574.0 |
| 39348622800 / test-shard (1) | 1202.0 | 22.0 | 3.0 | 1227.0 |

## Selection, skips and coverage

All-nine independent verification: **PASSED**. Receipts bind repository/head/tree/run/attempt; disjoint selections must have an exact complete union. Baseline JUnit maps exact test and skipped identities, including complex parametrized IDs. Checksums and XML line+branch counts independently enforce the unchanged configured coverage floor. Incomplete/failed samples never count as successes.

| Run | Exact nodes | Skips | Combined coverage % | Floor % |
|---|---:|---:|---:|---:|
| 39327880457 | 7860 | 5 | 90.7492926 | 90.0 |
| 39329539133 | 7860 | 5 | 90.7507587 | 90.0 |
| 39333414151 | 7860 | 5 | 90.7478266 | 90.0 |
| 39335368946 | 7860 | 5 | 90.7507587 | 90.0 |
| 39336736635 | 7860 | 5 | 90.7478266 | 90.0 |
| 39341306737 | 7860 | 5 | 90.7507587 | 90.0 |
| 39343615694 | 7860 | 5 | 90.7507587 | 90.0 |
| 39344792432 | 7860 | 5 | 90.7507587 | 90.0 |
| 39348622800 | 7860 | 5 | 90.7478266 | 90.0 |

Coverage is not assumed byte-identical. In the excluded b564 epoch, baseline/four differed by exactly one line and branch at calibrate.py:56–57 (0.002932 percentage points), with identical denominators and test/skip selection. Fresh-process real-loader reproduction established a module/cache-order mechanism: the shared sr_module fixture leaves score_review registered, and the production loader restores an existing module object. Actual CI test order/key attribution was unproven because per-test coverage contexts and baseline raw data were absent. This historical diagnostic identifies shared process-local fixture state as a maintenance smell; it is not proof of failing production behavior and does not authorize deleting valuable tests or excluding coverage. New-epoch percentages are reported independently above.

## Runner, cache, queue and other workflows

Authenticated [organization standard-runner settings](https://github.com/organizations/existential-birds/settings/actions/hosted-runners) displayed an effective shared concurrency limit of 20 at 2026-10-20T00:41:28.355690+00:00, with All jobs usage 2/20 and the final sample's two Linux shard jobs active. The sanitized capture and its digest accompany this report. This current UI observation is separate from study peak overlap and does not establish historical settings. [GitHub limits](https://docs.github.com/en/actions/reference/limits#job-concurrency-limits-for-github-hosted-runners). Observed overlap is a lower bound, not the configured cap. Raw setup logs establish hosted image/runtime class; the frozen workflow does not emit processor inventory, so no hardware model/count is invented.

| Run | Image version | Python | Cache hits / misses across jobs |
|---|---|---|---|
| 39327880457 | 20261012.334.1, 20261018.339.1 | 3.14.8 | 6 / 0 |
| 39329539133 | 20261012.334.1 | 3.14.8 | 1 / 0 |
| 39333414151 | 20261012.334.1, 20261018.339.1 | 3.14.8 | 4 / 0 |
| 39335368946 | 20261012.334.1, 20261018.339.1 | 3.14.8 | 6 / 0 |
| 39336736635 | 20261012.334.1, 20261018.339.1 | 3.14.8 | 1 / 0 |
| 39341306737 | 20261012.334.1, 20261018.339.1 | 3.14.8 | 4 / 0 |
| 39343615694 | 20261012.334.1, 20261018.339.1 | 3.14.8 | 6 / 0 |
| 39344792432 | 20261012.334.1 | 3.14.8 | 1 / 0 |
| 39348622800 | 20261012.334.1, 20261018.339.1 | 3.14.8 | 4 / 0 |

Repository activity query captured 177 runs across every workflow/event in its explicit 24-hour lookback plus this epoch's window; peak observed overlap is 5. Window complete.
Other organization activity and older queued runs beyond that stated bound are not inferred absent. Co-load is disclosed; no sample is silently retried or discarded.

## Controlled residual phase evidence

The 6 verified treatment runs retain setup/call/teardown durations for every node. Phase totals sum worker-report seconds across each configuration's samples; they describe executed work, not gate wall time. Baseline JUnit does not provide equivalent phase breakdowns.

| Jobs | Setup seconds | Call seconds | Teardown seconds | Call share |
|---|---:|---:|---:|---:|
| 2 | 372.7 | 19269.9 | 14.9 | 98.03% |
| 4 | 385.4 | 20562.0 | 14.9 | 98.09% |

Slowest nodes by median total duration across the controlled treatments:

| Node | Median setup seconds | Median call seconds | Median teardown seconds |
|---|---:|---:|---:|
| [`tests/test_benchmark_harbor_package.py::test_audit_execution_proofs_harbor_gated`](https://github.com/existential-birds/daydream/blob/e395675b66aa0b7e9cf2b882495b352d6cf0e6ce/tests/test_benchmark_harbor_package.py) | 0.0041 | 105.83 | 0.0006 |
| [`tests/deep_orchestrator/test_review_reuse_path.py::test_editing_a_recorded_frontier_file_invalidates_snapshot_bound_coverage`](https://github.com/existential-birds/daydream/blob/e395675b66aa0b7e9cf2b882495b352d6cf0e6ce/tests/deep_orchestrator/test_review_reuse_path.py) | 0.0538 | 38.34 | 0.0012 |
| [`tests/deep_orchestrator/test_review_capture_and_retry.py::test_structure_keeps_bounded_interaction_review_and_patch_navigation[files]`](https://github.com/existential-birds/daydream/blob/e395675b66aa0b7e9cf2b882495b352d6cf0e6ce/tests/deep_orchestrator/test_review_capture_and_retry.py) | 0.0029 | 30.01 | 0.0007 |
| [`tests/deep_orchestrator/test_review_capture_and_retry.py::test_parallel_scope_attribution_overflow_retains_findings_and_admitted_identities`](https://github.com/existential-birds/daydream/blob/e395675b66aa0b7e9cf2b882495b352d6cf0e6ce/tests/deep_orchestrator/test_review_capture_and_retry.py) | 0.0033 | 29.46 | 0.0025 |
| [`tests/deep_orchestrator/test_review_capture_and_retry.py::test_structure_keeps_bounded_interaction_review_and_patch_navigation[bytes]`](https://github.com/existential-birds/daydream/blob/e395675b66aa0b7e9cf2b882495b352d6cf0e6ce/tests/deep_orchestrator/test_review_capture_and_retry.py) | 0.0042 | 28.91 | 0.0009 |

The long Harbor case builds a wheel and two Docker images, executes isolated containers and invokes the Harbor metric subprocess. Review/reuse journeys exercise real Git, filesystem snapshots and artifact publication. Call-phase dominance locates the remaining work inside test bodies; it does not diagnose CPU, disk or network causality. Historical qualitative operation counts below identify concrete operations without substituting local timings for CI measurements.

## Historical residual profiling, with exact source binding

Local cProfile/audit operation counts were collected on **b56429690a6f2dcf876527d8088cc59c9d33241d**, not this new study SHA. No local duration is an acceptance benchmark. The selected tests deliberately repeat real runner calls to preserve cache invalidation and retained evidence contracts. Git blob identities for relevant production owners are recorded separately; all inspected production files are identical in the new source.

| Historical journey | Runner calls | Body Git processes | Tree manifests | Regular reads/copies | Atomic writes | fsync calls |
|---|---:|---:|---:|---|---:|---:|
| Plan change/reuse | 3 | 92 | 343 | 5497/1264 | 407 | 2977 |
| Frontier invalidation | 2 | 64 | 270 | 7749/1983 | 520 | 3863 |

The historical call chain is ReviewRun.run → runner._run_workspace → private artifact session → deep flow → freeze/publish/close. Repeated safe reads, copies, atomic publication and fsync identify concrete remaining operations, including warm reviewer reuse. Neither chosen journey issued git worktree add/remove. This does not diagnose CPU-versus-disk causality. Production optimization and broad test pruning remain outside this CI change.

The 97-versus-b564 imbalance comparison is also excluded-epoch diagnostic evidence only. It found broad per-node execution variance rather than a single heavy/Docker-selected slow shard; it is not behavior or median evidence for the new epoch.

Historical source: [runner](https://github.com/existential-birds/daydream/blob/b56429690a6f2dcf876527d8088cc59c9d33241d/daydream/runner.py#L501), [artifact owner](https://github.com/existential-birds/daydream/blob/b56429690a6f2dcf876527d8088cc59c9d33241d/daydream/artifacts/filesystem.py#L301).

## Portable reproduction and raw evidence

See [reproduction appendix](reproduction-appendix.md), [compact evidence](compact-evidence.json), and [launch manifest](launch-manifest.json).

[Archive delivery receipt](archive-receipt.json): `daydream-ci-sharding-1644-e395675b-evidence.zip`, 62,631,963 bytes, SHA256 `79cf2501ee7d31169a43726d887c61f1e6ea2b4a5361e20e664ec37d791c09b9`. This archive was delivered locally to Downloads; it has not been published to external durable storage. It contains all 27 original artifact ZIPs, raw metadata/log snapshots, pinned source files, historical supporting evidence and the independent nine-run identity/floor check.

GitHub artifact retention currently expires between `2027-01-17T21:29:23Z` and `2027-01-18T00:36:17Z` unless deleted earlier; exact IDs, stable source/archive URLs, sizes, expiry and byte digests are recorded in compact evidence. Preserve the local ZIP for verification after expiry. Reviewers can also reconstruct the inputs from GitHub while retained using the appendix. Links and hashes alone cannot recover expired bytes.

This published narrative adds the delivery receipt and final interpretation to the archived generated report. Compact evidence and the launch manifest are byte-identical to their archived copies. The delivery receipt remains outside the ZIP, avoiding a self-referential archive hash. No large raw evidence files are committed.
