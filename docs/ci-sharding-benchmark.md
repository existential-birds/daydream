# Full CI sharding measurement

Issue [#1644](https://github.com/existential-birds/daydream/issues/1644) requires
controlled measurements on free standard GitHub-hosted `ubuntu-24.04` runners.
The original historical runs span different revisions and cannot establish a
causal speedup. Actual controlled results remain pending until the procedure
below has completed; changing the workflow alone does not prove the target.

## Controlled configurations

Use one pushed commit for every measurement. Dispatch `ci.yml` with `test_jobs`
set to `1`, `2`, or `4`. One job runs the original sequential gate, including
`make test` with eight pytest workers, as the existing baseline; two and four
jobs each use four workers. All configurations use the
same root collection, dependency lock, Python/native CLI versions, branch
coverage instrumentation, aggregate coverage floor, and static gate. The baseline
retains sequential checks in `legacy-check`; treatments split `static-checks`,
`test-shard`, and `coverage`. A cheap planner and final required `check` job are
shared measurement overhead in every configuration. The baseline adds JUnit
reporting through `PYTEST_REPORT_ARGS=--junitxml=baseline-tests.xml` on the root
pytest invocation, preserving test selection and coverage instrumentation.
These arguments do not enter `PYTEST_ADDOPTS`, so nested pytest subprocesses do
not inherit the reporting destination. RL stays separately dispatched. Do not
compare a bare pytest run with a covered gate.

Run each configuration at least three times. Alternate configurations rather
than running all baselines first, and avoid deliberately overlapping benchmark
runs: account concurrency contention is part of the measurement and can obscure
the effect of sharding. Keep cache policy identical and record cold/cache-warm
runs and unrelated repository activity. Retain failures instead of replacing
them silently with faster successes.

## Launch and capture

`scripts/ci_benchmark.py` uses the authenticated `gh` CLI and Python's standard
library. The following launches one measurement and returns once its run ID is
known; it does not wait for CI to finish:

```bash
python3 scripts/ci_benchmark.py launch \
  --repo existential-birds/daydream --ref YOUR_BRANCH --sha FULL_COMMIT_SHA \
  --jobs 1 --manifest /tmp/daydream-ci-benchmark/manifest.json
```

After that run finishes, repeat with `--jobs 2`, then `--jobs 4`, and repeat the
cycle three times. `--repeat 3` is available when intentionally measuring
contention, but those overlapping runs must be identified in the analysis.
The helper checks the remote ref before every dispatch and binds discovery to
the exact SHA, ref, workflow, and event. Keep the branch fixed until all runs
have been dispatched. A run that races a branch update is rejected by reporting.
Do not dispatch another copy after a discovery timeout: the manifest retains
the pending dispatch. Resolve its identity explicitly:

```bash
python3 scripts/ci_benchmark.py resolve \
  --manifest /tmp/daydream-ci-benchmark/manifest.json --run-id RUN_ID
```

Collect a snapshot at any point, including while runs remain active:

```bash
python3 scripts/ci_benchmark.py report \
  --manifest /tmp/daydream-ci-benchmark/manifest.json \
  --output /tmp/daydream-ci-benchmark/results
```

The output preserves raw run/job API responses, derived `measurements.json`, and
`report.md`. Commit the final report and compact metrics with source run links;
retain the raw snapshots as accompanying evidence. Run attempts must not be
rerun under the same ID during benchmark capture; use fresh dispatches for each
sample. For ordinary CI recovery, choose **Re-run all jobs**. Receipts bind every
shard to an immutable workflow attempt, so **Re-run failed jobs** would mix the
new attempt with previously successful shard artifacts and must fail validation.
Full reruns replace shard evidence and coverage reports.

## Interpretation and evidence

Gate wall time starts at run `created_at` and ends at the required `check` job's
completion, including initial queue delay and dependencies. The report records
each shard's pytest duration, setup before `Run pytest shard`, total job time,
and delay between planner completion and shard job start. That dispatch delay
includes GitHub scheduling; the API does not expose a separate ready timestamp.
The baseline's `Run tests` step includes script unit tests after pytest, preserving
the old complete invocation; its pre-test time includes sequential static checks.
Treatments account for those unit tests in the static gate. Total runner minutes
sum all non-skipped job durations, including provisioning,
coverage, and final gate jobs. Imbalance is maximum shard pytest duration divided
by mean shard pytest duration. The report shows observed peak job overlap.

Record the account/repository concurrency limit from its authoritative settings
or administrator evidence, and inspect active jobs during four-job treatments.
Observed overlap is a lower bound on available concurrency, not the configured
limit. Explain competing runs, long dispatch delays, and whether all four shard
jobs can actually overlap. Compare both median end-to-end gate time and runner
minutes; include setup and artifact overhead instead of only pytest time.

Inspect artifacts and logs for every run to verify identical complete collection,
disjoint shard selections, identical skips, all expected coverage inputs, the
unchanged aggregate floor, and success of the remaining gate steps. Timing
metadata alone cannot prove those contracts. Confirm runner labels and the
runner-image hardware/setup log, not only an action's display name.

The target is at least a 25% reduction in median full-gate wall time relative to
one job. Investigate if it is missed. Compare two against four jobs explicitly;
document queue/concurrency effects if four fail to improve before proposing a
smaller count. Three passing timing samples are necessary but do not alone prove
correctness or the target. Baseline runs publish `baseline-tests.xml` alongside
`coverage.xml` in `coverage-report`. Compare the exact JUnit testcase identities
and skipped identities against each treatment's full collection and union of
executed shard reports. A manifest node is skipped when its setup or call report
has outcome `skipped`. Preserve parametrized IDs when mapping JUnit `classname`
and `name` back to pytest node IDs; verify the mapping against the full node ID
collection and reject ambiguous mappings. Require disjoint shard selections and
exact identity/skip equality for every measured run. Matching totals alone do
not establish equivalent selection or skips.

## Residual bottlenecks

Historical [run 39292965299](https://github.com/existential-birds/daydream/actions/runs/39292965299)
on the PR #1643 revision reported 7,783 passes, five skips, and 2,364.35 seconds
inside pytest. Its slowest test was
`tests/test_benchmark_harbor_package.py::test_audit_execution_proofs_harbor_gated`
at 144.58 seconds. It builds the wheel and two Docker images, exercises isolated
containers, and launches the Harbor metric subprocess. Deep review/reuse
journeys dominate the remaining longest tests: the `[files]` bounded-interaction
case took 89.16 seconds, and the recorded-frontier invalidation journey took
74.33 seconds. These figures describe one historical environment, not a diagnosis
of CPU versus disk or controlled treatment effects.

Retain per-node setup/call/teardown durations in controlled runs. Compare fixture
setup and real Git/worktree/subprocess/artifact operations in the slowest shards
to explain remaining tails; unequal test costs can expose uneven groups even
when total test counts look balanced. Keep regression coverage and real
filesystem/process boundaries. Production optimization and broad test pruning
belong in separate changes with their own evidence.
