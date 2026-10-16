# Branch Improve concurrency (#1404)

[Raw matched samples](branch-improve-concurrency.json) record the fixture patch,
implementation patch, calls, provenance, retries, outcomes and timings. These are
controlled synthetic-provider measurements on macOS arm64 / Python 3.14.3.
They do not measure live model latency or establish a live-provider speedup.

| Capacity | Median end to end | Median audit | Median vet |
|---|---:|---:|---:|
| 1 | 1.919701 s | 0.482664 s | 0.502821 s |
| 3 | 1.244135 s | 0.170465 s | 0.175306 s |

Three matched repetitions alternated capacity 1/3, 3/1, 1/3. Each sample entered
production `runner.run` with `flow_name="improve"`, branch focus and standard
effort, using a fresh copy of `tests.conftest.improve_branch_target`. The committed
billing diff, model identity (`mock-model`), payloads and effort were identical.
The only case difference was the synthetic backend's capacity. Each audit/vet
provider turn waited 50 ms before returning its fixed response. No model
executable, SDK request or network provider ran. The process environment remained
`PYTEST_WORKERS=1 DAYDREAM_FANOUT_CONCURRENCY=1`; only the fake advertised capacity
three. There was one benchmark process and no concurrent heavy workload.

The existing offline test fixtures supplied `make_config`, dataset isolation and
the capable fake `Backend` at `runner.create_backend`. Git, snapshot creation,
agent orchestration, event loop, task groups, lifecycle recording, grounding,
vetting, selection and artifact/plan writes remained real. Both cases used
`run_eval=False`, `archive=False` and `non_interactive=True`. The end-to-end timer
enclosed the awaited runner call; fixture copying and comparisons were outside
it. Observational wrappers around the unchanged production phase scopes measured
audit and vet separately. Those stage measurements include host overhead.

Every sample attempted eight audit and eight vet turns, reached its advertised
peak of one or three in each phase, and completed without retries or failures.
Eight candidates retained four introduced and four inherited labels. The same
normal aggregation produced seven packages containing all eight members (docs
and DX shared a package), with three introduced and four inherited package labels.
Five selected plans succeeded in each sample. Candidate/finding caps, confidence,
vet batch size and prioritization were not changed for the measurement.

All six samples had identical recon, coverage, audit, vetted, selection, report
and plan artifacts after normalizing only source/snapshot paths, session IDs,
trajectory paths and the exact `Daydream run` identity footer. List order was
preserved. Timing-bearing trajectories and plan diagnostics were retained raw
and excluded from semantic equality. The JSON includes hashes of each normalized
artifact set. This small fixture measures scheduling and actual host overhead
under a controlled wait; it does not model provider saturation, rate limiting,
heterogeneous real investigation times or real failure rates.

The production regression additionally uses event barriers rather than elapsed
speed assertions. Its paired standard/deep runs emit 90 candidates and six vet
batches (20/20/5 per category), reverse completion, and compare finding IDs,
selection, reports and durable rejection records. Separate cases cover quick
filtering/caps, tier/backend serial limits, isolated audit/vet failure, missing
verdicts, and cancellation with queued children. The identical standard regression
fails on baseline `7a5f8c38`: eight audits are planned but only one enters the
provider, leaving the three-party barrier unreleased until the test deadline.
After the repair the affected Improve/prioritization/diagram suites pass (204 tests).
