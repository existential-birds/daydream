# Overloaded-stack activation: issue #1533 evidence

[Issue #1533](https://github.com/existential-birds/daydream/issues/1533) and its
[Python reproduction](https://github.com/existential-birds/daydream/issues/1533#issuecomment-6149017336)
describe concentrated scale2 diffs left under one language reviewer. This change
activates the existing planner when post-collapse assignments can usefully split
under resolved bounds. Explicit overrides, scale4 activation, tiny/shallow gates,
whole-change Structure, cumulative scope budgets and the existing concurrency
limiter retain their contracts.

## Reproducible planning

[Planner records](evidence/1533-planner.json) contain full per-file patch weights,
primary assignments, frontiers, exact target identities and repeated-plan checks.
The archive patches were byte-identical to Git diffs at these frozen revisions:

| Workload | Target HEAD | Merge base | Patch |
| --- | --- | --- | --- |
| Go, archive `45cf14e3-87e4-4091-815e-ce6a7467e662` | `9605f0e5165c3247cb56e21d43bd26bbc698c1a7` | `6f518b8cefb6d7481f07d0d6d86b4835b695d681` | 23 files, 191,317 bytes, 3,803 lines, scale2 |
| Python, archive `57efd76c-703d-44b8-b95e-8ddca768d2c6` | `0e613bdbeed4104af5e4322fe5e508088f75fe82` | `dca928e089ec8428c7a03812752369b9d4ce7c50` | 17 files, 79,554 bytes, 1,399 lines, scale2 |

Native graphs were built from each exact target, with zero Go edges and 21 Python
edges. Graph-free and native replay both preserve one nonstructural primary owner
per changed path, contiguous stable shard IDs, sorted disjoint bounded supporting
frontiers and whole-change Structure. Automatic and explicit-enable assignments
are identical at a matching cap. Bounds remain five files, 12,288 bytes, cap16,
frontier8; cap8 is a calibration configuration, not a new default.

| Workload / mode | Total scopes (language scopes) | Aggregate investigation allowance | Aggregate starts |
| --- | ---: | ---: | ---: |
| Go, explicit off | 4 (1) | 3,840s | 384 |
| Go, automatic cap16 | 14 (9) | 13,440s | 1,344 |
| Go, automatic cap8 | 9 (4) | 8,640s | 864 |
| Python, explicit off | 3 (1) | 2,880s | 288 |
| Python, automatic cap16 | 11 (8) | 10,560s | 1,056 |
| Python, automatic cap8 | 9 (6) | 8,640s | 864 |

Every scale2 reviewer retains 960s and 96 starts; totals include Structure outside
the nonstructural cap. The shared pipeline allowance stays 5,400s with 600s
reserved for synthesis. File/byte targets remain soft under the cap, and large
files remain intact. These planner results establish assignments and allowances,
not completed reviews, model quality or savings.

To repeat planning, check out an immutable target at the table's HEAD and retain
its base object. With this branch's source imported, read the complete archived
`diff.patch`, assert equality to `git diff --no-ext-diff BASE HEAD`, derive changed
paths and `file_change_bytes(diff)`, and invoke `_prepare_review_stacks` with
`RunConfig(target=TARGET, output_mode="review", diagram="off", ...)`. Compare
`deep_shard_enabled=False`, omitted, and `True`, with default cap and cap8. Build
native graphs against TARGET, never the current Daydream checkout for historical
files. Repeat each plan and count ownership with `Counter`, rather than a set union.

## Live comparison method

[Comparison records](evidence/1533-review-comparisons.json) preserve all 60 trials:
24 exploratory controls, 26 refined controls and ten exact frozen reviews. Each
has actual planned/outcome scopes, revision/diff binding, phase outcomes, elapsed
time, requests, tool starts, prompt/completion/cached tokens, cost, allowances and
independent finding adjudication. Absolute private paths and raw prompts are
excluded. Fully completed comparisons require every planned scope and required
phase to be complete; exit zero and empty findings do not establish completeness.

All trials used genuine `runner.run` and Pi/OpenRouter
`deepseek/deepseek-v4.1-flash`, medium effort, balanced latency, default review
profile digest `781622aa97a644eac6b13bb22238e5b67fb399811000859dd0eb0e4bc9f64239`,
diagram off, host review reuse off and no file config. No external model seam was
stubbed. Constructor instrumentation on later trials delegates unchanged to the
real backend and records hint10/capacity10; earlier empty observation lists mean
that instrumentation was absent, not that capacity was measured differently.
Provider prompt caching remained active and is reported separately.

The isolated experiment source was detached baseline
`63e57d1c7d4880e916a41f3096e1d97e8469b665` plus the production orchestrator/config/
run-config patches. Its entire production Python tree matches this branch:
`71102d6610e13f47fa4f1321c1a7c0123a515c6b5b612bfbd81261c245b94432`.
The digest hashes sorted relative `daydream/**/*.py` paths followed by file bytes;
the baseline SHA alone does not identify the patched experiment source. Trials
clone immutable fixtures, assert unchanged tracked source/HEAD afterward and bind
typed coverage to the full captured patch including its final newline.

For each language, repeat1 ran off then auto and repeat2 auto then off. Independent
language lanes and synthetic/frozen lanes overlapped. Additional Go-local and
frozen-Python third pairs ran concurrently after an incomplete outcome. All failed
rows remain in the records; additional pairs are disclosed. The small sample,
shared provider/host load, caching and order variability prevent an isolated
scheduling claim or generalized completion guarantee.

The committed [`measure_overloaded_review.py`](../scripts/measure_overloaded_review.py)
creates signed temporary Git fixtures and measures live trials. For example:

```sh
make lockcheck
uv run python scripts/measure_overloaded_review.py \
  --prepare /tmp/sharding-python-clean --language python --variant clean
uv run python scripts/measure_overloaded_review.py \
  --source "$PWD" --fixture /tmp/sharding-python-clean \
  --output /tmp/sharding-python-clean-off-first --mode off
uv run python scripts/measure_overloaded_review.py \
  --source "$PWD" --fixture /tmp/sharding-python-clean \
  --output /tmp/sharding-python-clean-auto-first --mode auto
```

Repeat with new output directories, reverse order, and prepare `local`/`cross` for
both languages. For a frozen target, supply its immutable worktree as `--fixture`
and its exact merge base as `--base-commit`; the source stays this branch. `cap8`
is also supported by the tool, but the live results here compare off versus default
automatic cap16. Cap8 evidence here is planning only. Fresh signed fixture commit
identities may differ; compare fixture contents and pin one HEAD within every pair.

## Quality controls and observed timing

Each refined fixture changes seven helper files, adds tests and explicitly deletes
the legacy test path. Real before/after tests cover original APIs and new helpers.
Clean baselines/features pass; local variants fail only the empty-mean family and
cross-file variants only the millisecond/second expiry family. Separate executable
gold probes were never installed in reviewed fixtures or prompts. Python expiry
and its caller span primary shards with bounded frontiers. Go preserves the
same-package context limitation tracked by #1495; no graph-discovery change is
claimed.

Refined controls produced 18/18 expected unique-root observations, no false
positives in eight clean reviews, and 44 raw findings including 26 repetitions or
supporting observations. This is 100% seeded-root recall/precision in this small
matrix, not an exhaustive guarantee. Twenty-five of 26 reviews were complete.
`go-local-auto-2` returned malformed output for `go#0`; successful siblings retained
the real seed. It is excluded from completed timing. The extra Go-local pair
provides another completed automatic observation without removing that failure.

Scale1 controls retain 480s/48 starts per reviewer and a 2,700s shared pipeline.
Off plans have three scopes (1,440s/144 aggregate starts); auto plans have four
(1,920s/192). More aggregate opportunity is explicit.

| Completed control | Off seconds, mean (range), n | Auto seconds, mean (range), n |
| --- | --- | --- |
| Go clean | 129.3 (117.6–141.1), 2 | 111.0 (110.6–111.5), 2 |
| Go local | 179.0 (120.9–272.6), 3 | 121.7 (118.8–124.7), 2 |
| Go cross-file | 212.9 (137.7–288.0), 2 | 126.7 (108.2–145.2), 2 |
| Python clean | 118.8 (84.8–152.7), 2 | 100.3 (68.0–132.6), 2 |
| Python local | 84.2 (60.9–107.5), 2 | 118.9 (90.9–146.8), 2 |
| Python cross-file | 144.7 (102.9–186.4), 2 | 131.2 (129.4–133.0), 2 |

Python local automatic review was slower. Requests and token use also vary; the
JSON reports individual values, completed/all-trial means, ranges and sample
standard deviations. Unsupported observations and repeated findings never inflate
seed recall. Initial exploratory fixtures lacked comprehensive API tests and had
a Go subsecond/large-limit calibration problem. They were retained separately:
16/16 seeds, four unsupported false positives, one additional valid missing-test
observation, 34 repetitions/support observations and 21/24 complete reviews. Two
arbiter exhaustions and one schema failure remain recorded. The refined fixtures
fix the calibration before the fixed repeated matrix; production source stayed
unchanged throughout. Early reporting errors in patch newline hashing and typed
failure collection were corrected against captured artifacts, not by rerunning or
discarding unfavorable reviews.

## Exact frozen live workloads

Nine of ten fresh reviews completed. `python-off-2` exited successfully with empty
findings, but its Python scope explicitly declined three files and records
`evidence_incomplete`. That 854.9s observation stays in the evidence and is excluded
from completed timing; the extra third pair supplies a second complete off review.
These fresh runs do not convert the original archives' 960s incomplete cutoffs
into completed baselines.

| Completed frozen mode | Individual seconds | Mean ± sample SD | Mean requests | Mean prompt / completion tokens |
| --- | --- | --- | ---: | ---: |
| Go off | 788.9, 548.8 | 668.9 ± 169.8 | 140 | 5,660,209 / 174,146 |
| Go auto | 436.3, 463.0 | 449.7 ± 18.9 | 207 | 8,469,931 / 267,341 |
| Python off | 975.5, 494.2 | 734.9 ± 340.3 | 103.5 | 5,547,484 / 186,327 |
| Python auto | 343.3, 472.0, 346.1 | 387.2 ± 73.5 | 176.3 | 8,844,876 / 229,484 |

Observed mean wall time fell, while requests, tokens and aggregate allowance grew.
This comparison measures the changed default policy as a whole, including that
allowance change; it does not isolate scheduling savings. No provider invocation
error records were observed in the refined or frozen studies. Malformed admission
and declined evidence are separate coverage failures, not provider outages.

Frozen targets are unseeded real changes without an exhaustive clean oracle, so
recall/precision percentages are undefined. Independent exact-source adjudication
confirmed a Discovery timeout comment inconsistency and a migration-runner
dependency-trigger gap in Go auto results. The latter was confirmed by source and
path-filter inspection, not an actual failing CI reproduction; one report also
overstated manifest routing and misnamed the hook stage. A malformed-quote
redaction observation from off1 and an implementation-specific explicit-unlock
test recommendation from auto2 did not demonstrate contract defects. The unlock
adjudication considered the separate in-process keeper test and
[PostgreSQL session-lock semantics](https://www.postgresql.org/docs/16/explicit-locking.html#ADVISORY-LOCKS).
Python's empty findings establish neither exhaustive cleanliness nor defect recall.

## Regression and review evidence

The Go and Python real-runner activation journeys fail at the exact baseline
because terminal coverage contains one language scope, then pass with multiple
actual scopes here. Supplemental graph-contract regressions demonstrate zero graph
builds for fitting, singleton, cap-coarsened and explicit-false plans, one shared
native graph for diagram/sharding, and intact graph-failure planning.

Event/barrier tests prove constrained capacity and host ceiling10 without timing
sleeps. Cancellation closes active streams, rejects aborted streamed decisions,
retains admitted progress and prevents complete publication/cache stores. Ordinary
sibling failures preserve successful siblings. Public ReviewRun artifacts verify
stable records/revisions, plan-inventory cache invalidation and sanctioned large
inline transport. Existing serial-stage tests explicitly disable sharding to retain
their original lifecycle assertions; transport tests continue exercising automatic
shards. No shared fixture harness or consumer production module changed.

`$test-audit` covered all nine changed test files, including the capture/retry and
findings-artifact consumers. Independent preservation review found no weakened
assertions. `$code-review` compared against exact baseline
`63e57d1c7d4880e916a41f3096e1d97e8469b665` on separate Standards and Spec axes;
source/test reviews reported zero remaining findings after graph checks were added.

Validation: lock consistency passed before uv commands; the requested affected
suite passed (234 tests at the initial integration gate), targeted additions and
209 consumer tests passed, typecheck passed for 722 sources, and the final local
`make check` passed with **7,037 passed, 14 skipped**, 91% coverage. Docker-backed
actionlint, lint, dead-code, naming and lock checks passed. An earlier full gate
exposed 15 serial-script/scope-name regressions; all were reproduced against a
passing baseline and repaired without removing the original assertions.

Local skips were four opt-in live smoke tests, three Darwin cases covered by other
platform-specific tests, one Unicode filesystem normalization case, five unsupported
non-UTF-8 filename cases and one unavailable Prime RL workspace test. The live
comparisons above are separate real-provider measurements. This change does not
touch `rl/daydream_review`; its standalone gate is outside `make check`.
