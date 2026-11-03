# Per-stack review admission measurements

The [stage-aware review contract](../extensions.md#stage-aware-review-builders-api-8) owns
behavior. This offline benchmark measures what queued per-stack scopes keep in memory; it
does not execute providers or estimate a whole review's RAM or quality.

## Method

`scripts/benchmark_captured_source.py run --phase` captures a fixture, writes its patch and
hunk index, then runs the real `phase_per_stack_reviews` over N scopes that partition the
changed files round-robin, like shards. A scripted backend with fan-out concurrency C blocks
each admitted reviewer in `execute` until C have entered. At that moment C scopes are admitted
and N − C are queued; the worker counts live `StageInputFactory` objects and records
tracemalloc's retained bytes relative to the moment before the phase started. Every
invocation then fails permanently, so no provider runs and no retry backoff is timed.

A warm-up phase over one scope loads lazy imports first, so the retained delta compares scope
state only. Each case has one traced repetition and three untraced repetitions. The untraced
runs give the phase runtime and the peak of summed per-process private memory (Linux USS,
sampled every 10 ms) across the worker's process tree. Summed RSS (`tree_rss_sum_peak`) is
diagnostic only, not private RAM, because RSS double-counts shared pages.

Baseline: `2688970` (v0.33.0). Change: this PR. Both ran the same harness against identical
fixtures: `phase-128`, `phase-1024` and `phase-4096` have 64 changed files with 128, 1,024 or
4,096 changed lines each (heads `7ecf7168`, `d536465e`, `3f752f25`). Environment: Python 3.14.6,
Git 2.43.0, Linux 6.18 x86_64, 4 logical CPUs, 15.7 GiB RAM, ext4. Runs were sequential with
warm caches and no machine-wide isolation.

## Results

Patch is the captured diff size. Live factories and the retained delta come from the traced
run; private memory and runtime are medians of the untraced runs.

| Patch | N | C | Live factories (base → new) | Python retained while queued, MiB (base → new) | Process-tree USS peak, MiB (base → new) | Phase runtime, s (base → new) |
|---|---:|---:|---:|---:|---:|---:|
| 0.17 MiB | 4 | 1 | 4 → 1 | 2.36 → 0.50 | 89.89 → 88.04 | 0.47 → 0.47 |
| 0.17 MiB | 4 | 4 | 4 → 4 | 3.11 → 1.58 | 89.09 → 87.41 | 0.51 → 0.47 |
| 0.17 MiB | 16 | 1 | 16 → 1 | 7.22 → 0.57 | 95.15 → 88.70 | 1.53 → 1.60 |
| 0.17 MiB | 16 | 4 | 16 → 4 | 7.97 → 1.39 | 95.62 → 89.18 | 1.53 → 1.47 |
| 0.17 MiB | 64 | 1 | 64 → 1 | 26.66 → 1.14 | 117.66 → 92.61 | 5.76 → 5.79 |
| 0.17 MiB | 64 | 4 | 64 → 4 | 27.37 → 1.85 | 118.06 → 92.85 | 5.81 → 5.58 |
| 1.27 MiB | 4 | 1 | 4 → 1 | 12.25 → 0.76 | 102.39 → 96.36 | 0.81 → 0.70 |
| 1.27 MiB | 4 | 4 | 4 → 4 | 12.96 → 2.63 | 101.97 → 95.73 | 0.77 → 0.75 |
| 1.27 MiB | 16 | 1 | 16 → 1 | 43.49 → 0.62 | 134.90 → 95.95 | 2.51 → 1.95 |
| 1.27 MiB | 16 | 4 | 16 → 4 | 44.19 → 1.61 | 134.93 → 96.21 | 2.26 → 1.97 |
| 1.27 MiB | 64 | 1 | 64 → 1 | 168.45 → 1.16 | 264.34 → 95.61 | 8.42 → 7.45 |
| 1.27 MiB | 64 | 4 | 64 → 4 | 169.16 → 1.92 | 264.62 → 97.26 | 8.94 → 7.07 |
| 5.02 MiB | 4 | 1 | 4 → 1 | 46.63 → 1.86 | 170.01 → 171.10 | 3.86 → 2.77 |
| 5.02 MiB | 4 | 4 | 4 → 4 | 47.34 → 7.00 | 170.51 → 170.17 | 3.61 → 2.68 |
| 5.02 MiB | 16 | 1 | 16 → 1 | 167.86 → 0.89 | 273.64 → 169.38 | 7.08 → 4.59 |
| 5.02 MiB | 16 | 4 | 16 → 4 | 168.57 → 2.70 | 273.52 → 169.50 | 6.93 → 4.80 |
| 5.02 MiB | 64 | 1 | 64 → 1 | 652.81 → 1.22 | 759.32 → 169.73 | 21.83 → 13.67 |
| 5.02 MiB | 64 | 4 | 64 → 4 | 653.52 → 2.19 | 761.51 → 170.06 | 21.25 → 13.61 |

Before the change, every planned scope built a factory before waiting for its fan-out slot,
holding its own copy of the patch, a per-file block map and the parsed index. Retained memory
grew by about 0.40, 2.60 and 10.10 MiB per added queued scope for the three patch sizes, at
either concurrency. After the change the slope stays within ±0.08 MiB per scope in every case,
live factories equal admitted concurrency, and retained memory grows only with C × scope-local
parts. For the 5 MiB patch with 64 scopes, retained memory drops from about 653 MiB to about
2 MiB and the process-tree peak from about 760 MiB to 170 MiB. The remaining peak comes from
capture and the shared change, not from queued scopes. Runtime is unchanged for the smallest
patch and falls for larger ones, because scopes no longer parse the whole patch and index.

## Reproduce

Point the benchmark at a directory outside the checkout and at the implementation to measure:

```bash
export DAYDREAM_CAPTURE_BENCH_ROOT=/path/outside/the/checkout
export DAYDREAM_CAPTURE_BENCH_CODE=/path/to/implementation/checkout
python scripts/benchmark_captured_source.py phase-fixtures
python scripts/benchmark_captured_source.py run --fixture phase-1024 --phase --scopes 16 --concurrency 4 --trace --repeat 1
python scripts/benchmark_captured_source.py run --fixture phase-1024 --phase --scopes 16 --concurrency 4 --repeat 3
```

Repeat over `phase-128`, `phase-1024` and `phase-4096`, scopes 4, 16 and 64, and concurrency
1 and 4, once for each implementation. Runs append to `$DAYDREAM_CAPTURE_BENCH_ROOT/results.jsonl`
and keep raw logs and samples under `raw/`.
