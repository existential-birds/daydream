# Captured-source streaming measurements

The [runtime contract](../runtime.md#captured-local-source) owns behavior and limits.
This offline benchmark measures capture, complete validation, and factory admission;
it does not execute providers or estimate a whole review's RAM or quality.

## Recorded comparison

Baseline: `07cd9774482f0e69d17b4f7e66abc385f362f5ce`; recorded comparison base:
`481b9a1323e386bde80e213f9bacd4b8328b2184`. Both implementations used identical,
independent source fixtures, including the original complete physical object store.
Environment: Python 3.14.3, Apple Git 2.50.1, macOS 26.6.1 arm64, 16 logical CPUs,
48 GiB RAM, local APFS. Runs were sequential with warm caches, without cache flushing
or machine-wide isolation. Each case has three untraced repetitions and separate
tracemalloc repetitions. Medians and observed ranges are not population percentiles.

| Untraced control | Paired baseline median capture / validate | Streaming median capture / validate | Streaming range |
|---|---:|---:|---|
| Recorded checkout, 1,054 native files | 4.779 / 1.971 s | 1.320 / 0.247 s | capture 1.310–1.565; validate 0.240–0.256 s |
| 1,024 files, approximately 1 MiB | 4.592 / 1.878 s | 0.967 / 0.233 s | capture 0.925–1.102; validate 0.224–0.245 s |

An earlier streaming iteration measured 1.757 / 0.314 s on the recorded checkout,
missing the capture target. Bounded LF scanning replaced byte-at-a-time raw header
consumption before the final measurements above. Those earlier samples remain in
the external evidence; they are not included in the final implementation's medians.

Across every required 64 MiB/16-file, 240 MiB/16-file, and unchanged 64 MiB-leaf
repetition, traced capture peaks were at most **2.319 MiB** and complete-validation
peaks at most **2.171 MiB**. Original capture peaks were 324.45, 746.56, and
320.16 MiB respectively. Tracemalloc measures Python allocations, including retained
metadata, and excludes Git/native memory; its perturbed timings are kept separate.

| Control | Final process-tree footprint median | Reduction from original investigation |
|---|---:|---:|
| Recorded checkout | 98.438 MiB | 48.86% (original 192.49 MiB) |
| 64 MiB / 16 files | 68.688 MiB | 85.46% (original 472.39 MiB) |
| 240 MiB / 16 files | 68.735 MiB | 95.68% (original 1,589.38 MiB) |

Footprint is the maximum sampled simultaneous sum of macOS
`proc_pid_rusage(RUSAGE_INFO_V0).ri_phys_footprint` for worker and discovered
descendants, sampled approximately every 10 ms. It is kernel per-task accounting,
not total unique machine RAM. Short-lived peaks/children can be missed. Summed RSS
is diagnostic only; cross-process USS was unavailable. Native Git can reconstruct
an entire admitted blob; concurrent readers multiply that component.

One submission with 1/8/32 target decisions performs **one** complete traversal,
instead of 3/10/34. Eight-/32-target admission medians were 0.0642/0.0667 s,
96.94%/99.08% below paired baseline. Four eight-target submissions at concurrency
1/2/4 perform **four** traversals in every repetition. Complete native/blob payload
verification falls from 80.046082 to 8.004608 MiB; these are logical verification
bytes, not physical device I/O. Four-submission medians were 0.2562/0.1494/0.1116 s.
Preparation and terminal checks are independent and remain required.

Each local capture acceptance/full-validation operation launches one raw reader,
independent of file count. Required payload coverage matches baseline. All source,
diff, inventory, target, and endpoint identities match. Recorded-checkout identity:

```text
source_id:          7490271d4b507acd7d398ab75d827ee419bd8c54a6137c091d840c701e308bd6
diff SHA-256:       45ea5817fe73bdf3dee36ca343a3681ba0efd78631b5bd3752de08697340fed8
inventories SHA:    b21a92c2cc7e3f361583594d3be814d4b74c29d0d2d353e04e4be35c24395012
patch bytes:        333734
changed targets:    81
original objects:   127256855 logical bytes / 1562 physical files
```

Logical disk peaks use approximately 100 ms samples plus completed-capture
boundaries, excluding input fixtures, logs, and pytest artifacts. Allocated blocks
are recorded separately. No final measured disk peak exceeded paired baseline;
accepted recorded-checkout disk fell from 138.614 to 135.058 MiB by avoiding redundant
loose blobs already established in copied inventories. Full history still consumes
disk approximately one byte per retained byte and remains part of admission.
Nested instrumentation timings are inclusive; do not add Git time to its owner.

## Buffer and integrity proof

Regular-file copying/reading requests at most 1 MiB. A loop may briefly retain its
previous chunk while allocating the next. Native comparison adds one same-sized
comparison chunk; loose-object compression consumes chunks synchronously and retains
only the bounded compressor state/result. Raw transport retains at most 64 KiB per
pipe buffer plus bounded framing (256-byte headers, 4 KiB requests). These phases
join before returning, and the native tree is the sole working-content disk spool.
The observed manifest retains fingerprints/digests, not regular bodies. Together
these keep simultaneous per-operation Python payload buffers below 8 MiB, independent
of history/source/leaf size. Canonical textual retrieval results (up to 8 MiB) and
necessary manifest/diff/index metadata have separate existing bounds.

Complete old/new read digests and source immutability are checked separately from
measurement. `integrity` records every readable side/path/OID/body SHA-256, explicit
absence/unavailability, native byte digest, and complete source bytes/modes before
and after, including HEAD, index, refs and object files. Compare its JSON outputs
between implementations. The owning tests independently prove net subjects,
confinement, bounded instability, semantic index refresh, off-target corruption,
framing, cancellation/reap, competing bytes, atomic admission and retained typed
terminal findings. Neither hashes nor performance measurements establish model
understanding or a sandbox.

## Reproduce

Use the repository virtualenv. Keep fixtures/results outside every checkout. The
sampler and fixture environment recording require macOS; integrity checks use
ordinary Python/Git. No provider runs. The script supports the recorded baseline
as well as the current owner; its baseline instrumentation is developer tooling,
not a second runtime capture path.

```bash
capture_lab=$(mktemp -d /private/tmp/daydream-capture-bench.XXXXXX)
export DAYDREAM_CAPTURE_BENCH_ROOT="$capture_lab"
python_bin="$PWD/.venv/bin/python"
benchmark_script="$PWD/scripts/benchmark_captured_source.py"
recorded_head=07cd9774482f0e69d17b4f7e66abc385f362f5ce
recorded_base=481b9a1323e386bde80e213f9bacd4b8328b2184
"$python_bin" "$benchmark_script" fixtures
"$python_bin" "$benchmark_script" recorded-fixture --source "$PWD" \
  --head "$recorded_head" --base "$recorded_base"
mkdir "$capture_lab/baseline-code"
git archive "$recorded_head" | tar -x -C "$capture_lab/baseline-code"
for label in before after; do
  mkdir -p "$capture_lab/$label/fixtures"
  cp "$capture_lab/fixtures/"*.json "$capture_lab/$label/fixtures/"
done
```

Set `DAYDREAM_CAPTURE_BENCH_CODE` to the extracted baseline or implementation
checkout, and `DAYDREAM_CAPTURE_BENCH_ROOT` to `before` or `after`. Both refer to the
same source fixtures. Run each command separately for both versions:

```bash
"$python_bin" "$benchmark_script" run --fixture real-head --repeat 3
"$python_bin" "$benchmark_script" run --fixture real-head --trace --repeat 3
"$python_bin" "$benchmark_script" integrity --fixture real-head
```

Repeat timing/allocation runs for `bytes-64`, `bytes-240`, `large-leaf`,
`files-1024`, and `h128-packed`; timing controls also include `h0-loose`,
`h32-loose`, `h32-packed`, and `h128-loose`. For admission, use `targets-32` with
`--scopes 1 --admissions 1 --targets 1`, then 8 and 32 targets. Use `targets-8`
with `--scopes 4 --targets 8 --admissions 4 --concurrency 1`, then 2 and 4,
with separate traced runs. Run `integrity` for each source fixture too.

Fixture formulas, seed 2176 and packed command
`git repack -a -d --window=0 --threads=1` match the original investigation. Stable
oversized-store/source refusal controls are `over-store` and `over-source` (`run`
returns nonzero and records the refusal). Existing run logs/proofs are never
silently overwritten. Saved `results.jsonl`, raw events, process samples, fixture
parameters and environment are the evidence, not console prose.

The recorded fixture copies the provided repository's complete physical store
without cloning/repacking and materializes raw tracked revision bytes without
checkout filters or archive substitutions. Later repository history changes that
input store: report its actual population and compare both implementations against
that same frozen fixture instead of claiming the original disk/timing baseline.
Schemas, source-ID serialization, extension API 8 and stage/cache contract 15 are
unchanged; package version work remains release work. Queued scope/parsed-patch
allocation is separately owned by #1711.
