# Review runtime, diagnostics, and measurements

Daydream bounds model investigation, reserves time to serialize usable evidence,
and publishes validated partial reviews with explicit incomplete warnings.
Publication and fix execution have separate lifetimes. A timeout never establishes
that a review is clean.

## Investigation and finalization

The host enforces time and tool-start limits across backends, independently of
native turn-limit support. The existing 60-minute outer ceilings remain.

| Role | Investigation | Tool starts | Finalization reserve |
|---|---:|---:|---:|
| Exploration specialist | 300 s | 16 | 120 s |
| Intent | 120 s | 12 | 60 s |
| Alternatives | 300 s | 24 | 90 s |
| Language, generic, or structural reviewer | 480 s | 48 | 120 s |
| Arbiter | 120 s | 16 | 60 s |
| Suppression or supervision | 120 s | 12 | 60 s |
| Merge | 180 s | 16 | 60 s |

Diff exploration has a 450-second outer timeout for investigation, finalization,
and cleanup. Increasing the general invocation budget does not override shorter
phase limits.

The whole-review model deadline defaults to 2,700 seconds for modest diffs.
Changes exceeding 1,000, 5,000, or 10,000 lines, or 64, 256, or 512 KiB, scale
that deadline and per-role limits by 2×, 4×, or 6×. The latter two tiers enable
sharding unless explicitly disabled. An explicit review profile retains its
configured whole-review deadline:

```toml
schema_version = 1
name = "ci-review"

[pipeline]
review_wall_budget_s = 2700
```

The deadline enters the profile digest and includes queueing, retries,
exploration, intent, discovery, adjudication, merge, and diagram requests.
Discovery reserves five minutes per workload multiplier for synthesis; below
25 minutes it reserves 20% of the total. Zero allows deterministic publication
of an explicitly incomplete review without model dispatch. Resumes receive a
fresh deadline but retain persisted incomplete markers.

At cutoff, the host first uses a complete schema-valid response or the latest
valid assistant-turn checkpoint. Otherwise, a fresh invocation receives the
caller task, assigned files, output contract, supplied context, and completed
source evidence. It receives no discovery strategy, fresh-read requirement, or
speculative notes. Original budget-stop reasons remain attached. Finalization
and retries are clamped to remaining absolute time; no invocation starts when
no time remains. Stopping one invocation does not cancel its siblings.

Finalization controls are invocation-local:

| Backend | Controls | Boundary |
|---|---|---|
| Pi | `--no-tools`, serialization system prompt, thinking capped at `low` | Native tools are removed; the absolute deadline bounds generation. |
| Claude | `tools=[]`, strict empty MCP configuration, serializer guard, `low` effort | `allowed_tools=[]` alone does not remove tools under bypass permissions. |
| Codex | Reasoning capped at `low`, host zero-tool guard | No native tool-disable guarantee; read-only sandbox tools remain available. |
| Osprey or custom | Host deadline and zero-tool guard | Native controls require an advertised optional capability. |

Explicit lower supported effort settings remain intact. The host tool guard
stops events after emission; it is not a pre-execution security boundary.
Effective reasoning and native options appear in request events and spans.

The evidence capsule reserves 24 KB for declared task/context, 24 KB for captured
inputs (12 KB each), and 48 KB for completed tool results. Exact-path capture
validates UTF-8, rejects symlinks, and checks identity/hash and call mode. It never
extracts arbitrary paths from prompt prose. Completed blocks retain call/result
association, errors, exit status, cancellation, and truncation. Reads deduplicate;
late results cannot evict foundational evidence. Bounds and omissions remain
explicit. Pending starts are bounded too.

Partial stack records retain `incomplete: true` and stable finding identities
through adjudication and merge resume. Invalid JSON, reasoning, and unverified
notes never become findings. Invocation-local checkpoints do not recover from
an operating-system kill. Incomplete warnings prevent approval and resolution
of absent prior findings.

## Review scope and Pi transport

Exploration maps conventions, dependencies, and tests rather than repeating
correctness review. Reviewers make a finite pass, investigate concrete candidates,
and keep rejected candidates closed unless new evidence changes their premise.
They do not install dependencies or repair the review environment.

For default-policy changes with at most three non-test source files and a diff
at most 64 KiB, exploration uses static impact mapping and repository guidance
without model mapping calls. Larger changes and custom exploration strategies
retain specialist mapping. All changed files still reach primary reviewers.
Cache identity includes exploration strategy identity.

The built-in structural prompt owns the packaged default design-alternatives
check. Its presence suppresses the separate default alternatives invocation and
writes an empty compatibility artifact. Custom alternatives strategies, custom
structural prompt builders (including wrappers), and runs without structural
review retain their independent pass. Failures remain incomplete; resumes retain
prior alternatives and the structural design responsibility.

Automatic uncovered-file sweeps and reviewer-read coverage accounting have been
removed. Findings, failed-stack warnings, bounded recovery, and source-grounded
diagram validation remain. Evaluation retains cost, timing, finding quality,
duplicates, and citation-location checks.

Pi discovery uses the actual live session diff path for every diff size. Intent,
alternatives, per-stack, structural, and adjudication prompts retain this path
instead of embedding the diff. Reviewers inspect it and source with read-only
tools. Small reviews use the same path and explicit structural review. Per-stack
reruns replace stale structural outputs; resumes load the current assignment.

A durable diff reference has a separate 128 MiB streaming-validation bound.
Other captured inputs retain limits of 1 MiB per file and 4 MiB combined.
Admission checks regular, non-symlink UTF-8 files, identity/hash, backend, cwd,
and read-only mode. Every retry revalidates the reference. Reconstructing
`<repo>/.daydream/diff.patch` can select a detached public tree; use the admitted
session path. INLINE transports retain bounded capture and isolation.

Pi passes ordinary logical prompts and schema appendices through private UTF-8
`@file` references, and system instructions through a separate file-path option.
Files close before spawn and survive until child teardown, then are removed on
success, failure, cancellation, or generator close. Tools-disabled execution
uses stdin. Request events retain logical prompts, independent of filenames.
Pi's file processor adds its own `<file name="...">` wrapper to model input.

Recovery finalization cannot fetch evidence. Pointer-only diffs are excluded
from content capture; reference metadata, established findings, and bounded
completed reads remain available. Uninvestigated files remain incomplete.

## Diagnostics and rollout

Use CLI exit status and validated findings as publication gates. A manifest can
record pipeline success before a later findings-export failure. External
trajectories and local archives survive handled failures; platform/process kills
can prevent finalization or upload.

`--dump-artifacts` copies clean or advisory-only bundles unchanged. Blocking
secret findings trigger sanitization of a separate private copy and a rescan;
only a copy without remaining blockers is exported. Original evidence and
archives retain their bytes. Accepted dumps keep `bundle/manifest.json`, merge
into their destination, and preserve unrelated files. The scan covers generated
bundle files, not unrelated destination content.

Scanner/sanitizer failures or remaining blockers withhold the dump with a
warning and leave the destination unchanged. They preserve review exit status,
findings, trajectories, and the local archive. Archive integrity and output
publication failures remain fatal. Upload workflows must handle a missing dump.

For rollout:

1. Pin analysis and posting/validation to the same Daydream revision.
2. Log `gh --version` to diagnose repository-metadata compatibility.
3. Keep the initial 60-minute analysis timeout and 45-minute model budget;
   setup, cleanup, and exports need the remaining headroom.
4. Set dedicated diagnostic paths:

   ```sh
   --trajectory "$RUNNER_TEMP/daydream-debug/trajectory.json" \
   --dump-artifacts "$RUNNER_TEMP/daydream-debug/bundle"
   ```

5. Upload `daydream-debug/` with `if: always()`, preserving the findings artifact.
   Exclude credentials, home directories, provider auth, and runtime directories.
6. Retain strict findings schema/head validation and posting gates. Diagnostics
   never authorize approval or stale-finding resolution from an incomplete review.

## Measurement evidence

[measure_review_runtime.py](../scripts/measure_review_runtime.py) creates isolated
repositories and artifacts and never posts reviews. Reproduce its initial
baseline with fresh output directories:

```sh
mkdir -p /tmp/daydream-runtime-baseline
git archive ebd15ca95666dd2545c70fb290991abc2e52a735 daydream | tar -x -C /tmp/daydream-runtime-baseline
uv run python scripts/measure_review_runtime.py --source /tmp/daydream-runtime-baseline --output /tmp/before
uv run python scripts/measure_review_runtime.py --source . --output /tmp/after
```

The [initial sample](measurements/review-runtime.json) retained both seeded
regressions in all four runs, with no extras. Tools fell from 17 to 8 across two
pairs; generation intervals fell from 9 to 6. Mean elapsed time fell from
55.86 to 48.97 seconds, but the second pair was slower. Provider/transport
intervals do not separate inference from queueing.

The [follow-up sample](measurements/review-runtime-followup.json) adds a clean
pagination fixture whose before/after implementations agreed across 864 input
combinations. Default and two-tool runs retained both seeded defects and valid
clean results. Reserve-only finalizers retained budget-stop reasons: the
candidate completed both fixtures while the baseline timed out on one. The
candidate supplied diff-grounded results with no source tools, which does not
establish complete coverage. Adjudication remains conservative about stopped
phases even when finalized output is valid.

A deterministic 200-read workload falls from 2,000 virtual seconds to 490 seconds
and 48 reads, retaining the validated finding and incomplete warning. Hung-stream
and hung-finalizer tests check real-clock bounds. Runner/protocol-child tests
cover live-session diff references, 3,690,129-byte diff blocks, findings through
merge/publication, large prompts, and concurrent cancellation.

These small fixtures establish bounded work and functioning contracts. They do
not establish equivalent production recall, model convergence, or broad latency
improvement. Limits can reduce coverage on large changes; warnings expose that
tradeoff rather than proving it harmless.

### Incident references

The historical logs identified the failures these boundaries address:

| Run | Observed failure | Evidence limit |
|---|---|---|
| [35601480761](https://github.com/shelfspace-app/shelfspace-mono/actions/runs/35601480761/job/106338360088?pr=2826) | Alternatives and structure exhausted 1,800 s; alternatives aborted publication. | 245 logged tool starts; no trajectory artifact. Time cannot be attributed to inference, tools, or queueing. |
| [35621233648](https://github.com/shelfspace-app/shelfspace-mono/actions/runs/35621233648/job/106404625753?pr=2826) | Review/merge finished in about 14½ min; finalizers reused investigation prompts and export rejected malformed head metadata. | Benchmark-framing causation and the runner's gh version were not established. |
| [35633835751](https://github.com/shelfspace-app/shelfspace-mono/actions/runs/35633835751/job/106446456074?pr=2826) | Review completed in about 12 min with six cutoffs; blocking credential matches prevented diagnostics publication. | No artifacts recovered. A non-secret PostgreSQL template reproduced a blocker, but actual matched bytes remain unknown. |

An isolated replay later published findings, trajectory, and a sanitized dump
with incomplete warnings intact. Reviewers still hit limits and reopened
rejected candidates; removing advisory stages did not establish convergence.

GitHub metadata handling accepts absent/exactly empty `nameWithOwner` and derives
validated owner/name components. Nulls, malformed populated slugs, and
contradictions fail closed; case-only differences agree. The related upstream
projection fix appeared in [gh v2.89.0](https://github.com/cli/cli/releases/tag/v2.89.0).

## Latency profiles and the per-profile report

Deep runs choose wonder and arbiter spend with `fast`, `balanced` (default), or
`forensic`. The route is the monotone maximum of profile and mandatory diff risk
floors. Security, concurrency, persistence, interface, and migration risk cannot
lower effort. `forensic` retains existing behavior. Decisions are recorded in
`.daydream/deep/latency-routing.json`.

Compare the fixed corpus:

```sh
uv run python -m daydream.eval.latency_report --corpus tests/fixtures/latency_profiles/manifest.json
```

| Metric | Meaning |
|---|---|
| `runs` | Observed corpus runs under the profile. |
| `phase_latency_seconds.<phase>.p50` / `.p90` | Nearest-rank percentiles of per-run wall time. Wonder uses `alternatives`; arbiter uses the aggregate `deep` bucket, with legacy `arbiter` honored for older hand-authored corpora. |
| `high_severity_recall` | Golden high-severity `(file, line)` pairs present among shipped high findings, over all golden high pairs. |
| `false_positive_rate` | Shipped items matching no golden pair, over all shipped items. |
| `contested.kept` / `.dropped` | Routing-record contested targets that survived or disappeared. |
| `shipped_by_lens` | Merge-schema `lens` attribution. The separate `citations` block reports corroborating `(Sources: ...)` prose. |
| `calibration.surface_signals` | Trigger lists used by routing. |

`review_runtime` separates cold fix and warm reuse via explicit `sample_group`.
Its header names the corpus, sample size, nearest-rank p50/p90, and
`Target: 5-15 min for a small follow-up fix`. Missing groups remain `ungrouped`.
The committed reuse sample contains one cold and one warm case. Refresh both
on the organization's review VM before rerunning the command. The report makes
no claim beyond named cases; raw pre-merge `analyze_findings.per_lens` is separate.
