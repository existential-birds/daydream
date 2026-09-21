# Review runtime: incident, changes, and measurements

This change keeps the partial-review fix from `ebd15ca9` and bounds the work
that precedes publication. It also removes repeated context reads and conflicting
output instructions. Increasing a reviewer's timeout from 30 to 60 minutes did
not address the underlying work expansion or fit a 60-minute Actions job.

## Incident baseline

[Shelfspace run 35601480761, PR 2826](https://github.com/shelfspace-app/shelfspace-mono/actions/runs/35601480761/job/106338360088?pr=2826)
used Daydream `8d94fdb1`, Pi, OpenRouter, and
`deepseek/deepseek-v4.1-flash`. The downloaded Actions log was retained at
`/tmp/daydream-35601480761.log`. The GitHub artifacts API returned an empty
artifact list; there is no trajectory available for retrospective attribution.

| Observed interval (UTC) | Elapsed | Logged tool calls |
|---|---:|---:|
| Exploration, 12:46:56–12:51:49 | about 293 s | 55: 36 read, 15 grep, 3 find, 1 ls |
| Intent, 12:51:49–12:53:31 | about 102 s | 6 read |
| Alternatives + review fan-out, 12:53:31–13:23:31 | 1,800 s | 184: 68 read, 114 bash, 2 write |
| Total | about 36 min 36 s | 245 |

The minute/second boundaries above are rounded from log timestamps. Tool
counts come from the `🎠` tool headers, not lines mentioning tools in prose.
Thirteen reads name the shared exploration summary, affected-file map, or
intent artifact. Multiline argument reconstruction shows repeated source and
diff reads too, but arbitrary bash commands prevent a complete source-read count.
For example, reproduce the tool totals with:

```sh
gh run view 35601480761 --repo shelfspace-app/shelfspace-mono --log > /tmp/daydream-35601480761.log
rg -o '🎠 [[:alnum:]_]+' /tmp/daydream-35601480761.log | sort | uniq -c
```

Both alternatives and structure hit their wall limits at 13:23:31. The old
alternatives failure then aborted the run before merge/publication. Investigation
late in the run included upstream Actions release pages and Daydream source.
These are observed activities; their individual contribution to elapsed time is
unknown. Log panels interleave agents and lack correlated request/result timing.
We cannot assign the 30 minutes to inference, tools, retries, or provider queueing.

## Architectural causes

- Pi accepts `max_turns` but does not enforce it. Exploration's nominal turn
  limit therefore did not bound Pi's work. The host tool budget was unlimited.
- Alternatives and structure could repeat much of the local correctness review.
  Their strategies encouraged open-ended investigation without a clear stopping
  condition after disproving candidates.
- Each fresh reviewer followed pointers to the same intent and exploration
  artifacts. Model round trips were used to transport already available context.
- Per-stack prompts asked for a markdown file and prose verdict lines while the
  host consumed a structured JSON response. Alternatives requested a numbered
  list despite its JSON schema. The two logged writes are consistent with this
  conflicting contract; the log cannot quantify its reasoning cost.
- Discovery had only a hard cutoff. A reviewer could spend its entire allowance
  investigating and return no usable findings. Completed stack records survived
  only after the previous budget fix; valid partial records from an unfinished
  stack were still discarded.
- The critical path is exploration → intent → max(alternatives, stack fan-out)
  → uncovered sweep → arbitration/suppression → merge → supervision/diagrams
  → findings and artifact publication. Separate per-agent limits did not bound
  that complete chain or time spent waiting for a fan-out slot.

Host concurrency is already present: Pi's default fan-out hint is ten, and its
transport is per invocation. The incident's three review stacks fit below that
cap. Alternatives has a separate sibling invocation. There is no evidence that
a host-side serial backend lock caused the incident, so concurrency was not
raised. Provider-side queueing remains unknown. Single-stack alternatives stay
serial because their findings enter that review through its artifact pointer.

## Implemented behavior

The host enforces investigation time and tool-start limits across backends,
independently of native turn-limit support. Existing 60-minute hard ceilings
remain as outer safeguards. Earlier limits now bound useful work:

| Role | Investigation | Tool starts | Reserved finalization |
|---|---:|---:|---:|
| Exploration specialist | 120 s | 16 | 30 s |
| Intent | 120 s | 12 | 60 s |
| Alternatives | 300 s | 24 | 90 s |
| Each language/generic/structural reviewer | 480 s | 48 | 120 s |
| Each uncovered-file sweep | 90 s | 10 | 30 s |
| Arbiter | 120 s | 16 | 60 s |
| Suppression/supervision | 120 s | 12 | 60 s |
| Merge | 180 s | 16 | 60 s |

A whole-review model deadline defaults to 2,700 seconds. It includes queueing,
retries, exploration, intent, discovery, adjudication, merge, and optional diagram
requests. Discovery stops five minutes earlier, reserving time for synthesis.
For budgets below 25 minutes the synthesis reserve is 20% of the total. A resumed
run gets a fresh deadline; persisted incomplete markers remain in force. Fix
execution and artifact publication do not inherit this model deadline.

Configure the total using an explicit review profile passed with
`--review-profile /path/to/profile.toml`:

```toml
schema_version = 1
name = "ci-review"

[pipeline]
review_wall_budget_s = 2700
```

The deadline is included in the profile digest. Zero permits deterministic
publication of an explicitly incomplete review without dispatching model work.

At an investigation cutoff, the host first retains an already complete,
schema-valid response or the latest schema-valid assistant-turn checkpoint.
Otherwise it gives the same backend a fresh finalization request containing the
original inputs and a bounded capsule of completed tool evidence. That request
gets no investigative tool allowance and cannot exceed its reserved time or the
shared deadline. A tool-start limit is an event-level stop, not a pre-execution
security boundary. Existing backend access restrictions remain authoritative.

The evidence capsule retains at most 48 KB of completed tool blocks and 12 KB of
unfinished notes. Long individual outputs are marked truncated. Failed retry
attempts cannot contribute evidence/checkpoints to later attempts. Input capture
is revalidated before finalization; capture failures still propagate. Finalized
partial results require full schema validation, not merely a salvageable shape.
Closing a budget-stopped stream does not call backend-wide cancellation.

The full hook suite also exposed a signal-handling race: an interrupt delivered
inside an AnyIO task-group body can arrive at the CLI wrapped in an exception
group. Pure interrupt groups now take the normal shutdown/exit-130 path; mixed
groups still propagate their other failures. The existing real-signal recorder
test and deterministic grouped-interrupt regressions cover this behavior.

Partial stack records carry `incomplete: true`, survive adjudication rewrites
and merge resume, and retain their stable finding identities. Their declared
clean verdicts are discarded. Warnings still prevent approval or resolution of
absent prior findings. Invalid or unfinished JSON is never checkpointed as a
finding. Checkpoints are invocation-local until the phase persists them; this is
not durable recovery from an operating-system kill during investigation.

Small, whole intent/summary/file-map artifacts are shared inline (4 KiB each,
with pointer fallback for larger/missing/invalid files). They do not count as
source-read receipts. Source evidence and coverage gates remain in place.
Reviewers return JSON; the host writes the review artifacts. Default strategies
separate intent/design, local correctness, and cross-module responsibilities and
give concrete conditions for ending an investigation.

## Comparable measurements and quality

The [measurement script](../scripts/measure_review_runtime.py) creates independent
temporary repositories and artifacts. It never posts reviews. Both versions use
Pi/OpenRouter with the incident model and the same two seeded regressions:
empty-batch division by zero and authorization changed from AND to OR.

| Pair | Baseline elapsed / tools / generations | Updated elapsed / tools / generations |
|---|---|---|
| 1 | 85.080 s / 8 / 4 | 40.077 s / 4 / 3 |
| 2 | 26.642 s / 9 / 5 | 57.862 s / 4 / 3 |

All four runs returned both seeded defects, no additional findings, and a
`has_findings` verdict for the assigned source. No budget limit fired. Tools
fell from 17 to 8 across the two pairs (53%); provider generation intervals fell
from 9 to 6 (33%). Mean elapsed time fell from 55.86 to 48.97 seconds, but the
second pair was slower. With two pairs and this much variance, there is no
statistically supported latency claim. This is evidence of reduced work on a
small fixture, not a production-recall evaluation of Shelfspace PR 2826.

Observed generation intervals accounted for 76.287/17.925 baseline seconds and
34.299/50.512 updated seconds. Summed tool intervals were under 0.1 seconds in
each run. These intervals include provider/transport time and cannot separate
inference from provider queues. The remaining wall time includes startup,
request preparation, rendering, and teardown; it is not labeled as model time.
See [machine-readable measurements](measurements/review-runtime.json).

Reproduce the baseline without creating a branch or worktree:

```sh
mkdir -p /tmp/daydream-runtime-baseline
git archive ebd15ca95666dd2545c70fb290991abc2e52a735 daydream | tar -x -C /tmp/daydream-runtime-baseline
uv run python scripts/measure_review_runtime.py --source /tmp/daydream-runtime-baseline --output /tmp/before
uv run python scripts/measure_review_runtime.py --source . --output /tmp/after
```

Use fresh output paths. Each run writes its prompt, result, timing events, and
trajectory. The first measured candidate preceded the host-managed output-path
hint; the second includes it. The baseline and candidate used the same installed
dependencies. Raw experiment artifacts remain under `/tmp/daydream-runtime-*`.

A deterministic regression also feeds the same 200-read workload through the
agent with ten virtual seconds per round trip. The original path uses 200 reads
and 2,000 seconds; bounded investigation plus finalization uses 48 reads and
490 seconds. The validated finding survives with an incomplete warning. This
measures deadline/work reduction, not live model quality. Real-clock hung-stream
and hung-finalizer tests verify that silent backends are bounded too.

Large changes can lose review coverage under these bounds. Incomplete warnings,
the existing uncovered-file sweep, and preserved findings make that tradeoff
visible; they do not establish equivalent recall. More representative isolated
reviews and production trajectories are needed before tuning limits upward or
claiming a broad quality or latency improvement.

## Shelfspace rollout

Shelfspace still pins `8d94fdb1`, and its analyze job has a 60-minute timeout.
Neither its workflow nor its existing uncommitted `--verbose` edit was changed.
After this PR is reviewed and merged, pin analysis and posting/validation to the
same new Daydream revision. Deploying only `ebd15ca9` leaves a single reviewer's
60-minute allowance equal to the entire job limit and is insufficient.

Keep the analyze timeout at 60 minutes with the 45-minute model pipeline budget
initially. The remaining 15 minutes cover checkout/install, host processing,
bounded cleanup, final artifact export/upload, and variability. This is headroom,
not a hard guarantee on every host operation. If runner setup regularly consumes
that margin, lower the profile budget or increase the job timeout based on those
measurements. Do not equate the per-agent timeout with the required job timeout.

For rollout diagnostics, add an external live trajectory and finalized bundle:

```sh
--trajectory "$RUNNER_TEMP/daydream-debug/trajectory.json" \
--dump-artifacts "$RUNNER_TEMP/daydream-debug/bundle"
```

Upload that diagnostic directory in a separate `if: always()` artifact step,
alongside the existing findings artifact. The current upload contains only
`findings/findings.json` and cannot diagnose a failed run. Keep posting conditional
on a valid findings artifact. A platform kill can still prevent later upload
steps, which is another reason to leave job-level headroom.
