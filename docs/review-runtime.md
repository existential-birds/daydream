# Review runtime: incident, changes, and measurements

The runtime bounds introduced in PR #1282 preserve partial reviews and reserve
time for publication. This follow-up makes finalization a separate serialization
task, retains its required evidence, and fixes compatible GitHub metadata handling.
It does not increase the timeouts. The incidents below distinguish observed
behavior from hypotheses; prompt wiring alone does not prove model convergence.

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

## Follow-up incident: September 21, 2026

[Shelfspace run 35621233648, PR 2826](https://github.com/shelfspace-app/shelfspace-mono/actions/runs/35621233648/job/106404625753?pr=2826)
installed Daydream `6ef81035674eb099594b535cbe8ed61ee86b6ee0` (merged PR #1282).
Local logs were available as `/tmp/daydream-run-35621233648.log` and
`/tmp/daydream-review-35621233648.log`.

Confirmed from those logs and the installed revision:

- Review and merge finished in roughly 14½ minutes. Neither the 45-minute global
  model budget nor the 60-minute Actions timeout was exhausted.
- Exploration reached tool limits. Alternatives and all three review stacks
  reached investigation limits. Generic/react finalizers also timed out.
- The structural finalizer returned zero findings after **14,157 completion
  tokens**. Reviewers repeatedly reopened disproved candidates and explicitly
  speculated about a planted bug or an expected evaluation finding.
- Finalization reused the discovery prompt, its original allowance and fresh-read
  instructions. Pi ignored `max_turns`; the host zero-tool guard stopped tool
  events but did not remove Pi's native tools. Shared backend mutation would
  also have affected concurrent sibling calls, so it is not a valid remedy.
- Merge succeeded. Findings export then failed with
  `invalid PR row: malformed head repository slug`.

The production prompt's SlopCodeBench framing could have encouraged benchmark
assumptions, but that causal connection is **not proven**. The framing has been
removed; concrete consequence and repository-convention requirements remain.

GitHub CLI versions affected by [the upstream projection bug](https://github.com/cli/cli/commit/5ed8cf0faa80caff3eeb2edc6d32ef539e477a48)
could return `nameWithOwner: ""` while providing valid repository `name` and owner
`login`. The fix shipped in [gh v2.89.0](https://github.com/cli/cli/releases/tag/v2.89.0).
Daydream now treats absent/exactly empty slugs as unavailable and derives a
validated owner/name pair. Null, wrong types, malformed nonempty slugs and invalid
fallback components still fail. Contradictory populated identities fail closed;
case-only differences agree. Deleted-fork fallback and reviewed-head validation
remain in place. Errors distinguish missing, malformed and contradictory metadata
without logging whole API responses. Both PR lookup paths and findings export
have regressions using the affected response shape. The incident runner's gh
version was not captured, so this exact explanation for its export failure
remains a hypothesis.

## Original runtime causes

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

Automatic uncovered-file sweeps and reviewer-read coverage accounting have been removed.
Review findings, failed-stack warnings, and bounded recovery remain; diagram citations
are validated directly against repository source. Evaluation retains cost, timing,
finding quality, duplicates, and citation-location checks.

The host enforces investigation time and tool-start limits across backends,
independently of native turn-limit support. Existing 60-minute hard ceilings
remain as outer safeguards. Earlier limits now bound useful work:

| Role | Investigation | Tool starts | Reserved finalization |
|---|---:|---:|---:|
| Exploration specialist | 300 s | 16 | 120 s |
| Intent | 120 s | 12 | 60 s |
| Alternatives | 300 s | 24 | 90 s |
| Each language/generic/structural reviewer | 480 s | 48 | 120 s |
| Arbiter | 120 s | 16 | 60 s |
| Suppression/supervision | 120 s | 12 | 60 s |
| Merge | 180 s | 16 | 60 s |

Diff exploration has a 450-second outer timeout, allowing investigation,
finalization, and stream cleanup. Its former 120/30-second limits interrupted
Pi dependency mapping in the September 22 Sirona run while it was still
assembling results, then expired again during finalization. The 16-tool limit
and static shortcut for modest changes remain in place. Increasing the general
invocation wall budget alone does not override these shorter phase limits.

A whole-review model deadline defaults to 2,700 seconds for modest diffs. Diffs
over 1,000, 5,000, and 10,000 lines (or 64, 256, and 512 KiB) scale the default
deadline and per-role investigation, finalization, and tool-call limits by 2×,
4×, and 6× respectively. The 4× and 6× tiers enable the existing deep-review
sharder unless explicitly disabled. An explicit review profile retains its
configured whole-review deadline. The deadline includes queueing,
retries, exploration, intent, discovery, adjudication, merge, and optional diagram
requests. Discovery reserves five minutes per workload multiplier for synthesis.
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

At an investigation cutoff, the first fallback is an already complete,
schema-valid response or the latest schema-valid assistant-turn checkpoint.
Otherwise a fresh invocation receives a **standalone finalization contract**:
explicit caller task/output semantics, assigned files, schema (or plain-text
semantics), authoritative supplied context and completed source evidence.
Discovery strategies, stale allowances, fresh-read demands, research/test
instructions and speculative notes are not copied into it. Every review-limited
phase supplies this contract, including exploration and plain-text intent.

The request keeps the original incomplete/budget-stop reason. Its deadline is
clamped to remaining absolute time; investigation displays its actual clamped
allowance. If time is shorter than the reserve, no investigative invocation is
dispatched. If no time remains, no model invocation starts. Retries consume the
same deadlines and clear failed-attempt evidence. Closing a stopped invocation
does not cancel unrelated siblings sharing the backend.

Finalization controls are invocation-local, with effective reasoning and typed
native options recorded in request events and exported span attributes:

| Backend | Finalization controls | Verification / limit |
|---|---|---|
| Pi | `--no-tools`, replacement serialization system prompt, thinking capped at `low` | Installed 0.85.1 help confirms built-in and extension tools are removed. `max_turns` is unsupported; the absolute deadline bounds generation. |
| Claude | `tools=[]` → native `--tools ''`, strict empty MCP configuration, existing guards plus serializer-only guard, `low` effort | SDK 0.2.152 and its bundled CLI 2.1.259 completed a local protocol fixture with only `StructuredOutput` available. `allowed_tools=[]` alone is insufficient under bypass permissions. |
| Codex | Invocation-local `-c model_reasoning_effort=...` capped at `low`; host zero-tool guard retained | Installed 0.155.1 supports the override. No native tool-disable control is claimed; read-only sandbox access still permits tools. |
| Osprey/custom | Existing execution interface plus host deadline/zero-tool guard | Native finalization controls are unsupported unless the backend explicitly advertises the optional capability. |

Explicitly lower supported settings are preserved (`off`/`minimal` for Pi,
`none`/`minimal` for Codex; Claude's lowest supported effort is `low`). Opaque native
ambient reasoning defaults are not introspected. The host tool guard is an
event-level stop, not a pre-execution security boundary. A real-provider Claude
smoke was blocked by expired OAuth; the local protocol test establishes native
schema/tool compatibility, not model convergence or provider behavior.

The evidence capsule reserves up to 24 KB for declared task/context, 24 KB for
captured sanctioned inputs (12 KB per input), and 48 KB for completed tool
results. Callers prioritize structural diff/intent, adjudication targets and
merge records before advisory artifacts. Exact-path inputs originally contain
paths and identity, not text: bounded bytes are read through the existing
no-follow, UTF-8-validating capture boundary, with full input identity/hash checks
and call-mode/backend revalidation. Standalone phase calls use the same boundary.
No arbitrary path is extracted from prompt prose.

Completed tool blocks preserve call/result association, errors, exit status,
cancellation and native/host truncation. Identical reads deduplicate; later
searches cannot evict already retained foundational evidence. Unique late results
can still be omitted once the bound fills, with explicit omission markers. Pending
starts and retained block counts are bounded too. Missing evidence cannot establish
clean coverage. Partial JSON, reasoning and unverified notes never become findings.

Shared stopping guidance applies after both default and custom strategies: no
defect is guaranteed, a substantiated empty result succeeds, rejected candidates
stay rejected unless new evidence changes their premise, and every tool invocation
(including each parallel-batch member) counts. Exploration produces mappings rather
than another review. Completed source evidence from the same logical review can
satisfy finalizer grounding; a path or speculation cannot. Maintainability findings
need concrete consequences or established conventions, with no arbitrary extraction
threshold. Config-flow tracing guides investigation, not extra schema-breaking prose.

The full hook suite also exposed a signal-handling race: an interrupt delivered
inside an AnyIO task-group body can arrive at the CLI wrapped in an exception
group. Pure interrupt groups now take the normal shutdown/exit-130 path; mixed
groups still propagate their other failures. The existing real-signal recorder
test and deterministic grouped-interrupt regressions cover this behavior.

Partial stack records carry `incomplete: true`, survive adjudication rewrites
and merge resume, and retain their stable finding identities. Warnings still
prevent approval or resolution of
absent prior findings. Invalid or unfinished JSON is never checkpointed as a
finding. Checkpoints are invocation-local until the phase persists them; this is
not durable recovery from an operating-system kill during investigation.

Small, whole intent/summary/file-map artifacts are shared inline (4 KiB each,
with pointer fallback for larger/missing/invalid files). Direct source validation remains in place.
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

Large changes can lose review coverage under these bounds. Incomplete warnings
and preserved findings make that tradeoff visible; they do not establish equivalent
recall. More representative isolated
reviews and production trajectories are needed before tuning limits upward or
claiming a broad quality or latency improvement.

## Follow-up measurements

The extended [measurement script](../scripts/measure_review_runtime.py) uses
isolated temporary repositories and never posts reviews. Alongside the original
two-defect fixture, a nonempty pagination refactor preserves tenant filtering,
archive exclusion, ordering, exclusive cursors and limits. The clean fixture's
before/after implementations agreed on 864 combinations of ordering, tenant,
cursor and limit. This fixture is larger than the two-function defect example,
but is still a small synthetic review.

On September 21, baseline `6ef81035` and the candidate used Pi 0.85.1,
OpenRouter `deepseek/deepseek-v4.1-flash`, explicit `high` investigation thinking
and matching limits. Finalization's lower effort is part of the treatment. Each
row is one matched pair; some independent requests overlapped, and provider
queue/inference time cannot be separated. Values are **baseline → candidate**.

| Regime / fixture | Elapsed seconds | Tool starts | Prompt tokens | Completion tokens | Result |
|---|---:|---:|---:|---:|---|
| Default / clean | 14.409 → 72.816 | 8 → 5 | 26,331 → 25,956 | 1,159 → 1,255 | Both valid complete empty results |
| Default / two defects | 70.933 → 39.121 | 4 → 3 | 17,389 → 17,116 | 927 → 920 | Both retain 2/2 grounded defects, no extras |
| Two-tool / clean | 42.274 → 28.554 | 2 → 2 | 11,641 → 11,442 | 315 → 428 | Both valid complete empty results |
| Two-tool / two defects | 7.148 → 34.560 | 2 → 2 | 11,266 → 10,895 | 881 → 780 | Both retain 2/2 grounded defects, no extras |
| Reserve only / clean | 20.169 → 7.419 | 0 → 0 | 5,733 → 1,992 | 393 → 808 | Both valid empty results with `not_reviewed` |
| Reserve only / two defects | 60.045 → 38.946 | 0 → 0 | unavailable → 1,824 | unavailable → 1,109 | Baseline finalizer times out; candidate returns both diff-grounded defects |

Default limits were 480 s investigation / 120 s reserve / 48 tools; the two-tool
runs used 120 s / 60 s / 2 tools. None hit a budget stop: both models often stopped
within the tighter allowance. To exercise finalization independently, reserve-only
runs supplied zero investigation time and 60 s finalization. All four retained
`wall_budget_exceeded`. The candidate's two finalizers returned strict schema-valid
results with effective `low` thinking and `no_tools=true`; the baseline completed
one and timed out on one. No token metric was emitted for the timed-out baseline,
so its usage is unknown, not zero.

The reserve-only defect fixture supplies both changed functions and their contracts
inline. Its returned findings cite that code, but no source tools ran and complete
coverage is not established. Production preserves incomplete warnings through
merge/publication. Adjudication callers remain
conservative: a budget stop can cause them to discard even a valid finalized result,
retaining earlier findings and incomplete status instead of treating the phase as
complete.

These results show working output and bounded serialization on these fixtures.
Latency is mixed, and the sample supports **no broad latency or recall improvement
claim**. Prompt-string tests verify wiring, not convergence. See the
[machine-readable follow-up measurements](measurements/review-runtime-followup.json)
for limits, prompt digests, effective options, finalizer outcomes and adjudicated
finding retention. The script now records source digests too; that field was added
after this sample. Raw trajectories and fixtures remain in the listed `/tmp`
directories, outside the repository.

## Diagnostics lifecycle and later Shelfspace rollout

Daydream already exposes `--trajectory PATH` and `--dump-artifacts DIRECTORY`.
A regression now drives a failure during findings export after merge and verifies
that the external trajectory and archived diagnostic bundle survive the nonzero
CLI exit. No new exporter is introduced. One existing distinction matters: the
manifest's `pipeline_status` can say `succeeded` when the pipeline completed but
later findings export failed. Use CLI exit status and validated findings as the
publication gates, not that manifest field alone. A process/platform kill can
still prevent finalization or later upload steps.

`--dump-artifacts` copies exact assembled bundle bytes by default, including
credential-shaped strings and binary files, without an archive scan. For shared
diagnostics, explicitly add `--sanitize-dump-artifacts`. That option scans the
bundle, sanitizes a separate private copy when blocking findings occur, and
rescans before release. Accepted dumps preserve bundle layout and manifest
session ID; sanitized derivatives also include `dump-sanitization.json`. The
original archive and review evidence retain their bytes, subject to existing
upstream trajectory redaction.

With explicit sanitization, remaining blocking findings, scanner errors or
sanitizer failures withhold the dump without discarding completed review outputs
or changing the review exit status. Copy/I/O and archive-integrity failures keep
their fatal finalization behavior. Dumps merge into the destination, preserving
unrelated files; sanitization refusal preserves the prior destination.

The archive scanner covers limited credential patterns and is no guarantee of
safe sharing. Direct run-bundle uploads still scan by default. The operator's
CLI/API-only `--allow-archive-secrets` override permits credential findings only
for those direct uploads; scanner failures always refuse upload. It does not
change explicit dump sanitization, hydration or adjudication publication.

The Shelfspace upload job currently assumes `bundle/manifest.json` exists.
Sanitized bundles satisfy that contract after a Daydream pin update, but an
irreducible refusal still leaves that separate job without a bundle to upload.
Handling a missing bundle is a separate workflow change. This Daydream change
does not update Shelfspace's workflow or pin, or change the direct Hugging Face
destination configuration.

Shelfspace was not modified in this session; its PR #2826 must be rolled out
separately after these Daydream changes become available. Pin analysis and
posting/validation to the **same fixed Daydream revision**. Keep the 60-minute
analyze timeout and 45-minute model budget initially; setup, cleanup and exports
consume the remaining headroom, which is not an unconditional host-operation
runtime guarantee.

For that later rollout:

1. Log `gh --version` before analysis so metadata compatibility is diagnosable.
2. Export trajectory and bundle under a dedicated runner temporary directory:

   ```sh
   --trajectory "$RUNNER_TEMP/daydream-debug/trajectory.json" \
   --dump-artifacts "$RUNNER_TEMP/daydream-debug/bundle" \
   --sanitize-dump-artifacts
   ```

3. Upload `daydream-debug/` in a separate artifact step with `if: always()`,
   including when findings export fails. Preserve the existing findings artifact.
4. Keep credentials, provider auth and runtime directories outside that upload.
   Do not upload all of `$RUNNER_TEMP`, a home directory, or CLI credential state.
5. Preserve strict findings schema/head validation and existing posting gates.
   Diagnostics availability does not authorize posting, approval or resolving
   absent prior findings from an incomplete review.

## Third incident: export rejection and repeated investigation

[Shelfspace run 35633835751, PR 2826](https://github.com/shelfspace-app/shelfspace-mono/actions/runs/35633835751/job/106446456074?pr=2826)
installed `bb7efbc7956b80ab19c8858dc659cced5c9f0de3`. The reviewed head was
`406a4ca638533fa6ff560373ce1d9631c902284e`: ten files, 503 additions and nine
deletions, mostly a workflow, its tests and documentation. Analysis ran from
17:44:43 to 17:56:39 UTC (about twelve minutes). The Actions log contains 160
tool-start headers: 91 reads, 53 shell commands, 12 greps, three finds and one
listing. These are logged starts, not necessarily completed tool executions.

Six cutoffs occurred: three during exploration, then alternatives' wall limit,
structure's tool limit and React's wall limit. Merge still completed and findings
were prepared. The fatal exit came afterward: diagnostic publication rejected
four blocking `url_credential` matches in the generic/React trajectories. A fifth
`env_var` match in the diff was advisory. Finalization rolled back publication;
the findings upload was skipped and the diagnostics directory was absent. The
GitHub artifacts API returned no artifacts, so the four matched values cannot
be recovered from this run.

The reviewed Makefile contains a PostgreSQL connection template whose username
and host are Make-variable expansions and whose password is the literal `***`.
Both reviewers read that file. The existing scanner reproduced a blocking match
on that non-secret template, and the live trajectory redactor left it unchanged.
This is a plausible explanation for the incident matches, not proof of their
exact contents. Separate synthetic regressions reproduce a broader mismatch:
the live redactor did not cover all credential URL forms that the publication
scanner rejected.

Budget enforcement itself worked. Work allocation and investigation scope were
poorly matched to the change:

- The sole detected language absorbed root Makefile/JSON and standalone Node
  scripts into React: seven assigned files versus three generic files.
- All exploration specialists received the expanded affected-file list. The
  test mapper treated imported/context files as additional changed targets.
- Reviewers repeatedly reconsidered dismissed concerns, read sibling stacks,
  and attempted dependency installation after missing-package test failures.
- The structural diff exceeded its inline allowance and dropped five of ten
  blocks. Bounded finalization evidence also omitted late reads, so a long
  investigation did not guarantee recoverable complete coverage.

The fixes align live redaction with publication checks while preserving rejection
of real credentials, correct ownership of root infrastructure, and separate
changed exploration targets from known context. Oversized exploration diffs receive
bounded per-file changed-line excerpts with explicit omissions; these advisory
excerpts do not establish review coverage. This gives mapping agents without a
shell tool the changed behavior they previously had to infer from whole files.
Changed tests, documentation, configuration and build manifests remain mapping
evidence rather than additional test-mapper source targets. One confirmed covering
test per source completes the mapping task; a no-source change skips that specialist.
Review prompts define a finite pass, allow extra reads only for concrete candidates,
and prohibit dependency installation or environment repair during review.

For nontrivial default-policy changes with at most three non-test source files
and a diff at most 64 KiB, exploration uses the static impact map and bounded repository
guidance directly. It makes no model mapping calls. Larger changes and custom
exploration strategies retain the specialist path. All changed files still
reach the primary reviewers; the optimization removes advisory discovery work,
not correctness review. Exploration caches include strategy identity so cached
default results cannot suppress a custom exploration policy.

When structural review uses the built-in structural prompt builder, it also owns
the packaged default design alternatives check. The host omits the independent
default alternatives invocation and writes its empty compatibility artifact.
Custom alternatives strategies, custom structural prompt builders (including
wrappers), and runs without structural review retain their independent pass.
Structural failures
retain incomplete-review warnings; resume preserves prior alternatives and the
structural design responsibility. This removes a duplicated investigation without
raising limits or dropping design review.

An isolated replay of the same head/base with Pi 0.86.1 and the same OpenRouter
model confirmed that removing advisory stages alone was insufficient: all three
primary reviewers still reached their wall limits. Their effective user prompts
contained the stopping rules, but Pi's system addendum retained unconditional
search-first advice and a fixed, inaccurate tool allowance. The model repeatedly
reopened dismissed candidates and speculated about planted bugs. This is direct
evidence of non-convergence, not proof that prompt priority was its only cause.
Pi now receives the computed investigation allowance and stopping contract in
its system addendum for bounded review calls, including repository-scoped search
and explicit closure of resolved candidates. The fixed tool-count hint and
unconditional search-first instruction are removed. Fix calls and explicit
reasoning-effort settings retain their existing behavior.

That replay nevertheless completed publication with exit status zero: findings,
trajectory and diagnostic bundle were written, the scan reported zero blocking
matches and one advisory diff match, and incomplete-review warnings survived.

Existing wall/tool limits, finalization reserves and honest incomplete-coverage
warnings remain in force.
Prompt and routing tests establish these contracts; they do not by themselves
establish model convergence or equivalent recall. Evidence retention remains
bounded, and this change does not promise that every review completes before a
limit.


### Pi discovery and prompt transport

Pi discovery uses the actual live session diff path for every diff size. Intent,
alternatives, per-stack, structural and adjudication prompts
retain the admitted path instead of embedding diff content. Reviewers inspect the
relevant diff sections and source checkout with their read-only tools. Small Pi
reviews use this same path; the former tools-disabled finite evidence packet and
primary-owned structural delegation are no longer dispatched. Structural review
runs explicitly. A per-stack rerun replaces stale structural outputs; resumes
load records for the current structural reviewer assignment.

A durable diff reference has a separate 128 MiB streaming-validation resource
limit. It does not consume captured prompt-input allowances (1 MiB per exact file,
4 MiB combined), which still apply to other inputs. Admission validates a regular,
non-symlink UTF-8 file, retains identity and hash, and binds it to backend, cwd and
read-only mode. Each retry revalidates the reference, including its content hash.
Use the supplied session path: reconstructing `<repo>/.daydream/diff.patch` can
point into the detached public tree. INLINE transports keep their existing
bounded capture and isolation behavior.

Recovery finalization remains a tools-disabled serialization step. Pointer-only
diffs are excluded from sanctioned content capture; reference metadata, established
findings and bounded evidence from completed investigation reads remain available.
An uninvestigated file stays incomplete. Finalization must not fetch new evidence.

Normal Pi execution writes the logical prompt and schema appendix to a private,
invocation-local UTF-8 temporary file and passes its absolute `@file` reference.
Dynamic review system instructions use the separate system-prompt path argument.
Both files are closed before spawning, retained until child teardown and removed
on success, write/spawn failure, cancellation or generator close. Tools-disabled
execution continues to send the logical prompt over stdin. `RequestEvent` retains
the useful logical user and system prompts, independent of transport filenames.

The benchmark-pinned Pi 0.84.3 parser and file processor were checked directly
with a greater-than-3-MB UTF-8 prompt: `@file` becomes a `<file name="...">` wrapper
around the file contents. The effective model input therefore includes Pi's
wrapper; it is not byte-identical to the logical prompt. Pi also resolves existing
system-prompt file paths directly, without an `@` prefix.

Deterministic runner tests cover small and 3,690,129-byte single-file diff blocks,
live-session admission, child-visible references, and findings through merge and
report. Separate real protocol-child tests exercise large
arbitrary prompt transport and concurrent cancellation. These checks establish
workflow and transport behavior without claiming live-model recall or convergence.

## Latency profiles and the per-profile report

Deep runs choose wonder and arbiter spend through a named latency profile
(`fast`, `balanced`, `forensic`; `balanced` by default). A route is the
monotone max of the profile's floor and the diff's mandatory risk floors, so a
security-, concurrency-, persistence-, interface-, or migration-shaped diff is
never routed cheap, and `forensic` is defined as today's behaviour. Each run
persists its decision to `.daydream/deep/latency-routing.json`.

A fixed, hand-built corpus under `tests/fixtures/latency_profiles/` and one
documented command compare the profiles:

```sh
uv run python -m daydream.eval.latency_report --corpus tests/fixtures/latency_profiles/manifest.json
```

The report prints JSON with a section per profile. Each metric means exactly
one thing:

- `runs`: how many corpus runs were observed under that profile.
- `phase_latency_seconds.<phase>.p50` / `.p90`: nearest-rank percentiles of the
  per-run `wall_clock_seconds` for the wonder (`alternatives`) and arbiter
  phases. Real runs carry no `arbiter` timing bucket -- every arbiter call runs
  inside the `deep` phase -- so the arbiter number is read from
  `timing.phase_timings.deep`, which aggregates the whole deep phase (arbiter,
  suppression, supervision, and review). A legacy `arbiter` bucket
  is honoured only for hand-authored corpora that predate the pipeline keying.
- `high_severity_recall`: golden `(file, line)` pairs found among shipped items
  at `high` severity, over the corpus's golden high-severity pairs.
- `false_positive_rate`: shipped items matching no golden pair, over all shipped
  items.
- `contested.kept` / `.dropped`: arbiter targets the routing record names as
  contested that did or did not survive to the shipped set.
- `shipped_by_lens`: shipped items attributed by the merge schema's `lens`
  field; the top-level `citations` block separately reports the corroborating
  `(Sources: ...)` prose and its coverage.
- `calibration.surface_signals`: the committed trigger lists the route uses,
  restated so the report is the evidence used to revise them.

### Cold and warm review reuse

The same corpus carries an explicit `sample_group` on its reuse-measurement
cases, so the report's `review_runtime` block keeps a cold fix and a warm reuse
loop apart instead of averaging them together. Its `header` names the corpus
(`latency-profiles` here), the sample size observed in each group (`n=1` cold and
`n=1` warm in the committed corpus), the per-series p50/p90 -- using the same
nearest-rank helper as the per-profile block -- and the stated target,
`Target: 5-15 min for a small follow-up fix`. A case without `sample_group`
reports as `ungrouped`, so the existing comparison cases still render as before.

To refresh the measurement, record a cold run and a warm rerun into the same
corpus, then rerun the command above. The numbers are only meaningful when both
runs were measured on the org's review VM, never on an orchestrator host, and the
report makes no claim beyond the cases it names.

The corpus is small and fixed, so the report makes no statistical claim beyond
those exact runs: it states the observed subset and its coverage rather than
extrapolating. `analyze_findings.per_lens` remains raw pre-merge attribution and
is unaffected by this shipped-lens report.
