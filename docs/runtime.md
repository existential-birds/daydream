# Review runtime contracts

Read this before changing review completion, publication, recovery, or validation.
Use [configuration](../daydream/config.py), [resolution](../daydream/run_config.py),
and `daydream --help-all` for settings rather than copying defaults into docs.

## Terminal review findings contract

Ordinary `--findings-out` uses schema version 2 with committed `terminal_result` version 1.
Local captured review coverage uses terminal version 2 and a distinct `source_id`;
commit-bound findings envelopes reject that identity instead of relabeling it.
Immutable PR exports use schema 3, retaining original PR provenance separately
from the materialized analyzed revision (see [immutable PR inputs](#immutable-pr-inputs)).
Diagram artifacts carry no review coverage; unsupported schemas are rejected.

Completion requires positive success or host no-op evidence for every planned
scope and required phase. Valid unfinished work is incomplete; unusable analysis
or invalid projection is failed. Finding count, CLI success, warning absence,
and archive status cannot establish completeness. Review coverage is frozen
independently of later fix/test/publication state. Reviewer source references
prove correspondence, not understanding, reads, or executed checks.

Publish findings and typed coverage atomically, bound to the captured run, head,
merge base, diff key, and scope inventory. Preserve admitted partial findings and
failure reasons. Incomplete/failed results can post COMMENT notices but cannot
approve or resolve stale threads. When a complete review with no findings has
nothing else to post and approval is not enabled, it posts a COMMENT saying it is
complete if the latest Daydream review on its head or an ancestor, by the same login
and carrying the Daydream footer, is such a notice. Earlier reviews are never
edited, dismissed or minimized.
Cache/resume must validate the same identities and current typed target
attribution; missing/damaged proofs restart review.
If the expected source becomes unavailable, terminal export retains validated
findings and coverage using the bound captured diff, with conservative file/body
placement instead of live-source line resolution. This does not resume review
or establish completeness.

Approval requires explicit opt-in and no blocking severity among all retained
findings, including deduplication matches. Immutable schema-3 findings without
severity block approval; new findings post as COMMENT, while an all-matched review
may post nothing. Complete-empty immutable reviews remain eligible for approval
when coverage and warning checks permit it. Ordinary schema-2 omitted-severity
behavior and classified-severity policy are unchanged.

Retained source attribution has an 8 MiB run allocation, with the existing
512-decision/256 KiB per-scope and four-reference/8192-byte per-decision bounds.
Admission, restore, and terminal validation enforce the same allocation atomically.
Findings exports reserve a separate 1 MiB for all other fields and reject raw inputs
over 9 MiB before parsing. Writer, loader, and optional-context trimming use the
same published UTF-8 field accounting; trimming preserves every retained decision.
Producer and poster versions must support this policy together. Larger plans can
still exhaust these fixed bounds and retain truthful incomplete coverage. Committed
review shards its planned targets into fixed scopes (see below) so these per-scope
bounds never limit its population.

Ordinary commit-bound capture pins repository, PR, comparison and head independently
and visibly excludes local tracked/untracked edits. Dirtiness does not block it.
Later PR advancement does not invalidate captured analysis
or its export. Immutable input performs no initial live lookup. Publication always
requires an independently trusted target. Use a unique, initially
absent output path per invocation; replacement requires a known expected run ID
because a failed replacement may preserve an older complete artifact.

## Committed-review request/result

`daydream.committed_review.run_committed_review(CommittedReviewRequest(source, base))`
reviews a captured committed diff through a native host-tool backend, Pi or Claude
([capabilities](extensions.md#invocation-scoped-host-tools)); discovery and challenge
backends without host tools are refused before source preparation. `local=True` selects the net
base-to-final-working change described below. Committed capture excludes working-tree
edits, external diff drivers and text conversion; inert materialization supplies raw
Git blobs even when the original checkout uses filters/EOL conversion.

The complete planned population is computed from the full captured patch before
any prompt bounding. Host-dispatched investigation work, followed by host-scheduled
fresh challenge, navigates it under the request's
[assignment policy](#assignment-policy-and-work):

- System instructions, which native compaction never summarizes, carry the stable
  policy, the assignment identity, pending challenge claims and a bounded
  (16 KiB) first catalog page with `status`, `total_targets`, `total_files` and
  omitted counts. Catalog and decided markers there are a dispatch-time snapshot.
- `review_source` is a charged, read-only host lookup over frozen canonical inputs:
  with `target_id`/`path` null it pages the whole catalog; `path` lists one changed
  path's targets; `target_id` returns the target's `source_options`, ranges (for a
  continuation fragment, its `source_summary` claim-line span and split-line
  `fragment_line_offset`) and its exact captured change bytes in 32 KiB pages with
  per-page and whole-content SHA-256. Cursors are opaque and bound to their query. Unknown targets and
  unchanged paths return `status: unavailable` data, not errors. It never reads
  live checkout bytes, follows links or mutates review state, and is refused once
  assignment authority closes. A single line over 50 KB, unreachable through stock
  `read`, is reachable through its exact continuation fragments.
- `review_admit` admits one bounded batch (at most 128 targets or verdicts plus
  claims, within the shared 128 KiB argument bound) through the same synchronous
  closure-first path as `structured_output`, which submits the final batch and ends
  the assignment. Each batch is validated and installed atomically (coverage
  capacity before review state), decides each target at most once and answers
  with a compact receipt. Admitted decisions and claims survive later omission,
  unavailability, retries or failure.
- Both tools report current host capacity after charging the call (remaining tool
  starts, remaining time to the original deadline and protected reserves),
  labelled as distinct from dispatch-time snapshots.

A reviewed target decision or challenge verdict with empty `evidence_refs` records
the target's assigned sources or the judged claim's evidence; models cite source IDs
only to locate claims or add evidence.
At sealing every planned target has exactly one typed decision: an admitted model
decision or a host-authored `not_reviewed` decision whose reason starts with
`unsubmitted:`. Coverage shards targets in inventory order into scopes of 12
(`committed:NNNNN`, `shard` = index); a full shard of worst-case decisions (four
references, 1024-character reasons) fits the 256 KiB scope bound. Review reserves
that typed unfinished attribution before model work and refuses, before any model
work, a plan whose minimal attribution exceeds the 8 MiB run allocation. Claim lines must fall inside a cited `source_range` or, for exact
continuation fragments, inside that fragment's summarized complete-line span; the
canonical fragment identity is unchanged.

Before provider work the native
extension reports the selected model's `contextWindow`/`maxTokens` and compaction
settings (Pi 1.4.0 defaults when unset, labelled as such). The host requires the
instructions and prompt, the largest single tool result (64 KiB), one recommended
32 KiB batch, the compaction reserve and kept context to fit, estimated at three
bytes per token, and requires output for that batch; otherwise it fails with an
actionable `native_capacity` error. Claude exposes no output limit before work, so it
has no capacity preflight; see [Claude host tools](#claude-host-tools). Larger batches up to the 128 KiB argument bound
remain admissible; an oversized `review_admit` is a correctable rejection. Daydream never writes native model registries or settings.

Review alone owns planned targets, semantic claim revisions, mandatory challenge
and derived coverage. Every publication claim receives counterevidence-seeking
challenge in a distinct native conversation against captured source. Changes to
claim, trigger, consequence, location or grounds renew that obligation; titles do
not. Retain confirmed, substantively rejected and unresolved concerns separately.
Missing/failed challenge is unfinished work. Whole-submission admission rechecks authority,
revisions and limits before installation; closure precedes cleanup and sealing.
Cancellation installs interrupted coverage and retained concerns before propagating.

An assignment remains current before native startup and between permitted live
attempts. Closing an attempt's callback endpoint does not finish that assignment;
explicit revocation, supersession or sealing cannot be undone by another native
session. Review checks the current assignment identity after prospective source
validation at the same synchronous ordering point as installation. Stopped or
crashed reviews start afresh; retained output is evidence, not resumable authority.
The original review clock includes source preparation.

Without a hosted policy, Pi's CLI resolves model/provider/thinking/trust/authentication in the original
configuration context; stock read/search tools, model-visible cwd and AGENTS context
use captured source. Daydream forwards explicit phase/environment overrides and
rejects drift in opaque configuration identities, while credentials remain live.
Claude's SDK resolves its own authentication and user settings live; its opaque identity
covers only the selected model, effort and pinned SDK.
Profile discovery/arbitration strategies cannot disable host obligations.

The run's artifact session stages the report then atomically writes the
[versioned](#captured-local-source) `committed-review.json`: claims/challenge evidence, work records, typed coverage,
run/source and complete planned-work identity. Prior/competing bytes survive publication failure.
`CommittedReviewResult.to_dict()` copies the sealed payload; `output_installed`
and `capture_state` report installation and trajectory independently. Admission,
native settlement/cleanup, completeness, output, capture and delivery are separate.
Explicit unsupported exports fail before dispatch. Immutable PR exports use the
same artifact session as described below. Ordinary CLI routing/default cutover
and external delivery remain separate integration work.

### Assignment policy and work

`CommittedReviewRequest.assignment_policy="broad"` (the default) dispatches one
discovery work over every target. `"adaptive"` partitions the inventory with
`StageInputFactory.assignment_batches` (whole parts, at most four files within the
transport's assignment byte allowance, inventory order) into concurrent local discovery work under
`anyio.CapacityLimiter(effective_fanout_concurrency(10, backend))`, each in a forked
trajectory. It decomposes only when every batch keeps a usable two-start allocation
after the cross-assignment floor; otherwise it runs one broad assignment. There is no
numeric threshold, CLI route (#1679) or default change (#1641). `settings.assignment_policy`
records `name`, `decomposed`, `planned_assignments` and `cross_assignment`
(`required` or `not_applicable`).

After decomposition, follow-up work inherits a closed local work's unfinished targets
only when that work admitted progress; work that failed native-capacity preflight
without progress is split in two instead. The host then always dispatches one
`cross_assignment` work over the original captured source and full catalog, even when
local work found nothing. It inherits every unfinished target, receives compact
`retained_conclusions` and pending claims as open concerns, and searches for defects
spanning assignments. With one broad assignment the `cross_assignment` phase is a
recorded host no-op.

Review keeps one host-owned record per work (result `work`: `work_id`, `kind` of
`discovery`/`follow_up`/`cross_assignment`/`challenge`, `parents`, `state`, `targets`,
`claims`, `superseded_by`, `admitted_targets`, `admitted_claims`, `verdicts`). Only the
owning work decides a target or judges a claim revision; catalog entries name each
undecided target's owner. New work supersedes closed work only by taking exactly all
of its unfinished targets and pending claim revisions; the old binding stays closed and
its admissions keep their attribution. Claims carry `work_id`/`revised_by` and
verdicts `work_id`; a current-revision verdict must come from a context other than that
revision's author, and each target decision is credited to one work.

Siblings admit independently. Preparation validates a delta; installation, at one
synchronous point, rechecks only touched identities (target still undecided and owned
by this work, claim revision current, retained claim/coverage capacity) and then mints
claim ordinals. A sibling conflict is a correctable rejection; closure or deadline
remains fatal.

All work draws on one [budget ledger](#budget-ledger) for the review window; each
dispatch runs on its own child ledger carved from it, so parallel siblings never share a
count. Its limits come from the review profile only: `[pipeline]
reviewer_investigation_budget_s`, `review_wall_budget_s` and `reviewer_tool_call_budget`,
plus a fixed 120 s finalization reserve. Investigation receives half of the
tool allowance and challenge the remainder. Under decomposition, cross-assignment keeps
`max(2, half // 4)` starts; local work runs until 3/8 of the review window,
cross-assignment until 1/2 and challenge for the rest. Otherwise discovery runs until
1/2. `finalization_s` is always reserved. Earlier work cannot spend later reserves;
unspent capacity carries forward. Challenge takes every pending revision not held by
live work in a fresh context. It is re-dispatched for new revisions, after a round
that resolved a verdict or changed a claim, or once after an evidence-free failure; an
unresolved verdict keeps the revision with its work, so the next round supersedes it.

Each shard's status derives from its targets and the current (non-superseded) work
responsible for them; required phases are `challenge` and `cross_assignment`.
Superseded failed work whose responsibilities were transferred and accepted elsewhere
remains evidence but does not make the review incomplete. The staged deep path keeps
its own rule ([stage-aware builders](extensions.md#stage-aware-review-builders-api-8)).

### Captured local source

[`git_ops.captured_source`](../daydream/git_ops/captured_source.py) owns accepted
before/after contents, eligible population, scope, full target inventory and native
materialization. Ordinary review mode uses this same owner; comment and commit-bound
`--findings-out` select an independently pinned committed PR subject instead, with a
visible exclusion notice when the checkout is dirty. Improve retains its separate
audit-workspace population policy. Fixing-flow migration is outside this contract.

Local capture compares the pinned merge base with final working bytes, combining
committed and later tracked edits. The index is not another change layer. Undone
edits disappear, staged files use their final working contents, and nonignored
untracked files join the eligible population. Generated Daydream artifacts stay out.
Ignored files are not implicitly expanded; a separate explicit justified selection
is required under the access policy. Dirty supporting helpers outside changed
targets are captured too. Scope exclusions bound assigned changes while eligible
supporting context remains available. Complete targets precede prompt bounding.
Local capture refuses omitted `skip-worktree` paths, including sparse-checkout
exclusions, before dispatch; materialize those working contents and rerun.

Preparation reads a confined population, materializes its accepted contents, and
rechecks actual enumeration, file identity and bytes. Detected instability retries
at most three times, then asks the operator to stop concurrent writers and rerun.
This is not an instantaneous filesystem-wide transaction. Preparation never changes
source refs, index or user files. Native regular files and canonical supporting
references read the same private source after preparation; later original edits
cannot change that subject. Private index stat/cache refresh is permitted; indexed
paths, modes, blobs, stages and semantic flags remain bound to the accepted source.
Required missing/corrupt input prevents dispatch;
later integrity failure retains admitted findings and unfinished coverage. Optional
absent/unsupported context remains explicitly unavailable, with no committed or
live-byte substitution. Work blocked by unavailable input cannot count as complete.

Capture streams physical object files and regular working leaves in 1 MiB chunks.
The private native tree is the sole working-content spool; acceptance retains
fingerprints and SHA-256 digests rather than file bodies. One bounded raw Git
reader per materialization/verification operation checks object framing and OIDs,
drains stderr, enforces a shared deadline, and joins before dispatch/finalization.
Full validation covers both separately admitted sides (up to 512 MiB of unique
blob payload), native contents/population/modes, semantic index and storage.
Output from reviewer checks that accepted `.gitignore` rules ignore stays outside
the population. Ignore files created after capture excuse nothing, so a tool's
self-ignoring cache needs an accepted rule; other additions fail validation and
name the changed path.
Selected textual reads retain their independent limit. Private creation conflicts
and replacements are refused without deleting competing bytes.

Each native submission resolves its complete prospective payload through its
existing stage-input factory. Callback-local host facts avoid repeated supporting
body reads; the resolver closes before one complete capture/canonical-input check.
Only then does Review synchronously recheck original authority/deadline and install.
Empty selectors and sibling callbacks get the same final guard. Standalone/stored
reference checks, preparation, and terminal validation remain independent; no
validation success survives a callback. Stack and stage preparation validate on a
worker thread because a full check on the shared loop starves sibling invocations'
host-tool setup timers. The writable capture is not a sandbox.

Full history still consumes private disk, up to the physical object-store limits
in [Immutable PR inputs](#immutable-pr-inputs). Python payload buffers are
fixed-size; Git may reconstruct a whole admitted blob, and concurrent readers
multiply native memory. Manifest/diff memory remains bounded by its existing
policies. See the [measurements and reproduction commands](performance/captured-source.md).

Local `AnalyzedRevision.head_sha` remains real HEAD provenance. `source_id` binds
the accepted before/after contents, modes, eligible population and exclusions;
new-side canonical references use that identity, while old-side references use the
pinned base. True rename paths, deletion old sides, explicit absence, binary and
header-only metadata remain represented. Ordinary capture admits at most 100,000
entries and 256 MiB per side; text lookup retains the existing 8 MiB limit, so
larger native files remain captured while optional textual context is unavailable.
Symlinks/submodules retain metadata without native traversal or expansion. Initialized
submodules use their final checked-out commit; uninitialized directories retain the
recorded gitlink commit. Nested dirty files are not captured. This materialization
is not an OS sandbox. Native configuration/trust/authentication
still resolve in the original project, while native cwd/read/search use the capture.

Local program results use schema 9 and terminal/coverage version 2; committed
results 7, immutable results 8 and their committed terminal version 1 stay distinct.
Results 1–3 predate bounded navigation and sharded per-target coverage, and 4–6 predate
work records; they are rejected with a restart instruction rather than converted.
Mismatched versions/identities and local-to-PR relabeling are explicitly rejected.
Private stage/cache contract 17 invalidates older cache entries. Captured review
paths reject `--start-at` and do not restore judgment shards: a complete shard
marker does not establish that its enclosing review completed. Stopped reviews
restart; retained evidence cannot restore assignment authority. Replace incompatible
inputs by starting a new review; no historical converter is provided. The optional
request/revision fields retain extension API 8. This changes local review semantics
under the current `0.y.z` policy and requires release notes; package version bumps
remain release work. Queued-scope admission and the shared canonical change follow the
[stage-aware review contract](extensions.md#stage-aware-review-builders-api-8).

Run-record capture keeps `head_sha` as real provenance and retains local `source_id`
in the task and final state. `captured_working` identifies its population policy;
legacy committed/tracked scope labels cannot stand in for it. Current run-record
readers validate this binding; earlier readers reject the new fields/population.

### Hosted operator policy

Pass `daydream.hosted_review.HostedReviewPolicy` as
`CommittedReviewRequest(..., hosted_policy=policy)` to the same
`run_committed_review` entry point. The trusted host supplies an explicit
`ResolvedProfile` (which also supplies the review limits), complete child
`environment`, isolated `agent_dir`, `private_base` for the normal artifact owner,
and an initially absent `output_dir` whose parent exists. These canonical absolute
roots must be disjoint from one another and from the source. Request `config` is
replaced by this policy; repository configuration, phase overrides and ambient
environment cannot supply hosted authority.

| Supported hosted execution | Required value |
|---|---|
| Backend | Native Pi |
| Provider / model | Operator-selected nonblank strings supported by the isolated native configuration |
| Native baseline | Official Pi 1.4.0 / Node 24.18.1 |
| Model tools | Stock `read`, `find`, `ls`, `grep`; charged host `review_source` lookup and `review_admit` batches; bounded `structured_output` admission (policy `pi-read-only-v2`) |
| Source preparation | Explicit `trusted_prepared_source=True` operator attestation |

Prepare `agent_dir` separately from target and ambient user configuration. Only
regular operator-approved native provider configuration, authentication and inert settings
belong there. The registry must contain the exact operator-selected provider/model
IDs; identifiers are preserved without trimming or normalization, and implicit
fuzzy model substitution is rejected before provider work. The hosted defaults
and preferred example are `nous` / `deepseek/deepseek-v4.1-flash-1017`; these are
configurable choices, not an allowlist. Invalid or blank selections are rejected
before source preparation. Ordinary Pi defaults are unchanged.
Resource settings/directories, system-prompt files and executable
loader environment overrides are rejected. Native authentication and credential
refresh remain native; Daydream never copies or changes persistent user credentials.
`HOME`, `PI_CODING_AGENT_DIR` and provider selection are fixed to this isolated
context. Use a trusted executable `PATH` and explicitly supply credential variables
needed by the operator's native provider configuration; do not copy an uncontrolled
ambient environment.

The native process starts in that operator context with forced-untrusted project
behavior and executable resource loading disabled. Packaged helpers bind stock
readers to captured source without promoting project, ancestor or global instruction
files into system policy. Instructions remain readable review data. Startup
checks verify the supported versions and exact tool set before provider work;
`UnsupportedHostedPolicy` identifies unavailable support without switching harnesses.
If native readiness fails during an invocation, already admitted work and partial
coverage are sealed through the normal output session before that error propagates.
Every investigation work, fresh challenge/rechallenge and retry keeps the same policy
and [budget ledger](#budget-ledger). Tool admission intersects grants before effects, and domain
submission may change only the assigned in-memory Review state.

Trusted host writers publish `review-output.md`, `committed-review.json` and
`trajectory.json` (or its partial counterpart) under `output_dir` using the same
private session and conflict-safe installation. Existing source public artifacts
are restored unchanged. Diagnostics stay under `private_base`; output destinations
do not become model write grants. Recorded policy/configuration identities omit
credentials and callback capabilities. This additive API retains the committed result
schema and extension API 8.

For ordinary hosted committed input, the preparation attestation is required **before Git worktree materialization**:
the operator must control/audit source Git configuration, checkout hooks and filters.
Later raw-byte validation cannot prove inert preparation. This API does not safely
prepare arbitrary local Git repositories. Read/search policy is not filesystem
confinement or an egress firewall; stock readers can follow symlinks. The host owns
infrastructure isolation, scoped credentials, staging, deadlines, concurrency and
cleanup. Use the immutable input contract below for inert source materialization;
other-harness/CLI integration boundaries still apply.

### Immutable PR inputs

The same embedding API accepts
`CommittedReviewRequest(..., hosted_policy=policy, immutable_input=input)` with
`daydream.immutable_review.ImmutableReviewInput.from_dict(envelope)`. The trusted
host supplies schema version 1; repository configuration cannot provide identity,
source attestation or execution authority. The canonical envelope is frozen before
model work and retained in the immutable [result version](#captured-local-source) of
`committed-review.json`. Both result versions retain the local analyzed revision,
source references, complete planned-work binding and independent execution outcomes.

```python
from daydream.committed_review import CommittedReviewRequest, run_committed_review
from daydream.immutable_review import ImmutableReviewInput

identity = ImmutableReviewInput.from_dict(captured_envelope)
result = await run_committed_review(CommittedReviewRequest(
    source=prepared_checkout,
    base=identity.to_dict()["materialization"]["requested_base_sha"],
    hosted_policy=policy,
    immutable_input=identity,
    immutable_bundle=frozen_bundle,  # None for an exact object-only input
))
```

`source` is the prepared checkout whose HEAD, raw bytes and modes are validated;
`immutable_bundle` supplies the actual frozen bytes when `bundle_sha256` is set.
The host's output directory receives `review-output.md`, `committed-review.json`,
strict `findings.json` and trajectory evidence through the same artifact session.

The envelope has two strict groups:

| Group | Binding |
|---|---|
| `provenance` | Original `repository`, positive integer `pr_number`, `head_sha`, `requested_base_sha`, resolved `comparison_base_sha`, and bounded historical `title`, `body`, `context` |
| `materialization` | `kind` (`exact` or `synthetic`), actual local head/requested-base/comparison commits, head/comparison trees, current execution `review_diff_key`, and optional `bundle_sha256` |

`requested_base_sha` is the captured base tip; `comparison_base_sha` is the resolved
comparison/merge base. They can differ. Validated commits and trees establish
source identity. Compute `review_diff_key` from `immutable_review_diff` for the
current execution; it binds the patch actually reviewed, without requiring a
saved rendered-diff checksum to survive Daydream upgrades. The inert Git owner
uses an independent Git view with the captured head's tracked attributes,
ignoring ambient/local configuration and `.git/info/attributes`.
Title is bounded to 512 UTF-8 bytes, body to
8192 and context to 65536. Unsupported versions, sizes, plans, missing objects and
inconsistent evidence are errors before dispatch, never successful empty reviews.
The [inert Git owner](../daydream/git_ops/immutable.py) owns source limits:
100,000 files and objects, 8 MiB per source blob and 256 MiB of source bytes per
tree. Bundles and physical object stores, including metadata, are bounded at
2 GiB; the aggregate reconstructed object population also stops at 2 GiB before
fsck or tree reads. Git checks each allocation against 2 GiB plus one
byte for object terminators; this is a per-allocation guard, not a total memory
limit. Historical blobs retain the object-store limit. Both comparison and head
trees are checked before diff expansion; each diff capture stops at 256 MiB of
subprocess output. Control and blob captures are bounded while reading.

Cloud supplies exact original local objects. Their commit identities, merge-base
relationship, trees, raw checkout bytes and current execution diff key must match.
Harbor supplies the same envelope with `kind="synthetic"`: exactly two synthetic
commits, a root base and its single-child head, preserving the attested comparison
and head trees. The frozen bundle bytes must match their
digest. Original commits need not be present for this projection. Trusted compiler
evidence connects original provenance to those local objects; synthetic commits
and admitted references retain their actual identities throughout review/export.

The immutable Git owner creates independent storage and materializes raw blobs
without executing target/ambient hooks, filters, configuration or checkout
resources. Subsequent source, classification and artifact queries use that inert
storage. This boundary is separate from contribution Git: commit, push and
verification hooks remain mandatory. Host infrastructure isolation still applies.

Immutable analysis and export use only captured/local evidence. They perform no
GitHub discovery or credential resolution, including identity preflight, and never
refresh a head, base, PR body or remote. PR advancement/closure cannot relabel or
invalidate captured analysis. Trusted repository/PR/provenance stays on result and
ATIF control surfaces; only the approved bounded historical context enters model
assignments. Model-facing profile serialization never carries trusted provenance.

Strict immutable findings schema 3 binds the outer original repository/PR/head to
the envelope and typed coverage to its local head/comparison/diff key. Export does
not grant publication authority. Its publisher must independently supply and
validate repository, PR, original head, requested base and comparison SHA;
`post-findings` adds `--base-sha` and `--comparison-sha` for these facts. Current
openness, grants and publication eligibility belong to the external publisher.
Valid partial findings, unresolved challenges and diagnostics survive failures;
complete-empty still requires positive complete coverage or host no-diff evidence.
An irreconcilable competing output raises the artifact owner's explicit recovery
error while preserving the competing bytes and private sealed evidence. It cannot
authorize a successful candidate or delivery.

This is an additive minor feature under the current `0.y.z` policy. Immutable
request 1, committed result 2 and findings 3 must be consumed together; ordinary
result 1/findings 2 and extension API 8 retain their existing contracts. The bounded
Harbor adapter consumes this API; ordinary CLI/bot migration (#1679), broader
Harbor/native migration (#1639), and supported-default cutover (#1641) remain
separately owned. This contract does not claim other-backend or large-plan parity.

## Budget ledger

Each budget owner (an invocation, a staged scope or stage such as
`stage:python/integration`, or committed work such as `committed:challenge`) has one
[`BudgetLedger`](../daydream/review_budget.py): an absolute `clock.monotonic()` deadline
and one charged tool-call count. `run_agent` takes it as `budget=` and has no other
time, tool-call or turn limit; no backend receives a turn limit. A relative `wall_s` is
anchored once when `run_agent` starts after admission, so queue time is not charged,
and is never re-based. A carved child allocation charges and refunds its ancestors
too. `reserved` records calls already withheld for later work; `start_floor` refuses a
dispatch whose remaining calls cannot fit its minimum work. Without host tools,
`run_agent` charges each observed tool start; a `HostToolBinding` charges the same
ledger when it admits a native start (Pi's `/admit`, Claude's PreToolUse hook) before
the start runs, and its deadline reads the ledger.
Starts refused for the protected submission slot or over the limit are observed,
never charged.

Renewal rule: compaction or summarization never renews a budget. When a retryable
provider failure leads to a retry, the failed attempt's tool calls are refunded when
the retry dispatches, because it is a fresh native conversation that must redo the
lost work; a retry stopped before dispatch refunds nothing. Elapsed time is never refunded. Retries stay bounded by the
attempt count, the absolute deadline and the outage circuit; a failure after possible
host acceptance still permits no retry. Paths that do not retry are never refunded,
including a staged schema re-dispatch and a committed challenge re-dispatch. Committed
batches admitted before a provider failure stay installed while that attempt's starts
are refunded with the rest of the attempt.

Every budget stop records its owner, dimension (`time` or `tool_calls`), limit, amount
spent, amount remaining at dispatch and calls reserved for later work, as durations and
counts only, and appears on the trace once
([observability](observability.md#budget-stops)), including a finalization turn's stop
and a dispatch refused before any agent runs. Staged review also leads its coverage
diagnostic with those numbers. A read cut short by the ledger deadline and a binding's
deadline or allocation failure are budget stops (`wall_budget_exceeded`,
`tool_call_budget_exceeded`), not `StreamStalledError`; idle, setup, drain and cleanup
timeouts remain liveness checks.

## Pi host-tool lifetime and limits

Pi keeps its stock read/search tools, offsets and pagination. Ordinary committed reviews
bind those tools, the model-visible working directory and native AGENTS context
to the captured source while native configuration and its trust policy remain
rooted in the original source.
Existing private artifact/sanctioned-input owners furnish frozen diff parts and
source context.
Every staged review backend selects assigned source IDs or supporting
path/side/LF ranges; the host constructs canonical `evidence_refs` from the full
assignment inventory and pinned regular Git blobs with the same admission code for
all backends. Exact continuation fragments retain their null source ranges and
are cited as fragments, never by a continuation range; an assigned range may only
narrow its offered range. Supporting attribution cannot satisfy assigned coverage.
Correspondence is never proof of a read, understanding or an executed check.
Pi submits through its bound host tool and corrects rejected selectors within the
conversation (below). Other backends return selectors once as final structured
output, so an invalid selection rejects the whole stage as `evidence_incomplete`
source attribution while earlier admitted findings, decisions and truthful
incomplete coverage remain; unavailable storage stays fatal for the invocation.

The existing `structured_output` terminal tool calls one invocation-local host
callback. Only a sole terminal call in a completed native generation may enter
admission. The host validates required judgments, original captured inputs, source
selectors and shared retention limits before installing the entire stage. A
correctable payload rejection returns a correlated native error naming the
field/selector, violated constraint and concrete correction. It changes no
findings, decisions, notes or retention capacity. Correction uses the same
conversation and [budget ledger](#budget-ledger). A rejected real tool start
still consumes that allowance, except a protected-slot denial (below); exhausted
capacity cannot invite another call.
Successful acceptance installs state once, returns canonical output and permits
native termination; duplicate accepted callbacks cannot admit or charge retention
again. Unbound structured calls retain their ordinary serializer behavior.

Recheck open authority and the original deadline immediately before synchronous
installation, with no await or detached thread between that check and mutation.
Closure-first denies mutation; acceptance-first retains admitted work even when
native settlement later fails. A possibly accepted callback cannot be replayed.
Confirmed host/input/lifecycle failures are terminal: retain the failed condition
and underlying cause, close affected authority, and explain that changing the
submission cannot repair it. Native continuation and before-settle submission
recovery stop, and backend/agent retry preserves the nonretryable host failure
instead of relabeling it as stream truncation. Frozen-input validation never
recaptures changed bytes under earlier judgments. Unknown callback failures are
indeterminate, even when raised as `ValueError`; never infer safe replay from a
native error flag or missing receipt.

A valid contradiction of a previously closed reviewer judgment is terminal
`evidence_incomplete`. Preserve prior findings and decisions, stop native
continuation and retry, and never invite removing the contradiction to obtain
complete coverage. Foreign or duplicate contradiction IDs remain correctable
payload faults.

Ordinary transient preacceptance retries follow the [renewal rule](#budget-ledger). Attempt
closure revokes its endpoint before cleanup; only an unaccepted, nonfatal binding
with completed cleanup may open another attempt. Explicit capability closure is
permanent. A retired endpoint cannot revoke or mutate a later attempt. The
binding owns the first terminal failure and cannot reopen after it. `/admit` and
`/submit` enforce readiness, correlation, authority, deadline and cumulative
capacity with the same recovery semantics. Frames, arguments, results,
items, request concurrency and cleanup are bounded; overflow rejects whole payloads.
Frame, readiness and admission I/O keep their short transport allowance. Terminal
source validation and submission use the original assignment deadline. Native Pi
queries its remaining allowance through the authenticated status endpoint and
subtracts the full round-trip time, without sharing clock origins or renewing
authority. Lost receipts still prohibit replay and retain incomplete coverage.
Every real native tool start counts, including failed submissions, with no separate
eight-call ceiling or hidden nested tool execution.

Committed-review bindings set `protected_submission=True`: the last start belongs to
`structured_output`. Any other start that would take it, ordinary or host tool and
including parallel-batch members, is counted in `denied_starts` and refused with
correctable `capacity_protected` without charging or closing authority; a bounded
number of denials then fails closed. `capacity()` reports both counts; the ledger
observes denied starts without charging them. Pi and Claude deny prospectively at admission,
so a denied start is still observed and reported as [overshoot](observability.md#committed-review-evidence)
that never reduces later reserves. Codex and Osprey serve no host tools and only observe starts.

A binding may also declare charged nonterminal host tools (`HostTool`). Pi registers
them in the existing extension; `/admit` charges each start, and `/call` serves
that admitted identity exactly once under the same readiness, authority, deadline
and correlation rules. Read-only tools never mutate domain state; mutating tools
install at the same synchronous boundary as acceptance, and their unexpected
failures are indeterminate. Their results carry exact frozen bytes as a separate
text block and report current capacity. `/ready` exact-matches the declared
schemas and runs the binding's native-capacity preflight before provider work.
Compaction evidence is described in [observability](observability.md); compaction
is neither completion nor settlement evidence. The read-only Pi search guidance
names only the tools it can run.

Close authority before shutdown and join/isolate callback transports and evidence
writers before freezing. Native reads use the existing process cleanup. Acceptance,
native settlement/EOF/zero exit and cleanup are separate outcomes; acceptance or a
receipt alone cannot establish successful execution. Preserve admitted findings,
usage and truthful incomplete coverage; see
[lifecycle evidence](observability.md#pi-host-tool-lifecycle-evidence).

Endpoint credentials stay outside persisted configuration/public evidence.
Bindings and resources belong to one invocation; sibling shutdown is isolated.
Enforcement works without recording or tracing. See
[supported capabilities](extensions.md#invocation-scoped-host-tools).

### Claude host tools

Claude serves the same binding through the Agent SDK on both client startup routes.
Each attempt gets a fresh in-process SDK MCP server named `daydream` with
`review_source`, `review_admit` and `structured_output` at their exact bound schemas
(`mcp__daydream__<tool>` to the model), loaded directly with results inline up to the
host result bound. Native tools are `Read`, `Grep`, `Glob` and `Bash`; strict MCP
configuration excludes other servers, and native `StructuredOutput` is not offered.
The dangerous-command and background-Bash guards stay on; the read-only guard also
allows exactly the three host names, never a wildcard (`allowed_tools` only preapproves
under the bypass permission mode). Improve's audit root, finalization, continuation,
agents, writes and persisted sessions are refused before any client starts. The
assignment's system instructions become Claude's system prompt.

- Readiness: before the first provider request, this exact server instance must have
  served `tools/list` and `get_mcp_status()` must report it connected with exactly the
  bound tool names, within the 5 s setup allowance and the ledger deadline. A missing or
  mismatched registration, or setup that times out before the ledger deadline, fails
  `native_readiness`; if the ledger deadline passes first, it fails `deadline_expired`.
  The server builds its tools from the bound schemas, so the CLI exposes no separate
  schema to compare.
- Admission: a PreToolUse hook admits and charges every hooked native start before it
  runs; the MCP handler serves only the start whose `claudecode/toolUseId` metadata
  matches, else `native_correlation`. A raising SDK hook fails open, so hooks and the
  handler never raise: rejections become denials or tool errors that reach the model.
  A start the CLI refuses before PreToolUse (an unoffered tool or its own input
  validation) is admitted when its tool result arrives, so it is charged once like any start.
- Terminal rule: the CLI runs a generation's first tool before later blocks arrive, so
  Claude cannot require a sole terminal call. Acceptance closes authority at once; later
  starts are denied without charge or failure and observed once as overshoot, and a
  sibling call admitted before acceptance is refused without failing the assignment.
- Settlement: a successful result after acceptance settles; a native error keeps the
  admission and usage and is not retried. A host fatal interrupts the conversation,
  which is not an instant stop. Every exit, including cancellation, closes the attempt
  before its own client disconnects; consumer closure also interrupts and drains that
  client first. Disconnecting closes the SDK bridge within its 5 s grace, and a retired
  attempt's hook or handler changes nothing.
- Capacity: no output limit is available before work, so no capacity preflight runs.
  The terminal context window and output limit are recorded, never estimated
  ([evidence](observability.md#claude-host-tool-lifecycle-evidence)).

## Validation and repair

The host admits one canonical test recipe before an accepted actionable fix run.
Missing/invalid commands stop before VERIFY/FIX/TEST, repair, or commit. Manifest
suggestions and model prose cannot authorize execution. Required suite declarations
refer to that command; a focused test cannot satisfy the canonical gate.

Evidence reuse requires matching command, cwd, runner, interpreter, configuration,
tree, and revision. Crossing a commit also requires strict post-commit validation.
Reuse never replaces hooks, strict post-hook checks, or the exact push receipt.
Local tests, push verification, and exact-SHA remote CI are separate outcomes.
Only completed repair jobs authorize green results or commits; preserve their
captured policy, consumed budget, authorized-tree checkpoints, and prior evidence.
See [repair jobs](extensions.md#repair-jobs-test-step).

## Artifacts and recovery

Use the run's single artifact owner/session. Live artifacts are private; finalized
`.daydream/` and `.review-output.md` belong to the source checkout even when work
runs in an operational worktree. The private run directory may also hold the
[`local` trace](observability.md#ownership-and-evidence), written until the run span
ends after finalization and never frozen, published, archived or captured.
Join writers before freezing evidence. Tracked
artifact paths cause refusal; competing bytes survive conflicts. When the source
identity remains valid, output-publication failure restores prior checkout artifacts.
Optional archive, evaluation, index,
or upload failure preserves completed review outputs and emits safe diagnostics.
An unavailable expected checkout is distinct from a foreign path, symlink,
wrong filesystem type, or replacement directory. Publication and restoration
require the original source directory identity, independently of any surviving
operational worktree. When restoration is unsafe, retain canonical baselines,
frozen bytes, transaction journals and conflict bytes; never recreate the source
or install into its replacement. Recovery requires the transaction's captured
source identity; pending older journals without that proof are preserved and
refused rather than adopted or migrated.
`WorkspaceOwnership` holds checkout and Git descriptors from workspace setup
through cleanup. Its source capability owns strict journal proof and anchored
publication/restoration operations; its code capability guards live reads and
Git routing around each operation. Consumers request these operations and retain
their own refusal, frozen-evidence fallback, or warning response. An artifact
session borrows this authority; closing that session does not end workspace
ownership. Standalone analysis holds a temporary code capability with the same
directory checks, admitting plain directories without requiring Git.
Ephemeral cleanup verifies the original source and holds its Git common directory
through subprocess dispatch; inherited Git routing cannot select replacement
metadata. Unsafe cleanup retains the operational worktree and warns without
replacing a primary failure.

Completed fork trajectories stay on disk, not in RAM. The run's recorder registry
keeps one reference per completed child: its trajectory id, logical path, the
private file actually written, and that file's size and SHA-256. Run snapshots keep
the root and still-running forks as prepared bytes, because every flush rewrites
their `.partial` paths. Every consumer reads documents through one checked read:
prepared bytes come from memory, and a reference is re-read without following links
and must match its size and digest; identity-checked reads also verify
trajectory/session identity. Retaining a snapshot, which also happens from the
signal handler, checks only reference metadata and performs no I/O. Freeze checks
each reference in place and against the frozen copy's manifest. A missing or changed
child is dropped from that snapshot with a warning, and its file is left in place;
live run info drops it from its view the same way. Realistic causes are an older
retained partial snapshot whose child file a later fork with the same legacy
descriptor overwrote, or an edit in private state. Intact bytes with the wrong
identity remain a fatal freeze error. A run that lost a child still installs its
outputs and archive, but optional dataset capture skips it rather than record
incomplete trajectory evidence. Consumers after freeze read references from the
frozen copy, where a mismatch is an integrity failure.

Finalization holds one trajectory document at a time. After freeze, one summary
pass (`summarize_run`) reads each selected document once through its
identity-checked read and keeps only its metadata without `steps` plus compact
step facts; timing is reduced once from that summary, and evaluation and the
archive manifest share it. Nothing is persisted or recorded live for it.
Malformed step content fails only the evaluation that reads it. Dataset capture
validates, encodes and stages each checked document outside the record store's
lock, refusing an oversized record at the first document that crosses the size
cap; the lock covers only creating and locking that stage, interrupted-write
recovery, the comparison with an existing shard and publication, and recovery and
readers leave a live writer's stage alone.
The record bytes equal the in-memory canonical record. Live run info and
`daydream summarize` fold documents into the run-info table one at a time. The
recorder serializes each document once, straight from the ATIF model to bytes,
with ASCII escapes and absent fields omitted. Float spelling may differ from
older documents (`0.000015` rather than `1.5e-05`); decoded values are
identical. Serialization is not streamed, because every prepared document is
retained or written whole. [The finalization measurement](../scripts/measure_finalization_memory.py)
reports each stage's peak memory and runtime and the archive's bytes read.

Finalization reports local JSONL capture, configured upload, archive/evaluation,
and public installation separately. Later failure cannot revoke a successful
capture receipt or imply successful public installation. Valid frozen analysis
can be archived with explicitly unavailable live-source quality (null metrics
and an unavailable diagnostic); a valid operational workspace can still supply
quality independently of public-source availability when its current Git routing
matches the captured worktree and common directories. Genuine evaluation or
integrity failures continue to abort atomic archive installation. Body failures
and cancellation remain primary when finalization also fails.

Pass generated model inputs through exact sanctioned paths or bounded inline
contents; private storage is discovery isolation, not an OS sandbox. Never broaden
read roots or reconstruct public paths. Explicit external `--trajectory` supports
live atomic updates; other explicit exports wait for finalization. Diagnostic dumps
preserve assembled bytes, including sensitive/binary content; publication scanning
is independent. Archive indexes require the current complete schema or an empty
index; preserve unsupported indexes and use a fresh directory.

## Diagrams

Models supply evidence-bearing JSON; the host grounds it against immutable head
source and renders Mermaid. Forcing a kind changes eligibility, never grounding.
Flowchart grounding proves statements/calls exist, not arrow execution order.
Review diagram failures warn and retain findings; diagram-only failures exit 1.
Custom flow ordering remains authoritative. Read [diagram flow](extensions.md#diagram)
and [grounding](../daydream/deep/diagram_grounding/) before changing this boundary.
