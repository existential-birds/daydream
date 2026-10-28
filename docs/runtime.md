# Review runtime contracts

Read this before changing review completion, publication, recovery, or validation.
Use [configuration](../daydream/config.py), [resolution](../daydream/run_config.py),
and `daydream --help-all` for settings rather than copying defaults into docs.

## Terminal review findings contract

Ordinary `--findings-out` uses schema version 2 with `terminal_result` version 1.
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
approve or resolve stale threads. Cache/resume must validate the same identities
and current typed target attribution; missing/damaged proofs restart review.
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
still exhaust these fixed bounds and retain truthful incomplete coverage.

Ordinary commit-bound capture requires a clean checkout and an initially matching
live publication target. Later PR advancement does not invalidate captured analysis
or its export. Immutable input performs no initial live lookup. Publication always
requires an independently trusted target. Use a unique, initially
absent output path per invocation; replacement requires a known expected run ID
because a failed replacement may preserve an older complete artifact.

## Committed-review request/result

`daydream.committed_review.run_committed_review(CommittedReviewRequest(source, base))`
reviews a captured committed diff through native Pi. It excludes working-tree edits,
external diff drivers and text conversion; native regular files must match raw Git
blobs, including under checkout filters/EOL conversion. Plans over 512 assignment
parts are refused before dispatch.

Review alone owns planned targets, semantic claim revisions, mandatory challenge
and derived coverage. Every publication claim receives counterevidence-seeking
challenge in a distinct native conversation against captured source. Changes to
claim, trigger, consequence, location or grounds renew that obligation; titles do
not. Retain confirmed, substantively rejected and unresolved concerns separately.
Missing/failed challenge is unfinished work. Discovery cannot consume protected
challenge/finalization allowances. Whole-submission admission rechecks authority,
revisions and limits before installation; closure precedes cleanup and sealing.
Cancellation installs interrupted coverage and retained concerns before propagating.

Without a hosted policy, Pi's CLI resolves model/provider/thinking/trust/authentication in the original
configuration context; stock read/search tools, model-visible cwd and AGENTS context
use captured source. Daydream forwards explicit phase/environment overrides and
rejects drift in opaque configuration identities, while credentials remain live.
Profile discovery/arbitration strategies cannot disable host obligations.

The run's artifact session stages the report then atomically writes schema-1
`committed-review.json`: claims/challenge evidence, typed coverage, run/source and
complete planned-work identity. Prior/competing bytes survive publication failure.
`CommittedReviewResult.to_dict()` copies the sealed payload; `output_installed`
and `capture_state` report installation and trajectory independently. Admission,
native settlement/cleanup, completeness, output, capture and delivery are separate.
Explicit unsupported exports fail before dispatch. Immutable PR exports use the
same artifact session as described below. Ordinary CLI routing/default cutover
and external delivery remain separate integration work.

### Hosted operator policy

Pass `daydream.hosted_review.HostedReviewPolicy` as
`CommittedReviewRequest(..., hosted_policy=policy)` to the same
`run_committed_review` entry point. The trusted host supplies an explicit
`ResolvedProfile`, `ReviewLimits`, complete child `environment`, isolated
`agent_dir`, `private_base` for the normal artifact owner, and an initially absent
`output_dir` whose parent exists. These canonical absolute roots must be disjoint
from one another and from the source. Request `config` and `limits` are replaced
by this policy; repository configuration, phase overrides and ambient environment
cannot supply hosted authority.

| Supported hosted execution | Required value |
|---|---|
| Backend | Native Pi |
| Provider / model | Operator-selected nonblank strings supported by the isolated native configuration |
| Native baseline | Official Pi 1.4.0 / Node 24.18.1 |
| Model tools | Stock `read`, `find`, `ls`, `grep`; bounded `structured_output` admission |
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
Every discovery, fresh challenge/rechallenge and retry keeps the same policy and
cumulative allowances. Tool admission intersects grants before effects, and domain
submission may change only the assigned in-memory Review state.

Trusted host writers publish `review-output.md`, `committed-review.json` and
`trajectory.json` (or its partial counterpart) under `output_dir` using the same
private session and conflict-safe installation. Existing source public artifacts
are restored unchanged. Diagnostics stay under `private_base`; output destinations
do not become model write grants. Recorded policy/configuration identities omit
credentials and callback capabilities. This additive API retains committed result
schema 1 and extension API 8; it is a minor feature addition.

For ordinary hosted committed input, the preparation attestation is required **before Git worktree materialization**:
the operator must control/audit source Git configuration, checkout hooks and filters.
Later raw-byte validation cannot prove inert preparation. This API does not safely
prepare arbitrary local Git repositories. Read/search policy is not filesystem
confinement or an egress firewall; stock readers can follow symlinks. The host owns
infrastructure isolation, scoped credentials, staging, deadlines, concurrency and
cleanup. Use the immutable input contract below for inert source materialization;
the 512-assignment limit and other-harness/CLI integration boundaries still apply.

### Immutable PR inputs

The same embedding API accepts
`CommittedReviewRequest(..., hosted_policy=policy, immutable_input=input)` with
`daydream.immutable_review.ImmutableReviewInput.from_dict(envelope)`. The trusted
host supplies schema version 1; repository configuration cannot provide identity,
source attestation or execution authority. The canonical envelope is frozen before
model work and retained in schema-2 `committed-review.json`. Ordinary committed
results remain schema 1. Both result versions retain the local analyzed revision,
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
subprocess output. Control and blob captures are bounded while reading. The
committed program separately admits at most 512 assignment parts.

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

## Pi host-tool lifetime and limits

Pi keeps its stock read/search tools, offsets and pagination. Ordinary committed reviews
bind those tools, the model-visible working directory and native AGENTS context
to the captured source while native configuration and its trust policy remain
rooted in the original source.
Existing private artifact/sanctioned-input owners furnish frozen diff parts and
source context.
Models select assigned source IDs or supporting path/side/LF ranges; the host
constructs canonical `evidence_refs` from the full assignment inventory and pinned
regular Git blobs. Exact continuation fragments retain their null source ranges.
Supporting attribution cannot satisfy assigned coverage. Correspondence is never
proof of a read, understanding or an executed check.

The existing `structured_output` terminal tool calls one invocation-local host
callback. Only a sole terminal call in a completed native generation may enter
admission. The host validates required judgments, original captured inputs, source
selectors and shared retention limits before installing the entire stage. A
correctable payload rejection returns a correlated native error naming the
field/selector, violated constraint and concrete correction. It changes no
findings, decisions, notes or retention capacity. Correction uses the same
conversation and cumulative wall/tool allowance. A rejected real tool start
still consumes that allowance; exhausted capacity cannot invite another call.
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

Ordinary transient preacceptance retries use existing cumulative policy. The
binding owns the first terminal failure and cannot reopen after it. `/admit` and
`/submit` enforce readiness, correlation, authority, deadline and cumulative
capacity with the same recovery semantics. Frames, arguments, results,
items, request concurrency and cleanup are bounded; overflow rejects whole payloads.
Every real native tool start counts, including failed submissions, with no separate
eight-call ceiling or hidden nested tool execution.

Close authority before shutdown and join/isolate callback transports and evidence
writers before freezing. Native reads use the existing process cleanup. Acceptance,
native settlement/EOF/zero exit and cleanup are separate outcomes; acceptance or a
receipt alone cannot establish successful execution. Preserve admitted findings,
usage and truthful incomplete coverage; see
[lifecycle evidence](observability.md#pi-host-tool-lifecycle-evidence).

Endpoint credentials stay outside persisted configuration/public evidence.
Bindings and resources belong to one invocation; sibling shutdown is isolated.
Enforcement works without recording or tracing. See
[supported capabilities](extensions.md#invocation-scoped-pi-host-tools).

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
runs in an operational worktree. Join writers before freezing evidence. Tracked
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
