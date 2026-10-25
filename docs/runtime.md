# Review runtime contracts

Read this before changing review completion, publication, recovery, or validation.
Use [configuration](../daydream/config.py), [resolution](../daydream/run_config.py),
and `daydream --help-all` for settings rather than copying defaults into docs.

## Terminal review findings contract

`--findings-out` uses schema version 2 with `terminal_result` version 1.
Producer and poster must use the same reviewed revision. Diagram artifacts carry
no review coverage; old/unknown schemas are rejected.

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

Commit-bound exports require a clean checkout and an independently trusted
publication target. A moved PR head fails before export. Use a unique, initially
absent output path per invocation; replacement requires a known expected run ID
because a failed replacement may preserve an older complete artifact.

## Committed-review request/result

`daydream.committed_review.run_committed_review(CommittedReviewRequest(source, base))`
is the production package entry. It defaults to native Pi and review-only policy;
`config` resolves global/file and `review`/`arbiter` phase overrides before work.
It freezes requested native model/provider/thinking selections, retains native
credential refresh, and rejects later settings/model-registry changes. Requested
settings and native observations in the trajectory remain distinct evidence.
The resolved profile supplies `discovery.generic_fallback` and `arbitration`
strategies; protected host obligations cannot be disabled by profile prose.

The package-level committed-review program owns one captured committed change,
its complete planned targets, admitted semantic claim revisions and challenge
obligations. It uses native Pi investigation followed by a distinct native
conversation seeking counterevidence for every proposed publication claim.
Confidence and investigation completion cannot waive challenge. A changed claim,
trigger, consequence, location or supporting grounds requires fresh challenge;
presentation-only title changes preserve the semantic revision.

Confirmed findings, substantively rejected concerns with their counterevidence,
and unresolved concerns remain distinct in the sealed result. Missing or failed
challenge leaves unfinished work. Discovery cannot consume the protected
challenge and finalization allowance. Admission validates a complete submission
before synchronously rechecking authority, revisions and shared limits at install;
closing authority precedes cleanup and sealing. Accepted work survives later
native settlement failure.

The run's artifact session installs `committed-review.json` with schema version 1:
canonical claims and challenge evidence, typed terminal coverage, run ID, captured
head/base/diff identity and planned-work identity share one atomic JSON write.
The report is staged before that write. The session owns final publication and
preserves competing bytes on conflict. Required-output installation, native
settlement, review completion, capture and external delivery have independent
outcomes. The existing findings envelope 2 and terminal-result schema 1 retain
their contracts; this package-level entry does not change CLI routing.
`CommittedReviewResult.to_dict()` returns a copy of the sealed payload;
`output_installed` reports public installation, and `capture_state` reports the
trajectory snapshot independently. Dataset upload/external exports and delivery
are left to route/consumer integration; explicitly requested unsupported export
routes fail before model dispatch.

## Pi host-tool lifetime and limits

Pi keeps its stock read/search tools, offsets and pagination. Existing private
artifact/sanctioned-input owners furnish frozen diff parts and source context.
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
artifact paths cause refusal; competing bytes survive conflicts. Output-publication
failure restores prior checkout artifacts. Optional archive, evaluation, index,
or upload failure preserves completed review outputs and emits safe diagnostics.

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
