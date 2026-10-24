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

## Pi host-tool lifetime and limits

A requested Pi host-tool binding grants one invocation an exact set of file
paths and a finite absolute deadline. Registration does not grant directory-wide
access. The operation returns complete UTF-8 contents; non-UTF-8 files reject.
The host validates the schema, operation, path, open authority, remaining
capacity, and deadline before reading. Component-wise no-follow opens reject
symlink traversal, and reads require regular files. A separate read process
allows cancellation to end filesystem work before evidence freezes.

Definition/schema, transport frames, arguments, results, item counts, cumulative
calls/work, callback lifetime, concurrent requests, and reads have finite limits.
Overflow rejects the whole result; it never returns a truncated authoritative
file. Serialized MCP read results stay below Pi's direct MCP truncation threshold.
The server does not queue excess callbacks. Native admission charges each tool
call once, including rejection; delivery of the same native event cannot charge
it again. Progress and retries cannot renew authority or its allowance.

Close authority before awaiting shutdown. Join or isolate read processes,
callbacks, and evidence writers before recorder/span finalization. Record
authority closure, native settlement, and cleanup independently: endpoint
shutdown, a successful tool, assistant prose, or child exit alone cannot establish
native completion. Bound host-tool invocations require positive native settlement
even when no structured output schema was requested. Preserve observed usage and
truthful failure/incomplete outcomes on every exit path.

The endpoint and credentials are private invocation capabilities, excluded from
persisted configuration and public request evidence. Cached backend instances
do not own sibling grants or callbacks. Enforcement does not require recording
or tracing. See [the supported capability table](extensions.md#invocation-scoped-pi-host-tools).

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
