# Extension contract (`daydream_ext`)

Read before changing extension behavior or implementing a fork extension.
Current API: **8**, supported range **8..8**. Exact exported types/methods live in
[api.py](../daydream/extensions/api.py), [registry.py](../daydream/extensions/registry.py),
and [builtins.py](../daydream/extensions/builtins.py); consult builder signatures
for prompt names/kwargs instead of maintaining another inventory.

## Implement and validate

1. Create `daydream_ext/__init__.py` with `DAYDREAM_EXT_API = 8` and
   `register(registry)`. Built-ins are seeded first; registries are per-run.
2. After creating the package, run `uv sync --reinstall-package daydream` for
   editable discovery. Fork wheels include it through Hatch configuration.
3. Run `daydream ext validate`, then exercise the affected flow. Validation
   resolves registrations without running phases, exporters, or model calls.

Discovery prefers `$DAYDREAM_EXT_DIR/__init__.py`, then importable `daydream_ext`,
then built-ins. Invalid versions/imports/registration fail before agent work.
Breaking registry, step, prompt, stable data, or supervision contracts require
an API bump; additive optional fields/kwargs do not. Raise both supported bounds
for a hard break; validate migrated extensions after upgrading Daydream.

## Flows and custom phases

Use `FlowStep` with async `run(ctx)`: return `None`, `Stop(exit_code)`, or
`BreakLoop()`. `enabled(ctx)` gates execution; configuration uses `config_phase`
or the step name. Custom phase keys work in `[tool.daydream.phases.NAME]` or
`.daydream.toml`'s `[phases.NAME]`. Resolution is CLI → phase → global → backend;
custom phases skip built-in phase defaults.

`set_flow` resolves references at preflight; insert/remove require existing anchors
immediately. Replace a `LoopGroup` to edit its members. Custom flows use `--flow`;
review/comment/shallow are modes of `deep`. Fork stack rules win first-match
routing, while profiles/builders supply review behavior. The preamble's workspace,
identity, diff, and recorder work precedes every extension step.

### Built-in flows and steps

Read [registration](../daydream/extensions/builtins.py) and
[deep assembly](../daydream/deep/orchestrator.py) for ordered names/config keys.
Review/comment stop before fixes. Fix verification preserves retained fixes,
stops regressions, records unresolved outcomes, and repeats after tests alter
the tree. Models cannot authorize commits through completion prose.

### Diagram

The diagram-only flow is exploration → diagram → post-diagram and shares its
first two step objects with deep. Replacing either affects both flows. It preserves
prior deep recovery artifacts/diff keys and records diagram identity, never stale
review findings. Custom deep ordering keeps diagram generation serial; observers
after `diagram` see joined, attached output. Read [grounding](runtime.md#diagrams).

### Improve

Read [assembly](../daydream/improve/orchestrator.py) and
[service discovery](../daydream/services.py). All model phases require Claude's
independent snapshot-root guard. Custom survey strategy text keeps two-pass recon;
default discovery is combined. Primary manifests define assignment ownership,
not reads. Gitlinks carry path/commit references, not captured nested content;
retain explicit unavailable coverage. Issue publication is opt-in and reconciles
existing issues before creation.

### Repair jobs (`test` step)

Consume all three `run_agent` return values; abort reasons outrank model claims.
Only budget-interrupted turns may continue, and only completed jobs authorize
green results/commits. Preserve consumed bounds, prior execution evidence,
authorized-tree checkpoints captured before restoration, and exclusive job
ownership. Corrupt checkpoints block recovery; evidence stores continuation-token
digests rather than tokens. Scope widening requires the turn's canonical path
request/evidence and coordinator approval, never test-output paths alone.
Read [coordinator](../daydream/deep/repair_coordinator.py),
[job](../daydream/deep/repair_job.py), [outcomes](../daydream/phases/repair_outcome.py),
[checkpoints](../daydream/phases/repair_checkpoint.py), and
[test evidence](../daydream/phases/test_evidence.py) before replacing this step.

## Prompts

Overrides replace whole builders and must accept supplied optional kwargs and
preserve host schemas. Wrapping/replacing `structural` opts out of alternatives
folding; custom strategy text alone does not. Merge receives language plus retained
structural pools; its validated synthesis is authoritative. Preserve source UIDs;
structural whole-file privileges require actual structural attribution.

### Stage-aware review builders (API 8)

The host invokes `per-stack`, `structural`, or `generic-fallback` anew per stage
with `review_stage`; overrides always run. Extension API and internal stage/cache
contract versions are independent. Preserve these supplied groups:

| Fields | Meaning |
|---|---|
| `stage`, `scope_id`, `analyzed_revision` | Frozen first-pass/integration/triage identity; local source includes `source_id` |
| `assigned_files`, `source_refs`, `source_options`, `assignment_parts`, `source_summary` | Exact targets, canonical source identities and host source IDs; offsets are UTF-8 diff-body bytes |
| `supporting_bundle`, `supporting_catalog` | Bounded admitted context/pointers and availability; no directory-wide access |
| `remaining_work`, `advisory_tool_call_target`, `remaining_tool_calls`, `hard_tool_call_allowance` | Dispatch-time residual work, pace and enforced budgets; future reserves already excluded |
| `context_inputs`, `context_transport`, `context_statuses`, `canonical_input_identities` | Transport and original captured input bindings |
| `notes`, `note_context` | String notes plus aligned reviewer/source attribution; assertions, not completion/read receipts |

Custom builders opt into bundles with `review_input_bundle = True`; otherwise
`diff_path` and adjacent `hunk-index.json` retain real-file semantics. Required
assignment bytes precede optional context. The host atomically validates target
IDs, schemas, frozen hashes, and backend/cwd/mode bindings. Reviewed targets need
assigned `evidence_refs`; optional forgiving `supporting_files` never veto valid
primary decisions. Native transport schemas remain strict. Read
[admission](../daydream/review_investigation.py) and
[input bounds](../daydream/review_stage_inputs.py).

Inline staged dispatch packs original immutable parts into at most 64 KiB of
sanctioned input, including wrappers. First-pass groups assign at most four files;
the structural interaction stage retains its whole-scope inventory. Original
part sizing remains 12 KiB inline and 24 KiB for exact paths, preserving target
IDs, hashes, selectors, and canonical bytes. Other input callers retain the
12 KiB default; staged review selects a fixed 64 KiB allowance that remains bound
through revalidation. Exact-path resource limits and backend read roots
retain their existing policy.

Open/unresolved candidates receive one triage round. Closed reviewer assertions,
contradictions, unfinished targets, and isolated failures preserve their original
authority and incomplete coverage; later success cannot erase earlier failures.
Final-turn output selection and native structured submission remain host-owned;
malformed envelopes cannot recover nested fragments. Native Pi's private stage
contract 15 uses selector submissions and one host-accepted canonical result after
clean native settlement. Its model schema is distinct from retained canonical
output; other backends retain their canonical reference schema. The private cache
identity changes with this contract, while extension API 8 and persisted canonical
evidence remain compatible. Budgets/reserves remain host policy; feedback is
system instruction, never a synthetic user turn. Read
[output selection](../daydream/agent_output.py) and the relevant backend.
Configured reviewer time and the shared pipeline deadline cover preparation,
every stage, and retries. Explicit caller wall or absolute deadlines can tighten
that same cumulative bound.

For staged stack and structural reviewers, `--review-profile` accepts
`[pipeline].reviewer_tool_call_budget`, a nonnegative integer base allowance
independent of those time limits. The host applies existing diff-size scaling
once, counts every observed tool start across stages and retries, and retains
incomplete coverage on exhaustion; zero stops review dispatch. Changing the
allowance changes profile identity and invalidates incompatible cached reviews.
The default remains defined in [configuration](../daydream/config.py).

### Committed-review program

The [committed-review API](runtime.md#committed-review-requestresult) owns review
independently of extension flow steps/builders. Optional `CommittedReviewRequest.local` and `AnalyzedRevision.source_id` retain
extension API 8; captured local and committed subjects follow the
[captured-source contract](runtime.md#captured-local-source). Broader route integration is #1679; default cutover is #1641 after #1686
comparison/owner acceptance; final retirement is #1687.

### Plan-writer

`verification_commands` stays `Sequence[str]`; typed command evidence is in
`recon_summary`. Request `PlanWriterResult` using the supplied `PLAN_AUTHOR_SCHEMA`
and `PLAN_WRITER_CONTRACT_INSTRUCTIONS`. Host assembly owns deterministic repair
and pointered issues. Legacy Markdown-only output blocks plan publication.

## Invocation-scoped Pi host tools

Pi review stages bind `HostToolBinding(deadline=..., accept=...)` through
`run_agent(host_tools=..., output_schema=..., result_schema=...)`. The synchronous
callback prepares and atomically admits one complete canonical stage or raises
`HostToolRejected` for a correctable preacceptance submission fault. It raises
`HostToolFatal` for a confirmed host/input/lifecycle failure, preserving its cause;
unexpected callback exceptions have an indeterminate mutation outcome and cannot
authorize replay. `output_schema` describes model selectors;
`result_schema` describes canonical retained output. The callback, counters and
deadline belong to the live assignment, including preacceptance retries. Each
attempt owns its endpoint and credentials. `close_attempt()` revokes that endpoint;
`close_authority()` permanently prevents another session. Consumers must close
assignment authority when cancelling, finishing or superseding its work. Neither
method discards an accepted result. Shared investigation budgets retain retry
recovery spending across conversations as well as observed native tool starts.

The packaged native extensions and bounded authenticated loopback endpoint require
Pi 1.4.0 for the verified completion/tool APIs. They confirm the active terminal
tool/schema and intended endpoint before provider work. Missing/overridden tools,
tools-disabled or incompatible modes fail explicitly. There is no filesystem
reader, filesystem grant protocol or MCP runtime extra. Bindings may declare charged
nonterminal host tools; committed review uses `review_source` and `review_admit`
(see [committed review](runtime.md#committed-review-requestresult)), while staged
deep review declares none.
Stock Pi reads/searches retain their native behavior and read-only restrictions.
Ordinary unbound calls retain their supported version/configuration behavior.
Improve still requires Claude's independent snapshot-root guard.

Assigned `source_options` accompany all supplied parts and bounded deferred
Structure catalogs. Models copy compact `source:N` handles in
`{source_id, basis, source_range}`, with null retaining the exact host identity,
or select supporting `{path, side, source_range, basis}`. One deterministic map
per captured review scope binds identical source identities to the same handle
across that inventory's targets and stages. Admission accepts only exact handles
belonging to the target's frozen inventory; existing exact `source:<SHA-256>`
selectors remain compatible.
The full frozen assignment inventory resolves IDs, including later parts;
abbreviated prompt examples do not narrow authority. The host supplies revisions,
hashes, rename paths, roles and fragment metadata, validating original canonical
inputs again before acceptance. Optional `supporting_files` associates relevance
and remains forgiving. See [source/admission lifetime and limits](runtime.md#pi-host-tool-lifetime-and-limits).

Successful host acceptance returns the canonical result and terminates normally.
Correctable errors identify the rejected field/selector and a concrete correction
within the original allowance. Terminal outcomes explain why changing the
submission cannot resolve the failure, close authority, and stop native generation
and submission recovery. Binding-local fatal state remains authoritative; native
details cannot grant acceptance or permission to retry. Rejected calls remain
native history and trajectory evidence while contributing no admitted semantic
state. Accepted work survives later native execution failure, which still marks
execution incomplete. No possibly accepted mutation is replayed.
Native provider/model/authentication, genuine conversation roles, ordinary tools,
and sibling ownership remain with their existing owners.

This additive callback keyword and source-option metadata retain extension API 8.
The private callback error envelope and typed exception contract are introduced
by this unreleased change and ship together with the native extensions. Callers
must use `HostToolRejected` to certify a correctable preacceptance fault; arbitrary
`ValueError` does not certify that no mutation occurred. No released callback
protocol needs migration.
The superseded reader was unpublished PR machinery and has no released migration.
Prompt overrides must preserve the supplied response schema; sanctioned inputs and
real `diff_path` / adjacent `hunk-index.json` semantics retain their contracts.

## Tool supervision

Register one synchronous `(name, tool_input, *, phase) -> ToolDecision` callable.
Vetoes need nonblank reasons and close only the current invocation, retain its
partial trajectory, and return `tool_vetoed:<name>`. Exceptions are extension
failures, not retryable backend failures. Supervision observes `ToolStartEvent`;
backend dispatch timing remains host-owned. Extension tool supervision conflicts
with built-in `tool_supervisor = "rules"`; Claude's always-on guards still apply.
Held findings are excluded from artifacts, PR posts, and fixes; round-trip the
whole payload, including `held` and identity fields, when filtering items.

## Trace exporters

Factories are synchronous `(ObservabilityConfig) -> SpanExporter` and return fresh
exporters with finite timeouts/shutdown. Registration/validation never contacts
providers or requires credentials. Operator environment owns credentials; runtime
owns flush/shutdown, spans, redaction, and content policy. Vendor compatibility
mutates copies only. See [observability](observability.md).

## Renderers

`finding(finding, ctx)` returns inner finding Markdown; `summary(ctx)` returns
inner summary Markdown. Preserve summary finding `body_block` dedup markers and
`ctx.diagrams` verbatim. `agent_prompt` is compatibility-only and empty.
The host adds comment count, incomplete warnings, reviewed commit, footer, and
review event. Empty/invalid output or exceptions fall back to the built-in renderer.

## Working artifact paths

Use the same `ctx.artifacts` session and supplied data paths throughout a flow;
join writers before finalization. Pass that session and `allow_standalone=False`
to artifact helpers. Custom outputs belong under `runs/<session_id>/`; map live
and durable paths with session methods instead of reconstructing public paths.
Generated model inputs require `prepare_sanctioned_inputs`, bound to exact files,
backend/cwd/mode and unchanged bytes. Read [runtime contracts](runtime.md#artifacts-and-recovery)
and [input transport](../daydream/prompt_budget.py) for limits. Standalone calls
must opt in explicitly; production flows retain session ownership. A workspace
opened by the host carries its ownership authority through artifact-session close
and worktree cleanup. Live-code consumers use `ctx.artifacts.code_workspace.run`
for guarded operations; no separate path/identity or validation callback is needed.
Existing session routing/validation methods and standalone analysis paths remain
available under API 8.

Pass `ctx.run_context` to agents/built-in phases. Use `confirm`/`choice` with safe
defaults for unattended runs. GitHub helpers receive `ctx.github_execution.auth`
explicitly; keep that capability outside data, logs, and serializers. Only the
nonsecret login may appear in ordinary identity metadata.

## Stable `ctx.data` keys

These remain readable under API 8; other keys are internal:

| Keys | Contract |
|---|---|
| `diff`, `diff_path`, `tier` | Review text, path, size tier |
| `exploration_dir`, `intent_path`, `alts_path` | Supplied pre-scan/analysis paths |
| `items_file` | Canonical JSON Path after load-items; rewrite items here and preserve held/identities |
| `items` | Parsed findings populated by fix-gate; absent earlier |
| `diagrams` | Optional blocks/payload/results mapping; access with `.get()` |
| `import_graph` | Changed-file dependency sets, empty when unavailable |
| `intent_authoritative` | Fresh head-matched PR-body authority; defaults false on resume |

### Finding identity

Preserve `uid` (pre-merge record), `item_uid` (durable merged item), and
`source_uids` (provenance) through filtering/rewriting. `id` is a renumberable
display position. Several items may share sources; explicit empty provenance
remains empty. Mint identities only at host birth/validation, never from content.
Duplicate record UIDs are fatal. Accessors live in
[records.py](../daydream/deep/records.py).

## Extension boundaries

Backends and coupled host parse/test/commit/control-loop prompts are not registry
extension seams. Host dispatch, preamble, validation, identities, and publication
remain authoritative under overrides. Fork stack rules are additive; they cannot
remove the built-in routing table.
