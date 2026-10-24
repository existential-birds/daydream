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
| `stage`, `scope_id`, `analyzed_revision` | Frozen first-pass/integration/triage identity |
| `assigned_files`, `source_refs`, `assignment_parts`, `source_summary` | Exact targets, selectable source identities/ranges/fragments; offsets are UTF-8 diff-body bytes |
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

Open/unresolved candidates receive one triage round. Closed reviewer assertions,
contradictions, unfinished targets, and isolated failures preserve their original
authority and incomplete coverage; later success cannot erase earlier failures.
Final-turn output selection and native structured submission remain host-owned;
malformed envelopes cannot recover nested fragments. Native Pi uses the latest
successful finalized submission after clean settlement, never prose or fallback
from a host-invalid success. Budgets/reserves remain host policy; feedback is
system instruction, never a synthetic user turn. Read
[output selection](../daydream/agent_output.py) and the relevant backend.

### Plan-writer

`verification_commands` stays `Sequence[str]`; typed command evidence is in
`recon_summary`. Request `PlanWriterResult` using the supplied `PLAN_AUTHOR_SCHEMA`
and `PLAN_WRITER_CONTRACT_INSTRUCTIONS`. Host assembly owns deterministic repair
and pointered issues. Legacy Markdown-only output blocks plan publication.

## Invocation-scoped Pi host tools

Trusted callers may pass an explicit `host_tools` binding to `run_agent`.
Install the `host-tools` extra to select the official SDK's bundled FastMCP:
`uv tool install --editable '.[host-tools]'`. Repository checks install all extras.
Definitions and schemas describe the operation; grants, deadlines, counters,
transport credentials, and callbacks belong to that single invocation. The first
operation reads an explicitly permitted regular file. This seam does not select
captured review sources or admit findings; those remain Review responsibilities.
Binding lifetime does not determine the lifetime of a review assignment.

| Harness/version | Required host-tool capability |
|---|---|
| Pi 1.4.0 on POSIX | Native direct MCP through the official Python MCP SDK's bundled FastMCP; ordinary and read-only invocations |
| Claude, Codex, Osprey | Explicit host-tool bindings are unsupported; ordinary calls remain supported |

Bound calls require Pi 1.4.0 exactly; other versions reject before provider work.
Calls without a host binding retain their existing version behavior. CI installs
the same verified version for the native contract tests.

The extra pins official `mcp==1.30.0`. MCP 2 removes bundled FastMCP, so ordinary
calls can coexist with consumers requiring SDK 2, while a requested binding
without the supported SDK rejects before model work. The standalone RL lock
retains its SDK 2 dependency; its changes only synchronize extra metadata.

The operation requires directory-relative no-follow file opens on POSIX. Bound
calls reject unsupported platforms before model work. Daydream does not replay a
bound invocation in a fresh attempt; native provider recovery remains subject to
the original host deadline and cumulative tool allowance.

Native provider, model, settings, and authentication remain in Pi's existing
configuration path. The issue #1620 delivery review uses the current native
default, verified on 2026-10-24 as `nous` /
`deepseek/deepseek-v4.1-flash-1017`, without a model override or settings change.

Pi proves the intended endpoint's native initialization/tool-list exchange and
checks actual active tool names, schemas, and direct exposure before provider
work. Failed initialization, configuration overrides, disabled or missing tools,
and incompatible required modes fail explicitly. Tools-disabled, finalization,
and continued-session calls cannot request this binding. Improve still requires
Claude's independently enforced snapshot-root guard.

The host checks authorization and finite bounds before filesystem work. A native
schema-valid request can still be rejected, and a correlated native error lets
Pi correct it within the original allowance. Ordinary synchronous supervision
remains an observation of native events; it is not the host operation's
authorization boundary. See [host-tool lifetime and limits](runtime.md#pi-host-tool-lifetime-and-limits).

This optional keyword is additive under extension API 8. Existing prompt
overrides, sanctioned inputs, and real `diff_path`/adjacent `hunk-index.json`
semantics retain their contracts. No extension migration or cache-version change
is required.

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
must opt in explicitly; production flows retain session ownership.

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
