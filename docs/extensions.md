# Extension contract (`daydream_ext`)

Use a top-level `daydream_ext` package to customize flows, prompts, stack routing,
tool supervision, trace exporters, and findings rendering without editing `daydream/`.
Current API: **8**, supported range **8..8**.

## Implement and validate

1. Create `daydream_ext/__init__.py` next to `daydream/`, exporting
   `DAYDREAM_EXT_API = 8` and `register(registry)`.
2. Register the needed customization. Built-ins are seeded before `register()`;
   the registry is rebuilt once per run.
3. After first creating the package, run `uv sync --reinstall-package daydream`
   so the editable install discovers it. Fork wheels already include the package
   through the upstream Hatch configuration.
4. Run `daydream ext validate`. It loads the extension, reports its source/API,
   tool supervisor and registry inventory, and resolves every flow and stack rule.
   Broken references exit 1. No target repository or exporter credentials are needed.
5. Exercise the affected flow against a test repository; registry validation does
   not execute phases or validate their output.

This extension filters findings before publication and fixing:

```python
import json

from daydream.extensions import FlowStep, Registry
from daydream.flows.engine import FlowContext

DAYDREAM_EXT_API = 8

async def filter_items(ctx: FlowContext) -> None:
    path = ctx.data["items_file"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"] = [item for item in payload["items"] if item["severity"] != "low"]
    path.write_text(json.dumps(payload), encoding="utf-8")

def register(registry: Registry) -> None:
    registry.register_phase(FlowStep(name="filter-items", run=filter_items))
    registry.insert_after("deep", anchor="load-items", step="filter-items")
```

Round-trip the complete payload and retained item dictionaries: later steps may
have added top-level `held` findings or host-assigned identity fields.

### Loading and compatibility

Discovery order: `$DAYDREAM_EXT_DIR/__init__.py` (loaded fresh each run),
`import daydream_ext`, then built-ins only. An absent extension is normal.
A present extension with an absent/out-of-range version raises
`ExtensionVersionError`; import, registration, or missing-`register` failures raise
`ExtensionError`. These failures name the source and exit 1 before workspace,
recorder, or agent work.

Breaking changes to registry methods/types/signals, flow or step names, prompt
names/kwargs, stable `ctx.data` keys, or tool-supervision/findings semantics require
an API bump. Additive steps, slots, prompts, fields, and optional kwargs do not.
An increased ceiling with the old floor retains older supported contracts; a
hard break raises both. Upgrade Daydream first, then adapt the extension, update
`DAYDREAM_EXT_API`, and validate. The loader accepts only the inclusive
`MIN_SUPPORTED_EXTENSION_API_VERSION..EXTENSION_API_VERSION` range.

For exported symbols and exact type definitions, read
[`extensions/__init__.py`](../daydream/extensions/__init__.py) and
[`extensions/api.py`](../daydream/extensions/api.py); for registry methods, read
[`extensions/registry.py`](../daydream/extensions/registry.py).

## Flows and custom phases

Phase names share one global registry namespace. `FlowStep(name, run,
config_phase=None, enabled=None)` runs an async callable returning `None` to
continue, `Stop(exit_code)` to end the flow, or `BreakLoop()` to leave the current
loop group. `enabled(ctx)` gates execution. `phase_key` is `config_phase` or the
step name; composite steps may resolve another backend key inside their body.

Use these operations inside `register(registry)`:

```python
from daydream.extensions import FlowStep, LoopGroup, StackRule

registry.register_phase(FlowStep(name="my-gate", run=my_gate))
registry.insert_after("deep", anchor="intent", step="my-gate")
# Alternatively: registry.insert_before("deep", anchor="fix-gate", step="my-gate")
registry.remove("deep", "arbiter")
registry.register_phase(FlowStep(name="verify", run=my_verify), replace=True)
registry.set_flow("my-audit", ["my-gate"])
registry.add_stack(StackRule("proto", ("*.proto",)))
```

`set_flow` resolves entries during `run_flow` preflight or `ext validate`, so
registration order is unrestricted. Insert/remove operations check anchors
immediately. To reorder, remove/reinsert entries or replace the complete flow.
Steps inside a `LoopGroup` cannot be addressed directly by insert/remove; replace
its group entry. A loop is declared as:

```python
LoopGroup(name="my-loop", steps=("my-fix", "my-verify"), max_iterations=lambda ctx: 3)
```

Run a custom flow with `daydream --flow my-audit /path/to/project` or
`RunConfig(flow_name="my-audit")`. Built-in `--flow` names
(`deep`/`review`/`shallow`/`improve`) route through their dedicated helpers.
`--review`, `--comment`, and `--shallow` are modes of `deep`, not separate
registered flows. Unregistered names fail resolution.

Fork stack rules match changed files before the built-in extension table, in
registration order (first match wins). They supply routing metadata; the resolved
profile strategy and prompt hooks supply review behavior.

### Built-in flows and steps

Names below are stable even when mode, tier, or resume gates skip a step.
Workspace/identity resolution, diff computation, recorder setup, stack detection,
and resume checks precede all steps; the first insertion point is `exploration`.

#### `deep`

| # | Step | Registered step key |
|---|------|------------|
| 1 | `exploration` | `exploration` |
| 2 | `intent` | `intent` |
| 3 | `per-stack-reviews` | `per_stack_review` |
| 4 | `per-stack-parse` | `parse` |
| 5 | `arbiter` | `arbiter` |
| 6 | `cross-stack-merge` | `merge` |
| 7 | `single-stack-merge` | `single-stack-merge` |
| 8 | `load-items` | `load-items` |
| 9 | `supervise` | `supervise` |
| 10 | `diagram` | `diagram` |
| 11 | `findings-out` | `findings-out` |
| 12 | `post-review` | `post-review` |
| 13 | `fix-gate` | `fix-gate` |
| 14 | `verify` | `verify` |
| 15 | `fix` | `fix` |
| 16 | `fix-verify` | `fix-verify` |
| 17 | `test` | `test` |
| 18 | `commit` | `fix` |
| 19 | `remote-ci` | `remote-ci` |

- Review/comment modes stop after `post-review`; shallow forces one stack;
  the default mode runs the fix pipeline.
- `per-stack-reviews` also runs the alternatives pass on fresh multi-stack runs,
  as a sibling task. Its backend/config key is `wonder`; it has no separate step.
- Cross-stack merge can resume an unsharded arbiter session only when the arbiter
  returned a continuation token and both phases use the same backend instance.
  The resumed prompt requires rereading the record files rewritten by arbitration.
- `fix` and `fix-verify` form `LoopGroup("fix-verify-loop", ...)`, capped at
  three rounds. The latter audits retained hunks, uses the `verify` backend/config
  key, and invokes the `fix-verify` prompt. At the cap, `unresolved`/`wrong_target`
  verdicts warn and persist in `fix-outcomes.json`; `regressed` stops the run.
  Tests/healing that change the tree trigger verification again; newly actionable
  findings stop publication. Commit messages list only verified resolutions.
- Diagrams are disabled by default. `--diagram auto`, a named kind, or repository
  diagram `mode = "auto"` enables generation; `--start-at fix` skips it.
  Compatible deep flows start generation alongside independent review work when
  capacity permits, admitting primary reviewers first. The final `diagram` step
  joins generation and idempotently attaches its blocks after synthesis and
  supervision. It retains standalone generation for custom flows and diagram-only
  runs. Generation records deterministic eligibility, authors eligible kinds
  read-only (at most one repair each), and verifies against the head tree; when no
  kind is eligible, it makes no agent calls.
  Early scheduling requires the unchanged built-in prefix through `diagram`;
  inserted/replaced phases in that prefix retain serial generation. Observers
  after `diagram` see completed and attached results. Completed packaged-builder
  generation is restored on per-stack/merge resume only when the snapshot,
  eligibility/settings, author inputs and model contract match its checkpoint.
  Shared admission considers enabled model phases only and ends at the final
  diagram step, preserving later fix and custom phase capacity.
  Deterministic eligibility precedes backend resolution; ineligible generation
  needs no model backend. Capacity planning failures retain serial dispatch so
  optional overlap cannot move later phase construction errors ahead of review.
- `.review-output.md` cleanup runs after successful `run_flow`, subject to mode
  and `config.findings_out is None`; it is not an insertable step.

#### `diagram`

Order: `exploration` → `diagram` → `post-diagram`. The first two reuse the same
step objects as `deep`, so replacing either affects both flows. `post-diagram`
writes a `kind="diagram"` findings artifact when `--findings-out` is set;
otherwise it posts a standalone PR issue comment with a hidden
`daydream-diagram` marker per kind. It always ends the flow. Diagram-only runs
reuse the deep preamble and preserve previous deep recovery artifacts.
They neither clear `.daydream/deep/` nor replace its `diff-key`. Recorder/manifest
identity stays `DaydreamRunFlow.DIAGRAM`, so surviving merged findings are not
captured as results of the diagram-only run.

#### `improve`

| # | Step | Registered step key |
|---|------|------------|
| 1 | `recon` | `recon` |
| 2 | `audit` | `audit` |
| 3 | `vet` | `vet` |
| 4 | `select-plans` | `select-plans` |
| 5 | `write-plans` | `plan_write` |
| 6 | `publish-improve-issues` | `recon` |
| 7 | `improve-report` | `recon` |

The default `recon` step combines survey and reconnaissance in one bounded
model investigation, reusing host inventory and service discovery. Its admitted
`recon_summary` includes conventions (with descriptions/source paths), guidelines,
languages/frameworks, intent documents and host-validated command records.
A changed `exploration.repository_survey` strategy retains the two-pass
repository-survey/recon compatibility path; a partial profile that leaves this
strategy unchanged uses combined discovery. Independent `repo_scan` calls keep
their existing contract. Recon lifecycle metadata and `recon.json` identify
`combined` or `survey-recon`; only the fallback emits a `repo-survey` phase.

Configuration includes `improve_effort`, `improve_focus`, `improve_scope`, and
`improve_plan_description`. Issue publication requires
`[tool.daydream.improve.github] publish_issues = true`.

Improve audit builders retain their existing keyword arguments. Their `group`
mapping now includes `primary_manifest` (an exact inline `primary_files` list or a
snapshot-relative file descriptor with hash, path count, and line count) and
`primary_manifest_sanctioned`. The latter means `run_agent` supplies the descriptor
through sanctioned inputs; builders need not duplicate it. Large manifest bytes
live only in a host-created directory inside the existing audit root, with a
1 MiB cap and one path per line for paginated native Read access. Durable copies contain ownership paths, never the disposable descriptor.
Group partition entries count owned slices while retaining original names, roots,
and services. Context remains available for concrete dependency questions; it is
not additional ownership. Branch manifests list removed paths in `deleted_files`
for diff-only investigation, and counts include those changed paths.
Tracked gitlinks appear in `submodule_references` as path/commit metadata, not
`primary_files`; their nested contents are absent from the snapshot. The manifest
byte limit and descriptor counts include these references. Completed coverage
separates `primary_file_count` from `submodule_reference_count` and records
`unavailable_submodule_contents` with unknown nested file counts. Planned coverage
does not claim this metadata before the captured index has been inspected.
`primary_file_count` includes ordinary file-path assignments and diff-only deletions.

### Agent calls and phase configuration

Use `run_agent()` with `ctx.backend_for(phase_name)`, `ctx.work.repo`, and
`run_context=ctx.run_context`; this preserves backend selection, budgets,
supervision, and trajectory recording. Custom prompt builders can be registered
with `override_prompt` and retrieved through `ctx.registry.prompt(name)`.

Arbitrary phase names already work in `[tool.daydream.phases.<name>]` in
`pyproject.toml` or `[phases.<name>]` in `.daydream.toml`:

```toml
[tool.daydream.phases.my-gate]
model = "claude-sonnet-5"
```

Custom phases skip the built-in phase-default tier. Precedence remains CLI
model/effort → phase table → config-file global → backend default. See
[reasoning effort](../README.md#reasoning-effort) for the complete chain.

### Repair jobs (`test` step)

A replacement test/repair phase must consume all three `run_agent` return values
(text, continuation token, abort reason). Host aborts take precedence over model
completion claims. `RepairOutcome` classifies turns as `candidate_complete`
(host-validated structured completion), `diagnosis_unresolved`,
`budget_interrupted`, `execution_error`, or `scope_blocked`. Only budget
interruptions may continue in a later execution; this enum does not add public
`ReasonCode` values.

Keep host-observed evidence for every execution in `TestAndHealResult.repairs`
and `test-verdict.json`. A repair job can report green or authorize a commit only
in `completed` state. Capture checkpoints from the live authorized-path tree
before restoration; corrupt checkpoints block recovery. Preserve prior execution
evidence and the captured policy on resume. Continuation tokens are represented
by digests in evidence, never stored verbatim there.

Scope widening requires the repair turn's own canonical repository-relative path
request and per-path evidence, approved by the coordinator through
`AuthorizedFixFootprint.authorize_widened_path(action="approve_scope",
origin="scope_request")`. Test-output paths alone grant no authorization.
Only the host repair coordinator dispatches further executions; one owner may
hold a job at a time.

When replacing this step, read the exact records and lifecycle in
[`repair_outcome.py`](../daydream/phases/repair_outcome.py),
[`test_evidence.py`](../daydream/phases/test_evidence.py),
[`repair_checkpoint.py`](../daydream/phases/repair_checkpoint.py),
[`repair_job.py`](../daydream/deep/repair_job.py), and
[`repair_coordinator.py`](../daydream/deep/repair_coordinator.py).
Operator bounds are in [repair job budgets](../README.md#repair-job-budgets).

## Prompts

`registry.override_prompt(name, builder)` replaces the whole prompt; overrides
receive the built-in kwargs. Accept optional additions explicitly or through
`**kwargs`. All arguments below are keyword-only except those marked positional.
`intent_authoritative: bool = False` is passed to `per-stack`, `structural`,
`generic-fallback`, `arbiter`, and `merge`; exact-signature overrides must accept
it. It is false when absent on resume.

Overriding `structural`, even with a wrapper around the built-in builder, opts
out of host alternatives folding: the host runs alternatives separately.
Custom structural strategy text can still fold alternatives if the builder
remains the built-in function.

| Prompt | Kwargs |
|--------|--------|
| `intent` | `strategy`, `diff_path`, `branch`, `log`, `exploration_dir`, `pr_description`, `inline_diff`, `inline_exploration_summary` |
| `alternatives` | `strategy`, `intent_summary`, `diff_path`, `exploration_dir`, `inline_diff` |
| `fix` | `test_output`, `feedback_items` (both positional), `repo`, `concise_mode` |
| `per-stack` | `strategy`, `stack_name`, `files`, `diff_path`, `intent_path`, `alternatives_path`, `output_path`, `cwd`, `exploration_dir`, `prior_commits`, `inline_diff`, `intent_authoritative`, `include_alternatives`, `frontier_files`, `review_stage` |
| `structural` | `strategy`, `files`, `diff_path`, `intent_path`, `alternatives_path`, `output_path`, `cwd`, `exploration_dir`, `prior_commits`, `intent_authoritative`, `include_alternatives`, `review_stage` |
| `generic-fallback` | `strategy`, `files`, `diff_path`, `intent_path`, `alternatives_path`, `output_path`, `cwd`, `exploration_dir`, `is_docs_only`, `prior_commits`, `inline_diff`, `intent_authoritative`, `include_alternatives`, `frontier_files`, `review_stage` |
| `arbiter` | `strategy`, `arbiter_input_path`, `diff_path`, `intent_path`, `alternatives_path`, `cwd`, `exploration_dir`, `intent_authoritative` |
| `supervise` | `strategy`, `supervise_input_path`, `diff_path`, `intent_path`, `alternatives_path`, `cwd`, `exploration_dir` |
| `suppression` | `strategy`, `suppression_input_path`, `diff_path`, `intent_path`, `alternatives_path`, `cwd`, `exploration_dir` |
| `merge` | `strategy`, `per_stack_records_paths`, `intent_path`, `alternatives_path`, `dedup_candidates_path`, `output_path`, `exploration_dir`, `failed_stacks`, `structural_records_path`, `intent_authoritative`, `resumed_from_arbiter` |
| `verify` | `strategy`, `items`, `cwd`, `output_path` (accepted, ignored — the host writes the verdicts file) |
| `fix-verify` | `items`, `changed_hunks`, `cwd`, `round_number` |
| `diagram_sequence` | `diff_path`, `inline_diff`, `files_by_module`, `cwd`, `exploration_dir`, `schema`; inline transports (resolved from the backend and cwd) require `exploration_dir=None`, `clone_mode`, `inline_exploration`, `inline_dependencies` |
| `diagram_flowchart` | `diff_path`, `inline_diff`, `candidate_roots`, `forced`, `cwd`, `exploration_dir`, `schema`; inline kwargs as for `diagram_sequence` |
| `audit` | `category`, `strategy`, `group`, `scope_note`, `recon_summary`, `cwd`, `tier` |
| `vet` | `strategy`, `findings`, `cwd` |
| `plan-writer` | `finding`, `recon_summary`, `verification_commands`, `cwd` |

Merge builders receive language paths in `per_stack_records_paths` and the retained
meta-stack through `structural_records_path`. Every model-owned merge receives
both pools as sanctioned inputs, including resumed arbitration and bounded,
tools-free finalization. Successful synthesis owns the validated published set;
the host does not append omitted structural records. Combine duplicate defects
with all contributing `source_uids`, grounded evidence and affected files; keep
distinct concerns. Whole-file structural evidence/location privileges require
retained structural source attribution, not merely `lens="structural"`.
Structural-only/tiny-diff host bypass and failure salvage retain their host
folding policy. Completed merge reuse keys include the synthesis policy so
pre-policy output cannot be restored on fresh runs; merge resume recomputes it.

### Stage-aware review builders (API 8)

The host calls `per-stack`, `structural`, or `generic-fallback` anew for each stage,
passing `review_stage`, a mapping of host-owned assignment and admitted state.
Overrides are always invoked. Extension API 8 is separate from internal stage
contract 11, which invalidates older review caches. The stage response now includes `supporting_files: []` alongside
`targets`, `notes`, `candidates` and `contradictions`; builders must preserve this supplied strict response contract.
Native transports require `supporting_files` as an array or null. The host admission schema differs only for this
optional metadata: absent, null or malformed references supply no usable association and never repair or invalidate
otherwise valid primary target/finding judgments. Transport request schemas remain strict.

| Host field group | Contract |
| --- | --- |
| Identity: `stage`, `scope_id`, `analyzed_revision` | `first_pass`, `integration`, or `triage`, bound to the public reviewer and frozen snapshot. |
| Assignment: `assigned_files` | Current assignment, also passed as the builder's `files`. |
| Bundle: `supporting_bundle` | Bounded diff/index/binding or integration context. Built-ins use it; custom builders opt in with `review_input_bundle = True`. Otherwise `diff_path` and adjacent `hunk-index.json` retain their real file semantics. |
| Navigation: `supporting_catalog` | Exact pointers to admitted Structure diff projections; `status` and `omitted_part_count` declare partial availability. Navigation grants no directory access. |
| Residual work: `remaining_work` | Estimated residual assignments, files and stages; native Pi also exposes current submission/total start floors, remaining submission floor and reserved future starts. |
| Budget: `advisory_tool_call_target`, `remaining_tool_calls`, `hard_tool_call_allowance` | Dispatch-time snapshots: suggested pace, remaining reviewer cumulative allowance and the current invocation's hard allocation. Future-stage reserves are already excluded from that allocation. |
| Context: `context_inputs`, `context_transport`, `context_statuses`, `canonical_input_identities` | Admitted supporting inputs, transport/availability and original artifact identities. Omitted optional context is advisory; corrupted required inputs block admission. |
| Semantic notes: `notes`, `note_context` | `notes` preserves API 8's string list. Aligned context entries identify `note_index`, validated `supporting_files`, source anchors in `files` and host `origin` (scope, stage, sequence, reviewer judgment origin and analyzed revision). These are attributed assertions, never completion/read/test evidence. |

The owning artifact session projects frozen diff/index inputs. Required
assignment bytes take priority over bounded shared context and optional Structure
parts/catalogs. Custom builders opt into bundles with `review_input_bundle = True`;
otherwise `diff_path` and adjacent `hunk-index.json` keep their real-file semantics.
Read [review budgets](../README.md#review-budgets) for assignment/transport caps,
retry rules, and source-read policy.

Admission requires strict schemas, exact target/candidate IDs, valid dispositions
and findings, meaningful claims, and nonempty `not_reviewed` reasons. The host
checks revision/input hashes, complete transport, backend/cwd/mode, confinement,
and unchanged captures. Failures, exhausted bounds, corrupted required inputs,
handoff overflow, and unfinished decisions remain unsuccessful or incomplete.
Optional `supporting_files` are repository-relative source associations for the submitted notes. The host normalizes
safe references and excludes unavailable files, symlinks, Git metadata and generated review artifacts without vetoing
otherwise valid target/finding decisions. Associations grant neither read access nor reviewed coverage; no source-read
receipt is required. Relevance metadata, attribution and rendered decisions count toward the shared 64 KiB/128-item
semantic handoff. Closed decisions and host target acknowledgements retain their independent authority.
Findings and snapshot-bound typed coverage are
[atomically published](../README.md#terminal-review-findings-contract).

Discovery candidates marked `open` or `unresolved` receive one triage round
within the cumulative allowance/deadline. Triage receives admitted judgments and
may investigate further. Closed decisions retain `judgment_origin: reviewer`;
host admission validates their contract, not semantic correctness. Unresolved
triage and source-supported contradictions keep affected coverage incomplete
without reopening closed decisions or discarding admitted findings.
An isolated backend invocation failure leaves its batch unfinished and retains its diagnostic in the scope outcome
and attempt trajectory. Later independent assignment/triage batches may continue under the original cumulative
bounds and outage admission. Later success cannot clear prior failure evidence or settle a failed candidate group;
admission failures and shared stopping conditions still stop further work.
The scope's existing reason codes and diagnostics retain both the first failure and any later stopping failure,
including a cumulative budget or outage refusal before another stage starts; per-stage trajectories retain other attempts.

The host selects text from the final assistant turn, including explicit empty
turns and bounded overflow. Staged text accepts prose, closed JSON Markdown
fences, and one transparent single-property object wrapper whose value validates
and whose root has no expected schema fields. It consumes complete outer JSON
spans, ignores unrelated complete roots, and requires one unambiguous result.
Competing results, output-shaped invalid roots, wrapper siblings, lists, recursive
wrappers, truncation, malformed tails, and unfinished fences cannot recover nested
results. Only one complete schema-invalid text payload passing the domain
rejection guard qualifies for one fresh retry; syntax, envelope, ambiguity,
domain, and native failures do not. Domain admission precedes presentation;
backend events remain raw trajectory evidence.

Static stage-prompt and initial-preamble budget values are dispatch-time
snapshots; API 8 fields and kwargs keep their names and values. Native Pi reserves
minimum assignment reads/submissions for undispatched batches and known triage
chunks. The current invocation allocation already excludes that reserve; do not
subtract it again. Stages and transport retries are skipped if their minimum
needs cannot fit. Concrete unresolved assigned checks may exceed advisory targets
within the allocation/deadline. Host limits count observed starts rather than
prospectively admitting tools; budget stops preserve admitted findings and
truthful incomplete coverage.

Bounded native Pi calls replace one live budget note before each model request.
It gives one attempt-local remaining-call count, including final submission;
failed calls, parallel members, and submissions each count. Transport retries
reset the native counter against their effective attempt allowance; cumulative
host accounting and recorded evidence remain authoritative. Unbounded calls
remain unbounded.

Use the refreshed count without recounting history. Reserve one
`structured_output` call; continue concrete unresolved assigned checks while
tool/time capacity permits, and submit when checks settle. Unused capacity is
expected. With only submission capacity left, submit truthful incomplete
decisions for unfinished work. At zero, report exhaustion without implying
submission capacity. Feedback does not infer completion, force settlement, add
capacity, change ordinary tool access, or rewrite already-proposed batches.
Native output authority, stage/domain admission, and host enforcement remain
unchanged.

Pi retains read-only `read`/`find`/`ls`/`grep` and materialized before-side source;
Codex uses a disposable read-only checkout. Claude/Codex may inspect captured
revisions with Git. Native Pi accepts the last successfully finalized
`structured_output` submission only after clean EOF/reaping/settlement. Failed
submissions do not replace a success; a selected host-invalid success cannot fall
back to an older submission, and prose cannot supply native output. One hidden
missing-submission reminder may use the original deadline/allowance. Read-only
calls loading the packaged extension isolate it from ambient extensions.
No-schema and validation-opt-out text behavior is unchanged. Improve recon and
plan authoring retain validation-opt-out text and downstream partial-object repair.
See [trace semantics](observability-fields.md#effective-request-configuration).

### `plan-writer` compatibility and output contract

`verification_commands` remains `Sequence[str]` of literal repository commands;
`recon_summary` contains typed records with IDs, working directories,
applicability, expected results, and evidence. Use these records for detailed
command guidance without adding a required kwarg.

Request `PlanWriterResult` structured data. Overrides can compose
`PLAN_WRITER_CONTRACT_INSTRUCTIONS` from `daydream.improve.prompts`.
The host always supplies `PLAN_AUTHOR_SCHEMA`; `assemble_plan` validates,
repairs deterministically, reports remaining defects as pointered `AssemblyIssue`
values, and expands the host-owned plan consumed by `render_plan`.
Legacy `{markdown: ...}` output yields `AUTHOR_SCHEMA_INVALID` issues,
indexes the finding as `BLOCKED`, and writes no plan. Sanitized diagnostics carry
stable metadata/error codes. Prompt overrides cannot weaken this boundary.
These unchanged kwargs/output enforcement keep extension API 8.

## Tool supervision

Register one synchronous callable:

```python
from daydream.extensions import ToolDecision

def supervise(name, tool_input, *, phase):
    if name == "Write":
        return ToolDecision(veto=True, reason="writes require approval")
    return ToolDecision(veto=False)

# Inside register(registry):
registry.register_tool_supervisor(supervise)
```

Signature: `(name: str, tool_input: dict[str, Any], *, phase: DaydreamPhase)
-> ToolDecision`. A veto requires a nonblank reason, closes only the current
invocation's event stream, records the partial turn, and returns
`tool_vetoed:<name>` as its abort reason. Other invocations continue. Supervisor
exceptions propagate as extension failures, without backend retry.

Supervision occurs when `run_agent` receives a backend `ToolStartEvent`;
backend dispatch timing remains host-controlled. Duplicate, non-callable, or
async registrations fail. With no supervisor, this hook does nothing.

Built-in findings/tool rules and configuration are documented under
[supervisor settings](../README.md#supervisor-settings). Findings actions are
`allow`, `drop`, `edit`, and `hold`; held findings move from `items` to top-level
`held` in `merged-items.json` and appear under **Held Findings** in the report,
but are excluded from findings artifacts, PR posts, and fixes. Round-trip the
whole JSON payload when filtering. Extension tool supervision conflicts with
`tool_supervisor = "rules"`; choose one. Claude's always-on guards remain active.

## Trace exporters

`registry.register_trace_exporter(name, factory, replace=False)` registers a
synchronous `(config: ObservabilityConfig) -> SpanExporter` factory. Select it
with `--trace-to name` or `DAYDREAM_TRACE_TO=name`. Names start with a lowercase
letter, contain lowercase letters/digits/dots/underscores/hyphens, and are at most
64 characters. Existing names require `replace=True`. Query factories/inventory
with `trace_exporter(name)` and `trace_exporter_names()`.

`ObservabilityConfig` contains `destinations: tuple[str, ...]`,
`capture_content: bool`, and `service_name: str`. Resolve transport settings and
credentials from the operator environment. Registration/validation must not
instantiate exporters, require credentials, or contact providers. Return a fresh
exporter per call with finite transport timeouts and bounded shutdown; runtime
resolves selected names before construction and owns flush/shutdown.

The host owns spans, task context, content policy, and redaction. Add provider
compatibility attributes to a copy, preserving shared spans. Export failures
produce sanitized diagnostics and preserve the review outcome. Read
[observability](observability.md) for lifecycle, content, and operator recipes.

## Renderers

`registry.override_renderer(name, fn)` replaces either slot:

| Slot | Signature | Output |
|------|-----------|--------|
| `finding` | `(finding: CommentFinding, ctx: FindingRenderContext) -> str` | Inner finding Markdown |
| `summary` | `(ctx: SummaryContext) -> str` | Body between approval line and footer |

`CommentFinding` exposes `path`, `line`, `title`, `body`, `is_cross_stack`,
`severity`, `confidence`, and `fingerprint`. `ctx.placement` is `inline`,
`file_level`, or `summary`; the finding renderer runs for all three.
`SummaryContext` supplies `findings: tuple[SummaryFinding, ...]`, `agent_prompt`,
`review_info`, and optional `diagrams`. Each summary finding has `finding` and
host-rendered `body_block`.

Preserve each `body_block` verbatim: it contains the dedup marker needed to
prevent duplicate posts. Include `ctx.diagrams` verbatim to retain grounded
diagrams in posted summaries. The host wraps renderer output with the footer,
separators/details scaffolding, approval line, and review event decision;
renderers control only inner content. Exceptions, empty output, or non-string
returns log a warning naming the slot and fall back to its built-in renderer.

## Working artifact paths

Production/custom flows own an active `ctx.artifacts` session. Generated files
are private during execution, frozen at finalization, then published as
`.daydream/` and `.review-output.md` in the source checkout.

Runner finalization reports its active stages and elapsed outcome on stderr,
including optional evaluation. These presentation messages are separate from
the frozen evidence and its timing; custom flows must still join artifact
writers before returning control for finalization.

- Use supplied `ctx.data` paths; write new outputs through the same session.
  Direct public writes can fail the model-cwd check before a later dispatch.
  Working paths expire with the session.
- Pass `session=ctx.artifacts, allow_standalone=False` to `artifact_dir_for`
  and `review_output_path_for`, even with a bound session. Pass
  `artifact_session=ctx.artifacts` and
  `allow_standalone=ctx.allow_standalone_artifacts` to built-in phases.
  Keep runtime ownership choices outside `ctx.data`.
- Put custom outputs in `run_directory(root, session_id)` under
  `runs/<session_id>/`. Legacy adoption accepts only registered top-level names;
  published custom paths reopen only when their whole subtree matches canonical
  recovery. Use `ctx.artifacts.durable_path_for(live_path, repo=ctx.work.repo)`
  for handoffs and `live_path_for(public_path, repo=ctx.work.repo)` for writes.
  These validate ownership, routing, and ancestry; finalization-only dumps have
  no live write route.
- Generated model inputs require `prepare_sanctioned_inputs`: exact named files
  bound to backend/cwd/read-only mode, unchanged before each attempt. Strict
  Claude audit roots, read-only Codex clones, and sandboxed Osprey receive inline
  contents; other modes receive exact paths. This isolates cwd discovery, not
  unrestricted filesystem access, and grants no directory-wide read permission.
- Inline inputs allow 12,288 combined UTF-8 payload bytes. Exact paths allow
  512 files, 1 MiB/file, and 8 MiB combined, validated by streaming hashes.
  Pi's host-supplied exact-path `diff` has a separate 128 MiB limit, stays in the
  exact-file allowlist, and is never copied into recovery finalization context.
  Missing, changed, non-regular, invalid UTF-8, and oversized inputs fail before
  backend entry; they are not truncated.
- Intentional standalone calls opt into `allow_standalone=True`; without a bound
  session this uses public paths. Direct `run_deep`/recorder callers opting in
  must have no active session. Runner/custom flows retain session ownership and
  model-cwd checks.

Example: create a private note and dispatch a read-only agent with its captured input.

```python
from daydream.agent import run_agent
from daydream.artifact_visibility import artifact_dir_for
from daydream.extensions import FlowStep, Registry
from daydream.flows.engine import FlowContext
from daydream.prompt_budget import prepare_sanctioned_inputs
from daydream.trajectory import DaydreamPhase, run_directory

DAYDREAM_EXT_API = 8

async def explain_note(ctx: FlowContext) -> None:
    assert ctx.artifacts is not None
    note = run_directory(
        artifact_dir_for(ctx.work.repo, session=ctx.artifacts, allow_standalone=False),
        ctx.artifacts.layout.session_id,
    ) / "extension-note.txt"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("Review this repository's error-handling conventions.", encoding="utf-8")
    backend = ctx.backend_for("review")
    inputs = prepare_sanctioned_inputs(
        backend, ctx.work.repo, {"note": note}, read_only=True,
    )
    await run_agent(
        backend,
        ctx.work.repo,
        "Use the sanctioned note and inspect the relevant source files.",
        phase=DaydreamPhase.REVIEW,
        read_only=True,
        sanctioned_inputs=inputs,
        run_context=ctx.run_context,
    )

def register(registry: Registry) -> None:
    registry.register_phase(FlowStep(name="explain-note", run=explain_note))
    registry.set_flow("explain-note", ["explain-note"])
```

### Interaction and output policy

Pass `ctx.run_context` to agents and built-in phases. It owns frozen
`assume`/`interactive`/`quiet`/`log_mode` policy and backend registrations.
Use `confirm(..., safe_default=False)` and `choice()` for input so unattended
runs take safe defaults; menu choices support explicit `assume_yes`/`assume_no`
mappings. The runner binds the context, so shared console output and extensions
omitting the argument still inherit its policy. Standalone callers may use
`bind_run_context(RunContext(InteractionPolicy(...)))`; binding restores the
previous context on exit. With no explicit/bound context, defaults are interactive,
non-quiet, non-log, and have no assumed answer.

Pass `ctx.github_execution.auth` explicitly to GitHub helpers, e.g.
`git_ops.gh_api(ctx.work.repo, "/user", auth=ctx.github_execution.auth)`.
It supplies a complete subprocess environment, without merging ambient credentials.
Keep this runtime capability out of `ctx.data`, logs, trajectories, and serializers.
Only the non-secret login is exposed as `ctx.run_context.github_identity.login`
and `ctx.config.identity`. Standalone constructors omitting `github_execution`
inherit the parent environment.

## Stable `ctx.data` keys

Deep steps use a checked internal view over the same dictionary; extension
writes remain visible. These keys are readable under API 8; other keys are
internal and may change without a version bump.

| Key | Meaning |
|-----|---------|
| `diff` | Diff text under review |
| `diff_path` | Diff file path |
| `tier` | Diff-size tier for fan-out gates |
| `exploration_dir` | Pre-scan output directory or `None` |
| `intent_path` | Intent-analysis output path |
| `alts_path` | Alternatives-review output path |
| `items_file` | Canonical `{"items": [...]}` JSON `Path`, available after `load-items`; may also contain `held`. Rewrite `items` here before downstream consumers. |
| `items` | Parsed findings, populated by `fix-gate` from the potentially rewritten `items_file`; absent before that step. |
| `diagrams` | Optional `{"blocks": str, "payload": dict, "results": dict}`. `blocks` is Markdown; `payload` contains eligibility/results with rendered `mermaid` removed; `results` is keyed by `sequence`/`flowchart`. Use `.get("diagrams")`. |
| `import_graph` | `{changed file: set of changed files it imports}`, or `{}` on unavailable grammar, parse failure, or 5s budget. Empty denies only the diagram cross-module rule. |
| `intent_authoritative` | True when a fresh head-matched nonblank PR description grounded intent. Absent on resume; read with `.get("intent_authoritative", False)`. |

### Finding identity

Preserve host-assigned fields when rewriting records/items:

| Field | Meaning | Accessor in `daydream.deep.records` |
|-------|---------|------------------------------------|
| `uid` (`stack:ordinal`) | Pre-merge record identity, minted at record birth | `record_uid` |
| `source_uids` | Records a merged item derives from; provenance, not item identity | `item_source_uids` |
| `item_uid` (`item:n`) | Durable merged-item identity, minted once after validation | `item_uid` |
| `id` | Human-facing finding number, normalized to dense `1..N`; may change | — |

Cross-stack merged items normally have no own `uid`; `record_uid` returns `""`
there. `item_source_uids` prefers explicit attribution, falls back to the item's
`uid`, and preserves an explicit empty list. Several items may share provenance.
`item_uid` survives filtering/renumbering and is absent from `MERGED_ITEMS_SCHEMA`.
Round-trip identities rather than minting or deriving them from content;
duplicate record `uid`s are fatal.

## Extension boundaries

- Backends are built-in (`claude`, `codex`, `pi`, `osprey`); the registry has no
  backend-registration hook.
- Prompt replacement is wholesale. Parse/test/commit/setup-investigator/
  failure-summarizer prompts are coupled to host schemas/control loops and are
  not registered.
- Fork `StackRule`s are additive and win per file; the built-in `_EXT_TO_STACK`
  table cannot be modified or removed through the registry.
- Host dispatch timing, preamble, output validation, identities, and publication
  guards remain authoritative when phases or prompts are overridden.
