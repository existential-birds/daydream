# Extension contract (`daydream_ext`)

Daydream's extension seam lets a fork customize which phases run, prompts,
stack routing, tool supervision, trace destinations, and the canonical findings surface — entirely
from a top-level `daydream_ext` package, without editing any file under
`daydream/`. This document is the versioned
contract: the module shape daydream loads, the exact name inventories a fork
programs against, and the policy for when those names may change.

Current contract version: **`EXTENSION_API_VERSION = 7`** (supported: `7..7`).

## Extension module contract

A fork creates one package next to `daydream/`:

```text
daydream_ext/
└── __init__.py
```

`__init__.py` must export exactly two things:

```python
DAYDREAM_EXT_API = 7          # must be within daydream's supported range

def register(registry):       # receives a daydream.extensions.Registry
    ...                       # mutate flows / prompts / stacks here
```

`register(registry)` runs once per daydream run, after `register_builtins()`
has seeded the registry with everything daydream does today, so the extension
sees (and may mutate) the full built-in state through the same API the
built-ins used.

### Public API symbols

The `daydream.extensions` package exports these contract symbols:

| Symbol | Purpose |
|--------|---------|
| `EXTENSION_API_VERSION` | Running extension contract version |
| `MIN_SUPPORTED_EXTENSION_API_VERSION` | Oldest extension contract version still accepted (range floor) |
| `BreakLoop` | End the current loop group and continue the flow |
| `CommentFinding` | Public view of one review finding passed to a `"finding"` renderer |
| `ExtensionError` | Base error for extension failures |
| `ExtensionVersionError` | Error for an absent or incompatible extension version |
| `FindingRenderContext` | Placement context (`"inline"`/`"file_level"`/`"summary"`) passed with a finding |
| `FlowStep` | Named async flow step |
| `LoopGroup` | Repeated ordered group of flow steps |
| `ObservabilityConfig` | Immutable operator-selected trace destinations, content policy, and service name |
| `Registry` | Per-run extension registry |
| `StackRule` | Fork-defined changed-file-to-stack routing metadata |
| `Stop` | End a flow with an exit code |
| `SummaryContext` | Input to a `"summary"` renderer (findings, agent prompt, review info) |
| `SummaryFinding` | One finding in a `SummaryContext` (public finding plus host-rendered `body_block`) |
| `ToolDecision` | Continue or veto a tool invocation |
| `ToolSupervisor` | Callable protocol for tool supervision |
| `TraceExporterFactory` | Synchronous factory for a per-run OpenTelemetry span exporter |
| `UnresolvedExtensionError` | Error for a missing registered name |
| `build_registry` | Seed and load a per-run registry |
| `get_registry` | Read the current async context's registry |
| `set_registry` | Set the current async context's registry |

### Discovery order

1. `$DAYDREAM_EXT_DIR` — explicit path to the package directory and the test
   seam. Daydream loads
   `<dir>/__init__.py` fresh on every run — never via `sys.modules` — so
   repeat runs and tests never see a stale module.
2. `import daydream_ext` — the fork extension package.
3. No extension — builtins-only registry. Absence is silent and normal.

A *present-but-broken* extension is a loud, named error before any workspace,
recorder, or agent work happens: a missing or mismatched `DAYDREAM_EXT_API`
raises `ExtensionVersionError` naming the module source path, the declared
version, and the supported range; a missing `register`, an import failure, or an exception inside
`register()` raises `ExtensionError` with the original message. All of them
exit the run with code 1.

### Packaging

Upstream's `pyproject.toml` pre-declares `daydream_ext` in
`[tool.hatch.build.targets.wheel] packages`; hatchling silently tolerates the
declared-but-absent package upstream and includes it when a fork ships it. So
a fork adds the package with zero upstream-file edits and wheels keep working.

Editable-install note: after first *creating* the `daydream_ext` package in a
fork, run `uv sync --reinstall-package daydream` so the editable install picks
up the new top-level package.

## Versioning policy

`EXTENSION_API_VERSION` (in `daydream/extensions/api.py`) is a single integer.
It bumps on **any** breaking change to:

- the registry API (`Registry` methods, `FlowStep` / `LoopGroup` / `StackRule`
  fields, the `Stop` / `BreakLoop` signals, the error hierarchy),
- flow names or step names,
- prompt names or their kwargs,
- the documented stable `ctx.data` keys below,
- the tool-supervisor decision and findings-file semantics below.

The loader accepts any `DAYDREAM_EXT_API` within the inclusive range
`[MIN_SUPPORTED_EXTENSION_API_VERSION, EXTENSION_API_VERSION]` — the floor is
the oldest contract the tool still understands and `EXTENSION_API_VERSION` is
the ceiling. A declared version above the ceiling (newer than the tool
understands), below the floor (a contract the tool has dropped), or absent is a
loud `ExtensionVersionError` that exits the run with code 1.

On a bump, advance `EXTENSION_API_VERSION`. An **additive** bump leaves the
floor where it is, widening the supported window so a not-yet-upgraded
extension keeps loading — that window is the deprecation window. A
**hard-breaking** bump raises the floor to the new version in the same release,
since no older extension can run against the changed contract. Deprecating an
aged-out version later is one edit: raise the floor.

Upgrade ordering: upgrade the tool before the extension. A newer tool runs an
older in-range extension (the rolling-upgrade window), but an older tool cannot
run a newer contract — forward compatibility is unachievable by any gate.

Additive changes (new steps, new slots, new prompts, new optional kwargs) do
not bump the version.

### Migration guidance

To migrate a fork to a new contract version: adjust the registered steps, prompts, and `StackRule`/findings surface to
the new inventory, then declare the new `DAYDREAM_EXT_API` and validate with
`daydream ext validate`. A hard-breaking bump (where ceiling and floor rise
together) requires the declaration and the new surface in the same change; an
additive bump only widens the window and needs no code migration.

## Tool supervision

An extension may register one synchronous callable for the lifetime of the
per-run `Registry`. Daydream calls it for each `ToolStartEvent` emitted by a
backend:

```python
from daydream.extensions import ToolDecision

def supervise(name, tool_input, *, phase):
    return ToolDecision(veto=False)

def register(r):
    r.register_tool_supervisor(supervise)
```

The callable has the signature
`(name: str, tool_input: dict[str, Any], *, phase: DaydreamPhase) -> ToolDecision`.
`name` is the tool name, `tool_input` is the backend-provided input mapping,
and `phase` identifies the current `DaydreamPhase`. The callable is
synchronous; it must not be declared with `async def`.

Return `ToolDecision(veto=False)` to let the invocation continue. Return
`ToolDecision(veto=True, reason="...")` to abort the current agent turn; a veto
requires a non-blank reason. Daydream closes the current invocation's event
stream, records the partial turn, and returns a `tool_vetoed:<name>` budget
reason to the caller. Other invocations sharing the backend continue running.
If the supervisor raises, the failure propagates as an extension failure rather
than being treated as a backend retry.

`register_tool_supervisor` accepts only one callable per registry. A second
registration or a non-callable value raises `ExtensionError`. If an extension
does not register a supervisor, tool supervision is a no-op. Supervision runs
when `run_agent` receives the backend's `ToolStartEvent`; it does not change
backend-specific dispatch timing or add an earlier backend hook. The built-in
rule supervisor uses this same turn-level enforcement point; it cannot intercept
a tool before the backend emits its start event.

### Built-in supervisor configuration

The built-in findings supervisor is disabled by default. Set
`supervisor = "rules"` to drop findings whose repository-relative `file` matches
one of `supervisor_deny_globs`, or set `supervisor = "llm"` for one batched
adjudication call. The LLM call uses the model configured at
`[tool.daydream.phases.supervise]` (or `[phases.supervise]` in `.daydream.toml`)
and defaults to the Sonnet tier for supported backends.

```toml
supervisor = "rules"
supervisor_deny_globs = ["vendor/**", "generated/**"]
tool_supervisor = "rules"
tool_bash_deny = ["rm -rf", "git push --force"]

[phases.supervise]
model = "claude-sonnet-5"
```

The built-in tool supervisor applies the shared file globs to `Write` and
`Edit`, and applies `tool_bash_deny` as regular expressions to `Bash` commands.
Claude's always-on guards (dangerous-command and background-Bash) remain
active; this configuration adds rules and does not replace them.

Supervisor actions are `allow`, `drop`, `edit`, and `hold`. A held finding is
removed from the actionable `items` list and stored under the top-level `held`
key in `merged-items.json`; the rendered report keeps it under **Held Findings**.
All downstream readers continue to consume `items`, so held findings do not
reach findings artifacts, PR posts, or fix prompts.

Only one tool supervisor may be registered per run. If an extension registers a
tool supervisor while `tool_supervisor = "rules"` enables the built-in one, the
run fails at registry construction with a conflict error. Choose the extension
policy or the built-in policy.

## Trace exporters

Register a named destination through the same seam used by the built-in `otlp`,
`langsmith`, and `honeyhive` exporters:

```python
import os

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SpanExporter

from daydream.extensions import ObservabilityConfig, Registry

DAYDREAM_EXT_API = 7

def company_exporter(config: ObservabilityConfig) -> SpanExporter:
    return OTLPSpanExporter(
        endpoint=os.environ["COMPANY_OTLP_TRACES_ENDPOINT"],
        headers={"Authorization": "Bearer " + os.environ["COMPANY_OTLP_TOKEN"]},
        timeout=5,
    )

def register(registry: Registry) -> None:
    registry.register_trace_exporter("company", company_exporter)
```

Select it with `daydream . --trace-to company` or `DAYDREAM_TRACE_TO=company`.
`TraceExporterFactory` has the signature
`(config: ObservabilityConfig) -> SpanExporter`. Its configuration fields are
`destinations: tuple[str, ...]`, `capture_content: bool`, and `service_name: str`.
Factories resolve their own transport settings from the operator environment.
Do not read destinations or credentials from the repository being reviewed.

`register_trace_exporter(name, factory, replace=False)` requires a synchronous
callable and a unique lowercase name starting with a letter. Names may contain
digits, dots, underscores, and hyphens and are at most 64 characters long.
Use `replace=True` to replace an existing destination. `trace_exporter(name)`
returns its factory; `trace_exporter_names()` returns the registered inventory.

Registration and `daydream ext validate` do not instantiate exporters, require
credentials, or contact providers. The runtime resolves all selected names
before calling factories, then owns each returned exporter's flush and shutdown.
Return a fresh exporter for each call. Exporters must use finite transport
timeouts and implement shutdown without unbounded blocking.

The host owns span creation, task context, content policy, and redaction. An
exporter receives the same portable spans as other destinations; provider-specific
compatibility attributes may be added to a copy without mutating the shared span.
Export failures produce sanitized diagnostics and preserve the review outcome.
See [observability](observability.md) for the lifecycle and content contract.

## Inventories

### Flows and steps

Three flows are registered: `deep` (the single PR-process flow), `improve`, and
`diagram` (the `--diagram-only` grounded-diagram flow).
Each step's *registered step key* is its `FlowStep.phase_key`
(`FlowStep.config_phase`, defaulting to the step name). The registry and flow
lifecycle use this key. A composite step can deliberately resolve a different
backend key inside its body.

**Naming convention:** phase names are one global registry namespace. The `deep`
flow owns the plain names. The `review` and `shallow` flows were collapsed
into modes of `deep` (#330): `--review`, `--comment`, and `--shallow` run the
review spine, with shallow forcing a single stack. Fork-defined flows should
follow the same convention: pick globally unique step names, and use
`config_phase` to reuse an existing config key.

#### `deep` (the single PR-process flow, #330)

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

The steps are gated by the run's mode (`ctx.data["mode"]`), set in the dispatch
preamble. `review` / `comment` run the review spine and stop after `post-review`;
`shallow` forces `single_stack_mode`; `loop` is the default fixing flow;
`diagram` runs the separate `diagram` flow below over the same preamble.

`.review-output.md` cleanup is not a flow step; it runs in `run_deep` after
`run_flow` returns, tied to a successful outcome, an applicable mode, and
`config.findings_out is None`.

`fix` and `fix-verify` (issue #744) are wrapped together in a `LoopGroup`
(`fix-verify-loop`) capped at 3 rounds: each round dispatches findings, a
read-only `fix-verify` step audits the round's changed hunks and returns one
verdict per finding, and actionable verdicts re-dispatch in the next round
until none remain (or the budget is spent). `fix-verify` has registered step
key `fix-verify`; its retained-tree verifier resolves model and backend
overrides from `[tool.daydream.phases.verify]` and uses the separately
registered `fix-verify` prompt.

At the round limit, `unresolved` and `wrong_target` findings remain in
`fix-outcomes.json` and produce a warning; the retained patch continues through
tests, commit, and push. A `regressed` verdict still stops the run. If tests or
healing change the retained tree, verification runs again: previously unresolved
findings may remain unresolved, but newly actionable findings stop publication.
Tree identity, scope enforcement, test validation, and Git hooks still gate
publication. Commit messages list only findings verified as resolved.

`per-stack-reviews` runs the TTT alternative-review (wonder) as well: on a fresh
multi-stack run the two are siblings in one task group, so wonder has no step of
its own. Its per-phase config key is still `wonder`
(`[tool.daydream.phases.wonder]`), resolved inside the step.

`diagram` (issue #1113) decides deterministically which grounded diagram kinds
apply, runs one read-only author agent per eligible kind (plus at most one
repair turn each), verifies every proposed element against the head tree, and
writes `.daydream/deep/diagram.json` + `diagram.md`. It is gated off on a
`--start-at fix` resume and by `--diagram off` / `[tool.daydream.diagram] mode
= "off"`; otherwise it always runs and records its eligibility decision, even
when no kind is eligible (which costs zero agent calls). The rendered blocks
reach the PR summary through `ctx.diagrams` and `review-output.md` through a
`## Diagrams` section.

#### Repair jobs (`test` step, issue #1210)

Step 17 (`test`) owns repair work. When the retained tree's suite is red, the
step may dispatch bounded *repair turns*; those turns are one **repair job** that
outlives a single call, and the contract below is what a fork's own phase needs
in order to participate honestly. Nothing here extends the `Backend` protocol:
`run_agent` still returns exactly three values, and the repair turn consumes all
three — the final text, the continuation token, and the abort reason.

**The five-case outcome vocabulary.** `daydream.phases.repair_outcome.RepairOutcome`
is a `StrEnum`, and it is the *host's* classification, never the model's claim.
A host abort always wins over whatever the turn said: prose claiming a fix inside
an interrupted turn is partial diagnosis, not completion.

| Member | `value` | Reached when |
|--------|---------|--------------|
| `CANDIDATE_COMPLETE` | `candidate_complete` | The turn's own structured payload claimed a complete candidate and the host validated it. Reachable only from the explicit completion path. |
| `DIAGNOSIS_UNRESOLVED` | `diagnosis_unresolved` | The turn ended on its own without narrowing the failure — including a blank turn, because silence is not a success claim. |
| `BUDGET_INTERRUPTED` | `budget_interrupted` | A budget stopped the turn. The only case a later execution may continue from. |
| `EXECUTION_ERROR` | `execution_error` | A non-budget host stop (transport or backend failure). |
| `SCOPE_BLOCKED` | `scope_blocked` | The turn's only candidate was outside the authorization policy and no widening was granted. |

`RepairOutcome` is **not** a `ReasonCode` extension and adds no new public stop
reason. `repair_reason_code(abort_reason)` maps a turn's abort reason onto the
existing vocabulary (budget reasons through `reason_for_budget`, an already-valid
code through `ReasonCode`, anything unrecognised to `ReasonCode.BACKEND_FAILURE`,
and a turn that ended on its own to `None`).

**The evidence record.** Every repair turn produces one
`daydream.phases.test_evidence.RepairAttemptEvidence` (frozen dataclass). Its
fields are host observations, not turn claims — `changed_paths` is what the tree
actually holds, `output_tree_key` is read from the tree, and a value the host
cannot produce is left empty rather than invented:

| Field | Type | Meaning |
|-------|------|---------|
| `job_id` | `str` | The repair job's identity (`repair-<session_id>`). |
| `execution_id` | `str` | This one bounded execution. |
| `run_id` | `str` | The run the execution belongs to. |
| `outcome` | `RepairOutcome` | The host's classification (above). |
| `abort_reason` | `str \| None` | The host's stop reason, verbatim; `None` when the turn ended on its own. |
| `backend_name` / `model` | `str` | The backend that actually served the repair turn (the FIX backend, not the TEST one). |
| `execution_elapsed_s` / `job_elapsed_s` | `float` | Wall seconds for this execution, and for the job so far. |
| `input_tree_key` / `output_tree_key` | `str` | Full-delta tree identity before and after the turn. |
| `changed_paths` | `tuple[str, ...]` | Authorized paths the tree actually changed. |
| `checkpoint_ref` | `str \| None` | The durable checkpoint written for this turn, if any. |
| `focused_evidence` | `tuple[str, ...]` | Bounded, redacted excerpts of the failure output the turn worked from. |
| `scope_request` | `Mapping \| None` | The turn's parsed scope request (see below). |
| `continuation_ref` | `str \| None` | Digest of the continuation token, never the token itself. |
| `diagnostics` | `tuple[str, ...]` | Named degradation of the host's own accounting; never a swallowed failure. |

`reason_code` is a derived property (the converged public reason for the stop).
`payload()` is its JSON form, used verbatim in `test-verdict.json`'s `repairs`
array.

**The phase result.** `TestAndHealResult` gained `repairs:
tuple[RepairAttemptEvidence, ...] = ()` alongside `attempts`, so a caller can
tell an interrupted repair from a completed one without re-reading the
transcript. The field defaults empty, so a phase that never runs a repair turn is
unchanged. The `test` step merges the records of **every** execution of the job
before writing the verdict, so a green run that went through an interrupted
repair reports that interruption.

**The scope-request return path.** A repair turn may not edit outside the
authorized scope; it may *ask*. When a correct fix needs a repository-relative
path the turn cannot edit, the fix prompt instructs it to name the path in its
final message with the `file:line` evidence that requires it and stop there.
`repair_scope_request(repo, payload)` parses that request: it canonicalizes every
named path through `repository_paths` (absolute paths, traversal, and symlink
crossings raise `InvalidRepositoryFilePath` rather than being dropped), keeps
bounded redacted per-path `evidence`, records `evidence_source`, and sorts the
paths so two equivalent requests read identically. A payload that is not an
object, or whose `paths` is not a list, requests nothing; a malformed `paths`
list is an error, never a narrower-than-asked authorization.

Granting is the coordinator's call, never the parser's: only the turn's *own*
request with its own per-path evidence may widen the policy, through
`AuthorizedFixFootprint.authorize_widened_path(..., action="approve_scope",
origin="scope_request")` — the same widening shape as a policy-approved
generated path, recorded under its own `origin="scope_request"`. A path
that merely appears in test output is not evidence of authorization, and a
request that cannot be canonicalized grants nothing.

**Durable state.** Two artifacts under the run's deep directory, both private
during the run and published at finalization like every other generated output:

- `.daydream/deep/repair-checkpoint.json` (`DeepArtifact.REPAIR_CHECKPOINT`) — the
  authorized-path-only patch captured from the **live tree** before any
  restoration, plus structured facts, digests, and bounded redacted excerpts. It
  never carries raw model prose, environment values, or an unbounded command
  line. Reading is deliberately asymmetric: a *corrupt* checkpoint is a recovery
  blocker (`CheckpointRead.blocked`), never an empty job.
- `.daydream/deep/repair-job.json` (`DeepArtifact.REPAIR_JOB`) — the job's
  `RepairJobRecord`: `state`, the captured `policy` (`execution_s`,
  `job_total_s`, `max_executions`, `reserve_s`, `max_cost_usd`), `consumed_s`,
  `cumulative_cost_usd`, `granted_allowance_s`, `executions`, `scope_request`,
  `unchanged_evidence`, `progress_evidence`, `next_experiment`,
  `completed_experiments`, `disproven_hypotheses`, `last_transition_reason`,
  `checkpoint_ref`, and `diagnostics`. Both files carry
  a `format_version`; both are merged read-modify-write, in the shape of
  `deep/routing_record.py`, so a resume never erases an earlier execution's
  evidence.

Job states are `running`, `paused`, `ready_to_resume`, `validating`, `completed`,
`blocked`, and `exhausted`. A job is **structurally fail-closed**: only
`completed` may report a passing verdict or authorize a commit
(`cannot_report_green`). No absolute clock reading is ever persisted —
`execution_allowance_s()` derives each execution's allowance from
the stored consumption, so every deadline is process-local. A job's bounds are
stored, not re-resolved, so a resumed job cannot inherit a broader policy than it
was granted. A green run that never repaired writes no job record and takes no
owner lock, so an ordinary run's artifact set is unchanged.

**Ownership and continuation.** `daydream.deep.repair_coordinator` is the only
component that dispatches a further execution, from inside the existing
`runner.run` flow. Exactly one live repair owner may exist per job: an in-process
owner registry plus the exclusive workspace `flock` exclude a second process, and
a lock-contention result warns and continues instead of running a second worker.

#### `diagram` (`daydream --diagram-only KIND <target>`)

| # | Step | Registered step key |
|---|------|------------|
| 1 | `exploration` | `exploration` |
| 2 | `diagram` | `diagram` |
| 3 | `post-diagram` | `post-diagram` |

The first two steps are the same registered `FlowStep` objects the `deep` flow
uses, so a fork that overrides `diagram` overrides it in both flows.
`post-diagram` is the flow's deliverable: it writes the `kind = "diagram"`
findings artifact and stops when `--findings-out` is set, and otherwise posts a
standalone PR issue comment (never a review) carrying one hidden
`daydream-diagram` marker per kind. It always ends the flow.

The `diagram` flow reuses the deep spine's preamble (workspace, diff, hunk
index, exploration cache) but deliberately does **not** clear
`.daydream/deep/`, so a diagram-only run never destroys a previous deep
review's resumable artifacts.

#### `improve` (`daydream improve <target>`)

| # | Step | Registered step key |
|---|------|------------|
| 1 | `recon` | `recon` |
| 2 | `audit` | `audit` |
| 3 | `vet` | `vet` |
| 4 | `select-plans` | `select-plans` |
| 5 | `write-plans` | `plan_write` |
| 6 | `publish-improve-issues` | `recon` |
| 7 | `improve-report` | `recon` |

The improve run configuration also carries `improve_effort`, `improve_focus`,
`improve_scope`, and `improve_plan_description`. Issue publication is gated by
`[tool.daydream.improve.github] publish_issues = true`; when disabled, the
publication step is a no-op.

Steps carry `enabled` predicates internally (tier gates, mode gates,
resume points); a step listed here may be skipped for a given run, but the
name is stable.

### Prompts

Overriding `structural`, including wrapping the built-in builder, opts out of host
alternatives folding. The host runs the alternatives pass independently and
invokes the custom structural reviewer.
Custom structural strategy text still supports alternatives folding when the
prompt builder itself remains the built-in function.

The 17 registered prompt names and the exact kwargs their builders receive.
Overrides receive the same kwargs and must accept the current contract. All
kwargs are keyword-only except where noted.

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

#### Stage-aware review builders (API 7)

The registered `per-stack`, `structural`, and `generic-fallback` names are
unchanged. The host invokes the selected builder anew for **every** stage with
`review_stage`, a mapping of host-owned assignment and admitted state. API 6
extensions are incompatible and fail at load time with the declared and
supported versions; migrate the callable to accept this kwarg and declare
`DAYDREAM_EXT_API = 7`. Overrides are invoked, never silently replaced.

- `stage` is `first_pass`, `integration`, or `triage`; `scope_id` and
  `analyzed_revision` bind the work to the public reviewer and frozen snapshot.
- `assigned_files` and the builder's `files` are the current assignment. Language
  and generic batches group nearby directories within four files and the actual
  24,576-byte rendered assignment cap for exact paths or the 12,288-byte inline
  input allowance, including wrappers, scoped index and snapshot bindings.
  Complete files are preferred when they fit; required assignment bytes take
  priority over shared context. The stage factory
  keeps exact assignment references outside the separate inline shared-context
  allowance; only the shared bytes actually inlined and their wrappers consume it.
  It reads complete frozen canonical diff/index artifacts and writes bounded inputs
  atomically through the owning `ArtifactSession`; it never uses a truncated
  in-memory diff. `diff_path` points at the current stage projection, with its
  scoped `hunk-index.json`. Sanctioned inline bytes carry that same content;
  `inline_diff` remains optional. Other paths are
  supporting context for concrete candidates, not additional audit targets.
  Structure begins with a whole-change interaction assignment and the synthetic
  target `integration:structure`. Its honest compact inventory and bounded
  supporting diff parts orient boundary traces, rather than alphabetical file
  batches. Omitted supporting parts are explicitly unavailable.
- Contract **4** adds fields under `review_stage` without changing any API-7
  callable signature. `source_access` describes frozen before/after source
  windows, original and rename-side paths, revisions, ranges and permitted read
  methods. `read_required` indicates mandatory fresh work; false alone does not
  prove reuse (the before side may be optional). `admitted_source_windows` binds
  verified complete receipts from successful stages in this reviewer/snapshot.
  Clean stages retain useful source and notes. First-pass handoffs carry receipt
  metadata without repeating source bodies; triage retains relevant compact
  excerpts. Host receipt authority is distinct from current agent knowledge:
  omitted bodies and compaction do not erase retained receipts. Targeted rereads
  remain appropriate for understanding a concrete concern or uncovered source.
  Exact interval unions may cover
  later windows; unknown/opaque ranges and failed attempts never authorize reuse.
  Every assigned unit still requires a fresh review decision. Schema retries
  obtain new evidence under the same absolute allowance/deadline.
- `supporting_bundle` is available to built-in builders. A custom builder can
  opt in by declaring `review_input_bundle = True` on its callable. Without this
  explicit capability its `diff_path` remains a real diff file and adjacent
  `hunk-index.json` remains a real index, with their original semantics. They are
  never aliases for a bundle. First-pass bundles contain the bounded assignment
  diff, index and binding; integration bundles contain the compact whole-change
  inventory and binding with navigation instructions for deferred diff parts.
  Bundle reading supplies supporting context only.
  `supporting_catalog` exposes targeted bounded Structure diff projections where
  exact transport permits them. Its exact pointer leads to bounded 12,000-byte
  JSON catalogs containing `parts` (`file`, existing `target_id`, exact `path`)
  or `catalogs` (exact child `path`, `part_count`). The full part inventory remains
  available without repeating every pointer in the initial prompt. Every catalog
  and part is captured supporting input through `ArtifactSession`; no directory
  access is granted. Catalogs guide navigation, never an exhaustive reading
  checklist. Structure resolves concrete changed-boundary concerns at their
  relevant owners and submits; settled contract checks stay settled unless new
  contradictory evidence appears. Inline omissions remain honestly unavailable.
  Recipe-capable Structure omits duplicate source projection paths, retaining
  concrete before/after ranges and native selectors for file and canonical part
  IDs. `context_availability` summarizes complete source/part/catalog counts;
  explicit partial and unavailable statuses remain in `context_statuses`.
- Additive API-7 `source_catalog` names the exact captured Structure source
  catalog root (`path`, `window_count`, `status`, `read_required:false`). Leaves
  contain `windows` preserving frozen metadata, all target aliases and exact
  source-tool arguments; index nodes contain exact child paths/window counts.
  The separate `source-catalog-*` namespace cannot overwrite diff catalogs.
  Each file remains <=12,000 bytes and participates in the existing owning
  session, count/aggregate capture, confinement and post-invocation revalidation.
  Catalog metadata grants no source evidence or directory access.
- Additive `access_guide` is a JSON string <=8 KiB assembled after input capture.
  It persists in stage system guidance through Pi compaction, retaining exact
  catalog/assignment/shared-context pointers and required source arguments.
  Complete inline context is labeled honestly; optional omissions are explicit.
  Persistent stage identities also retain `admitted_source_windows`; their
  verified covered ranges govern fresh-read obligations when an access guide
  was prepared before receipt reuse. The guide preserves navigation arguments.
  Custom integration builders keep their full initial `source_access` and do not
  acquire a new guide-size admission gate; unsupported persistent access remains
  explicitly unrepresented. Recipe-capable exact integration also receives its
  source catalog with the original required-window flags.
  Output target IDs and readable source selectors remain distinct. Full host
  `source_access` and legacy custom-builder transport contracts are preserved.
- `response_contract` carries the strict invocation schema and explicit
  four-key skeleton. Triage targets are exactly `[]`; discovery candidates may
  be nonempty with `candidate_id: ""`. Independent identity, snapshot, grounds
  and contradiction admission remains authoritative. Syntax/identity failures
  are terminal, with no repair or semantic retry. `remaining_work` estimates
  residual assignments/files/stages and separately labels current-stage fresh
  source/read costs. These estimates prove transport feasibility only; native
  quality/capacity needs matched cold measurements under unchanged numeric limits.
- Claude, Codex and Pi expose the same invocation-owned
  `read_source(target_id, side)` reader. The host serves bounded frozen windows
  through authenticated loopback MCP; Pi's packaged native tool delegates to it.
  The server starts before provider execution and joins requests after provider
  teardown, including cancellation. It serves no private directory or shell access.
  Source projections are atomically owned/revalidated by `ArtifactSession` and
  typed in `PreparedSanctionedInputs`; legacy prepared inputs default to
  supporting. Source bytes/ranges are independently verified, including rename
  sides and bounded enclosing context. Inline prompt bytes alone are not native
  receipts. Codex's independent read-only snapshot is preserved; its required
  source reads no longer depend on shell syntax or merged stderr. Owned MCP names
  and associated single-text frames normalize through existing AgentEvents.
  Foreign MCP tools receive no reader identity. Shell commands cannot establish
  source authority while the owned reader is active. Tracked repository-relative
  paths can select after-side dependencies at captured HEAD. Existing independently
  verified file reads remain supported. Osprey retains frozen-file transport pending
  existential-birds/osprey#1195. Pi's native
  bounded-read LF representation, including its exact continuation footer, is
  verified against frozen bytes; native results and completion flags remain
  unchanged. Unknown wrappers, forged footers and truncation supply no coverage.
  Complete older cache entries miss staged contract 7.
- `assignment_parts` describes each required target: `target_id`, `file`,
  one-based `part_index`, `part_count`, and `kind` (`file`, `hunk`, or
  `continuation`). Split hunks include zero-based `hunk_index`, `old_start`,
  `old_count`, `new_start`, and `new_count`. Continuations also carry ordered
  `segment_index`/`segment_count`, UTF-8 `fragment_offset`/`fragment_bytes`,
  and `old_line`/`new_line`/`fragment_line_offset` mapping. The ranges identify
  the parent hunk; a continuation never claims to contain it whole. Every
  required target must succeed before host file coverage is complete.
  Stage indexes match canonical hunks by old/new ranges, including explicit
  `old_only` entries for pure deletions omitted from the posting index; a
  deletion never shifts the authority of subsequent hunk assignments.
- `assigned_target_ids` must be acknowledged exactly. `assigned_candidate_ids`
  scopes triage; only those candidates and their relevant admitted notes and
  evidence are supplied. `closed_candidate_ids` are handles for reporting
  contradictions, not permission to reopen decisions. Additive `closed_decisions`
  carries each admitted closed candidate's host ID, file/line, disposition,
  bounded conclusion and relevant frozen evidence references. These summaries
  also persist in stage system guidance through Pi compaction. They reuse settled
  decisions without substituting for receipt admission or granting source access.
  No new discovery occurs during triage.
- `remaining_work` keeps source/read estimates separate from native submission
  and total start floors. These are optimistic transport hints, not quality proof.
- `advisory_tool_call_target` suggests a compact stage workload;
  `remaining_tool_calls` is the hard remaining cumulative allowance.
  `observed_tool_starts` includes retries and the received start exceeding the
  hard allowance. Native parallel or buffered execution can precede observation.
  Each supplied bound source read uses its own tool call/result and preserves
  the host's quoted arguments. Parallel reads remain separate calls; compound
  shell commands and concatenated multi-file output supply no bound receipts.
  The reviewer deadline and total allowance remain unchanged across stages.
  Reviewers use the existing remaining-work estimates and allowance to preserve
  capacity for submission, remaining assignments and open-candidate triage.
- `context_inputs` lists only admitted sanctioned labels; `context_transport`
  describes inline bytes or exact paths. `context_statuses` records complete,
  partial or unavailable supporting context. `canonical_input_identities`
  binds generated inputs to the original artifact hashes and analyzed revision.
  Shared intent and exploration are
  admitted whole within the transport budget. Missing context is advisory and
  cannot establish evidence; any necessary pointer reads consume tool calls.
  Triage receives admitted candidate context rather than whole-review inputs.
- `attempt` is 1 or 2, with `max_attempts=2` on legacy paths and 1 for native Pi.
  Only a normally completed,
  complete object on a legacy output path rejected at the strict schema selection boundary may receive
  one fresh stage attempt. `schema_rejection` contains bounded validator
  `category`, host `schema_path`, `error_count`, and `candidate_count`, never
  rejected values or arbitrary unknown property names. The selected registered
  builder runs anew for every stage and attempt. Rejected evidence, notes and
  candidates are discarded; necessary source grounding must be obtained anew.
  There is no serializer recovery, conversation continuation, reserve, extra
  calls, deadline reset or `max_turns`. Native tools-enabled Pi instead owns
  schema correction within its invocation; its errors consume the same allowance
  and deadline and never trigger this host fresh attempt. Backend transport
  retries remain separate.
  Retry eligibility also checks the untouched rejected candidate for independent
  terminal identity, contradiction, grounding and handoff failures. A schema
  error combined with one of these failures does not permit a fresh attempt.

These fields are additive within API 7; exact callable signatures need no new
kwargs. Builder/strategy selection uses the canonical base stack (`generic#0`
routes to `generic-fallback`, `rust#0` retains Rust policy), while public shard
identity remains intact in coverage, artifacts and UIDs.

`ResultEvent.structured_output_origin` is an additive optional backend field:
`native` (the default) means an authoritative structured payload; `text` means
the backend inferred a candidate from assistant text. Staged callers validate
the complete final assistant turn for text-derived candidates, excluding earlier
planning/tool-turn prose. Native structured results remain authoritative despite
prose. The final-turn text buffer is bounded to 128 KiB; overflow is terminal,
never a truncated candidate or a schema-retry witness. Pi activates native output from existing schema/validation/tools arguments;
`RequestEvent` with typed `PiRequestConfig.schema_emulated=False` and a schema
identifies native tool validation, not provider-constrained decoding. Native mode
accepts only the last successful finalized `structured_output` details in
transcript order after EOF/reap and clean settlement, bounded to 128 KiB. Failed
submissions do not replace prior successes; a selected host-invalid success
cannot fall back to earlier output. Both assistant-text fallback and prose
checkpoints are disabled. Mixed batches and native corrections remain ordinary
charged work. Invocation-authorized submission tools are output control, excluded
from source receipts and compact source retention; their recoverable errors do
not invalidate source evidence. No-schema, no-tools/finalization and explicit
validation opt-out preserve their text behavior. API remains 7. Pi forwards typed native tool truncation,
exit, status and cancellation metadata; printed notices are not capture authority.

Build the stage prompt from its assignment and semantic policy. Do not append a
narrow stage to a conflicting terminal-review prompt. Scope dependency tracing,
test-quality review, configuration tracing, and verification to current targets
or assigned candidates. Operator judgment policy still applies within this
scope. Custom structural builders and custom alternatives retain their separate
alternatives behavior.

Stage output must match `daydream.phases.schemas.REVIEW_STAGE_SCHEMA` exactly:
`targets`, `notes`, `candidates`, and `contradictions`. The host assigns discovery
candidate IDs and serializes terminal findings; do not request an `issues`
serializer. Read complete enclosing symbols in targeted segments. Invocation-local
complete associated native receipts govern admission independently of compact
presentation. Full retention is bounded to 2 MiB per result and 8 MiB across live
and admitted reviewer captures, including bounded metadata; overflow stays typed
incomplete evidence. Native metadata is bounded to 2 KiB, with a completion
metadata reservation charged at each start. Full receipt/pending counts are
bounded to 4,096 (8 MiB / 2 KiB), independently of the compact view's 64 pending
calls and 128 blocks. These are host memory bounds, not model call allowances.
Compact views retain the 12,000-byte per-output and
48,000-byte aggregate caps, with explicit partial/omitted markers. Supporting
inputs are classified only from revalidated prepared identities, never basenames;
mixed/opaque calls are treated conservatively. Supporting reads cannot establish
source coverage. Irrelevant failed searches alone do not invalidate complete source.
A completed unavailable Pi built-in `read` requires the supplied frozen recipe,
typed `no_extensions=True`, one unambiguous supported operand with positive integer
offset/limit, no furnished source/supporting identity, and no-follow absence.
Checkout-local operands additionally require a host-only complete strict HEAD
inventory bound to the recipe's full revision; default empty inventories and Git
errors never establish absence. Frozen-before identity does not imply current
source availability. Case/Unicode/prefix/file-URL aliases and broken symlinks stay blocking.
For `read_source`, the owned executed zero-match branch returns `isError:true`
with bounded disposition and packet digest details. The Pi adapter checks the
exact invocation bytes and owned-tool association; evidence independently counts
zero recipe matches and checks original arguments and native completion flags.
`source_tool_enabled` records the invocation-local packaged-tool request fact.
Unknown, initialization, markerless/wrong-digest, ambiguous and valid-source errors
cannot recover. Native assistant length termination binds bounded actual call IDs
to typed `ToolStartEvent.input_incomplete` before synthetic tool events; later
finish reasons cannot erase this fact. Incomplete calls supply no coverage/reuse.
Recovery is stored at matched completion before fatal source counters increment.
Failed receipts/starts remain charged; required source, furnished-pointer failures,
cancellation, truncation, capture overflow and unmatched/pending calls remain
blocking. A later good read never clears an actual source failure.
Every reviewed assertion, including an empty-candidate claim, needs complete
source evidence and explicit valid host-assigned coverage. Free-form citations
are not authenticated by receipts. Triage sees only relevant admitted partial
views and may obtain targeted rereads under the same cumulative budget.
Structure's whole-change interaction assignment may use relevant source evidence
without a file-by-file audit, but every candidate still needs complete evidence
for its own file and any terminal finding file. Retargeting requires meaningful
source reads for the published location. Missing or non-string candidate grounds are insufficient
evidence and cannot qualify for a schema-only retry.
Native capture loss, schema, identity, snapshot, grounds, and contradiction checks
remain strict. Failed or cancelled invocations admit no
output, even if they emitted valid JSON before failing; earlier successful-stage
findings survive with incomplete coverage. Every actual attempt records logical
stage, attempt, observed starts, remaining hard allowance, advisory target,
admission, safe schema rejection, retained bytes and compact clipping separately
from native truncation and host retention overflow. Diagnostics distinguish
quantitative exhaustion, schema rejection, capture loss and admission failure.
Tools-enabled native Pi may add one hidden missing-submission reminder at its
public settlement boundary after completed prose, preserving existing boundary
entries. Reminder state spans native continuations; successful finalized serializer
calls suppress it. Repeated prose remains missing output. Native validation
correction stays Pi-owned, with no host attempt or allowance reset.
Stage contract **7** invalidates older complete cache entries. ATIF is
optional recording, never the admission authority; other evidence finalization
consumers preserve their existing behavior.

#### `plan-writer` compatibility and output contract

The `plan-writer` override keeps its existing keyword-only callable contract.
In particular, `verification_commands` remains a `Sequence[str]` of literal
repository command strings so an existing override may continue to join or
render those values directly. The serialized `recon_summary` contains the full
typed recon command records, including ids, working directories, applicability,
expected results, and evidence. Override authors should use those typed records
when composing detailed command guidance; adding a required `recon_commands`
kwarg would break exact-signature builders.

A plan-writer must ask the backend to return `PlanWriterResult` structured
data, not authored Markdown. `PLAN_WRITER_CONTRACT_INSTRUCTIONS` is available
from `daydream.improve.prompts` for overrides that want to compose the built-in
typed-contract guidance. Regardless of prompt content, the host supplies its
own `PLAN_AUTHOR_SCHEMA` as the backend `output_schema` for the plan-write
call. `daydream.improve.assemble.assemble_plan` is the single validation
boundary: it validates the authored object against that schema, applies the
deterministic repairs, collects every remaining authoring defect as a pointered
`AssemblyIssue`, and only then expands the result into the host-owned assembled
plan shape that `render_plan` consumes. A wholesale prompt override cannot
replace or weaken that boundary.

Legacy override output containing `{markdown: ...}` fails closed: every
required authoring field is absent, so assembly returns one
`AUTHOR_SCHEMA_INVALID` issue per missing key, the finding is indexed as
`BLOCKED`, sanitized diagnostics record only stable metadata and error codes,
and no plan file is written. There is intentionally no Markdown-to-typed
adapter. Override authors must update their prompt to request
`PlanWriterResult`.

This compatibility repair does not bump `EXTENSION_API_VERSION` or its support
floor. The documented prompt name and kwargs are unchanged, and the legacy
`verification_commands` value shape is preserved. The authoritative output
validation is a host safety boundary, not a
fork-selectable schema. A future release that renames this kwarg or replaces it
with required typed command kwargs must follow the breaking-change version
policy above.

### Renderers

`override_renderer(name, fn)` restyles the Markdown of PR review comments
without touching `daydream/`. It mirrors `override_prompt`: the fork registers a
callable against a slot name, and `pr_review` calls it in place of the built-in
default. Two slots are registered by `register_builtins`:

| Slot | Signature | Returns |
|------|-----------|---------|
| `"finding"` | `fn(finding: CommentFinding, ctx: FindingRenderContext) -> str` | The inner human block for one finding |
| `"summary"` | `fn(ctx: SummaryContext) -> str` | The body between the approval line and the footer |

The `"finding"` renderer is invoked for every inline comment, every file-level
comment, and every finding inside the summary's by-file section;
`ctx.placement` is `"inline"`, `"file_level"`, or `"summary"` respectively, so a
fork can vary its output per placement. Its inputs:

- `CommentFinding` — the public view of one finding: `path`, `line`
  (`int | None`), `title`, `body`, `is_cross_stack`, `severity`
  (`str | None`), `confidence` (`str | None`), `fingerprint` (`str | None`).
- `FindingRenderContext` — `placement: str`.

The `"summary"` renderer receives one `SummaryContext`:

- `findings: tuple[SummaryFinding, ...]` — each `SummaryFinding` carries its
  `finding: CommentFinding` and a `body_block: str` that the host has already
  rendered (the finding marker is embedded in `body_block`).
- `agent_prompt: str` — the consolidated agent prompt (empty when there is
  nothing to fix).
- `review_info: str` — the fully-wrapped review-info `<details>` block.
- `diagrams: str | None` — the host-rendered grounded-diagram blocks (issue
  #1113), or `None` when the run rendered none. Already-folded
  `<details>` blocks containing host-generated mermaid; the model never
  authors this markdown.

#### Host-owned invariants

Renderers return only the inner content. The host owns, and always injects
around whatever a renderer returns, the parts that dedup and identity depend on:

- the per-finding dedup marker,
- the `DAYDREAM_FOOTER` trailer,
- the `---` separators and `<details>` scaffolding,
- the approval line, and
- the review `event` decision (approve / comment / request-changes).

A renderer therefore cannot drop the footer, `<details>` scaffolding, approval
line, or `event` decision. It **can** drop `ctx.diagrams`: a custom `"summary"`
renderer that never emits it silently discards the run's grounded diagrams from
the posted comment (they still land in `.daydream/deep/diagram.md` and in
`review-output.md`). Include `ctx.diagrams` verbatim, near the top of your
output, to keep them. However, **the per-finding dedup marker lives inside
`body_block`** (it is embedded by the host before `body_block` is passed to the
renderer via `SummaryFinding`). A custom `"summary"` renderer that omits
`body_block` from its output will drop those markers, causing duplicate
re-posting on the next run. Always include `body_block` verbatim in the
rendered output.

#### Fallback and warning

The call goes through a safe wrapper. If the registered renderer raises any
`Exception`, or returns a non-`str` (or empty) result, `pr_review`
**falls back** to the built-in default renderer for that slot and logs a
`logging.getLogger(__name__)` warning naming the slot (`"finding"` or
`"summary"`) and the failure. Rendering never aborts the comment build. The
built-in defaults (`default_render_finding`, `default_render_summary`) reproduce
today's Markdown byte-for-byte, so an unregistered slot and a failed override
both yield the stock output.

### Working artifact paths

During a run, Daydream's generated files live in private source-owned storage,
not in the checkout. Production runner calls and custom-flow steps run inside an
active artifact session (`ctx.artifacts`); the session freezes at finalization
and then publishes `.daydream/` and `.review-output.md` into the source
checkout.

Working contracts for extension steps:

- Write working outputs through the active artifact session, never directly to
  `<source>/.daydream` or `.review-output.md`.
- Use the paths already supplied in `ctx.data`; do not reconstruct them from
  `ctx.work.repo`, and do not retain them after the session ends.
- Before each model dispatch, Daydream rejects generated files in the model's
  working directory. A direct public write can therefore fail a later phase even
  when the write itself succeeded.
- Model inputs that come from generated files are passed as **sanctioned
  inputs**: exact named files bound to the selected backend, cwd, and read-only
  mode. Their captured bytes must remain unchanged before each dispatch attempt.
- Strict Claude audit roots, read-only Codex clones, and sandboxed Osprey
  receive the captured contents inline. Other supported modes receive the exact
  file paths. Neither transport makes an unrestricted backend's filesystem
  inaccessible beyond its cwd.
- Inline inputs have a combined limit of 12,288 UTF-8 payload bytes per model
  call. Exact-path inputs have separate validation limits: 512 files, 1 MiB per
  file, and 8 MiB combined. Missing, changed, non-regular, invalid UTF-8, or
  over-limit inputs fail before backend entry instead of being truncated.
  Exact-path validation streams and hashes the named files without retaining
  their full contents.
  Pi's exact-path `diff` input is a pointer-only durable artifact: it has a
  separate 128 MiB streaming-validation limit and does not consume the 1 MiB
  captured-file or 8 MiB aggregate allowances. It remains in the exact-file
  allowlist, is UTF-8 validated and stream-hashed again before each attempt,
  and is never copied into recovery finalization context. Its path must be the
  active session path supplied by the host. All other required inputs retain
  their existing limits; INLINE transports retain their isolation contract.
- Private storage is cwd-rooted discovery isolation, not an OS sandbox; no
  transport grants access to an entire runtime directory.
- Intentional standalone phase calls pass `allow_standalone=True` to keep
  the legacy paths without an active artifact session. Production runner and
  custom-flow calls bind a session and cannot opt out of the model-cwd check.

API v6 retains `ctx.artifacts`, the stable data keys, and verifier routing
headings. Extensions using implicit artifact lookup must update their calls
as shown below.

A complete example — write a private note, prepare it as a sanctioned input,
and dispatch one read-only agent turn:

```python
from daydream.agent import run_agent
from daydream.artifact_visibility import artifact_dir_for
from daydream.extensions import FlowStep, Registry
from daydream.flows.engine import FlowContext
from daydream.prompt_budget import prepare_sanctioned_inputs
from daydream.trajectory import DaydreamPhase, run_directory

DAYDREAM_EXT_API = 7

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

Inside the runner, pass `session=ctx.artifacts, allow_standalone=False` to
`artifact_dir_for(ctx.work.repo, ...)`. It routes to the owning session's
private directory while the session is live, so the note is
invisible to the model's cwd and to git during the run. At finalization the
whole `.daydream/` subtree is published back into the checkout (the same
contract as `review_output_path_for`: private while the session is active,
published under the checkout's untracked `.daydream/` at finalization).
`review_output_path_for(ctx.work.repo, session=ctx.artifacts,
allow_standalone=False)` routes the review-output file the same way: private
while the session is active, published as `.review-output.md` at finalization.

Both accessors require an explicit session by default, even when a session is
bound to the current task. Intentional standalone callers pass
`allow_standalone=True`; this compatibility mode may use a bound session or,
without one, the repository's public artifact path. Direct `run_deep` callers
without runner-owned artifacts also opt in with `allow_standalone=True`, and
must have no active artifact session. The recorder factory has the same
restriction. Active flows must pass their owning artifacts to these entry
points and to handoff helpers so writable and durable paths share one owner.
Direct flow contexts carry that choice in `allow_standalone_artifacts`; its
default is false. Pass `artifact_session=ctx.artifacts` and
`allow_standalone=ctx.allow_standalone_artifacts` to built-in phases that use
generated paths. Keep these runtime choices outside `ctx.data`.

For durable handoff references, use
`ctx.artifacts.durable_path_for(live_path, repo=ctx.work.repo)`; use
`ctx.artifacts.live_path_for(public_path, repo=ctx.work.repo)` for the writable destination.
These methods validate the owning session, repository, registered route, and
path ancestry. They reject unrelated paths and destinations without a live
write route, including finalization-only artifact dumps.

Place extension artifacts under the current `runs/<session_id>/` run directory,
composed with `daydream.trajectory.run_directory(root, session_id)` rather than
retyped. Legacy adoption accepts only registered top-level artifact names; an
unknown sibling is rejected even when the tree also contains a recognized
directory.
Previously published custom paths can reopen when their complete subtree
matches the validated canonical recovery copy.

### Interaction and output policy

Runner-created flows receive `ctx.run_context`, which owns a frozen policy
(`assume`, `interactive`, `quiet`, and `log_mode`) and active backend registrations.
Pass it to `run_agent` and built-in phase calls. Use its `confirm()` and `choice()`
methods for input so unattended runs take the call site's safe default.

Runner-owned GitHub authentication is available separately as
`ctx.github_execution.auth`. Pass it explicitly to GitHub helpers, for example
`git_ops.gh_api(ctx.work.repo, "/user", auth=ctx.github_execution.auth)`.
Existing extension constructors may omit `github_execution`; their GitHub calls
then inherit the live parent environment. A runner-created context carries the
credential selected for that run. Only its non-secret login is available as
`ctx.run_context.github_identity.login` and `ctx.config.identity`.

`github_execution` is a runtime capability excluded from the context's repr.
Keep it out of `ctx.data`, logs, trajectories, and artifact serializers. Explicit
authentication supplies a complete subprocess environment; helpers do not merge
it with ambient credentials.

For example, `ctx.run_context.confirm("Apply this change?", safe_default=False)`
honors a forced answer, declines unattended changes, and otherwise prompts.
`choice()` accepts explicit `assume_yes` and `assume_no` mappings for menus;
free-form input without those mappings follows the run's interactivity policy.

The runner binds this context for the run, so the shared console and API v6
extensions that omit the new argument still use the emitting run's policy.
Built-in phases also bind an explicitly supplied `run_context` for the entire
invocation, including output before and after agent execution.
Standalone callers can use `with bind_run_context(RunContext(InteractionPolicy(...)))`
from `daydream.run_context` to scope several calls together. Without an explicit
or bound context, standalone calls use a fresh interactive, non-quiet, non-log
default with no assumed answer. Binding restores the previous context on exit,
including exceptions. This additive field does not change API version 6,
`ctx.artifacts`, or the shared `ctx.data` mapping.

### Stable `ctx.data` keys

Built-in deep steps use an internal `DeepState` view over this same dictionary.
The view checks a value when a step reads it and writes back to the existing
key. It does not copy the mapping or cache its values, so extension writes remain
visible to later steps. Extensions continue to use `ctx.data` under API v6.

Steps share state through `FlowContext.data`. Forks may **read** these keys;
every other key is internal and may change without a version bump:

| Key | Meaning |
|-----|---------|
| `diff` | The diff text under review |
| `diff_path` | Path to the diff file on disk |
| `tier` | Diff-size tier driving the deep fan-out gates |
| `exploration_dir` | Exploration pre-scan output directory (or None) |
| `intent_path` | Path to the intent-analysis output |
| `alts_path` | Path to the alternatives-review output |
| `items_file` | `Path` published after `load-items`; it contains canonical `{"items": [...]}` JSON and may include top-level `held`. An extension may read this file and rewrite its `items` before downstream consumers run. |
| `items` | Parsed finding items, populated by `fix-gate` from the (potentially rewritten) `items_file`. Not present before `fix-gate` runs; rewriting `items_file` before that step is sufficient to affect all consumers. |
| `diagrams` | Issue #1113. Published by the `diagram` step as `{"blocks": str, "payload": dict, "results": dict}`: `blocks` is the rendered markdown, `payload` is `diagram.json`'s content (`{"eligibility", "results"}`) with every rendered `mermaid` string removed, and `results` is the per-kind result dict keyed by `"sequence"` / `"flowchart"`. Absent when the diagram step did not run, so read it with `.get("diagrams")`. |
| `import_graph` | Issue #1113. `{changed file: set of changed files it imports}` over the reviewed diff, or `{}` when no graph could be built (no grammar, a parse failure, or the 5s build budget). Advisory: an empty graph denies the diagram's cross-module rule and nothing else. |
| `intent_authoritative` | `bool` — `True` when a fresh, head-matched PR description with non-whitespace content grounded the intent phase; absent (hence read with `.get("intent_authoritative", False)`) on a `--start-at` resume because `_step_intent` is skipped in that case. Controls whether the deep review prompts carry the author-intent precedence rule. |

This keyword-only addition to the five in-scope prompt builders does not bump
`EXTENSION_API_VERSION` or its support floor — it is an additive kwarg per the
versioning policy above. The host passes `intent_authoritative` on every call
to `per-stack`, `structural`, `generic-fallback`, `arbiter`, and `merge`, so an
exact-signature override of one of those five prompts must add
`intent_authoritative: bool = False` (or an equivalent `**kwargs` catch-all) to
its own signature; an override that omits it raises `TypeError` and aborts that
phase's fan-out. Read `ctx.data["intent_authoritative"]` with `.get(…, False)`
to handle the absent-on-resume case correctly.

#### Host-assigned record `uid`

Per-stack review records, and the finding items behind `items` / `items_file`,
may carry a host-assigned `uid` string (`stack:ordinal`, e.g. `python:1`). It is
the record's *referential* identity — "which record object is this?" — minted at
record birth by `daydream/deep/records.py` and used by dedup, arbitration,
suppression and the per-stack records rewrite. This is an **additive** field and
does not bump `EXTENSION_API_VERSION`: a fork that round-trips whole record dicts
carries it through automatically, and a `{**item, ...}` spread preserves it.

Merged items additionally carry `source_uids`: the list of record `uid`s the
finding derives from. A merged item is a synthesis and may consolidate several
records, so its provenance is a list where a record's identity is a single
value. Read it with `daydream.deep.records.item_source_uids`, which resolves the
explicit attribution first and falls back to the item's own `uid`, so one
accessor is correct for merge-agent items, for items that bypassed the merge
agent (the single-stack path, host-appended structural records, the salvage
path), and for artifacts written before the field existed. An empty list means
the merge agent declined to attribute the item — a real answer, not an error.

Merged items also carry `item_uid` (`item:n`) — the item's **own durable
identity**, distinct from both of the above. `id` is the human-facing finding
number and `normalize_items` reassigns it to a dense `1..N` sequence by design,
so it is not a stable handle; `item_uid` is minted once and never reassigned.
This matters directly to a fork: if you rewrite `items_file` and renumber, every
`id` shifts, but `item_uid` survives the round-trip. Read it with
`daydream.deep.records.item_uid`, and preserve it when you rewrite an item —
minting a fresh one would defeat the point. It is host-minted post-validation
and is deliberately absent from `MERGED_ITEMS_SCHEMA`.

Three keys can therefore sit on one merged item, answering three different
questions: `uid` (which record it was born as — structural and single-stack
items only), `item_uid` (which shipped finding this is), and `source_uids`
(which records it was made of). Do not substitute one for another; in
particular provenance is not identity, since two items may cite the same record.

Two constraints for a fork that reads it:

- `uid` itself is a **pre-merge** handle. The cross-stack merge agent re-emits
  items from scratch, so a multi-stack merged item has no `uid` *of its own* —
  use `source_uids` for its derivation, and note that `id` is already globally
  unique on every merged item. `daydream.deep.records.record_uid` returns `""`
  for "no pre-merge identity"; after a multi-stack merge that is the common
  case, not an error.
- Never mint one yourself, and never derive one from record content. A
  content-derived key gets *less* discriminating as two records get more
  similar, which is precisely the condition every consumer of this field runs
  under. A duplicate `uid` is a fatal error the host reports and stops on.

## Recipes

All recipes go inside `register(registry)` in `daydream_ext/__init__.py`.

### Insert a phase

```python
from daydream.extensions import FlowStep

async def _my_gate(ctx):
    ...  # return None to continue, Stop(code) to end the flow

def register(r):
    r.register_phase(FlowStep(name="my_gate", run=_my_gate))
    r.insert_after("deep", anchor="intent", step="my_gate")
    # or: r.insert_before("deep", anchor="fix-gate", step="my_gate")
```

### Filter findings and supervise tools

This recipe inserts a step after `load-items` to rewrite the canonical
findings file, and registers a singleton supervisor for tool invocations:

```python
import json

from daydream.extensions import FlowStep, ToolDecision

DAYDREAM_EXT_API = 7

async def _filter_items(ctx):
    items_file = ctx.data["items_file"]
    payload = json.loads(items_file.read_text())
    payload["items"] = [item for item in payload["items"] if item["severity"] != "low"]
    items_file.write_text(json.dumps(payload))

def _supervise(name, tool_input, *, phase):
    if name == "Write":
        return ToolDecision(veto=True, reason="writes require a separate approval policy")
    return ToolDecision(veto=False)

def register(r):
    r.register_phase(FlowStep(name="filter-items", run=_filter_items))
    r.insert_after("deep", anchor="load-items", step="filter-items")
    r.register_tool_supervisor(_supervise)
```

The inserted step runs before `findings-out`, `post-review`, and the fix
consumers, so their reads observe the rewritten canonical JSON.

> **Note — preserve `payload` when inserting after `supervise`.**  The
> recipe above anchors at `load-items`, where `items_file` contains only
> `{"items": [...]}`.  If you move the anchor to after `supervise`, the
> file will already contain a top-level `held` key (items withheld by the
> supervisor).  Rewriting the file as a fresh `{"items": filtered}` dict at
> that point silently drops the held list.  Always round-trip through the
> full payload dict as shown — `payload = json.loads(...); payload["items"]
> = ...; write_text(json.dumps(payload))` — so any keys the runtime wrote
> are preserved.

### Disable a phase

```python
r.remove("deep", "arbiter")
```

### Replace a phase

```python
r.register_phase(FlowStep(name="verify", run=_my_verify), replace=True)
```

### Reorder a flow

Remove-and-reinsert individual steps, or set the whole flow at once:

```python
from daydream.extensions import LoopGroup

r.set_flow("deep", ["exploration", "intent", "per-stack-reviews",
                    "per-stack-parse", "cross-stack-merge", "load-items",
                    "supervise", "post-review", "fix-gate", "verify",
                    LoopGroup(
                        name="fix-verify-loop",
                        steps=("fix", "fix-verify"),
                        max_iterations=lambda ctx: 3,
                    ),
                    "test", "commit"])
```

Flow entries are resolved against registered phases by `run_flow`'s pre-flight
pass (and `daydream ext validate`), not at `set_flow` time, so registration
order does not matter. `insert_before` / `insert_after` / `remove` validate
their anchors eagerly.

### Selecting a flow

The built-in PR-process modes all run the `deep` flow; `--shallow` and
`--review`/`--comment` are mode gates on it, not separate flow names (#330).
The other registered flows are `diagram` (the `--diagram-only` grounded-diagram
flow) and `improve` (`daydream improve <target>`).

A newly registered flow is dispatched by name with `--flow <name>` (or
`RunConfig(flow_name=...)`):

```python
r.set_flow("ro-audit", ["ro_audit"])
# daydream --flow ro-audit /path/to/project
```

A built-in name passed to `--flow` (`deep`/`review`/`shallow`/`improve`) routes to its
dedicated helper, so behavior matches the corresponding flag. An unregistered
name errors with the same resolve check `daydream ext validate` runs.

### Add a stack

```python
from daydream.extensions import StackRule

r.add_stack(StackRule("proto", ("*.proto",)))
```

`StackRule` is routing metadata only: a stack name and changed-file patterns.
Fork rules are evaluated per changed file *before* the built-in extension table
(registration order, first match wins). Review behavior comes from the resolved
profile strategy and the registered prompt hooks.

### Override a prompt

```python
r.override_prompt("per-stack", my_builder)  # receives the exact built-in kwargs
```

Override is wholesale: the builder's return value is the whole prompt. There
is no append/compose hook (the internal suffix helpers compose into built-in
builders' outputs and are replaced along with them).

### Custom phase with its own prompt and per-phase config

```python
from daydream.extensions import FlowStep, get_registry

DAYDREAM_EXT_API = 7

def _ro_prompt(*, policy):
    return f"RO-GATE {policy}"

async def _ro(ctx):
    from daydream.agent import run_agent
    from daydream.trajectory import DaydreamPhase
    prompt = get_registry().prompt("ro_gate")(policy="read-only")
    await run_agent(ctx.backend_for("ro_gate"), ctx.work.repo, prompt,
                    phase=DaydreamPhase.REVIEW)

def register(r):
    r.register_phase(FlowStep(name="ro_gate", run=_ro))
    r.override_prompt("ro_gate", _ro_prompt)
    r.insert_after("deep", anchor="intent", step="ro_gate")
```

Per-phase model/backend/reasoning-effort config needs no extension code —
`[tool.daydream.phases.<name>]` in `pyproject.toml` or `.daydream.toml` already
accepts arbitrary phase names:

```toml
[tool.daydream.phases.ro_gate]
model = "claude-sonnet-5"
```

A fork-defined phase has no entry in the built-in `PHASE_DEFAULT_MODELS` /
`PHASE_DEFAULT_EFFORT` tables, so it skips only that tier: CLI `--model` /
`--reasoning-effort` still win, then the phase table, then the config-file
global, then the backend default. Set `model` / `reasoning_effort` on the phase table
to pin it (see the README's [Reasoning Effort](../README.md#reasoning-effort)
section for the precedence chain).

### Validate the registry

```bash
daydream ext validate
```

Loads the extension, reports its source and API version, reports whether a tool
supervisor is `registered` or `none`, resolve-checks every flow entry and stack
rule, and prints a registry summary. Broken references exit 1
naming the broken piece. Runs anywhere — no target repo needed.

## Exclusions (Version 6)

- **No backend registration.** Backends are the built-in `Backend`
  implementations (claude, codex, pi, osprey); forks cannot register new ones.
- **Backend dispatch timing is host-controlled.** A tool supervisor runs after
  `run_agent` receives a backend `ToolStartEvent`; extensions cannot move that
  check earlier or later in a backend's internal dispatch pipeline.
- **No prompt append.** Prompt override is wholesale only.
- **Parse/test/commit/setup-investigator/failure-summarizer prompts are not
  registered** — they are schema- and control-loop-coupled.
- **The built-in extension→stack table (`_EXT_TO_STACK`) is not overridable.**
  Fork `StackRule`s are additive and win per file, but built-in mappings
  cannot be modified or removed.
- **The preamble is not insertable-before.** Workspace/identity resolution,
  diff computation, trajectory-recorder setup, stack detection, and resume
  artifact checks run before any flow step; phases begin at exploration.
