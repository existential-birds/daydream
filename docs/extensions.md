# Extension contract (`daydream_ext`)

Daydream's extension seam lets a fork customize which phases run, which skills
those phases use, the prompts, stack routing, tool supervision, and the
canonical findings surface — entirely from a top-level `daydream_ext` package,
without editing any file under `daydream/`. This document is the versioned
contract: the module shape daydream loads, the exact name inventories a fork
programs against, and the policy for when those names may change. A drift-guard
test (`tests/test_extension_contract_doc.py`) pins this document to the
registered inventories in the code.

Current contract version: **`EXTENSION_API_VERSION = 5`** (supported: `5..5`).

## Extension module contract

A fork creates one package next to `daydream/`:

```text
daydream_ext/
└── __init__.py
```

`__init__.py` must export exactly two things:

```python
DAYDREAM_EXT_API = 5          # must be within daydream's supported range

def register(registry):       # receives a daydream.extensions.Registry
    ...                       # mutate flows / skills / prompts / stacks here
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
| `DAYDREAM_SERVICE_V1` | Versioned executor/publisher contract this daydream implements |
| `MIN_SUPPORTED_DAYDREAM_SERVICE_V1` | Oldest executor/publisher contract accepted (range floor) |
| `BreakLoop` | End the current loop group and continue the flow |
| `ExtensionError` | Base error for extension failures |
| `ExtensionVersionError` | Error for an absent or incompatible extension version |
| `FlowStep` | Named async flow step |
| `LoopGroup` | Repeated ordered group of flow steps |
| `Registry` | Per-run extension registry |
| `StackRule` | Fork-defined changed-file-to-skill routing rule |
| `Stop` | End a flow with an exit code |
| `ToolDecision` | Continue or veto a tool invocation |
| `ToolSupervisor` | Callable protocol for tool supervision |
| `UnresolvedExtensionError` | Error for a missing registered name |
| `ArtifactEnvelope` | Bounded review outcome from an executor (DAYDREAM_SERVICE_V1) |
| `ExecutionRef` | Opaque execution identity (kind/version/handle/attempt) |
| `ExecutionSnapshot` | Lifecycle observation of one execution |
| `ExecutionStatus` | Neutral execution lifecycle position |
| `ExecutorCapability` | Capability an executor may declare/require |
| `ExecutorError` | Base error for the executor seam |
| `ExecutorJob` | Neutral job handed to `ReviewExecutor.start` |
| `LocalExecutor` | Hermetic filesystem/time-based conformance executor |
| `ReviewExecutor` | The versioned executor port (start/inspect/cancel/collect/release) |
| `ScriptedExecutor` | Hermetic in-memory step-based conformance executor |
| `build_registry` | Seed and load a per-run registry |
| `get_registry` | Read the current async context's registry |
| `set_registry` | Set the current async context's registry |

### Discovery order

1. `$DAYDREAM_EXT_DIR` — explicit path to the package directory (matching the
   `$DAYDREAM_SKILLS_DIR` convention; also the test seam). Daydream loads
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
- skill slot names,
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

### Changelog

- **Version 5** — **hard-breaking**. The `review` / `shallow` / `pr-feedback`
  flows are gone; they collapsed into modes of the single `deep` flow (#330).
  Their flow names, the `review` prompt slot, and the `phase:review` skill-slot
  binding no longer exist. The `deep` flow inventory gained the feedback prefix
  steps (`fetch-feedback`, `parse-feedback`, `fix-items`, `commit-push`,
  `respond-feedback`) and the mode gates (review/comment/shallow/feedback are
  `ctx.data["mode"]` values, not registered flows). Removing flow names and a
  prompt slot is a breaking change to the flow and prompt inventories, so per
  the policy above the floor rises to `5` in the same release: the supported
  range is `5..5` and every fork must declare `DAYDREAM_EXT_API = 5`. Forks
  that referenced `review`/`shallow`/`pr-feedback` flows, `--flow review`,
  `override_prompt("review", …)`, or `override_skill("phase:review", …)` must
  retarget: shallow/comment/review run the `deep` flow (replacing the per-stack
  prompt, or inserting a step anchored to a `deep` step name), and the feedback
  prefix is a mode of `deep`, not a separately registered flow.
  Also removed in this series: the `cleanup` step name is gone from the `deep`
  flow inventory — terminal `.review-output.md` cleanup now runs as a
  success-path helper invoked by the review spine after `run_flow` returns
  (gated on a zero exit, the loop/shallow/review/comment modes, and no
  `--findings-out`), not as a registered step (#335). Forks that did
  `r.remove("deep", "cleanup")` or
  `insert_after("deep", anchor="cleanup", ...)` must retarget — `cleanup` is
  no longer a step name (the step table below reflects the reduced inventory).
- **Version 4** — **hard-breaking**. The `alternatives` step is removed from
  the `deep` flow: the TTT alternative-review (wonder) now runs concurrently
  with the per-stack fan-out inside the `per-stack-reviews` step, so on a fresh
  multi-stack run the reviewers no longer wait for it. Removing a step name is a
  breaking change to the flow inventory, so per the policy above the floor rises
  to `4` in the same release: the supported range was `4..4` at the time.
  Forks that did `r.remove("deep", "alternatives")` or
  `insert_after("deep", anchor="alternatives", ...)` must retarget —
  `alternatives` is no longer a step name. `[tool.daydream.phases.wonder]`
  is unchanged: the config key survives, resolved inside `per-stack-reviews`.
  Additive in the same release (no bump of their own): the `include_alternatives`
  kwarg on the `per-stack` / `structural` / `generic-fallback` prompts, the
  `inline_diff` kwarg on the `intent` / `alternatives` prompts, the
  `resumed_from_arbiter` kwarg on the `merge` prompt, and the `verify` prompt's
  `output_path` becoming accepted-but-ignored. (Version 5 later raised the
  floor to `5`, so 4 is now aged out.)
- **Version 3** — additive. Adds the `improve` flow and its steps, the
  `audit:<category>[:<stack>]` skill slots, and three new prompt slots: `audit`
  (kwargs `category`, `skill_invocation`, `group`, `scope_note`,
  `recon_summary`, `cwd`, `tier`, where `group` is a partition-group mapping of
  `name`, `stack`, `file_count`, `partitions[{name, root, file_count, service}]`),
  `vet`, and `plan-writer`. Every one of these names is new in
  this release, so no v1 or v2 extension can have overridden them. No existing
  flow name, step name, prompt name, prompt kwarg, skill slot, or `Registry`
  method changed. The floor therefore stays at `1`: the supported range is
  `1..3` at the time. (Version 4 later raised the floor to `4`, and version 5
  to `5`, so 1-3 are now aged out.)
- **Version 2** — adds the synchronous tool-supervisor seam, the
  `ToolDecision` result, and the public `items_file` findings surface. Aged out
  by the version-4 floor raise.
- **Version 1** — initial flow, skill, prompt, stack, loader, and validation
  contract. Aged out by the version-4 floor raise.

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
Claude's always-on dangerous-command guard remains active; this configuration
adds rules and does not replace it.

Supervisor actions are `allow`, `drop`, `edit`, and `hold`. A held finding is
removed from the actionable `items` list and stored under the top-level `held`
key in `merged-items.json`; the rendered report keeps it under **Held Findings**.
All downstream readers continue to consume `items`, so held findings do not
reach findings artifacts, PR posts, or fix prompts.

Only one tool supervisor may be registered per run. If an extension registers a
tool supervisor while `tool_supervisor = "rules"` enables the built-in one, the
run fails at registry construction with a conflict error. Choose the extension
policy or the built-in policy.

## Executors and publishers (DAYDREAM_SERVICE_V1 seam)

The `Registry` also carries a *versioned executor/publisher seam* for the
executor-neutral review service (Plan 008 / issue #357). It is additive on the
extension contract — it does **not** change the meaning of `Backend` (the
model-agent driver) and does not bump `EXTENSION_API_VERSION`. The seam lets a
fork register compute/workspace adapters and trusted publishers through the
same `register(registry)` entrypoint it already uses.

```python
from daydream.extensions import LocalExecutor, ScriptedExecutor

def register(r):
    r.register_executor("local", LocalExecutor("."))
    r.register_executor("scripted", ScriptedExecutor())
    r.register_publisher("github-checks", MyTrustedPublisher())
```

`register_executor(name, executor)` enforces **capability admission at
registration**: the executor must be a conformant `ReviewExecutor` (implements
`start` / `inspect` / `cancel` / `collect` / `release` and exposes `kind` and
`capabilities`) and must declare every required capability (see
`ExecutorCapability`). `adapter_version` is not validated at registration; a
duck-typed executor missing it passes registration and fails later with an
`AttributeError` when the execution bridge reads it. A weak executor, a duplicate name, or an out-of-range
service contract version raises `ExtensionError`. Registered executors resolve
via `executor(name)`, `executor_if_registered(name)`, and `executor_names()`.

`register_publisher(name, publisher)` names a trusted publisher object
(credential-safety is the publisher leaf's contract); resolve via
`publisher(name)`, `publisher_if_registered(name)`, `publisher_names()`. Unlike
executors, publishers are not capability-admitted in this registry.

The neutral contract models, ports, capability admission, and vendor-error
mapping live in `daydream.executors.contract` / `daydream.executors.protocol`
and are versioned as `DAYDREAM_SERVICE_V1`. Any adapter registers behind the
same seam and must pass the common conformance suite
(`tests/test_executor_contract.py`); see `docs/executors/executor-registration.md`.

## Inventories

### Flows and steps

Two flows are registered: `deep` (the single PR-process flow) and `improve`.
Each step's *config key* is its `[tool.daydream.phases.<key>]` key
(`FlowStep.config_phase`, defaulting to the step name) — the key per-phase
model/backend overrides resolve against.

**Naming convention:** phase names are one global registry namespace. The `deep`
flow owns the plain names. The `review` / `shallow` / `pr-feedback` flows were
collapsed into modes of `deep` (#330) — `--review`, `--comment`, and `--shallow`
run the review spine (stopping after `post-review` for review/comment, forcing a
single stack for shallow), and `daydream feedback <pr#>` runs only the feedback
prefix. Fork-defined flows should follow the same convention: pick globally
unique step names, and use `config_phase` to reuse an existing config key.

#### `deep` (the single PR-process flow, #330)

| # | Step | Config key |
|---|------|------------|
| 1 | `fetch-feedback` | `pr_feedback` |
| 2 | `parse-feedback` | `parse` |
| 3 | `fix-items` | `fix` |
| 4 | `commit-push` | `review` |
| 5 | `respond-feedback` | `pr_feedback` |
| 6 | `exploration` | `exploration` |
| 7 | `intent` | `intent` |
| 8 | `per-stack-reviews` | `per_stack_review` |
| 9 | `per-stack-parse` | `parse` |
| 10 | `uncovered-sweep` | `parse` |
| 11 | `arbiter` | `arbiter` |
| 12 | `cross-stack-merge` | `merge` |
| 13 | `single-stack-merge` | `single-stack-merge` |
| 14 | `load-items` | `load-items` |
| 15 | `supervise` | `supervise` |
| 16 | `findings-out` | `findings-out` |
| 17 | `post-review` | `post-review` |
| 18 | `fix-gate` | `fix-gate` |
| 19 | `verify` | `verify` |
| 20 | `fix` | `fix` |
| 21 | `test` | `test` |
| 22 | `commit` | `fix` |

The steps are gated by the run's mode (`ctx.data["mode"]`), set in the dispatch
preamble: `feedback` runs only the prefix (steps 1-5, ending at
`respond-feedback`); `review` / `comment` run the review spine and stop after
`post-review` (the fix cycle is gated off, and `post-review` auto-posts for
`comment`); `shallow` forces `single_stack_mode` (no arbiter / cross-stack
merge); `loop` is the unchanged default fixing everything.

`.review-output.md` cleanup is not a flow step; it runs in `_run_review_spine`
after `run_flow` returns, tied to a successful outcome (`exit_code == 0`), the
mode gate (loop/shallow/review/comment — never `feedback`, which writes no
report), and `config.findings_out is None` (a `--findings-out` run keeps the
rendered report the user asked it to produce). It is skipped entirely on any
non-zero (failure) exit so evidence survives, and removes the report per
`--cleanup` / `--no-cleanup` / interactive-or-unattended prompt defaults (#335).

`per-stack-reviews` runs the TTT alternative-review (wonder) as well: on a fresh
multi-stack run the two are siblings in one task group, so wonder has no step of
its own. Its per-phase config key is still `wonder`
(`[tool.daydream.phases.wonder]`), resolved inside the step.

`uncovered-sweep` (issue #309) re-reviews diff files no per-stack reviewer read
with a cheap second-pass agent; it resolves its backend via the `parse` phase
key (the cheapest tier) and is gated off on `--start-at merge`/`fix` resumes.

#### `improve` (`daydream improve <target>`)

| # | Step | Config key |
|---|------|------------|
| 1 | `recon` | `recon` |
| 2 | `audit` | `audit` |
| 3 | `vet` | `vet` |
| 4 | `select-plans` | `select-plans` |
| 5 | `write-plans` | `plan_write` |
| 6 | `improve-report` | `recon` |

The improve run configuration also carries `improve_effort`, `improve_focus`,
`improve_scope`, and `improve_plan_description`.

Steps carry `enabled` predicates internally (tier gates, mode gates,
resume points); a step listed here may be skipped for a given run, but the
name is stable.

### Skill slots

| Slot | Built-in value |
|------|----------------|
| `stack:python` | `beagle-python:review-python` |
| `stack:react` | `beagle-react:review-frontend` |
| `stack:elixir` | `beagle-elixir:review-elixir` |
| `stack:go` | `beagle-go:review-go` |
| `stack:rust` | `beagle-rust:review-rust` |
| `stack:ios` | `beagle-ios:review-ios` |
| `structural` | `beagle-core:review-structure` |
| `pr-feedback-fetch` | `beagle-core:fetch-pr-feedback` |
| `pr-feedback-respond` | `beagle-core:respond-pr-feedback` |
| `audit:correctness:python` | `beagle-python:review-python` |
| `audit:correctness:react` | `beagle-react:review-frontend` |
| `audit:correctness:elixir` | `beagle-elixir:review-elixir` |
| `audit:correctness:go` | `beagle-go:review-go` |
| `audit:correctness:rust` | `beagle-rust:review-rust` |
| `audit:correctness:ios` | `beagle-ios:review-ios` |
| `audit:security:elixir` | `beagle-elixir:elixir-security-review` |
| `audit:performance:elixir` | `beagle-elixir:elixir-performance-review` |
| `audit:tests:python` | `beagle-python:pytest-code-review` |
| `audit:tests:go` | `beagle-go:go-testing-code-review` |
| `audit:tests:rust` | `beagle-rust:rust-testing-code-review` |
| `audit:tests:elixir` | `beagle-elixir:exunit-code-review` |
| `audit:tech-debt` | `beagle-core:review-structure` |

`phase:<name>` is the phase-bound slot convention: no `phase:*` slot is
registered by default, but when a fork binds one, the phase resolves its skill
from it (a custom phase reads its own `phase:<name>` slot). The built-in deep
flow binds no `phase:*` slots — the per-stack reviewer resolves its skill from
the `stack:<name>` slots (or a fork `StackRule`), never from a `phase:*` slot.

### Prompts

The 14 registered prompt names and the exact kwargs their builders receive
(an override gets the same kwargs). All kwargs are keyword-only except where
noted.

| Prompt | Kwargs |
|--------|--------|
| `intent` | `diff_path`, `branch`, `log`, `exploration_dir`, `pr_description`, `inline_diff` |
| `alternatives` | `intent_summary`, `diff_path`, `exploration_dir`, `inline_diff` |
| `fix` | `test_output`, `feedback_items` (both positional), `repo`, `concise_mode` |
| `per-stack` | `skill_invocation`, `stack_name`, `files`, `diff_path`, `intent_path`, `alternatives_path`, `output_path`, `cwd`, `exploration_dir`, `prior_commits`, `inline_diff`, `intent_authoritative`, `include_alternatives` |
| `structural` | `skill_invocation`, `files`, `diff_path`, `intent_path`, `alternatives_path`, `output_path`, `cwd`, `exploration_dir`, `prior_commits`, `intent_authoritative`, `include_alternatives` |
| `generic-fallback` | `files`, `diff_path`, `intent_path`, `alternatives_path`, `output_path`, `cwd`, `exploration_dir`, `is_docs_only`, `prior_commits`, `inline_diff`, `intent_authoritative`, `include_alternatives` |
| `arbiter` | `arbiter_input_path`, `diff_path`, `intent_path`, `alternatives_path`, `cwd`, `exploration_dir`, `intent_authoritative` |
| `supervise` | `supervise_input_path`, `diff_path`, `intent_path`, `alternatives_path`, `cwd`, `exploration_dir` |
| `suppression` | `suppression_input_path`, `diff_path`, `intent_path`, `alternatives_path`, `cwd`, `exploration_dir` |
| `merge` | `per_stack_records_paths`, `intent_path`, `alternatives_path`, `dedup_candidates_path`, `output_path`, `exploration_dir`, `failed_stacks`, `structural_records_path`, `intent_authoritative`, `resumed_from_arbiter` |
| `verify` | `items`, `cwd`, `output_path` (accepted, ignored — the host writes the verdicts file) |
| `audit` | `category`, `skill_invocation`, `group`, `scope_note`, `recon_summary`, `cwd`, `tier` |
| `vet` | `findings`, `cwd` |
| `plan-writer` | `finding`, `recon_summary`, `verification_commands`, `cwd` |

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

### Stable `ctx.data` keys

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

DAYDREAM_EXT_API = 5

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
r.set_flow("deep", ["exploration", "intent", "per-stack-reviews",
                    "per-stack-parse", "cross-stack-merge", "load-items",
                    "supervise", "post-review", "fix-gate", "verify",
                    "fix", "test", "commit"])
```

Flow entries are resolved against registered phases by `run_flow`'s pre-flight
pass (and `daydream ext validate`), not at `set_flow` time, so registration
order does not matter. `insert_before` / `insert_after` / `remove` validate
their anchors eagerly.

### Selecting a flow

The built-in PR-process modes all run the `deep` flow; `--shallow`,
`--review`/`--comment`, and `daydream feedback <pr#>` are mode gates on it,
not separate flow names (#330). The only other registered flow is `improve`
(`daydream improve <target>`).

A newly registered flow is dispatched by name with `--flow <name>` (or
`RunConfig(flow_name=...)`):

```python
r.set_flow("ro-audit", ["ro_audit"])
# daydream --flow ro-audit /path/to/project
```

A built-in name passed to `--flow` (`deep`/`review`/`shallow`/`improve`) routes to its
dedicated helper, so behavior matches the corresponding flag. `feedback` is
not selectable via `--flow` (it needs a PR number and bot identity — use
`daydream feedback`). An unregistered name errors with the same resolve check
`daydream ext validate` runs.

### Remap a built-in stack's skill

```python
r.override_skill("stack:python", "ro-python:review-python")
```

### Add a stack

```python
from daydream.extensions import StackRule

r.add_stack(StackRule("proto", ("*.proto",), "ro-proto:review-proto"))
```

Fork stack rules are evaluated per changed file *before* the built-in
extension table (registration order, first match wins), and fork-registered
stacks bypass the installed-Beagle-plugin availability check.

### Override the structural or pr-feedback skills

```python
r.override_skill("structural", "ro-core:review-structure")
r.override_skill("pr-feedback-fetch", "ro-core:fetch-pr-feedback")
r.override_skill("pr-feedback-respond", "ro-core:respond-pr-feedback")
```

### Bind a skill to a custom phase

A fork-defined step resolves its skill from a `phase:<name>` slot it binds
itself:

```python
r.override_skill("phase:ro_gate", "ro-core:gate-skill")
```

The built-in deep flow binds no `phase:*` slots — its per-stack reviewer
resolves the `stack:<name>` slot, never a `phase:*` slot — so this binding
only takes effect in a step that calls `registry.skill("phase:<name>")`.

### Override a prompt

```python
r.override_prompt("per-stack", my_builder)  # receives the exact built-in kwargs
```

Override is wholesale: the builder's return value is the whole prompt. There
is no append/compose hook (the internal suffix helpers compose into built-in
builders' outputs and are replaced along with them).

### Custom phase with its own prompt, skill, and per-phase config

```python
from daydream.extensions import FlowStep, get_registry

DAYDREAM_EXT_API = 5

def _ro_prompt(skill):
    return f"RO-GATE {skill}"

async def _ro(ctx):
    from daydream.agent import run_agent
    from daydream.trajectory import DaydreamPhase
    r = get_registry()
    prompt = r.prompt("ro_gate")(skill=r.skill("phase:ro_gate"))
    await run_agent(ctx.backend_for("ro_gate"), ctx.work.repo, prompt,
                    phase=DaydreamPhase.REVIEW)

def register(r):
    r.register_phase(FlowStep(name="ro_gate", run=_ro))
    r.override_prompt("ro_gate", _ro_prompt)
    r.override_skill("phase:ro_gate", "ro-core:gate-skill")
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
supervisor is `registered` or `none`, resolve-checks every flow entry, skill
slot, and stack rule, and prints a registry summary. Broken references exit 1
naming the broken piece. Runs anywhere — no target repo needed.

## Exclusions (Version 3)

- **No backend registration.** Backends are the built-in `Backend`
  implementations (claude, codex, pi); forks cannot register new ones.
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
