# Extension contract (`daydream_ext`)

Daydream's extension seam lets a fork customize which phases run, prompts,
stack routing, tool supervision, and the canonical findings surface — entirely
from a top-level `daydream_ext` package, without editing any file under
`daydream/`. This document is the versioned
contract: the module shape daydream loads, the exact name inventories a fork
programs against, and the policy for when those names may change. A drift-guard
test (`tests/test_extension_contract_doc.py`) pins this document to the
registered inventories in the code.

Current contract version: **`EXTENSION_API_VERSION = 6`** (supported: `6..6`).

## Extension module contract

A fork creates one package next to `daydream/`:

```text
daydream_ext/
└── __init__.py
```

`__init__.py` must export exactly two things:

```python
DAYDREAM_EXT_API = 6          # must be within daydream's supported range

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
| `Registry` | Per-run extension registry |
| `StackRule` | Fork-defined changed-file-to-stack routing metadata |
| `Stop` | End a flow with an exit code |
| `SummaryContext` | Input to a `"summary"` renderer (findings, agent prompt, review info) |
| `SummaryFinding` | One finding in a `SummaryContext` (public finding plus host-rendered `body_block`) |
| `ToolDecision` | Continue or veto a tool invocation |
| `ToolSupervisor` | Callable protocol for tool supervision |
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

To migrate a fork to a new contract version: read the matching `### Changelog`
entry, adjust the registered steps, prompts, and `StackRule`/findings surface to
the new inventory, then declare the new `DAYDREAM_EXT_API` and validate with
`daydream ext validate`. A hard-breaking bump (where ceiling and floor rise
together) requires the declaration and the new surface in the same change; an
additive bump only widens the window and needs no code migration.

### Changelog

- **Version 6** — **hard-breaking**. Removes the feedback command and its deep-flow
  prefix, and removes all extension-owned agent capability selection. `StackRule`
  now contains only `stack_name` and `patterns`; review behavior comes from the
  resolved review profile and prompt hooks. The corresponding registry methods
  and prompt kwargs are gone. Because older extensions cannot run against this
  contract, both the ceiling and floor rise together: the supported range is
  `6..6`, and every fork must declare `DAYDREAM_EXT_API = 6`.
- **Version 5** — **hard-breaking**. The `review` and `shallow` flows collapsed
  into modes of the single `deep` flow, and the `review` prompt slot was removed.
  The `cleanup` step also moved out of the registered flow and into the review
  spine's success path. The supported range was `5..5` at the time.
- **Version 4** — **hard-breaking**. Removed the `alternatives` step from the
  `deep` flow when alternative review moved inside `per-stack-reviews`. The
  supported range was `4..4` at the time.
- **Version 3** — additive. Added the `improve` flow and the `audit`, `vet`, and
  `plan-writer` prompt slots.
- **Version 2** — added the synchronous tool-supervisor seam, the
  `ToolDecision` result, and the public `items_file` findings surface.
- **Version 1** — initial flow, prompt, stack, loader, and validation contract.

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

## Inventories

### Flows and steps

Three flows are registered: `deep` (the single PR-process flow), `improve`, and
`diagram` (the `--diagram-only` grounded-diagram flow).
Each step's *config key* is its `[tool.daydream.phases.<key>]` key
(`FlowStep.config_phase`, defaulting to the step name) — the key per-phase
model/backend overrides resolve against.

**Naming convention:** phase names are one global registry namespace. The `deep`
flow owns the plain names. The `review` and `shallow` flows were collapsed
into modes of `deep` (#330): `--review`, `--comment`, and `--shallow` run the
review spine, with shallow forcing a single stack. Fork-defined flows should
follow the same convention: pick globally unique step names, and use
`config_phase` to reuse an existing config key.

#### `deep` (the single PR-process flow, #330)

| # | Step | Config key |
|---|------|------------|
| 1 | `exploration` | `exploration` |
| 2 | `intent` | `intent` |
| 3 | `per-stack-reviews` | `per_stack_review` |
| 4 | `per-stack-parse` | `parse` |
| 5 | `uncovered-sweep` | `parse` |
| 6 | `arbiter` | `arbiter` |
| 7 | `cross-stack-merge` | `merge` |
| 8 | `single-stack-merge` | `single-stack-merge` |
| 9 | `load-items` | `load-items` |
| 10 | `supervise` | `supervise` |
| 11 | `diagram` | `diagram` |
| 12 | `findings-out` | `findings-out` |
| 13 | `post-review` | `post-review` |
| 14 | `fix-gate` | `fix-gate` |
| 15 | `verify` | `verify` |
| 16 | `fix` | `fix` |
| 17 | `fix-verify` | `verify` |
| 18 | `test` | `test` |
| 19 | `commit` | `fix` |

The steps are gated by the run's mode (`ctx.data["mode"]`), set in the dispatch
preamble. `review` / `comment` run the review spine and stop after `post-review`;
`shallow` forces `single_stack_mode`; `loop` is the default fixing flow;
`diagram` runs the separate `diagram` flow below over the same preamble.

`.review-output.md` cleanup is not a flow step; it runs in `_run_review_spine`
after `run_flow` returns, tied to a successful outcome, an applicable mode, and
`config.findings_out is None`.

`fix` and `fix-verify` (issue #744) are wrapped together in a `LoopGroup`
(`fix-verify-loop`) capped at 3 rounds: each round dispatches findings, a
read-only `fix-verify` step audits the round's changed hunks and returns one
verdict per finding, and actionable verdicts re-dispatch in the next round
until none remain (or the budget is spent). `fix-verify` uses the `verify`
phase config key and its own registered `fix-verify` prompt.

`per-stack-reviews` runs the TTT alternative-review (wonder) as well: on a fresh
multi-stack run the two are siblings in one task group, so wonder has no step of
its own. Its per-phase config key is still `wonder`
(`[tool.daydream.phases.wonder]`), resolved inside the step.

`uncovered-sweep` (issue #309) re-reviews diff files no per-stack reviewer read
with a cheap second-pass agent; it resolves its backend via the `parse` phase
key (the cheapest tier) and is gated off on `--start-at merge`/`fix` resumes.

`diagram` (issue #1113) decides deterministically which grounded diagram kinds
apply, runs one read-only author agent per eligible kind (plus at most one
repair turn each), verifies every proposed element against the head tree, and
writes `.daydream/deep/diagram.json` + `diagram.md`. It is gated off on a
`--start-at fix` resume and by `--diagram off` / `[tool.daydream.diagram] mode
= "off"`; otherwise it always runs and records its eligibility decision, even
when no kind is eligible (which costs zero agent calls). The rendered blocks
reach the PR summary through `ctx.diagrams` and `review-output.md` through a
`## Diagrams` section.

#### `diagram` (`daydream --diagram-only KIND <target>`)

| # | Step | Config key |
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

| # | Step | Config key |
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

The 17 registered prompt names and the exact kwargs their builders receive
(an override gets the same kwargs). All kwargs are keyword-only except where
noted.

| Prompt | Kwargs |
|--------|--------|
| `intent` | `strategy`, `diff_path`, `branch`, `log`, `exploration_dir`, `pr_description`, `inline_diff`, `inline_exploration_summary` |
| `alternatives` | `strategy`, `intent_summary`, `diff_path`, `exploration_dir`, `inline_diff` |
| `fix` | `test_output`, `feedback_items` (both positional), `repo`, `concise_mode` |
| `per-stack` | `strategy`, `stack_name`, `files`, `diff_path`, `intent_path`, `alternatives_path`, `output_path`, `cwd`, `exploration_dir`, `prior_commits`, `inline_diff`, `intent_authoritative`, `include_alternatives`, `frontier_files` |
| `structural` | `strategy`, `files`, `diff_path`, `intent_path`, `alternatives_path`, `output_path`, `cwd`, `exploration_dir`, `prior_commits`, `intent_authoritative`, `include_alternatives` |
| `generic-fallback` | `strategy`, `files`, `diff_path`, `intent_path`, `alternatives_path`, `output_path`, `cwd`, `exploration_dir`, `is_docs_only`, `prior_commits`, `inline_diff`, `intent_authoritative`, `include_alternatives`, `frontier_files` |
| `arbiter` | `strategy`, `arbiter_input_path`, `diff_path`, `intent_path`, `alternatives_path`, `cwd`, `exploration_dir`, `intent_authoritative` |
| `supervise` | `strategy`, `supervise_input_path`, `diff_path`, `intent_path`, `alternatives_path`, `cwd`, `exploration_dir` |
| `suppression` | `strategy`, `suppression_input_path`, `diff_path`, `intent_path`, `alternatives_path`, `cwd`, `exploration_dir` |
| `merge` | `strategy`, `per_stack_records_paths`, `intent_path`, `alternatives_path`, `dedup_candidates_path`, `output_path`, `exploration_dir`, `failed_stacks`, `structural_records_path`, `intent_authoritative`, `resumed_from_arbiter` |
| `verify` | `strategy`, `items`, `cwd`, `output_path` (accepted, ignored — the host writes the verdicts file) |
| `fix-verify` | `items`, `changed_hunks`, `cwd`, `round_number` |
| `diagram_sequence` | `diff_path`, `inline_diff`, `files_by_module`, `cwd`, `exploration_dir`, `schema` |
| `diagram_flowchart` | `diff_path`, `inline_diff`, `candidate_roots`, `forced`, `cwd`, `exploration_dir`, `schema` |
| `audit` | `category`, `strategy`, `group`, `scope_note`, `recon_summary`, `cwd`, `tier` |
| `vet` | `strategy`, `findings`, `cwd` |
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

DAYDREAM_EXT_API = 6

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

DAYDREAM_EXT_API = 6

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
