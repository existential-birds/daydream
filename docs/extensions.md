# Extension contract (`daydream_ext`)

Daydream's extension seam lets a fork customize which phases run, which skills
those phases use, and the prompts — entirely from a top-level `daydream_ext`
package, without editing any file under `daydream/`. This document is the
versioned contract: the module shape daydream loads, the exact name
inventories a fork programs against, and the policy for when those names may
change. A drift-guard test (`tests/test_extension_contract_doc.py`) pins this
document to the registered inventories in the code.

Current contract version: **`EXTENSION_API_VERSION = 1`**.

## Extension module contract

A fork creates one package next to `daydream/`:

```text
daydream_ext/
└── __init__.py
```

`__init__.py` must export exactly two things:

```python
DAYDREAM_EXT_API = 1          # must equal daydream's EXTENSION_API_VERSION

def register(registry):       # receives a daydream.extensions.Registry
    ...                       # mutate flows / skills / prompts / stacks here
```

`register(registry)` runs once per daydream run, after `register_builtins()`
has seeded the registry with everything daydream does today, so the extension
sees (and may mutate) the full built-in state through the same API the
built-ins used.

### Discovery order

1. `$DAYDREAM_EXT_DIR` — explicit path to the package directory (matching the
   `$DAYDREAM_SKILLS_DIR` convention; also the test seam). Daydream loads
   `<dir>/__init__.py` fresh on every run — never via `sys.modules` — so
   repeat runs and tests never see a stale module.
2. `import daydream_ext` — the fork extension package.
3. No extension — builtins-only registry. Absence is silent and normal.

A *present-but-broken* extension is a loud, named error before any workspace,
recorder, or agent work happens: a missing or mismatched `DAYDREAM_EXT_API`
raises `ExtensionVersionError` naming the module source path and both
versions; a missing `register`, an import failure, or an exception inside
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
- the documented stable `ctx.data` keys below.

The loader hard-rejects an extension whose `DAYDREAM_EXT_API` does not equal
the running daydream's `EXTENSION_API_VERSION` — there is no compatibility
range. Additive changes (new steps, new slots, new prompts, new optional
kwargs) do not bump the version.

### Changelog

- **Version 1** — initial contract: the four flows below as registered step
  lists; skill slots; the 10 named prompts; `$DAYDREAM_EXT_DIR` /
  `daydream_ext` discovery; `daydream ext validate`.

## Inventories

### Flows and steps

Four flows are registered: `deep`, `shallow`, `review`, and `pr-feedback`.
Each step's *config key* is its `[tool.daydream.phases.<key>]` key
(`FlowStep.config_phase`, defaulting to the step name) — the key per-phase
model/backend overrides resolve against.

**Naming convention:** phase names are one global registry namespace shared by
every flow. The deep flow owns the plain names; when another flow has a step
whose natural name is already taken, that step gets a flow-qualified name
(`review-exploration`, `shallow-fix`, …) while `config_phase` keeps the
original per-phase config key. Fork-defined flows should follow the same
convention: pick globally unique step names, and use `config_phase` to reuse
an existing config key.

#### `deep` (the default review → fix → test pipeline)

| # | Step | Config key |
|---|------|------------|
| 1 | `exploration` | `exploration` |
| 2 | `intent` | `intent` |
| 3 | `alternatives` | `wonder` |
| 4 | `per-stack-reviews` | `per_stack_review` |
| 5 | `per-stack-parse` | `parse` |
| 6 | `arbiter` | `arbiter` |
| 7 | `cross-stack-merge` | `merge` |
| 8 | `single-stack-merge` | `single-stack-merge` |
| 9 | `load-items` | `load-items` |
| 10 | `findings-out` | `findings-out` |
| 11 | `post-review` | `post-review` |
| 12 | `fix-gate` | `fix-gate` |
| 13 | `verify` | `verify` |
| 14 | `fix` | `fix` |
| 15 | `test` | `test` |
| 16 | `commit` | `fix` |

#### `shallow` (`--shallow` single-skill loop)

| # | Step | Config key |
|---|------|------------|
| 1 | `shallow-exploration` | `exploration` |
| 2 | `loop-preflight` | `loop-preflight` |
| 3 | loop group `iterate` (up to `--loop` N passes): `review` → `parse` → `shallow-fix` → `shallow-test` → `commit-iteration` | `review` / `parse` / `fix` / `test` / `fix` |
| 4 | `loop-exhausted` | `loop-exhausted` |
| 5 | `summary` | `summary` |
| 6 | `commit-gate` | `review` |

#### `review` (`--review` / `--comment` review-only)

| # | Step | Config key |
|---|------|------------|
| 1 | `review-exploration` | `exploration` |
| 2 | `review-intent` | `review` |
| 3 | `review-alternatives` | `review` |
| 4 | `emit-findings` | `emit-findings` |
| 5 | `no-issues-exit` | `no-issues-exit` |
| 6 | `post-comments` | `post-comments` |

#### `pr-feedback` (`daydream feedback <pr#>`)

| # | Step | Config key |
|---|------|------------|
| 1 | `fetch-feedback` | `pr_feedback` |
| 2 | `parse-feedback` | `parse` |
| 3 | `fix-items` | `fix` |
| 4 | `commit-push` | `review` |
| 5 | `respond-feedback` | `pr_feedback` |

Steps carry `enabled` predicates internally (tier gates, `--loop` mode,
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

`phase:<name>` is the phase-bound slot convention: no `phase:*` slot is
registered by default, but when a fork binds one, the phase resolves its skill
from it (e.g. `phase:review` feeds shallow skill resolution; a custom phase
reads its own `phase:<name>` slot).

### Prompts

The 11 registered prompt names and the exact kwargs their builders receive
(an override gets the same kwargs). All kwargs are keyword-only except where
noted.

| Prompt | Kwargs |
|--------|--------|
| `review` | `skill_invocation`, `diff_instruction`, `review_output_path`, `exploration_dir`, `prior_commits` |
| `intent` | `diff_path`, `branch`, `log`, `exploration_dir`, `pr_description` |
| `alternatives` | `intent_summary`, `diff_path`, `exploration_dir` |
| `fix` | `test_output`, `feedback_items` (both positional), `repo`, `concise_mode` |
| `per-stack` | `skill_invocation`, `stack_name`, `files`, `diff_path`, `intent_path`, `alternatives_path`, `output_path`, `cwd`, `exploration_dir`, `prior_commits`, `inline_diff` |
| `structural` | `skill_invocation`, `files`, `diff_path`, `intent_path`, `alternatives_path`, `output_path`, `cwd`, `exploration_dir`, `prior_commits` |
| `generic-fallback` | `files`, `diff_path`, `intent_path`, `alternatives_path`, `output_path`, `cwd`, `exploration_dir`, `is_docs_only`, `prior_commits`, `inline_diff` |
| `arbiter` | `arbiter_input_path`, `diff_path`, `intent_path`, `alternatives_path`, `cwd`, `exploration_dir` |
| `suppression` | `suppression_input_path`, `diff_path`, `intent_path`, `alternatives_path`, `cwd`, `exploration_dir` |
| `merge` | `per_stack_records_paths`, `intent_path`, `alternatives_path`, `dedup_candidates_path`, `output_path`, `exploration_dir`, `failed_stacks`, `structural_records_path` |
| `verify` | `items`, `cwd`, `output_path` |

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
| `items` | Parsed, merged actionable finding items |

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

### Disable a phase

```python
r.remove("deep", "alternatives")
```

### Replace a phase

```python
r.register_phase(FlowStep(name="verify", run=_my_verify), replace=True)
```

### Reorder a flow

Remove-and-reinsert individual steps, or set the whole flow at once:

```python
r.set_flow("review", ["review-intent", "review-exploration", "review-alternatives",
                      "emit-findings", "no-issues-exit", "post-comments"])
```

Flow entries are resolved against registered phases by `run_flow`'s pre-flight
pass (and `daydream ext validate`), not at `set_flow` time, so registration
order does not matter. `insert_before` / `insert_after` / `remove` validate
their anchors eagerly.

### Selecting a flow

The built-in flows dispatch through their existing flags and defaults: the
default run selects `deep`, `--shallow` selects `shallow`, `--review`/`--comment`
select `review`, and `daydream feedback <pr#>` selects `pr-feedback`.

A newly registered flow is dispatched by name with `--flow <name>` (or
`RunConfig(flow_name=...)`):

```python
r.set_flow("ro-audit", ["ro_audit"])
# daydream --flow ro-audit /path/to/project
```

A built-in name passed to `--flow` (`deep`/`shallow`/`review`) routes to its
dedicated helper, so behavior matches the corresponding flag. `pr-feedback` is
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

### Bind a skill to a phase

```python
r.override_skill("phase:review", "ro-python:review-python")
```

Shallow skill resolution precedence: `--skill` (CLI) > `phase:review` slot
(extension) > interactive menu, or the "Missing --skill" error when
non-interactive. The slot sits at the config tier of the usual
`CLI > config > default` chain.

### Override a prompt

```python
r.override_prompt("review", my_builder)  # receives the exact built-in kwargs
```

Override is wholesale: the builder's return value is the whole prompt. There
is no append/compose hook (the internal suffix helpers compose into built-in
builders' outputs and are replaced along with them).

### Custom phase with its own prompt, skill, and per-phase config

```python
from daydream.extensions import FlowStep, get_registry

DAYDREAM_EXT_API = 1

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

Per-phase model/backend config needs no extension code — `[tool.daydream.phases.<name>]`
in `pyproject.toml` or `.daydream.toml` already accepts arbitrary phase names:

```toml
[tool.daydream.phases.ro_gate]
model = "claude-sonnet-4-5"
```

### Validate the registry

```bash
daydream ext validate
```

Loads the extension, reports its source and API version, resolve-checks every
flow entry, skill slot, and stack rule, and prints a per-namespace summary.
Broken references exit 1 naming the broken piece. Runs anywhere — no target
repo needed.

## Exclusions (v1 limits)

- **No backend discovery.** Backends are the built-in `Backend`
  implementations (claude, codex, pi); forks cannot register new ones.
- **No prompt append.** Prompt override is wholesale only.
- **Parse/test/commit/setup-investigator/failure-summarizer prompts are not
  registered** — they are schema- and control-loop-coupled.
- **The built-in extension→stack table (`_EXT_TO_STACK`) is not overridable.**
  Fork `StackRule`s are additive and win per file, but built-in mappings
  cannot be modified or removed.
- **The preamble is not insertable-before.** Workspace/identity resolution,
  diff computation, trajectory-recorder setup, stack detection, and resume
  artifact checks run before any flow step; phases begin at exploration.
