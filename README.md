# daydream

[![DOI](https://zenodo.org/badge/1147075973.svg)](https://doi.org/10.5281/zenodo.21614348) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/existential-birds/daydream)

Daydream is an automated code-review agent. It reviews a code change, applies fixes, and runs the test suite to validate the result. It records every agent action as a structured trajectory.

The goal of daydream is an open-weight code-review model. Daydream trains this model on the trajectory archive that it collects from its own runs. Training is a staged recipe: reward construction, then SFT cold-start, then RFT (rejection fine-tuning), then online RL. Daydream benchmarks the model against commercial code-review bots on a held-out PR replay corpus.

## Requirements

Daydream requires the following tools:

- Python 3.12.13 or newer
- [uv](https://docs.astral.sh/uv/)
- The [Claude Code](https://claude.ai/code) command line interface

The following tools are optional:

- [GitHub CLI](https://cli.github.com/) (`gh`) for PR feedback and `--comment` mode
- [Codex CLI](https://openai.com/codex) for the `codex` backend
- [Pi CLI](https://pi.dev) for the `pi` backend
- Osprey CLI for the `osprey` backend

## Quick start

Clone the repository and install the `daydream` command:

```bash
git clone https://github.com/existential-birds/daydream.git
cd daydream
uv tool install --editable .
```

This command installs a `daydream` executable in the uv tool directory. On Linux and macOS this
directory is `~/.local/bin`. If your shell shows `daydream: command not found`, the directory is not
on your `PATH`. Run `uv tool update-shell`, then open a new shell.

The `--editable` flag makes the installed command read the source from this clone. A `git pull` is
sufficient for a code change. Install the command again after a pull that changes the dependencies in
`pyproject.toml`:

```bash
git pull
uv tool install --reinstall --editable .    # necessary only for a dependency change
```

### Run without an installed command

`uv sync` does **not** make a `daydream` command on your `PATH`. It builds the project virtualenv and
writes the executable to `.venv/bin/daydream`. Use `uv run` to run daydream from the clone without a
tool install:

```bash
uv sync
uv run daydream /path/to/project
```

`uv run` is only available in the clone. In the remainder of this README, `daydream ...` and
`uv run daydream ...` are equivalent.

## Usage

Run `daydream /path/to/project` to review, fix, and test a project. The command `daydream review /path/to/project` performs the same action.

The default flow is the deep multi-stack pipeline. This pipeline performs the following stages:

1. Pre-scan the repository for imports and conventions.
2. Analyze the author intent.
3. Review each stack with the applicable review profile.
4. Review alternative approaches.
5. Resolve conflicts between findings.
6. Merge the findings across stacks.
7. Verify the findings.
8. Fix the identified issues.
9. Run the test suite natively on the local host to validate the fixes.
10. Commit and push through ordinary Git commands and repository hooks.
11. Verify GitHub CI for the exact pushed repository, PR, branch, and commit SHA.

Local verification, push verification, and remote CI are recorded as distinct
states. Remote CI polls every 10 seconds, allows 120 seconds for checks to
register, and has a 30-minute completion bound. Required checks determine the
result; advisory failures and pending checks remain visible but do not turn a
green required policy red. Daydream reports `no_ci` only after the fixed PR/SHA
has no required policy, active workflow, check run, or legacy status through
the bounded registration window. Every failed or incomplete remote result exits
1 and writes `.daydream/deep/remote-ci-handoff.json` with the next action.

Commit and push hooks always run. The local platform facts and GitHub-reported
check names in the artifacts describe only what was observed; they are not proof
of another operating system or broader test coverage.

Use the common commands for the common tasks:

```bash
daydream /path/to/project                    # review, fix, and test
daydream --comment /path/to/project          # review, then post inline PR comments
daydream --review /path/to/project           # write a report only; no fixes or PR comments
daydream --shallow /path/to/project          # review one stack in one pass
daydream --yes /path/to/project              # apply fixes without prompting
daydream --diagram-only flowchart /path/to/project       # post one grounded flowchart comment, then exit
daydream --review-profile review.toml /path/to/project   # explicit review profile
daydream -s python /path/to/project                      # force a specific stack
```

Profile precedence is: explicit `--review-profile <path>` > env `DAYDREAM_REVIEW_PROFILE` > repo-committed `file_config.review_profile` > built-in default. The `--comment` mode posts inline PR comments and exits; `--review` writes a report and exits. Neither runs the fix cycle.

The profile selects analysis settings, but backend, provider, model, reasoning effort, and safety/scoring are host-owned invariants outside the profile — they come from the host, not the profile.

Run `daydream --help` to see the common flags. Run `daydream --help-all` to see the full advanced surface.

## Observability

Send Daydream traces to LangSmith, HoneyHive, or any OTLP-compatible platform:

```bash
export LANGSMITH_API_KEY="your-key"
export LANGSMITH_PROJECT="daydream"
daydream /path/to/project --trace-to langsmith
```

Tracing uses OpenLLMetry 0.62.3 and captures runs, phases, agent attempts, tool calls,
prompts, responses, and available token/cost metadata across all four backends.
It is off by default. Use `--trace-content metadata` to omit conversation and tool
content, or `--no-tracing` to override tracing enabled in your environment.

Repeat `--trace-to` to send the same trace to several destinations. Extensions can
register additional exporters through the existing `daydream_ext` seam.
See [observability setup](docs/observability.md) for provider recipes, content handling,
and the scope of captured data.

## Audit a repository and write implementation plans

The `improve` command audits a whole repository. It verifies each candidate finding, prioritizes the findings by impact, and writes self-contained implementation plans. Every model turn runs against an independent, disposable Git snapshot outside the target and uses Claude's strict tool-root guard. Daydream writes only host-owned run artifacts under `.daydream/` and advisory plans under `daydream_plans/`. It does not modify tracked source files. These `.daydream/` paths are finalized output in the source checkout; during the run the live artifacts stay in private source-owned storage ([Output files](#output-files) describes the lifecycle).

Improve currently requires the `claude` backend for all of its model phases. Codex, Pi, and Osprey improve configurations are refused before a model executable starts until those drivers can prove an equivalent root-confinement capability. Other review flows retain support for all four backends. The independent repository prevents shared Git refs, objects, indexes, and remotes; the Claude hook separately mediates model tool access. Neither a detached worktree nor a disposable clone by itself is a filesystem sandbox, and this tool-layer policy is not an OS sandbox.

Snapshot preparation refuses inherited Git repository, external-diff, trace, executable-path, or arbitrary configuration overrides, including empty values. Well-formed indexed `commit.gpgsign` and `core.excludesFile` settings and the exact default `GIT_EXEC_PATH` that Git itself exports to hook processes are the only exceptions and pass through unchanged. Unsupported overrides fail before creating the snapshot; caller settings and hooks are never silently changed. This also applies to Codex's disposable read-only snapshots; ordinary Git commands retain their existing environment behavior.

```bash
daydream improve /path/to/project
daydream improve --effort deep --scope "apps/*" /path/to/project
daydream improve --focus security /path/to/project
```

### Effort tiers

`--effort` selects the audit breadth. It does not change the model or the reasoning effort.

| Tier | Audit coverage |
|------|----------------|
| `quick` | Correctness, security, tests, and tech debt. Serial, HIGH-confidence findings only. Cap near six. |
| `standard` | All eight categories. Concurrency ceiling of ten. This is the default. |
| `deep` | All eight categories. Concurrency ceiling of ten. Includes LOW-confidence investigation items. |

On a large repository the audit fans out over partition groups. A partition group is a bounded, stack-homogeneous slice of the tree. Each agent searches one group for one category. The `standard` tier audits at most eight groups per run. The `deep` tier is unbounded. The `quick` tier audits the whole repository as one group.

```toml
[tool.daydream.improve]
partition_max_files = 400
max_partition_groups = 8
```

The report names whatever a bound leaves out. The file `.daydream/improve/coverage.json` also names it. Coverage is never silently truncated.

### Focus modes

| Focus | Behavior |
|-------|----------|
| `security` | Audit only security |
| `performance` | Audit only performance |
| `tests` | Audit only test coverage |
| `branch` | Audit the merge-base diff. Label each finding as introduced or inherited. |

Use `--scope SERVICE_OR_GLOB` to restrict the audit to matching detected services. The glob matches a
service's root path itself, so `apps/billing` selects `apps/billing` while `apps/billing/*` matches nothing.

A directory under a conventional root (`apps/`, `services/`, `packages/`, `crates/`, `cmd/`) counts as a
service when it holds `pyproject.toml`, `package.json`, `go.mod`, `Cargo.toml`, `mix.exs`,
`requirements.txt`, `requirements.in`, `setup.py`, `setup.cfg`, or `tox.ini`. Set
`[tool.daydream.improve] service_roots` to declare roots explicitly when a repository's layout does not
match; declared roots replace discovery rather than adding to it.

### Plan subcommands

The `plan` subcommand runs reconnaissance and writes one plan for the supplied request. It does not run the category audit.

```bash
daydream improve plan "add rate limiting" /path/to/project
```

Each audit writes its report under `.daydream/improve/`. Durable output under `daydream_plans/` contains numbered plan files, an index, a rendered `README.md`, and `rejected.json`. Daydream honors the **Status** cell of a plan in `README.md`. That cell outranks `.index.json`.

### Publish plans as GitHub issues

A repository can opt into unattended issue publication:

```toml
[tool.daydream.improve.github]
publish_issues = true
```

When enabled, improve first writes and validates each plan locally. It then copies the plan into the corresponding GitHub issue. It creates no plan branch, commit, or push. A stable marker in each issue makes reruns idempotent. Daydream reconciles both open and closed issues before it creates any new issue. If reconciliation fails, publication stops.

Only one improve publisher may run against a repository at a time. Use a repository-scoped GitHub Actions concurrency group:

```yaml
concurrency:
  group: daydream-improve-${{ github.repository }}
  cancel-in-progress: false
```

## Training data

This section is for machine learning researchers. Daydream is a data-collection system. It turns every run into a labeled trajectory. You project these trajectories into JSONL datasets. You use these datasets in a staged training recipe.

### Trajectories

Daydream records every agent interaction as an [ATIF v1.7](https://www.harborframework.com/docs/agents/trajectory-format) trajectory. The trajectory records the review pipeline, the model output, the tool calls, the cost, and the result.

During a run, Daydream keeps its working artifacts outside the target checkout. After finalization, it publishes the trajectory at `<project>/.daydream/runs/<session-id>/trajectory.json` and parallel sub-trajectories in the sibling `trajectories/` directory. Daydream archives the complete run bundle at `~/.daydream/archive/runs/<session-id>/`. The bundle contains the trajectory, the manifest, the review output, the diff, and the evaluation analysis. An SQLite index at `~/.daydream/archive/index.db` supports cross-project querying.

Publication is all-or-nothing at finalization: a run that refuses publication (for example, because strict archive finalization failed) restores the checkout's prior artifacts instead of leaving a partial bundle. The archived bundle only exists once archiving has succeeded.

### Corpus commands

The data-pipeline verbs live under the `corpus` namespace:

```bash
daydream corpus harvest                              # annotate all archived runs
daydream corpus harvest --dry-run
daydream corpus build --bundle-root BUNDLE_ROOT --annotation-bundle-root ANNOTATION_BUNDLE_ROOT \
  --license-policy LICENSE_POLICY --out PROJECTION_DIR/corpus.jsonl   # project a curated bundle into a frozen-corpus projection
daydream corpus build --bundle-root BUNDLE_ROOT --annotation-bundle-root ANNOTATION_BUNDLE_ROOT \
  --license-policy LICENSE_POLICY --out PROJECTION_DIR/corpus.jsonl --dry-run    # print the projection summary, write nothing
daydream corpus label <session-id> --outcome accepted  # manual outcome override
daydream corpus calibrate-reward ...                   # deterministic reward-calibration artifact (see docs/calibration.md)
daydream corpus hydrate-hub --source-repo org/ds --source-revision <commit-sha> \
  --destination-repo org/ds --stage-dir /tmp/daydream-hydrate --dry-run
```

`corpus hydrate-hub` discovers the producer's canonical Hub layout,
`<session-id>/manifest.json` plus `trajectory.json`, and also accepts the
tested legacy `bundles/<session-id>/...` layout. Accepted sessions are
normalized into `downloads/<revision>/bundles/<session-id>/`; unrelated
top-level metadata and derived `curated/**` or `annotations/**` files are
ignored. The dry-run summary reports discovered, admitted, rejected, and
accounted candidate counts. A non-dry publication (omit `--dry-run`) also
requires `--license-policy`: the per-repo license admission gate runs at
hydration and rejected sessions are excluded before publication (issue
#1094); a dry-run may omit it. License-evidence enrichment fills legacy
bundles' missing evidence from the GitHub license API, which requires
`GITHUB_TOKEN` in the environment (read-only; env-only, never on a URL or
argv); a non-dry publication fails closed without it. If a pinned revision
contains run-shaped
manifests but no complete candidates, hydration fails with layout diagnostics
instead of reporting a successful zero/zero plan.

The pipeline has three stages:

1. **Harvest.** Walk the archive. Write one bitemporal annotation per run. Each annotation contains a label, an intrinsic reward, and a valid-at timestamp.
2. **Label.** Override the automated outcome for a run. The manual label beats the automated label.
3. **Build.** Project the per-finding annotations into a frozen-corpus projection directory (`corpus build`). Add the lineage manifest.

`calibrate-reward` validates a pinned calibration bundle and emits a deterministic, versioned reward-calibration artifact (input wire format: [docs/calibration.md](docs/calibration.md)).

`corpus adjudicate publish-state` creates an immutable recoverable checkpoint after every labeling batch. `publish-final` constructs and publishes the content-addressed per-finding annotation bundle, and `download-final` verifies its exact success commit into a fresh destination. Use `publish-final --dry-run` to validate the complete final identity without contacting the Hub. The empty-VM recovery and verified-publication procedure is in [docs/runbooks/annotation-final-publish.md](docs/runbooks/annotation-final-publish.md); downstream corpus and training operations remain separate.

The harvest stage clones the target repository into a local cache before scoring. Setting `DAYDREAM_GIT_TOKEN` (for example, a GitHub PAT with read access) authenticates clones of private repos. The token is injected out-of-band via git config environment variables (`http.extraHeader`). It is never embedded in the remote URL and never on the command line. Without the token, the harvest stage performs a plain clone via the ambient credential helper. A failed clone emits a warning and never blocks the harvest run. The token is only needed for private repos. See docs/runbooks/credential-remediation.md for operational guidance.

The build stage applies a temporal-leakage guard. It prevents future data from leaking into the past. It applies C5, C8, and C9 filters. It stratifies the corpus by stack and caps the projected stack/repository/profile shares.

### Scoring

The harvest stage scores each trajectory. The intrinsic reward is a composite:

- Correctness, weight 0.6
- Grounding, weight 0.4
- A length ramp (a penalty)

The format-valid check dominates. A trajectory that fails the format check receives no reward. Daydream records a posterior-cost axis as a sibling. It is never folded into the intrinsic reward.

### Upload to a private Hugging Face dataset

Upload of trajectories to Hugging Face is opt-in. Only the operator selects the destination. It comes from two sources, highest first:

1. The `--trajectory-hub-repo` CLI flag
2. The `DAYDREAM_TRAJECTORY_HUB_REPO` environment variable

Daydream ignores a `trajectory_hub_repo` key in the target checkout file config. When unset, nothing leaves your machine.

When set, daydream uploads every run's complete archive bundle to the dataset repo as a per-run folder. The upload requires the `huggingface_hub` package and a valid `HF_TOKEN`. If either is missing, or the upload fails, the run is never aborted. Daydream emits a one-line warning and leaves the bundle un-uploaded. On the first upload daydream creates the dataset repo private. It reuses an existing repo with its current visibility.

```sh
export DAYDREAM_TRAJECTORY_HUB_REPO="existentialbirds/daydream-trajectories"
export HF_TOKEN="hf_..."   # required for upload to proceed
```

### Training roadmap

Training follows the staged recipe from the open-weight model epic (issue #86):

| Stage | What it does |
|-------|--------------|
| 0. Reward construction | Train a reward model on the accept/reject labels. Validate it offline against held-out labels before any RL run. |
| 1. SFT cold-start | Supervised fine-tuning as a warm start. bf16 LoRA at rank 64 to 128. |
| 2. RFT | Rejection-sample against the stage-0 reward. Fold the winners back in as dense supervision. |
| 3. Online RL | GRPO against the stage-0 reward. bf16 LoRA at rank 16 to 32. |

SFT is a cold-start stage only. Online RL is the center of gravity of the recipe.

The repository contains an RL recipe and a verifiers environment:

- `rl/train/rl.toml` is a GRPO recipe. It uses prime-rl 0.7.0, LoRA rank 16, and batch size 128. The train and eval sets are two separate manifests of PR snapshots. Training uses the `pi` backend only.
- `rl/daydream_review/` is a verifiers v1 environment. One rollout is one headless deep run. The reward combines the intrinsic composite and a non-regression metric over the test suite.

The corpus paths and the base model in these files are placeholders. The real training set is tracked as issue #164. Daydream has scaffolded and validated the pipeline, but it has not yet harvested production data. The training pipeline itself is tracked as issue #91. The launch record — the measured corpora/split digests, base-model sizing rationale, hardware, wall time, and cost accounting from the validation runs — lives in [docs/training-launch.md](docs/training-launch.md).

### Evaluation

The evaluation framework has two arms:

- **Recall.** Compare daydream findings against a human gold baseline. Report inter-annotator agreement (Krippendorff alpha) and PR-level bootstrap confidence intervals.
- **Quality.** Track erosion and verbosity metrics. The fix-phase quality gate uses these metrics to flag degraded fixes.

Private PR benchmarks run on Harbor. See [docs/benchmark.md](docs/benchmark.md) for the benchmark runbook.

## Architecture

Daydream runs a deep multi-stack review pipeline. The pipeline runs exploration, intent analysis, alternative review, per-stack reviews, an arbiter pass, cross-stack merge, and recommendation verification. A `--shallow` mode reviews one stack in a single pass for simpler projects.

Daydream records and archives every run as an ATIF v1.7 trajectory. The `--no-archive` flag skips archival. A bitemporal corpus pipeline harvests, scores, and projects these trajectories into JSONL datasets.

The [project page](https://existentialbirds.com/projects/daydream) documents the full architectural details.

### Backends

Daydream supports four backends. Each implements the same `Backend` protocol and emits the same event stream:

| Backend | Driver |
|---------|--------|
| `claude` | In-process Claude agent SDK. This is the default. |
| `codex` | Codex CLI in a disposable read-only clone |
| `pi` | Pi CLI (Nous DeepSeek models) |
| `osprey` | Osprey CLI |

All four remain available to review flows. Repository-wide `improve` is intentionally stricter and currently accepts only Claude, whose SDK hook provides the required snapshot-root tool boundary.

Select a backend with `--backend`. The selection order, highest first, is:

**CLI `--backend` > config-file phase override > config-file global > built-in default.**

There is no environment-variable tier. `DAYDREAM_MODEL` and `DAYDREAM_BACKEND` are not read.

### Extensions

A fork can extend daydream. A top-level `daydream_ext` package exposes a `register(registry)` function. The function can add phases, reorder flow steps, override prompts, and register stack rules. The extension API is version 6. Verify an extension with `daydream ext validate`. See [docs/extensions.md](docs/extensions.md).

## Configuration

Configuration lives in the target repository root. Daydream reads two sources and merges them per key:

1. `pyproject.toml` under `[tool.daydream]` — lower precedence
2. `.daydream.toml` at the repository root — higher precedence

The dotfile uses bare top-level keys. It wins on scalar conflicts.

```toml
# pyproject.toml  →  [tool.daydream]
[tool.daydream]
model = "claude-opus-5"     # global default across phases
backend = "claude"          # global default backend
scope_issue_filing = false  # default; set true to file out-of-scope work as GitHub issues

[tool.daydream.phases.fix]  # per-phase override
backend = "codex"
model = "gpt-5.6-terra"
reasoning_effort = "medium"
```

```toml
# .daydream.toml  (top-level keys; no [tool.daydream] prefix)
model = "claude-opus-5"

[phases.fix]
backend = "codex"
```

The resolution order, highest first, is:

**CLI > config file (phase, then global) > built-in per-backend default.**

### Out-of-scope issue filing

`scope_issue_filing` (default `false`) opts a repository into filing out-of-scope findings and reverted out-of-scope edits as GitHub issues. By default daydream makes no GitHub writes for out-of-scope work — findings are excluded from the fix pass and out-of-scope edits are reverted regardless; only the issue filing is gated. Enable it in the target repo's config: `[tool.daydream] scope_issue_filing = true`, or per-run with `daydream --file-scope-issues /path/to/project`.

### Per-phase settings

Phase names are the flow-step config keys: `exploration`, `intent`, `wonder`, `per_stack_review`, `arbiter`, `merge`, `review`, `parse`, `fix`, `test`, `verify`, `supervise`, `diagram`, and more. Any name is accepted, including phases a fork defines.

### Reasoning effort

`reasoning_effort` is accepted as a global key and per phase. The accepted levels are `low`, `medium`, `high`, `xhigh`, and `max`. Each backend maps the level to its own knob:

| Backend | Knob |
|---------|------|
| `claude` | `ClaudeAgentOptions.effort` → CLI `--effort` |
| `codex` | `-c model_reasoning_effort=<level>` |
| `pi` | `--thinking <level>` |
| `osprey` | `--effort <level>` |

The resolution order, highest first, is:

**`--reasoning-effort` > config file (phase, then global) > built-in per-phase default.**

### Supervisor settings

Supervisor settings are config-file-only:

| Key | Default | Semantics |
|-----|---------|-----------|
| `supervisor` | `"off"` | Findings supervisor mode: `"off"`, `"rules"`, or `"llm"`. |
| `supervisor_deny_globs` | `[]` | Repository-relative globs shared by findings and tool rules. |
| `tool_supervisor` | `"off"` | Built-in tool policy mode: `"off"` or `"rules"`. |
| `tool_bash_deny` | `[]` | Regular expressions for Bash commands the policy vetoes. |

Configure the LLM supervisor model under `[tool.daydream.phases.supervise]`.

### Uncovered-diff-file sweep

A second-pass reviewer covers diff files that no per-stack reviewer read:

| Key | Default | Semantics |
|-----|---------|-----------|
| `uncovered_sweep` | `true` | Toggle the second pass. |
| `uncovered_sweep_max_files` | `10` | Cap on swept files per run. `0` sweeps nothing. |
| `uncovered_sweep_min_hunk_lines` | `5` | Minimum added/removed hunk lines to be sweepable. `0` removes the floor. |

### Quality gate

The fix-phase anti-degradation quality gate prevents a fix from degrading a file:

| Key | Default | Semantics |
|-----|---------|-----------|
| `quality_gate_enabled` | `true` | Toggle the gate. |
| `quality_gate_erosion_delta` | `0.05` | Per-file erosion-delta threshold. |
| `quality_gate_verbosity_delta` | `0.05` | Per-file verbosity-delta threshold. |
| `quality_gate_erosion_absolute` | `0.05` | Absolute post-fix erosion threshold. |
| `quality_gate_verbosity_absolute` | `0.05` | Absolute post-fix verbosity threshold. |

The gate is fail-open. A flagged file surfaces as a warning plus a manifest record. It never aborts a run. Daydream clamps the thresholds to finite non-negative numbers. An invalid value degrades to the named default.

### Diagrams

A review can post grounded mermaid diagrams — a sequence diagram, a flowchart, or both — folded into the PR summary comment and into `review-output.md`. The model never writes mermaid. It proposes a structured JSON spec in which every participant, message, node, and edge carries `file:line` evidence (plus a `symbol` where one applies). The host verifies that evidence against the head tree, and a pure renderer draws only what survived.

Diagram settings are config-file-only:

| Key | Default | Semantics |
|-----|---------|-----------|
| `mode` | `"auto"` | `"auto"` renders every eligible kind; `"off"` disables diagrams. A repo file may enable or suppress, never force a kind — `sequence`, `flowchart`, and `both` are per-invocation and CLI-only. |
| `min_code_files` | `3` | Changed-code-file floor for the sequence cross-module rule. |
| `min_modules` | `2` | Distinct-module floor for the sequence cross-module rule. |
| `min_branch_points` | `3` | Changed-branch-point floor for the flowchart rule. |
| `service_roots` | `[]` | Repository-relative globs naming service roots for participant grouping. Empty falls back to `[tool.daydream.improve] service_roots`, then to layout inference. |

```toml
# pyproject.toml  →  [tool.daydream.diagram]
[tool.daydream.diagram]
mode = "auto"
min_branch_points = 4
service_roots = ["services/*"]
```

```toml
# .daydream.toml  (top-level keys; no [tool.daydream] prefix)
[diagram]
mode = "off"
```

The diagram author's model and reasoning effort come from `[tool.daydream.phases.diagram]`.

Two flags override the file config for one run:

- `--diagram KIND` selects `auto`, `sequence`, `flowchart`, `both`, or `off` for the review paths. It is on the `--help-all` tier.
- `--diagram-only KIND` selects `auto`, `sequence`, `flowchart`, or `both` and is its own output mode: it runs the pre-scan and the diagram phase, posts one standalone PR comment, and exits. It is mutually exclusive with `--comment` and `--review`, and daydream rejects it together with `--diagram`, `--flow`, or `--start-at`.

The self-hosted bot serves the same two kinds from a PR comment: `@<bot> add sequence diagram` (alias `@<bot> add sequence`) and `@<bot> add flowchart`. Each runs a diagram-only pass over the approved head SHA and posts one standalone comment carrying no findings. A prior bot diagram comment of the same kind is minimized as outdated rather than edited.

Which diagram and why:

| Situation | Sequence | Flowchart |
|---|---|---|
| Deep review, cross-module or cross-service, no branch-heavy function | rendered | skipped |
| Deep review, single module, one function with ≥ 3 changed branch points | skipped | rendered |
| Deep review, both signals | rendered | rendered (below the sequence block) |
| `--diagram sequence` / `@<bot> add sequence diagram` | forced | skipped |
| `--diagram flowchart` / `@<bot> add flowchart` | skipped | forced (candidates become all changed functions when none meets the threshold) |
| `--diagram both` | forced | forced |
| `--diagram off` / `mode = "off"` | skipped | skipped |

A forced kind still goes through grounding and may still be omitted. Forcing changes eligibility, never verification.

**How grounding works.** Every cited path must resolve inside the repository, the file must exist at head, the line must be in range, the cited symbol must appear on that line (a ±3-line snap is attempted and recorded), and the file must carry a completed read in the diagram phase's own trajectory — an unread file is never drawn. A sequence message additionally requires its evidence file to belong to the `from` participant and, for an internal target, the callee to be defined in a `to` participant file. A flowchart node must sit inside the chosen root function's tree-sitter range; a `decision` line must be a real branch statement in that file's language, an `end` line a real return/raise/throw/panic/exit, and a `subroutine` symbol must be on the cited call-site line and defined somewhere in the repository. Ungrounded elements are pruned together with whatever depended on them, the author gets exactly one repair turn with the reason codes, the render caps are applied, and a diagram left below its floor (3 messages, 2 participants, and 1 message in a changed hunk; 4 nodes including a start, an end, and a grounded decision) is omitted with a stated reason instead of drawn thin. Every decision — eligibility signals, per-element reason codes, prune and cap counts — is recorded in `.daydream/deep/diagram.json`, and the rendered blocks in `.daydream/deep/diagram.md`.

Flowchart grounding proves that each node is a real statement of the stated kind inside the root function, and that each subroutine call exists at its call site and has a definition. It does **not** prove the arrows. Edge order is checked only for structural validity — a decision's fan-out, and both endpoints being grounded — and no control-flow graph is extracted or compared, so the sequencing of a flowchart is the model's reading of the function rather than a verified execution order. A diagram failure in a review path is fail-open: it warns and the review continues, in both the review run and the `post-findings` poster, which drops a rejected diagram payload and still posts the findings. Under `--diagram-only` the diagram is the deliverable, so a failure exits 1 after the artifact is written.

### Cost pricing

When a backend does not report a USD cost directly, daydream synthesizes the cost from token counts. The resolution order, highest first, is:

**backend-reported cost > user `prices.toml` > built-in price table > `-`.**

To override the built-in prices, create `~/.daydream/prices.toml`:

```toml
# USD per 1M tokens. User entries override built-ins per model.
[prices."gpt-5.6-sol"]
input = 4.50
cached_input = 0.45
output = 27.00
```

The `DAYDREAM_PRICES_FILE` environment variable overrides that path.

## GitHub App identity

By default, GitHub reads and writes run under the identity of the `gh` CLI. To post as a bot, supply GitHub App credentials:

```bash
export DAYDREAM_APP_ID=12345
export DAYDREAM_APP_PRIVATE_KEY="$(cat daydream-bot.private-key.pem)"  # raw PEM content
```

When both variables are set, each run mints a short-lived installation access token. Daydream attributes posts to `<app-slug>[bot]`. It displays the active identity before any GitHub action.

The behavior notes are:

- Neither variable set → ambient `gh` identity.
- Only one set → abort with an error naming the missing one.
- Posting runs abort if daydream cannot determine owner/repo, or if token minting fails.
- Daydream redacts the private key and minted tokens from logs and trajectory files.

## Self-hosted review bot

Daydream can run as a self-hosted PR review bot. It runs in your own repository's GitHub Actions and posts under your own GitHub App identity. The `daydream setup` command automates most of the install: App registration, secret deposit, and a workflow PR. Clicking **Install** on the new App stays manual, because GitHub requires it.

```bash
daydream setup /path/to/repo --repo OWNER/REPO    # one-command bot setup
daydream setup /path/to/repo --verify             # read-only install audit
```

See [docs/self-hosted-bot-setup.md](docs/self-hosted-bot-setup.md) for details.

## Non-interactive mode

`--non-interactive` runs unattended. It takes each prompt's safe default. On a test failure it writes a `handoff.md` and exits non-zero. Otherwise it declines fixes and exits zero. It is orthogonal to `--yes`: `--non-interactive` controls whether daydream may block on stdin, while `--yes` pre-decides every yes/no gate as "yes". A non-TTY or CI environment auto-enables non-interactive mode.

## Output files

These paths contain finalized output in the source checkout. Live artifacts stay under `~/.daydream/runtime/<source-key>/`; temporary linked worktrees use the separate `~/.daydream/workspaces/<source-key>/operational/` tree. Both use the source checkout's identity, including when a run uses an ephemeral worktree. Daydream does not add a symlink from the checkout to private storage.

| Path | Description |
|------|-------------|
| `.daydream/runs/<id>/trajectory.json` | ATIF v1.7 trajectory |
| `.daydream/runs/<id>/trajectories/` | Forked sub-trajectories from parallel fan-outs |
| `.daydream/diff.patch` | Unified diff captured at run start |
| `.daydream/deep/` | Deep pipeline artifacts |
| `.daydream/deep/test-verdict.json` | Native local-test result and local host facts |
| `.daydream/deep/push-verdict.json` | Session-bound ordinary push attempt and exact SHA |
| `.daydream/deep/remote-ci-verdict.json` | Bounded GitHub CI evidence for the exact pushed target |
| `.daydream/deep/remote-ci-handoff.json` | Next action for failed or incomplete remote CI |
| `.daydream/exploration/` | Cached pre-scan grounding |
| `.review-output.md` | Review findings (removed with `--cleanup`) |
| `~/.daydream/archive/runs/<id>/` | Archived run: manifest, trajectory, review output, evaluation, deep artifacts |
| `~/.daydream/archive/index.db` | SQLite index for cross-project querying |

Daydream moves existing untracked `.daydream/` and `.review-output.md` artifacts into private storage for the run. It restores or merges them during finalization. Tracked files at these artifact paths cause a preflight refusal; Daydream does not detach tracked source files. If another process changes an output, Daydream retains the competing bytes and reports a conflict instead of overwriting them. Recovery remains tied to the source checkout after temporary worktree cleanup.

An explicit `--trajectory` path outside the source checkout receives live, atomic full and partial trajectory updates. Other explicit outputs remain deferred until finalization; `--dump-artifacts` merges files without replacing the whole destination directory. External trajectory publication requires the destination filesystem to support the checked atomic operations. An unsupported destination fails before model dispatch; there is no non-atomic fallback.

Private storage prevents generated artifacts from appearing in ordinary cwd-rooted discovery. It is not an OS sandbox. When a later phase needs a generated file, Daydream passes its exact approved path or, for isolated backend modes, its bounded contents inline. It does not grant access to an entire runtime directory.

The `.daydream/exploration/` cache is reused on an exact key match. The key excludes uncommitted edits. A near-match never counts as a hit, because a stale hit would misground every review prompt. The `--shallow` and `--review` modes delete the directory. Alternating modes degrade to a cache miss, never to stale grounding.

## Development

**New to contributing? Read [CONTRIBUTING.md](CONTRIBUTING.md) — setup, commands, the required gate, and PR workflow.**

```bash
make install
make hooks      # install git hooks
make lint       # ruff linter
make typecheck  # mypy
make test       # pytest
make deadcode   # vulture dead-code scan
make coverage-report  # verify coverage.xml after make test
make check-naming  # naming-convention check
make actionlint # workflow YAML checks via Docker
make rl-check   # standalone RL: lockcheck + ruff + mypy + pytest
make check      # the required gate (rl-check is separate; run it when touching rl/)
```

`make install` runs `uv sync --all-extras`. This builds the virtualenv for the targets above. Like
`uv sync`, it does not make a `daydream` command on your `PATH`. See [Quick start](#quick-start).

`make hooks` installs two gates: a commit-time gate that runs ruff on the staged
Python files, and the pre-push gate (the hook verifies commit signatures first,
then delegates to `make check` — the quality-gate portion of that hook). A running Docker daemon
is required for `make actionlint` (the workflow YAML checks run the pinned
container); when no daemon is available that target is skipped with a note and
exits 0, so `make check` still succeeds without a daemon (CI always runs
actionlint).

Editors that support [EditorConfig](https://editorconfig.org) pick up the root
`.editorconfig` automatically (UTF-8, LF, final newline; 4-space Python, 2-space
YAML, 4-space TOML, tabs in Makefiles, preserved Markdown hard breaks).

See [docs/coverage.md](docs/coverage.md) for the coverage gate and ratchet procedure.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
