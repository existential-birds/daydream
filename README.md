# daydream

[![DOI](https://zenodo.org/badge/1147075973.svg)](https://doi.org/10.5281/zenodo.21614348) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/existential-birds/daydream)

Daydream is an automated code-review agent. It reviews a code change, applies fixes, and runs the test suite to validate the result. It records every agent action as a structured trajectory.

The goal of daydream is an open-weight code-review model. Daydream trains this model on immutable JSONL run records and their observation history. Training is a staged recipe: reward construction, then SFT cold-start, then RFT (rejection fine-tuning), then online RL. Daydream benchmarks the model against commercial code-review bots on a held-out PR replay corpus.

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

Fix verification retries unresolved findings for up to three rounds. Findings
that remain unresolved are reported in `.daydream/deep/fix-outcomes.json`; the
retained changes continue through tests, commit, and push. Detected regressions
still stop publication, and commit messages list only verified fixes.

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
daydream --latency-profile fast /path/to/project          # route wonder/arbiter cheap
daydream -s python /path/to/project                      # force a specific stack
```

Profile precedence is: explicit `--review-profile <path>` > env `DAYDREAM_REVIEW_PROFILE` > repo-committed `file_config.review_profile` > built-in default. The `--comment` mode posts inline PR comments and exits; `--review` writes a report and exits. Neither runs the fix cycle.

The profile selects analysis settings, but backend, provider, model, reasoning effort, and safety/scoring are host-owned invariants outside the profile — they come from the host, not the profile.

Run `daydream --help` to see the common flags. Run `daydream --help-all` to see the full advanced surface.

## Diagnostics

Unexpected fatal errors print a concise error panel; `--verbose` additionally
writes the redacted exception chain to `stderr` (bounded at 64 KiB with a
single explicit truncation marker) and keeps the redacted agent-event stream
on `stdout`:

```bash
daydream --verbose /path/to/project >events.log 2>diagnostics.log
```

Verbose mode changes neither retries nor exit codes. `--log` was removed; use
`--verbose` instead. Verbose output is diagnostic data: credentials and
usernames are redacted, but repository paths, package paths, commands, and
Daydream stack frames may remain.

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

During a run, Daydream keeps its working artifacts outside the target checkout. After finalization, it publishes the trajectory at `<project>/.daydream/runs/<session-id>/trajectory.json` and parallel sub-trajectories in the sibling `trajectories/` directory. Daydream archives the complete run bundle at `~/.daydream/archive/runs/<session-id>/`. The bundle contains the trajectory, the manifest, the review output, the diff, and the evaluation analysis. A local SQLite index at `~/.daydream/archive/index.db` records finalized run metadata for diagnostics. Cross-project training queries read pinned JSONL snapshots through `LocalRecordStore.read_snapshot`; see [Corpus commands](#corpus-commands).

Runtime output publication is all-or-nothing at finalization: invalid run identity, frozen evidence, findings projection, or output publication restores the checkout's prior artifacts. Archive, evaluation, index, and upload failures emit a sanitized data-collection diagnostic and preserve the review result and completed outputs. The archived bundle only exists once archiving has succeeded; handoffs do not promise a manifest from optional persistence.

New runs can also collect raw JSONL evidence with `--capture-data`. Capture is off by default.
Passing `--dataset-store DIR` also enables capture and chooses private local storage
(default with `--capture-data`: `~/.daydream/dataset`). An explicit
`--trajectory-hub-repo` or `DAYDREAM_TRAJECTORY_HUB_REPO` also enables capture.
`--no-capture-data` disables collection and upload even with a configured Hub destination.
Capture preserves evidence as produced without additional redaction or secret scanning.
The user controls this local data and is responsible for reviewing it before sharing.
Collection works with `--no-archive` and tracing disabled.
It retains the original analyzed revisions and input diff, trajectories, structured claims and
verification/fix associations at the existing frozen finalization boundary, including cooperative
interruption. SIGKILL and power loss before that boundary can lose uncaptured evidence. A
persistence refusal emits a sanitized collection diagnostic and preserves completed review
outputs. Existing archives remain available; this workflow accepts newly collected JSONL data
only. Harvesting, adjudication, and frozen corpus projection consume this same record store.

### Corpus commands

The canonical evidence path is capture → `LocalRecordStore` → immutable JSONL
publication → exact-commit download → record-based harvest/adjudication → offline corpus.

```bash
daydream corpus dataset publish --trajectory-hub-repo OWNER/REPO --store ~/.daydream/dataset
daydream corpus dataset status --trajectory-hub-repo OWNER/REPO --store ~/.daydream/dataset
daydream corpus dataset download --trajectory-hub-repo OWNER/REPO --revision <40-character-commit-sha> --output /tmp/daydream-records
daydream corpus dataset snapshot --store /tmp/daydream-records --observed-before <ISO_TIMESTAMP>
daydream corpus harvest --store /tmp/daydream-records --snapshot-id <SNAPSHOT_ID>
daydream corpus label <run-id> --store /tmp/daydream-records --outcome accepted
daydream corpus dataset snapshot --store /tmp/daydream-records --observed-before <LATER_ISO_TIMESTAMP>
daydream corpus build --store /tmp/daydream-records --snapshot-id <ENRICHED_SNAPSHOT_ID> \
  --license-policy LICENSE_POLICY --out PROJECTION_DIR/corpus.jsonl
daydream corpus calibrate-reward ...  # docs/calibration.md
```

Snapshots freeze observe-time membership. An optional `--valid-before` controls
valid-time eligibility without removing later-valid observations from pinned
history. Select a new snapshot after appending enrichment or judgments; previous
snapshots remain unchanged. Publication uses the same shards, manifest, retry
queue, and conflict handling for both run records and observations. Downloads
verify every record at the requested commit. No archive conversion is performed.

Harvest retains intrinsic reward axes, outcome priors, rubric evidence, and
version pins in `daydream.observation.v2` harvest annotations. Human run labels
and per-finding judgments append independently; human decisions take precedence
and model suggestions require human review. Adjudication queues are disposable
local views of immutable evidence. See the [record dataset workflow](docs/runbooks/record-dataset.md).

Corpus projection runs offline against a selected snapshot and a pinned license
policy. It preserves license and exclusion admission, gold/silver eligibility,
trace segmentation, temporal guards, deterministic splits, and stack/repository/
profile caps. Source lineage uses `record-snapshot-v1`: snapshot membership pins
run and observation digests, and downloaded evidence also pins its HF commit.
Run identity, host finding identity, fingerprint, and training record identity
remain distinct. Derived corpus exports are separate from canonical evidence.

The harvest stage can clone target repositories into a local cache. Setting
`DAYDREAM_GIT_TOKEN` authenticates private clones through git configuration
in the child environment; it never appears in URLs or command arguments.

### Scoring

Capture and sealed RL runs read producer artifacts through the same scoring owner. Harvest reads those captured signals from immutable run records and adds separate outcome evidence. Intrinsic scoring uses verifier correctness and a length penalty:

```text
correctness = mean(consistent: 1.0, uncertain: 0.5, contradicts: 0.0)
length_penalty = clip((length - 2000) / 8000, 0, 1)
composite = round(clip(correctness - 0.2 * length_penalty, 0, 1), 4)
```

The format-valid check dominates: invalid format yields `0.0`. Missing or empty verifier verdicts leave the composite uncomputable (`None`); missing length contributes no penalty. The current reward version is `2026.10.01-1`. Grounding is no longer a reward axis. Daydream records posterior cost separately from the intrinsic reward; `w_fp = 0.3` remains a training-time combination parameter and is never subtracted here.

RFT winner thresholds support `composite`, `correctness_per_finding` (mean score), and `length_penalty`. Each threshold is a minimum; a length threshold selects at least that much penalty, not a maximum verbosity limit.

### Upload to a private Hugging Face dataset

Upload of trajectories to Hugging Face is opt-in. Runs and `corpus dataset` commands share one
operator-selected repository setting. It comes from two sources, highest first:

1. The `--trajectory-hub-repo` CLI flag
2. The `DAYDREAM_TRAJECTORY_HUB_REPO` environment variable

Daydream ignores a `trajectory_hub_repo` key in the target checkout file config. When unset, nothing leaves your machine.

When configured, Daydream captures run records in its private local JSONL store and publishes
bounded immutable shards with a manifest in one Hub commit. Run records and append-only
observations are separate datasets in the operator's chosen private repository.
Credentials alone never enable automatic uploads.
Create the destination as a private dataset before uploading; existing public repositories are rejected.
Malformed records, incompatible schemas, unsafe paths, blocking secret findings, scanner
failures, and incomplete records are refused before publication. Complete oversized records
produce an explicit error and are never truncated. Advisory scan findings may proceed.

Upload requires the optional `huggingface_hub` package and HF credentials (`HF_TOKEN`
or a cached Hugging Face login). Unavailable HF,
interrupted uploads, and uncertain responses leave evidence queued locally. Retrying checks
record and shard content identity, and commit conflicts reload the latest manifest without
replacing existing entries. The review result and completed outputs survive upload failures;
collection diagnostics contain no matched credentials or exception payloads.

```sh
export DAYDREAM_TRAJECTORY_HUB_REPO="OWNER/REPO"
export HF_TOKEN="hf_..."
daydream --review /path/to/project

daydream corpus dataset publish --trajectory-hub-repo OWNER/REPO --store ~/.daydream/dataset
daydream corpus dataset status --trajectory-hub-repo OWNER/REPO --store ~/.daydream/dataset
daydream corpus dataset download --trajectory-hub-repo OWNER/REPO --revision <40-character-commit-sha> --output /tmp/daydream-records
```

The `corpus dataset publish`, `status`, and `download` commands require an operator-selected
repository through `--trajectory-hub-repo OWNER/REPO` or `DAYDREAM_TRAJECTORY_HUB_REPO`; the flag takes
precedence over the environment variable. Without either, the command stops before accessing
the store or contacting HF. `status` reports JSON counts
for queued, published, and failed records, plus sanitized errors and the last confirmed commit.
Publication returns a failure exit code while affected evidence remains queued. Downloads
require an exact commit, verify manifest/shard checksums and schema contracts, and produce a
validated local record store directly. They never reconstruct run directories or an archive
index. A downloaded store can be read through `LocalRecordStore.select_snapshot` and
`read_snapshot`, preserving temporal eligibility and observation precedence.

Local evidence preserves raw producer content. The credential scanner recognizes limited
patterns; a passing scan does not establish that content is safe to share. Training admission
continues to apply its separate license and eligibility policies.

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

Daydream records and archives every run as an ATIF v1.7 trajectory. The `--no-archive` flag skips archival. Captured run records feed a bitemporal harvest, adjudication, and frozen corpus pipeline; local archives remain available for diagnostics.

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

Reviewer tool telemetry does not determine review completeness or diagram validity. Failed or budget-limited reviewers produce partial-result warnings; validated findings survive recovery. Diagram citations are checked directly against repository source.

Select a backend with `--backend`. The selection order, highest first, is:

**CLI `--backend` > config-file phase override > config-file global > built-in default.**

There is no environment-variable tier. `DAYDREAM_MODEL` and `DAYDREAM_BACKEND` are not read.

### Extensions

A fork can extend daydream. A top-level `daydream_ext` package exposes a `register(registry)` function. The function can add phases, reorder flow steps, override prompts, and register stack rules. The extension API is version 8. Verify an extension with `daydream ext validate`. See [docs/extensions.md](docs/extensions.md).

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

### Latency profiles

Deep runs choose wonder and arbiter spend through a named latency profile. Set it
per run with `--latency-profile <name>`, or for a repository:

```toml
# pyproject.toml  →  [tool.daydream]
[tool.daydream]
latency_profile = "fast"

# .daydream.toml  (top-level keys; no [tool.daydream] prefix)
latency_profile = "fast"
```

| Name | Wonder | Arbiter | Sharding |
|------|--------|---------|----------|
| `fast` | skip | `medium` | on |
| `balanced` (default) | `medium` | `high` | on |
| `forensic` | `high` | `xhigh` | off — today's behaviour |

A profile sets an effort *floor*, never a ceiling: a security-, concurrency-,
persistence-, interface-, or migration-shaped diff raises the route, so a
sensitive diff is never routed cheap. An unrecognised name **fails safe upward**
to `forensic` and the run records the fallback — it never routes cheaper than the
default. A deliberate `--reasoning-effort` pin still outranks the profile.

The route selects effort only on Codex, the one backend whose deep-review phases
use the built-in effort table; on Claude and Pi it still selects wonder and
arbiter scheduling while their effort stays the backend default. A selection
that fits one group is unsharded and keeps the pre-profile arbiter effort
(Codex `xhigh`, the Claude/Pi ambient default) whatever the profile: the
route's arbiter effort is a per-group knob. Each run writes
its decision to `.daydream/deep/latency-routing.json`, and the archived
`evaluation.json` carries the selected profile. Compare profiles with:

```sh
uv run python -m daydream.eval.latency_report --corpus tests/fixtures/latency_profiles/manifest.json
```

The report covers the recorded corpus only; its arbiter latency includes the
whole deep phase.

The corpus's `selection_cases` produce a `verify_selection` block comparing the
conservative verifier (`verify_all`) with the selective mode. It reports, per mode: the selected item
and backend-call counts (a mode with no selected item makes no call), how many
the mode skips, how many skipped items the archived arm had verdicted
`contradicts` or `uncertain` (the offline counterfactual), the fraction of each
case's golden high-severity anchors the mode still verifies, and the reverted or
failed fix count from `fix-outcomes.json`. The proposed arm's latency scales the
archived measured verify wall-clock by the selected-item ratio and is labelled a
projection, never a second measurement. `flip_allowed` turns on the
contradiction counter and the recall anchor alone -- latency is reported, not
gated.

### Test recipe and evidence reuse

A run resolves the project's test command and environment once, in the deep
preamble, and writes the result to `.daydream/deep/test-recipe.json`. The same
resolved facts feed the host test run, the pre-push hook run, and every agent
prompt, so no phase re-discovers them.

`test_required_suites` declares the suite ids the single configured
`test_command` is the authoritative gate for. It is declaration-only: there is
no second runner, and a narrowed `-k`/file check can never satisfy the required
contract.

```toml
# pyproject.toml  →  [tool.daydream]
[tool.daydream]
test_command = "uv run pytest"
test_required_suites = ["python", "integration"]
```

A green host run may stand in for a fresh validation at a later gate only when
the whole typed execution identity still matches — command, package cwd, runner,
interpreter, config-input digest, tree key, and revision. Across a commit the
tree key alone is not enough: the created commit must have passed the strict
post-commit verification. Either way the gate prints a line naming the decision
(`reused matching evidence` or `ran real validation (<result>: <component>)`)
and records it in `.daydream/deep/evidence-reuse.json`, keyed by gate. Reuse
never replaces the pre-push hook, the post-hook strict check, or the push
receipt check.

### Recommendation verifier settings

Selection-gated recommendation verification is the default: the verifier is
rendered only the findings that genuinely need an independent second pass
(mandatory risk categories, contested or weakly-evidenced adjudications, and
unadjudicated findings). The two knobs are config-file-only:

| Key | Default | Semantics |
|-----|---------|-----------|
| `verify_all` | `false` | `true` restores the conservative mode exactly: every non-exempt finding is verifier-rendered, and no finding is selection-skipped. |
| `extra_risk_categories` | `[]` | Validated against the mandatory risk-category vocabulary, which is also the whole declared vocabulary today. It cannot widen or narrow selection: a name that is already mandatory is a no-op, and an unknown name fails the run. |

```toml
# pyproject.toml  →  [tool.daydream]
[tool.daydream]
verify_all = false                 # the default; true restores today's conservative verifier
extra_risk_categories = ["security"]   # `security` is already mandatory: validated, no-op

# .daydream.toml  (top-level keys; no [tool.daydream] prefix)
verify_all = false
extra_risk_categories = ["security"]
```

Precedence is the standard one: **CLI (none for these keys) > config file > built-in
default**. An absent key uses the built-in default; `verify_all = true` in
either file restores conservative verification exactly. An unrecognised
`extra_risk_categories` entry **fails the run loudly** before the verify pass
rather than silently widening or narrowing selection. The mandatory category
vocabulary is the one shared with the diff-routing risk floors (`security`,
`concurrency`, `persistence`, `public-interface`, `migration`).

The default was flipped only after the evidence gate above went green: the
report command
`uv run python -m daydream.eval.latency_report --corpus tests/fixtures/latency_profiles/manifest.json`
emits `flip_allowed: true` on the contradiction-counter and recall-anchor axes
(its latency figures are a labelled projection, reported but not gated).

### Supervisor settings

Supervisor settings are config-file-only:

| Key | Default | Semantics |
|-----|---------|-----------|
| `supervisor` | `"off"` | Findings supervisor mode: `"off"`, `"rules"`, or `"llm"`. |
| `supervisor_deny_globs` | `[]` | Repository-relative globs shared by findings and tool rules. |
| `tool_supervisor` | `"off"` | Built-in tool policy mode: `"off"` or `"rules"`. |
| `tool_bash_deny` | `[]` | Regular expressions for Bash commands the policy vetoes. |

Configure the LLM supervisor model under `[tool.daydream.phases.supervise]`.

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

### Review budgets

The review model pipeline defaults to 45 minutes for modest diffs, including
queueing and retries, with the last five minutes reserved for synthesis. Diffs
over 1,000, 5,000, and 10,000 lines (or 64, 256, and 512 KiB) receive 2×, 4×,
and 6× review time and per-role tool-call allowances respectively. The 4× and
6× tiers also enable deep-review sharding unless explicitly disabled. An explicit review
profile keeps its configured whole-review deadline. Set
`pipeline.review_wall_budget_s` in a `--review-profile` TOML file to change it.
The nonstructural role cap coarsens the largest packed stacks by repeatedly merging adjacent groups with the smallest
combined diff-byte weight (leftmost on ties). Final shard names are contiguous and frontiers are recomputed. File/byte
targets are soft under the cap; files stay indivisible, each stack retains at least one role, and whole-change Structure
is outside the cap. More groups increase aggregate model opportunity: the captured 81-file Python workload grows from
one to eight groups, 288 to 2,304 maximum starts, and 48 to 384 nominal role-minutes. Shared pipeline time, queueing and
concurrency still limit execution; these are neither billing estimates nor guarantees of cold review completion.

Per-stack reviewers share eight minutes and 48 observed tool starts across stages and retries. Language/generic
assignments prefer nearby complete files, at most four per batch, within 24,576 bytes for exact paths or 12,288 bytes
inline, including wrappers, scoped diff/index and snapshot bindings. Exact assignment pointers do not consume the
separate inline allowance for shared bytes and their wrappers. Oversized files split into hunks and ordered
continuations with old/new ranges and exact fragment offsets; every required part must succeed for its file to be
complete. Required assignments take priority over separately bounded context.

Each stage gets a fresh prompt. Structure reviews whole-change interactions using a compact inventory, bounded diff
parts and supporting documentation; discovery candidates marked `open` or `unresolved` receive one finite triage round.
A candidate still `unresolved` after triage keeps coverage incomplete. Call targets guide pace, while the cumulative
allowance and absolute deadline remain hard limits. For native Pi stages, the host withholds minimum input-read and
submission capacity for undispatched assignments and submission capacity for known pending triage. Each invocation
has a hard allowance within that reservation and may borrow above its advisory target only within that allowance.
The host stops before dispatch if the current stage's minimum input-read and submission needs cannot fit. Reservations
do not guarantee model completion: observed tool starts can arrive after execution, and a stage that spends its own
allowance without submitting remains incomplete. Native Pi supplies bounded live feedback on invocation-local tool
starts before each model request, including failed tools and submissions; every member of a parallel batch counts.
This improves pacing without changing tool admission or the recorded provider proposals. One fresh full-stage retry
is permitted after a normally completed text-mode
invocation fails strict schema validation and passes the domain rejection guard; only safe validator metadata
carries over, never rejected output. Syntax, envelope, ambiguity and native-output failures are terminal. Other
admission failures are terminal. Failed attempts admit no semantic state, and later failure preserves findings from
successful stages while marking unfinished work incomplete.

Optional Structure projections and their navigation catalogs share the remaining exact-input allowance
(512 files and 8 MiB) after required/shared context. Catalogs list admitted pointers only; partial status and
omitted-part counts describe unavailable supporting context without aborting the interaction review.

Valid decisions can rely on supplied diff/context without source reads; ordinary file/Git tools remain available,
recorded and charged, but do not authenticate claims or establish completion. Findings and typed coverage are
snapshot-bound and atomically published. `complete` requires success or explicit host no-op for every planned scope and
required phase; it does not guarantee exhaustive defect discovery. The
[stage contract](docs/extensions.md#stage-aware-review-builders-api-8) specifies identity/capture checks, triage state,
backend transports and native output rules. Stage contract 9 invalidates older cached reviews; extension API 8 requires
staged builders.

The host resolves only the final assistant turn. Staged text accepts one complete schema-valid result with ordinary
prose, a closed JSON Markdown fence, or one transparent object wrapper (one property, no expected schema root fields).
Complete unrelated JSON is ignored. Competing results, malformed or truncated envelopes/tails and unfinished fences
are rejected without recovering nested fragments. Payload and assignment/domain validation still govern admission;
accepted host results are presented once, while raw provider events remain trajectory evidence.

When a review agent exhausts its time or tool-call budget, Daydream continues with
completed reviewers' findings and validated partial checkpoints, and marks the
report **Review incomplete**. If the
merge agent times out, the host consolidates surviving stack records. Incomplete
reviews still produce a `--findings-out` artifact and exit successfully; the poster
publishes a comment even when there are no findings. It never approves an
incomplete review or resolves prior findings absent from that partial result.
Optional arbiter, suppression, and supervisor budget stops also mark the review
incomplete and preserve each stage's existing missing-verdict policy. Ordinary
backend errors and malformed merge responses retain their failure behavior.

The findings artifact carries optional `review_warnings`. Update both the analyze
and posting jobs to the same Daydream revision. Leave CI headroom beyond the
model deadline for setup, host processing, and artifact publication; a 60-minute
job with the default model budget leaves 15 minutes for those operations. An
external job cancellation cannot use the graceful budget-exhaustion path.
For diagnosis, capture `--trajectory PATH` and `--dump-artifacts DIRECTORY`.
Inspect diagnostic bundles for credentials before sharing them.

### Retry recovery

A retry may spend only the recovery budget it was given, never the invocation's useful-work time. The allowance is a **cumulative retry-overhead budget**, measured in seconds. It is **additional to the invocation deadline** (the per-turn wall budget), it starts at the **first retryable failure** of an invocation, and it then charges every backoff sleep plus the backend time of every retry against itself. A spent allowance re-raises the current failure without dispatching again. Because it only bounds retry overhead, it **never caps** an otherwise healthy invocation: a 300 s allowance does not shorten a healthy 1800 s fix turn, since no retryable failure ever activates it. The retry decides its delay from a server `Retry-After` hint when one is present, otherwise from full jitter whose per-failure cap is the smaller of the configured maximum delay and the remaining allowance/deadline. The hint is read from a status-prefixed JSON provider error's `metadata.headers.Retry-After` (numeric seconds only; the header name is matched case-insensitively), and a date-formatted header yields no hint and falls back to bounded jitter. An admitted hint is waited **in full** and is never shortened by the exponential cap or the configured maximum jitter delay: it is evaluated against the remaining recovery allowance and invocation deadline only, and a hint longer than either stops the retry ladder with the existing insufficient-budget reason rather than being clamped. The allowance is **clamped** by the invocation deadline and by the fix file-group budget rather than re-basing either one.

| Key | Default | Semantics |
|-----|---------|-----------|
| `retry_recovery_allowance_s` | `300` | Cumulative retry-overhead budget, in seconds, for one invocation. `0` disables retry recovery: the first retryable failure ends the ladder, without disabling attempts. The default is not yet tuned against outage data. |
| `group_max_wall_s` | `600` | Per-file-group wall-clock ceiling for the fix phase. The allowance composes with it by clamping: a retry can never spend past the group's remaining wall time. |

The same allowance is configurable per run through environment variables. The retry ladder's other knobs are env-only:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DAYDREAM_PI_RETRY_ATTEMPTS` | `20` | Retry attempts for a backend that declares no `RetryPolicy`. |
| `DAYDREAM_PI_RETRY_BASE_DELAY_S` | `10.0` | Base of the exponential backoff for pi (`2.0` for claude/codex). |
| `DAYDREAM_PI_RETRY_MAX_DELAY_S` | `120.0` | Maximum delay a single backoff may reach. |
| `DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S` | `300` | Cumulative retry-overhead budget, in seconds. Applies on the plain-CLI path directly and on the embedded path through the `RetryPolicy` built by `BackendExecutionInput.from_environment`. |

Resolution precedence, highest first, is: a backend `RetryPolicy.retry_recovery_allowance_s`, then a backend `retry_recovery_allowance_s` attribute, then the explicit argument (only the fix phase threads a config-file value), then `DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S`, then the `300` default. A backend that declares a `RetryPolicy` declares its retry settings *completely*: no ambient `DAYDREAM_PI_RETRY_*` value is consulted for it. The embedded/benchmark construction path is not excluded from the operator knob — `BackendExecutionInput.from_environment` materialises `DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S` into the `RetryPolicy` it builds, so the env value reaches those runs through the top precedence tier (an invalid value warns and stays undeclared).

That materialisation makes the effective winner for a pair of declared sources path-dependent, deliberately and visibly: on the **plain-CLI path** a pi run declares no `RetryPolicy`, so the ambient variable sits at the bottom tier and a repo `retry_recovery_allowance_s` value — threaded as the explicit argument — wins; on the **embedded/benchmark path** the same variable is materialised at the top tier, so it beats the repo value. Declare one source when a single effective allowance matters, and read the allowance the run actually used from its retry telemetry.

The unit is seconds. An invalid value degrades to the default with a warning, never silently becoming an effective bound. Contradictory combinations are refused before dispatch — a non-zero allowance with retries disabled (`retry_recovery_allowance_s > 0` alongside `retry_attempts = 0`), or a base delay above the maximum delay. Permanent failures — authentication, schema, and tool-policy vetoes — retry zero times regardless of the allowance, and ladder exhaustion still surfaces the last failure unchanged.

```toml
# pyproject.toml  →  [tool.daydream]
[tool.daydream]
retry_recovery_allowance_s = 120

# .daydream.toml  (top-level keys; no [tool.daydream] prefix)
retry_recovery_allowance_s = 120
```

### Repair job budgets

When a test run fails, the test phase may dispatch bounded **repair turns** to
diagnose the failure and edit the authorized scope. Those turns form a *repair
job*: each interrupted turn leaves authorized work on the tree, a durable
checkpoint of that work, and a record of what the job has already spent, so the
next bounded execution resumes instead of starting over. The job's own bounds are
configurable:

| Key | Default | Semantics |
|-----|---------|-----------|
| `repair_execution_wall_s` | `1800.0` | Wall-clock ceiling, in seconds, for **one** repair execution. Reaching it ends the turn mid-work; the partial edits are kept, recorded, and carried into the job's checkpoint rather than verified. |
| `repair_job_wall_s` | `7200.0` | Cumulative wall-clock ceiling, in seconds, across **every** execution of one job. A 60-second reserve is held back from this total so the final validating execution has somewhere to run; the reserve is not operator-configurable. |
| `repair_max_executions` | `4` | How many bounded executions one job may run. An execution that repeats the same candidate against the same failure is charged as no progress and ends the job, so this is a ceiling and not a retry count. |
| `repair_grant_job_wall_s` | `0` | An operator's **additional finite** allowance for a job that ran out of time. It applies only to an *exhausted* job, never to one that went blocked, and it is recorded beside — never in place of — the seconds already spent. |

**These three values are the issue's proposals adopted as configurable
defaults, not measured optimal values.** No measurement of real repairs produced
1800/7200/4, and they are not tuned against outage data. Raising the total
allowance above a single execution's ceiling is a deliberate, reviewable policy
change rather than a tuning detail: a bounded second attempt is the entire point
of a repair job, so a job total equal to one execution would allow exactly one.
A repository that wants the previous behavior back sets
`repair_max_executions = 1` — or bounds `repair_job_wall_s` — explicitly.

The bounds are resolved once, when the job starts, and **stored in the job
record**. A resumed job therefore reads its own stored policy and can never
quietly inherit a broader one than it was granted. Only consumed seconds are
persisted, never an absolute clock reading, so every deadline is reconstructed
process-locally and a record written by an earlier process stays meaningful.

The job never runs unbounded: an execution without a *discriminating* result (a
completed experiment, or a candidate patch a focused test supports) is recorded as
no progress with the unchanged evidence named, and the job ends `blocked` or
`exhausted` — keeping its checkpoint and its evidence, and never committing,
pushing, or repeating. Only a `completed` job may report a passing verdict or
authorize a commit. A job a second process finds already owned does not start a
second worker; that run warns and continues with the evidence it has.

```toml
# pyproject.toml  →  [tool.daydream]
[tool.daydream]
repair_execution_wall_s = 1800.0
repair_job_wall_s = 7200.0
repair_max_executions = 4
repair_grant_job_wall_s = 0

# .daydream.toml  (top-level keys; no [tool.daydream] prefix)
repair_execution_wall_s = 1800.0
repair_job_wall_s = 7200.0
repair_max_executions = 4
repair_grant_job_wall_s = 0
```

Budgets and counts accept only finite non-negative values and preserve zero, so
`repair_max_executions = 0` legitimately grants no execution. An invalid value
degrades to the named default; it never silently becomes a different effective
bound.

### Diagrams

A review can post grounded mermaid diagrams — a sequence diagram, a flowchart, or both — folded into the PR summary comment and into `review-output.md`. The model never writes mermaid. It proposes a structured JSON spec in which every participant, message, node, and edge carries `file:line` evidence (plus a `symbol` where one applies). The host verifies that evidence against the head tree, and a pure renderer draws only what survived. PR comments show the diagrams without evidence tables or audit captions.

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

**How grounding works.** Generation checks repository paths, source lines, symbols, and definitions directly against the source checkout. Sequence calls must match their participants; flowchart nodes must belong to the chosen function and match their statement kinds. Unsupported elements and their dependents are pruned after one repair turn, then render caps and minimum diagram sizes apply. Diagrams below their minimum size are omitted. `.daydream/deep/diagram.json` stores eligibility, final specs, and omission reasons; `diagram.md` stores the rendered blocks. Posting validates the final specs and immutable head source, then renders safe Mermaid. Per-element audit reports and proposed-spec copies are not persisted.

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
| `.daydream/deep/test-verdict.json` | Native local-test result, per-repair-turn records, and local host facts |
| `.daydream/deep/repair-job.json` | Repair job state: bound, consumption, execution count, and last transition reason |
| `.daydream/deep/repair-checkpoint.json` | The interrupted repair's authorized work, captured before any restoration |
| `.daydream/deep/push-verdict.json` | Session-bound ordinary push attempt and exact SHA |
| `.daydream/deep/evidence-reuse.json` | Per-gate reuse decision: reused or revalidated, with the deciding component |
| `.daydream/deep/remote-ci-verdict.json` | Bounded GitHub CI evidence for the exact pushed target |
| `.daydream/deep/remote-ci-handoff.json` | Next action for failed or incomplete remote CI |
| `.daydream/exploration/` | Cached pre-scan grounding |
| `.review-output.md` | Review findings (removed with `--cleanup`) |
| `~/.daydream/archive/runs/<id>/` | Archived run: manifest, trajectory, review output, evaluation, deep artifacts |
| `~/.daydream/archive/index.db` | Local diagnostic index of finalized run metadata |

Daydream moves existing untracked `.daydream/` and `.review-output.md` artifacts into private storage for the run. It restores or merges them during finalization. Tracked files at these artifact paths cause a preflight refusal; Daydream does not detach tracked source files. If another process changes an output, Daydream retains the competing bytes and reports a conflict instead of overwriting them. Recovery remains tied to the source checkout after temporary worktree cleanup.

An explicit `--trajectory` path outside the source checkout receives live, atomic full and partial trajectory updates. Other explicit outputs remain deferred until finalization; `--dump-artifacts` merges files without replacing the whole destination directory. External trajectory publication requires the destination filesystem to support the checked atomic operations. An unsupported destination fails before model dispatch; there is no non-atomic fallback.

`--dump-artifacts DIR` always copies the finalized bundle's exact assembled bytes, including credential-shaped strings and binary files, without running the archive scanner or sanitizer. It preserves bundle layout, manifest session ID and unrelated destination files, and works with `--no-archive`. Byte preservation does not undo upstream trajectory redaction. Copy/I/O and archive-integrity failures retain the existing fatal finalization behavior and rollback.

Review diagnostic content before sharing; diagnostic dumping preserves the assembled evidence.

The canonical JSONL publisher scans complete run and observation records before private publication, refusing blocking secret findings with value-free diagnostics. Local records, archive evidence, and review output protections remain independent of publication retries.

Logs and local archives can contain sensitive information, so upload only the intended outputs.

Private storage prevents generated artifacts from appearing in ordinary cwd-rooted discovery. It is not an OS sandbox. When a later phase needs a generated file, Daydream passes its exact approved path or, for isolated backend modes, its bounded contents inline. It does not grant access to an entire runtime directory.

The `.daydream/exploration/` cache is reused on an exact key match. The key excludes uncommitted edits. A near-match never counts as a hit, because a stale hit would misground every review prompt. The `--shallow` and `--review` modes delete the directory. Alternating modes degrade to a cache miss, never to stale grounding.

### Terminal review findings contract

`--findings-out` emits schema version 2. Review artifacts require a `terminal_result` (nested version 1); diagram artifacts carry no code-review coverage. Readers reject older or unknown versions. Producer and poster must use the same reviewed version.

Analysis is `complete` only with positive completion or host no-op evidence for every planned reviewer and required phase. Valid work with unfinished coverage is `incomplete`; unusable analysis or an invalid findings projection is `failed`. Findings count, warning absence, CLI success and archive status cannot establish completeness. `pipeline_state` covers review finalization independently of later fixes or publication.

Coverage records run identity, exact stack/shard outcomes, captured head, diff merge base and diff key, plus initial PR base tip when available. Backend/authentication, missing/malformed output, evidence and policy failures remain typed; model-turn, host wall/tool and pipeline budgets have distinct reasons.

Findings and coverage are validated and atomically published together. Valid partial findings survive normal finalization; incomplete/failed results publish COMMENT notices and cannot authorize approval or stale-thread resolution. Commit-bound exports require a clean checkout and use captured identity and diff for placement. Interactive dirty reviews remain supported; publishers validate the independently trusted target.

Every artifact Daydream writes declares the commit its analysis read, for `--review` and `--diagram-only` exports alike. A PR head that has moved away from the analyzed checkout ends the run non-zero, before any artifact is written. The commit is captured once at run start; no export path re-resolves it.

Use a unique, initially absent output path for each invocation. Reusing a path requires an independently known expected run ID: failed replacement can preserve an old complete file. Missing/truncated output, process destruction and failed installation provide no valid current result.

Resume and cache proofs require typed coverage for the exact revision and scope inventory. Missing, damaged or mismatched proofs restart review; same-revision partial resumes preserve failures, successful scope reruns clear them, and complete reuse receives the current run ID.

## Development

**New to contributing? Read [CONTRIBUTING.md](CONTRIBUTING.md) — setup, commands, the required gate, and PR workflow.**

Editors that support [EditorConfig](https://editorconfig.org) pick up the root
`.editorconfig` automatically (UTF-8, LF, final newline; 4-space Python, 2-space
YAML, 4-space TOML, tabs in Makefiles, preserved Markdown hard breaks).

See [docs/coverage.md](docs/coverage.md) for the coverage gate and ratchet procedure.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
