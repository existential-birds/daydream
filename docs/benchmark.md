# Benchmark Runbook

`daydream bench` scores daydream's deep-review findings against [Martian's Code Review Benchmark](https://github.com/withmartian/code-review-benchmark) offline set: the 26 evaluable Python/Go/TS PRs (6 Sentry + 10 Grafana + 10 Cal.com). Per PR it acquires a local checkout, runs `daydream --non-interactive` as a subprocess, deterministically maps the merged findings into the benchmark's `benchmark_data.json`, then drives the benchmark's step2/2.5/3 modules to produce precision/recall. The benchmark repo itself is never modified by code; only its `results/` data is injected.

This runbook takes you from nothing to a scored result.

## Prerequisites

- **A benchmark checkout.** Clone the benchmark beside this repo so its offline harness sits at `../code-review-benchmark/offline/`. That `offline/` directory is the `--benchmark-repo` path; the step2/2.5/3 modules read `results/benchmark_data.json` relative to it.
- **`daydream` installed.** Run `uv sync` so the `daydream` console script is on `PATH` (the harness invokes it as a subprocess).
- **`git` and `gh` on `PATH`.** `git` performs the blobless clone and `pull/N/head` fetch per PR.
- **The Beagle plugin** installed in Claude Code (see the [Quickstart](../README.md#quickstart)). Deep review needs the stack-specific skills.
- **A backend for the reviewer under test.** By default the reviewer runs daydream's built-in default backend (Claude). To benchmark another backend, select it with `--reviewer-backend` (see [Selecting the reviewer backend](#selecting-the-reviewer-backend)). The `pi` backend driving a GLM model over OpenRouter additionally needs the `pi` CLI on `PATH` and the OpenRouter provider extension registered with `pi` (installed once via `pi install`); the run forwards `--reviewer-provider` to the reviewer as the `PI_PROVIDER` environment variable.
- **The judge credential, exported.** Scoring requires one env var and accepts two optional overrides:
  - `MARTIAN_API_KEY`: an OpenRouter `sk-or-…` key (or a withmartian key). **Required for `--score`.**
  - `MARTIAN_BASE_URL`: the OpenAI-compatible judge endpoint. Defaults to `https://api.withmartian.com/v1` (default set by the withmartian step modules, not by daydream); set to `https://openrouter.ai/api/v1` when using an OpenRouter key.
  - `MARTIAN_MODEL`: the judge model id. Defaults to `openai/gpt-4o-mini` (default set by the withmartian step modules, not by daydream); should match `--model` for comparable results.

  These may live in a `.env` file in the directory you run `daydream bench` from; it is auto-loaded at bench entry (`python-dotenv`, searching from the cwd upward). Already-exported shell variables win, so an inline `MARTIAN_API_KEY=… daydream bench …` still overrides the `.env`; a missing or malformed `.env` is a silent no-op.

  ```bash
  # .env beside your invocation
  MARTIAN_API_KEY=sk-or-…
  MARTIAN_MODEL=anthropic/claude-opus-4-5-20251101
  ```

## Configuration file (`[tool.daydream.bench]`)

Repeating `--benchmark-repo`, the judge `--model`, and the full reviewer flag set on every invocation gets old. A `[tool.daydream.bench]` table in the `pyproject.toml` (or `.daydream.toml`) of the directory you run `daydream bench` from supplies defaults **under** the CLI flags. Precedence is always **CLI flag > config file > built-in default**: an explicit flag always wins; the config only fills a flag you omit.

```toml
[tool.daydream.bench]
benchmark-repo = "../code-review-benchmark/offline"   # makes --benchmark-repo optional
model = "anthropic/claude-opus-4-5-20251101"           # judge model when --model is omitted

# Named reviewer presets: each expands to --reviewer-backend / -model / -provider.
[tool.daydream.bench.reviewers.glm]
backend = "pi"
model = "z-ai/glm-5.2"
provider = "openrouter"
```

config-only: there are no built-in reviewer names or model ids baked into daydream; a preset exists only if you define its table.

### `--reviewer <name>` expands a preset

`--reviewer glm` looks up `[tool.daydream.bench.reviewers.glm]`, applies its `backend`/`model`/`provider` as the reviewer fields, and derives `--tool-label` as `daydream-glm` ; its findings file under a distinct results key automatically (see [`--tool-label` isolates per-backend results](#--tool-label-isolates-per-backend-results)). Explicit `--reviewer-backend`/`-model`/`-provider` or `--tool-label` flags still override the preset (CLI > config). An unknown `--reviewer` name is a usage error.

With the table above, the full GLM sweep over one PR collapses to:

```bash
daydream bench --reviewer glm --only grafana --limit 1
```

`benchmark-repo` and the judge `model` come from config; `--reviewer glm` supplies the backend/model/provider and the `daydream-glm` label. (Scoring is on by default, so `MARTIAN_API_KEY` must be present; see [Prerequisites](#prerequisites).)

## Smoke subset

Wire the pipeline cheaply before spending on the paid judge. Run two Grafana PRs with scoring off:

> **Note:** On first run each PR repo is blobless-cloned from GitHub. The clone is subject to a 60 s timeout; on a slow connection a large repo (Grafana, Sentry) can hit that limit and surface as a `GitError`, aborting the sweep. If you see a clone timeout, retry once the network is faster, or pre-clone the repos manually and point `--benchmark-repo` at a local mirror.

```bash
daydream bench --benchmark-repo ../code-review-benchmark/offline --only grafana --limit 2 --no-score
```

This acquires the checkouts, runs deep review, and injects two `daydream` reviews into `benchmark_data.json`; no judge calls. When the wiring looks right, add the judge:

```bash
daydream bench --benchmark-repo ../code-review-benchmark/offline --only grafana --limit 2 --score
```

`--only` matches a source-repo name (`sentry`, `grafana`, `cal.com`) or a golden-URL substring. `--limit N` caps how many of the selected PRs run.

## Full sweep

Drop `--only` and `--limit` to run all 26 evaluable PRs:

```bash
daydream bench --benchmark-repo ../code-review-benchmark/offline --score
```

This is the load-bearing, money-spending run: 26 deep reviews plus 26 judge passes.

## Watching progress (`--verbose`)

A deep review of one PR runs for minutes. By default each PR shows a live spinner with the PR label and reviewer, then a completion line with the elapsed time and finding count:

```text
▶ [1/2] Reviewing https://github.com/grafana/grafana/pull/1234 · reviewer daydream…
Reviewed https://github.com/grafana/grafana/pull/1234 in 4m12s · 3 findings
```

Pass `-v`/`--verbose` to stream the underlying `daydream --non-interactive` subprocess output live instead of the spinner (streaming and a spinner can't share one console, so verbose replaces the spinner; the announce and completion lines stay):

```bash
daydream bench --reviewer glm --only grafana --limit 1 --verbose
```

## Selecting the reviewer backend

The harness benchmarks daydream itself, but the *reviewer under test*: the backend/model that produces the findings, is selectable. This is independent of `--model`, which only names the **judge**. Four flags control the reviewer:

- `--reviewer-backend {claude,codex,pi}`: the backend daydream runs its deep review on. Forwarded to the per-PR subprocess as `--backend`. Omit to use daydream's built-in default (Claude).
- `--reviewer-model <id>`: the reviewer model id. Forwarded as `--model` to the reviewer subprocess. Omit to use the backend's default.
- `--reviewer-provider <name>`: the reviewer provider, forwarded to the reviewer subprocess as the `PI_PROVIDER` environment variable (never as an argv flag). Used by the `pi` backend to route a model through a specific provider, e.g. `openrouter` to run GLM via OpenRouter. Requires the OpenRouter provider extension registered with `pi` (see Prerequisites).
- `--tool-label <label>`: the results key this reviewer's findings are filed under (default: `daydream`).

> **Note:** There is no `--provider` flag on the main `daydream` CLI; the reviewer provider crosses the subprocess boundary only as `PI_PROVIDER`. Pass it to the benchmark as `--reviewer-provider`, not `--provider`.

Example: benchmark daydream driven by GLM (`glm-5.2`) on the `pi` backend, routed through OpenRouter, filed under a distinct label:

```bash
daydream bench --benchmark-repo ../code-review-benchmark/offline \
  --reviewer-backend pi --reviewer-model glm-5.2 --reviewer-provider openrouter \
  --tool-label daydream-glm --only grafana --limit 1 --score
```

### `--tool-label` isolates per-backend results

Every reviewer's findings are injected into `benchmark_data.json` and scored under its `--tool-label`. The label is the **only** thing keeping two reviewer backends from overwriting each other:

- A PR is skipped on re-run when a review with the *same* `--tool-label` already exists. Two backends sharing one label would mean the second never runs (the first's review is "already present").
- The judge writes each tool's scores into a leaf keyed by the tool label inside `evaluations.json`. Sharing a label silently merges/overwrites the two backends' scores.

So when benchmarking more than the default reviewer, give each backend a distinct label (`daydream` for the default, `daydream-glm` for the GLM/pi reviewer, etc.). Reviews and score leaves for different labels coexist in the same corpus and the same `results/<judge>/` directory, side by side.

## Where the number lands

For each scored PR the harness writes a leaf (keyed by the reviewer's `--tool-label`, default `daydream`) into:

```text
<benchmark-repo>/results/<sanitized-model>/evaluations.json
```

`<sanitized-model>` is the `--model` (judge) id with `/` replaced by `_`. For the default judge the directory is `results/anthropic_claude-opus-4.5/` (the dot is preserved). Inside each PR's entry the scores are filed under the reviewer's `--tool-label` (default `daydream`; e.g. `daydream-glm` for a GLM reviewer). Each leaf carries `tp`, `fp`, `fn`, `precision`, and `recall` for that PR.

The command also prints to stdout:

- per-PR tp/fp/fn counts,
- the aggregate precision/recall over all scored PRs (`precision = ΣTP / (ΣTP + ΣFP)`, `recall = ΣTP / (ΣTP + ΣFN)`), and
- the **N scored** count.

Aggregate scores use **micro-averaging** (pool all TP/FP/FN, then divide), the same method used in the published Martian benchmark numbers, so results are directly comparable.

## Incremental re-runs

Re-running is resumable and idempotent. `benchmark_data.json` is saved after each PR, so an interrupted sweep can be resumed. A PR that already has a `tool:"daydream"` review is **skipped**: no checkout, no review, no judge call. Pass `--force` to re-run injected PRs and replace their findings.

**Run one sweep per benchmark repo at a time.** Each save acquires an exclusive `benchmark_data.json.lock` file to serialise concurrent writers on the same machine, but two sweeps sharing the same `--benchmark-repo` would still race at the read-inject-write level: the second run reads a stale corpus, overwrites the first run's injections, and you lose results. Start a second sweep only after the first has finished (or been interrupted).

## Comparability caveat

When comparing results, what matters is the reviewer model (the model doing the review) and the judge model (the model scoring the findings) are both reported. A different reviewer model produces different findings; a different judge model scores the same findings differently. The offline HTML report at `bench/benchmark-report/runs/latest/index.html` shows both for each run.

The `daydream bench` `--model` flag sets the judge model label (for the results directory). The judge's underlying model is set by `MARTIAN_MODEL`. The runner's model is set by `--reviewer-model`. Both are documented in the report metadata.

## First measured baseline (provisional)

A first full sweep was run on **2026-06-04** to validate the harness end-to-end. These numbers are a **single-sweep provisional baseline**, not a published result. For the published commercial-bot numbers these are ultimately measured against, see the [Martian Code Review Benchmark leaderboard (offline mode)](https://codereview.withmartian.com/?mode=offline). Daydream ran on `claude-opus-4-5` (the `daydream-owl-alpha` tool label) with default deep multi-stack review.

The benchmark report generator (`make benchmark-report`) renders an offline comparison from the same `results/` data against 42 competing review tools on the 22-PR subset daydream covered:

| Metric | Value | Rank (of 42 tools) |
|---|---|---|
| Precision (micro) | 0.206 | 34 |
| Recall (micro) | 0.590 | 10 |
| F1 (micro) | 0.305 | 30 |
| PRs scored | 22/26 | |
| TP / FP / FN | 36 / 139 / 25 | |

The full per-reviewer scorecard, per-PR breakdown, and cost comparison are in the self-contained HTML report at `bench/benchmark-report/runs/latest/index.html`.

Caveats:

- **Single sweep.** No variance band; the LLM judge runs at `temperature: 0.0` but is not fully deterministic.
- **4 PRs unscored.** The offline set has 26 evaluable PRs; daydream's sweep covered 22 (the remainder exceeded per-PR time caps or hit transient failures). The sweep is resumable.
- **Precision gap.** 36 TP against 139 FP. Precision (0.206) sits below the README's 50% target. Recall (0.590) is competitive, ranking 10th of 42 tools. The precision gap is what the training milestone is meant to close.
- **Tied to this setup.** Daydream model, judge model, date all move the number.
