# Benchmark Runbook

`daydream bench` scores daydream's deep-review findings against [Martian's Code Review Benchmark](https://github.com/withmartian/code-review-benchmark) offline set — the 26 evaluable Python/Go/TS PRs (6 Sentry + 10 Grafana + 10 Cal.com). Per PR it acquires a local checkout, runs `daydream --non-interactive` as a subprocess, deterministically maps the merged findings into the benchmark's `benchmark_data.json`, then drives the benchmark's step2/2.5/3 modules to produce precision/recall. The benchmark repo itself is never modified by code — only its `results/` data is injected.

This runbook takes you from nothing to a scored result.

## Prerequisites

- **A benchmark checkout.** Clone the benchmark beside this repo so its offline harness sits at `../code-review-benchmark/offline/`. That `offline/` directory is the `--benchmark-repo` path; the step2/2.5/3 modules read `results/benchmark_data.json` relative to it.
- **`daydream` installed.** Run `uv sync` so the `daydream` console script is on `PATH` (the harness invokes it as a subprocess).
- **`git` and `gh` on `PATH`.** `git` performs the blobless clone and `pull/N/head` fetch per PR.
- **The Beagle plugin** installed in Claude Code (see the [Quickstart](../README.md#quickstart)) — deep review needs the stack-specific skills.
- **The judge credential, exported.** Scoring requires one env var and accepts two optional overrides:
  - `MARTIAN_API_KEY` — an OpenRouter `sk-or-…` key (or a withmartian key). **Required for `--score`.**
  - `MARTIAN_BASE_URL` — the OpenAI-compatible judge endpoint. Defaults to `https://api.withmartian.com/v1`; set to `https://openrouter.ai/api/v1` when using an OpenRouter key.
  - `MARTIAN_MODEL` — the judge model id. Defaults to `openai/gpt-4o-mini`; should match `--model` for comparable results.

  The harness reads `os.environ` only — it does **not** parse a `.env` file. If you keep these in a `.env`, you must export them into the shell first:

  ```bash
  set -a; source .env; set +a
  ```

## Smoke subset

Wire the pipeline cheaply before spending on the paid judge. Run two Grafana PRs with scoring off:

```bash
daydream bench --benchmark-repo ../code-review-benchmark/offline --only grafana --limit 2 --no-score
```

This acquires the checkouts, runs deep review, and injects two `daydream` reviews into `benchmark_data.json` — no judge calls. When the wiring looks right, add the judge:

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

## Where the number lands

For each scored PR the harness writes a `daydream` leaf into:

```
<benchmark-repo>/results/<sanitized-model>/evaluations.json
```

`<sanitized-model>` is the `--model` id with `/` replaced by `_`. For the default model the directory is `results/anthropic_claude-opus-4.5/` (the dot is preserved). Each leaf carries `tp`, `fp`, `fn`, `precision`, and `recall` for that PR.

The command also prints to stdout:

- per-PR tp/fp/fn counts,
- the aggregate precision/recall over all scored PRs (`precision = ΣTP / (ΣTP + ΣFP)`, `recall = ΣTP / (ΣTP + ΣFN)`), and
- the **N scored** count.

## Incremental re-runs

Re-running is resumable and idempotent. `benchmark_data.json` is saved after each PR, so an interrupted sweep can be resumed. A PR that already has a `tool:"daydream"` review is **skipped** — no checkout, no review, no judge call. Pass `--force` to re-run injected PRs and replace their findings.

## Comparability caveat

The `--model` value names the per-model results directory; it does **not** select the judge model. The judge model is selected by the `MARTIAN_MODEL` environment variable consumed by the scoring step. To compare against published benchmark numbers, both the `MARTIAN_MODEL` value and the `--model` label must match the published run. Using a different judge — or a different id string for the same underlying model — lands in a different `results/<dir>` and is not directly apples-to-apples.

The published Martian run used the dated id `anthropic/claude-opus-4-5-20251101` → `results/anthropic_claude-opus-4-5-20251101/`; the default here is `anthropic/claude-opus-4.5` → `results/anthropic_claude-opus-4.5/`, a distinct directory. Scores are model-determined rather than string-determined, and the judge prompt and `temperature: 0.0` are identical, so the same underlying model under a different id string is broadly comparable. But routing through a different gateway can shift outputs slightly (system-prompt injection, sampling, schema handling), so for the closest apples-to-apples comparison run without structured output and note the gateway difference. To reuse the existing published directory exactly, set `--model` to that dated id only if OpenRouter accepts it as a model id.
