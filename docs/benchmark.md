# Review benchmarks

## Public ReviewBench

Use `daydream benchmark reviewbench --help`. Export performs no inference;
reviewing and judging require experiment authorization. Keep inputs/results out
of capture and training, preserve failed evidence, and use fresh output paths.

Install a candidate wheel with the packaged
[Harbor lock](../daydream/benchmark/reviewbench/harbor-runtime.lock). Use Node 20+
and a clean ReviewBench checkout at `4f04392d6bf4941ab6f8a3ea0c15b449f4458864`
with `npm ci`. Run both commands from that candidate installation:

```bash
daydream benchmark reviewbench harbor-export /private/export \
  --source ~/ReviewBench --wheel /path/to/candidate.whl --rounds 3 \
  --platform linux/arm64 -- --backend pi /path/to/operator-config-root
daydream benchmark reviewbench harbor-run /private/export \
  --reviewer-pi-home /private/reviewer-auth --pi-home /private/judge-auth \
  --jobs-dir /private/jobs --job-name comparison --set test --rounds 3
```

For Codex, select `--backend codex` and replace `--reviewer-pi-home` with
`--codex-home`. Reviewer and judge need separate native credential stores outside
exports/jobs; the common judge uses `openai-codex` / `gpt-6.1-sol`. Docker must
support the selected architecture, nested sandboxing and exact-host isolation.

`--set test` selects the supplied 25 cases; `--set full` selects all 219.
Repeated `--task EXACT_NAME` selects a subset. Harbor owns independent rounds;
the supplied checker and metric own admission, scoring and reports. Profiles use
schema 3; preserve older exports without migration. Offline integration proves
no review-quality or performance improvement.

Inspect interrupted evidence without inference with
`python -m daydream.benchmark.reviewbench.diagnostics JOB/TRIAL/agent/daydream`.
The last valid checkpoint preserves observed activity; Harbor owns timeout
classification. Checkpoints establish neither completed reviews nor scores.

## Private Harbor benchmark

Requires the `benchmark` extra and local Docker Desktop on macOS with live
nftables preflight; other runtimes/network fallbacks are unsupported. Pi reviewer
and judge use OpenRouter with separate allowlists. Obtain authorization for
private-source transmission under the provider's handling policy. Paid Oracle
and calibration runs stay outside CI.

Workflow: `init WORKSPACE --repo OWNER/REPO` → `import-prs` → `curate` → `validate` → `build-harbor` →
`run --oracle --yes` → `run --yes` → `objective --run-id RUN_ID`.

- Curate gold explicitly; advisory signals are not approval. Compilation requires
  current canonical task approval. Re-anchored/unavailable evidence needs manual
  editing. Keep gold hidden from reviewers and credentials outside tasks/evidence.
- Pin candidate/wheel, compiled lock, model/backend, judge and scoring identities.
  Only trusted control-plane or packaged profiles are admitted. Profile/allowlist/
  lock changes invalidate the Oracle receipt; regenerate it before a normal run.
- `objective` reads an exact completed ledgered run without inference. `aggregate`
  pools compatible runs' TP/FP/FN before deriving micro metrics; never average
  repository F1. Cleanup-pending runs and verifier errors are ineligible.
  `calibrate-judge` is advisory agreement analysis, not a correctness gate.
- Preserve sealed transaction boundaries. Recover stale inputs with
  `import-prs --refresh`; inspect `upgrade --dry-run` before legacy upgrades.
  Verification needs the private mirror; unreplayable cases cannot be curated ready.
  `clean --derived` preserves source/gold; `clean --all --yes` deletes them.

Change owners: [CLI](../daydream/benchmark/cli.py),
[environment policy](../daydream/benchmark/harbor/env_policy.py),
[curation](../daydream/benchmark/curation.py),
[storage/recovery](../daydream/benchmark/storage.py),
[run gates](../daydream/benchmark/harbor/run.py),
[objective/aggregation schema](../daydream/benchmark/harbor/objective.py).
