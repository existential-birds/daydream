# Review benchmarks

## Public ReviewBench

Install the packaged [Harbor lock](../daydream/benchmark/reviewbench/harbor-runtime.lock),
Node 20+, and ReviewBench at `4f04392d6bf4941ab6f8a3ea0c15b449f4458864` with `npm ci`.
Export performs no inference; reviewing/judging require separate authorization.
Keep inputs/results out of capture/training and use fresh output paths:

```bash
daydream benchmark reviewbench harbor-export /private/export \
  --source ~/ReviewBench --wheel /path/to/candidate.whl --rounds 3 \
  --platform linux/arm64 -- --backend pi /path/to/operator-config-root
daydream benchmark reviewbench harbor-run /private/export \
  --reviewer-pi-home /private/reviewer-auth --pi-home /private/judge-auth \
  --jobs-dir /private/jobs --job-name comparison --set test --rounds 3
```

For Codex, use `--backend codex` and `--codex-home`. Keep independent native
reviewer/judge credentials outside exports/jobs. The judge uses
`openai-codex` / `gpt-6.1-sol`; Docker needs the selected architecture, nested
sandboxing and exact-host isolation. The wheel/hash identifies the candidate;
package extensions in that wheel. Export/launch need no matching installation.

`--set test` selects the supplied 25 cases; `--set full` all 219; repeated
`--task EXACT_NAME` a subset. Harbor owns rounds; supplied scoring/checkers own
results. Schema 4 requires fresh exports and candidate wheels; preserve old evidence.
Offline proof establishes no review-quality or performance improvement.
Read interruptions with `python -m daydream.benchmark.reviewbench.diagnostics JOB/TRIAL/agent/daydream`.
Checkpoints retain exact recorder bytes; Harbor owns timeout classification and completion/scoring.

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
