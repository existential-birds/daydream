# Review benchmarks

## Public ReviewBench

Install the packaged [Harbor lock](../daydream/benchmark/reviewbench/harbor-runtime.lock), Node 20+,
and ReviewBench at `4f04392d6bf4941ab6f8a3ea0c15b449f4458864` with `npm ci`.
Tracked source must match that commit; extra files under `scripts/` are rejected.
Unrelated untracked files elsewhere are allowed.
Export performs no inference; model/judge runs require separate authorization.
Keep inputs/results outside capture/training and use fresh paths:

```bash
daydream benchmark reviewbench harbor-export /private/export \
  --source ~/ReviewBench --wheel /path/to/candidate.whl --rounds 3 \
  --platform linux/arm64 -- --backend pi /path/to/operator-config-root
daydream benchmark reviewbench harbor-run /private/export \
  --reviewer-pi-home /private/reviewer-auth --pi-home /private/judge-auth \
  --jobs-dir /private/jobs --job-name comparison --set test --rounds 3
```

Pi requires configured `nous`; for Codex use `--backend codex` / `--codex-home`.
Keep independent native credentials outside exports/jobs. The judge uses `openai-codex` / `gpt-6.1-sol`.
Docker needs the selected architecture, nested sandboxing and exact-host isolation.
The wheel/hash identifies the candidate; package extensions there. No matching installation is needed.
Exports freeze reviewer retry, idle, and Codex lifecycle controls for native execution.
`--set test`: supplied 25; `--set full`: all 219; repeated `--task EXACT_NAME`: a subset.
Harbor owns rounds/timeouts; supplied scoring/checkers own results. Use fresh schema-4 wheels/exports;
regenerate earlier exports without frozen execution controls and preserve old evidence.
Ordinary timeout cancellation joins writers and exports partial root/child trajectories, phase cancellation events,
native output, `execution.json`, and process receipts. Cleanup retains a 20-second graceful shutdown allowance;
forced termination can leave detailed review evidence unavailable. New runs create no periodic checkpoint artifacts;
preserve historical artifacts.

## Private Harbor benchmark

Requires the `benchmark` extra, macOS Docker Desktop and live nftables preflight; no network/runtime fallback.
Pi reviewer/judge use separate OpenRouter allowlists. Authorize private-source transmission under provider policy;
paid Oracle/calibration runs stay outside CI.
Workflow: `init WORKSPACE --repo OWNER/REPO` → `import-prs` → `curate` → `validate` → `build-harbor` → `run --oracle --yes` → `run --yes` → `objective --run-id RUN_ID`.

- Curate gold explicitly: advisory signals are not approval. Compilation requires canonical task approval;
  re-anchored/unavailable evidence needs manual editing. Hide gold from reviewers; keep credentials outside tasks/evidence.
- Pin wheel/model/judge/scoring identities and locks; admit only control-plane or packaged profiles.
  Profile/allowlist/lock changes invalidate the Oracle receipt; regenerate it before normal runs.
- `objective` reads exact completed ledgered runs. Cleanup-pending runs/verifier errors are ineligible.
  `aggregate` pools compatible TP/FP/FN before micro metrics; never average F1. Calibration is advisory.
- Preserve sealed transactions: `import-prs --refresh` recovers stale inputs; inspect `upgrade --dry-run` first.
  Private mirror required; unreplayable cases stay unready. `clean --derived` keeps source/gold; `--all --yes` deletes them.

Owners: [CLI](../daydream/benchmark/cli.py), [environment](../daydream/benchmark/harbor/env_policy.py),
[curation](../daydream/benchmark/curation.py), [storage](../daydream/benchmark/storage.py),
[run gates](../daydream/benchmark/harbor/run.py), [aggregation](../daydream/benchmark/harbor/objective.py).
