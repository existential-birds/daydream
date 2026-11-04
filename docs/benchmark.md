# Review benchmarks

## Public ReviewBench

ReviewBench and other public benchmarks run in
[code-review-benchmarks](https://github.com/existential-birds/code-review-benchmarks),
which pins one Daydream build and reviews each pull request with it in Harbor.

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

Normal Harbor runs require an explicit, nonblank `DAYDREAM_REVIEW_MODEL` with the
exact operator-selected OpenRouter model ID; its spelling is preserved unchanged.

The bounded Pi reviewer consumes the shared
[immutable PR embedding API](runtime.md#immutable-pr-inputs), using official Pi
1.4.0 and Node 24.18.1. Compilation preserves exactly two synthetic commits with
the original comparison/head trees. Validated Git objects establish saved source
identity; the current execution computes its own review diff key. Legacy saved
`snapshot.diff_sha256` is accepted as ignored metadata, and new snapshots omit it.
Bundle and exported-artifact checksums still bind their exact bytes. Trusted runtime metadata
delivers only the immutable provenance/materialization subset and frozen bundle;
the full lock, gold and verifier metadata remain private. Original repository/PR
identity is retained in result/ATIF metadata, while source references identify the
actual synthetic commits. Only approved bounded historical PR context is model
input. Analysis requires no remote, GitHub discovery or GitHub credentials.

The entrypoint keeps review-only effects and local trajectories/diagnostics.
Only a complete, validated current result with successful native settlement,
output installation and capture can install a candidate through the atomic
candidate owner. Partial/error output cannot install a successful candidate.
Unsupported backends and oversized plans fail explicitly. This adapter does not
authorize fixing, posting, archive/upload/training, or claim comprehensive native
parity; #1639 owns that migration. Ordinary candidate publication remains supported.

Owners: [CLI](../daydream/benchmark/cli.py), [environment](../daydream/benchmark/harbor/env_policy.py),
[curation](../daydream/benchmark/curation.py), [storage](../daydream/benchmark/storage.py),
[run gates](../daydream/benchmark/harbor/run.py), [aggregation](../daydream/benchmark/harbor/objective.py).
