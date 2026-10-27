# Private PR benchmarks

Read before operating or changing the Harbor benchmark. Install the `benchmark`
extra in Daydream's environment. The supported runtime is local Docker Desktop
on macOS with live nftables preflight; custom bridges/proxies, air-gapped Business
containers, OrbStack, Linux/cloud hosts, and public-network fallbacks are unsupported.
The reviewer is Pi; reviewer and judge use OpenRouter through separate allowlists.
Live runs send private source to that provider: the operator must accept its
handling/retention policy before launch. Paid Oracle and judge-calibration runs
are never executed in CI.

## Workflow

Use `daydream benchmark COMMAND --help` for flags. Keep the workspace private.

```bash
daydream benchmark init ~/bench --repo OWNER/REPO --reviewer-host openrouter.ai --judge-host openrouter.ai
daydream benchmark import-prs ~/bench --pr 123
daydream benchmark curate ~/bench --case CASE_ID
daydream benchmark validate ~/bench
daydream benchmark build-harbor ~/bench --daydream-wheel dist/daydream.whl
daydream benchmark run ~/bench --oracle --yes
daydream benchmark run ~/bench --yes
daydream benchmark objective ~/bench --run-id RUN_ID --json -
```

Curate gold explicitly; prioritization and resolved/outdated signals are advisory.
Exact inline acceptance requires the derived authoring anchor at the selected
original head, usable location/title, and neither outdated nor dismissed state.
Re-anchored or unavailable evidence needs manual editing. Refresh preserves
curation. Frozen tasks require current canonical task approval before compilation.
`validate` exits 0 for ready, 2 for valid but incomplete, and 1 for corrupt.

The Oracle receipt must match the compiled lock before a normal benchmark run;
allowlist/profile/lock changes invalidate it. `calibrate-judge` is optional
agreement analysis against an unverified fixture, never a correctness or run gate.

## Trust and results

Gold is hidden from the reviewer. Credentials remain runtime capabilities, outside
compiled tasks and trajectories. The environment declaration is
[daydream/benchmark/harbor/env_policy.py](../daydream/benchmark/harbor/env_policy.py).
Only an explicit trusted control-plane candidate profile or packaged default is
admitted; ambient and target repository profiles cannot alter it. Pin wheel,
compiled lock, candidate, model/backend, judge, and scoring identities.

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

`objective` reads an exact completed ledgered run; it invokes no models.
`aggregate` pools completed compatible runs from a manifest:

```json
{"schema_version": 1, "entries": [{"workspace": "/private/bench", "run_id": "RUN_ID"}]}
```

Pool TP/FP/FN before deriving micro precision/recall/F1; never average repository
F1. Missing, duplicated, incompatible, or comparison-ineligible runs refuse the
whole suite. Outputs contain opaque identities/counts, no private source or gold.
Location/severity axes cover matched pairs only and do not affect reward. Absent
axes add no scored pairs; exported rates remain zero. `location_pairs_scored` and
`severity_pairs_scored` distinguish no evidence from measured performance. See
[objective](../daydream/benchmark/harbor/objective.py) for the output schema.

## Recovery

Writes are locked, atomic, and journaled. Preserve the before/after transaction
boundary rather than hand-editing sealed snapshots. Unreplayable cases cannot be
promoted by curation. Use `import-prs --refresh` for stale inputs and
`upgrade --dry-run` before upgrading legacy provenance; verification requires the
private mirror and never stamps unproven ready state. Authored gold stays intact.

`clean --derived` removes disposable caches/jobs/trajectories while preserving
source/gold; `clean --all --yes` also deletes curated material. Cleanup-pending
runs are not complete objectives. Verifier errors make comparisons ineligible.
