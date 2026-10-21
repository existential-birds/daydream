# Review benchmarks

## Supplied ReviewBench with native Daydream

Use `daydream benchmark reviewbench COMMAND --help`. This evaluation-only adapter
calls the ordinary `runner.run` reviewer; it preserves native backend, provider,
authentication, model, per-phase configuration and review-profile precedence.
It adds no reviewer engine selector and changes no reviewer defaults. Benchmark
inputs and results never enter dataset capture or training.

Pin the supplied source and install its Node dependencies (Node 20+):

```bash
git clone https://github.com/review-bench/ReviewBench.git ~/ReviewBench
git -C ~/ReviewBench checkout --detach 4f04392d6bf4941ab6f8a3ea0c15b449f4458864
cd ~/ReviewBench
npm ci
```

Keep that checkout unchanged; the adapter verifies its revision and source hashes.
The starter set is exactly `corpus/test/test.json` (25 cases); `--set full` selects
all 219 entries in `corpus/manifest.json`. Create a fresh candidate plan with no
inference:

```bash
daydream benchmark reviewbench plan ~/reviewbench-runs --source ~/ReviewBench \
  --candidate native-current --producer daydream-native --set test --rounds 1
daydream benchmark reviewbench status ~/reviewbench-runs/native-current
```

Plans record the executing Daydream package version, Git revision when available,
package artifact hash, producer, benchmark revision and exact case membership.
Use the candidate installation/checkout itself to create and run its plan; another
revision is a new plan, not a relabeled result. Package-byte changes invalidate a
plan. Configuration is frozen on the first generation call and must agree across
its rounds. Effective native configuration and profile digests accompany every
terminal outcome. The freeze includes native provider/thinking/retry controls and
the operator configuration root's public Pi model defaults. Case-local Pi defaults
cannot replace them. Actual request model/provider/effort identities are retained
separately from configured phase intent, including optional review phases.
Available prompt/completion/cache/cost totals use the recorder's authoritative
whole-run accounting, including joined forks exactly once. Missing measurements
remain unavailable; no cleanup duration is relabeled as reviewer time. Credentials
and credential-bearing files are never serialized or fingerprinted.
Native configuration-directory, provider-switch and endpoint selectors accompany
the adapter's supported host controls. Native client file contents and internal
provider decisions can remain opaque; this limitation is recorded alongside
observed request identities and does not change supplied scoring or averaging.

**Generation incurs reviewer calls. Launch only with separate experiment
authorization.** After `--`, use ordinary native review flags and a configuration
root. Its repository file policy is loaded through the existing CLI parser;
explicit flags retain their usual precedence. Native provider/authentication
settings are inherited, including Pi's configured `nous` and
`deepseek/deepseek-v4.1-flash`; no OpenRouter substitution occurs.

```bash
daydream benchmark reviewbench run ~/reviewbench-runs/native-current --round 1 \
  -- --backend pi /path/to/operator-config-root
```

`--pr INDEX` selects a zero-based entry in the selected supplied set, matching
`try-agent.sh`; `--repository PATH` optionally supplies that single case's local
frozen repository. Otherwise the adapter fetches full supplied base/head commits
from the corresponding `review-bench/OWNER_REPO` mirror. Every review uses an
independent disposable clone at HEAD, with golden/reference material outside
reviewer inputs. No fixes, pushes, PR publication or dataset capture run. This
native offline workflow does not claim the portal container/egress isolation or
official leaderboard eligibility; separately authorized experiments must arrange
the supplied isolation and audit requirements for the selected native harness.

The adapter verifies actual source identity, analyzed HEAD, merge base and full
patch fingerprint, rather than trusting a supplied SHA label. Native ignore rules
that change the analyzed patch produce a visible comparison failure. It consumes the
actual finalized terminal result without manufacturing a publication PR.
Complete empty, incomplete, failed and absent results remain distinct. Retained
partial findings and conversion limitations remain in `outcomes/`. Canonical
line hints are converted only when the captured diff proves a HEAD coordinate;
deleted-side-only, ambiguous, relocated and unlocatable hints are omissions.
Any omission marks the case `conversion_limited` and excludes it from scorable
findings, rather than presenting discarded findings as a successful empty review.

Each `round-N/findings/` is one evaluator input. Never pass a parent containing
multiple rounds: the supplied loader recursively merges identical PR identities.
Artifacts are created at new paths and prior outputs are preserved on failure;
use a new candidate plan for retries or changed inputs.
The evaluator installs `evaluator.json` last; interrupted publication leaves an
incomplete directory that reports reject, so preserve it and use a fresh plan.
`status` reports expected, produced, completed-empty, failed, missing and evaluated memberships independently
of supplied overlap-only scoring. Subsets are identified as subsets.

**Judging incurs model calls for ordinary nonempty reviews.** Select its standalone
provider and model explicitly after separate experiment authorization:

```bash
daydream benchmark reviewbench judge ~/reviewbench-runs/native-current --round 1 \
  --provider anthropic --model YOUR_JUDGE_MODEL
daydream benchmark reviewbench report ~/reviewbench-runs/native-current
```

The supplied evaluator owns matching, classification and metric formulas. Reports
use its `printSummary` and preserve raw grounded/augmented precision and recall,
macro/micro metrics, per-PR/severity/category breakdowns, ratios/nulls, `eval_config`
and available usage. Evaluator provenance records the explicit workflow, provider,
model, source revision, candidate/reference bytes and prompt hashes. Standalone
and public Harbor evaluator profiles differ; neither substitutes for the other.
Augmented recall has an agent-dependent denominator. Supplied duration means the
successful reviewer attempt, not total publication/cleanup latency; this native
adapter currently leaves that measurement unavailable.

For the documented final comparison, make a fresh `--set full --rounds 3` plan,
run/judge rounds 1, 2 and 3 independently, then `report` averages corresponding
supplied score ratios with null availability counts while retaining all three raw
results. `RB_ATTEMPT` is an upstream retry counter, not an independent round.
These results inform owner acceptance of tradeoffs; no custom reward, quality
floor, automatic winner, calibration or qualification gate is added.

Offline verification uses synthetic fixtures and the pinned upstream parser,
scorer/aggregate/summary and its completed-empty judge fast path, without model
connections. To rerun optional source-backed integration tests:

```bash
uv lock --check
DAYDREAM_REVIEWBENCH_SOURCE=~/ReviewBench uv run pytest tests/test_reviewbench_*.py
```

## Private Harbor benchmark

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
