# Review benchmarks

## Public Harbor with Codex subscription access

This is a distinct **local custom evaluation profile**, not the published
`review-bench/reviewbench-public@v1` judge configuration. It uses all 219 supplied
cases and the unchanged upstream matcher, classifier, deduplication, scorer and
independent-round metric. Daydream's ordinary native Codex backend reviews each
frozen repository. A separate sandbox runs the upstream Pi judge with
`openai-codex` subscription authentication. This supports private matched
before/after comparisons; it claims neither leaderboard eligibility nor a quality
improvement. The private `daydream benchmark` Harbor workflow remains separate.

Build the candidate wheel, then install the **public** runtime into a separate
environment. Released Harbor 0.24.0 lacks the upload-metric API imported by the
upstream plugin; the shipped lock pins ReviewBench
`4f04392d6bf4941ab6f8a3ea0c15b449f4458864` (`0.1.0rc10`) and its Harbor fork
`6ce6abee0b9fa94a50c88f0557d030a419998d78` (`0.24.0`). It also freezes resolved
Python dependencies for the host. The reviewer image uses a separate
`reviewer-runtime.lock` exported from Daydream's locked core dependency graph,
installs without dependency caches, and never installs the ReviewBench package
or its Git checkout. Answer assets are delivered to the separate verifier.
Harbor stays outside core runtime dependencies.

```bash
uv build --wheel --out-dir /tmp/reviewbench-wheels
uv venv --python 3.14.3 /tmp/reviewbench-runtime
uv pip install --python /tmp/reviewbench-runtime/bin/python \
  -r daydream/benchmark/reviewbench/harbor-runtime.lock \
  /tmp/reviewbench-wheels/daydream-0.30.0-py3-none-any.whl
```

Use a clean pinned ReviewBench checkout with `npm ci` already completed (as below).
Export the **entire** corpus without inference, from the installed candidate:

```bash
/tmp/reviewbench-runtime/bin/daydream benchmark reviewbench harbor-export \
  /tmp/reviewbench-full --source /path/to/ReviewBench \
  --wheel /tmp/reviewbench-wheels/daydream-0.30.0-py3-none-any.whl \
  --reviewer-model gpt-6.1-sol --reviewer-effort high --rounds 3 \
  --platform linux/amd64
/tmp/reviewbench-runtime/bin/reviewbench-check-dataset /tmp/reviewbench-full
```

The profile freezes wheel/package bytes, upstream revision, PR title/body and
base/HEAD, native model/effort, default review profile through package identity,
judge settings/prompts/source, images, Python/Codex/uv versions, dependency lock,
bootstrap/compose policy, custom evaluation IDs and task/dataset metric digests.
Candidate identity and wheel equivalence include the shipped `daydream_ext`
code and resources. Controlled benchmarks reject `DAYDREAM_EXT_DIR` overrides
and default extensions resolving outside the candidate package: package the
desired extension as `daydream_ext` and unset the override. Ordinary Daydream
extension loading is unchanged. Extension identity is checked again after
generation; changed executable inputs make the review unscorable.
Public reviewer admission and execution use isolated Python
startup, so task checkout modules and `PYTHONPATH` cannot replace the installed
adapter. The native reviewer still works in the frozen task repository.
Choose the native Docker architecture explicitly. The default `linux/amd64`
matches upstream's documented platform; `--platform linux/arm64` selects the
custom native ARM64 profile. Keep that choice fixed across the before/after
comparison. Export downloads Moby's Docker seccomp profile at a pinned revision,
verifies its SHA-256, and adds the bounded Codex bootstrap rules. The complete
policy and provenance are frozen in the exported dataset; running it needs no
policy download. The reviewer uses this pinned Docker seccomp profile with
only the namespace bootstrap allowances required by Codex's nested Bubblewrap
sandbox. Docker capabilities stay at their default, and Codex applies its own
read-only filesystem and seccomp restrictions. The selected architecture must
support the actual sandbox, including nested seccomp; running an amd64 binary
alone is insufficient.

Harbor's exact-host kernel isolation is also required. Both roles allow only
`chatgpt.com` and `auth.openai.com`; the runner checks isolation before inference
and preserves that network policy.

Create **two dedicated ChatGPT subscription logins**, one with native Codex and
one with the pinned Pi CLI. They may use the same ChatGPT account, but must come
from separate sign-in flows. Copying or converting one session into two stores
does not make it independent. API-key logins, API fallback, and alternate
providers remain prohibited. These directories must contain only `auth.json`
and runtime lock sidecars, outside all dataset and result directories.

The following setup launches sign-in flows, not inference. Use new, empty
destinations; do not overwrite an existing login. Sign in to ChatGPT for Codex:

```bash
umask 077
mkdir -p "$HOME/.local/share/daydream/reviewbench"
mkdir "$HOME/.local/share/daydream/reviewbench/reviewer"
reviewbench_codex_setup=$(mktemp -d)
CODEX_HOME="$reviewbench_codex_setup" npx --yes --package @openai/codex@0.163.0 \
  codex -c 'cli_auth_credentials_store="file"' login --device-auth
CODEX_HOME="$reviewbench_codex_setup" npx --yes --package @openai/codex@0.163.0 \
  codex -c 'cli_auth_credentials_store="file"' login status
mv "$reviewbench_codex_setup/auth.json" "$HOME/.local/share/daydream/reviewbench/reviewer/auth.json"
rm -r "$reviewbench_codex_setup"
unset reviewbench_codex_setup
```

Next, use the clean pinned ReviewBench checkout with `npm ci` completed. Its
`node_modules/.bin/pi` is **0.75.5**, separate from any globally installed Pi.
Start Pi, enter `/login`, choose **ChatGPT Plus/Pro (Codex Subscription)**, complete a
new sign-in, then `/quit`. Do not send a prompt or import Codex credentials.

```bash
mkdir "$HOME/.local/share/daydream/reviewbench/judge"
reviewbench_pi_setup=$(mktemp -d)
PI_CODING_AGENT_DIR="$reviewbench_pi_setup" /path/to/ReviewBench/node_modules/.bin/pi
# After completing /login and /quit:
mv "$reviewbench_pi_setup/auth.json" "$HOME/.local/share/daydream/reviewbench/judge/auth.json"
rm -r "$reviewbench_pi_setup"
unset reviewbench_pi_setup
chmod 700 "$HOME/.local/share/daydream/reviewbench/"{reviewer,judge}
chmod 600 "$HOME/.local/share/daydream/reviewbench/"{reviewer,judge}/auth.json
```

Run each block only after the preceding command succeeds. These one-time moves
retire the temporary sources; do not copy a session and leave another active
consumer. Reserve both stores for this workflow. Reauthentication follows the
same separate sign-in procedure with all jobs stopped; never delete an active
lock or run an unrelated Codex/Pi client against these stores. The launcher
rejects identical refresh tokens, but cannot prove that different tokens came
from independent sign-ins.

The reviewer mounts only its persistent credential directory at
`/run/reviewbench-codex`. A reviewer-only `flock` covers the whole invocation,
since native Codex does not provide cross-process refresh coordination. Its
fresh private `CODEX_HOME` links directly to persistent `auth.json`; pinned Codex
writes that file in place. There is no source borrowing, conversion, copyback,
or cross-runtime lease. Rotation survives failed invocations and disposable-home
cleanup. Native fan-out and Harbor trial concurrency remain frozen at one.

The verifier mounts only the Pi credential directory at `/run/reviewbench-pi`.
Every judge process uses `/run/reviewbench-pi/auth.json`, with Pi's lock sidecars
on the same writable mount. Pi's existing storage backend owns refresh and lock
recovery. A tiny Node preload binds the pinned evaluator's zero-argument
`AuthStorage.create()` calls to its supported storage API at that path; upstream Pi and
evaluator files remain unchanged. Controlled `NODE_OPTIONS` propagates the preload
through the verifier's evaluator subprocess; inherited Node options are scrubbed.
The binding retries Pi's short synchronous startup lock acquisition for up to
60 seconds so a concurrent refresh cannot masquerade as missing authentication.
Pi still owns every lock and refresh. `PI_CODING_AGENT_DIR` still points to a new
temporary directory for each invocation's models/configuration. Sessions,
workspaces, logs and role homes remain fresh and isolated, and only sanitized
classifier evidence is retained. No credential store is archived.

This changes the unreleased public launch interface: replace `--auth-runtime`
with `--pi-home`, and point `--codex-home` at the dedicated persistent reviewer
store. Re-export datasets with the new candidate wheel because bootstrap,
compose and evaluation identities change. Old shared leases are not migrated;
create fresh independent logins rather than reusing ambiguous recovery state.

`--reviewer-effort` configures the native base effort. Ordinary phase-specific
and arbitration effort overrides remain in effect. Retained invocation evidence
records the actual model and effort used.

**The following command launches reviewer and judge model work.** Select one
task from the full export for integration proof:

```bash
/tmp/reviewbench-runtime/bin/daydream benchmark reviewbench harbor-run \
  /tmp/reviewbench-full \
  --codex-home "$HOME/.local/share/daydream/reviewbench/reviewer" \
  --pi-home "$HOME/.local/share/daydream/reviewbench/judge" --jobs-dir /tmp/reviewbench-jobs \
  --job-name one-task --n-tasks 1 --rounds 1
/tmp/reviewbench-runtime/bin/reviewbench-check-job \
  /tmp/reviewbench-jobs/one-task \
  /tmp/reviewbench-jobs/one-task/reviewbench-metric.py \
  --output /tmp/reviewbench-jobs/one-task/checked.json
```

`--task NAME_OR_GLOB` selects a supplied case; `--n-tasks` limits selection after
filtering. The wrapper invokes Harbor with the public ReviewBench plugin and the
Daydream public agent, without a bare Codex agent or the private Daydream scorer.
For an independently authorized final comparison, omit task filters and use
`--rounds 3`; each task has one separate attempt per round and corresponding
round scores are averaged by the supplied metric. Source membership stays 219;
selected, missing, failed and scored memberships remain distinct. No full run
is authorized merely by exporting its profile.

The supplied reviewer/verifier wall limits remain 900/7200 seconds. Matcher
reasoning stays low, classifier medium. The explicit Pi model metadata declares
8192 max tokens, but its Codex transport does **not** send `max_output_tokens`;
no enforced token cap is claimed. Supported wall/reasoning controls remain in
place. Current model metadata comes from [official OpenAI documentation](https://developers.openai.com/api/docs/models/gpt-6.1-sol);
account availability must also be demonstrated by the actual authenticated run.

Retain `reviewbench-plan.json`, `reviewbench-metric.py`, trial results, findings,
`agent/daydream/execution.json`, native trajectories, verifier logs,
`scores.json`, `scores.details.json` and `reward.json`. The native evidence bundle
under `agent/daydream/runs/<session>/` preserves the root and joined sibling
trajectories with their original logical references; `agent/daydream/trajectory.json`
is a root alias. Temporary archive assembly stays private and is removed after
export. The separate verifier retains sanitized classifier session JSONL and a
digest/model/effort/available-usage index under `verifier/judge-sessions/`.
Session digests cover retained sanitized bytes; credentials are excluded.
The supplied matcher uses an in-memory session and its evaluator does not retain
matcher usage; those measurements remain explicitly unavailable. Judge evidence
contains no authentication store or role home. Partial or conversion-limited
reviews cannot publish findings, and existing output is never reused as a new
success. A valid empty review may require no judge calls: integration acceptance
also needs genuine solver findings and at least one real semantic Codex request.
Classifier execution is conditional on unmatched findings. Supplied null metrics
stay null; token counts and API-equivalent cost estimates are evidence, not
subscription cash charges. Benchmark data never enters capture/training.

Remaining live validation after setup: authorize one selected task, confirm a
nonempty reviewer result triggers a genuine semantic matcher request, and choose
an unmatched finding to exercise the classifier. Confirm model/effort and
retained evidence with the supplied checker. Run a second fresh job using the
same two stores to verify authenticated reuse; verify native refresh persistence
when rotation occurs, without printing or retaining tokens. Synthetic offline
tests cover rotation and contention; they cannot establish account availability,
real provider refresh, Docker mount ownership, or the supported sandbox on the
operator's host. Never infer live success from an empty review or export alone.

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
The package hash covers `daydream` and its shipped `daydream_ext` code/resources;
external extension overrides are refused rather than assigned a partial identity.
Use the candidate installation/checkout itself to create and run its plan; another
revision is a new plan, not a relabeled result. Package-byte changes invalidate a
plan. Standalone generation rechecks the candidate before each case and before
publishing findings; a changed candidate leaves a failed outcome with retained
review evidence and no scorable findings. Configuration is frozen on the first
generation call and must agree across its rounds. Effective native configuration
and profile digests accompany every terminal outcome. The freeze includes native
provider/thinking/retry controls and
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
line hints refer to HEAD and are checked against the captured new-side diff.
Deleted-file, relocated and unlocatable hints remain omissions.
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
