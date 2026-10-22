# Review benchmarks

Use `daydream benchmark [reviewbench] COMMAND --help` for flags. Export performs
no inference; benchmark generation and judging require separate experiment
authorization. Keep all benchmark inputs/results out of capture and training.
Preserve failed evidence and use fresh output paths for changed candidates or
configuration.

## Supplied ReviewBench through Harbor

Harbor is the sole public ReviewBench workflow. The supplied exporter owns frozen
corpus/context/snapshots, references, instructions and judging assets. Harbor owns
selection, trial scheduling and independent attempts; the supplied ReviewBench
plugin freezes task/evaluation/round membership. Its metric and completed-job
checker own scoring and reports. Daydream supplies the ordinary native reviewer,
conservative finalized-finding conversion and the necessary candidate,
authentication, isolation and evidence integration.

These are custom local evaluations, not official leaderboard submissions. The
[comparison acceptance](https://github.com/existential-birds/daydream/issues/1672#issuecomment-6303577385)
uses the exact supplied 25-task test set with three independent rounds per
candidate (75 trials each). Full219 remains available through the same path.
Keep the judge, supplied prompts/scorer and isolation fixed; identify each
producer's package, profile and native settings. Test25 does not establish quality
on roughly 20,000-line reviews. References are incomplete and augmented recall has
an agent-dependent denominator; offline integration checks establish no quality or
performance improvement.

### Source and candidate setup

Use a clean pinned source checkout with Node 20+:

```bash
git clone https://github.com/review-bench/ReviewBench.git ~/ReviewBench
git -C ~/ReviewBench checkout --detach 4f04392d6bf4941ab6f8a3ea0c15b449f4458864
(cd ~/ReviewBench && npm ci)
```

Build the candidate wheel and install it with
[`harbor-runtime.lock`](../daydream/benchmark/reviewbench/harbor-runtime.lock)
in a separate Python environment. Released Harbor lacks the required metric API.
Run export and launch from that candidate installation. Candidate identity binds
all shipped code/resources and packaged `daydream_ext`; unset `DAYDREAM_EXT_DIR`.
The host contains Harbor/ReviewBench. The reviewer image installs only the core
Python graph plus its selected native executable; verifier sources, references and
retained source caches stay out of it. The Pi reviewer has its own pinned npm
dependency graph, independent from the supplied judge's Pi SDK.

Export always keeps all 219 tasks and the full metric catalog. Select the host's
supported Docker architecture (`linux/amd64` or `linux/arm64`). Actual nested
Bubblewrap/seccomp and Harbor exact-host kernel isolation must work. The exporter
receives separate reviewer/judge hosts; runtime egress stays confined to those
exact hosts.

Pass ordinary native reviewer arguments and an operator configuration root after
`--`. Project/global settings and explicit model/provider/effort overrides retain
native precedence. For example, freeze the configured Pi/Nous reviewer or native
Codex reviewer into separate exports using the same candidate wheel:

```bash
daydream benchmark reviewbench harbor-export /private/reviewbench-pi \
  --source ~/ReviewBench --wheel /path/to/candidate.whl --rounds 3 \
  --platform linux/arm64 -- --backend pi /path/to/operator-config-root

daydream benchmark reviewbench harbor-export /private/reviewbench-codex \
  --source ~/ReviewBench --wheel /path/to/candidate.whl --rounds 3 \
  --platform linux/arm64 -- --backend codex /path/to/operator-config-root
reviewbench-check-dataset /private/reviewbench-pi
reviewbench-check-dataset /private/reviewbench-codex
```

The required Pi arm preserves the operator's `nous` /
`deepseek/deepseek-v4.1-flash` configuration and native authentication, including
its selected custom provider definition. It does not change ordinary fallback
models. Unsupported selected capabilities fail before model work. Ordinary
Claude/Osprey use remains available outside this public Pi/Codex integration;
Osprey retains its ambient transport.

### Independent authentication and judge

Both reviewer arms use the same supplied Pi-path judge with explicit
`openai-codex` / `gpt-6.1-sol` subscription policy, matcher/classifier settings,
prompts and raw scoring. Reviewer selection never determines judge model/provider.
Credentials stay in native role capabilities/mounts, outside profiles, hashes,
agent kwargs, exports and jobs.

For the Codex arm, use a dedicated persistent auth-only `CODEX_HOME`. Create its
subscription login separately with the pinned Codex CLI and file credential store.
For the Pi arm, use its native reviewer authentication; keep its directory distinct
from the judge. Do not convert/copy Codex sessions, force OpenRouter or change
operator settings/billing routes.

The independent judge directory uses the pinned supplied source's
`node_modules/.bin/pi`: `/login` → ChatGPT Plus/Pro (Codex Subscription) → `/quit`.
Reviewer and judge subscription stores need independently created sessions and
native refresh locks. Keep stores outside exports/jobs, reserve them for this
workflow and stop jobs before reauthentication. Do not change credentials merely
to validate this integration.

### Exact selection and independent rounds

After experiment authorization, the canonical test journey is:

```bash
daydream benchmark reviewbench harbor-run /private/reviewbench-pi \
  --reviewer-pi-home /private/pi-reviewer-auth --pi-home /private/judge-auth \
  --jobs-dir /private/reviewbench-jobs --job-name pi-test25 --set test --rounds 3

daydream benchmark reviewbench harbor-run /private/reviewbench-codex \
  --codex-home /private/codex-reviewer-auth --pi-home /private/judge-auth \
  --jobs-dir /private/reviewbench-jobs --job-name codex-test25 --set test --rounds 3
```

`--set test` selects the exact 25 full case identities from verified
[`corpus/test/test.json`](https://github.com/review-bench/ReviewBench/blob/4f04392d6bf4941ab6f8a3ea0c15b449f4458864/corpus/test/test.json).
`--set full --rounds 3` selects all 219 (657 trials). The launcher checks delivered
case/profile identities before model work and supplies exact names to Harbor.
The plugin retains `source_prs=219` even for test25.

For a smaller smoke run, replace `--set test` with one or more repeated
`--task EXACT_EXPORTED_NAME` flags and `--rounds 1`. Subsets are identified as
subsets, not complete comparisons. Globs/count limits cannot name the supplied
test set. Harbor attempts are independent rounds; retries/best-of are not a
substitute. Keep all raw trial/round evidence.

After Harbor exits zero the launcher explicitly invokes the supplied completed-job
checker. Harbor logs plugin-finalization exceptions as warnings, so plugin
invocation alone cannot admit a successful comparison. The supplied metric
computes round means, population standard deviations and null arithmetic; there
is no second Daydream evaluator, receipt, averaging or report format.

Retain plans/metrics, native terminal configuration/usage and joined trajectories,
verifier logs/judge sessions, raw scores and supplied summary. Distinguish expected,
produced, completed-empty, failed, incomplete, conversion-limited, missing and
evaluated work; unconvertible findings remain visible omissions. Native ignore
rules that change the analyzed patch cause a visible comparison failure. Missing
measurements stay unavailable; observed zero remains zero. API-equivalent estimates
are not subscription charges. Native private staging remains required before
no-clobber publication to Harbor logs; the log mount is not native artifact storage.

### Compatibility and offline checks

This redesign removes PR-only standalone `plan`, `run`, `status`, `judge` and
`report` commands, their receipts/readback and custom reporting. Only
`harbor-export`/`harbor-run` remain. Ordinary native arguments replace the export's
PR-only reviewer-model/effort flags, and exact `--set`/repeatable `--task` replaces
`--n-tasks`. Policy/task profiles use schema 3: create fresh exports and preserve
old evidence in its original directory. Incompatible formats are rejected without
migration. No version bump is included; ordinary runtime interfaces are preserved.

Source-backed offline checks:

```bash
PATH="$PWD/.venv/bin:$PATH" DAYDREAM_REVIEWBENCH_SOURCE=~/ReviewBench \
  uv run pytest tests/test_reviewbench_*.py tests/test_templates_packaging.py tests/test_json_utils.py
```

Check `uv lock --check` first. `DAYDREAM_REVIEWBENCH_RUNTIME` selects a separately
installed pinned host Python for actual Harbor/plugin planning proof;
`DAYDREAM_REVIEWBENCH_REVIEWER_IMAGE` enables real image/seccomp checks.
`DAYDREAM_REVIEWBENCH_NATIVE_PI` selects the pinned native Pi package directory for
authentication/lock checks. Build the current candidate wheel, rebind that runtime
and rebuild its image before using these checks as evidence. Offline checks launch no benchmark inference.

Change owners: [launch/selection](../daydream/benchmark/reviewbench/harbor_cli.py),
[export/profile](../daydream/benchmark/reviewbench/harbor_profile.py),
[native execution](../daydream/benchmark/reviewbench/native.py),
[adapter](../daydream/benchmark/reviewbench/harbor_agent.py),
[authentication](../daydream/benchmark/reviewbench/harbor_auth.py),
[judge binding](../daydream/benchmark/reviewbench/judge_auth.mjs).

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
