# Review benchmarks

Use `daydream benchmark [reviewbench] COMMAND --help` for flags. Planning/export
performs no inference; generation and judging require experiment authorization.
Keep benchmark inputs/results out of capture and training. Preserve failed evidence
and use fresh output paths for retries or changed candidates/configuration.

## Supplied ReviewBench setup

Both ReviewBench workflows require an unchanged pinned source checkout and Node 20+:

```bash
git clone https://github.com/review-bench/ReviewBench.git ~/ReviewBench
git -C ~/ReviewBench checkout --detach 4f04392d6bf4941ab6f8a3ea0c15b449f4458864
(cd ~/ReviewBench && npm ci)
```

Create and execute plans/exports from the candidate installation itself. Candidate
identity includes shipped `daydream_ext`; package extensions and unset
`DAYDREAM_EXT_DIR`. Changed package bytes require a new plan/export.

## Public Harbor with Codex subscription access

Custom local evaluation: native Codex reviewer, separate upstream Pi judge,
unchanged supplied scoring. Results support matched comparisons, not official
leaderboard eligibility. Keep architecture and evaluation profile fixed across
candidates; standalone and public Harbor judge profiles are distinct.

Build the wheel and install it with
[`harbor-runtime.lock`](../daydream/benchmark/reviewbench/harbor-runtime.lock)
in a separate Python environment. Released Harbor lacks the required metric API.
Run these commands from that environment, using the same candidate wheel:

```bash
daydream benchmark reviewbench harbor-export /private/reviewbench-full \
  --source ~/ReviewBench --wheel /path/to/candidate.whl --rounds 3 \
  --platform linux/amd64
reviewbench-check-dataset /private/reviewbench-full
```

Export always includes the full supplied corpus. Select the host's supported
Docker architecture explicitly (`linux/amd64` or `linux/arm64`); actual nested
Bubblewrap/seccomp and Harbor exact-host kernel isolation must work. Preserve
the frozen sandbox/network policy.

Before launching, create two dedicated ChatGPT subscription logins with Codex
pinned by [`harbor_build.py`](../daydream/benchmark/reviewbench/harbor_build.py)
and Pi from the pinned source checkout:

- Reviewer: fresh temporary `CODEX_HOME`; run Codex with
  `-c 'cli_auth_credentials_store="file"' login --device-auth`, then `login status`.
- Judge: fresh temporary `PI_CODING_AGENT_DIR`; use the pinned source checkout's
  `node_modules/.bin/pi`, `/login` → ChatGPT Plus/Pro (Codex Subscription), `/quit`.
- Move each `auth.json` into its own new persistent directory outside datasets/jobs;
  retire temporary sources. Keep only auth and runtime lock sidecars there, with
  directory/file permissions 700/600. Separate sign-ins may use the same account;
  copied/converted sessions, API keys and provider fallback are inadmissible.
- Reserve both stores for this workflow. Stop jobs before reauthentication; keep
  active locks intact and unrelated Codex/Pi clients away from these stores.

After authorization, prove one selected task before a full comparison:

```bash
daydream benchmark reviewbench harbor-run /private/reviewbench-full \
  --codex-home /private/reviewer-auth --pi-home /private/judge-auth \
  --jobs-dir /private/reviewbench-jobs --job-name one-task --n-tasks 1 --rounds 1
```

The wrapper runs the supplied job checker. Live acceptance requires nonempty
reviewer findings, a real semantic matcher request, an unmatched finding exercising
the classifier, and a second fresh job reusing both stores. Verify model/effort,
retained evidence and refresh persistence when rotation occurs. Export, empty
reviews and offline tests cannot establish live integration success. For an
authorized final comparison, omit task filters and use `--rounds 3`.

Retain the complete job evidence, including plan/metric, native trajectories,
execution, verifier logs/judge sessions and scores; exclude credentials. Preserve
missing/failed/scored membership and null measurements. API-equivalent costs are
not subscription charges.

Change owners: [export/profile](../daydream/benchmark/reviewbench/harbor_profile.py),
[launch/checks](../daydream/benchmark/reviewbench/harbor_cli.py),
[reviewer](../daydream/benchmark/reviewbench/harbor_agent.py),
[authentication](../daydream/benchmark/reviewbench/harbor_auth.py),
[judge binding](../daydream/benchmark/reviewbench/judge_auth.mjs).

## Supplied ReviewBench with native Daydream

Evaluation-only ordinary native reviewer/configuration/authentication; arrange
isolation separately. This workflow supplies no portal container/egress isolation
or leaderboard eligibility.

```bash
daydream benchmark reviewbench plan ~/reviewbench-runs --source ~/ReviewBench \
  --candidate native-current --producer daydream-native --set test --rounds 1
daydream benchmark reviewbench status ~/reviewbench-runs/native-current
# Generation and judging launch models; authorize each experiment first.
daydream benchmark reviewbench run ~/reviewbench-runs/native-current --round 1 \
  -- --backend pi /path/to/operator-config-root
daydream benchmark reviewbench judge ~/reviewbench-runs/native-current --round 1 \
  --provider anthropic --model YOUR_JUDGE_MODEL
daydream benchmark reviewbench report ~/reviewbench-runs/native-current
```

Arguments after `--` retain native precedence. `--pr` is a zero-based selected-set
index; `--repository` supplies that single case's frozen repository. Otherwise
mirror commits are fetched. Inspect `status` and `outcomes/`: complete-empty,
failed, missing and conversion-limited reviews remain distinct; only complete,
convertible results are scorable.

Each `round-N/findings/` is one evaluator input. Passing a multiple-round parent
silently merges identical PR identities. Preserve interrupted evaluations and
create a fresh plan; `evaluator.json` is published last.

Final comparisons use fresh `--set full --rounds 3` plans: independently run/judge
rounds 1–3, then `report` averages corresponding supplied ratios. `RB_ATTEMPT` is
a retry counter, not a round. Disclose subsets/nulls; augmented recall has an
agent-dependent denominator.

Change owners: [CLI](../daydream/benchmark/reviewbench/cli.py),
[source/cases](../daydream/benchmark/reviewbench/contract.py),
[native execution/conversion](../daydream/benchmark/reviewbench/native.py),
[evaluation/reporting](../daydream/benchmark/reviewbench/evaluator.py).
Source-backed offline tests:
`DAYDREAM_REVIEWBENCH_SOURCE=~/ReviewBench uv run pytest tests/test_reviewbench_*.py`.

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

Change owners: [CLI](../daydream/benchmark/cli.py),
[environment policy](../daydream/benchmark/harbor/env_policy.py),
[curation](../daydream/benchmark/curation.py),
[storage/recovery](../daydream/benchmark/storage.py),
[run gates](../daydream/benchmark/harbor/run.py),
[objective/aggregation schema](../daydream/benchmark/harbor/objective.py).
