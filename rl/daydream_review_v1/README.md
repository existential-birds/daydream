# daydream-review-v1

A [verifiers](https://github.com/PrimeIntellect-ai/verifiers) **v1 environment**:
one rollout is one headless daydream deep review→fix→test run inside a sandbox,
scored on two axes — daydream's own intrinsic trajectory composite, and a
deterministic re-run of the repository's test suite against the fixed tree.

Standalone uv project on purpose: verifiers pulls ~100 packages and must never
enter daydream's lockfile. `daydream` is a path dependency (`../..`, editable) so
the reward imports the *same* `score_trajectory` the offline training pipeline
uses — one scorer, so an online reward and an offline label cannot disagree about
the same run.

- Taskset id / module: `daydream-review-v1` / `daydream_review_v1`
- Pins: `verifiers==0.2.1`, prime-rl `v0.7.0` (see `../train/README.md`)

```bash
cd rl/daydream_review_v1
uv sync
uv run ruff check . && uv run mypy daydream_review_v1 tests && uv run pytest
```

## How a rollout works

```text
Taskset.load()          harvested corpus + images/manifest.toml, C5-enforced
  → Runtime             one container per rollout, from the PR's own image
  → Harness.launch()    daydream --non-interactive --yes --backend <b> --model <m> --base <sha>
       every model turn ↴
  → interception server (host) → the policy endpoint
  → Task.score()        while the sandbox is still live
       intrinsic_composite  archived run dir → score_trajectory()
       fix_tests_pass       re-run the repo's own suite, exit code
```

Nothing clones at rollout time and no rollout carries credentials: the repository
is baked into the image at the task's head SHA with `origin` pointing at an
in-container mirror, so daydream's terminal commit-and-push stays inside the
container.

## Backends

`--backend` is a harness config key, not a rewrite. The interception server serves
all three wire dialects simultaneously and the route selects the format, so each
backend only needs to learn a base URL and a key:

| `backend` | dialect | how the endpoint gets in |
|---|---|---|
| `claude` (default) | Anthropic Messages | `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` |
| `codex` | OpenAI Responses | `$CODEX_HOME/config.toml` provider block + `CODEX_INTERCEPT_KEY` |
| `pi` | Chat Completions | an installed pi provider extension + `VF_INTERCEPT_API_KEY` |

`claude` is the day-one default only because its CLI is what the base image
already carries. `osprey` lands as one more class when daydream ships that
backend.

## Adding a repository

1. **Harvest a corpus.** Any repo NOT in `daydream/training/schema/exclusion.txt`:

   ```bash
   daydream bench harvest --repo OWNER/REPO --bot "<bot-login>[bot]" --out ./corpus-train
   ```

   Target merged-PR snapshots. The image build requires the suite to be green at
   the head commit, and a merged PR in a CI-disciplined repo was green at merge.

2. **Add a manifest entry** in `images/manifest.toml`. `image` is a docker
   repository name with NO tag — the tag is the task's 12-char head SHA, so one
   image is exactly one PR snapshot.

3. **Build the images.** The last layer runs the repository's own suite at the
   head commit; a red baseline fails the build and produces nothing, because
   `fix_tests_pass` would otherwise be rewarding noise.

   ```bash
   python images/build_images.py --corpus ./corpus-train
   python images/build_images.py --only OWNER/REPO          # one repo
   python images/build_images.py --red                      # prove the gate fails
   ```

## Running an eval

Local smoke — real daydream, real interception, canned upstream, meaningless
rewards, about ten seconds:

```bash
python -m daydream_review_v1.fixture /tmp/daydream-rl-smoke/repo
mkdir -p /tmp/daydream-rl-smoke/archive /tmp/daydream-rl-smoke/home
python -m daydream_review_v1.stub_upstream --port 8399 &
uv run eval @ configs/eval-stub.toml
```

Docker runtime, the real rollout shape:

```bash
python images/build_images.py --only existential-birds/daydream-rl-fixture
uv run eval @ configs/eval-docker.toml --client.base-url <your endpoint>
```

**Two flags are not optional offline.** `uv run eval` defaults to `push = true`
(it uploads the run to Prime's platform) and to an upstream of
`https://api.pinference.ai/api/v1`, so any offline run must set `push = false`
and an explicit `--client.base-url`. The URL must keep its `/v1` suffix: the
client string-appends the dialect's path, and omitting it yields an HTML 404.
Both are baked into `configs/*.toml`.

## Things worth knowing before you trust a number

- **A rollout that finds NOTHING scores `intrinsic_composite` 1.0.** Grounding is
  vacuously perfect over an empty finding set. `score_trajectory` is reused
  verbatim on purpose, so the fix belongs to the Stage-0 rubric (#91); until then,
  watch the `n_findings` metric next to the reward. Reward climbing while
  `n_findings` falls is the policy learning to say nothing.
- **The correctness axis exists only when the fix gate was accepted.**
  `deep/recommendation-verdicts.json` is written only on that branch, so a
  review-only rollout scores on grounding and format alone.
  `trace.info["reward_breakdown"]["axes_present"]` records which axes were live.
- **Never pass `--no-eval` or `--no-archive`.** The grounding axis comes from the
  archive-time eval pass; without it the axis is silently null.
- **`golden_overlap` is a crude localisation proxy**, not a reward. It feeds
  #91's rubric design and is deliberately never summed.
- **Rollout cost is real.** A deep run is minutes and dollars. The task caps the
  harness at 5400s; daydream bounds each phase at 1800s of its own.
