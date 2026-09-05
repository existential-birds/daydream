# daydream-review-v1

A [verifiers](https://github.com/PrimeIntellect-ai/verifiers) **v1 environment**:
one rollout is one headless daydream deep review→fix→test run inside a sandbox,
scored on a single reward axis — daydream's own intrinsic trajectory composite.
The post-change suite result is recorded as the ``suite_non_regression`` metric
(a green suite proves the tree did not regress, not that the reported defect was
repaired, so it earns no training signal of its own). Scoring consumes only
sealed artifacts: the supervisor seals the archived run dir after the agent's
write window, the reward verifies the seal against the staged copy before
trusting any value, and the suite re-run executes against a separate
root-owned read-only checkout under a distinct non-root verifier identity —
never the agent-mutable tree.

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
       intrinsic_composite  sealed archived run dir → score_trajectory()
       suite_non_regression  metric: re-run the repo suite in a read-only checkout
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

`claude` is the default backend — for smoke runs and the real eval rollout
(`configs/eval-docker.toml` also sets `backend = "claude"`) — because its CLI is
what the base image already carries and its injection needs no provisioning file.
`osprey` lands as one more class when daydream ships that backend.

**Only `pi` can train.** Two facts compose. The interception server passes the
agent's dialect straight through to the upstream — the URL is
`base_url + dialect.upstream_path` (`verifiers/clients/eval.py:95`), no cross-dialect
adapter. And at training time the upstream is verifiers' renderer client, which
tokenizes with an HF chat template, calls vLLM's `/inference/v1/generate` (this
is how `token_ids` and `logprobs` reach the trace), and raises
`NotImplementedError` on any dialect but Chat Completions
(`verifiers/clients/train.py:233-239`).

So `codex` (Responses) and `claude` (Anthropic Messages) work for an **eval**
against a hosted provider and cannot be used for a **training run** at these
pins. `pi` speaks Chat Completions end to end. It is also the backend proven on a
real model: 79 captured turns, 3 findings, a fix applied and committed,
`suite_non_regression` 1.0 from a genuine in-sandbox suite re-run.

Measure a baseline under the **same backend you will train under**. The agent
scaffold is part of what the number describes; comparing a codex-scaffold
baseline against a pi-scaffold trained run moves two variables at once.

## Adding a repository

1. **Provide a harvested corpus.** Any repo NOT in `daydream/training/schema/exclusion.txt`:

   ```bash
   # corpus shape: an index.json (harvested-PR records) plus a results/ dir.
   # The old `daydream bench harvest` command was removed with the Martian stack
   # (issue-785); point `--corpus` at an existing harvested-corpus directory.
   ```

   Target merged-PR snapshots. The image build requires the suite to be green at
   the head commit, and a merged PR in a CI-disciplined repo was green at merge.

2. **Add a manifest entry** in `images/manifest.toml`. `image` is a docker
   repository name with NO tag — the tag is the task's 12-char head SHA, so one
   image is exactly one PR snapshot.

   Every repository with a non-empty `setup_cmds` must satisfy four rules:

   - The dependency lockfile is **committed at the head SHA** baked into the
     image, so the image installs exactly the dependency set that commit pins.
   - The package manager runs in a mode that **rejects lock drift** (e.g.
     `uv sync --locked`): a lock that is missing or no longer matches fails the
     build rather than silently resolving a different dependency set.
   - The generated dependency environment lives **outside `/work/repo`** (e.g.
     `UV_PROJECT_ENVIRONMENT=/opt/repo-venv`), so the baked checkout stays
     clean and the environment survives into the image.
   - `test_command` invokes the environment produced by that locked setup (e.g.
     `/opt/repo-venv/bin/python -m pytest -q`), so the green-baseline gate and
     the rollout's `suite_non_regression` re-run both exercise the locked dependencies.

   An unavailable or stale lock is an **image-build failure** — never permission
   to **fall back to unconstrained pip**.

   Every manifest entry must also declare a **required, nonempty
   `protected_test_paths`** array: the literal repository-relative files or
   directories that constitute the repository's test oracle. A missing or empty
   inventory is a manifest-load error, never a silently unprotected task.
   Coverage must include the test sources **and every runner-config file
   `test_command` can load** — including absent filenames whose later creation
   could alter collection (e.g. `conftest.py`, `pytest.ini`, `pyproject.toml`,
   `setup.cfg`, `tox.ini`). At scoring time `suite_non_regression` compares
   these paths against the image's baked head SHA, fail-closed: a changed or
   unverifiable oracle records **zero non-regression telemetry** and the
   repository's mutable `test_command` is not executed. That covers any tracked difference, any
   non-ignored untracked file under a protected path or an untracked root
   `sitecustomize.py` (imported at startup by every `python` run, so one that
   `sys.exit(0)`s makes a suite that never ran look green), a
   `skip-worktree`/`assume-unchanged` flag on a protected file, any change to
   the ignore rules the probes honor (tracked or new untracked `.gitignore`
   files, or a rule written to `.git/info/exclude`), or a Git error.

3. **Build the images.** The last layer runs the repository's own suite at the
   head commit; a red baseline fails the build and produces nothing, because
   `suite_non_regression` would otherwise be measuring noise.

   ```bash
   uv run python images/build_images.py --corpus ./corpus-train
   uv run python images/build_images.py --only OWNER/REPO      # one repo
   uv run python images/build_images.py --red --only existential-birds/daydream-rl-fixture  # prove the gate fails (requires a fixture PR selected)
   ```

   ``--red`` requires at least one selected fixture PR (``fixture://daydream-rl-fixture``)
   and refuses ``--base-only``; both refusals exit status 2 before any build. Mixed
   selections are accepted but non-fixture repositories in them are never mutated
   by ``--red``.

## Running an eval

Local smoke — real daydream, real interception, canned upstream, meaningless
rewards, about ten seconds:

```bash
uv run python -m daydream_review_v1.fixture /tmp/daydream-rl-smoke/repo
mkdir -p /tmp/daydream-rl-smoke/archive /tmp/daydream-rl-smoke/home
uv run python -m daydream_review_v1.stub_upstream --port 8399 &
uv run eval @ configs/eval-stub.toml
```

Docker runtime, the real rollout shape:

```bash
uv run python images/build_images.py --only existential-birds/daydream-rl-fixture
uv run eval @ configs/eval-docker.toml -m <your policy model id> --client.base-url <your endpoint>
```

**Two flags are not optional offline.** `uv run eval` defaults to `push = true`
(it uploads the run to Prime's platform) and to an upstream of
`https://api.pinference.ai/api/v1`, so any offline run must set `push = false`
and an explicit `--client.base-url`. The URL must keep its `/v1` suffix: the
client string-appends the dialect's path, and omitting it yields an HTML 404.
Both are baked into `configs/*.toml` — `push = false` in both eval-stub and
eval-docker; the explicit `--client.base-url` is in `configs/eval-stub.toml`
(the docker config has no `[client]` block and takes it from the CLI).

## Things worth knowing before you trust a number

- **A rollout that finds NOTHING scores `intrinsic_composite` 0.0, not 1.0.**
  `analyze_grounding` returns `grounding_rate = None` over an empty finding set
  (undefined, not vacuously perfect), so no credit axis is present and
  `score_trajectory` returns `composite = None`, mapped to 0.0 at the reward
  boundary. A correct "nothing wrong here" therefore scores the same as a broken
  run; any positive floor for a genuinely clean review is reward design and
  belongs to the Stage-0 rubric (#91). Keep watching the `n_findings` metric
  next to the reward — reward climbing while `n_findings` falls is the policy
  learning to say nothing.
- **The correctness axis exists only when the fix gate was accepted.**
  `deep/recommendation-verdicts.json` is written only on that branch, so a
  review-only rollout scores on grounding and format alone.
  `trace.info["reward_breakdown"]["axes_present"]` records which axes were live.
- **Never pass `--no-eval` or `--no-archive`.** The grounding axis comes from the
  archive-time eval pass; without it the axis is silently null.
- **Grounding is location-aware, not just file-aware.** A finding counts as
  grounded only when the agent actually read the file it cites *and* the line it
  cites resolves inside (or within tolerance of) a diff hunk. A finding pinned to
  a real file at a line the diff never touched is ungrounded, so the axis can drop
  without the policy having invented a filename. `grounding_rate` therefore has
  its own version stamp discipline: it is part of what
  `daydream/training/reward.py`'s `REWARD_VERSION` identifies, and the pin in
  `tests/test_rewards.py::test_reward_version_is_pinned` is what fails when the
  predicate moves under a run.
- **`golden_overlap` is a crude localisation proxy**, not a reward. It feeds
  #91's rubric design and is deliberately never summed.
- **Scoring trusts only sealed state.** The supervisor seals the archived run
  dir (with the candidate diff) after the launch returns; the reward verifies
  the seal against its single staged copy before any value is trusted, and a
  tampered archive zeroes `intrinsic_composite` and records
  `suite_non_regression` 0.0 — never honest telemetry. The container launches
  daydream through the root-owned `run-as-agent` wrapper (setpriv down to the
  non-root `agent` user), so no backend CLI subprocess can write the sealed
  surfaces, and the suite re-run runs under a distinct non-root `verifier`
  identity against a root-owned read-only checkout.
- **Rollout cost is real.** A deep run is minutes and dollars. The task caps the
  harness at 5400s; daydream bounds each phase at 1800s of its own.

## Vendored-verifiers skew (AC10)

**The skew:** this env pins `verifiers==0.2.1`, but prime-rl `v0.7.0` vendors
**verifiers 0.2.0** (submodule `deps/verifiers`). Task 0's spike validated that
the API surface this env uses is compatible across 0.2.0/0.2.1 in both
directions (env suite: 172 passed against the vendored copy; `rl` dry run green
against 0.2.1), so the resolution is **pin discipline** — no fork, no version
relaxation.

**Resolution:** keep `verifiers==0.2.1` pinned exactly in
`rl/daydream_review_v1/pyproject.toml` and `uv.lock`. The gate test
`tests/test_vendored_verifiers_suite.py` re-runs the env suite with the vendored
0.2.0 copy shadowing the pin via `PYTHONPATH` precedence (no venv mutation).

**Running the gate:**

```bash
export PRIME_RL_VENDORED_VERIFIERS=<prime-rl>/deps/verifiers
cd rl/daydream_review_v1 && uv run pytest tests/test_vendored_verifiers_suite.py -q
```

Expected: PASS. When `PRIME_RL_VENDORED_VERIFIERS` is unset the gate skips
LOUDLY with instructions — never silently. A training claim is only valid when
this gate has run green.
