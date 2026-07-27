# Training the review policy with prime-rl

`rl.toml` here is the GRPO recipe for the `daydream-review-v1` environment. It is
config only — no package, no lockfile — because **prime-rl cannot be consumed as a
dependency**. At v0.7.0 prime-rl resolves verifiers from an editable path source
pointing at its own submodule (`[tool.uv.sources] verifiers = { path = "deps/verifiers" }`),
so a `prime-rl @ git+…` line in a pyproject does not resolve. The supported shape
is a prime-rl workspace checkout that you install our environment package into.

## Workspace setup

```bash
git clone --recurse-submodules https://github.com/PrimeIntellect-ai/prime-rl
cd prime-rl
git checkout v0.7.0

# Three of the four submodules are declared with `git@github.com:` URLs, so on a
# machine with no SSH key to GitHub the recursive clone above silently leaves
# deps/verifiers empty — and a `uv sync` against an empty path source fails in a
# way that does not mention submodules. They are public repositories, so point
# them at HTTPS for this checkout. (A `url.<base>.insteadOf` rewrite is NOT
# enough here; set the submodule URLs themselves.)
for m in verifiers renderers research-environments; do
  git config "submodule.$m.url" "https://github.com/PrimeIntellect-ai/$m.git"
done
git submodule update --init --recursive

uv sync --all-packages
uv pip install -e /abs/path/to/daydream/rl/daydream_review_v1
```

The loader needs nothing more than an importable top-level module whose name is
the taskset id with underscores — no entry points, no registry. Dropping the
package under `deps/verifiers/environments/` works too (it is inside prime-rl's
workspace glob), but an editable install is less surprising.

## Validate before spending a GPU-hour

```bash
uv run rl @ /abs/path/to/daydream/rl/train/rl.toml --dry-run
```

`--dry-run` runs every pydantic validator — including renderer resolution, the
LoRA/weight-broadcast interlock, and the batch-size/group-size divisibility check
— and returns before it ever touches `pynvml`, so it needs no GPU. Resolved
configs land in `<output_dir>/configs/`.

## The verifiers skew you must check

prime-rl v0.7.0 vendors verifiers at fork commit `6c64ce6`, which **predates the
`0.2.1` this environment develops against**. Two different verifiers, one package.
Before trusting any training claim, run the environment's own suite inside the
prime-rl workspace venv, not just in `rl/daydream_review_v1`:

```bash
cd prime-rl
uv run pytest /abs/path/to/daydream/rl/daydream_review_v1/tests
```

Nothing in `daydream_review_v1` may import verifiers internals beyond the
documented v1 surface (`Taskset`, `Task`, `TaskData`, `TaskConfig`,
`TasksetConfig`, `Harness`, `HarnessConfig`, `Runtime`, `Trace`, `reward`,
`metric`, `ProgramResult`, `TaskTimeout`, `ModelContext`, `State`). If something
is missing under the older submodule, pin a newer submodule in a prime-rl fork or
align to the intersection — never paper over it with `try`/`except ImportError`,
which would let a silently-degraded rollout train.

## What the recipe deliberately does not pin

- **The base model.** `[model] name` is a placeholder that exists only because
  prime-rl requires the key. The real model is chosen by criteria and passed at
  launch: `--model.name <id>` (SPEC C1).
- **`target_modules`.** SPEC C2 requires prime-rl's default list, so the recipe
  omits the key rather than restating it.
- **The algorithm.** GRPO is prime-rl's default at this tag, so there is no
  `[orchestrator.algo]` block. Valid types are `grpo|echo|max_rl|opd|opsd|sft`;
  there is no `custom` algorithm — a custom *loss* is `[trainer.loss]
  type = "custom"`.

## Corpora

`[[orchestrator.train.env]]` and `[[orchestrator.eval.env]]` point at **two
different corpus directories**. That is the whole train/eval split: there is no
split flag, and the taskset refuses the five held-out benchmark repositories
outright, in both corpora, with no bypass (SPEC C5).

Effective upstream concurrency is roughly `pool.max_workers × fanout_concurrency`
— each rollout runs its own parallel exploration and per-stack fan-out.
