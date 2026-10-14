# Training the review policy with prime-rl

Three configs live here:

- `rl.toml` — the Stage-2/3 GRPO recipe for the `daydream-review` environment.
- `sft.toml` — the Stage-1 dataset-SFT recipe over the labeled corpus (separate
  `sft` entrypoint, not `rl @`).
- `rft.toml` — the deterministic RFT replay reference parameters
  (daydream-private, passed at launch via `RftConfig` to
  `daydream.training.rft.run_rft()`; the file is reference-only, not prime-rl).

## Integration paused

These recipes are retained as reference configurations. The Daydream package
and standalone RL environment require Python 3.14; the environment pins
verifiers 0.4.0. Prime-RL v0.7.0 and the latest v0.10.0 require Python 3.12,
so their workspace setup cannot install this Daydream release. The previous
workspace installation and dry-run instructions have been retired. Do not
force an install with ignored dependency or interpreter constraints.

Standalone environment validation remains supported:

```bash
cd /abs/path/to/daydream
make rl-check
```

Resume training only after selecting a compatible training workspace,
validating these reference configs with its real dry-run entrypoint, and
passing the complete environment suite inside that workspace. Earlier
vendored-verifier and training dry-run measurements apply to the previous
Python 3.12 / verifier 0.2.x environment.

Stage-2 deterministic RFT remains available through Daydream's offline,
GPU-free `daydream train` path; it does not require Prime-RL.

## Manifests

`[[orchestrator.train.env]]` and `[[orchestrator.eval.env]]` point at **two
different manifest files** (`--taskset.manifest-path`). That is the whole
train/eval split: there is no split flag, and the taskset refuses the five
held-out benchmark repositories outright, in both manifests, with no bypass
(SPEC C5).

Effective upstream concurrency is roughly `pool.max_workers × fanout_concurrency`
— each rollout runs its own parallel exploration and per-stack fan-out.
