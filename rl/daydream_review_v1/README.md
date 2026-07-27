# daydream-review-v1

A [verifiers](https://github.com/PrimeIntellect-ai/verifiers) **v1 environment** that runs
daydream's deep review→fix→test loop as a rollout Harness inside a sandbox, and scores it on
two axes: daydream's own intrinsic trajectory composite, and a deterministic in-sandbox
re-run of the repository's test suite.

Standalone uv project on purpose: verifiers pulls ~100 packages and must never enter
daydream's lockfile. `daydream` itself is a path dependency (`../..`, editable) so the reward
imports the *same* `score_trajectory` the training pipeline uses — one source of truth.

- Taskset id / module: `daydream-review-v1` / `daydream_review_v1`
- Pins: `verifiers==0.2.1`, prime-rl `v0.7.0`

Full usage — corpus harvesting, manifest entries, image builds, eval and training — is
documented in Phase 6 of `.beagle/plans/verifiers-env-164/plan.md` and expanded here as the
phases land.

## Quick start

```bash
cd rl/daydream_review_v1
uv sync
uv run ruff check .
uv run pytest
```
