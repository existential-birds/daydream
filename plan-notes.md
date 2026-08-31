# plan-notes.md — Task 0 spike findings (prime-rl dry path, dataset-SFT schema, verifiers skew)

Workspace used for the spike: `/home/exedev/prime-rl`, cloned from
`https://github.com/PrimeIntellect-ai/prime-rl`, checkout **v0.7.0** (d334ea529),
submodules re-pointed at HTTPS per `rl/train/README.md`, `uv sync --all-packages`
(see "Python version" caveat below), then
`uv pip install -e /home/exedev/repo/rl/daydream_review_v1`.
All probes run on a GPU-less Linux x86_64 box (no `nvidia-smi`).

## Step 1 — canonical dry path: CONFIRMED

```
uv run rl @ /home/exedev/repo/rl/train/rl.toml --dry-run
# INFO Training from scratch, cleaning any stale rollouts and broadcasts
# INFO Wrote subconfigs to outputs/configs
# SUCCESS Dry run complete. To start an RL run locally, remove --dry-run from your command.
exit 0
```

Resolved `outputs/configs/inference.toml` came back with `enable_lora = true` and
`max_lora_rank = 16` (derived from the trainer LoRA block), matching the README.

**pynvml nuance (correction to the README's wording):** `-X importtime` on the dry
run shows `pynvml` **is imported** (transitively, early in prime-rl config loading),
so a strict `assert 'pynvml' not in sys.modules` after the run FAILS. What actually
holds — and what matters — is that the dry run **never touches a GPU**: it succeeds
on a machine with no GPU driver at all. The claim to carry forward is "dry run is
GPU-free (no GPU query/driver needed)", not "pynvml is never imported".

Python-version caveat for the recipe: prime-rl's lockfile resolves fine on the
system 3.12.3, but our env package requires `>=3.12.13`, so the workspace venv was
recreated with `uv venv --python 3.12.14 --clear && uv sync --all-packages` before
installing `daydream_review_v1`. Future task instructions that say "install our env
package into a prime-rl checkout" need that step on hosts with 3.12.<13.

## Step 2 — dataset-SFT schema: CONFIRMED (shape differs from the RL config)

- **Entrypoint is a separate verb**: `sft` (`prime_rl.entrypoints.sft:main`), i.e.
  `uv run sft @ <cfg>.toml --dry-run` — NOT `rl @ ...`. There is also a bare
  `trainer` entrypoint. Schema lives in
  `packages/prime-rl-configs/src/prime_rl/configs/sft.py` (`SFTConfig`).
- **Dataset mode is `[data] type = "sft"`** (default) with `data.name` = HF dataset
  name **or local path**; alternates are `[data] type = "fake"` (bench) — the
  discriminated union is `FakeDataConfig | SFTDataConfig` on `type`. Columns:
  the loader (`src/prime_rl/trainer/sft/data.py::SFTDataset._process`) accepts
  either a `messages` column (whole-chat) **or both `prompt` and `completion`
  columns**; anything else raises. Tool columns `tools`/`tool_defs` accepted
  (OAI or verifiers shape). So prompt/completion JSONL works directly.
- **Live-teacher vs dataset distinction is clean and config-level**: the
  orchestrator-side algorithm union (`rl @` runs, `configs/algorithm.py`) declares
  `type: grpo|max_rl|opd|opsd|sft|echo` — `[orchestrator.algo] type = "sft"` is the
  live-teacher variant inside an RL run; the dataset SFT path never has an
  orchestrator block at all (different entrypoint + different config class). M10's
  assertion test can distinguish them by entrypoint/config-class, not by guesswork.
- **LoRA**: `SFTConfig.model.lora` (same `LoRAConfig` as RL: `rank`, `alpha`,
  `target_modules`, ...). Probed `rank = 64` and `rank = 100`: both validate; SFT
  keeps the rank verbatim (no vLLM `max_lora_rank` rounding — that auto-setup only
  exists on the inference config). `rank = 128` is inside the valid band for the
  same reason. `save_adapter_separately = true` validated with LoRA enabled.
- **Renderer**: `[renderer] name = "default"` is accepted on `SFTConfig`
  (`renderer: RendererConfig | None`; `None` falls back to the legacy incremental
  token-mask path). Resolved `outputs/configs/sft.toml` round-trips
  `name = "default"`, the LoRA block, and the local dataset path.
- Probe file: `/tmp/sftprobe/sft-probe.toml` + 2-line prompt/completion JSONL;
  `uv run sft @ /tmp/sftprobe/sft-probe.toml --dry-run` → exit 0,
  `SUCCESS Dry run complete.` One schema gotcha: top-level `seq_len` is not a valid
  SFT key (it lives under `[data]` / `[model]`); the CLI error is a helpful
  "did you mean --model.seq-len".

## Step 3 — vendored-verifiers skew: MILDER THAN FEARED, resolved per AC10

- prime-rl v0.7.0's submodule `deps/verifiers` is **verifiers 0.2.0**
  (hatch-vcs version from the tag); our env package pins `verifiers==0.2.1`.
- **Direction A (env pinned 0.2.1 inside the prime-rl venv):** installing
  `daydream_review_v1` editable swaps 0.2.0 → 0.2.1; the `rl` dry run still
  succeeds (exit 0). prime-rl at this tag tolerates 0.2.1.
- **Direction B (env suite run against vendored 0.2.0):** installed
  `deps/verifiers` editable (+ `daydream_review_v1 --no-deps` so the pin doesn't
  fight), then ran the env's full suite:
  `python -m pytest tests -q` → **172 passed, 1 skipped** (the skip is the
  docker-gated slow marker). **No import errors, no version errors, no API-skew
  failures.** (First `-x` attempt from the prime-rl cwd showed one failure in
  `test_mypy_strict_config.py` — that test reads a relative `pyproject.toml`, a
  CWD artifact, not a skew issue; from the env's own directory it passes.)
- **Decision for AC10:** the env's API surface used at these pins is compatible
  across 0.2.0/0.2.1 in both directions. Keep `verifiers==0.2.1` in the env's
  pyproject (root lockfile never sees it); inside a prime-rl workspace either let
  uv swap to 0.2.1 (validated) or install the env `--no-deps` on top of the
  vendored 0.2.0 (validated). No fork, no patch, no version relaxation needed.

## Verdict

No Key Decision invalidated: dry path works GPU-free at v0.7.0, dataset-SFT exists
as a separate `sft` entrypoint taking prompt/completion JSONL with LoRA rank up to
128 and `renderer.name = "default"`, and the verifiers skew resolves by pin
discipline alone. Proceed to Task 1.

## Issue-1055 Task 0 spike (2026-08-30, branch eb/daydream/issue-1055 @ HEAD)

- Baseline: `pytest tests/test_corpus_v2.py tests/test_training_adjudication_preview_harvest.py -x -q` → **48 passed**.
- Wiring import check: `run_build_corpus_v2`, `append_label_observation`, `build_export_entries` all import cleanly → `wiring OK`. (`_verify_snapshot_pinned` was in the spike-time import list but is gone at HEAD — the merged branch deleted the function.)
- Failure-mode capture: **no tests reference `_verify_snapshot_pinned`** — moot at HEAD since the function no longer exists; there is no existing digest-membership contract coverage to replace — Task 9 writes the two-bundle contract tests fresh.
- Consumer scan: stale at HEAD — `_verify_snapshot_pinned` was previously anchored at `daydream/training/corpus_v2/projector.py:374` (sole call site, inside `run_build_corpus_v2`), with a docstring mention in `adjudication/preview.py:8`, but the merged branch deleted the function and both anchors are gone. No consumers remain.

## Verdict (issue-1055 spike)

Baseline green, wiring intact. Spike passes; proceed to Task 1.
