# Training launch: corpora, model, hardware, wall time, costs

This is the launch record for the four-stage training pipeline (reward gate →
dataset SFT → deterministic RFT → online GRPO). Every number below comes from
an artifact produced by a fixture-scale validation run (the stage manifest of
the 50-record projection-fixture run) or from a measured environment — nothing
is a placeholder. No production training run has completed; where a value is a
plan rather than a measurement, the source is named as such.

## Corpus

The training pipeline's input is a frozen-corpus projection directory produced
by `daydream corpus build` (a deterministic per-finding projection over
immutable run records plus their eligible observation history). The validated run trained on
the committed 50-record projection fixture produced by
`tests/fixtures/training/build_projection_50.py` (both gold classes present on
the frozen holdout side, silver `process-trace` and `task-only` records
included). Its identity digests, as recorded in the stage manifest of that
fixture-scale validation run:

| Field | Value |
|---|---|
| `run_identity.corpus_digest` | `f8f254b73fee23de638619b33d84467a8912bd7b9d1f597ef8986d1625e20132` |
| `run_identity.split_digest` | `903de487711c6be182d582cee898570c5635c75e2c6c3dbcdf622af1236b33ee` |
| `run_identity.reward_version` | `2026.09.04-1` |

That version describes the historical fixture run, not the current reducer.
The current intrinsic reward version is `2026.10.01-1`: verifier correctness
is the mean of `consistent = 1.0`, `uncertain = 0.5`, and `contradicts = 0.0`.
The composite is `round(clip(correctness - 0.2 * length_penalty, 0, 1), 4)`,
where `length_penalty = clip((length - 2000) / 8000, 0, 1)`.
Invalid format floors it at `0.0`; missing verdicts make it uncomputable
(`None`), and missing length contributes no penalty. Grounding is no longer
a reward axis. Posterior cost remains separate; `w_fp = 0.3` does not enter
the intrinsic composite. Historical observations retain their original version.
Current RFT threshold axes are `composite`, `correctness_per_finding` (mean),
and `length_penalty`, all interpreted as minimums.

Corpus-side loading goes through `daydream.training.stacks.load_v2_projection`,
which validates repository identity and enforces the C5 benchmark exclusion list before any
record is returned, and re-verifies the projection's `_SUCCESS` marker,
split-digest lineage, and split drift on every load.

The planned real-corpus runs use the same loader over a frozen projection built
from a frozen private record snapshot. Current projections retain explicit or
claim-derived stack identity and native profile/configuration provenance from
those records. The committed fixture is a
CI-scale stand-in for that projection. Each projected record carries the full
RFT task identity (`base_sha`/`head_sha`/`diff`) — the projector embeds the raw
diff body on every record from the run record's immutable `original_task` section, so
Stage 2 replays tasks from the frozen record itself with no archive
materialization.

### Splits

Stage 0 freezes the split before training: splits are assigned deterministically
at projection time under the lineage's pinned salt and holdout/validation rates
(the fixture uses `holdout` 0.2 / `validation` 0.2, salt
`issue-1081-fixture-salt`), written to the per-split JSONL manifests and pinned
in `lineage.json` with digests. The loader recomputes each record's split from
its record id and refuses any drift, so the Stage-0 gate consumes the frozen
split as-is — it is never re-frozen, and resume validation
(`validate_resume`) detects a stale or drifted split.

## Frozen-corpus projection: real-corpus training

For real-corpus training, the input is a frozen-corpus projection directory
produced by the projector. The real-corpus command sequence is:

```bash
daydream corpus build --store RECORD_STORE --snapshot-id SNAPSHOT_ID --out PROJECTION_DIR/corpus.jsonl
```

```bash
daydream train --projection PROJECTION_DIR --out OUT_DIR --dry-run
```

(Drop `--dry-run` for the real run. `--projection` is the pipeline's only
training input: the legacy `--corpus` path is gone.)

Dataset builders own permission and licensing decisions. Use the captured repository,
base/head commit, and diff provenance when deciding whether to include or share records.
Daydream does not fetch licenses or authorize training use.

The projection directory is the immutable input contract for the run:

- **`_SUCCESS`** — the completeness marker written last by the projector; a
  directory without it is refused before any record is read.
- **`lineage.json`** — pins the split salt, holdout/validation rates, and
  provenance digests the loader re-checks.
- **Split digests** — per-split JSONL sha256s plus a deterministic
  directory-level digest over the sorted `(relpath, sha256(file_bytes))`
  pairs; the directory digest is the frozen `run_identity.corpus_digest`.
- **`base_sha` / `head_sha`** — the per-record task-identity git SHAs, used by
  Stage-2 RFT to rebuild replay tasks; full-SHA values are validated before
  any task rebuild.
- **C5 benchmark isolation** — the projection loader canonicalizes repository identity
  and re-applies the benchmark exclusion list on every load.
- **Fail-closed drift** — the split recorded on each record's `lineage.split`
  is recomputed from the record id under the lineage's pinned salt/rates, and
  any disagreement refuses the entire load with the offending record id
  named. Stage 0 consumes the projector's frozen split as-is (it is never
  re-frozen), so drift is a hard stop, never a silent re-split.

Because the directory-level digest is a pure function of the directory bytes,
the same projection always yields the same run identity — a re-run over a
modified directory aborts at the resume guard instead of training on drifted
data.

## Model

| Field | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen3-8B` | `PipelineConfig` default, recorded in the stage manifest's `run_identity.base_model` |
| Fine-tune | bf16 LoRA, rank 64 | `run_identity.lora_rank` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` | `run_identity.lora_targets` |
| Optimizer / LR | adamw, 1e-5 | `run_identity.optimizer`, `.learning_rate` |
| Max sequence length | 32768 | `run_identity.max_seq_len` |
| Renderer | `default` (never stock `qwen3`) | `run_identity.tokenizer_renderer` |

C1 sizing rationale (localization-first): review quality in this corpus is
dominated by localization — grounded, diff-anchored findings — not by long-form
generation. A rank-64 bf16 LoRA over Qwen3-8B with 32768-token sequences fits
comfortably on a single 80 GB accelerator (matching the shipped `sft.toml` /
`rl.toml` recipes at `seq_len = 32768`), which keeps the whole SFT→RFT→GRPO
loop on one GPU and makes per-finding economics favorable. Current reward
credit comes from verifier correctness, reduced by the length penalty described
above; the earlier fixture's grounding weights are historical provenance.
The model and hardware measurements here do not establish comparative quality
under the current reward formula.

## Hardware

The offline stages (Stage-0 gate, all dry-path validation, CI) ran on the
development VM: AMD EPYC 9554P 64-core, 7 GiB RAM, **no GPU**. The committed
projection dry run is covered by `tests/test_training_dry_fixture.py`. The
separate `tests/training/test_stage1_sft_config.py` checks the SFT recipe's
dry run when a prime-rl workspace is available.

GPU stages (Stage-1 dataset SFT, Stage-2 deterministic RFT replay, Stage-3
online GRPO) are planned for a single-GPU 80 GB node (H100 or A100 80 GB);
rank-64 bf16 LoRA on Qwen3-8B at 32768 tokens fits that budget with optimizer
states offloaded (the shipped recipes train at `seq_len = 32768`). This is the documented plan for the GPU run, not a
measurement from this machine.

## Wall time

Measured: the full coordinator run over the 50-record fixture — Stage-0 gate
train + evaluate + split freeze, stages 1–3 dry — completes in about
0.5 s end-to-end on the CPU-only VM above (Stopwatch over `run_pipeline(dry_run=True)`).

Expected GPU wall time (plan, not measurement): Stage-1 SFT over the real
corpus at rank 64 is expected in the low tens of minutes per epoch on a single
80 GB accelerator; Stage-2 deterministic replay is GPU-free offline replay; the
Stage-3 GRPO run is bounded by prime-rl's own schedule in `rl/train/rl.toml`.
These numbers are pinned in the run manifest when the GPU run happens.

## Stage-0 gate result (validation run)

From the stage manifest's `stages.stage0.gate`:

| Field | Value |
|---|---|
| Separation | 0.7499203696024055 (threshold `min_separation` 0.1) |
| Calibration | 0.928531093214983 (threshold `min_calibration` 0.5) |
| Held-out rows | 10 |
| Label ratio (reported / actual) | 0.7 / 0.9 |
| Model fingerprint | `38e7a8cd` |
| Evidence digest | `33ff119b16a98b3acd2be5b5c49e7b83d5e231138ab410874dd16b60d4e32ed4` |
| Verdict | **passed** |

The 0.1 / 0.5 thresholds are documented config values (`GateConfig`); their
final numeric pinning is the calibration run's result, not this document's.
All numbers in this document come from fixture-scale validation runs; no
production training run has completed.
