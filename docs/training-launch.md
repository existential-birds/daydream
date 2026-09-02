# Training launch: corpora, model, hardware, wall time, costs

This is the launch record for the four-stage training pipeline (reward gate →
dataset SFT → deterministic RFT → online GRPO). Every number below comes from
an artifact produced by a validation run (the stage manifest of the 50-record
fixture run) or from a measured environment — nothing is a placeholder. Where
a value is a plan rather than a measurement, the source is named as such.

## Corpus

The validated run trains on the committed 50-record corpus fixture at
`tests/fixtures/training/records-50/records.jsonl` (35 accepted / 15 rejected,
0.7 accepted ratio, `pi` backend), produced by the Task-12 fixture harness. Its
identity digests, as recorded in the stage manifest of the validation run:

| Field | Value |
|---|---|
| `run_identity.corpus_digest` | `80cfdda8293d5854216ed1845c6228e6d8ada013152a54d924309813704854ce` |
| `run_identity.split_digest` | `fe0a7a9559493b0cb4ef3795e5a66b4fcf8eeb718e12b44c790b4420a28a5bde` |
| `run_identity.reward_version` | `2026.05.28-2` |

Corpus-side loading goes through `daydream.training.stacks.load_dataset`,
which fail-closes on the C5 exclusion list and C8 copyleft opt-in before any
record is returned.

The planned real-archive runs use the same loader over `run_build_corpus`
exports of the private PR archive (`--include-all-labels`, both label classes
exported). The committed fixture is a CI-scale stand-in — not byte-identical to
an export: it carries the Stage-0 gold outcome fields
(`comment_id`/`text`/`label`) plus the M16 lineage field set, and the
coordinator normalizes both the fixture shape and the exporter's
(`session_id`/`review_output`/`outcome_label`) shape before Stage 0. Stage-2's
RFT task identity (`base_sha`/`head_sha`/`diff`) is rebuilt from the frozen
record; since the schema-v1 export carries no raw `diff` body, the coordinator
materializes it from the record's `fix_diff_ref` pointer to the archived
`diff.patch` (the reviewed-INPUT diff), so the same exporter shape is runnable
through Stage 2 — no bespoke `diff` column is required.

### Splits

Stage 0 freezes the split before training (M16): 40 train / 10 held-out rows
(`held_out_fraction` 0.2, seed 0), frozen to
`labels.jsonl.gate-split.json` with digest
`fe0a7a9559493b0cb4ef3795e5a66b4fcf8eeb718e12b44c790b4420a28a5bde` —
identical to `run_identity.split_digest`, which is how resume validation
(`validate_resume`) detects a stale or drifted split (AC4).

## Corpus v2: real-corpus training from a frozen projection

For real-corpus training, the input is a frozen corpus-v2 projection directory
produced by the projector, not the v1 JSONL export. The real-corpus command
sequence is:

```bash
daydream corpus build-v2 --bundle-root <dir> --annotation-bundle-root <dir> --license-policy <file> --out <proj_dir>
```

```bash
daydream train --corpus-v2 <proj_dir> --out <out_dir> --dry-run
```

(Drop `--dry-run` for the real run. `--corpus-v2` is mutually exclusive with
`--corpus` on the train parser; the v1 `--corpus` path below remains valid and
unchanged.)

The projection directory is the immutable input contract for the run:

- **`_SUCCESS`** — the completeness marker written last by the projector; a
  directory without it is refused before any record is read.
- **`lineage.json`** — pins the split salt, holdout/validation rates, and
  provenance digests the loader re-checks.
- **Split digests** — per-split JSONL sha256s plus a deterministic
  directory-level digest over the sorted `(relpath, sha256(file_bytes))`
  pairs; the directory digest replaces the v1 single-file corpus digest in
  `run_identity.corpus_digest`.
- **`base_sha` / `head_sha`** — the per-record task-identity git SHAs, used by
  Stage-2 RFT to rebuild replay tasks; full-SHA values are validated before
  any task rebuild.
- **C5/C8 re-application** — the v2 loader re-applies the C5 exclusion list
  and the C8 copyleft opt-in gate fail-closed on every load; the projector's
  decision is never trusted on its own.
- **Fail-closed drift** — the split recorded on each record's `lineage.split`
  is recomputed from the record id under the lineage's pinned salt/rates, and
  any disagreement refuses the entire load with the offending record id
  named. Stage 0 consumes the projector's frozen split as-is (it is never
  re-frozen), so drift is a hard stop, never a silent re-split.

Because the directory-level digest is a pure function of the directory bytes,
the same projection always yields the same run identity — a re-run over a
modified directory aborts at the resume guard instead of training on drifted
data.

## Legacy traces

Legacy rows — runs admitted under the reply-count / merge-presence gold policy
before the reply-classifier policy version existed — are tagged explicitly at
load time: `stacks.load_dataset` sets `legacy_policy=True` on every record
whose `labeler_policy_version` is absent or null (M23). The tag is metadata,
never a drop; the loader's only refusals are the C5/C8 fail-closed gates.
Current-policy SFT prefers native-profile traces: selection filters on
`legacy_policy=False` first and falls back to legacy rows only when the
native-profile pool is empty (no accepted rows at all). No skill-era contract
appears in the corpus or this pipeline.

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
loop on one GPU and makes per-finding economics favorable, while preserving the
grounding-weighted reward signal (`w_grounding` 0.4, `w_fp` 0.3,
`w_correctness` 0.6 in `run_identity.reward_weights`). Larger bases would
multiply GPU-hours for gains on axes the rubric down-weights; smaller bases
measurably lose thread-level localization on the held-out split.

## Hardware

The offline stages (Stage-0 gate, all dry-path validation, CI) ran on the
development VM: AMD EPYC 9554P 64-core, 7 GiB RAM, **no GPU** — the dry path
imports no pynvml and never initializes CUDA (asserted by
`tests/training/test_coordinator_fixture_ci.py::test_ci_dry_path_has_no_gpu_imports`).

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

## Cost accounting

Per-run costs are recorded by `daydream.training.costs.record_stage_costs` and
aggregated by `summarize_costs` into two metrics:

- **`usd_per_review`** — total recorded USD divided by the number of reviews.
- **`usd_per_finding_that_mattered`** — total USD divided by findings that
  survived to an accepted/contested label (the denominator is findings a
  maintainer actually engaged with, per the module contract).

Measured: the 50-record fixture validation run recorded **$0.00 total LLM
spend** (dry path, no paid backend calls), so both metrics were reported as
zero-spend rather than estimated. The real-archive GPU runs will pin these
numbers in their stage manifests; this section is updated with those measured
values at launch.

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
