# Reward calibration (`daydream corpus calibrate-reward`)

Issue #999 ships one documented, GPU-free command that validates a pinned
calibration bundle (wire format below), computes deterministic
reward-calibration statistics with confidence intervals, and emits a
byte-reproducible, versioned calibration artifact. The command **never
mutates any reward default** — every candidate weight and threshold comes
from the flags you pass.

## Scope boundary

The calibration tool is analysis-only. It is the answer to the calibration
question raised in issue #114: it tells you what candidate weights and
thresholds *would* score on a pinned corpus, so a human can decide before
choosing values for the reward model. It does not choose values for you, does
not write to `daydream/training/` reward defaults, and does not touch the
live reward used by the harvest pipeline.

## Input bundle (wire format)

`--corpus-dir` must hold exactly three files:

- `corpus.jsonl` — one JSON object per line, each with `schema_version: "2"`,
  `record_id`, `session_id`, `repo_slug`, `reward_version`, and a `lineage`
  object carrying `split`, `as_of`, `valid_at`, `license_decision`, and the
  label version stamps (`labeler_policy_version`, `reply_classifier_version`,
  `rubric_schema_version`). `record_id` must be unique across lines, and each
  record's stored `lineage.split` must match the split deterministically
  re-derived from the bundle salt and split rates.
- `lineage.json` — a single object with `schema_version: "corpus-v2"`, `salt`,
  `holdout_rate`, `val_rate`, `as_of`, `valid_at`.
- `SHA256SUMS` — lines of `<sha256>  <name>` covering the bundle files.

`--gold-labels` and `--breakdowns` must be JSON objects keyed by the same
`record_id` values.

This wire format is **not** what `daydream corpus build` emits: build output
holds `schema_version: "1"` records without `record_id` or a per-record
`lineage`, writes a lineage manifest with `trajectory_set_hash` /
`labeler_version` / `reward_version` / `as_of` / `created_at` (no salt or
split rates), and produces no `SHA256SUMS`. Pointing `calibrate-reward`
directly at a build output directory therefore fails the first gate by
design; the in-repo reference producer of the calibration wire format is
`tests/fixtures/training/calibration/build_fixture.py`.

## Invocation

The exact command, run against the synthetic test fixture as a dry-run-style
example (the fixture lives at `tests/fixtures/training/calibration/`):

```bash
daydream corpus calibrate-reward \
  --corpus-dir tests/fixtures/training/calibration/corpus \
  --gold-labels tests/fixtures/training/calibration/gold.json \
  --breakdowns tests/fixtures/training/calibration/breakdowns.json \
  --stage0-scores tests/fixtures/training/calibration/stage0-scores-aligned.json \
  --model-digest sha256:calibration-stage0-model-1 \
  --out /tmp/calibration-out \
  --run-id cal-example-1 \
  --seed 7 \
  --candidate w_fp=0.1,0.3,0.5
```

Required flags: `--corpus-dir`, `--gold-labels`, `--breakdowns`, `--out`,
`--run-id`, `--seed`, and at least one repeatable `--candidate AXIS=V1,V2,...`.

Optional flags:

| Flag | Meaning |
|------|---------|
| `--stage0-scores PATH` | Stage-0 score JSON keyed by `record_id` (`{"score": float, "model_digest": str}`) |
| `--model-digest DIGEST` | Digest of the Stage-0 model; required with `--stage0-scores` |
| `--grid-points N` | Grid resolution per candidate axis (default 9) |
| `--bootstrap-resamples N` | Bootstrap resample count for AUC confidence intervals (default 1000) |

## Determinism

Given the same inputs and the same `--seed`, the emitted artifact is
byte-reproducible. The artifact records the SHA256 of every input file, the
corpus lineage (`schema_version: corpus-v2`, `content_digests`, `as_of`,
`valid_at`, `salt`, split rates), and the resolved version stamps. The
artifact schema version is `calibration-artifact-v1`.

## Validation is fail-closed

The command fails with a non-zero exit and a named error for: missing or
digest-mismatched bundle files, unrecognized or absent version stamps,
records whose stored split does not match the split re-derived from
`lineage.json` via `daydream.training.calibration.assign_split`,
gold/breakdown records that do not join cleanly on `record_id`, Stage-0
scores whose `model_digest` does not match `--model-digest` (and a missing
`--model-digest` whenever `--stage0-scores` is given), and a re-run that
would collide with an existing `--run-id` in the output directory. C5-excluded
repos are rejected as a defense-in-depth check on legacy corpora. All
statistics are computed with the Python standard library only.