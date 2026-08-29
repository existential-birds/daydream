# 50-record training fixture

A committed 50-record corpus-format JSONL fixture (`records.jsonl`) for the
training coordinator's GPU-free CI dry path (`tests/training/test_coordinator_fixture_ci.py`,
`.github/workflows/training-dry.yml`). Modeled on the corpus record shape
produced by `daydream.training.corpus.run_build_corpus` and the fixture layout
of `rl/daydream_review_v1/tests/fixtures/corpus-mini/`.

## Contents

- `records.jsonl` — 50 records, each carrying the full M16 lineage field set
  (`session_id`, `evidence_tier`, `base_sha`, `head_sha`, `diff_identity`,
  `daydream_version`, `profile_digest`, `detected_stack`, `label_source`,
  `label_version`, `reward_version`, `split`) plus the Stage-0 gold outcome
  fields (`comment_id`, `text`, `label`).
- `manifest.json` — fixture metadata (counts, ratio, harness config).

## Label mix

The fixture mirrors the archive's accepted/rejected ratio at roughly
**70% accepted / 30% rejected** (35 accepted / 15 rejected). Accepted rows
carry grounded, actionable review text; rejected rows carry noise chatter, so
the Stage-0 outcome model can separate the classes and the gate passes.

## Constraints honored

- **C5/C8**: every `repo_slug` (`acme/tooling-*`) is synthetic and appears on
  neither the exclusion nor the copyleft list, so the fail-closed loaders in
  `daydream.training.stacks` admit the fixture.
- **M24**: the harness config in `manifest.json` pins `backend` to `pi`.
- **GPU-free**: the coordinator's `dry_run=True` path over this fixture never
  imports pynvml or initializes CUDA.
