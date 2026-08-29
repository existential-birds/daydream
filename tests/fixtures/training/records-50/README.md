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
  fields (`comment_id`, `text`, `label`) with the current reply-classifier
  policy version (`labeler_policy_version`) so the gold-admission guard admits
  them (M23), and a frozen `diff` body for the Stage-2 RFT replay identity
  (`id`/`base_sha`/`head_sha`/`diff`).
- `manifest.json` — fixture metadata (counts, ratio, harness config).

## Label mix

The fixture mirrors the archive's accepted/rejected ratio at roughly
**70% accepted / 30% rejected** (35 accepted / 15 rejected). Accepted rows
are distinct grounded, actionable findings naming a concrete defect and its
consequence; rejected rows are distinct hedged, non-actionable chatter. No two
rows share a literal template, so the Stage-0 outcome model must separate the
classes on distributional word/char-trigram signal (mechanism-and-consequence
vocabulary versus hedge vocabulary) rather than a memorized phrase — which
is what the CI stage-0 gate measures.

## Constraints honored

- **C5/C8**: every `repo_slug` (`acme/tooling-*`) is synthetic and appears on
  neither the exclusion nor the copyleft list, so the fail-closed loaders in
  `daydream.training.stacks` admit the fixture.
- **M24**: the harness config in `manifest.json` pins `backend` to `pi`.
- **GPU-free**: the coordinator's `dry_run=True` path over this fixture never
  imports pynvml or initializes CUDA.
