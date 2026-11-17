# Handoff: issue #1898 — remove training stages, outcome model, reward calibration and capture-time reward

Prepared read-only against `origin/main` = `a35b484a97487248f4a120ffa8827f14f159d172`
(fetched 2026-11-17). This file lives only on branch `claude/adoring-curie-kxi5qr`;
read it with `git show origin/claude/adoring-curie-kxi5qr:handoff/issue-1898.md`.
It must NOT be part of the implementation PR: branch from `origin/main`, not from this branch.

Source: https://github.com/existential-birds/daydream/issues/1898 (no comments as of preparation).
Related: #94 (keeps harvest, adjudication, labeler signals, record evidence/identity,
exclusion and corpus projection, and reuses projection splits, checksums and as-of cutoff).

## User directives (override defaults)

- Seek **maximum removal** of code, tests, docs and fixtures.
- **Write no new tests, including regression tests.** Edit surviving tests only to drop
  references to removed code or fields. Deleting tests is expected.
- Run `$test-audit` (`.agents/skills/test-audit/SKILL.md`) on every test file changed.

## Decisions made during preparation

1. **Capture-time reward: drop it** (the issue's preferred option). No kept code reads it for a
   decision. All readers are in removed code (calibration, coordinator, lineage, reward_model,
   corpus.py) or only carry/validate it (dataset.py schema, projector pass-through,
   record-schema.json, trace-analysis skill's section listing). `rubric.derive_outcome_label` never
   imports reward. Harvest derives labels at `training/harvest.py:426-427`, before `score_trajectory` runs.
   Drop the whole run-record `scoring` section, not only the reward fields inside it.
   Its other contents duplicate data captured elsewhere or have no kept reader:
   - `verifier_verdicts` duplicates `verification`.
   - `format_valid` duplicates `collection_diagnostics`.
   - `length` is only the reward's length proxy.
   - `review_text` is a rendering of the findings.
2. **Schema versions:** follow the #1903 precedent (`a280b3f`, CHANGELOG `[Unreleased]` line ~47).
   A shape change bumps the version, earlier records are refused with the existing message, and
   nothing is converted. Bump `daydream.run.v3` → `daydream.run.v4` and
   `daydream.observation.v3` → `daydream.observation.v4`, because the harvest-annotation and
   run-label payloads change. Add no compatibility readers. `run.v3` is unreleased (0.36.0 shipped
   `run.v2`), so the disruption is limited to the current dev cycle.
3. **`training/corpus.py` is NOT only used by `reward_model.py`** (the issue is wrong here):
   `corpus_projection/projector.py:15` imports `_is_posterior_leak` and `_trajectory_set_hash`.
   Move them into `projector.py`. `_is_posterior_leak` has one caller (`projector.py:80`) and may be
   folded into `_refuse_posterior_evidence`. `_trajectory_set_hash` has one caller
   (`projector.py:377`) and can be inlined as a two-line sha256. Delete `_is_admitted_outcome_gold`
   and `_OUTCOME_GOLD_LABELS`, which have no kept user. Then delete `corpus.py`.
4. **`training/admission.py`**: after `stacks.py` goes, only `projector.py:14,219,221` uses its two
   reason codes. Make them module constants in `projector.py` and delete `admission.py`.
5. **Corpus projection output** (kept for #94): keep the splits, `SHA256SUMS`, `_SUCCESS`, lineage
   and as_of. Delete the whole `if scoring["status"] == "available":` branch (`projector.py:295-300`),
   which writes `reward_version`, `intrinsic_reward`, `verification` and `scoring`. Also delete the
   row fields `has_posterior`, `decisive_only` and `decisive_mix` (`projector.py:301-303`); only the
   removed outcome-gold admission read them. Remove all seven properties from
   `training/schema/record-schema.json:266-289`, including any `required` entries. Bumping the
   projected `schema_version` "3" is optional: only the removed `stacks.py` checked it.
6. **No `daydream train` tombstone.** `cli.py:124-130` rejects `bench`/`benchmark` explicitly. Adding
   `train` there would add code, so don't. A removed verb now falls through to review parsing.
7. `scripts/check-naming.sh` keeps its versioned-name patterns. They guard a naming decision
   (#1093), not removed code.

## Deletions (whole files)

Production:
- `daydream/training/{coordinator,stacks,lineage,gate,reward_model,calibration,corpus,admission,reward}.py`
- `daydream/dataset_scoring.py` (entirely reward input)
- `daydream/commands/{train,calibrate}.py`
- `docs/training-launch.md`, `docs/calibration.md`. Neither holds adjudication-rubric content #94 needs.
  Projection invariants already live in `docs/runbooks/record-dataset.md:41-51`, so nothing moves.

Tests (about 870 lines and 23 test functions):
- `tests/test_training_coordinator_projection.py`, `tests/test_training_coordinator.py` (not in the
  issue's list; it tests the `train` CLI), `tests/test_training_gate.py`
- `tests/test_training_reward_model.py`, `tests/test_training_lineage_guard.py`, `tests/test_stacks_load.py`
- `tests/test_stacks_gate.py`, `tests/test_corpus_admission_units.py`, `tests/test_calibration.py`
- `tests/test_calibration_cli_smoke.py`, `tests/test_cli_corpus_calibrate.py` (not in the issue's
  list), `tests/test_training_reward.py`
- the whole `tests/fixtures/training/` tree (21 files; it only holds `calibration/`, generated by its
  own `build_fixture.py`)

## Production edits

### Lane A: stages, calibration, CLI, docs and changelog

- **`daydream/cli.py`**
  - Line 12: drop `train` from the import.
  - Line ~28: drop the `"train"` handler entry. `KNOWN_VERBS` derives from this dict.
- **`daydream/commands/corpus.py`**
  - Line 13: drop `calibrate` from the import.
  - Lines ~161, ~166 and ~172: drop the `calibrate-reward` dispatch entry, its mention in the usage
    string and its help line.
  - Line ~136: change the observation literal to `daydream.observation.v4`.
- **`daydream/commands/common.py:~168`**: change the docstring "corpus/train subcommand parser" to
  "corpus subcommand parser". `_add_dry_run_argument` stays.
- **`daydream/json_utils.py:131-158`**: delete `atomic_write_pair`; its only caller was
  `calibration.py`. Keep `_publish_staged`, which `dataset.py:763` uses, and `_stage_bytes`, which
  tests monkeypatch.
- **`scripts/measure_finalization_memory.py:96`**: delete `import daydream.dataset_scoring`.
- **`scripts/ci_test_durations.json`**: delete the entries for node ids this PR deletes (`ci_shard.py`
  tolerates unknown ids).
- **`README.md:59-62`**
  - Delete "The training pipeline prepares reward, SFT and adapter-handoff artifacts; production
    training has not completed."
  - Delete `, [training](docs/training-launch.md)`.
  - Keep the dataset-workflow link and the `hub` extra sentence.
- **`CONTRIBUTING.md:~211`**: delete the "Corpus / training launches" bullet.
- **`AGENTS.md`**
  - Line ~221: in the "Capture, harvest, adjudication…" row, drop `, [calibration](docs/calibration.md)`.
  - Line ~222: delete the "Training launches" row.
  - Owner row ~113 (`dataset*.py, training/, archive/, eval/`): already training-neutral; reword only
    if something there is now stale.
- **`docs/runbooks/record-dataset.md:14`**: change "scoring annotations" to "outcome annotations".
- **`docs/observability.md:~71`**: change `daydream.run.v3` to `daydream.run.v4`.
- **`pyproject.toml:~106`**: the comment "Other fixtures/training helpers stay checked" is stale;
  reword it. No dependency, extra, mypy, vulture, coverage or package entry exists only for removed
  code: the `hub` extra serves `daydream/hub.py`, and the `training/schema/**` include serves the
  projector and exclusion.
- **`CHANGELOG.md` `[Unreleased]`**: add `### Removed` before `### Fixed`, matching 0.36.0's Changed →
  Removed → Fixed order. Use the fuller style of 0.36.0 line ~168: `- **Breaking (0.y.z):** …
  ([#1898](https://github.com/existential-birds/daydream/issues/1898))`, with sub-bullets that state:
  - `daydream train` and `daydream corpus calibrate-reward` are gone.
  - The training stages, outcome model and capture-time reward are deleted.
  - New records are `daydream.run.v4` with no `scoring` section, and `daydream.observation.v4` with no
    reward fields in harvest annotations.
  - Projections drop the reward, posterior and decisive fields.
  - Earlier stores and Hub destinations are refused and not converted; use a fresh store or
    destination.

  Leave historical CHANGELOG lines alone.

### Lane B: capture-time reward and record schema

- **`daydream/dataset.py`**
  - Delete `_BREAKDOWN`, the `"scoring"` entry of `_RUN_PAYLOADS`, the scoring validator block
    (~402-415) and the `without_review_context` import (line 26).
  - Bump the `run.v3` and `observation.v3` literals (lines 116, 134, 166-167, 190, 197, 207, 245 and 301).
  - In `_HARVEST_ANNOTATION`, drop `reward_version`, `reward_json`, `composite_reward`, `has_posterior`
    and `reviewer_logins`.
  - In the `run-label` payload, drop `outcome_prior` and `outcome_prior_n`, which have no producer.
    Drop `reviewer_logins` there too if a grep confirms nothing writes or reads it.
- **`daydream/dataset_capture.py`** (lines 22, 102, 153-159, 192, 209-231)
  - Delete `capture_scoring`, the scoring assembly and the `"scoring"` evidence.
  - Bump the literal at line 192.
  - Remove any review-text read or redaction helpers that only fed scoring.
- **`daydream/dataset_hub.py:32`**: change `_SCHEMAS` to v4.
- **`daydream/deep/render.py:84-89`**: delete `without_review_context`; it becomes dead.
- **`daydream/training/harvest.py`** (lines 37, 57, 62-66, 100-111, 339-458, 724-781 and 857)
  - Delete `_PRIOR_SUFFICIENCY_THRESHOLD`, the two reward Protocol methods, `reviewer_prior`,
    `read_scoring_inputs`, the prior acquisition, the scoring block, and the
    `reward_version`/`reward_json`/`composite_reward`/`has_posterior`/`reviewer_logins` fields in
    `AnnotationPayload`/`build_annotation`.
  - Update the docstrings and the v4 literal.
  - Keep `reviewer_logins_signal`: the rubric uses it to qualify replies (lines 390-402).
- **`daydream/training/harvest_types.py`** (lines 11, 122-137 and 161): drop `scoring_inputs`,
  `pooled_prior`, `prior_n`, and `reviewer_logins` if it becomes unused.
- **`daydream/training/adjudication/observations.py:71`**: change the literal to v4.
- **`daydream/training/corpus_projection/projector.py`**: apply decisions 3-5.
- **`daydream/training/corpus_projection/tiers.py:2`** and **`daydream/training/labeler_versions.py:1`**:
  remove reward wording from the docstrings.
- **`daydream/training/schema/record-schema.json:266-289`**: apply decision 5.
- **`.agents/skills/daydream-trace-analysis/`**: change v3 to v4 in `references/format.md:6`,
  `scripts/trace.py:4,94`, `scripts/cohort.py:95-96`, `scripts/fixture_evidence.py:65` and the
  `scripts/test_*.py` literals. Drop `scoring` from `cohort.py:17` `SECTIONS` and `format.md:17,28-29`.
  The gate runs these tests via `make test-scripts`.
- Vulture runs at confidence 80 and will not flag dead functions, so grep by hand after edits. Check
  every symbol the deleted code imported, plus `labeler_versions` constants used only by reward.

## Test edits (Lane C, edits only: no new tests)

- **`tests/test_atomic_write_migration.py`**
  - Drop the imports at lines 18 and 21.
  - Drop the calibration half of `test_projection_and_calibration_failures_leave_no_stray_temp`
    (33-47), keep the projection half and rename the test.
  - Fix the docstring at lines 1-2.
- **`tests/test_corpus_projection.py`**
  - Drop the `load_dataset_v2` import (line 18).
  - Drop the `reward` parameter and argument (lines 35, 43-44, 51-53).
  - Drop lines 149 and 166.
  - Delete `test_v2_loader_rejects_incomplete_wrong_version_and_malformed_outputs` (202-219).
  - Drop the reward keys and `breakdown`, plus `has_posterior`/`reviewer_logins`, at 243-248 and 263.
  - Drop "task_reward" from the test name at line 82.
  - Drop assertions on the removed projection fields.
- **`tests/test_training_harvest.py`**
  - Drop the imports at lines 16 and 28, the scoring override at 63-65, the `has_posterior` assertion
    at 257 and the `reward_json`/`posterior_cost` assertion at 515.
  - Delete `pure_annotation` and `test_record_reviewer_prior_…` (336-410), then the imports that
    become unused.
  - Update the docstring at line 1 and the version literal at 392.
- **`tests/test_dataset_store.py`**
  - Remove the scoring runs and ids in `test_current_records_read_all_history_without_creating_snapshot`
    (41-68).
  - Remove the scoring row at 108-109.
  - Delete `test_scoring_inputs_without_review_evidence_earn_no_credit` (328-330).
  - Remove the imports that become unused (`replace`, `insert_review_context`, `capture_scoring`,
    `assemble_scoring_inputs`, `ScoringInputs`).
- **`tests/harness/record_projection.py:9,11,38,50`**: drop the scoring construction.
- **`tests/test_archive_data_capture.py:217`**: drop the `persisted_breakdown` assertion.
- **`tests/test_archive.py:223`**: reword the "reward" docstring, which pairs with the comment at
  `daydream/archive/manifest.py:350` (that one is about evaluation metrics; leave it unless stale).
- **Version literals → v4**: `tests/harness/dataset.py:9,19`, `tests/test_dataset_hub.py:51-52,204,238`,
  `tests/test_dataset_capture.py:128` and `tests/test_corpus_projection.py:254`. Re-grep
  `run\.v3|observation\.v3` across the repo.
- **Final grep must be empty**, excluding CHANGELOG history and `scripts/check-naming.sh`:
  `git grep -nE "training\.(coordinator|stacks|lineage|gate|reward_model|corpus|calibration|admission|reward)\b|dataset_scoring|ScoringInputs|score_trajectory|REWARD_VERSION|DEFAULT_WEIGHTS|FP_PENALTY_MAP|reward_policy|reward_json|composite_reward|persisted_breakdown|intrinsic_reward|calibrate-reward|calibrate_reward|load_dataset_v2|load_v2_projection|training-launch\.md|calibration\.md|run\.v3|observation\.v3"`

## Lanes and ownership

- **Lane A:** `daydream/cli.py`, `daydream/commands/*`, `daydream/json_utils.py`, the stage and
  calibration module deletions, the docs, README, AGENTS, CONTRIBUTING, CHANGELOG, `pyproject.toml`
  and `scripts/`.
- **Lane B:** `daydream/dataset*.py`, `daydream/deep/render.py`, `daydream/training/reward.py`,
  `harvest*.py`, `labeler_versions.py`, `adjudication/observations.py`, `corpus.py`, `admission.py`,
  `corpus_projection/*`, `schema/record-schema.json` and the trace-analysis skill.
- **Lane C:** everything under `tests/`.
- **Shared contract** (fixed above; no lane changes it): the v4 names, the removed field list and the
  projector-local helpers.
- The lead integrates the lanes and runs the full gate.

## Validation

- **Focused checks:**
  - `uv run pytest tests/test_corpus_projection.py tests/test_training_harvest.py tests/test_dataset_store.py tests/test_dataset_capture.py tests/test_dataset_hub.py tests/test_atomic_write_migration.py tests/test_archive_data_capture.py tests/test_cli_adjudicate.py tests/test_cli_corpus_namespace.py`
  - `make lint typecheck deadcode test-scripts check-naming`
- **Acceptance smoke:** `uv run daydream corpus --help` lists `dataset,harvest,build,label,adjudicate`
  and no `calibrate-reward`. `uv run daydream corpus calibrate-reward` exits 2 with usage.
- **Full gate:** `make check`, which the pre-push hook also runs; install hooks with `make hooks`.
  Budget: under 2 minutes on a typical container.
- **Coverage:** `fail_under = 90` (measured 90.59% earlier). Never lower it, and do not add tests.
  If removal drops it below 90, first remove more dead or uncovered code from the touched area. If it
  still fails, report a blocker; don't write tests.
- **Simplification sweep:** required because the PR touches more than 5 files. Run
  `.agents/skills/daydream-deadcode-sweep/SKILL.md` in PR mode before final validation, and record it
  in the Test Plan.

## Publication and review

- Make signed Conventional Commits, e.g.
  `refactor!: remove training stages, reward calibration and capture-time reward (#1898)`.
  Stage paths explicitly and never bypass hooks.
- Start the PR from `.github/PULL_REQUEST_TEMPLATE.md` using the `pr` skill. Use `Closes #1898`, state
  SemVer as breaking (0.y.z) and keep every checklist item.
- After opening the PR, run the Daydream review:
  `daydream . --backend pi --comment --non-interactive --capture-data`. Add `--base <sha>` only if the
  default base differs from the verified PR base.
- Then run `fetch-pr-feedback --include-author --include-self`, then `receive-feedback`, then
  `respond-pr-feedback`. Loop per `.agents/skills/prepare-implement/SKILL.md` "PR review loop".

## Risks and notes

- `--capture-data` writes v4 records to `~/.daydream/dataset`. If that store already holds v3 records,
  reads refuse them by design. Use a fresh store (`dataset_store_path`) and don't convert. Report it as
  environmental if it blocks the review.
- Issue #94's body links `docs/calibration.md`, which this PR deletes. Mention that in the PR so the
  owner can update #94; don't edit #94.
- Leave the run-record `verification` section in place. It is verifier evidence, not reward.
- Out of scope: renaming `daydream/training/`, and the record-pipeline changes #94 needs.
