# Plan 001 implementation report

Baseline: `dad490231cff36241836377057237820f83ce6e0`. Existing uncommitted retirement annotations were preserved as a starting snapshot before implementation. No local or remote user datasets were removed or converted.

## Result

The production evidence path is now capture → `LocalRecordStore` → immutable JSONL shards and one HF manifest → exact-commit download → record harvest/adjudication → offline frozen corpus projection. `dataset_hub.py` remains the publication owner; `hub.py` is the sole lazy HF SDK boundary.

Annotation checkpoint publication/restoration, its pointers and merge machinery, final-bundle transport, hydration, folder uploading, and authoritative SQLite annotation history are deleted. No compatibility modules, command aliases, archive conversion, or replacement checkpoint workflow remain. Local annotation queues and exports are disposable views of canonical records.

Schema changes are explicit: `daydream.observation.v2` retains the complete harvest annotation; `daydream.hub.v2` declares permitted record schemas; `daydream.snapshot.v2` pins download provenance alongside membership and temporal cutoffs. Existing canonical JSONL versions remain readable through the same workflow. This does not retain the retired archive/bundle pipeline.

Training source identity is `record-snapshot-v1`, hashing run, trajectory, segment, and host item UID. Fingerprints remain external comment joins. Distinct host findings with the same fingerprint remain distinct. The identity change and resulting split membership are versioned; normalized pre-cutover examples retain task, finding, license, provenance, evidence, and reward semantics.

## Deletion and retention map

| Removed or replaced owner | Production consumer | Replacement and preserved contract | Owning proof |
| --- | --- | --- | --- |
| `dataset_hub_client.py`, `archive/hub.py` SDK/resolver | Canonical publisher/download, runner | `hub.py`: explicit operator destination, confirmed private metadata, exact lowercase commits, authenticated downloads, normal blob symlinks, confirmed absence, value-free failures, guarded atomic commit, commit-only 412 conflicts | `test_hub.py`; `test_dataset_hub_run.py` destination precedence and hostile checkout config |
| Unused `upload_run_bundle` and its backoff/globals/fake hooks | None | Deleted; capture/publication still retains review outputs and retryable failures | `test_dataset_hub_run.py`; archive lifecycle/dump tests |
| `archive/hydrate.py`, `commands/hydrate.py`, production `FakeHub`, curation schema/builders and remote resume/success ledgers | Former hydration and annotation transport | Canonical `DatasetUploader`/`download_snapshot`; immutable shards, manifest union, lost-response retry, exact download integrity | `test_dataset_hub.py`; `test_dataset_commands.py::test_cli_rival_manifest_commit_preserves_both_record_sets` now includes judgments from both writers |
| SQLite acquisition, label history, reviewer prior and PR/base backfill | Harvest, adjudication, corpus | Frozen record acquisition and append-only typed enrichment/annotations/judgments; human precedence, explicit unknown, intrinsic/posterior separation, strict temporal priors | `test_training_harvest.py`; `test_cli_label.py`; `test_harvest_record_ownership.py` |
| Checkpoint `publish-state`/`resume-state`, `publish-final`/`download-final`, state merging/restoration and final-bundle modules | Retired adjudication CLI | Deleted; ordinary dataset publication persists durable judgments | `test_cli_corpus_namespace.py` rejects retired verbs with unchanged filesystem; `test_record_dataset_journey.py` |
| Sessions/index/bundle readers and materialization authority | Adjudication queue, preview, report, export | Record views preserve complete finding population, human/model roles, evidence schemes, changed-evidence requeue, reports, and disposable exports | `test_cli_adjudicate.py`; `test_training_adjudication_preview_harvest.py` immutable snapshot drift and old-ledger refusal |
| Corpus `bundle.py`, batch artifacts and annotation bundle linkage | Corpus build and training coordinator | Offline snapshot projection; task/diff/revisions, native profile/stack, captured scoring, license admission, gold/silver/task/process eligibility, deterministic caps/splits and lineage | `test_corpus_projection.py`; process, reproducibility and cap siblings; normalized pre-cutover fixture comparison |
| Staging license enrichment and sanitizer derivatives | Former hydration | Neutral commit-pinned license acquisition plus shared admission rules; canonical publication secret scan; diagnostic dumps retain assembled raw bytes | `test_license_evidence.py`; harvest license failure tests; archive dump and dataset refusal owners |
| Calibration curation sidecars | Calibration readers/fixtures | Derived canonical corpus output with checksums and lineage repository/license decisions; reward policy unchanged | `test_calibration.py`; `test_calibration_consumes_canonical_record_projection_and_preserves_reward_policy` |
| Legacy folder/bundle/checkpoint fixtures and feature-only tests | Tests only | Deleted after consumer cutover; canonical fake remains in `tests/harness/dataset_hub.py` | Dataset, adjudication, corpus and complete journey keepers above |

The local diagnostic archive, evaluation/dump lifecycle, shared redaction/scanning, safe Git boundary, and runtime artifact ownership/freeze/publication remain live. Diagnostic manifest version is 2.0 and its runs-only index is version 9. Unsupported existing indexes remain byte-preserved and are refused; use a fresh archive directory. No historical archive migration was added.

Capture now retains branch/base/source context, native profile identity, and the existing raw finding body representation. Derived corpus/annotation outputs and acquisition caches reject overlap with record-store namespaces, including symlink aliases. A harvest provider must match the requested store, snapshot, and dry-run policy before acquisition.

## Validation

- `uv lock --check`: passed before dependency operations.
- Initial preservation baseline: 258 tests passed.
- First migration run: 675 passed, 19 failed; source defects and test setup failures were repaired.
- Broad repaired run: 717 passed, three restored-test expectations failed; canonical URL normalization and absent posterior fields were corrected without changing reward policy.
- Final focused consumer/transport/schema tests: 171 passed.
- Final harvest ownership and public CLI journey selection: 43 passed.
- Isolated same-harness controls: both reviewed source regressions failed before their fixes (digest scheme rejection and historical automatic conflict); the prior timestamp regression also failed with the original comparator. Direct import control reproduced the cycle before neutral identity relocation.
- Independent read-only preservation review: passed with no unresolved confirmed code findings. It covered transport, store/schema/capture, harvest/adjudication, corpus/calibration, deleted coverage and runtime/archive safeguards.
- `make lint`, `make typecheck`, `make deadcode`, `git diff --check`: passed.
- Full `make check`: passed — 8,980 tests passed, 14 skipped; branch coverage 90.47% (required 86%). Docker actionlint and the unchanged naming gate passed. Two remaining CLI parser fixtures were migrated before this successful rerun.

HF/GitHub boundaries are simulated in behavioral tests; no live HF dataset publication or deletion was performed. RL recipes and code were not changed. The initial harvest test setup used the default cache; all retained harvest CLI keepers now explicitly isolate their caches. Existing cache contents were not deleted.

Production changes: 1,944 lines added and 8,855 deleted (net reduction 6,911). Tests/support: 4,150 added and 11,836 deleted (net reduction 7,686). Counts include new files and are separate from documentation/tooling.

Landing uses ordinary signed commit/push commands with repository hooks enabled. The pre-push hook repeats the complete gate before transmitting the commit.
