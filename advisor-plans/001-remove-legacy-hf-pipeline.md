# Remove the legacy Hugging Face pipeline

Status: implemented. See [the implementation report](001-work-report.md) for the final workflow, preservation review, and passing gate.

## Executor instructions

Implement this plan in a new session. The user requires one HF pipeline and removal of all unused legacy HF code. Complete the consumer cutover and deletion; consolidating SDK imports alone is insufficient. Do not stop after exploration or produce another plan instead of implementing.

Planned against commit `dad49023`, 2026-10-06. Priority P1; effort L; risk HIGH because active training consumers still depend on the old pipeline. There are no prerequisite plans.

The post-cutover removal is tracked in [GitHub issue #1475](https://github.com/existential-birds/daydream/issues/1475). Its cleanup gate requires training/producer record cutover (#1469, #1470, #1471, #1472, #1474); TODO comments in the source link the legacy blocks to that issue. This plan includes the prerequisite migration work, while issue #1475 tracks the subsequent retirement.

**Confirmed feature removal:** the user explicitly authorized removal of annotation checkpoint recovery after identifying `corpus adjudicate resume-state` and its paired `publish-state` command. Delete those commands, their checkpoint/pointer/state restoration machinery, and tests whose only contract is that retired feature. Do not migrate or rebuild checkpoint recovery in JSONL. Durable published run/observation evidence remains readable through the normal dataset download workflow.

Start with `git status --short` and `git diff --stat dad49023..HEAD -- daydream tests README.md CLAUDE.md docs`. Preserve existing changes. Refresh the caller inventory when the checkout has changed; do not blindly apply stale line numbers. Read applicable AGENTS.md instructions, CLAUDE.md, CONTRIBUTING.md, Makefile, and the available codebase-design/test-audit skills. Never bypass Git hooks: no `--no-verify`, disabling environment variables, configuration overrides, or equivalent mechanisms. Fix ordinary hook failures on the working branch or report the blocker. Do not commit, push, publish, or delete external data without separate authorization.

## Required result

The canonical data path is:

`frozen run capture -> LocalRecordStore runs/observations -> immutable JSONL shards and one HF manifest -> exact-commit download -> record-based harvest/adjudication -> offline frozen corpus projection`.

All HF access uses one production SDK client. All evidence publication and pinned downloads use the existing JSONL publication workflow, extended only where a concrete retained requirement needs it. No second folder uploader, annotation bundle publisher, annotation checkpoint recovery, remote per-session resume ledger, parallel curation identity, or remote verification-success marker protocol remains. Disposable local annotation queues and derived corpus exports are allowed; they are not additional canonical evidence stores.

This is a replacement of the old pipeline. Historical archive conversion, fallback readers, deprecated aliases, compatibility writers, and automatic deletion of historical data are not required. Preserve training and adjudication semantics. Keep runtime artifacts, review output publication, shared redaction, and live local diagnostic/archive functionality outside the HF pipeline intact.

## Verified current state

Three read-only exploration agents covered transport, producer flows, and training/adjudication consumers. The implementation session must recheck these facts at its current HEAD.

- `daydream/archive/hub.py:40` defines `upload_run_bundle()` with `upload_folder()`. It has no production callers. Its only live helper is `resolve_hub_repo()`, imported by `daydream/runner.py:290`.
- `daydream/run_artifacts.py:403-409` uses `DatasetUploader` after frozen record capture; `daydream/commands/dataset.py:17,41` uses the same publisher. `daydream/dataset_hub.py:278` and `:349` are live publication/download logic, not obsolete code.
- `daydream/dataset_hub_client.py:51` and `daydream/archive/hydrate.py:1739` contain independent HF clients, SDK loaders, metadata reads, downloads, and guarded commit implementations.
- `daydream/commands/corpus.py:346` still dispatches `hydrate-hub`. `daydream/archive/hydrate.py:2167-2168` constructs source/destination clients. This entire workflow is legacy but still reachable.
- `daydream/training/adjudication/cli.py:582,619,683,703` constructs the old client for checkpoint publish/resume and final publish/download. `training/adjudication/publish.py` remains active and must be replaced before deletion.
- `daydream/archive/hydrate_client.py:19` contains `FakeHub`, imported only by tests and test fixture builders. It belongs in test support, not the production package.
- `daydream/training/harvest.py:727` queries archive SQLite rows. `training/adjudication/materialize.py:44,69,77` selects sessions or a hydrated SQLite index. `canonical.py:304` appends SQLite label observations. `final_bundle.py:358,408` loads a curated bundle and SQLite history. `training/corpus_projection/projector.py:521,535,536` reads curated batches, file artifacts, and trajectory documents. No training consumer currently reads `LocalRecordStore`.

Load-bearing excerpts:

```python
# daydream/run_artifacts.py:408-409 -- keep this canonical producer workflow
store = LocalRecordStore(config.dataset_store_path or Path.home() / ".daydream" / "dataset")
status = DatasetUploader(store, config.trajectory_hub_repo).upload()

# daydream/training/adjudication/materialize.py:44-48 -- replace both old input paths
def index_sessions(index_root: Path) -> tuple[list[dict[str, Any]], str]:
    if (index_root / _SESSIONS_OUT_FILENAME).is_file():
        return _load_sessions(index_root)
    return _sessions_from_hydrated_stage(index_root)

# daydream/dataset.py:357-365 -- reuse the existing semantic reducer
def effective_judgment(self, run_id: str, item_uid: str) -> dict[str, Any]:
    # Filters eligible finding judgments, then:
    return effective_adjudication(history)
```

The new record schema supports finding judgments and PR/base/license enrichment. Verify whether retained annotation and scoring evidence fits the schema; explicitly version any necessary extension. Do not silently omit history or fabricate missing evidence. Annotation checkpoint state is outside the retained requirements.

## Scope

Modify the dataset store/publication modules, producer integration/configuration, `daydream/commands/{dataset,corpus,hydrate,calibrate,common}.py`, archive HF modules, training harvest/adjudication/corpus readers, and their direct callers, tests, harnesses, fixture builders, and documentation as required by this cutover. A neutral `daydream/hub.py` is the suggested home for the one SDK client. Remove `archive/hub.py`, `dataset_hub_client.py`, and `archive/hydrate_client.py` after moving their retained responsibilities. Do not leave import aliases or forwarding adapters at their former paths.

Do not change model recipes, reward policy, benchmark methodology, backend execution, runtime artifact ownership/freeze/publication protections, or vendored ATIF implementation. Do not delete local archives or remote repositories. Preserve historical changelog entries. Shared license/identity rules are not disposable because they were located in a legacy module; relocate them where live callers need them.

## Implementation sequence

### 1. Establish the preservation inventory

Trace production and CLI callers separately from tests, including `daydream_ext`, scripts, configuration, packaged workflows, docs, and fixture builders. Write a deletion/retention map in the work report: symbol, production caller, removal or replacement, retained contract, and owning test. Vulture scans production and tests together, so tests can conceal dead production code.

Run `uv lock --check` before any `uv run` or dependency sync. Establish focused baseline results for dataset, harvest, adjudication, projection, calibration, and archive lifecycle tests. Record genuine pre-existing failures; never remove failing tests just to make the gate pass.

**Verify:** the inventory accounts for each current HF SDK implementation and every supported HF CLI entrypoint; focused baseline commands finish with recorded results.

### 2. Preserve one client and delete the already-dead uploader

Move destination resolution and required SDK behavior into the canonical HF module. Keep one lazy SDK loader, validated repository metadata and commit IDs, authenticated exact-revision downloads, and guarded atomic tree commits. Prefer bytes directly over writing temporary files solely for an adapter's input signature.

Preserve the stronger current contract: private destinations require confirmed `private is True`; commit IDs must be full lowercase hashes; only confirmed remote absence counts as missing; auth/outage/local-cache errors propagate as value-free diagnostics; ordinary HF cache symlinks may resolve to regular blobs; only create-commit HTTP 412 is a typed concurrent-update conflict. Never persist arbitrary SDK exception payloads. Credentials alone must not enable uploads.

Remove `upload_run_bundle()`, its globals/backoff/warning machinery, and `archive/hub.py` after moving its live resolver. Retarget live dataset consumers directly. Temporary migration of active legacy consumers is acceptable within this implementation, but the finished tree must have no legacy client, compatibility imports, or parallel uploader.

**Verify:** dataset run/CLI tests pass through the canonical client; missing dependency, private destination, exact pin, missing file, cache failure, uncertain success, and real commit conflict cases retain coverage.

### 3. Move harvest and annotation history onto records

Replace archive-path and SQLite acquisition with validated run records. Reuse pure harvest/reward reductions and existing GitHub evidence acquisition. Append PR/base/license enrichment and automated/human judgments as typed observations, without modifying sealed run evidence. Preserve the distinction between run identity, host finding identity, fingerprint, and externally derived record identity.

Update `training/harvest.py`, adjudication materialization, canonical persistence, observation writing, preview/queue/report/export/snapshot callers, corpus label/harvest commands, and calibration readers. Use frozen `SnapshotRecords` for selected evidence. Reuse `effective_adjudication`; preserve human precedence, valid/observed cutoffs, changed-evidence requeueing, missing/unanswered findings, full population accounting, model suggestions requiring human review, and intrinsic/posterior reward separation.

Avoid rebuilding bundle directories or SQLite to satisfy old readers. Small in-memory translations into existing pure reducers are acceptable. SQLite may remain only for unrelated live functionality or a justified disposable cache, never as authoritative annotation history.

**Verify:** real CLI harvest/label/adjudication paths operate on a new captured/downloaded record store, without a legacy archive index. Precedence, temporal eligibility, reward, identity, and completeness tests pass.

### 4. Remove annotation checkpoint recovery and unify evidence publication

Delete `corpus adjudicate publish-state` and `resume-state`, their parser options/handlers, checkpoint batches/pointers, three-way checkpoint state merge, and state restoration. This feature removal is authorized; it does not require a replacement command, checkpoint format, or total-machine-loss recovery integration test.

Publish annotation observations through the same JSONL queue, shards, manifest, conflict handling, and download workflow as run records. Replace the old annotation final-bundle transport and curation identity with the canonical record/snapshot identity. Remove standalone legacy `publish-final`/`download-final` transport commands in favor of canonical dataset publish/download; keep any useful pure annotation report/export logic without remote transport.

Preserve durable published judgment history, concurrent observation union, idempotent retries, and exact-commit evidence download integrity. These are canonical dataset contracts, not annotation checkpoint recovery. Do not persist transient UI/queue/checkpoint state merely to reproduce the removed feature. Do not reproduce old `curated/<id>`, annotation checkpoint pointers, bundle `_SUCCESS`, or remote resume-ledger machinery behind new names. Review annotation tests for their retained behavioral contracts; delete feature-only checkpoint/recovery tests and port independent annotation/publication contracts.

**Verify:** retired checkpoint commands are absent from dispatch/help and rejected by the CLI without network or filesystem side effects. A real CLI JSONL journey publishes judgments, downloads a pinned dataset, and builds deterministic corpus output from it. The old total-VM-loss checkpoint test can be removed with the retired feature; it need not be ported.

### 5. Cut frozen corpus projection over to records

Replace curated-bundle input in `training/corpus_projection/projector.py`, associated bundle readers/configuration, and `corpus build`. Read retained task/diff/revisions, trajectories, findings, verification/scoring, and provenance directly from frozen records, with the eligible observation overlay. Update calibration and training coordinator callers of the replaced inputs.

Preserve training output contracts, reward calculations, license/exclusion/copy-left admission, gold/silver eligibility, trace segmentation, deterministic splits and membership, repository/stack/profile caps, and reproducible lineage. Projection must run offline once the record snapshot and policies are present. Where storage-derived identity changes are unavoidable, explicitly version/document the change and prove stable deduplication and membership; do not silently change examples.

**Verify:** compare normalized outputs for equivalent synthetic evidence represented in the new records; retain independent tests of eligibility, deterministic splits/caps, task/process examples, temporal leakage, and license refusal. The primary CLI journey reaches frozen corpus output without any hydration/bundle index.

### 6. Delete the superseded HF pipeline and its support

After production callers have moved, remove `archive/hydrate.py` and `commands/hydrate.py`, old corpus `hydrate-hub` dispatch, bundle-only discovery/assembly/hydration/curation/index rebuilding, remote batch/resume/checksum/success publication, the old SDK client/factory/protocols, and legacy annotation bundle publication. Remove the explicitly retired annotation checkpoint commands without implementing replacements. Delete or simplify `training/adjudication/{publish,final_bundle}.py` and `training/corpus_projection/bundle.py` according to their remaining semantic responsibilities; file names are not a reason to retain obsolete implementations.

Inspect `archive/hydrate_rules.py` and related schema/readers: preserve or relocate live license, repository-identity, and other pure shared rules before removing bundle-only code. Do not sweep local archive/evaluation/dump code into this deletion without proving it is solely superseded HF support.

Remove obsolete upload tests in `tests/test_hub_upload.py` while carrying destination precedence/hostile checkout-config tests to live coverage. Remove `_upload_fixture` and deleted-uploader monkeypatches from `test_archive_integration.py` and `test_archive.py`; preserve their meaningful local persistence/no-network outcomes. Remove `TARGET_HUB_KEY_CONFIG` only if no live tests need it. Move useful fake-HF behavior into `tests/harness`, preferably one serialization-aware canonical fake, and delete production `hydrate_client.py`, obsolete bundle fixture builders, and tests that only preserve deleted machinery. Never delete a still-required behavioral test without a named replacement.

Update README, CLAUDE.md, relevant corpus/calibration/training docs, runbooks, CLI help, and shipped workflows to the final single workflow. Historical changelog prose may still name old code.

**Verify:** caller searches find no production references to the removed uploader, modules, old hydration dispatch, annotation checkpoint/final-bundle transport, or compatibility paths. All HF SDK imports/client construction occur in the canonical client. No tests keep deleted production code alive.

## Validation and completion

Use the existing real-path standard: enter through `runner.run` or public CLI with real temporary Git/filesystem/event-loop behavior; stub only external model/HF/GitHub seams. Retain focused transport/schema/fault tests where they protect independent contracts. Read the test-audit skill before changing tests and obtain an independent read-only preservation review before finishing.

The primary acceptance test covers capture, private JSONL publication, pinned download, enrichment/judgments, publication of observations through the same pipeline, and offline frozen corpus creation. Annotation checkpoint recovery and total-machine-loss reconstruction of transient annotation state are intentionally removed. Cover interruption, lost successful responses, two writers, identity conflicts, corrupt manifests/shards, failed local persistence, oversized complete records, public destinations, blocking secrets, absent licenses, and preserved review outputs.

Verification commands come from the current Makefile:

- First: `uv lock --check` (exit 0 before dependency operations).
- Focused: `uv run pytest <retained-and-migrated-owner-test-paths>`; start with `tests/test_dataset_store.py`, `test_dataset_capture.py`, `test_dataset_hub.py`, `test_dataset_hub_run.py`, `test_dataset_commands.py`, `test_training_harvest.py`, `test_training_adjudication_precedence.py`, and the new canonical record-to-corpus integration journey. Include affected projection/calibration/CLI/archive siblings and any renamed transport tests. Do not keep checkpoint feature tests merely to retain deleted code.
- Then: `make lint`, `make typecheck`, `make deadcode`, and `git diff --check` (all exit 0).
- Final required gate: `make check` (all required checks pass; explicitly report any Docker-dependent actionlint skip). Run `make rl-check` if the separately locked RL project changes. Never weaken coverage settings, exclusions, verification hooks, or tests to make removal pass.

Done means one SDK client, one canonical HF evidence publication/download workflow, record-based training consumers, no annotation checkpoint feature, no remaining superseded HF production/test scaffolding, updated docs, and the complete gate plus preservation review. Report production versus test/support deletions separately, retained shared functionality, tests actually run, any unavailable proof, and remaining work honestly. Update the plan status in `advisor-plans/README.md`.

## Blockers and maintenance

If an external/extension contract truly requires an old workflow, or required evidence cannot be represented without changing training semantics, report the concrete contract and affected callers. Do not resolve it with a hidden legacy fallback or silently discard evidence. Investigate and fix normal implementation/test failures; if progress requires an unapproved semantic change, obtain that decision while continuing independent authorized work.

Future evidence types and annotation commands should enter through the record store and canonical publisher. Reviewers should scrutinize published observation history, snapshot eligibility, schema changes, consumer independence from legacy paths, and tests that previously modeled different transaction behavior from production. Do not reintroduce annotation checkpoint recovery under a new command name.
