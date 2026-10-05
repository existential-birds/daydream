# Local run evidence

Enable collection for a new run:

```sh
daydream --review --capture-data --dataset-store ~/.daydream/dataset --no-archive --no-tracing /path/to/project
```

`--capture-data` is an operator choice; capture remains disabled by default. A store path alone
does not enable it. `--no-capture-data` explicitly disables collection. These shared options also
apply to Improve and custom flows. The existing archive workflow retains its default. Choose a
store outside the reviewed checkout and Daydream's private runtime/workspace directories;
overlapping paths are refused before collection writes any bytes.

Capture consumes the existing frozen run and artifact boundary after producers finish. It
preserves the original task acquired before model work separately from final Git state and
the recommended fix patch. Root and child trajectory documents retain their producer identities,
invocation and step membership, and fork registration evidence; document-array position does
not establish execution order. Structured claims and item/source links are collected independently
of `--findings-out`. Missing, unproduced, failed, withheld, and intentionally empty evidence have
distinct representations. A missing verifier does not earn correctness credit; intrinsic reward
and later posterior cost remain separate.

Shared redaction applies before captured-content digests are calculated. Original task
`source_diff_sha256` identifies the producer's original input bytes; `diff_sha256` verifies the
retained credential-safe diff. Recommended patches likewise retain separate source and captured
digests. A redacted copy is never presented as content verified by its source-body hash. Remaining
blocking secret findings withhold the record. `run_record_schema()` and `observation_record_schema()`
publish the current JSON Schema contracts; `parse_run()` and `parse_observation()` additionally
validate digests and identity associations.

Cooperative interruption captures available evidence before workspace teardown. SIGKILL or power
loss before finalization can lose uncaptured run evidence. Store writes use private storage,
serialized writers, and atomic durable commits. The JSONL store does not reconstruct archive
directories, require a trace destination, or treat SQLite as observation history.

The public entry point is `daydream.dataset.LocalRecordStore`. Commit and append return a
`CommitResult`: `committed=True` means the complete record was persisted; `False` means an identical
record already exists. Failures raise `StoreError` with a sanitized `code`. Neither result means
the data was published to a remote service. The default per-record limit is 64 MiB; the Python API
can select a different `max_record_bytes`. Oversized records are refused in full.

```python
from pathlib import Path
from daydream.dataset import LocalRecordStore

store = LocalRecordStore(Path.home() / ".daydream" / "dataset")
snapshot = store.select_snapshot(
    observed_before="2026-10-04T23:59:59+00:00",
    valid_before="2026-10-04T23:59:59+00:00",
)
records = store.read_snapshot(snapshot.snapshot_id)
for run in records.runs:
    print(run.run_id, run.outcome, run.original_task.status)
```

`records.observations` retains the selected history; `records.eligible_observations` additionally
applies the valid-time cutoff. Supply `run_ids` to `select_snapshot` to select particular captured
runs. Keep the returned snapshot identity to repeat the same read after new data arrives.

Run records are immutable. Later run labels, finding judgments, and PR/base/license enrichment
are typed observations with source identity, author role, valid time, observation time, evidence
digests, and policy versions. Missing license evidence supplies no training admission. Model
suggestions require review and do not override a human decision. Correction text has its own
captured-content digest and redaction provenance, separate from the original reply-body hash
and the unchanged semantic classifier/deduplication projection.

Append a typed observation to a captured run, then select a new snapshot:

```python
from daydream.dataset import ObservationRecord, RunLabelPayload, semantic_evidence_digest
from daydream.training.labeler_versions import HUMAN_LABELER_VERSION, LABELER_POLICY_VERSION, RUBRIC_SCHEMA_VERSION

evidence = {"source": "operator review", "result": "accepted"}
store.append_observation(ObservationRecord(
    observation_id="operator-review-2026-10-05",
    run_id=records.runs[0].run_id,
    valid_at="2026-10-05T12:00:00+00:00",
    observed_at="2026-10-05T12:05:00+00:00",
    source=HUMAN_LABELER_VERSION,
    author="operator",
    role="rater",
    policy_version=LABELER_POLICY_VERSION,
    rubric_version=RUBRIC_SCHEMA_VERSION,
    semantic_evidence=evidence,
    evidence_digest=semantic_evidence_digest(evidence),
    payload=RunLabelPayload(label="accepted"),
))
later = store.select_snapshot(observed_before="2026-10-05T23:59:59+00:00")
assert len(store.read_snapshot(later).observations) >= 1
assert store.read_snapshot(snapshot.snapshot_id).observations == records.observations
```

A `FindingJudgmentPayload` additionally targets the captured `item_uid`, which is independent
of the finding fingerprint or its display number. `SnapshotRecords.effective_judgment(run_id,
item_uid)` uses the existing adjudication reducer over eligible pinned history; it retains conflict
and review requirements. Append enrichment with `EnrichmentPayload(kind="license", evidence=...)`;
the other kinds are `pr` and `base`. The payloads preserve raw evidence for downstream consumers; appending a label
does not project training examples or implement an adjudication command.

A dataset snapshot pins exact run and observation membership, content/shard digests, and temporal
cutoffs. Observation time selects acquired history; valid time determines evidence eligibility.
Appending later evidence cannot change a saved snapshot. Conflicting immutable identities,
orphan references, malformed records, unsupported schemas and sizes, privacy refusal, and
interrupted persistence produce diagnostics instead of a partial committed record.

Privacy and dataset persistence failures preserve completed review results and outputs. Runtime
artifact integrity and publication failures continue to fail closed. Historical migration,
Hugging Face publication, live GitHub harvesting, adjudication commands, corpus cutover, and
review improvement reports are downstream work.
