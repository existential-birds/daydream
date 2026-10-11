# Record dataset workflow

Capture writes immutable run evidence under the private local dataset store.
Credentials alone do not enable remote publication; configure an explicit private
HF dataset using `--trajectory-hub-repo` or `DAYDREAM_TRAJECTORY_HUB_REPO`.

```bash
daydream corpus dataset publish --store RECORD_STORE --trajectory-hub-repo OWNER/REPO
daydream corpus dataset download --trajectory-hub-repo OWNER/REPO --revision COMMIT_SHA --output FRESH_STORE
daydream corpus dataset snapshot --store FRESH_STORE --observed-before ISO_TIMESTAMP
```

Use the printed snapshot ID for harvest, adjudication, and corpus build. Harvest
appends PR/base enrichment and scoring annotations without rewriting run
evidence. Human run labels append with `corpus label`. Per-finding labels append
through the adjudication CLI; use `corpus adjudicate --help` for queue and preview
inputs. Queues can be recreated from records and contain no remote checkpoint.

After annotation, publish through the same `dataset publish` command. Download
its exact commit into a fresh store and select a new snapshot with the intended
observe-time and optional valid-time cutoffs. This retains all eligible durable
judgment history. Local queue state is disposable and is not published.

Harvest retains each acquired reply in the finding observation's `reply_captures`
collection, including replies excluded from voting. Each entry carries its exact
`source_reply_id`, source `body_sha256`, content availability, and, when available,
the full `text` with an independently verified `captured_sha256`. Available empty
text is preserved; missing or malformed bodies carry an explicit status and reason.
Available creation/edit times, thread IDs, and source URLs are retained as provenance.
The source hash belongs to the semantic evidence; the captured hash verifies the
retained text. Capture metadata does not participate in the semantic digest.

Read all replies offline from the selected snapshot, alongside the recorded claims
and original code context:

```python
from pathlib import Path
from daydream.dataset import LocalRecordStore
from daydream.training.record_evidence import sessions_from_snapshot

records = LocalRecordStore(Path("FRESH_STORE")).read_snapshot("SNAPSHOT_ID")
for run in records.runs:
    print(run["findings"], run["original_task"])
for session in sessions_from_snapshot(records):
    for finding in session["resolutions"]:
        print(finding["item_uid"], finding["disposition"], finding["evidence"])
        for reply in finding["reply_captures"]:
            print(reply["source_reply_id"], reply["status"], reply.get("text"))
```

Finding materialization and adjudication exports also retain these captures when a
human label supplies the effective disposition. Adding captured content to the same
source evidence appends immutable history and keeps matching human judgments valid.
A source reply edit changes the semantic digest and reopens the old judgment. Earlier
snapshot IDs retain their original text and membership. Publish and exact-commit
download preserve complete reply captures through the common dataset workflow;
the existing publication secret scanner checks retained reply text too.

```bash
daydream corpus build --store FRESH_STORE --snapshot-id SNAPSHOT_ID \
  --out PROJECTION_DIR/corpus.jsonl
```

Projection is offline. Admission validates captured repository identity and rejects
C5 benchmark holdouts while preserving evidence eligibility and temporal guards.
Dataset builders make permission and licensing decisions using recorded repository,
base/head commit, and diff provenance. Split policies and caps retain deterministic
behavior. Lineage uses `record-snapshot-v2`, retaining snapshot
membership digests and, for downloaded stores, exact HF source provenance.

Publication errors preserve local evidence and emit sanitized diagnostics. Retry
`dataset publish` after resolving transport failures. Conflicting record identities
and corrupt manifests/shards fail closed; never repair them by editing sealed
records. Existing local archives and remote repositories are not migrated or deleted.
