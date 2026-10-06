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
appends PR/base/license enrichment and scoring annotations without rewriting run
evidence. Human run labels append with `corpus label`. Per-finding labels append
through the adjudication CLI; use `corpus adjudicate --help` for queue and preview
inputs. Queues can be recreated from records and contain no remote checkpoint.

After annotation, publish through the same `dataset publish` command. Download
its exact commit into a fresh store and select a new snapshot with the intended
observe-time and optional valid-time cutoffs. This retains all eligible durable
judgment history. Local queue state is disposable and is not published.

```bash
daydream corpus build --store FRESH_STORE --snapshot-id SNAPSHOT_ID \
  --license-policy LICENSE_POLICY --out PROJECTION_DIR/corpus.jsonl
```

Projection is offline. Admission requires trustworthy repository/license evidence
and a pinned policy; missing licenses fail closed. Exact repository copyleft
opt-ins use repeatable `--allow-copyleft OWNER/REPO`. Split policies and caps retain
deterministic behavior. Lineage uses `record-snapshot-v1`, retaining snapshot
membership digests and, for downloaded stores, exact HF source provenance.

Publication errors preserve local evidence and emit sanitized diagnostics. Retry
`dataset publish` after resolving transport failures. Conflicting record identities
and corrupt manifests/shards fail closed; never repair them by editing sealed
records. Existing local archives and remote repositories are not migrated or deleted.
