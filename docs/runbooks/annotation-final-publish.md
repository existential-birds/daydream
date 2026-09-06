# Publish and verify an annotation snapshot

Use this runbook to checkpoint human adjudication, recover it on an empty
replacement VM, publish a content-addressed final annotation bundle, and
verify that bundle by downloading its exact Hub revision.

This procedure stops at verified annotation publication. For downstream
corpus construction and training, follow `docs/training-launch.md` after the
final download succeeds.

## Prerequisites

You need:

- a private source dataset repository containing the archived run bundles;
- a private annotation dataset repository;
- the exact lowercase 40-hex source revision;
- the checked-in production license policy;
- Hub credentials in your environment; and
- `GITHUB_TOKEN` in your environment if license enrichment must query GitHub.

Never put credentials in a URL, command argument, state file, or manifest.
The commands below use these placeholders:

- `<source-revision>`: the exact source dataset commit;
- `<curation-id>`: the stable curation identity produced by hydration;
- `<archive-index-digest>`: the SHA-256 digest of the hydrated archive index;
- `<evidence-observed-at>`: the ISO-8601 evidence observation time;
- `<final-snapshot-id>`: the ID printed by `publish-final`; and
- `<success-commit-oid>`: the Hub commit printed by `publish-final`.

## Prepare and checkpoint the first VM

### Hydrate the pinned source

First inspect the complete admission accounting without publishing:

```bash
daydream corpus hydrate-hub --source-repo org/run-bundles --source-revision <source-revision> --destination-repo org/run-bundles --stage-dir /tmp/daydream-hydrate --license-policy daydream/training/schema/license-policy-production.json --dry-run
```

Gate the next step on complete record accounting: every discovered record
must have an admission or exclusion decision, and the license decisions must
match the production policy. Copyleft repositories remain excluded unless
you explicitly opt in with `--allow-copyleft`; when needed, pass that option
to both the dry run and the subsequent hydration commands.

After this gate passes, hydrate the pinned source into the local index:

```bash
daydream corpus hydrate-hub --source-repo org/run-bundles --source-revision <source-revision> --destination-repo org/run-bundles --stage-dir /tmp/daydream-hydrate --license-policy daydream/training/schema/license-policy-production.json
```

Hydration writes the curation bundle under
`/tmp/daydream-hydrate/curated/<curation-id>/`. Record the curation ID; it is
the only local-independent identity required to discover the latest durable
checkpoint.

### Materialize and adjudicate the preview

Materialize the preview snapshot from the pinned hydrated index:

```bash
daydream corpus adjudicate materialize --index-root /tmp/daydream-hydrate --out-dir /tmp/snapshot --curation-id <curation-id> --sanitized-hub-commit <source-revision> --source-hub-commit <source-revision> --archive-index-digest <archive-index-digest> --evidence-observed-at <evidence-observed-at>
```

Build the queue and preview ledger, then record human decisions:

```bash
daydream corpus adjudicate build --index-root /tmp/snapshot --state-dir /tmp/state
```

```bash
daydream corpus adjudicate export --index-root /tmp/snapshot --state-dir /tmp/state --dry-run
```

```bash
daydream corpus adjudicate label --state-dir /tmp/state --batch 10 --disposition accepted --rationale verified-against-diff-context --labeler alice
```

Repeat `label` with the appropriate disposition, rationale, and labeler until
the coverage report shows the intended adjudication state.

### Import surviving observation history

If a local archive or backup contains additional `label_observations`, merge
it into the hydrated archive and publish the resulting SQLite history with
the state checkpoint:

```bash
daydream corpus adjudicate import-local-observations --archive-root /tmp/local-archive --index-root /tmp/snapshot --archive-dir /tmp/daydream-hydrate --state-dir /tmp/state --publish --manifest /tmp/snapshot/preview-manifest.json --hub-repo org/annotation-snapshot
```

The import keeps the source archive read-only, merges into the hydrated
`index.db`, copies that merged index into the checkpoint state, and then uses
the same immutable checkpoint protocol as `publish-state`.

### Publish every adjudication batch

Publish after every ordinary labeling batch:

```bash
daydream corpus adjudicate publish-state --state-dir /tmp/state --manifest /tmp/snapshot/preview-manifest.json --hub-repo org/annotation-snapshot
```

Every successful invocation creates one immutable content-addressed batch and
updates the stable curation pointer in the same guarded commit. The command
prints both the batch ID and the actual checkpoint revision. Re-publishing
identical state is idempotent.

## Recover on an empty replacement VM

Assume all first-VM paths are gone. Create no destination directory before
running `resume-state`; the command installs only a fully verified checkpoint.

```bash
daydream corpus adjudicate resume-state --curation-id <curation-id> --destination /tmp/state --hub-repo org/annotation-snapshot
```

This command discovers one repository revision from the stable curation
pointer, pins every download to it, verifies all declared digests, and restores
`preview-manifest.json` as data. Missing state, authentication failure, Hub
failure, corruption, or an existing destination exits nonzero.

Re-hydrate the original source revision on the replacement VM:

```bash
daydream corpus hydrate-hub --source-repo org/run-bundles --source-revision <source-revision> --destination-repo org/run-bundles --stage-dir /tmp/daydream-hydrate --license-policy daydream/training/schema/license-policy-production.json
```

Re-materialize the same preview from those pinned inputs:

```bash
daydream corpus adjudicate materialize --index-root /tmp/daydream-hydrate --out-dir /tmp/snapshot --curation-id <curation-id> --sanitized-hub-commit <source-revision> --source-hub-commit <source-revision> --archive-index-digest <archive-index-digest> --evidence-observed-at <evidence-observed-at>
```

## Build and publish the final bundle

Select the archive once and use it for both harvest and final publication.
If the checkpoint restored `/tmp/state/index.db`, use it directly to preserve
its SQLite-only history; do not copy it into the newly hydrated tree. If you
did not import a backup and the checkpoint has no optional index, use the
newly rehydrated, pinned source index instead. Keep `/tmp/state` as the state
directory in both cases so your restored annotation decisions are applied.

```bash
ANNOTATION_ARCHIVE_DIR=/tmp/daydream-hydrate
if [ -f /tmp/state/index.db ]; then
  ANNOTATION_ARCHIVE_DIR=/tmp/state
fi
daydream corpus adjudicate harvest-snapshot --index-root /tmp/daydream-hydrate --materialize-dir /tmp/snapshot --archive-dir "$ANNOTATION_ARCHIVE_DIR" --state-dir /tmp/state
```

Validate the complete final bundle without contacting the Hub:

```bash
daydream corpus adjudicate publish-final --index-root /tmp/daydream-hydrate --materialize-dir /tmp/snapshot --archive-dir "$ANNOTATION_ARCHIVE_DIR" --curation-bundle-dir /tmp/daydream-hydrate/curated/<curation-id> --hub-repo org/annotation-snapshot --state-dir /tmp/state --dry-run
```

The dry run builds the same semantic bundle as publication, checks the 80%
human-adjudication gate, and prints the computed final snapshot ID. It does not
construct a Hub client or upload any bytes.

Publish only after the dry run passes:

```bash
daydream corpus adjudicate publish-final --index-root /tmp/daydream-hydrate --materialize-dir /tmp/snapshot --archive-dir "$ANNOTATION_ARCHIVE_DIR" --curation-bundle-dir /tmp/daydream-hydrate/curated/<curation-id> --hub-repo org/annotation-snapshot --state-dir /tmp/state
```

Record the printed final snapshot ID and Hub commit. The Hub commit is the
actual success-marker commit, not the source revision or a synthetic digest.

## Verify the exact final revision

Choose a destination that does not exist. Download the final snapshot using
the exact values printed by `publish-final`:

```bash
daydream corpus adjudicate download-final --curation-id <curation-id> --snapshot-id <final-snapshot-id> --revision <success-commit-oid> --destination /tmp/annotation-bundle --hub-repo org/annotation-snapshot
```

The command pins every read to the requested success commit, verifies the
publication manifest, semantic-file digests, `SHA256SUMS`, and `_SUCCESS`
binding, then installs the complete directory in one final replacement. After
establishing ownership, a failed install attempts to remove only the staging or
installed tree whose identity it still owns. A concurrent replacement is
preserved. Failure before ownership can be established may leave a temporary
pathname for operator inspection.

Annotation publication is complete only when this command succeeds and prints
the same final snapshot ID and success commit that `publish-final` reported.
