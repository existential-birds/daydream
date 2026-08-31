# Annotation Final-Publish Runbook (issue #1078, M9)

This runbook walks the daydream operator through publishing the immutable,
per-finding annotation snapshot to the private Hub and projecting it into a
corpus-v2 training bundle — entirely from the `daydream corpus adjudicate`
CLI. There are no hand-authored JSON files and no `python -c` imports: every
artifact in the chain is produced and validated by a supported command.

Audience: the daydream operator on a fresh VM, holding the private Hub dataset
repo name and provider credentials in their environment.

Two operator-safety constraints frame every step:

- **Success is defined only by the end of step 8.** A run is finished when a
  clean-download verification passes and the bundle's `_SUCCESS` marker is
  present. Anything before that — including a green `--dry-run` — is not
  success.
- **The VM is expendable; the Hub checkpoint is not.** Step 2 recovers all
  published adjudication state from the Hub after VM loss. Re-run it any time
  you are unsure the local state is intact.

Every command below is a literal, single-line CLI invocation that parses against the real parser (enforced by
`tests/test_cli_adjudicate.py::test_runbook_commands_parse_against_real_parser`).

---

## 1. Hydrate the source index

Bring the producer's archived run bundles down from the Hub into a local,
normalized index root:

```bash
daydream corpus hydrate-hub --source-repo org/run-bundles --source-revision <commit-sha> --destination-repo org/run-bundles --stage-dir /tmp/daydream-hydrate
```

The resulting `downloads/<revision>/` tree is the hydrated index root this
runbook refers to as `INDEX_ROOT`. It must contain `sessions.jsonl`.

## 2. Restore adjudication state after VM loss

If you are resuming on a fresh VM (or doubt the local state), restore the
published adjudication state — digest-verified — from the Hub checkpoint:

```bash
daydream corpus adjudicate resume-state --manifest /tmp/state/preview-manifest.json --stage-dir /tmp/state
```

The manifest is the `preview-manifest.json` written by the last
`publish-state` run; pin it with the state (keep it outside the expendable VM
path, e.g. in the Hub repo or an operator-controlled store). After this step
`/tmp/state` holds the restored observations and queue.

## 3. Materialize the annotation snapshot

Materialize one record per finding — automatic decisive, human-decisive, and
non-decisive alike — into the snapshot directory:

```bash
daydream corpus adjudicate materialize --index-root /tmp/daydream-hydrate/downloads/<revision> --out-dir /tmp/snapshot
```

`/tmp/snapshot` receives `sessions.jsonl` and the `preview-manifest.json` that
pins it. This snapshot is the input to both the canonical harvest (step 5) and
the final bundle (steps 6–7).

## 4. Build, label, and publish the human queue

Build the unresolved-only operator queue, record human observations, and push
state back to the Hub. These three commands repeat per labeling session:

```bash
daydream corpus adjudicate build --index-root /tmp/daydream-hydrate/downloads/<revision> --state-dir /tmp/state
```

```bash
daydream corpus adjudicate label --state-dir /tmp/state --batch 10 --disposition accepted --rationale verified-against-diff-context --labeler alice
```

```bash
daydream corpus adjudicate publish-state --state-dir /tmp/state --manifest /tmp/snapshot/preview-manifest.json
```

`build` never shows decisive records — those are adjudicated automatically.
`label` records provenance (who, why, when) that the final bundle carries
forward. `publish-state` uploads additively, so it is safe to re-run.

## 5. Canonical harvest with the drift gate

Run the canonical harvest: the drift gate checks the complete record set
(decisive records included) against the queue, merges precedence, and appends
label observations to the archive index:

```bash
daydream corpus adjudicate harvest-snapshot --index-root /tmp/daydream-hydrate/downloads/<revision> --materialize-dir /tmp/snapshot --archive-dir /tmp/archive --state-dir /tmp/state
```

A drift failure here means the snapshot and the queue disagree; fix the
upstream labeling rather than bypassing the gate.

## 6. Construct and validate the final bundle (dry run)

Build the staging bundle — annotations, sessions, observation history,
coverage report, and generated lineage — and validate every gate without
publishing:

```bash
daydream corpus adjudicate publish-final --index-root /tmp/daydream-hydrate/downloads/<revision> --materialize-dir /tmp/snapshot --archive-dir /tmp/archive --curation-bundle-dir /tmp/daydream-hydrate/downloads/<revision> --hub-repo org/annotation-snapshot --dry-run
```

The private-repo gate, secret scan, and SHA256SUMS construction all run here.
Only proceed when the dry run is green.

## 7. Publish the final bundle

Repeat the same command without `--dry-run` to upload the bundle additively to
the Hub (the `_SUCCESS` marker is written last):

```bash
daydream corpus adjudicate publish-final --index-root /tmp/daydream-hydrate/downloads/<revision> --materialize-dir /tmp/snapshot --archive-dir /tmp/archive --curation-bundle-dir /tmp/daydream-hydrate/downloads/<revision> --hub-repo org/annotation-snapshot
```

## 8. Verify by clean download

Success is defined only here. Re-download the published revision from the Hub
(e.g. a clean clone of the dataset repo), re-run the same command with
`--dry-run` against the downloaded tree, and confirm the bundle's `_SUCCESS`
marker is present in the published layout. If either check fails, the publish
is not done — re-run step 7.

## 9. Project corpus v2

Project the curated bundle and the verified annotation bundle into the frozen
corpus-v2 training records:

```bash
daydream corpus build-v2 --bundle-root /tmp/daydream-hydrate/downloads/<revision> --annotation-bundle-root /tmp/annotation-bundle --out /tmp/corpus-v2/corpus-v2.jsonl
```

`build-v2` self-verifies and cross-links the annotation bundle before
projecting, applies per-tier caps, and writes the split manifests and lineage
beside the output. A green run here is the end of the pipeline.
