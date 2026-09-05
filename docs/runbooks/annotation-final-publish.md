# Annotation Final-Publish Runbook (issue #1078, M9)

This runbook walks the daydream operator through publishing the immutable,
per-finding annotation snapshot to the private Hub and projecting it into a
corpus-v2 training bundle — entirely from the `daydream corpus adjudicate`
CLI. There are no hand-authored JSON files and no `python -c` imports: every
artifact in the chain is produced and validated by a supported command.

Audience: the daydream operator on a fresh VM, holding the private Hub dataset
repo name and provider credentials in their environment.

Two operator-safety constraints frame every step:

- **Success is defined only by the end of step 10.** A run is finished when a
  clean-download verification passes, the bundle's `_SUCCESS` marker is
  present, and a green `daydream train --corpus-v2` dry run consumes the
  frozen projection. Anything before that — including a green
  `--dry-run` on an intermediate command — is not success.
- **The VM is expendable; the Hub checkpoint is not.** Step 2 recovers all
  published adjudication state from the Hub after VM loss. Re-run it any time
  you are unsure the local state is intact.

Every `daydream corpus adjudicate` command below (steps 1–8, including the
lettered sub-steps 3a–3b) is a literal, single-line CLI invocation that parses against the real parser (enforced by
`tests/test_cli_adjudicate.py::test_runbook_commands_parse_against_real_parser`). The step 9 (`corpus build-v2`) and step 10 (`train --corpus-v2`) commands are not covered by that parser check — `tests/test_training_docs_v2.py` only asserts their presence.

---

## 1. Hydrate the source index

Bring the producer's archived run bundles down from the Hub into a local,
normalized index root. First run the identical command with `--dry-run`:

```bash
daydream corpus hydrate-hub --source-repo org/run-bundles --source-revision <commit-sha> --destination-repo org/run-bundles --stage-dir /tmp/daydream-hydrate --license-policy daydream/training/schema/license-policy-production.json --allow-copyleft <owner/repo> --dry-run
```

The non-dry publication below is gated on that dry-run: proceed only after a
completed dry-run reports full record accounting (the discovered-candidate
tally, the license-admission gate's per-code and per-repo counts matching over
the license-adjudicated population — imported sessions plus license-gate
rejections; ingest/fixture rejections are reported separately, never counted
as adjudicated) and an admitted count you accept.

```bash
daydream corpus hydrate-hub --source-repo org/run-bundles --source-revision <commit-sha> --destination-repo org/run-bundles --stage-dir /tmp/daydream-hydrate --license-policy daydream/training/schema/license-policy-production.json --allow-copyleft <owner/repo>
```

`--license-policy` is required on every non-dry publication — the command
refuses to run without it. `--allow-copyleft` opts in, by exact `owner/repo`
slug (case-insensitive; repeat the flag once per slug, e.g.
`--allow-copyleft a/b --allow-copyleft c/d`), specific copyleft-licensed
repositories the production policy would otherwise reject; omit it when no
such exception is intended.

License-evidence enrichment (which fills legacy bundles' missing evidence
from the live GitHub license API during both the dry-run and the
publication) requires `GITHUB_TOKEN` in the environment — exported,
read-only, and never placed on a URL or argv. Export it before running
either command above, or the step fails after download/ingest with a clear
`GITHUB_TOKEN is not set` error instead of a silent empty-token 401.

The stage dir itself is the hydrated index root this runbook refers to as
`INDEX_ROOT`: hydration writes the SQLite index (`index.db`) and one sanitized
per-run trajectory (`runs/<session_id>/trajectory.json`) at the stage root,
with the pinned source snapshot under `downloads/<revision>`. A hydrated
staging archive has no `sessions.jsonl` — the commands below derive sessions
from the index plus trajectories.

## 2. Restore adjudication state after VM loss

If you are resuming on a fresh VM (or doubt the local state), restore the
published adjudication state — digest-verified — from the Hub checkpoint:

```bash
daydream corpus adjudicate resume-state --manifest /tmp/state/preview-manifest.json --stage-dir /tmp/state --hub-repo org/annotation-snapshot
```

The manifest is the `preview-manifest.json` written by the last
`publish-state` run; pin it with the state (keep it outside the expendable VM
path, e.g. in the Hub repo or an operator-controlled store). After this step
`/tmp/state` holds the restored observations and queue.

## 3. Materialize the annotation snapshot

Materialize one record per finding — automatic decisive, human-decisive, and
non-decisive alike — into the snapshot directory:

```bash
daydream corpus adjudicate materialize --index-root /tmp/daydream-hydrate --out-dir /tmp/snapshot --curation-id <curation-id> --sanitized-hub-commit <commit-sha> --source-hub-commit <commit-sha> --archive-index-digest <archive-index-digest> --evidence-observed-at <evidence-observed-at>
```

All five pin components are mandatory — a missing one is an exit-1 data
problem, not a usage error, and nothing is written. `--curation-id` is the
single `cur-*` directory hydration wrote under
`/tmp/daydream-hydrate/curated/` (recorded as `curation_id` in that
directory's `curation-manifest.json`); `--source-hub-commit` is the pinned
`<commit-sha>` from step 1 (recorded as `source_hub_commit` in the same
curation manifest), and `--sanitized-hub-commit` is the same pinned revision.
`--archive-index-digest` is the 64-hex sha256 of the hydrated archive index;
`--evidence-observed-at` is the ISO-8601 observation timestamp.

`/tmp/snapshot` receives `sessions.jsonl` and the `preview-manifest.json` that
pins it. This snapshot is the input to both the canonical harvest (step 5) and
the final bundle (steps 6–7).

## 3a. Preview the ledger — the pre-harvest drift check

Before labeling (or re-harvesting on a resume), build the digest-pinned preview
ledger over the materialized snapshot and validate the export rows it feeds —
without writing anything:

```bash
daydream corpus adjudicate export --index-root /tmp/snapshot --state-dir /tmp/state --dry-run
```

This runs the read-only preview over the snapshot's `sessions.jsonl` and pins
`preview-ledger.json` (canonical JSON, byte-identical for an identical index)
in `/tmp/state`. The ledger is the drift reference the canonical harvest's
drift gate (step 5) adjudicates the queue against — a re-run of this command
against a changed snapshot reports any drifted record ids instead of silently
merging them, which is exactly what you want to catch before labeling or
harvesting. Re-run it any time the snapshot or the observations change.

## 3b. Import surviving local history into the hydrated archive

Before the canonical harvest, merge any surviving local archive/backup roots'
immutable `label_observations` histories into the hydrated stage's archive
index — the same `index.db` (under `INDEX_ROOT`, i.e. `/tmp/daydream-hydrate`)
that step 5 appends the harvest's observations into, so imported and
harvested resolutions meet in one place. First plan the import without
writing anything:

```bash
daydream corpus adjudicate import-local-observations --archive-root /tmp/local-archive --index-root /tmp/snapshot --archive-dir /tmp/daydream-hydrate --state-dir /tmp/state --dry-run
```

`--archive-root` (repeatable) is each local archive/backup root holding an
`index.db` with a `label_observations` history; `--index-root` is the pinned
materialized snapshot the import links session identity and per-finding
evidence against. The dry run inventories the sources, runs the identity
linkage and content-digest dedupe against the pinned index, and reports the
reason-coded per-session accounting (the buckets always sum to the source row
count) while writing nothing.

Proceed with the real run only after the planned accounting looks right:

```bash
daydream corpus adjudicate import-local-observations --archive-root /tmp/local-archive --index-root /tmp/snapshot --archive-dir /tmp/daydream-hydrate --state-dir /tmp/state
```

The real run is read-only over the sources: it appends the surviving
(survived-dedupe) rows — after a fail-closed redaction + secret scan — into
the `--archive-dir` archive and writes the digest-stable import report and
ledger beside the scan artifacts in `--state-dir`. The import never writes
the state-dir `index.db`; the hydrated archive is the single merge target.
To checkpoint the merged state for fresh-VM resume, add
`--publish --manifest /tmp/snapshot/preview-manifest.json` (mutually
exclusive with `--dry-run`) after the real run has succeeded once.

## 4. Build, label, and publish the human queue

Build the unresolved-only operator queue, record human observations, and push
state back to the Hub. These three commands repeat per labeling session:

```bash
daydream corpus adjudicate build --index-root /tmp/snapshot --state-dir /tmp/state
```

```bash
daydream corpus adjudicate label --state-dir /tmp/state --batch 10 --disposition accepted --rationale verified-against-diff-context --labeler alice
```

```bash
daydream corpus adjudicate publish-state --state-dir /tmp/state --manifest /tmp/snapshot/preview-manifest.json --hub-repo org/annotation-snapshot
```

`build` consumes the materialized snapshot (the hydrated stage root has no
`sessions.jsonl`; the snapshot written by step 3 does) and never shows
decisive records — those are adjudicated automatically.
`label` records provenance (who, why, when) that the final bundle carries
forward. `publish-state` uploads additively, so it is safe to re-run.

## 5. Canonical harvest with the drift gate

Run the canonical harvest: the drift gate checks the complete record set
(decisive records included) against the queue, merges precedence, and appends
label observations to the hydrated stage's archive index (`--archive-dir` is
the same `index.db` hydration wrote the runs into — no separate archive
directory is materialized):

```bash
daydream corpus adjudicate harvest-snapshot --index-root /tmp/daydream-hydrate --materialize-dir /tmp/snapshot --archive-dir /tmp/daydream-hydrate --state-dir /tmp/state
```

A drift failure here means the snapshot and the queue disagree; fix the
upstream labeling rather than bypassing the gate.

## 6. Construct and validate the final bundle (dry run)

Build the staging bundle — annotations, sessions, observation history,
coverage report, and generated lineage — and validate every gate without
publishing:

```bash
daydream corpus adjudicate publish-final --index-root /tmp/daydream-hydrate --materialize-dir /tmp/snapshot --archive-dir /tmp/daydream-hydrate --curation-bundle-dir /tmp/daydream-hydrate/curated/<curation-id> --hub-repo org/annotation-snapshot --state-dir /tmp/state --dry-run
```

The dry run validates the staging bundle's construction, coverage, and lineage
only, and prints the 80% human-adjudication admission-gate verdict (PASS/FAIL
over the outcome-bearing numerator/denominator). The private-repo gate, the
admission-gate refusal, the secret scan, and SHA256SUMS construction run at
real publish time (step 7). Only proceed when the dry run is green — a red
gate means more outcome-bearing records must be adjudicated and the bundle
rebuilt before the Hub can ever see it.

`--curation-bundle-dir` is the hydration-produced curated bundle root — the
single `cur-*` directory under `/tmp/daydream-hydrate/curated/` (the curation
id is derived deterministically from the pinned source commit and is recorded
as `curation_id` in `/tmp/snapshot/preview-manifest.json`).

## 7. Publish the final bundle

Repeat the same command without `--dry-run` to upload the bundle additively to
the Hub (the `_SUCCESS` marker is written last). A bundle whose coverage
report fails the 80% human-adjudication admission gate is refused (the
handler exits 1) before any byte is uploaded: adjudicate more outcome-bearing
records and rebuild before re-publishing.

```bash
daydream corpus adjudicate publish-final --index-root /tmp/daydream-hydrate --materialize-dir /tmp/snapshot --archive-dir /tmp/daydream-hydrate --curation-bundle-dir /tmp/daydream-hydrate/curated/<curation-id> --hub-repo org/annotation-snapshot --state-dir /tmp/state
```

## 8. Verify by clean download

Success is defined only here. Clean-download the published final bundle tree —
`annotations/<curation-id>/<snapshot-id>/final/` in the dataset repo (the ids
come from `/tmp/snapshot/preview-manifest.json`) — into a fresh directory
(e.g. `/tmp/annotation-bundle`, which step 9 consumes), verify every file
against the bundle's published `SHA256SUMS`, and confirm the `_SUCCESS` marker
is present in that layout. Any checksum mismatch or a missing `_SUCCESS` means
the publish is not done — re-run step 7.

## 9. Project corpus v2

Project the curated bundle and the verified annotation bundle into the frozen
corpus-v2 training records:

```bash
daydream corpus build-v2 --bundle-root /tmp/daydream-hydrate/curated/<curation-id> --annotation-bundle-root /tmp/annotation-bundle --license-policy <license-policy.json> --out /tmp/corpus-v2/corpus-v2.jsonl
```

`build-v2` self-verifies the annotation bundle (`_SUCCESS`, `SHA256SUMS`,
`lineage.json`, `annotations.jsonl`) and cross-links it against the curated
bundle root before projecting — `--bundle-root` is the hydration-produced
curated bundle (`/tmp/daydream-hydrate/curated/<curation-id>/`, which carries
`_SUCCESS`, `SHA256SUMS`, and `curation-manifest.json`), not the raw
`downloads/<revision>` snapshot tree. `--license-policy` is the digest-pinned
license-policy JSON every record's per-repo license decision is resolved from
(it is required — the command refuses to run without it). It applies per-tier
caps and writes the split manifests and lineage beside the output.

## 10. Train on the frozen corpus-v2 projection

The projection directory written by step 9 is a frozen, immutable training
input — do not edit, filter, or re-split it. Train the four-stage pipeline
against it directly:

```bash
daydream train --corpus-v2 /tmp/corpus-v2 --out /tmp/train-out --dry-run
```

(Drop `--dry-run` for the real run; `--corpus-v2` is mutually exclusive with
the v1 `--corpus` flag.) The loader fail-closes before any stage runs unless
the directory's `_SUCCESS` marker and `lineage.json` are present, and it
re-applies the C5 exclusion list and the C8 copyleft opt-in gate fail-closed
on every load — the projector's decisions are never trusted on their own. The split is
recomputed from each record id under the lineage's pinned salt and rates and
compared against the recorded `lineage.split`; any drift refuses the entire
load with the offending record id named, so Stage 0 always consumes exactly
the projector's frozen split. Stage-2 RFT rebuilds replay tasks from the
records' task-identity git SHAs (`base_sha`/`head_sha`), which are validated
as full 40-hex SHAs before any rebuild.

A green `daydream train --corpus-v2` dry run here is the end of the pipeline:
the successfully frozen corpus-v2 bundle feeds Stage-0, SFT, and RFT directly,
with no manual conversion to the legacy JSONL schema.
