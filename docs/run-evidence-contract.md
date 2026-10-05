# Run evidence contract for the JSONL workflow

The first intended use of collected run evidence is a report showing review mistakes so the operator can improve Daydream. Open-weight review-model training remains a later use of the same evidence. This document inventories the current consumers for [issue #1467](https://github.com/existential-birds/daydream/issues/1467); it does not specify the report's presentation or implement record capture.

## Scope and settled decisions

- Support newly collected data only. The approved scope amendment in #1467 supersedes the historical migration requirements in [the parent issue](https://github.com/existential-birds/daydream/issues/1465). Existing historical data is not automatically deleted.
- Capture the existing frozen run snapshot at finalization and cooperative interruption. Evidence not captured before SIGKILL or power loss may be lost; local persistence still requires atomic, recoverable writes. See [ADR 0001](adr/0001-run-evidence-capture-boundary.md).
- A selected dataset snapshot retains exact run/observation membership, shard digests, and the temporal cutoffs used for selection. Later evidence belongs to another snapshot. See [ADR 0002](adr/0002-pin-dataset-snapshot-membership.md).
- Review improvement reports are the first downstream use. Preserving training and adjudication semantics remains required. See [ADR 0003](adr/0003-prioritize-review-improvement-reports.md).
- Dataset capture and publication are optional side effects. The prerequisite [issue #1466](https://github.com/existential-birds/daydream/issues/1466) isolates these failures from completed review outputs while retaining runtime artifact integrity protections.

## Existing schemas are not the new capture contract

[The current training-record schema](../daydream/training/schema/record-schema.json) describes projected per-finding examples, including outcome-finding, process-trace, and task-only records. It does not describe a complete captured run or the full observation history. [The adjudication observation validator](../daydream/training/adjudication/observations.py) describes per-finding judgments, rather than all run-level labels and later enrichment.

The new workflow needs distinct versioned run and observation schemas. Corpus projection remains a transformation of that evidence into training examples; capture must not require an example to satisfy training admission first.

## Evidence inventory

These are semantic requirements. Field names and nesting are implementation details, and unavailable evidence must have an explicit state rather than a fabricated value.

| Evidence family | Evidence to retain | Current consumers and sources |
|---|---|---|
| Task identity and input | Credential-safe repository identity and host, PR identity when known, original base/head revisions, original reviewed diff and digest, changed files, and evidence availability. The original input diff and the recommended fix patch have different meanings. | [Harvest inputs](../daydream/training/harvest_types.py), [task projection](../daydream/training/corpus_projection/projector.py), [RFT task reconstruction](../daydream/training/rft.py) |
| Root and child trajectories | Complete exposed trajectory documents, session/trajectory identities, parent-child references, step order, attempts, effective requests, responses, tools, partial states, and available timing/usage. Retain child registration order independently of physical record ordering. | [Trajectory recorder](../daydream/trajectory/recorder.py), [segment construction](../daydream/training/corpus_projection/segments.py), [process-trace projection](../daydream/training/corpus_projection/projector.py) |
| Review claims and their derivation | Structured per-stack claims and merged findings, finding text and location, existing host-assigned record/item identities, source UID links, fingerprints, and any terminal publication disposition. Preserve claims that verification or merging removed where the backend/artifacts expose them. | [Review identities](../daydream/deep/records.py), [finding projection](../daydream/training/corpus_projection/projector.py), [per-finding outcomes](../daydream/training/labeler_signals.py) |
| Verification and intrinsic scoring | Raw verifier verdicts and explanations, unverified assumptions where supplied, structured-format validity, final review text or its exact scoring length, and score policy/version/configuration when scores are persisted. Missing verdicts remain uncomputable, distinct from zero credit. | [Scoring-input assembly](../daydream/training/harvest.py), [reward computation](../daydream/training/reward.py), [verification schema](../daydream/phases/schemas.py), [reward calibration](../daydream/training/calibration.py) |
| Fix and later commit evidence | Recommended patch separately from original task diff, relevant revision identities, applied/total hunk counts, inspected commit window, local-commit verdict, and acquisition failures or unavailable states. | [Fix/outcome signals](../daydream/training/labeler_signals.py), [harvest reduction](../daydream/training/harvest.py) |
| GitHub outcome observations | PR state, merge time and author when known; finding/comment association; reply identity, author and association, creation time, qualification reason, classifier result and version, evidence digest, and acquisition time. Retain reviewer identities and the prior population inputs used by current reduction. | [GitHub signals](../daydream/training/labeler_signals.py), [annotation reduction](../daydream/training/harvest.py), [rubric semantics](../daydream/training/rubric.py) |
| Human judgments and corrections | Target run/finding identity, evidence and digest, disposition, author/labeler, role, rationale, policy/rubric version, valid time, observation time, and complete append-only history. Keep model suggestions distinguishable from decisive human judgments. | [Judgment history](../daydream/training/adjudication/observations.py), [precedence and reopening](../daydream/training/adjudication/precedence.py), [canonical annotation records](../daydream/training/adjudication/snapshot.py) |
| Provenance and admission | Backend/model and effective profile/configuration, producer version/provenance, available license evidence with source and acquisition time, and policy identities used for derived decisions. Unknown license evidence remains unknown and cannot imply training admission. | [Run identity](../daydream/run_snapshot.py), [profile provenance](../daydream/training/corpus_projection/provenance.py), [license admission](../daydream/training/corpus_projection/license.py), [training lineage](../daydream/training/lineage.py) |
| Execution and collection state | Flow identity, execution outcome, evidence completeness/availability, selected snapshot cutoff, optional trace identity, and sanitized collection diagnostics. A successful review does not imply successful capture or upload. | [Run finalization](../daydream/run_artifacts.py), [trajectory snapshots](../daydream/trajectory/recorder.py), [observability lifecycle](observability.md) |

## Concrete gaps and traps

### Correction text is not currently retained by reply evidence

[`_reply_evidence`](../daydream/training/labeler_signals.py) currently retains reply metadata, a body hash, and the classifier label, but not the reply body. A hash cannot explain a correction or allow a future analysis to examine its reasoning. New GitHub outcome observations should retain the available redacted reply text alongside its source identity and content digest, subject to the same privacy checks as other evidence. This belongs to later outcome acquisition; run finalization cannot capture a reply that does not yet exist.

Adding text to evidence must not silently change current classifier, evidence-digest, or deduplication semantics. Version the enriched representation and preserve the inputs the current reducers use. Record missing or withheld text explicitly.

The existing `body_sha256` hashes the original reply body; it does not verify a redacted copy. Retain separately identified redacted captured content, its digest, and redaction provenance. [`reply_evidence_digest`](../daydream/training/labeler_versions.py) hashes every canonical evidence field, so adding text to the current evidence dictionary would change reopening/deduplication behavior even for unchanged source text. Keep the current semantic evidence projection separate from captured text, or explicitly version a replacement projection.

The schema and local round-trip support belong to #1467. Live GitHub reply acquisition and unchanged-evidence harvest behavior belong to [#1469](https://github.com/existential-birds/daydream/issues/1469); human correction commands and resolution/export belong to [#1470](https://github.com/existential-birds/daydream/issues/1470).

### Identity has several meanings

[`deep.records`](../daydream/deep/records.py) assigns particular review claims `uid` values, merged claims `item_uid` values, and `source_uids` provenance. A finding fingerprint matches similar defects; it is not interchangeable with those identities. The [training identity](../daydream/training/corpus_projection/identity.py) additionally depends on session, trajectory, segment, and fingerprint.

Preserve these inputs and their associations before choosing the final field layout. A storage change must not silently detach judgments, renumber segments, or change the current projected record identity and deterministic splits. No new finding identity scheme has been selected in this discussion.

### Missing evidence is not a negative outcome

An unanswered comment, a missing comment, ambiguous feedback, and an explicitly rejected finding are different states. PR merge status does not supply a finding label. A local-branch rejection indicates that the recommended change did not land in the inspected commits; it does not establish that the original bug claim was false. A verifier contradiction or uncertainty retains its evidence and source rather than becoming an unsupported factual mistake label.

Missed-bug analysis needs independent reference evidence. A run with no findings does not establish that the reviewed change contained no bugs. See the current [reference-based recall calculation](../daydream/eval/latency_report.py).

### Finalization cannot reconstruct the original task after fixes

The [deep orchestrator](../daydream/deep/orchestrator.py) already saves the original diff before the review/fix flow. Capture must serialize that retained input. Re-reading the final working tree or current HEAD can describe Daydream's changes instead of the change it reviewed. The schema needs distinct original-input and recommended-patch evidence with their revision identities and digests.

The [deep orchestrator](../daydream/deep/orchestrator.py) already constructs `AnalyzedRevision` for built-in deep and shallow reviews, independently of whether a findings export is requested. The [current archive finalizer](../daydream/archive/finalize.py) separately acquires Git context at finalization. Serialize the producer's analyzed revision alongside the original diff; do not replace it with final Git state or add a duplicate capture lifecycle.

For dirty local reviews, the diff can include tracked worktree changes in addition to committed changes. Base/head revisions alone do not describe that complete input. Preserve the full original diff and its available input-scope/dirty-state provenance. [Workspace anchors](../daydream/workspace.py) and the [diff resolver's base](../daydream/git_ops/queries.py) can differ, so use the identity of the task actually analyzed. Runs that fail before task acquisition retain explicit unavailable evidence; later base enrichment stays an observation.

Retain the analyzed task identity when input acquisition finishes. Cooperative cancellation can skip normal [review terminal finalization](../daydream/deep/review_terminal.py), so cancellation capture must not depend exclusively on terminal coverage being written. Failure before recorder startup can leave no trajectory document; represent unproduced evidence honestly for supported captured runs, without inventing an empty successful review. Invalid CLI invocations before a run exists do not require fabricated run records.

### Findings capture must be independent of optional output files

The portable [findings envelope](../daydream/findings.py) is an optional routed output. It carries fingerprints and terminal review completeness, but the host-assigned item/source identity graph lives in the deep artifacts. Assemble raw run evidence from the frozen sources rather than requiring an optional `findings.json` output. Preserve [fix outcomes](../daydream/deep/fix_steps.py) by item identity and represent legitimately absent patches after declined or failed fixes.

Raw merged items do not necessarily contain a stored fingerprint. Associate the existing fingerprint calculation from [PR review projection](../daydream/pr_review.py) with the canonical finding while retaining its item/source UIDs. Numeric verifier issue IDs, verifier selection decisions, durable merged item IDs, and fix-outcome item IDs must keep their existing associations; an optional portable export does not carry the entire graph.

Preserve exposed removal/derivation evidence from arbitration, suppression, deduplication, and structural folding. [Dropped speculative findings](../daydream/phases/findings.py) retain full items and source associations even when final stack records no longer contain them. Serialize the relevant evidence from these frozen sources rather than copying entire artifact directories or fabricating the history of unrecorded claims.

### Attempt and tool order already have tracing-independent sources

The [agent runner](../daydream/agent.py) records separate trajectory invocations for retry attempts. [Invocation summaries](../daydream/trajectory/invocation.py) retain invocation identity and step membership, and step identities preserve tool/content order independently of tracing. Serialize that evidence directly; this satisfies #1467's requirement to preserve recorded attempt/tool ordering.

Explicit logical retry-group/ordinal metadata is currently present in the [tracing attempt scope](../daydream/observability/spans.py), but #1467 does not require introducing a new retry-group identity scheme. Do not infer such groups from adjacent steps or depend on trace readback. Keep that additional producer instrumentation outside this slice unless an actual required consumer demonstrates the need.

Frozen document arrays are root-first and sorted by child identity. That is serialization order, not execution order. Preserve step, invocation, fork/dispatch, and child registration identities and order from the documents.

### Temporal selection and precedence remain part of reading

Pin exact observation membership and preserve both when evidence became valid and when Daydream acquired it. Apply the current [temporal leakage rules](../daydream/training/corpus.py) and [human precedence](../daydream/training/adjudication/precedence.py) to the selected history. New observations cannot change an earlier dataset snapshot, and a later model suggestion cannot erase a human correction.

### JSONL does not provide store guarantees

The public record store must provide serialized mutations, complete validated records, atomic/recoverable persistence, identity/content deduplication, conflict diagnostics, and immutable snapshot reads. [The current observation appender](../daydream/training/adjudication/observations.py) appends and flushes text; that alone does not satisfy the new durability contract.

Reuse shared redaction, credential-safe repository identity, and blocking secret-scan rules against the new serialized payload. [Existing privacy helpers](../daydream/archive/sanitize.py) contain useful behavior, but their directory interface must not force a hidden archive reconstruction. Use private local storage, withhold rejected data, diagnose unsupported oversized records without truncating them, and keep dataset failures isolated from review publication.

## Acceptance journey for #1467

Exercise the public run entry point with a real local filesystem and an offline backend/network substitute:

1. Execute a new run with capture enabled and tracing disabled. Retain original task evidence before fixes, then capture the selected frozen run snapshot.
2. Read the validated run through the public dataset entry point without an archive bundle or trace readback. Check original task identity, structured claims, verification/scoring inputs, trajectory relationships/order, and completeness.
3. Append a typed run or finding observation, select an explicit dataset snapshot, and read the observation through the same public store. A later observation appears only in a newly selected snapshot.
4. Repeat an identical commit/append and verify that it does not duplicate evidence. Conflicting content for an immutable identity produces a diagnostic.
5. Exercise cooperative interruption, absent evidence, malformed/unknown schemas, interrupted persistence, oversized records, privacy rejection, and capture opt-out. Incomplete persistence is never reported as published; dataset failures preserve the established review outcome and completed output protections.

Existing consumers are the evidence inventory for this slice; switching all harvest/corpus readers to the new records belongs to the later cutover. HF publication and the review improvement report are downstream work, while the records must retain their required inputs now.

## Suggested clarifications for issue #1467

These additions make the existing acceptance criteria testable without adding report design or a new retry protocol:

- [ ] Serialize the producer's original analyzed repository/base/head identity and full input diff/digest, with committed-versus-tracked-worktree provenance where available. Retain this evidence immediately after acquisition so cooperative interruption preserves it. Finalization-time Git state remains separate; an intentionally empty input and unavailable/unproduced input are distinct.
- [ ] Capture canonical claims and existing record/item/source identity links independently of `findings_out`. Use the current finding fingerprint calculation and retain associations to verifier selection/verdicts and item-keyed fix outcomes. Preserve available held/removed claim evidence and explicit completeness without inventing claims that were never recorded.
- [ ] Preserve recorded root/child references, fork registration order, invocation IDs and step membership, and tool/content ordering from frozen trajectory documents. Demonstrate capture with tracing disabled and a retry; logical retry-group/ordinal instrumentation is not an additional requirement.
- [ ] Give the common observation schema distinct run-label, finding-judgment, and PR/base/license-enrichment payloads with explicit targets, source/policy versions, valid/observed times, and evidence digests. Support redacted correction text with its own captured-content digest and provenance, separately from current source-body hashes and semantic classifier/dedup evidence.
- [ ] Through the public local store, verify immutable snapshot membership and both observation-time and evidence-valid-time selection; round-trip typed evidence and preserve human precedence and model-suggestion review requirements. New observations cannot alter an earlier snapshot.
- [ ] Exercise serialized writers, interrupted record commits, identical retry no-ops, same-identity content conflicts, oversized/malformed/unknown-schema records, and privacy refusals through observable public behavior. Persistence failures preserve the existing review/output dispositions. Incremental capture for SIGKILL/power-loss recovery is outside scope.

## Existing verification seams to extend

| Behavior | Existing test seam | New observable check |
|---|---|---|
| Original versus recommended change | [Run data capture](../tests/test_archive_data_capture.py), [review completion](../tests/deep_orchestrator/test_review_completion.py), [Git input resolution](../tests/test_git_ops.py) | A real flow commits a fix after acquiring its input; the public record still has the original producer-resolved base/head/diff, distinct from final revision/recommended patch. A dirty input preserves its exact captured scope. |
| Cooperative interruption and ephemeral source identity | [Runner lifecycle](../tests/test_runner.py) | Captured evidence survives supported cancellation and ephemeral workspace teardown; unavailable evidence remains explicit and source identity does not become the temporary model cwd. |
| Claims without optional export and frozen trajectory order | [Findings artifact journey](../tests/deep_orchestrator/test_findings_artifacts.py), [retained claim identities](../tests/deep_orchestrator/test_record_identity_and_retained_tree.py), [agent retries](../tests/test_agent_retry.py), [trajectory serialization](../tests/test_trajectory.py) | Capture a run with no findings export and tracing disabled; its public record retains claims, verifier/fix associations, root/child provenance, ordered invocations and tool evidence, including the incomplete first attempt in a fail-then-succeed retry. |
| Observation identity, precedence, and correction content | [Observation history](../tests/test_training_adjudication_observations.py), [human precedence](../tests/test_training_adjudication_precedence.py), [snapshot identity](../tests/test_training_adjudication_snapshot.py), [reply evidence](../tests/test_training_labeler_signals.py) | The new local store round-trips run labels, judgments and enrichment; identical retries are no-ops, later model suggestions preserve human precedence, and redacted text has a correctly identified content digest. |
| Temporal selection and frozen membership | [Corpus leakage](../tests/test_corpus_leakage.py), [projection reproducibility](../tests/test_corpus_projection_reproducibility.py) | A saved snapshot reads the same run/observation membership after append; valid-time and observation-time checks retain their distinct meanings. |

Use the public run/store entry points with real local files and an offline backend/network substitute. Existing helper-level fixtures can isolate malformed evidence, but they do not replace the primary end-to-end capture/read/append journey.
