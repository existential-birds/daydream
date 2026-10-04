## Improvement

<!-- Concrete problem, source evidence, maintenance value, and the proposed simplification. -->

Finding: FINDING_ID
Record: `improve_curated/FILENAME.md`
Published record: IMMUTABLE_BLOB_URL_OR_EXPLICITLY_NOT_YET_PUSHED

## Implementation scope and acceptance

<!-- Preserve the finding's contracts and boundaries. Keep unresolved details as questions for the implementer. -->

- [ ] The identified unnecessary mechanism or duplication is removed or consolidated.
- [ ] FINDING_SPECIFIC_PRESERVED_BEHAVIOR_AND_CONTRACTS
- [ ] FINDING_SPECIFIC_VERIFICATION_AND_OBSERVABLE_OUTCOME
- [ ] Actual fix-start identity, implementation commits, validation commands/results,
      and deviations are recorded in the finding without replacing discovery evidence.

## For the future implementation session

Use `implement` on this issue. Read its comments and linked finding before editing.
If the file is absent, restore the exact embedded snapshot below after verifying its
SHA-256; its path must stay inside `improve_curated/`. Append this issue's URL to
`resolution.issues`. Preserve a newer existing record and reconcile discrepancies.

Revalidate the opportunity in current code. Record the actual pre-edit fix-start
commit and any exact dirty-state snapshot in the finding's Resolution and validation
section. The discovery/benchmark baseline remains unchanged even if main has moved.
Record implementation evidence and exact checks/results there as work completes.
Use ordinary hook-enabled commits; fix failures or report them, never bypass hooks.

The user will invoke `create-improve-pr` afterward to publish the draft fix PR and
link this issue. Keep this issue open while queued or under implementation.

## Discovery provenance

<!-- Summarize the original observed source identity, frozen baseline applicability,
     dirty-state limitations, and any missing source artifact. Do not claim that an
     embedded Markdown record or its checksum preserves the repository source. -->

This issue contains reference answers. Exclude it, the finding record, fix PRs, and
future history from the evaluated agent's discovery environment. The label
`improve-training` identifies the workflow; eligibility still requires verified
source provenance and human confirmation.

<!-- improve-issue-metadata:start -->
```yaml
schema_version: 1
kind: improve_issue
finding_id: "REPLACE_WITH_FINDING_ID"
repository: "REPLACE_WITH_CANONICAL_REPOSITORY_URL"
document_path: "improve_curated/REPLACE_WITH_FILENAME.md"
document_commit: null # Full pushed SHA only when it contains this exact record.
document_url: null # Verified commit-pinned blob URL, otherwise null.
snapshot_sha256: "REPLACE_WITH_SHA256_OF_EMBEDDED_FINDING_UTF8_BYTES"
```
<!-- improve-issue-metadata:end -->

## Finding snapshot at issue publication

<!-- Embed the complete finding, including YAML frontmatter and final newline,
     in a fence longer than any backtick run in the content. Hash that exact content.
     On retry, retain this snapshot and record later corrections separately. -->
