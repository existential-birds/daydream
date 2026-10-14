---
# Canonical schema for new findings; preserve existing extensions when updating.
# Scalars are strings unless noted; unknown optional scalar values are YAML null.
# All [] fields are lists of strings, except validation (entry shape below).
schema_version: 1 # Integer schema version; retain 1.
id: "REPLACE_WITH_STABLE_ID" # improve-YYYYMMDD-<8 lowercase hex>; stable and unique.
title: "REPLACE_WITH_CONCRETE_OPPORTUNITY" # Nonempty string.
repository: null
discovered_at: null # Actual ISO date/timestamp when known; never invent one.
origin:
  kind: human # human | agent | mixed; discoverer, not document author.
  session_ref: null
  human_confirmation: null # Or {by, at, evidence}; by/evidence required, at may be null.
status: proposed # proposed | confirmed | rejected; confirmed requires human evidence.
observed:
  commit: null # Full verified commit SHA, or null; dirty state needs separate identity.
  working_tree: unknown # clean | dirty | unknown
  snapshot_ref: null # Durable exact snapshot reference with checksum/retrieval details.
  dirty_paths: []
benchmark:
  commit: null # Full verified commit SHA, or null; dirty state needs separate identity.
  snapshot_ref: null # Durable exact snapshot reference with checksum/retrieval details.
  verified: false # Boolean; true only after checking this exact immutable baseline.
affected_paths: []
affected_symbols: []
related_findings: []
resolution:
  status: open # open | in_progress | fixed; an open fix PR stays in_progress.
  issues: [] # Canonical issue URLs; absent in older records means [].
  prs: [] # Canonical PR URLs.
  fix_commits: [] # Full verified commit SHAs.
  validation: [] # {command, result, checked_commit}; command/result strings, SHA or null.
  # Explain null checked_commit with exact checked snapshot identity or an explicit gap
  # in Resolution and validation. Never attribute checks to untested source states.
---

# REPLACE_WITH_CONCRETE_OPPORTUNITY

## Evidence and root cause

Describe the unnecessary mechanism or duplicated responsibility and the relationship
between affected symbols. Cite paths/symbols and, when useful, commit-pinned lines or
small code excerpts. State which observations are verified and which are hypotheses.

## Maintenance value

Explain what becomes easier to understand, change, or validate. Distinguish a
substantial simplification from cosmetic deletion or unsupported feature removal.

## Proposed simplification and preserved contracts

Describe what can disappear or be consolidated, any existing implementation to reuse,
and the observable behavior, interfaces, persisted data, or failure handling to retain.
State open questions and conditions that would invalidate the proposal.

## Discovery trail and human judgment

Record actual searches, comparisons, relevant negative results, and user corrections
available in the active session. Include human confirmation/rejection evidence and
confidence with its basis; do not invent a thought process or a numerical certainty.

## Discovery matching criteria

- Essential opportunity and concrete evidence a successful discovery must identify:
- Acceptable alternative descriptions or simplifications:
- Insufficient matches, invalid proposals, and behavior-breaking alternatives:
- Scope/overlap with related findings (one root cause receives one match):

## Source and benchmark provenance

Explain the observed pre-fix state, dirty-file relevance, and any gaps in reproducing
it. If a benchmark is verified, state how the opportunity was checked in that exact
snapshot. An immutable archive/manifest reference should include its checksum and
retrieval location. Record baseline eligibility as unresolved when evidence is missing.

## Resolution and validation

Not yet implemented. Later record fix PRs, what actually changed, checks and their
exact tested source identity, failures/limitations, and merge/fix commit evidence.
Do not replace discovery-time evidence with the final solution.

## Savings

- Estimated at discovery: Not yet measured.
- Observed after implementation: Not yet measured.

If measured, separate production code, tests/support, and documentation; state the
comparison commits and additions/deletions. Describe removed concepts alongside LOC.

## Amendments

Append dated corrections, judgment changes, and provenance updates here.
