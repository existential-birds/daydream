## Summary

<!-- Explain the concrete problem, why the simplification matters, and observed behavior after the fix. Include a small diff sketch/diagram only when useful. -->

- Findings: FINDING_ID — [finding document](IMMUTABLE_DOCUMENT_URL)
- Implementation issues: ISSUE_URLS (use a closing reference only for fully resolved issues).
- Root cause and evidence: ...
- Intended simplification: ...
- Observed result and preserved contracts: ...
- Equivalent discoveries, alternatives, and exclusions: see the linked finding's matching guidance; note any new evidence here.

## Evidence

<!-- Include only commands and before/after observations actually obtained. Explain not-run/failed/blocked checks and any validation gaps. -->

| Command / observation | Result | Checked commit or exact snapshot |
| --- | --- | --- |
| ... | ... | ... |

Optional size changes: source ..., tests ..., docs ...; comparison endpoints and exclusions ... . Size reduction is supporting evidence, not a reward target.

## Benchmark provenance

Original observation: ... . Frozen benchmark: ... . Verified applicability and limitations: ... . Implementation began from ... . Target/head SHAs below are recorded at the stated time and may advance.

The linked findings and implementation issues, this PR, its fix diff, and later repository history contain reference answers. Evaluate discovery against an isolated frozen source snapshot that excludes those materials. Unknown or unverified provenance is not a benchmark-ready case.

<!-- improve-fix-metadata:start -->
```yaml
schema_version: 1
kind: improve_fix_pr
recorded_at: "REPLACE_WITH_UTC_ISO_TIMESTAMP"
repository: "REPLACE_WITH_CANONICAL_REPOSITORY_URL"
findings:
  - id: "REPLACE_WITH_FINDING_ID"
    issues: [] # Canonical URLs from resolution.issues; absent in older records means [].
    document_path: "improve_curated/REPLACE_WITH_FILENAME.md"
    document_commit: "REPLACE_WITH_PUSHED_FULL_SHA"
    document_url: "REPLACE_WITH_IMMUTABLE_BLOB_URL"
    observed: # Copy exactly from finding frontmatter.
      commit: "REPLACE_WITH_FULL_SHA"
      working_tree: clean # clean | dirty | unknown
      snapshot_ref: null
      dirty_paths: []
    benchmark: # Copy exactly; do not infer from the PR base.
      commit: null
      snapshot_ref: null
      verified: false
fix_start:
  commit: null # Full SHA when known; does not capture dirty changes alone.
  working_tree: unknown # clean | dirty | unknown
  snapshot_ref: null # Durable exact source snapshot identity, including checksum if external.
  dirty_paths: []
  provenance_notes: null
git:
  target_branch: "REPLACE_WITH_TARGET_BRANCH"
  base_commit: "REPLACE_WITH_RESOLVED_TARGET_FULL_SHA"
  merge_base_commit: "REPLACE_WITH_ACTUAL_MERGE_BASE_FULL_SHA"
  head_commit: "REPLACE_WITH_PUSHED_PR_HEAD_FULL_SHA"
implementation:
  commit: "REPLACE_WITH_IMPLEMENTATION_FULL_SHA"
  intended_simplification: "REPLACE_WITH_INTENT"
  observed_result: "REPLACE_WITH_OBSERVED_RESULT"
  preserved_contracts: []
validation: [] # Entries: {command, result, checked_commit}; use null + notes below if not a commit.
validation_notes: [] # Explain source snapshots, gaps, and later metadata-only commits as needed.
size_changes: null # Optional mapping: from, to, source/tests/docs {added, deleted}, exclusions.
limitations: []
```
<!-- improve-fix-metadata:end -->

## Merge Danger

**Door:** one-way or two-way, with explanation where necessary.

**Blast Radius:** affected behavior and any material risk, migration, rollback, or unresolved limitation.
