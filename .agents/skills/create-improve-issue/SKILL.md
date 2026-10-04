---
name: create-improve-issue
description: Create a GitHub implementation issue from a confirmed improve_curated finding, label it improve-training, and preserve the context and provenance needed by a later implementation session.
---

# Create an Improve implementation issue

Queue a captured, human-confirmed improvement for a later session. The workflow is
`publish-improve-finding` → this skill → `implement` on the issue → `create-improve-pr`.
Issue publication includes the `improve-training` label and a local backlink; it
does not implement the fix, commit/push files, open a PR, or start training. Follow
separately authorized actions without asking for the same permission again.

Read `improve_curated/AGENTS.md`, the selected finding, and
[ISSUE_TEMPLATE.md](ISSUE_TEMPLATE.md). Resolve the finding from the user's path/ID
or the active session; ask only when the selection is ambiguous. Do not publish
every file in the directory just because several findings exist.

## Prepare the handoff

- Require `status: confirmed` and actual `origin.human_confirmation` evidence. An
  explicit confirmation already in the session can be recorded; never ask for it
  again or infer it merely from a request to save a candidate. For an unconfirmed
  or rejected record, resolve that judgment with the user before opening a fix issue.
- If capture has not happened, use the sibling `publish-improve-finding` skill
  first. Preserve the finding's stable ID, filename, observed source, benchmark,
  matching criteria, and human provenance. Unknown benchmark identity may remain
  unknown; a useful implementation issue need not yet be benchmark-eligible.
- Make the issue useful to a fresh `implement` session: explain the concrete
  problem, evidence, maintenance value, proposed simplification, preserved contracts,
  acceptance criteria, and relevant checks. Keep uncertain implementation details
  as investigation questions; do not invent a completed design or validation result.
- Include the full finding as a frozen publication-time snapshot, plus its path,
  stable ID, and SHA-256 of the exact UTF-8 bytes embedded. Retain its final newline
  and use a Markdown fence longer than any backtick run inside the record. This
  gives a later session the evidence even if the local file is uncommitted.
- If the exact record already exists at a pushed commit, also include its verified
  commit-pinned blob URL and document commit. Otherwise leave those fields `null`;
  never link to a nonexistent blob or substitute `observed.commit`, which identifies
  source code rather than the finding document. Do not push merely to obtain a link.
- Check the material being published for credentials and unrelated private session
  content. Preserve relevant evidence without exposing those values. An issue body
  or its checksum is not an archive of the repository's source state.

## Publish once and link back

1. Resolve the GitHub repository from the finding and local remote, checking any
   mismatch before publication. Inspect `resolution.issues` (absent means `[]`) and
   search issues in **all states** by the exact finding ID. Read candidate bodies
   to verify identity. Reuse a matching open issue; do not create duplicates after
   a partial earlier run. A closed match or conflicting matches need explicit
   resolution, not silent reopening or a replacement issue.
2. Ensure the exact label `improve-training` exists. Reuse it unchanged if present;
   otherwise create it with color `6F42C1` and description
   `Human-confirmed improvement findings for discovery benchmarks and training`.
   A label-list search may be fuzzy: compare the returned name exactly. If a
   concurrent create reports that it already exists, read it back. Stop on other
   authorization/publication errors rather than repeatedly retrying mutations.
3. Prepare the complete issue body in a temporary UTF-8 file. Use `gh issue create
   --repo OWNER/REPO --title TITLE --label improve-training --body-file PATH` with
   safely quoted arguments or equivalent structured API input. Honor explicitly
   requested assignees or project placement; do not infer an epic or bulk assignment.
   When reusing an issue, preserve its original finding snapshot and discussion;
   add missing handoff context or dated corrections without erasing others' work.
   Ensure the label is applied to the reused issue too.
4. Read back the canonical issue URL and append it once to the local record's
   `resolution.issues`, adding that list to older records. Leave `resolution.status`
   as `open` while merely queued. Do not change confirmation or baseline fields.
   Add a dated amendment if needed. This backlink is a local edit until separately
   committed/pushed; report that fact accurately.
5. Verify the remote issue body, exact label, finding ID, snapshot checksum, and
   local backlink. The embedded snapshot intentionally predates the new backlink;
   keep its hash accurate rather than regenerating it to include its own issue URL.
   On an ambiguous create timeout, search/read back by ID before another attempt.
   If the issue exists but the backlink fails, report its URL and repair the local
   record on retry rather than opening another issue.

## Handoff requirements

The issue must tell the future implementation agent to:

- Read the full issue and comments; locate the named finding or restore the embedded
  snapshot after checking its hash. Validate the recorded path is a Markdown file
  inside `improve_curated/`; never overwrite a newer local record. Append the current
  issue URL to the restored record's `resolution.issues`.
- Inspect current code and revalidate the opportunity because it may have drifted.
  Preserve the original observation/benchmark and record the actual fix-start SHA
  plus exact dirty-state identity if needed **before editing**. Keep that evidence
  in the finding's Resolution and validation section for `create-improve-pr`.
- Implement the agreed scope, record exact checks/results and implementation commits,
  and preserve all issue links. Normal repository hooks apply to any authorized
  commits/pushes; never bypass a failing hook. Do not automatically close the issue
  or publish a PR as a side effect of this issue-creation skill.
- Use `create-improve-pr` when the user invokes it to publish the completed fix with
  finding links, benchmark metadata, and an appropriate issue-closing reference.

Return the issue URL, selected finding ID/path, and any provenance gaps or incomplete
backlink work. Issue bodies and embedded findings are hidden reference answers for
discovery evaluation; the `improve-training` label does not certify benchmark eligibility.
