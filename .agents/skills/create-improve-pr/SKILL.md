---
name: create-improve-pr
description: "Implement or finish a curated improvement and publish a focused draft PR with reproducible benchmark provenance and bidirectional links to improve_curated findings. Use when asked to create an improve fix PR."
---

# Create an Improve fix PR

Turn selected `improve_curated/` findings into a focused, reviewable **draft PR by default**. Preserve enough evidence to later evaluate blind discovery of the original improvement. This skill handles implementation and publication when invoked for a finding; it does not authorize merging.

## Inputs and authority

Read the repository instructions, `improve_curated/AGENTS.md`, the selected finding documents, and relevant check/PR conventions. If `.agents/skills/pr/SKILL.md` is present, follow its concise Summary, Evidence, and Merge Danger conventions, augmented with the benchmark information below. Otherwise use this directory's self-contained `PR_TEMPLATE.md`; the optional local `pr` skill is not required on a fresh clone.

- Use findings identified by the user or unambiguously selected in the active session. Ask for scope only when multiple unrelated choices remain; do not manufacture a finding to fill the template.
- If no document exists, use `../publish-improve-finding/SKILL.md` to capture the active session's finding **before implementation**. Read its template for the canonical finding schema.
- Invocation authorizes the scoped fix, appropriate validation, ordinary commits/pushes, and draft PR creation. Preserve unrelated user work. Never force-push, merge, mark ready, or expand into unrelated refactors without authorization.
- Group findings only when one coherent change resolves them. Keep independent opportunities in separate PRs. A fix PR is evidence for a discovery benchmark; its diff is not the discovery task input.

## 1. Capture provenance before changing code

Inspect the Git status, current branch, full `HEAD` SHA, remotes, and any existing PR for that branch. Check for ongoing work by other agents. Use an isolated worktree/branch when necessary; do not reset, discard, or indiscriminately stage changes. Do not move another session's uncommitted work into your PR.

When resuming an existing PR or interrupted publication, read its existing metadata block, linked finding records, and verified branch history **before recording provenance or editing source**. Preserve the original `fix_start`, finding observation, and frozen benchmark. Do not recapture the current HEAD as the fix start merely because this invocation began there. Retain the original implementation commit and validation evidence on documentation-only/publication retries. If existing provenance is missing or contradictory, recover it from evidence and explain the correction; leave unknowns explicit rather than replacing them with current state.

Keep these identities separate:

| Identity | Meaning |
| --- | --- |
| Finding `observed` | Original state in which the opportunity was discovered; retain this evidence unchanged. |
| Finding `benchmark` | Frozen target in which this finding was independently verified to exist; may remain unknown/unverified. |
| PR `fix_start` | Exact source state immediately before this implementation started. |
| PR `git.base_commit` | Resolved target-branch SHA at the stated recording time. |
| PR `git.merge_base_commit` | Actual merge base of the fix branch and recorded target commit. |
| PR `implementation.commit` | Commit containing the completed implementation covered by source validation. |
| PR `git.head_commit` | Pushed PR head when the description was refreshed, including later documentation-only commits. |

Record full SHAs. Never infer the original finding or benchmark snapshot from the current PR merge base. Never silently replace a frozen benchmark because main advanced.

For a clean fix start, record `fix_start.commit` and `working_tree: clean`. A dirty starting tree is **not captured by HEAD**. Identify which changes belong to the finding, preserve the exact applicable state with a durable, retrievable snapshot plus checksum or an explicitly scoped ordinary checkpoint commit, and record its reference and dirty paths. Do not archive secrets or unrelated private files. If an exact state cannot be preserved safely, record it as unknown/incomplete and explain the limitation; do not claim reproducibility or benchmark eligibility. Prefer an isolated clean start when the fix can be reproduced there.

If implementation already occurred, recover the genuine pre-fix state from verified history or an existing snapshot. Reproduce the opportunity there where feasible. Do not invent a before-state, backdate observations, or imply an after-the-fact discovery was recorded beforehand.

Check the selected finding's `benchmark.verified` evidence. Preserve an existing valid baseline; if its applicability has not been verified, keep it unverified. This need not block a useful fix PR, but the description must state the benchmark limitation.

## 2. Implement and validate

Resolve the documented root cause with the smallest coherent simplification. Preserve the contracts listed in the finding. Record deviations from the proposed fix and why they were necessary. Distinguish intended simplification from the observed outcome.

Run relevant checks required by repository instructions and appropriate targeted validation. Record exact commands, outcomes, and the source state checked. Include any actual before/after evidence; do not claim a baseline failure or passing check you did not observe. Distinguish failed, blocked, and not-run checks and explain limitations.

When useful, measure additions/deletions against the exact fix start and report source, tests, and documentation separately. Identify exclusions such as generated files and lockfiles. Net LOC reduction is supporting evidence, never the optimization target or proposed training reward.

Stage only the intended changes. Commit and push through normal repository hooks. **Never use `--no-verify`, hook-disabling environment variables, or another bypass.** If a hook fails, fix it on the current branch and retry the ordinary command, or stop and report the blocker. An unrelated or pre-existing failure is not an exception. Record any consequential hook changes and rerun validation if they change the source after checks.

Record the implementation commit once the fix and required checks are complete. Checks run on a working tree may be attributed to that commit only after verifying its source tree matches what was checked; otherwise use `checked_commit: null` with an exact snapshot reference and explanation. Do not claim a check covered later runtime edits.

## 3. Draft the PR description

Use `PR_TEMPLATE.md` in this directory. Replace placeholders with observed facts and use YAML `null`/empty lists for unknown or inapplicable values. Keep `schema_version: 1` and `kind: improve_fix_pr` stable. Use one metadata block delimited by the template's HTML comments.

The human-readable description must explain:

- Finding IDs and immutable links to their Markdown documents; the concrete evidence, root cause, and why the change matters.
- Original observation and benchmark identities, with verification and any dirty/incomplete-snapshot limitations.
- Intended simplification, observed result, preserved behavior/contracts, and any remaining scope.
- Discovery matching notes: reference the finding's alternatives, equivalent discoveries, and exclusions. A later evaluator must match the underlying opportunity, not exact prose or this implementation.
- Exact validation commands/results and checked state, plus material risks and limitations. Preserve the repository's concise Summary, Evidence, and Merge Danger sections.

Copy each finding's `observed` and `benchmark` values exactly into its metadata entry; do not improve uncertain provenance by inference. `status: confirmed` requires the human confirmation recorded by the publishing skill. Do not label your own findings human-confirmed.

Use GitHub blob URLs pinned to a **pushed full commit SHA** for finding documents, such as `https://github.com/OWNER/REPO/blob/FULL_SHA/improve_curated/FILE.md`. Verify the document exists at that commit. A branch URL may be added for convenience but cannot replace the immutable link. Preserve the repository's actual owner/name; do not assume this skill always runs in Daydream.

## 4. Publish and persist both directions

Use the authenticated repository and explicit target branch. Inspect existing PRs for the selected head branch and finding IDs/URLs before creating anything. Resume the matching existing PR when appropriate, including after an interrupted earlier invocation. Report a closed/merged PR or an ambiguous match instead of creating a duplicate silently. Do not convert an existing ready PR to draft without authorization.

1. Commit/push the implementation and selected finding documents through ordinary hooks. Capture the pushed head. Write the exact initial PR body to a temporary UTF-8 file using a file-writing API or a quoted heredoc. Use `gh pr create --draft --base TARGET --head BRANCH --title TITLE --body-file PATH`, with safely quoted arguments. For a matching existing PR, use `gh pr edit NUMBER --body-file PATH`. Never interpolate untrusted Markdown into shell command text or use command substitution to construct a body.
2. Obtain the canonical PR URL and number. Add its URL once to each finding's `resolution.prs`. Add the implementation SHA once to `resolution.fix_commits`; append truthful `{command, result, checked_commit}` entries to `resolution.validation`. Keep `resolution.status: in_progress` while the PR remains open; change it to `fixed` only when the resolution is actually complete under the collection's instructions. Preserve original observation, benchmark identity, human provenance, and discovery criteria.
3. Commit/push the finding backlink update with ordinary hooks. These changes belong on the PR branch so the backlink is already remotely available before merge. Verify the update contains only the intended metadata/documentation changes. If runtime code changed, rerun affected checks and refresh the implementation/validation records.
4. Refresh the **remote PR body** to pin finding document links and `git.head_commit` to the resulting pushed commit. Keep `implementation.commit` and validation's checked state accurate. Record later docs-only checks separately where required; do not imply source checks ran against a later commit when they did not. Updating the PR description requires no new Git commit.
5. Read back the PR state/body and each finding at its pinned remote commit. Verify draft state for new PRs, correct target/head, both link directions, and all recorded SHAs/URLs. Report any partial publication or blocker explicitly.

Avoid a self-reference loop: finding documents store the stable PR URL and the prior implementation commit, **not their own enclosing commit SHA** or final PR head. Only the external PR body records the final head/doc commit. Do not repeatedly commit simply to update a hash embedded in that same commit.

If creation succeeds but a subsequent backlink commit/push fails, leave the draft PR intact, report its URL and missing backlink, and retry idempotently when the blocker is resolved. Do not claim the workflow complete until the reciprocal link is remotely persisted and the body refreshed.

On a documentation-only retry, perform only missing publication/backlink/body steps; reuse the established implementation commit and source-validation results. Append validation idempotently: do not duplicate an existing entry with the same command, result, and checked commit/source identity, and do not erase earlier failures when recording a later pass. If new runtime/source modifications are needed, retain the original `fix_start` and frozen benchmark, record the additional implementation commit in the finding's `resolution.fix_commits`, and rerun affected checks. Update the PR's `implementation.commit` to the newly validated implementation state while retaining the earlier commit and its role in the human-readable history/validation notes. Neither kind of retry establishes a new original baseline.

## Completion report

Return the PR URL, linked finding IDs, concise change/validation results, and any provenance or publication limitations. Do not merge. Remind future benchmark builders through the PR metadata/template that finding documents, fix diffs, PR descriptions, and future history are reference answers: exclude them from the evaluated agent's target snapshot.
