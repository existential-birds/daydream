# Response Format

Account for every source item in a concise table or prose. Preserve identifiers, URLs, source snapshot, current verification target, evidence, and results in the handoff even when the human summary is shorter.

Identify the selected PR's base/head repositories and branches, acquired remote
head, and exact verification worktree/snapshot. When checking an ahead or dirty
local candidate, record its ancestry and owned patch separately from the remote
PR tree. Label evidence and outcomes by the tree checked; a local candidate fix
remains local until its publication on the selected PR is confirmed.

## Verification summary

| ID / source | Assessment | Evidence | Next action |
|-------------|------------|----------|-------------|
| Original comment ID or local label | VALID / INVALID / UNCERTAIN / NEEDS CLARIFICATION | Current file:line or command result with relevant conditions | Authorized fix, evidenced rejection, missing evidence, or specific clarification |
| Context-only source ID | Context; no substantive finding | Boilerplate, prior fix reply, or supporting discussion | Retain as evidence; no fix or reply |

Include concrete proposed changes and work chunks with their finding IDs and file ownership. Follow [Fix-launch decision](SKILL.md#fix-launch-decision) after verification. Preserve explicit triage-only scope and existing implementation authorization. An explicit user subset controls the fix set; identify omitted items as user-excluded. Record overrides separately from assessments, including the requested change and approving instruction.

## Final result and handoff

| ID / source | Disposition | Change or reason | Evidence / validation |
|-------------|-------------|------------------|-----------------------|
| Preserved identifier and comment URL | Implemented and verified | Cause corrected at current file:line | Relevant test command and passing result |
| Preserved identifier and comment URL | Invalid; no override requested | Claim disproved for stated conditions | Current code/command evidence |
| Preserved identifier and comment URL | User override implemented and verified | Approved change; retain original assessment | Override instruction and passing validation |
| Preserved identifier and comment URL | Changed; validation incomplete | Patch applied; validation could not complete | Command failure and remaining work |
| Preserved identifier and comment URL | Blocked / uncertain / needs clarification | Specific impediment or missing evidence | Attempted checks and required input |
| Preserved identifier and comment URL | User-excluded / outside authorized scope | Exclusion or scope boundary | Relevant user instruction |

Omit unused categories. A one-item result can be a short evidenced sentence. Never omit unresolved items or present them as implemented/rejected just to fit a binary table.

Report combined-tree validation once, with actual commands/results and unavailable or skipped checks. Mention residual risks or blockers. Review worker diffs and evidence before claiming a fix. Do not imply tests, commits, pushes, replies, or thread resolution happened unless verified.

After verified fixes, follow [Push and response handoff](SKILL.md#push-and-response-handoff). Include the publication target, commit scope, and published SHA/link as each becomes available. Proceed with authorized publication and response actions; include a decision question only when existing authorization does not cover that action. Record declined or pending decisions and failed publication accurately.

For `respond-pr-feedback`, retain original REST/node comment IDs, review IDs, thread IDs, URLs, reply parents, original locations, and source head SHA, selected base/head repository and branch identities, exact target worktree path, and remote/candidate snapshot identities. Include assessment, implementation disposition, evidence, tested current tree/commit, open issues, and whether the patch is only local, committed, or present on the PR's remote head. This skill does not post replies or resolve threads; those actions need separate authorization.
