---
name: prepare-implement
description: Gather GitHub issue or spec context with subagents and produce a self-contained goal prompt for a fresh implementation session.
---

# Prepare Implement

Prepare the requested issue for a fresh session, then return only one copyable
`/goal` prompt. The preparing agent gathers context read-only; the new session
creates its branch, implements the issue, and delivers a merge-ready PR. Create
additional artifacts only when the user requests them.

## Establish the source and baseline

- Read the complete issue/spec and comments, repository `AGENTS.md` and its
  referenced guidance. Resolve linked prerequisites and scope amendments;
  distinguish superseded requirements from current acceptance criteria.
- Inspect the repository path, remote, current branch, exact HEAD, and worktree
  status. Record unrelated edits, including untracked files, for the handoff.
  Record the user's requested base or, otherwise, the current checkout's HEAD.
  Resolve its exact SHA from available local refs; identify remote-tracking refs
  as locally observed rather than freshly fetched. If the base is unavailable,
  carry its reference and that limitation into the prompt for the new session.
- Leave the checkout, refs, and files unchanged. Branch creation and any fetch
  needed to establish the requested base belong in the generated prompt.

## Fan out context gathering

Use subagents for distinct, bounded, read-only investigations, split by the
issue's actual seams: runtime/contracts, consumers/architecture, tests/fixtures.
Give each agent the source reference and repository guidance. Request concrete
paths and symbols, preserved invariants, dependencies, public test seams,
fixture regeneration commands, verified check commands, and risks. Preparation
agents leave file edits and implementation to the next session.

Collect every agent's findings and reconcile shared contracts before writing the
prompt. Resolve conflicting findings with focused reads. Propose disjoint work
lanes and one owner for shared schemas, test harnesses, or generated fixtures.
Distinguish checks actually run from checks merely identified; context gathering
does not establish passing tests. Ask for missing scope decisions only when they
materially affect the goal; carry unresolved questions into the handoff.

## Generate the goal prompt

Return a single fenced text block beginning with `/goal Deliver a merge-ready PR
for ...`, using the actual source and desired behavior. Use another outcome only
when the user requests it. Include all of the following inside the prompt so it
stands alone in the new session:

- **Source and scope:** repository path, issue/spec reference, current acceptance
  criteria and amendments, scope boundaries, and non-obvious agent findings.
  Reference the complete source, while spelling out decisions and guarantees
  the new session cannot infer from that source alone. Include unresolved risks
  and assumptions; avoid copying the entire ticket.
- **Fresh branch:** observed branch, exact HEAD, worktree status, and requested
  base/ref with its observed SHA or resolution limitation. Instruct the new
  session to reread repository guidance and verify current state before edits;
  fetch if needed to establish the requested remote base, then create and check
  out a descriptive new branch using repository conventions and the issue number.
  Record the actual base SHA. If the suggested name exists, choose a unique name
  unless the user explicitly requested resuming it. Preserve unrelated work and
  exclude it from commits; use an isolated worktree if it prevents mixing edits.
- **Implementation fan-out:** use worker subagents with explicit, disjoint
  file/module ownership. Coordinate shared contracts first and assign shared
  files and integration to one owner. Tell every worker that others are editing
  the codebase and to accommodate their changes without reverting them. Tailor
  the work lanes to gathered findings, and delegate an independent read-only
  review of the integrated diff. Use the repository's `code-review` skill when
  available; otherwise review against repository standards, the issue/spec, and
  the gathered contracts directly.
- **Validation:** name existing public acceptance journeys, relevant test files,
  fixture rebuild commands, and check commands verified from repository files.
  Follow repository testing policy, run affected checks during implementation,
  and complete the applicable full verification gate before publication. Fix
  independent review findings and repeat affected checks after later edits.
  Distinguish actual failures or unavailable checks from passing evidence.
- **Publication:** explicitly stage related paths, make signed Conventional
  Commits, and push with normal Git hooks enabled. Never bypass hooks, including
  failures that predate this work: fix them on the branch and retry, or report
  the blocker. Use the repository's `pr` skill for an issue-linked PR body and
  open a ready-for-review PR after the applicable checks pass.
- **PR review loop:** after opening the PR, verify its selected base/head
  repositories, refs, and SHAs, and use the implementation checkout bound to that
  head. Run `daydream . --backend codex --comment --non-interactive` by default.
  Confirm its comparison base resolves to the verified PR base SHA; if the
  default is wrong or stale, including for a non-default PR target,
  append only `--base <verified ref/SHA>`. Fetch the required base if needed
  without disturbing user work, and verify any chosen ref resolves to that exact
  PR base SHA. Record the actual command, selected base/head SHAs, resolved review
  comparison/merge-base, and command result. Refresh moved PR refs before review.
  Use
  `fetch-pr-feedback` with `--include-author --include-self`, then
  `receive-feedback`, then `respond-pr-feedback` to fetch, evaluate, address, and
  reply to the results. Fetch posted evidence even if Daydream partially fails;
  retain the failure as a blocker until a successful review covers the current
  HEAD. Include feedback from every reviewer and bot, including `greptile-apps`,
  across inline threads, submitted reviews, and general PR comments. Distinguish
  reviewer findings from self-authored agent replies and PR descriptions rather
  than treating every fetched message as a new finding. Verify findings
  against the current code before accepting or rejecting them. Fix confirmed
  issues, rerun affected checks, commit and push with hooks, then post
  evidence-based replies and resolve eligible threads according to those skills.
  Rerun Daydream after changes so its review covers the latest PR HEAD, refresh
  other reviewer feedback and required CI, and continue until all actionable
  feedback is addressed and the latest HEAD satisfies the completion criteria.
  Recheck known queued or in-progress reviewer/bot runs even when they are not
  required CI; an empty fetch does not complete a pending review. Wait for those
  known runs or report an external blocker, without waiting for bots that are
  unconfigured or have no requested run.
  Review failures, pending required checks, and unavailable feedback remain
  explicit blockers; absence of feedback alone is not proof of review success.
- **Authority and completion:** the pasted goal authorizes scoped implementation
  and verified review fixes, signed commits, pushes, opening/updating the PR,
  reviewer replies, and eligible thread resolutions. Continue those actions
  without redundant approval requests within the goal's scope. Merge-ready means
  all acceptance criteria completed, applicable local checks and required CI
  passing on the latest HEAD, independent and Daydream reviews completed with
  verified findings addressed, a pushed branch and open ready-for-review PR,
  and a clean working tree apart from preserved unrelated work. Report the PR
  URL, final HEAD, checks, and feedback dispositions. Stop before merging; if an
  external blocker prevents completion, report it without claiming merge readiness.

Replace placeholders with gathered facts and concrete commands; do not invent
commands or claim checks passed during preparation. The final response contains
only the goal prompt, with all branch instructions inside it.
