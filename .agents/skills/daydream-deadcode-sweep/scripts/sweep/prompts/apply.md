# Code-simplification applier — __CHUNK_LABEL__

Candidates: __FINDINGS__
Artifact directory: __APPLY_DIR__

Implement complete changes that maximize net code removed while preserving
supported behavior and making the remaining implementation easier to understand.
Substantial internal refactors and smaller implementations of fully used behavior
are allowed. The coordinator has selected globally ranked, non-overlapping candidates;
estimated savings and confidence are leads, not proof.

## Before editing

- Read repository instructions and reopen every affected implementation, caller,
  reuse target, and supporting contract. Revalidate needs_revalidation findings.
  Distinguish production/test callers and supported external or dynamic entrypoints.
- State what disappears and which implementation or owner remains. Preserve CLI,
  supported APIs, persisted formats, observable outputs/side effects, failure paths,
  authorization boundaries, cancellation, and any required immutable snapshots.
- Consider deleting/inlining a layer and reusing an existing implementation before
  adding a helper. Count the helper plus all migrated callers/tests. Choose fewer
  concepts and clear ownership; do not introduce a configurable framework to save
  a few repeated lines.
- For a library replacement, recheck the pinned API's ownership, identity, lifetime,
  failure/cancellation, filtering/default, and persisted-format semantics relevant
  to the removed implementation. Retain a distinct guard when the library does not
  supply it; verify the migration through independent surviving contract tests.
- Defer candidates whose behavior or migration cost is unresolved. Record a precise
  next step rather than taking a speculative deletion to CI.

## Apply a complete simplification

In full-repository mode, net-negative means total insertions < total deletions across
the committed PR. In PR mode the PR may grow, but each simplification must remove more
than it adds. Count all replacement code, callers, tests, and support files. No minimum savings;
prefer the largest well-supported changes that can be finished and verified.
File moves, whitespace removal, compressed statements, fixture minification, and
documentation cuts that conceal code growth do not qualify as simplification.

New implementation code, helpers, and meaningful tests are allowed when required
for the smaller design. Preserve supported behavior; feature removal is outside
scope. Migrate callers completely and remove the old path rather than keeping an
unnecessary compatibility alias. Afterward inspect newly orphaned imports, types,
parameters, tests, fixtures, schemas, configuration, and dependencies.

Reuse existing test fixtures where appropriate. Parameterize only equivalent
scenarios. For each deleted test, name the surviving independent assertion and
failure path, or the proven-dead behavior removed with it. Do not derive expected
output from the same implementation being tested. Keep explanations of why,
invariants, security, compatibility, and non-obvious architecture.

Work one connected change at a time. Leave coherent checkpoints before time runs
out. A partial batch may ship only if each retained change preserves behavior and
passes verification; half-migrated code is not a completed simplification.

## Verification and commits

Never disable or bypass commit/push hooks, signatures, checks, or the coverage
floor, including for supposedly unrelated failures. Do not change Git identity.
When commits are authorized, run ordinary commits. A hook failure is a failure;
preserve the work and report it.

For dependency changes, trace the graph through sibling path-dependency projects
and generated deployment/runtime locks, including hash or source-fingerprint
headers. Find authoritative generators in repository scripts/docs, regenerate all
affected outputs, and check regeneration is stable. Include these files in the
change's net cost. A changed sibling lock requires that project's separate gate;
run `make rl-check` when RL inputs change. Do this before the final full gate so
ordinary uv commands cannot silently repair an omitted lock during verification.

Append affected surviving test FILE paths, one canonical repo-relative path per
line, to __APPLY_DIR__/tests-__CHUNK_LABEL__.txt. Include unchanged consumers of
changed production code/shared helpers. Exclude deleted files. The root and
rl/daydream_review projects use separate environments.

Run the smallest meaningful affected tests while implementing. Add characterization
tests when a substantial refactor lacks evidence for an important existing behavior.
The coordinator follows the skill's verification procedure, including the final
make check.
Normal pre-push hooks still run make check; avoid duplicating the entire suite per
finding. CI remains required for merge.

If a simplification breaks behavior or checks, correct it or undo only that change,
including restoring code/tests when needed. Net reduction is measured across the
remaining sweep changes; never delete more live behavior to repair a deletion. Update the
summary/outcome records if a previously applied change is reversed.

When the user has requested commits or shipping, commit completed work with a
conventional message describing the actual simplification and additions/deletions/net. Otherwise leave local changes uncommitted for review.
Include relevant verification evidence. Do not claim completion because a timeout
checkpoint was auto-committed.

## Durable outcomes and report

Append one JSON object per candidate to
__APPLY_DIR__/outcomes-__CHUNK_LABEL__.jsonl:

{"id":"candidate id supplied in findings","status":"applied","reason":"migrated all callers; preserved failure path verified by named test"}

Use applied, rejected, or deferred. Rejected/deferred rows need evidence or a
specific unresolved question/next step. If repair changes an earlier decision,
append its corrected outcome to outcomes-repair.jsonl. Unapplied findings remain
in durable memory; the next run should not have to discover them again.

As each change is verified, append a plain semantic bullet to
__APPLY_DIR__/summary-bullets.md. Name the area and what disappeared or became
simpler; no heading, filler, commit-subject echo, or claims about untouched files.
One honest bullet is better than padding. Keep bullets concise; remove any made
inaccurate by repair. Take the file breakdown from `git diff --numstat <base>...HEAD`.

Write __APPLY_DIR__/report-__CHUNK_LABEL__.md with applied/deferred/rejected IDs,
preserved contracts, verification actually run, and actual additions/deletions/net
against the branch point. Distinguish whole-PR totals from this chunk. Do not claim
unrun gates or CI passed.
