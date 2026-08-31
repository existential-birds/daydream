"""Out-of-scope issue routing for the deep fix loop (issue #336, extracted in #338).

The fix loop auto-fixes only findings inside the reviewed diff. Findings on files
outside the diff, and post-fix edits outside the diff, are routed to tracked
GitHub issues (best-effort, gated by the ``scope_issue_filing`` opt-in, default
off — issue #1056) and excluded from auto-fix. This module owns that
machinery: scope-finding fingerprint -> marker -> cross-run dedup -> filing, plus
the post-fix residual revert net (whose filing path carries the same
fingerprint -> marker -> dedup -> file chain, issue #1051). Extracted from
``deep/orchestrator.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.agent import console
from daydream.generated_files import is_generated_file
from daydream.ui import print_warning

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext
    from daydream.workspace import WorkContext


def _file_scope_issue(repo: Path, *, title: str, body: str, noun: str, ident: str) -> None:
    """Best-effort: file one scope-related GitHub issue; never raise.

    Shared by the out-of-scope *finding* router (pre-fix gate) and the
    reverted *edit* filer (post-fix residual net). Issue #336 — both file a
    tracked issue for work the fix loop will not land in the PR. Filing is
    best-effort: a failed ``gh issue create`` (no auth, cross-org, offline)
    logs a warning and the scope decision (exclude / revert) stands regardless.
    """
    from daydream import git_ops

    try:
        url = git_ops.gh_issue_create(repo, title=title, body=body)
        print_warning(console, f"Filed out-of-scope {noun} as issue: {url}")
    except Exception as exc:  # noqa: BLE001 -- best-effort issue filing
        print_warning(console, f"Could not file out-of-scope {noun} '{ident}' as issue: {exc}")


def _scope_finding_fingerprint(item: dict[str, Any]) -> str:
    """Stable cross-run identity for an out-of-scope finding.

    Reuses :func:`daydream.pr_review.compute_fingerprint` (path + normalized
    description + rationale), so minor wording drift across runs still maps to
    the same fingerprint and a re-review reproducing an out-of-scope finding is
    recognized as already-filed.
    """
    from daydream.pr_review import compute_fingerprint

    return compute_fingerprint(
        item.get("file") or "",
        item.get("description", ""),
        item.get("evidence", ""),
    )


def _scope_finding_marker(fingerprint: str) -> str:
    """Hidden HTML comment embedding a finding fingerprint in an issue body.

    Distinct from :func:`daydream.pr_review.finding_marker` (the PR-comment
    store) so the two stores never collide: scope issues are deduped only
    against scope issues, PR comments only against PR comments.
    """
    return f"<!-- daydream-scope-finding: {fingerprint} -->"


def _scope_already_filed(repo: Path, marker: str) -> bool:
    """Best-effort: has an open issue already filed this scope marker?

    Shared by the out-of-scope *finding* router (pre-fix gate, issue #336) and
    the reverted *edit* filer (post-fix residual net, issue #1051). GitHub is
    the store: scan open issues for the item's fingerprint marker and skip
    filing when present, so a re-run/resume re-deriving the same item does not
    re-file a duplicate issue. Best-effort by construction: ``gh_issue_list``
    itself returns an empty list on any failure (no auth, offline, cross-org),
    so a failed lookup degrades to filing (the prior behavior), never to
    silently dropping the item.

    The marker is computed once by the caller and threaded in, so the item's
    fingerprint is not recomputed for both the dedup lookup and the issue body.
    """
    from daydream import git_ops

    issues = git_ops.gh_issue_list(repo, search="out-of-scope")
    return any(marker in (issue.get("body") or "") for issue in issues)


def _file_out_of_scope_issue(ctx: FlowContext, item: dict[str, Any]) -> None:
    """Best-effort: file one out-of-scope finding as a tracked GitHub issue.

    Issue #336 — the fix loop auto-fixes only findings inside the reviewed
    diff. A finding on a file outside the diff is still *valid*, so instead of
    silently dropping it we route it to a GitHub issue where it stays tracked.
    Cross-run dedup: the finding's fingerprint is embedded as a hidden marker
    in the body, and a re-run/resume skips filing when an open issue already
    carries it, so the same out-of-scope finding is filed once, not on every
    run. Filing is best-effort by design: a failed ``gh issue create`` (no auth,
    cross-org, offline) logs a warning and the item is still excluded from
    auto-fix — the scope decision never depends on issue-filing success.
    """
    file = item.get("file") or "<unknown>"
    description = item.get("description", "No description")
    evidence = item.get("evidence", "")
    # Compute the fingerprint marker once and thread it into both the dedup
    # lookup and the issue body, rather than recomputing it for each.
    marker = _scope_finding_marker(_scope_finding_fingerprint(item))
    if _scope_already_filed(ctx.work.repo, marker):
        return
    title = f"[daydream] out-of-scope finding: {file}"
    body = (
        f"{description}\n\n"
        f"- File: `{file}`\n"
        f"- Line: {item.get('line', '?')}\n"
        f"- Evidence: {evidence}\n\n"
        "Filed by daydream fix loop: out of scope for PR.\n"
        f"{marker}"
    )
    _file_scope_issue(ctx.work.repo, title=title, body=body, noun="finding", ident=file)


def _scope_edit_fingerprint(path: str, patch: str) -> str:
    """Stable cross-run identity for a reverted out-of-scope edit (issue #1051).

    Keyed on the file path plus the edit's changed content lines (the spec's
    "file path plus diff content" option): a re-run that reproduces the same
    residual edit on the same file maps to the same fingerprint and is
    recognized as already-filed. The raw ``git diff`` evidence embeds volatile
    hunk offsets (``@@ -x,y +z,w @@``) and context lines, so fingerprinting the
    raw patch would re-file a duplicate whenever a re-run reproduces the edit
    at shifted offsets — the duplicate-issue regression #1051 set out to close.
    Stripping the hunk headers and context lines (keeping only the ``+``/``-``
    content lines) mirrors how :func:`daydream.pr_review.compute_fingerprint`
    excludes line numbers so code shifts do not change a finding's identity.
    Distinct from the finding-path fingerprint so the two stores never collide.
    """
    from daydream.pr_review import compute_fingerprint

    changed_lines = [
        line
        for line in patch.splitlines()
        if (line.startswith("+") or line.startswith("-")) and not line.startswith(("+++", "---"))
    ]
    return compute_fingerprint(path, "\n".join(changed_lines), "")


def _scope_edit_marker(fingerprint: str) -> str:
    """Hidden HTML comment embedding an edit fingerprint in an issue body.

    Distinct prefix from ``_scope_finding_marker`` so the finding store and
    the edit store never collide, mirroring the split between
    ``_scope_finding_marker`` and ``pr_review.finding_marker``.
    """
    return f"<!-- daydream-scope-edit: {fingerprint} -->"


def _file_reverted_edit_issue(repo: Path, path: str, patch: str) -> None:
    """Best-effort: file one reverted out-of-scope edit as a tracked GitHub issue.

    Issue #336 — the post-fix residual check reverts edits the fix pass made
    outside the reviewed diff. The reverted edit is still *valid* work, so it
    is filed as an issue (with the diff as evidence) instead of being lost.
    Cross-run dedup (issue #1051): the edit's fingerprint is embedded as a
    hidden marker in the body, and a re-run/resume skips filing when an open
    issue already carries it. Filing is best-effort: a failed ``gh issue
    create`` logs a warning and the revert stands regardless.
    """
    # Compute the fingerprint marker once and thread it into both the dedup
    # lookup and the issue body, rather than recomputing it for each.
    marker = _scope_edit_marker(_scope_edit_fingerprint(path, patch))
    if _scope_already_filed(repo, marker):
        return
    title = f"[daydream] out-of-scope edit reverted: {path}"
    body = (
        f"The fix pass edited `{path}`, which is outside the reviewed diff. "
        f"The edit was reverted and is filed here for review.\n\n"
        f"```diff\n{patch}\n```\n\n"
        "Filed by daydream fix loop: out of scope for PR.\n"
        f"{marker}"
    )
    _file_scope_issue(repo, title=title, body=body, noun="edit", ident=path)


def _scope_filing_note(file_scope_issues: bool) -> str:
    """Filing-status note shared by the gate and residual-net messages.

    Issue #1056 — the pre-fix gate (``_step_fix_gate``) and the post-fix
    residual net (``_revert_out_of_scope_edits``) both branch on the same
    ``scope_issue_filing`` opt-in to report whether out-of-scope items were
    filed as GitHub issues or just excluded/reverted. Only the trailing note
    differs between the two branches at every message site, so the pair is
    rendered through this single helper instead of per-module if/else ladders
    that could drift on future edits.
    """
    return "filed as issue(s)" if file_scope_issues else "(issue filing disabled)"


def _revert_out_of_scope_edits(
    work: WorkContext,
    *,
    pre_fix_ref: str,
    snapshot_captured: bool,
    pre_fix_untracked: set[str],
    changed_files: set[str] | None,
    finding_files: set[str],
    file_scope_issues: bool = False,
) -> list[str] | None:
    """Revert post-fix edits outside the reviewed diff; conditionally file issues.

    Issue #336 (Task 4): after ``phase_fix_parallel`` returns, enumerate the
    files the fix pass actually edited (vs the pre-fix snapshot) and subtract
    the allowed set — the finding files, the reviewed-diff file set, and
    newly-created generated files. Every residual is reverted unconditionally to
    its pre-fix content (``git checkout <ref> -- <file>``, the same mechanism
    the generated-file guard uses) so the commit step can never land an
    edit outside the reviewed diff.

    Filing is OPT-IN (issue #1056): when *file_scope_issues* is true each
    reverted residual is additionally filed as a best-effort GitHub issue
    carrying the edit's diff as evidence (with cross-run dedup, issue #1051);
    when false (the default) the revert alone happens and a warning notes that
    issue filing is disabled. The revert and the fail-close never depend on
    filing.

    Only TRACKED edits are considered: a path present at *pre_fix_ref*. Newly
    created untracked files (e.g. the test harness's ``.fixed-*`` sentinels)
    have no pre-fix content to restore to and are governed by the generated-file
    guard and the leftover-untracked bookkeeping instead.

    Args:
        work: The run's workspace (``work.repo`` is the git working dir).
        pre_fix_ref: ``git stash create`` SHA captured before fixes, or ``HEAD``
            when the pre-fix tracked tree was clean.
        snapshot_captured: Whether the pre-fix snapshot exists. When False the
            baseline is untrustworthy — skip with a warning rather than guess.
        pre_fix_untracked: Untracked paths present before the fix pass.
        changed_files: The reviewed diff's file set, or ``None`` when no diff
            context is available (resume) — then only finding-file scope is
            enforced and a warning is logged.
        finding_files: Files named by the findings being fixed.

    Returns:
        The repo-relative paths that were reverted (empty when none), or ``None``
        when any revert failed — matching the generated-file guard, the caller
        aborts the run so an unreverted out-of-scope edit never reaches commit.
    """
    from daydream import git_ops
    from daydream.git_ops import GitError

    if not snapshot_captured:
        print_warning(
            console,
            "Post-fix scope check skipped: no trustworthy pre-fix snapshot "
            "(cannot judge which edits the fix pass made).",
        )
        return []

    repo = work.repo
    try:
        edited = set(
            git_ops.changed_files_against(repo, pre_fix_ref, preexisting_untracked=pre_fix_untracked)
        )
    except GitError as exc:
        print_warning(console, f"Post-fix scope check skipped: could not enumerate edited files: {exc}")
        return []

    allowed = set(finding_files)
    if changed_files is not None:
        allowed |= set(changed_files)
    else:
        print_warning(
            console,
            "Post-fix scope check: no reviewed-diff file set; enforcing finding-file scope only.",
        )
    # Newly-created generated files (e.g. new migrations) are deliberately
    # permitted by the generated-file guard; never treat them as residuals here.
    allowed |= {path for path in edited if is_generated_file(path)}

    residual: list[str] = []
    for path in sorted(edited):
        # Only tracked files (present at the pre-fix ref) are revertible to a
        # pre-fix baseline; skip untracked new files (see docstring).
        try:
            git_ops.show(repo, pre_fix_ref, path)
        except GitError:
            continue
        if path in allowed:
            continue
        residual.append(path)

    restoration_failed = False
    for path in residual:
        patch = ""
        # Capture the diff as evidence BEFORE reverting (the revert destroys
        # it) — but only when filing is opted in: with filing disabled the
        # patch has no consumer, so skip the per-residual ``git diff``
        # subprocess entirely (issue #1056). The revert and fail-close below
        # never depend on the capture.
        if file_scope_issues:
            try:
                patch = git_ops.diff_worktree_against(repo, pre_fix_ref, [path])
            except GitError as exc:
                print_warning(console, f"Could not diff out-of-scope edit '{path}': {exc}")
        # Revert unconditionally — same mechanism as the generated-file guard.
        try:
            git_ops.restore_paths_from_ref(repo, pre_fix_ref, [path])
        except GitError as exc:
            print_warning(console, f"Could not revert out-of-scope edit '{path}': {exc}")
            # Fail-close to match the sibling generated-file guard: an
            # unreverted out-of-scope edit must never reach the commit step
            # (the whole point of this net), so signal abort rather than leave
            # the edit in the worktree. Continue so remaining residuals are
            # still best-effort reverted before the abort is signalled.
            restoration_failed = True
            continue
        if file_scope_issues:
            _file_reverted_edit_issue(repo, path, patch)
    if residual:
        print_warning(
            console,
            f"Reverted {len(residual)} post-fix edit(s) outside the reviewed diff "
            f"{_scope_filing_note(file_scope_issues)}: {residual}.",
        )
    return None if restoration_failed else residual


def _resolve_changed_files(ctx: FlowContext) -> set[str] | None:
    """Resolve the reviewed-diff file set for the fix-loop scope bound.

    Issue #336 — the reviewed-diff file set is computed once in the
    ``_run_review_spine`` preamble and carried in ``ctx.data["changed_files"]``.
    On a ``--start-at fix`` resume that lost it but retained ``diff`` it is
    recomputed via ``_diff_changed_files``. Shared by ``_step_fix_gate`` (the
    pre-fix partition) and ``_step_fix`` (the post-fix residual net + the
    "Allowed files" prompt clause) so the gate and the net agree on the allowed
    set — a divergence left the residual net strictly weaker than the gate on
    the resume path.
    """
    from daydream.deep.orchestrator import _diff_changed_files, _read_full_diff

    changed_files: set[str] | None = ctx.data.get("changed_files")
    if changed_files is None:
        # Issue #644 — ctx.data["diff"] is the gather-time BOUNDED diff; the
        # resume path must resolve the scope from the FULL diff.patch (via
        # diff_path) so a truncated-away file is not silently excluded from
        # the auto-fix scope. Fall back to the bounded text only when the ctx
        # carries no diff_path (defensive legacy path).
        try:
            diff_str = _read_full_diff(ctx) or ""
        except OSError:
            diff_str = ctx.data.get("diff") or ""
        if diff_str:
            changed_files = set(_diff_changed_files(diff_str))
    return changed_files
