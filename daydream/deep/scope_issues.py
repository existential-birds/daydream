"""Out-of-scope issue routing for the deep fix loop (issue #336, extracted in #338).

The fix loop auto-fixes only findings inside the reviewed diff. Findings on files
outside the diff, and post-fix edits outside the diff, are routed to tracked
GitHub issues (best-effort) and excluded from auto-fix. This module owns that
machinery: scope-finding fingerprint -> marker -> cross-run dedup -> filing, plus
the post-fix residual revert net. Extracted from ``deep/orchestrator.py``.
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


def _scope_finding_already_filed(repo: Path, marker: str) -> bool:
    """Best-effort: has an open issue already filed this out-of-scope finding?

    Issue #336 — out-of-scope findings are never fixed, so a re-run/resume
    re-derives them and would re-file a duplicate issue every time. GitHub is
    the store: scan open issues for the finding's fingerprint marker and skip
    filing when present. Best-effort — a failed ``gh issue list`` returns
    ``False`` so the call degrades to filing (the prior behavior) rather than
    silently dropping the finding.

    The marker is computed once by the caller (``_file_out_of_scope_issue``)
    and threaded in, so the finding's fingerprint is not recomputed for both
    the dedup lookup and the issue body.
    """
    from daydream import git_ops

    try:
        issues = git_ops.gh_issue_list(repo, search="out-of-scope")
    except Exception:  # noqa: BLE001 -- best-effort dedup lookup
        return False
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
    if _scope_finding_already_filed(ctx.work.repo, marker):
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


def _file_reverted_edit_issue(repo: Path, path: str, patch: str) -> None:
    """Best-effort: file one reverted out-of-scope edit as a tracked GitHub issue.

    Issue #336 — the post-fix residual check reverts edits the fix pass made
    outside the reviewed diff. The reverted edit is still *valid* work, so it
    is filed as an issue (with the diff as evidence) instead of being lost.
    Filing is best-effort: a failed ``gh issue create`` logs a warning and the
    revert stands regardless.
    """
    title = f"[daydream] out-of-scope edit reverted: {path}"
    body = (
        f"The fix pass edited `{path}`, which is outside the reviewed diff. "
        f"The edit was reverted and is filed here for review.\n\n"
        f"```diff\n{patch}\n```\n\n"
        "Filed by daydream fix loop: out of scope for PR."
    )
    _file_scope_issue(repo, title=title, body=body, noun="edit", ident=path)


def _revert_out_of_scope_edits(
    work: WorkContext,
    *,
    pre_fix_ref: str,
    snapshot_captured: bool,
    pre_fix_untracked: set[str],
    changed_files: set[str] | None,
    finding_files: set[str],
) -> list[str] | None:
    """Revert post-fix edits outside the reviewed diff; file an issue per residual.

    Issue #336 (Task 4): after ``phase_fix_parallel`` returns, enumerate the
    files the fix pass actually edited (vs the pre-fix snapshot) and subtract
    the allowed set — the finding files, the reviewed-diff file set, and
    newly-created generated files. Every residual is reverted unconditionally to
    its pre-fix content (``git checkout <ref> -- <file>``, the same mechanism
    the generated-file guard uses) and filed as a best-effort GitHub issue
    carrying the edit's diff as evidence, so the commit step can never land an
    edit outside the reviewed diff.

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
        # Capture the diff as evidence BEFORE reverting (the revert destroys it).
        try:
            patch = git_ops.diff_worktree_against(repo, pre_fix_ref, [path])
        except GitError as exc:
            patch = ""
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
        _file_reverted_edit_issue(repo, path, patch)
    if residual:
        print_warning(
            console,
            f"Reverted {len(residual)} post-fix edit(s) outside the reviewed diff and "
            f"filed issue(s): {residual}.",
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
    from daydream.deep.orchestrator import _diff_changed_files

    changed_files: set[str] | None = ctx.data.get("changed_files")
    if changed_files is None:
        diff_str = ctx.data.get("diff") or ""
        if diff_str:
            changed_files = set(_diff_changed_files(diff_str))
    return changed_files
