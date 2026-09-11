"""Authorized fix-footprint enforcement and residual-edit issue filing.

Canonical finding paths are admitted by :class:`AuthorizedFixFootprint` even
when they are outside the reviewed diff. Edits outside that run-wide footprint
are restored here and, when ``scope_issue_filing`` is enabled, filed as tracked
GitHub issues with cross-run fingerprint deduplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from daydream.agent import console
from daydream.deep.state import DeepState
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.git_ops import INHERIT_GITHUB_AUTH, GitHubAuth, GitPathState
from daydream.ui import print_warning

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext
    from daydream.workspace import WorkContext


@dataclass(frozen=True)
class ScopeEnforcementResult:
    """Outcome of one strict run-wide authorization enforcement boundary."""

    retained_paths: frozenset[str]
    mutated: bool


def enforce_authorized_fix_footprint(
    work: WorkContext,
    stable_ref: str,
    footprint: AuthorizedFixFootprint,
    *,
    preexisting_untracked: dict[str, GitPathState],
    preexisting_gitlinks: tuple[GitPathState, ...] = (),
    phase: str,
    round_number: int | None,
    file_scope_issues: bool = False,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> ScopeEnforcementResult:
    """Restore every run-external edit while preserving authorized siblings.

    Enumeration, capture, restoration, and verification are strict. Protected
    pre-existing untracked paths are compared by their stored blob/type/mode
    state and restored even though they remain untracked throughout the run.
    The index is snapshotted at entry and restored byte-for-byte after the
    worktree-only recovery operation.
    """
    from daydream import git_ops

    repo = work.repo
    index_before = git_ops.snapshot_index(repo)
    tracked_changed = set(git_ops.changed_paths_z(repo, stable_ref, include_untracked=False))
    gitlink_baseline = {state.path: state for state in preexisting_gitlinks}
    current_gitlinks = {
        state.path: state
        for state in git_ops.snapshot_worktree_paths(repo, gitlink_baseline)
    }
    mutated_gitlinks = {
        path
        for path, baseline in gitlink_baseline.items()
        if current_gitlinks[path] != baseline
    }
    current_untracked = git_ops.snapshot_untracked_paths(repo, include_runtime_artifacts=False)
    current_protected: dict[str, GitPathState] = {}
    for path in preexisting_untracked:
        if path in current_untracked:
            current_protected[path] = current_untracked[path]
        else:
            current_protected[path] = git_ops.snapshot_worktree_paths(repo, [path])[0]
    mutated_protected = {
        path
        for path, baseline in preexisting_untracked.items()
        if current_protected[path] != baseline
    }
    new_untracked = set(current_untracked) - set(preexisting_untracked)
    residual_tracked = tracked_changed - set(footprint.run_allowed_paths)
    # A pre-run checkout different from the superproject entry is owner state,
    # not a run-created residual.  Preserve it unless the run changed it.
    residual_tracked -= {
        path
        for path, baseline in gitlink_baseline.items()
        if current_gitlinks[path] == baseline
    }
    residual_new = new_untracked - set(footprint.run_allowed_paths)
    protected_gitlink_mutations = mutated_gitlinks - set(footprint.run_allowed_paths)
    to_restore = (
        residual_tracked
        | residual_new
        | mutated_protected
        | protected_gitlink_mutations
    )

    filing_evidence: list[tuple[str, str]] = []
    if file_scope_issues:
        paths_to_file = sorted(
            residual_tracked - set(preexisting_untracked),
            key=lambda value: value.encode("utf-8", "surrogateescape"),
        )
        for path in paths_to_file:
            try:
                patch = git_ops.diff_worktree_against(repo, stable_ref, [path])
            except Exception:  # noqa: BLE001 -- optional evidence must not block restoration
                print_warning(
                    console,
                    "Could not capture optional out-of-scope edit evidence; "
                    "continuing to restoration without filing.",
                )
            else:
                filing_evidence.append((path, patch))

    if to_restore:
        stable_states = git_ops.snapshot_commit_paths(
            repo,
            stable_ref,
            residual_tracked - set(preexisting_untracked) - set(gitlink_baseline),
        )
        rollback = git_ops.WorktreeRollbackSnapshot(
            ref=stable_ref,
            index=index_before,
            path_states=tuple(
                (
                    *stable_states,
                    *(gitlink_baseline[path] for path in protected_gitlink_mutations),
                )
            ),
            untracked={path: preexisting_untracked[path] for path in mutated_protected},
        )
        git_ops.restore_group_from_snapshot(repo, rollback, to_restore)

        for path in sorted(to_restore, key=lambda value: value.encode("utf-8", "surrogateescape")):
            if path in residual_new and path not in preexisting_untracked:
                action = "remove"
                reason = "removed a new untracked path outside the authorized run footprint"
            else:
                action = "restore"
                reason = (
                    "restored protected pre-existing untracked state"
                    if path in preexisting_untracked
                    else (
                        "restored protected pre-existing gitlink checkout"
                        if path in gitlink_baseline
                        else "restored tracked content outside the authorized run footprint"
                    )
                )
            footprint.record_git_event(
                action=action,
                path=path,
                origin="guard",
                phase=phase,
                round_number=round_number,
                reason=reason,
            )

    if git_ops.snapshot_index(repo) != index_before:
        raise git_ops.GitError("scope enforcement changed the repository index")
    untracked_after = git_ops.snapshot_untracked_paths(repo, include_runtime_artifacts=False)
    for path, baseline in preexisting_untracked.items():
        current = (
            untracked_after[path]
            if path in untracked_after
            else git_ops.snapshot_worktree_paths(repo, [path])[0]
        )
        if current != baseline:
            raise git_ops.GitError("scope enforcement did not restore protected untracked state")
    remaining_tracked = set(git_ops.changed_paths_z(repo, stable_ref, include_untracked=False))
    final_gitlinks = {
        state.path: state
        for state in git_ops.snapshot_worktree_paths(repo, gitlink_baseline)
    }
    for path, baseline in gitlink_baseline.items():
        if path in protected_gitlink_mutations and final_gitlinks[path] != baseline:
            raise git_ops.GitError("scope enforcement did not restore protected gitlink state")
        if final_gitlinks[path] == baseline:
            remaining_tracked.discard(path)
    remaining_untracked = set(untracked_after) - set(preexisting_untracked)
    residual_after = (remaining_tracked | remaining_untracked) - set(footprint.run_allowed_paths)
    if residual_after:
        raise git_ops.GitError("scope enforcement left paths outside the authorized run footprint")
    retained = ((remaining_tracked | remaining_untracked) & set(footprint.run_allowed_paths)) - set(
        preexisting_untracked
    )
    for path, patch in filing_evidence:
        try:
            _file_reverted_edit_issue(repo, path, patch, auth=auth)
        except Exception:  # noqa: BLE001 -- all issue filing is best-effort
            print_warning(
                console,
                "Could not file optional out-of-scope edit evidence after restoration.",
            )
    return ScopeEnforcementResult(retained_paths=frozenset(retained), mutated=bool(to_restore))


def _file_scope_issue(
    repo: Path,
    *,
    title: str,
    body: str,
    noun: str,
    ident: str,
    auth: GitHubAuth,
) -> None:
    """Best-effort: file one scope-related GitHub issue; never raise.

    The residual-edit filer records work the footprint guard restored instead
    of landing it in the PR. Filing is best-effort: a failed ``gh issue create``
    (no auth, cross-org, offline) logs a warning and the restore stands.
    """
    from daydream import git_ops

    try:
        url = git_ops.gh_issue_create(repo, title=title, body=body, auth=auth)
        print_warning(console, f"Filed out-of-scope {noun} as issue: {url}")
    except Exception as exc:  # noqa: BLE001 -- best-effort issue filing
        print_warning(console, f"Could not file out-of-scope {noun} '{ident}' as issue: {exc}")


def _scope_already_filed(repo: Path, marker: str, *, auth: GitHubAuth) -> bool:
    """Best-effort: has an open issue already filed this scope marker?

    GitHub is the store: scan open issues for the reverted edit's fingerprint
    marker and skip filing when present, so a re-run reproducing the same edit
    does not file a duplicate issue. Best-effort by construction:
    ``gh_issue_list`` returns an empty list on failure, so a failed lookup
    degrades to filing rather than silently dropping the restored edit.

    The marker is computed once by the caller and threaded in, so the edit's
    fingerprint is not recomputed for both the dedup lookup and the issue body.
    """
    from daydream import git_ops

    issues = git_ops.gh_issue_list(repo, search="out-of-scope", auth=auth)
    return any(marker in (issue.get("body") or "") for issue in issues)


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
    Its scope-edit marker namespace stays distinct from PR-comment findings.
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

    The distinct ``daydream-scope-edit`` prefix prevents collisions with the
    PR-comment finding store.
    """
    return f"<!-- daydream-scope-edit: {fingerprint} -->"


def _file_reverted_edit_issue(
    repo: Path, path: str, patch: str, *, auth: GitHubAuth
) -> None:
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
    if _scope_already_filed(repo, marker, auth=auth):
        return
    title = f"[daydream] out-of-scope edit reverted: {path}"
    body = (
        f"The fix pass edited `{path}`, which is outside the reviewed diff. "
        f"The edit was reverted and is filed here for review.\n\n"
        f"```diff\n{patch}\n```\n\n"
        "Filed by daydream fix loop: out of scope for PR.\n"
        f"{marker}"
    )
    _file_scope_issue(repo, title=title, body=body, noun="edit", ident=path, auth=auth)


def _resolve_changed_files(ctx: FlowContext) -> set[str] | None:
    """Resolve the complete reviewed-diff file set for fix-cycle provenance.

    Issue #336 — the reviewed-diff file set is computed once in the
    ``_run_review_spine`` preamble and carried in ``ctx.data["changed_files"]``.
    On a ``--start-at fix`` resume that lost it but retained ``diff`` it is
    recomputed via ``_diff_changed_files``. The fix gate records these paths as
    reviewed origins in the authorized footprint, and the quality gate uses the
    Python subset. Canonical finding paths are authorized independently.
    """
    deep_state = DeepState(ctx.data)
    from daydream.deep.diff import _diff_changed_files, _read_full_diff

    changed_files: set[str] | None = deep_state.changed_files_or_none
    if changed_files is None:
        # Issue #644 — ctx.data["diff"] is the gather-time BOUNDED diff; the
        # resume path must resolve provenance from the FULL diff.patch (via
        # diff_path) so a truncated-away file is not silently omitted. Fall
        # back to the bounded text only when the ctx carries no diff_path.
        try:
            diff_str = _read_full_diff(ctx) or ""
        except OSError:
            diff_str = deep_state.diff_or_empty or ""
        if diff_str:
            changed_files = set(_diff_changed_files(diff_str))
    return changed_files
