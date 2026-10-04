"""Post canonical review findings after resolving their PR placement.

Live reviews resolve local Git evidence; artifact posting validates event-derived
identity and reconciles prior bot comments before submitting a review."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from daydream import git_ops
from daydream.extensions import (
    get_registry,
)
from daydream.git_ops import INHERIT_GITHUB_AUTH, GitError, GitHubAuth
from daydream.pr_comment_renderer import render_run_info
from daydream.reviews.diagrams import (
    diagram_comment_kinds as diagram_comment_kinds,
    post_diagram_artifact as post_diagram_artifact,
    post_diagram_comment_to_pr as post_diagram_comment_to_pr,
    render_diagram_blocks_from_payload as render_diagram_blocks_from_payload,
    render_diagram_comment_body as render_diagram_comment_body,
    validate_diagram_payload as validate_diagram_payload,
)
from daydream.reviews.identity import (
    DAYDREAM_FOOTER as DAYDREAM_FOOTER,
    DAYDREAM_REPO_URL as DAYDREAM_REPO_URL,
    DIAGRAM_MARKER_RE as DIAGRAM_MARKER_RE,
    FINDING_MARKER_RE as FINDING_MARKER_RE,
    diagram_marker as diagram_marker,
    finding_marker as finding_marker,
    parse_diagram_markers as parse_diagram_markers,
    parse_finding_markers as parse_finding_markers,
)
from daydream.reviews.models import (
    ClassifiedIssues as ClassifiedIssues,
    ClassifiedReviewResult as ClassifiedReviewResult,
    FileCommentPayload as FileCommentPayload,
    InlineReviewComment as InlineReviewComment,
    ItemFields as ItemFields,
    ParsedIssue as ParsedIssue,
    PostStatus as PostStatus,
    PRInfo as PRInfo,
    ReviewEvent as ReviewEvent,
    ReviewPayload as ReviewPayload,
    ReviewPostResult as ReviewPostResult,
    SubmissionStatus as SubmissionStatus,
)
from daydream.reviews.rendering import (
    ReviewRenderers as ReviewRenderers,
    build_payload_for_event as build_payload_for_event,
    default_render_finding as default_render_finding,
    default_render_summary as default_render_summary,
    format_comment_body as format_comment_body,
    resolve_review_renderers as resolve_review_renderers,
)
from daydream.run_context import RunContext, bind_resolved_run_context, resolve_run_context
from daydream.severity import normalize_severity
from daydream.ui import print_error, print_info, print_success, print_warning

if TYPE_CHECKING:
    from rich.console import Console

    from daydream.findings import ArtifactFinding


@bind_resolved_run_context
async def post_review_to_pr_from_report(
    target_dir: Path,
    merged_items_path: Path,
    *,
    console: Console,
    run_info: str,
    renderers: ReviewRenderers,
    post: bool = False,
    approve_on_clean: bool = False,
    pr_number: int | None = None,
    diagram_blocks: str | None = None,
    review_warnings: tuple[str, ...] = (),
    run_context: RunContext | None = None,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> PostStatus:
    """Post canonical findings from every lens, including structural findings.

    Callers supply captured run-info and renderers. An explicit pr_number disables
    branch discovery; lookup errors fail, while missing PRs return NO_PR.
    post skips interactive confirmation; approve_on_clean enables severity-gated approval."""
    run_context = resolve_run_context(run_context)
    if not merged_items_path.exists():
        return PostStatus.NOTHING_TO_POST
    try:
        items = json.loads(merged_items_path.read_text()).get("items", [])
    except (OSError, json.JSONDecodeError):
        # A corrupt/partially-written merged-items.json must not crash the run
        # with an unhandled JSONDecodeError; treat it like a missing file and
        # skip the post cleanly (#400).
        print_warning(
            console,
            f"Could not read {merged_items_path.name}; skipping PR post.",
        )
        return PostStatus.NOTHING_TO_POST
    issues = parsed_issues_from_items(items)
    if not issues and not approve_on_clean and not review_warnings and not (diagram_blocks and diagram_blocks.strip()):
        print_info(console, "No parseable issues in review output; skipping PR post.")
        return PostStatus.NOTHING_TO_POST
    return await _post(
        target_dir,
        issues,
        console=console,
        run_info=run_info,
        renderers=renderers,
        post=post,
        approve_on_clean=approve_on_clean,
        pr_number=pr_number,
        diagram_blocks=diagram_blocks,
        review_warnings=review_warnings,
        run_context=run_context,
        auth=auth,
    )


def _severity_off_vocabulary(raw: dict[str, Any]) -> bool:
    """Distinguish an asserted unknown severity from an omitted one for approval."""
    value = raw.get("severity")
    return (
        isinstance(value, str)
        and bool(value.strip())
        and normalize_severity(value) is None
    )


def extract_item_fields(
    raw: dict[str, Any],
) -> ItemFields | None:
    """Normalize one canonical finding; reject an empty file path."""
    path = str(raw.get("file", "")).strip()
    if not path:
        return None
    line = raw.get("line")
    line_int = int(line) if isinstance(line, int) and not isinstance(line, bool) else None
    description = str(raw.get("description", "")).strip()
    rationale = str(raw.get("rationale", "")).strip()
    severity = normalize_severity(raw.get("severity"))
    confidence = str(raw.get("confidence", "")).strip().upper() or None
    is_cross_stack = str(raw.get("lens", "")).strip() == "cross-stack"
    location_distrust = bool(raw.get("location_distrust"))
    before = raw.get("severity_before_demotion")
    severity_before_demotion = str(before).strip().lower() or None if before else None
    return ItemFields(
        path=path,
        line_int=line_int,
        description=description,
        rationale=rationale,
        severity=severity,
        confidence=confidence,
        is_cross_stack=is_cross_stack,
        location_distrust=location_distrust,
        severity_before_demotion=severity_before_demotion,
        severity_off_vocabulary=_severity_off_vocabulary(raw),
    )


def parsed_issues_from_items(items: list[dict[str, Any]]) -> list[ParsedIssue]:
    """Convert all canonical lenses to postable findings with severity and provenance."""
    out: list[ParsedIssue] = []
    for raw in items:
        fields = extract_item_fields(raw)
        if fields is None:
            continue
        # Title is the description; rationale (when present and distinct)
        # becomes the body so the agent prompt has context.
        body_parts: list[str] = []
        if fields.severity:
            body_parts.append(f"**Severity:** {fields.severity}")
        if fields.confidence:
            body_parts.append(f"**Confidence:** {fields.confidence}")
        if fields.location_distrust:
            # First production reader of the location-distrust signal (issue
            # #972 R2): the demotion is visible on the report, never silent.
            if fields.severity_before_demotion:
                body_parts.append(
                    "**Location:** unverified citation "
                    f"(severity demoted from {fields.severity_before_demotion})"
                )
            else:
                body_parts.append("**Location:** unverified citation (severity demoted)")
        if fields.rationale and fields.rationale != fields.description:
            body_parts.append(fields.rationale)
        body = "\n\n".join(body_parts)
        out.append(
            ParsedIssue(
                path=fields.path,
                line=fields.line_int,
                title=fields.description,
                body=body,
                is_cross_stack=fields.is_cross_stack,
                confidence=fields.confidence,
                severity=fields.severity,
                fingerprint=compute_fingerprint(
                    fields.path, fields.description, fields.rationale
                ),
                location_distrust=fields.location_distrust,
                severity_before_demotion=fields.severity_before_demotion,
                severity_off_vocabulary=fields.severity_off_vocabulary,
            )
        )
    return out


def _head_repo_slug_from_row(row: dict[str, Any]) -> str | None:
    """Resolve the head repository, rejecting malformed or contradictory identities.

    A deleted fork (null repository) allows base-repository link fallback. Older gh
    versions omit nameWithOwner or emit an empty string; validated owner/name may
    reconstruct it. Populated fields must agree case-insensitively."""
    if "headRepository" not in row or "headRepositoryOwner" not in row:
        raise GitError("invalid PR row: missing requested head repository metadata")
    head_owner = row["headRepositoryOwner"]
    owner_login: str | None = None
    if head_owner is not None:
        if not isinstance(head_owner, dict) or not isinstance(head_owner.get("login"), str):
            raise GitError("invalid PR row: malformed head repository owner")
        owner_login = head_owner["login"]
        if git_ops.split_owner_repo(f"{owner_login}/repository") is None:
            raise GitError("invalid PR row: malformed head repository owner")
    head_repo = row["headRepository"]
    if head_repo is None:
        return None
    if not isinstance(head_repo, dict):
        raise GitError("invalid PR row: headRepository must be an object or null")
    repo_name = head_repo.get("name")
    if "name" in head_repo and (
        not isinstance(repo_name, str)
        or git_ops.split_owner_repo(f"owner/{repo_name}") is None
    ):
        raise GitError("invalid PR row: malformed head repository name")
    if "nameWithOwner" in head_repo:
        name_with_owner = head_repo["nameWithOwner"]
        if not isinstance(name_with_owner, str):
            raise GitError("invalid PR row: head repository nameWithOwner must be a string")
        if name_with_owner != "":
            slug = git_ops.split_owner_repo(name_with_owner)
            if slug is None:
                raise GitError("invalid PR row: malformed head repository slug")
            if (
                (owner_login is not None and slug[0].casefold() != owner_login.casefold())
                or (isinstance(repo_name, str) and slug[1].casefold() != repo_name.casefold())
            ):
                raise GitError("invalid PR row: contradictory head repository identity")
            return name_with_owner
    if owner_login is not None and isinstance(repo_name, str):
        return f"{owner_login}/{repo_name}"
    raise GitError("invalid PR row: unavailable head repository slug requires owner and name")


def _pr_info_from_row(
    target_dir: Path, row: dict[str, Any], *, auth: GitHubAuth = INHERIT_GITHUB_AUTH
) -> PRInfo:
    """Resolve the base posting repository and fork-aware head link.

    Raise GitError for invalid PR metadata, repository context, or local merge base."""
    from daydream.archive.git_safe import normalize_remote_url

    number = row.get("number")
    head_sha = row.get("headRefOid")
    head_ref = row.get("headRefName")
    base_ref = row.get("baseRefName")
    url = row.get("url")
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise GitError("invalid PR row: number must be a positive integer")
    if not isinstance(head_sha, str):
        raise GitError("invalid PR row: headRefOid must be a string")
    if not isinstance(head_ref, str) or not head_ref:
        raise GitError("invalid PR row: headRefName must be a non-empty string")
    if not isinstance(base_ref, str) or not base_ref:
        raise GitError("invalid PR row: baseRefName must be a non-empty string")
    if not isinstance(url, str) or not url:
        raise GitError("invalid PR row: url must be a non-empty string")
    head_repo = _head_repo_slug_from_row(row)
    git_ops.validate_branch_name(target_dir, head_ref)
    git_ops.validate_branch_name(target_dir, base_ref)

    owner, repo = git_ops.gh_repo_view_required(target_dir, auth=auth)
    base_slug = f"{owner}/{repo}"
    matching_remotes: list[str] = []
    for remote, raw_url in git_ops.remote_urls(target_dir).items():
        identity, _safe_url = normalize_remote_url(raw_url)
        if identity is not None and identity.lower() == base_slug.lower():
            matching_remotes.append(f"refs/remotes/{remote}/{base_ref}")
    base_sha = git_ops.resolve_pr_merge_base(
        target_dir,
        matching_remotes,
        f"refs/heads/{base_ref}",
        head_sha,
    )
    return PRInfo(
        number=number,
        head_sha=head_sha,
        base_sha=base_sha,
        base_ref=base_ref,
        head_ref=head_ref,
        owner=owner,
        repo=repo,
        url=url,
        head_repo=head_repo,
        pr_base_sha=row.get("baseRefOid") if isinstance(row.get("baseRefOid"), str) else None,
    )


def capture_pr_base_tip(
    target_dir: Path, pr: PRInfo, *, auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> str | None:
    """Read optional base-tip evidence without requiring newer gh JSON fields.

    A PR that advanced between reads cannot supply evidence for this snapshot.
    The required identity and merge base already come from the initial lookup.
    """
    if pr.pr_base_sha is not None:
        return pr.pr_base_sha
    try:
        data = git_ops.gh_api(target_dir, f"repos/{pr.owner}/{pr.repo}/pulls/{pr.number}", auth=auth)
    except GitError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("head"), dict):
        return None
    if data["head"].get("sha") != pr.head_sha or not isinstance(data.get("base"), dict):
        return None
    sha = data["base"].get("sha")
    return sha if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}", sha) else None


def find_open_pr(
    target_dir: Path, *, auth: GitHubAuth = INHERIT_GITHUB_AUTH
) -> PRInfo | None:
    """Find the branch's open PR; return None for absence and raise GitError on failure."""
    branch = git_ops.current_branch(target_dir)
    if not branch:
        return None
    rows = git_ops.gh_pr_list_for_branch(target_dir, branch, auth=auth)
    if not rows:
        return None
    return _pr_info_from_row(target_dir, rows[0], auth=auth)


def find_pr_by_number(
    target_dir: Path, pr_number: int, *, auth: GitHubAuth = INHERIT_GITHUB_AUTH
) -> PRInfo | None:
    """Find an explicit PR without branch fallback; raise GitError on lookup failure."""
    data = git_ops.gh_pr_view(target_dir, pr_number, auth=auth)
    if data is None:
        return None
    return _pr_info_from_row(target_dir, data, auth=auth)


_ANCHOR_TOKEN = re.compile(r"`([^`\n]{3,80})`|\b([A-Za-z_][A-Za-z0-9_]{4,})\b")


def extract_anchors(text: str, *, prefer_quoted: bool = False) -> list[str]:
    """Select at most eight unique anchor tokens, breaking length ties by first appearance.

    The default longest-first ordering is frozen into cross-run fingerprints.
    prefer_quoted prioritizes backtick identifiers for line resolution, so longer
    prose words cannot crowd them out of the cap."""
    seen: list[str] = []
    quoted: set[str] = set()
    for m in _ANCHOR_TOKEN.finditer(text):
        token = m.group(1) or m.group(2)
        if not token:
            continue
        if m.group(1):
            quoted.add(token)
        if token not in seen:
            seen.append(token)
    if prefer_quoted:
        # Stable sort, so first-appearance order still breaks length ties.
        seen.sort(key=lambda t: (t not in quoted, -len(t)))
    else:
        # Longest-first improves hit quality (generic words lose to identifiers).
        seen.sort(key=len, reverse=True)
    return seen[:8]


def compute_fingerprint(path: str, description: str, rationale: str) -> str:
    """Hash path, normalized prose, and sorted anchors into a cross-run identity.

    Line numbers and rendered severity/confidence badges are deliberately excluded.
    Prose preserves word order; anchor order is immaterial."""
    normalized_description = " ".join(description.strip().lower().split())
    normalized_rationale = " ".join(rationale.strip().lower().split())
    canonical = "\n".join(
        [
            path,
            normalized_description,
            "\n".join(sorted(extract_anchors(f"{description}\n{rationale}"))),
            normalized_rationale,
        ]
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _in_hunk(line: int, hunks: list[tuple[int, int]]) -> bool:
    """Whether ``line`` falls inside any head-side ``(start, end)`` hunk range."""
    return any(start <= line <= end for start, end in hunks)


def resolve_line(
    target_dir: Path,
    head_sha: str,
    issue: ParsedIssue,
    hunks: list[tuple[int, int]] | None = None,
) -> int | None:
    """Locate a finding at head: preserve valid in-hunk hints, then search anchors.

    Out-of-hunk hints require an anchor within five lines. Whole-file search prefers
    in-hunk matches, using an out-of-hunk match only when no changed-line match exists.
    Missing files, empty files, and absent anchors return None."""
    try:
        raw = git_ops.show(target_dir, head_sha, issue.path)
    except GitError:
        return None
    lines = raw.decode(errors="replace").splitlines()
    if not lines:
        return None

    ranges = hunks or []

    # Step 1: an in-hunk hint is authoritative -- pass it straight through.
    if issue.line is not None and 1 <= issue.line <= len(lines) and _in_hunk(issue.line, ranges):
        return issue.line

    anchors = extract_anchors(f"{issue.title}\n{issue.body}", prefer_quoted=True)

    # Step 2: verify an out-of-hunk hint against nearby anchors.
    if issue.line is not None and 1 <= issue.line <= len(lines):
        lo = max(1, issue.line - 5)
        hi = min(len(lines), issue.line + 5)
        for anchor in anchors:
            if any(anchor in lines[i - 1] for i in range(lo, hi + 1)):
                return issue.line
        # Hint didn't verify; fall through to full-file search.

    # Step 3: full-file search, in-hunk hits first.
    out_of_hunk: int | None = None
    for anchor in anchors:
        for i, line in enumerate(lines, start=1):
            if anchor not in line:
                continue
            if _in_hunk(i, ranges):
                return i
            if out_of_hunk is None:
                out_of_hunk = i

    return out_of_hunk


# Splits a unified diff on each `diff --git` header so we can pick out the
# block for a single file from a full-PR diff.
_DIFF_BLOCK_SPLIT = re.compile(r"(?m)^(?=diff --git )")

# Max distance (in lines) from a diff-hunk boundary that still counts as
# "within" the hunk for PR-comment placement.
HUNK_TOLERANCE: int = 3


def file_hunks(
    target_dir: Path,
    base_sha: str,
    head_sha: str,
    path: str,
    *,
    pr_number: int | None = None,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> list[tuple[int, int]]:
    """Read inclusive head-side hunk ranges, falling back to gh for unreachable bases.

    The GitHub fallback selects this path's block from the full PR diff."""
    try:
        diff_text = git_ops.diff_paths(
            target_dir, base_sha, head_sha, [path], unified=3, merge_base_diff=False
        )
    except GitError:
        diff_text = (
            _gh_pr_diff_for_path(target_dir, pr_number, path, auth=auth)
            if pr_number is not None else ""
        )
    return _parse_hunks(diff_text)


def _gh_pr_diff_for_path(
    target_dir: Path,
    pr_number: int,
    path: str,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> str:
    """Fetch the PR's full diff via `gh pr diff` and return just the block for `path`."""
    try:
        full_diff = git_ops.gh_pr_diff(target_dir, pr_number, auth=auth)
    except GitError:
        return ""
    # Pick the `diff --git a/<path> b/<path>` block.
    needle_a = f"a/{path} "
    needle_b = f"b/{path}\n"
    for block in _DIFF_BLOCK_SPLIT.split(full_diff):
        if not block.startswith("diff --git "):
            continue
        header_line = block.split("\n", 1)[0]
        if (
            needle_a in header_line
            or header_line.endswith(f"b/{path}")
            or needle_b in header_line
        ):
            return block
    return ""


_DIFF_GIT_HEADER = re.compile(r"(?m)^diff --git a/.+ b/(.+)$")


def pr_changed_files(
    target_dir: Path, pr: PRInfo, *, auth: GitHubAuth = INHERIT_GITHUB_AUTH
) -> set[str]:
    """Read head-side changed paths, falling back to gh when local diff is unavailable.

    This set gates file comments: GitHub rejects paths outside the PR diff."""
    changed = set(git_ops.diff_name_only(target_dir, pr.base_sha, pr.head_sha))
    if changed:
        return changed
    try:
        full_diff = git_ops.gh_pr_diff(target_dir, pr.number, auth=auth)
    except GitError:
        return set()
    return set(_DIFF_GIT_HEADER.findall(full_diff))


def _parse_hunks(diff_text: str) -> list[tuple[int, int]]:
    """Read inclusive head-side ranges using the shared unified-diff parser."""
    from daydream.hunk_index import head_side_ranges, parse_hunks

    return head_side_ranges(parse_hunks(diff_text))


def snap_to_hunk(
    line: int, hunks: list[tuple[int, int]], tolerance: int = HUNK_TOLERANCE
) -> int | None:
    """Preserve in-hunk lines; snap nearby lines to the nearest boundary within tolerance.

    The pre-report location validator owns citation authority. This final placement
    check uses the live diff so GitHub receives a valid changed line."""
    # Shared two-sided boundary-distance primitive (same as the pre-report
    # validator) so posting and the validator agree on what ``in hunk`` /
    # ``near boundary`` means (issue #745).
    from daydream.hunk_index import range_distance

    best: int | None = None
    best_dist = tolerance + 1
    for start, end in hunks:
        if start <= line <= end:
            return line
        dist = range_distance(line, start, end)
        candidate = start if line < start else end
        if dist <= tolerance and dist < best_dist:
            best = candidate
            best_dist = dist
    return best


def classify(
    target_dir: Path,
    pr: PRInfo,
    issues: list[ParsedIssue],
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    renderers: ReviewRenderers | None = None,
    snapshot_diff: str | None = None,
) -> ClassifiedIssues:
    """Place findings inline, then at file level, then in the review body.

    Only changed files accept file comments; body-only findings have no repliable
    thread for the labeler. Inline relocation annotates the caller-owned issue body."""
    renderers = renderers if renderers is not None else resolve_review_renderers(get_registry())
    out = ClassifiedIssues()
    hunks_cache: dict[str, list[tuple[int, int]]] = {}
    if snapshot_diff is not None:
        from daydream.hunk_index import head_side_ranges_by_file, parse_hunks

        hunks_cache = head_side_ranges_by_file(parse_hunks(snapshot_diff))
        changed_files = set(git_ops.diff_name_only_strict(target_dir, pr.base_sha, pr.head_sha))
    else:
        changed_files = pr_changed_files(target_dir, pr, auth=auth)

    def _unplaced(issue: ParsedIssue) -> None:
        if issue.path in changed_files:
            out.file_level.append(issue)
        else:
            out.body_only.append(issue)

    for issue in issues:
        if issue.is_cross_stack:
            _unplaced(issue)
            continue
        # Skip the file_hunks() diff lookup -- a git-diff subprocess call with
        # a gh-pr-diff network fallback on GitError -- for a path that cannot
        # resolve at head_sha at all (e.g. deleted or renamed away in this
        # PR). resolve_line would reject such a path via this same git show
        # regardless of what hunks it was handed, so there is nothing for the
        # hunk lookup to buy here.
        try:
            git_ops.show(target_dir, pr.head_sha, issue.path)
        except GitError:
            _unplaced(issue)
            continue
        # The hunk ranges are resolved BEFORE line resolution, not after: they
        # are what lets `resolve_line` pass an already-valid in-hunk line
        # through untouched instead of re-deriving it from prose (issue #1102).
        if issue.path not in hunks_cache:
            hunks_cache[issue.path] = ([] if snapshot_diff is not None else file_hunks(
                target_dir,
                pr.base_sha,
                pr.head_sha,
                issue.path,
                pr_number=pr.number,
                auth=auth,
            ))
        hunks = hunks_cache[issue.path]
        line = resolve_line(target_dir, pr.head_sha, issue, hunks)
        if line is None:
            _unplaced(issue)
            continue
        snapped = snap_to_hunk(line, hunks)
        if snapped is None:
            _unplaced(issue)
            continue
        _note_relocation(issue, snapped)
        out.inline.append(_inline_comment(issue, snapped, renderers))
        out.inline_issues.append(issue)
    return out


# Marks the posting-time relocation annotation so it is appended at most once
# even if a caller classifies the same issue objects twice.
_PLACEMENT_NOTE_PREFIX = "**Placement:** posted on line "


def _note_relocation(issue: ParsedIssue, posted_line: int) -> None:
    """Record a changed positive citation once without changing its fingerprint.

    The body annotation also reaches findings artifacts. Fingerprints use the original
    description/rationale, and zero/negative whole-file sentinels are not citations."""
    if issue.line is None or issue.line <= 0 or issue.line == posted_line:
        return
    if _PLACEMENT_NOTE_PREFIX in issue.body:
        return
    note = f"{_PLACEMENT_NOTE_PREFIX}{posted_line}; reviewer cited line {issue.line}."
    issue.body = f"{issue.body}\n\n{note}" if issue.body else note


def _inline_comment(issue: ParsedIssue, line: int, renderers: ReviewRenderers) -> InlineReviewComment:
    """Build one immutable inline review comment for the review payload."""
    return InlineReviewComment(
        path=issue.path,
        line=line,
        side="RIGHT",
        body=format_comment_body(issue, "inline", renderers),
    )


@dataclass(frozen=True)
class ClassifiedReviewPlan:
    """Immutable, authorized input to the shared review write operation."""

    pr: PRInfo
    inline: tuple[InlineReviewComment, ...]
    inline_issues: tuple[ParsedIssue, ...]
    file_level: tuple[ParsedIssue, ...]
    body_only: tuple[ParsedIssue, ...]
    event: ReviewEvent
    run_info: str
    renderers: ReviewRenderers
    diagram_blocks: str | None
    review_warnings: tuple[str, ...] = ()

    @classmethod
    def from_classified(
        cls,
        pr: PRInfo,
        classified: ClassifiedIssues,
        *,
        event: ReviewEvent,
        run_info: str,
        renderers: ReviewRenderers,
        diagram_blocks: str | None = None,
        review_warnings: tuple[str, ...] = (),
    ) -> ClassifiedReviewPlan:
        """Snapshot a mutable classified review after the caller authorizes it."""
        return cls(
            pr=pr,
            inline=tuple(classified.inline),
            inline_issues=tuple(replace(issue) for issue in classified.inline_issues),
            file_level=tuple(replace(issue) for issue in classified.file_level),
            body_only=tuple(replace(issue) for issue in classified.body_only),
            event=event,
            run_info=run_info,
            renderers=renderers,
            diagram_blocks=diagram_blocks,
            review_warnings=review_warnings,
        )


class ReviewTransport(Protocol):
    """Explicit capability for the two kinds of GitHub review writes."""

    def post_file_comment(self, pr: PRInfo, payload: FileCommentPayload) -> bool: ...

    def post_review(self, pr: PRInfo, payload: ReviewPayload) -> ReviewPostResult: ...


def _file_comment_payload_dict(payload: FileCommentPayload) -> dict[str, Any]:
    return asdict(payload)


def _review_payload_dict(payload: ReviewPayload) -> dict[str, Any]:
    return {
        "event": payload.event.value,
        "commit_id": payload.commit_id,
        "body": payload.body,
        "comments": [asdict(comment) for comment in payload.comments],
    }


@dataclass(frozen=True)
class GitHubReviewTransport:
    """GitHub review writes bound to one repository checkout and auth source."""

    target_dir: Path
    auth: GitHubAuth = field(repr=False)

    def post_file_comment(self, pr: PRInfo, payload: FileCommentPayload) -> bool:
        endpoint = f"/repos/{pr.owner}/{pr.repo}/pulls/{pr.number}/comments"
        try:
            git_ops.gh_api(
                self.target_dir,
                endpoint,
                method="POST",
                input_data=_file_comment_payload_dict(payload),
                auth=self.auth,
            )
        except GitError:
            return False
        return True

    def post_review(self, pr: PRInfo, payload: ReviewPayload) -> ReviewPostResult:
        endpoint = f"/repos/{pr.owner}/{pr.repo}/pulls/{pr.number}/reviews"
        try:
            data = git_ops.gh_api(
                self.target_dir,
                endpoint,
                method="POST",
                input_data=_review_payload_dict(payload),
                auth=self.auth,
            )
        except GitError as exc:
            safe_error = "GitHub review submission failed"
            if exc.preserved_payload_path is not None:
                safe_error += (
                    " (request payload preserved at "
                    f"{exc.preserved_payload_path})"
                )
            return ReviewPostResult(review_url=None, safe_error=safe_error)
        if not isinstance(data, dict):
            return ReviewPostResult(review_url=None, safe_error=None)
        url = data.get("html_url")
        return ReviewPostResult(
            review_url=str(url) if url else None,
            safe_error=None,
        )


# Severities that must never let an opted-in review post as an approval.
# Fails closed: any string outside ``_NON_BLOCKING_SEVERITIES`` blocks. The
# findings schema permits arbitrary strings, so unknown/off-vocabulary labels
# ("critical", "blocker", "major", ...) must conservatively block approval.
_NON_BLOCKING_SEVERITIES = frozenset({"low"})


def _severity_blocks_approval(severity: str | None) -> bool:
    """Unknown severities block approval; low and omitted severities do not."""
    return severity is not None and severity.lower() not in _NON_BLOCKING_SEVERITIES


def _finding_blocks_approval(
    severity: str | None,
    location_distrust: bool,
    severity_off_vocabulary: bool = False,
    severity_before_demotion: str | None = None,
) -> bool:
    """Block on asserted unknown severity or a blocking severity before location demotion."""
    return (
        severity_off_vocabulary
        or _severity_blocks_approval(severity)
        or (location_distrust and _severity_blocks_approval(severity_before_demotion))
    )


def _is_clean_review(classified: ClassifiedIssues, approve_on_clean: bool) -> bool:
    """Approve only when explicitly enabled and every finding passes the severity gate."""
    if not approve_on_clean:
        return False
    return not any(
        _finding_blocks_approval(
            issue.severity,
            issue.location_distrust,
            issue.severity_off_vocabulary,
            issue.severity_before_demotion,
        )
        for issue in classified.all_issues()
    )


def post_classified_review(
    plan: ClassifiedReviewPlan,
    *,
    transport: ReviewTransport,
) -> ClassifiedReviewResult:
    """Submit ordered file comments, fold failures, then post one final review."""
    posted: list[ParsedIssue] = []
    folded: list[ParsedIssue] = []
    for finding in plan.file_level:
        payload = FileCommentPayload(
            commit_id=plan.pr.head_sha,
            path=finding.path,
            subject_type="file",
            body=format_comment_body(finding, "file_level", plan.renderers),
        )
        if transport.post_file_comment(plan.pr, payload):
            posted.append(finding)
        else:
            folded.append(finding)

    final_classified = ClassifiedIssues(
        inline=list(plan.inline),
        inline_issues=list(plan.inline_issues),
        file_level=list(posted),
        body_only=[*plan.body_only, *folded],
    )
    review_payload = build_payload_for_event(
        plan.pr,
        final_classified,
        event=plan.event,
        run_info=plan.run_info,
        renderers=plan.renderers,
        diagram_blocks=plan.diagram_blocks,
        review_warnings=plan.review_warnings,
    )
    review_result = transport.post_review(plan.pr, review_payload)
    posted_review = review_result.review_url is not None
    return ClassifiedReviewResult(
        status=SubmissionStatus.POSTED if posted_review else SubmissionStatus.FAILED,
        review_url=review_result.review_url,
        posted_file_level=tuple(posted),
        folded_file_level=tuple(folded),
        final_review_posted=posted_review,
        safe_error=None if posted_review else review_result.safe_error,
    )


def _resolve_pr(
    target_dir: Path,
    console: Console,
    pr_number: int | None,
    *,
    auth: GitHubAuth,
) -> PRInfo | None:
    """Resolve the pinned PR or current branch; warn on absence, propagate lookup failures."""
    if pr_number is not None:
        pr = find_pr_by_number(target_dir, pr_number, auth=auth)
        if pr is None:
            print_warning(
                console,
                f"PR #{pr_number} not found via `gh pr view`; skipping PR post.",
            )
            return None
        return pr
    pr = find_open_pr(target_dir, auth=auth)
    if pr is None:
        print_warning(
            console,
            "No open PR found for the current branch; skipping PR post.",
        )
        return None
    return pr


@bind_resolved_run_context
async def _post(
    target_dir: Path,
    issues: list[ParsedIssue],
    *,
    console: Console,
    run_info: str,
    renderers: ReviewRenderers,
    post: bool = False,
    approve_on_clean: bool = False,
    pr_number: int | None = None,
    diagram_blocks: str | None = None,
    review_warnings: tuple[str, ...] = (),
    run_context: RunContext | None = None,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> PostStatus:
    run_context = resolve_run_context(run_context)
    try:
        pr = _resolve_pr(target_dir, console, pr_number, auth=auth)
    except GitError as exc:
        print_error(console, "PR Lookup Failed", str(exc))
        return PostStatus.FAILED
    if pr is None:
        return PostStatus.NO_PR

    classified = classify(target_dir, pr, issues, auth=auth, renderers=renderers)
    if (
        classified.is_empty()
        and not approve_on_clean
        and not review_warnings
        and not (diagram_blocks and diagram_blocks.strip())
    ):
        print_info(
            console, "No postable issues after classification; skipping PR post."
        )
        return PostStatus.NOTHING_TO_POST

    inline_files = sorted({c.path for c in classified.inline})
    summary = (
        f"{len(classified.inline)} inline on "
        f"{', '.join(inline_files) if inline_files else '(none)'}, "
        f"{len(classified.file_level)} file-level, "
        f"{len(classified.body_only)} folded into body"
    )
    clean = not review_warnings and _is_clean_review(classified, approve_on_clean)
    event_note = " — will post event: APPROVE" if clean else ""
    print_info(console, f"PR #{pr.number}: {summary}{event_note}")

    if not post and not run_context.confirm(
        safe_default=False,
        question=(
            "Post an APPROVE review for this clean PR? [y/N]"
            if clean
            else "Post these as a PR review? [y/N]"
        ),
        default="n",
        console=console,
    ):
        print_info(console, "Skipped posting to PR.")
        return PostStatus.NOTHING_TO_POST

    plan = ClassifiedReviewPlan.from_classified(
        pr,
        classified,
        event=ReviewEvent.APPROVE if clean else ReviewEvent.COMMENT,
        run_info=run_info,
        renderers=renderers,
        diagram_blocks=diagram_blocks,
        review_warnings=review_warnings,
    )
    result = post_classified_review(
        plan,
        transport=GitHubReviewTransport(target_dir=target_dir, auth=auth),
    )
    if result.folded_file_level:
        print_warning(
            console,
            f"{len(result.folded_file_level)} file-level comment(s) failed to post; "
            "folded into the review body.",
        )

    if result.status is SubmissionStatus.FAILED:
        suffix = f" ({result.safe_error})" if result.safe_error else ""
        # File-level comments post before the review, so some may already be
        # live on the PR — saying "no comments were posted" would be false.
        already = (
            f" {len(result.posted_file_level)} file-level comment(s) were already posted."
            if result.posted_file_level
            else " No comments were posted."
        )
        print_warning(console, f"Failed to post PR review;{already}{suffix}")
        return PostStatus.FAILED

    print_success(console, f"Posted review: {result.review_url}")
    return PostStatus.POSTED


def post_findings_from_artifact(
    target_dir: Path,
    artifact_path: Path,
    *,
    pr_number: int,
    head_sha: str,
    repo: str,
    console: Console,
    bot_login: str | None = None,
    approve_on_clean: bool = False,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> int:
    """Validate artifact identity against event facts, reconcile bot comments, and post.

    No agent work or prompting occurs here. Explicit bot_login overrides the environment;
    missing identity disables REST dedup. All current findings, including already posted
    matches, participate in the approval gate. Invalid optional diagrams are dropped;
    a standalone diagram artifact fails if its only deliverable is invalid.
    Return 0 for success/no new findings, 1 for rejected artifacts or GitHub failures."""
    # Late imports: ``findings`` and ``reconcile`` both import this module at
    # module level (one-way by design), so the poster flow resolves them at
    # call time — the same no-cycle pattern as ``daydream.deep.orchestrator``.
    from daydream.findings import FindingsValidationError, load_findings_artifact
    from daydream.reconcile import fetch_prior_findings, partition, resolve_threads

    # Resolve the effective bot login in ONE place (here), so both the CLI
    # and any library caller get the env fallback. Precedence: explicit param
    # over $DAYDREAM_BOT_HANDLE. An unresolved login degrades dedup safely.
    effective_login = bot_login or os.environ.get("DAYDREAM_BOT_HANDLE") or None
    if effective_login is None:
        print_warning(
            console,
            "BOT_LOGIN_UNRESOLVED: no --bot-login and $DAYDREAM_BOT_HANDLE is unset; "
            "prior-finding dedup is degraded (GraphQL still protected by viewerDidAuthor; "
            "REST dedup unavailable) — may double-post, will never suppress.",
        )

    try:
        artifact = load_findings_artifact(
            artifact_path,
            expected_repo=repo,
            expected_pr_number=pr_number,
            expected_head_sha=head_sha,
        )
    except FindingsValidationError as exc:
        print_error(console, "Findings Artifact Rejected", str(exc))
        return 1

    # The synthetic PRInfo depends only on this function's own arguments, so it
    # is built once here -- above the diagram branch, which needs it too.
    owner, repo_name = repo.split("/", 1)
    pr = PRInfo(
        number=pr_number,
        head_sha=head_sha,
        base_sha="",
        base_ref="",
        head_ref="",
        owner=owner,
        repo=repo_name,
        url="",
    )

    # Issue #1113: a diagram artifact carries no fingerprints, so it must never
    # reach the reconcile path below -- ``partition([], prior)`` would classify
    # EVERY prior review finding as stale and ``resolve_threads`` would minimize
    # the bot's real open findings. Branch before ``fetch_prior_findings``.
    if artifact.kind == "diagram":
        return post_diagram_artifact(
            artifact,
            pr,
            target_dir,
            console=console,
            bot_login=effective_login,
            auth=auth,
        )

    diagram_blocks: str | None = None
    if isinstance(artifact.diagrams, dict):
        problem = validate_diagram_payload(
            artifact.diagrams,
            target_dir=target_dir,
            head_sha=pr.head_sha,
            repo_slug=repo,
            auth=auth,
        )
        # Issue #1176: a diagram problem must not discard findings that passed
        # their own schema, fingerprint and event-fact validation. Dropping the
        # diagram already keeps it off GitHub, which is the whole of what the
        # confused-deputy gate requires; exiting 1 only adds the lost comment.
        if problem is not None:
            print_warning(console, f"Diagram dropped (the findings are still posted): {problem}")
        else:
            diagram_blocks = render_diagram_blocks_from_payload(artifact.diagrams) or None

    try:
        prior = fetch_prior_findings(
            target_dir, repo, pr_number, bot_login=effective_login, auth=auth
        )
    except GitError as exc:
        print_error(console, "Prior-Finding Inventory Failed", str(exc))
        return 1

    review_warnings = artifact.review_warnings + artifact.coverage_notice
    plan = partition([f.fingerprint for f in artifact.findings], prior)
    if plan.stale and artifact.analysis_complete and not review_warnings:
        resolved, failed = resolve_threads(target_dir, plan.stale, auth=auth)
        print_info(console, f"Stale findings minimized: {resolved} succeeded, {failed} failed.")

    new_fingerprints = set(plan.new)
    renderers = resolve_review_renderers(get_registry())
    classified = ClassifiedIssues()
    for finding in artifact.findings:
        if finding.fingerprint not in new_fingerprints:
            continue
        issue = _issue_from_artifact_finding(finding)
        if finding.placement == "inline" and finding.line is not None:
            classified.inline.append(_inline_comment(issue, finding.line, renderers))
            classified.inline_issues.append(issue)
        elif finding.placement == "file":
            classified.file_level.append(issue)
        else:
            classified.body_only.append(issue)

    # The approval decision must fail closed over EVERY finding the current
    # review still carries — not just the ones being posted this run. A
    # high-severity finding already posted (matched) and still live on the PR
    # is invisible to ``classified`` (built from new fingerprints only), so
    # without this a new low-only batch could post APPROVE over the bot's own
    # open high finding (#343 R2 F2b). Matched findings are never re-posted
    # as comments — only their severities count here.
    can_approve = approve_on_clean and artifact.analysis_complete and not review_warnings and not any(
        _finding_blocks_approval(
            finding.severity,
            finding.location_distrust,
            finding.severity_off_vocabulary,
            finding.severity_before_demotion,
        )
        for finding in artifact.findings
    )

    if classified.is_empty() and not can_approve and diagram_blocks is None and not review_warnings:
        print_info(
            console,
            f"No new findings to post ({len(plan.matched)} already on PR #{pr_number}).",
        )
        return 0

    submission_plan = ClassifiedReviewPlan.from_classified(
        pr,
        classified,
        event=ReviewEvent.APPROVE if can_approve else ReviewEvent.COMMENT,
        run_info=(
            artifact.run_info if artifact.run_info is not None else render_run_info(())
        ),
        renderers=renderers,
        diagram_blocks=diagram_blocks,
        review_warnings=review_warnings,
    )
    result = post_classified_review(
        submission_plan,
        transport=GitHubReviewTransport(target_dir=target_dir, auth=auth),
    )
    if result.folded_file_level:
        print_info(
            console,
            f"{len(result.folded_file_level)} file-level comment(s) failed to post; "
            "folded into the review body.",
        )

    if result.status is SubmissionStatus.FAILED:
        suffix = f" ({result.safe_error})" if result.safe_error else ""
        already = (
            f"{len(result.posted_file_level)} file-level comment(s) were already posted."
            if result.posted_file_level
            else "No comments were posted."
        )
        print_error(console, "PR Review Post Failed", f"{already}{suffix}")
        return 1
    print_success(console, f"Posted review: {result.review_url}")
    return 0


def _issue_from_artifact_finding(finding: ArtifactFinding) -> ParsedIssue:
    """Restore validated issue fields; artifact placement requires no local PR Git objects."""
    return ParsedIssue(**{member.name: getattr(finding, member.name) for member in fields(ParsedIssue)})
