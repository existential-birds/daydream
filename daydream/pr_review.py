"""Post daydream review findings as inline comments on a target PR.

Shared by deep-review mode (reads canonical `merged-items.json`), comment
mode (`--comment`) (consumes alt-review issues directly), and
`scripts/redrive_post.py` (a thin CLI over the same canonical
`merged-items.json` that a redrive must post).

Flow:
    1. Locate the target PR: by explicit number via `gh pr view` when
       `pr_number` is supplied (consumer: `scripts/redrive_post.py`), else
       the current branch's open PR via `gh pr list`.
    2. Parse issues (from canonical merged items or alt-issue dicts).
    3. Resolve each issue to a real head-SHA line via anchor grep.
    4. Classify into inline (line within a diff hunk), file-level (file in
       the diff but no line home), or body-only (last resort).
    5. Render comment bodies, embedding a hidden `daydream-finding` marker
       per fingerprinted issue (cross-run dedup; see `finding_marker`).
    6. Build a single review payload, show a summary, ask y/n.
    7. On yes, POST to `/repos/<owner>/<repo>/pulls/<num>/reviews`.

Everything is best-effort: failures warn and return, never raise.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonschema

import daydream
from daydream import git_ops
from daydream.agent import get_assume, get_non_interactive, resolve_or_prompt
from daydream.config import DIAGRAM_KINDS
from daydream.extensions import (
    CommentFinding,
    FindingRenderContext,
    SummaryContext,
    SummaryFinding,
    get_registry,
)
from daydream.git_ops import GitError
from daydream.pr_comment_renderer import render_run_info_block
from daydream.severity import normalize_severity
from daydream.trajectory import TrajectoryRecorder, get_current_recorder
from daydream.ui import print_error, print_info, print_success, print_warning

if TYPE_CHECKING:
    from rich.console import Console

    from daydream.findings import ArtifactFinding, FindingsArtifact


_logger = logging.getLogger(__name__)


# --- Data shapes ------------------------------------------------------------


class PostStatus(Enum):
    """Outcome of a PR-post attempt.

    The deep flow treats every non-``POSTED`` state as warn-and-continue;
    comment mode (``--comment``) treats ``NO_PR`` and ``FAILED`` as a failed
    run because posting is its deliverable (#8).
    """

    POSTED = "posted"
    NOTHING_TO_POST = "nothing-to-post"
    NO_PR = "no-pr"
    FAILED = "failed"


@dataclass
class ParsedIssue:
    """One issue to evaluate for PR posting.

    Attributes:
        path: File path relative to repo root.
        line: Line hint from the source, if any. May be stale.
        title: Short issue title (first line of the body).
        body: Full issue body (rationale + recommendation).
        is_cross_stack: True when the issue spans multiple stacks.
        confidence: Normalised HIGH / MEDIUM / LOW, if known.
        severity: Normalised high / medium / low, if known.
        fingerprint: Deterministic SHA256 identity for cross-run dedup. Set on
            canonical merged findings and alt-review issues; None on other
            construction paths.
        location_distrust: True when location validation demoted this finding
            (its citation was beyond tolerance), issue #972 R2. Renders a
            demotion note; blocks approval only when the finding was demoted
            from a blocking original severity (see ``severity_before_demotion``).
        severity_before_demotion: The original severity before location-
            validation demotion, if any; the approval gate compares it against
            the blocking set so an initially-low or never-asserted severity
            stays non-blocking despite the demotion mark.
        severity_off_vocabulary: True when this issue carried a present severity
            string outside the canonical vocabulary (e.g. ``"critical"``). The
            boundary folds such labels into ``None`` for ``severity`` (so the
            canonical render path stays clean), but the gate must still fail
            closed on them rather than read them as an omitted severity.
    """

    path: str
    line: int | None
    title: str
    body: str
    is_cross_stack: bool = False
    confidence: str | None = None
    severity: str | None = None
    fingerprint: str | None = None
    location_distrust: bool = False
    severity_before_demotion: str | None = None
    severity_off_vocabulary: bool = False


@dataclass
class PRInfo:
    """Details about the open PR for the current branch."""

    number: int
    head_sha: str
    base_sha: str
    base_ref: str
    owner: str
    repo: str
    url: str
    # Slug of the PR's head repository (``owner/repo``) — the fork when the
    # PR head lives in a fork. Used ONLY for the reviewed-commit link, which
    # must point at the repo that actually holds the head commit; comments
    # always post to the base repo via ``owner``/``repo``.
    head_repo: str | None = None


@dataclass(frozen=True)
class ItemFields:
    """Named fields extracted from a single canonical merged item.

    Returned by :func:`extract_item_fields` in place of the former positional
    7-tuple so callers can use attribute access instead of positional unpacking.
    """

    path: str
    line_int: int | None
    description: str
    rationale: str
    severity: str | None
    confidence: str | None
    is_cross_stack: bool
    location_distrust: bool = False
    severity_before_demotion: str | None = None
    severity_off_vocabulary: bool = False


@dataclass
class _ClassifiedIssues:
    inline: list[dict[str, Any]] = field(default_factory=list)
    body_only: list[ParsedIssue] = field(default_factory=list)
    # Parallel list to `inline`: the original ParsedIssue for each inline
    # comment. Used to roll severity/confidence into the summary body.
    inline_issues: list[ParsedIssue] = field(default_factory=list)
    # Findings with no diff-line home whose file is still part of the PR
    # diff. Posted as file-level review comments so they land in
    # `/pulls/{n}/comments` as repliable threads the labeler can read back.
    file_level: list[ParsedIssue] = field(default_factory=list)

    def is_empty(self) -> bool:
        """True when nothing would be posted in any placement.

        Checks ``inline`` (the rendered comment dicts) rather than
        ``inline_issues``, so the guard holds even if the two parallel lists
        ever drift.
        """
        return not (self.inline or self.file_level or self.body_only)

    def total(self) -> int:
        """Count of findings across every placement."""
        return len(self.inline) + len(self.file_level) + len(self.body_only)

    def all_issues(self) -> list[ParsedIssue]:
        """Every classified :class:`ParsedIssue`, for severity/confidence rollups."""
        return [*self.inline_issues, *self.file_level, *self.body_only]


# --- Public entry points ----------------------------------------------------


async def post_review_to_pr_from_report(
    target_dir: Path,
    merged_items_path: Path,
    *,
    console: Console,
    post: bool = False,
    approve_on_clean: bool = False,
    pr_number: int | None = None,
    diagram_blocks: str | None = None,
) -> PostStatus:
    """Read canonical `merged-items.json` and offer to post to the PR.

    Builds issues from the canonical item list (every lens, including
    structural) via :func:`parsed_issues_from_items` rather than re-parsing
    the rendered markdown — the regex parser silently dropped structural
    findings, which live under ``## Structural Review``.

    ``post=True`` bypasses the interactive confirm gate (comment mode, #330).

    ``approve_on_clean=True`` (issue #343) opts into posting
    ``event: "APPROVE"`` when the review has zero high/medium findings.

    ``pr_number``: when set, resolves the target PR by explicit number via
    ``gh pr view`` (redrive) instead of the current branch's open PR;
    a failing explicit lookup returns :attr:`PostStatus.NO_PR` with no
    fallback to current-branch discovery.

    ``diagram_blocks`` (issue #1113): host-rendered grounded-diagram markdown,
    threaded through to :func:`build_payload`.

    Returns:
        A :class:`PostStatus` describing the outcome so the caller can decide
        whether a non-posting run is a failure (comment mode) or a
        warn-and-continue (default deep flow).
    """
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
    if not issues and not approve_on_clean and not (diagram_blocks and diagram_blocks.strip()):
        print_info(console, "No parseable issues in review output; skipping PR post.")
        return PostStatus.NOTHING_TO_POST
    return await _post(
        target_dir,
        issues,
        console=console,
        post=post,
        approve_on_clean=approve_on_clean,
        pr_number=pr_number,
        diagram_blocks=diagram_blocks,
    )


# --- Parsers ---------------------------------------------------------------


DAYDREAM_REPO_URL = "https://github.com/existential-birds/daydream"
DAYDREAM_FOOTER = (
    f"<sub>🧙 Posted by [daydream v{daydream.__version__}]({DAYDREAM_REPO_URL})</sub>"
)

# Hidden HTML-comment marker embedded in posted comment bodies so later runs
# can recognise their own findings (cross-run dedup). Invisible in rendered
# markdown, present in the raw body fetched via the API.
FINDING_MARKER_RE = re.compile(r"<!-- daydream-finding: ([0-9a-f]{64}) -->")


def finding_marker(fingerprint: str) -> str:
    """Render the hidden finding marker comment for a fingerprint."""
    return f"<!-- daydream-finding: {fingerprint} -->"


def parse_finding_markers(text: str) -> list[str]:
    """Return all finding fingerprints embedded in ``text``, in order."""
    return FINDING_MARKER_RE.findall(text)


# Hidden marker for a standalone grounded-diagram comment (issue #1113). One
# per rendered kind, so a later diagram-only run of the SAME kind can find and
# minimize its own prior comment without touching the other kind's.
DIAGRAM_MARKER_RE = re.compile(r"<!-- daydream-diagram: ([a-z]+) ([0-9a-f]{7,40}) -->")


def diagram_marker(kind: str, head_sha: str) -> str:
    """Render the hidden diagram marker comment for one kind at one head."""
    return f"<!-- daydream-diagram: {kind} {head_sha} -->"


def parse_diagram_markers(text: str) -> list[tuple[str, str]]:
    """Return all ``(kind, head_sha)`` diagram markers in ``text``, in order."""
    return [(kind, sha) for kind, sha in DIAGRAM_MARKER_RE.findall(text)]


def _normalize_severity(raw: dict[str, Any]) -> str | None:
    """Normalize a raw item's severity against the canonical vocabulary.

    Total: never raises. Present-but-null severities (the wire schema emits
    ``severity: null``) and omitted keys both map to ``None``, as do unknown
    or non-string values — never the string ``"none"``. Unknown string
    severities (e.g. ``"critical"``) also map to ``None`` here so the
    canonical render path stays clean; callers must pair this with
    :func:`_severity_off_vocabulary` so a present-but-off-vocabulary label
    still fails closed at the approval gate instead of looking like a model
    that omitted severity.
    """
    return normalize_severity(raw.get("severity"))


def _severity_off_vocabulary(raw: dict[str, Any]) -> bool:
    """True when ``raw`` carries a present, non-empty severity string outside
    the canonical vocabulary (e.g. ``"critical"``).

    ``_normalize_severity`` folds such labels into ``None``, but a raw
    off-vocabulary label is a severity the model asserted — it must not be
    indistinguishable from an omitted severity at the approval gate. The gate
    blocks on this flag, restoring the documented fail-closed invariant for
    off-vocabulary labels (issue #972).
    """
    value = raw.get("severity")
    return (
        isinstance(value, str)
        and bool(value.strip())
        and normalize_severity(value) is None
    )


def alt_issues_to_parsed(alt_issues: list[dict[str, Any]]) -> list[ParsedIssue]:
    """Convert `phase_alternative_review` dicts into ParsedIssue objects.

    Alt issues have a `files: list[str]` field and no line hint. When
    multiple files are listed we emit one issue per file (classifier will
    fold file-level issues into the review body).

    Every emitted issue carries a stable cross-run ``fingerprint`` computed
    from the file path, title, and description (``recommendation`` is
    excluded from identity), so the per-file fan-out yields one distinct
    fingerprint per file.
    """
    out: list[ParsedIssue] = []
    for raw in alt_issues:
        files = raw.get("files") or []
        if not files:
            continue
        title = str(raw.get("title", "")).strip()
        description = str(raw.get("description", "")).strip()
        recommendation = str(raw.get("recommendation", "")).strip()
        severity = _normalize_severity(raw)
        confidence = str(raw.get("confidence", "")).strip().upper() or None
        body_parts = []
        if severity:
            body_parts.append(f"**Severity:** {severity}")
        if confidence:
            body_parts.append(f"**Confidence:** {confidence}")
        if description:
            body_parts.append(description)
        if recommendation:
            body_parts.append(f"**Recommendation:** {recommendation}")
        body = "\n\n".join(body_parts)
        for path in files:
            out.append(
                ParsedIssue(
                    path=str(path),
                    line=None,
                    title=title,
                    body=body,
                    confidence=confidence,
                    severity=severity,
                    severity_off_vocabulary=_severity_off_vocabulary(raw),
                    fingerprint=compute_fingerprint(str(path), title, description),
                )
            )
    return out


def extract_item_fields(
    raw: dict[str, Any],
) -> ItemFields | None:
    """Extract and normalise fields from a single canonical merged item.

    Returns an :class:`ItemFields` instance, or ``None`` when ``file`` is
    empty (so callers can simply ``continue``).
    """
    path = str(raw.get("file", "")).strip()
    if not path:
        return None
    line = raw.get("line")
    line_int = int(line) if isinstance(line, int) and not isinstance(line, bool) else None
    description = str(raw.get("description", "")).strip()
    rationale = str(raw.get("rationale", "")).strip()
    severity = _normalize_severity(raw)
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
    """Convert canonical merged items into ParsedIssue objects.

    Maps every lens — per-stack, cross-stack, and structural — to a postable
    ParsedIssue, carrying ``severity`` (and ``confidence`` when present)
    through to the tag/emoji rendering path. Unlike :func:`parse_report`,
    nothing is filtered by section, so structural findings post too.

    Each item is one canonical finding with ``file``/``line`` already
    resolved, so (unlike :func:`alt_issues_to_parsed`) there is no multi-file
    fan-out.
    """
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


# --- Git / gh helpers ------------------------------------------------------


def _current_branch(target_dir: Path) -> str | None:
    try:
        return git_ops.current_branch(target_dir)
    except GitError:
        return None


def _head_repo_slug_from_row(row: dict[str, Any]) -> str | None:
    """The ``owner/repo`` slug that holds the PR's head commit, or ``None``.

    ``gh pr list --json headRepository,headRepositoryOwner`` rows carry the
    head repository — the fork for fork-head PRs, where the head commit
    actually exists. ``gh pr view`` rows carry neither key, so callers fall
    back to the base-repo slug for those.
    """
    head_repo = row.get("headRepository")
    if not isinstance(head_repo, dict):
        return None
    name_with_owner = head_repo.get("nameWithOwner")
    if isinstance(name_with_owner, str) and "/" in name_with_owner:
        return name_with_owner
    head_owner = row.get("headRepositoryOwner")
    if (
        isinstance(head_owner, dict)
        and isinstance(head_owner.get("login"), str)
        and isinstance(head_repo.get("name"), str)
    ):
        return f"{head_owner['login']}/{head_repo['name']}"
    return None


def _pr_info_from_row(target_dir: Path, row: dict[str, Any]) -> PRInfo | None:
    """Build :class:`PRInfo` from a ``gh`` PR row, resolving the owner/repo slug.

    ``owner``/``repo`` (the posting target) come from ``gh repo view`` — the
    base repository, which hosts the PR's comments. ``head_repo`` (the
    reviewed-commit link target) comes from the row's head-repository entry
    when present, so fork-head PRs link to the fork that holds the commit.

    Returns None when the owner/repo slug cannot be resolved.
    """
    # Owner/repo lookup via `gh repo view` — the base repo hosts the PR.
    slug = git_ops.gh_repo_view(target_dir)
    if slug is None:
        return None
    owner, repo = slug
    return PRInfo(
        number=int(row["number"]),
        head_sha=row["headRefOid"],
        base_sha=row["baseRefOid"],
        base_ref=row.get("baseRefName", ""),
        owner=owner,
        repo=repo,
        url=row.get("url", ""),
        head_repo=_head_repo_slug_from_row(row),
    )


def find_open_pr(target_dir: Path) -> PRInfo | None:
    """Locate the open PR for the current branch. Returns None if not found."""
    branch = _current_branch(target_dir)
    if not branch:
        return None
    rows = git_ops.gh_pr_list_for_branch(target_dir, branch)
    if not rows:
        return None
    return _pr_info_from_row(target_dir, rows[0])


def find_pr_by_number(target_dir: Path, pr_number: int) -> PRInfo | None:
    """Resolve :class:`PRInfo` for an explicit PR number via ``gh pr view``.

    Used when the caller pins the target PR (``--pr-number``) instead of
    deriving it from the current branch like :func:`find_open_pr`.

    Returns:
        The resolved :class:`PRInfo`, or ``None`` when the PR or the
        owner/repo slug cannot be resolved.
    """
    data = git_ops.gh_pr_view(target_dir, pr_number)
    if data is None:
        return None
    return _pr_info_from_row(target_dir, data)


# --- Line resolution + hunk classification --------------------------------


_ANCHOR_TOKEN = re.compile(r"`([^`\n]{3,80})`|\b([A-Za-z_][A-Za-z0-9_]{4,})\b")


def extract_anchors(text: str, *, prefer_quoted: bool = False) -> list[str]:
    """Pull candidate anchor tokens from issue text, capped at the first 8.

    Two orderings share one extraction pass:

    ``prefer_quoted=False`` (default) sorts every token longest-first.
    :func:`compute_fingerprint` hashes this selection, so which 8 tokens
    survive the cap is part of a finding's cross-run identity -- changing it
    would re-fingerprint every open finding and defeat the reconcile dedup.
    This ordering is therefore frozen.

    ``prefer_quoted=True`` puts backtick-quoted tokens first (longest-first
    within each group), then bare words. Longest-first alone does the opposite
    of what it claims once a rationale carries ordinary English words: prose
    like ``environment``/``immediately`` outranks a short backticked
    identifier such as ``ttl`` and pushes it past the cap, so the one token
    that actually appears on the cited line never reaches line resolution
    (issue #1102). :func:`resolve_line` asks for this ordering.
    """
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
    """Compute a stable SHA256 fingerprint identifying a finding across runs.

    Hashes the canonical raw fields — never the rendered comment body, which
    carries volatile severity/confidence badges. The fingerprint combines the
    file path, normalized description (the finding title), sorted anchor
    tokens from description + rationale, and normalized rationale. Anchor
    tokens are sorted (order-insensitive code symbols); description and
    rationale preserve word order so differently-worded findings do not
    collide. The line number is excluded so code shifts do not change a
    finding's identity.
    """
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
    """Resolve the true line in the head commit for an issue.

    ``hunks`` are the head-side ``(start, end)`` diff ranges for ``issue.path``
    (see :func:`file_hunks`). They are what makes this path the
    no-op-on-valid backstop ``daydream.deep.location_validator`` documents it
    as: without them a correct, already-validated in-hunk line was re-derived
    from prose tokens and could be relocated to the first anchor hit anywhere
    in the file, after which :func:`snap_to_hunk` dropped the finding off the
    diff entirely (issue #1102).

    Tries (in order):
      1. Line hint inside a hunk: return it unchanged. The merge-time
         validator already confirmed it against the persisted hunk index, so
         anchor verification could only move a line that is known good.
      2. Line hint outside every hunk (or no ``hunks`` supplied): trust it
         only when an anchor appears within +/-5 lines.
      3. Whole-file anchor search, preferring the first hit that lands inside
         a hunk. An out-of-hunk hit is returned only when no anchor hits any
         hunk, so a comment is never relocated onto unchanged code while a
         changed-line candidate exists.

    Returns None if the file doesn't exist at head or no anchor matches.
    """
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
) -> list[tuple[int, int]]:
    """Return (start, end) inclusive line ranges on the head side for `path`.

    Primary path: ``git diff <base_sha>..<head_sha> -- <path>``.

    Fallback path: when the git invocation fails (returncode != 0 or raises --
    common when ``base_sha`` has been rewritten out of the local history) and a
    ``pr_number`` is available, re-derive the hunks from ``gh pr diff <num>``.
    The gh diff is a full PR diff, so we slice out the block for ``path``
    before parsing hunks to avoid attributing other files' hunks to this one.

    Args:
        base_sha: Base commit SHA (may be unreachable locally after a rebase).
        pr_number: Optional PR number; enables the ``gh pr diff`` fallback.
    """
    git_failed = False
    diff_text = ""
    try:
        diff_text = git_ops.diff_paths(
            target_dir, base_sha, head_sha, [path], unified=3, merge_base_diff=False
        )
    except GitError:
        git_failed = True

    if git_failed and pr_number is not None:
        diff_text = _gh_pr_diff_for_path(target_dir, pr_number, path)

    return _parse_hunks(diff_text)


def _gh_pr_diff_for_path(target_dir: Path, pr_number: int, path: str) -> str:
    """Fetch the PR's full diff via `gh pr diff` and return just the block for `path`."""
    try:
        full_diff = git_ops.gh_pr_diff(target_dir, pr_number)
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


def pr_changed_files(target_dir: Path, pr: PRInfo) -> set[str]:
    """Return the head-side paths touched by ``pr``.

    Primary path is ``git diff <base>..<head>``; when the local clone cannot
    resolve ``base_sha`` (rewritten by a rebase) the full PR diff from
    ``gh pr diff`` is parsed instead — the same two-tier strategy as
    :func:`file_hunks`.

    GitHub rejects a file-level review comment whose path is not part of the
    PR diff (HTTP 422), so this set gates file-level placement in
    :func:`classify`.
    """
    changed = set(git_ops.diff_name_only(target_dir, pr.base_sha, pr.head_sha))
    if changed:
        return changed
    try:
        full_diff = git_ops.gh_pr_diff(target_dir, pr.number)
    except GitError:
        return set()
    return set(_DIFF_GIT_HEADER.findall(full_diff))


def _parse_hunks(diff_text: str) -> list[tuple[int, int]]:
    """Head-side inclusive hunk ranges for a (single-file) diff block.

    Delegates to the shared unified-diff parser in ``daydream.hunk_index``
    (``head_side_ranges(parse_hunks(...))``) so pr_review, quote_scrub and
    coverage all count from the same source. The contract is unchanged: a
    ``list[tuple[int, int]]`` of ``(new_start, new_start + count - 1)`` ranges
    in diff order.
    """
    from daydream.hunk_index import head_side_ranges, parse_hunks

    return head_side_ranges(parse_hunks(diff_text))


def snap_to_hunk(
    line: int, hunks: list[tuple[int, int]], tolerance: int = HUNK_TOLERANCE
) -> int | None:
    """Return a valid in-hunk line for a PR comment, or None if too far.

    A no-op-on-valid backstop (issue #745): the pre-report location validator
    (``daydream.deep.location_validator``) owns pre-report authority; posting
    keeps this snap against the LIVE branch diff for placement, passing valid
    lines through unchanged.

    If ``line`` falls inside a hunk, return it unchanged. If it is within
    ``tolerance`` lines of a hunk boundary, snap to the nearest boundary
    so the GitHub API receives a line that actually appears in the diff.
    Returns ``None`` when the line is beyond tolerance of every hunk.
    """
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


# --- Classification + payload build ---------------------------------------


def classify(
    target_dir: Path, pr: PRInfo, issues: list[ParsedIssue]
) -> _ClassifiedIssues:
    """Split issues into inline, file-level, and body-only placements.

    A finding with no resolvable diff-line home is not automatically demoted
    to the review body: when its file is part of the PR diff it becomes a
    file-level comment instead. The review body is invisible to the
    ``/pulls/{n}/comments`` endpoint the labeler reads back, so body-only is
    the placement of last resort — used only when GitHub could not accept a
    comment for that path at all.

    Side effect: for issues placed inline, this mutates the caller-owned
    ``ParsedIssue.body`` of the objects in ``issues`` in place, via
    ``_note_relocation``, whenever the posted line differs from the line the
    issue cited. Callers should not rely on ``issue.body`` remaining
    unchanged after this call.
    """
    out = _ClassifiedIssues()
    hunks_cache: dict[str, list[tuple[int, int]]] = {}
    changed_files = pr_changed_files(target_dir, pr)

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
            hunks_cache[issue.path] = file_hunks(
                target_dir,
                pr.base_sha,
                pr.head_sha,
                issue.path,
                pr_number=pr.number,
            )
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
        out.inline.append(_inline_comment(issue, snapped))
        out.inline_issues.append(issue)
    return out


# Marks the posting-time relocation annotation so it is appended at most once
# even if a caller classifies the same issue objects twice.
_PLACEMENT_NOTE_PREFIX = "**Placement:** posted on line "


def _note_relocation(issue: ParsedIssue, posted_line: int) -> None:
    """Annotate ``issue`` when it is posted on a line it did not cite.

    Anchor resolution and hunk snapping can both move a finding off its
    reported line. Overwriting it silently leaves the run's own artifacts
    showing a citation the posted comment does not use (issue #1102), so the
    relocation is recorded in the body -- the same in-place, non-destructive
    annotation the pre-report ``deep.location_validator`` writes for its
    demotions. The note reaches the findings artifact too, because
    ``findings._finding_dict`` serialises ``body``. Fingerprints are computed
    from description/rationale and never the body, so annotating cannot change
    a finding's cross-run identity.

    ``issue.line <= 0`` (including the ``0`` whole-file sentinel used by
    structural findings) is not a cited line and is never reported as one.
    """
    if issue.line is None or issue.line <= 0 or issue.line == posted_line:
        return
    if _PLACEMENT_NOTE_PREFIX in issue.body:
        return
    note = f"{_PLACEMENT_NOTE_PREFIX}{posted_line}; reviewer cited line {issue.line}."
    issue.body = f"{issue.body}\n\n{note}" if issue.body else note


def _inline_comment(issue: ParsedIssue, line: int) -> dict[str, Any]:
    """Build one inline review-comment dict for the review payload."""
    return {
        "path": issue.path,
        "line": line,
        "side": "RIGHT",
        "body": _format_inline_body(issue),
    }


_SEVERITY_EMOJI: dict[str, str] = {
    "high": "⚠️",
    "medium": "🔵",
    "low": "💡",
}


def _severity_emoji(severity: str | None) -> str:
    """Map a severity level to an emoji prefix."""
    if not severity:
        return ""
    return _SEVERITY_EMOJI.get(severity.lower(), "")


def _issue_header(issue: CommentFinding, *, prefix: str = "", always_bold: bool = False) -> str:
    """Compose the emoji/title/tag header line for one issue."""
    emoji = _severity_emoji(issue.severity)
    title_prefix = f"{emoji} " if emoji else ""
    header = f"{title_prefix}**{prefix}{issue.title}**" if issue.title or always_bold else ""
    tags = _format_tag_line(issue)
    if header and tags:
        return f"{header} | {tags}"
    return header or tags


def default_render_finding(finding: CommentFinding, ctx: FindingRenderContext) -> str:
    """Render the inner human block for one finding (header + body + agent prompt).

    Placement-parameterized (``"inline"``, ``"file_level"``, ``"summary"``) so
    it reproduces today's inline, file-level, and summary inner text exactly.
    It never emits :data:`DAYDREAM_FOOTER` or the hidden finding marker — those
    stay host-owned and are injected by the callers.
    """
    if ctx.placement == "inline":
        parts = [p for p in (_issue_header(finding), finding.body) if p]
        parts.append(_build_agent_prompt(finding))
        return "\n\n".join(parts)
    prefix = "[cross-stack] " if finding.is_cross_stack else ""
    if ctx.placement == "summary":
        summary_parts = [_issue_header(finding, prefix=prefix, always_bold=True)]
        if finding.body:
            summary_parts.append(f"\n{finding.body}\n")
        summary_parts.append(_build_agent_prompt(finding))
        return "\n".join(summary_parts)
    # file_level (default placement).
    header = _issue_header(finding, prefix=prefix, always_bold=True)
    parts = [p for p in (header, finding.body) if p]
    parts.append(_build_agent_prompt(finding))
    return "\n\n".join(parts)


def _comment_finding(issue: ParsedIssue) -> CommentFinding:
    """Map an internal :class:`ParsedIssue` to the public :class:`CommentFinding`."""
    return CommentFinding(
        path=issue.path,
        line=issue.line,
        title=issue.title,
        body=issue.body,
        is_cross_stack=issue.is_cross_stack,
        severity=issue.severity,
        confidence=issue.confidence,
        fingerprint=issue.fingerprint,
    )


def _render_finding(issue: ParsedIssue, placement: str) -> str:
    """Render one finding's inner block through the registered ``"finding"`` renderer.

    Falls back to :func:`default_render_finding` (and warns) when the custom
    renderer raises or returns a non-``str``/empty result, so a broken fork
    can never break posting.
    """
    cf = _comment_finding(issue)
    ctx = FindingRenderContext(placement=placement)
    _fn = get_registry().renderer("finding")
    _label = "builtin" if _fn is default_render_finding else "custom"
    try:
        result = _fn(cf, ctx)
    except Exception as exc:  # noqa: BLE001 - any fork error degrades to the default
        _logger.warning("%s 'finding' renderer failed (%s); using default", _label, exc)
        return default_render_finding(cf, ctx)
    if not isinstance(result, str) or not result:
        _logger.warning(
            "%s 'finding' renderer failed (returned %r); using default", _label, result
        )
        return default_render_finding(cf, ctx)
    return result


def _format_inline_body(issue: ParsedIssue) -> str:
    parts = [_render_finding(issue, "inline"), DAYDREAM_FOOTER]
    if issue.fingerprint:
        parts.append(finding_marker(issue.fingerprint))
    return "\n\n".join(parts).strip()


def _format_file_level_body(issue: ParsedIssue) -> str:
    """Render the body of a file-level review comment.

    Carries the same :data:`DAYDREAM_FOOTER` badge and hidden finding marker
    as an inline comment, so the labeler's author check and fingerprint join
    recognise it without any read-side special-casing.
    """
    parts = [_render_finding(issue, "file_level"), DAYDREAM_FOOTER]
    if issue.fingerprint:
        parts.append(finding_marker(issue.fingerprint))
    return "\n\n".join(parts).strip()


def _format_tag_line(issue: CommentFinding) -> str:
    """Render severity/confidence badges for a single issue, if set."""
    bits: list[str] = []
    if issue.severity:
        bits.append(f"severity: `{issue.severity}`")
    if issue.confidence:
        bits.append(f"confidence: `{issue.confidence}`")
    return " · ".join(bits)


def _build_agent_prompt(issue: CommentFinding) -> str:
    """Build a collapsible AI-agent-friendly prompt for a single issue."""
    loc = f"`{issue.path}`"
    if issue.line:
        loc += f" around line {issue.line}"
    instruction = issue.title
    if issue.body:
        # First meaningful body line as added context.
        first_line = issue.body.strip().split("\n")[0].strip()
        if first_line and first_line != issue.title:
            instruction = f"{instruction}: {first_line}" if instruction else first_line
    return (
        "<details>\n"
        "<summary>🔮 Prompt for AI Agents</summary>\n\n"
        "```\n"
        "Verify each finding against the current code and only fix it if needed.\n\n"
        f"In {loc}, {instruction}\n"
        "```\n\n"
        "</details>"
    )


def _summary_body_block(issue: ParsedIssue) -> str:
    """Render one non-inline finding's block for the summary section.

    Routes the inner human block through the ``"finding"`` renderer seam
    (placement ``"summary"``) then appends the host-owned finding marker. The
    result is byte-identical to the finding's flattened header/body/prompt/marker
    sequence in the pre-seam summary section.
    """
    block = _render_finding(issue, "summary")
    if issue.fingerprint:
        block = f"{block}\n{finding_marker(issue.fingerprint)}"
    return block


def _render_body_section(findings: tuple[SummaryFinding, ...]) -> str:
    """Assemble the by-file collapsible ``<details>`` non-inline findings section.

    Shared by :func:`_format_body_section` (the ``ParsedIssue`` entry point) and
    :func:`default_render_summary` (the ``SummaryContext`` entry point). Consumes
    each finding's host-rendered ``body_block`` (marker already embedded) as a
    single unit, preserving the exact whitespace of the pre-seam layout.
    """
    if not findings:
        return ""
    grouped: dict[str, list[SummaryFinding]] = {}
    for sf in findings:
        grouped.setdefault(sf.finding.path, []).append(sf)
    total = len(findings)
    parts: list[str] = [
        "<details>",
        f"<summary>📋 Non-inline findings ({total})</summary><blockquote>\n",
    ]
    for filepath, file_findings in grouped.items():
        parts.append("<details>")
        parts.append(
            f"<summary>{filepath} ({len(file_findings)})</summary><blockquote>\n"
        )
        for i, sf in enumerate(file_findings):
            parts.append(sf.body_block)
            if i < len(file_findings) - 1:
                parts.append("\n---\n")
        parts.append("\n</blockquote></details>")
    parts.append("\n</blockquote></details>")
    return "\n".join(parts)


def _summary_findings(body_only: list[ParsedIssue]) -> tuple[SummaryFinding, ...]:
    """Map non-inline :class:`ParsedIssue` objects to public :class:`SummaryFinding`s."""
    return tuple(
        SummaryFinding(finding=_comment_finding(issue), body_block=_summary_body_block(issue))
        for issue in body_only
    )


def _format_body_section(body_only: list[ParsedIssue]) -> str:
    """Render the by-file non-inline findings section from internal issues.

    Retained as the ``ParsedIssue`` entry point (approval-snapshot guard);
    delegates to :func:`_render_body_section` so the default summary renderer
    and this path share one scaffolding implementation.
    """
    return _render_body_section(_summary_findings(body_only))


def default_render_summary(ctx: SummaryContext) -> str:
    """Render the summary body between the approval line and the footer.

    Reproduces today's markdown byte-for-byte: ``**Code Review Summary**``, the
    grounded-diagram blocks (issue #1113, when the run rendered any), the
    by-file non-inline findings section, the consolidated agent prompt (when
    non-empty), then the fully-wrapped review-info block. Never emits the
    approval line, the ``event`` decision, or :data:`DAYDREAM_FOOTER` — those
    stay host-owned in :func:`build_payload`.
    """
    chunks: list[str] = ["**Code Review Summary**"]
    if ctx.diagrams:
        chunks.append(ctx.diagrams)
    section = _render_body_section(ctx.findings)
    if section:
        chunks.append(section)
    if ctx.agent_prompt:
        chunks.append(ctx.agent_prompt)
    chunks.append(ctx.review_info)
    return "\n\n".join(chunks)


def _render_summary(ctx: SummaryContext) -> str:
    """Render the summary body through the registered ``"summary"`` renderer.

    Falls back to :func:`default_render_summary` (and warns) when the custom
    renderer raises or returns a non-``str``/empty result, so a broken fork can
    never break posting.
    """
    try:
        result = get_registry().renderer("summary")(ctx)
    except Exception as exc:  # noqa: BLE001 - any fork error degrades to the default
        _logger.warning("custom 'summary' renderer failed (%s); using default", exc)
        return default_render_summary(ctx)
    if not isinstance(result, str) or not result:
        _logger.warning(
            "custom 'summary' renderer failed (returned %r); using default", result
        )
        return default_render_summary(ctx)
    return result


def _count_labels(
    issues: list[ParsedIssue], attr: str, order: tuple[str, ...]
) -> list[str]:
    """Return ordered `N LABEL` strings for non-empty counts."""
    counts: dict[str, int] = {}
    for issue in issues:
        val = getattr(issue, attr)
        if val:
            counts[val] = counts.get(val, 0) + 1
    out: list[str] = []
    for key in order:
        n = counts.get(key, 0)
        if n:
            out.append(f"{n} {key}")
    return out


def _build_consolidated_prompt(
    classified: _ClassifiedIssues,
    pr: PRInfo,
) -> str:
    """Build a single collapsible prompt block that tells AI agents to fetch and fix review comments."""
    total = classified.total()

    prompt_body = (
        f"Fix the {total} review comment(s) posted on this PR.\n"
        "\n"
        "Fetch the comments manually:\n"
        f"1. gh api repos/{pr.owner}/{pr.repo}/pulls/{pr.number}/comments\n"
        f"2. gh api repos/{pr.owner}/{pr.repo}/issues/{pr.number}/comments\n"
        "\n"
        "These endpoints return all comments on the PR. Focus on the most\n"
        "recent review — ignore older review threads that have already been\n"
        "addressed. For each comment: read the referenced file, verify the\n"
        "finding against the current code, and fix it if valid. Skip false\n"
        "positives. Commit all fixes when done."
    )

    return (
        "<details>\n"
        "<summary>🔮 Prompt for all review comments with AI agents</summary>\n\n"
        f"```\n{prompt_body}\n```\n\n"
        "</details>"
    )


def _resolve_trajectory_paths(
    recorder: TrajectoryRecorder | None,
) -> tuple[list[Path], tempfile.TemporaryDirectory[str] | None]:
    """Resolve trajectory file paths to feed the enriched-comment renderer.

    Returns the parent trajectory plus any sibling fork files (deep mode).
    Because :meth:`TrajectoryRecorder._write` only fires at ``__aexit__``,
    the parent file does not yet exist when the PR comment is composed
    mid-run; we therefore snapshot the in-memory parent ATIF Trajectory to a
    tempfile so the renderer can read it like any other on-disk trajectory.
    Sibling forks have already exited and written by post-time, so we glob
    for them.

    Discovery rule: parent path is taken from ``recorder.path``; siblings
    are every ``*.json`` under
    ``<target_dir>/.daydream/runs/<session_id>/trajectories/`` (every fork
    in the run dir belongs to this run by construction — no prefix
    filtering required).

    The returned ``TemporaryDirectory`` (when not ``None``) MUST be kept
    alive by the caller until the renderer finishes; closing it deletes
    the snapshot file.
    """
    if recorder is None:
        return [], None
    paths: list[Path] = []
    tmpdir: tempfile.TemporaryDirectory[str] | None = None
    try:
        # Snapshot the in-memory parent trajectory to a tempfile (parent
        # file isn't written until __aexit__).
        if recorder.steps:
            trajectory = recorder.build_trajectory()
            tmpdir = tempfile.TemporaryDirectory(prefix="daydream-traj-snapshot-")
            snapshot = Path(tmpdir.name) / "parent.json"
            snapshot.write_text(
                json.dumps(trajectory.to_json_dict(), indent=2), encoding="utf-8"
            )
            paths.append(snapshot)
        # Discover sibling fork trajectories on disk (deep mode).
        sibling_dir = (
            recorder.target_dir
            / ".daydream"
            / "runs"
            / recorder.session_id
            / "trajectories"
        )
        if sibling_dir.is_dir():
            for sibling in sorted(sibling_dir.glob("*.json")):
                if sibling.is_file():
                    paths.append(sibling)
    except Exception:  # noqa: BLE001 - renderer treats [] as missing data
        if tmpdir is not None:
            with contextlib.suppress(Exception):
                tmpdir.cleanup()
        return [], None
    return paths, tmpdir


_REVIEWED_COMMIT_MARKER = "- **Reviewed commit:**"


def _strip_forged_reviewed_commit_lines(run_info: str) -> str:
    """Drop forged reviewed-commit lines from run-info text (issue 2).

    The reviewed-commit line is the single source of truth for which commit
    was reviewed, built solely from validated ``PRInfo.head_sha``. The
    untrusted artifact ``run_info`` string must never render a second,
    forged line, so any line (optionally indented) starting with the marker
    is removed from the run-info block before composition.
    """
    return "\n".join(
        line
        for line in run_info.splitlines()
        if not line.lstrip().startswith(_REVIEWED_COMMIT_MARKER)
    )


def _render_review_info_block() -> str:
    """Render the enriched run-info block, falling back to a brief note.

    Wraps :func:`render_run_info_block` with one extra safety net beyond
    its own internal try/except: if any unexpected exception escapes (e.g.
    snapshot write failure), we degrade to a 'run details unavailable'
    note. The comment must still post (K8 / M9).
    """
    try:
        recorder = get_current_recorder()
        paths, tmpdir = _resolve_trajectory_paths(recorder)
        try:
            return render_run_info_block(paths)
        finally:
            if tmpdir is not None:
                with contextlib.suppress(Exception):
                    tmpdir.cleanup()
    except Exception:  # noqa: BLE001 - posting must never crash on snapshot/discovery
        return f"*run details unavailable*\n\n<sub>Generated by daydream v{daydream.__version__}</sub>"


# Severities that must never let an opted-in review post as an approval.
# Fails closed: any string outside ``_NON_BLOCKING_SEVERITIES`` blocks. The
# findings schema permits arbitrary strings, so unknown/off-vocabulary labels
# ("critical", "blocker", "major", ...) must conservatively block approval.
_NON_BLOCKING_SEVERITIES = frozenset({"low"})


def _severity_blocks_approval(severity: str | None) -> bool:
    """Whether one finding's severity blocks an approval (issue #343).

    Fail-closed: any severity outside ``_NON_BLOCKING_SEVERITIES`` blocks —
    the findings schema permits any string, so off-vocabulary labels must not
    slip an approval through. ``None`` (a model that omitted severity)
    deliberately does not block.
    """
    return severity is not None and severity.lower() not in _NON_BLOCKING_SEVERITIES


def _finding_blocks_approval(
    severity: str | None,
    location_distrust: bool,
    severity_off_vocabulary: bool = False,
    severity_before_demotion: str | None = None,
) -> bool:
    """Whether one finding blocks an approval, demotion-aware (issue #972 R2).

    A finding marked ``location_distrust=True`` was judged at a higher severity
    and demoted by location validation (its citation was beyond tolerance);
    the demoted severity must not silently make it non-blocking. The demotion
    mark is written for any beyond-tolerance record regardless of its original
    severity, so the gate only re-blocks when the pre-demotion severity carried
    via ``severity_before_demotion`` was itself blocking; an originally-low or
    never-asserted severity stays non-blocking. This check is deliberately
    separate from ``_severity_blocks_approval`` (and NOT folded into
    ``_NON_BLOCKING_SEVERITIES``) so off-vocabulary severity strings keep
    failing closed for their own reason. ``severity_off_vocabulary`` carries
    that signal: a present-but-off-canonical label (e.g. ``"critical"``) is
    folded into ``None`` at the boundary but was still a severity the model
    asserted, so it blocks rather than reading as an omitted severity.
    """
    if location_distrust and _severity_blocks_approval(severity_before_demotion):
        return True
    if severity_off_vocabulary:
        return True
    return _severity_blocks_approval(severity)


def _is_clean_review(classified: _ClassifiedIssues, approve_on_clean: bool) -> bool:
    """Whether an opted-in review may post as an approval (issue #343).

    True only when ``approve_on_clean`` is set AND no finding carries a
    blocking severity. Severity is matched case-insensitively against
    ``_NON_BLOCKING_SEVERITIES``: any string outside it ("high", "medium",
    "critical", "blocker", "major", ...) blocks — the findings schema permits
    any string, so off-vocabulary labels must conservatively block — while
    ``None`` (a model that omitted severity) deliberately does not.
    """
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


def build_payload(
    pr: PRInfo,
    classified: _ClassifiedIssues,
    *,
    run_info_override: str | None = None,
    approve_on_clean: bool = False,
    diagram_blocks: str | None = None,
) -> dict[str, Any]:
    """Assemble the review payload for `POST /repos/.../pulls/<n>/reviews`.

    The review body uses collapsible sections so large reviews stay readable:
        **Code Review Summary**
        <details> Non-inline findings grouped by file
        <details> Consolidated AI agent prompt
        <details> Review info (reviewed commit + enriched run-info + version footer)
        Footer (🧙 Posted by daydream vX.Y.Z)

    Args:
        run_info_override: Pre-rendered run-info markdown to use in place of
            the live recorder block (``post-findings`` posts from artifact
            data; there is no recorder in that process). ``None`` renders the
            live block as usual.
        approve_on_clean: Opt-in approval (issue #343). When True AND the
            classified review has no blocking severity findings (anything
            other than ``"low"`` or omitted, fail-closed — see
            ``_NON_BLOCKING_SEVERITIES``), the payload's event becomes
            ``"APPROVE"`` with a prepended approval line; otherwise the event
            stays ``"COMMENT"`` and the body is byte-identical to the
            non-approve path.
        diagram_blocks: Host-rendered grounded-diagram markdown (issue #1113),
            handed to the ``"summary"`` renderer as ``ctx.diagrams`` so it
            lands directly under the summary header. ``None`` (the default)
            renders a byte-identical no-diagram body.
    """
    all_issues_with_inline_meta = classified.all_issues()

    clean = _is_clean_review(classified, approve_on_clean)

    # Consolidated AI agent prompt (host-built; empty means "omit").
    agent_prompt = (
        _build_consolidated_prompt(classified, pr) if not classified.is_empty() else ""
    )

    # Collapsible review info: a linked reviewed-commit line (from
    # ``pr.head_sha``, so readers can tell whether findings apply to the
    # PR's current head), then enriched run-info (rollup + per-phase
    # breakdown + version footer, owned by the renderer), then the
    # conditional severity/confidence breakdown. The renderer emits its own
    # ``<sub>Generated by daydream...</sub>`` footer, so don't double it.
    enriched_run_info = run_info_override if run_info_override is not None else _render_review_info_block()
    # The commit link targets the repo that holds the head commit: the fork
    # for fork-head PRs (``head_repo``), else the base repo from
    # ``owner``/``repo``. ``owner``/``repo`` themselves stay the base repo —
    # that is where the review comment posts.
    commit_slug = pr.head_repo or f"{pr.owner}/{pr.repo}"
    commit_line = (
        f"{_REVIEWED_COMMIT_MARKER} [`{pr.head_sha[:7]}`]"
        f"(https://github.com/{commit_slug}/commit/{pr.head_sha})"
    )
    extra_info_lines: list[str] = []
    severity_parts = _count_labels(
        all_issues_with_inline_meta, "severity", ("high", "medium", "low")
    )
    if severity_parts:
        extra_info_lines.append("- **Severity:** " + ", ".join(severity_parts))
    confidence_parts = _count_labels(
        all_issues_with_inline_meta, "confidence", ("HIGH", "MEDIUM", "LOW")
    )
    if confidence_parts:
        extra_info_lines.append("- **Confidence:** " + ", ".join(confidence_parts))
    review_info = f"{commit_line}\n\n{_strip_forged_reviewed_commit_lines(enriched_run_info)}"
    if extra_info_lines:
        review_info = f"{review_info}\n\n" + "\n".join(extra_info_lines)
    review_info_block = (
        "<details>\n"
        "<summary>ℹ️ Review info</summary>\n\n"
        f"{review_info}\n\n"
        "</details>"
    )

    summary_ctx = SummaryContext(
        findings=_summary_findings(classified.body_only),
        agent_prompt=agent_prompt,
        review_info=review_info_block,
        diagrams=diagram_blocks or None,
    )
    summary_body = _render_summary(summary_ctx)

    body_chunks: list[str] = []
    if clean:
        body_chunks.append("✅ **Deep review passed with no high/medium findings.**")
    body_chunks.append(summary_body)
    # DAYDREAM_FOOTER is the bottom-of-comment "🧙 Posted by daydream"
    # badge — distinct from the renderer's "Generated by daydream" line
    # inside the review-info block.
    body_chunks.append(DAYDREAM_FOOTER)

    payload: dict[str, Any] = {
        "event": "APPROVE" if clean else "COMMENT",
        "commit_id": pr.head_sha,
        "body": "\n\n".join(body_chunks),
        "comments": classified.inline,
    }
    return payload


# --- Core orchestration ---------------------------------------------------


def _resolve_pr(
    target_dir: Path,
    console: Console,
    pr_number: int | None,
) -> PRInfo | None:
    """Resolve the target PR for posting.

    Handles both the current-branch discovery path (:func:`find_open_pr`) and
    the explicitly-pinned path (:func:`find_pr_by_number`). On failure it
    prints a warning describing the cause — distinguishing a missing PR from a
    failed owner/repo slug resolution — and returns ``None``.

    Returns:
        The resolved :class:`PRInfo`, or ``None`` (after warning) when the PR
        cannot be resolved.
    """
    if pr_number is not None:
        pr = find_pr_by_number(target_dir, pr_number)
        if pr is None:
            # Distinguish "PR not found" from "owner/repo slug unresolvable"
            # so the operator gets a precise diagnostic rather than a generic
            # "could not resolve" message.
            if git_ops.gh_pr_view(target_dir, pr_number) is None:
                print_warning(
                    console,
                    f"PR #{pr_number} not found via `gh pr view`; skipping PR post.",
                )
            else:
                print_warning(
                    console,
                    f"PR #{pr_number} resolved via `gh pr view` but the owner/repo "
                    "slug could not be resolved via `gh repo view`; skipping PR post.",
                )
            return None
        return pr
    pr = find_open_pr(target_dir)
    if pr is None:
        print_warning(
            console,
            "No open PR found for the current branch; skipping PR post.",
        )
        return None
    return pr


async def _post(
    target_dir: Path,
    issues: list[ParsedIssue],
    *,
    console: Console,
    post: bool = False,
    approve_on_clean: bool = False,
    pr_number: int | None = None,
    diagram_blocks: str | None = None,
) -> PostStatus:
    pr = _resolve_pr(target_dir, console, pr_number)
    if pr is None:
        return PostStatus.NO_PR

    classified = classify(target_dir, pr, issues)
    if (
        classified.is_empty()
        and not approve_on_clean
        and not (diagram_blocks and diagram_blocks.strip())
    ):
        print_info(
            console, "No postable issues after classification; skipping PR post."
        )
        return PostStatus.NOTHING_TO_POST

    inline_files = sorted({c["path"] for c in classified.inline})
    summary = (
        f"{len(classified.inline)} inline on "
        f"{', '.join(inline_files) if inline_files else '(none)'}, "
        f"{len(classified.file_level)} file-level, "
        f"{len(classified.body_only)} folded into body"
    )
    clean = _is_clean_review(classified, approve_on_clean)
    event_note = " — will post event: APPROVE" if clean else ""
    print_info(console, f"PR #{pr.number}: {summary}{event_note}")

    if not post and not resolve_or_prompt(
        assume=get_assume(),
        interactive=not get_non_interactive(),
        safe_default=False,
        question=(
            "Post an APPROVE review for this clean PR? [y/N]"
            if clean
            else "Post these as a PR review? [y/N]"
        ),
        default="n",
    ):
        print_info(console, "Skipped posting to PR.")
        return PostStatus.NOTHING_TO_POST

    # File-level comments post first: a failure here has to fall back into the
    # review body, which is built below.
    posted, failed = _submit_file_level_comments(target_dir, pr, classified.file_level)
    if failed:
        classified.file_level = posted
        classified.body_only.extend(failed)
        print_warning(
            console,
            f"{len(failed)} file-level comment(s) failed to post; folded into the review body.",
        )

    payload = build_payload(
        pr, classified, approve_on_clean=approve_on_clean, diagram_blocks=diagram_blocks
    )
    review_url, error_msg = _submit_review(target_dir, pr, payload)
    if review_url is None:
        # ``error_msg`` carries the GitError text from git_ops, which includes
        # the preserved tempfile path on failure (see git_ops.gh_api).
        suffix = f" ({error_msg})" if error_msg else ""
        # File-level comments post before the review, so some may already be
        # live on the PR — saying "no comments were posted" would be false.
        already = (
            f" {len(classified.file_level)} file-level comment(s) were already posted."
            if classified.file_level
            else " No comments were posted."
        )
        print_warning(console, f"Failed to post PR review;{already}{suffix}")
        return PostStatus.FAILED

    print_success(console, f"Posted review: {review_url}")
    return PostStatus.POSTED


def _submit_file_level_comments(
    target_dir: Path, pr: PRInfo, issues: list[ParsedIssue]
) -> tuple[list[ParsedIssue], list[ParsedIssue]]:
    """POST each issue as a file-level review comment; return the ones that failed.

    GitHub rejects ``subject_type`` inside a batch review payload (HTTP 422),
    so file-level comments must be posted one at a time against
    ``/pulls/{n}/comments``. Each one becomes a top-level, repliable thread on
    that endpoint — the surface :func:`index_pr_review_comments` reads — which
    is what makes these findings labelable at all.

    Failures are returned rather than raised so the caller can fold them back
    into the review body: a finding that cannot get its own thread must still
    reach the maintainer.

    Returns:
        ``(posted, failed)`` partitioning ``issues`` in input order.
    """
    endpoint = f"/repos/{pr.owner}/{pr.repo}/pulls/{pr.number}/comments"
    posted: list[ParsedIssue] = []
    failed: list[ParsedIssue] = []
    for issue in issues:
        payload = {
            "commit_id": pr.head_sha,
            "path": issue.path,
            "subject_type": "file",
            "body": _format_file_level_body(issue),
        }
        try:
            git_ops.gh_api(target_dir, endpoint, method="POST", input_data=payload)
        except GitError:
            failed.append(issue)
        else:
            posted.append(issue)
    return posted, failed


def _submit_review(
    target_dir: Path, pr: PRInfo, payload: dict[str, Any]
) -> tuple[str | None, str | None]:
    """POST the review payload via ``gh api``.

    Returns:
        ``(html_url, None)`` on success, ``(None, error_message)`` on failure.
        The error message — when present — includes the preserved-payload path
        produced by :func:`daydream.git_ops.gh_api` so callers can surface it.
    """
    endpoint = f"/repos/{pr.owner}/{pr.repo}/pulls/{pr.number}/reviews"
    try:
        data = git_ops.gh_api(target_dir, endpoint, method="POST", input_data=payload)
    except GitError as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, None
    url = data.get("html_url")
    return (str(url) if url else None), None


def post_diagram_comment_to_pr(
    target_dir: Path,
    pr: PRInfo,
    *,
    body: str,
    kinds: list[str],
    bot_login: str | None,
) -> tuple[str | None, str | None]:
    """Post a standalone grounded-diagram issue comment on a PR (issue #1113).

    A diagram is not a finding: it has no line anchor, no severity, and no
    thread to resolve, so it posts as a plain issue comment rather than a
    review. The body carries one hidden ``daydream-diagram`` marker per kind
    plus :data:`DAYDREAM_FOOTER`, which is how the *next* diagram-only run of
    the same kind recognises and minimizes this comment.

    Prior comments are minimized only after the new one posts successfully, so
    a failed replacement cannot remove the PR's only live diagram.
    Minimization is best-effort: an unresolved ``bot_login`` skips it with a
    warning (mirroring ``BOT_LOGIN_UNRESOLVED``) rather than trusting an
    unattributed comment, and a failed mutation warns and continues — a stale
    fold is a cosmetic problem, a missing diagram is the run's deliverable.

    Args:
        target_dir: Repository directory the ``gh`` calls run in.
        pr: Resolved target PR.
        body: Rendered markdown (diagram blocks and/or omission notices).
        kinds: The diagram kinds this comment speaks for; markers are written
            for exactly these, and only prior comments carrying one of them are
            minimized.
        bot_login: Bot login for author attribution, or None.

    Returns:
        ``(html_url, None)`` on success, ``(None, error_message)`` on failure.
    """
    from daydream.agent import console
    from daydream.reconcile import fetch_prior_diagram_comments, minimize_comment

    repo_slug = f"{pr.owner}/{pr.repo}"
    prior = []
    if bot_login is None:
        print_warning(
            console,
            "BOT_LOGIN_UNRESOLVED: no bot login resolvable; skipping minimization of "
            "prior diagram comments (a stale diagram may stay unfolded, but no "
            "comment is ever minimized on an unproven author).",
        )
    else:
        try:
            prior = fetch_prior_diagram_comments(
                target_dir, repo_slug, pr.number, bot_login=bot_login
            )
        except GitError as exc:
            print_warning(console, f"Could not inventory prior diagram comments: {exc}")
            prior = []

    markers = "\n".join(diagram_marker(kind, pr.head_sha) for kind in kinds)
    chunks = [chunk for chunk in (markers, body.strip(), DAYDREAM_FOOTER) if chunk]
    endpoint = f"/repos/{pr.owner}/{pr.repo}/issues/{pr.number}/comments"
    try:
        data = git_ops.gh_api(
            target_dir, endpoint, method="POST", input_data={"body": "\n\n".join(chunks)}
        )
    except GitError as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, None
    url = data.get("html_url")
    if not url:
        return None, None

    current_kinds = set(kinds)
    for comment in prior:
        if not set(comment.kinds) & current_kinds:
            continue
        if not minimize_comment(target_dir, comment.node_id):
            print_warning(
                console,
                f"Failed to minimize prior diagram comment {comment.node_id}",
            )
    return str(url), None


def diagram_comment_kinds(payload: dict[str, Any]) -> list[str]:
    """Return the diagram kinds a standalone comment speaks for (issue #1113).

    A comment speaks for the kinds the run's eligibility decision marked
    eligible -- the kinds it actually attempted -- in render order. That is the
    right marker set: a ``--diagram-only flowchart`` run must never minimize a
    prior sequence-diagram comment, and a run whose requested kind ended up
    omitted must still supersede its own prior comment for that kind.

    Args:
        payload: The ``diagram.json`` payload (``{"eligibility", "results"}``).

    Returns:
        A subset of :data:`~daydream.config.DIAGRAM_KINDS`, sequence first.
    """
    eligibility = payload.get("eligibility")
    if not isinstance(eligibility, dict):
        return []
    kinds: list[str] = []
    for kind in DIAGRAM_KINDS:
        decision = eligibility.get(kind)
        if isinstance(decision, dict) and decision.get("eligible"):
            kinds.append(kind)
    return kinds


def _diagram_results(payload: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    """Extract the per-kind result dicts from a ``diagram.json`` payload.

    Total: a missing or malformed ``results`` object yields ``None`` for every
    kind, which the renderer reads as "no block", never as an error.
    """
    raw = payload.get("results")
    results: dict[str, dict[str, Any] | None] = {}
    for kind in DIAGRAM_KINDS:
        value = raw.get(kind) if isinstance(raw, dict) else None
        results[kind] = value if isinstance(value, dict) else None
    return results


def render_diagram_blocks_from_payload(payload: dict[str, Any]) -> str:
    """Render just the folded diagram blocks from a ``diagram.json`` payload.

    Used for the review-comment slot, which carries blocks only: an omission
    notice belongs on an explicit request (``--diagram-only`` / a mention
    command), where silence would be ambiguous, not on every deep review.
    """
    from daydream.deep.diagram_render import render_diagram_blocks

    return render_diagram_blocks(_diagram_results(payload))


def render_diagram_comment_body(payload: dict[str, Any]) -> str:
    """Render the standalone diagram comment body from a ``diagram.json`` payload.

    The single renderer for both phases: the diagram-only run posts through it
    directly, and ``post-findings`` re-renders through it from the artifact, so
    the two produce identical markdown for identical specs. Mermaid is always
    re-derived from ``spec_final`` -- a stored ``mermaid`` string is never read.

    Rendered kinds contribute their folded block; a requested kind that was
    skipped, failed, or fell below its grounding floor contributes its omission
    notice, because on an explicit request silence is indistinguishable from a
    broken run.

    Args:
        payload: The ``diagram.json`` payload.

    Returns:
        The comment body, or a short "nothing was eligible" line when the run
        requested no kind at all.
    """
    from daydream.deep.diagram_render import render_omission_notice

    results = _diagram_results(payload)
    chunks: list[str] = []
    blocks = render_diagram_blocks_from_payload(payload)
    if blocks:
        chunks.append(blocks)
    for kind in diagram_comment_kinds(payload):
        result = results.get(kind)
        if result is None or result.get("status") == "rendered":
            continue
        notice = render_omission_notice(kind, result)
        if notice:
            chunks.append(notice)
    if not chunks:
        chunks.append(
            "No grounded diagram was eligible for this pull request: the change "
            "does not cross a module or service boundary and no changed function "
            "gained enough branch points to be worth charting."
        )
    return "\n\n".join(chunks)


def _diagram_grounding_problem(
    kind: str, spec: dict[str, Any], grounding: Any
) -> str | None:
    """Validate the host-produced grounding report attached to a rendered spec."""
    from daydream.deep.diagram_grounding import REASON_CODES

    if not isinstance(grounding, dict):
        return f"{kind} rendered result has no grounding attestation"
    elements = grounding.get("elements")
    summary = grounding.get("summary")
    capped = grounding.get("capped")
    if not isinstance(elements, list):
        return f"{kind} grounding attestation has no 'elements' array"
    if not isinstance(summary, dict):
        return f"{kind} grounding attestation has no 'summary' object"
    if not isinstance(capped, dict):
        return f"{kind} grounding attestation has no 'capped' object"

    allowed_elements = (
        {"participant", "message", "block", "branch"}
        if kind == "sequence"
        else {"root", "node", "edge"}
    )
    allowed_caps = (
        {"participants", "messages", "blocks"}
        if kind == "sequence"
        else {"nodes", "edges"}
    )
    checks: list[dict[str, Any]] = []
    seen_refs: set[tuple[str, str]] = set()
    for index, raw in enumerate(elements):
        if not isinstance(raw, dict):
            return f"{kind} grounding attestation element {index} is not an object"
        element = raw.get("element")
        ref = raw.get("ref")
        grounded = raw.get("grounded")
        reason = raw.get("reason")
        final_index = raw.get("final_index")
        if element not in allowed_elements or not isinstance(ref, str) or not ref:
            return f"{kind} grounding attestation element {index} has an invalid identity"
        identity = (element, ref)
        if identity in seen_refs:
            return f"{kind} grounding attestation repeats {element} reference {ref!r}"
        seen_refs.add(identity)
        if not isinstance(grounded, bool):
            return f"{kind} grounding attestation for {element} {ref!r} has no boolean verdict"
        if grounded and reason is not None:
            return f"{kind} grounding attestation for {element} {ref!r} is contradictory"
        if not grounded and reason not in REASON_CODES:
            return f"{kind} grounding attestation for {element} {ref!r} has an invalid reason"
        if final_index is not None and (
            isinstance(final_index, bool) or not isinstance(final_index, int) or final_index < 0
        ):
            return f"{kind} grounding attestation for {element} {ref!r} has an invalid final index"
        if final_index is not None and not grounded:
            return f"{kind} grounding attestation publishes ungrounded {element} {ref!r}"
        strength = raw.get("strength")
        snapped_line = raw.get("snapped_line")
        if strength not in (None, "definition", "token"):
            return f"{kind} grounding attestation for {element} {ref!r} has an invalid strength"
        if snapped_line is not None and (
            isinstance(snapped_line, bool)
            or not isinstance(snapped_line, int)
            or snapped_line < 1
        ):
            return f"{kind} grounding attestation for {element} {ref!r} has an invalid snapped line"
        if not isinstance(raw.get("in_changed_hunk"), bool):
            return f"{kind} grounding attestation for {element} {ref!r} has an invalid hunk verdict"
        defined_at = raw.get("defined_at")
        if defined_at is not None and (not isinstance(defined_at, str) or not defined_at):
            return f"{kind} grounding attestation for {element} {ref!r} has an invalid definition"
        checks.append(raw)

    for count_name in ("proposed", "grounded_first_pass", "repaired", "pruned"):
        value = summary.get(count_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return (
                f"{kind} grounding attestation summary has an invalid "
                f"{count_name!r} count"
            )
    if summary["proposed"] != len(checks):
        return f"{kind} grounding attestation summary does not match its elements"
    if summary["pruned"] != sum(not check["grounded"] for check in checks):
        return f"{kind} grounding attestation pruned count does not match its verdicts"
    for collection, count in capped.items():
        if (
            collection not in allowed_caps
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            return f"{kind} grounding attestation has an invalid {collection!r} cap count"

    claimed: set[int] = set()

    def claim(element: str, final_index: int, ref: str | None = None) -> dict[str, Any] | None:
        matches = [
            (index, check)
            for index, check in enumerate(checks)
            if check["element"] == element
            and check["final_index"] == final_index
            and (ref is None or check["ref"] == ref)
        ]
        if len(matches) != 1:
            return None
        index, check = matches[0]
        claimed.add(index)
        return check

    if kind == "sequence":
        participants = spec["participants"]
        messages = spec["messages"]
        blocks = spec["blocks"]
        names = {participant["name"] for participant in participants}
        if len(names) != len(participants):
            return "sequence spec_final has duplicate participant names"
        for index, participant in enumerate(participants):
            if claim("participant", index, participant["name"]) is None:
                return f"sequence grounding attestation does not cover participant at final index {index}"
        for index, message in enumerate(messages):
            if message["from"] not in names or message["to"] not in names:
                return f"sequence spec_final message {index} has an unknown participant"
            if claim("message", index) is None:
                return f"sequence grounding attestation does not cover message at final index {index}"
        for block_index, block in enumerate(blocks):
            block_check = claim("block", block_index)
            if block_check is None:
                return f"sequence grounding attestation does not cover block at final index {block_index}"
            prefix = f"{block_check['ref']}."
            for branch_index, branch in enumerate(block["branches"]):
                if any(
                    isinstance(message_index, bool)
                    or message_index >= len(messages)
                    for message_index in branch["messages"]
                ):
                    return f"sequence spec_final block {block_index} cites an unknown message"
                matches = [
                    (index, check)
                    for index, check in enumerate(checks)
                    if check["element"] == "branch"
                    and check["final_index"] == branch_index
                    and check["ref"].startswith(prefix)
                ]
                if len(matches) != 1:
                    return (
                        "sequence grounding attestation does not cover branch at "
                        f"final index {block_index}.{branch_index}"
                    )
                claimed.add(matches[0][0])
        if grounding.get("root_range") is not None:
            return "sequence grounding attestation has an unexpected root range"
    else:
        root = spec["root"]
        nodes = spec["nodes"]
        edges = spec["edges"]
        root_range = grounding.get("root_range")
        if (
            not isinstance(root_range, list)
            or len(root_range) != 2
            or any(isinstance(line, bool) or not isinstance(line, int) for line in root_range)
            or root_range[0] != root["line"]
            or root_range[1] < root_range[0]
        ):
            return "flowchart grounding attestation has an invalid root range"
        if claim("root", 0, root["name"]) is None:
            return "flowchart grounding attestation does not cover root at final index 0"
        node_ids = {node["id"] for node in nodes}
        if len(node_ids) != len(nodes):
            return "flowchart spec_final has duplicate node ids"
        for index, node in enumerate(nodes):
            evidence = node["evidence"]
            if evidence["file"] != root["file"] or not (
                root_range[0] <= evidence["line"] <= root_range[1]
            ):
                return f"flowchart spec_final node {index} lies outside its root"
            if claim("node", index, node["id"]) is None:
                return f"flowchart grounding attestation does not cover node at final index {index}"
        for index, edge in enumerate(edges):
            if edge["from"] not in node_ids or edge["to"] not in node_ids:
                return f"flowchart spec_final edge {index} has an unknown endpoint"
            if claim("edge", index, f"{edge['from']}->{edge['to']}") is None:
                return f"flowchart grounding attestation does not cover edge at final index {index}"
    if any(
        check["final_index"] is not None and index not in claimed
        for index, check in enumerate(checks)
    ):
        return f"{kind} grounding attestation contains an unmatched published element"
    return None


def _diagram_head_evidence_problem(
    kind: str,
    spec: dict[str, Any],
    target_dir: Path,
    head_sha: str,
    sources: dict[str, list[str]],
) -> str | None:
    """Return a problem when a rendered diagram citation is absent at ``head_sha``."""
    from daydream.tree_sitter_index import (
        is_branch_line,
        is_executable_statement_line,
        is_terminal_line,
        language_for_path,
    )

    def check(evidence: dict[str, Any]) -> str | None:
        path = evidence["file"]
        lines = sources.get(path)
        if lines is None:
            try:
                lines = git_ops.show(target_dir, head_sha, path).decode(
                    errors="replace"
                ).splitlines()
            except GitError:
                return f"{kind} diagram evidence is missing from immutable head: {path}"
            sources[path] = lines
        if evidence["line"] > len(lines):
            return (
                f"{kind} diagram evidence line {evidence['line']} is missing from "
                f"immutable head: {path}"
            )
        return None

    if kind == "sequence":
        paths = {
            path
            for participant in spec["participants"]
            for path in participant["files"]
        }
        paths.update(message["evidence"]["file"] for message in spec["messages"])
        paths.update(
            branch["evidence"]["file"]
            for block in spec["blocks"]
            for branch in block["branches"]
        )
        with tempfile.TemporaryDirectory() as snapshot_dir:
            snapshot_root = Path(snapshot_dir)
            for path in paths:
                try:
                    source = git_ops.show(target_dir, head_sha, path)
                except GitError:
                    return f"sequence diagram evidence is missing from immutable head: {path}"
                snapshot_path = snapshot_root / path
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                snapshot_path.write_bytes(source)

            from daydream.deep.diagram_grounding import RepoSymbols, ground_sequence

            report = ground_sequence(
                spec,
                repo_root=snapshot_root,
                hunk_ranges={},
                read_paths=paths,
                symbols=RepoSymbols(snapshot_root),
            )
        if report.spec_final != spec:
            return "sequence diagram evidence is not grounded in immutable head"
    else:
        problem = check(spec["root"])
        if problem is not None:
            return problem
        for node in spec["nodes"]:
            evidence = node["evidence"]
            problem = check(evidence)
            if problem is not None:
                return problem
            source = "\n".join(sources[evidence["file"]]).encode()
            language = language_for_path(evidence["file"])
            line = evidence["line"]
            try:
                grounded = (
                    is_terminal_line(language, source, line)
                    if node["kind"] == "end"
                    else is_branch_line(language, source, line)
                    if node["kind"] == "decision"
                    else is_executable_statement_line(language, source, line)
                )
            except Exception:
                grounded = False
            if not grounded:
                return (
                    f"flowchart diagram evidence line {line} is not a valid "
                    f"{node['kind']} node in immutable head: {evidence['file']}"
                )
    return None


def validate_diagram_payload(
    payload: dict[str, Any],
    *,
    target_dir: Path | None = None,
    head_sha: str | None = None,
) -> str | None:
    """Validate each rendered spec and its grounding attestation before posting.

    The privileged poster re-renders mermaid from model-derived specs, so those
    specs are re-validated here -- against the same hand-written schema the
    author agent answered -- before a single character reaches GitHub.
    ``FINDINGS_SCHEMA`` deliberately keeps ``diagrams`` permissive (it must not
    couple to four modules' ``to_dict`` shapes), and this is the check that
    makes that safe.

    Args:
        payload: The artifact's ``diagrams`` payload.
        target_dir: Repository used to read immutable source evidence, when posting.
        head_sha: Event-derived commit used to read immutable source evidence.

    Returns:
        None when every rendered kind validates, else a human error message.
    """
    from daydream.config import (
        DIAGRAM_MAX_BLOCKS,
        DIAGRAM_MAX_EDGES,
        DIAGRAM_MAX_MESSAGES,
        DIAGRAM_MAX_NODES,
        DIAGRAM_MAX_PARTICIPANTS,
    )
    from daydream.deep.diagram_schema import FLOWCHART_SPEC_SCHEMA, SEQUENCE_SPEC_SCHEMA

    schemas = {"sequence": SEQUENCE_SPEC_SCHEMA, "flowchart": FLOWCHART_SPEC_SCHEMA}
    caps = {
        "sequence": {
            "participants": DIAGRAM_MAX_PARTICIPANTS,
            "messages": DIAGRAM_MAX_MESSAGES,
            "blocks": DIAGRAM_MAX_BLOCKS,
        },
        "flowchart": {"nodes": DIAGRAM_MAX_NODES, "edges": DIAGRAM_MAX_EDGES},
    }
    results = payload.get("results")
    if not isinstance(results, dict):
        return "diagrams payload has no 'results' object"
    sources: dict[str, list[str]] = {}
    for kind in DIAGRAM_KINDS:
        result = results.get(kind)
        if not isinstance(result, dict) or result.get("status") != "rendered":
            continue
        try:
            jsonschema.validate(result.get("spec_final"), schemas[kind])
        except jsonschema.ValidationError as exc:
            return f"{kind} spec_final failed schema validation: {exc.message}"
        spec = result["spec_final"]
        for collection, cap in caps[kind].items():
            size = len(spec[collection])
            if size > cap:
                return f"{kind} spec_final exceeds {collection} render cap: {size} > {cap}"
        if result.get("reason") is not None or result.get("omit_reasons") not in (None, []):
            return f"{kind} rendered result contradicts its status"
        problem = _diagram_grounding_problem(kind, spec, result.get("grounding"))
        if problem is not None:
            return problem
        if target_dir is not None and head_sha is not None:
            problem = _diagram_head_evidence_problem(
                kind, spec, target_dir, head_sha, sources
            )
            if problem is not None:
                return problem
    return None


def _post_diagram_artifact(
    artifact: "FindingsArtifact",
    pr: PRInfo,
    target_dir: Path,
    *,
    console: Console,
    bot_login: str | None,
) -> int:
    """Post a ``kind == "diagram"`` artifact as a standalone PR comment.

    The Phase B half of the diagram-only flow. Validates the model-derived
    specs, re-renders the mermaid from them, then posts one issue comment --
    never a review, and never through ``build_payload``.

    Returns:
        ``0`` on success, ``1`` on a rejected payload or a failed POST.
    """
    payload = artifact.diagrams
    if not isinstance(payload, dict):
        print_error(
            console,
            "Diagram Artifact Rejected",
            "artifact declares kind 'diagram' but carries no 'diagrams' payload",
        )
        return 1
    problem = validate_diagram_payload(
        payload, target_dir=target_dir, head_sha=pr.head_sha
    )
    if problem is not None:
        print_error(console, "Diagram Artifact Rejected", problem)
        return 1
    body = render_diagram_comment_body(payload)
    kinds = diagram_comment_kinds(payload)
    url, error = post_diagram_comment_to_pr(
        target_dir, pr, body=body, kinds=kinds, bot_login=bot_login
    )
    if url is None:
        suffix = f" ({error})" if error else ""
        print_error(console, "Diagram Comment Post Failed", f"No comment was posted.{suffix}")
        return 1
    print_success(console, f"Posted diagram comment: {url}")
    return 0


def post_findings_from_artifact(
    artifact_path: Path,
    *,
    pr_number: int,
    head_sha: str,
    repo: str,
    console: Console,
    bot_login: str | None = None,
    approve_on_clean: bool = False,
) -> int:
    """Post a Phase A findings artifact to the PR (the Phase B privileged poster).

    Unattended CI flow behind ``daydream post-findings``: validate the
    untrusted artifact against the event-derived facts (confused-deputy gate,
    before any GitHub write), reconcile against the bot's own prior comments
    via hidden fingerprint markers, minimize stale findings, then re-render
    and post only the new ones through the existing review payload path.
    No prompting and no ATIF trajectory — there is no agent work here.

    Args:
        artifact_path: Path to the ``--findings-out`` artifact.
        pr_number: Event-derived target PR number.
        head_sha: Event-derived PR head SHA.
        repo: Event-derived ``owner/repo`` slug.
        console: Rich console for status output.
        bot_login: Optional bot login (App slug) for prior-finding author
            filtering. When ``None``, falls back to ``$DAYDREAM_BOT_HANDLE``;
            if still unresolved, dedup degrades safely (GraphQL is still
            protected by ``viewerDidAuthor``; REST dedup is unavailable) and
            a warning is printed. Never suppresses on an unresolved login.
        approve_on_clean: Opt-in approval (issue #343). When True AND no
            finding in the artifact carries a blocking severity — new findings
            AND already-posted (matched) findings still live on the PR — the
            review is posted with ``event: "APPROVE"``; otherwise ``event:
            "COMMENT"``. Matched findings never re-post as comments; only
            their severities participate in the approval decision.

    Returns:
        ``0`` on success (including "no new findings"); ``1`` when the
        artifact fails validation, the prior-finding inventory fails, or the
        review POST fails.
    """
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

    target_dir = Path.cwd()
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
        owner=owner,
        repo=repo_name,
        url="",
    )

    # Issue #1113: a diagram artifact carries no fingerprints, so it must never
    # reach the reconcile path below -- ``partition([], prior)`` would classify
    # EVERY prior review finding as stale and ``resolve_threads`` would minimize
    # the bot's real open findings. Branch before ``fetch_prior_findings``.
    if artifact.kind == "diagram":
        return _post_diagram_artifact(
            artifact, pr, target_dir, console=console, bot_login=effective_login
        )

    diagram_blocks: str | None = None
    if isinstance(artifact.diagrams, dict):
        problem = validate_diagram_payload(
            artifact.diagrams, target_dir=target_dir, head_sha=pr.head_sha
        )
        if problem is not None:
            print_error(console, "Diagram Payload Rejected", problem)
            return 1
        diagram_blocks = render_diagram_blocks_from_payload(artifact.diagrams) or None

    try:
        prior = fetch_prior_findings(target_dir, repo, pr_number, bot_login=effective_login)
    except GitError as exc:
        print_error(console, "Prior-Finding Inventory Failed", str(exc))
        return 1

    plan = partition([f.fingerprint for f in artifact.findings], prior)
    if plan.stale:
        resolved, failed = resolve_threads(target_dir, plan.stale)
        print_info(console, f"Stale findings minimized: {resolved} succeeded, {failed} failed.")

    new_fingerprints = set(plan.new)
    classified = _ClassifiedIssues()
    for finding in artifact.findings:
        if finding.fingerprint not in new_fingerprints:
            continue
        issue = _issue_from_artifact_finding(finding)
        if finding.placement == "inline" and finding.line is not None:
            classified.inline.append(_inline_comment(issue, finding.line))
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
    can_approve = approve_on_clean and not any(
        _finding_blocks_approval(
            finding.severity,
            finding.location_distrust,
            finding.severity_off_vocabulary,
            finding.severity_before_demotion,
        )
        for finding in artifact.findings
    )

    if classified.is_empty() and not can_approve and diagram_blocks is None:
        print_info(
            console,
            f"No new findings to post ({len(plan.matched)} already on PR #{pr_number}).",
        )
        return 0

    posted_files, failed_files = _submit_file_level_comments(target_dir, pr, classified.file_level)
    if failed_files:
        classified.file_level = posted_files
        classified.body_only.extend(failed_files)
        print_info(
            console,
            f"{len(failed_files)} file-level comment(s) failed to post; folded into the review body.",
        )

    payload = build_payload(
        pr,
        classified,
        run_info_override=artifact.run_info,
        approve_on_clean=can_approve,
        diagram_blocks=diagram_blocks,
    )
    review_url, error_msg = _submit_review(target_dir, pr, payload)
    if review_url is None:
        suffix = f" ({error_msg})" if error_msg else ""
        already = (
            f"{len(posted_files)} file-level comment(s) were already posted."
            if posted_files
            else "No comments were posted."
        )
        print_error(console, "PR Review Post Failed", f"{already}{suffix}")
        return 1
    print_success(console, f"Posted review: {review_url}")
    return 0


def _issue_from_artifact_finding(finding: ArtifactFinding) -> ParsedIssue:
    """Rebuild a :class:`ParsedIssue` from a validated artifact finding.

    The artifact carries raw issue fields with placement already resolved by
    Phase A's :func:`classify`, so rendering here needs no PR git objects.
    """
    return ParsedIssue(
        path=finding.path,
        line=finding.line,
        title=finding.title,
        body=finding.body,
        is_cross_stack=finding.is_cross_stack,
        confidence=finding.confidence,
        severity=finding.severity,
        fingerprint=finding.fingerprint,
        location_distrust=finding.location_distrust,
        severity_before_demotion=finding.severity_before_demotion,
        severity_off_vocabulary=finding.severity_off_vocabulary,
    )
