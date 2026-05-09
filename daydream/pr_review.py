"""Post daydream review findings as inline comments on the current branch's PR.

Shared by deep-review mode (parses `.review-output.md`) and
trust-the-technology mode (consumes alt-review issues directly).

Flow:
    1. Locate the open PR for the current branch via `gh pr list`.
    2. Parse issues (from report markdown or alt-issue dicts).
    3. Resolve each issue to a real head-SHA line via anchor grep.
    4. Classify into inline (line within a diff hunk) vs body-only.
    5. Build a single review payload, show a summary, ask y/n.
    6. On yes, POST to `/repos/<owner>/<repo>/pulls/<num>/reviews`.

Everything is best-effort: failures warn and return, never raise.
"""

from __future__ import annotations

import contextlib
import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import daydream
from daydream import git_ops
from daydream.git_ops import GitError
from daydream.pr_comment_renderer import render_run_info_block
from daydream.trajectory import TrajectoryRecorder, get_current_recorder
from daydream.ui import print_info, print_success, print_warning, prompt_user

if TYPE_CHECKING:
    from rich.console import Console


# --- Data shapes ------------------------------------------------------------


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
    """

    path: str
    line: int | None
    title: str
    body: str
    is_cross_stack: bool = False
    confidence: str | None = None
    severity: str | None = None


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


@dataclass
class _ClassifiedIssues:
    inline: list[dict[str, Any]] = field(default_factory=list)
    body_only: list[ParsedIssue] = field(default_factory=list)
    # Parallel list to `inline`: the original ParsedIssue for each inline
    # comment. Used to roll severity/confidence into the summary body.
    inline_issues: list[ParsedIssue] = field(default_factory=list)


# --- Public entry points ----------------------------------------------------


async def post_review_to_pr_from_report(
    target_dir: Path,
    report_path: Path,
    *,
    console: Console,
) -> None:
    """Parse a deep-mode `.review-output.md` and offer to post to the PR.

    Args:
        target_dir: Repo root.
        report_path: Path to the merged review report.
        console: Rich console for user-facing output.
    """
    if not report_path.exists():
        return
    text = report_path.read_text()
    issues = parse_report(text)
    if not issues:
        print_info(console, "No parseable issues in review output; skipping PR post.")
        return
    await _post(target_dir, issues, console=console)


async def post_review_to_pr_from_alt_issues(
    target_dir: Path,
    alt_issues: list[dict[str, Any]],
    *,
    console: Console,
    plan_data: dict[str, Any] | None = None,
) -> None:
    """Convert alt-review issues (from `--comment`) and offer to post to the PR.

    Args:
        target_dir: Repo root.
        alt_issues: Issue dicts from `phase_alternative_review`.
        console: Rich console for user-facing output.
        plan_data: Structured plan from ``phase_generate_plan`` (``--plan``).
            When provided the consolidated agent prompt in the PR comment
            includes per-issue change instructions from the plan.
    """
    issues = alt_issues_to_parsed(alt_issues)
    if not issues:
        return
    await _post(target_dir, issues, console=console, plan_data=plan_data)


# --- Parsers ---------------------------------------------------------------


_ISSUES_HEADER = re.compile(r"^## (?:Cross-Stack Issues|Issues)\s*$", re.MULTILINE)
_XSTACK_HEADER = re.compile(r"^## Cross-Stack Issues\s*$", re.MULTILINE)
# Tolerates bold markers in either position: "**Confidence:** HIGH",
# "**Confidence**: HIGH", or bare "Confidence: HIGH".
_CONFIDENCE_LINE = re.compile(
    r"[Cc]onfidence[:*\s]+(HIGH|MEDIUM|LOW|High|Medium|Low|high|medium|low)"
)
_SEVERITY_LINE = re.compile(
    r"[Ss]everity[:*\s]+(high|medium|low|HIGH|MEDIUM|LOW|High|Medium|Low)"
)

DAYDREAM_REPO_URL = "https://github.com/existential-birds/daydream"
DAYDREAM_FOOTER = (
    f"<sub>🧙 Posted by [daydream v{daydream.__version__}]({DAYDREAM_REPO_URL})</sub>"
)
_NEXT_SECTION = re.compile(r"^## ", re.MULTILINE)
# Matches "N. [path:line] Title" or "N. [path] Title" with optional leading
# `[cross-stack]` marker. Tolerates Markdown bold/italic wrapping the whole
# head (`N. **[path] title**`) and multiple comma-separated paths in the
# bracket. The closing `**`/`__` is only stripped when an opening marker was
# matched (conditional backref on `bold`), so bold inside the title is
# preserved when the head itself isn't wrapped.
_ISSUE_HEAD = re.compile(
    r"^(?P<num>\d+)\.\s+"
    r"(?P<bold>\*\*|__)?"
    r"(?:\[cross-stack\]\s+)?"
    r"\[(?P<paths>[^\]]+)\]\s+"
    r"(?P<title>.+?)"
    r"(?(bold)(?P=bold)|)"
    r"\s*$",
    re.MULTILINE,
)


def _split_paths(paths: str) -> list[tuple[str, int | None]]:
    """Parse the bracket contents into (path, line) pairs.

    Accepts either a single `path[:line]` or a comma-separated list, so
    `a.ts:1, b.go:41, c.py:48` fans out into three pairs.
    """
    out: list[tuple[str, int | None]] = []
    for chunk in paths.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        path, sep, line_str = chunk.partition(":")
        path = path.strip()
        line: int | None = None
        if sep:
            try:
                line = int(line_str.strip())
            except ValueError:
                # Malformed line hint -- keep the raw chunk as the path.
                path = chunk
        out.append((path, line))
    return out


def parse_report(text: str) -> list[ParsedIssue]:
    """Extract issues from a deep-mode merged review report.

    Recognises the `## Issues` and `## Cross-Stack Issues` sections and
    reads each numbered entry. Cross-stack entries (or entries whose
    title starts with `[cross-stack]`) are tagged accordingly.
    """
    issues: list[ParsedIssue] = []
    xstack_match = _XSTACK_HEADER.search(text)
    xstack_start = xstack_match.start() if xstack_match else -1

    for header_match in _ISSUES_HEADER.finditer(text):
        section_start = header_match.end()
        # Find where this section ends (next "## " or EOF).
        rest = text[section_start:]
        next_section = _NEXT_SECTION.search(rest)
        section_end = (
            section_start + next_section.start() if next_section else len(text)
        )
        section_text = text[section_start:section_end]
        section_is_xstack = header_match.start() == xstack_start

        matches = list(_ISSUE_HEAD.finditer(section_text))
        for i, m in enumerate(matches):
            body_start = m.end()
            body_end = (
                matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
            )
            body = section_text[body_start:body_end].strip()
            title = m.group("title").strip()
            is_xstack = section_is_xstack or title.lower().startswith("[cross-stack]")
            confidence = _extract_confidence(body)
            severity = _extract_severity(body)
            # Fan out multi-path brackets into one ParsedIssue per file, mirroring
            # alt_issues_to_parsed. Single-path entries produce a single issue.
            for path, line in _split_paths(m.group("paths")):
                issues.append(
                    ParsedIssue(
                        path=path,
                        line=line,
                        title=title,
                        body=body,
                        is_cross_stack=is_xstack,
                        confidence=confidence,
                        severity=severity,
                    )
                )

    return issues


def _extract_confidence(text: str) -> str | None:
    m = _CONFIDENCE_LINE.search(text)
    return m.group(1).upper() if m else None


def _extract_severity(text: str) -> str | None:
    m = _SEVERITY_LINE.search(text)
    return m.group(1).lower() if m else None


def alt_issues_to_parsed(alt_issues: list[dict[str, Any]]) -> list[ParsedIssue]:
    """Convert `phase_alternative_review` dicts into ParsedIssue objects.

    Alt issues have a `files: list[str]` field and no line hint. When
    multiple files are listed we emit one issue per file (classifier will
    fold file-level issues into the review body).
    """
    out: list[ParsedIssue] = []
    for raw in alt_issues:
        files = raw.get("files") or []
        if not files:
            continue
        title = str(raw.get("title", "")).strip()
        description = str(raw.get("description", "")).strip()
        recommendation = str(raw.get("recommendation", "")).strip()
        severity = str(raw.get("severity", "")).strip().lower() or None
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
                )
            )
    return out


# --- Git / gh helpers ------------------------------------------------------


def _current_branch(target_dir: Path) -> str | None:
    try:
        return git_ops.current_branch(target_dir)
    except GitError:
        return None


def find_open_pr(target_dir: Path) -> PRInfo | None:
    """Locate the open PR for the current branch. Returns None if not found."""
    branch = _current_branch(target_dir)
    if not branch:
        return None
    rows = git_ops.gh_pr_list_for_branch(target_dir, branch)
    if not rows:
        return None
    row = rows[0]
    # Owner/repo lookup via `gh repo view` (handles fork cases cleanly).
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
    )


# --- Line resolution + hunk classification --------------------------------


_ANCHOR_TOKEN = re.compile(r"`([^`\n]{3,80})`|\b([A-Za-z_][A-Za-z0-9_]{4,})\b")


def extract_anchors(issue: ParsedIssue) -> list[str]:
    """Pull candidate anchor tokens from an issue body (longest first).

    Prefers backtick-quoted identifiers (e.g. `foo_bar`) since those are
    the most specific signals of code the reviewer cited. Falls back to
    any alphanumeric word of length >=5.
    """
    seen: list[str] = []
    for m in _ANCHOR_TOKEN.finditer(f"{issue.title}\n{issue.body}"):
        token = m.group(1) or m.group(2)
        if token and token not in seen:
            seen.append(token)
    # Longest-first improves hit quality (generic words lose to identifiers).
    seen.sort(key=len, reverse=True)
    return seen[:8]


def resolve_line(target_dir: Path, head_sha: str, issue: ParsedIssue) -> int | None:
    """Resolve the true line in the head commit for an issue.

    Tries (in order):
      1. If the issue has a line hint, verify the anchor appears within
         +/-5 lines; trust it on match.
      2. Otherwise search the whole file at head for the first anchor hit.
    Returns None if the file doesn't exist at head or no anchor matches.
    """
    try:
        raw = git_ops.show(target_dir, head_sha, issue.path)
    except GitError:
        return None
    lines = raw.decode(errors="replace").splitlines()
    if not lines:
        return None

    anchors = extract_anchors(issue)

    # Step 1: verify hint.
    if issue.line is not None and 1 <= issue.line <= len(lines):
        for anchor in anchors:
            lo = max(1, issue.line - 5)
            hi = min(len(lines), issue.line + 5)
            if any(anchor in lines[i - 1] for i in range(lo, hi + 1)):
                return issue.line
        # Hint didn't verify; fall through to full-file search.

    # Step 2: full-file search.
    for anchor in anchors:
        for i, line in enumerate(lines, start=1):
            if anchor in line:
                return i

    return None


_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
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
        target_dir: Repo root.
        base_sha: Base commit SHA (may be unreachable locally after a rebase).
        head_sha: Head commit SHA.
        path: Repo-relative file path.
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


def _parse_hunks(diff_text: str) -> list[tuple[int, int]]:
    hunks: list[tuple[int, int]] = []
    for m in _HUNK_HEADER.finditer(diff_text):
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) else 1
        if count == 0:
            continue
        hunks.append((start, start + count - 1))
    return hunks


def snap_to_hunk(
    line: int, hunks: list[tuple[int, int]], tolerance: int = HUNK_TOLERANCE
) -> int | None:
    """Return a valid in-hunk line for a PR comment, or None if too far.

    If ``line`` falls inside a hunk, return it unchanged. If it is within
    ``tolerance`` lines of a hunk boundary, snap to the nearest boundary
    so the GitHub API receives a line that actually appears in the diff.
    Returns ``None`` when the line is beyond tolerance of every hunk.

    Args:
        line: Candidate line number on the head side.
        hunks: (start, end) inclusive ranges from ``_parse_hunks``.
        tolerance: Max distance from a hunk boundary to still snap.

    Returns:
        A line number guaranteed to be inside a hunk, or None.
    """
    best: int | None = None
    best_dist = tolerance + 1
    for start, end in hunks:
        if start <= line <= end:
            return line
        if line < start:
            dist = start - line
            candidate = start
        else:
            dist = line - end
            candidate = end
        if dist <= tolerance and dist < best_dist:
            best = candidate
            best_dist = dist
    return best


# --- Classification + payload build ---------------------------------------


def classify(
    target_dir: Path, pr: PRInfo, issues: list[ParsedIssue]
) -> _ClassifiedIssues:
    """Split issues into inline vs body-only based on diff hunks."""
    out = _ClassifiedIssues()
    hunks_cache: dict[str, list[tuple[int, int]]] = {}
    for issue in issues:
        if issue.is_cross_stack:
            out.body_only.append(issue)
            continue
        line = resolve_line(target_dir, pr.head_sha, issue)
        if line is None:
            out.body_only.append(issue)
            continue
        if issue.path not in hunks_cache:
            hunks_cache[issue.path] = file_hunks(
                target_dir,
                pr.base_sha,
                pr.head_sha,
                issue.path,
                pr_number=pr.number,
            )
        snapped = snap_to_hunk(line, hunks_cache[issue.path])
        if snapped is None:
            out.body_only.append(issue)
            continue
        out.inline.append(
            {
                "path": issue.path,
                "line": snapped,
                "side": "RIGHT",
                "body": _format_inline_body(issue),
            }
        )
        out.inline_issues.append(issue)
    return out


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


def _format_inline_body(issue: ParsedIssue) -> str:
    emoji = _severity_emoji(issue.severity)
    title_prefix = f"{emoji} " if emoji else ""
    header = f"{title_prefix}**{issue.title}**" if issue.title else ""
    tags = _format_tag_line(issue)
    header_line = f"{header} | {tags}" if header and tags else header or tags
    parts = [p for p in (header_line, issue.body) if p]
    agent_prompt = _build_agent_prompt(issue)
    parts.append(agent_prompt)
    parts.append(DAYDREAM_FOOTER)
    return "\n\n".join(parts).strip()


def _format_tag_line(issue: ParsedIssue) -> str:
    """Render severity/confidence badges for a single issue, if set."""
    bits: list[str] = []
    if issue.severity:
        bits.append(f"severity: `{issue.severity}`")
    if issue.confidence:
        bits.append(f"confidence: `{issue.confidence}`")
    return " · ".join(bits)


def _build_agent_prompt(issue: ParsedIssue) -> str:
    """Build a collapsible AI-agent-friendly prompt for a single issue."""
    loc = f"`{issue.path}`"
    if issue.line:
        loc += f" around line {issue.line}"
    # Combine title and body into a condensed instruction.
    instruction = issue.title
    if issue.body:
        # Take the first meaningful line of the body as additional context.
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


def _group_by_file(issues: list[ParsedIssue]) -> dict[str, list[ParsedIssue]]:
    """Group issues by file path, preserving insertion order."""
    groups: dict[str, list[ParsedIssue]] = {}
    for issue in issues:
        groups.setdefault(issue.path, []).append(issue)
    return groups


def _format_body_section(body_only: list[ParsedIssue]) -> str:
    if not body_only:
        return ""
    grouped = _group_by_file(body_only)
    total = len(body_only)
    parts: list[str] = [
        "<details>",
        f"<summary>📋 Non-inline findings ({total})</summary><blockquote>\n",
    ]
    for filepath, file_issues in grouped.items():
        parts.append("<details>")
        parts.append(
            f"<summary>{filepath} ({len(file_issues)})</summary><blockquote>\n"
        )
        for i, issue in enumerate(file_issues):
            prefix = "[cross-stack] " if issue.is_cross_stack else ""
            emoji = _severity_emoji(issue.severity)
            title_prefix = f"{emoji} " if emoji else ""
            tags = _format_tag_line(issue)
            header = f"{title_prefix}**{prefix}{issue.title}**"
            header_line = f"{header} | {tags}" if tags else header
            parts.append(header_line)
            if issue.body:
                parts.append(f"\n{issue.body}\n")
            parts.append(_build_agent_prompt(issue))
            if i < len(file_issues) - 1:
                parts.append("\n---\n")
        parts.append("\n</blockquote></details>")
    parts.append("\n</blockquote></details>")
    return "\n".join(parts)


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
    *,
    plan_data: dict[str, Any] | None = None,
) -> str:
    """Build a single collapsible prompt block that tells AI agents to fetch and fix review comments."""
    total = len(classified.inline_issues) + len(classified.body_only)

    plan_section = _format_plan_for_prompt(plan_data) if plan_data else ""

    prompt_body = (
        f"Fix the {total} review comment(s) posted on this PR.\n"
        "\n"
        "If the /beagle-core:fetch-pr-feedback skill is available, run:\n"
        f"/beagle-core:fetch-pr-feedback --pr {pr.number}\n"
        "\n"
        "Otherwise, fetch the comments manually:\n"
        f"1. gh api repos/{pr.owner}/{pr.repo}/pulls/{pr.number}/comments\n"
        f"2. gh api repos/{pr.owner}/{pr.repo}/issues/{pr.number}/comments\n"
        "\n"
        "These endpoints return all comments on the PR. Focus on the most\n"
        "recent review — ignore older review threads that have already been\n"
        "addressed. For each comment: read the referenced file, verify the\n"
        "finding against the current code, and fix it if valid. Skip false\n"
        "positives. Commit all fixes when done."
    )
    if plan_section:
        prompt_body += "\n\n" + plan_section

    return (
        "<details>\n"
        "<summary>🔮 Prompt for all review comments with AI agents</summary>\n\n"
        f"```\n{prompt_body}\n```\n\n"
        "</details>"
    )


def _format_plan_for_prompt(plan_data: dict[str, Any]) -> str:
    """Render structured plan data as actionable instructions for the agent prompt."""
    plan = plan_data.get("plan", plan_data)
    plan_issues = plan.get("issues", [])
    if not plan_issues:
        return ""

    lines = ["## Implementation Plan", ""]
    for issue in plan_issues:
        title = issue.get("title", "Untitled")
        lines.append(f"### #{issue.get('id', '?')} {title}")
        changes = issue.get("changes", [])
        for change in changes:
            action = change.get("action", "modify")
            file_path = change.get("file", "?")
            desc = change.get("description", "")
            lines.append(f"- {action} `{file_path}`: {desc}")
            refs = change.get("references", [])
            for ref in refs:
                lines.append(f"  - ref: `{ref.get('file', '')}:{ref.get('symbol', '')}`")
        lines.append("")

    return "\n".join(lines)


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


def build_payload(
    pr: PRInfo,
    classified: _ClassifiedIssues,
    *,
    plan_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the review payload for `POST /repos/.../pulls/<n>/reviews`.

    The review body uses collapsible sections so large reviews stay readable:
        **Code Review Summary**
        <details> Non-inline findings grouped by file
        <details> Consolidated AI agent prompt
        <details> Review info (enriched run-info + version footer)
        Footer (🧙 Posted by daydream vX.Y.Z)
    """
    all_issues_with_inline_meta = [*classified.body_only]
    all_issues_with_inline_meta.extend(classified.inline_issues)

    body_chunks: list[str] = []
    body_chunks.append("**Code Review Summary**")

    body_section = _format_body_section(classified.body_only)
    if body_section:
        body_chunks.append(body_section)

    # Consolidated AI agent prompt.
    if classified.inline_issues or classified.body_only:
        body_chunks.append(_build_consolidated_prompt(classified, pr, plan_data=plan_data))

    # Collapsible review info: enriched run-info (rollup + per-phase
    # breakdown + version footer, owned by the renderer) followed by the
    # existing severity/confidence breakdown. The renderer emits its own
    # ``<sub>Generated by daydream...</sub>`` footer, so don't double it.
    enriched_run_info = _render_review_info_block()
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
    review_info = enriched_run_info
    if extra_info_lines:
        review_info = f"{review_info}\n\n" + "\n".join(extra_info_lines)
    body_chunks.append(
        "<details>\n"
        "<summary>ℹ️ Review info</summary>\n\n"
        f"{review_info}\n\n"
        "</details>"
    )

    # DAYDREAM_FOOTER is the bottom-of-comment "🧙 Posted by daydream"
    # badge — distinct from the renderer's "Generated by daydream" line
    # inside the review-info block.
    body_chunks.append(DAYDREAM_FOOTER)

    payload: dict[str, Any] = {
        "commit_id": pr.head_sha,
        "event": "COMMENT",
        "body": "\n\n".join(body_chunks),
        "comments": classified.inline,
    }
    return payload


# --- Core orchestration ---------------------------------------------------


async def _post(
    target_dir: Path,
    issues: list[ParsedIssue],
    *,
    console: Console,
    plan_data: dict[str, Any] | None = None,
) -> None:
    pr = find_open_pr(target_dir)
    if pr is None:
        print_warning(
            console,
            "No open PR found for the current branch; skipping PR post.",
        )
        return

    classified = classify(target_dir, pr, issues)
    if not classified.inline and not classified.body_only:
        print_info(
            console, "No postable issues after classification; skipping PR post."
        )
        return

    inline_files = sorted({c["path"] for c in classified.inline})
    summary = (
        f"{len(classified.inline)} inline on "
        f"{', '.join(inline_files) if inline_files else '(none)'}, "
        f"{len(classified.body_only)} folded into body"
    )
    print_info(console, f"PR #{pr.number}: {summary}")

    answer = prompt_user(console, "Post these as a PR review? [y/N]", "n")
    if answer.strip().lower() not in ("y", "yes"):
        print_info(console, "Skipped posting to PR.")
        return

    payload = build_payload(pr, classified, plan_data=plan_data)
    review_url, error_msg = _submit_review(target_dir, pr, payload)
    if review_url is None:
        # ``error_msg`` carries the GitError text from git_ops, which includes
        # the preserved tempfile path on failure (see git_ops.gh_api).
        suffix = f" ({error_msg})" if error_msg else ""
        print_warning(
            console,
            f"Failed to post PR review; no comments were posted.{suffix}",
        )
        return

    print_success(console, f"Posted review: {review_url}")


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
