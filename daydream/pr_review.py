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

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    mode_label: str = "deep review",
) -> None:
    """Parse a deep-mode `.review-output.md` and offer to post to the PR.

    Args:
        target_dir: Repo root.
        report_path: Path to the merged review report.
        console: Rich console for user-facing output.
        mode_label: Short label for the review mode (used in body template).
    """
    if not report_path.exists():
        return
    text = report_path.read_text()
    issues = parse_report(text)
    if not issues:
        print_info(console, "No parseable issues in review output; skipping PR post.")
        return
    await _post(target_dir, issues, console=console, mode_label=mode_label)


async def post_review_to_pr_from_alt_issues(
    target_dir: Path,
    alt_issues: list[dict[str, Any]],
    *,
    console: Console,
    mode_label: str = "trust-the-technology",
) -> None:
    """Convert alt-review issues (from `--ttt`) and offer to post to the PR.

    Args:
        target_dir: Repo root.
        alt_issues: Issue dicts from `phase_alternative_review`.
        console: Rich console for user-facing output.
        mode_label: Short label for the review mode (used in body template).
    """
    issues = alt_issues_to_parsed(alt_issues)
    if not issues:
        return
    await _post(target_dir, issues, console=console, mode_label=mode_label)


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
DAYDREAM_FOOTER = f"<sub>🧙 Posted by [daydream]({DAYDREAM_REPO_URL})</sub>"
_NEXT_SECTION = re.compile(r"^## ", re.MULTILINE)
# Matches "N. [path:line] Title" or "N. [path] Title" with optional
# leading `[cross-stack]` marker.
_ISSUE_HEAD = re.compile(
    r"^(?P<num>\d+)\.\s+"
    r"(?:\[cross-stack\]\s+)?"
    r"\[(?P<path>[^\]:]+)(?::(?P<line>\d+))?\]\s+"
    r"(?P<title>.+?)\s*$",
    re.MULTILINE,
)


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
        section_end = section_start + next_section.start() if next_section else len(text)
        section_text = text[section_start:section_end]
        section_is_xstack = header_match.start() == xstack_start

        matches = list(_ISSUE_HEAD.finditer(section_text))
        for i, m in enumerate(matches):
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
            body = section_text[body_start:body_end].strip()
            title = m.group("title").strip()
            line_str = m.group("line")
            issues.append(
                ParsedIssue(
                    path=m.group("path").strip(),
                    line=int(line_str) if line_str else None,
                    title=title,
                    body=body,
                    is_cross_stack=section_is_xstack or title.lower().startswith("[cross-stack]"),
                    confidence=_extract_confidence(body),
                    severity=_extract_severity(body),
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


def _run(
    cmd: list[str], cwd: Path, *, timeout: int = 15, input_bytes: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Run a subprocess with bytes IO; all args hardcoded or derived from JSON."""
    return subprocess.run(  # noqa: S603 - args are hardcoded or from parsed gh/git output
        cmd,
        cwd=cwd,
        capture_output=True,
        input=input_bytes,
        timeout=timeout,
        shell=False,
    )


def _current_branch(target_dir: Path) -> str | None:
    try:
        r = _run(["git", "branch", "--show-current"], target_dir, timeout=5)
        return r.stdout.decode().strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def find_open_pr(target_dir: Path) -> PRInfo | None:
    """Locate the open PR for the current branch. Returns None if not found."""
    branch = _current_branch(target_dir)
    if not branch:
        return None
    try:
        r = _run(
            [
                "gh",
                "pr",
                "list",
                "--head",
                branch,
                "--state",
                "open",
                "--json",
                "number,headRefOid,baseRefOid,baseRefName,url,headRepository,headRepositoryOwner",
            ],
            target_dir,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if r.returncode != 0:
        return None
    try:
        rows = json.loads(r.stdout.decode() or "[]")
    except json.JSONDecodeError:
        return None
    if not rows:
        return None
    row = rows[0]
    # Owner/repo lookup via `gh repo view` (handles fork cases cleanly).
    owner, repo = _repo_owner_name(target_dir)
    if owner is None or repo is None:
        return None
    return PRInfo(
        number=int(row["number"]),
        head_sha=row["headRefOid"],
        base_sha=row["baseRefOid"],
        base_ref=row.get("baseRefName", ""),
        owner=owner,
        repo=repo,
        url=row.get("url", ""),
    )


def _repo_owner_name(target_dir: Path) -> tuple[str | None, str | None]:
    try:
        r = _run(
            ["gh", "repo", "view", "--json", "owner,name"],
            target_dir,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None, None
    if r.returncode != 0:
        return None, None
    try:
        data = json.loads(r.stdout.decode())
    except json.JSONDecodeError:
        return None, None
    owner = data.get("owner", {}).get("login")
    name = data.get("name")
    return owner, name


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


def resolve_line(
    target_dir: Path, head_sha: str, issue: ParsedIssue
) -> int | None:
    """Resolve the true line in the head commit for an issue.

    Tries (in order):
      1. If the issue has a line hint, verify the anchor appears within
         +/-5 lines; trust it on match.
      2. Otherwise search the whole file at head for the first anchor hit.
    Returns None if the file doesn't exist at head or no anchor matches.
    """
    try:
        r = _run(["git", "show", f"{head_sha}:{issue.path}"], target_dir, timeout=10)
    except (subprocess.SubprocessError, OSError):
        return None
    if r.returncode != 0:
        return None
    lines = r.stdout.decode(errors="replace").splitlines()
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
    try:
        r = _run(
            [
                "git",
                "diff",
                "--unified=3",
                f"{base_sha}..{head_sha}",
                "--",
                path,
            ],
            target_dir,
            timeout=20,
        )
        if r.returncode == 0:
            diff_text = r.stdout.decode(errors="replace")
        else:
            git_failed = True
            diff_text = ""
    except (subprocess.SubprocessError, OSError):
        git_failed = True
        diff_text = ""

    if git_failed and pr_number is not None:
        diff_text = _gh_pr_diff_for_path(target_dir, pr_number, path)

    return _parse_hunks(diff_text)


def _gh_pr_diff_for_path(target_dir: Path, pr_number: int, path: str) -> str:
    """Fetch the PR's full diff via `gh pr diff` and return just the block for `path`."""
    try:
        r = _run(
            ["gh", "pr", "diff", str(pr_number)],
            target_dir,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return ""
    if r.returncode != 0:
        return ""
    full_diff = r.stdout.decode(errors="replace")
    # Pick the `diff --git a/<path> b/<path>` block.
    needle_a = f"a/{path} "
    needle_b = f"b/{path}\n"
    for block in _DIFF_BLOCK_SPLIT.split(full_diff):
        if not block.startswith("diff --git "):
            continue
        header_line = block.split("\n", 1)[0]
        if needle_a in header_line or header_line.endswith(f"b/{path}") or needle_b in header_line:
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


def within_hunk(line: int, hunks: list[tuple[int, int]], tolerance: int = 3) -> bool:
    return any(start - tolerance <= line <= end + tolerance for start, end in hunks)


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
        if not within_hunk(line, hunks_cache[issue.path]):
            out.body_only.append(issue)
            continue
        out.inline.append(
            {
                "path": issue.path,
                "line": line,
                "side": "RIGHT",
                "body": _format_inline_body(issue),
            }
        )
        out.inline_issues.append(issue)
    return out


def _format_inline_body(issue: ParsedIssue) -> str:
    header = f"**{issue.title}**" if issue.title else ""
    tags = _format_tag_line(issue)
    parts = [p for p in (header, tags, issue.body) if p]
    return "\n\n".join(parts + [DAYDREAM_FOOTER]).strip()


def _format_tag_line(issue: ParsedIssue) -> str:
    """Render severity/confidence badges for a single issue, if set."""
    bits: list[str] = []
    if issue.severity:
        bits.append(f"severity: `{issue.severity}`")
    if issue.confidence:
        bits.append(f"confidence: `{issue.confidence}`")
    return " · ".join(bits)


def _format_body_section(body_only: list[ParsedIssue]) -> str:
    if not body_only:
        return ""
    lines = ["## Non-inline findings\n"]
    for issue in body_only:
        prefix = "[cross-stack] " if issue.is_cross_stack else ""
        loc = f"`{issue.path}`" + (f":{issue.line}" if issue.line else "")
        tags = _format_tag_line(issue)
        tag_block = f"\n{tags}\n" if tags else ""
        lines.append(f"### {prefix}{issue.title} ({loc})\n{tag_block}\n{issue.body}\n")
    return "\n".join(lines)


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


def build_payload(
    pr: PRInfo, mode_label: str, classified: _ClassifiedIssues
) -> dict[str, Any]:
    """Assemble the review payload for `POST /repos/.../pulls/<n>/reviews`.

    The review body follows a fixed template so every run looks the same:
        🧙 Daydream <mode_label> review
        N inline, M non-inline
        Severity / Confidence breakdowns (when known)
        Non-inline findings (when any)
        Repo link footer
    """
    all_issues_with_inline_meta = [*classified.body_only]
    # Inline comments live as dicts in classified.inline; we still want
    # their confidence/severity in the summary counts, so peek at the
    # original ParsedIssue list by cross-referencing path+line is brittle.
    # Instead, carry counts we already have from body_only plus inline
    # metadata we stamp at classify time via the ParsedIssue objects.
    # (See classify: we store the original issue on _ClassifiedIssues.)
    all_issues_with_inline_meta.extend(classified.inline_issues)

    inline_count = len(classified.inline)
    body_count = len(classified.body_only)

    body_chunks: list[str] = []
    body_chunks.append(f"🧙 [Daydream]({DAYDREAM_REPO_URL}) {mode_label} review")
    body_chunks.append(f"{inline_count} inline comment(s), {body_count} non-inline finding(s).")

    severity_parts = _count_labels(
        all_issues_with_inline_meta, "severity", ("high", "medium", "low")
    )
    if severity_parts:
        body_chunks.append("**Severity:** " + ", ".join(severity_parts))

    confidence_parts = _count_labels(
        all_issues_with_inline_meta, "confidence", ("HIGH", "MEDIUM", "LOW")
    )
    if confidence_parts:
        body_chunks.append("**Confidence:** " + ", ".join(confidence_parts))

    body_section = _format_body_section(classified.body_only)
    if body_section:
        body_chunks.append(body_section)

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
    mode_label: str,
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
        print_info(console, "No postable issues after classification; skipping PR post.")
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

    payload = build_payload(pr, mode_label, classified)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"pr-{pr.number}-review-",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as tf:
        tf.write(json.dumps(payload, indent=2))
        payload_path = Path(tf.name)

    review_url = _submit_review(target_dir, pr, payload_path)
    if review_url is None:
        print_warning(
            console,
            f"Failed to post PR review; no comments were posted. Payload kept at {payload_path}",
        )
        return

    # Success: clean up the tmp payload (gate 3).
    payload_path.unlink(missing_ok=True)
    print_success(console, f"Posted review: {review_url}")


def _submit_review(
    target_dir: Path, pr: PRInfo, payload_path: Path
) -> str | None:
    """POST the review payload via `gh api`. Returns html_url or None on failure."""
    try:
        r = _run(
            [
                "gh",
                "api",
                "--method",
                "POST",
                f"/repos/{pr.owner}/{pr.repo}/pulls/{pr.number}/reviews",
                "--input",
                str(payload_path),
            ],
            target_dir,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if r.returncode != 0:
        return None
    try:
        data = json.loads(r.stdout.decode())
    except json.JSONDecodeError:
        return None
    url = data.get("html_url")
    return str(url) if url else None
