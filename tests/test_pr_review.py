"""Unit tests for daydream.pr_review."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops, pr_review
from daydream.pr_review import (
    ParsedIssue,
    PRInfo,
    _parse_hunks,
    alt_issues_to_parsed,
    build_payload,
    classify,
    extract_anchors,
    parse_report,
    snap_to_hunk,
)

# gh-gated marker: gh isn't always installed in CI, so tests that need to
# stub out gh's subprocess (rather than use real git) are skipped if missing.
_gh_available = shutil.which("gh") is not None
gh_required = pytest.mark.skipif(not _gh_available, reason="gh CLI not installed")


def _git(repo: Path, *args: str) -> str:
    """Local helper for tests that need to script git directly."""
    proc = subprocess.run(  # noqa: S603 - arguments are not user-controlled
        ["git", *args],  # noqa: S607 - git is a trusted command
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()

REPORT_FIXTURE = """\
# Review

## Per-Stack Context

Some prose.

## Issues
1. [src/foo.py:42] **Null check missing**
   The function `compute_total` dereferences `items` without a guard.
   **Severity:** high
   **Confidence:** HIGH
2. [src/bar.ts:10] **Type mismatch**
   Variable `count` is typed `string` but assigned a number.
   Severity: medium
   Confidence: MEDIUM
3. [docs/README.md] File-level note
   No line hint; general doc feedback.

## Cross-Stack Issues
4. [cross-stack] [src/api.py:5] **Contract drift**
   Python payload shape diverges from TS `ApiResponse` definition.
"""


def test_parse_report_extracts_all_sections() -> None:
    issues = parse_report(REPORT_FIXTURE)
    assert len(issues) == 4
    assert issues[0].path == "src/foo.py"
    assert issues[0].line == 42
    assert issues[0].title == "**Null check missing**"
    assert not issues[0].is_cross_stack
    assert issues[0].severity == "high"
    assert issues[0].confidence == "HIGH"

    # Parsing also picks up un-bolded `Severity: medium` form.
    assert issues[1].severity == "medium"
    assert issues[1].confidence == "MEDIUM"

    assert issues[2].path == "docs/README.md"
    assert issues[2].line is None
    assert issues[2].severity is None
    assert issues[2].confidence is None

    assert issues[3].is_cross_stack
    assert issues[3].path == "src/api.py"
    assert issues[3].line == 5


def test_inline_body_has_footer_and_tags() -> None:
    issue = ParsedIssue(
        path="a.py",
        line=10,
        title="Null deref",
        body="rationale here",
        confidence="HIGH",
        severity="high",
    )
    body = pr_review._format_inline_body(issue)
    assert "**Null deref**" in body
    assert "severity: `high`" in body
    assert "confidence: `HIGH`" in body
    assert body.rstrip().endswith("</sub>")
    assert pr_review.DAYDREAM_REPO_URL in body
    # Severity emoji prefix.
    assert "⚠️" in body
    # Collapsible AI agent prompt.
    assert "🔮 Prompt for AI Agents" in body
    assert "<details>" in body


def test_parse_report_handles_no_issues_section() -> None:
    assert parse_report("# Nothing here") == []


BOLD_HEAD_REPORT = """\
# Review

## Issues
1. **[admin-dashboard/middleware.ts:30] Missing audit logging for admin denials**
   When a non-admin is rejected the middleware silently redirects.
   **Severity:** medium
2. __[src/bar.ts:12] Italic-bold wrapper form__
   Parser must accept `__...__` the same as `**...**`.

## Cross-Stack Issues
3. **[cross-stack] [admin-dashboard/lib/auth.ts:1, book-service/admin_auth.go:41, backend/deps.py:48] Inconsistent auth source**
   Frontend reads the JWT claim, backends query the database.
   **Confidence:** HIGH
"""


def test_parse_report_strips_bold_wrapper_from_head() -> None:
    """`** [path] title **` used to drop the whole issue; now it parses cleanly."""
    issues = parse_report(BOLD_HEAD_REPORT)
    # 2 regular + 3 fan-out from cross-stack multi-path
    assert len(issues) == 5
    first = issues[0]
    assert first.path == "admin-dashboard/middleware.ts"
    assert first.line == 30
    # Title no longer carries leading/trailing `**` markers.
    assert first.title == "Missing audit logging for admin denials"
    assert first.severity == "medium"


def test_parse_report_accepts_underscore_bold() -> None:
    issues = parse_report(BOLD_HEAD_REPORT)
    second = issues[1]
    assert second.path == "src/bar.ts"
    assert second.line == 12
    assert second.title == "Italic-bold wrapper form"


def test_parse_report_fans_out_multi_path_cross_stack() -> None:
    issues = parse_report(BOLD_HEAD_REPORT)
    xstack = [i for i in issues if i.is_cross_stack]
    assert [i.path for i in xstack] == [
        "admin-dashboard/lib/auth.ts",
        "book-service/admin_auth.go",
        "backend/deps.py",
    ]
    assert [i.line for i in xstack] == [1, 41, 48]
    # All share title, body, severity/confidence from the single source entry.
    assert len({i.title for i in xstack}) == 1
    assert xstack[0].title == "Inconsistent auth source"
    assert all(i.confidence == "HIGH" for i in xstack)


def test_alt_issues_to_parsed_produces_one_per_file() -> None:
    alt = [
        {
            "id": 1,
            "title": "Extract helper",
            "description": "Duplicated logic",
            "recommendation": "Move to util",
            "severity": "low",
            "files": ["a.py", "b.py"],
            "confidence": "HIGH",
            "rationale": "r",
        }
    ]
    issues = alt_issues_to_parsed(alt)
    assert [i.path for i in issues] == ["a.py", "b.py"]
    assert all(i.line is None for i in issues)
    assert "Recommendation" in issues[0].body
    assert "Severity" in issues[0].body
    assert issues[0].severity == "low"
    assert issues[0].confidence == "HIGH"


def test_alt_issues_to_parsed_skips_no_files() -> None:
    assert alt_issues_to_parsed([{"title": "t", "files": []}]) == []


def test_extract_anchors_prefers_long_tokens() -> None:
    issue = ParsedIssue(
        path="x.py",
        line=None,
        title="Null check",
        body="The function `compute_total` dereferences `items` in handleRequest",
    )
    anchors = extract_anchors(issue)
    # Backtick tokens should appear; longest first.
    assert "compute_total" in anchors
    assert "handleRequest" in anchors
    assert anchors == sorted(anchors, key=len, reverse=True)


def test_parse_hunks() -> None:
    diff = (
        "diff --git a/x.py b/x.py\n"
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1,3 +10,5 @@\n"
        " old\n"
        "+new1\n"
        "+new2\n"
        "@@ -20 +30,2 @@\n"
        "+new3\n"
    )
    assert _parse_hunks(diff) == [(10, 14), (30, 31)]


def test_snap_to_hunk_inside_returns_unchanged() -> None:
    hunks = [(10, 20), (30, 40)]
    assert snap_to_hunk(15, hunks) == 15
    assert snap_to_hunk(10, hunks) == 10
    assert snap_to_hunk(40, hunks) == 40


def test_snap_to_hunk_within_tolerance_snaps_to_boundary() -> None:
    hunks = [(90, 105)]
    # Line 89 is 1 below hunk start -> snap to 90
    assert snap_to_hunk(89, hunks) == 90
    # Line 87 is 3 below hunk start -> snap to 90
    assert snap_to_hunk(87, hunks) == 90
    # Line 108 is 3 above hunk end -> snap to 105
    assert snap_to_hunk(108, hunks) == 105


def test_snap_to_hunk_beyond_tolerance_returns_none() -> None:
    hunks = [(90, 105)]
    assert snap_to_hunk(86, hunks) is None
    assert snap_to_hunk(109, hunks) is None


def test_snap_to_hunk_between_two_hunks() -> None:
    """Line between hunks snaps to the nearest boundary."""
    hunks = [(80, 98), (106, 120)]
    # Line 105 is 7 past first hunk end (too far) but 1 before second start
    assert snap_to_hunk(105, hunks) == 106
    # Line 100 is 2 past first hunk end -> snap to 98
    assert snap_to_hunk(100, hunks) == 98
    # Line 102 is 4 past first hunk end (too far) and 4 before second (too far)
    assert snap_to_hunk(102, hunks) is None


def test_snap_to_hunk_empty_hunks() -> None:
    assert snap_to_hunk(10, []) is None


@pytest.fixture
def pr() -> PRInfo:
    return PRInfo(
        number=42,
        head_sha="head123",
        base_sha="base456",
        base_ref="main",
        owner="acme",
        repo="widgets",
        url="https://github.com/acme/widgets/pull/42",
    )


def test_classify_splits_inline_vs_body(monkeypatch: pytest.MonkeyPatch, pr: PRInfo) -> None:
    issues = [
        ParsedIssue(path="a.py", line=10, title="t1", body="anchor_one"),
        ParsedIssue(path="b.py", line=99, title="t2", body="anchor_two"),
        ParsedIssue(path="c.py", line=None, title="t3", body="xstack", is_cross_stack=True),
    ]

    def fake_resolve(_td: Path, _sha: str, issue: ParsedIssue) -> int | None:
        return issue.line

    def fake_hunks(
        _td: Path,
        _base: str,
        _head: str,
        path: str,
        *,
        pr_number: int | None = None,
    ) -> list[tuple[int, int]]:
        if path == "a.py":
            return [(8, 12)]  # 10 is inside
        if path == "b.py":
            return [(1, 5)]  # 99 is outside
        return []

    monkeypatch.setattr(pr_review, "resolve_line", fake_resolve)
    monkeypatch.setattr(pr_review, "file_hunks", fake_hunks)

    result = classify(Path("."), pr, issues)
    assert len(result.inline) == 1
    assert result.inline[0]["path"] == "a.py"
    assert result.inline[0]["line"] == 10
    assert result.inline[0]["side"] == "RIGHT"
    assert len(result.inline_issues) == 1
    assert result.inline_issues[0].path == "a.py"
    body_paths = [i.path for i in result.body_only]
    assert set(body_paths) == {"b.py", "c.py"}


def test_classify_snaps_tolerance_line_to_hunk_boundary(
    monkeypatch: pytest.MonkeyPatch, pr: PRInfo
) -> None:
    """Line 89 near hunk (90, 105) should become inline at line 90, not 89."""
    issues = [
        ParsedIssue(path="conftest.py", line=89, title="t1", body="anchor_one"),
        ParsedIssue(path="scripts/modernize-app.py", line=105, title="t2", body="anchor_two"),
    ]

    def fake_resolve(_td: Path, _sha: str, issue: ParsedIssue) -> int | None:
        return issue.line

    def fake_hunks(
        _td: Path,
        _base: str,
        _head: str,
        path: str,
        *,
        pr_number: int | None = None,
    ) -> list[tuple[int, int]]:
        if path == "conftest.py":
            return [(90, 105)]  # 89 is 1 below start
        if path == "scripts/modernize-app.py":
            return [(80, 98), (106, 120)]  # 105 is 1 before second hunk
        return []

    monkeypatch.setattr(pr_review, "resolve_line", fake_resolve)
    monkeypatch.setattr(pr_review, "file_hunks", fake_hunks)

    result = classify(Path("."), pr, issues)
    assert len(result.inline) == 2
    # conftest.py:89 snapped to hunk start 90
    assert result.inline[0]["path"] == "conftest.py"
    assert result.inline[0]["line"] == 90
    # modernize-app.py:105 snapped to second hunk start 106
    assert result.inline[1]["path"] == "scripts/modernize-app.py"
    assert result.inline[1]["line"] == 106


def test_build_payload_shape(pr: PRInfo) -> None:
    classified = pr_review._ClassifiedIssues(
        inline=[{"path": "a.py", "line": 10, "side": "RIGHT", "body": "x"}],
        body_only=[
            ParsedIssue(
                path="b.py",
                line=None,
                title="File note",
                body="desc",
                confidence="MEDIUM",
                severity="low",
            )
        ],
        inline_issues=[
            ParsedIssue(
                path="a.py",
                line=10,
                title="t",
                body="b",
                confidence="HIGH",
                severity="high",
            )
        ],
    )
    payload = build_payload(pr, "deep review", classified)
    assert payload["commit_id"] == "head123"
    assert payload["event"] == "COMMENT"
    assert payload["comments"] == classified.inline

    body = payload["body"]
    # Title header.
    assert "Code Review Summary" in body
    # Footer has wizard emoji + daydream repo link.
    assert "🧙" in body
    assert pr_review.DAYDREAM_REPO_URL in body
    # Mode label lives in collapsible review info now.
    assert "deep review" in body
    # Actionable count header.
    assert "Actionable comments posted: 1" in body
    # Counts line.
    assert "1 inline comment(s), 1 non-inline finding(s)" in body
    # Severity/confidence in collapsible review info section.
    assert "**Severity:**" in body and "1 high" in body and "1 low" in body
    assert "**Confidence:**" in body and "1 HIGH" in body and "1 MEDIUM" in body
    # Non-inline section grouped by file in <details>.
    assert "Non-inline findings" in body
    assert "b.py" in body
    # Consolidated AI agent prompt references fetch commands with PR details.
    assert "🔮 Prompt for all review comments" in body
    assert "/beagle-core:fetch-pr-feedback --pr 42" in body
    assert "repos/acme/widgets/pulls/42/comments" in body
    # Review info collapsible.
    assert "ℹ️ Review info" in body
    # Footer.
    assert body.rstrip().endswith("</sub>")


def test_find_open_pr_returns_none_on_empty_list(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path
) -> None:
    """Real git tells us the branch; gh wrapper returns no PRs -> None.

    Stubbed at the git_ops gh wrapper layer (not subprocess) so it works
    without a real GitHub remote or gh auth.
    """
    monkeypatch.setattr(git_ops, "gh_pr_list_for_branch", lambda *_a, **_k: [])
    assert pr_review.find_open_pr(git_repo) is None


def test_find_open_pr_returns_pr_info(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path
) -> None:
    """Real git for branch; gh wrappers stubbed for the PR + repo lookups."""
    rows = [
        {
            "number": 7,
            "headRefOid": "h",
            "baseRefOid": "b",
            "baseRefName": "main",
            "url": "u",
        }
    ]
    monkeypatch.setattr(git_ops, "gh_pr_list_for_branch", lambda *_a, **_k: rows)
    monkeypatch.setattr(git_ops, "gh_repo_view", lambda _r: ("o", "r"))
    info = pr_review.find_open_pr(git_repo)
    assert info is not None
    assert info.number == 7
    assert info.owner == "o"
    assert info.repo == "r"


class _FakeConsole:
    def print(self, *_a: Any, **_k: Any) -> None:
        pass


@pytest.mark.asyncio
async def test_post_skips_when_no_pr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td: None)
    warnings: list[str] = []
    monkeypatch.setattr(
        pr_review,
        "print_warning",
        lambda _c, msg: warnings.append(msg),
    )
    await pr_review._post(
        tmp_path,
        [ParsedIssue(path="x.py", line=1, title="t", body="b")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        mode_label="deep review",
    )
    assert warnings and "No open PR" in warnings[0]


@pytest.mark.asyncio
async def test_post_succeeds_and_prints_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pr: PRInfo
) -> None:
    """On a successful submit the URL is forwarded to print_success."""
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td: pr)
    monkeypatch.setattr(
        pr_review,
        "classify",
        lambda *_a, **_k: pr_review._ClassifiedIssues(
            inline=[{"path": "a.py", "line": 1, "side": "RIGHT", "body": "x"}],
            body_only=[],
        ),
    )
    monkeypatch.setattr(pr_review, "prompt_user", lambda *_a, **_k: "y")

    captured: dict[str, Any] = {}

    def fake_submit(
        _td: Path, _pr: PRInfo, payload: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        captured["payload"] = payload
        return "https://github.com/acme/widgets/pull/42#pullrequestreview-1", None

    monkeypatch.setattr(pr_review, "_submit_review", fake_submit)
    successes: list[str] = []
    monkeypatch.setattr(
        pr_review,
        "print_success",
        lambda _c, msg: successes.append(msg),
    )
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)

    await pr_review._post(
        tmp_path,
        [ParsedIssue(path="a.py", line=1, title="t", body="b")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        mode_label="deep review",
    )
    # The payload that would be POSTed was assembled and forwarded.
    assert captured["payload"]["commit_id"] == pr.head_sha
    assert captured["payload"]["event"] == "COMMENT"
    assert successes and "pullrequestreview" in successes[0]


@pytest.mark.asyncio
async def test_post_warns_with_preserved_payload_path_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pr: PRInfo
) -> None:
    """When submit returns an error, the warning surfaces git_ops's preserved-path text."""
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td: pr)
    monkeypatch.setattr(
        pr_review,
        "classify",
        lambda *_a, **_k: pr_review._ClassifiedIssues(
            inline=[{"path": "a.py", "line": 1, "side": "RIGHT", "body": "x"}],
            body_only=[],
        ),
    )
    monkeypatch.setattr(pr_review, "prompt_user", lambda *_a, **_k: "y")
    err = "gh api /repos/acme/widgets/pulls/42/reviews failed: HTTP 422 (payload preserved at /tmp/x.json)"
    monkeypatch.setattr(
        pr_review, "_submit_review", lambda *_a, **_k: (None, err)
    )
    warnings: list[str] = []
    monkeypatch.setattr(
        pr_review,
        "print_warning",
        lambda _c, msg: warnings.append(msg),
    )
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)

    await pr_review._post(
        tmp_path,
        [ParsedIssue(path="a.py", line=1, title="t", body="b")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        mode_label="deep review",
    )
    assert warnings
    assert "no comments were posted" in warnings[0].lower()
    # The git_ops error text -- including the preserved payload path -- is forwarded.
    assert "payload preserved at /tmp/x.json" in warnings[0]


@pytest.mark.asyncio
async def test_post_skipped_when_user_declines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pr: PRInfo
) -> None:
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td: pr)
    monkeypatch.setattr(
        pr_review,
        "classify",
        lambda *_a, **_k: pr_review._ClassifiedIssues(
            inline=[{"path": "a.py", "line": 1, "side": "RIGHT", "body": "x"}],
            body_only=[],
        ),
    )
    monkeypatch.setattr(pr_review, "prompt_user", lambda *_a, **_k: "n")
    submit_called = False

    def fake_submit(*_a: Any, **_k: Any) -> tuple[str | None, str | None]:
        nonlocal submit_called
        submit_called = True
        return "x", None

    monkeypatch.setattr(pr_review, "_submit_review", fake_submit)
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)

    await pr_review._post(
        tmp_path,
        [ParsedIssue(path="a.py", line=1, title="t", body="b")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        mode_label="deep review",
    )
    assert not submit_called


def _commit_file(repo: Path, path: str, contents: str, message: str) -> str:
    """Write *path* under *repo*, commit it, and return the new HEAD SHA."""
    file_path = repo / path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(contents)
    _git(repo, "add", path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def test_resolve_line_verifies_hint(git_repo: Path) -> None:
    """Real-git path: show pulls the file at HEAD and the anchor verifies the hint."""
    text = "\n".join(f"line_{i} extra" for i in range(1, 21)) + "\n"
    sha = _commit_file(git_repo, "x.py", text, "add x.py")
    issue = ParsedIssue(path="x.py", line=10, title="t", body="`line_10`")
    assert pr_review.resolve_line(git_repo, sha, issue) == 10


def test_resolve_line_full_search_when_hint_bad(git_repo: Path) -> None:
    """Hint points to line 2, but the anchor is at line 15 -- full-file search wins."""
    text = "\n".join(f"row_{i}" for i in range(1, 21)) + "\n"
    sha = _commit_file(git_repo, "x.py", text, "add x.py")
    issue = ParsedIssue(path="x.py", line=2, title="t", body="`row_15`")
    assert pr_review.resolve_line(git_repo, sha, issue) == 15


def test_resolve_line_none_when_missing_file(git_repo: Path) -> None:
    """git show fails for a path that doesn't exist at HEAD -> None."""
    sha = _git(git_repo, "rev-parse", "HEAD")
    issue = ParsedIssue(path="gone.py", line=1, title="t", body="b")
    assert pr_review.resolve_line(git_repo, sha, issue) is None


# --- file_hunks git-diff + gh-pr-diff fallback ----------------------------


_GH_PR_DIFF = (
    "diff --git a/x.py b/x.py\n"
    "--- a/x.py\n"
    "+++ b/x.py\n"
    "@@ -1,3 +10,5 @@\n"
    " old\n"
    "+new1\n"
    "+new2\n"
    "diff --git a/other.py b/other.py\n"
    "--- a/other.py\n"
    "+++ b/other.py\n"
    "@@ -1 +1,2 @@\n"
    "+noise\n"
)


def test_file_hunks_uses_git_diff_when_it_succeeds(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path
) -> None:
    """Happy path: real git diff yields hunks; gh fallback is not consulted."""
    # Build base, then add 5 lines on a feature branch starting at line N.
    _commit_file(
        git_repo, "x.py", "\n".join(f"line {i}" for i in range(1, 30)) + "\n", "baseline"
    )
    base = _git(git_repo, "rev-parse", "HEAD")
    lines = [f"line {i}" for i in range(1, 30)]
    # Insert two new lines after position 20 to create a clear hunk.
    lines[19:19] = ["NEW1", "NEW2"]
    (git_repo / "x.py").write_text("\n".join(lines) + "\n")
    _git(git_repo, "add", "x.py")
    _git(git_repo, "commit", "-m", "add 2 lines")
    head = _git(git_repo, "rev-parse", "HEAD")

    # If the gh fallback fires we want to know about it.
    gh_called = False

    def boom(*_a: Any, **_k: Any) -> str:
        nonlocal gh_called
        gh_called = True
        return ""

    monkeypatch.setattr(git_ops, "gh_pr_diff", boom)

    hunks = pr_review.file_hunks(git_repo, base, head, "x.py", pr_number=42)
    assert hunks  # at least one hunk
    # The fallback was not needed because real git diff succeeded.
    assert gh_called is False


def test_file_hunks_falls_back_to_gh_when_base_unreachable(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path
) -> None:
    """When git diff fails (base_sha unreachable), gh pr diff rescues the hunks.

    Uses real git (which raises GitError on the bogus base SHA) and stubs the
    git_ops gh wrapper (no remote/auth required).
    """
    monkeypatch.setattr(
        git_ops, "gh_pr_diff", lambda _r, _n: _GH_PR_DIFF
    )
    hunks = pr_review.file_hunks(
        git_repo, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "HEAD", "x.py", pr_number=42
    )
    # Must come from the x.py block only -- the other.py hunk starts at line 1
    # and must NOT leak into x.py's result.
    assert hunks == [(10, 14)]


def test_file_hunks_no_fallback_without_pr_number(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path
) -> None:
    """Without a pr_number, file_hunks returns empty instead of calling gh.

    Uses real git (which fails on the bogus base) plus a guard on the gh
    wrapper to confirm no fallback is invoked.
    """
    gh_called = False

    def boom(*_a: Any, **_k: Any) -> str:
        nonlocal gh_called
        gh_called = True
        return ""

    monkeypatch.setattr(git_ops, "gh_pr_diff", boom)
    hunks = pr_review.file_hunks(
        git_repo, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "HEAD", "x.py"
    )
    assert hunks == []
    assert gh_called is False


def test_file_hunks_gh_fallback_handles_subprocess_error(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path
) -> None:
    """If gh itself errors out, file_hunks returns empty without raising.

    Real git is allowed to fail, then the stubbed gh wrapper raises GitError
    -- pr_review must swallow it and return [].
    """

    def raise_git_error(*_a: Any, **_k: Any) -> str:
        raise git_ops.GitError("gh blew up")

    monkeypatch.setattr(git_ops, "gh_pr_diff", raise_git_error)
    hunks = pr_review.file_hunks(
        git_repo, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "HEAD", "x.py", pr_number=42
    )
    assert hunks == []
