"""Unit tests for daydream.pr_review."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from daydream import pr_review
from daydream.pr_review import (
    ParsedIssue,
    PRInfo,
    _parse_hunks,
    alt_issues_to_parsed,
    build_payload,
    classify,
    extract_anchors,
    parse_report,
    within_hunk,
)

REPORT_FIXTURE = """\
# Review

## Per-Stack Context

Some prose.

## Issues
1. [src/foo.py:42] **Null check missing**
   The function `compute_total` dereferences `items` without a guard.
2. [src/bar.ts:10] **Type mismatch**
   Variable `count` is typed `string` but assigned a number.
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

    assert issues[2].path == "docs/README.md"
    assert issues[2].line is None
    assert not issues[2].is_cross_stack

    assert issues[3].is_cross_stack
    assert issues[3].path == "src/api.py"
    assert issues[3].line == 5


def test_parse_report_handles_no_issues_section() -> None:
    assert parse_report("# Nothing here") == []


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


def test_within_hunk_tolerance() -> None:
    hunks = [(10, 14)]
    assert within_hunk(13, hunks)
    assert within_hunk(16, hunks, tolerance=3)  # 14 + 2
    assert not within_hunk(20, hunks, tolerance=3)


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

    def fake_hunks(_td: Path, _base: str, _head: str, path: str) -> list[tuple[int, int]]:
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
    body_paths = [i.path for i in result.body_only]
    assert set(body_paths) == {"b.py", "c.py"}


def test_build_payload_shape(pr: PRInfo) -> None:
    classified = pr_review._ClassifiedIssues(
        inline=[{"path": "a.py", "line": 10, "side": "RIGHT", "body": "x"}],
        body_only=[
            ParsedIssue(path="b.py", line=None, title="File note", body="desc")
        ],
    )
    payload = build_payload(pr, "Test summary.", classified)
    assert payload["commit_id"] == "head123"
    assert payload["event"] == "COMMENT"
    assert payload["comments"] == classified.inline
    assert "Test summary." in payload["body"]
    assert "Non-inline findings" in payload["body"]
    assert "1 inline comment(s), 1 non-inline finding(s)" in payload["body"]


def test_find_open_pr_returns_none_on_empty_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(cmd: list[str], *_a: Any, **_k: Any) -> subprocess.CompletedProcess[bytes]:
        if cmd[:3] == ["git", "branch", "--show-current"]:
            return subprocess.CompletedProcess(cmd, 0, b"feat/x\n", b"")
        if cmd[:3] == ["gh", "pr", "list"]:
            return subprocess.CompletedProcess(cmd, 0, b"[]", b"")
        raise AssertionError(f"unexpected {cmd}")

    monkeypatch.setattr(pr_review.subprocess, "run", fake_run)
    assert pr_review.find_open_pr(tmp_path) is None


def test_find_open_pr_returns_pr_info(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = [
        {
            "number": 7,
            "headRefOid": "h",
            "baseRefOid": "b",
            "baseRefName": "main",
            "url": "u",
        }
    ]
    repo = {"owner": {"login": "o"}, "name": "r"}

    def fake_run(cmd: list[str], *_a: Any, **_k: Any) -> subprocess.CompletedProcess[bytes]:
        if cmd[:3] == ["git", "branch", "--show-current"]:
            return subprocess.CompletedProcess(cmd, 0, b"f\n", b"")
        if cmd[:3] == ["gh", "pr", "list"]:
            return subprocess.CompletedProcess(cmd, 0, json.dumps(rows).encode(), b"")
        if cmd[:3] == ["gh", "repo", "view"]:
            return subprocess.CompletedProcess(cmd, 0, json.dumps(repo).encode(), b"")
        raise AssertionError(f"unexpected {cmd}")

    monkeypatch.setattr(pr_review.subprocess, "run", fake_run)
    info = pr_review.find_open_pr(tmp_path)
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
        summary_prefix="s",
    )
    assert warnings and "No open PR" in warnings[0]


@pytest.mark.asyncio
async def test_post_cleans_tmp_file_on_success(
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
    monkeypatch.setattr(pr_review, "prompt_user", lambda *_a, **_k: "y")

    captured: dict[str, Path] = {}

    def fake_submit(_td: Path, _pr: PRInfo, payload_path: Path) -> str | None:
        captured["path"] = payload_path
        assert payload_path.exists()
        return "https://github.com/acme/widgets/pull/42#pullrequestreview-1"

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
        summary_prefix="s",
    )
    # Tmp file was deleted after success.
    assert not captured["path"].exists()
    assert successes and "pullrequestreview" in successes[0]


@pytest.mark.asyncio
async def test_post_warns_and_keeps_file_on_failure(
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
    monkeypatch.setattr(pr_review, "prompt_user", lambda *_a, **_k: "y")
    monkeypatch.setattr(pr_review, "_submit_review", lambda *_a, **_k: None)
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
        summary_prefix="s",
    )
    assert warnings
    assert "no comments were posted" in warnings[0].lower()
    # Tmp path mentioned in warning still exists.
    # Extract path from warning message.
    import re as _re

    match = _re.search(r"(/\S+\.json)", warnings[0])
    assert match
    assert Path(match.group(1)).exists()


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

    def fake_submit(*_a: Any, **_k: Any) -> str | None:
        nonlocal submit_called
        submit_called = True
        return "x"

    monkeypatch.setattr(pr_review, "_submit_review", fake_submit)
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)

    await pr_review._post(
        tmp_path,
        [ParsedIssue(path="a.py", line=1, title="t", body="b")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        summary_prefix="s",
    )
    assert not submit_called


def test_resolve_line_verifies_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    file_text = "\n".join(f"line_{i} extra" for i in range(1, 21)).encode()

    def fake_run(cmd: list[str], *_a: Any, **_k: Any) -> subprocess.CompletedProcess[bytes]:
        assert cmd[:2] == ["git", "show"]
        return subprocess.CompletedProcess(cmd, 0, file_text, b"")

    monkeypatch.setattr(pr_review.subprocess, "run", fake_run)
    issue = ParsedIssue(path="x.py", line=10, title="t", body="`line_10`")
    assert pr_review.resolve_line(tmp_path, "head", issue) == 10


def test_resolve_line_full_search_when_hint_bad(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    file_text = "\n".join(f"row_{i}" for i in range(1, 21)).encode()

    def fake_run(cmd: list[str], *_a: Any, **_k: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(cmd, 0, file_text, b"")

    monkeypatch.setattr(pr_review.subprocess, "run", fake_run)
    # Hint at line 2, but anchor is at line 15.
    issue = ParsedIssue(path="x.py", line=2, title="t", body="`row_15`")
    assert pr_review.resolve_line(tmp_path, "head", issue) == 15


def test_resolve_line_none_when_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(cmd: list[str], *_a: Any, **_k: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(cmd, 128, b"", b"fatal")

    monkeypatch.setattr(pr_review.subprocess, "run", fake_run)
    issue = ParsedIssue(path="gone.py", line=1, title="t", body="b")
    assert pr_review.resolve_line(tmp_path, "head", issue) is None
