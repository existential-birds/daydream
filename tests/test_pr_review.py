"""Unit tests for daydream.pr_review."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops, pr_comment_renderer, pr_review
from daydream.findings import ArtifactFinding
from daydream.git_ops import GitError
from daydream.pr_review import (
    DAYDREAM_FOOTER,
    ParsedIssue,
    PRInfo,
    ReviewRenderers,
    _format_body_section,
    _format_file_level_body,
    _format_inline_body,
    _parse_hunks,
    alt_issues_to_parsed,
    build_payload,
    classify,
    default_render_finding,
    default_render_summary,
    extract_anchors,
    parse_finding_markers,
    parsed_issues_from_items,
    resolve_review_renderers,
    snap_to_hunk,
)
from daydream.run_context import InteractionPolicy, RunContext
from tests.harness.git_helpers import git as _git

# gh-gated: tests that stub gh's subprocess are skipped when gh is not installed.
_gh_available = shutil.which("gh") is not None
gh_required = pytest.mark.skipif(not _gh_available, reason="gh CLI not installed")

SNAP = Path(__file__).parent / "fixtures" / "comment_snapshots"
BUILTIN_RENDERERS = ReviewRenderers(default_render_finding, default_render_summary)


def test_finding_and_summary_markdown_is_byte_stable() -> None:
    i = ParsedIssue(
        path="a.py", line=3, title="T", body="B rationale", severity="high", confidence="HIGH", fingerprint="a" * 64
    )
    assert _format_inline_body(i, renderers=BUILTIN_RENDERERS) == (SNAP / "inline.md").read_text()
    assert (
        _format_file_level_body(replace(i, is_cross_stack=True), renderers=BUILTIN_RENDERERS)
        == (SNAP / "file_level.md").read_text()
    )
    section = _format_body_section(
        [replace(i, line=None), replace(i, path="b.py", line=None, fingerprint="b" * 64)], renderers=BUILTIN_RENDERERS
    )
    assert section == (SNAP / "summary_body.md").read_text()


def test_custom_finding_renderer_flows_into_inline_body_with_host_invariants() -> None:
    from daydream.extensions import Registry
    from daydream.extensions.builtins import register_builtins

    reg = Registry()
    register_builtins(reg)
    reg.override_renderer("finding", lambda finding, ctx: f"CUSTOM::{ctx.placement}::{finding.title}")
    body = _format_inline_body(
        ParsedIssue(path="a.py", line=3, title="T", body="B", fingerprint="a" * 64),
        renderers=resolve_review_renderers(reg),
    )
    assert "CUSTOM::inline::T" in body
    assert DAYDREAM_FOOTER in body
    assert parse_finding_markers(body) == ["a" * 64]


def test_finding_renderer_falls_back_and_warns_on_error(caplog: pytest.LogCaptureFixture) -> None:
    from daydream.extensions import Registry
    from daydream.extensions.builtins import register_builtins

    def boom(finding: Any, ctx: Any) -> str:
        raise RuntimeError("boom")

    reg = Registry()
    register_builtins(reg)
    reg.override_renderer("finding", boom)
    with caplog.at_level("WARNING"):
        body = _format_inline_body(
            ParsedIssue(
                path="a.py",
                line=3,
                title="T",
                body="B rationale",
                severity="high",
                confidence="HIGH",
                fingerprint="a" * 64,
            ),
            renderers=resolve_review_renderers(reg),
        )
    assert body == (SNAP / "inline.md").read_text()
    assert "finding" in caplog.text and "boom" in caplog.text


_FIXTURE = Path(__file__).parent / "fixtures" / "trajectories" / "single_phase_claude.json"


def test_custom_summary_renderer_can_build_collapsible_per_finding_list(
    pr: PRInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daydream.extensions import Registry
    from daydream.extensions.builtins import register_builtins

    def summary_renderer(ctx: Any) -> Any:
        rows = [
            f"<details><summary>{f.finding.path} — {f.finding.title}</summary>\n{f.body_block}\n</details>"
            for f in ctx.findings
        ]
        return "**Custom Summary**\n\n" + "\n".join(rows)

    reg = Registry()
    register_builtins(reg)
    reg.override_renderer("summary", summary_renderer)
    classified = pr_review._ClassifiedIssues(
        body_only=[ParsedIssue(path="b.py", line=None, title="File note", body="desc", fingerprint="b" * 64)]
    )
    payload = build_payload(
        pr,
        classified,
        renderers=resolve_review_renderers(reg),
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    body = payload["body"]
    assert "**Custom Summary**" in body
    assert "<summary>b.py — File note</summary>" in body
    assert parse_finding_markers(body) == ["b" * 64]
    assert body.rstrip().endswith("</sub>")


def test_custom_finding_renderer_flows_into_summary_section(pr: PRInfo, monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.extensions import Registry
    from daydream.extensions.builtins import register_builtins

    reg = Registry()
    register_builtins(reg)
    reg.override_renderer("finding", lambda finding, ctx: f"CUSTOM::{ctx.placement}::{finding.title}")
    classified = pr_review._ClassifiedIssues(
        body_only=[ParsedIssue(path="b.py", line=None, title="File note", body="desc", fingerprint="b" * 64)]
    )
    body = build_payload(
        pr,
        classified,
        renderers=resolve_review_renderers(reg),
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )["body"]
    assert "CUSTOM::summary::File note" in body
    assert parse_finding_markers(body) == ["b" * 64]


def test_summary_renderer_falls_back_and_warns_on_error(
    pr: PRInfo, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from daydream.extensions import Registry
    from daydream.extensions.builtins import register_builtins

    def boom(ctx: Any) -> str:
        raise RuntimeError("kaboom")

    reg = Registry()
    register_builtins(reg)
    reg.override_renderer("summary", boom)
    classified = pr_review._ClassifiedIssues(
        body_only=[
            ParsedIssue(
                path="b.py",
                line=None,
                title="File note",
                body="desc",
                confidence="MEDIUM",
                severity="low",
                fingerprint="b" * 64,
            )
        ]
    )
    default_body = build_payload(
        pr, classified, renderers=BUILTIN_RENDERERS, run_info=pr_comment_renderer.render_run_info_block([_FIXTURE])
    )["body"]
    with caplog.at_level("WARNING"):
        body = build_payload(
            pr,
            classified,
            renderers=resolve_review_renderers(reg),
            run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
        )["body"]
    assert body == default_body
    assert "summary" in caplog.text and "kaboom" in caplog.text


def test_structural_item_becomes_parsed_issue() -> None:
    items = [{"id": 1, "lens": "structural", "file": "big.py", "line": 1,
              "description": "1k-line file", "severity": "high",
              "confidence": "HIGH", "rationale": "r"}]
    issues = parsed_issues_from_items(items)
    assert [(i.path, i.line) for i in issues] == [("big.py", 1)]   # structural posts


def test_inline_body_has_footer_and_tags() -> None:
    issue = ParsedIssue(
        path="a.py",
        line=10,
        title="Null deref",
        body="rationale here",
        confidence="HIGH",
        severity="high",
    )
    body = pr_review._format_inline_body(issue, renderers=BUILTIN_RENDERERS)
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


def test_inline_body_carries_parseable_marker() -> None:
    issue = ParsedIssue(path="a.py", line=3, title="T", body="B", fingerprint="ab12" * 16)
    body = _format_inline_body(issue, renderers=BUILTIN_RENDERERS)
    assert parse_finding_markers(body) == ["ab12" * 16]
    assert DAYDREAM_FOOTER in body  # marker does not displace the footer


def test_no_marker_without_fingerprint() -> None:
    assert (
        parse_finding_markers(
            _format_inline_body(ParsedIssue(path="a.py", line=3, title="T", body="B"), renderers=BUILTIN_RENDERERS)
        )
        == []
    )


def test_body_section_markers_one_per_fingerprinted_issue() -> None:
    issues = [ParsedIssue(path="a.py", line=None, title=f"T{i}", body="B", fingerprint=f"{i:064x}") for i in range(2)]
    assert parse_finding_markers(_format_body_section(issues, renderers=BUILTIN_RENDERERS)) == [
        f"{i:064x}" for i in range(2)
    ]


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
    anchors = extract_anchors(
        "Null check\nThe function `compute_total` dereferences `items` in handleRequest"
    )
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


def test_parse_hunks_uses_shared_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    import daydream.hunk_index as hunk_index

    calls = {"n": 0}
    real = hunk_index.parse_hunks

    def spy(diff_text: Any) -> Any:
        calls["n"] += 1
        return real(diff_text)

    monkeypatch.setattr(hunk_index, "parse_hunks", spy)
    diff = (
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
        "@@ -1,3 +10,5 @@\n old\n+new1\n+new2\n@@ -20 +30,2 @@\n+new3\n"
    )
    assert _parse_hunks(diff) == [(10, 14), (30, 31)]
    assert calls["n"] == 1, "_parse_hunks must delegate to the shared parser"


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
        head_ref="feature",
        owner="acme",
        repo="widgets",
        url="https://github.com/acme/widgets/pull/42",
    )


def test_agent_prompt_has_no_skill_advertising(pr: PRInfo) -> None:
    """M5: the consolidated AI-agent prompt carries no /beagle-core skill reference."""
    body = pr_review._build_consolidated_prompt(pr_review._ClassifiedIssues(), pr)
    assert "/beagle-core:fetch-pr-feedback" not in body
    assert "/beagle-core:" not in body


def test_classify_splits_inline_vs_body(monkeypatch: pytest.MonkeyPatch, pr: PRInfo) -> None:
    issues = [
        ParsedIssue(path="a.py", line=10, title="t1", body="anchor_one"),
        ParsedIssue(path="b.py", line=99, title="t2", body="anchor_two"),
        ParsedIssue(path="c.py", line=None, title="t3", body="xstack", is_cross_stack=True),
    ]

    def fake_resolve(
        _td: Path, _sha: str, issue: ParsedIssue, hunks: list[tuple[int, int]] | None = None
    ) -> int | None:
        # classify must resolve the hunks first and hand them down (issue #1102).
        assert hunks is not None, "classify called resolve_line without the file's hunks"
        return issue.line

    def fake_hunks(
        _td: Path,
        _base: str,
        _head: str,
        path: str,
        *,
        pr_number: int | None = None,
        **_kwargs: Any,
    ) -> list[tuple[int, int]]:
        if path == "a.py":
            return [(8, 12)]  # 10 is inside
        if path == "b.py":
            return [(1, 5)]  # 99 is outside
        return []

    monkeypatch.setattr(git_ops, "show", lambda *_a, **_k: b"")
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

    def fake_resolve(
        _td: Path, _sha: str, issue: ParsedIssue, hunks: list[tuple[int, int]] | None = None
    ) -> int | None:
        assert hunks is not None, "classify called resolve_line without the file's hunks"
        return issue.line

    def fake_hunks(
        _td: Path,
        _base: str,
        _head: str,
        path: str,
        *,
        pr_number: int | None = None,
        **_kwargs: Any,
    ) -> list[tuple[int, int]]:
        if path == "conftest.py":
            return [(90, 105)]  # 89 is 1 below start
        if path == "scripts/modernize-app.py":
            return [(80, 98), (106, 120)]  # 105 is 1 before second hunk
        return []

    monkeypatch.setattr(git_ops, "show", lambda *_a, **_k: b"")
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


def test_build_payload_reviewed_commit_line_first_in_review_info(pr: PRInfo, monkeypatch: pytest.MonkeyPatch) -> None:
    """M1/M4: the reviewed-commit line is rendered, fully linked (S1),
    and precedes Model/Cost and Severity/Confidence in the review-info block."""
    classified = pr_review._ClassifiedIssues(
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
    )
    # Feed the enriched renderer a real fixture trajectory so run-info
    # fields (Model/Cost/Tokens) render instead of the fallback stub.
    payload = build_payload(
        pr, classified, renderers=BUILTIN_RENDERERS, run_info=pr_comment_renderer.render_run_info_block([_FIXTURE])
    )
    body = payload["body"]

    expected = (
        f"- **Reviewed commit:** [`{pr.head_sha[:7]}`](https://github.com/{pr.owner}/{pr.repo}/commit/{pr.head_sha})"
    )
    assert expected in body
    # Ordering: the commit line precedes the enriched run-info fields
    # (Model/Cost) and the conditional Severity/Confidence lines inside
    # the Review info block. Whole-body indices suffice: the commit line
    # appears exactly once, and all compared markers first appear inside
    # this block.
    assert body.index(expected) < body.index("- **Model:**")
    assert body.index(expected) < body.index("- **Severity:**")
    assert body.index(expected) < body.index("- **Confidence:**")


def test_build_payload_reviewed_commit_survives_run_info_fallback(pr: PRInfo, monkeypatch: pytest.MonkeyPatch) -> None:
    """M2: renderer degradation to 'run details unavailable' must not hide
    the reviewed-commit line (host-injection, Key Decision 2)."""
    payload = build_payload(
        pr, pr_review._ClassifiedIssues(), renderers=BUILTIN_RENDERERS, run_info=pr_comment_renderer._render_fallback()
    )
    body = payload["body"]
    assert "*run details unavailable*" in body  # degraded run-info present
    assert (
        f"- **Reviewed commit:** [`{pr.head_sha[:7]}`]"
        f"(https://github.com/{pr.owner}/{pr.repo}/commit/{pr.head_sha})" in body
    )


def test_build_payload_reviewed_commit_links_fork_for_fork_head_pr(
    pr: PRInfo,
) -> None:
    """Fork-head PRs: the reviewed-commit link points at the fork that holds
    the head commit, while owner/repo (the POST target) stay the base repo."""
    fork_pr = replace(pr, head_repo="forky/widgets")
    payload = build_payload(
        fork_pr, pr_review._ClassifiedIssues(), run_info="test run info", renderers=BUILTIN_RENDERERS
    )
    body = payload["body"]
    assert "- **Reviewed commit:** [`head123`](https://github.com/forky/widgets/commit/head123)" in body
    assert "https://github.com/acme/widgets/commit/head123" not in body


def test_build_payload_blocks_forged_reviewed_commit_line(
    pr: PRInfo,
) -> None:
    """A hostile run_info containing a crafted reviewed-commit line
    must not render a second, forged line below the trusted one (issue 2)."""
    payload = build_payload(
        pr,
        pr_review._ClassifiedIssues(),
        run_info=(
            "test run info\n"
            "- **Reviewed commit:** [`deadbee`](https://github.com/evil/widgets/commit/" + "e" * 40 + ")\n"
            "*run details unavailable*"
        ),
        renderers=BUILTIN_RENDERERS,
    )
    body = payload["body"]
    commit_lines = [line for line in body.splitlines() if line.startswith("- **Reviewed commit:**")]
    assert len(commit_lines) == 1
    assert "- **Reviewed commit:** [`head123`](https://github.com/acme/widgets/commit/head123)" in body
    assert "evil/widgets" not in body
    assert "e" * 40 not in body


def test_build_payload_shape(pr: PRInfo, monkeypatch: pytest.MonkeyPatch) -> None:
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

    payload = build_payload(
        pr, classified, renderers=BUILTIN_RENDERERS, run_info=pr_comment_renderer.render_run_info_block([_FIXTURE])
    )
    assert payload["commit_id"] == "head123"
    assert payload["event"] == "COMMENT"
    assert payload["comments"] == classified.inline

    body = payload["body"]
    assert "- **Reviewed commit:** [`head123`](https://github.com/acme/widgets/commit/head123)" in body
    # Title header.
    assert "**Code Review Summary**" in body
    # Bottom-of-comment wizard footer (DAYDREAM_FOOTER) carries the version.
    assert "🧙 Posted by [daydream v" in body
    assert pr_review.DAYDREAM_REPO_URL in body
    # Mode line is gone everywhere.
    assert "**Mode:**" not in body
    # Severity/confidence still surface inside the collapsible block.
    assert "**Severity:**" in body and "1 high" in body and "1 low" in body
    assert "**Confidence:**" in body and "1 HIGH" in body and "1 MEDIUM" in body
    # Non-inline section grouped by file in <details>.
    assert "Non-inline findings" in body
    assert "b.py" in body
    # Consolidated AI agent prompt references manual fetch commands with PR details.
    assert "🔮 Prompt for all review comments" in body
    assert "/beagle-core:" not in body
    assert "repos/acme/widgets/pulls/42/comments" in body
    # Review info collapsible.
    assert "ℹ️ Review info" in body
    # Renderer fields (M1, M2): rollup labels and per-phase table shell.
    assert "- **Model:**" in body
    assert "- **Cost:**" in body
    assert "- **Tokens:**" in body
    assert "- **Steps / tool calls:**" in body
    assert "<details><summary>Per-phase breakdown</summary>" in body
    assert "| Phase | Model | Tools | Input (cached) | Output | Cost |" in body
    # Renderer-owned version footer appears once, inside the review-info shell.
    assert body.count("Generated by daydream v") == 1
    # Footer is the last block.
    assert body.rstrip().endswith("</sub>")


def test_build_payload_approves_when_clean_and_enabled(pr: PRInfo, monkeypatch: pytest.MonkeyPatch) -> None:
    """approve_on_clean=True + zero high/medium findings -> event APPROVE."""
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
                confidence="LOW",
                severity="low",
            )
        ],
    )

    payload = build_payload(
        pr,
        classified,
        approve_on_clean=True,
        renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert payload["event"] == "APPROVE"
    assert "no high/medium findings" in payload["body"]
    # F3: the reviewed SHA is pinned on every payload, APPROVE included.
    assert payload["commit_id"] == pr.head_sha
    # Approval prefix added; rest of body format intact.
    assert "**Code Review Summary**" in payload["body"]
    # The approval line is first, before the summary header.
    assert payload["body"].index("no high/medium findings") < payload["body"].index("**Code Review Summary**")


def test_build_payload_keeps_comment_when_high_finding(pr: PRInfo, monkeypatch: pytest.MonkeyPatch) -> None:
    """approve_on_clean=True but a high-severity finding -> event COMMENT."""
    classified = pr_review._ClassifiedIssues(
        inline=[{"path": "a.py", "line": 10, "side": "RIGHT", "body": "x"}],
        body_only=[
            ParsedIssue(
                path="b.py",
                line=None,
                title="File note",
                body="desc",
                confidence="HIGH",
                severity="high",
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

    payload = build_payload(
        pr,
        classified,
        approve_on_clean=True,
        renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert payload["event"] == "COMMENT"
    assert "no high/medium findings" not in payload["body"]


def test_build_payload_keeps_comment_when_medium_finding(pr: PRInfo, monkeypatch: pytest.MonkeyPatch) -> None:
    """approve_on_clean=True but a medium-severity finding -> event COMMENT."""
    classified = pr_review._ClassifiedIssues(
        inline=[{"path": "a.py", "line": 10, "side": "RIGHT", "body": "x"}],
        body_only=[
            ParsedIssue(
                path="b.py",
                line=None,
                title="File note",
                body="desc",
                confidence="MEDIUM",
                severity="medium",
            )
        ],
        inline_issues=[
            ParsedIssue(
                path="a.py",
                line=10,
                title="t",
                body="b",
                confidence="MEDIUM",
                severity="medium",
            )
        ],
    )

    payload = build_payload(
        pr,
        classified,
        approve_on_clean=True,
        renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert payload["event"] == "COMMENT"
    assert "no high/medium findings" not in payload["body"]


def test_build_payload_none_severity_does_not_crash_on_approve_check(
    pr: PRInfo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A None-severity issue must not crash the clean computation."""
    classified = pr_review._ClassifiedIssues(
        inline=[{"path": "a.py", "line": 10, "side": "RIGHT", "body": "x"}],
        inline_issues=[ParsedIssue(path="a.py", line=10, title="t", body="b", severity=None)],
    )

    payload = build_payload(
        pr,
        classified,
        approve_on_clean=True,
        renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert payload["event"] == "APPROVE"


@pytest.mark.parametrize("off_vocabulary_severity", ["critical", "blocker"])
def test_build_payload_keeps_comment_when_off_vocabulary_severity(
    pr: PRInfo,
    monkeypatch: pytest.MonkeyPatch,
    off_vocabulary_severity: str,
) -> None:
    """F1: approve_on_clean=True but an off-vocabulary severity -> event COMMENT.

    The findings schema permits any string, so 'critical'/'blocker' must block
    the approval (fail-closed) just like 'high'/'medium'.
    """
    classified = pr_review._ClassifiedIssues(
        inline=[{"path": "a.py", "line": 10, "side": "RIGHT", "body": "x"}],
        inline_issues=[
            ParsedIssue(
                path="a.py",
                line=10,
                title="t",
                body="b",
                confidence="HIGH",
                severity=off_vocabulary_severity,
            )
        ],
    )

    payload = build_payload(
        pr,
        classified,
        approve_on_clean=True,
        renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert payload["event"] == "COMMENT"
    assert "no high/medium findings" not in payload["body"]


def test_non_blocking_severities_fail_closed() -> None:
    """F1: any severity outside _NON_BLOCKING_SEVERITIES blocks; low and None do not."""
    assert pr_review._NON_BLOCKING_SEVERITIES == frozenset({"low"})
    for off_vocabulary in (
        "high",
        "medium",
        "critical",
        "blocker",
        "major",
        "warning",
        "info",
        "INFO",
        " High ",
    ):
        assert pr_review._severity_blocks_approval(off_vocabulary) is True
    assert pr_review._severity_blocks_approval("low") is False
    assert pr_review._severity_blocks_approval("LOW") is False
    assert pr_review._severity_blocks_approval(None) is False


def test_find_open_pr_returns_none_on_empty_list(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path
) -> None:
    """Real git tells us the branch; gh wrapper returns no PRs -> None.

    Stubbed at the git_ops gh wrapper layer (not subprocess) so it works
    without a real GitHub remote or gh auth.
    """
    monkeypatch.setattr(git_ops, "gh_pr_list_for_branch", lambda *_a, **_k: [])
    assert pr_review.find_open_pr(git_repo) is None


def _local_pr_row(repo: Path, *, head_owner: str = "o") -> tuple[dict[str, Any], str, str]:
    base = git_ops.head_sha(repo)
    _git(repo, "checkout", "-b", "feature")
    (repo / "feature.py").write_text("value = 1\n")
    _git(repo, "add", "feature.py")
    _git(repo, "commit", "-m", "feature")
    head = git_ops.head_sha(repo)
    return (
        {
            "number": 7,
            "headRefOid": head,
            "headRefName": "feature",
            "baseRefName": "main",
            "url": "https://github.com/o/r/pull/7",
            "headRepository": {"name": "r", "nameWithOwner": f"{head_owner}/r"},
            "headRepositoryOwner": {"login": head_owner},
        },
        base,
        head,
    )


def test_find_open_pr_returns_pr_info(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path
) -> None:
    """Real git for branch; gh wrappers stubbed for the PR + repo lookups."""
    row, base, head = _local_pr_row(git_repo)
    monkeypatch.setattr(git_ops, "gh_pr_list_for_branch", lambda *_a, **_k: [row])
    monkeypatch.setattr(
        git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("o", "r")
    )
    info = pr_review.find_open_pr(git_repo)
    assert info is not None
    assert (info.number, info.head_sha, info.base_sha, info.owner, info.repo) == (
        7, head, base, "o", "r",
    )


def test_find_open_pr_captures_head_repo_for_fork_pr(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path
) -> None:
    """Fork-head PR: the row's headRepository/headRepositoryOwner (the fork)
    is captured as ``head_repo`` for the reviewed-commit link, while
    owner/repo (the POST target) stay the base repo from ``gh repo view``."""
    row, _, _ = _local_pr_row(git_repo, head_owner="forky")
    row["headRepository"] = {"name": "widgets", "nameWithOwner": "forky/widgets"}
    monkeypatch.setattr(git_ops, "gh_pr_list_for_branch", lambda *_a, **_k: [row])
    monkeypatch.setattr(
        git_ops,
        "gh_repo_view_required",
        lambda _r, **_kwargs: ("acme", "widgets"),
    )
    info = pr_review.find_open_pr(git_repo)
    assert info is not None
    assert (info.owner, info.repo) == ("acme", "widgets")
    assert info.head_repo == "forky/widgets"
    assert info.head_ref == "feature"


def test_find_pr_by_number_returns_none_when_pr_missing(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path
) -> None:
    """An unresolvable PR number short-circuits before the repo lookup."""
    monkeypatch.setattr(git_ops, "gh_pr_view", lambda *_a, **_k: None)
    assert pr_review.find_pr_by_number(git_repo, 7) is None


def test_find_pr_by_number_raises_when_slug_unresolved(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path
) -> None:
    """A resolvable PR but failed owner/repo lookup is a hard error."""
    row, _, _ = _local_pr_row(git_repo)
    monkeypatch.setattr(git_ops, "gh_pr_view", lambda *_a, **_k: row)

    def fail_slug(_repo: Path, **_kwargs: Any) -> tuple[str, str]:
        raise GitError("gh repo view failed: auth")

    monkeypatch.setattr(git_ops, "gh_repo_view_required", fail_slug)
    with pytest.raises(GitError, match="auth"):
        pr_review.find_pr_by_number(git_repo, 7)


def test_find_pr_by_number_assembles_pr_info(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path
) -> None:
    """Valid lookups assemble a fully-populated PRInfo from the gh view row."""
    row, base, head = _local_pr_row(git_repo, head_owner="forky")
    monkeypatch.setattr(git_ops, "gh_pr_view", lambda *_a, **_k: row)
    monkeypatch.setattr(
        git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("o", "r")
    )
    info = pr_review.find_pr_by_number(git_repo, 7)
    assert info is not None
    assert (
        info.number, info.head_sha, info.base_sha, info.base_ref,
        info.owner, info.repo, info.url, info.head_repo,
    ) == (
        7, head, base, "main", "o", "r", "https://github.com/o/r/pull/7", "forky/r",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("number", True),
        ("number", 0),
        ("headRefOid", "HEAD"),
        ("headRefOid", "deadbeef"),
        ("headRefName", ""),
        ("headRefName", 7),
        ("headRefName", "feature~1"),
        ("baseRefName", ""),
        ("baseRefName", "main~1"),
        ("url", ""),
        ("url", 7),
        ("headRepository", "fork/r"),
        ("headRepository", {"nameWithOwner": "fork/r/extra"}),
        ("headRepository", {}),
        ("headRepository", {"nameWithOwner": 7, "name": "r"}),
        ("headRepositoryOwner", {}),
        ("headRepositoryOwner", {"login": 7}),
        ("headRepositoryOwner", {"login": ""}),
        ("headRepositoryOwner", "fork"),
    ],
)
def test_pr_info_rejects_malformed_row_fields(
    monkeypatch: pytest.MonkeyPatch,
    git_repo: Path,
    field: str,
    value: Any,
) -> None:
    row, _, _ = _local_pr_row(git_repo)
    row[field] = value
    monkeypatch.setattr(
        git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("o", "r")
    )
    with pytest.raises(GitError, match="invalid PR row|base branch|exact PR head"):
        pr_review._pr_info_from_row(git_repo, row)


@pytest.mark.parametrize("missing_field", ["headRepository", "headRepositoryOwner"])
def test_pr_info_rejects_missing_requested_head_metadata(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path, missing_field: str,
) -> None:
    row, _, _ = _local_pr_row(git_repo)
    del row[missing_field]
    monkeypatch.setattr(
        git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("o", "r")
    )
    with pytest.raises(GitError, match="invalid PR row"):
        pr_review._pr_info_from_row(git_repo, row)


def test_pr_info_rejects_malformed_owner_even_with_null_head_repository(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path,
) -> None:
    row, _, _ = _local_pr_row(git_repo)
    row["headRepository"] = None
    row["headRepositoryOwner"] = {"login": 7}
    monkeypatch.setattr(
        git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("o", "r")
    )
    with pytest.raises(GitError, match="invalid PR row"):
        pr_review._pr_info_from_row(git_repo, row)


@pytest.mark.parametrize("head_owner", [None, {"login": "former-owner"}])
def test_pr_info_accepts_null_same_repo_head_metadata(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path, head_owner: Any,
) -> None:
    row, _, _ = _local_pr_row(git_repo)
    row["headRepository"] = None
    row["headRepositoryOwner"] = head_owner
    monkeypatch.setattr(
        git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("o", "r")
    )

    assert pr_review._pr_info_from_row(git_repo, row).head_repo is None


def test_find_open_pr_propagates_current_branch_failure(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path
) -> None:
    def fail_branch(_repo: Path) -> str | None:
        raise GitError("cannot read current branch")

    monkeypatch.setattr(git_ops, "current_branch", fail_branch)
    with pytest.raises(GitError, match="cannot read current branch"):
        pr_review.find_open_pr(git_repo)


@pytest.mark.parametrize("lookup", ["branch", "number"])
def test_fork_pr_uses_upstream_base_for_both_lookup_paths(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path, lookup: str
) -> None:
    row, base, _ = _local_pr_row(git_repo, head_owner="forky")
    _git(git_repo, "remote", "add", "origin", "https://github.com/forky/r.git")
    _git(git_repo, "remote", "add", "upstream", "https://github.com/acme/widgets.git")
    _git(git_repo, "update-ref", "refs/remotes/upstream/main", base)
    monkeypatch.setattr(
        git_ops,
        "gh_repo_view_required",
        lambda _r, **_kwargs: ("acme", "widgets"),
    )
    monkeypatch.setattr(git_ops, "gh_pr_list_for_branch", lambda *_a, **_k: [row])
    monkeypatch.setattr(git_ops, "gh_pr_view", lambda *_a, **_k: row)

    info = (
        pr_review.find_open_pr(git_repo)
        if lookup == "branch"
        else pr_review.find_pr_by_number(git_repo, 7)
    )
    assert info is not None
    assert info.base_sha == base
    assert info.head_repo == "forky/r"


def test_pr_base_remote_matching_is_credential_safe(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path
) -> None:
    row, base, _ = _local_pr_row(git_repo)
    tree = _git(git_repo, "write-tree")
    unrelated = _git(git_repo, "commit-tree", tree, "-m", "unrelated base")
    _git(
        git_repo,
        "remote",
        "add",
        "origin",
        "https://user:top-secret@github.com/acme/widgets.git?token=private",
    )
    _git(git_repo, "update-ref", "refs/remotes/origin/main", unrelated)
    monkeypatch.setattr(
        git_ops,
        "gh_repo_view_required",
        lambda _r, **_kwargs: ("acme", "widgets"),
    )

    with pytest.raises(GitError) as excinfo:
        pr_review._pr_info_from_row(git_repo, row)
    assert "top-secret" not in str(excinfo.value)
    assert "token=private" not in str(excinfo.value)

    _git(
        git_repo,
        "remote",
        "set-url",
        "origin",
        "https://user:fork-secret@github.com/forky/widgets.git?token=fork",
    )
    assert pr_review._pr_info_from_row(git_repo, row).base_sha == base


class _FakeConsole:
    def print(self, *_a: Any, **_k: Any) -> None:
        pass


def _assumed_context(answer: str) -> RunContext:
    return RunContext(InteractionPolicy(assume=answer))


@pytest.mark.asyncio
async def test_post_skips_when_no_pr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td, **_kwargs: None)
    warnings: list[str] = []
    monkeypatch.setattr(
        pr_review,
        "print_warning",
        lambda _c, msg: warnings.append(msg),
    )
    status = await pr_review._post(
        tmp_path,
        [ParsedIssue(path="x.py", line=1, title="t", body="b")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert warnings and "No open PR" in warnings[0]
    assert status == pr_review.PostStatus.NO_PR


@pytest.mark.asyncio
async def test_post_fails_with_safe_diagnostic_when_pr_lookup_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_lookup(_target_dir: Path, **_kwargs: Any) -> PRInfo | None:
        raise GitError("gh pr list failed: authentication required")

    monkeypatch.setattr(pr_review, "find_open_pr", fail_lookup)
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        pr_review,
        "print_error",
        lambda _console, title, message: errors.append((title, message)),
    )

    status = await pr_review._post(
        tmp_path,
        [ParsedIssue(path="x.py", line=1, title="t", body="b")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )

    assert status == pr_review.PostStatus.FAILED
    assert errors == [("PR Lookup Failed", "gh pr list failed: authentication required")]


@pytest.mark.asyncio
async def test_post_succeeds_and_prints_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pr: PRInfo) -> None:
    """On a successful submit the URL is forwarded to print_success."""
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td, **_kwargs: pr)
    monkeypatch.setattr(
        pr_review,
        "classify",
        lambda *_a, **_k: pr_review._ClassifiedIssues(
            inline=[{"path": "a.py", "line": 1, "side": "RIGHT", "body": "x"}],
            body_only=[],
        ),
    )
    captured: dict[str, pr_review.ClassifiedReviewPlan] = {}

    def fake_submit(
        plan: pr_review.ClassifiedReviewPlan, *, transport: pr_review.ReviewTransport
    ) -> pr_review.ClassifiedReviewResult:
        captured["plan"] = plan
        return pr_review.ClassifiedReviewResult(
            status=pr_review.SubmissionStatus.POSTED,
            review_url="https://github.com/acme/widgets/pull/42#pullrequestreview-1",
            posted_file_level=(),
            folded_file_level=(),
            final_review_posted=True,
            safe_error=None,
        )

    monkeypatch.setattr(pr_review, "post_classified_review", fake_submit)
    successes: list[str] = []
    monkeypatch.setattr(
        pr_review,
        "print_success",
        lambda _c, msg: successes.append(msg),
    )
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)

    status = await pr_review._post(
        tmp_path,
        [ParsedIssue(path="a.py", line=1, title="t", body="b")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        run_context=_assumed_context("yes"),
        renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert captured["plan"].pr.head_sha == pr.head_sha
    assert captured["plan"].event is pr_review.ReviewEvent.COMMENT
    assert successes and "pullrequestreview" in successes[0]
    assert status == pr_review.PostStatus.POSTED


@pytest.mark.asyncio
async def test_post_payload_approves_when_clean_and_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pr: PRInfo,
) -> None:
    """_post with approve_on_clean=True + clean classified -> APPROVE payload."""
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td, **_kwargs: pr)
    monkeypatch.setattr(
        pr_review,
        "classify",
        lambda *_a, **_k: pr_review._ClassifiedIssues(
            inline=[{"path": "a.py", "line": 1, "side": "RIGHT", "body": "x"}],
            inline_issues=[
                ParsedIssue(
                    path="a.py",
                    line=1,
                    title="t",
                    body="b",
                    confidence="LOW",
                    severity="low",
                )
            ],
        ),
    )
    captured: dict[str, pr_review.ClassifiedReviewPlan] = {}

    def fake_submit(
        plan: pr_review.ClassifiedReviewPlan, *, transport: pr_review.ReviewTransport
    ) -> pr_review.ClassifiedReviewResult:
        captured["plan"] = plan
        return pr_review.ClassifiedReviewResult(
            status=pr_review.SubmissionStatus.POSTED,
            review_url="https://github.com/acme/widgets/pull/42#pullrequestreview-1",
            posted_file_level=(),
            folded_file_level=(),
            final_review_posted=True,
            safe_error=None,
        )

    monkeypatch.setattr(pr_review, "post_classified_review", fake_submit)
    monkeypatch.setattr(pr_review, "print_success", lambda *_a, **_k: None)
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)

    status = await pr_review._post(
        tmp_path,
        [ParsedIssue(path="a.py", line=1, title="t", body="b", severity="low")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        approve_on_clean=True,
        run_context=_assumed_context("yes"),
        renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert captured["plan"].event is pr_review.ReviewEvent.APPROVE
    assert status == pr_review.PostStatus.POSTED


@pytest.mark.asyncio
async def test_post_warns_with_preserved_payload_path_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pr: PRInfo,
) -> None:
    """When submit returns an error, the warning surfaces git_ops's preserved-path text."""
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td, **_kwargs: pr)
    monkeypatch.setattr(
        pr_review,
        "classify",
        lambda *_a, **_k: pr_review._ClassifiedIssues(
            inline=[{"path": "a.py", "line": 1, "side": "RIGHT", "body": "x"}],
            body_only=[],
        ),
    )
    err = "GitHub review submission failed (request payload preserved at /tmp/x.json)"
    monkeypatch.setattr(
        pr_review,
        "post_classified_review",
        lambda _plan, *, transport: pr_review.ClassifiedReviewResult(
            status=pr_review.SubmissionStatus.FAILED,
            review_url=None,
            posted_file_level=(),
            folded_file_level=(),
            final_review_posted=False,
            safe_error=err,
        ),
    )
    warnings: list[str] = []
    monkeypatch.setattr(
        pr_review,
        "print_warning",
        lambda _c, msg: warnings.append(msg),
    )
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)

    status = await pr_review._post(
        tmp_path,
        [ParsedIssue(path="a.py", line=1, title="t", body="b")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        run_context=_assumed_context("yes"),
        renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert warnings
    assert "no comments were posted" in warnings[0].lower()
    # The structured safe error preserves the request payload path.
    assert "payload preserved at /tmp/x.json" in warnings[0]
    assert status == pr_review.PostStatus.FAILED


def test_github_transport_surfaces_only_structured_preserved_payload_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pr: PRInfo,
) -> None:
    error = GitError("remote response leaked secret=never-show-this")
    error.preserved_payload_path = Path("/tmp/review.json")

    def fail_gh(*_args: Any, **_kwargs: Any) -> object:
        raise error

    monkeypatch.setattr(git_ops, "gh_api", fail_gh)

    result = pr_review.GitHubReviewTransport(
        target_dir=tmp_path,
        auth=git_ops.INHERIT_GITHUB_AUTH,
    ).post_review(
        pr,
        pr_review.ReviewPayload(
            event=pr_review.ReviewEvent.COMMENT,
            commit_id=pr.head_sha,
            body="review body",
            comments=(),
        ),
    )

    assert result.review_url is None
    assert result.safe_error == "GitHub review submission failed (request payload preserved at /tmp/review.json)"
    assert "secret" not in result.safe_error


@pytest.mark.asyncio
async def test_post_skipped_when_user_declines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pr: PRInfo) -> None:
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td, **_kwargs: pr)
    monkeypatch.setattr(
        pr_review,
        "classify",
        lambda *_a, **_k: pr_review._ClassifiedIssues(
            inline=[{"path": "a.py", "line": 1, "side": "RIGHT", "body": "x"}],
            body_only=[],
        ),
    )
    submit_called = False

    def fake_submit(
        _plan: pr_review.ClassifiedReviewPlan, *, transport: pr_review.ReviewTransport
    ) -> pr_review.ClassifiedReviewResult:
        nonlocal submit_called
        submit_called = True
        return pr_review.ClassifiedReviewResult(
            status=pr_review.SubmissionStatus.POSTED,
            review_url="x",
            posted_file_level=(),
            folded_file_level=(),
            final_review_posted=True,
            safe_error=None,
        )

    monkeypatch.setattr(pr_review, "post_classified_review", fake_submit)
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)

    status = await pr_review._post(
        tmp_path,
        [ParsedIssue(path="a.py", line=1, title="t", body="b")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        run_context=_assumed_context("no"),
        renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert not submit_called
    assert status == pr_review.PostStatus.NOTHING_TO_POST


@pytest.mark.asyncio
async def test_post_review_from_report_empty_items_is_nothing_to_post(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Empty merged items skip the post cleanly (NOTHING_TO_POST, not a failure)."""
    merged = tmp_path / "merged-items.json"
    merged.write_text(json.dumps({"items": []}))
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)

    status = await pr_review.post_review_to_pr_from_report(
        tmp_path,
        merged,
        console=_FakeConsole(),  # type: ignore[arg-type]
        renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert status == pr_review.PostStatus.NOTHING_TO_POST


@pytest.mark.asyncio
async def test_post_review_from_report_empty_items_posts_diagram(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pr: PRInfo,
) -> None:
    merged = tmp_path / "merged-items.json"
    merged.write_text(json.dumps({"items": []}))
    blocks = "<details><summary><h3>Flowchart</h3></summary>\nX\n</details>"
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td, **_kwargs: pr)
    monkeypatch.setattr(pr_review, "classify", lambda *_a, **_k: pr_review._ClassifiedIssues())
    captured: dict[str, pr_review.ClassifiedReviewPlan] = {}

    def fake_submit(
        plan: pr_review.ClassifiedReviewPlan, *, transport: pr_review.ReviewTransport
    ) -> pr_review.ClassifiedReviewResult:
        captured["plan"] = plan
        return pr_review.ClassifiedReviewResult(
            status=pr_review.SubmissionStatus.POSTED,
            review_url="https://github.com/acme/widgets/pull/42#pullrequestreview-1",
            posted_file_level=(),
            folded_file_level=(),
            final_review_posted=True,
            safe_error=None,
        )

    monkeypatch.setattr(pr_review, "post_classified_review", fake_submit)
    monkeypatch.setattr(pr_review, "print_success", lambda *_a, **_k: None)
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)

    status = await pr_review.post_review_to_pr_from_report(
        tmp_path,
        merged,
        console=_FakeConsole(),  # type: ignore[arg-type]
        post=True,
        diagram_blocks=blocks,
        renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )

    assert status == pr_review.PostStatus.POSTED
    assert captured["plan"].event is pr_review.ReviewEvent.COMMENT
    assert captured["plan"].diagram_blocks == blocks


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


# --- Diff-aware line resolution (issue #1102) -----------------------------
#
# `resolve_line` is documented (deep/location_validator.py:1-10) as a
# no-op-on-valid backstop behind the merge-time location validator. Before
# #1102 it never saw the diff, so a correct in-hunk line was re-derived from
# prose tokens and could be relocated to the first anchor hit anywhere in the
# file -- after which `snap_to_hunk` dropped the finding off the diff.

# The finding from the issue's reproduction: a prose-heavy rationale whose one
# code identifier (`ttl`) is short enough that longest-first ranking cuts it.
_PROSE_HEAVY_ISSUE = ParsedIssue(
    path="cache.yaml",
    line=12,
    title="Expiration collapses from an hour to a minute",
    body=(
        "The new override sets `ttl` far below every other environment, so cached "
        "entries become unavailable almost immediately and the downstream service "
        "absorbs the additional request volume."
    ),
)

# `prod:`'s TTL drops from an hour to a minute on line 12; every other block
# keeps 3600, so the anchor token `ttl` occurs on four lines and the only
# prose-matching token (`Expiration`) sits on line 1, outside the hunk.
_CACHE_YAML_BASE = (
    "# Expiration policy for the shared cache tier.\n"
    "defaults:\n"
    "  ttl: 3600\n"
    "\n"
    "staging:\n"
    "  ttl: 3600\n"
    "\n"
    "worker:\n"
    "  ttl: 3600\n"
    "\n"
    "prod:\n"
    "  ttl: 3600\n"
)
_CACHE_YAML_HEAD = _CACHE_YAML_BASE.replace("prod:\n  ttl: 3600\n", "prod:\n  ttl: 60\n")


def _pr_for(base_sha: str, head_sha: str) -> PRInfo:
    """A PRInfo pointing at two real commits in a temp repo."""
    return PRInfo(
        number=7,
        head_sha=head_sha,
        base_sha=base_sha,
        base_ref="main",
        head_ref="feature",
        owner="acme",
        repo="widgets",
        url="https://github.com/acme/widgets/pull/7",
    )


def test_extract_anchors_prefer_quoted_keeps_short_backticked_identifier() -> None:
    """Quoted-first ranking keeps a 3-char identifier that prose would crowd out."""
    text = f"{_PROSE_HEAVY_ISSUE.title}\n{_PROSE_HEAVY_ISSUE.body}"
    assert extract_anchors(text, prefer_quoted=True)[0] == "ttl"
    # Bare words still rank longest-first behind the quoted tokens.
    bare = extract_anchors(text, prefer_quoted=True)[1:]
    assert bare == sorted(bare, key=len, reverse=True)


def test_extract_anchors_default_ordering_stays_frozen_for_fingerprints() -> None:
    """The default (fingerprint-hashed) selection is unchanged by #1102.

    ``compute_fingerprint`` hashes the default token set, so re-ranking it
    would re-identify every open finding and defeat the reconcile dedup. The
    prose-heavy rationale must still crowd ``ttl`` out of the default cap --
    that is exactly the selection the existing fingerprints were built from.
    """
    text = f"{_PROSE_HEAVY_ISSUE.title}\n{_PROSE_HEAVY_ISSUE.body}"
    anchors = extract_anchors(text)
    assert anchors == [
        "environment",
        "unavailable",
        "immediately",
        "Expiration",
        "downstream",
        "additional",
        "collapses",
        "override",
    ]
    assert "ttl" not in anchors


def test_resolve_line_trusts_in_hunk_hint_without_any_anchor_match(git_repo: Path) -> None:
    """An in-hunk line hint is returned unchanged even when no anchor matches.

    This is the "no-op-on-valid" contract: the merge-time validator already
    confirmed the line against the hunk index, so posting must not re-derive it.
    """
    text = "\n".join(f"row_{i}" for i in range(1, 21)) + "\n"
    sha = _commit_file(git_repo, "x.py", text, "add x.py")
    issue = ParsedIssue(path="x.py", line=10, title="Latency regression", body="prose only")
    # No anchor from the title/body occurs anywhere in the file.
    assert pr_review.resolve_line(git_repo, sha, issue) is None
    assert pr_review.resolve_line(git_repo, sha, issue, [(8, 12)]) == 10


def test_resolve_line_prefers_in_hunk_anchor_hit_over_first_file_hit(git_repo: Path) -> None:
    """With no usable hint, an in-hunk anchor hit beats the first hit in the file."""
    lines = [f"row_{i}" for i in range(1, 21)]
    lines[1] = "marker_token  # pre-existing, unchanged"
    lines[17] = "marker_token  # the changed line"
    sha = _commit_file(git_repo, "x.py", "\n".join(lines) + "\n", "add x.py")
    issue = ParsedIssue(path="x.py", line=None, title="t", body="`marker_token`")
    # Without hunks the first hit (line 2) wins, as before.
    assert pr_review.resolve_line(git_repo, sha, issue) == 2
    # With hunks, the in-hunk hit (line 18) wins over the earlier out-of-hunk one.
    assert pr_review.resolve_line(git_repo, sha, issue, [(16, 20)]) == 18


def test_resolve_line_returns_out_of_hunk_hit_when_no_in_hunk_candidate(
    git_repo: Path,
) -> None:
    """An out-of-hunk anchor hit is still returned when no anchor hits a hunk."""
    lines = [f"row_{i}" for i in range(1, 21)]
    lines[1] = "marker_token  # pre-existing, unchanged"
    sha = _commit_file(git_repo, "x.py", "\n".join(lines) + "\n", "add x.py")
    issue = ParsedIssue(path="x.py", line=None, title="t", body="`marker_token`")
    assert pr_review.resolve_line(git_repo, sha, issue, [(16, 20)]) == 2


def test_classify_keeps_prose_heavy_in_hunk_finding_inline(git_repo: Path) -> None:
    """Real-path #1102 repro: a correct in-hunk citation stays inline.

    Drives the real :func:`classify` over a real two-commit repo -- real
    ``git diff`` for both the changed-file set and the hunk ranges, no mocks.
    The finding cites line 12, the only changed line. Before the fix the eight
    surviving anchors were all rationale prose, none of them within +/-5 lines
    of line 12, so the hint was discarded and the full-file search relocated
    the comment to ``Expiration`` on line 1 -- 8 lines outside the hunk, so
    ``snap_to_hunk`` returned None and the finding lost its line.
    """
    base = _commit_file(git_repo, "cache.yaml", _CACHE_YAML_BASE, "add cache.yaml")
    head = _commit_file(git_repo, "cache.yaml", _CACHE_YAML_HEAD, "cut prod ttl")
    issue = replace(_PROSE_HEAVY_ISSUE)

    result = classify(git_repo, _pr_for(base, head), [issue])

    assert [c["line"] for c in result.inline] == [12], (
        f"in-hunk citation was not posted on its own line; "
        f"file_level={[i.path for i in result.file_level]} "
        f"body_only={[i.path for i in result.body_only]}"
    )
    assert result.inline[0]["path"] == "cache.yaml"
    assert not result.file_level
    assert not result.body_only
    # Nothing moved, so nothing is annotated.
    assert "**Placement:**" not in result.inline[0]["body"]


def test_classify_annotates_relocated_line(git_repo: Path) -> None:
    """A snapped line is recorded in the body instead of overwritten silently.

    Real-path: the finding cites line 15, two lines outside the real hunk
    (17, 23). ``snap_to_hunk`` moves the comment to 17, so the posted line is
    not the cited one -- and the relocation must be visible on the comment and
    in the issue body the findings artifact serialises (issue #1102).
    """
    lines = [f"row_{i}" for i in range(1, 31)]
    lines[14] = "settle_window = 5  # cited here"
    base = _commit_file(git_repo, "x.py", "\n".join(lines) + "\n", "add x.py")
    lines[19] = "row_20_changed"
    head = _commit_file(git_repo, "x.py", "\n".join(lines) + "\n", "change row 20")
    issue = ParsedIssue(path="x.py", line=15, title="t", body="`settle_window` is too small")

    result = classify(git_repo, _pr_for(base, head), [issue])

    assert [c["line"] for c in result.inline] == [17]
    note = "**Placement:** posted on line 17; reviewer cited line 15."
    assert note in issue.body, "the relocation was not recorded on the finding"
    assert note in result.inline[0]["body"], "the posted comment does not show the relocation"
    # Re-classifying the same objects must not stack duplicate notes.
    classify(git_repo, _pr_for(base, head), [issue])
    assert issue.body.count("**Placement:**") == 1


def test_classify_skips_file_hunks_for_path_missing_at_head(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path
) -> None:
    """A path that doesn't exist at head_sha (deleted/renamed) never calls file_hunks.

    ``resolve_line`` would reject such a path via its own ``git show`` no
    matter what hunks it was handed, so `classify` should discover the path
    is unresolvable up front instead of paying for the file_hunks() diff
    lookup (and its gh-pr-diff network fallback) first (issue #1102 follow-up).
    """
    base = _commit_file(git_repo, "gone.py", "x = 1\n", "add gone.py")
    _git(git_repo, "rm", "gone.py")
    _git(git_repo, "commit", "-m", "delete gone.py")
    head = _git(git_repo, "rev-parse", "HEAD")
    issue = ParsedIssue(path="gone.py", line=1, title="t", body="`x`")

    def fail_if_called(*_a: Any, **_k: Any) -> list[tuple[int, int]]:
        raise AssertionError("file_hunks should not be called for a path missing at head_sha")

    monkeypatch.setattr(pr_review, "file_hunks", fail_if_called)

    result = classify(git_repo, _pr_for(base, head), [issue])

    assert not result.inline
    assert [i.path for i in result.file_level] == ["gone.py"]


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
        git_ops, "gh_pr_diff", lambda _r, _n, **_kwargs: _GH_PR_DIFF
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


def test_demoted_high_finding_still_blocks_approval(pr: PRInfo) -> None:
    """R2.2: a judged-high finding demoted to low by location validation must
    still block APPROVE — demotion is visible, never approval-silencing."""
    classified = pr_review._ClassifiedIssues(
        inline=[{"path": "a.py", "line": 10, "side": "RIGHT", "body": "x"}],
        inline_issues=[
            ParsedIssue(
                path="a.py",
                line=10,
                title="t",
                body="b",
                severity="low",
                location_distrust=True,
                severity_before_demotion="high",
            )
        ],
    )
    payload = build_payload(
        pr,
        classified,
        approve_on_clean=True,
        renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert payload["event"] == "COMMENT"


def test_demoted_low_finding_does_not_block_approval(pr: PRInfo) -> None:
    """#336: a demoted finding whose ORIGINAL severity was already low (or never
    asserted) must not block APPROVE — the location_distrust mark is written for
    every beyond-tolerance record, but only a demotion from a blocking severity
    keeps the gate closed."""
    for before in ("low", None):
        classified = pr_review._ClassifiedIssues(
            inline=[{"path": "a.py", "line": 10, "side": "RIGHT", "body": "x"}],
            inline_issues=[
                ParsedIssue(
                    path="a.py",
                    line=10,
                    title="t",
                    body="b",
                    severity="low",
                    location_distrust=True,
                    severity_before_demotion=before,
                )
            ],
        )
        payload = build_payload(
            pr,
            classified,
            approve_on_clean=True,
            renderers=BUILTIN_RENDERERS,
            run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
        )
        assert payload["event"] == "APPROVE"


def test_demoted_low_finding_approval_gate_does_not_block() -> None:
    """#336: the approval-gate unit check itself — location_distrust alone no
    longer blocks once the original severity was low or never asserted."""
    assert pr_review._finding_blocks_approval(
        "low", True, False, "low"
    ) is False
    assert pr_review._finding_blocks_approval(
        "low", True, False, None
    ) is False
    assert pr_review._finding_blocks_approval(
        "low", True, False, "high"
    ) is True
    assert pr_review._finding_blocks_approval(
        "low", True, False, "medium"
    ) is True


def test_parsed_issues_carry_location_distrust_and_report_note() -> None:
    """R2 visibility: the demotion reaches the human-read issue body and the
    machine-readable flag rides on the ParsedIssue."""
    items = [
        {
            "file": "a.py",
            "line": 10,
            "description": "off-citation",
            "rationale": "r",
            "severity": "low",
            "confidence": "LOW",
            "severity_before_demotion": "high",
            "location_distrust": True,
        }
    ]
    issues = parsed_issues_from_items(items)
    assert len(issues) == 1
    assert issues[0].location_distrust is True
    assert "**Location:** unverified citation (severity demoted from high)" in issues[0].body


@pytest.mark.parametrize("raw", [{"severity": None}, {}])  # present-but-null ≡ omitted (R4.1)
def test_null_severity_coerces_to_none_not_none_string(raw: dict[str, Any]) -> None:
    fields = pr_review.extract_item_fields({"file": "a.py", "line": 1, **raw})
    assert fields is not None
    assert fields.severity is None  # not the string "none"


def test_null_severity_does_not_block_approval(pr: PRInfo, monkeypatch: pytest.MonkeyPatch) -> None:
    # SUPERVISE_SCHEMA emits severity: null — must approve like omitted.
    classified = pr_review._ClassifiedIssues(
        inline=[{"path": "a.py", "line": 10, "side": "RIGHT", "body": "x"}],
        inline_issues=[ParsedIssue(path="a.py", line=10, title="t", body="b", severity=None)],
    )

    payload = build_payload(
        pr,
        classified,
        approve_on_clean=True,
        renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert payload["event"] == "APPROVE"
    assert "**Severity:** none" not in payload["body"]  # no phantom label rendered


def test_artifact_off_vocabulary_severity_blocks_approval() -> None:
    """R1/F1: an off-vocabulary label ('critical') folded to None at the Phase A
    boundary must still block the artifact-backed approve gate.

    The raw off-vocabulary signal rides through the artifact as
    ``severity_off_vocabulary`` so the poster's gate fails closed even though
    ``severity`` reads ``None``.
    """
    finding = ArtifactFinding(
        fingerprint="f" * 64,
        path="a.py",
        line=10,
        placement="inline",
        title="t",
        body="b",
        severity=None,  # "critical" was folded to None by Phase A normalization
        confidence="HIGH",
        is_cross_stack=False,
        severity_off_vocabulary=True,
    )
    issue = pr_review._issue_from_artifact_finding(finding)
    assert issue.severity is None
    assert issue.severity_off_vocabulary is True
    # The approval gate blocks on the off-vocabulary signal even with severity None.
    assert pr_review._finding_blocks_approval(
        issue.severity, issue.location_distrust, issue.severity_off_vocabulary
    ) is True


def test_artifact_folding_to_none_not_off_vocabulary_does_not_block() -> None:
    """Present-but-null severity (wire ``severity: null``) is not an asserted
    off-vocabulary label: it models an omitted severity and does not block."""
    finding = ArtifactFinding(
        fingerprint="f" * 64,
        path="a.py",
        line=10,
        placement="inline",
        title="t",
        body="b",
        severity=None,
        confidence="HIGH",
        is_cross_stack=False,
        severity_off_vocabulary=False,
    )
    issue = pr_review._issue_from_artifact_finding(finding)
    assert pr_review._finding_blocks_approval(
        issue.severity, issue.location_distrust, issue.severity_off_vocabulary
    ) is False


# --- Grounded diagrams (issue #1113) ----------------------------------------


def test_diagram_marker_round_trip() -> None:
    """The hidden marker is invisible in rendered markdown and parses back exactly."""
    from daydream.pr_review import diagram_marker, parse_diagram_markers

    body = "\n\n".join(
        [
            diagram_marker("sequence", "a" * 40),
            diagram_marker("flowchart", "a" * 40),
            "<details>the diagrams</details>",
        ]
    )
    assert parse_diagram_markers(body) == [
        ("sequence", "a" * 40),
        ("flowchart", "a" * 40),
    ]
    # A finding marker is a different namespace and must not cross-parse.
    assert parse_diagram_markers(pr_review.finding_marker("f" * 64)) == []
    assert pr_review.parse_finding_markers(diagram_marker("sequence", "a" * 40)) == []


def test_diagram_replacement_post_failure_keeps_prior_comment(
    pr: PRInfo, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from daydream.git_ops import GitError
    from daydream.reconcile import PriorDiagramComment

    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        "daydream.reconcile.fetch_prior_diagram_comments",
        lambda *_a, **_k: [PriorDiagramComment("IC_prior", ("sequence",))],
    )

    def minimize(_target: Path, node_id: str, **_kwargs: Any) -> bool:
        calls.append(("minimize", node_id))
        return True

    monkeypatch.setattr(
        "daydream.reconcile.minimize_comment",
        minimize,
    )

    def fail_post(*_args: Any, **_kwargs: Any) -> Any:
        calls.append(("post", None))
        raise GitError("simulated POST failure")

    monkeypatch.setattr(git_ops, "gh_api", fail_post)

    result = pr_review.post_diagram_comment_to_pr(
        tmp_path,
        pr,
        body="diagram",
        kinds=["sequence"],
        bot_login="daydream",
    )

    assert result == (None, "simulated POST failure")
    assert calls == [("post", None)]


def test_diagram_replacement_posts_before_minimizing_matching_prior_comment(
    pr: PRInfo, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from daydream.reconcile import PriorDiagramComment

    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        "daydream.reconcile.fetch_prior_diagram_comments",
        lambda *_a, **_k: [
            PriorDiagramComment("IC_matching", ("sequence",)),
            PriorDiagramComment("IC_other_kind", ("flowchart",)),
        ],
    )

    def minimize(_target: Path, node_id: str, **_kwargs: Any) -> bool:
        calls.append(("minimize", node_id))
        return True

    monkeypatch.setattr(
        "daydream.reconcile.minimize_comment",
        minimize,
    )

    def post(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        calls.append(("post", None))
        return {"html_url": "https://github.com/acme/widgets/issues/42#issuecomment-1"}

    monkeypatch.setattr(git_ops, "gh_api", post)

    result = pr_review.post_diagram_comment_to_pr(
        tmp_path,
        pr,
        body="diagram",
        kinds=["sequence"],
        bot_login="daydream",
    )

    assert result == (
        "https://github.com/acme/widgets/issues/42#issuecomment-1",
        None,
    )
    assert calls == [("post", None), ("minimize", "IC_matching")]


def test_build_payload_places_diagram_blocks_under_the_header(pr: PRInfo, monkeypatch: pytest.MonkeyPatch) -> None:
    """``diagram_blocks`` lands directly under ``**Code Review Summary**``."""
    classified = pr_review._ClassifiedIssues(
        body_only=[
            ParsedIssue(
                path="b.py",
                line=None,
                title="File note",
                body="desc",
                fingerprint="b" * 64,
            )
        ]
    )
    blocks = "<details><summary><h3>Sequence Diagram</h3></summary>\nX\n</details>"

    payload = pr_review.build_payload(
        pr,
        classified,
        diagram_blocks=blocks,
        renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    body = payload["body"]
    header = "**Code Review Summary**"
    assert body[body.index(header) + len(header) :].lstrip().startswith(blocks)
    # Absent (the default) is byte-identical to the pre-#1113 body.
    assert pr_review.build_payload(
        pr, classified, renderers=BUILTIN_RENDERERS, run_info=pr_comment_renderer.render_run_info_block([_FIXTURE])
    )["body"] == body.replace(f"{header}\n\n{blocks}", header)


def test_custom_summary_renderer_receives_and_may_drop_diagrams(pr: PRInfo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec test 16: ``ctx.diagrams`` reaches a fork renderer, which owns it.

    The host cannot inject the blocks around a custom renderer's output (they
    belong inside the summary body), so a renderer that ignores ``ctx.diagrams``
    drops them -- which is exactly what docs/extensions.md warns about.
    """
    from daydream.extensions import Registry
    from daydream.extensions.builtins import register_builtins

    seen: list[str | None] = []

    def keeps(ctx: Any) -> str:
        seen.append(ctx.diagrams)
        return f"**Custom**\n\n{ctx.diagrams}"

    def drops(ctx: Any) -> str:
        seen.append(ctx.diagrams)
        return "**Custom**"

    classified = pr_review._ClassifiedIssues()
    blocks = "<details><summary><h3>Flowchart</h3></summary>\nX\n</details>"
    for renderer, expect_present in ((keeps, True), (drops, False)):
        reg = Registry()
        register_builtins(reg)
        reg.override_renderer("summary", renderer)
        body = pr_review.build_payload(
            pr,
            classified,
            diagram_blocks=blocks,
            renderers=resolve_review_renderers(reg),
            run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
        )["body"]
        assert (blocks in body) is expect_present
    assert seen == [blocks, blocks]


def test_explicit_run_info_payload_is_byte_stable(pr: PRInfo) -> None:
    """Pin the host envelope, approval, rollup, markers and diagram placement."""
    classified = pr_review._ClassifiedIssues(
        body_only=[
            ParsedIssue(
                path="a.py",
                line=None,
                title="Fixture note",
                body="Fixture rationale",
                severity="low",
                confidence="HIGH",
                fingerprint="a" * 64,
            )
        ]
    )
    payload = build_payload(
        pr,
        classified,
        run_info="Fixture run info",
        approve_on_clean=True,
        diagram_blocks="<details>Fixture diagram</details>",
        renderers=BUILTIN_RENDERERS,
    )
    payload["body"] = payload["body"].replace(DAYDREAM_FOOTER, "<DAYDREAM_FOOTER>")
    assert payload == json.loads((SNAP / "explicit_payload.json").read_text())


def test_payload_uses_only_explicit_renderers_and_run_info(
    pr: PRInfo, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daydream.extensions import Registry, SummaryContext
    from daydream.extensions.builtins import register_builtins

    seen: list[SummaryContext] = []

    def summary(ctx: SummaryContext) -> str:
        seen.append(ctx)
        return default_render_summary(ctx)

    registry = Registry()
    register_builtins(registry)
    registry.override_renderer("summary", summary)
    renderers = resolve_review_renderers(registry)
    classified = pr_review._ClassifiedIssues(body_only=[ParsedIssue(
        path="a.py", line=None, title="Explicit finding", body="Explanation",
        severity="low", confidence="HIGH",
    )])
    run_info = pr_comment_renderer.render_run_info_block([_FIXTURE])

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("payload assembly accessed ambient state or the filesystem")

    registry.override_renderer("summary", forbidden)
    monkeypatch.setattr(pr_review, "get_registry", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr("tempfile.TemporaryDirectory", forbidden)
    monkeypatch.setattr("daydream.trajectory.get_current_recorder", forbidden)
    first = build_payload(pr, classified, run_info=run_info, renderers=renderers)
    second = build_payload(pr, classified, run_info=run_info, renderers=renderers)
    assert first == second
    assert seen[0] == seen[1]
    assert seen[0].findings[0].finding.title == "Explicit finding"
    assert run_info in seen[0].review_info
