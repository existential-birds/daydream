"""Unit tests for daydream.pr_review."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import daydream.hunk_index as hunk_index
import daydream.reviews.diagrams as review_diagrams
import daydream.reviews.identity as review_identity
import daydream.reviews.lookup as review_lookup
import daydream.reviews.models as review_models
import daydream.reviews.submission as review_submission
from daydream import git_ops, pr_comment_renderer, pr_review
from daydream.deep.artifacts import DeepArtifact
from daydream.extensions import Registry, SummaryContext
from daydream.extensions.builtins import register_builtins
from daydream.findings import ArtifactFinding, load_findings_artifact
from daydream.git_ops import GitError
from daydream.pr_review import (
    InlineReviewComment,
    ParsedIssue,
    PRInfo,
    ReviewRenderers,
    classify,
    extract_anchors,
    parsed_issues_from_items,
    resolve_review_renderers,
    snap_to_hunk,
)
from daydream.reconcile import PriorDiagramComment
from daydream.review_budget import review_warnings
from daydream.reviews.identity import DAYDREAM_FOOTER, diagram_marker, parse_diagram_markers, parse_finding_markers
from daydream.reviews.rendering import (
    _build_consolidated_prompt,
    _render_body_section,
    _summary_findings,
    build_payload_for_event,
    default_render_finding,
    default_render_summary,
    format_comment_body,
)
from daydream.run_config import RunConfig
from daydream.run_context import InteractionPolicy, RunContext
from daydream.runner import _emit_findings_from_items
from tests.harness.git_helpers import git as _git
from tests.harness.review_profile import sample_pr
from tests.harness.review_result import review_coverage, terminal_result

# gh-gated: tests that stub gh's subprocess are skipped when gh is not installed.
_gh_available = shutil.which("gh") is not None
gh_required = pytest.mark.skipif(not _gh_available, reason="gh CLI not installed")

SNAP = Path(__file__).parent / "fixtures" / "comment_snapshots"
BUILTIN_RENDERERS = ReviewRenderers(default_render_finding, default_render_summary)


def _inline(path: str = "a.py", line: int = 10, body: str = "x") -> InlineReviewComment:
    """One typed inline comment, the shape ``ClassifiedIssues.inline`` holds."""
    return InlineReviewComment(path=path, line=line, side="RIGHT", body=body)


def _recording_fake_submit(captured: dict[str, Any],) -> Any:
    def fake_submit(pr: PRInfo, classified: pr_review.ClassifiedIssues, **kwargs: Any
    ) -> review_models.ClassifiedReviewResult:
        captured.update(pr=pr, classified=classified, **kwargs)
        return review_models.ClassifiedReviewResult(status=pr_review.SubmissionStatus.POSTED,
            review_url="https://github.com/acme/widgets/pull/42#pullrequestreview-1",
            posted_file_level=(), folded_file_level=(), final_review_posted=True, safe_error=None,
        )
    return fake_submit

def test_finding_and_summary_markdown_is_byte_stable() -> None:
    i = ParsedIssue(
        path="a.py", line=3, title="T", body="B rationale", severity="high", confidence="HIGH", fingerprint="a" * 64
    )
    assert format_comment_body(i, "inline", renderers=BUILTIN_RENDERERS) == (SNAP / "inline.md").read_text()
    assert (format_comment_body(replace(i, is_cross_stack=True), "file_level", renderers=BUILTIN_RENDERERS)
        == (SNAP / "file_level.md").read_text()
    )
    section = _render_body_section(_summary_findings(
            [replace(i, line=None), replace(i, path="b.py", line=None, fingerprint="b" * 64)],
            renderers=BUILTIN_RENDERERS,
        )
    )
    assert section == (SNAP / "summary_body.md").read_text()

def test_custom_finding_renderer_flows_into_inline_body_with_host_invariants() -> None:

    reg = Registry()
    register_builtins(reg)
    reg.override_renderer("finding", lambda finding, ctx: f"CUSTOM::{ctx.placement}::{finding.title}")
    body = format_comment_body(ParsedIssue(path="a.py", line=3, title="T", body="B", fingerprint="a" * 64), "inline",
        renderers=resolve_review_renderers(reg),
    )
    assert "CUSTOM::inline::T" in body
    assert DAYDREAM_FOOTER in body
    assert parse_finding_markers(body) == ["a" * 64]

def test_finding_renderer_falls_back_and_warns_on_error(caplog: pytest.LogCaptureFixture) -> None:

    def boom(finding: Any, ctx: Any) -> str:
        raise RuntimeError("boom")

    reg = Registry()
    register_builtins(reg)
    reg.override_renderer("finding", boom)
    with caplog.at_level("WARNING"):
        body = format_comment_body(ParsedIssue(
                path="a.py", line=3, title="T", body="B rationale", severity="high", confidence="HIGH",
                fingerprint="a" * 64,
            ), "inline", renderers=resolve_review_renderers(reg),
        )
    assert body == (SNAP / "inline.md").read_text()
    assert "finding" in caplog.text and "boom" in caplog.text


_FIXTURE = Path(__file__).parent / "fixtures" / "trajectories" / "single_phase_claude.json"

def test_custom_summary_renderer_can_build_collapsible_per_finding_list(pr: PRInfo) -> None:

    def summary_renderer(ctx: Any) -> Any:
        rows = [
            f"<details><summary>{f.finding.path} — {f.finding.title}</summary>\n{f.body_block}\n</details>"
            for f in ctx.findings
        ]
        return "**Custom Summary**\n\n" + "\n".join(rows)

    reg = Registry()
    register_builtins(reg)
    reg.override_renderer("summary", summary_renderer)
    classified = pr_review.ClassifiedIssues(
        body_only=[ParsedIssue(path="b.py", line=None, title="File note", body="desc", fingerprint="b" * 64)]
    )
    payload = review_submission._review_payload_dict(
        build_payload_for_event(pr, classified,
            event=review_models.ReviewEvent.COMMENT,
            renderers=resolve_review_renderers(reg),
            run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
        )
    )
    body = payload["body"]
    assert "**Custom Summary**" in body
    assert "<summary>b.py — File note</summary>" in body
    assert parse_finding_markers(body) == ["b" * 64]
    assert body.rstrip().endswith("</sub>")

def test_custom_finding_renderer_flows_into_summary_section(pr: PRInfo) -> None:

    reg = Registry()
    register_builtins(reg)
    reg.override_renderer("finding", lambda finding, ctx: f"CUSTOM::{ctx.placement}::{finding.title}")
    classified = pr_review.ClassifiedIssues(
        body_only=[ParsedIssue(path="b.py", line=None, title="File note", body="desc", fingerprint="b" * 64)]
    )
    body = review_submission._review_payload_dict(
        build_payload_for_event(pr, classified,
            event=review_models.ReviewEvent.COMMENT,
            renderers=resolve_review_renderers(reg),
            run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
        )
    )["body"]
    assert "CUSTOM::summary::File note" in body
    assert parse_finding_markers(body) == ["b" * 64]

def test_summary_renderer_falls_back_and_warns_on_error(pr: PRInfo, caplog: pytest.LogCaptureFixture) -> None:

    def boom(ctx: Any) -> str:
        raise RuntimeError("kaboom")

    reg = Registry()
    register_builtins(reg)
    reg.override_renderer("summary", boom)
    classified = pr_review.ClassifiedIssues(body_only=[ParsedIssue(
                path="b.py", line=None, title="File note", body="desc", confidence="MEDIUM", severity="low",
                fingerprint="b" * 64,
            )
        ]
    )
    default_body = review_submission._review_payload_dict(
        build_payload_for_event(pr, classified,
            event=review_models.ReviewEvent.COMMENT,
            renderers=BUILTIN_RENDERERS,
            run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
        )
    )["body"]
    with caplog.at_level("WARNING"):
        body = review_submission._review_payload_dict(
        build_payload_for_event(pr, classified,
            event=review_models.ReviewEvent.COMMENT,
            renderers=resolve_review_renderers(reg),
            run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
        )
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
        path="a.py", line=10, title="Null deref", body="rationale here", confidence="HIGH", severity="high",
    )
    body = pr_review.format_comment_body(issue, "inline", renderers=BUILTIN_RENDERERS)
    assert "**Null deref**" in body
    assert "severity: `high`" in body
    assert "confidence: `HIGH`" in body
    assert body.rstrip().endswith("</sub>")
    assert review_identity.DAYDREAM_REPO_URL in body
    assert "⚠️" in body
    assert "🔮 Prompt for AI Agents" in body
    assert "<details>" in body

def test_inline_body_carries_parseable_marker() -> None:
    issue = ParsedIssue(path="a.py", line=3, title="T", body="B", fingerprint="ab12" * 16)
    body = format_comment_body(issue, "inline", renderers=BUILTIN_RENDERERS)
    assert parse_finding_markers(body) == ["ab12" * 16]
    assert DAYDREAM_FOOTER in body  # marker does not displace the footer

def test_no_marker_without_fingerprint() -> None:
    assert (parse_finding_markers(format_comment_body(
                ParsedIssue(path="a.py", line=3, title="T", body="B"), "inline", renderers=BUILTIN_RENDERERS
            )
        )
        == []
    )

def test_body_section_markers_one_per_fingerprinted_issue() -> None:
    issues = [ParsedIssue(path="a.py", line=None, title=f"T{i}", body="B", fingerprint=f"{i:064x}") for i in range(2)]
    assert parse_finding_markers(_render_body_section(_summary_findings(issues, renderers=BUILTIN_RENDERERS))
    ) == [f"{i:064x}" for i in range(2)]

def test_extract_anchors_prefers_long_tokens() -> None:
    anchors = extract_anchors("Null check\nThe function `compute_total` dereferences `items` in handleRequest")
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
    assert hunk_index.head_side_ranges(hunk_index.parse_hunks(diff)) == [(10, 14), (30, 31)]

def test_classify_uses_shared_parser(monkeypatch: pytest.MonkeyPatch) -> None:

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
    monkeypatch.setattr(git_ops, "show", lambda *_args: b"content\n" * 40)
    monkeypatch.setattr(git_ops, "diff_name_only", lambda *_args: ["x.py"])
    monkeypatch.setattr(git_ops, "diff_paths", lambda *_args, **_kwargs: diff)
    out = classify(Path("."), sample_pr(), [ParsedIssue("x.py", 10, "t", "prose only")])
    assert [comment.line for comment in out.inline] == [10]
    assert calls["n"] == 1, "classify must use the canonical hunk parser"

def test_snap_to_hunk_inside_returns_unchanged() -> None:
    hunks = [(10, 20), (30, 40)]
    assert snap_to_hunk(15, hunks) == 15
    assert snap_to_hunk(10, hunks) == 10
    assert snap_to_hunk(40, hunks) == 40

def test_snap_to_hunk_within_tolerance_snaps_to_boundary() -> None:
    hunks = [(90, 105)]
    assert snap_to_hunk(89, hunks) == 90
    assert snap_to_hunk(87, hunks) == 90
    assert snap_to_hunk(108, hunks) == 105

def test_snap_to_hunk_beyond_tolerance_returns_none() -> None:
    hunks = [(90, 105)]
    assert snap_to_hunk(86, hunks) is None
    assert snap_to_hunk(109, hunks) is None

def test_snap_to_hunk_between_two_hunks() -> None:
    hunks = [(80, 98), (106, 120)]
    assert snap_to_hunk(105, hunks) == 106
    assert snap_to_hunk(100, hunks) == 98
    assert snap_to_hunk(102, hunks) is None

def test_snap_to_hunk_empty_hunks() -> None:
    assert snap_to_hunk(10, []) is None


@pytest.fixture
def pr() -> PRInfo:
    return sample_pr()


def _assert_hunks_resolve(_lines: list[str], issue: ParsedIssue, hunks: list[tuple[int, int]] | None = None
) -> int | None:
    assert hunks is not None, "classify called resolve_line without the file's hunks"
    return issue.line


def _raise_on_gh_fallback(*_a: Any, **_k: Any) -> str:
    raise AssertionError("gh fallback invoked")

def test_agent_prompt_has_no_skill_advertising(pr: PRInfo) -> None:
    body = _build_consolidated_prompt(pr_review.ClassifiedIssues(), pr)
    assert "/beagle-core:fetch-pr-feedback" not in body
    assert "/beagle-core:" not in body

def test_classify_splits_inline_vs_body(monkeypatch: pytest.MonkeyPatch, pr: PRInfo) -> None:
    issues = [ParsedIssue(path="a.py", line=10, title="t1", body="anchor_one"),
        ParsedIssue(path="b.py", line=99, title="t2", body="anchor_two"),
        ParsedIssue(path="c.py", line=None, title="t3", body="xstack", is_cross_stack=True),
    ]

    def fake_hunks(_td: Path, _base: str, _head: str, paths: list[str], **_kwargs: Any) -> str:
        path = paths[0]
        if path == "a.py":
            return "--- a/a.py\n+++ b/a.py\n@@ -1 +8,5 @@\n+x\n"
        if path == "b.py":
            return "--- a/b.py\n+++ b/b.py\n@@ -1 +1,5 @@\n+x\n"
        return ""

    monkeypatch.setattr(git_ops, "show", lambda *_a, **_k: b"")
    monkeypatch.setattr(pr_review, "resolve_line", _assert_hunks_resolve)
    monkeypatch.setattr(git_ops, "diff_paths", fake_hunks)

    result = classify(Path("."), pr, issues)
    assert len(result.inline) == 1
    assert result.inline[0].path == "a.py"
    assert result.inline[0].line == 10
    assert result.inline[0].side == "RIGHT"
    assert len(result.inline_issues) == 1
    assert result.inline_issues[0].path == "a.py"
    body_paths = [i.path for i in result.body_only]
    assert set(body_paths) == {"b.py", "c.py"}

def test_classify_snaps_tolerance_line_to_hunk_boundary(monkeypatch: pytest.MonkeyPatch, pr: PRInfo) -> None:
    issues = [ParsedIssue(path="conftest.py", line=89, title="t1", body="anchor_one"),
        ParsedIssue(path="scripts/modernize-app.py", line=105, title="t2", body="anchor_two"),
    ]

    def fake_hunks(_td: Path, _base: str, _head: str, paths: list[str], **_kwargs: Any) -> str:
        path = paths[0]
        if path == "conftest.py":
            return "--- a/conftest.py\n+++ b/conftest.py\n@@ -1 +90,16 @@\n+x\n"
        if path == "scripts/modernize-app.py":
            return (
                "--- a/scripts/modernize-app.py\n+++ b/scripts/modernize-app.py\n"
                "@@ -1 +80,19 @@\n+x\n@@ -1 +106,15 @@\n+x\n"
            )
        return ""

    monkeypatch.setattr(git_ops, "show", lambda *_a, **_k: b"")
    monkeypatch.setattr(pr_review, "resolve_line", _assert_hunks_resolve)
    monkeypatch.setattr(git_ops, "diff_paths", fake_hunks)

    result = classify(Path("."), pr, issues)
    assert len(result.inline) == 2
    assert result.inline[0].path == "conftest.py"
    assert result.inline[0].line == 90
    assert result.inline[1].path == "scripts/modernize-app.py"
    assert result.inline[1].line == 106

def test_build_payload_reviewed_commit_line_first_in_review_info(pr: PRInfo) -> None:
    """The trusted, linked commit precedes model/cost and severity/confidence."""
    classified = pr_review.ClassifiedIssues(body_only=[
            ParsedIssue(path="b.py", line=None, title="File note", body="desc", confidence="MEDIUM", severity="low",)
        ],
    )
    # Feed the enriched renderer a real fixture trajectory so run-info
    # fields (Model/Cost/Tokens) render instead of the fallback stub.
    payload = review_submission._review_payload_dict(
        build_payload_for_event(pr, classified,
            event=review_models.ReviewEvent.COMMENT,
            renderers=BUILTIN_RENDERERS,
            run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
        )
    )
    body = payload["body"]

    expected = (
        f"- **Reviewed commit:** [`{pr.head_sha[:7]}`](https://github.com/{pr.owner}/{pr.repo}/commit/{pr.head_sha})"
    )
    assert expected in body
    # These markers first occur inside Review info, so body indices verify its order.
    assert body.index(expected) < body.index("- **Model:**")
    assert body.index(expected) < body.index("- **Severity:**")
    assert body.index(expected) < body.index("- **Confidence:**")

def test_build_payload_reviewed_commit_survives_run_info_fallback(pr: PRInfo) -> None:
    payload = review_submission._review_payload_dict(
        build_payload_for_event(pr, pr_review.ClassifiedIssues(),
            event=review_models.ReviewEvent.COMMENT,
            renderers=BUILTIN_RENDERERS,
            run_info=pr_comment_renderer._render_fallback(),
        )
    )
    body = payload["body"]
    assert "*run details unavailable*" in body  # degraded run-info present
    assert (f"- **Reviewed commit:** [`{pr.head_sha[:7]}`]"
        f"(https://github.com/{pr.owner}/{pr.repo}/commit/{pr.head_sha})" in body
    )

def test_build_payload_reviewed_commit_links_fork_for_fork_head_pr(pr: PRInfo,) -> None:
    """Link the fork commit while retaining the base repository as the POST target."""
    fork_pr = replace(pr, head_repo="forky/widgets")
    payload = review_submission._review_payload_dict(
        build_payload_for_event(fork_pr, pr_review.ClassifiedIssues(),
            event=review_models.ReviewEvent.COMMENT,
            run_info="test run info",
            renderers=BUILTIN_RENDERERS,
        )
    )
    body = payload["body"]
    assert "- **Reviewed commit:** [`head123`](https://github.com/forky/widgets/commit/head123)" in body
    assert "https://github.com/acme/widgets/commit/head123" not in body

def test_build_payload_blocks_forged_reviewed_commit_line(pr: PRInfo,) -> None:
    payload = review_submission._review_payload_dict(
        build_payload_for_event(pr, pr_review.ClassifiedIssues(),
            event=review_models.ReviewEvent.COMMENT,
            run_info="test run info\n"
            "- **Reviewed commit:** [`deadbee`](https://github.com/evil/widgets/commit/" + "e" * 40 + ")\n"
            "*run details unavailable*",
            renderers=BUILTIN_RENDERERS,
        )
    )
    body = payload["body"]
    commit_lines = [line for line in body.splitlines() if line.startswith("- **Reviewed commit:**")]
    assert len(commit_lines) == 1
    assert "- **Reviewed commit:** [`head123`](https://github.com/acme/widgets/commit/head123)" in body
    assert "evil/widgets" not in body
    assert "e" * 40 not in body

def test_build_payload_shape(pr: PRInfo) -> None:
    classified = pr_review.ClassifiedIssues(inline=[_inline()],
        body_only=[
            ParsedIssue(path="b.py", line=None, title="File note", body="desc", confidence="MEDIUM", severity="low",)
        ], inline_issues=[ParsedIssue(path="a.py", line=10, title="t", body="b", confidence="HIGH", severity="high",)],
    )

    payload = review_submission._review_payload_dict(
        build_payload_for_event(pr, classified,
            event=review_models.ReviewEvent.COMMENT,
            renderers=BUILTIN_RENDERERS,
            run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
        )
    )
    assert payload["commit_id"] == "head123"
    assert payload["event"] == "COMMENT"
    assert payload["comments"][0]["path"] == "a.py"

    body = payload["body"]
    assert "- **Reviewed commit:** [`head123`](https://github.com/acme/widgets/commit/head123)" in body
    assert "**Code Review Summary**" in body
    assert "🧙 Posted by [daydream v" in body
    assert review_identity.DAYDREAM_REPO_URL in body
    assert "**Mode:**" not in body
    assert "**Severity:**" in body and "1 high" in body and "1 low" in body
    assert "**Confidence:**" in body and "1 HIGH" in body and "1 MEDIUM" in body
    assert "Non-inline findings" in body
    assert "b.py" in body
    assert "🔮 Prompt for all review comments" in body
    assert "/beagle-core:" not in body
    assert "repos/acme/widgets/pulls/42/comments" in body
    assert "ℹ️ Review info" in body
    assert "- **Model:**" in body
    assert "- **Cost:**" in body
    assert "- **Tokens:**" in body
    assert "- **Steps / tool calls:**" in body
    assert "<details><summary>Per-phase breakdown</summary>" in body
    assert "| Phase | Model | Tools | Input (cached) | Output | Cost |" in body
    assert body.count("Generated by daydream v") == 1
    assert body.rstrip().endswith("</sub>")


def _classified_with_severity(severity: str, confidence: str, *, body_confidence: str | None = None,
) -> pr_review.ClassifiedIssues:
    return pr_review.ClassifiedIssues(
        inline=[_inline()], body_only=[ParsedIssue(path="b.py", line=None, title="File note", body="desc",
                               confidence=body_confidence or confidence, severity=severity)],
        inline_issues=[ParsedIssue(path="a.py", line=10, title="t", body="b",
                                   confidence=confidence, severity=severity)],
    )


def _approval_payload(pr: PRInfo, classified: pr_review.ClassifiedIssues) -> dict[str, Any]:
    return review_submission._review_payload_dict(
        build_payload_for_event(pr, classified,
            event=(
                review_models.ReviewEvent.APPROVE if pr_review._is_clean_review(classified, True)
                else review_models.ReviewEvent.COMMENT
            ),
            renderers=BUILTIN_RENDERERS,
            run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
        )
    )

def test_build_payload_approves_when_clean_and_enabled(pr: PRInfo) -> None:
    payload = _approval_payload(pr, _classified_with_severity("low", "LOW", body_confidence="MEDIUM"))
    assert payload["event"] == "APPROVE"
    assert "no high/medium findings" in payload["body"]
    assert payload["commit_id"] == pr.head_sha
    assert "**Code Review Summary**" in payload["body"]
    assert payload["body"].index("no high/medium findings") < payload["body"].index("**Code Review Summary**")

@pytest.mark.parametrize("severity", ["high", "medium"])
def test_build_payload_keeps_comment_when_blocking_finding(pr: PRInfo, severity: str) -> None:
    payload = _approval_payload(pr, _classified_with_severity(severity, severity.upper()))
    assert payload["event"] == "COMMENT"
    assert "no high/medium findings" not in payload["body"]

def test_build_payload_none_severity_does_not_crash_on_approve_check(pr: PRInfo,) -> None:
    classified = pr_review.ClassifiedIssues(
        inline=[_inline()], inline_issues=[ParsedIssue(path="a.py", line=10, title="t", body="b", severity=None)],
    )

    payload = _approval_payload(pr, classified)
    assert payload["event"] == "APPROVE"

@pytest.mark.parametrize("off_vocabulary_severity", ["critical", "blocker"])
def test_build_payload_keeps_comment_when_off_vocabulary_severity(pr: PRInfo, off_vocabulary_severity: str,) -> None:
    classified = pr_review.ClassifiedIssues(inline=[_inline()],
        inline_issues=[ParsedIssue(
                path="a.py", line=10, title="t", body="b", confidence="HIGH", severity=off_vocabulary_severity,
            )
        ],
    )

    payload = _approval_payload(pr, classified)
    assert payload["event"] == "COMMENT"
    assert "no high/medium findings" not in payload["body"]

def test_non_blocking_severities_fail_closed() -> None:
    """F1: any severity outside _NON_BLOCKING_SEVERITIES blocks; low and None do not."""
    assert pr_review._NON_BLOCKING_SEVERITIES == frozenset({"low"})
    for off_vocabulary in ("high", "medium", "critical", "blocker", "major", "warning", "info", "INFO", " High ",):
        assert pr_review._severity_blocks_approval(off_vocabulary) is True
    assert pr_review._severity_blocks_approval("low") is False
    assert pr_review._severity_blocks_approval("LOW") is False
    assert pr_review._severity_blocks_approval(None) is False

def test_find_open_pr_returns_none_on_empty_list(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    """Use real branch discovery with an empty gh response and no remote/auth."""
    monkeypatch.setattr(git_ops, "gh_pr_list_for_branch", lambda *_a, **_k: [])
    assert pr_review.find_open_pr(git_repo) is None


def _local_pr_row(repo: Path, *, head_owner: str = "o") -> tuple[dict[str, Any], str, str]:
    base = git_ops.head_sha(repo)
    _git(repo, "checkout", "-b", "feature")
    (repo / "feature.py").write_text("value = 1\n")
    _git(repo, "add", "feature.py")
    _git(repo, "commit", "-m", "feature")
    head = git_ops.head_sha(repo)
    return ({"number": 7, "headRefOid": head, "headRefName": "feature", "baseRefName": "main",
            "url": "https://github.com/o/r/pull/7", "headRepository": {"name": "r", "nameWithOwner": f"{head_owner}/r"},
            "headRepositoryOwner": {"login": head_owner},
        }, base, head,
    )

def test_find_open_pr_returns_pr_info(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    """Real git for branch; gh wrappers stubbed for the PR + repo lookups."""
    row, base, head = _local_pr_row(git_repo)
    monkeypatch.setattr(git_ops, "gh_pr_list_for_branch", lambda *_a, **_k: [row])
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("o", "r"))
    info = pr_review.find_open_pr(git_repo)
    assert info is not None
    assert (info.number, info.head_sha, info.base_sha, info.owner, info.repo) == (7, head, base, "o", "r",)

def test_find_open_pr_captures_head_repo_for_fork_pr(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    """Capture fork identity for commit links while posting to the base repository."""
    row, _, _ = _local_pr_row(git_repo, head_owner="forky")
    row["headRepository"] = {"name": "widgets", "nameWithOwner": "forky/widgets"}
    monkeypatch.setattr(git_ops, "gh_pr_list_for_branch", lambda *_a, **_k: [row])
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("acme", "widgets"),)
    info = pr_review.find_open_pr(git_repo)
    assert info is not None
    assert (info.owner, info.repo) == ("acme", "widgets")
    assert info.head_repo == "forky/widgets"
    assert info.head_ref == "feature"

def test_find_pr_by_number_returns_none_when_pr_missing(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    """An unresolvable PR number short-circuits before the repo lookup."""
    monkeypatch.setattr(git_ops, "gh_pr_view", lambda *_a, **_k: None)
    assert pr_review.find_pr_by_number(git_repo, 7) is None

def test_find_pr_by_number_raises_when_slug_unresolved(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    """A resolvable PR but failed owner/repo lookup is a hard error."""
    row, _, _ = _local_pr_row(git_repo)
    monkeypatch.setattr(git_ops, "gh_pr_view", lambda *_a, **_k: row)

    def fail_slug(_repo: Path, **_kwargs: Any) -> tuple[str, str]:
        raise GitError("gh repo view failed: auth")

    monkeypatch.setattr(git_ops, "gh_repo_view_required", fail_slug)
    with pytest.raises(GitError, match="auth"):
        pr_review.find_pr_by_number(git_repo, 7)

def test_find_pr_by_number_assembles_pr_info(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    row, base, head = _local_pr_row(git_repo, head_owner="forky")
    monkeypatch.setattr(git_ops, "gh_pr_view", lambda *_a, **_k: row)
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("o", "r"))
    info = pr_review.find_pr_by_number(git_repo, 7)
    assert info is not None
    assert (info.number, info.head_sha, info.base_sha, info.base_ref, info.owner, info.repo, info.url, info.head_repo,
    ) == (7, head, base, "main", "o", "r", "https://github.com/o/r/pull/7", "forky/r",)

@pytest.mark.parametrize("lookup", ["branch", "number"])
@pytest.mark.parametrize("include_empty_slug", [False, True])
def test_pr_lookup_and_findings_export_accept_unavailable_head_slug(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path, tmp_path: Path, lookup: str, include_empty_slug: bool,
) -> None:
    """Both gh lookup routes preserve validated head identity through export."""

    row, base, head = _local_pr_row(git_repo)
    row["headRepository"] = {"id": "R_fixture", "name": "shelfspace-mono"}
    if include_empty_slug:
        row["headRepository"]["nameWithOwner"] = ""
    row["headRepositoryOwner"] = {"login": "shelfspace-app"}
    monkeypatch.setattr(git_ops, "gh_pr_list_for_branch", lambda *_a, **_k: [row])
    monkeypatch.setattr(git_ops, "gh_pr_view", lambda *_a, **_k: row)
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda *_a, **_k: ("o", "r"))
    info = (pr_review.find_open_pr(git_repo) if lookup == "branch" else pr_review.find_pr_by_number(git_repo, 7))
    assert info is not None
    assert (info.head_repo, info.base_sha, info.head_sha) == ("shelfspace-app/shelfspace-mono", base, head)

    output = tmp_path / "findings.json"
    config = RunConfig(findings_out=str(output), pr_number=7 if lookup == "number" else None)
    assert _emit_findings_from_items(
        git_repo, config, [], run_info="", renderers=BUILTIN_RENDERERS, terminal_result=terminal_result(head_sha=head),
        captured_pr=info, snapshot_diff=git_ops.diff_paths(git_repo, base, head, ["."]),
    ) == 0
    artifact = load_findings_artifact(output, expected_repo="o/r", expected_pr_number=7, expected_head_sha=head,)
    assert artifact.findings == []

    # Captured export identity still requires an available PR head.
    output.unlink()
    row["headRefOid"] = "f" * 40
    with pytest.raises(GitError, match="bad object"):
        _emit_findings_from_items(
            git_repo, config, [], run_info="", renderers=BUILTIN_RENDERERS,
            terminal_result=terminal_result(head_sha="f" * 40), captured_pr=replace(info, head_sha="f" * 40),
            snapshot_diff=git_ops.diff_paths(git_repo, base, head, ["."]),
        )
    assert not output.exists()

@pytest.mark.parametrize("value", [None, False, 7, " ", "fork/r/extra", "fork/", "/r"])
def test_head_slug_fallback_does_not_mask_invalid_present_value(value: Any) -> None:
    row = {"headRepository": {"name": "r", "nameWithOwner": value}, "headRepositoryOwner": {"login": "fork"}}
    with pytest.raises(GitError, match="invalid PR row"):
        review_lookup._head_repo_slug_from_row(row)

@pytest.mark.parametrize("component", ["name", "login"])
@pytest.mark.parametrize("value", [None, False, 7, "", " ", "a/b", "a\nb"])
def test_head_slug_fallback_rejects_invalid_components(component: str, value: Any) -> None:
    row = {"headRepository": {"name": "r", "nameWithOwner": ""}, "headRepositoryOwner": {"login": "fork"}}
    row["headRepository" if component == "name" else "headRepositoryOwner"][component] = value
    with pytest.raises(GitError, match="invalid PR row"):
        review_lookup._head_repo_slug_from_row(row)

@pytest.mark.parametrize("slug", ["other/r", "fork/other"])
def test_head_slug_rejects_contradictory_valid_identity(slug: str) -> None:
    with pytest.raises(GitError, match="contradictory head repository identity") as error:
        review_lookup._head_repo_slug_from_row({
            "headRepository": {"name": "r", "nameWithOwner": slug, "id": "private-value"},
            "headRepositoryOwner": {"login": "fork"},
        })
    assert "private-value" not in str(error.value)
    assert slug not in str(error.value)

def test_head_slug_allows_case_differences() -> None:
    assert review_lookup._head_repo_slug_from_row({
        "headRepository": {"name": "Widgets", "nameWithOwner": "FORK/widgets"},
        "headRepositoryOwner": {"login": "fork"},
    }) == "FORK/widgets"

@pytest.mark.parametrize(("field", "value"),
    [("number", True), ("number", 0), ("headRefOid", "HEAD"), ("headRefOid", "deadbeef"), ("headRefName", ""),
        ("headRefName", 7), ("headRefName", "feature~1"), ("baseRefName", ""), ("baseRefName", "main~1"), ("url", ""),
        ("url", 7), ("headRepository", "fork/r"), ("headRepository", {"nameWithOwner": "fork/r/extra"}),
        ("headRepository", {}), ("headRepository", {"nameWithOwner": 7, "name": "r"}), ("headRepositoryOwner", {}),
        ("headRepositoryOwner", {"login": 7}), ("headRepositoryOwner", {"login": ""}), ("headRepositoryOwner", "fork"),
    ],
)
def test_pr_info_rejects_malformed_row_fields(monkeypatch: pytest.MonkeyPatch, git_repo: Path, field: str, value: Any,
) -> None:
    row, _, _ = _local_pr_row(git_repo)
    row[field] = value
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("o", "r"))
    with pytest.raises(GitError, match="invalid PR row|base branch|exact PR head"):
        review_lookup._pr_info_from_row(git_repo, row)

@pytest.mark.parametrize("missing_field", ["headRepository", "headRepositoryOwner"])
def test_pr_info_rejects_missing_requested_head_metadata(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path, missing_field: str,
) -> None:
    row, _, _ = _local_pr_row(git_repo)
    del row[missing_field]
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("o", "r"))
    with pytest.raises(GitError, match="invalid PR row"):
        review_lookup._pr_info_from_row(git_repo, row)

def test_pr_info_rejects_malformed_owner_even_with_null_head_repository(monkeypatch: pytest.MonkeyPatch, git_repo: Path,
) -> None:
    row, _, _ = _local_pr_row(git_repo)
    row["headRepository"] = None
    row["headRepositoryOwner"] = {"login": 7}
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("o", "r"))
    with pytest.raises(GitError, match="invalid PR row"):
        review_lookup._pr_info_from_row(git_repo, row)

@pytest.mark.parametrize("head_owner", [None, {"login": "former-owner"}])
def test_pr_info_accepts_null_same_repo_head_metadata(monkeypatch: pytest.MonkeyPatch, git_repo: Path, head_owner: Any,
) -> None:
    row, _, _ = _local_pr_row(git_repo)
    row["headRepository"] = None
    row["headRepositoryOwner"] = head_owner
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("o", "r"))

    assert review_lookup._pr_info_from_row(git_repo, row).head_repo is None

def test_find_open_pr_propagates_current_branch_failure(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    def fail_branch(_repo: Path) -> str | None:
        raise GitError("cannot read current branch")

    monkeypatch.setattr(git_ops, "current_branch", fail_branch)
    with pytest.raises(GitError, match="cannot read current branch"):
        pr_review.find_open_pr(git_repo)

@pytest.mark.parametrize("lookup", ["branch", "number"])
def test_fork_pr_uses_upstream_base_for_both_lookup_paths(monkeypatch: pytest.MonkeyPatch, git_repo: Path, lookup: str
) -> None:
    row, base, _ = _local_pr_row(git_repo, head_owner="forky")
    _git(git_repo, "remote", "add", "origin", "https://github.com/forky/r.git")
    _git(git_repo, "remote", "add", "upstream", "https://github.com/acme/widgets.git")
    _git(git_repo, "update-ref", "refs/remotes/upstream/main", base)
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("acme", "widgets"),)
    monkeypatch.setattr(git_ops, "gh_pr_list_for_branch", lambda *_a, **_k: [row])
    monkeypatch.setattr(git_ops, "gh_pr_view", lambda *_a, **_k: row)

    info = (pr_review.find_open_pr(git_repo) if lookup == "branch" else pr_review.find_pr_by_number(git_repo, 7))
    assert info is not None
    assert info.base_sha == base
    assert info.head_repo == "forky/r"

def test_pr_base_remote_matching_is_credential_safe(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    row, base, _ = _local_pr_row(git_repo)
    tree = _git(git_repo, "write-tree")
    unrelated = _git(git_repo, "commit-tree", tree, "-m", "unrelated base")
    _git(git_repo, "remote", "add", "origin", "https://user:top-secret@github.com/acme/widgets.git?token=private",)
    _git(git_repo, "update-ref", "refs/remotes/origin/main", unrelated)
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("acme", "widgets"),)

    with pytest.raises(GitError) as excinfo:
        review_lookup._pr_info_from_row(git_repo, row)
    assert "top-secret" not in str(excinfo.value)
    assert "token=private" not in str(excinfo.value)

    _git(git_repo, "remote", "set-url", "origin", "https://user:fork-secret@github.com/forky/widgets.git?token=fork",)
    assert review_lookup._pr_info_from_row(git_repo, row).base_sha == base


class _FakeConsole:
    def print(self, *_a: Any, **_k: Any) -> None:
        pass


def _assumed_context(answer: str) -> RunContext:
    return RunContext(InteractionPolicy(assume=answer))

@pytest.mark.asyncio
async def test_post_skips_when_no_pr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td, **_kwargs: None)
    warnings: list[str] = []
    monkeypatch.setattr(pr_review, "print_warning", lambda _c, msg: warnings.append(msg),)
    status = await pr_review._post(tmp_path, [ParsedIssue(path="x.py", line=1, title="t", body="b")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        renderers=BUILTIN_RENDERERS, run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert warnings and "No open PR" in warnings[0]
    assert status == pr_review.PostStatus.NO_PR

@pytest.mark.asyncio
async def test_post_fails_with_safe_diagnostic_when_pr_lookup_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_lookup(_target_dir: Path, **_kwargs: Any) -> PRInfo | None:
        raise GitError("gh pr list failed: authentication required")

    monkeypatch.setattr(pr_review, "find_open_pr", fail_lookup)
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(pr_review, "print_error", lambda _console, title, message: errors.append((title, message)),)

    status = await pr_review._post(tmp_path, [ParsedIssue(path="x.py", line=1, title="t", body="b")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        renderers=BUILTIN_RENDERERS, run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )

    assert status == pr_review.PostStatus.FAILED
    assert errors == [("PR Lookup Failed", "gh pr list failed: authentication required")]

@pytest.mark.asyncio
async def test_post_succeeds_and_prints_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pr: PRInfo) -> None:
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td, **_kwargs: pr)
    monkeypatch.setattr(
        pr_review, "classify", lambda *_a, **_k: pr_review.ClassifiedIssues(inline=[_inline(line=1)], body_only=[],),
    )
    captured: dict[str, Any] = {}
    fake_submit = _recording_fake_submit(captured)
    monkeypatch.setattr(pr_review, "post_classified_review", fake_submit)
    successes: list[str] = []
    monkeypatch.setattr(pr_review, "print_success", lambda _c, msg: successes.append(msg),)
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)

    status = await pr_review._post(tmp_path, [ParsedIssue(path="a.py", line=1, title="t", body="b")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        run_context=_assumed_context("yes"), renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert captured["pr"].head_sha == pr.head_sha
    assert captured["event"] is pr_review.ReviewEvent.COMMENT
    assert successes and "pullrequestreview" in successes[0]
    assert status == pr_review.PostStatus.POSTED

@pytest.mark.asyncio
async def test_post_payload_approves_when_clean_and_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pr: PRInfo,
) -> None:
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td, **_kwargs: pr)
    monkeypatch.setattr(pr_review, "classify",
        lambda *_a, **_k: pr_review.ClassifiedIssues(inline=[_inline(line=1)],
            inline_issues=[ParsedIssue(path="a.py", line=1, title="t", body="b", confidence="LOW", severity="low",)],
        ),
    )
    captured: dict[str, Any] = {}
    fake_submit = _recording_fake_submit(captured)
    monkeypatch.setattr(pr_review, "post_classified_review", fake_submit)
    monkeypatch.setattr(pr_review, "print_success", lambda *_a, **_k: None)
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)

    status = await pr_review._post(tmp_path, [ParsedIssue(path="a.py", line=1, title="t", body="b", severity="low")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        approve_on_clean=True, run_context=_assumed_context("yes"), renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert captured["event"] is pr_review.ReviewEvent.APPROVE
    assert status == pr_review.PostStatus.POSTED

@pytest.mark.asyncio
async def test_post_warns_with_preserved_payload_path_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pr: PRInfo,
) -> None:
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td, **_kwargs: pr)
    monkeypatch.setattr(
        pr_review, "classify", lambda *_a, **_k: pr_review.ClassifiedIssues(inline=[_inline(line=1)], body_only=[],),
    )
    err = "GitHub review submission failed (request payload preserved at /tmp/x.json)"
    monkeypatch.setattr(pr_review, "post_classified_review",
        lambda _pr, _classified, **_kwargs: review_models.ClassifiedReviewResult(
            status=pr_review.SubmissionStatus.FAILED, review_url=None, posted_file_level=(), folded_file_level=(),
            final_review_posted=False, safe_error=err,
        ),
    )
    warnings: list[str] = []
    monkeypatch.setattr(pr_review, "print_warning", lambda _c, msg: warnings.append(msg),)
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)

    status = await pr_review._post(tmp_path, [ParsedIssue(path="a.py", line=1, title="t", body="b")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        run_context=_assumed_context("yes"), renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert warnings
    assert "no comments were posted" in warnings[0].lower()
    # The structured safe error preserves the request payload path.
    assert "payload preserved at /tmp/x.json" in warnings[0]
    assert status == pr_review.PostStatus.FAILED

def test_github_transport_surfaces_only_structured_preserved_payload_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pr: PRInfo,
) -> None:
    error = GitError("remote response leaked secret=never-show-this")
    error.preserved_payload_path = Path("/tmp/review.json")

    def fail_gh(*_args: Any, **_kwargs: Any) -> object:
        raise error

    monkeypatch.setattr(git_ops, "gh_api", fail_gh)

    result = pr_review.GitHubReviewTransport(target_dir=tmp_path, auth=git_ops.INHERIT_GITHUB_AUTH,).post_review(pr,
        review_models.ReviewPayload(
            event=pr_review.ReviewEvent.COMMENT, commit_id=pr.head_sha, body="review body", comments=(),
        ),
    )

    assert result.review_url is None
    assert result.safe_error == "GitHub review submission failed (request payload preserved at /tmp/review.json)"
    assert "secret" not in result.safe_error

@pytest.mark.asyncio
async def test_post_skipped_when_user_declines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pr: PRInfo) -> None:
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td, **_kwargs: pr)
    monkeypatch.setattr(
        pr_review, "classify", lambda *_a, **_k: pr_review.ClassifiedIssues(inline=[_inline(line=1)], body_only=[],),
    )
    submit_called = False

    def fake_submit(_pr: PRInfo, _classified: pr_review.ClassifiedIssues, **_kwargs: Any
    ) -> review_models.ClassifiedReviewResult:
        nonlocal submit_called
        submit_called = True
        return review_models.ClassifiedReviewResult(
            status=pr_review.SubmissionStatus.POSTED, review_url="x", posted_file_level=(), folded_file_level=(),
            final_review_posted=True, safe_error=None,
        )

    monkeypatch.setattr(pr_review, "post_classified_review", fake_submit)
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)

    status = await pr_review._post(tmp_path, [ParsedIssue(path="a.py", line=1, title="t", body="b")],
        console=_FakeConsole(),  # type: ignore[arg-type]
        run_context=_assumed_context("no"), renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert not submit_called
    assert status == pr_review.PostStatus.NOTHING_TO_POST

@pytest.mark.asyncio
async def test_post_review_from_report_empty_items_is_nothing_to_post(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Empty merged items skip the post cleanly (NOTHING_TO_POST, not a failure)."""
    merged = tmp_path / "merged-items.json"
    merged.write_text(json.dumps({"items": []}))
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)

    status = await pr_review.post_review_to_pr_from_report(tmp_path, merged,
        console=_FakeConsole(),  # type: ignore[arg-type]
        renderers=BUILTIN_RENDERERS, run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    assert status == pr_review.PostStatus.NOTHING_TO_POST

@pytest.mark.asyncio
async def test_post_review_from_report_empty_items_posts_diagram(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pr: PRInfo,
) -> None:
    merged = tmp_path / "merged-items.json"
    merged.write_text(json.dumps({"items": []}))
    blocks = "<details><summary><h3>Flowchart</h3></summary>\nX\n</details>"
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td, **_kwargs: pr)
    monkeypatch.setattr(pr_review, "classify", lambda *_a, **_k: pr_review.ClassifiedIssues())
    captured: dict[str, Any] = {}
    fake_submit = _recording_fake_submit(captured)
    monkeypatch.setattr(pr_review, "post_classified_review", fake_submit)
    monkeypatch.setattr(pr_review, "print_success", lambda *_a, **_k: None)
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)

    status = await pr_review.post_review_to_pr_from_report(tmp_path, merged,
        console=_FakeConsole(),  # type: ignore[arg-type]
        post=True, diagram_blocks=blocks, renderers=BUILTIN_RENDERERS,
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )

    assert status == pr_review.PostStatus.POSTED
    assert captured["event"] is pr_review.ReviewEvent.COMMENT
    assert captured["diagram_blocks"] == blocks

@pytest.mark.asyncio
@pytest.mark.parametrize("failed_reviewer", [False, True], ids=["phase-budget", "provider-failure"])
async def test_incomplete_live_review_posts_even_without_findings_and_cannot_approve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pr: PRInfo, failed_reviewer: bool,
) -> None:
    merged = tmp_path / "merged-items.json"
    merged.write_text(json.dumps({"items": []}))
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td, **_kwargs: pr)
    monkeypatch.setattr(pr_review, "classify", lambda *_a, **_k: pr_review.ClassifiedIssues())
    captured: dict[str, Any] = {}
    monkeypatch.setattr(pr_review, "post_classified_review", _recording_fake_submit(captured))
    warnings: tuple[str, ...] = ("Alternatives: wall_budget_exceeded",)
    if failed_reviewer:
        coverage = review_coverage(scope_ids=("python",), phases=())
        coverage.record_scope("python", "failed", reasons=("backend_failure",),
                              diagnostic="RuntimeError: provider unavailable")
        DeepArtifact.REVIEW_COVERAGE.at(tmp_path).write_text(json.dumps(coverage.to_dict()))
        warnings = review_warnings(tmp_path)
    status = await pr_review.post_review_to_pr_from_report(
        tmp_path, merged, console=_FakeConsole(),  # type: ignore[arg-type]
        post=True, approve_on_clean=True, review_warnings=warnings, renderers=BUILTIN_RENDERERS, run_info="test run",
    )
    assert status == pr_review.PostStatus.POSTED
    assert captured["event"] is pr_review.ReviewEvent.COMMENT
    assert captured["review_warnings"] == warnings


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
    assert pr_review.resolve_line(git_ops.show(git_repo, sha, issue.path).decode().splitlines(), issue) == 10

def test_resolve_line_full_search_when_hint_bad(git_repo: Path) -> None:
    """Hint points to line 2, but the anchor is at line 15 -- full-file search wins."""
    text = "\n".join(f"row_{i}" for i in range(1, 21)) + "\n"
    sha = _commit_file(git_repo, "x.py", text, "add x.py")
    issue = ParsedIssue(path="x.py", line=2, title="t", body="`row_15`")
    assert pr_review.resolve_line(git_ops.show(git_repo, sha, issue.path).decode().splitlines(), issue) == 15

def test_resolve_line_none_for_empty_content() -> None:
    issue = ParsedIssue(path="gone.py", line=1, title="t", body="b")
    assert pr_review.resolve_line([], issue) is None


# Valid in-hunk locations must survive prose matching: an earlier anchor outside
# the diff would otherwise move the finding and cause snap_to_hunk to drop it.

# The finding from the issue's reproduction: a prose-heavy rationale whose one
# code identifier (`ttl`) is short enough that longest-first ranking cuts it.
_PROSE_HEAVY_ISSUE = ParsedIssue(path="cache.yaml", line=12, title="Expiration collapses from an hour to a minute",
    body=("The new override sets `ttl` far below every other environment, so cached "
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
    return PRInfo(number=7, head_sha=head_sha, base_sha=base_sha, base_ref="main", head_ref="feature", owner="acme",
        repo="widgets", url="https://github.com/acme/widgets/pull/7",
    )

def test_extract_anchors_prefer_quoted_keeps_short_backticked_identifier() -> None:
    """Quoted-first ranking keeps a 3-char identifier that prose would crowd out."""
    text = f"{_PROSE_HEAVY_ISSUE.title}\n{_PROSE_HEAVY_ISSUE.body}"
    assert extract_anchors(text, prefer_quoted=True)[0] == "ttl"
    # Bare words still rank longest-first behind the quoted tokens.
    bare = extract_anchors(text, prefer_quoted=True)[1:]
    assert bare == sorted(bare, key=len, reverse=True)

def test_extract_anchors_default_ordering_stays_frozen_for_fingerprints() -> None:
    """Default anchor ordering determines fingerprints and must remain stable.

    Rationale prose must still crowd ttl out of the cap; re-ranking would
    re-identify existing findings and defeat reconciliation."""
    text = f"{_PROSE_HEAVY_ISSUE.title}\n{_PROSE_HEAVY_ISSUE.body}"
    anchors = extract_anchors(text)
    assert anchors == [
        "environment", "unavailable", "immediately", "Expiration", "downstream", "additional", "collapses", "override",
    ]
    assert "ttl" not in anchors

def test_resolve_line_trusts_in_hunk_hint_without_any_anchor_match(git_repo: Path) -> None:
    text = "\n".join(f"row_{i}" for i in range(1, 21)) + "\n"
    sha = _commit_file(git_repo, "x.py", text, "add x.py")
    issue = ParsedIssue(path="x.py", line=10, title="Latency regression", body="prose only")
    # No anchor from the title/body occurs anywhere in the file.
    assert pr_review.resolve_line(git_ops.show(git_repo, sha, issue.path).decode().splitlines(), issue) is None
    assert pr_review.resolve_line(git_ops.show(git_repo, sha, issue.path).decode().splitlines(), issue, [(8, 12)]) == 10

def test_resolve_line_prefers_in_hunk_anchor_hit_over_first_file_hit(git_repo: Path) -> None:
    lines = [f"row_{i}" for i in range(1, 21)]
    lines[1] = "marker_token  # pre-existing, unchanged"
    lines[17] = "marker_token  # the changed line"
    sha = _commit_file(git_repo, "x.py", "\n".join(lines) + "\n", "add x.py")
    issue = ParsedIssue(path="x.py", line=None, title="t", body="`marker_token`")
    # Without hunks the first hit (line 2) wins, as before.
    assert pr_review.resolve_line(git_ops.show(git_repo, sha, issue.path).decode().splitlines(), issue) == 2
    # With hunks, the in-hunk hit (line 18) wins over the earlier out-of-hunk one.
    captured_lines = git_ops.show(git_repo, sha, issue.path).decode().splitlines()
    assert pr_review.resolve_line(captured_lines, issue, [(16, 20)]) == 18

def test_resolve_line_returns_out_of_hunk_hit_when_no_in_hunk_candidate(git_repo: Path,) -> None:
    lines = [f"row_{i}" for i in range(1, 21)]
    lines[1] = "marker_token  # pre-existing, unchanged"
    sha = _commit_file(git_repo, "x.py", "\n".join(lines) + "\n", "add x.py")
    issue = ParsedIssue(path="x.py", line=None, title="t", body="`marker_token`")
    assert pr_review.resolve_line(git_ops.show(git_repo, sha, issue.path).decode().splitlines(), issue, [(16, 20)]) == 2

def test_classify_keeps_prose_heavy_in_hunk_finding_inline(git_repo: Path) -> None:
    """A valid line-12 citation survives prose-only anchors in a real Git diff.

    Searching those anchors would relocate it to line 1, outside snap tolerance,
    incorrectly converting an inline finding into a body-only finding."""
    base = _commit_file(git_repo, "cache.yaml", _CACHE_YAML_BASE, "add cache.yaml")
    head = _commit_file(git_repo, "cache.yaml", _CACHE_YAML_HEAD, "cut prod ttl")
    issue = replace(_PROSE_HEAVY_ISSUE)

    result = classify(git_repo, _pr_for(base, head), [issue])

    assert [c.line for c in result.inline] == [12], (f"in-hunk citation was not posted on its own line; "
        f"file_level={[i.path for i in result.file_level]} "
        f"body_only={[i.path for i in result.body_only]}"
    )
    assert result.inline[0].path == "cache.yaml"
    assert not result.file_level
    assert not result.body_only
    assert "**Placement:**" not in result.inline[0].body

def test_classify_annotates_relocated_line(git_repo: Path) -> None:
    lines = [f"row_{i}" for i in range(1, 31)]
    lines[14] = "settle_window = 5  # cited here"
    base = _commit_file(git_repo, "x.py", "\n".join(lines) + "\n", "add x.py")
    lines[19] = "row_20_changed"
    head = _commit_file(git_repo, "x.py", "\n".join(lines) + "\n", "change row 20")
    issue = ParsedIssue(path="x.py", line=15, title="t", body="`settle_window` is too small")

    result = classify(git_repo, _pr_for(base, head), [issue])

    assert [c.line for c in result.inline] == [17]
    note = "**Placement:** posted on line 17; reviewer cited line 15."
    assert note in issue.body, "the relocation was not recorded on the finding"
    assert note in result.inline[0].body, "the posted comment does not show the relocation"
    classify(git_repo, _pr_for(base, head), [issue])
    assert issue.body.count("**Placement:**") == 1

def test_classify_skips_hunk_read_for_path_missing_at_head(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    """Reject absent head paths before reading hunks or invoking gh fallback."""
    base = _commit_file(git_repo, "gone.py", "x = 1\n", "add gone.py")
    _git(git_repo, "rm", "gone.py")
    _git(git_repo, "commit", "-m", "delete gone.py")
    head = _git(git_repo, "rev-parse", "HEAD")
    issue = ParsedIssue(path="gone.py", line=1, title="t", body="`x`")

    def fail_if_called(*_a: Any, **_k: Any) -> list[tuple[int, int]]:
        raise AssertionError("Git diff should not be read for a path missing at head_sha")

    monkeypatch.setattr(git_ops, "diff_paths", fail_if_called)

    result = classify(git_repo, _pr_for(base, head), [issue])

    assert not result.inline
    assert [i.path for i in result.file_level] == ["gone.py"]


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

def test_classify_uses_git_diff_when_it_succeeds(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    base = _commit_file(git_repo, "x.py", "\n".join(f"line {i}" for i in range(1, 30)) + "\n", "baseline")
    lines = [f"line {i}" for i in range(1, 30)]
    lines[19:19] = ["NEW1", "NEW2"]
    head = _commit_file(git_repo, "x.py", "\n".join(lines) + "\n", "add two lines")
    monkeypatch.setattr(git_ops, "gh_pr_diff", _raise_on_gh_fallback)
    issue = ParsedIssue("x.py", 20, "new line", "`NEW1`")
    out = classify(git_repo, _pr_for(base, head), [issue])
    assert [c.line for c in out.inline] == [20]


def test_classify_falls_back_to_gh_when_base_unreachable(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    head = _commit_file(git_repo, "x.py", "\n".join(f"line {i}" for i in range(1, 30)) + "\n", "baseline")
    monkeypatch.setattr(git_ops, "gh_pr_diff", lambda _r, _n, **_kwargs: _GH_PR_DIFF)
    out = classify(git_repo, _pr_for("deadbeef" * 5, head), [
        ParsedIssue("x.py", 11, "valid", "prose only"),
        ParsedIssue("x.py", 1, "foreign hunk", "prose only"),
    ])
    assert [c.line for c in out.inline] == [11]
    assert [i.title for i in out.file_level] == ["foreign hunk"]


def test_classify_snapshot_does_not_fall_back_to_gh(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    base = _commit_file(git_repo, "x.py", "line 1\n", "baseline")
    head = _commit_file(git_repo, "x.py", "line 2\n", "change")
    monkeypatch.setattr(git_ops, "gh_pr_diff", _raise_on_gh_fallback)
    monkeypatch.setattr(git_ops, "diff_paths", _raise_on_gh_fallback)
    out = classify(git_repo, _pr_for(base, head), [ParsedIssue("x.py", 1, "t", "`line`")], snapshot_diff="")
    assert not out.inline
    assert [i.path for i in out.file_level] == ["x.py"]


def test_classify_gh_fallback_handles_subprocess_error(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    head = _commit_file(git_repo, "x.py", "line 1\n", "baseline")
    def raise_git_error(*_a: Any, **_k: Any) -> str:
        raise git_ops.GitError("gh blew up")
    monkeypatch.setattr(git_ops, "gh_pr_diff", raise_git_error)
    out = classify(git_repo, _pr_for("deadbeef" * 5, head), [ParsedIssue("x.py", 1, "t", "`line`")])
    assert not out.inline
    assert [i.path for i in out.body_only] == ["x.py"]


def test_demoted_high_finding_still_blocks_approval(pr: PRInfo) -> None:
    classified = pr_review.ClassifiedIssues(inline=[_inline()],
        inline_issues=[ParsedIssue(path="a.py", line=10, title="t", body="b", severity="low", location_distrust=True,
                severity_before_demotion="high",
            )
        ],
    )
    payload = _approval_payload(pr, classified)
    assert payload["event"] == "COMMENT"

def test_demoted_low_finding_does_not_block_approval(pr: PRInfo) -> None:
    """Location distrust alone does not block originally low or unasserted severity."""
    for before in ("low", None):
        classified = pr_review.ClassifiedIssues(inline=[_inline()],
            inline_issues=[ParsedIssue(
                    path="a.py", line=10, title="t", body="b", severity="low", location_distrust=True,
                    severity_before_demotion=before,
                )
            ],
        )
        payload = review_submission._review_payload_dict(
        build_payload_for_event(pr, classified,
            event=(
                review_models.ReviewEvent.APPROVE if pr_review._is_clean_review(classified, True)
                else review_models.ReviewEvent.COMMENT
            ),
            renderers=BUILTIN_RENDERERS,
            run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
        )
    )
        assert payload["event"] == "APPROVE"

def test_demoted_low_finding_approval_gate_does_not_block() -> None:
    assert pr_review._finding_blocks_approval("low", True, False, "low") is False
    assert pr_review._finding_blocks_approval("low", True, False, None) is False
    assert pr_review._finding_blocks_approval("low", True, False, "high") is True
    assert pr_review._finding_blocks_approval("low", True, False, "medium") is True

def test_parsed_issues_carry_location_distrust_and_report_note() -> None:
    items = [{"file": "a.py", "line": 10, "description": "off-citation", "rationale": "r", "severity": "low",
            "confidence": "LOW", "severity_before_demotion": "high", "location_distrust": True,
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

def test_null_severity_does_not_block_approval(pr: PRInfo) -> None:
    # SUPERVISE_SCHEMA emits severity: null — must approve like omitted.
    classified = pr_review.ClassifiedIssues(
        inline=[_inline()], inline_issues=[ParsedIssue(path="a.py", line=10, title="t", body="b", severity=None)],
    )

    payload = _approval_payload(pr, classified)
    assert payload["event"] == "APPROVE"
    assert "**Severity:** none" not in payload["body"]  # no phantom label rendered

def test_artifact_off_vocabulary_severity_blocks_approval() -> None:
    """Unknown raw severity still blocks after schema normalization produces None.

    The artifact retains severity_off_vocabulary for the posting gate."""
    finding = ArtifactFinding(fingerprint="f" * 64, path="a.py", line=10, placement="inline", title="t", body="b",
        severity=None,  # "critical" was folded to None by Phase A normalization
        confidence="HIGH", is_cross_stack=False, severity_off_vocabulary=True,
    )
    issue = pr_review._issue_from_artifact_finding(finding)
    assert issue.severity is None
    assert issue.severity_off_vocabulary is True
    # The approval gate blocks on the off-vocabulary signal even with severity None.
    assert pr_review._finding_blocks_approval(issue.severity, issue.location_distrust, issue.severity_off_vocabulary
    ) is True

def test_artifact_folding_to_none_not_off_vocabulary_does_not_block() -> None:
    """Explicit null severity is unasserted, not an unknown label that blocks approval."""
    finding = ArtifactFinding(
        fingerprint="f" * 64, path="a.py", line=10, placement="inline", title="t", body="b", severity=None,
        confidence="HIGH", is_cross_stack=False, severity_off_vocabulary=False,
    )
    issue = pr_review._issue_from_artifact_finding(finding)
    assert pr_review._finding_blocks_approval(issue.severity, issue.location_distrust, issue.severity_off_vocabulary
    ) is False


def test_diagram_marker_round_trip() -> None:
    """The hidden marker is invisible in rendered markdown and parses back exactly."""

    body = "\n\n".join(
        [diagram_marker("sequence", "a" * 40), diagram_marker("flowchart", "a" * 40),
            "<details>the diagrams</details>",
        ]
    )
    assert parse_diagram_markers(body) == [("sequence", "a" * 40), ("flowchart", "a" * 40)]
    # A finding marker is a different namespace and must not cross-parse.
    assert parse_diagram_markers(review_identity.finding_marker("f" * 64)) == []
    assert review_identity.parse_finding_markers(diagram_marker("sequence", "a" * 40)) == []


def _record_minimize(calls: list[tuple[str, str | None]]) -> Any:
    def minimize(_target: Path, node_id: str, **_kwargs: Any) -> bool:
        calls.append(("minimize", node_id))
        return True

    return minimize

def test_diagram_replacement_post_failure_keeps_prior_comment(
    pr: PRInfo, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr("daydream.reconcile.fetch_prior_diagram_comments",
        lambda *_a, **_k: [PriorDiagramComment("IC_prior", ("sequence",))],
    )

    monkeypatch.setattr("daydream.reconcile.minimize_comment", _record_minimize(calls),)

    def fail_post(*_args: Any, **_kwargs: Any) -> Any:
        calls.append(("post", None))
        raise GitError("simulated POST failure")

    monkeypatch.setattr(git_ops, "gh_api", fail_post)

    result = review_diagrams.post_diagram_comment_to_pr(
        tmp_path, pr, body="diagram", kinds=["sequence"], bot_login="daydream",
    )

    assert result == (None, "simulated POST failure")
    assert calls == [("post", None)]

def test_diagram_replacement_posts_before_minimizing_matching_prior_comment(
    pr: PRInfo, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr("daydream.reconcile.fetch_prior_diagram_comments",
        lambda *_a, **_k: [
            PriorDiagramComment("IC_matching", ("sequence",)), PriorDiagramComment("IC_other_kind", ("flowchart",)),
        ],
    )

    monkeypatch.setattr("daydream.reconcile.minimize_comment", _record_minimize(calls),)

    def post(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        calls.append(("post", None))
        return {"html_url": "https://github.com/acme/widgets/issues/42#issuecomment-1"}

    monkeypatch.setattr(git_ops, "gh_api", post)

    result = review_diagrams.post_diagram_comment_to_pr(
        tmp_path, pr, body="diagram", kinds=["sequence"], bot_login="daydream",
    )

    assert result == ("https://github.com/acme/widgets/issues/42#issuecomment-1", None,)
    assert calls == [("post", None), ("minimize", "IC_matching")]

def test_build_payload_places_diagram_blocks_under_the_header(pr: PRInfo) -> None:
    classified = pr_review.ClassifiedIssues(
        body_only=[ParsedIssue(path="b.py", line=None, title="File note", body="desc", fingerprint="b" * 64,)]
    )
    blocks = "<details><summary><h3>Sequence Diagram</h3></summary>\nX\n</details>"

    payload = review_submission._review_payload_dict(
        build_payload_for_event(pr, classified,
            event=review_models.ReviewEvent.COMMENT,
            diagram_blocks=blocks,
            renderers=BUILTIN_RENDERERS,
            run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
        )
    )
    body = payload["body"]
    header = "**Code Review Summary**"
    assert body[body.index(header) + len(header) :].lstrip().startswith(blocks)
    # Absent (the default) is byte-identical to the pre-#1113 body.
    assert review_submission._review_payload_dict(
        build_payload_for_event(pr, classified,
            event=review_models.ReviewEvent.COMMENT,
            renderers=BUILTIN_RENDERERS,
            run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
        )
    )["body"] == body.replace(f"{header}\n\n{blocks}", header)

def test_custom_summary_renderer_receives_and_may_drop_diagrams(pr: PRInfo) -> None:
    """Custom renderers own ctx.diagrams; the host cannot restore blocks they omit."""

    seen: list[str | None] = []

    def keeps(ctx: Any) -> str:
        seen.append(ctx.diagrams)
        return f"**Custom**\n\n{ctx.diagrams}"

    def drops(ctx: Any) -> str:
        seen.append(ctx.diagrams)
        return "**Custom**"

    classified = pr_review.ClassifiedIssues()
    blocks = "<details><summary><h3>Flowchart</h3></summary>\nX\n</details>"
    for renderer, expect_present in ((keeps, True), (drops, False)):
        reg = Registry()
        register_builtins(reg)
        reg.override_renderer("summary", renderer)
        body = review_submission._review_payload_dict(
        build_payload_for_event(pr, classified,
            event=review_models.ReviewEvent.COMMENT,
            diagram_blocks=blocks,
            renderers=resolve_review_renderers(reg),
            run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
        )
    )["body"]
        assert (blocks in body) is expect_present
    assert seen == [blocks, blocks]

def test_explicit_run_info_payload_is_byte_stable(pr: PRInfo) -> None:
    """Pin the host envelope, approval, rollup, markers and diagram placement."""
    classified = pr_review.ClassifiedIssues(body_only=[ParsedIssue(
                path="a.py", line=None, title="Fixture note", body="Fixture rationale", severity="low",
                confidence="HIGH", fingerprint="a" * 64,
            )
        ]
    )
    payload = review_submission._review_payload_dict(
        build_payload_for_event(pr, classified,
            event=(
                review_models.ReviewEvent.APPROVE if pr_review._is_clean_review(classified, True)
                else review_models.ReviewEvent.COMMENT
            ),
            run_info="Fixture run info",
            diagram_blocks="<details>Fixture diagram</details>",
            renderers=BUILTIN_RENDERERS,
        )
    )
    payload["body"] = payload["body"].replace(DAYDREAM_FOOTER, "<DAYDREAM_FOOTER>")
    assert payload == json.loads((SNAP / "explicit_payload.json").read_text())

def test_payload_uses_only_explicit_renderers_and_run_info(pr: PRInfo, monkeypatch: pytest.MonkeyPatch,) -> None:

    seen: list[SummaryContext] = []

    def summary(ctx: SummaryContext) -> str:
        seen.append(ctx)
        return default_render_summary(ctx)

    registry = Registry()
    register_builtins(registry)
    registry.override_renderer("summary", summary)
    renderers = resolve_review_renderers(registry)
    classified = pr_review.ClassifiedIssues(body_only=[ParsedIssue(
        path="a.py", line=None, title="Explicit finding", body="Explanation", severity="low", confidence="HIGH",
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
    first = review_submission._review_payload_dict(
        build_payload_for_event(pr, classified,
            event=review_models.ReviewEvent.COMMENT,
            run_info=run_info,
            renderers=renderers,
        )
    )
    second = review_submission._review_payload_dict(
        build_payload_for_event(pr, classified,
            event=review_models.ReviewEvent.COMMENT,
            run_info=run_info,
            renderers=renderers,
        )
    )
    assert first == second
    assert seen[0] == seen[1]
    assert seen[0].findings[0].finding.title == "Explicit finding"
    assert run_info in seen[0].review_info


def test_initial_pr_lookup_preserves_base_tip_separately_from_merge_base(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path,
) -> None:
    row, merge_base, _head = _local_pr_row(git_repo)
    tip = "c" * 40
    row["baseRefOid"] = tip
    monkeypatch.setattr(git_ops, "gh_pr_view", lambda *_args, **_kwargs: row)
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda *_args, **_kwargs: ("o", "r"))
    info = pr_review.find_pr_by_number(git_repo, 7)
    assert info is not None
    assert info.base_sha == merge_base
    assert info.pr_base_sha == tip
    assert info.pr_base_sha != info.base_sha
    assert "baseRefOid" not in git_ops.GH_PR_VIEW_FIELDS
    assert "baseRefOid" not in git_ops.GH_PR_LIST_FIELDS


@pytest.mark.parametrize("advanced", [False, True])
def test_base_tip_api_read_rejects_advanced_pr(
    monkeypatch: pytest.MonkeyPatch, pr: PRInfo, advanced: bool,
) -> None:
    tip = "c" * 40
    calls: list[str] = []
    def api(_repo: Path, endpoint: str, **_kwargs: Any) -> dict[str, Any]:
        calls.append(endpoint)
        return {"head": {"sha": "d" * 40 if advanced else pr.head_sha}, "base": {"sha": tip}}
    monkeypatch.setattr(git_ops, "gh_api", api)
    assert review_lookup.capture_pr_base_tip(Path("."), pr) == (None if advanced else tip)
    assert calls == [f"repos/{pr.owner}/{pr.repo}/pulls/{pr.number}"]


@pytest.mark.parametrize("path", ["é.py", "spaced name.py", 'quoted"name.py'])
def test_snapshot_placement_decodes_git_quoted_paths(git_repo: Path, path: str) -> None:
    from tests.harness.git_helpers import git
    base = git_ops.head_sha(git_repo)
    (git_repo / path).write_text("defect = True\n")
    git(git_repo, "add", path)
    git(git_repo, "commit", "-m", "add quoted path")
    head = git_ops.head_sha(git_repo)
    pr = PRInfo(7, head, base, "main", "feature", "o", "r", "https://github.com/o/r/pull/7")
    diff = git_ops.diff_paths(git_repo, base, head, ["."])
    issue = ParsedIssue(path=path, line=1, title="Defect", body="defect")
    placed = classify(git_repo, pr, [issue], snapshot_diff=diff)
    assert [(finding.path, finding.line) for finding in placed.inline] == [(path, 1)]
    assert placed.body_only == []


def test_classify_captures_remote_diff_once_after_native_base_failure(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path,
) -> None:
    _commit_file(git_repo, "x.py", "\n".join(f"line {i}" for i in range(30)), "x")
    head = _commit_file(git_repo, "other.py", "line 1\nline 2\n", "other")
    calls: list[int] = []
    def remote(_repo: Path, number: int, **_kwargs: Any) -> str:
        calls.append(number)
        return _GH_PR_DIFF
    monkeypatch.setattr(git_ops, "gh_pr_diff", remote)
    out = classify(git_repo, _pr_for("deadbeef" * 5, head), [
        ParsedIssue("x.py", 11, "x", "prose only"),
        ParsedIssue("other.py", 1, "other", "prose only"),
        ParsedIssue("x.py", 12, "x again", "prose only"),
    ])
    assert [(c.path, c.line) for c in out.inline] == [("x.py", 11), ("other.py", 1), ("x.py", 12)]
    assert len(calls) == 1


def test_classify_failed_remote_acquisition_remains_retryable_for_later_path(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path,
) -> None:
    _commit_file(git_repo, "x.py", "line 1\n", "x")
    head = _commit_file(git_repo, "other.py", "line 1\n", "other")
    calls: list[int] = []
    def remote(_repo: Path, number: int, **_kwargs: Any) -> str:
        calls.append(number)
        if len(calls) == 1:
            raise GitError("remote temporarily unavailable")
        return _GH_PR_DIFF
    monkeypatch.setattr(git_ops, "diff_name_only", lambda *_args: ["x.py", "other.py"])
    monkeypatch.setattr(git_ops, "gh_pr_diff", remote)
    out = classify(git_repo, _pr_for("deadbeef" * 5, head), [
        ParsedIssue("x.py", 1, "first", "prose only"),
        ParsedIssue("other.py", 1, "later", "prose only"),
        ParsedIssue("x.py", 1, "cached failure", "prose only"),
    ])
    assert [(c.path, c.line) for c in out.inline] == [("other.py", 1)]
    assert [i.title for i in out.file_level] == ["first", "cached failure"]
    assert len(calls) == 2


def test_classify_remote_foreign_malformed_path_does_not_affect_selected_file(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path,
) -> None:
    head = _commit_file(git_repo, "x.py", "\n".join(f"line {i}" for i in range(30)), "x")
    foreign = 'diff --git a/broken b/broken\n--- a/broken\n+++ "b/\\377"\n@@ -1 +1 @@\n+x\n'
    monkeypatch.setattr(git_ops, "gh_pr_diff", lambda *_args, **_kwargs: foreign + _GH_PR_DIFF)
    out = classify(git_repo, _pr_for("deadbeef" * 5, head), [ParsedIssue("x.py", 11, "t", "prose only")])
    assert [(c.path, c.line) for c in out.inline] == [("x.py", 11)]


def test_classify_resolves_anchors_from_the_same_head_bytes_it_admitted(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path,
) -> None:
    base = _commit_file(git_repo, "x.py", "original\n", "base")
    head = _commit_file(git_repo, "x.py", "captured_marker\n", "head")
    real_show = git_ops.show
    calls: list[str] = []
    def capture(repo: Path, sha: str, path: str) -> bytes:
        calls.append(path)
        if len(calls) > 1:
            raise GitError("a second acquisition is unavailable")
        return real_show(repo, sha, path)
    monkeypatch.setattr(git_ops, "show", capture)
    out = classify(git_repo, _pr_for(base, head), [ParsedIssue("x.py", None, "t", "`captured_marker`")])
    assert [c.line for c in out.inline] == [1]
    assert calls == ["x.py"]


def test_classify_unchanged_rename_preserves_native_per_path_hunks(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path,
) -> None:
    base = _commit_file(git_repo, "old.py", "\n".join(f"line {i}" for i in range(30)), "base")
    _git(git_repo, "mv", "old.py", "new.py")
    _git(git_repo, "commit", "-m", "rename")
    head = _git(git_repo, "rev-parse", "HEAD")
    monkeypatch.setattr(git_ops, "gh_pr_diff", _raise_on_gh_fallback)
    out = classify(git_repo, _pr_for(base, head), [ParsedIssue("new.py", 15, "t", "prose only")])
    assert [(c.path, c.line) for c in out.inline] == [("new.py", 15)]
