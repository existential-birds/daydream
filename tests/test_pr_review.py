"""Unit tests for daydream.pr_review."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops, pr_comment_renderer, pr_review
from daydream.extensions import Registry
from daydream.extensions.builtins import register_builtins
from daydream.findings import load_findings_artifact
from daydream.git_ops import GitError
from daydream.pr_review import (
    InlineReviewComment,
    ParsedIssue,
    PRInfo,
    ReviewRenderers,
    classify,
    default_render_finding,
    default_render_summary,
    extract_anchors,
    parse_finding_markers,
    resolve_review_renderers,
    snap_to_hunk,
)
from daydream.reconcile import PriorDiagramComment
from daydream.reviews.rendering import (
    format_comment_body,
)
from daydream.run_config import RunConfig
from daydream.run_context import InteractionPolicy, RunContext
from daydream.runner import _emit_findings_from_items
from tests.harness.git_helpers import git as _git
from tests.harness.review_payload import payload_for
from tests.harness.review_profile import sample_pr
from tests.harness.review_result import terminal_result

# gh-gated: tests that stub gh's subprocess are skipped when gh is not installed.

SNAP = Path(__file__).parent / "fixtures" / "comment_snapshots"
BUILTIN_RENDERERS = ReviewRenderers(default_render_finding, default_render_summary)


def _inline(path: str = "a.py", line: int = 10, body: str = "x") -> InlineReviewComment:
    """One typed inline comment, the shape ``ClassifiedIssues.inline`` holds."""
    return InlineReviewComment(path=path, line=line, side="RIGHT", body=body)


def _recording_fake_submit(captured: dict[str, pr_review.ClassifiedReviewPlan],) -> Any:
    def fake_submit(plan: pr_review.ClassifiedReviewPlan, *, transport: pr_review.ReviewTransport
    ) -> pr_review.ClassifiedReviewResult:
        captured["plan"] = plan
        return pr_review.ClassifiedReviewResult(status=pr_review.SubmissionStatus.POSTED,
            review_url="https://github.com/acme/widgets/pull/42#pullrequestreview-1",
            posted_file_level=(), folded_file_level=(), final_review_posted=True, safe_error=None,
        )
    return fake_submit



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
_RUN_INFO = pr_comment_renderer.render_run_info_block([_FIXTURE])
# Input findings handed to _post/post_review_to_pr_from_report below. They are inert whenever
# `classify` is stubbed (see _stub_post): the stubbed classifier never reads them.
_POST_ISSUE = ParsedIssue(path="a.py", line=1, title="t", body="b")
_NO_ISSUES = pr_review.ClassifiedIssues()

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
    payload = payload_for(pr, classified, renderers=resolve_review_renderers(reg),
        run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
    )
    body = payload["body"]
    assert "**Custom Summary**" in body
    assert "<summary>b.py — File note</summary>" in body
    assert parse_finding_markers(body) == ["b" * 64]
    assert body.rstrip().endswith("</sub>")


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
    default_body = payload_for(
        pr, classified, renderers=BUILTIN_RENDERERS, run_info=pr_comment_renderer.render_run_info_block([_FIXTURE])
    )["body"]
    with caplog.at_level("WARNING"):
        body = payload_for(pr, classified, renderers=resolve_review_renderers(reg),
            run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
        )["body"]
    assert body == default_body
    assert "summary" in caplog.text and "kaboom" in caplog.text




def test_no_marker_without_fingerprint() -> None:
    assert (parse_finding_markers(format_comment_body(
                ParsedIssue(path="a.py", line=3, title="T", body="B"), "inline", renderers=BUILTIN_RENDERERS
            )
        )
        == []
    )


def test_extract_anchors_prefers_long_tokens() -> None:
    anchors = extract_anchors("Null check\nThe function `compute_total` dereferences `items` in handleRequest")
    # Backtick tokens should appear; longest first.
    assert "compute_total" in anchors
    assert "handleRequest" in anchors
    assert anchors == sorted(anchors, key=len, reverse=True)




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


def _assert_hunks_resolve(_td: Path, _sha: str, issue: ParsedIssue, hunks: list[tuple[int, int]] | None = None
) -> int | None:
    assert hunks is not None, "classify called resolve_line without the file's hunks"
    return issue.line


def _raise_on_gh_fallback(*_a: Any, **_k: Any) -> str:
    raise AssertionError("gh fallback invoked")


def test_classify_splits_inline_vs_body(monkeypatch: pytest.MonkeyPatch, pr: PRInfo) -> None:
    issues = [ParsedIssue(path="a.py", line=10, title="t1", body="anchor_one"),
        ParsedIssue(path="b.py", line=99, title="t2", body="anchor_two"),
        ParsedIssue(path="c.py", line=None, title="t3", body="xstack", is_cross_stack=True),
    ]

    def fake_hunks(_td: Path, _base: str, _head: str, path: str, *, pr_number: int | None = None, **_kwargs: Any,
    ) -> list[tuple[int, int]]:
        if path == "a.py":
            return [(8, 12)]  # 10 is inside
        if path == "b.py":
            return [(1, 5)]  # 99 is outside
        return []

    monkeypatch.setattr(git_ops, "show", lambda *_a, **_k: b"")
    monkeypatch.setattr(pr_review, "resolve_line", _assert_hunks_resolve)
    monkeypatch.setattr(pr_review, "file_hunks", fake_hunks)

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

    def fake_hunks(_td: Path, _base: str, _head: str, path: str, *, pr_number: int | None = None, **_kwargs: Any,
    ) -> list[tuple[int, int]]:
        if path == "conftest.py":
            return [(90, 105)]  # 89 is 1 below start
        if path == "scripts/modernize-app.py":
            return [(80, 98), (106, 120)]  # 105 is 1 before second hunk
        return []

    monkeypatch.setattr(git_ops, "show", lambda *_a, **_k: b"")
    monkeypatch.setattr(pr_review, "resolve_line", _assert_hunks_resolve)
    monkeypatch.setattr(pr_review, "file_hunks", fake_hunks)

    result = classify(Path("."), pr, issues)
    assert len(result.inline) == 2
    assert result.inline[0].path == "conftest.py"
    assert result.inline[0].line == 90
    assert result.inline[1].path == "scripts/modernize-app.py"
    assert result.inline[1].line == 106



def test_build_payload_reviewed_commit_links_fork_for_fork_head_pr(pr: PRInfo,) -> None:
    """Link the fork commit while retaining the base repository as the POST target."""
    fork_pr = replace(pr, head_repo="forky/widgets")
    payload = payload_for(
        fork_pr, pr_review.ClassifiedIssues(), run_info="test run info", renderers=BUILTIN_RENDERERS
    )
    body = payload["body"]
    assert "- **Reviewed commit:** [`head123`](https://github.com/forky/widgets/commit/head123)" in body
    assert "https://github.com/acme/widgets/commit/head123" not in body
















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


def test_find_pr_by_number_raises_when_slug_unresolved(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    """A resolvable PR but failed owner/repo lookup is a hard error."""
    row, _, _ = _local_pr_row(git_repo)
    monkeypatch.setattr(git_ops, "gh_pr_view", lambda *_a, **_k: row)

    def fail_slug(_repo: Path, **_kwargs: Any) -> tuple[str, str]:
        raise GitError("gh repo view failed: auth")

    monkeypatch.setattr(git_ops, "gh_repo_view_required", fail_slug)
    with pytest.raises(GitError, match="auth"):
        pr_review.find_pr_by_number(git_repo, 7)


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
        pr_review._head_repo_slug_from_row(row)

@pytest.mark.parametrize("component", ["name", "login"])
@pytest.mark.parametrize("value", [None, False, 7, "", " ", "a/b", "a\nb"])
def test_head_slug_fallback_rejects_invalid_components(component: str, value: Any) -> None:
    row = {"headRepository": {"name": "r", "nameWithOwner": ""}, "headRepositoryOwner": {"login": "fork"}}
    row["headRepository" if component == "name" else "headRepositoryOwner"][component] = value
    with pytest.raises(GitError, match="invalid PR row"):
        pr_review._head_repo_slug_from_row(row)

@pytest.mark.parametrize("slug", ["other/r", "fork/other"])
def test_head_slug_rejects_contradictory_valid_identity(slug: str) -> None:
    with pytest.raises(GitError, match="contradictory head repository identity") as error:
        pr_review._head_repo_slug_from_row({
            "headRepository": {"name": "r", "nameWithOwner": slug, "id": "private-value"},
            "headRepositoryOwner": {"login": "fork"},
        })
    assert "private-value" not in str(error.value)
    assert slug not in str(error.value)

def test_head_slug_allows_case_differences() -> None:
    assert pr_review._head_repo_slug_from_row({"headRepository": {"name": "Widgets", "nameWithOwner": "FORK/widgets"},
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
        pr_review._pr_info_from_row(git_repo, row)

@pytest.mark.parametrize("missing_field", ["headRepository", "headRepositoryOwner"])
def test_pr_info_rejects_missing_requested_head_metadata(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path, missing_field: str,
) -> None:
    row, _, _ = _local_pr_row(git_repo)
    del row[missing_field]
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("o", "r"))
    with pytest.raises(GitError, match="invalid PR row"):
        pr_review._pr_info_from_row(git_repo, row)

def test_pr_info_rejects_malformed_owner_even_with_null_head_repository(monkeypatch: pytest.MonkeyPatch, git_repo: Path,
) -> None:
    row, _, _ = _local_pr_row(git_repo)
    row["headRepository"] = None
    row["headRepositoryOwner"] = {"login": 7}
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("o", "r"))
    with pytest.raises(GitError, match="invalid PR row"):
        pr_review._pr_info_from_row(git_repo, row)

@pytest.mark.parametrize("head_owner", [None, {"login": "former-owner"}])
def test_pr_info_accepts_null_same_repo_head_metadata(monkeypatch: pytest.MonkeyPatch, git_repo: Path, head_owner: Any,
) -> None:
    row, _, _ = _local_pr_row(git_repo)
    row["headRepository"] = None
    row["headRepositoryOwner"] = head_owner
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("o", "r"))

    assert pr_review._pr_info_from_row(git_repo, row).head_repo is None

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
        pr_review._pr_info_from_row(git_repo, row)
    assert "top-secret" not in str(excinfo.value)
    assert "token=private" not in str(excinfo.value)

    _git(git_repo, "remote", "set-url", "origin", "https://user:fork-secret@github.com/forky/widgets.git?token=fork",)
    assert pr_review._pr_info_from_row(git_repo, row).base_sha == base


class _FakeConsole:
    def print(self, *_a: Any, **_k: Any) -> None:
        pass


def _assumed_context(answer: str) -> RunContext:
    return RunContext(InteractionPolicy(assume=answer))


def _stub_post(monkeypatch: pytest.MonkeyPatch, pr: PRInfo, classified: pr_review.ClassifiedIssues, *,
               submit: Any = None, messages: list[str] | None = None,
               ) -> dict[str, pr_review.ClassifiedReviewPlan]:
    """Stub every ``pr_review._post`` seam and return the dict that receives the submitted plan.

    Pass *submit* to replace the recording stub (e.g. to return ``FAILED``); pass *messages* to
    collect ``print_success``/``print_warning`` output that is otherwise dropped.
    """
    captured: dict[str, pr_review.ClassifiedReviewPlan] = {}
    collect = messages.append if messages is not None else lambda *_a: None
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td, **_kwargs: pr)
    monkeypatch.setattr(pr_review, "classify", lambda *_a, **_k: classified)
    monkeypatch.setattr(pr_review, "post_classified_review", submit or _recording_fake_submit(captured))
    monkeypatch.setattr(pr_review, "print_info", lambda *_a, **_k: None)
    monkeypatch.setattr(pr_review, "print_success", lambda _c, msg: collect(msg))
    monkeypatch.setattr(pr_review, "print_warning", lambda _c, msg: collect(msg))
    return captured


def _post_kwargs(answer: str | None = "yes", **overrides: Any) -> dict[str, Any]:
    """The console/run-context/renderers/run-info every ``pr_review._post`` call shares."""
    return {"console": _FakeConsole(), "run_context": _assumed_context(answer) if answer is not None else None,
            "renderers": BUILTIN_RENDERERS, "run_info": _RUN_INFO, **overrides}




@pytest.mark.asyncio
async def test_post_payload_approves_when_clean_and_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pr: PRInfo,
) -> None:
    clean = pr_review.ClassifiedIssues(
        inline=[_inline(line=1)],
        inline_issues=[ParsedIssue(path="a.py", line=1, title="t", body="b", confidence="LOW", severity="low")],
    )
    captured = _stub_post(monkeypatch, pr, clean)

    status = await pr_review._post(tmp_path, [_POST_ISSUE], **_post_kwargs(approve_on_clean=True))
    assert captured["plan"].event is pr_review.ReviewEvent.APPROVE
    assert status == pr_review.PostStatus.POSTED





@pytest.mark.asyncio
async def test_post_review_from_report_empty_items_posts_diagram(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pr: PRInfo,
) -> None:
    merged = tmp_path / "merged-items.json"
    merged.write_text(json.dumps({"items": []}))
    blocks = "<details><summary><h3>Flowchart</h3></summary>\nX\n</details>"
    captured = _stub_post(monkeypatch, pr, _NO_ISSUES)

    status = await pr_review.post_review_to_pr_from_report(tmp_path, merged,
        console=_FakeConsole(),  # type: ignore[arg-type]
        post=True, diagram_blocks=blocks, renderers=BUILTIN_RENDERERS, run_info=_RUN_INFO,
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
    sha = _git(git_repo, "rev-parse", "HEAD")
    issue = ParsedIssue(path="gone.py", line=1, title="t", body="b")
    assert pr_review.resolve_line(git_repo, sha, issue) is None


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
    assert pr_review.resolve_line(git_repo, sha, issue) is None
    assert pr_review.resolve_line(git_repo, sha, issue, [(8, 12)]) == 10

def test_resolve_line_prefers_in_hunk_anchor_hit_over_first_file_hit(git_repo: Path) -> None:
    lines = [f"row_{i}" for i in range(1, 21)]
    lines[1] = "marker_token  # pre-existing, unchanged"
    lines[17] = "marker_token  # the changed line"
    sha = _commit_file(git_repo, "x.py", "\n".join(lines) + "\n", "add x.py")
    issue = ParsedIssue(path="x.py", line=None, title="t", body="`marker_token`")
    # Without hunks the first hit (line 2) wins, as before.
    assert pr_review.resolve_line(git_repo, sha, issue) == 2
    # With hunks, the in-hunk hit (line 18) wins over the earlier out-of-hunk one.
    assert pr_review.resolve_line(git_repo, sha, issue, [(16, 20)]) == 18

def test_resolve_line_returns_out_of_hunk_hit_when_no_in_hunk_candidate(git_repo: Path,) -> None:
    lines = [f"row_{i}" for i in range(1, 21)]
    lines[1] = "marker_token  # pre-existing, unchanged"
    sha = _commit_file(git_repo, "x.py", "\n".join(lines) + "\n", "add x.py")
    issue = ParsedIssue(path="x.py", line=None, title="t", body="`marker_token`")
    assert pr_review.resolve_line(git_repo, sha, issue, [(16, 20)]) == 2

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

def test_classify_skips_file_hunks_for_path_missing_at_head(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    """Reject absent head paths before reading hunks or invoking gh fallback."""
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


def test_file_hunks_falls_back_to_gh_when_base_unreachable(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    """A gh-wrapper fake supplies hunks after real Git rejects the base SHA."""
    monkeypatch.setattr(git_ops, "gh_pr_diff", lambda _r, _n, **_kwargs: _GH_PR_DIFF)
    hunks = pr_review.file_hunks(git_repo, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "HEAD", "x.py", pr_number=42)
    # Must come from the x.py block only -- the other.py hunk starts at line 1
    # and must NOT leak into x.py's result.
    assert hunks == [(10, 14)]

def test_file_hunks_no_fallback_without_pr_number(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    monkeypatch.setattr(git_ops, "gh_pr_diff", _raise_on_gh_fallback)
    hunks = pr_review.file_hunks(git_repo, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "HEAD", "x.py")
    assert hunks == []

def test_file_hunks_gh_fallback_handles_subprocess_error(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    """After real Git fails, a GitError from gh degrades to an empty hunk list."""

    def raise_git_error(*_a: Any, **_k: Any) -> str:
        raise git_ops.GitError("gh blew up")

    monkeypatch.setattr(git_ops, "gh_pr_diff", raise_git_error)
    hunks = pr_review.file_hunks(git_repo, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "HEAD", "x.py", pr_number=42)
    assert hunks == []












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

    result = pr_review.post_diagram_comment_to_pr(
        tmp_path, pr, body="diagram", kinds=["sequence"], bot_login="daydream",
    )

    assert result == (None, "simulated POST failure")
    assert calls == [("post", None)]



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
        body = payload_for(pr, classified, diagram_blocks=blocks, renderers=resolve_review_renderers(reg),
            run_info=pr_comment_renderer.render_run_info_block([_FIXTURE]),
        )["body"]
        assert (blocks in body) is expect_present
    assert seen == [blocks, blocks]




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
    assert pr_review.capture_pr_base_tip(Path("."), pr) == (None if advanced else tip)
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
