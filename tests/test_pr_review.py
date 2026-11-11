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
from daydream.github_app import GitHubIdentity
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
)
from daydream.reconcile import PriorDiagramComment
from daydream.review_reporting import render_github_review_context
from daydream.reviews.rendering import (
    build_payload_for_event,
    format_comment_body,
)
from daydream.run_config import RunConfig
from daydream.run_context import InteractionPolicy, RunContext
from daydream.runner import _write_findings_for_parsed
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import git as _git
from tests.harness.review_payload import payload_for
from tests.harness.review_profile import sample_pr
from tests.harness.review_result import review_coverage, terminal_result

# gh-gated: tests that stub gh's subprocess are skipped when gh is not installed.

SNAP = Path(__file__).parent / "fixtures" / "comment_snapshots"
BUILTIN_RENDERERS = ReviewRenderers(default_render_finding, default_render_summary)



def _case_dir(tmp_path: Path, index: int) -> Path:
    case = tmp_path / f"case-{index}"
    case.mkdir()
    return case


def _inline(path: str = "a.py", line: int = 10, body: str = "x") -> InlineReviewComment:
    """One typed inline comment, the shape ``ClassifiedIssues.inline`` holds."""
    return InlineReviewComment(path=path, line=line, side="RIGHT", body=body)


def _recording_fake_submit(captured: dict[str, pr_review.ClassifiedReviewPlan],) -> Any:
    def fake_submit(plan: pr_review.ClassifiedReviewPlan, *, transport: pr_review.ReviewTransport
    ) -> pr_review.ClassifiedReviewResult:
        captured["plan"] = plan
        return pr_review.ClassifiedReviewResult(
            review_url="https://github.com/acme/widgets/pull/42#pullrequestreview-1",
            posted_file_level=(), folded_file_level=(), safe_error=None,
        )
    return fake_submit


_FIXTURE = Path(__file__).parent / "fixtures" / "trajectories" / "single_phase_claude.json"
_RUN_INFO = pr_comment_renderer.render_run_info_block([_FIXTURE])
# Input findings handed to _post/post_review_to_pr_from_report below. They are inert whenever
# `classify` is stubbed (see _stub_post): the stubbed classifier never reads them.
_POST_ISSUE = ParsedIssue(path="a.py", line=1, title="t", body="b")
_NO_ISSUES = pr_review.ClassifiedIssues()


def _registry_with(kind: str, renderer: Any) -> ReviewRenderers:
    registry = Registry()
    register_builtins(registry)
    registry.override_renderer(kind, renderer)
    return resolve_review_renderers(registry)


def test_custom_renderers_are_framed_by_the_host_and_fall_back_on_error(
    caplog: pytest.LogCaptureFixture, pr: PRInfo,
) -> None:
    """A working summary renderer is framed with host count, commit, markers and footer; it owns diagrams.

    A failing finding or summary renderer falls back to the built-in output and warns.
    """
    note = ParsedIssue(path="b.py", line=None, title="File note", body="desc", confidence="MEDIUM",
                       severity="low", fingerprint="b" * 64)
    classified = pr_review.ClassifiedIssues(body_only=[note])

    def collapsible(ctx: Any) -> str:
        rows = [f"<details><summary>{f.finding.path} — {f.finding.title}</summary>\n{f.body_block}\n</details>"
                for f in ctx.findings]
        return "**Custom Summary**\n\n" + "\n".join(rows)

    body = payload_for(pr, classified, renderers=_registry_with("summary", collapsible), run_info=_RUN_INFO)["body"]
    assert body.startswith("Actionable comments posted: 1\n\n")
    assert body.index("Reviewed commit") < body.index("**Custom Summary**")
    assert "<summary>b.py — File note</summary>" in body
    assert parse_finding_markers(body) == ["b" * 64]
    assert body.rstrip().endswith("</sub>")

    blocks = "<details><summary><h3>Flowchart</h3></summary>\nX\n</details>"
    seen: list[str | None] = []
    for keep in (True, False):
        def diagram_renderer(ctx: Any, keep: bool = keep) -> str:
            seen.append(ctx.diagrams)
            return f"**Custom**\n\n{ctx.diagrams}" if keep else "**Custom**"
        rendered = payload_for(pr, _NO_ISSUES, diagram_blocks=blocks, renderers=_registry_with(
            "summary", diagram_renderer), run_info=_RUN_INFO)["body"]
        assert (blocks in rendered) is keep  # the host cannot restore blocks a renderer omits
    assert seen == [blocks, blocks]

    def boom_finding(finding: Any, ctx: Any) -> str:
        raise RuntimeError("boom")

    def boom_summary(ctx: Any) -> str:
        raise RuntimeError("kaboom")

    with caplog.at_level("WARNING"):
        finding_body = format_comment_body(ParsedIssue(
            path="a.py", line=3, title="T", body="B rationale", severity="high", confidence="HIGH",
            fingerprint="a" * 64,
        ), "inline", renderers=_registry_with("finding", boom_finding))
        summary_body = payload_for(pr, classified, renderers=_registry_with("summary", boom_summary),
                                   run_info=_RUN_INFO)["body"]
    assert finding_body == (SNAP / "inline.md").read_text()
    assert summary_body == payload_for(pr, classified, renderers=BUILTIN_RENDERERS, run_info=_RUN_INFO)["body"]
    assert "boom" in caplog.text and "kaboom" in caplog.text


@pytest.fixture
def pr() -> PRInfo:
    return sample_pr()


def _raise_on_gh_fallback(*_a: Any, **_k: Any) -> str:
    raise AssertionError("gh fallback invoked")


def _payload_keeps_warning_display_bounded_case(pr: PRInfo, with_terminal_result: bool) -> None:
    """Payload assembly must not restore diagnoses omitted by the bounded summary."""
    warnings = tuple(f"stack-{index:02}: <unsafe> & reviewer timed out" for index in range(12))
    summary = None
    if with_terminal_result:
        coverage = review_coverage(scope_ids=('structural',), phases=('structural', 'merge'))
        coverage.record_scope('structural', 'failed', reasons=('backend_failure',))
        coverage.record_phase('structural', 'failed', reasons=('synthesis_failure',))
        coverage.record_phase('merge', 'failed', reasons=('synthesis_failure',))
        warnings = ('structural: synthesis transport failed', 'structural: reviewer backend failed',
                    'structural: additional evidence unavailable', 'structural: synthesis transport failed',
                    *warnings, "Transport evidence unavailable.")
        summary = render_github_review_context(coverage.finalize("failed"), review_warnings=warnings)
    body = build_payload_for_event(
        pr, pr_review.ClassifiedIssues(), event=pr_review.ReviewEvent.COMMENT,
        run_info="Fixture run info", renderers=BUILTIN_RENDERERS,
        review_warnings=warnings, review_summary=summary,
    ).body

    if with_terminal_result:
        for warning in warnings[:3]:
            assert body.count(warning) == 1
    assert body.count("reviewer timed out") == (4 if with_terminal_result else 8)
    assert f"{9 if with_terminal_result else 4} additional" in body
    assert "&lt;unsafe&gt; &amp;" in body
    assert ('merge: synthesis failure' in body) is with_terminal_result
    assert "stack-11:" not in body
    assert "Transport evidence unavailable." not in body


def test_payload_keeps_warning_display_bounded(pr: PRInfo) -> None:
    for with_terminal_result in [False, True]:
        _payload_keeps_warning_display_bounded_case(pr=pr, with_terminal_result=with_terminal_result)


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


def test_pr_lookup_and_findings_export_accept_unavailable_head_slug(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path, tmp_path: Path,
) -> None:
    """Both gh lookup routes preserve validated head identity through export; lookup failures are hard errors."""

    row, base, head = _local_pr_row(git_repo)
    row["headRepository"] = {"id": "R_fixture", "name": "shelfspace-mono", "nameWithOwner": ""}
    row["headRepositoryOwner"] = {"login": "shelfspace-app"}
    monkeypatch.setattr(git_ops, "gh_pr_list_for_branch", lambda *_a, **_k: [row])
    monkeypatch.setattr(git_ops, "gh_pr_view", lambda *_a, **_k: row)
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda *_a, **_k: ("o", "r"))
    branch_info = pr_review.find_open_pr(git_repo)
    info = pr_review.find_pr_by_number(git_repo, 7)
    assert info is not None and branch_info == info
    assert (info.head_repo, info.base_sha, info.head_sha) == ("shelfspace-app/shelfspace-mono", base, head)

    output = tmp_path / "findings.json"
    config = RunConfig(findings_out=str(output), pr_number=7)
    assert _write_findings_for_parsed(
        git_repo, config, [], run_info="", renderers=BUILTIN_RENDERERS, terminal_result=terminal_result(head_sha=head),
        captured_pr=info, snapshot_diff=git_ops.diff_paths(git_repo, base, head, ["."]),
    ) == 0
    artifact = load_findings_artifact(output, expected_repo="o/r", expected_pr_number=7, expected_head_sha=head,)
    assert artifact.findings == []

    # Captured export identity still requires an available PR head.
    output.unlink()
    row["headRefOid"] = "f" * 40
    with pytest.raises(GitError, match="bad object"):
        _write_findings_for_parsed(
            git_repo, config, [], run_info="", renderers=BUILTIN_RENDERERS,
            terminal_result=terminal_result(head_sha="f" * 40), captured_pr=replace(info, head_sha="f" * 40),
            snapshot_diff=git_ops.diff_paths(git_repo, base, head, ["."]),
        )
    assert not output.exists()

    def fail_slug(_repo: Path, **_kwargs: Any) -> tuple[str, str]:
        raise GitError("gh repo view failed: auth")

    monkeypatch.setattr(git_ops, "gh_repo_view_required", fail_slug)
    with pytest.raises(GitError, match="auth"):
        pr_review.find_pr_by_number(git_repo, 7)

    def fail_branch(_repo: Path) -> str | None:
        raise GitError("cannot read current branch")

    monkeypatch.setattr(git_ops, "current_branch", fail_branch)
    with pytest.raises(GitError, match="cannot read current branch"):
        pr_review.find_open_pr(git_repo)


def test_pr_info_rejects_malformed_or_contradictory_head_identity(
    git_repo: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for slug in ("other/r", "fork/other"):
        with pytest.raises(GitError, match="contradictory head repository identity") as error:
            pr_review._head_repo_slug_from_row({
                "headRepository": {"name": "r", "nameWithOwner": slug, "id": "private-value"},
                "headRepositoryOwner": {"login": "fork"},
            })
        assert "private-value" not in str(error.value)
        assert slug not in str(error.value)
    valid, _, _ = _local_pr_row(git_repo)
    fake_gh.set_response("repo-view", value="o/r")
    malformed: list[tuple[str, Any]] = [
        ("number", True), ("number", 0), ("headRefOid", "HEAD"), ("headRefName", ""), ("headRefName", "feature~1"),
        ("baseRefName", ""), ("baseRefName", "main~1"), ("url", ""), ("headRepository", "fork/r"),
        ("headRepository", {"nameWithOwner": "fork/r/extra", "name": "r"}), ("headRepository", {}),
        ("headRepository", {"nameWithOwner": 7, "name": "r"}), ("headRepository", {"nameWithOwner": "", "name": None}),
        ("headRepository", {"nameWithOwner": "", "name": ""}), ("headRepository", {"nameWithOwner": "", "name": " "}),
        ("headRepository", {"nameWithOwner": "", "name": "a/b"}), ("headRepositoryOwner", {"login": ""}),
        ("headRepositoryOwner", "fork"),
    ]
    for index, (field, value) in enumerate(malformed, start=1):
        fake_gh.serve_pr_view({**valid, field: value})
        try:
            with pytest.raises(GitError, match="invalid PR row|base branch|exact PR head"):
                pr_review.find_pr_by_number(git_repo, 7)
        except BaseException as exc:
            exc.add_note(f"case: {field}={value!r}")
            raise
        assert len(fake_gh.command_calls("pr view")) == index, field
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("o", "r"))
    for missing_field in ("headRepository", "headRepositoryOwner"):
        row = dict(valid)
        del row[missing_field]
        with pytest.raises(GitError, match="invalid PR row"):
            pr_review._pr_info_from_row(git_repo, row)
    # A same-repository PR may report null head metadata.
    for head_owner in (None, {"login": "former-owner"}):
        assert pr_review._pr_info_from_row(
            git_repo, {**valid, "headRepository": None, "headRepositoryOwner": head_owner}).head_repo is None


def test_fork_pr_uses_upstream_base_for_both_lookup_paths(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    row, base, _ = _local_pr_row(git_repo, head_owner="forky")
    _git(git_repo, "remote", "add", "origin", "https://github.com/forky/r.git")
    _git(git_repo, "remote", "add", "upstream", "https://github.com/acme/widgets.git")
    _git(git_repo, "update-ref", "refs/remotes/upstream/main", base)
    monkeypatch.setattr(git_ops, "gh_repo_view_required", lambda _r, **_kwargs: ("acme", "widgets"),)
    monkeypatch.setattr(git_ops, "gh_pr_list_for_branch", lambda *_a, **_k: [row])
    monkeypatch.setattr(git_ops, "gh_pr_view", lambda *_a, **_k: row)

    for info in (pr_review.find_open_pr(git_repo), pr_review.find_pr_by_number(git_repo, 7)):
        assert info is not None
        assert info.base_sha == base
        assert (info.owner, info.repo, info.head_repo) == ("acme", "widgets", "forky/r")
    # Commit links follow the fork's head repository while the review posts to the base repository.
    assert info is not None
    head = info.head_sha
    body = payload_for(info, pr_review.ClassifiedIssues(), run_info="run info", renderers=BUILTIN_RENDERERS)["body"]
    assert f"- **Reviewed commit:** [`{head[:7]}`](https://github.com/forky/r/commit/{head})" in body
    assert f"https://github.com/acme/widgets/commit/{head}" not in body


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


def _stub_post(monkeypatch: pytest.MonkeyPatch, pr: PRInfo, classified: pr_review.ClassifiedIssues,
               ) -> dict[str, pr_review.ClassifiedReviewPlan]:
    """Stub every ``pr_review._post`` seam and return the dict that receives the submitted plan."""
    captured: dict[str, pr_review.ClassifiedReviewPlan] = {}
    monkeypatch.setattr(pr_review, "find_open_pr", lambda _td, **_kwargs: pr)
    monkeypatch.setattr(pr_review, "classify", lambda *_a, **_k: classified)
    monkeypatch.setattr(pr_review, "post_classified_review", _recording_fake_submit(captured))
    return captured


def _post_kwargs(answer: str | None = "yes", **overrides: Any) -> dict[str, Any]:
    """The console/run-context/renderers/run-info every ``pr_review._post`` call shares."""
    return {"console": _FakeConsole(), "run_context": _assumed_context(answer) if answer is not None else None,
            "renderers": BUILTIN_RENDERERS, "run_info": _RUN_INFO, **overrides}


async def _post_payload_approves_when_clean_and_enabled_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pr: PRInfo,
    complete: bool,
) -> None:
    clean = pr_review.ClassifiedIssues(
        inline=[_inline(line=1)],
        inline_issues=[ParsedIssue(path="a.py", line=1, title="t", body="b", confidence="LOW", severity="low")],
    )
    captured = _stub_post(monkeypatch, pr, clean)

    status = await pr_review._post(
        tmp_path, [_POST_ISSUE], **_post_kwargs(approve_on_clean=True, review_complete=complete)
    )
    assert captured["plan"].event is (pr_review.ReviewEvent.APPROVE if complete else pr_review.ReviewEvent.COMMENT)
    assert status == pr_review.PostStatus.POSTED


async def test_publication_decisions_follow_review_state(tmp_path: Path, pr: PRInfo) -> None:
    """Approval, empty-report diagrams, failed diagram replacement and base-tip reads follow captured state."""
    # An authorized plan snapshots its issues; later classification mutation cannot change the submission.
    mutable = pr_review.ClassifiedIssues(inline=[_inline()], inline_issues=[ParsedIssue(path="a.py", line=10,
                                         title="planned", body="b")], body_only=[ParsedIssue(path="b.py", line=None,
                                         title="planned note", body="d")])
    plan = pr_review.ClassifiedReviewPlan.from_classified(pr, mutable, event=pr_review.ReviewEvent.COMMENT,
                                                          run_info=_RUN_INFO, renderers=BUILTIN_RENDERERS)
    mutable.inline_issues[0].title = mutable.body_only[0].title = "mutated"
    mutable.inline.append(_inline(line=20))
    assert [issue.title for issue in (*plan.inline_issues, *plan.body_only)] == ["planned", "planned note"]
    assert len(plan.inline) == 1
    # A rationale that only repeats the description is not posted twice.
    repeated, = pr_review.parsed_issues_from_items([{"file": "a.py", "line": 1, "description": "Same text",
                                                     "rationale": "Same text", "severity": "low"}])
    assert repeated.title == "Same text" and "Same text" not in repeated.body
    with pytest.MonkeyPatch.context() as mp:
        warnings: list[str] = []
        mp.setattr(pr_review, "find_pr_by_number", lambda *_a, **_k: None)
        mp.setattr(pr_review, "print_warning", lambda _c, message: warnings.append(message))
        assert pr_review._resolve_pr(tmp_path, _FakeConsole(), 7,  # type: ignore[arg-type]
                                     auth=git_ops.StaticGitHubAuth({})) is None
        assert warnings == ["PR #7 not found via `gh pr view`; skipping PR post."]
    for index, flag in enumerate([False, True]):
        with pytest.MonkeyPatch.context() as mp:
            await _post_payload_approves_when_clean_and_enabled_case(
                monkeypatch=mp, tmp_path=_case_dir(tmp_path, index), pr=pr, complete=flag)
        with pytest.MonkeyPatch.context() as mp:
            await _post_review_from_report_empty_items_posts_diagram_case(
                monkeypatch=mp, tmp_path=_case_dir(tmp_path, 2 + index), pr=pr, with_diagram=flag)
        with pytest.MonkeyPatch.context() as mp:
            _base_tip_api_read_rejects_advanced_pr_case(monkeypatch=mp, pr=pr, advanced=flag)
    with pytest.MonkeyPatch.context() as mp:
        _diagram_replacement_post_failure_keeps_prior_comment(pr=pr, monkeypatch=mp, tmp_path=_case_dir(tmp_path, 4))


async def test_pinned_publication_revalidates_and_places_the_captured_comparison(
    monkeypatch: pytest.MonkeyPatch,
    git_repo: Path,
) -> None:
    """Real placement covers dirty bytes; prior _post tests replaced classification and missed reanchoring."""
    row, base, head = _local_pr_row(git_repo)
    pinned = PRInfo(
        number=7,
        owner="o",
        repo="r",
        base_sha=base,
        pr_base_sha=base,
        head_sha=head,
        base_ref="main",
        head_ref="feature",
        url=row["url"],
    )
    snapshot = git_ops.diff_paths(git_repo, base, head, ["."])
    (git_repo / "feature.py").write_text('value = "LOCAL_ONLY"\n' * 10)
    (git_repo / "untracked.py").write_text("LOCAL_ONLY = True\n")
    issue = ParsedIssue(
        path="feature.py", line=1, title="Pinned defect", body="value = 1", severity="medium", confidence="HIGH"
    )
    moved = {
        None: pinned,
        "head": replace(pinned, head_sha=base),
        "base": replace(pinned, pr_base_sha=head),
        "repository": replace(pinned, owner="other"),
    }
    for movement, current_pr in moved.items():
        captured: dict[str, pr_review.ClassifiedReviewPlan] = {}
        lookups: list[PRInfo] = [pinned, current_pr]
        monkeypatch.setattr(pr_review, "find_pr_by_number", lambda *_a, **_k: lookups.pop(0))
        monkeypatch.setattr(pr_review, "post_classified_review", _recording_fake_submit(captured))
        status = await pr_review._post(
            git_repo,
            [issue],
            **_post_kwargs(post=True, captured_pr=pinned, snapshot_diff=snapshot, publication_dir=git_repo),
        )
        assert lookups == []
        if movement:
            assert status is pr_review.PostStatus.FAILED and not captured
        else:
            plan = captured["plan"]
            assert status is pr_review.PostStatus.POSTED
            assert plan.pr == pinned and [(item.path, item.line) for item in plan.inline] == [("feature.py", 1)]
            assert not plan.file_level and not plan.body_only
        assert (git_repo / "feature.py").read_text() == 'value = "LOCAL_ONLY"\n' * 10
        assert (git_repo / "untracked.py").read_text() == "LOCAL_ONLY = True\n"


async def test_complete_pinned_comment_review_supersedes_its_unfinished_notice(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path, fake_gh: FakeGh, tmp_path: Path,
) -> None:
    """The --comment route answers its own unfinished notice only after a complete pinned review.

    The bot-route matrix proves notice selection; this proves the live route's completeness gate and identity."""
    row, base, head = _local_pr_row(git_repo)
    pinned = PRInfo(number=7, owner="o", repo="r", base_sha=base, pr_base_sha=base, head_sha=head,
                    base_ref="main", head_ref="feature", url=row["url"])
    monkeypatch.setattr(pr_review, "find_pr_by_number", lambda *_a, **_k: pinned)
    notice = build_payload_for_event(pinned, pr_review.ClassifiedIssues(), event=pr_review.ReviewEvent.COMMENT,
                                     run_info=_RUN_INFO, renderers=BUILTIN_RENDERERS,
                                     review_warnings=("structure: Review limit exhausted",)).body
    fake_gh.set_response("GET", "repos/o/r/pulls/7/reviews",
                         [{"id": 1, "user": {"login": "Octocat"}, "commit_id": head, "body": notice}])
    merged = tmp_path / "merged-items.json"
    merged.write_text(json.dumps({"items": []}))
    context = RunContext(InteractionPolicy(assume="yes"), github_identity=GitHubIdentity("octocat"))
    for complete in (False, True):
        status = await pr_review.post_review_to_pr_from_report(
            git_repo, merged, console=_FakeConsole(),  # type: ignore[arg-type]
            post=True, renderers=BUILTIN_RENDERERS, run_info=_RUN_INFO, run_context=context, captured_pr=pinned,
            snapshot_diff=git_ops.diff_paths(git_repo, base, head, ["."]), review_complete=complete,
            publication_dir=git_repo,
        )
        posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
        if not complete:
            assert status is pr_review.PostStatus.NOTHING_TO_POST and not fake_gh.calls("GET") and not posts
    assert status is pr_review.PostStatus.POSTED
    assert len(posts) == 1 and posts[0].payload["event"] == "COMMENT" and posts[0].payload["commit_id"] == head
    assert f"✅ **Review complete.** No findings in the complete review of `{head[:7]}`" in posts[0].payload["body"]


async def _post_review_from_report_empty_items_posts_diagram_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pr: PRInfo,
    with_diagram: bool,
) -> None:
    merged = tmp_path / "merged-items.json"
    merged.write_text(json.dumps({"items": []}))
    blocks = "<details><summary><h3>Flowchart</h3></summary>\nX\n</details>"
    captured = _stub_post(monkeypatch, pr, _NO_ISSUES)

    status = await pr_review.post_review_to_pr_from_report(
        tmp_path,
        merged,
        console=_FakeConsole(),  # type: ignore[arg-type]
        post=True,
        diagram_blocks=blocks if with_diagram else None,
        renderers=BUILTIN_RENDERERS,
        run_info=_RUN_INFO,
        review_summary="## Review outcome\n\nNo confirmed findings. Recorded analysis complete.",
    )

    if not with_diagram:
        assert status == pr_review.PostStatus.NOTHING_TO_POST
        assert not captured
        return
    assert status == pr_review.PostStatus.POSTED
    assert captured["plan"].event is pr_review.ReviewEvent.COMMENT
    assert captured["plan"].diagram_blocks == blocks
    assert captured["plan"].review_summary is not None
    assert "No confirmed findings" in captured["plan"].review_summary


def _commit_file(repo: Path, path: str, contents: str, message: str) -> str:
    """Write *path* under *repo*, commit it, and return the new HEAD SHA."""
    file_path = repo / path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(contents)
    _git(repo, "add", path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def test_classify_places_findings_against_real_changed_ranges(git_repo: Path) -> None:
    """A two-hunk diff owns tolerance, nearest-boundary, anchor search and non-inline placement."""
    lines = [f"row_{index}" for index in range(1, 41)]
    lines[1], lines[3] = "row_2 marker_token", "row_4 outside_only_token"
    _commit_file(git_repo, "x.py", "\n".join(lines) + "\n", "base source")
    _commit_file(git_repo, "cache.yaml", _CACHE_YAML_BASE, "base cache policy")
    base = _commit_file(git_repo, "stable.py", "stable_marker\n", "unchanged source")
    lines[9] = "row_10 changed"
    lines[23] = "row_24 changed marker_token"
    _commit_file(git_repo, "x.py", "\n".join(lines) + "\n", "two changed ranges")
    head = _commit_file(git_repo, "cache.yaml", _CACHE_YAML_HEAD, "cut prod ttl")
    issues = [
        ParsedIssue(path="x.py", line=10, title="in hunk", body="prose has no anchor"),
        ParsedIssue(path="x.py", line=4, title="before within tolerance", body="`row_4`"),
        ParsedIssue(path="x.py", line=15, title="nearest earlier hunk", body="`row_15`"),
        ParsedIssue(path="x.py", line=20, title="nearest later hunk", body="`row_20`"),
        ParsedIssue(path="x.py", line=30, title="after within tolerance", body="`row_30`"),
        ParsedIssue(path="x.py", line=2, title="bad hint", body="`row_24`"),
        ParsedIssue(path="x.py", line=3, title="before outside tolerance", body="`row_3`"),
        ParsedIssue(path="x.py", line=17, title="between outside tolerance", body="`row_17`"),
        ParsedIssue(path="x.py", line=31, title="after outside tolerance", body="`row_31`"),
        ParsedIssue(path="stable.py", line=1, title="unchanged", body="`stable_marker`"),
        ParsedIssue(path="cross-stack", line=None, title="cross stack", body="context", is_cross_stack=True),
        ParsedIssue(path="x.py", line=None, title="duplicate anchor", body="`marker_token`"),
        ParsedIssue(path="x.py", line=None, title="outside-only anchor", body="`outside_only_token`"),
        replace(_PROSE_HEAVY_ISSUE),
        replace(_PROSE_HEAVY_ISSUE, line=None, title="Expiration collapses from an hour to a minute without hint"),
    ]
    result = classify(git_repo, _pr_for(base, head), issues)
    assert [comment.line for comment in result.inline] == [10, 7, 13, 21, 27, 24, 24, 7, 15, 15]
    assert [comment.path for comment in result.inline] == ["x.py"] * 8 + ["cache.yaml"] * 2
    assert all(comment.side == "RIGHT" for comment in result.inline)
    assert [issue.title for issue in result.inline_issues[-4:]] == [
        "duplicate anchor",
        "outside-only anchor",
        "Expiration collapses from an hour to a minute",
        "Expiration collapses from an hour to a minute without hint",
    ]
    assert all("**Placement:**" not in comment.body for comment in result.inline[-2:])
    assert [issue.title for issue in result.file_level] == [
        "before outside tolerance",
        "between outside tolerance",
        "after outside tolerance",
    ]
    assert [issue.title for issue in result.body_only] == ["unchanged", "cross stack"]
    assert "posted on line 24; reviewer cited line 2" in result.inline[5].body
    assert "<!-- daydream-finding:" not in result.inline[0].body
    # Default anchor ordering feeds persisted fingerprints; rationale prose still crowds `ttl` out of the cap.
    assert extract_anchors(f"{_PROSE_HEAVY_ISSUE.title}\n{_PROSE_HEAVY_ISSUE.body}") == [
        "environment", "unavailable", "immediately", "Expiration", "downstream", "additional", "collapses", "override",
    ]
    note = "**Placement:** posted on line 7; reviewer cited line 4."
    assert note in issues[1].body and note in result.inline[1].body
    classify(git_repo, _pr_for(base, head), issues)
    assert issues[1].body.count("**Placement:**") == 1


_PROSE_HEAVY_ISSUE = ParsedIssue(
    path="cache.yaml",
    line=15,
    title="Expiration collapses from an hour to a minute",
    body=(
        "The new override sets `ttl` far below every other environment, so cached "
        "entries become unavailable almost immediately and the downstream service "
        "absorbs the additional request volume."
    ),
)

# `prod:`'s TTL drops from an hour to a minute on line 15; every other block
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
    "\n\n\n\n"  # Earlier ttl hits stay outside prod's three-line diff context.
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


def test_file_hunks_fall_back_to_the_pr_diff_only_when_git_cannot_read_the_base(
    monkeypatch: pytest.MonkeyPatch,
    git_repo: Path,
) -> None:
    """A gh-wrapper fake supplies hunks after real Git rejects the base SHA; failures degrade to no hunks."""
    missing_base = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    monkeypatch.setattr(git_ops, "gh_pr_diff", _raise_on_gh_fallback)
    assert pr_review.file_hunks(git_repo, missing_base, "HEAD", "x.py") == []
    monkeypatch.setattr(git_ops, "gh_pr_diff", lambda _r, _n, **_kwargs: _GH_PR_DIFF)
    # Must come from the x.py block only; the other.py hunk must NOT leak into x.py's result.
    assert pr_review.file_hunks(git_repo, missing_base, "HEAD", "x.py", pr_number=42) == [(10, 14)]

    def raise_git_error(*_a: Any, **_k: Any) -> str:
        raise git_ops.GitError("gh blew up")

    monkeypatch.setattr(git_ops, "gh_pr_diff", raise_git_error)
    assert pr_review.file_hunks(git_repo, missing_base, "HEAD", "x.py", pr_number=42) == []
    sha = _git(git_repo, "rev-parse", "HEAD")
    assert pr_review.resolve_line(git_repo, sha, ParsedIssue(path="gone.py", line=1, title="t", body="b")) is None


def _record_minimize(calls: list[tuple[str, str | None]]) -> Any:
    def minimize(_target: Path, node_id: str, **_kwargs: Any) -> bool:
        calls.append(("minimize", node_id))
        return True

    return minimize


def _diagram_replacement_post_failure_keeps_prior_comment(
    pr: PRInfo, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        "daydream.reconcile.fetch_prior_diagram_comments",
        lambda *_a, **_k: [PriorDiagramComment("IC_prior", ("sequence",))],
    )

    monkeypatch.setattr(
        "daydream.reconcile.minimize_comment",
        _record_minimize(calls),
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


def _base_tip_api_read_rejects_advanced_pr_case(
    monkeypatch: pytest.MonkeyPatch,
    pr: PRInfo,
    advanced: bool,
) -> None:
    tip = "c" * 40
    calls: list[str] = []

    def api(_repo: Path, endpoint: str, **_kwargs: Any) -> dict[str, Any]:
        calls.append(endpoint)
        return {"head": {"sha": "d" * 40 if advanced else pr.head_sha}, "base": {"sha": tip}}

    monkeypatch.setattr(git_ops, "gh_api", api)
    assert pr_review.capture_pr_base_tip(Path("."), pr) == (None if advanced else tip)
    assert calls == [f"repos/{pr.owner}/{pr.repo}/pulls/{pr.number}"]

    def unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise GitError("gh api failed")

    monkeypatch.setattr(git_ops, "gh_api", unavailable)
    assert pr_review.capture_pr_base_tip(Path("."), pr) is None
