"""Focused contracts for Improve's headless GitHub issue publisher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops, runner
from daydream.config_file import DaydreamFileConfig
from daydream.improve.publish import (
    ImprovePublishError,
    IssuePublisher,
    issue_body,
    member_fingerprint_marker,
    member_marker,
    package_marker,
)
from daydream.runner import RunConfig


def _issue(package_id: str, *, state: str = "open", number: int = 7) -> dict[str, object]:
    return {
        "number": number,
        "title": "Improve code reuse",
        "body": f"{package_marker(package_id)}\n\nplan",
        "url": f"https://github.com/acme/widgets/issues/{number}",
        "state": state,
    }


def test_issue_body_preserves_complete_plan_markdown() -> None:
    plan = "# Plan\n\nKeep leading structure and final newline.\n"

    body = issue_body("reuse-handler", plan)

    assert body == f"{package_marker('reuse-handler')}\n\n{plan}"
    assert package_marker("reuse-handler") == (
        "<!-- daydream-improve: package=reuse-handler -->"
    )


def test_issue_body_embeds_stable_member_aliases_before_the_complete_plan() -> None:
    plan = "# Plan\n\nDelete duplicate code.\n"
    aliases = ("member-v1:aaa", "member-v1:bbb")

    body = issue_body("reuse-handler", plan, member_aliases=aliases)

    marker_block, embedded_plan = body.split("\n\n", 1)
    assert marker_block.splitlines() == [
        package_marker("reuse-handler"),
        member_marker("member-v1:aaa"),
        member_marker("member-v1:bbb"),
    ]
    assert embedded_plan == plan


def test_package_marker_rejects_comment_injection() -> None:
    with pytest.raises(ValueError, match="package_id"):
        package_marker("package -->\nmalicious")


def test_connect_fails_closed_when_existing_issues_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        git_ops,
        "gh_issue_list_strict",
        lambda *args, **kwargs: (_ for _ in ()).throw(git_ops.GitError("offline")),
    )
    created = False

    def unexpected_create(*args: Any, **kwargs: Any) -> str:
        nonlocal created
        created = True
        return ""

    monkeypatch.setattr(git_ops, "gh_issue_create", unexpected_create)

    with pytest.raises(ImprovePublishError, match="safely reconcile"):
        IssuePublisher.connect(tmp_path, repo_slug="acme/widgets")
    assert created is False


def test_connect_infers_repository_and_lists_open_and_closed_issues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(git_ops, "gh_repo_view", lambda repo: ("acme", "widgets"))

    def list_strict(repo: Path, *, state: str, repo_slug: str) -> list[dict[str, Any]]:
        captured.update(repo=repo, state=state, repo_slug=repo_slug)
        return []

    monkeypatch.setattr(git_ops, "gh_issue_list_strict", list_strict)

    publisher = IssuePublisher.connect(tmp_path)

    assert publisher.repo_slug == "acme/widgets"
    assert captured == {
        "repo": tmp_path,
        "state": "all",
        "repo_slug": "acme/widgets",
    }


def test_existing_closed_issue_is_reused_without_creating_a_duplicate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# Complete plan\n", encoding="utf-8")
    monkeypatch.setattr(
        git_ops,
        "gh_issue_list_strict",
        lambda *args, **kwargs: [_issue("reuse-handler", state="closed")],
    )
    monkeypatch.setattr(
        git_ops,
        "gh_issue_create",
        lambda *args, **kwargs: pytest.fail("must not create a duplicate issue"),
    )
    publisher = IssuePublisher.connect(tmp_path, repo_slug="acme/widgets")

    result = publisher.publish(
        package_id="reuse-handler",
        title="Reuse the existing handler",
        plan_path=plan_path,
    )

    assert result.disposition == "existing"
    assert result.issue_url.endswith("/7")


def test_all_member_aliases_reconcile_a_regrouped_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# Complete plan\n", encoding="utf-8")
    existing = _issue("older-package")
    existing["body"] = issue_body(
        "older-package",
        "old plan",
        member_aliases=("member-v1:aaa", "member-v1:bbb"),
    )
    monkeypatch.setattr(
        git_ops,
        "gh_issue_list_strict",
        lambda *args, **kwargs: [existing],
    )
    monkeypatch.setattr(
        git_ops,
        "gh_issue_create",
        lambda *args, **kwargs: pytest.fail("must reuse complete alias coverage"),
    )
    publisher = IssuePublisher.connect(tmp_path, repo_slug="acme/widgets")

    result = publisher.publish(
        package_id="regrouped-package",
        title="Reuse the existing handler",
        plan_path=plan_path,
        member_aliases=("member-v1:aaa", "member-v1:bbb"),
    )

    assert result.disposition == "existing"


def test_partial_member_alias_overlap_fails_instead_of_creating_duplicate_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# Expanded plan\n", encoding="utf-8")
    existing = _issue("older-package")
    existing["body"] = issue_body(
        "older-package",
        "old plan",
        member_aliases=("member-v1:shared",),
    )
    monkeypatch.setattr(
        git_ops,
        "gh_issue_list_strict",
        lambda *args, **kwargs: [existing],
    )
    monkeypatch.setattr(
        git_ops,
        "gh_issue_create",
        lambda *args, **kwargs: pytest.fail("must not create overlapping work"),
    )
    publisher = IssuePublisher.connect(tmp_path, repo_slug="acme/widgets")

    with pytest.raises(ImprovePublishError, match="partially covers"):
        publisher.publish(
            package_id="expanded-package",
            title="Expand reuse cleanup",
            plan_path=plan_path,
            member_aliases=("member-v1:shared", "member-v1:new"),
        )


def test_matching_package_marker_cannot_hide_stale_member_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# Expanded plan\n", encoding="utf-8")
    existing = _issue("same-package")
    existing["body"] = issue_body(
        "same-package",
        "old plan",
        member_aliases=("member-v1:old",),
    )
    monkeypatch.setattr(
        git_ops,
        "gh_issue_list_strict",
        lambda *args, **kwargs: [existing],
    )
    monkeypatch.setattr(
        git_ops,
        "gh_issue_create",
        lambda *args, **kwargs: pytest.fail("must not publish stale coverage"),
    )
    publisher = IssuePublisher.connect(tmp_path, repo_slug="acme/widgets")

    with pytest.raises(ImprovePublishError, match="stale or overlapping"):
        publisher.publish(
            package_id="same-package",
            title="Expanded cleanup",
            plan_path=plan_path,
            member_aliases=("member-v1:old", "member-v1:new"),
        )


def test_colliding_member_aliases_require_every_raw_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# Two distinct cleanups\n", encoding="utf-8")
    existing = _issue("same-package")
    existing["body"] = issue_body(
        "same-package",
        "one cleanup",
        member_aliases=("member-v1:shared",),
        member_fingerprints=("raw-first",),
    )
    monkeypatch.setattr(
        git_ops,
        "gh_issue_list_strict",
        lambda *args, **kwargs: [existing],
    )
    monkeypatch.setattr(
        git_ops,
        "gh_issue_create",
        lambda *args, **kwargs: pytest.fail("one alias cannot cover two members"),
    )
    publisher = IssuePublisher.connect(tmp_path, repo_slug="acme/widgets")

    with pytest.raises(ImprovePublishError, match="stale or overlapping"):
        publisher.publish(
            package_id="same-package",
            title="Two distinct cleanups",
            plan_path=plan_path,
            member_aliases=("member-v1:shared", "member-v1:shared"),
            member_fingerprints=("raw-first", "raw-second"),
        )

    assert member_fingerprint_marker("raw-second") not in str(existing["body"])


def test_publish_creates_issue_with_the_complete_local_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.md"
    plan = "# Complete plan\n\n- Delete the redundant adapter.\n"
    plan_path.write_text(plan, encoding="utf-8")
    monkeypatch.setattr(git_ops, "gh_issue_list_strict", lambda *args, **kwargs: [])
    captured: dict[str, object] = {}

    def create(repo: Path, **kwargs: object) -> str:
        captured.update(repo=repo, **kwargs)
        return "https://github.com/acme/widgets/issues/12"

    monkeypatch.setattr(git_ops, "gh_issue_create", create)
    publisher = IssuePublisher.connect(tmp_path, repo_slug="acme/widgets")

    result = publisher.publish(
        package_id="delete-adapter",
        title="Delete the redundant adapter",
        plan_path=plan_path,
    )

    assert result.disposition == "created"
    assert result.issue_url.endswith("/12")
    assert captured["repo"] == tmp_path
    assert captured["repo_slug"] == "acme/widgets"
    assert captured["title"] == "Delete the redundant adapter"
    assert captured["body"] == issue_body("delete-adapter", plan)


def test_ambiguous_create_failure_reconciles_before_returning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# Complete plan\n", encoding="utf-8")
    lookups = iter([[], [_issue("reuse-handler", number=19)]])
    monkeypatch.setattr(
        git_ops,
        "gh_issue_list_strict",
        lambda *args, **kwargs: next(lookups),
    )
    monkeypatch.setattr(
        git_ops,
        "gh_issue_create",
        lambda *args, **kwargs: (_ for _ in ()).throw(git_ops.GitTimeoutError("response lost")),
    )
    publisher = IssuePublisher.connect(tmp_path, repo_slug="acme/widgets")

    result = publisher.publish(
        package_id="reuse-handler",
        title="Reuse the existing handler",
        plan_path=plan_path,
    )

    assert result.disposition == "reconciled"
    assert result.issue_url.endswith("/19")


def test_ambiguous_create_failure_without_a_marker_remains_a_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# Complete plan\n", encoding="utf-8")
    monkeypatch.setattr(git_ops, "gh_issue_list_strict", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        git_ops,
        "gh_issue_create",
        lambda *args, **kwargs: (_ for _ in ()).throw(git_ops.GitTimeoutError("response lost")),
    )
    publisher = IssuePublisher.connect(tmp_path, repo_slug="acme/widgets")

    with pytest.raises(ImprovePublishError, match="no matching issue"):
        publisher.publish(
            package_id="reuse-handler",
            title="Reuse the existing handler",
            plan_path=plan_path,
        )


def test_duplicate_package_markers_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# Complete plan\n", encoding="utf-8")
    monkeypatch.setattr(
        git_ops,
        "gh_issue_list_strict",
        lambda *args, **kwargs: [
            _issue("reuse-handler", number=2),
            _issue("reuse-handler", number=3),
        ],
    )
    monkeypatch.setattr(
        git_ops,
        "gh_issue_create",
        lambda *args, **kwargs: pytest.fail("ambiguous state must not create"),
    )
    publisher = IssuePublisher.connect(tmp_path, repo_slug="acme/widgets")

    with pytest.raises(ImprovePublishError, match="Multiple GitHub issues"):
        publisher.publish(
            package_id="reuse-handler",
            title="Reuse the existing handler",
            plan_path=plan_path,
        )


def test_strict_issue_lookup_paginates_and_filters_pull_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    rows: list[dict[str, object]] = [
        {
            "number": number,
            "title": f"Issue {number}",
            "body": None,
            "html_url": f"https://github.com/acme/widgets/issues/{number}",
            "state": "closed" if number % 2 else "open",
        }
        for number in range(1, 102)
    ]
    rows.append(
        {
            "number": 102,
            "title": "A pull request",
            "body": "",
            "html_url": "https://github.com/acme/widgets/pull/102",
            "state": "open",
            "pull_request": {"url": "api"},
        }
    )

    def api(repo: Path, endpoint: str, **kwargs: object) -> list[dict[str, object]]:
        captured.update(repo=repo, endpoint=endpoint, **kwargs)
        return rows

    monkeypatch.setattr(git_ops, "gh_api", api)

    issues = git_ops.gh_issue_list_strict(
        tmp_path,
        state="all",
        repo_slug="acme/widgets",
    )

    assert len(issues) == 101
    assert issues[0]["body"] == ""
    assert captured["paginate"] is True
    assert captured["jq"] == ".[]"
    assert captured["idempotent"] is True
    assert "state=all" in str(captured["endpoint"])
    assert "per_page=100" in str(captured["endpoint"])


def test_runner_treats_configured_improve_as_a_posting_flow() -> None:
    enabled = RunConfig(
        flow_name="improve",
        file_config=DaydreamFileConfig(improve_github_publish_issues=True),
    )
    disabled = RunConfig(flow_name="improve", file_config=DaydreamFileConfig())

    assert runner._run_posts_to_github(enabled) is True
    assert runner._run_posts_to_github(disabled) is False
