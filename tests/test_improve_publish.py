"""Focused contracts for Improve's headless GitHub issue publisher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.git_ops import github as git_github
from daydream.improve.publish import (
    ImprovePublishError,
    IssuePublisher,
    PublishResult,
    issue_body,
    member_fingerprint_marker,
    package_marker,
)


def _issue(package_id: str, *, state: str = "open", number: int = 7) -> dict[str, object]:
    return {"number": number, "title": "Improve code reuse",
        "body": f"{package_marker(package_id)}\n\nplan",
        "url": f"https://github.com/acme/widgets/issues/{number}", "state": state,
    }

def _stub_existing_issue(monkeypatch: pytest.MonkeyPatch, issues: list[dict[str, object]], *, create_failure: str,
) -> None:
    """Stub the issue listing and make any create attempt a test failure."""
    monkeypatch.setattr(git_ops, "gh_issue_list_strict", lambda *args, **kwargs: issues)
    monkeypatch.setattr(git_ops, "gh_issue_create", lambda *args, **kwargs: pytest.fail(create_failure),)

def _publish(tmp_path: Path, *, package_id: str, title: str,
    plan: str = "# Complete plan\n",
    **kwargs: Any,
) -> PublishResult:
    """Write a plan and publish it through a connected test publisher."""
    plan_path = tmp_path / "plan.md"
    plan_path.write_text(plan, encoding="utf-8")
    publisher = IssuePublisher.connect(tmp_path, repo_slug="acme/widgets")
    return publisher.publish(package_id=package_id, title=title, plan_path=plan_path, **kwargs)



def test_package_marker_rejects_comment_injection() -> None:
    with pytest.raises(ValueError, match="package_id"):
        package_marker("package -->\nmalicious")

def test_connect_fails_closed_when_existing_issues_cannot_be_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        git_ops, "gh_issue_list_strict", lambda *args, **kwargs: (_ for _ in ()).throw(git_ops.GitError("offline")),
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

def test_connect_infers_repository_and_lists_open_and_closed_issues(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(git_ops, "gh_repo_view", lambda repo, **_kwargs: ("acme", "widgets"))
    def list_strict(repo: Path, *, state: str, repo_slug: str, **_kwargs: Any) -> list[dict[str, Any]]:
        captured.update(repo=repo, state=state, repo_slug=repo_slug)
        return []
    monkeypatch.setattr(git_ops, "gh_issue_list_strict", list_strict)
    publisher = IssuePublisher.connect(tmp_path)
    assert publisher.repo_slug == "acme/widgets"
    assert captured == {"repo": tmp_path, "state": "all", "repo_slug": "acme/widgets",}

def test_existing_closed_issue_is_reused_without_creating_a_duplicate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _stub_existing_issue(
        monkeypatch, [_issue("reuse-handler", state="closed")], create_failure="must not create a duplicate issue",
    )
    result = _publish(tmp_path, package_id="reuse-handler", title="Reuse the existing handler")
    assert result.disposition == "existing"
    assert result.issue_url.endswith("/7")

def test_all_member_aliases_reconcile_a_regrouped_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,) -> None:
    existing = _issue("older-package")
    existing["body"] = issue_body("older-package", "old plan", member_aliases=("member:aaa", "member:bbb"),)
    _stub_existing_issue(monkeypatch, [existing], create_failure="must reuse complete alias coverage")
    result = _publish(tmp_path, package_id="regrouped-package", title="Reuse the existing handler",
        member_aliases=("member:aaa", "member:bbb"),
    )
    assert result.disposition == "existing"

def test_partial_member_alias_overlap_fails_instead_of_creating_duplicate_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    existing = _issue("older-package")
    existing["body"] = issue_body("older-package", "old plan", member_aliases=("member:shared",),)
    _stub_existing_issue(monkeypatch, [existing], create_failure="must not create overlapping work")
    with pytest.raises(ImprovePublishError, match="partially covers"):
        _publish(tmp_path, package_id="expanded-package", title="Expand reuse cleanup",
            plan="# Expanded plan\n",
            member_aliases=("member:shared", "member:new"),
        )

def test_matching_package_marker_cannot_hide_stale_member_coverage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    existing = _issue("same-package")
    existing["body"] = issue_body("same-package", "old plan", member_aliases=("member:old",),)
    _stub_existing_issue(monkeypatch, [existing], create_failure="must not publish stale coverage")
    with pytest.raises(ImprovePublishError, match="stale or overlapping"):
        _publish(tmp_path, package_id="same-package", title="Expanded cleanup",
            plan="# Expanded plan\n",
            member_aliases=("member:old", "member:new"),
        )

def test_colliding_member_aliases_require_every_raw_fingerprint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    existing = _issue("same-package")
    existing["body"] = issue_body(
        "same-package", "one cleanup", member_aliases=("member:shared",), member_fingerprints=("raw-first",),
    )
    _stub_existing_issue(monkeypatch, [existing], create_failure="one alias cannot cover two members")

    with pytest.raises(ImprovePublishError, match="stale or overlapping"):
        _publish(tmp_path, package_id="same-package", title="Two distinct cleanups",
            plan="# Two distinct cleanups\n",
            member_aliases=("member:shared", "member:shared"), member_fingerprints=("raw-first", "raw-second"),
        )

    assert member_fingerprint_marker("raw-second") not in str(existing["body"])


def test_ambiguous_create_failure_reconciles_before_returning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,) -> None:
    lookups = iter([[], [_issue("reuse-handler", number=19)]])
    monkeypatch.setattr(git_ops, "gh_issue_list_strict", lambda *args, **kwargs: next(lookups),)
    monkeypatch.setattr(git_ops, "gh_issue_create",
        lambda *args, **kwargs: (_ for _ in ()).throw(git_ops.GitTimeoutError("response lost")),
    )
    result = _publish(tmp_path, package_id="reuse-handler", title="Reuse the existing handler")
    assert result.disposition == "reconciled"
    assert result.issue_url.endswith("/19")

def test_ambiguous_create_failure_without_a_marker_remains_a_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(git_ops, "gh_issue_list_strict", lambda *args, **kwargs: [])
    monkeypatch.setattr(git_ops, "gh_issue_create",
        lambda *args, **kwargs: (_ for _ in ()).throw(git_ops.GitTimeoutError("response lost")),
    )
    with pytest.raises(ImprovePublishError, match="no matching issue"):
        _publish(tmp_path, package_id="reuse-handler", title="Reuse the existing handler")

def test_duplicate_package_markers_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,) -> None:
    _stub_existing_issue(monkeypatch, [_issue("reuse-handler", number=2), _issue("reuse-handler", number=3)],
        create_failure="ambiguous state must not create",
    )
    with pytest.raises(ImprovePublishError, match="Multiple GitHub issues"):
        _publish(tmp_path, package_id="reuse-handler", title="Reuse the existing handler")

def test_strict_issue_lookup_paginates_and_filters_pull_requests(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    rows: list[dict[str, object]] = [{"number": number, "title": f"Issue {number}", "body": None,
            "html_url": f"https://github.com/acme/widgets/issues/{number}", "state": "closed" if number % 2 else "open",
        }
        for number in range(1, 102)
    ]
    rows.append({"number": 102, "title": "A pull request", "body": "",
            "html_url": "https://github.com/acme/widgets/pull/102", "state": "open", "pull_request": {"url": "api"},
        }
    )

    def api(repo: Path, endpoint: str, **kwargs: object) -> list[dict[str, object]]:
        captured.update(repo=repo, endpoint=endpoint, **kwargs)
        return rows

    monkeypatch.setattr(git_github, "gh_api", api)

    issues = git_ops.gh_issue_list_strict(tmp_path, state="all", repo_slug="acme/widgets",)

    assert len(issues) == 101
    assert issues[0]["body"] == ""
    assert captured["paginate"] is True
    assert captured["jq"] == ".[]"
    assert captured["idempotent"] is True
    assert "state=all" in str(captured["endpoint"])
    assert "per_page=100" in str(captured["endpoint"])
