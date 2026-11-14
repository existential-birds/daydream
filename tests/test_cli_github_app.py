"""CLI bootstrap uses local identity before any authenticated GitHub request.

Real Git, CLI event loop, workspace, JWT signing and artifacts; only GitHub's
subprocess transport and the model Backend are fake.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from daydream import cli
from daydream.dataset import LocalRecordStore
from daydream.reviews.identity import finding_marker
from tests.harness.dataset import read_records
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import bare_remote, git
from tests.harness.rsa import generate_rsa_pem
from tests.harness.stub_backend import StubBackend

_TOKEN = "ghs_scoped_installation_test_token"
_AMBIENT = "ghp_unusable_ambient_test_token"


@dataclass
class _Transport:
    gh: FakeGh
    requests: list[tuple[Path, list[str], dict[str, str]]] = field(default_factory=list, repr=False)
    require_app: bool = True


@pytest.fixture
def transport(fake_gh: FakeGh, feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch) -> _Transport:
    boundary = _Transport(fake_gh)
    fake_gh.set_response("GET", "app/installations", [
        {"id": 99, "account": {"login": "another-owner"}, "app_slug": "unrelated-app"},
        {"id": 71, "account": {"login": "AcMe"}, "app_slug": "review-app"},
    ])
    fake_gh.set_response("POST", "app/installations/71/access_tokens", {
        "token": _TOKEN, "expires_at": "2099-01-01T00:00:00Z",
    })
    fake_gh.set_response("GET", "user", {"login": "ambient-user"})
    fake_gh.set_response("repo-view", value="acme/widgets")
    fake_gh.serve_open_pr(feature_branch_repo)
    real_run = subprocess.run

    def request(args: Any, *positional: Any, **kwargs: Any) -> Any:
        if isinstance(args, (list, tuple)) and args and args[0] == "gh":
            environment = dict(kwargs.get("env") or os.environ)
            boundary.requests.append((Path(kwargs.get("cwd") or Path.cwd()), list(args), environment))
            if boundary.require_app and environment.get("GH_TOKEN") in (None, _AMBIENT):
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="ambient gh authentication unavailable")
        return real_run(args, *positional, **kwargs)

    monkeypatch.setattr(subprocess, "run", request)
    for name in ("GH_REPO", "GH_HOST", "DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GH_TOKEN", _AMBIENT)
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-alternate-token")
    monkeypatch.setenv("DAYDREAM_APP_ID", "42")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", generate_rsa_pem())
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: StubBackend(feature_branch_repo))
    return boundary


def _run(repo: Path, trajectory: Path, *flags: str) -> int | str | None:
    with pytest.raises(SystemExit) as result:
        cli.main([str(repo), "--non-interactive", "--trajectory", str(trajectory),
                  *(() if "--capture-data" in flags else ("--no-capture-data",)), *flags])
    return result.value.code


def _assert_app_requests(transport: _Transport) -> None:
    listing, exchange = transport.requests[:2]
    jwt_token = listing[2]["GH_TOKEN"]
    authorization = f"Authorization: Bearer {jwt_token}"
    assert listing[1] == ["gh", "api", "-H", authorization, "--paginate", "--jq", "(.[]) | @json", "/app/installations"]
    assert exchange[1][:8] == ["gh", "api", "-H", authorization, "/app/installations/71/access_tokens",
                               "--method", "POST", "--input"]
    assert len(exchange[1]) == 9 and exchange[0] == listing[0]
    assert exchange[2]["GH_TOKEN"] == jwt_token
    private_key = serialization.load_pem_private_key(os.environ["DAYDREAM_APP_PRIVATE_KEY"].encode(), password=None)
    decoded = pyjwt.decode(jwt_token, cast(rsa.RSAPublicKey, private_key.public_key()), algorithms=["RS256"])
    assert decoded["iss"] == "42" and decoded["exp"] - decoded["iat"] == 600
    for _cwd, argv, environment in transport.requests:
        assert environment["GH_TOKEN"] != _AMBIENT, argv
        assert environment["GH_REPO"] == "acme/widgets"
        assert environment["GH_HOST"] == "github.com"
        assert "GITHUB_TOKEN" not in environment
        assert "DAYDREAM_APP_PRIVATE_KEY" not in environment
    assert os.environ["GH_TOKEN"] == _AMBIENT
    assert os.environ["GITHUB_TOKEN"] == "ambient-alternate-token"


def test_cli_app_failure_is_redacted_without_ambient_fallback(
    feature_branch_repo: Path, tmp_path: Path, transport: _Transport, capsys: pytest.CaptureFixture[str],
) -> None:
    git(feature_branch_repo, "remote", "add", "origin", "https://github.com/acme/widgets.git")
    secret = "ghs_sensitive_installation_response_token"
    transport.gh.set_response("POST", "app/installations/71/access_tokens", {"__error__": secret})

    assert _run(feature_branch_repo, tmp_path / "trajectory.json", "--comment", "--verbose") == 1

    captured = capsys.readouterr()
    assert "App token resolution failed" in captured.out + captured.err
    assert secret not in captured.out + captured.err
    _assert_app_requests(transport)
    assert not (feature_branch_repo / ".review-output.md").exists()


def test_cli_invalid_app_key_aborts_without_exposing_secret(
    feature_branch_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    transport: _Transport, capsys: pytest.CaptureFixture[str],
) -> None:
    git(feature_branch_repo, "remote", "add", "origin", "https://github.com/acme/widgets.git")
    secret = "malformed-private-key-secret"
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", secret)

    assert _run(feature_branch_repo, tmp_path / "trajectory.json", "--comment", "--verbose") == 1

    captured = capsys.readouterr()
    assert "App token resolution failed" in captured.out + captured.err
    assert secret not in captured.out + captured.err
    assert transport.requests == []
    assert not (feature_branch_repo / ".review-output.md").exists()


def test_cli_no_app_uses_gh_selected_repository(
    feature_branch_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transport: _Transport,
) -> None:
    """CLI artifacts must follow gh's upstream selection even when origin is a fork.

    Alias-only coverage misses recognizable github.com origins winning over gh.
    """
    transport.require_app = False
    monkeypatch.delenv("DAYDREAM_APP_ID")
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY")
    git(feature_branch_repo, "remote", "add", "origin", "https://github.com/contributor/widgets.git")
    output = tmp_path / "trajectory.json"

    assert _run(feature_branch_repo, output, "--review", "--base", "main") == 0

    assert json.loads(output.read_text())["extra"]["pr_repo"] == "acme/widgets"
    assert (feature_branch_repo / ".review-output.md").is_file()
    assert not transport.gh.calls("POST", "app/installations/71/access_tokens")
    assert all(env["GH_TOKEN"] == _AMBIENT for _cwd, _argv, env in transport.requests)


def test_cli_requested_branch_ignores_current_checkout_pr(
    feature_branch_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    transport: _Transport,
) -> None:
    """CLI branch selection must bind capture and publication before detached review.

    Direct runner tests omit CLI provenance discovery, which can otherwise turn
    the current checkout's different PR into an explicit target for this run.
    """
    repo = feature_branch_repo
    current_head = git(repo, "rev-parse", "HEAD")
    base = git(repo, "rev-parse", "main")
    git(repo, "checkout", "-b", "requested", "main")
    (repo / "main.py").write_text("def hello():\n    return 'galaxy'\n")
    git(repo, "add", "main.py")
    git(repo, "commit", "-m", "requested branch change")
    requested_head = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "feature")
    remote = bare_remote(tmp_path / "origin.git")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "origin", "main", "feature", "requested")
    monkeypatch.setenv("GH_REPO", "acme/widgets")
    requested_pr = {
        "number": 73, "headRefOid": requested_head, "headRefName": "requested", "baseRefName": "main",
        "url": "https://github.com/acme/widgets/pull/73", "headRepository": {"name": "widgets"},
        "headRepositoryOwner": {"login": "acme"},
    }
    current_pr = {**requested_pr, "number": 7, "headRefOid": current_head, "headRefName": "feature",
                  "url": "https://github.com/acme/widgets/pull/7"}
    transport.gh.set_response("GET", "repos/acme/widgets/pulls/73", {
        "head": {"sha": requested_head}, "base": {"sha": base},
    })
    request = subprocess.run

    def route_pr_lookup(args: Any, *positional: Any, **kwargs: Any) -> Any:
        if isinstance(args, (list, tuple)) and args[:3] == ["gh", "pr", "view"]:
            transport.gh.serve_pr_view(requested_pr if args[3] == "73" else current_pr)
        elif isinstance(args, (list, tuple)) and args[:3] == ["gh", "pr", "list"]:
            branch = args[args.index("--head") + 1]
            transport.gh.set_response("pr-list", value=[
                pr for pr in (requested_pr, current_pr) if pr["headRefName"] == branch
            ])
        return request(args, *positional, **kwargs)

    monkeypatch.setattr(subprocess, "run", route_pr_lookup)
    output = tmp_path / "trajectory.json"
    store = LocalRecordStore(tmp_path / "records")
    assert _run(repo, output, "--comment", "--branch", "requested", "--base", "main", "--approve-on-clean",
                "--capture-data", "--dataset-store", str(store.root)) == 0

    assert json.loads(output.read_text())["extra"]["pr_number"] == 73
    # The sealed review, not the fake, attests which commit discovery and its fresh challenge analyzed.
    sealed = json.loads((repo / ".daydream/committed-review.json").read_text())
    assert sealed["analyzed_revision"]["head_sha"] == requested_head
    assert [entry["assignment"] for entry in sealed["execution"]] == ["discovery", "challenge"]
    assert sealed["terminal_result"]["analysis_state"] == "complete"
    assert [finding["title"] for finding in sealed["findings"]] == ["Sample issue"]
    reviews = transport.gh.calls("POST", "repos/acme/widgets/pulls/73/reviews")
    assert len(reviews) == 1 and reviews[0].payload["commit_id"] == requested_head
    # A confirmed finding asserts no severity, so an opted-in approval is withheld.
    assert reviews[0].payload["event"] == "COMMENT"
    (comment,) = reviews[0].payload["comments"]
    assert comment["path"] == "main.py" and "**Severity:**" not in comment["body"]
    # Capture and publication read the same canonical item, so they agree on its identity.
    (captured,) = read_records(store).runs
    (item,) = captured["findings"]["value"]["items"]
    assert finding_marker(item["fingerprint"]) in comment["body"]
    assert item["item_uid"] == sealed["findings"][0]["item_uid"]
    assert captured["findings"]["value"]["terminal_coverage"]["planned_scopes"][0]["scope_id"] == "committed:00000"
    assert not transport.gh.calls("POST", "repos/acme/widgets/pulls/7/reviews")
    _assert_app_requests(transport)
    # Branch PR discovery runs on the original checkout with the session's minted authentication.
    lookups = [(cwd, env) for cwd, argv, env in transport.requests if argv[:3] == ["gh", "pr", "list"]]
    assert lookups and all(cwd == repo and env["GH_TOKEN"] == _TOKEN for cwd, env in lookups)
    assert git(repo, "branch", "--show-current") == "feature"
    assert git(repo, "rev-parse", "HEAD") == current_head
