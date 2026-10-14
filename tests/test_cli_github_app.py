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
from typing import Any

import pytest

from daydream import cli
from daydream.backends import ResultEvent, TextEvent
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import bare_remote, git
from tests.harness.rsa import generate_rsa_pem
from tests.harness.stub_backend import review_stage_result
from tests.test_github_app_integration import _minimal_backend

_TOKEN = "ghs_scoped_installation_test_token"
_AMBIENT = "ghp_unusable_ambient_test_token"


@dataclass
class _Transport:
    gh: FakeGh
    requests: list[tuple[Path, list[str], dict[str, str]]] = field(default_factory=list, repr=False)
    require_app: bool = True


@pytest.fixture
def transport(fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch) -> _Transport:
    boundary = _Transport(fake_gh)
    fake_gh.set_response("GET", "app/installations", [
        {"id": 71, "account": {"login": "acme"}, "app_slug": "review-app"},
    ])
    fake_gh.set_response("POST", "app/installations/71/access_tokens", {
        "token": _TOKEN, "expires_at": "2099-01-01T00:00:00Z",
    })
    fake_gh.set_response("GET", "user", {"login": "ambient-user"})
    fake_gh.set_response("repo-view", value="acme/widgets")
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
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: _minimal_backend())
    return boundary


def _run(repo: Path, trajectory: Path, *flags: str) -> int | str | None:
    with pytest.raises(SystemExit) as result:
        cli.main([str(repo), "--shallow", "--stack", "python", "--non-interactive",
                  "--trajectory", str(trajectory), "--no-capture-data", *flags])
    return result.value.code


def _assert_app_requests(transport: _Transport) -> None:
    assert transport.requests
    for _cwd, argv, environment in transport.requests:
        assert environment["GH_TOKEN"] != _AMBIENT, argv
        assert environment["GH_REPO"] == "acme/widgets"
        assert environment["GH_HOST"] == "github.com"
        assert "GITHUB_TOKEN" not in environment
        assert "DAYDREAM_APP_PRIVATE_KEY" not in environment
    assert os.environ["GH_TOKEN"] == _AMBIENT
    assert os.environ["GITHUB_TOKEN"] == "ambient-alternate-token"


@pytest.mark.parametrize("origin,override", [
    ("https://github.com/acme/widgets.git", None),
    ("git@github.com:acme/widgets.git", None),
    ("ssh://git@github.com/acme/widgets.git", None),
    ("https://github.com/other/disagrees.git", "acme/widgets"),
    ("https://github.com/other/disagrees.git", "github.com/acme/widgets"),
    (None, "https://github.com/acme/widgets"),
])
def test_cli_app_bootstrap_without_ambient_auth(
    feature_branch_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    transport: _Transport, origin: str | None, override: str | None,
) -> None:
    repo = feature_branch_repo
    if origin is not None:
        git(repo, "remote", "add", "origin", origin)
    if override is not None:
        monkeypatch.setenv("GH_REPO", override)
    invoking = tmp_path / "unrelated-cwd"
    invoking.mkdir()
    monkeypatch.chdir(invoking)
    output = tmp_path / "trajectory.json"

    assert _run(repo, output, "--comment", "--base", "main") == 0

    document = json.loads(output.read_text())
    assert document["extra"]["pr_repo"] == "acme/widgets"
    assert (repo / ".review-output.md").is_file()
    assert transport.gh.calls("POST", "app/installations/71/access_tokens")[0].payload == {"repositories": ["widgets"]}
    _assert_app_requests(transport)


@pytest.mark.parametrize("origin,override,host", [
    (None, None, None),
    ("https://gitlab.com/acme/widgets.git", None, None),
    ("https://secret-user:remote-secret@foreign.invalid/acme/widgets.git", None, None),
    ("https://github.com/acme/widgets/extra", None, None),
    ("https://github.com/acme/widgets.git", "malformed", None),
    ("https://github.com/acme/widgets.git", "foreign.invalid/acme/widgets", None),
    ("https://github.com/acme/widgets.git", None, "foreign.invalid"),
])
def test_cli_invalid_identity_aborts_before_mint(
    feature_branch_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    transport: _Transport, capsys: pytest.CaptureFixture[str],
    origin: str | None, override: str | None, host: str | None,
) -> None:
    if origin is not None:
        git(feature_branch_repo, "remote", "add", "origin", origin)
    if override is not None:
        monkeypatch.setenv("GH_REPO", override)
    if host is not None:
        monkeypatch.setenv("GH_HOST", host)
    output = tmp_path / "trajectory.json"

    assert _run(feature_branch_repo, output, "--comment") == 1

    captured = capsys.readouterr()
    assert "Cannot determine owner/repo" in captured.out + captured.err
    assert "remote-secret" not in captured.out + captured.err
    assert transport.requests == []
    assert not (feature_branch_repo / ".review-output.md").exists()


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


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_app_pr_target_survives_detached_workspace(
    feature_branch_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    transport: _Transport, explicit: bool,
) -> None:
    repo = feature_branch_repo
    remote = bare_remote(tmp_path / "origin.git")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "origin", "main", "feature")
    monkeypatch.setenv("GH_REPO", "acme/widgets")
    head = git(repo, "rev-parse", "HEAD")
    transport.gh.set_response("pr-view", value={
        "number": 73, "headRefOid": head, "headRefName": "feature", "baseRefName": "main",
        "url": "https://github.com/acme/widgets/pull/73", "headRepository": {"name": "widgets"},
        "headRepositoryOwner": {"login": "acme"},
    })
    def respond(cwd: Any, prompt: str, output_schema: Any = None, *args: Any) -> list[Any]:
        issue = {"id": 1, "file": "main.py", "line": 2, "severity": "medium",
                 "confidence": "HIGH", "description": "Changed greeting breaks callers",
                 "rationale": "Callers require world", "evidence": "return 'universe'"}
        return [TextEvent(text="Review complete."), ResultEvent(
            structured_output=review_stage_result(prompt, [issue]) if output_schema else None, continuation=None,
        )]

    monkeypatch.setattr("daydream.runner.create_backend", lambda *a, **kw: ScriptedBackend(responder=respond))
    output = tmp_path / "trajectory.json"

    assert _run(repo, output, "--comment", "--worktree", "--base", "main",
                *(["--pr-number", "73"] if explicit else [])) == 0

    document = json.loads(output.read_text())
    assert document["extra"]["pr_number"] == 73
    assert document["extra"]["pr_repo"] == "acme/widgets"
    views = [(cwd, argv, env) for cwd, argv, env in transport.requests if argv[1:3] == ["pr", "view"]]
    assert views
    if not explicit:
        assert views[0][0] == repo
        assert views[0][2]["GH_TOKEN"] == _TOKEN
    else:
        assert all(argv[3] == "73" for _cwd, argv, _env in views)
    assert any(cwd != repo for cwd, _argv, _env in views)
    assert transport.gh.calls("POST", "repos/acme/widgets/pulls/73/reviews")
    _assert_app_requests(transport)
    assert git(repo, "branch", "--show-current") == "feature"


@pytest.mark.parametrize("app_configured,mode", [(False, "--review"), (True, "--review"), (False, "--comment")])
def test_cli_read_only_preserves_ambient_auth_without_minting(
    feature_branch_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    transport: _Transport, app_configured: bool, mode: str,
) -> None:
    transport.require_app = False
    if not app_configured:
        monkeypatch.delenv("DAYDREAM_APP_ID")
        monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY")
    output = tmp_path / "trajectory.json"

    assert _run(feature_branch_repo, output, mode, "--base", "main") == 0

    assert output.is_file()
    assert (feature_branch_repo / ".review-output.md").is_file()
    assert not transport.gh.calls("POST", "app/installations/71/access_tokens")
    assert all(env["GH_TOKEN"] == _AMBIENT for _cwd, _argv, env in transport.requests)


def test_cli_ambiguous_origin_aborts_before_mint(
    feature_branch_repo: Path, tmp_path: Path, transport: _Transport,
) -> None:
    git(feature_branch_repo, "remote", "add", "origin", "https://github.com/acme/widgets.git")
    git(feature_branch_repo, "config", "--add", "remote.origin.url", "https://github.com/acme/other.git")

    assert _run(feature_branch_repo, tmp_path / "trajectory.json", "--comment") == 1

    assert transport.requests == []
    assert not (feature_branch_repo / ".review-output.md").exists()


def test_cli_no_app_retains_identity_for_ssh_alias(
    feature_branch_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transport: _Transport,
) -> None:
    transport.require_app = False
    monkeypatch.delenv("DAYDREAM_APP_ID")
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY")
    git(feature_branch_repo, "remote", "add", "origin", "git@work-github:acme/widgets.git")
    output = tmp_path / "trajectory.json"

    assert _run(feature_branch_repo, output, "--review", "--base", "main") == 0

    assert json.loads(output.read_text())["extra"]["pr_repo"] == "acme/widgets"
    assert (feature_branch_repo / ".review-output.md").is_file()
    assert not transport.gh.calls("POST", "app/installations/71/access_tokens")
    assert all(env["GH_TOKEN"] == _AMBIENT for _cwd, _argv, env in transport.requests)


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
