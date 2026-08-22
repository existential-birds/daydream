"""Tests for ``daydream benchmark import-prs``.

Task 0 spike: the load-bearing claim that the importer's ``gh``/``git``
calls, made through :mod:`daydream.git_ops` with ``cwd=<workspace root>``,
are intercepted by the in-process ``fake_gh`` router. Subsequent
collection/normalization/projection/orchestration tasks build on this seam.
"""

import json

import pytest


def test_preflight_gh_and_ls_remote_route_through_fake(tmp_path, fake_gh):
    from daydream.benchmark import github_import as gi

    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    ws = tmp_path / "ws"
    ws.mkdir()
    status = gi._run_gh_preflight_status(ws)       # gh auth status --hostname github.com
    login = gi._run_gh_api_user(ws)                # gh api user
    cred = gi._gh_auth_git_credential(ws)          # gh auth git-credential
    refs = gi._git_ls_remote(ws, "https://github.com/o/r.git")  # git ls-remote
    assert status.returncode == 0
    assert login == {"login": "octocat", "type": "User"}
    assert "password=" in cred
    assert "refs/heads/head" in refs