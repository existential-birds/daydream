"""Import normalized evidence from explicit private GitHub PRs.

Task 0 spike: prove the importer's ``gh``/``git`` calls route through
:mod:`daydream.git_ops` (so the in-process ``fake_gh`` router intercepts
them) before any collection logic is written. The functions here are the
thin preflight call sites; later tasks build the full
:func:`fetch_and_normalize` / :func:`preflight` / :func:`run_import_prs`
surface on top of them.
"""

from __future__ import annotations

import json
from pathlib import Path

from daydream import git_ops


def _run_gh_preflight_status(root: Path):
    """Run ``gh auth status --hostname github.com`` (exit code is the contract)."""
    return git_ops._run_gh(root, ["auth", "status", "--hostname", "github.com"])


def _run_gh_api_user(root: Path) -> dict:
    """Return the authenticated GitHub user record from ``gh api user``."""
    proc = git_ops._run_gh(root, ["api", "user"])
    if proc.returncode != 0:
        raise git_ops.GitError(f"gh api user failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def _gh_auth_git_credential(root: Path) -> str:
    """Run the command-scoped git credential helper and return its protocol text."""
    proc = git_ops._run_gh(root, ["auth", "git-credential"])
    if proc.returncode != 0:
        raise git_ops.GitError(f"gh auth git-credential failed: {proc.stderr.strip()}")
    return proc.stdout


def _git_ls_remote(root: Path, url: str) -> str:
    """Run an authenticated ``git ls-remote <url>`` and return the refs text."""
    proc = git_ops._run_git(root, ["ls-remote", url])
    return proc.stdout