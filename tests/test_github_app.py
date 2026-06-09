import subprocess
from pathlib import Path
from unittest.mock import patch

from daydream import git_ops


def test_run_gh_injects_token_env_when_set():
    """When the token-env singleton is set, _run_gh passes it to subprocess.run."""
    captured = {}

    def spy_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    git_ops.set_gh_token_env({"GH_TOKEN": "ghs_test123", "PATH": "/usr/bin"})
    try:
        with patch("subprocess.run", side_effect=spy_run):
            git_ops._run_gh(Path("/tmp"), ["version"])
    finally:
        git_ops.reset_gh_token_env()

    assert captured["env"]["GH_TOKEN"] == "ghs_test123"


def test_run_gh_passes_none_env_when_unset():
    """With no token-env set, _run_gh passes env=None (parent inheritance)."""
    captured = {}

    def spy_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    git_ops.reset_gh_token_env()
    with patch("subprocess.run", side_effect=spy_run):
        git_ops._run_gh(Path("/tmp"), ["version"])

    assert captured.get("env") is None


def test_token_env_accessors_roundtrip():
    """set/get/reset behave as a simple module singleton."""
    git_ops.reset_gh_token_env()
    assert git_ops.get_gh_token_env() is None
    git_ops.set_gh_token_env({"GH_TOKEN": "x"})
    assert git_ops.get_gh_token_env() == {"GH_TOKEN": "x"}
    git_ops.reset_gh_token_env()
    assert git_ops.get_gh_token_env() is None
