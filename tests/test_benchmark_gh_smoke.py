"""Opt-in live ``gh`` preflight smoke test (env-gated, never on by default).

Proves the real authenticated path against git + installed gh: preflight
resolves an opaque ``R_kgD...`` repository node id, ``ls-remote`` authenticates
through the real ``gh auth git-credential`` helper, and no credential leaks into
``benchmark.yaml``. Off by default — CI never runs this (no ``DAYDREAM_LIVE_GH``)
so the suite requires no network credentials.
"""

import json
import os
import shutil

import pytest

from daydream.benchmark import github_import as gi
from daydream.benchmark.storage import load_yaml_strict
from daydream.benchmark.workspace import init_workspace

_run = shutil.which("gh") is not None and os.environ.get("DAYDREAM_LIVE_GH") == "1"
pytestmark = pytest.mark.skipif(
    not _run, reason="live gh smoke test disabled (DAYDREAM_LIVE_GH=1 + gh required)"
)


def _seed_manifest(ws, repository: str) -> None:
    init_workspace(ws, repository, ["h1.example.com"], ["h2.example.com"])


def test_private_preflight_smoke_with_installed_gh(tmp_path):
    """A public repo the operator can read; proves the real authenticated path."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _seed_manifest(ws, "existential-birds/daydream")
    out = gi.preflight(ws, pr_count=0)
    assert out.login
    assert out.repository_id.startswith("R_kgD") and out.visibility in ("public", "private")
    raw = load_yaml_strict(ws / "benchmark.yaml")
    assert raw["source"]["repository_id"].startswith("R_kgD")
    assert "password=" not in json.dumps(raw)     # no credential leakage into the manifest