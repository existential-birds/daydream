"""Tests for lazy ``base_sha`` materialization in older archive manifests.

Each test monkeypatches ``daydream.git_ops.merge_base`` to keep the unit
under test pure — no shelling out, no live clones. The function-under-test
calls that exact symbol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.training.base_sha import materialize_base_sha


def _write_manifest(
    manifest_path: Path,
    *,
    base_sha: str | None,
    head_sha: str,
    branch: str,
) -> None:
    """Write the minimal manifest needed for base-SHA materialization tests."""
    manifest_path.write_text(
        json.dumps(
            {
                "code_context": {
                    "base_sha": base_sha,
                    "base_branch": "main",
                    "head_sha": head_sha,
                    "branch": branch,
                    "changed_files": [],
                },
                "git": {"base_branch": "main", "head_sha": head_sha},
            }
        )
    )


def test_materialize_writes_sha_into_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve a missing merge base and persist it into the archive manifest."""
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, base_sha=None, head_sha="abc123", branch="feat/x")
    monkeypatch.setattr(
        "daydream.git_ops.merge_base",
        lambda repo, base, head: "deadbeefcafef00d" if (base, head) == ("main", "abc123") else None,
    )
    result = materialize_base_sha(manifest_path, repo_clone=tmp_path / "fake-clone")
    assert result == "deadbeefcafef00d"
    rewritten = json.loads(manifest_path.read_text())
    assert rewritten["code_context"]["base_sha"] == "deadbeefcafef00d"


def test_materialize_returns_none_when_merge_base_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, base_sha=None, head_sha="abc", branch="x")
    monkeypatch.setattr("daydream.git_ops.merge_base", lambda *a, **k: None)
    result = materialize_base_sha(manifest_path, repo_clone=tmp_path / "fake-clone")
    assert result is None
    assert json.loads(manifest_path.read_text())["code_context"]["base_sha"] is None


def test_materialize_is_noop_when_base_sha_already_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, base_sha="existing-sha", head_sha="abc", branch="x")
    called: list[bool] = []

    def _record_merge_base(*a: Any, **k: Any) -> str:
        called.append(True)
        return "should-not-be-used"

    monkeypatch.setattr("daydream.git_ops.merge_base", _record_merge_base)
    result = materialize_base_sha(manifest_path, repo_clone=tmp_path / "fake-clone")
    assert result == "existing-sha"
    assert called == []
