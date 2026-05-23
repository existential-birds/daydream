"""Tests for lazy ``base_sha`` materialization in older archive manifests.

Each test monkeypatches ``daydream.git_ops.merge_base`` to keep the unit
under test pure — no shelling out, no live clones. The function-under-test
calls that exact symbol.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daydream.training.base_sha import materialize_base_sha


def test_materialize_writes_sha_into_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "code_context": {
                    "base_sha": None,
                    "base_branch": "main",
                    "head_sha": "abc123",
                    "branch": "feat/x",
                    "changed_files": [],
                },
                "git": {"base_branch": "main", "head_sha": "abc123"},
            }
        )
    )
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
    manifest_path.write_text(
        json.dumps(
            {
                "code_context": {
                    "base_sha": None,
                    "base_branch": "main",
                    "head_sha": "abc",
                    "branch": "x",
                    "changed_files": [],
                },
                "git": {"base_branch": "main", "head_sha": "abc"},
            }
        )
    )
    monkeypatch.setattr("daydream.git_ops.merge_base", lambda *a, **k: None)
    result = materialize_base_sha(manifest_path, repo_clone=tmp_path / "fake-clone")
    assert result is None
    assert json.loads(manifest_path.read_text())["code_context"]["base_sha"] is None


def test_materialize_is_noop_when_base_sha_already_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "code_context": {
                    "base_sha": "existing-sha",
                    "base_branch": "main",
                    "head_sha": "abc",
                    "branch": "x",
                    "changed_files": [],
                },
                "git": {"base_branch": "main", "head_sha": "abc"},
            }
        )
    )
    called: list[bool] = []
    monkeypatch.setattr(
        "daydream.git_ops.merge_base",
        lambda *a, **k: called.append(True) or "should-not-be-used",
    )
    result = materialize_base_sha(manifest_path, repo_clone=tmp_path / "fake-clone")
    assert result == "existing-sha"
    assert called == []
