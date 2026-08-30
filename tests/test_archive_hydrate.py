"""Unit/component tests for the #982 hydrate module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from daydream.archive import hydrate
from daydream.archive.hydrate_client import FakeHub

SNAPSHOT = {
    "bundles/sess-a/manifest.json": b'{"session_id": "sess-a"}',
    "bundles/sess-a/trajectory.json": b"{}",
}


def make_fake_hub(tmp_path: Path) -> FakeHub:
    return FakeHub(repo_id="org/private-ds", private=True, files=dict(SNAPSHOT))


def test_fake_hub_roundtrip_and_revision(tmp_path: Path) -> None:
    hub = make_fake_hub(tmp_path)
    hub.commit_revision("abc123def4567890")  # fake pins a "commit sha"
    info = hub.repo_info(revision="abc123def4567890")
    assert info.private is True
    assert info.sha == "abc123def4567890"
    assert sorted(hub.list_repo_files()) == sorted(SNAPSHOT)
    data = hub.download_file("bundles/sess-a/manifest.json", revision="abc123def4567890")
    assert json.loads(data)["session_id"] == "sess-a"


def test_hf_client_missing_extra_is_fatal_and_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing huggingface_hub is fatal for hydrate (operator command), redacted, never leaks argv."""
    monkeypatch.setattr(hydrate, "_import_hf_hub", lambda: None)  # simulate ImportError
    with pytest.raises(hydrate.HubUnavailableError) as excinfo:
        hydrate._make_client("org/private-ds")
    msg = str(excinfo.value)
    assert "huggingface-hub" in msg
    assert "HF_TOKEN" not in msg  # no token material in the error path


def test_hf_client_requires_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(hydrate.HubUnavailableError, match="HF_TOKEN"):
        hydrate._make_client("org/private-ds", token_present=False)
