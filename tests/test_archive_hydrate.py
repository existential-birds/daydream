"""Unit/component tests for the #982 hydrate module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from daydream.archive import hydrate
from daydream.archive import hydrate_rules
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


class TestCurationManifestSchema:
    SCHEMA = Path(hydrate_rules.__file__).parent.parent / "training" / "schema" / "curation-manifest-v1.json"
    FIXTURE = Path(__file__).parent / "fixtures" / "training" / "curation-manifest-v1-fixture.json"

    def _validate(self, instance: dict) -> None:
        from jsonschema import Draft202012Validator

        Draft202012Validator(json.loads(self.SCHEMA.read_text())).validate(instance)

    def test_fixture_validates(self) -> None:
        self._validate(json.loads(self.FIXTURE.read_text()))

    def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        doc = json.loads(self.FIXTURE.read_text())
        doc["batches"][0]["artifact_relpath"] = str(tmp_path)  # absolute VM-local path
        with pytest.raises(Exception):
            self._validate(doc)

    def test_schema_version_mismatch_flagged(self) -> None:
        doc = json.loads(self.FIXTURE.read_text())
        doc["schema_version"] = "999"
        with pytest.raises(Exception):
            self._validate(doc)

    def test_required_contract_fields(self) -> None:
        doc = json.loads(self.FIXTURE.read_text())
        for field in ("schema_version", "source_hub_commit", "curation_id",
                      "sanitizer_version", "hydration_index_schema_version",
                      "admission_policy_version", "batches"):
            assert field in doc
        batch = doc["batches"][0]
        assert {"session_id", "content_digest", "status", "reason_code",
                "artifact_relpath"} <= set(batch)


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
