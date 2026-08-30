"""Unit/component tests for the #982 hydrate module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from daydream.archive import hydrate, hydrate_rules
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

    def _validate(self, instance: dict[str, object]) -> None:
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


class TestHydrateRules:
    def test_curation_id_deterministic(self) -> None:
        kwargs = dict(source_commit="a" * 40, sanitizer_version="1.0.0",
                      index_schema_version="1", admission_policy_version="1")
        first = hydrate_rules.derive_curation_id(**kwargs)
        assert first == hydrate_rules.derive_curation_id(**kwargs)
        assert first == hydrate_rules.derive_curation_id(**kwargs)  # stable across calls
        assert hydrate_rules.CURATION_ID_RE.fullmatch(first)

    def test_curation_id_inputs_sensitive(self) -> None:
        base = dict(source_commit="a" * 40, sanitizer_version="1.0.0",
                    index_schema_version="1", admission_policy_version="1")
        ids = {hydrate_rules.derive_curation_id(**{**base, k: v})
               for k, v in [("source_commit", "b" * 40), ("sanitizer_version", "1.0.1"),
                            ("admission_policy_version", "2")]}
        assert len(ids) == 3  # every input changes the id
        assert hydrate_rules.derive_curation_id(**base) not in ids

    def test_fixture_exclusion_reason_codes(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.json").write_text('{"session_id": "s", "source_path": "/tmp/pytest-of-user/x"}')
        codes = hydrate_rules.fixture_exclusion_codes(tmp_path)
        assert "fixture_pytest_path" in codes

    def test_pipeline_status_policy(self) -> None:
        # evidence present → revalidated value; absent → stable code, never success
        assert hydrate_rules.legacy_pipeline_status(
            pipeline_status="unknown", deep_artifacts={"status": "succeeded"}) == "succeeded"
        assert hydrate_rules.legacy_pipeline_status(
            pipeline_status="unknown", deep_artifacts=None) == ("excluded", "pipeline_status_evidence_absent")
        assert hydrate_rules.legacy_pipeline_status(pipeline_status="failed", deep_artifacts=None) == "failed"


class TestResolveRevision:
    def test_exact_sha_passes_through(self) -> None:
        hub = make_fake_hub(Path("/tmp"))
        hub.commit_revision("a" * 40)
        assert hydrate.resolve_source_revision(hub, "a" * 40, exploratory=False) == "a" * 40

    def test_short_prefix_resolves_to_full_sha(self) -> None:
        hub = make_fake_hub(Path("/tmp"))
        hub.commit_revision("abcdef1234567890" + "0" * 24)
        assert hydrate.resolve_source_revision(hub, "abcdef12", exploratory=False) == "abcdef1234567890" + "0" * 24

    def test_moving_branch_rejected_without_optin(self) -> None:
        hub = make_fake_hub(Path("/tmp"))
        hub.commit_revision("a" * 40, ref="main")  # branch ref, not immutable
        with pytest.raises(hydrate.MovingBranchError, match="exploratory"):
            hydrate.resolve_source_revision(hub, "main", exploratory=False)

    def test_moving_branch_allowed_with_exploratory_flag(self) -> None:
        hub = make_fake_hub(Path("/tmp"))
        hub.commit_revision("b" * 40, ref="main")
        assert hydrate.resolve_source_revision(hub, "main", exploratory=True) == "b" * 40

    def test_unknown_revision_fails_closed(self) -> None:
        hub = make_fake_hub(Path("/tmp"))
        with pytest.raises(hydrate.HydrationError):
            hydrate.resolve_source_revision(hub, "deadbeef" * 5, exploratory=False)

    def test_ambiguous_prefix_fails_closed(self) -> None:
        hub = make_fake_hub(Path("/tmp"))
        hub.commit_revision("ab12" + "1" * 36)
        hub.commit_revision("ab12" + "2" * 36)
        with pytest.raises(hydrate.HydrationError, match="ambiguous"):
            hydrate.resolve_source_revision(hub, "ab12", exploratory=False)

    def test_client_error_is_redacted(self) -> None:
        class LeakyHub(FakeHub):
            def repo_info(self, revision: str | None = None) -> hydrate.RepoInfo:
                raise hydrate.HubDownloadError("boom https://user:hf_token_secret@huggingface.co/x")

        hub = LeakyHub(repo_id="org/private-ds")
        with pytest.raises(hydrate.HydrationError) as excinfo:
            hydrate.resolve_source_revision(hub, "c" * 40, exploratory=False)
        assert "hf_token_secret" not in str(excinfo.value)
