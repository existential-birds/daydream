"""Unit/component tests for the #982 hydrate module."""
from __future__ import annotations

import hashlib
import json
import pathlib
from pathlib import Path
from typing import Any

import pytest

from daydream.archive import hydrate, hydrate_rules
from daydream.archive.hydrate_client import FakeHub

SNAPSHOT = {
    "bundles/sess-a/manifest.json": b'{"session_id": "sess-a"}',
    "bundles/sess-a/trajectory.json": b"{}",
}


def make_fake_hub(tmp_path: Path) -> FakeHub:
    hub = FakeHub(repo_id="org/private-ds", private=True, files=dict(SNAPSHOT))
    hub.commit_revision("a" * 40)
    return hub


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

    def test_derive_curation_id_v2_binds_policy_inputs(self) -> None:
        from daydream.archive.hydrate_rules import derive_curation_id_v2
        base = dict(
            source_commit="a" * 40,
            policy_digest="d" * 64,
            policy_version="prod-1",
            allow_copyleft=frozenset({"acme/widget"}),
            exclusions_digest="e" * 64,
        )
        cid = derive_curation_id_v2(**base, decisions_digest="f" * 64, distribution_digest="0" * 64)
        assert cid.startswith("cur-") and len(cid) == 20 and cid[4:].isalnum()
        # Any change to any bound input changes the id (identity-breaking by design).
        for key, value in [
            ("policy_digest", "d" * 63 + "0"),
            ("policy_version", "prod-2"),
            ("allow_copyleft", frozenset({"acme/other"})),
            ("exclusions_digest", "e" * 63 + "0"),
            ("decisions_digest", "f" * 63 + "0"),
            ("distribution_digest", "0" * 63 + "1"),
        ]:
            assert derive_curation_id_v2(**{**base, key: value}) != cid
        # v1 ids are untouched: existing prefixes keep the old derivation.
        from daydream.archive.hydrate_rules import derive_curation_id
        assert derive_curation_id("a" * 40, "1", "1", "1") == derive_curation_id("a" * 40, "1", "1", "1")

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

    def test_symbolic_ref_not_case_folded(self) -> None:
        """Case-sensitive refs are never silently mapped to a differently-cased name."""
        hub = make_fake_hub(Path("/tmp"))
        hub.commit_revision("b" * 40, ref="main")
        with pytest.raises(hydrate.HydrationError, match="unknown revision"):
            hydrate.resolve_source_revision(hub, "Main", exploratory=True)  # no silent fold to 'main'
        hub.commit_revision("c" * 40, ref="Main")
        assert hydrate.resolve_source_revision(hub, "Main", exploratory=True) == "c" * 40
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


class TestDownloadSnapshot:
    def test_canonical_root_sessions_are_normalized_and_derived_outputs_ignored(
        self, tmp_path: Path
    ) -> None:
        hub = FakeHub(
            repo_id="org/private-ds",
            private=True,
            files={
                "sess-root/manifest.json": b'{"session_id": "sess-root"}',
                "sess-root/trajectory.json": b"{}",
                "sess-root/deep/merged-items.json": b"{}",
                "README.md": b"archive metadata",
                "bundle/manifest.json": b'{"session_id": "sess-root"}',
                "bundle/trajectory.json": b"{}",
                "curated/cur-ignored/batches/old/manifest.json": b"{}",
                "annotations/latest/sessions.jsonl": b"{}\n",
            },
        )
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage" / "downloads"

        result = hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage)

        assert result.discovered == 1
        assert result.run_shaped_manifests == 1
        normalized = stage / ("a" * 40) / "bundles" / "sess-root"
        assert (normalized / "manifest.json").read_bytes() == b'{"session_id": "sess-root"}'
        assert (normalized / "trajectory.json").read_bytes() == b"{}"
        assert (normalized / "deep" / "merged-items.json").read_bytes() == b"{}"
        assert not (stage / ("a" * 40) / "sess-root").exists()
        assert not (stage / ("a" * 40) / "README.md").exists()
        assert not (stage / ("a" * 40) / "curated").exists()
        assert not (stage / ("a" * 40) / "annotations").exists()

    def test_run_shaped_manifest_without_required_artifacts_fails_closed(self, tmp_path: Path) -> None:
        hub = FakeHub(
            repo_id="org/private-ds",
            private=True,
            files={"sess-incomplete/manifest.json": b'{"session_id": "sess-incomplete"}'},
        )
        hub.commit_revision("a" * 40)

        with pytest.raises(hydrate.HydrationError, match="zero candidates|trajectory.json"):
            hydrate.download_snapshot(
                hub, revision="a" * 40, stage_dir=tmp_path / "stage" / "downloads"
            )

    def test_bronze_tree_is_never_discovered(self, tmp_path: Path) -> None:
        hub = FakeHub(
            repo_id="org/private-ds",
            private=True,
            files={
                "sess-a/manifest.json": b'{"session_id": "sess-a"}',
                "sess-a/trajectory.json": b"{}",
                "bronze/manifest.json": b'{"bronze": true}',
                "bronze/trajectory.json": b"{}",
            },
        )
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage" / "downloads"

        result = hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage)

        # M10: bronze raw-ingest content is immutable companion data, never
        # a canonical session candidate.
        assert result.discovered == 1
        assert result.run_shaped_manifests == 1
        assert not (stage / ("a" * 40) / "bundles" / "bronze").exists()

    def test_legacy_reserved_root_is_never_discovered(self, tmp_path: Path) -> None:
        hub = FakeHub(
            repo_id="org/private-ds",
            private=True,
            files={
                "bundles/sess-a/manifest.json": b'{"session_id": "sess-a"}',
                "bundles/sess-a/trajectory.json": b"{}",
                # Same shape as a complete legacy session under a reserved
                # root: excluded, exactly like its top-level sibling.
                "bundles/curated/manifest.json": b'{"session_id": "curated"}',
                "bundles/curated/trajectory.json": b"{}",
                "bundles/curated/batches/old/items.jsonl": b"{}\n",
            },
        )
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage" / "downloads"

        result = hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage)

        assert result.discovered == 1
        assert result.run_shaped_manifests == 1
        assert not (stage / ("a" * 40) / "bundles" / "curated").exists()
        assert (stage / ("a" * 40) / "bundles" / "sess-a" / "manifest.json").exists()

    def test_revision_without_any_manifest_fails_closed(self, tmp_path: Path) -> None:
        hub = FakeHub(
            repo_id="org/private-ds",
            private=True,
            files={"README.md": b"not an archive revision\n"},
        )
        hub.commit_revision("a" * 40)

        with pytest.raises(
            hydrate.NoSessionCandidatesError, match="zero candidates"
        ):
            hydrate.download_snapshot(
                hub, revision="a" * 40, stage_dir=tmp_path / "stage" / "downloads"
            )

    def test_duplicate_session_id_across_layouts_fails_closed(self, tmp_path: Path) -> None:
        hub = FakeHub(
            repo_id="org/private-ds",
            private=True,
            files={
                "sess-dup/manifest.json": b'{"session_id": "sess-dup"}',
                "sess-dup/trajectory.json": b"{}",
                "bundles/sess-dup/manifest.json": b'{"session_id": "sess-dup"}',
                "bundles/sess-dup/trajectory.json": b"{}",
            },
        )
        hub.commit_revision("a" * 40)

        with pytest.raises(
            hydrate.StageError, match="multiple source layouts normalize to the same session"
        ):
            hydrate.download_snapshot(
                hub, revision="a" * 40, stage_dir=tmp_path / "stage" / "downloads"
            )

    def test_clean_download_layout_and_digests(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage" / "downloads"
        hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage)
        manifest = json.loads((stage / ("a" * 40) / "_download_manifest.json").read_text())
        assert len(manifest["artifacts"]) == len(SNAPSHOT)
        assert manifest["candidate_sessions"] == ["sess-a"]
        art = manifest["artifacts"][0]
        assert art["relpath"].startswith("bundles/")  # paths relative to snapshot root
        expected = hashlib.sha256(SNAPSHOT[art["relpath"]]).hexdigest()
        assert art["sha256"] == expected
        assert (stage / ("a" * 40) / art["relpath"]).read_bytes() == SNAPSHOT[art["relpath"]]

    def test_resume_skips_verified_artifacts(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage" / "downloads"
        first = hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage)
        assert first.downloaded == len(SNAPSHOT)
        # interrupt: remove one artifact file, rerun — only it is re-fetched
        manifest_path = stage / ("a" * 40) / "_download_manifest.json"
        art = json.loads(manifest_path.read_text())["artifacts"][0]
        (stage / ("a" * 40) / art["relpath"]).unlink()
        hub.downloaded_log.clear()
        second = hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage)
        assert second.downloaded == 1
        assert second.skipped == len(SNAPSHOT) - 1

    def test_digest_mismatch_rejects_artifact(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        hub.commit_revision("a" * 40)
        hub.files["bundles/sess-a/manifest.json"] = b'{"session_id": "sess-a", "tampered": true}'
        stage = tmp_path / "stage" / "downloads"
        with pytest.raises(hydrate.StageError, match="digest"):
            hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage,
                                      expect={"bundles/sess-a/manifest.json": "0" * 64})

    def test_traversal_relpath_rejected(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        hub.files["../../escape.txt"] = b"pwned"  # hostile relpath is part of the pinned snapshot
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage" / "downloads"
        with pytest.raises(hydrate.StageError, match="traversal|escapes"):
            hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage)

    def test_traversal_from_manifest_never_writes(self, tmp_path: Path) -> None:
        """A hostile Hub listing pointing outside the staging root writes nothing outside."""
        hub = make_fake_hub(tmp_path)
        hub.files["bundles/../../outside.txt"] = b"pwned"  # pinned with the snapshot revision
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage" / "downloads"
        with pytest.raises(hydrate.StageError):
            hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage)
        assert not (tmp_path / "outside.txt").exists()
        assert not (tmp_path / "stage" / "outside.txt").exists()


class TestPublish:
    def _staged(self, tmp_path: Path) -> Path:
        hub = make_fake_hub(tmp_path)
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage"
        hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage / "downloads")
        hydrate.ingest_bundles(stage, revision="a" * 40)
        hydrate.dedupe_admitted(stage, revision="a" * 40)
        hydrate.build_import_ledger(stage, revision="a" * 40, source_commit="a" * 40)
        return stage

    def test_public_destination_hard_fails(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        hub.private = False
        stage = self._staged(tmp_path)
        cid = hydrate_rules.derive_curation_id(
            source_commit="a" * 40, sanitizer_version="1", index_schema_version="1",
            admission_policy_version="1")
        with pytest.raises(hydrate.PublicDestinationError, match="private"):
            hydrate.publish_batches(hub, stage, curation_id=cid)
        assert not any(p.startswith("curated/") for p in hub.files)  # nothing published

    def test_batches_and_ledger_under_additive_prefix(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        stage = self._staged(tmp_path)
        cid = hydrate_rules.derive_curation_id(
            source_commit="a" * 40, sanitizer_version="1", index_schema_version="1",
            admission_policy_version="1")
        hydrate.publish_batches(hub, stage, curation_id=cid)
        prefix = f"curated/{cid}/"
        uploaded = [p for p in hub.files if p.startswith(prefix)]
        assert any(p.startswith(prefix + "batches/") for p in uploaded)
        assert any(p == prefix + "resume/ledger.jsonl" for p in uploaded)
        assert any(p == prefix + "SHA256SUMS" for p in uploaded)
        # Bronze safety (M10/M13): only curated/… paths are ever written.
        assert not any(p.startswith("bronze") or p.startswith("runs/") for p in hub.files)
        ledger = json.loads(hub.files[prefix + "resume/ledger.jsonl"].decode().splitlines()[0])
        assert ledger["session_id"] == "sess-a"
        assert ledger["batch_digest"]
        assert ledger["source_commit"] == "a" * 40
        # Idempotent re-upload: same content lands at the same content-addressed paths.
        before = dict(hub.files)
        hydrate.publish_batches(hub, stage, curation_id=cid)
        assert hub.files == before

    def test_remote_ledger_checkpoint_enables_resume(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        stage = self._staged(tmp_path)
        cid = "cur-" + "0" * 16
        hydrate.publish_batches(hub, stage, curation_id=cid)
        # a fresh VM with empty disk discovers the remote ledger and skips completed batches
        fresh = tmp_path / "fresh"
        state = hydrate.resume_state(hub, curation_id=cid, stage_dir=fresh)
        assert state.completed_sessions == {"sess-a"}
        assert state.redownloaded == []  # digests verified remotely, nothing refetched


def _staged_with(tmp_path: Path, remote_urls: dict[str, str | None]) -> Path:
    """Stage a tree: ingest + index admitted bundles carrying the given remote URLs."""
    files: dict[str, bytes] = {}
    for sid, url in remote_urls.items():
        manifest: dict[str, str] = {"session_id": sid}
        if url is not None:
            manifest["remote_url"] = url
        files[f"bundles/{sid}/manifest.json"] = json.dumps(manifest).encode()
        files[f"bundles/{sid}/trajectory.json"] = b"{}"
    hub = FakeHub(repo_id="org/private-ds", private=True, files=files)
    hub.commit_revision("a" * 40)
    stage = tmp_path / "stage"
    hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage / "downloads")
    hydrate.ingest_bundles(stage, revision="a" * 40)
    hydrate.rebuild_index(stage)
    return stage


class TestResolutionMap:
    def test_map_records_slug_and_pinned_sha_no_raw_urls(self, tmp_path: Path) -> None:
        stage = _staged_with(tmp_path, remote_urls={
            "sess-a": "https://github.com/octo/repo",
            "sess-b": None,
        })
        cmap = hydrate.build_resolution_map(stage, source_commit="a" * 40)
        entry = cmap["octo/repo"]
        assert entry["pinned_sha"].startswith("a" * 8) or entry["pinned_sha"] == "a" * 40
        assert "raw_url" not in entry and "source_path" not in entry
        assert "https://github.com/octo/repo" not in json.dumps(cmap)  # raw URL is data, not output

    def test_non_allowlisted_host_reported_not_cloned(self, tmp_path: Path) -> None:
        stage = _staged_with(tmp_path, remote_urls={"sess-c": "https://gitlab.com/x/y"})
        cmap = hydrate.build_resolution_map(stage, source_commit="a" * 40)
        assert cmap["unavailable"] == ["sess-c"]     # reported, no fallback, no clone attempted
        assert "gitlab.com" not in json.dumps(cmap)  # redacted from published metadata

    def test_no_clone_during_hydration(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import daydream.git_ops as git_ops

        def boom(*a: object, **k: object) -> None:
            raise AssertionError("hydration must not clone")

        monkeypatch.setattr(git_ops, "clone_with_token", boom)
        monkeypatch.setattr(git_ops, "fetch", boom)
        stage = _staged_with(tmp_path, remote_urls={"sess-a": "https://github.com/octo/repo"})
        hydrate.build_resolution_map(stage, source_commit="a" * 40)  # must not raise


class TestIngestAndIndex:
    def _staged(self, tmp_path: Path, revision: str = "a" * 40) -> Path:
        hub = make_fake_hub(tmp_path)
        hub.commit_revision(revision)
        stage = tmp_path / "stage"
        hydrate.download_snapshot(hub, revision=revision, stage_dir=stage / "downloads")
        return stage

    def test_clean_bundle_ingested_and_indexed_staging_local(self, tmp_path: Path) -> None:
        stage = self._staged(tmp_path)
        results = hydrate.ingest_bundles(stage, revision="a" * 40)
        assert [r.status for r in results] == ["admitted"]
        row_dir = stage / "runs" / "sess-a"
        assert row_dir.is_dir()
        # index row carries staging-local paths only
        from daydream.archive.index import query_runs
        hydrate.rebuild_index(stage)
        rows = query_runs(stage)
        assert len(rows) == 1
        assert rows[0]["archive_path"].startswith(str(stage))     # inside staging root
        assert "downloads" not in rows[0]["archive_path"]         # not the raw download path
        assert rows[0]["source_path"] is None or not Path(rows[0]["source_path"]).is_absolute() or \
            rows[0]["source_path"].startswith(str(stage))

    def test_dirty_bundle_quarantined_never_visible(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        hub.files["bundles/sess-bad/manifest.json"] = \
            b'{"session_id": "sess-bad", "remote_url": "https://user:hunter2@github.com/o/r"}'
        hub.files["bundles/sess-bad/trajectory.json"] = b"{}"
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage"
        hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage / "downloads")
        results = hydrate.ingest_bundles(stage, revision="a" * 40)
        bad = [r for r in results if r.session_id == "sess-bad"]
        assert bad and bad[0].status == "quarantined"
        assert bad[0].reason_code == "secrets_scan_dirty"
        # never visible to the index / harvest
        hydrate.rebuild_index(stage)
        from daydream.archive.index import query_runs
        assert all(row["session_id"] != "sess-bad" for row in query_runs(stage))
        assert (stage / "quarantine" / "sess-bad").exists()

    def test_embedded_paths_never_dereferenced(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        hub.files["bundles/sess-evil/manifest.json"] = (
            b'{"session_id": "sess-evil", "archive_path": "/etc", "source_path": "/usr",'
            b' "remote_url": "file:///etc/passwd"}'
        )
        hub.files["bundles/sess-evil/trajectory.json"] = b"{}"
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage"
        hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage / "downloads")
        results = hydrate.ingest_bundles(stage, revision="a" * 40)
        evil = [r for r in results if r.session_id == "sess-evil"]
        assert evil and evil[0].status == "quarantined"   # non-allowlisted host fails closed
        # nothing outside staging was touched
        assert not pathlib.Path("/etc/passwd.git").exists()

    def test_traversal_session_id_quarantined_before_any_write(self, tmp_path: Path) -> None:
        """A traversal-bearing manifest session id is quarantined pre-write (M4)."""
        hub = FakeHub(repo_id="org/private-ds", private=True, files={
            "bundles/sess-evil/manifest.json":
                b'{"session_id": "../escape", "remote_url": "https://github.com/o/r"}',
            "bundles/sess-evil/trajectory.json": b"{}",
        })
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage"
        hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage / "downloads")
        results = hydrate.ingest_bundles(stage, revision="a" * 40)
        evil = [r for r in results if r.session_id == "../escape"]
        assert evil and evil[0].status == "quarantined"
        assert evil[0].reason_code == hydrate_rules.REASON_CODE_PATH_TRAVERSAL
        assert not (stage / "runs").exists()          # never admitted
        assert not (stage / "quarantine").exists()    # sanitize gate never saw it
        assert not (tmp_path / "escape").exists()     # nothing outside staging

    def test_produced_nested_manifest_indexed_without_crash(self, tmp_path: Path) -> None:
        """Real manifests nest git.* and carry a nested daydream provenance dict."""
        from daydream.archive.manifest import Manifest
        from daydream.archive.provenance import ExecutableProvenance

        manifest = Manifest(
            session_id="sess-real",
            remote_url="https://github.com/octo/nested-repo",
            repo_slug="octo/nested-repo",
            source_path="/orig/absolute/path",
            daydream=ExecutableProvenance(version="1.0", install_source="editable"),
        )
        hub = FakeHub(repo_id="org/private-ds", private=True, files={
            "bundles/sess-real/manifest.json": json.dumps(manifest.to_dict()).encode(),
            "bundles/sess-real/trajectory.json": b"{}",
        })
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage"
        hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage / "downloads")
        results = hydrate.ingest_bundles(stage, revision="a" * 40)
        assert [r.status for r in results] == ["admitted"]
        hydrate.rebuild_index(stage)  # must not raise: nested daydream dict is dropped
        from daydream.archive.index import query_runs
        rows = [r for r in query_runs(stage) if r["session_id"] == "sess-real"]
        assert len(rows) == 1
        assert rows[0]["repo_slug"] == "octo/nested-repo"  # nested git.remote_url read

    def test_bundles_exclude_git_dirs(self, tmp_path: Path) -> None:
        """Task 0B constraint: no .git ships inside hydrated bundles (harvest priority-1 safety)."""
        stage = self._staged(tmp_path)
        hydrate.ingest_bundles(stage, revision="a" * 40)
        assert not list((stage / "runs").rglob(".git"))


class TestFinalizeAndVerify:
    def _config(self, tmp_path: Path) -> hydrate.HydrateHubConfig:
        return hydrate.HydrateHubConfig(
            source_repo="org/private-ds", source_revision="a" * 40,
            destination_repo="org/private-ds", stage_dir=tmp_path / "stage")

    def _staged(self, tmp_path: Path) -> Path:
        hub = make_fake_hub(tmp_path)
        stage = tmp_path / "stage"
        hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage / "downloads")
        hydrate.ingest_bundles(stage, revision="a" * 40)
        hydrate.dedupe_admitted(stage, revision="a" * 40)
        hydrate.build_import_ledger(stage, revision="a" * 40, source_commit="a" * 40)
        return stage

    def test_verify_failure_never_publishes_success_marker(self, tmp_path: Path) -> None:
        """A post-publication verification failure leaves no published _SUCCESS."""
        class CorruptingHub(FakeHub):
            def download_file(self, path_in_repo: str, revision: str | None = None) -> bytes:
                data = super().download_file(path_in_repo, revision)
                if "/batches/" in path_in_repo and path_in_repo.endswith("trajectory.json"):
                    return data + b"\ncorrupted"
                return data

        hub = CorruptingHub(repo_id="org/private-ds", private=True, files=dict(SNAPSHOT))
        hub.commit_revision("a" * 40)
        with pytest.raises(hydrate.VerificationError):
            hydrate.run_hydrate_hub(hydrate.HydrateHubConfig(
                source_repo="org/private-ds", source_revision="a" * 40,
                destination_repo="org/private-ds", stage_dir=tmp_path / "stage"), client=hub)
        assert not any(p.endswith("_SUCCESS") for p in hub.uploaded_paths)

    def test_success_marker_last_and_output_sha_captured(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        stage = self._staged(tmp_path)
        summary = hydrate.run_hydrate_hub(hydrate.HydrateHubConfig(
            source_repo="org/private-ds", source_revision="a" * 40,
            destination_repo="org/private-ds", stage_dir=stage), client=hub)
        order = hub.commit_order
        assert order[-1]["contains"] == ["curated/" + summary.curation_id + "/_SUCCESS"]
        assert summary.output_commit_sha and len(summary.output_commit_sha) == 40

    def test_verify_cycle_reproduces_dry_run_counts(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        self._staged(tmp_path)
        summary = hydrate.run_hydrate_hub(self._config(tmp_path), client=hub)
        assert summary.verified is True
        assert summary.dry_run_admitted == summary.verify_admitted  # M19 count equality

    def test_no_success_on_skipped_upload(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        hub.fail_uploads = True
        self._staged(tmp_path)
        with pytest.raises(hydrate.HydrationError):
            hydrate.run_hydrate_hub(self._config(tmp_path), client=hub)
        assert "cur-" not in "".join(hub.uploaded_paths) or \
            not any(p.endswith("_SUCCESS") for p in hub.uploaded_paths)

    def test_no_success_on_public_destination(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        hub.private = False
        with pytest.raises(hydrate.PublicDestinationError):
            hydrate.run_hydrate_hub(self._config(tmp_path), client=hub)
        assert not any(p.endswith("_SUCCESS") for p in hub.uploaded_paths)


class TestDedupeAndLedger:
    def _staged(self, tmp_path: Path, revision: str = "a" * 40) -> Path:
        hub = make_fake_hub(tmp_path)
        hub.commit_revision(revision)
        stage = tmp_path / "stage"
        hydrate.download_snapshot(hub, revision=revision, stage_dir=stage / "downloads")
        hydrate.ingest_bundles(stage, revision=revision)
        return stage

    def test_collision_durable_across_reruns(self, tmp_path: Path) -> None:
        """A collision re-quarantines on every later run; the mutated derivative
        is never re-admitted over the published baseline (M7 durability)."""
        hub = make_fake_hub(tmp_path)
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage"
        hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage / "downloads")
        hydrate.ingest_bundles(stage, revision="a" * 40)
        hydrate.dedupe_admitted(stage, revision="a" * 40)  # run 1: admit baseline
        baseline_manifest = (stage / "runs" / "sess-a" / "manifest.json").read_bytes()
        # mutate + re-download + re-ingest (runs 2 and 3 see the same mutated tree)
        hub.mutate_bundle("a" * 40, "sess-a", b'{"tampered": true}')
        for _ in range(2):
            staged = stage / "downloads" / ("a" * 40) / "bundles" / "sess-a" / "manifest.json"
            staged.unlink()
            hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage / "downloads")
            hydrate.ingest_bundles(stage, revision="a" * 40)
            rerun = hydrate.dedupe_admitted(stage, revision="a" * 40)
            assert rerun.collisions == 1   # every later run re-quarantines
            assert rerun.admitted == 0     # the mutated derivative is never admitted
        # the admitted derivative is the restored baseline, byte for byte
        restored = stage / "runs" / "sess-a" / "manifest.json"
        assert restored.read_bytes() == baseline_manifest
        from daydream.archive.index import query_runs
        assert len(query_runs(stage)) == 1  # one session row, never overwritten

    def test_idempotent_rerun_no_duplicates(self, tmp_path: Path) -> None:
        stage = self._staged(tmp_path)
        first = hydrate.dedupe_admitted(stage, revision="a" * 40)
        assert first.admitted == 1 and first.collisions == 0
        second = hydrate.dedupe_admitted(stage, revision="a" * 40)  # same content re-run
        assert second.admitted == 1 and second.collisions == 0
        from daydream.archive.index import query_runs
        assert len(query_runs(stage)) == 1  # no duplicate rows

    def test_identity_collision_quarantined(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage"
        hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage / "downloads")
        hydrate.ingest_bundles(stage, revision="a" * 40)
        hydrate.dedupe_admitted(stage, revision="a" * 40)
        # same session identity, different content -> quarantine, never overwrite
        hub.files["bundles/sess-a/trajectory.json"] = b'{"different": true}'
        hub.commit_revision("a" * 40)  # re-pin the mutated tree at the same revision
        stage2_dir = stage / "downloads"
        # re-download only the changed file then re-ingest into the same stage
        (stage2_dir / ("a" * 40) / "bundles/sess-a/trajectory.json").unlink()
        hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage2_dir)
        hydrate.ingest_bundles(stage, revision="a" * 40)
        result = hydrate.dedupe_admitted(stage, revision="a" * 40)
        assert result.collisions == 1
        assert (stage / "quarantine" / "sess-a.conflict").exists() or \
            result.collision_ids == ["sess-a"]
        from daydream.archive.index import query_runs
        rows = [r for r in query_runs(stage) if r["session_id"] == "sess-a"]
        assert len(rows) == 1  # original retained, never overwritten

    def test_fixture_bundle_excluded_with_code(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        hub.files["bundles/sess-fixture/manifest.json"] = (
            b'{"session_id": "sess-fixture", "source_path": "/tmp/pytest-of-user/test_x"}'
        )
        hub.files["bundles/sess-fixture/trajectory.json"] = b"{}"
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage"
        hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage / "downloads")
        hydrate.ingest_bundles(stage, revision="a" * 40)
        result = hydrate.dedupe_admitted(stage, revision="a" * 40)
        assert ("sess-fixture", "fixture_pytest_path") in result.excluded
        assert not (stage / "runs" / "sess-fixture").exists()  # never indexed/harvest-visible

    def test_ledger_is_value_free(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        hub.files["bundles/sess-bad/manifest.json"] = (
            b'{"session_id": "sess-bad", "remote_url": "https://user:hunter2@github.com/o/r"}'
        )
        hub.files["bundles/sess-bad/trajectory.json"] = b"{}"
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage"
        hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage / "downloads")
        hydrate.ingest_bundles(stage, revision="a" * 40)
        ledger = hydrate.build_import_ledger(stage, revision="a" * 40, source_commit="a" * 40)
        text = json.dumps(ledger)
        assert "hunter2" not in text and "user:" not in text  # no matched secret values (M11)
        assert ledger["pinned_revision"] == "a" * 40
        assert {"imported", "quarantined", "excluded", "rejections"} <= set(ledger)
        assert any(e["session_id"] == "sess-bad" for e in ledger["quarantined"])
        # ledger file persisted under the curated prefix, atomically
        ledger_path = stage / "curated" / ledger["curation_id"] / "import-ledger.json"
        assert ledger_path.is_file()
        assert json.loads(ledger_path.read_text())["pinned_revision"] == "a" * 40


class TestLicenseAdmissionGate:
    """Issue #1080 task 3: per-repo license decisions at hydration admission."""

    REV = "a" * 40

    @staticmethod
    def _session_manifest(sid: str, slug: str, spdx: str | None = None) -> bytes:
        data: dict[str, object] = {
            "session_id": sid,
            "git": {"remote_url": f"https://github.com/{slug}", "repo_slug": slug},
        }
        if spdx is not None:
            data["license_evidence"] = {"spdx_id": spdx, "source": "manifest"}
        return json.dumps(data).encode()

    def _seed_stage(self, tmp_path: Path, sessions: dict[str, bytes]) -> Path:
        files: dict[str, bytes] = {}
        for sid, manifest in sessions.items():
            files[f"bundles/{sid}/manifest.json"] = manifest
            files[f"bundles/{sid}/trajectory.json"] = b"{}"
        hub = FakeHub(repo_id="org/private-ds", private=True, files=files)
        hub.commit_revision(self.REV)
        stage = tmp_path / "stage"
        hydrate.download_snapshot(hub, revision=self.REV, stage_dir=stage / "downloads")
        hydrate.ingest_bundles(stage, revision=self.REV)
        hydrate.dedupe_admitted(stage, revision=self.REV)
        return stage

    def _write_policy(self, tmp_path: Path) -> Path:
        policy_path = tmp_path / "license-policy.json"
        policy_path.write_text(json.dumps({
            "policy_version": "1",
            "spdx_decisions": {"MIT": "accepted", "GPL-3.0-only": "rejected"},
        }))
        return policy_path

    def _built_manifest(self, stage: Path) -> dict[str, Any]:
        ledger = hydrate.build_import_ledger(stage, revision=self.REV, source_commit=self.REV)
        return hydrate._curation_manifest_doc(
            stage, curation_id=str(ledger["curation_id"]), source_commit=self.REV, ledger=ledger
        )

    def test_manifest_batches_carry_repo_slug_and_license_evidence(self, tmp_path: Path) -> None:
        stage = self._seed_stage(tmp_path, {
            "sess-a": self._session_manifest("sess-a", "owner/repo-a", spdx="MIT"),
        })
        manifest = self._built_manifest(stage)
        admitted = [b for b in manifest["batches"] if b["status"] == "admitted"]
        assert admitted, "seed must produce at least one admitted batch"
        for batch in admitted:
            assert batch["repo_slug"] == "owner/repo-a"
            assert batch["license_evidence"] == {"spdx_id": "MIT", "source": "manifest"}

    def test_hydration_excludes_c5_repo_at_admission(self, tmp_path: Path) -> None:
        stage = self._seed_stage(tmp_path, {
            "sess-a": self._session_manifest("sess-a", "owner/repo-a", spdx="MIT"),
            "sess-c5": self._session_manifest("sess-c5", "getsentry/sentry", spdx="MIT"),
        })
        hydrate.apply_license_gate(
            stage, revision=self.REV,
            license_policy_path=self._write_policy(tmp_path), allow_copyleft=frozenset())
        manifest = self._built_manifest(stage)
        c5 = next(
            (b for b in manifest["batches"]
             if b["status"] == "excluded"
             and b["reason_code"] == hydrate_rules.REASON_CODE_C5_EXCLUDED_REPO),
            None,
        )
        assert c5 is not None, "C5 gate failed to exclude the sentry repo (needs C5 batch)"
        assert c5["session_id"] == "sess-c5"
        assert c5["artifact_relpath"].startswith("excluded/")
        assert not (stage / "runs" / "sess-c5").exists()
        assert (stage / "excluded" / "sess-c5").exists()
        # the admitted MIT repo is untouched
        admitted = [b for b in manifest["batches"] if b["status"] == "admitted"]
        assert [b["session_id"] for b in admitted] == ["sess-a"]

    def test_hydration_c5_gate_catches_non_canonical_slug_spelling(self, tmp_path: Path) -> None:
        # The license gate compares the canonical owner/repo identity: a
        # manifest that stamps the clone URL as repo_slug cannot bypass C5
        # (issue #1080, fail-closed).
        c5_manifest = {
            "session_id": "sess-c5",
            "git": {
                "remote_url": "https://github.com/getsentry/sentry",
                "repo_slug": "https://github.com/getsentry/sentry",
            },
            "license_evidence": {"spdx_id": "MIT", "source": "manifest"},
        }
        stage = self._seed_stage(tmp_path, {
            "sess-a": self._session_manifest("sess-a", "owner/repo-a", spdx="MIT"),
            "sess-c5": json.dumps(c5_manifest).encode(),
        })
        hydrate.apply_license_gate(
            stage, revision=self.REV,
            license_policy_path=self._write_policy(tmp_path), allow_copyleft=frozenset())
        manifest = self._built_manifest(stage)
        c5 = next(
            (b for b in manifest["batches"]
             if b["status"] == "excluded"
             and b["reason_code"] == hydrate_rules.REASON_CODE_C5_EXCLUDED_REPO),
            None,
        )
        assert c5 is not None, "C5 gate failed to exclude the sentry repo (needs C5 batch)"
        assert c5["session_id"] == "sess-c5"
        assert not (stage / "runs" / "sess-c5").exists()
        assert (stage / "excluded" / "sess-c5").exists()

    def test_hydration_excludes_unopted_copyleft_and_missing_evidence(self, tmp_path: Path) -> None:
        stage = self._seed_stage(tmp_path, {
            "sess-gpl": self._session_manifest("sess-gpl", "owner/gpl-repo", spdx="GPL-3.0-only"),
            "sess-noev": self._session_manifest("sess-noev", "owner/noev-repo"),
        })
        hydrate.apply_license_gate(
            stage, revision=self.REV,
            license_policy_path=self._write_policy(tmp_path), allow_copyleft=frozenset())
        manifest = self._built_manifest(stage)
        rows = {b["session_id"]: (b["status"], b["reason_code"])
                for b in manifest["batches"]}
        assert rows["sess-gpl"] == ("excluded", hydrate_rules.REASON_CODE_C8_COPYLEFT_UNOPTED)
        assert rows["sess-noev"] == ("excluded", hydrate_rules.REASON_CODE_LICENSE_EVIDENCE_MISSING)

    def test_allow_copyleft_opt_in_admits_exact_slug_only(self, tmp_path: Path) -> None:
        stage = self._seed_stage(tmp_path, {
            "sess-opted": self._session_manifest("sess-opted", "owner/gpl-repo", spdx="GPL-3.0-only"),
            "sess-similar": self._session_manifest("sess-similar", "owner/similar-gpl-repo", spdx="GPL-3.0-only"),
        })
        hydrate.apply_license_gate(
            stage, revision=self.REV,
            license_policy_path=self._write_policy(tmp_path), allow_copyleft={"owner/gpl-repo"})
        manifest = self._built_manifest(stage)
        rows = {b["session_id"]: (b["status"], b["reason_code"])
                for b in manifest["batches"]}
        assert rows["sess-opted"] == ("admitted", None)
        assert rows["sess-similar"] == ("excluded", hydrate_rules.REASON_CODE_C8_COPYLEFT_UNOPTED)

    def test_license_gate_refuses_missing_policy_fail_closed(self, tmp_path: Path) -> None:
        stage = self._seed_stage(tmp_path, {
            "sess-a": self._session_manifest("sess-a", "owner/repo-a", spdx="MIT"),
        })
        with pytest.raises(ValueError, match="license_policy_path"):
            hydrate.apply_license_gate(
                stage, revision=self.REV, license_policy_path=None, allow_copyleft=frozenset())
        # refusal is pre-work: nothing moved, nothing recorded
        assert (stage / "runs" / "sess-a").exists()
        assert not (stage / "excluded").exists()

    def test_gate_wired_into_run_hydrate_hub_and_manifest_rows(self, tmp_path: Path) -> None:
        """run_hydrate_hub threads the policy through; the C5 repo never publishes."""
        hub = FakeHub(repo_id="org/private-ds", private=True, files={
            "bundles/sess-a/manifest.json": self._session_manifest("sess-a", "owner/repo-a", spdx="MIT"),
            "bundles/sess-a/trajectory.json": b"{}",
            "bundles/sess-c5/manifest.json": self._session_manifest("sess-c5", "getsentry/sentry", spdx="MIT"),
            "bundles/sess-c5/trajectory.json": b"{}",
        })
        hub.commit_revision(self.REV)
        stage = tmp_path / "stage"
        summary = hydrate.run_hydrate_hub(hydrate.HydrateHubConfig(
            source_repo="org/private-ds", source_revision=self.REV,
            destination_repo="org/private-ds", stage_dir=stage,
            license_policy_path=str(self._write_policy(tmp_path)), allow_copyleft=frozenset(),
        ), client=hub)
        assert summary.verified is True
        assert summary.dry_run_admitted == 1 and summary.dry_run_rejected == 1
        doc = json.loads(hub.download_file(
            f"curated/{summary.curation_id}/curation-manifest.json", revision=summary.output_commit_sha))
        rows = {b["session_id"]: b for b in doc["batches"]}
        assert rows["sess-a"]["repo_slug"] == "owner/repo-a"
        assert rows["sess-a"]["license_evidence"] == {"spdx_id": "MIT", "source": "manifest"}
        assert rows["sess-c5"]["reason_code"] == hydrate_rules.REASON_CODE_C5_EXCLUDED_REPO
        assert rows["sess-c5"]["status"] == "excluded"


class TestAdmissionSummary:
    """Issue #1080 task 9 (S2): per-repo human summary over the admission ledger."""

    def test_admission_summary_buckets_every_session(self, tmp_path: Path) -> None:
        gate = TestLicenseAdmissionGate()
        stage = gate._seed_stage(tmp_path, {
            "sess-a": gate._session_manifest("sess-a", "owner/repo-a", spdx="MIT"),
            "sess-c5": gate._session_manifest("sess-c5", "getsentry/sentry", spdx="MIT"),
            "sess-gpl": gate._session_manifest("sess-gpl", "owner/gpl-repo", spdx="GPL-3.0-only"),
            "sess-noev": gate._session_manifest("sess-noev", "owner/noev-repo"),
        })
        hydrate.apply_license_gate(
            stage, revision=gate.REV,
            license_policy_path=gate._write_policy(tmp_path), allow_copyleft=frozenset())
        ledger = hydrate.build_import_ledger(stage, revision=gate.REV, source_commit=gate.REV)
        summary = hydrate.license_admission_summary(ledger)
        assert summary["admitted"] == 1
        assert summary["c5_excluded"] == 1
        assert summary["c8_copyleft_unopted"] == 1
        assert summary["license_evidence_missing"] == 1
        # M8: the buckets partition the license-gate sessions by construction.
        assert sum(summary.values()) == 4

    def test_admission_summary_pure_helper_buckets_seed_entries(self) -> None:
        entries: list[tuple[str, str | None]] = [
            ("s1", None),
            ("s2", hydrate_rules.REASON_CODE_C5_EXCLUDED_REPO),
            ("s3", hydrate_rules.REASON_CODE_C8_COPYLEFT_UNOPTED),
            ("s4", hydrate_rules.REASON_CODE_LICENSE_EVIDENCE_MISSING),
            ("s5", hydrate_rules.REASON_CODE_REPO_IDENTITY_MISSING),
        ]
        summary = hydrate.admission_summary_buckets(entries)
        assert summary["admitted"] == 1
        assert summary["c5_excluded"] == 1
        assert summary["c8_copyleft_unopted"] == 1
        assert summary["license_evidence_missing"] == 2  # identity-missing folds in
        assert sum(summary.values()) == 5


def test_repo_commit_unresolved_is_a_license_bucket_code():
    from daydream.archive.hydrate import admission_summary_buckets
    from daydream.archive.hydrate_rules import (
        EXCLUSION_CODES,
        REASON_CODE_REPO_COMMIT_UNRESOLVED,
    )
    assert REASON_CODE_REPO_COMMIT_UNRESOLVED == "repo_commit_unresolved"
    assert REASON_CODE_REPO_COMMIT_UNRESOLVED in EXCLUSION_CODES
    buckets = admission_summary_buckets([("s1", REASON_CODE_REPO_COMMIT_UNRESOLVED)])
    assert buckets["license_evidence_missing"] == 1 and sum(buckets.values()) == 1
