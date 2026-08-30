"""Unit/component tests for the #982 hydrate module."""
from __future__ import annotations

import hashlib
import json
import pathlib
from pathlib import Path

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


class TestDownloadSnapshot:
    def test_clean_download_layout_and_digests(self, tmp_path: Path) -> None:
        hub = make_fake_hub(tmp_path)
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage" / "downloads"
        hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage)
        manifest = json.loads((stage / ("a" * 40) / "_download_manifest.json").read_text())
        assert len(manifest["artifacts"]) == len(SNAPSHOT)
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
        hub.commit_revision("a" * 40)
        hub.files["../../escape.txt"] = b"pwned"
        stage = tmp_path / "stage" / "downloads"
        with pytest.raises(hydrate.StageError, match="traversal|escapes"):
            hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage)

    def test_traversal_from_manifest_never_writes(self, tmp_path: Path) -> None:
        """A hostile Hub listing pointing outside the staging root writes nothing outside."""
        hub = make_fake_hub(tmp_path)
        hub.commit_revision("a" * 40)
        hub.files["bundles/../../outside.txt"] = b"pwned"
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
        hub.commit_revision("a" * 40)
        stage = tmp_path / "stage"
        hydrate.download_snapshot(hub, revision="a" * 40, stage_dir=stage / "downloads")
        results = hydrate.ingest_bundles(stage, revision="a" * 40)
        evil = [r for r in results if r.session_id == "sess-evil"]
        assert evil and evil[0].status == "quarantined"   # non-allowlisted host fails closed
        # nothing outside staging was touched
        assert not pathlib.Path("/etc/passwd.git").exists()

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
