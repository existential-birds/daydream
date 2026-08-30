"""Offline end-to-end hydrate tests over a local fake Hub snapshot (M22).

No network: the snapshot is materialized in-memory by build_hub_snapshot.py and
served by hydrate_client.FakeHub. Scenarios: clean import, interruption/resume,
identity collision, path traversal, deterministic re-index — plus bronze
immutability (M10).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from daydream.archive import hydrate, hydrate_rules
from daydream.archive.hydrate_client import FakeHub
from tests.fixtures.training.build_hub_snapshot import SNAPSHOT_REVISION, build_snapshot

REVISION = SNAPSHOT_REVISION  # 40-hex pinned by the fixture builder

# Index fields that legitimately differ between two staging roots (staging-local
# paths) or across runs (row timestamps) — determinism is asserted on the rest.
_VOLATILE_ROW_KEYS = frozenset({"archive_path", "source_path", "created_at", "updated_at"})


@pytest.fixture()
def hub() -> FakeHub:
    return build_snapshot()  # repo org/private-ds, private, bundles for 3 sessions


def _config(stage: Path) -> hydrate.HydrateHubConfig:
    return hydrate.HydrateHubConfig(
        source_repo="org/private-ds",
        source_revision=REVISION,
        destination_repo="org/private-ds",
        stage_dir=stage,
    )


def _index_rows(stage: Path) -> list[dict[str, object]]:
    from daydream.archive.index import query_runs

    return [
        {k: v for k, v in row.items() if k not in _VOLATILE_ROW_KEYS}
        for row in query_runs(stage)
    ]


class TestCleanImport:
    def test_hydrate_discovers_all_snapshot_sessions(self, hub: FakeHub, tmp_path: Path) -> None:
        stage = tmp_path / "stage"
        summary = hydrate.run_hydrate_hub(_config(stage), client=hub)
        assert summary.verified
        # M1: the hydrated staging archive is harvest-discoverable from disk alone.
        from daydream.archive.index import count_runs

        assert count_runs(stage) == 3

    def test_bronze_never_mutated(self, hub: FakeHub, tmp_path: Path) -> None:
        before = dict(hub.files)
        hydrate.run_hydrate_hub(_config(tmp_path / "stage"), client=hub)
        # M10: files only gain curated/ keys; bronze and every pre-existing key
        # are byte-identical after the run.
        for key, content in before.items():
            if key.startswith("curated/"):
                continue
            assert hub.files.get(key) == content, f"pre-existing Hub file {key!r} was mutated"
        new_keys = set(hub.files) - set(before)
        assert new_keys and all(k.startswith("curated/") for k in new_keys)
        assert not any(k.startswith("bronze/") for k in new_keys)


class TestInterruptionResume:
    def test_resume_after_kill_completes_without_duplicates(self, hub: FakeHub, tmp_path: Path) -> None:
        stage = tmp_path / "stage"
        # interrupt: download only, then "kill" — no ingest, no ledger, no index.
        hydrate.download_snapshot(hub, revision=REVISION, stage_dir=stage / "downloads")
        # fresh VM simulation: empty disk except the remote ledger — resume via
        # resume_state (Hub ledger is the only canonical resume state, M15).
        curation_id = hydrate_rules.derive_curation_id(
            source_commit=REVISION, sanitizer_version=hydrate_rules.SANITIZER_VERSION,
            index_schema_version=hydrate_rules.HYDRATION_INDEX_SCHEMA_VERSION,
            admission_policy_version=hydrate_rules.ADMISSION_POLICY_VERSION,
        )
        state = hydrate.resume_state(
            hub, curation_id=curation_id, stage_dir=tmp_path / "fresh"
        )
        assert state.completed_sessions is not None
        summary = hydrate.run_hydrate_hub(_config(stage), client=hub)
        assert summary.verified
        from daydream.archive.index import count_runs

        assert count_runs(stage) == 3  # no duplicate sessions after resume


class TestCollision:
    def test_same_identity_different_content_quarantines(self, hub: FakeHub, tmp_path: Path) -> None:
        stage = tmp_path / "stage"
        hydrate.run_hydrate_hub(_config(stage), client=hub)
        hub.mutate_bundle(REVISION, "sess-a", b'{"tampered": true}')  # same identity, new content
        # simulate the resumed download re-fetching the mutated bundle
        staged = stage / "downloads" / REVISION / "bundles" / "sess-a" / "manifest.json"
        staged.unlink()
        rerun = hydrate.run_hydrate_hub(_config(stage), client=hub)  # collision is reported, not silent
        from daydream.archive.index import query_runs

        rows = [r for r in query_runs(stage) if r["session_id"] == "sess-a"]
        assert len(rows) == 1  # original intact
        # the collision is a recorded rejection with the stable reason code
        ledger = json.loads(
            (stage / "curated" / rerun.curation_id / "import-ledger.json").read_text()
        )
        quarantined = {e["session_id"]: e["reason_code"] for e in ledger["quarantined"]}
        assert quarantined.get("sess-a") == hydrate_rules.REASON_CODE_IDENTITY_COLLISION
        # the published curation manifest lists the collided session exactly once,
        # as quarantined at the real collision directory (never double-listed)
        manifest = json.loads(
            hub.files[f"curated/{rerun.curation_id}/curation-manifest.json"].decode()
        )
        sess_rows = [b for b in manifest["batches"] if b["session_id"] == "sess-a"]
        assert len(sess_rows) == 1
        assert sess_rows[0]["status"] == "quarantined"
        assert sess_rows[0]["reason_code"] == hydrate_rules.REASON_CODE_IDENTITY_COLLISION
        assert sess_rows[0]["artifact_relpath"] == "quarantine/sess-a.conflict"


class TestPathTraversal:
    def test_hostile_snapshot_escapes_nothing(self, tmp_path: Path) -> None:
        hub = build_snapshot(hostile=True)  # adds ../../ and /etc-style relpaths
        with pytest.raises(hydrate.HydrationError):
            hydrate.run_hydrate_hub(_config(tmp_path / "stage"), client=hub)
        assert not (tmp_path / "escape.txt").exists()
        assert not (tmp_path / "etc").exists()


class TestDeterministicReindex:
    def test_rerun_yields_identical_index_and_curation_id(self, hub: FakeHub, tmp_path: Path) -> None:
        s1 = hydrate.run_hydrate_hub(_config(tmp_path / "a"), client=hub)
        s2 = hydrate.run_hydrate_hub(_config(tmp_path / "b"), client=hub)
        assert s1.curation_id == s2.curation_id  # M14
        rows1 = sorted(json.dumps(r, sort_keys=True) for r in _index_rows(tmp_path / "a"))
        rows2 = sorted(json.dumps(r, sort_keys=True) for r in _index_rows(tmp_path / "b"))
        assert rows1 == rows2  # deterministic re-index (modulo staging-local paths)
