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
from tests.fixtures.training.build_hub_snapshot import (
    PINNED_POLICY_FIXTURE,
    PINNED_REVISION,
    REPO_ID,
    SNAPSHOT_REVISION,
    build_pinned_snapshot,
    build_snapshot,
)

REVISION = SNAPSHOT_REVISION  # 40-hex pinned by the fixture builder


def _write_policy(tmp_path: Path) -> str:
    """Minimal permissive policy: the non-dry pipeline fail-closes without one
    (issue #1094), and the snapshot's declared MIT evidence admits."""
    policy = tmp_path / "license-policy.json"
    policy.write_text(json.dumps({"policy_version": "1", "spdx_decisions": {"MIT": "accepted"}}))
    return str(policy)


@pytest.fixture(autouse=True)
def _offline_enrichment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the enrichment stage offline: resolve every repo to MIT."""
    from daydream.archive import license_enrich

    def resolve(repo_slug: str, repo_commit: str | None) -> license_enrich.EnrichedEvidence | None:
        return resolve_evidence(repo_slug)

    def resolve_evidence(repo_slug: str) -> license_enrich.EnrichedEvidence:
        return license_enrich.EnrichedEvidence(
            spdx_id="MIT", source=f"fake:{repo_slug}", repo_commit="c" * 40
        )

    class FakeResolver:
        def resolve(
            self, repo_slug: str, repo_commit: str | None
        ) -> license_enrich.EnrichedEvidence | None:
            return resolve(repo_slug, repo_commit)

    monkeypatch.setattr(license_enrich, "_make_license_resolver", lambda: FakeResolver())


def _v2_curation_id(hub: FakeHub, tmp_path: Path) -> str:
    """Probe the production post-gate v2 identity derivation (issue #1094):
    the pipeline's curation id comes from the resolved policy binding, not
    the historical v1 derivation, so resume probing must use the same path."""
    stage = tmp_path / "identity-probe"
    hydrate.download_snapshot(hub, revision=REVISION, stage_dir=stage / "downloads")
    hydrate.ingest_bundles(stage, revision=REVISION)
    hydrate.dedupe_admitted(stage, revision=REVISION)
    from daydream.archive import license_enrich

    class FakeResolver:
        def resolve(
            self, repo_slug: str, repo_commit: str | None
        ) -> license_enrich.EnrichedEvidence | None:
            return license_enrich.EnrichedEvidence(
                spdx_id="MIT", source=f"fake:{repo_slug}", repo_commit="c" * 40
            )

    license_enrich.enrich_license_evidence(stage, revision=REVISION, resolver=FakeResolver())
    hydrate.restamp_admitted_digests(stage, revision=REVISION)
    hydrate.apply_license_gate(
        stage, revision=REVISION, license_policy_path=_write_policy(tmp_path),
        allow_copyleft=frozenset(),
    )
    binding = hydrate.resolve_curation_identity(
        stage, source_commit=REVISION, license_policy_path=_write_policy(tmp_path),
        allow_copyleft=frozenset(),
    )
    return str(binding["curation_id"])

# Index fields that legitimately differ between two staging roots (staging-local
# paths) or across runs (row timestamps) — determinism is asserted on the rest.
_VOLATILE_ROW_KEYS = frozenset({"archive_path", "source_path", "created_at", "updated_at"})


@pytest.fixture()
def hub() -> FakeHub:
    return build_snapshot()  # repo org/private-ds, private, bundles for 3 sessions


def _config(stage: Path, policy_dir: Path | None = None) -> hydrate.HydrateHubConfig:
    return hydrate.HydrateHubConfig(
        source_repo="org/private-ds",
        source_revision=REVISION,
        destination_repo="org/private-ds",
        stage_dir=stage,
        license_policy_path=_write_policy(stage.parent if policy_dir is None else policy_dir),
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
        assert summary.dry_run_discovered == 3
        assert summary.dry_run_admitted + summary.dry_run_rejected == 3

        snapshot_dir = stage / "downloads" / REVISION
        assert (snapshot_dir / "bundles" / "sess-a" / "manifest.json").is_file()
        assert not (snapshot_dir / "sess-a").exists()
        assert not (snapshot_dir / "README.md").exists()
        assert not (snapshot_dir / "curated").exists()
        assert not (snapshot_dir / "annotations").exists()

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
        # The candidate prefix is the post-gate v2 curation id (issue #1094):
        # probe it through the production derivation, never the v1 inputs.
        curation_id = _v2_curation_id(hub, tmp_path)
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


# ---------------------------------------------------------------------------
# Pinned archive fixture (issue #1094 Task 10): enrichment -> gate -> v2
# identity -> publication over a committed, digest-pinned snapshot + policy.
# ---------------------------------------------------------------------------


class _PinnedResolver:
    """Canned per-slug enrichment results; records queried slugs."""

    def __init__(self, results: dict[str, object]) -> None:
        self.results = results
        self.queried: list[str] = []

    def resolve(self, repo_slug: str, repo_commit: str | None):  # type: ignore[no-untyped-def]
        from daydream.archive import license_enrich

        self.queried.append(repo_slug)
        evidence = self.results.get(repo_slug)
        if evidence is None:
            return None
        assert isinstance(evidence, license_enrich.EnrichedEvidence)
        return evidence


class _ReplayResolver:
    """Fresh-VM replay resolver: serves only from the published
    ``license-evidence.jsonl`` cache bytes. There is no live source to query —
    a slug absent from the published cache fails the test (never re-queried).
    """

    def __init__(self, published_cache: bytes) -> None:
        self.by_slug: dict[str, dict[str, object]] = {}
        for line in published_cache.decode("utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            slug = entry.get("repo_slug")
            if isinstance(slug, str):
                self.by_slug.setdefault(slug, entry)
        self.served: list[str] = []

    def resolve(self, repo_slug: str, repo_commit: str | None):  # type: ignore[no-untyped-def]
        from daydream.archive import license_enrich

        entry = self.by_slug.get(repo_slug)
        assert entry is not None, (
            f"replay resolver asked for {repo_slug!r}, which the published "
            "cache never resolved — the live source would be re-queried"
        )
        self.served.append(repo_slug)
        if entry.get("status") != "resolved":
            return None
        return license_enrich.EnrichedEvidence(
            spdx_id=str(entry["spdx_id"]),
            source=str(entry["source"]),
            repo_commit=str(entry["repo_commit"]),
        )


def _pinned_resolver() -> _PinnedResolver:
    from daydream.archive import license_enrich

    commit = "d" * 40
    return _PinnedResolver({
        "acme/widget": license_enrich.EnrichedEvidence(
            spdx_id="MIT", source=f"github:acme/widget@{commit}", repo_commit=commit,
        ),
        "acme/copyleft": license_enrich.EnrichedEvidence(
            spdx_id="GPL-3.0-only", source=f"github:acme/copyleft@{commit}", repo_commit=commit,
        ),
        "ghost/nope": None,  # unresolvable at the source: stable-code miss
        "getsentry/sentry": license_enrich.EnrichedEvidence(
            spdx_id="MIT", source=f"github:getsentry/sentry@{commit}", repo_commit=commit,
        ),
    })


def run_pinned_fixture_hydration(
    stage_root: Path,
    *,
    hub: FakeHub | None = None,
    resolver: object | None = None,
) -> hydrate.HydrateSummary:
    """One full ``run_hydrate_hub`` pass over the pinned snapshot + policy fixture.

    ``resolver`` (default: the canned pinned resolver) is injected through the
    production ``_make_license_resolver`` seam.
    """
    from daydream.archive import license_enrich

    hub = hub if hub is not None else build_pinned_snapshot()
    resolver = resolver if resolver is not None else _pinned_resolver()
    stage_root.mkdir(parents=True, exist_ok=True)
    policy = stage_root / "license-policy.json"
    policy.write_bytes(PINNED_POLICY_FIXTURE.read_bytes())
    config = hydrate.HydrateHubConfig(
        source_repo=REPO_ID,
        source_revision=PINNED_REVISION,
        destination_repo=REPO_ID,
        stage_dir=stage_root / "stage",
        license_policy_path=str(policy),
        allow_copyleft=frozenset(),
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(license_enrich, "_make_license_resolver", lambda: resolver)
        return hydrate.run_hydrate_hub(config, client=hub)


class TestPinnedArchiveFixture:
    def test_pinned_archive_fixture_full_path_enrich_gate_identity_publish(
        self, tmp_path: Path
    ) -> None:
        # Pinned snapshot fixture (committed builder bytes, digest-pinned
        # revision) + pinned policy fixture: enrichment fills legacy evidence,
        # the gate excludes per stable code, the v2 identity binds, publication
        # + verify pass, _SUCCESS lands last.
        hub = build_pinned_snapshot()
        summary = run_pinned_fixture_hydration(tmp_path, hub=hub)
        assert summary.verified is True
        assert summary.license_admission["admitted"] >= 1          # nonzero admitted data
        assert summary.license_admission["c8_copyleft_unopted"] == 1
        assert summary.license_admission["license_evidence_missing"] >= 1
        # Full record accounting over the pinned five-session matrix.
        assert summary.license_admission == {
            "admitted": 2,               # declared MIT + enriched MIT
            "c5_excluded": 1,            # getsentry/sentry (C5 preempts evidence)
            "c8_copyleft_unopted": 1,    # GPL-3.0-only, no exact-slug opt-in
            "license_evidence_missing": 1,  # unresolvable repo
        }
        assert summary.dry_run_discovered == 5
        assert (
            summary.dry_run_admitted
            + summary.dry_run_rejected
            == summary.dry_run_discovered
        )

        # Fresh-VM replay: identical decisions and identity from the published
        # cache + pinned inputs — the replay resolver reads the published
        # license-evidence.jsonl and is never re-queried against a live source.
        from daydream.archive.license_enrich import _PUBLISHED_CACHE_NAME

        published_key = f"curated/{summary.curation_id}/{_PUBLISHED_CACHE_NAME}"
        assert published_key in hub.files  # published under the v2 prefix
        replay_resolver = _ReplayResolver(hub.files[published_key])
        replay = run_pinned_fixture_hydration(
            tmp_path / "replay", hub=build_pinned_snapshot(), resolver=replay_resolver,
        )
        assert replay.curation_id == summary.curation_id
        assert replay.license_admission == summary.license_admission
        assert replay.verified is True
        # Every enrichment slug was served from the published cache, never the live source.
        assert set(replay_resolver.served) == {
            "acme/widget", "acme/copyleft", "ghost/nope", "getsentry/sentry",
        }


class TestDeterministicReindex:
    def test_rerun_yields_identical_index_and_curation_id(self, hub: FakeHub, tmp_path: Path) -> None:
        s1 = hydrate.run_hydrate_hub(_config(tmp_path / "a"), client=hub)
        s2 = hydrate.run_hydrate_hub(_config(tmp_path / "b"), client=hub)
        assert s1.curation_id == s2.curation_id  # M14
        rows1 = sorted(json.dumps(r, sort_keys=True) for r in _index_rows(tmp_path / "a"))
        rows2 = sorted(json.dumps(r, sort_keys=True) for r in _index_rows(tmp_path / "b"))
        assert rows1 == rows2  # deterministic re-index (modulo staging-local paths)
