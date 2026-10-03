"""The CLI consumes the captured source population, not mutable cache receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream import cli
from daydream.archive import hydrate, license_enrich
from tests.test_cli_hydrate import _FakeLicenseResolver, _hydrate_args, _seed_three_repo_hub


def _configure_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, Path, Path]:
    hub = _seed_three_repo_hub()
    hub.commit_revision("a" * 40)
    stage = tmp_path / "stage"
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"policy_version": "1", "spdx_decisions": {"MIT": "accepted"}}))
    monkeypatch.setenv("HF_TOKEN", "fixture-token")
    monkeypatch.setenv("GITHUB_TOKEN", "fixture-token")
    monkeypatch.setattr(hydrate, "_make_client", lambda _repo: hub)
    return hub, stage, policy


@pytest.mark.parametrize("dry_run", [True, False])
def test_cli_late_fabricated_manifest_does_not_inflate_source_population(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], dry_run: bool,
) -> None:
    hub, stage, policy = _configure_cli(monkeypatch, tmp_path)

    class Resolver(_FakeLicenseResolver):
        def resolve(self, repo_slug: str, repo_commit: str | None) -> Any:
            # A real external license lookup runs after source download/ingest.
            # Simulate another local actor replacing only the generated receipts.
            folder = next((stage / "downloads").iterdir())
            manifest = folder / "_download_manifest.json"
            data = json.loads(manifest.read_bytes())
            if "fabricated" not in data["candidate_sessions"]:
                data["candidate_sessions"].append("fabricated")
                data["discovery"]["run_shaped_manifests"] = 999
                data["discovery"]["incomplete_manifests"] = ["fabricated (missing trajectory.json)"]
                manifest.write_text(json.dumps(data))
                results = folder / "_ingest_results.json"
                doc = json.loads(results.read_bytes())
                doc["results"].append({
                    "session_id": "fabricated", "status": "quarantined", "reason_code": "bundle_unreadable",
                })
                results.write_text(json.dumps(doc))
            return super().resolve(repo_slug, repo_commit)

    monkeypatch.setattr(license_enrich, "_make_license_resolver", lambda: Resolver(("acme/widget",)))
    with pytest.raises(SystemExit) as exited:
        cli.main(["corpus", "hydrate-hub", *_hydrate_args(
            stage, source_repo="org/private-ds", destination_repo="org/private-ds",
            policy=policy, dry_run=dry_run,
        )])
    assert exited.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "discovered 3 candidate(s)" in output
    assert "admitted 2" in output and "rejected 1 batch(es)" in output
    assert "yield reduced" not in output and "999" not in output
    ledger = json.loads(next(stage.glob("curated/*/import-ledger.json")).read_bytes())
    assert ledger["tallies"]["discovered"] == 3
    assert ledger["tallies"]["accounted"] == 3
    assert ledger["tallies"]["run_shaped_manifests"] == 3
    assert ledger["tallies"]["incomplete_manifests"] == []
    assert "fabricated" not in {row["session_id"] for row in ledger["rejections"]}
    assert not (stage / "runs" / "fabricated").exists()
    assert bool(hub.uploaded_paths) is not dry_run


@pytest.mark.parametrize("failure", ["cache", "ledger"])
def test_cli_cache_and_ledger_failures_keep_publication_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], failure: str,
) -> None:
    hub, stage, policy = _configure_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(
        license_enrich, "_make_license_resolver", lambda: _FakeLicenseResolver(("acme/widget",)),
    )
    events = []
    original_cache = license_enrich.publish_enrichment_cache

    def publish_cache(*args: Any, **kwargs: Any) -> Any:
        events.append("cache")
        if failure == "cache":
            raise OSError("fixture cache publication refused")
        return original_cache(*args, **kwargs)

    def publish_ledger(*args: Any, **kwargs: Any) -> Any:
        events.append("ledger")
        raise OSError("fixture ledger publication refused")

    monkeypatch.setattr(license_enrich, "publish_enrichment_cache", publish_cache)
    monkeypatch.setattr(hydrate, "build_import_ledger", publish_ledger)
    with pytest.raises(SystemExit) as exited:
        cli.main(["corpus", "hydrate-hub", *_hydrate_args(
            stage, source_repo="org/private-ds", destination_repo="org/private-ds", policy=policy,
        )])
    assert exited.value.code == 1
    assert events == (["cache"] if failure == "cache" else ["cache", "ledger"])
    assert not list(stage.glob("curated/*/import-ledger.json"))
    assert bool(list(stage.glob("curated/*/license-evidence.jsonl"))) is (failure == "ledger")
    assert hub.uploaded_paths == []
    assert "hydration verified" not in capsys.readouterr().out
