"""Enrichment stage: evidence from an authorized immutable source, cache + provenance, env-only creds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.archive.license_enrich import (
    EnrichedEvidence,
    enrich_license_evidence,
)


def seed_admitted_runs(
    stage: Path, specs: list[tuple[str, str | None, dict[str, str] | None]]
) -> None:
    """Seed admitted derivatives directly under ``stage/runs/`` (the enrichment input set)."""
    for sid, slug, evidence in specs:
        d = stage / "runs" / sid
        d.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"session_id": sid}
        if slug is not None:
            data["git"] = {"remote_url": f"https://github.com/{slug}", "repo_slug": slug}
        if evidence is not None:
            data["license_evidence"] = evidence
        (d / "manifest.json").write_text(json.dumps(data), encoding="utf-8")


def _as_entries(stage: Path) -> dict[str, dict[str, Any]]:
    cache = stage / "_enrich" / "evidence.jsonl"
    entries: dict[str, dict[str, Any]] = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            entries[str(e["session_id"])] = e
    return entries


class FakeResolver:
    """Resolver seam: canned per-slug results; records queried slugs for dedupe assertions."""

    def __init__(self, results: dict[str, EnrichedEvidence | None]):
        self.results = results
        self.queried: list[str] = []

    def resolve(self, repo_slug: str, repo_commit: str | None) -> EnrichedEvidence | None:
        self.queried.append(repo_slug)
        return self.results.get(repo_slug)


def _make_resolver(spdx: str = "MIT") -> FakeResolver:
    return FakeResolver({
        "acme/widget": EnrichedEvidence(
            spdx_id=spdx, source=f"github:acme/widget@{'b' * 40}", repo_commit="b" * 40,
        ),
    })


def test_enrich_fills_missing_evidence_and_skips_declared(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    seed_admitted_runs(stage, [
        ("sess-legacy", "acme/widget", None),
        ("sess-declared", "acme/widget", {"spdx_id": "Apache-2.0", "source": "producer"}),
    ])
    resolver = _make_resolver()
    evidence = enrich_license_evidence(stage, revision="a" * 40, resolver=resolver)
    # The legacy record gains declared evidence identical in shape to producer evidence.
    assert evidence["sess-legacy"]["spdx_id"] == "MIT"
    assert evidence["sess-legacy"]["source"] == f"github:acme/widget@{'b' * 40}"
    # Well-formed declared evidence is never re-derived (out of scope per spec).
    assert "sess-declared" not in [k for k in evidence if evidence[k].get("origin") == "enriched"]
    assert "sess-declared" not in _as_entries(stage)
    # The enriched evidence was written into the session manifest for the gate to consume.
    manifest = json.loads((stage / "runs" / "sess-legacy" / "manifest.json").read_text())
    assert manifest["license_evidence"] == {
        "spdx_id": "MIT", "source": f"github:acme/widget@{'b' * 40}",
    }
    # Dedupe: one (repo, revision) queried even with two sessions in the same repo.
    assert resolver.queried == ["acme/widget"]


def test_enrich_publishes_cache_with_provenance_and_no_credentials(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    seed_admitted_runs(stage, [("sess-legacy", "acme/widget", None)])
    enrich_license_evidence(stage, revision="a" * 40, resolver=_make_resolver())
    cache_path = stage / "_enrich" / "evidence.jsonl"
    assert cache_path.is_file()
    import json
    entries = [json.loads(line) for line in cache_path.read_text().splitlines() if line.strip()]
    e = next(x for x in entries if x["session_id"] == "sess-legacy")
    assert e["status"] == "resolved" and e["spdx_id"] == "MIT"
    assert e["repo_commit"] == "b" * 40 and "github:acme/widget@" in e["source"]
    # No token or authenticated URL anywhere in the published bytes.
    raw = cache_path.read_text()
    assert "ghp_" not in raw and "token" not in raw.lower()


def test_enrich_records_stable_failure_codes_for_unresolvable(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    seed_admitted_runs(stage, [
        ("sess-noslug", None, None),               # no repo identity
        ("sess-unknown", "ghost/nope", None),      # resolver cannot resolve
    ])
    enrich_license_evidence(stage, revision="a" * 40, resolver=FakeResolver({}))
    codes = {sid: v["status"] for sid, v in _as_entries(stage).items()}
    # No-slug session: repo_identity_missing; unresolvable repo: license_evidence_missing.
    assert codes["sess-noslug"] == "repo_identity_missing"
    assert codes["sess-unknown"] == "license_evidence_missing"
    # Neither record gained evidence in its manifest — the gate rejects it downstream.
    for sid in ("sess-noslug", "sess-unknown"):
        manifest = json.loads((stage / "runs" / sid / "manifest.json").read_text())
        assert "license_evidence" not in manifest


def test_enrich_reuses_cached_resolution_across_runs(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    seed_admitted_runs(stage, [("sess-1", "acme/widget", None)])
    enrich_license_evidence(stage, revision="a" * 40, resolver=_make_resolver())
    # A second run (fresh resolver instance) must hit the cache, not the resolver.
    second = _make_resolver()
    seed_admitted_runs(stage, [("sess-2", "acme/widget", None)])
    evidence = enrich_license_evidence(stage, revision="a" * 40, resolver=second)
    assert second.queried == []
    assert evidence["sess-2"]["spdx_id"] == "MIT"


def test_enrichment_cache_copied_into_curated_prefix(tmp_path: Path) -> None:
    from daydream.archive.hydrate import _curated_dir
    stage = tmp_path / "stage"
    seed_admitted_runs(stage, [("sess-legacy", "acme/widget", None)])
    enrich_license_evidence(stage, revision="a" * 40, resolver=_make_resolver())
    from daydream.archive.license_enrich import publish_enrichment_cache
    publish_enrichment_cache(stage, revision="a" * 40)
    published = _curated_dir(stage, "a" * 40) / "license-evidence.jsonl"
    assert published.is_file()
    assert published.read_text() == (stage / "_enrich" / "evidence.jsonl").read_text()


def test_github_resolver_reads_token_from_env_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production adapter: GITHUB_TOKEN from the environment, never an argument or a URL."""
    from daydream.archive.license_enrich import GithubLicenseResolver
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secrettokenvalue")
    resolver = GithubLicenseResolver()
    # The token never appears in the request URL — only in the Authorization header.
    import urllib.request

    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, **kwargs: Any) -> Any:
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())

        class R:
            def __enter__(self) -> Any:
                return self

            def __exit__(self, *_args: Any) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps({
                    "license": {"spdx_id": "MIT"},
                    "commit_sha": "b" * 40,
                }).encode()

        return R()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    evidence = resolver.resolve("acme/widget", None)
    assert evidence is not None and evidence.spdx_id == "MIT"
    assert "ghp_secrettokenvalue" not in captured["url"]
    assert captured["headers"]["Authorization"] == "Bearer ghp_secrettokenvalue"
    assert evidence.source == f"github:acme/widget@{'b' * 40}"
