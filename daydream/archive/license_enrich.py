"""Enrich missing license evidence before the pure license-policy gate.

GitHub requests are commit-pinned. Tokens come only from the environment,
never URLs or persisted data; surfaced errors are redacted.

The persistent staging cache dedupes by repo/commit and records one status per
session. Its published ``license-evidence.jsonl`` pins this curation's evidence
for audit and replay. Fresh staging resolves again; changed upstream decisions
produce a different curation identity.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from daydream.archive.hydrate_admission import _curated_dir, _manifest_license_evidence, _manifest_repo_slug
from daydream.archive.hydrate_stage import _require_manifest_dict
from daydream.archive.hydrate_types import HydrationError
from daydream.training.corpus_projection.license import normalize_repo_slug
from daydream.trajectory import RUNS_DIRNAME, redact_text

_ENRICH_DIR = "_enrich"
_ENRICH_CACHE_NAME = "evidence.jsonl"
_PUBLISHED_CACHE_NAME = "license-evidence.jsonl"

_GITHUB_API = "https://api.github.com"
_GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
_RATE_LIMIT_ATTEMPTS = 3
_RATE_LIMIT_BASE_DELAY_S = 2.0

# A full 40-hex Git commit embedded in an evidence ``source`` string (e.g. the
# enrichment-produced ``github:<owner>/<repo>@<commit>`` form or a URL tail).
_FULL_COMMIT_IN_SOURCE_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{40}(?![0-9a-fA-F])")


def _commit_from_source(source: str) -> str | None:
    """Extract a lowercased full Git SHA from a declared source; never invent an unresolved pin."""
    match = _FULL_COMMIT_IN_SOURCE_RE.search(source or "")
    return match.group(0).lower() if match else None


@dataclass(frozen=True)
class EnrichedEvidence:
    """License evidence from an authorized immutable source."""

    spdx_id: str
    source: str
    repo_commit: str


@runtime_checkable
class RepoLicenseResolver(Protocol):
    """Resolver seam: authorized immutable license evidence for one repo slug."""

    def resolve(self, repo_slug: str, repo_commit: str | None) -> EnrichedEvidence | None: ...


class GithubLicenseResolver:
    """Resolve commit-pinned GitHub license evidence using the ambient token."""

    def resolve(self, repo_slug: str, repo_commit: str | None) -> EnrichedEvidence | None:
        token = os.environ.get(_GITHUB_TOKEN_ENV, "")
        if not token:
            # Require a token before HTTP or staging; errors never expose credentials.
            raise HydrationError(
                "GITHUB_TOKEN is not set; license evidence enrichment requires "
                "a GitHub API token for the license endpoint. Export "
                "GITHUB_TOKEN and retry."
            )
        repo_commit = repo_commit or self._resolve_head_commit(repo_slug, token)
        data = self._request_json(
            f"{_GITHUB_API}/repos/{repo_slug}/license"
            + (f"?ref={repo_commit}" if repo_commit else ""),
            token,
        )
        if data is None:
            return None
        spdx_id = (data.get("license") or {}).get("spdx_id") if isinstance(data, dict) else None
        # The contents-style license response carries the blob ``sha``, never a
        # commit — the commit identity is the ref this request was pinned at
        # (the resolver-resolved head commit or the caller-supplied commit).
        commit = repo_commit
        if not isinstance(spdx_id, str) or not spdx_id.strip():
            raise HydrationError(
                redact_text(f"license response for {repo_slug} carried no usable spdx_id")
            )
        if not isinstance(commit, str) or commit.strip() == "":
            # An unpinnable Git commit records repo_commit_unresolved instead of aborting
            # the corpus. None means unresolvable under the resolver contract.
            return None
        return EnrichedEvidence(
            spdx_id=spdx_id, source=f"github:{repo_slug}@{commit}", repo_commit=commit,
        )

    def _resolve_head_commit(self, repo_slug: str, token: str) -> str | None:
        """Default-branch head commit for the repo (None when unresolvable)."""
        data = self._request_json(f"{_GITHUB_API}/repos/{repo_slug}", token)
        if data is None:
            return None
        # GitHub's /repos/{owner}/{repo} carries ``default_branch`` as a plain
        # string; the head commit lives on the branch resource.
        default_branch = data.get("default_branch") if isinstance(data, dict) else None
        if not isinstance(default_branch, str) or not default_branch.strip():
            return None
        branch = self._request_json(
            f"{_GITHUB_API}/repos/{repo_slug}/branches/{default_branch}", token
        )
        if branch is None:
            return None
        commit = (branch.get("commit") or {}).get("sha") if isinstance(branch, dict) else None
        return commit if isinstance(commit, str) and commit.strip() else None

    def _request_json(self, url: str, token: str) -> dict[str, Any] | None:
        """GET one JSON document with bounded rate-limit backoff; 404 → None."""
        delay = _RATE_LIMIT_BASE_DELAY_S
        last_error: Exception | None = None
        for attempt in range(_RATE_LIMIT_ATTEMPTS):
            request = urllib.request.Request(url, headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            })
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return payload if isinstance(payload, dict) else None
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                remaining = exc.headers.get("X-RateLimit-Remaining") if exc.headers else None
                rate_limited = exc.code == 403 and remaining == "0"
                last_error = exc
                if attempt + 1 < _RATE_LIMIT_ATTEMPTS and (rate_limited or retry_after):
                    time.sleep(float(retry_after) if retry_after else delay)
                    delay *= 2
                    continue
                break
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < _RATE_LIMIT_ATTEMPTS:
                    time.sleep(delay)
                    delay *= 2
                    continue
                break
        raise HydrationError(
            redact_text(f"license source request failed for {url}: {last_error}")
        ) from last_error


def _make_license_resolver() -> RepoLicenseResolver:
    """Production resolver factory — the monkeypatch seam, like ``_make_client``."""
    return GithubLicenseResolver()


def _cache_path(stage: Path) -> Path:
    return stage / _ENRICH_DIR / _ENRICH_CACHE_NAME


def _load_cache(stage: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load the enrichment cache: latest entry per session and first resolved per repo."""
    by_session: dict[str, dict[str, Any]] = {}
    by_repo: dict[str, dict[str, Any]] = {}
    path = _cache_path(stage)
    if not path.is_file():
        return by_session, by_repo
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict) or not entry.get("session_id"):
            continue
        by_session[str(entry["session_id"])] = entry
        slug = entry.get("repo_slug")
        commit = entry.get("repo_commit")
        if entry.get("status") == "resolved" and isinstance(slug, str) and isinstance(commit, str):
            by_repo.setdefault(slug, entry)
    return by_session, by_repo


def _append_cache(path: Path, entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")


def _write_resolved(
    derivative: Path, data: dict[str, Any], entry: dict[str, Any]
) -> dict[str, str]:
    """Persist *entry*'s resolved evidence into the manifest; return the payload record."""
    spdx_id, source = str(entry["spdx_id"]), str(entry.get("source") or "")
    data["license_evidence"] = {"spdx_id": spdx_id, "source": source}
    (derivative / "manifest.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {
        "spdx_id": spdx_id,
        "source": source,
        "repo_commit": str(entry.get("repo_commit") or ""),
        "origin": "enriched",
    }


def enrich_license_evidence(
    stage: Path, *, resolver: RepoLicenseResolver
) -> dict[str, dict[str, str]]:
    """Enrich admitted manifests with missing license evidence, caching by repo/commit."""
    runs_dir = stage / RUNS_DIRNAME
    by_session, by_repo = _load_cache(stage)
    fresh: list[dict[str, Any]] = []
    resolved: dict[str, dict[str, str]] = {}
    if not runs_dir.is_dir():
        return resolved
    for derivative in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        data = _require_manifest_dict(derivative, label=f"admitted derivative {derivative.name}")
        sid = str(data.get("session_id") or derivative.name)
        declared = _manifest_license_evidence(data)
        if declared is not None:
            # Retain declared evidence; cache its full Git commit for the resolution map.
            # Without a pinned commit, report unreachable. Do not seed by_repo: evidence-less
            # siblings must still use the live resolver.
            commit = _commit_from_source(str(declared.get("source") or ""))
            raw_slug = _manifest_repo_slug(data)
            slug = normalize_repo_slug(raw_slug) if raw_slug else ""
            if commit is not None and slug:
                fresh.append({
                    "session_id": sid, "repo_slug": slug, "status": "resolved",
                    "spdx_id": declared["spdx_id"],
                    "source": str(declared.get("source") or ""),
                    "repo_commit": commit,
                })
            continue
        prior = by_session.get(sid)
        if prior is not None:
            if prior.get("status") == "resolved":
                # Ingest may restore an evidence-less manifest on reuse. Reapply cached evidence:
                # the admission gate reads manifests, never the cache.
                resolved[sid] = _write_resolved(derivative, data, prior)
            continue
        raw_slug = _manifest_repo_slug(data)
        slug = normalize_repo_slug(raw_slug) if raw_slug else ""
        if not slug:
            fresh.append({"session_id": sid, "status": "repo_identity_missing"})
            continue
        # Dedupe per (repo_slug, resolved repo_commit): any prior resolved entry for
        # this slug is reused — repeated sessions in one repo hit the cache, not the
        # resolver (the commit is produced by the resolver, so the slug keys the hit).
        cached_hit = by_repo.get(slug)
        entry: dict[str, Any]
        if cached_hit is not None:
            entry = {**cached_hit, "session_id": sid}
        else:
            evidence = resolver.resolve(slug, None)
            if evidence is None:
                # An identified repo without a pinned Git commit is unavailable; retain that
                # specific reason in the resolution map and license evidence-missing bucket.
                entry = {
                    "session_id": sid, "repo_slug": slug,
                    "status": "repo_commit_unresolved",
                }
            else:
                entry = {
                    "session_id": sid, "repo_slug": slug, "status": "resolved",
                    "spdx_id": evidence.spdx_id, "source": evidence.source,
                    "repo_commit": evidence.repo_commit,
                }
                by_repo.setdefault(slug, entry)
        fresh.append(entry)
        if entry.get("status") == "resolved":
            resolved[sid] = _write_resolved(derivative, data, entry)
    _append_cache(_cache_path(stage), fresh)
    return resolved


def publish_enrichment_cache(
    stage: Path, *, revision: str | None = None, curated_dir: Path | None = None,
) -> Path | None:
    """Publish the staging cache as license-evidence.jsonl for audit and replay."""
    cache = _cache_path(stage)
    if not cache.is_file():
        return None
    if curated_dir is None:
        curated_dir = _curated_dir(stage, str(revision))
    target = curated_dir / _PUBLISHED_CACHE_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(cache.read_bytes())
    return target
