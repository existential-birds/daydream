"""License evidence enrichment for issue #1094 (Task 3).

Fills legacy records' missing ``license_evidence`` from normalized repo identity
via an injectable :class:`RepoLicenseResolver` protocol before the license gate
runs. The production adapter targets the GitHub license API using a commit-pinned
request; ``GITHUB_TOKEN`` is read from the environment only and never persisted,
never placed on a URL, and never embedded in an error message (every exception
message passes through :func:`daydream.trajectory.redact_text`).

Enrichment is a separate stage so ``apply_license_gate`` stays a pure function
of policy + evidence: the gate consumes enriched evidence exactly like declared
evidence via the existing ``_session_identity`` manifest path.

Results are appended to ``stage/_enrich/evidence.jsonl`` — one JSON line per
session with a stable status code — deduped per ``(repo_slug, repo_commit)`` so
repeated sessions in one repo hit the cache, not the resolver. The cache
survives re-runs (load-before-query) and is published into the curated prefix
as ``license-evidence.jsonl`` by :func:`publish_enrichment_cache` so fresh-VM
replay from pinned inputs reproduces identical decisions.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from daydream.archive.hydrate import (
    HydrationError,
    _curated_dir,
    _manifest_license_evidence,
    _manifest_repo_slug,
    _read_manifest_dict,
)
from daydream.training.corpus_v2.license import normalize_repo_slug
from daydream.trajectory import redact_text

_ENRICH_DIR = "_enrich"
_ENRICH_CACHE_NAME = "evidence.jsonl"
_PUBLISHED_CACHE_NAME = "license-evidence.jsonl"

_GITHUB_API = "https://api.github.com"
_GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
_RATE_LIMIT_ATTEMPTS = 3
_RATE_LIMIT_BASE_DELAY_S = 2.0


@dataclass(frozen=True)
class EnrichedEvidence:
    """License evidence from an authorized immutable source.

    ``source`` is a provenance string of the form ``github:<owner>/<repo>@<full-commit>``
    — never a URL carrying credentials.
    """

    spdx_id: str
    source: str
    repo_commit: str


@runtime_checkable
class RepoLicenseResolver(Protocol):
    """Resolver seam: authorized immutable license evidence for one repo slug.

    ``None`` means unresolvable at this source — a recorded stable-code miss,
    not an exception; the gate rejects it downstream (fail-closed).
    """

    def resolve(self, repo_slug: str, repo_commit: str | None) -> EnrichedEvidence | None: ...


class GithubLicenseResolver:
    """Production adapter against the GitHub license API.

    ``GITHUB_TOKEN`` is read from ``os.environ`` only — never an argument, never
    persisted. 404 → ``None`` (a stable-code miss, not an exception). Rate-limit
    responses are retried with bounded exponential backoff (3 attempts, 2s base)
    honoring ``Retry-After``; every surfaced error message is redacted.
    """

    def resolve(self, repo_slug: str, repo_commit: str | None) -> EnrichedEvidence | None:
        token = os.environ.get(_GITHUB_TOKEN_ENV, "")
        repo_commit = repo_commit or self._resolve_head_commit(repo_slug, token)
        data = self._request_json(
            f"{_GITHUB_API}/repos/{repo_slug}/license"
            + (f"?ref={repo_commit}" if repo_commit else ""),
            token,
        )
        if data is None:
            return None
        spdx_id = (data.get("license") or {}).get("spdx_id") if isinstance(data, dict) else None
        commit = data.get("commit_sha") if isinstance(data, dict) else None
        if not isinstance(spdx_id, str) or not spdx_id.strip():
            raise HydrationError(
                redact_text(f"license response for {repo_slug} carried no usable spdx_id")
            )
        if not isinstance(commit, str) or commit.strip() == "":
            raise HydrationError(
                redact_text(f"license response for {repo_slug} carried no usable commit")
            )
        return EnrichedEvidence(
            spdx_id=spdx_id, source=f"github:{repo_slug}@{commit}", repo_commit=commit,
        )

    def _resolve_head_commit(self, repo_slug: str, token: str) -> str | None:
        """Default-branch head commit for the repo (None when unresolvable)."""
        data = self._request_json(f"{_GITHUB_API}/repos/{repo_slug}", token)
        if data is None:
            return None
        commit = (data.get("default_branch") or {}).get("sha") if isinstance(data, dict) else None
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


def _load_cache(stage: Path) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Load the enrichment cache: latest entry per session and per (slug, commit)."""
    by_session: dict[str, dict[str, Any]] = {}
    by_repo: dict[tuple[str, str], dict[str, Any]] = {}
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
            by_repo[(slug, commit)] = entry
    return by_session, by_repo


def _append_cache(path: Path, entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")


def enrich_license_evidence(
    stage: Path, *, revision: str, resolver: RepoLicenseResolver
) -> dict[str, dict[str, str]]:
    """Fill missing ``license_evidence`` on admitted derivatives under ``stage/runs/``.

    Records with well-formed declared evidence are left untouched (out of scope
    per spec). Legacy records with a canonical repo slug are resolved through
    ``resolver`` (dedup per ``(repo_slug, repo_commit)``); resolved evidence is
    written into the session's ``manifest.json`` so the gate consumes it exactly
    like declared evidence. Every processed session gets a stable-code cache
    entry: ``resolved`` | ``repo_identity_missing`` | ``license_evidence_missing``.

    Resolver network failures propagate as redacted :class:`HydrationError`s
    (fatal); a resolver returning ``None`` is a recorded miss, never an
    exception — the gate rejects it. Returns resolved evidence per session id
    (``spdx_id``/``source``/``repo_commit``/``origin: enriched``); ``revision``
    is accepted to mirror the other staging-stage signatures (dedup is per
    resolved repo commit, not the Hub dataset revision — repository commits in
    the resolution map come from the resolver, never the Hub revision).
    """
    runs_dir = stage / "runs"
    by_session, by_repo = _load_cache(stage)
    fresh: list[dict[str, Any]] = []
    resolved: dict[str, dict[str, str]] = {}
    if not runs_dir.is_dir():
        return resolved
    for derivative in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        data = _read_manifest_dict(derivative)
        if data is None:
            raise HydrationError(
                redact_text(f"admitted derivative {derivative.name} has an unreadable manifest")
            )
        sid = str(data.get("session_id") or derivative.name)
        if _manifest_license_evidence(data) is not None:
            continue  # well-formed declared evidence is never re-derived
        prior = by_session.get(sid)
        if prior is not None:
            if prior.get("status") == "resolved":
                resolved[sid] = {
                    "spdx_id": str(prior["spdx_id"]),
                    "source": str(prior["source"]),
                    "repo_commit": str(prior.get("repo_commit") or ""),
                    "origin": "enriched",
                }
            continue
        raw_slug = _manifest_repo_slug(data)
        slug = normalize_repo_slug(raw_slug) if raw_slug else ""
        if not slug:
            fresh.append({"session_id": sid, "status": "repo_identity_missing"})
            continue
        # Dedupe per (repo_slug, resolved repo_commit): any prior resolved entry for
        # this slug is reused — repeated sessions in one repo hit the cache, not the
        # resolver (the commit is produced by the resolver, so the slug keys the hit).
        cached_hit = next((e for (s, __), e in by_repo.items() if s == slug), None)
        entry: dict[str, Any]
        if cached_hit is not None:
            entry = {**cached_hit, "session_id": sid}
        else:
            evidence = resolver.resolve(slug, None)
            if evidence is None:
                entry = {
                    "session_id": sid, "repo_slug": slug,
                    "status": "license_evidence_missing",
                }
            else:
                entry = {
                    "session_id": sid, "repo_slug": slug, "status": "resolved",
                    "spdx_id": evidence.spdx_id, "source": evidence.source,
                    "repo_commit": evidence.repo_commit,
                }
                by_repo[(slug, evidence.repo_commit)] = entry
        fresh.append(entry)
        if entry.get("status") == "resolved":
            resolved[sid] = {
                "spdx_id": str(entry["spdx_id"]),
                "source": str(entry["source"]),
                "repo_commit": str(entry["repo_commit"]),
                "origin": "enriched",
            }
            manifest_path = derivative / "manifest.json"
            data["license_evidence"] = {
                "spdx_id": entry["spdx_id"], "source": entry["source"],
            }
            manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _append_cache(_cache_path(stage), fresh)
    return resolved


def publish_enrichment_cache(
    stage: Path, *, revision: str | None = None, curated_dir: Path | None = None,
) -> Path | None:
    """Copy the staging-internal enrichment cache into the curated prefix.

    The cache at ``stage/_enrich/`` is VM-local bookkeeping (mirroring
    ``_dedupe``'s published-set exclusion); the curated copy
    ``license-evidence.jsonl`` is the published pinned state a fresh-VM replay
    consumes. Returns the published path, or ``None`` when nothing was cached.

    Issue #1094: publication passes the post-gate v2 ``curated_dir`` (the
    curation id is only known after the gate); callers without one fall back
    to the pre-identity v1-shaped location.
    """
    cache = _cache_path(stage)
    if not cache.is_file():
        return None
    if curated_dir is None:
        curated_dir = _curated_dir(stage, str(revision))
    target = curated_dir / _PUBLISHED_CACHE_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(cache.read_bytes())
    return target
