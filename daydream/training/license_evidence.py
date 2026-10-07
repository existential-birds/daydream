"""Acquire immutable GitHub license evidence for append-only dataset observations.

Credentials come from the environment. Only commit-pinned license responses
produce evidence; failures expose value-free diagnostics.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from daydream.training.corpus_projection.license import normalize_repo_slug

_GITHUB_API = "https://api.github.com"
_GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
_RATE_LIMIT_ATTEMPTS = 3
_RATE_LIMIT_BASE_DELAY_S = 2.0


class LicenseEvidenceError(ValueError):
    """License acquisition failed without exposing credentials or response bodies."""


@dataclass(frozen=True)
class EnrichedEvidence:
    """License evidence from an authorized immutable source.

    ``source`` is a provenance string of the form ``github:<owner>/<repo>@<full-commit>``
    — never a URL carrying credentials.
    """

    spdx_id: str
    source: str
    repo_commit: str


class GithubLicenseResolver:
    """Resolve commit-pinned GitHub license evidence using the ambient token.

    404 is a miss. Rate limits receive three attempts with bounded exponential
    backoff and Retry-After support; surfaced errors are redacted.
    """

    def resolve(self, repo_slug: str, repo_commit: str | None) -> EnrichedEvidence | None:
        token = os.environ.get(_GITHUB_TOKEN_ENV, "")
        if not token:
            # Fail fast with a clear, credential-free message before any HTTP
            # request: an empty ``Authorization: Bearer`` header is answered 401
            # and would otherwise surface only as a redacted generic failure
            # after the frozen record acquisition has already run.
            raise LicenseEvidenceError(
                "GITHUB_TOKEN is not set; license evidence enrichment requires "
                "a GitHub API token for the license endpoint. Export "
                "GITHUB_TOKEN and retry."
            )
        repo_slug = normalize_repo_slug(repo_slug)
        if not repo_slug:
            raise LicenseEvidenceError("invalid_repository")
        repo_commit = repo_commit or self._resolve_head_commit(repo_slug, token)
        if repo_commit is None:
            return None
        if re.fullmatch(r"[0-9a-fA-F]{40}", repo_commit) is None:
            raise LicenseEvidenceError("invalid_repository_revision")
        repo_commit = repo_commit.lower()
        data = self._request_json(
            f"{_GITHUB_API}/repos/{repo_slug}/license"
            + f"?ref={repo_commit}",
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
            raise LicenseEvidenceError(
                "invalid_license_response"
            )
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
        return commit if isinstance(commit, str) and re.fullmatch(r"[0-9a-fA-F]{40}", commit) else None

    def _request_json(self, url: str, token: str) -> dict[str, Any] | None:
        """GET one JSON document with bounded rate-limit backoff; 404 → None."""
        delay = _RATE_LIMIT_BASE_DELAY_S
        for attempt in range(_RATE_LIMIT_ATTEMPTS):
            request = urllib.request.Request(url, headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            })
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("invalid response shape")
                return payload
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                remaining = exc.headers.get("X-RateLimit-Remaining") if exc.headers else None
                rate_limited = exc.code == 403 and remaining == "0"
                if attempt + 1 < _RATE_LIMIT_ATTEMPTS and (rate_limited or retry_after):
                    try:
                        retry_delay = float(retry_after) if retry_after else delay
                    except ValueError:
                        retry_delay = delay
                    time.sleep(min(120.0, max(0.0, retry_delay)))
                    delay *= 2
                    continue
                break
            except (urllib.error.URLError, TimeoutError, ValueError):
                if attempt + 1 < _RATE_LIMIT_ATTEMPTS:
                    time.sleep(delay)
                    delay *= 2
                    continue
                break
        raise LicenseEvidenceError(
            "license_source_request_failed"
        ) from None

