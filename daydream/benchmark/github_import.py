"""Import normalized evidence from explicit private GitHub PRs.

Task 0 spike: prove the importer's ``gh``/``git`` calls route through
:mod:`daydream.git_ops` (so the in-process ``fake_gh`` router intercepts
them) before any collection logic is written. The functions here are the
thin preflight call sites; later tasks build the full
:func:`fetch_and_normalize` / :func:`preflight` / :func:`run_import_prs`
surface on top of them.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daydream import git_ops
from daydream.benchmark import schema, storage


def _run_gh_preflight_status(root: Path):
    """Run ``gh auth status --hostname github.com`` (exit code is the contract)."""
    return git_ops._run_gh(root, ["auth", "status", "--hostname", "github.com"])


def _run_gh_api_user(root: Path) -> dict:
    """Return the authenticated GitHub user record from ``gh api user``."""
    proc = git_ops._run_gh(root, ["api", "user"])
    if proc.returncode != 0:
        raise git_ops.GitError(f"gh api user failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def _gh_auth_git_credential(root: Path) -> str:
    """Run the command-scoped git credential helper and return its protocol text."""
    proc = git_ops._run_gh(root, ["auth", "git-credential"])
    if proc.returncode != 0:
        raise git_ops.GitError(f"gh auth git-credential failed: {proc.stderr.strip()}")
    return proc.stdout


def _git_ls_remote(root: Path, url: str) -> str:
    """Run an authenticated ``git ls-remote <url>`` and return the refs text."""
    proc = git_ops._run_git(root, ["ls-remote", url])
    return proc.stdout


def _now_rfc3339() -> str:
    """Current UTC time as an RFC3339 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_ndjson(text: str) -> list[Any]:
    """Parse ``gh `` --jq '.[] | @json'`` NDJSON output into a list of values.

    A missing required/failed line or a non-JSON line surfaces as :class:`GitError`
    rather than being silently defaulted.
    """
    values: list[Any] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise git_ops.GitError(f"gh returned non-JSON line {lineno}: {exc}") from exc
    return values


def _rest(root: Path, endpoint: str) -> list[Any]:
    """Fetch one REST resource, paginating fully, returning the parsed NDJSON rows.

    ``--paginate`` walks Link headers so every page is retained (the fake serves
    the complete canned list in one NDJSON stream).
    """
    proc = git_ops._run_gh(root, ["api", "--paginate", endpoint, "--jq", ".[] | @json"])
    if proc.returncode != 0:
        raise git_ops.GitError(f"gh api {endpoint} failed: {proc.stderr.strip()}")
    return _parse_ndjson(proc.stdout)


def _as_author(raw: dict) -> dict:
    author = raw.get("user") or {}
    return {"login": author.get("login", ""), "type": author.get("type", "User")}


def _evidence_from_review(raw: dict[str, Any]) -> dict[str, Any]:
    db_id = int(raw["id"])
    submitted = raw.get("submitted_at")
    body = raw.get("body") or ""
    return {
        "source_id": f"github:review:{db_id}",
        "kind": "review",
        "database_id": db_id,
        "node_id": raw.get("node_id") or f"PRR_{db_id}",
        "author": _as_author(raw),
        "body": body,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "created_at": raw.get("created_at") or submitted,
        "updated_at": raw.get("updated_at") or submitted,
        "submitted_at": submitted,
        "commit_id": raw.get("commit_id"),
        "original_commit_id": raw.get("original_commit_id"),
        "state": raw.get("state"),
        "is_bot": (raw.get("user") or {}).get("type") == "Bot",
        "url": raw.get("html_url") or "",
    }


def _evidence_from_inline(raw: dict[str, Any]) -> dict[str, Any]:
    db_id = int(raw["id"])
    body = raw.get("body") or ""
    subject_type = raw.get("subject_type")
    if subject_type is None:
        subject_type = "file" if raw.get("path") is None else "line"
    return {
        "source_id": f"github:inline_comment:{db_id}",
        "kind": "inline_comment",
        "database_id": db_id,
        "node_id": raw.get("node_id") or f"DIFF_{db_id}",
        "author": _as_author(raw),
        "body": body,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
        "commit_id": raw.get("commit_id"),
        "original_commit_id": raw.get("original_commit_id"),
        "path": raw.get("path"),
        "original_path": raw.get("original_path"),
        "line": raw.get("line"),
        "start_line": raw.get("start_line"),
        "original_line": raw.get("original_line"),
        "thread_id": None,
        "reply_to_id": str(raw["in_reply_to_id"]) if raw.get("in_reply_to_id") is not None else None,
        "subject_type": subject_type,
        "side": raw.get("side"),
        "start_side": raw.get("start_side"),
        "is_bot": bool((raw.get("user") or {}).get("type") == "Bot"),
        "url": raw.get("html_url") or "",
    }


def _evidence_from_issue(raw: dict[str, Any]) -> dict[str, Any]:
    db_id = int(raw["id"])
    body = raw.get("body") or ""
    return {
        "source_id": f"github:issue_comment:{db_id}",
        "kind": "issue_comment",
        "database_id": db_id,
        "node_id": raw.get("node_id") or f"IC_{db_id}",
        "author": _as_author(raw),
        "body": body,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
        "is_bot": bool((raw.get("user") or {}).get("type") == "Bot"),
        "url": raw.get("html_url") or "",
    }


def _payload_sha256(records: list[schema.EvidenceRecord]) -> str:
    """sha256 over the canonical JSON of the evidence list."""
    canonical = json.dumps(
        [r.model_dump(mode="json") for r in records],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _repository_block(root: Path, owner_repo: str) -> dict[str, Any]:
    """The resolved repository identity for the import.

    Prefers the workspace manifest's resolved ``source`` (set by preflight) so
    the immutable repository id/visibility flow into the import; falls back to
    a default private identity when no manifest exists (unit tests drive
    ``fetch_and_normalize`` directly).
    """
    try:
        raw = storage.load_yaml_strict(root / "benchmark.yaml")
        source = raw.get("source") or {}
        visibility = source.get("visibility", "unresolved")
        return {
            "id": source.get("repository_id") or 0,
            "name_with_owner": source.get("repository") or owner_repo,
            "visibility": "public" if visibility == "public" else "private",
        }
    except Exception:
        return {"id": 0, "name_with_owner": owner_repo, "visibility": "private"}


def fetch_and_normalize(
    root: Path,
    owner_repo: str,
    number: int,
    heads: list[str],
) -> schema.ImportDocument:
    """Fetch one PR's full evidence set through REST and normalize it.

    Pulls the PR header, then every submitted review, top-level inline comment,
    and conversation comment in order. Every retrieved record is retained as an
    :class:`EvidenceRecord` with a stable source ID and body hash; ``is_bot`` is
    derived from the author type and never filters. Failure of any call raises
    :class:`GitError` — never a silent default.
    """
    del heads  # requested heads are consumed by the orchestrator, not the fetch
    header_rows = _rest(root, f"repos/{owner_repo}/pulls/{number}")
    if not header_rows:
        raise git_ops.GitError(f"gh gives no PR header for {owner_repo}#{number}")
    header = header_rows[0]

    evidence: list[dict[str, Any]] = []
    for raw in _rest(root, f"repos/{owner_repo}/pulls/{number}/reviews"):
        evidence.append(_evidence_from_review(raw))
    for raw in _rest(root, f"repos/{owner_repo}/pulls/{number}/comments"):
        evidence.append(_evidence_from_inline(raw))
    for raw in _rest(root, f"repos/{owner_repo}/issues/{number}/comments"):
        evidence.append(_evidence_from_issue(raw))

    records = [schema.EvidenceRecord.model_validate(e) for e in evidence]
    base = header.get("base") or {}
    head = header.get("head") or {}
    pull_request = {
        "number": header["number"],
        "url": header.get("url") or "",
        "title": header.get("title") or "",
        "state": header.get("state") or "",
        "base": {"sha": base.get("sha"), "ref": base.get("ref")},
        "head": {"sha": head.get("sha")},
        "created_at": header.get("created_at"),
        "updated_at": header.get("updated_at"),
        "author": _as_author(header),
    }
    return schema.ImportDocument.model_validate(
        {
            "schema_version": 1,
            "repository": _repository_block(root, owner_repo),
            "pull_request": pull_request,
            "evidence": [e.model_dump(mode="json") for e in records],
            "fetch": {
                "fetched_at": _now_rfc3339(),
                "etag": None,
                "payload_sha256": _payload_sha256(records),
            },
        }
    )