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
import re
from dataclasses import dataclass
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


class ImportTargetError(Exception):
    """An import-prs target (PR number/URL/file line/head SHA) failed to parse."""


_PR_URL_RE = re.compile(r"^https://github\.com/([^/]+)/([^/]+)/pull/(\d+)$")


@dataclass
class ImportTargets:
    """The deduplicated PR targets + requested heads for one import run."""

    pr_numbers: list[int]
    requested_heads: list[str]


def _parse_pr_token(token: str) -> int:
    """Parse one CLI arg or file line to a PR number, raising on anything else."""
    token = token.strip()
    if token.isdigit():
        return int(token)
    match = _PR_URL_RE.match(token)
    if match is not None:
        return int(match.group(3))
    raise ImportTargetError(
        f"invalid PR target {token!r} (expected a PR number or https://github.com/OWNER/REPO/pull/N)"
    )


def parse_import_targets(
    pr_args: list[str],
    pr_files: list[Path],
    heads: list[str],
) -> ImportTargets:
    """Resolve CLI ``--pr`` args + ``--pr-file`` lines + ``--head`` SHAs.

    Number/URL/file selections merge CLI-first then file in order and dedupe
    to the first-seen, stable order. ``requested_heads`` always starts with
    ``"final"`` (the PR's default head) followed by the validated 40-hex
    ``heads``. An unparseable token or malformed head raises
    :class:`ImportTargetError` naming the offending value.
    """
    numbers: list[int] = []
    seen: set[int] = set()

    def _add(number: int) -> None:
        if number not in seen:
            seen.add(number)
            numbers.append(number)

    for arg in pr_args:
        _add(_parse_pr_token(arg))
    for pr_file in pr_files:
        for line in Path(pr_file).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            _add(_parse_pr_token(line))

    validated_heads: list[str] = []
    for head in heads:
        if re.fullmatch(r"[0-9a-f]{40}", head) is None:
            raise ImportTargetError(f"invalid head SHA {head!r} (expected 40-hex)")
        validated_heads.append(head)
    return ImportTargets(pr_numbers=numbers, requested_heads=["final", *validated_heads])


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


_REVIEW_THREADS_QUERY = """
query ReviewThreads($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 50, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id isResolved isOutdated isResolvedBy subjectType path line originalLine side startSide
          comments(first: 100) {
            nodes { id databaseId body author { login type isBot } createdAt updatedAt url replyTo { id } }
          }
        }
      }
    }
  }
}
"""


def _evidence_from_thread(thread: dict[str, Any], comment: dict[str, Any]) -> dict[str, Any]:
    db_id = int(comment["databaseId"])
    body = comment.get("body") or ""
    author = comment.get("author") or {}
    subject = str(thread.get("subjectType") or "").lower()
    subject_type = subject if subject in ("line", "file") else None
    return {
        "source_id": f"github:thread_comment:{db_id}",
        "kind": "thread_comment",
        "database_id": db_id,
        "node_id": comment.get("id") or f"TH_{db_id}",
        "author": {"login": author.get("login", ""), "type": author.get("type", "User")},
        "body": body,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "created_at": comment.get("createdAt"),
        "updated_at": comment.get("updatedAt") or comment.get("createdAt"),
        "subject_type": subject_type,
        "side": thread.get("side"),
        "start_side": thread.get("startSide"),
        "path": thread.get("path"),
        "line": thread.get("line"),
        "original_line": thread.get("originalLine"),
        "resolved": bool(thread.get("isResolved", False)),
        "outdated": bool(thread.get("isOutdated", False)),
        "thread_id": thread.get("id"),
        "reply_to_id": (comment.get("replyTo") or {}).get("id"),
        "is_bot": bool(author.get("type") == "Bot"),
        "url": comment.get("url") or "",
    }


def _graphql_review_threads(root: Path, owner_repo: str, number: int) -> list[dict[str, Any]]:
    """Paginate every review thread (and reply) for a PR via GraphQL.

    A GraphQL ``errors`` block, a missing ``data.repository.pullRequest.reviewThreads``
    key, or a failed call raises :class:`GitError` — never a silent empty fallback.
    """
    owner, name = owner_repo.split("/", 1)
    all_nodes: list[dict[str, Any]] = []
    after: str | None = None
    while True:
        variables: dict[str, Any] = {"owner": owner, "name": name, "number": number}
        if after is not None:
            variables["after"] = after
        resp = git_ops.gh_api(
            root,
            "graphql",
            method="POST",
            idempotent=True,
            input_data={"query": _REVIEW_THREADS_QUERY, "variables": variables},
        )
        if not isinstance(resp, dict):
            raise git_ops.GitError("graphql reviewThreads returned a non-object response")
        if resp.get("errors"):
            raise git_ops.GitError(f"graphql reviewThreads failed: {resp['errors']}")
        try:
            threads = resp["data"]["repository"]["pullRequest"]["reviewThreads"]
        except (KeyError, TypeError) as exc:
            raise git_ops.GitError(f"graphql response missing reviewThreads: {exc}") from exc
        all_nodes.extend(threads.get("nodes") or [])
        page_info = threads.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        after = page_info.get("endCursor")
        if after is None:
            raise git_ops.GitError("graphql reviewThreads hasNextPage without an endCursor")
    return all_nodes


def _normalize_body(body: str) -> str:
    """Normalize line endings and strip trailing whitespace at the document end.

    CRLF/CR are converted to LF; internal Markdown whitespace is preserved.
    """
    return body.replace("\r\n", "\n").replace("\r", "\n").rstrip()


_MARKDOWN_PREFIX = re.compile(r"^(#{1,6}\s+|[-*]\s+)")


def _derive_title(body: str) -> str:
    """The bounded title from the first nonblank line of a body."""
    for line in body.split("\n"):
        if not line.strip():
            continue
        title = " ".join(line.split())
        return _MARKDOWN_PREFIX.sub("", title, count=1)
    return ""


def _title_ok(title: str) -> bool:
    """True when *title* is non-empty and within the 500 UTF-8 byte bound."""
    if not title:
        return False
    return 0 < len(title.encode("utf-8")) <= 500


def _anchor_location(
    evidence: schema.EvidenceRecord,
) -> tuple[schema.Location | None, str | None]:
    """Project a RIGHT-side inline anchor to a ``Location``, or the block reason.

    Returns ``(location, None)`` on a usable anchor, or ``(None, reason)`` when
    the anchor is unusable or the comment is not a right-side line anchor.
    """
    if evidence.subject_type == "file":
        return None, None
    if evidence.side == "LEFT" or evidence.start_side == "LEFT":
        return None, "side"
    path = evidence.original_path or evidence.path
    start = evidence.start_line or evidence.line
    end = evidence.line or evidence.start_line
    if not path or start is None or end is None or start < 1 or end < start:
        return None, "anchor"
    return schema.Location(path=path, start_line=start, end_line=end), None


def _project_one(evidence: schema.EvidenceRecord, head_sha: str) -> schema.Candidate:
    body = _normalize_body(evidence.body)
    title = _derive_title(body)
    title_ok = _title_ok(title)

    location: schema.Location | None = None
    exact = title_ok
    reason: str | None = None
    if evidence.kind == "inline_comment":
        loc, anchor_reason = _anchor_location(evidence)
        location = loc
        if anchor_reason is not None:
            exact = False
            reason = anchor_reason
    # review bodies are file-agnostic: no location, no side constraint

    if not title_ok:
        exact = False
        reason = "title"
    if evidence.commit_id != head_sha:
        exact = False
        reason = "commit"
    if evidence.outdated:
        exact = False
        reason = "outdated"
    if evidence.dismissed:
        exact = False
        reason = "dismissed"

    return schema.Candidate(
        source_id=evidence.source_id,
        title=title,
        body=body,
        severity=None,
        location=location,
        exact_acceptable=exact,
        not_exact_reason=reason if not exact else None,
    )


def project_candidates(
    doc: schema.ImportDocument, head_sha: str
) -> list[schema.Candidate]:
    """The deterministic §5 projection of root comments + non-pure reviews.

    Root inline comments with a non-empty body and ``COMMENTED`` /
    ``CHANGES_REQUESTED`` review bodies become candidates; pure approvals,
    replies, and conversation comments are retained as evidence only.
    Projection never raises — an underivable title, a LEFT-side anchor, an
    unusable anchor, an off-head commit, an outdated, or a dismissed record all
    set ``exact_acceptable`` low with a ``not_exact_reason``.
    """
    cands: list[schema.Candidate] = []
    for evidence in doc.evidence:
        if not evidence.body:
            continue
        if evidence.kind == "inline_comment":
            if evidence.reply_to_id is not None:
                continue  # replies are evidence, not candidates
        elif evidence.kind == "review":
            if evidence.state not in ("COMMENTED", "CHANGES_REQUESTED"):
                continue
        else:
            continue
        cands.append(_project_one(evidence, head_sha))
    return cands


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

    for thread in _graphql_review_threads(root, owner_repo, number):
        for comment in thread.get("comments", {}).get("nodes", []):
            evidence.append(_evidence_from_thread(thread, comment))

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