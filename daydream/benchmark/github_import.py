"""Import normalized evidence from explicit private GitHub PRs.

This module delivers the full import surface: parse ``--pr``/``--pr-file``/
``--head`` targets; run six ordered preflight checks (binaries, ``gh``
authentication, identity, repository read access + immutable identity); fetch
and normalize a PR's header, submitted reviews, inline comments and
conversation comments (REST) plus all review threads/replies (GraphQL);
deterministically project import-time candidates; and atomically persist one
import file, one materialized case per requested head, and the ledger through
a single crash-consistent :class:`storage.Transaction`.

Every ``gh``/``git`` call routes through :mod:`daydream.git_ops`, so the
in-process ``fake_gh`` router intercepts it. Rate-limit retries are bounded
(``Retry-After`` honored, 60s cap); a fetch that exhausts them marks that PR
``fetch_failed`` in the ledger rather than silently dropping evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

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
    return git_ops.gh_auth_git_credential(root)


def _git_ls_remote(root: Path, url: str) -> str:
    """Run an authenticated ``git ls-remote <url>`` and return the refs text."""
    return git_ops.git_ls_remote(root, url)


class PreflightError(Exception):
    """A preflight check failed with an exact ``{code, message}`` pair."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"preflight failed: {code}: {message}")
        self.code = code
        self.message = message


@dataclass
class PreflightResult:
    """The authenticated identity + resolved repository captured by preflight."""

    login: str
    repository_id: int
    visibility: str


def _run_repo_view(root: Path, repo_slug: str) -> dict[str, Any]:
    """Resolve a repository's identity and the caller's read access to it."""
    proc = git_ops._run_gh(
        root,
        ["repo", "view", repo_slug, "--json", "id,nameWithOwner,url,visibility,defaultBranchRef"],
    )
    if proc.returncode != 0:
        raise PreflightError("no_access", f"cannot read repository {repo_slug}: {proc.stderr.strip()}")
    try:
        view = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise PreflightError("repo_unresolved", f"repo view returned invalid JSON: {exc}") from exc
    if not isinstance(view, dict) or view.get("id") is None:
        raise PreflightError("repo_unresolved", f"repo view of {repo_slug} returned no identity")
    return view


def _resolve_identity(root: Path, repo_slug: str) -> tuple[int, str]:
    """Fill the immutable repository identity atomically on first success.

    Stage ``benchmark.yaml`` (the only mutated file) through one
    :class:`Transaction` under the workspace lock, so a crash restores the
    whole before- or after-state. A concurrent partial state is surfaced as
    :class:`PreflightError` rather than repaired.
    """
    view = _run_repo_view(root, repo_slug)
    repository_id = int(view["id"])
    visibility = str(view.get("visibility") or "").lower()
    if visibility not in ("public", "private"):
        raise PreflightError("repo_unresolved", f"unrecognized visibility {visibility!r}")
    with storage.WorkspaceLock(root):
        raw = storage.load_yaml_strict(root / "benchmark.yaml")
        current = raw.get("source") or {}
        if current.get("repository_id") is not None or current.get("visibility") != "unresolved":
            raise PreflightError("repo_unresolved", "repository identity is already resolved")
        raw["source"]["repository_id"] = repository_id
        raw["source"]["visibility"] = visibility
        with storage.Transaction(
            root, op_id="identity-" + repo_slug.replace("/", "_"), kind="identity"
        ) as tx:
            tx.stage("benchmark.yaml", yaml.safe_dump(raw, sort_keys=False).encode("utf-8"))
            tx.commit()
    return repository_id, visibility


def preflight(root: Path, pr_count: int) -> PreflightResult:
    """Run the six fixed-order preflight checks before any fetch.

    Fails the run at the first failing check with an exact ``{code, message}``
    pair (:class:`PreflightError`). Repository identity resolves once and is
    immutable for the workspace's lifetime.
    """
    root = Path(root)
    if shutil.which("git") is None or shutil.which("gh") is None:
        raise PreflightError("missing_binary", "git and gh binaries must be reachable")

    status = _run_gh_preflight_status(root)
    if status.returncode != 0:
        raise PreflightError("not_authenticated", "gh is not authenticated to github.com")

    try:
        user = _run_gh_api_user(root)
    except git_ops.GitError as exc:
        raise PreflightError("auth_failed", str(exc)) from exc
    login = user.get("login") if isinstance(user, dict) else None
    if not login:
        raise PreflightError("auth_failed", "gh api user returned no login")

    raw = storage.load_yaml_strict(root / "benchmark.yaml")
    source = raw.get("source") or {}
    repo_slug = source.get("repository") or ""
    repository_id = source.get("repository_id")
    visibility = source.get("visibility", "unresolved")
    if repository_id is None and visibility == "unresolved":
        repository_id, visibility = _resolve_identity(root, repo_slug)
    elif (repository_id is None) != (visibility == "unresolved"):
        raise PreflightError("repo_unresolved", "repository identity is in a partial/corrupt state")
    if visibility not in ("public", "private"):
        raise PreflightError("repo_unresolved", "repository identity did not resolve to public/private")
    if not isinstance(repository_id, int):
        raise PreflightError("incomplete_resolution", "repository identity lacks a numeric repository_id")

    try:
        _git_ls_remote(root, f"https://github.com/{repo_slug}.git")
    except git_ops.GitError as exc:
        raise PreflightError("git_preflight_failed", str(exc)) from exc

    print(f"authenticated identity: {login}")
    print(f"repository visibility: {visibility}")
    print(f"requested PR count: {pr_count}")
    print(f"local destination: {root / 'imports'}")
    return PreflightResult(login=login, repository_id=repository_id, visibility=visibility)


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


_RATE_LIMIT_ATTEMPTS = 3
_RATE_LIMIT_MAX_SLEEP_S = 60.0


def _call_with_rate_limit_retry(
    call: Callable[[], Any],
) -> tuple[Any, git_ops.RateLimitError | None]:
    """Run *call* up to 3 times, sleeping ``min(retry_after, 60)`` between rate-limit failures.

    A non-rate-limit failure returns immediately. Returns ``(result, error)``
    where *error* is the classified :class:`git_ops.RateLimitError` when the
    last result is a rate-limited failure and ``None`` otherwise. After the
    third rate-limit attempt the last (failed) result is returned so the
    orchestrator can mark that PR ``fetch_failed`` — never a silent placeholder.
    """
    last = None
    last_rate_limit: git_ops.RateLimitError | None = None
    for attempt in range(_RATE_LIMIT_ATTEMPTS):
        proc = call()
        if proc.returncode == 0:
            return proc, None
        error = git_ops._gh_error_for(f"gh call failed: {proc.stderr.strip()}", proc.stderr)
        if not isinstance(error, git_ops.RateLimitError):
            return proc, None
        retry_after = error.retry_after if error.retry_after is not None else _RATE_LIMIT_MAX_SLEEP_S
        wait = min(retry_after, _RATE_LIMIT_MAX_SLEEP_S)
        if attempt < _RATE_LIMIT_ATTEMPTS - 1:
            time.sleep(wait)
        last = proc
        last_rate_limit = error
    return last, last_rate_limit


def _fetch_with_retry(root: Path, owner_repo: str, number: int) -> dict[str, Any]:
    """Fetch the singular ``pulls/<number>`` object with the bounded retry policy.

    The singular pulls endpoint returns one object, so it is fetched with the
    ``@json`` filter (not the array-flattening ``.[]``) and parsed as a
    single JSON value. A failed call raises :class:`git_ops.GitError`; a
    rate-limit that exhausts the retry budget raises
    :class:`_ImportRateLimitError`.
    """
    endpoint = f"repos/{owner_repo}/pulls/{number}"
    proc, rate_limit = _call_with_rate_limit_retry(
        lambda: git_ops._run_gh(root, ["api", endpoint, "--jq", "@json"])
    )
    if proc.returncode != 0:
        if rate_limit is not None:
            raise _ImportRateLimitError(f"gh api {endpoint} rate limited: {proc.stderr.strip()}")
        raise git_ops.GitError(f"gh api {endpoint} failed: {proc.stderr.strip()}")
    header = json.loads(proc.stdout)
    if not isinstance(header, dict):
        raise git_ops.GitError(f"gh gives no PR header for {owner_repo}#{number}")
    return header


def _rest(root: Path, endpoint: str) -> list[Any]:
    """Fetch one REST resource, paginating fully, returning the parsed NDJSON rows.

    ``--paginate`` walks Link headers so every page is retained (the fake serves
    the complete canned list in one NDJSON stream).
    """
    proc, rate_limit = _call_with_rate_limit_retry(
        lambda: git_ops._run_gh(root, ["api", "--paginate", endpoint, "--jq", ".[] | @json"])
    )
    if proc.returncode != 0:
        if rate_limit is not None:
            raise _ImportRateLimitError(f"gh api {endpoint} rate limited: {proc.stderr.strip()}")
        raise git_ops.GitError(f"gh api {endpoint} failed: {proc.stderr.strip()}")
    return _parse_ndjson(proc.stdout)


class _ImportRateLimitError(Exception):
    """A fetch exhausted its rate-limit retries; the PR becomes ``fetch_failed``."""


def _as_author(raw: dict) -> dict:
    author = raw.get("user") or {}
    return {"login": author.get("login", ""), "type": author.get("type", "User")}


def _record_common(author: dict, body: str) -> dict[str, Any]:
    """The author + body-hash + bot block shared by every evidence record builder."""
    return {
        "author": {"login": author.get("login", ""), "type": author.get("type", "User")},
        "body": body,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "is_bot": author.get("type") == "Bot",
    }


def _evidence_from_review(raw: dict[str, Any]) -> dict[str, Any]:
    db_id = int(raw["id"])
    submitted = raw.get("submitted_at")
    body = raw.get("body") or ""
    fields = {
        "source_id": f"github:review:{db_id}",
        "kind": "review",
        "database_id": db_id,
        "node_id": raw.get("node_id") or f"PRR_{db_id}",
        "created_at": raw.get("created_at") or submitted,
        "updated_at": raw.get("updated_at") or submitted,
        "submitted_at": submitted,
        "commit_id": raw.get("commit_id"),
        "original_commit_id": raw.get("original_commit_id"),
        "state": raw.get("state"),
        "url": raw.get("html_url") or "",
    }
    fields.update(_record_common(raw.get("user") or {}, body))
    return fields


def _evidence_from_inline(raw: dict[str, Any]) -> dict[str, Any]:
    db_id = int(raw["id"])
    body = raw.get("body") or ""
    subject_type = raw.get("subject_type")
    if subject_type is None:
        subject_type = "file" if raw.get("path") is None else "line"
    fields = {
        "source_id": f"github:inline_comment:{db_id}",
        "kind": "inline_comment",
        "database_id": db_id,
        "node_id": raw.get("node_id") or f"DIFF_{db_id}",
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
        "url": raw.get("html_url") or "",
    }
    fields.update(_record_common(raw.get("user") or {}, body))
    return fields


def _evidence_from_issue(raw: dict[str, Any]) -> dict[str, Any]:
    db_id = int(raw["id"])
    body = raw.get("body") or ""
    fields = {
        "source_id": f"github:issue_comment:{db_id}",
        "kind": "issue_comment",
        "database_id": db_id,
        "node_id": raw.get("node_id") or f"IC_{db_id}",
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
        "url": raw.get("html_url") or "",
    }
    fields.update(_record_common(raw.get("user") or {}, body))
    return fields


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
    fields = {
        "source_id": f"github:thread_comment:{db_id}",
        "kind": "thread_comment",
        "database_id": db_id,
        "node_id": comment.get("id") or f"TH_{db_id}",
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
        "url": comment.get("url") or "",
    }
    fields.update(_record_common(author, body))
    return fields


def _graphql_with_rate_limit_retry(root: Path, variables: dict[str, Any]) -> dict[str, Any]:
    """Call the paginated reviewThreads GraphQL query honoring the rate-limit retry policy.

    REST paths flow through :func:`_call_with_rate_limit_retry` (3 attempts,
    honoring Retry-After); the GraphQL query must follow the same policy so a
    transient rate limit is retried and, when exhausted, surfaces as
    :class:`_ImportRateLimitError` (recorded as ``rate_limit`` in the ledger,
    not ``fetch``).
    """
    last_rate_limit: git_ops.RateLimitError | None = None
    for attempt in range(_RATE_LIMIT_ATTEMPTS):
        try:
            resp = git_ops.gh_api(
                root,
                "graphql",
                method="POST",
                idempotent=True,
                input_data={"query": _REVIEW_THREADS_QUERY, "variables": variables},
            )
            return resp
        except git_ops.RateLimitError as exc:
            last_rate_limit = exc
            if attempt < _RATE_LIMIT_ATTEMPTS - 1:
                wait = exc.retry_after if exc.retry_after is not None else _RATE_LIMIT_MAX_SLEEP_S
                time.sleep(min(wait, _RATE_LIMIT_MAX_SLEEP_S))
    assert last_rate_limit is not None
    raise _ImportRateLimitError(
        f"gh api graphql reviewThreads rate limited: {last_rate_limit}"
    ) from last_rate_limit


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
        resp = _graphql_with_rate_limit_retry(root, variables)
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


def _evidence_signature_from_doc(doc: schema.ImportDocument) -> frozenset[tuple[str, str]]:
    return frozenset((e.source_id, e.body_sha256) for e in doc.evidence)


def _evidence_signature_from_raw(raw: dict[str, Any]) -> frozenset[tuple[str, str]]:
    return frozenset(
        (str(e.get("source_id")), str(e.get("body_sha256"))) for e in raw.get("evidence", [])
    )


def _case_materialize(
    doc: schema.ImportDocument,
    number: int,
    requested_heads: list[str],
    import_file: str,
    import_sha256: str,
    *,
    prior_curations: dict[str, dict[str, Any]] | None = None,
    changed: bool = False,
) -> list[tuple[str, str, dict[str, Any]]]:
    """One materialized case document per requested head.

    When *changed* is True and a prior curated case exists, its curation is
    carried over and flipped to ``stale`` with attestation cleared — findings
    and exclusions are never overwritten by a refresh.
    """
    pull_request = doc.pull_request
    base_sha = (pull_request.get("base") or {}).get("sha")
    out: list[tuple[str, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for head_token in requested_heads:
        head_sha = head_token if head_token != "final" else (pull_request.get("head") or {}).get("sha")
        if not head_sha or head_sha in seen:
            continue
        seen.add(head_sha)
        case_id = schema.case_id_for(number, head_sha)
        candidates = project_candidates(doc, head_sha)
        curation: dict[str, Any] = {
            "state": "draft",
            "snapshot_attested": False,
            "clean_attested": False,
            "gold_status": None,
            "findings": [],
            "exclusions": [],
            "case_exclusion": None,
        }
        if prior_curations and case_id in prior_curations:
            prior = prior_curations[case_id]
            curation = dict(prior)
            # Preserve prior curation whenever a curated case exists (unchanged
            # re-import / refresh must NOT wipe findings or attestation). Only
            # demote to stale (and clear attestation) when the evidence actually
            # drifted (changed=True).
            if changed and prior.get("state") in ("ready", "stale"):
                curation["state"] = "stale"
                curation["snapshot_attested"] = False
        case_doc: dict[str, Any] = {
            "schema_version": 1,
            "case_id": case_id,
            "pull_request": pull_request,
            "snapshot": {
                "status": "imported",
                "policy": "final_pr_head",
                "requested_head": head_token,
                "original_base_sha": base_sha,
                "original_head_sha": head_sha,
                "error": None,
            },
            "source": {"import_file": import_file, "import_sha256": import_sha256},
            "curation": curation,
            "candidates": [c.model_dump(mode="json") for c in candidates],
        }
        out.append((case_id, f"cases/{case_id}.yaml", case_doc))
    return out


def _ledger_replace(raw: dict[str, Any], entry: dict[str, Any]) -> None:
    """Replace (or append) one ``pull_requests[]`` entry, keeping stable order."""
    raw["pull_requests"] = [
        e for e in raw.get("pull_requests", []) if e.get("number") != entry["number"]
    ]
    raw["pull_requests"].append(entry)


def _stamp_fetched(
    raw: dict[str, Any],
    number: int,
    import_file: str,
    import_sha256: str,
    requested_heads: list[str],
    case_ids: list[str],
) -> None:
    schema.validate_pr_transition(
        _pending_pr_state(raw, number), "fetched"
    )
    _ledger_replace(
        raw,
        {
            "number": number,
            "import_state": "fetched",
            "import_file": import_file,
            "import_sha256": import_sha256,
            "error": None,
            "requested_heads": requested_heads,
            "case_ids": case_ids,
        },
    )
    for case_id in case_ids:
        # Replace any prior index row for this case_id so a re-import of the same
        # PR (incl. the fetched->fetched --refresh path) never leaves duplicate
        # cases[] rows. Mirrors _ledger_replace's replace-by-key semantics.
        raw["cases"] = [
            c for c in raw.get("cases", []) if c.get("case_id") != case_id
        ]
        raw["cases"].append(
            {"case_id": case_id, "pr_number": number, "case_file": f"cases/{case_id}.yaml"}
        )
    raw["cases"] = _sorted_cases(raw["cases"])


def _stages_failed(raw: dict[str, Any], number: int, code: str, message: str) -> None:
    schema.validate_pr_transition(_pending_pr_state(raw, number), "fetch_failed")
    _ledger_replace(
        raw,
        {
            "number": number,
            "import_state": "fetch_failed",
            "import_file": None,
            "import_sha256": None,
            "error": {"code": code, "message": message},
            "requested_heads": [],
            "case_ids": [],
        },
    )


def _pending_pr_state(raw: dict[str, Any], number: int) -> str:
    for entry in raw.get("pull_requests", []):
        if entry.get("number") == number:
            return entry.get("import_state", "pending")
    return "pending"


def _manifest_entry(raw: dict[str, Any], number: int) -> dict[str, Any] | None:
    """The ledger entry for *number*, or None when not yet imported."""
    for entry in raw.get("pull_requests", []):
        if entry.get("number") == number:
            return entry
    return None


def _sorted_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _key(c: dict[str, Any]) -> tuple[int, str, str]:
        return (int(c["pr_number"]), schema.head_sha_from_case_id(c["case_id"]), c["case_id"])

    return sorted(cases, key=_key)


def _manifest_bytes(raw: dict[str, Any]) -> bytes:
    return yaml.safe_dump(raw, sort_keys=False).encode("utf-8")


def _stage_fetch_failure(
    root: Path, raw: dict[str, Any], number: int, code: str, message: str
) -> None:
    """Atomically flip a PR's ledger entry to ``fetch_failed``.

    Stages only ``benchmark.yaml`` through one :class:`Transaction`; a failed
    fetch materializes no import/case file (the whole before/after ledger state
    is atomic).
    """
    _stages_failed(raw, number, code, message)
    with storage.Transaction(root, op_id=f"import-{number}", kind="import") as tx:
        tx.stage("benchmark.yaml", _manifest_bytes(raw))
        tx.commit()


def _prior_import_state(
    root: Path, raw: dict[str, Any], number: int
) -> tuple[frozenset[tuple[str, str]] | None, dict[str, dict[str, Any]], str]:
    """The prior snapshot signature, prior case curations, and the import file path.

    A missing/unreadable prior snapshot yields a ``None`` signature (so a
    refresh cannot compare); an unreadable curation file is skipped, never fatal.
    """
    import_file = f"imports/pr-{number:06d}.json"
    existing = _manifest_entry(raw, number)
    prior_sig: frozenset[tuple[str, str]] | None = None
    prior_curations: dict[str, dict[str, Any]] = {}
    if existing is not None and existing.get("import_state") == "fetched":
        try:
            prior_raw = storage.load_json_strict(root / existing["import_file"])
            prior_sig = _evidence_signature_from_raw(prior_raw)
        except Exception:
            prior_sig = None
        for case_id in existing.get("case_ids", []):
            try:
                cur = storage.load_yaml_strict(root / "cases" / f"{case_id}.yaml").get("curation")
                if isinstance(cur, dict):
                    prior_curations[case_id] = cur
            except Exception:
                pass
    return prior_sig, prior_curations, import_file


def _import_one_pr(
    root: Path,
    raw: dict[str, Any],
    repo: str,
    number: int,
    requested_heads: list[str],
    *,
    refresh: bool,
) -> int:
    """Fetch + materialize one PR, or stage its failure: 0 on success, 1 on failure."""
    prior_sig, prior_curations, import_file = _prior_import_state(root, raw, number)
    try:
        doc = fetch_and_normalize(root, repo, number)
        changed = refresh and prior_sig is not None and prior_sig != _evidence_signature_from_doc(doc)
        import_bytes = json.dumps(doc.model_dump(mode="json"), indent=2).encode("utf-8")
        import_sha256 = hashlib.sha256(import_bytes).hexdigest()
        cases = _case_materialize(
            doc, number, requested_heads, import_file, import_sha256,
            prior_curations=prior_curations, changed=changed,
        )
        with storage.Transaction(root, op_id=f"import-{number}", kind="import") as tx:
            tx.stage(import_file, import_bytes)
            for _, case_path, case_doc in cases:
                tx.stage(case_path, yaml.safe_dump(case_doc, sort_keys=False).encode("utf-8"))
            _stamp_fetched(
                raw,
                number,
                import_file,
                import_sha256,
                requested_heads,
                [c[0] for c in cases],
            )
            tx.stage("benchmark.yaml", _manifest_bytes(raw))
            tx.commit()
        return 0
    except _ImportRateLimitError as exc:
        _stage_fetch_failure(root, raw, number, "rate_limit", str(exc))
        return 1
    except (git_ops.GitError, schema.TransitionError, storage.WorkspaceError, PreflightError) as exc:
        _stage_fetch_failure(root, raw, number, "fetch", str(exc))
        return 1


def run_import_prs(
    root: Path,
    pr_numbers: list[int],
    heads: list[str] | None = None,
    refresh: bool = False,
) -> int:
    """Import each PR's evidence into one atomic import ledger/case transaction.

    Runs startup recovery then preflight (identity idempotent), then for each PR
    writes its import file, one case per requested head, and the ledger through
    one :class:`Transaction` (``benchmark.yaml`` last). A failed fetch stages no
    import/case file — only a ledger flip to ``fetch_failed`` with an exact
    error. The overall exit is non-zero when any PR failed.
    """
    root = Path(root)
    requested_heads: list[str] = []
    seen_heads: set[str] = set()
    for head in ["final", *(heads or [])]:
        if head not in seen_heads:
            seen_heads.add(head)
            requested_heads.append(head)
    exit_code = 0
    with storage.WorkspaceLock(root):
        storage.recover_startup(root)
        preflight(root, len(pr_numbers))
        raw = storage.load_yaml_strict(root / "benchmark.yaml")
        repo = raw.get("source", {}).get("repository") or ""
        for number in pr_numbers:
            if _import_one_pr(root, raw, repo, number, requested_heads, refresh=refresh):
                exit_code = 1
    return exit_code


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
) -> schema.ImportDocument:
    """Fetch one PR's full evidence set through REST and normalize it.

    Pulls the PR header, then every submitted review, top-level inline comment,
    and conversation comment in order. Every retrieved record is retained as an
    :class:`EvidenceRecord` with a stable source ID and body hash; ``is_bot`` is
    derived from the author type and never filters. Failure of any call raises
    :class:`GitError` — never a silent default.
    """
    header = _fetch_with_retry(root, owner_repo, number)

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
