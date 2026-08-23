"""Import normalized evidence from explicit private GitHub PRs.

This module delivers the full import surface: parse ``--pr``/``--pr-file``/
``--head`` targets; run six ordered preflight checks (binaries, ``gh``
authentication, identity, repository read access + immutable identity); fetch
and normalize a PR's header, submitted reviews, inline comments and
conversation comments (REST) plus all review threads/replies (GraphQL);
deterministically project import-time candidates; and atomically persist one
import file, one frozen case per requested head (a ``ready|unreplayable``
snapshot dict + its deterministic bundle), and the ledger through a single
crash-consistent :class:`storage.Transaction`.

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
from typing import Any, Literal, cast

import yaml

from daydream import git_ops
from daydream.benchmark import schema, snapshot, storage


def _run_gh_preflight_status(root: Path):
    """Run ``gh auth status --hostname github.com`` (exit code is the contract)."""
    return git_ops._run_gh(root, ["auth", "status", "--hostname", "github.com"])


def _run_gh_api_user(root: Path) -> dict:
    """Return the authenticated GitHub user record from ``gh api user``."""
    proc = git_ops._run_gh(root, ["api", "user"])
    if proc.returncode != 0:
        raise git_ops.GitError(f"gh api user failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


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
    """The authenticated identity + verified repository captured by preflight."""

    login: str
    repository_id: str
    visibility: str


def _run_repo_view(root: Path, repo_slug: str) -> dict[str, Any]:
    """Fetch the repository's current identity and the caller's read access to it."""
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
    if not isinstance(view, dict):
        raise PreflightError("repo_unresolved", f"repo view of {repo_slug} returned no identity")
    return view


def _verify_repo_view(view: dict[str, Any], repo_slug: str) -> tuple[str, Literal["public", "private"]]:
    """Verify a repo view matches the workspace's exact repository identity.

    Returns ``(repository_id, visibility)``. Every mismatch, missing id, or
    unrecognized visibility fails closed with :class:`PreflightError` — the
    node id is an opaque string like ``R_kgD...`` and is never cast to int.
    """
    name_with_owner = view.get("nameWithOwner")
    # GitHub OWNER/REPO slugs are case-insensitive, so compare with case
    # folding: a repo initialized with non-canonical casing is valid.
    if (name_with_owner or "").lower() != repo_slug.lower():
        raise PreflightError(
            "repo_mismatch", f"repo view returned {name_with_owner!r}, expected {repo_slug!r}"
        )
    if (view.get("url") or "").lower() != f"https://github.com/{repo_slug}".lower():
        raise PreflightError(
            "repo_mismatch",
            f"repo view url {view.get('url')!r} does not canonicalize to https://github.com/{repo_slug}",
        )
    repository_id = view.get("id")
    if not isinstance(repository_id, str) or not repository_id.strip() or repository_id.strip().isdigit():
        raise PreflightError(
            "repo_unresolved", f"repo view of {repo_slug} returned no opaque node id"
        )
    visibility = str(view.get("visibility") or "").lower()
    if visibility not in ("public", "private"):
        raise PreflightError("repo_unresolved", f"unrecognized visibility {visibility!r}")
    return repository_id, cast(Literal["public", "private"], visibility)


def _persist_identity(root: Path, repo_slug: str, repository_id: str, visibility: str) -> None:
    """Stage the resolved identity atomically (the only mutation during resolve).

    Stage ``benchmark.yaml`` through one :class:`Transaction` under the
    workspace lock, so a crash restores the whole before- or after-state. A
    concurrent partial state is surfaced as :class:`PreflightError`.
    """
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


def preflight(root: Path, pr_count: int) -> PreflightResult:
    """Run the six fixed-order preflight checks before any fetch.

    Fails the run at the first failing check with an exact ``{code, message}``
    pair (:class:`PreflightError`). Exact repository identity + read access is
    re-verified **on every call** (every import and ``--refresh``) before any
    authoring-file, manifest, or bundle mutation; a mismatch or lost access
    fails closed.
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
    stored_repository_id = source.get("repository_id")
    stored_visibility = source.get("visibility", "unresolved")

    # Re-verify current identity + read access on every run, before mutation.
    view = _run_repo_view(root, repo_slug)
    repository_id, visibility = _verify_repo_view(view, repo_slug)

    needs_persist = stored_repository_id is None and stored_visibility == "unresolved"
    if (stored_repository_id is None) != (stored_visibility == "unresolved"):
        raise PreflightError("repo_unresolved", "repository identity is in a partial/corrupt state")
    elif not needs_persist and (stored_repository_id != repository_id or stored_visibility != visibility):
        raise PreflightError(
            "repo_mismatch",
            f"repository identity changed (stored {stored_repository_id}/{stored_visibility}, "
            f"verified {repository_id}/{visibility})",
        )

    try:
        _git_ls_remote(root, f"https://github.com/{repo_slug}.git")
    except git_ops.GitError as exc:
        raise PreflightError("git_preflight_failed", str(exc)) from exc

    # The read-access gate passed, so the fresh identity may now be persisted:
    # never stage the identity before repository read access is confirmed
    # (a failed gate must leave benchmark.yaml untouched).
    if needs_persist:
        _persist_identity(root, repo_slug, repository_id, visibility)

    # Record the successful verification (never on a failed run): a mode-0600
    # ledger so ``status`` can surface whether the last import/refresh actually
    # re-verified repository identity + read access.
    ledger = schema.PreflightLedger(
        last_verified_at=_now_rfc3339(),
        repository=repo_slug,
        repository_id=repository_id,
        visibility=visibility,
        matched=True,
    )
    storage.atomic_write_json(
        root / "runtime" / "preflight.json", ledger.model_dump(), mode=0o600
    )

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
    """The deduplicated PR targets + requested heads for one import run.

    ``requested_heads`` is the flat union (back-compat consumers); ``pr_heads``
    maps each requested PR number to its own head list (including ``"final"``)
    so a ``PR=<40-hex>`` binding is honored only for the PR it names.
    """

    pr_numbers: list[int]
    requested_heads: list[str]
    pr_heads: dict[int, list[str]]


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
    ``"final"`` (the PR's default head). A ``--head`` token is either a bare
    40-hex SHA (back-compat: applied to every requested PR) or
    ``<PR_NUMBER>=<40-hex>``, which ties an explicit head to *that* PR (the
    ``pr_heads`` map) and keeps a distinct import/snapshot target alongside
    the PR's default head. A bound ``PR`` number must itself be requested
    (else the binding is silently dropped) and otherwise pushes an
    :class:`ImportTargetError`. An unparseable token or malformed head raises
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

    per_pr: dict[int, list[str]] = {n: [] for n in numbers}
    all_valid: list[str] = []
    for head in heads:
        sha = head
        bound_pr: int | None = None
        if "=" in head:
            pr_part, _, rhs = head.partition("=")
            if not pr_part.isdigit() or not int(pr_part) > 0:
                raise ImportTargetError(
                    f"invalid head token {head!r} (expected PR=<40-hex>)"
                )
            bound_pr = int(pr_part)
            sha = rhs
        if re.fullmatch(r"[0-9a-f]{40}", sha) is None:
            raise ImportTargetError(
                f"invalid head SHA {head!r} (expected bare 40-hex or PR=<40-hex>)"
            )
        if bound_pr is not None:
            if bound_pr not in per_pr:
                raise ImportTargetError(
                    f"head {head!r} references PR {bound_pr} which is not among "
                    f"the requested PR targets"
                )
            per_pr[bound_pr].append(sha)
        else:
            # Bare 40-hex: back-compat, applied to every requested PR.
            for number in numbers:
                per_pr[number].append(sha)
        all_valid.append(sha)
    pr_heads: dict[int, list[str]] = {
        n: ["final", *dict.fromkeys(per_pr[n])] for n in numbers
    }
    return ImportTargets(
        pr_numbers=numbers,
        requested_heads=["final", *dict.fromkeys(all_valid)],
        pr_heads=pr_heads,
    )


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
        "review_id": str(raw["pull_request_review_id"]) if raw.get("pull_request_review_id") is not None else None,
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
            pageInfo { hasNextPage endCursor }
            nodes { id databaseId body author { login type isBot } createdAt updatedAt url replyTo { id } }
          }
        }
      }
    }
  }
}
"""

_THREAD_COMMENTS_QUERY = """
query ThreadComments($threadId: ID!, $commentsAfter: String) {
  node(id: $threadId) {
    ... on PullRequestReviewThread {
      comments(first: 100, after: $commentsAfter) {
        pageInfo { hasNextPage endCursor }
        nodes { id databaseId body author { login type isBot } createdAt updatedAt url replyTo { id } }
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


def _canonical_comment_from_thread(
    thread: dict[str, Any], comment: dict[str, Any]
) -> dict[str, Any]:
    """Build a canonical ``inline_comment`` record from GraphQL thread fields.

    Used for thread comments with no REST counterpart (never dropped, and never
    emitted with the ``thread_comment`` kind). The thread carries the anchors;
    commit anchors are absent because only REST exposes them.
    """
    rec = _evidence_from_thread(thread, comment)
    db_id = rec["database_id"]
    rec["source_id"] = f"github:inline_comment:{db_id}"
    rec["kind"] = "inline_comment"
    rec["commit_id"] = None
    rec["original_commit_id"] = None
    rec["review_id"] = None
    rec["dismissed"] = False
    return rec


def _reconcile_inline_evidence(
    inline_records: list[dict[str, Any]],
    thread_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge REST inline comments and GraphQL thread comments into one record per comment.

    The REST inline record is the canonical base — it carries the anchors only
    REST exposes (``commit_id``/``original_commit_id``, ``path``/``original_path``,
    line/start line, ``subject_type``, ``side``/``start_side``, ``url``). GraphQL
    thread state (``thread_id``, ``resolved``, ``outdated``, and the reply link
    when REST lacks it) is overlaid by ``database_id`` with a ``node_id``
    fallback, so every comment that exists in both feeds yields exactly one
    record. A thread comment with no REST counterpart is surfaced as a canonical
    ``inline_comment`` record built from thread fields — never dropped, never
    emitted as ``thread_comment``. Exactly one record is produced per comment
    database id.
    """
    inline_by_db = {rec["database_id"]: rec for rec in inline_records}
    inline_by_node = {rec["node_id"]: rec for rec in inline_records if rec.get("node_id")}
    canonical: list[dict[str, Any]] = [dict(rec) for rec in inline_records]
    canonical_by_db = {rec["database_id"]: rec for rec in canonical}
    for thread in thread_nodes:
        nodes = thread.get("comments", {}).get("nodes")
        if nodes is None:
            continue  # a thread with no comments contributes nothing
        for comment in nodes:
            if not isinstance(comment, dict) or "databaseId" not in comment:
                raise git_ops.GitError(
                    f"graphql review thread {thread.get('id')} comment node missing databaseId"
                )
            db_id = int(comment["databaseId"])
            base = inline_by_db.get(db_id)
            if base is None and comment.get("id"):
                base = inline_by_node.get(comment["id"])
            if base is not None:
                rec = canonical_by_db[base["database_id"]]
            else:
                rec = _canonical_comment_from_thread(thread, comment)
                canonical.append(rec)
            rec["thread_id"] = thread.get("id")
            rec["resolved"] = bool(thread.get("isResolved", False))
            rec["outdated"] = bool(thread.get("isOutdated", False))
            if rec.get("reply_to_id") is None:
                rec["reply_to_id"] = (comment.get("replyTo") or {}).get("id")
    return canonical


def _join_dismissal(
    canonical: list[dict[str, Any]], review_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Mark ``dismissed`` on comments whose review was dismissed.

    Dismissal is joined deterministically from the review set already fetched:
    REST review ``state == "DISMISSED"`` mapped through the comment's
    ``pull_request_review_id``. Pure dict join — a comment whose review id maps
    to no DISMISSED review (or to no review at all) simply keeps the default
    ``dismissed=False``.
    """
    states = {str(int(raw["id"])): raw.get("state") for raw in review_records}
    for rec in canonical:
        review_id = rec.get("review_id")
        if rec.get("kind") == "inline_comment" and review_id and states.get(review_id) == "DISMISSED":
            rec["dismissed"] = True
    return canonical


def _graphql_with_rate_limit_retry(
    root: Path, variables: dict[str, Any], *, query: str = _REVIEW_THREADS_QUERY
) -> dict[str, Any]:
    """Call one GraphQL query honoring the rate-limit retry policy.

    REST paths flow through :func:`_call_with_rate_limit_retry` (3 attempts,
    honoring Retry-After); GraphQL queries must follow the same policy so a
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
                input_data={"query": query, "variables": variables},
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


def _next_cursor(page_info: dict[str, Any], *, context: str) -> str | None:
    """Return the next ``endCursor``, or ``None`` when there is no next page.

    Fails closed: a connection reporting ``hasNextPage`` without an
    ``endCursor`` raises :class:`GitError` naming ``context`` instead of
    silently dropping a page.
    """
    if not page_info.get("hasNextPage"):
        return None
    after = page_info.get("endCursor")
    if after is None:
        raise git_ops.GitError(f"graphql {context} hasNextPage without an endCursor")
    return after


def _graphql_thread_comments(
    root: Path, thread_id: str, *, after: str | None = None
) -> dict[str, Any]:
    """Fetch one nested page of a review thread's comments via ``node(id:)``.

    An ``errors`` block, a missing/null ``data.node``, or a ``comments``
    connection lacking ``pageInfo``/``nodes`` raises :class:`GitError` with a
    message naming the thread id — never a silent empty fallback, so a malformed
    page can never drop replies.
    """
    variables: dict[str, Any] = {"threadId": thread_id}
    if after is not None:
        variables["commentsAfter"] = after
    resp = _graphql_with_rate_limit_retry(root, variables, query=_THREAD_COMMENTS_QUERY)
    if not isinstance(resp, dict):
        raise git_ops.GitError(
            f"graphql thread comments for {thread_id} returned a non-object response"
        )
    if resp.get("errors"):
        raise git_ops.GitError(f"graphql thread comments for {thread_id} failed: {resp['errors']}")
    try:
        node = resp["data"]["node"]
    except (KeyError, TypeError) as exc:
        raise git_ops.GitError(
            f"graphql response missing node for thread {thread_id}: {exc}"
        ) from exc
    if node is None:
        raise git_ops.GitError(f"graphql node(id:{thread_id}) returned null")
    try:
        comments = node["comments"]
    except (KeyError, TypeError) as exc:
        raise git_ops.GitError(
            f"graphql thread {thread_id} missing comments connection: {exc}"
        ) from exc
    if not (
        isinstance(comments, dict)
        and isinstance(comments.get("nodes"), list)
        and isinstance(comments.get("pageInfo"), dict)
    ):
        raise git_ops.GitError(f"graphql thread {thread_id} comments missing nodes/pageInfo")
    return comments


def _graphql_review_threads(root: Path, owner_repo: str, number: int) -> list[dict[str, Any]]:
    """Paginate every review thread (and reply) for a PR via GraphQL.

    The outer loop walks ``reviewThreads`` by cursor; a thread whose nested
    ``comments(first: 100)`` connection reports ``hasNextPage`` is completed by
    a per-thread ``node(id:)`` follow-up loop that walks strictly forward by
    ``endCursor``, so every reply past 100 is collected exactly once and page
    boundaries never reorder or drop comments. A GraphQL ``errors`` block, a
    missing ``data.repository.pullRequest.reviewThreads`` key, a failed call, or
    a nested ``comments`` connection missing ``pageInfo.hasNextPage`` raises
    :class:`GitError` — never a silent empty fallback.
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
        after = _next_cursor(page_info, context="reviewThreads")
        if after is None:
            break
    for thread in all_nodes:
        comments = thread.get("comments")
        if not isinstance(comments, dict):
            raise git_ops.GitError(
                f"graphql review thread {thread.get('id')} has no comments connection"
            )
        page_info = comments.get("pageInfo")
        if not isinstance(page_info, dict) or "hasNextPage" not in page_info:
            raise git_ops.GitError(
                f"graphql review thread {thread.get('id')} comments missing pageInfo.hasNextPage"
            )
        nested_after = _next_cursor(
            page_info, context=f"review thread {thread.get('id')} comments"
        )
        if nested_after is None:
            continue
        while True:
            page = _graphql_thread_comments(root, thread["id"], after=nested_after)
            comments.setdefault("nodes", []).extend(page["nodes"])
            page_info = page["pageInfo"]
            nested_after = _next_cursor(
                page_info, context=f"thread comments for {thread['id']}"
            )
            if nested_after is None:
                break
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
    """True when *title* is non-empty and within the 500-character bound."""
    if not title:
        return False
    return 0 < len(title) <= 500


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


def _payload_sha256(import_doc: dict[str, Any]) -> str:
    """sha256 over the canonical JSON of the complete normalized import.

    Spans every block of the persisted ``ImportDocument`` — ``schema_version``,
    ``repository``, the PR header (title/body/state/timestamps/head/base) and
    the evidence — except the self-referential ``fetch`` record that carries
    this digest itself, so any PR-intent or repository change flips it, not
    just evidence changes.
    """
    canonical = json.dumps(import_doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evidence_projection_hash(rec: dict[str, Any]) -> str:
    """sha256 over the sorted-JSON of the projection-relevant evidence fields.

    One digest per physical evidence record over exactly the fields that feed
    candidate projection and the curated review surface: body sha, author
    (login/type), commit anchors, path/line anchors, side, subject type,
    resolution state, dismissal, and review state. Values use ``.get()``
    defaults (``False`` for booleans, ``None`` for optional fields, ``""`` for
    strings) so an absent pre-canonicalization key equals the canonical
    default. ``kind``/``source_id``/``database_id``/``url``/timestamps are
    excluded: they are format-drift/metadata-sensitive and must not flip the
    signature.
    """
    author_raw = rec.get("author")
    author = author_raw if isinstance(author_raw, dict) else {}
    values: dict[str, Any] = {
        "body_sha256": str(rec.get("body_sha256") or ""),
        "author.login": str(author.get("login") or ""),
        "author.type": str(author.get("type") or ""),
        "commit_id": rec.get("commit_id"),
        "original_commit_id": rec.get("original_commit_id"),
        "path": rec.get("path"),
        "original_path": rec.get("original_path"),
        "line": rec.get("line"),
        "start_line": rec.get("start_line"),
        "original_line": rec.get("original_line"),
        "side": rec.get("side"),
        "start_side": rec.get("start_side"),
        "subject_type": rec.get("subject_type"),
        "resolved": bool(rec.get("resolved", False)),
        "outdated": bool(rec.get("outdated", False)),
        "dismissed": bool(rec.get("dismissed", False)),
        "state": rec.get("state"),
    }
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evidence_signature_from_doc(doc: schema.ImportDocument) -> frozenset[tuple[int, str]]:
    """Projection signature: one ``(database_id, projection_hash)`` per evidence record.

    Keyed on the physical comment id rather than ``source_id`` so the refresh
    stale check is immune to the canonical-record format change: pre-canonicalization
    files rekinded thread-only comments and persisted a comment that existed in
    both feeds twice, while the canonical format emits exactly one
    ``inline_comment`` per database id. The per-record hash spans the full
    projection-relevant provenance (body, author, commit/anchor fields, sides,
    subject type, resolution state, dismissal, review state) and excludes
    ``kind``/``source_id``/``database_id``/``url``/timestamps, so a duplicate
    pre-canon record collapses to one identical set element and a pure
    format/metadata change keeps prior curated cases — a genuine content
    change (a comment added/removed/edited, re-anchored, or re-resolved) still
    flips the digest. Deletion is carried by the set: a removed record's
    ``database_id`` simply disappears (no fallback hash).
    """
    return frozenset(
        (e.database_id, _evidence_projection_hash(e.model_dump(mode="json")))
        for e in doc.evidence
    )


def _evidence_signature_from_raw(raw: dict[str, Any]) -> frozenset[tuple[int, str]]:
    """The projection signature computed over a raw import document's dict.

    Shares the per-record hash helper with :func:`_evidence_signature_from_doc`
    so the persisted-file path and the typed-doc path always agree; a record
    appearing twice for one ``database_id`` (the pre-canonicalization duplicate)
    collapses to one identical set element, and a deleted record simply
    disappears from the set.
    """
    return frozenset(
        (int(e["database_id"]), _evidence_projection_hash(e))
        for e in raw.get("evidence", [])
    )


def _task_input_signature_from_doc(doc: schema.ImportDocument) -> str:
    """Deterministic sha256 over the header fields that feed the compiled context.

    Covers ``title``/``body``/``base``/``head`` (sha + ref) — the task-input
    contract a reviewer is shown — computed at refresh time without a full
    compile. A body/title/base/head change flips it; metadata-only changes
    (updated_at, html_url, merged state) do not.
    """
    sig = _task_input_signature_from_raw(doc.model_dump(mode="json"))
    assert sig is not None  # a typed doc always carries body + head.ref
    return sig


def _task_input_signature_from_raw(raw: dict[str, Any]) -> str | None:
    """The task-input signature computed over a raw import document's dict.

    A predate import file persisted ``head: {sha}`` without ``ref`` and no
    ``body``, so the task-input contract cannot be reconstructed from what it
    stored. Returns ``None`` for such files so the task-input arm of ``changed``
    stays inert until the file is re-persisted with the full header; a fresh
    file always carries both keys (``body`` may be ``""``, ``head.ref`` may be
    ``None``) and yields a comparable signature.
    """
    pr = raw.get("pull_request") or {}
    head = pr.get("head") or {}
    if "body" not in pr or "ref" not in head:
        return None
    base = pr.get("base") or {}
    payload = {
        "title": str(pr.get("title") or ""),
        "body": str(pr.get("body") or ""),
        "base_sha": base.get("sha"),
        "base_ref": base.get("ref"),
        "head_sha": head.get("sha"),
        "head_ref": head.get("ref"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _case_materialize(
    doc: schema.ImportDocument,
    number: int,
    requested_heads: list[str],
    import_file: str,
    import_sha256: str,
    *,
    root: Path | None = None,
    repo_slug: str = "",
    origin_url: str | None = None,
    prior_curations: dict[str, dict[str, Any]] | None = None,
    changed: bool = False,
) -> tuple[list[tuple[str, str, dict[str, Any]]], list[tuple[str, bytes]]]:
    """One materialized case document per requested head.

    When *root*/origin are provided the case ``snapshot`` is frozen via
    :func:`daydream.benchmark.snapshot.freeze_one` (a ``ready|unreplayable``
    dict) and any produced bundle is returned in the second element for the
    caller to stage atomically. When *changed* is True and a prior curated
    case exists, its curation is carried over and flipped to ``stale`` with
    attestation cleared — findings/exclusions are never overwritten by refresh.
    """
    pull_request = doc.pull_request
    base_sha = pull_request.base.sha
    out: list[tuple[str, str, dict[str, Any]]] = []
    bundle_drops: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for head_token in requested_heads:
        head_sha = head_token if head_token != "final" else pull_request.head.sha
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
            if changed and prior.get("state") in ("ready", "stale"):
                curation["state"] = "stale"
                curation["snapshot_attested"] = False
        if root is not None and origin_url is not None and base_sha and head_sha:
            policy = "final_pr_head" if head_token == "final" else "explicit_head"
            snapshot_doc, bundle_bytes = snapshot.freeze_one(
                root,
                repo_slug,
                number,
                base_tip=base_sha,
                head_sha=head_sha,
                policy=policy,
                requested_head=head_token,
                origin_url=origin_url,
            )
            if snapshot_doc.get("status") == "ready" and bundle_bytes is not None:
                bundle_drops.append((snapshot_doc["bundle_file"], bundle_bytes))
        else:
            snapshot_doc = {
                "status": "imported",
                "policy": "final_pr_head" if head_token == "final" else "explicit_head",
                "requested_head": head_token,
                "original_base_sha": base_sha,
                "original_head_sha": head_sha,
                "error": None,
            }
        case_doc: dict[str, Any] = {
            "schema_version": 2,
            "case_id": case_id,
            "pull_request": pull_request.model_dump(mode="json"),
            "snapshot": snapshot_doc,
            "source": {"import_file": import_file, "import_sha256": import_sha256},
            "curation": curation,
            "candidates": [c.model_dump(mode="json") for c in candidates],
        }
        out.append((case_id, f"cases/{case_id}.yaml", case_doc))
    return out, bundle_drops


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
) -> tuple[
    frozenset[tuple[int, str]] | None,
    str | None,
    dict[str, dict[str, Any]],
    str,
]:
    """The prior evidence signature, task-input signature, curations, and import path.

    A missing prior state (no ``fetched`` ledger entry, or no persisted
    import/case to read) yields ``None`` for both signatures and empty
    curations — the normal first-run path. A *present-but-corrupt* prior
    import or curation file is fatal: :class:`~daydream.benchmark.storage.WorkspaceCorrupt`
    from the strict loaders propagates so a refresh fails before any network
    fetch or mutation, never silently healing corrupt prior state to
    ``None``/``draft``.
    """
    import_file = f"imports/pr-{number:06d}.json"
    existing = _manifest_entry(raw, number)
    prior_sig: frozenset[tuple[int, str]] | None = None
    prior_task_sig: str | None = None
    prior_curations: dict[str, dict[str, Any]] = {}
    if existing is not None and existing.get("import_state") == "fetched":
        prior_import_file = existing.get("import_file")
        if not prior_import_file:
            # A fetched ledger entry must name its import document; one that
            # lacks or nulls import_file is corrupt prior state — without the
            # guard it escapes as a bare KeyError/TypeError instead of the
            # documented fail-closed WorkspaceCorrupt of the strict loaders.
            raise storage.WorkspaceCorrupt(
                f"{root}: fetched ledger entry for PR {number} is missing import_file"
            )
        # Ledger-derived authoring paths go through the containment gate (same
        # as every other workspace authoring read), so an absolute or escaping
        # import_file/case_id can never read outside the workspace root.
        prior_raw = storage.load_json_strict(
            storage.resolve_authoring_path(root, prior_import_file)
        )
        prior_sig = _evidence_signature_from_raw(prior_raw)
        prior_task_sig = _task_input_signature_from_raw(prior_raw)
        for case_id in existing.get("case_ids", []):
            cur = storage.load_yaml_strict(
                storage.resolve_authoring_path(root, f"cases/{case_id}.yaml")
            ).get("curation")
            if isinstance(cur, dict):
                prior_curations[case_id] = cur
    return prior_sig, prior_task_sig, prior_curations, import_file


def _import_one_pr(
    root: Path,
    raw: dict[str, Any],
    repo: str,
    number: int,
    requested_heads: list[str],
    *,
    refresh: bool,
    origin_url: str | None = None,
) -> int:
    """Fetch + materialize one PR, or stage its failure: 0 on success, 1 on failure."""
    prior_sig, prior_task_sig, prior_curations, import_file = _prior_import_state(root, raw, number)
    try:
        doc = fetch_and_normalize(root, repo, number)
        # stale only on an evidence change OR a task-input-contract change
        # (the title/body/base/head a reviewer was shown); a metadata-only change
        # updates checksums without staling gold.
        changed = (
            refresh
            and prior_sig is not None
            and (
                prior_sig != _evidence_signature_from_doc(doc)
                or (
                    prior_task_sig is not None
                    and prior_task_sig != _task_input_signature_from_doc(doc)
                )
            )
        )
        import_bytes = json.dumps(doc.model_dump(mode="json"), indent=2).encode("utf-8")
        import_sha256 = hashlib.sha256(import_bytes).hexdigest()
        cases, bundle_rels = _case_materialize(
            doc, number, requested_heads, import_file, import_sha256,
            root=root, repo_slug=repo, origin_url=origin_url,
            prior_curations=prior_curations, changed=changed,
        )
        with storage.Transaction(root, op_id=f"import-{number}", kind="import") as tx:
            tx.stage(import_file, import_bytes)
            for rel, content in bundle_rels:
                tx.stage(rel, content)
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


_UNSET_ORIGIN = object()


def run_import_prs(
    root: Path,
    pr_numbers: list[int],
    heads: list[str] | None = None,
    pr_heads: dict[int, list[str]] | None = None,
    refresh: bool = False,
    origin_url: str | None | object = _UNSET_ORIGIN,
) -> int:
    """Import each PR's evidence into one atomic import ledger/case transaction.

    Runs startup recovery then preflight (identity idempotent), then for each
    PR expects its import file, one case per requested head, and the ledger
    through one :class:`Transaction` (``benchmark.yaml`` last). *heads* is a
    flat back-compat list applied to every PR; *pr_heads* (from
    ``parse_import_targets``) maps each PR to its own requested heads so a
    ``PR=<40-hex>`` binding is honored for the PR it names only — when it is
    provided each PR resolves ``"final"`` plus its own explicit heads. When
    present, *origin_url* drives the snapshot freeze mirror fetch; when it is
    omitted entirely the origin is derived from the repository
    (``https://github.com/<repo>.git``). Passing ``origin_url=None``
    explicitly leaves the import hermetic — no snapshot freeze and no network
    git fetch. A failed fetch stages no import/case file — only a ledger flip
    to ``fetch_failed`` with an exact error. The overall exit is non-zero
    when any PR failed.
    """
    root = Path(root)
    flat_heads: list[str] = []
    seen_heads: set[str] = set()
    for head in ["final", *(heads or [])]:
        if head not in seen_heads:
            seen_heads.add(head)
            flat_heads.append(head)
    requested_by_pr: dict[int, list[str]] = {}
    for number in pr_numbers:
        if pr_heads is not None and pr_heads.get(number):
            requested_by_pr[number] = pr_heads[number]
        else:
            requested_by_pr[number] = list(flat_heads)
    exit_code = 0
    with storage.WorkspaceLock(root):
        storage.recover_startup(root)
        preflight(root, len(pr_numbers))
        raw = storage.load_yaml_strict(root / "benchmark.yaml")
        repo = raw.get("source", {}).get("repository") or ""
        if origin_url is _UNSET_ORIGIN:
            origin_url = f"https://github.com/{repo}.git" if repo else None
        effective_origin: str | None = (
            origin_url if isinstance(origin_url, str) or origin_url is None else None
        )
        for number in pr_numbers:
            if _import_one_pr(
                root, raw, repo, number, requested_by_pr[number], refresh=refresh, origin_url=effective_origin
            ):
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
            "id": source.get("repository_id") or "",
            "name_with_owner": source.get("repository") or owner_repo,
            "visibility": "public" if visibility == "public" else "private",
        }
    except Exception:
        return {"id": "", "name_with_owner": owner_repo, "visibility": "private"}


def fetch_and_normalize(
    root: Path,
    owner_repo: str,
    number: int,
) -> schema.ImportDocument:
    """Fetch one PR's full evidence set through REST and normalize it.

    Pulls the PR header, then every submitted review and conversation comment;
    each top-level inline comment is reconciled with its GraphQL thread state
    into one canonical ``inline_comment`` record (REST anchors plus joined
    thread id/resolved/outdated/dismissal), and review dismissal is joined by
    review id. Evidence is emitted in deterministic ``(database_id, created_at)``
    order, independent of REST/GraphQL page boundaries. The normalized
    ``pull_request`` block
    carries the complete header: number, url/html_url, title, body, state,
    merge/close timestamps, created/updated timestamps, author, exact
    base/head (sha + ref), and the persisted ``title_sha256``/``body_sha256``
    digests; the fetch ``payload_sha256`` spans the whole normalized import.
    Every retrieved record is retained as an :class:`EvidenceRecord` with a
    stable source ID and body hash; ``is_bot`` is derived from the author type
    and never filters. Failure of any call raises :class:`GitError` — never a
    silent default.
    """
    header = _fetch_with_retry(root, owner_repo, number)

    review_records = _rest(root, f"repos/{owner_repo}/pulls/{number}/reviews")
    inline_records = [_evidence_from_inline(raw) for raw in _rest(root, f"repos/{owner_repo}/pulls/{number}/comments")]
    threads = _graphql_review_threads(root, owner_repo, number)

    evidence: list[dict[str, Any]] = [_evidence_from_review(raw) for raw in review_records]
    evidence.extend(_join_dismissal(_reconcile_inline_evidence(inline_records, threads), review_records))
    for raw in _rest(root, f"repos/{owner_repo}/issues/{number}/comments"):
        evidence.append(_evidence_from_issue(raw))

    records = [schema.EvidenceRecord.model_validate(e) for e in evidence]
    # Canonical order: sort by (database_id, created_at) so persisted order and
    # payload_sha256 are independent of REST/GraphQL page boundaries.
    records.sort(key=lambda r: (r.database_id, r.created_at))
    record_dicts = [r.model_dump(mode="json") for r in records]
    base = header.get("base") or {}
    head = header.get("head") or {}
    title = header.get("title") or ""
    body = header.get("body") or ""          # null/empty -> "", Unicode/newlines preserved byte-for-byte
    pull_request = {
        "number": header["number"],          # KeyError propagates if absent — fail closed, never 0
        "url": header.get("url") or "",
        "html_url": header.get("html_url") or "",
        "title": title,
        "body": body,
        "state": header.get("state") or "",
        "title_sha256": hashlib.sha256(title.encode("utf-8")).hexdigest(),
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "base": {"sha": base.get("sha"), "ref": base.get("ref")},
        "head": {"sha": head.get("sha"), "ref": head.get("ref")},
        "created_at": header.get("created_at"),
        "updated_at": header.get("updated_at"),
        "merged_at": header.get("merged_at"),
        "closed_at": header.get("closed_at"),
        "author": _as_author(header),
    }
    import_doc = {
        "schema_version": 1,
        "repository": _repository_block(root, owner_repo),
        "pull_request": pull_request,
        "evidence": record_dicts,
    }
    return schema.ImportDocument.model_validate(
        {
            **import_doc,
            "fetch": {
                "fetched_at": _now_rfc3339(),
                "etag": None,
                "payload_sha256": _payload_sha256(import_doc),
            },
        }
    )
