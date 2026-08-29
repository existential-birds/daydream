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
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import ValidationError

from daydream import git_ops
from daydream.benchmark import curation as cu
from daydream.benchmark import schema, snapshot, storage
from daydream.benchmark.schema import EXTRACTION_VERSION


def _run_gh_preflight_status(root: Path) -> subprocess.CompletedProcess[str]:
    """Run ``gh auth status --hostname github.com`` (exit code is the contract)."""
    return git_ops._run_gh(root, ["auth", "status", "--hostname", "github.com"])


def _run_gh_api_user(root: Path) -> dict[str, Any]:
    """Return the authenticated GitHub user record from ``gh api user``."""
    proc = git_ops._run_gh(root, ["api", "user"])
    if proc.returncode != 0:
        raise git_ops.GitError(f"gh api user failed: {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    if not isinstance(data, dict):
        raise git_ops.GitError("gh api user returned a non-object payload")
    return data


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


def _as_author(raw: dict[str, Any]) -> dict[str, Any]:
    author = raw.get("user") or {}
    return {"login": author.get("login", ""), "type": author.get("type", "User")}


def _record_common(author: dict[str, Any], body: str) -> dict[str, Any]:
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
        "original_start_line": raw.get("original_start_line"),
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
          id isResolved isOutdated subjectType path line originalLine originalStartLine
          side: diffSide startSide: startDiffSide
          comments(first: 100) {
            pageInfo { hasNextPage endCursor }
            nodes { id databaseId body author { login type: __typename } createdAt updatedAt url replyTo { id } }
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
        nodes { id databaseId body author { login type: __typename } createdAt updatedAt url replyTo { id } }
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
        "original_start_line": thread.get("originalStartLine"),
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
            return cast(dict[str, Any], resp)  # gh_api returns raw JSON; the GraphQL body is a dict
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
    return str(after)


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
    """Project a RIGHT-side inline anchor exclusively from its authoring anchor.

    The strict versioned ``authoring_anchor`` (derived at case materialization
    from the authenticated mirror) is the single source of the projected
    ``Location``; the re-anchored observed fields (``path``/``start_line``/
    ``line``/``original_path``) never feed it. Returns ``(location, None)`` on
    a usable derived anchor, or ``(None, reason)`` from the fixed closed set
    — ``side`` (LEFT/mixed-side), ``history-unavailable`` (no anchor ever
    derived: import-only snapshot or a pre-anchor import), ``path-unavailable``
    / ``range-unavailable`` (derivation failed closed on exactly that). A
    locationless file-level comment returns ``(None, reason)`` too: its
    location is always None, but exact acceptance is gated by the same
    commit/anchor exactness as its line-level siblings — a missing anchor
    fails closed to ``history-unavailable``, a fail-closed derivation to its
    own status, and even a derived anchor cannot satisfy the exact-acceptance
    "usable authoring location" requirement, so it is ``range-unavailable``.
    """
    if evidence.subject_type == "file":
        anchor = evidence.authoring_anchor
        if anchor is None:
            # No strict anchor: never trust GitHub's re-anchored fields. As
            # with line comments, a missing anchor fails closed rather than
            # projecting exact off an unverifiable position.
            return None, "history-unavailable"
        if anchor.status != "derived":
            # 1:1 mapping of the fail-closed derivation status to the fixed
            # reason, matching the line-level branch.
            return None, anchor.status
        # Derived but locationless: no authoring line range exists for a
        # file-level comment, and inline exact acceptance requires a usable
        # authoring location — stay edit-required.
        return None, "range-unavailable"
    if evidence.side == "LEFT" or evidence.start_side == "LEFT":
        return None, "side"
    anchor = evidence.authoring_anchor
    if anchor is None:
        # No strict anchor: never trust GitHub's re-anchored fields. A missing
        # anchor means the authoring history was never derived (no mirror) or
        # does not exist in this import — fail closed to edit-required.
        return None, "history-unavailable"
    if anchor.status != "derived":
        # 1:1 mapping of the fail-closed derivation status to the fixed reason.
        return None, anchor.status
    if (
        anchor.start_line is None
        or anchor.end_line is None
        or anchor.start_line < 1
        or anchor.end_line < anchor.start_line
    ):
        return None, "range-unavailable"
    if anchor.path is None:
        return None, "path-unavailable"
    return (
        schema.Location(path=anchor.path, start_line=anchor.start_line, end_line=anchor.end_line),
        None,
    )


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
        elif (
            evidence.authoring_anchor is not None
            and evidence.authoring_anchor.status == "derived"
            and evidence.authoring_anchor.commit_id != head_sha
        ):
            # A derived anchor on any other commit means GitHub re-anchored
            # the comment (its re-anchored ``commit_id`` may even equal the
            # head). Exact acceptance is judged solely from the anchor.
            exact = False
            reason = "re-anchored"
    elif evidence.commit_id != head_sha:
        # review bodies are file-agnostic: no location, no side constraint,
        # and their single submission commit_id (reviews expose no inline
        # re-anchoring fields) still gates exact acceptance.
        exact = False
        reason = "commit"

    if not title_ok:
        exact = False
        reason = "title"
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
    Projection never raises — an underivable title, a LEFT-side anchor, a
    missing or fail-closed authoring anchor, a re-anchored inline record,
    an off-head review submission, an outdated, or a dismissed record all
    set ``exact_acceptable`` low with a ``not_exact_reason`` from the fixed
    closed set (``re-anchored``, ``history-unavailable``, ``path-unavailable``,
    ``range-unavailable``, ``side``, ``title``, ``commit``, ``outdated``,
    ``dismissed``). Inline exact acceptance derives solely from the authoring
    anchor's commit matching *head_sha*; review bodies key off their
    submission ``commit_id``. A locationless file-level comment is never
    exactly acceptable: it gates on the same anchor exactness, projecting its
    location as None and carrying the anchor's closed status
    (``history-unavailable`` when no anchor ever derived, ``range-unavailable``
    for a derived anchor that still offers no usable authoring location).
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
    (login/type), commit anchors, re-anchored path/line anchors, the strict
    authoring anchor (version/status/commit/path/range), sides, subject type,
    reply status (replies are evidence, never candidates), resolution
    state, dismissal, and review state. Values use ``.get()``
    defaults (``False`` for booleans, ``None`` for optional fields, ``""`` for
    strings) so an absent pre-canonicalization key equals the canonical
    default. ``kind``/``source_id``/``database_id``/``url``/timestamps are
    excluded: they are format-drift/metadata-sensitive and must not flip the
    signature. A pre-canonicalization record with no anchor key hashes like
    the canonical ``authoring_anchor``-less default, so a pure format/metadata
    change stays stable while a genuine anchor change flips the digest.
    """
    author_raw = rec.get("author")
    author = author_raw if isinstance(author_raw, dict) else {}
    anchor_raw = rec.get("authoring_anchor")
    anchor: dict[str, Any] | None
    if isinstance(anchor_raw, dict):
        anchor = {
            "version": anchor_raw.get("version"),
            "status": anchor_raw.get("status"),
            "commit_id": anchor_raw.get("commit_id"),
            "path": anchor_raw.get("path"),
            "start_line": anchor_raw.get("start_line"),
            "end_line": anchor_raw.get("end_line"),
        }
    else:
        anchor = None
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
        "original_start_line": rec.get("original_start_line"),
        "authoring_anchor": anchor,
        "side": rec.get("side"),
        "start_side": rec.get("start_side"),
        "subject_type": rec.get("subject_type"),
        "reply_to_id": rec.get("reply_to_id"),
        "resolved": bool(rec.get("resolved", False)),
        "outdated": bool(rec.get("outdated", False)),
        "dismissed": bool(rec.get("dismissed", False)),
        "state": rec.get("state"),
    }
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evidence_signature_from_doc(
    doc: schema.ImportDocument,
    *,
    downgrade_start_line: set[int] | None = None,
) -> frozenset[tuple[int, str]]:
    """Projection signature: one ``(database_id, projection_hash)`` per evidence record.

    Keyed on the physical comment id rather than ``source_id`` so the refresh
    stale check is immune to the canonical-record format change: pre-canonicalization
    files rekinded thread-only comments and persisted a comment that existed in
    both feeds twice, while the canonical format emits exactly one
    ``inline_comment`` per database id. The per-record hash spans the full
    projection-relevant provenance (body, author, commit/anchor fields, sides,
    subject type, reply status, resolution state, dismissal, review state) and excludes
    ``kind``/``source_id``/``database_id``/``url``/timestamps, so a duplicate pre-canon record
    collapses only when its projections are identical; when the thread copy lacks the commit
    anchors the two copies project differently, and the refresh changed-check treats the fresh
    canonical projection as unchanged while it matches any prior projection for that database id —
    so a pure format/metadata change keeps prior curated cases, while a genuine content
    change (a comment added/removed/edited, re-anchored, or re-resolved) still
    flips the digest. Deletion is carried by the set: a removed record's
    ``database_id`` simply disappears (no fallback hash).

    *downgrade_start_line* names the database ids whose prior import predates
    the ``original_start_line`` field (the key is absent from the persisted raw
    dict, e.g. a legacy multi-line comment); for exactly those ids the fresh
    canonical value is a pure schema-upgrade artifact, so the record is hashed
    without the key — identical to the prior absent-key form — and the
    one-time upgrade cannot flip the signature and stale curated cases.
    """
    return frozenset(
        (e.database_id, _evidence_projection_hash(_signature_dict(e, downgrade_start_line=downgrade_start_line)))
        for e in doc.evidence
    )


def _signature_dict(
    e: schema.EvidenceRecord, *, downgrade_start_line: set[int] | None = None
) -> dict[str, Any]:
    """One record's projection-hash dict, schema-upgrade normalized.

    Trades the canonical ``model_dump`` shape for one where ``original_start_line``
    is dropped when the record's id is in *downgrade_start_line* (see
    :func:`_evidence_signature_from_doc`) so legacy and fresh hashes agree.
    """
    rd = e.model_dump(mode="json")
    if downgrade_start_line is not None and e.database_id in downgrade_start_line:
        rd.pop("original_start_line", None)
    return rd


def _evidence_signature_from_raw(raw: dict[str, Any]) -> frozenset[tuple[int, str]]:
    """The projection signature computed over a raw import document's dict.

    Shares the per-record hash helper with :func:`_evidence_signature_from_doc`
    so the persisted-file path and the typed-doc path always agree; a record
    appearing twice for one ``database_id`` (the pre-canonicalization duplicate)
    may linger as two format-only projections when the thread copy lacks
    the commit anchors, and a deleted record simply disappears from the set.
    """
    return frozenset(
        (int(e["database_id"]), _evidence_projection_hash(e))
        for e in raw.get("evidence", [])
    )


def _backfill_prior_anchors(doc: schema.ImportDocument, prior_raw: dict[str, Any]) -> None:
    """Restore the prior import's persisted authoring anchors onto a fresh doc.

    ``fetch_and_normalize`` rebuilds every record from live GitHub, so an
    authoring anchor exists only in the persisted import document -- a refresh
    would otherwise compare an anchor-less fresh signature against the prior
    record's anchored one and flip every previously-derived id, staling every
    curated case that references it (the one-time anchor-era flip). Each fresh
    root inline record copies the prior record's anchor for the same physical
    comment (``database_id``) when one was persisted; a record the prior import
    left anchor-less stays unset so the mirror derivation in
    :func:`_derive_authoring_anchors` backfills exactly those (Task 7). The
    physical comment id is unique per record on both sides; a pre-canonical
    duplicate may appear twice for one id, and the first anchor-bearing copy
    wins -- the REST copy is the only kind that ever carries one. A record
    whose authoring_anchor is present but invalid (or that lacks its
    database_id) is corrupt prior state and raises
    :class:`~daydream.benchmark.storage.WorkspaceCorrupt` -- the caller stages
    a ledger failure instead of letting ValidationError/KeyError abort the run.
    """
    prior_by_id: dict[int, schema.AuthoringAnchor] = {}
    for e in prior_raw.get("evidence", []):
        anchor_raw = e.get("authoring_anchor")
        if not isinstance(anchor_raw, dict):
            continue
        try:
            db_id = int(e["database_id"])
            if db_id not in prior_by_id:
                prior_by_id[db_id] = schema.AuthoringAnchor.model_validate(anchor_raw)
        except (KeyError, ValidationError) as exc:
            # A persisted evidence record whose authoring_anchor is present but
            # invalid (or that lacks its database_id) is corrupt prior state:
            # fail closed via the module's WorkspaceCorrupt convention (caught
            # by the caller's WorkspaceError arm, which stages a ledger failure)
            # instead of letting ValidationError/KeyError escape the run
            # unhandled and crash the whole import.
            raise storage.WorkspaceCorrupt(
                "prior import evidence record has an invalid authoring_anchor"
                " or a missing database_id"
            ) from exc
    for record in doc.evidence:
        if record.kind != "inline_comment" or record.reply_to_id is not None:
            continue
        prior = prior_by_id.get(record.database_id)
        if prior is not None and record.authoring_anchor is None:
            record.authoring_anchor = prior


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


def _referenced_evidence_id(sid: str) -> int:
    """The trailing database id of one canonical ``github:<kind>:<id>`` source_id.

    Fail-closed, mirroring the schema's ``_canonical_source_id`` validator: a
    non-canonical source_id (a hand-edited or externally-mutated curation) is
    corrupt prior state and is never silently dropped from the referenced-
    evidence set, or the per-case stale gate would fail open and let
    referenced evidence change without the case flipping stale.
    """
    if not schema._SOURCE_ID_RE.fullmatch(sid):
        raise storage.WorkspaceCorrupt(
            f"curation references non-canonical source_id {sid!r}"
        )
    return int(sid.rsplit(":", 1)[-1])


def _referenced_evidence_ids(curation: dict[str, Any]) -> set[int]:
    """The physical database_ids a curation references via its source_ids.

    Derives each id from ``findings[].provenance.source_ids`` and
    ``exclusions[].source_id`` by parsing the canonical
    ``github:<kind>:<id>`` form to the trailing int. Used by the per-case
    stale decision so a case stales only when evidence *it references*
    changed; an unreferenced record never flips it. A non-canonical
    reference raises :class:`~daydream.benchmark.storage.WorkspaceCorrupt`
    (see :func:`_referenced_evidence_id`) instead of being skipped.
    """
    ids: set[int] = set()
    for finding in curation.get("findings", []):
        if not isinstance(finding, dict):
            continue
        provenance = finding.get("provenance") or {}
        for sid in provenance.get("source_ids", []):
            ids.add(_referenced_evidence_id(str(sid)))
    for exclusion in curation.get("exclusions", []):
        if not isinstance(exclusion, dict):
            continue
        ids.add(_referenced_evidence_id(str(exclusion.get("source_id") or "")))
    return ids


def _referenced_projection_changed(
    prior: dict[str, Any],
    prior_case_candidates: dict[str, dict[str, Any]],
    fresh_candidates: list[schema.Candidate],
) -> bool:
    """True when a referenced candidate re-projected differently on refresh.

    A preserved historical finding must keep byte-matching its candidate
    projection (title/body/location) or the next validation raises. The
    referenced-evidence *changed_ids* arm only catches projection changes
    driven by raw evidence edits; a projection-basis change carries no raw
    field flip: a record the prior import left anchor-less gains a
    mirror-derived authoring anchor, and its ``Location`` switches from the
    pre-anchor projection to the authoring-time one -- so it never enters
    *changed_ids* and the stale gate would silently re-project under the
    preserved curation. Compare each referenced candidate's persisted
    ``Location`` with the freshly projected one (among the byte-match fields
    the anchor derivation can only move the ``Location``); any change stales
    the case like the raw-evidence arms. A referenced candidate that vanished
    is a change too (the *changed_ids* arm already stales it; kept here so the
    flag stays monotone).
    """
    referenced = _referenced_evidence_ids(prior)
    if not referenced:
        return False
    fresh_by_source: dict[str, dict[str, Any]] = {
        c.source_id: c.model_dump(mode="json") for c in fresh_candidates
    }
    for source_id, prior_candidate in prior_case_candidates.items():
        if _referenced_evidence_id(source_id) not in referenced:
            continue
        fresh_candidate = fresh_by_source.get(source_id)
        if fresh_candidate is None:
            return True
        if prior_candidate.get("location") != fresh_candidate.get("location"):
            return True
    return False


def _anchor_fail_closed(
    status: Literal["history-unavailable", "path-unavailable", "range-unavailable"],
) -> schema.AuthoringAnchor:
    """One fail-closed authoring anchor: the fixed status, and nothing else.

    Every non-``derived`` status keeps all four data fields unset (the schema's
    ``_derived_iff_populated`` invariant), so a closed anchor can never be
    mistaken for a real authoring-time location.
    """
    return schema.AuthoringAnchor(
        version=1, status=status, commit_id=None, path=None, start_line=None, end_line=None,
    )


def _derive_one_anchor(
    record: schema.EvidenceRecord,
    mirror_repo: Path,
    head_sha: str,
) -> schema.AuthoringAnchor:
    """Derive one root inline record's strict authoring anchor, fail-closed.

    An ``original_commit_id`` is traced through the pinned mirror via
    :func:`daydream.benchmark.snapshot.derive_authoring_path` (the observed
    path when it exists in the authoring tree, else the unique rename between
    the authoring commit and the mapped head); every failure — a missing
    authoring commit, an ambiguous rename trace, a missing or inverted
    authoring range, a mirror-derived path the schema's relative-path rule
    rejects, or a hard git failure — maps to a fixed closed status with all
    data fields unset. Exactly one path may fill the anchor: a mirror-derived
    one. The observed ``original_path`` is reclassified as observed data and
    is never stored as the authoring path.
    """
    original_commit_id = record.original_commit_id
    if original_commit_id is None:
        return _anchor_fail_closed("history-unavailable")
    start = record.original_start_line or record.original_line
    end = record.original_line
    if start is None or end is None:
        return _anchor_fail_closed("range-unavailable")
    if start > end:
        # GitHub does not guarantee the authoring range ordering; the schema's
        # ``_derived_iff_populated`` validator rejects an inverted span, so
        # fail this record closed to its existing fixed status instead of
        # letting a ValidationError escape and abort the whole import run.
        return _anchor_fail_closed("range-unavailable")
    path = record.path or record.original_path
    if path is None:
        return _anchor_fail_closed("path-unavailable")
    try:
        authoring_path = snapshot.derive_authoring_path(
            mirror_repo, original_commit_id, path, head_sha
        )
    except snapshot.AnchorDerivationError as exc:
        # The closed reason rides on the exception; anything else is history.
        if exc.reason == "path-unavailable":
            return _anchor_fail_closed("path-unavailable")
        return _anchor_fail_closed("history-unavailable")
    except git_ops.GitError:
        # A hard git failure (subprocess/OS-level, e.g. a rename-trace diff
        # timeout) fails the anchor closed rather than aborting the import.
        return _anchor_fail_closed("history-unavailable")
    try:
        return schema.AuthoringAnchor(
            version=1, status="derived", commit_id=original_commit_id,
            path=authoring_path, start_line=start, end_line=end,
        )
    except ValidationError:
        # The observed original_* fields already validated (per-field >= 1 and
        # the ordering guard above); the mirror-derived authoring_path is the
        # only remaining input the shared relative-path rule can reject (git
        # permits filenames it forbids, e.g. ``:``). Fail this record closed
        # rather than aborting the import run: an anchor failure never kills
        # the import.
        return _anchor_fail_closed("path-unavailable")


def _extract_prioritization_facts(
    doc: schema.ImportDocument,
    mirror_repo: Path,
    base_tip: str,
    head_sha: str,
    candidate_ids: set[str],
) -> schema.PrioritizationFacts:
    """Per-evidence snapshot-comparison facts against the pinned head.

    Consumes exactly each record's ``authoring_anchor`` (the #826 contract —
    never GitHub's re-anchored fields). The commit relation classifies the
    record's authoring commit against the pinned head; the anchor delta
    measures the base_tip..head-classified interaction from that same
    authoring commit to the head, so an at-head comment's anchored region is
    by definition unchanged by the PR's own diff. Both helpers fail closed
    to ``unavailable``; a defensive per-record catch maps an unexpected git
    failure to ``unavailable`` on both axes without failing the import.
    """
    candidates: dict[str, schema.PrioritizationCandidate] = {}
    non_candidates: dict[str, schema.PrioritizationCandidate] = {}
    for record in doc.evidence:
        commit = record.commit_id or record.original_commit_id or head_sha
        relation: str = "unavailable"
        delta: str = "unavailable"
        try:
            relation = snapshot.commit_relation(mirror_repo, base_tip, head_sha, commit)
            anchor = (
                record.authoring_anchor.model_dump(mode="json")
                if record.authoring_anchor
                else None
            )
            delta = snapshot.anchor_delta(mirror_repo, commit, head_sha, anchor)
        except git_ops.GitError:
            pass
        entry = schema.PrioritizationCandidate(
            commit_relation=relation,  # type: ignore[arg-type]
            anchor_delta=delta,  # type: ignore[arg-type]
        )
        (candidates if record.source_id in candidate_ids else non_candidates)[
            record.source_id
        ] = entry
    return schema.PrioritizationFacts(
        extraction_version=EXTRACTION_VERSION,
        head_sha=head_sha,
        candidates=candidates,
        non_candidates=non_candidates,
    )


def _derive_authoring_anchors(
    doc: schema.ImportDocument,
    mirror_repo: Path,
    head_sha: str,
    changed_ids: set[int] | None = None,
) -> None:
    """Derive (or fail closed) the authoring anchor of every root inline record.

    Called once per materialized head, immediately before candidate projection
    and only when the freeze mirror is available: the same mirror-derived
    anchor the projection consumes (Task 5) is what the caller persists onto
    the import document. A record that already carries an anchor (restored by
    :func:`_backfill_prior_anchors` from the prior import document) keeps it —
    refresh backfills only records that are missing one (Task 7), so the
    persisted anchor is stable across refreshes and a restored anchor never
    re-stales curated cases on its own (it re-produces the identical candidate
    ``Location``) — except a record whose ``database_id`` sits in *changed_ids*
    (its projection hash flipped against the prior import, a genuine
    content/anchor edit): that record is re-derived so a stale anchor cannot
    linger on changed evidence. A record that genuinely gains its first anchor
    here re-projects its candidate differently; the per-case stale gate
    surfaces that via :func:`_referenced_projection_changed` instead of
    silently re-projecting preserved curation. Replies are evidence (never
    candidates), so they carry no anchor.
    """
    for record in doc.evidence:
        if record.kind != "inline_comment" or record.reply_to_id is not None:
            continue
        if record.authoring_anchor is None or (
            changed_ids is not None and record.database_id in changed_ids
        ):
            record.authoring_anchor = _derive_one_anchor(record, mirror_repo, head_sha)


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
    prior_candidates: dict[str, dict[str, dict[str, Any]]] | None = None,
    prior_pinned: dict[str, str] | None = None,
    prior_policy: dict[str, str] | None = None,
    changed_ids: set[int] | None = None,
    task_input_changed: bool = False,
) -> tuple[list[tuple[str, str, dict[str, Any]]], list[tuple[str, bytes]]]:
    """One materialized case document per requested head.

    When *root*/origin are provided the case ``snapshot`` is frozen via
    :func:`daydream.benchmark.snapshot.freeze_one` (a ``ready|unreplayable``
    dict) and any produced bundle is returned in the second element for the
    caller to stage atomically. An existing ``final`` case resolves to the
    pinned head from its prior ``snapshot.original_head_sha`` (head-immutable,
    so a live head advance reproduces the identical ``case_id``); only a first
    import with no prior ``final_pr_head`` pin uses the live
    ``pull_request.head.sha``. When a prior curated case exists, its curation
    is carried over; it is flipped to ``stale`` with attestation cleared iff
    *task_input_changed* (PR-wide), a referenced candidate's ``Location``
    changed on re-projection (*prior_candidates* — the projection-basis arm:
    a mirror-derived authoring anchor re-projects a record the prior import
    left anchor-less, a change no raw field carries into *changed_ids*), or
    its own referenced evidence ids intersect *changed_ids* (a referenced
    record changed or disappeared). An unreferenced evidence change never
    stales it and an untouched PR keeps it ready — findings/exclusions are
    never overwritten by refresh. On
    refresh, evidence records the prior import left anchor-less are backfilled
    with mirror-derived authoring anchors (or a fail-closed status) against
    the same pinned head — records already carrying a restored anchor keep it
    unless a genuine edit flipped their projection signature. A freeze
    of an existing ``ready|stale`` case that comes back ``unreplayable`` (the
    pinned head became unreachable after a force-push/rebased branch) raises
    :class:`~daydream.git_ops.GitError` instead of writing an unreplayable
    snapshot over the curated case — the refresh then fails like the sibling
    fetch-failure path (rc != 0, last-good linkage kept).

    On a ``ready`` freeze, per-evidence snapshot-comparison facts (commit
    relation + anchor delta against the pinned head, via
    :mod:`daydream.benchmark.snapshot`) are computed once here — where the
    authenticated mirror is already populated — and persisted on the case
    document under the additive ``prioritization`` key (schema_version stays
    2). Imported-status and unreplayable snapshots persist no key. Facts are
    read-projection input only and live on the case doc alone, so every hash
    surface (import payload, evidence signatures, projection hashes,
    staleness signatures) is untouched.
    """
    pull_request = doc.pull_request
    base_sha = pull_request.base.sha
    out: list[tuple[str, str, dict[str, Any]]] = []
    bundle_drops: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for head_token in requested_heads:
        head_sha = head_token if head_token != "final" else None
        if head_token == "final":
            # head-immutable: an existing final_pr_head case resolves to its
            # pinned commit; only a first import (no pin) uses the live head.
            pinned = _pinned_head_sha(prior_policy, prior_pinned)
            if pinned is not None:
                head_sha = pinned
            if head_sha is None:
                head_sha = pull_request.head.sha
        if not head_sha or head_sha in seen:
            continue
        seen.add(head_sha)
        case_id = schema.case_id_for(number, head_sha)
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
            curation = dict(prior_curations[case_id])
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
            elif snapshot_doc.get("status") != "ready" and prior_curations and (
                prior_curations.get(case_id) or {}
            ).get("state") in ("ready", "stale"):
                # A pinned head that became unreachable (force-push/rebased
                # branch) makes the re-freeze of a previously curated case
                # unreplayable. Never overwrite the curated case's snapshot
                # with an unreplayable dict: that would orphan its staged
                # bundle and silently flip the case out of ready while
                # _import_one_pr still returns 0. Fail the refresh like the
                # sibling fetch-failure path (rc 1, last-good linkage kept)
                # so the curated state and its bundle stay intact and indexed.
                error = snapshot_doc.get("error") or {}
                raise git_ops.GitError(
                    f"PR {number} freeze of curated case {case_id} is unreplayable "
                    f"({error.get('reason')}): {error.get('detail')}"
                )
            # Strict authoring anchors: derived from the authenticated mirror
            # (the same one the freeze just populated) immediately before
            # projection. Derivation mutates the typed doc's evidence records
            # in place and always fails closed — an anchor can never be
            # guessed, and an anchor failure never kills the import. On
            # refresh, records whose prior anchor was restored by the caller
            # keep it (only missing anchors are backfilled), unless their id is
            # in *changed_ids* (a genuine edit re-derives its anchor).
            _derive_authoring_anchors(doc, snapshot.mirror(root), head_sha, changed_ids)
        else:
            # Imported status: no freeze, no mirror — anchors stay unset
            # (projection treats the missing anchor as not-exact, Task 5).
            snapshot_doc = {
                "status": "imported",
                "policy": "final_pr_head" if head_token == "final" else "explicit_head",
                "requested_head": head_token,
                # both base SHAs carry the PR base tip at import — the merge
                # base is not yet computed and diverges on imported -> ready.
                "original_base_sha": base_sha,
                "requested_base_sha": base_sha,
                "original_head_sha": head_sha,
                "error": None,
            }
        # Projection runs after the freeze branch so per-head candidates
        # consume the derived authoring anchors (or their absence, closed).
        candidates = project_candidates(doc, head_sha)
        facts: schema.PrioritizationFacts | None = None
        if root is not None and origin_url is not None and snapshot_doc["status"] == "ready":
            facts = _extract_prioritization_facts(
                doc,
                snapshot.mirror(root),
                base_sha,
                head_sha,
                {c.source_id for c in candidates},
            )
        if prior_curations and case_id in prior_curations:
            prior = prior_curations[case_id]
            # The stale gate runs after projection so its third arm can see
            # the derived projections. Three independent signals: the PR-wide
            # task-input arm (title/body/base/head — refresh only), the
            # referenced-evidence arm (database_ids whose projection hash
            # changed or disappeared — refresh AND plain re-import), and the
            # projection-basis arm: a record the prior import left anchor-less
            # gains a mirror-derived anchor here, which re-projects its
            # Location from the authoring-time fields — a change no raw field
            # carries, so it never enters changed_ids, while a preserved
            # historical finding byte-matching the newly projected candidate
            # now fails. Feed that derivation-induced re-projection into the
            # stale gate rather than silently carrying preserved curation over
            # an invalidated candidate basis.
            should_stale = task_input_changed or _referenced_projection_changed(
                prior, (prior_candidates or {}).get(case_id, {}), candidates
            ) or (
                changed_ids is not None
                and bool(_referenced_evidence_ids(prior) & changed_ids)
            )
            if should_stale and prior.get("state") in ("ready", "stale"):
                curation["state"] = "stale"
                curation["snapshot_attested"] = False
                cu._invalidate_task_spec_approval(curation)
        case_doc: dict[str, Any] = {
            "schema_version": 2,
            "case_id": case_id,
            "pull_request": pull_request.model_dump(mode="json"),
            "snapshot": snapshot_doc,
            "source": {"import_file": import_file, "import_sha256": import_sha256},
            "curation": curation,
            "candidates": [c.model_dump(mode="json") for c in candidates],
        }
        if facts is not None:
            case_doc["prioritization"] = facts.model_dump(mode="json")
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
            "latest_error": None,  # a successful import/refresh clears the prior failed attempt
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
    prior_state = _pending_pr_state(raw, number)
    if prior_state == "fetched":
        # Non-destructive failed refresh (issue #813): a fetched PR keeps its
        # last-good linkage (import_file/import_sha256/requested_heads/case_ids)
        # and records the failed attempt separately in latest_error — it never
        # flips to bare fetch_failed, which would orphan its curated cases.
        entry = _manifest_entry(raw, number) or {}
        _ledger_replace(
            raw,
            {
                "number": number,
                "import_state": "fetched",
                "import_file": entry.get("import_file"),
                "import_sha256": entry.get("import_sha256"),
                "error": None,
                "latest_error": {"code": code, "message": message},
                "requested_heads": entry.get("requested_heads", []),
                "case_ids": entry.get("case_ids", []),
            },
        )
        return
    schema.validate_pr_transition(prior_state, "fetch_failed")
    _ledger_replace(
        raw,
        {
            "number": number,
            "import_state": "fetch_failed",
            "import_file": None,
            "import_sha256": None,
            "error": {"code": code, "message": message},
            "latest_error": None,
            "requested_heads": [],
            "case_ids": [],
        },
    )


def _pending_pr_state(raw: dict[str, Any], number: int) -> str:
    for entry in raw.get("pull_requests", []):
        if entry.get("number") == number:
            return str(entry.get("import_state", "pending"))
    return "pending"


def _manifest_entry(raw: dict[str, Any], number: int) -> dict[str, Any] | None:
    """The ledger entry for *number*, or None when not yet imported."""
    for entry in raw.get("pull_requests", []):
        if isinstance(entry, dict) and entry.get("number") == number:
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
    """Atomically stage a failed fetch on a PR's ledger entry.

    A first-import failure flips the entry to ``fetch_failed`` with an exact
    error; a failed refresh on an already-``fetched`` PR preserves its last-good
    linkage and records the attempt in ``latest_error`` instead. Stages only
    ``benchmark.yaml`` through one :class:`Transaction`; a failed fetch
    materializes no import/case file (the whole before/after ledger state is
    atomic).
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
    dict[str, dict[str, dict[str, Any]]],
    str,
    dict[str, str],
    dict[str, str],
    list[str],
]:
    """Prior import signatures, curations, candidates, path, pins, policies, heads.

    Returns the prior evidence signature, task-input signature, per-case
    curations, per-case projected candidates (``prior_candidates``:
    case_id -> source_id -> candidate dict — the candidate basis the prior
    curation's findings/exclusions were validated against), the import path,
    each prior case's pinned ``snapshot.original_head_sha`` (``prior_pinned``),
    each prior case's ``snapshot.policy`` (``prior_policy``), and the prior
    ledger entry's ``requested_heads``. A missing prior state (no ``fetched``
    ledger entry, or no persisted import/case to read) yields ``None`` for
    both signatures, empty curations/candidates/pins/policies/heads, and the
    default import path — the normal first-run path. A *present-but-corrupt*
    prior import or curation file is fatal: :class:`~daydream.benchmark.storage.WorkspaceCorrupt`
    from the strict loaders propagates so a refresh fails before any network
    fetch or mutation, never silently healing corrupt prior state to
    ``None``/``draft``. A ``ready``/``stale`` prior case missing its pinned
    head is also corrupt prior state — never a silent live-head default.
    """
    import_file = f"imports/pr-{number:06d}.json"
    existing = _manifest_entry(raw, number)
    prior_sig: frozenset[tuple[int, str]] | None = None
    prior_task_sig: str | None = None
    prior_curations: dict[str, dict[str, Any]] = {}
    prior_candidates: dict[str, dict[str, dict[str, Any]]] = {}
    prior_pinned: dict[str, str] = {}
    prior_policy: dict[str, str] = {}
    prior_requested_heads: list[str] = list(existing.get("requested_heads", [])) if existing else []
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
            case_raw = storage.load_yaml_strict(
                storage.resolve_authoring_path(root, f"cases/{case_id}.yaml")
            )
            cur = case_raw.get("curation")
            if isinstance(cur, dict):
                prior_curations[case_id] = cur
            candidates = case_raw.get("candidates")
            if isinstance(candidates, list):
                prior_candidates[case_id] = {
                    c["source_id"]: c
                    for c in candidates
                    if isinstance(c, dict) and c.get("source_id")
                }
            snapshot = case_raw.get("snapshot") or {}
            original_head_sha = snapshot.get("original_head_sha")
            if (
                isinstance(cur, dict)
                and cur.get("state") in ("ready", "stale")
                and not original_head_sha
            ):
                # A curated case must know which commit it was curated against;
                # an absent pin makes head-immutable refresh impossible without
                # silently re-anchoring to the live head.
                raise storage.WorkspaceCorrupt(
                    f"{root}: ready/stale case {case_id} is missing snapshot.original_head_sha"
                )
            if original_head_sha:
                prior_pinned[case_id] = original_head_sha
            policy = snapshot.get("policy")
            if policy:
                prior_policy[case_id] = policy
    return (
        prior_sig,
        prior_task_sig,
        prior_curations,
        prior_candidates,
        import_file,
        prior_pinned,
        prior_policy,
        prior_requested_heads,
    )


def _pinned_head_sha(
    prior_policy: dict[str, str] | None,
    prior_pinned: dict[str, str] | None,
) -> str | None:
    """Return the pinned head sha for an existing ``final_pr_head`` case, if any.

    Head-immutable task input: an existing ``final`` case resolves to the pinned
    head from its prior ``snapshot.original_head_sha``, so a live head advance
    neither re-anchors the case nor flips its task-input signature. Only a first
    import with no ``final_pr_head`` pin uses the live head (caller falls back).
    Shared by the materialize path and ``_import_one_pr`` so the pinning logic
    and its ``final_pr_head`` magic string live in exactly one place.
    """
    if prior_policy and prior_pinned:
        for prior_case_id, prior_pol in prior_policy.items():
            if prior_pol == "final_pr_head" and prior_case_id in prior_pinned:
                return prior_pinned[prior_case_id]
    return None


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
    """Fetch + materialize one PR, or stage its failure: 0 on success, 1 on failure.

    On refresh/re-import the persisted authoring anchors are backfilled onto
    the freshly fetched doc before the staleness comparison, so an anchor-era
    refresh neither stales curated cases on the one-time anchor backfill nor
    re-derives anchors the prior import already settled. Evidence records the
    prior import left anchor-less gain derived anchors (or a fail-closed
    status) whenever the freeze mirror is available — never guessed, never
    silently left exact. A record whose derived anchor re-projects its
    candidate differently than the prior import's candidate flips its curated
    case stale via the projection-basis arm in :func:`_case_materialize`, so
    refresh never silently re-projects preserved curation onto a candidate
    basis the findings no longer byte-match.
    """
    prior_sig, prior_task_sig, prior_curations, prior_candidates, import_file, prior_pinned, \
        prior_policy, prior_requested_heads = _prior_import_state(root, raw, number)
    try:
        doc = fetch_and_normalize(root, repo, number)
        # Head-immutable task input: an existing final_pr_head case pins the
        # refreshed doc's head to its snapshot.original_head_sha, so a live
        # head advance neither re-anchors the case nor flips the task-input
        # signature of the pinned case; only a first import (no pin) keeps the
        # live pull_request.head.sha.
        pinned = _pinned_head_sha(prior_policy, prior_pinned)
        if pinned is not None:
            doc.pull_request.head.sha = pinned
        # Anchor backfill for refresh/re-import: fetch_and_normalize rebuilds
        # every record from live GitHub, so a persisted authoring anchor exists
        # only on the prior import document. Restore those anchors onto the
        # fresh doc BEFORE the projection-signature comparison — the anchor
        # fields are in the projection whitelist, and comparing an anchor-less
        # fresh signature against an anchored prior one would flip every
        # previously-derived id and stale every curated case that references
        # it. Records the prior import left anchor-less stay unset;
        # _case_materialize then derives exactly those (plus genuinely changed
        # ids) when the mirror is available.
        prior_raw: dict[str, Any] | None = None
        if prior_sig is not None:
            prior_raw = storage.load_json_strict(
                storage.resolve_authoring_path(root, import_file)
            )
            _backfill_prior_anchors(doc, prior_raw)
        # Two independent stale signals computed here: the per-case
        # referenced-evidence arm (database_ids whose projection hash changed
        # or disappeared — runs on refresh AND plain re-import) and the
        # PR-wide task-input arm (the title/body/base/head a reviewer was
        # shown — refresh only). A third, per-case projection-basis arm fires
        # in _case_materialize when a derived authoring anchor re-projects a
        # referenced candidate differently (see _referenced_projection_changed).
        # A metadata-only change updates checksums without staling.
        task_input_changed = (
            refresh
            and prior_task_sig is not None
            and prior_task_sig != _task_input_signature_from_doc(doc)
        )
        changed_ids: set[int] | None = None
        # The referenced-evidence arm runs for ANY fetched PR, refresh or plain
        # re-import: a curated case whose own referenced evidence changed or
        # disappeared must stale even on a non-refresh import (the refresh
        # semantics cannot be bypassed). Only the PR-wide task-input arm stays
        # gated on *refresh* (Assumption 3: a plain re-import never stales on
        # title/body/base/head). A first import (no prior_sig) computes nothing.
        if prior_sig is not None:
            # Per-id projection-hash SETS instead of a dict() collapse: two
            # records for one database_id (the pre-canonicalization duplicate —
            # a REST inline copy and a GraphQL thread copy under the same id)
            # carry different projection hashes, and which tuple dict() keeps
            # depends on frozenset iteration order, which hash randomization
            # makes nondeterministic across processes. Comparing per-id hash
            # sets is order-independent. A database id counts as changed only
            # when its fresh canonical projection is NOT covered by the prior
            # projections: a genuine content/anchor/resolution edit, an
            # addition, or a deletion. The pre-canonical thread copy lacks the
            # commit anchors only REST exposes, so it is a pure format artifact
            # — the fresh REST-derived projection still matches a prior
            # projection and the first post-format refresh must NOT stale
            # curated gold. When every id is unique on both sides this reduces
            # exactly to the old single-hash comparison.
            prior_by_id: dict[int, set[str]] = {}
            for db_id, proj_hash in prior_sig:
                prior_by_id.setdefault(db_id, set()).add(proj_hash)
            new_by_id: dict[int, set[str]] = {}
            # One-time schema-era format upgrade: prior files written before
            # the authoring-range field existed persist no ``original_start_line``
            # key, while the canonical dump now carries a real value (e.g. 4 on
            # a multi-line comment). Hash the fresh side as if the field were
            # absent for exactly those ids so the upgrade cannot flip the
            # signature and stale curated cases referencing the record (the
            # anchor backfill covers the anchor field; this covers the raw
            # authoring-range field the anchor backfill does not restore).
            assert prior_raw is not None  # loaded above whenever prior_sig is not None
            legacy_without_start_line: set[int] = {
                int(e["database_id"])
                for e in prior_raw.get("evidence", [])
                if "original_start_line" not in e
            }
            for db_id, proj_hash in _evidence_signature_from_doc(
                doc, downgrade_start_line=legacy_without_start_line
            ):
                new_by_id.setdefault(db_id, set()).add(proj_hash)
            changed_ids = {
                db_id
                for db_id in set(prior_by_id) | set(new_by_id)
                if db_id not in new_by_id
                or not (new_by_id[db_id] <= prior_by_id.get(db_id, set()))
            }
        # The digest the materializer stamps into every case's source block is
        # computed here, before materialization mutates the doc. The persisted
        # import document is re-serialized afterwards so the derived authoring
        # anchors land on disk, and the ledger + case source blocks are
        # re-stamped to one digest over those final bytes.
        import_bytes = json.dumps(doc.model_dump(mode="json"), indent=2).encode("utf-8")
        import_sha256 = hashlib.sha256(import_bytes).hexdigest()
        # Refresh/re-import never orphans a previously pinned case: materialize
        # the union of the prior ledger heads and the newly-requested heads so
        # _stamp_fetched's cases[] rewrite keeps every curated case indexed.
        materialize_heads = requested_heads
        if prior_requested_heads:
            materialize_heads = list(dict.fromkeys([*prior_requested_heads, *requested_heads]))
        cases, bundle_rels = _case_materialize(
            doc, number, materialize_heads, import_file, import_sha256,
            root=root, repo_slug=repo, origin_url=origin_url,
            prior_curations=prior_curations, prior_candidates=prior_candidates,
            prior_pinned=prior_pinned, prior_policy=prior_policy,
            changed_ids=changed_ids, task_input_changed=task_input_changed,
        )
        # Re-serialize the mutated doc (authoring anchors derived on the typed
        # evidence records during materialization), recompute the fetch payload
        # digest over the same blocks, and keep every digest in lockstep.
        final_doc = doc.model_dump(mode="json")
        doc.fetch.payload_sha256 = _payload_sha256(
            {k: final_doc[k] for k in ("schema_version", "repository", "pull_request", "evidence")}
        )
        import_bytes = json.dumps(doc.model_dump(mode="json"), indent=2).encode("utf-8")
        import_sha256 = hashlib.sha256(import_bytes).hexdigest()
        for _, _, case_doc in cases:
            case_doc["source"]["import_sha256"] = import_sha256
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
                materialize_heads,
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
