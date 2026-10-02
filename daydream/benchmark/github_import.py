"""Import normalized evidence and frozen cases from explicit private GitHub PRs.

Preflight re-verifies access and immutable repository identity before mutations.
REST and GraphQL evidence, each requested head's snapshot/bundle, and the ledger
land in one crash-consistent transaction. Git/GitHub calls use git_ops; exhausted
bounded rate-limit retries record a fetch failure rather than dropping evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import ValidationError

from daydream import git_ops
from daydream.benchmark import curation as cu, schema, snapshot, storage
from daydream.benchmark.schema import EXTRACTION_VERSION
from daydream.git_ops import process as git_process
from daydream.pr_review import FINDING_MARKER_RE


def _run_gh_api_user(root: Path) -> dict[str, Any]:
    """Return the authenticated GitHub user record from ``gh api user``."""
    proc = git_process._run_gh(root, ["api", "user"], auth=git_ops.INHERIT_GITHUB_AUTH)
    if proc.returncode != 0:
        raise git_ops.GitError(f"gh api user failed: {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    if not isinstance(data, dict):
        raise git_ops.GitError("gh api user returned a non-object payload")
    return data


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
    proc = git_process._run_gh(
        root,
        ["repo", "view", repo_slug, "--json", "id,nameWithOwner,url,visibility,defaultBranchRef"],
        auth=git_ops.INHERIT_GITHUB_AUTH,
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
    """Require exact repository identity and recognized visibility; keep opaque node IDs as strings."""
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
    """Persist resolved identity under the workspace lock in one recoverable transaction."""
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
    """Run fixed-order binary, authentication, identity, and access checks.

    Every import/refresh re-verifies exact repository identity before authoring,
    manifest, or bundle mutation. The first failure raises its stable code/message.
    """
    root = Path(root)
    if shutil.which("git") is None or shutil.which("gh") is None:
        raise PreflightError("missing_binary", "git and gh binaries must be reachable")

    status = git_process._run_gh(root, ["auth", "status", "--hostname", "github.com"], auth=git_ops.INHERIT_GITHUB_AUTH)
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
        git_ops.git_ls_remote(root, f"https://github.com/{repo_slug}.git")
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
        last_verified_at=schema.rfc3339_now(),
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
    """Stable PR order with a compatibility head union and per-PR heads, each including final."""

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
    """Merge CLI PRs before file entries, preserving first-seen order.

    Every PR includes final. A bare lowercase 40-hex head applies to every PR;
    PR=<40-hex> applies only to that requested PR. Invalid tokens or bindings to
    unrequested PRs raise ImportTargetError with the offending value.
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


def _parse_ndjson(text: str) -> list[Any]:
    """Parse nonblank gh NDJSON rows; malformed JSON raises GitError with the line number."""
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
    """Return the last result and its rate-limit classification after at most three calls.

    Other failures return immediately. Between rate-limited attempts, honor
    Retry-After up to 60 seconds; callers record exhausted retries as fetch failures.
    """
    last = None
    last_rate_limit: git_ops.RateLimitError | None = None
    for attempt in range(_RATE_LIMIT_ATTEMPTS):
        proc = call()
        if proc.returncode == 0:
            return proc, None
        error = git_process._gh_error_for(f"gh call failed: {proc.stderr.strip()}", proc.stderr)
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
    """Fetch and parse the singular PR header with bounded retries and explicit rate-limit failures."""
    endpoint = f"repos/{owner_repo}/pulls/{number}"
    proc, rate_limit = _call_with_rate_limit_retry(
        lambda: git_process._run_gh(root, ["api", endpoint, "--jq", "@json"], auth=git_ops.INHERIT_GITHUB_AUTH)
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
    """Fetch every REST page as NDJSON, retaining all rows or propagating the failure."""
    proc, rate_limit = _call_with_rate_limit_retry(
        lambda: git_process._run_gh(
            root, ["api", "--paginate", endpoint, "--jq", ".[] | @json"], auth=git_ops.INHERIT_GITHUB_AUTH,
        )
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


def _rest_evidence(raw: dict[str, Any], kind: str, node_prefix: str) -> dict[str, Any]:
    """Normalize the shared REST identity, timestamps, author, and body contract."""
    db_id = int(raw["id"])
    return {
        "source_id": f"github:{kind}:{db_id}",
        "kind": kind,
        "database_id": db_id,
        "node_id": raw.get("node_id") or f"{node_prefix}_{db_id}",
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
        "url": raw.get("html_url") or "",
        **_record_common(raw.get("user") or {}, raw.get("body") or ""),
    }


def _evidence_from_review(raw: dict[str, Any]) -> dict[str, Any]:
    submitted = raw.get("submitted_at")
    return {
        **_rest_evidence(raw, "review", "PRR"),
        "created_at": raw.get("created_at") or submitted,
        "updated_at": raw.get("updated_at") or submitted,
        "submitted_at": submitted,
        **{key: raw.get(key) for key in ("commit_id", "original_commit_id", "state")},
    }


def _evidence_from_inline(raw: dict[str, Any]) -> dict[str, Any]:
    subject_type = raw.get("subject_type")
    if subject_type is None:
        subject_type = "file" if raw.get("path") is None else "line"
    return {
        **_rest_evidence(raw, "inline_comment", "DIFF"),
        **{key: raw.get(key) for key in (
            "commit_id", "original_commit_id", "path", "original_path", "line", "start_line",
            "original_line", "original_start_line", "side", "start_side",
        )},
        "thread_id": None,
        "review_id": str(raw["pull_request_review_id"]) if raw.get("pull_request_review_id") is not None else None,
        "reply_to_id": str(raw["in_reply_to_id"]) if raw.get("in_reply_to_id") is not None else None,
        "subject_type": subject_type,
    }


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
    """Normalize a GraphQL-only comment as inline evidence; REST-only commit anchors stay absent."""
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
    """Overlay thread state onto REST comments by database ID, falling back to node ID.

    Keep REST location/commit anchors; use GraphQL thread ID, resolution, outdated
    state, and a missing reply link. Retain unmatched GraphQL comments as canonical
    inline evidence so overlapping feeds do not duplicate known comments.
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
    """Join DISMISSED review state onto its inline comments by review ID."""
    states = {str(int(raw["id"])): raw.get("state") for raw in review_records}
    for rec in canonical:
        review_id = rec.get("review_id")
        if rec.get("kind") == "inline_comment" and review_id and states.get(review_id) == "DISMISSED":
            rec["dismissed"] = True
    return canonical


def _graphql_with_rate_limit_retry(
    root: Path, variables: dict[str, Any], *, query: str = _REVIEW_THREADS_QUERY
) -> dict[str, Any]:
    """Apply bounded Retry-After retries to GraphQL; exhausted limits retain the rate_limit ledger classification."""
    last_rate_limit: git_ops.RateLimitError | None = None
    for attempt in range(_RATE_LIMIT_ATTEMPTS):
        try:
            resp = git_ops.gh_api(
                root,
                "graphql",
                method="POST",
                idempotent=True,
                input_data={"query": query, "variables": variables}, auth=git_ops.INHERIT_GITHUB_AUTH,
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
    """Return the next cursor; hasNextPage without endCursor raises instead of dropping a page."""
    if not page_info.get("hasNextPage"):
        return None
    after = page_info.get("endCursor")
    if after is None:
        raise git_ops.GitError(f"graphql {context} hasNextPage without an endCursor")
    return str(after)


def _graphql_thread_comments(
    root: Path, thread_id: str, *, after: str | None = None
) -> dict[str, Any]:
    """Fetch a thread comment page; API errors or missing node/comments/nodes/pageInfo raise GitError."""
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
    """Collect every thread and every nested reply page in source order.

    Follow outer and per-thread cursors independently. API errors and missing
    required connections/page metadata raise instead of silently omitting evidence.
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


def _projected_body(body: str) -> str:
    """Normalize candidate text after removing canonical Daydream markers."""
    return _normalize_body(FINDING_MARKER_RE.sub("", body))


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
    return 0 < len(title) <= 500


def _anchor_location(
    evidence: schema.EvidenceRecord,
) -> tuple[schema.Location | None, str | None]:
    """Project location solely from a strict authoring anchor.

    Reject LEFT/mixed-side line comments. Missing or failed anchors retain their
    closed reason. File-level comments remain locationless and edit-required even
    with a derived anchor; observed GitHub re-anchored fields never supply location.
    """
    if evidence.subject_type != "file" and (evidence.side == "LEFT" or evidence.start_side == "LEFT"):
        return None, "side"
    anchor = evidence.authoring_anchor
    if anchor is None:
        return None, "history-unavailable"
    if anchor.status != "derived":
        return None, anchor.status
    if (
        evidence.subject_type == "file"
        or anchor.start_line is None
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
    body = _projected_body(evidence.body)
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
    """Project nonempty root inline comments and COMMENTED/CHANGES_REQUESTED reviews.

    Keep replies, approvals, and conversation comments as evidence only. Exactness
    uses authoring commit/location for inline records and submission commit for
    reviews, with fixed reasons for title, side, commit, outdated, dismissed, or
    unavailable anchors. File-level comments remain locationless and never exact.
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
    """Hash the entire canonical import except its self-referential fetch record, including repository and PR intent."""
    canonical = json.dumps(import_doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evidence_projection_hash(rec: dict[str, Any]) -> str:
    """Hash projection-relevant content, provenance, anchors, and review/thread state.

    Default missing fields to their canonical values. Exclude record kind, source id,
    database id, URLs, and timestamps so a metadata/schema-format change does not stale
    curation. An absent authoring anchor hashes as ``None``.
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
    """Return immutable ``(database_id, projection_hash)`` pairs for typed evidence.

    Physical ids survive canonicalization of old REST/GraphQL duplicates. Retain every
    distinct projection for one id: refresh changes that id only if the old set cannot
    cover the fresh projections, or the id disappears. For ``downgrade_start_line`` ids,
    omit the field introduced after their persisted imports to avoid false staleness.
    """
    return frozenset(
        (e.database_id, _evidence_projection_hash(_signature_dict(e, downgrade_start_line=downgrade_start_line)))
        for e in doc.evidence
    )


def _signature_dict(
    e: schema.EvidenceRecord, *, downgrade_start_line: set[int] | None = None
) -> dict[str, Any]:
    """Dump evidence for hashing, omitting only explicitly identified legacy range fields."""
    rd = e.model_dump(mode="json")
    if downgrade_start_line is not None and e.database_id in downgrade_start_line:
        rd.pop("original_start_line", None)
    return rd


def _evidence_signature_from_raw(raw: dict[str, Any]) -> frozenset[tuple[int, str]]:
    """Hash raw evidence with the same defaults, retaining distinct legacy projections per id."""
    return frozenset(
        (int(e["database_id"]), _evidence_projection_hash(e))
        for e in raw.get("evidence", [])
    )


def _backfill_prior_anchors(doc: schema.ImportDocument, prior_raw: dict[str, Any]) -> None:
    """Restore persisted anchors onto fresh root inline comments before comparison.

    The first anchor-bearing prior copy per physical id wins. Leave previously missing
    anchors for mirror derivation; invalid persisted anchors or missing database ids
    raise ``WorkspaceCorrupt`` instead of discarding prior state.
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
            # Report corrupt prior state through the caller's ledger-failure path.
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
    """Hash title, body, and base/head SHA+ref; timestamps, URLs, and merge metadata do not affect task input."""
    sig = _task_input_signature_from_raw(doc.model_dump(mode="json"))
    assert sig is not None  # a typed doc always carries body + head.ref
    return sig


def _task_input_signature_from_raw(raw: dict[str, Any]) -> str | None:
    """Return a comparable task-input hash only when the historical header is complete.

    Legacy imports missing body or head.ref return None, leaving that stale gate
    inactive until a complete header is persisted. Present empty/null values count.
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
    """Parse a canonical source ID; corrupt curation references raise rather than bypass the stale gate."""
    if not schema._SOURCE_ID_RE.fullmatch(sid):
        raise storage.WorkspaceCorrupt(
            f"curation references non-canonical source_id {sid!r}"
        )
    return int(sid.rsplit(":", 1)[-1])


def _referenced_evidence_ids(curation: dict[str, Any]) -> set[int]:
    """Collect physical IDs from finding provenance and exclusions, rejecting noncanonical references."""
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
    """Detect disappeared or relocated referenced candidates after projection.

    A newly derived anchor can change location without raw-evidence changes.
    Preserved historical findings must still byte-match, so this independently
    stales affected curation.
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
    """Construct a closed anchor status with every data field unset."""
    return schema.AuthoringAnchor(
        version=1, status=status, commit_id=None, path=None, start_line=None, end_line=None,
    )


def _derive_one_anchor(
    record: schema.EvidenceRecord,
    mirror_repo: Path,
    head_sha: str,
) -> schema.AuthoringAnchor:
    """Derive one root inline authoring anchor through the pinned mirror.

    Resolve original_commit_id to an existing path or unique rename and valid range.
    Missing history, ambiguous paths, invalid ranges/schema paths, and Git failures
    produce closed statuses with no data. Never trust observed original_path.
    """
    original_commit_id = record.original_commit_id
    if original_commit_id is None:
        return _anchor_fail_closed("history-unavailable")
    start = record.original_start_line or record.original_line
    end = record.original_line
    if start is None or end is None:
        return _anchor_fail_closed("range-unavailable")
    if start > end:
        # GitHub can supply inverted ranges; close this anchor before model validation.
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
        # Range fields already validated. The derived Git path can still violate
        # schema rules (for example ':'); close only this anchor.
        return _anchor_fail_closed("path-unavailable")


def _extract_prioritization_facts(
    doc: schema.ImportDocument,
    mirror_repo: Path,
    head_sha: str,
    candidate_ids: set[str],
) -> schema.PrioritizationFacts:
    """Compare strict authoring anchors with the pinned head, in authoring coordinates.

    Missing/closed anchors yield unavailable/locationless without Git probes.
    Git failures mark facts unavailable without failing import. Cache whole-tree
    diffs per authoring-commit/head pair; modified-path probes remain per record.
    """
    candidates: dict[str, schema.PrioritizationCandidate] = {}
    non_candidates: dict[str, schema.PrioritizationCandidate] = {}
    # Whole-tree diffs are shared by every record at the same authoring commit.
    diff_cache: dict[tuple[str, str], snapshot.AnchorDiff] = {}
    for record in doc.evidence:
        anchor = (
            record.authoring_anchor.model_dump(mode="json")
            if record.authoring_anchor
            else None
        )
        relation: str = "unavailable"
        delta: str = "locationless"
        if anchor is not None and anchor.get("status") == "derived":
            # A derived anchor carries the authoring commit (schema-enforced);
            # both probes run against that commit, never a re-anchored field.
            delta = "unavailable"
            try:
                relation = snapshot.commit_relation(
                    mirror_repo, head_sha, anchor["commit_id"]
                )
                delta = snapshot.anchor_delta(
                    mirror_repo, anchor["commit_id"], head_sha, anchor,
                    diff_cache=diff_cache,
                )
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


def _reuse_prior_facts(
    prior: dict[str, Any] | None,
    head_sha: str,
    changed_ids: set[int] | None,
    candidate_ids: set[str],
    evidence_ids: set[str],
) -> schema.PrioritizationFacts | None:
    """Reuse valid facts only when extraction version, head, evidence, and candidate split match.

    Projection changes can move the candidate split without raw-evidence edits.
    Missing, corrupt, or changed inputs trigger fresh extraction; unchanged inputs
    avoid repeated mirror probes.
    """
    if prior is None:
        return None
    if prior.get("extraction_version") != EXTRACTION_VERSION:
        return None
    if prior.get("head_sha") != head_sha:
        return None
    if changed_ids:
        return None
    if not isinstance(prior.get("candidates"), dict) or not isinstance(
        prior.get("non_candidates"), dict
    ):
        return None
    # Projection changes can alter membership without changing raw evidence.
    if set(prior["candidates"]) != candidate_ids:
        return None
    if set(prior["non_candidates"]) != evidence_ids - candidate_ids:
        return None
    try:
        return schema.PrioritizationFacts.model_validate(prior)
    except ValidationError:
        return None


def _derive_authoring_anchors(
    doc: schema.ImportDocument,
    mirror_repo: Path,
    head_sha: str,
    changed_ids: set[int] | None = None,
) -> None:
    """Derive root inline anchors before projection when the freeze mirror is available.

    Keep persisted anchors except for genuinely changed evidence IDs. Newly added
    anchors can move candidate locations, which the referenced-projection stale
    gate detects separately. Replies remain unanchored evidence.
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
    *,
    root: Path | None = None,
    repo_slug: str = "",
    origin_url: str | None = None,
    prior: _PriorImport | None = None,
    changed_ids: set[int] | None = None,
    task_input_changed: bool = False,
) -> tuple[list[tuple[str, str, dict[str, Any]]], list[tuple[str, bytes]]]:
    """Freeze and project each distinct requested head, preserving prior curation.

    Existing final-head cases retain their pinned commit. With an origin, freeze the
    snapshot and derive missing/changed authoring anchors before candidate projection;
    otherwise emit an imported snapshot. A curated ready/stale case that cannot be
    re-frozen fails rather than losing its last good snapshot or bundle.

    Stale only curated cases affected by task input, referenced evidence, or referenced
    candidate location changes. Findings and exclusions survive refresh. Prioritization
    facts are reused only for an identical head, extraction version, evidence, and
    candidate split; they never affect identity or staleness. Return cases and bundle
    bytes for one transaction; the caller stamps the final import digest after anchors
    have been serialized.
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
            pinned = prior.pinned_head if prior is not None else None
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
        prior_case = prior.cases.get(case_id) if prior is not None else None
        previous = prior_case.curation if prior_case is not None else None
        if previous is not None:
            curation = dict(previous)
        if root is not None and origin_url is not None and base_sha and head_sha:
            policy = "final_pr_head" if head_token == "final" else "explicit_head"
            if policy == "explicit_head" and pull_request.changed_files is None:
                raise git_ops.GitError(
                    f"PR {number} explicit-head freeze requires a complete changed-files inventory"
                )
            snapshot_doc, bundle_bytes = snapshot.freeze_one(
                root,
                repo_slug,
                number,
                base_tip=base_sha,
                head_sha=head_sha,
                policy=policy,
                requested_head=head_token,
                pr_changed_files=frozenset(pull_request.changed_files or ()),
                origin_url=origin_url,
            )
            if snapshot_doc.get("status") == "ready" and bundle_bytes is not None:
                bundle_drops.append((snapshot_doc["bundle_file"], bundle_bytes))
            elif snapshot_doc.get("status") != "ready" and (previous or {}).get("state") in ("ready", "stale"):
                # An unreachable pinned head fails refresh. Preserve the curated
                # snapshot and indexed bundle rather than installing unreplayable state.
                error = snapshot_doc.get("error") or {}
                raise git_ops.GitError(
                    f"PR {number} freeze of curated case {case_id} is unreplayable "
                    f"({error.get('reason')}): {error.get('detail')}"
                )
            elif snapshot_doc.get("status") == "unreplayable" and curation.get("state") != "excluded":
                curation["state"] = "unreplayable"
                curation["snapshot_attested"] = False
                curation["clean_attested"] = False
                curation["gold_status"] = "findings" if curation.get("findings") else None
                cu._invalidate_task_spec_approval(curation)
            # Restore exact anchors from the populated mirror before projection.
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
            # Reuse only facts matching the freshly projected candidate split.
            facts = _reuse_prior_facts(
                prior_case.facts if prior_case is not None else None,
                head_sha,
                changed_ids,
                {c.source_id for c in candidates},
                {r.source_id for r in doc.evidence},
            )
            if facts is None:
                facts = _extract_prioritization_facts(
                    doc,
                    snapshot.mirror(root),
                    head_sha,
                    {c.source_id for c in candidates},
                )
        if previous is not None:
            # Derived anchors can move a referenced candidate even when raw evidence is unchanged.
            should_stale = task_input_changed or _referenced_projection_changed(
                previous, prior_case.candidates if prior_case is not None else {}, candidates
            ) or (
                changed_ids is not None
                and bool(_referenced_evidence_ids(previous) & changed_ids)
            )
            if should_stale and previous.get("state") in ("ready", "stale"):
                curation["state"] = "stale"
                curation["snapshot_attested"] = False
                cu._invalidate_task_spec_approval(curation)
        case_doc: dict[str, Any] = {
            "schema_version": 2,
            "case_id": case_id,
            "pull_request": pull_request.model_dump(mode="json"),
            "snapshot": snapshot_doc,
            "source": {"import_file": import_file},
            "curation": curation,
            "candidates": [c.model_dump(mode="json") for c in candidates],
        }
        if facts is not None:
            case_doc["prioritization"] = facts.model_dump(mode="json")
        out.append((case_id, f"cases/{case_id}.yaml", case_doc))
    return out, bundle_drops


def _retired_snapshot_bundles(
    root: Path,
    manifest: dict[str, Any],
    number: int,
    new_cases: list[tuple[str, str, dict[str, Any]]],
) -> list[tuple[str, str]]:
    """Retire prior ready bundles only after a non-ready rewrite and only when no ready case still references them."""
    entry = _manifest_entry(manifest, number)
    if entry is None or entry.get("import_state") != "fetched":
        return []
    new_by_id = {case_id: case_doc for case_id, _, case_doc in new_cases}
    transitioned = {
        case_id
        for case_id, case_doc in new_by_id.items()
        if (case_doc.get("snapshot") or {}).get("status") != "ready"
    }
    if not transitioned:
        return []
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in manifest.get("cases", []):
        if isinstance(row, dict) and isinstance(row.get("case_id"), str):
            rows_by_id[row["case_id"]] = row
    old_docs: dict[str, schema.CaseDocument] = {}

    def old_doc(case_id: str) -> schema.CaseDocument:
        if case_id not in old_docs:
            row = rows_by_id.get(case_id)
            if not isinstance(row, dict) or not isinstance(row.get("case_file"), str):
                raise storage.WorkspaceCorrupt(
                    f"{root}: prior case {case_id} has no indexed case file"
                )
            raw = storage.load_yaml_strict(
                storage.resolve_authoring_path(root, row["case_file"])
            )
            prior_snapshot = raw.get("snapshot") if isinstance(raw, dict) else None
            if (
                isinstance(prior_snapshot, dict)
                and prior_snapshot.get("status") == "ready"
                and "base_resolution" not in prior_snapshot
            ):
                raise storage.WorkspaceCorrupt(
                    f"{root}: prior ready case {case_id} is missing snapshot.base_resolution; "
                    "run `daydream benchmark upgrade <workspace>` for this workspace "
                    "before refreshing"
                )
            try:
                old_docs[case_id] = schema.CaseDocument.model_validate(
                    schema._schema_ready(raw)
                )
            except Exception as exc:
                raise storage.WorkspaceCorrupt(
                    f"{root}: prior case {case_id} is not a valid case document"
                ) from exc
        return old_docs[case_id]

    candidates: dict[str, str] = {}
    for case_id in entry.get("case_ids", []):
        if case_id not in transitioned:
            continue
        prior_snapshot = old_doc(case_id).snapshot
        if not isinstance(prior_snapshot, schema.SnapshotReady):
            continue
        previous = candidates.get(prior_snapshot.bundle_file)
        if previous is not None and previous != prior_snapshot.bundle_sha256:
            raise storage.WorkspaceCorrupt(
                f"{root}: shared prior bundle has conflicting recorded digests"
            )
        candidates[prior_snapshot.bundle_file] = prior_snapshot.bundle_sha256

    if not candidates:
        return []
    retained_refs = {
        str((case_doc.get("snapshot") or {}).get("bundle_file"))
        for case_doc in new_by_id.values()
        if (case_doc.get("snapshot") or {}).get("status") == "ready"
        and (case_doc.get("snapshot") or {}).get("bundle_file")
    }
    for case_id in rows_by_id:
        if case_id in new_by_id:
            continue
        snapshot = old_doc(case_id).snapshot
        if isinstance(snapshot, schema.SnapshotReady):
            retained_refs.add(snapshot.bundle_file)
    return sorted(
        (bundle_file, digest)
        for bundle_file, digest in candidates.items()
        if bundle_file not in retained_refs
    )


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
        # Keep last-good linkage so a failed refresh cannot orphan curated cases.
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
    """Persist a fetch failure in the ledger without staging imports or cases.

    First imports become fetch_failed; failed refreshes preserve last-good linkage
    and record latest_error. The manifest rewrite is transactional.
    """
    _stages_failed(raw, number, code, message)
    with storage.Transaction(root, op_id=f"import-{number}", kind="import") as tx:
        tx.stage("benchmark.yaml", _manifest_bytes(raw))
        tx.commit()


@dataclass(frozen=True)
class _PriorCase:
    curation: dict[str, Any] | None
    candidates: dict[str, dict[str, Any]]
    snapshot: dict[str, Any]
    facts: dict[str, Any] | None


@dataclass
class _PriorImport:
    """One consistent read of prior evidence and its frozen case documents."""

    import_file: str
    requested_heads: list[str]
    document: dict[str, Any] | None = None
    evidence_signature: frozenset[tuple[int, str]] | None = None
    task_signature: str | None = None
    cases: dict[str, _PriorCase] = field(default_factory=dict)

    @property
    def pinned_head(self) -> str | None:
        for case in self.cases.values():
            if case.snapshot.get("policy") == "final_pr_head" and case.snapshot.get("original_head_sha"):
                return cast(str, case.snapshot["original_head_sha"])
        return None


def _prior_import_state(root: Path, raw: dict[str, Any], number: int) -> _PriorImport:
    """Load prior state before fetching; present-but-corrupt documents fail closed.

    Every authoring path passes containment checks. A ready/stale case must retain
    its original head: refresh must never silently fall back to the live PR head.
    """
    existing = _manifest_entry(raw, number)
    prior = _PriorImport(
        f"imports/pr-{number:06d}.json", list(existing.get("requested_heads", [])) if existing else [],
    )
    if existing is None or existing.get("import_state") != "fetched":
        return prior
    prior_import_file = existing.get("import_file")
    if not prior_import_file:
        raise storage.WorkspaceCorrupt(f"{root}: fetched ledger entry for PR {number} is missing import_file")
    prior.document = storage.load_json_strict(storage.resolve_authoring_path(root, prior_import_file))
    prior.evidence_signature = _evidence_signature_from_raw(prior.document)
    prior.task_signature = _task_input_signature_from_raw(prior.document)
    for case_id in existing.get("case_ids", []):
        case = storage.load_yaml_strict(storage.resolve_authoring_path(root, f"cases/{case_id}.yaml"))
        curation = case.get("curation")
        curation = curation if isinstance(curation, dict) else None
        candidates = case.get("candidates")
        snapshot = case.get("snapshot") or {}
        if (curation is not None and curation.get("state") in ("ready", "stale")
                and not snapshot.get("original_head_sha")):
            raise storage.WorkspaceCorrupt(f"{root}: ready/stale case {case_id} is missing snapshot.original_head_sha")
        facts = case.get("prioritization")
        prior.cases[case_id] = _PriorCase(
            curation=curation,
            candidates={c["source_id"]: c for c in candidates if isinstance(c, dict) and c.get("source_id")}
            if isinstance(candidates, list) else {},
            snapshot=snapshot,
            facts=facts if isinstance(facts, dict) else None,
        )
    return prior


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
    """Fetch, materialize, and commit one PR atomically; stage failures without losing prior linkage.

    Restore persisted anchors before comparing signatures, then derive missing or genuinely
    changed anchors during materialization. A changed referenced projection stales its
    case; metadata-only changes update digests without staling preserved curation.
    """
    prior = _prior_import_state(root, raw, number)
    import_file = prior.import_file
    try:
        # Refresh/re-import never orphans a previously pinned case. The same
        # union also decides whether a complete PR-file inventory is required:
        # a newly final-only refresh must still protect a retained explicit head.
        materialize_heads = requested_heads
        if prior.requested_heads:
            materialize_heads = list(dict.fromkeys([*prior.requested_heads, *requested_heads]))
        include_changed_files = any(head != "final" for head in materialize_heads)
        doc = fetch_and_normalize(
            root,
            repo,
            number,
            include_changed_files=include_changed_files,
        )
        # Existing final-head cases retain their frozen head even if the live PR advances.
        pinned = prior.pinned_head if prior is not None else None
        if pinned is not None:
            doc.pull_request.head.sha = pinned
        # Persisted anchors precede signature comparison; missing anchors are derived later.
        if prior.document is not None:
            _backfill_prior_anchors(doc, prior.document)
        # Task-input staleness is refresh-only; referenced evidence changes apply to every import.
        task_input_changed = (
            refresh
            and prior.task_signature is not None
            and prior.task_signature != _task_input_signature_from_doc(doc)
        )
        changed_ids: set[int] | None = None
        if prior.evidence_signature is not None:
            # A new (id, projection) pair changes that id; absent ids are deletions.
            # Prior duplicate projections are harmless when they cover every new pair.
            # Omitted legacy range fields are schema upgrades, not evidence edits.
            assert prior.document is not None
            legacy_without_start_line: set[int] = {
                int(e["database_id"])
                for e in prior.document.get("evidence", [])
                if "original_start_line" not in e
            }
            fresh = _evidence_signature_from_doc(doc, downgrade_start_line=legacy_without_start_line)
            prior_ids = {db_id for db_id, _ in prior.evidence_signature}
            fresh_ids = {db_id for db_id, _ in fresh}
            changed_ids = (prior_ids - fresh_ids) | {
                db_id for db_id, _ in fresh - prior.evidence_signature
            }
        # Refresh/re-import never orphans a previously pinned case: materialize
        # the union of the prior ledger heads and the newly-requested heads so
        # _stamp_fetched's cases[] rewrite keeps every curated case indexed.
        cases, bundle_rels = _case_materialize(
            doc, number, materialize_heads, import_file,
            root=root, repo_slug=repo, origin_url=origin_url,
            prior=prior,
            changed_ids=changed_ids, task_input_changed=task_input_changed,
        )
        retired_bundles = _retired_snapshot_bundles(root, raw, number, cases)
        # Re-serialize the mutated doc (authoring anchors derived on the typed
        # evidence records during materialization), recompute the fetch payload
        # digest over the same blocks, and keep every digest in lockstep.
        final_doc = doc.model_dump(mode="json")
        final_doc["fetch"]["payload_sha256"] = _payload_sha256(
            {k: final_doc[k] for k in ("schema_version", "repository", "pull_request", "evidence")}
        )
        import_bytes = json.dumps(final_doc, indent=2).encode("utf-8")
        import_sha256 = hashlib.sha256(import_bytes).hexdigest()
        for _, _, case_doc in cases:
            case_doc["source"]["import_sha256"] = import_sha256
        with storage.Transaction(root, op_id=f"import-{number}", kind="import") as tx:
            for rel, digest in retired_bundles:
                tx.retire(rel, expected_sha256=digest)
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
    """Recover, preflight, then atomically import each PR; any failed PR makes exit nonzero.

    Flat heads apply to every PR; pr_heads restricts explicit heads per PR, always
    including final. An omitted origin derives the GitHub URL; explicit None
    prevents snapshot freezing/network Git fetch. Commit the manifest last and
    preserve prior linkage on refresh failures.
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
    """Use preflight-resolved identity, with a private fallback only when no workspace manifest exists."""
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
    *,
    include_changed_files: bool = False,
) -> schema.ImportDocument:
    """Fetch the full PR header and all REST/GraphQL review evidence.

    Reconcile overlapping inline comments, thread state, and dismissal; retain all
    records including bots. Sort by database_id and created_at, independent of pages.
    Persist source/body hashes and the full header; the payload hash covers the
    complete normalized import. Any failed fetch raises without a substitute result.
    """
    header = _fetch_with_retry(root, owner_repo, number)
    changed_files = None
    if include_changed_files:
        changed_files = _normalize_changed_files(
            header,
            _rest(root, f"repos/{owner_repo}/pulls/{number}/files"),
        )

    review_records = _rest(root, f"repos/{owner_repo}/pulls/{number}/reviews")
    inline_records = [_evidence_from_inline(raw) for raw in _rest(root, f"repos/{owner_repo}/pulls/{number}/comments")]
    threads = _graphql_review_threads(root, owner_repo, number)

    evidence: list[dict[str, Any]] = [_evidence_from_review(raw) for raw in review_records]
    evidence.extend(_join_dismissal(_reconcile_inline_evidence(inline_records, threads), review_records))
    for raw in _rest(root, f"repos/{owner_repo}/issues/{number}/comments"):
        evidence.append(_rest_evidence(raw, "issue_comment", "IC"))

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
        "changed_files": changed_files,
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
                "fetched_at": schema.rfc3339_now(),
                "etag": None,
                "payload_sha256": _payload_sha256(import_doc),
            },
        }
    )


def _normalize_changed_files(header: dict[str, Any], rows: list[Any]) -> list[str]:
    """Return a complete canonical PR path inventory or fail closed."""
    valid_statuses = {
        "added",
        "removed",
        "modified",
        "renamed",
        "copied",
        "changed",
        "unchanged",
    }
    expected = header.get("changed_files")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 0:
        raise git_ops.GitError("PR changed_files count is missing or malformed")
    if expected > 3000:
        raise git_ops.GitError(
            f"PR changed_files count {expected} exceeds the 3000-file API inventory limit"
        )
    if len(rows) != expected:
        raise git_ops.GitError(
            f"PR changed_files inventory count mismatch: header={expected}, rows={len(rows)}"
        )

    current_names: set[str] = set()
    all_names: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise git_ops.GitError(f"PR changed_files row {index} is not an object")
        try:
            current = schema.exact_git_tree_path(row.get("filename"))
        except ValueError as exc:
            raise git_ops.GitError(f"PR changed_files row {index} has invalid filename: {exc}") from exc
        if current in current_names:
            raise git_ops.GitError(f"PR changed_files inventory repeats filename {current!r}")
        current_names.add(current)
        all_names.add(current)

        status = row.get("status")
        if not isinstance(status, str) or status not in valid_statuses:
            raise git_ops.GitError(
                f"PR changed_files row {index} has missing or unsupported status"
            )
        previous = row.get("previous_filename")
        if status in ("renamed", "copied") and previous is None:
            raise git_ops.GitError(
                f"PR changed_files {status} row {index} is missing previous_filename"
            )
        if status not in ("renamed", "copied") and previous is not None:
            raise git_ops.GitError(
                f"PR changed_files row {index} has unexpected previous_filename"
            )
        if previous is not None:
            try:
                all_names.add(schema.exact_git_tree_path(previous))
            except ValueError as exc:
                raise git_ops.GitError(
                    f"PR changed_files row {index} has invalid previous_filename: {exc}"
                ) from exc
    return sorted(all_names)
