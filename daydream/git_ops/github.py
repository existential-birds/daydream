"""GitHub CLI endpoint operations and bounded API pagination."""

from __future__ import annotations

import base64
import json
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

from daydream.git_ops import process, queries
from daydream.git_ops.auth import INHERIT_GITHUB_AUTH, GitHubAuth
from daydream.git_ops.models import GitError, GitHubPageLimits, GitHubRequestBudget, GitTimeoutError, PathAbsentError
from daydream.repository_paths import strip_dot_slash, valid_repository_file_path

# --- gh wrappers -------------------------------------------------------------


GH_PR_VIEW_FIELDS: tuple[str, ...] = (
    "number",
    "title",
    "body",
    "state",
    "headRefName",
    "baseRefName",
    "headRefOid",
    "url",
    "headRepository",
    "headRepositoryOwner",
)
GH_PR_LIST_FIELDS: tuple[str, ...] = (
    "number",
    "headRefName",
    "headRefOid",
    "baseRefName",
    "url",
    "headRepository",
    "headRepositoryOwner",
)
_GH_DIAGNOSTIC_LIMIT = 2_000
_MISSING_BRANCH_PR_RE = re.compile(r'^no pull requests found for branch "[^"\r\n]+"$')


def _safe_gh_diagnostic(stderr: str) -> str:
    """Return a redacted, bounded one-command diagnostic."""
    redacted = process._redact_sensitive_text(stderr.strip())
    if len(redacted) <= _GH_DIAGNOSTIC_LIMIT:
        return redacted
    return redacted[:_GH_DIAGNOSTIC_LIMIT] + "...[truncated]"


def _pr_view_is_absent(stderr: str, pr: int | None) -> bool:
    """Recognize only gh's two established PR-absence diagnostics."""
    diagnostic = stderr.strip()
    if pr is None:
        return _MISSING_BRANCH_PR_RE.fullmatch(diagnostic) is not None
    return diagnostic == (
        f"GraphQL: Could not resolve to a PullRequest with the number of {pr}. (repository.pullRequest)"
    )


def gh_pr_view(
    repo: Path,
    pr: int | None = None,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> dict[str, Any] | None:
    """Read a PR, inferring it from the current branch when no number is supplied.

    Return ``None`` only for a recognized missing-PR diagnostic. Other failures and
    malformed JSON raise ``GitError``.
    """
    args = ["pr", "view"]
    if pr is not None:
        args.append(str(pr))
    args.extend(
        [
            "--json",
            ",".join(GH_PR_VIEW_FIELDS),
        ]
    )
    proc = process._run_gh(repo, args, auth=auth, retries=process._gh_retries())
    if proc.returncode != 0:
        if _pr_view_is_absent(proc.stderr, pr):
            return None
        diagnostic = _safe_gh_diagnostic(proc.stderr) or "no diagnostic"
        raise process._gh_error_for(f"gh pr view failed: {diagnostic}", proc.stderr)
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise GitError("gh pr view returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise GitError("gh pr view expected a JSON object")
    return data


def gh_pr_list_for_branch(
    repo: Path,
    branch: str,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> list[dict[str, Any]]:
    """List open PRs for a head ref; only a successful empty query returns ``[]``."""
    proc = process._run_gh(
        repo,
        [
            "pr",
            "list",
            "--head",
            branch,
            "--state",
            "open",
            "--json",
            ",".join(GH_PR_LIST_FIELDS),
        ],
        auth=auth,
        retries=process._gh_retries(),
    )
    if proc.returncode != 0:
        diagnostic = _safe_gh_diagnostic(proc.stderr) or "no diagnostic"
        raise process._gh_error_for(f"gh pr list failed: {diagnostic}", proc.stderr)
    try:
        rows = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise GitError("gh pr list returned invalid JSON") from exc
    if not isinstance(rows, list):
        raise GitError("gh pr list expected a JSON list")
    if any(not isinstance(row, dict) for row in rows):
        raise GitError("gh pr list returned a non-object row")
    return rows


def gh_pr_diff(
    repo: Path,
    pr: int,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> str:
    """Return the PR's unified diff, raising ``GitError`` if ``gh pr diff`` fails."""
    proc = process._run_gh(
        repo,
        ["pr", "diff", str(pr)],
        auth=auth,
        retries=process._gh_retries(),
    )
    process._require_ok(proc, f"gh pr diff {pr} failed")
    return proc.stdout


def gh_repo_view(
    repo: Path,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> tuple[str, str] | None:
    """Return the ``(owner, name)`` slug, or ``None`` when it cannot be read."""
    try:
        return gh_repo_view_required(repo, auth=auth)
    except GitTimeoutError:
        raise
    except GitError:
        return None


def gh_repo_view_required(
    repo: Path,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> tuple[str, str]:
    """Return the current repository slug, raising on command or shape failure."""
    proc = process._run_gh(
        repo,
        ["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        auth=auth,
        retries=process._gh_retries(),
    )
    if proc.returncode != 0:
        diagnostic = _safe_gh_diagnostic(proc.stderr) or "no diagnostic"
        raise process._gh_error_for(f"gh repo view failed: {diagnostic}", proc.stderr)
    raw_slug = proc.stdout.rstrip("\r\n")
    slug = queries.split_owner_repo(raw_slug)
    if slug is None:
        raise GitError("gh repo view returned an invalid repository slug")
    return slug


def _parse_gh_json(stdout: str, jq: str | None, endpoint: str, *, payload_note: str = "") -> Any:
    """Parse NDJSON into a list with ``jq``, otherwise parse one JSON value.

    Invalid JSON raises ``GitError`` with ``payload_note`` appended to the message.
    """
    try:
        if jq is not None:
            return [json.loads(line) for line in stdout.splitlines() if line.strip()]
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise GitError(
            process._redact_sensitive_text(f"gh api {endpoint} returned invalid JSON: {exc}{payload_note}")
        ) from exc


_GITHUB_JSON_HEADERS: tuple[str, ...] = (
    "Accept: application/vnd.github+json",
    "X-GitHub-Api-Version: 2022-11-28",
)


async def _gh_api_read(
    repo: Path,
    endpoint: str,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    budget: GitHubRequestBudget,
) -> Any:
    """Read one GitHub endpoint with explicit version headers."""
    args = ["api"]
    for header in _GITHUB_JSON_HEADERS:
        args.extend(("-H", header))
    args.extend(("--method", "GET", endpoint))
    proc = await process._run_gh_async(repo, args, auth=auth, budget=budget)
    if proc.returncode != 0:
        diagnostic = _safe_gh_diagnostic(proc.stderr) or "no diagnostic"
        raise process._gh_error_for(f"gh api {endpoint} failed: {diagnostic}", proc.stderr)
    return _parse_gh_json(proc.stdout, None, endpoint)


def _validate_page_limits(limits: GitHubPageLimits) -> None:
    if limits.per_page <= 0 or limits.max_pages <= 0:
        raise GitError("GitHub pagination limits must be positive")


async def gh_api_bounded_pages(
    repo: Path,
    endpoint: str,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    envelope: str | None,
    limits: GitHubPageLimits,
    budget: GitHubRequestBudget,
) -> list[dict[str, Any]]:
    """Collect a manually paged GitHub list without silent truncation."""
    _validate_page_limits(limits)
    capacity = limits.per_page * limits.max_pages
    collected: list[dict[str, Any]] = []
    separator = "&" if "?" in endpoint else "?"

    for page in range(1, limits.max_pages + 1):
        page_endpoint = f"{endpoint}{separator}per_page={limits.per_page}&page={page}"
        value = await _gh_api_read(
            repo,
            page_endpoint,
            auth=auth,
            budget=budget,
        )
        total_count: int | None = None
        if envelope is None:
            rows = value
        else:
            if not isinstance(value, dict):
                raise GitError(f"gh api {endpoint} returned an invalid page shape")
            total_count = value.get("total_count")
            if not isinstance(total_count, int) or isinstance(total_count, bool) or total_count < 0:
                raise GitError(f"gh api {endpoint} returned an invalid page shape")
            if total_count > capacity:
                raise GitError(f"gh api {endpoint} exceeded pagination limit")
            rows = value.get(envelope)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise GitError(f"gh api {endpoint} returned an invalid page shape")
        if len(rows) > limits.per_page:
            raise GitError(f"gh api {endpoint} returned an invalid page shape")
        collected.extend(rows)
        if total_count is not None:
            if len(collected) > total_count:
                raise GitError(f"gh api {endpoint} returned an invalid page shape")
            if len(collected) == total_count:
                return collected
        if page == limits.max_pages and len(rows) == limits.per_page:
            raise GitError(f"gh api {endpoint} exceeded pagination limit")
        if len(rows) < limits.per_page:
            if total_count is not None and len(collected) != total_count:
                raise GitError(f"gh api {endpoint} returned an incomplete page")
            return collected

    raise GitError(f"gh api {endpoint} exceeded pagination limit")


async def gh_pr_ci_snapshot(
    repo: Path,
    owner: str,
    name: str,
    number: int,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    budget: GitHubRequestBudget,
) -> dict[str, Any]:
    """Return one PR object from the bounded asynchronous read boundary."""
    endpoint = f"repos/{owner}/{name}/pulls/{number}"
    value = await _gh_api_read(repo, endpoint, auth=auth, budget=budget)
    if not isinstance(value, dict):
        raise GitError(f"gh api {endpoint} returned an invalid top-level shape")
    return value


async def gh_active_branch_rules(
    repo: Path,
    owner: str,
    name: str,
    branch: str,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    limits: GitHubPageLimits,
    budget: GitHubRequestBudget,
) -> list[dict[str, Any]]:
    """Return active repository rules applying to a branch."""
    encoded_branch = quote(branch, safe="")
    endpoint = f"repos/{owner}/{name}/rules/branches/{encoded_branch}"
    return await gh_api_bounded_pages(
        repo,
        endpoint,
        auth=auth,
        envelope=None,
        limits=limits,
        budget=budget,
    )


async def gh_classic_required_checks(
    repo: Path,
    owner: str,
    name: str,
    branch: str,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    budget: GitHubRequestBudget,
) -> dict[str, Any] | None:
    """Return classic required checks, or None only for unprotected branches."""
    encoded_branch = quote(branch, safe="")
    endpoint = f"repos/{owner}/{name}/branches/{encoded_branch}/protection/required_status_checks"
    try:
        value = await _gh_api_read(repo, endpoint, auth=auth, budget=budget)
    except GitError as exc:
        absent = f"gh api {endpoint} failed: gh: Branch not protected (HTTP 404)"
        if type(exc) is GitError and str(exc) == absent:
            return None
        raise
    if not isinstance(value, dict):
        raise GitError(f"gh api {endpoint} returned an invalid top-level shape")
    return value


def gh_api(
    repo: Path,
    endpoint: str,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    method: str = "GET",
    paginate: bool = False,
    input_data: Any | None = None,
    jq: str | None = None,
    headers: dict[str, str] | None = None,
    idempotent: bool = False,
) -> Any:
    """Call ``gh api <endpoint>`` and parse JSON, or NDJSON when *jq* is set.

    A JSON *input_data* body travels through a temporary file: success removes
    it; failure preserves it and names its path in ``GitError``. *jq* flattens
    concatenated paginated arrays into one JSON value per line. Explicit
    authorization headers support App JWT calls. Only read callers set
    *idempotent* to retry timeouts: GraphQL mutations also use POST and cannot
    be inferred from *method*. Rate limits raise ``RateLimitError``.
    """
    header_args = [arg for name, value in (headers or {}).items() for arg in ("-H", f"{name}: {value}")]
    output_args: list[str] = []
    if paginate:
        output_args.append("--paginate")
    if jq is not None:
        output_args.extend(["--jq", f"({jq}) | @json"])
    retries = process._gh_retries() if idempotent else 0

    if input_data is None:
        method_args = ["-X", method.upper()] if method.upper() != "GET" else []
        args = ["api", *header_args, *method_args, *output_args, endpoint]
        proc = process._run_gh(repo, args, auth=auth, retries=retries)
        if proc.returncode != 0:
            raise process._gh_error_for(f"gh api {endpoint} failed: {proc.stderr.strip()}", proc.stderr)
        return _parse_gh_json(proc.stdout, jq, endpoint)

    # input_data path: serialise to a tempfile and shell out via `--input`.
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 - lifecycle managed below
        suffix=".json", mode="w", delete=False, encoding="utf-8"
    )
    tmp_path = Path(tmp.name)
    payload_note = f" (request payload preserved at {tmp_path})"
    succeeded = False
    try:
        try:
            json.dump(input_data, tmp)
        finally:
            tmp.close()
        args = ["api", *header_args, endpoint, "--method", method.upper(), "--input", str(tmp_path), *output_args]
        proc = process._run_gh(repo, args, auth=auth, retries=retries)
        if proc.returncode != 0:
            raise process._gh_error_for(
                f"gh api {endpoint} failed: {proc.stderr.strip()}{payload_note}",
                proc.stderr,
            )
        result = _parse_gh_json(proc.stdout, jq, endpoint, payload_note=payload_note)
        succeeded = True
        return result
    except GitError as exc:
        exc.preserved_payload_path = tmp_path
        raise
    finally:
        if succeeded:
            tmp_path.unlink(missing_ok=True)


# GitHub's own owner/repo name grammar. ``split_owner_repo`` only rejects
# whitespace and a wrong slash count, which would still let ``..`` or an
# encoded segment reshape the endpoint path built below.
_GITHUB_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")
_COMMIT_SHA_RE = re.compile(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}")


def _decode_github_base64(content: str, what: str) -> bytes:
    """Decode one line-wrapped GitHub base64 payload into raw bytes."""
    try:
        return base64.b64decode(content)
    except ValueError as exc:  # binascii.Error subclasses ValueError
        raise GitError(f"{what} returned undecodable base64 content") from exc


_GH_HTTP_404_RE = re.compile(r"\bHTTP 404\b")


def _gh_failure_is_absence(exc: GitError) -> bool:
    """Recognize only gh's established ``(HTTP 404)`` diagnostic as proven absence."""
    return _GH_HTTP_404_RE.search(str(exc)) is not None


def gh_file_at_ref(
    repo: Path,
    slug: str,
    ref: str,
    path: str,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> bytes:
    """Read immutable file bytes from GitHub without a checkout of the target repository.

    Validate the explicit slug/ref and untrusted relative path, then percent-encode the
    path. Only HTTP 404 or a response naming no file proves absence; all other malformed
    inputs and request failures remain ``GitError``. ``repo`` supplies gh's cwd only.
    """
    owner_repo = queries.split_owner_repo(slug)
    if owner_repo is None or not all(_GITHUB_NAME_RE.fullmatch(part) for part in owner_repo):
        raise GitError(f"invalid repository slug: {slug}")
    if _COMMIT_SHA_RE.fullmatch(ref) is None:
        raise GitError(f"invalid commit sha: {ref}")
    if not valid_repository_file_path(path):
        raise GitError("invalid repository file path")
    owner, name = owner_repo
    relative = strip_dot_slash(path)
    base = f"repos/{owner}/{name}"
    where = f"{slug}@{ref}:{relative}"
    try:
        payload = gh_api(
            repo,
            f"{base}/contents/{quote(relative)}?ref={ref}",
            auth=auth,
            idempotent=True,
        )
    except GitError as exc:
        if _gh_failure_is_absence(exc):
            raise PathAbsentError(str(exc)) from exc
        raise
    if not isinstance(payload, dict) or payload.get("type") != "file":
        raise PathAbsentError(f"{where} does not name a file")
    content = payload.get("content")
    if payload.get("encoding") == "base64" and isinstance(content, str):
        return _decode_github_base64(content, where)
    # Past the contents endpoint's 1 MiB inline ceiling GitHub answers with an
    # empty body and ``encoding: "none"``; the blob endpoint still serves it.
    blob_sha = payload.get("sha")
    if not isinstance(blob_sha, str) or _COMMIT_SHA_RE.fullmatch(blob_sha) is None:
        raise GitError(f"{where} carries no readable content")
    blob = gh_api(
        repo,
        f"{base}/git/blobs/{blob_sha}",
        auth=auth,
        idempotent=True,
    )
    blob_content = blob.get("content") if isinstance(blob, dict) else None
    if not isinstance(blob, dict) or blob.get("encoding") != "base64" or not isinstance(blob_content, str):
        raise GitError(f"{where} carries no readable blob content")
    return _decode_github_base64(blob_content, where)


# --- gh secret / variable / PR primitives ------------------------------------


def _scope_args(org: str | None, repo_slug: str | None) -> list[str]:
    """Require exactly one organization or repository scope."""
    if (org is None) == (repo_slug is None):
        raise GitError("exactly one of org or repo_slug must be provided")
    return ["--org", org] if org is not None else ["--repo", repo_slug or ""]


def gh_secret_set(
    repo: Path,
    name: str,
    value: str,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    org: str | None = None,
    repo_slug: str | None = None,
) -> None:
    """Set an Actions secret at exactly one scope, passing its value on stdin.

    Secret material never enters process listings. Scope or command failures raise.
    """
    args = ["secret", "set", name, *_scope_args(org, repo_slug)]
    proc = process._run_gh(repo, args, auth=auth, input_text=value)
    if proc.returncode != 0:
        raise process._gh_error_for(f"gh secret set {name} failed: {proc.stderr.strip()}", proc.stderr)


def gh_variable_set(
    repo: Path,
    name: str,
    value: str,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    org: str | None = None,
    repo_slug: str | None = None,
) -> None:
    """Set a non-secret Actions variable via ``--body`` at exactly one scope; failures raise."""
    args = ["variable", "set", name, "--body", value, *_scope_args(org, repo_slug)]
    proc = process._run_gh(repo, args, auth=auth)
    if proc.returncode != 0:
        raise process._gh_error_for(f"gh variable set {name} failed: {proc.stderr.strip()}", proc.stderr)


def _gh_name_list(
    repo: Path,
    kind: str,
    org: str | None,
    repo_slug: str | None,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> list[str]:
    """Run ``gh <kind> list --json name`` and return the names."""
    args = [kind, "list", "--json", "name", *_scope_args(org, repo_slug)]
    proc = process._run_gh(repo, args, auth=auth, retries=process._gh_retries())
    if proc.returncode != 0:
        raise process._gh_error_for(f"gh {kind} list failed: {proc.stderr.strip()}", proc.stderr)
    try:
        entries = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise GitError(f"gh {kind} list returned invalid JSON: {exc}") from exc
    return [entry["name"] for entry in entries]


def gh_secret_list(
    repo: Path,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    org: str | None = None,
    repo_slug: str | None = None,
) -> list[str]:
    """List secret names at exactly one scope; values remain unavailable and failures raise."""
    return _gh_name_list(repo, "secret", org, repo_slug, auth=auth)


def gh_variable_list(
    repo: Path,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    org: str | None = None,
    repo_slug: str | None = None,
) -> list[str]:
    """List variable names at exactly one scope; failed queries raise."""
    return _gh_name_list(repo, "variable", org, repo_slug, auth=auth)


def gh_pr_create(
    repo: Path,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    head: str,
    base: str,
    title: str,
    body: str,
    repo_slug: str | None = None,
) -> str:
    """Create a PR and return its URL; an explicit slug overrides gh's cwd context."""
    args = ["pr", "create", "--head", head, "--base", base, "--title", title, "--body", body]
    if repo_slug is not None:
        args += ["--repo", repo_slug]
    proc = process._run_gh(repo, args, auth=auth)
    if proc.returncode != 0:
        raise process._gh_error_for(f"gh pr create failed: {proc.stderr.strip()}", proc.stderr)
    return proc.stdout.strip()


def gh_issue_create(
    repo: Path,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    title: str,
    body: str,
    repo_slug: str | None = None,
    labels: list[str] | None = None,
) -> str:
    """Create an issue for an out-of-scope finding and return its URL.

    The body travels through ``--body-file`` and never appears on argv.
    *repo_slug* selects an explicit repository; labels are optional. Failed
    creation raises ``GitError``.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as bf:
        bf.write(body)
        body_path = bf.name
    args: list[str] = ["issue", "create"]
    if repo_slug is not None:
        args += ["--repo", repo_slug]
    args += ["--title", title, "--body-file", body_path]
    if labels:
        for label in labels:
            args += ["--label", label]
    try:
        proc = process._run_gh(repo, args, auth=auth)
    finally:
        try:
            Path(body_path).unlink()
        except OSError:
            pass
    if proc.returncode != 0:
        raise process._gh_error_for(f"gh issue create failed: {proc.stderr.strip()}", proc.stderr)
    return proc.stdout.strip()


def gh_issue_list(
    repo: Path,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    state: str = "open",
    search: str | None = None,
    limit: int = 100,
    repo_slug: str | None = None,
) -> list[dict[str, Any]]:
    """List issues for best-effort cross-run finding deduplication.

    Returns ``[]`` on lookup failure so a transient GitHub error does not
    block filing an out-of-scope finding. Rows carry number, title, body, and URL.
    """
    args: list[str] = [
        "issue",
        "list",
        "--state",
        state,
        "--json",
        "number,title,body,url",
        "--limit",
        str(limit),
    ]
    if search:
        args += ["--search", search]
    if repo_slug is not None:
        args += ["--repo", repo_slug]
    try:
        proc = process._run_gh(repo, args, auth=auth, retries=process._gh_retries())
    except GitError as exc:
        process._logger.warning("gh issue list failed (%s, returning []): %s", type(exc).__name__, exc)
        return []
    if proc.returncode != 0:
        return []
    try:
        rows = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return rows if isinstance(rows, list) else []


def gh_issue_list_strict(
    repo: Path,
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    state: str = "all",
    repo_slug: str,
) -> list[dict[str, Any]]:
    """List all repository issues with strict lookup and response validation.

    A failed or malformed lookup raises ``GitError`` because an empty result
    could cause duplicate writes. Pagination is unbounded, and GitHub's
    issue-endpoint pull request rows are excluded.
    """
    if state not in {"open", "closed", "all"}:
        raise GitError(f"invalid issue state {state!r}")
    parsed = queries.split_owner_repo(repo_slug)
    if parsed is None or "/" in parsed[1]:
        raise GitError(f"invalid GitHub repository slug {repo_slug!r}")
    owner, name = parsed
    endpoint = f"repos/{owner}/{name}/issues?state={state}&per_page=100"
    rows = gh_api(
        repo,
        endpoint,
        auth=auth,
        paginate=True,
        jq=".[]",
        idempotent=True,
    )
    if not isinstance(rows, list):
        raise GitError("GitHub issue lookup returned a non-list response")

    issues: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise GitError("GitHub issue lookup returned a non-object row")
        if row.get("pull_request") is not None:
            continue
        number = row.get("number")
        title = row.get("title")
        body = row.get("body")
        url = row.get("html_url", row.get("url"))
        row_state = row.get("state")
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or not isinstance(title, str)
            or body is not None
            and not isinstance(body, str)
            or not isinstance(url, str)
            or not url
            or not isinstance(row_state, str)
        ):
            raise GitError("GitHub issue lookup returned a malformed issue row")
        issues.append(
            {
                "number": number,
                "title": title,
                "body": body or "",
                "url": url,
                "state": row_state,
            }
        )
    return issues
