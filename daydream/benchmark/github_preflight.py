"""Validate GitHub access, workspace repository identity, and explicit import targets."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from daydream import git_ops
from daydream.benchmark import schema, storage
from daydream.git_ops import process as git_process


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


def preflight(root: Path, pr_count: int) -> None:
    """Run fixed-order binary, authentication, identity, and access checks."""
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
    """Merge CLI PRs before file entries, preserving first-seen order."""
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
