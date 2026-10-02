"""Resolve operator GitHub App credentials into scoped, refreshable run authentication.
JWTs use RS256; display identity stays separate from credential-bearing subprocess
input. Also supports App manifest conversion and metadata lookup.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jwt as pyjwt

from daydream import git_ops
from daydream.timeutil import parse_iso_timestamp

APP_ID_ENV = "DAYDREAM_APP_ID"
APP_PRIVATE_KEY_ENV = "DAYDREAM_APP_PRIVATE_KEY"


class GitHubAppError(Exception):
    """Credential configuration, repository resolution, or App-token failure requiring a
    run abort.
    """


@dataclass(frozen=True)
class AppCredentials:
    """Numeric App id and secret PEM-encoded RSA signing key."""

    app_id: int
    private_key: str = field(repr=False)


@dataclass(frozen=True)
class GitHubIdentity:
    """The non-secret GitHub login displayed for a run."""

    login: str


@dataclass(frozen=True)
class GitHubExecutionInput:
    """Credential-bearing GitHub subprocess input owned by one run."""

    auth: git_ops.GitHubAuth = field(
        default=git_ops.INHERIT_GITHUB_AUTH,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class ResolvedGitHubSession:
    """One resolved display identity and its separate execution input."""

    identity: GitHubIdentity
    execution: GitHubExecutionInput = field(repr=False, compare=False)


@dataclass(frozen=True)
class _InstallationToken:
    """A minted installation credential and its GitHub-provided expiry."""

    token: str = field(repr=False)
    identity: str
    expires_at: float


def resolve_credentials(
    environment: Mapping[str, str] | None = None,
) -> AppCredentials | None:
    """Read both App environment variables, returning None only when both are absent.
    Partial configuration or a noninteger App id raises ValueError naming the field.
    """
    source = os.environ if environment is None else environment
    app_id_raw = source.get(APP_ID_ENV)
    private_key = source.get(APP_PRIVATE_KEY_ENV)

    if app_id_raw is None and private_key is None:
        return None
    if app_id_raw is None:
        raise ValueError(f"{APP_ID_ENV} is required when {APP_PRIVATE_KEY_ENV} is set")
    if private_key is None:
        raise ValueError(f"{APP_PRIVATE_KEY_ENV} is required when {APP_ID_ENV} is set")

    try:
        app_id = int(app_id_raw)
    except ValueError as exc:
        raise ValueError(f"{APP_ID_ENV} must be an integer, got {app_id_raw!r}") from exc

    return AppCredentials(app_id=app_id, private_key=private_key)


def mint_jwt(app_id: int, private_key: str) -> str:
    """Sign an RS256 App JWT with a backdated issue time and ten-minute expiry."""
    iat = int(time.time()) - 60
    payload = {
        "iss": str(app_id),
        "iat": iat,
        "exp": iat + 600,
    }
    return pyjwt.encode(payload, private_key, algorithm="RS256")


_APP_AUTH_ENV_KEYS = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GH_ENTERPRISE_TOKEN",
    "GITHUB_ENTERPRISE_TOKEN",
    APP_PRIVATE_KEY_ENV,
)


def build_gh_env(
    token: str,
    *,
    base_environment: Mapping[str, str],
) -> dict[str, str]:
    """Copy the complete subprocess environment, remove ambient credential variables, and
    bind GH_TOKEN.
    """
    environment = dict(base_environment)
    for name in _APP_AUTH_ENV_KEYS:
        environment.pop(name, None)
    environment["GH_TOKEN"] = token
    return environment


def build_app_jwt_auth(
    app_id: int,
    private_key: str,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> tuple[git_ops.StaticGitHubAuth, dict[str, str]]:
    """Build explicit static auth and Bearer headers for one App JWT."""
    jwt_token = mint_jwt(app_id, private_key)
    base = os.environ if base_environment is None else base_environment
    auth = git_ops.StaticGitHubAuth(
        build_gh_env(jwt_token, base_environment=base)
    )
    return auth, {"Authorization": f"Bearer {jwt_token}"}


def _mint_installation_token(
    repo_dir: Path,
    app_id: int,
    private_key: str,
    owner: str,
    repo: str,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> _InstallationToken:
    """Find the owner installation and mint a repository-scoped token with its expiry.
    Missing App slug yields cosmetic identity unknown; installation/exchange failures
    raise ValueError.
    """
    auth, bearer = build_app_jwt_auth(
        app_id,
        private_key,
        base_environment=base_environment,
    )
    installation_id, identity = _find_installation(
        repo_dir,
        owner,
        repo,
        bearer,
        auth=auth,
    )
    token, expires_at = _exchange_for_token(
        repo_dir,
        installation_id,
        owner,
        repo,
        bearer,
        auth=auth,
    )
    return _InstallationToken(token=token, identity=identity, expires_at=expires_at)


def _find_installation(
    repo_dir: Path,
    owner: str,
    repo: str,
    headers: dict[str, str],
    *,
    auth: git_ops.GitHubAuth,
) -> tuple[int, str]:
    """List App installations and return ``(id, "{slug}[bot]")`` for *owner*."""
    try:
        installations = git_ops.gh_api(
            repo_dir,
            "/app/installations",
            paginate=True,
            jq=".[]",
            headers=headers,
            idempotent=True,
            auth=auth,
        )
    except git_ops.GitError as exc:
        raise ValueError(f"failed to list App installations: {exc}") from exc

    for entry in installations:
        account = entry.get("account") or {}
        login = account.get("login")
        if isinstance(login, str) and login.lower() == owner.lower():
            installation_id = entry.get("id")
            if not isinstance(installation_id, int):
                raise ValueError(f"installation for {owner!r} is missing an integer id")
            slug = entry.get("app_slug")
            identity = f"{slug}[bot]" if isinstance(slug, str) and slug else "unknown"
            return installation_id, identity
    raise ValueError(f"no App installation found for owner {owner!r} (repo {owner}/{repo})")


def _exchange_for_token(
    repo_dir: Path,
    installation_id: int,
    owner: str,
    repo: str,
    headers: dict[str, str],
    *,
    auth: git_ops.GitHubAuth,
) -> tuple[str, float]:
    """Mint a repository-scoped installation token; reject transport, token-shape, or
    expiry errors.
    """
    try:
        payload = git_ops.gh_api(
            repo_dir,
            f"/app/installations/{installation_id}/access_tokens",
            method="POST",
            input_data={"repositories": [repo]},
            headers=headers,
            auth=auth,
        )
    except git_ops.GitError as exc:
        raise ValueError(f"failed to mint installation token for {owner}/{repo}: {exc}") from exc

    token = payload.get("token") if isinstance(payload, dict) else None
    if not isinstance(token, str) or not token:
        raise ValueError(f"installation token response for {owner}/{repo} is missing the 'token' field")
    expires_at_raw = payload.get("expires_at") if isinstance(payload, dict) else None
    if not isinstance(expires_at_raw, str) or not expires_at_raw:
        raise ValueError(f"installation token response for {owner}/{repo} is missing the 'expires_at' field")
    try:
        expires_at = parse_iso_timestamp(expires_at_raw).timestamp()
    except ValueError as exc:
        raise ValueError(f"installation token response for {owner}/{repo} has invalid 'expires_at'") from exc
    return token, expires_at


def exchange_manifest_code(
    repo_dir: Path,
    code: str,
    *,
    auth: git_ops.GitHubAuth = git_ops.INHERIT_GITHUB_AUTH,
) -> tuple[AppCredentials, str]:
    """Exchange the credential-bearing manifest callback code for App id/key/slug. Require
    an object, integer id, and nonempty PEM; missing cosmetic slug becomes unknown.
    Conversion failures raise GitHubAppError.
    """
    try:
        payload = git_ops.gh_api(
            repo_dir,
            f"/app-manifests/{code}/conversions",
            method="POST",
            auth=auth,
        )
    except git_ops.GitError as exc:
        raise GitHubAppError(f"failed to exchange App manifest code: {exc}") from exc

    if not isinstance(payload, dict):
        raise GitHubAppError("App manifest conversion response is not a JSON object")

    app_id = payload.get("id")
    if not isinstance(app_id, int):
        raise GitHubAppError("App manifest conversion response is missing an integer 'id' field")

    private_key = payload.get("pem")
    if not isinstance(private_key, str) or not private_key:
        raise GitHubAppError("App manifest conversion response is missing the 'pem' field")

    slug = payload.get("slug")
    slug = slug if isinstance(slug, str) and slug else "unknown"

    return AppCredentials(app_id=app_id, private_key=private_key), slug


def get_app_metadata(
    repo_dir: Path,
    app_id: int,
    private_key: str,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read /app with matching explicit JWT authentication and Bearer header. Ambient
    credentials remain untouched; transport or non-object responses raise
    GitHubAppError.
    """
    auth, bearer = build_app_jwt_auth(
        app_id,
        private_key,
        base_environment=base_environment,
    )
    try:
        payload = git_ops.gh_api(
            repo_dir,
            "/app",
            headers=bearer,
            idempotent=True,
            auth=auth,
        )
    except git_ops.GitError as exc:
        raise GitHubAppError(f"failed to read App metadata: {exc}") from exc

    if not isinstance(payload, dict):
        raise GitHubAppError("App metadata response is not a JSON object")
    return payload


def resolve_user_identity(
    repo_dir: Path,
    *,
    auth: git_ops.GitHubAuth = git_ops.INHERIT_GITHUB_AUTH,
) -> str:
    """Read the active user login; cosmetic lookup failures return unknown without aborting
    the run.
    """
    try:
        login = git_ops.gh_api(
            repo_dir,
            "/user",
            idempotent=True,
            auth=auth,
        ).get("login")
    except Exception:  # noqa: BLE001 - identity display is cosmetic; never abort a run
        return "unknown"
    if isinstance(login, str) and login:
        return login
    return "unknown"


def resolve_run_identity(
    target_dir: Path,
    pr_repo: str | None,
    *,
    is_posting: bool,
    base_environment: Mapping[str, str] | None = None,
) -> ResolvedGitHubSession:
    """Validate credential configuration even for read-only runs. No-App/read-only sessions
    inherit live parent auth unless a complete base environment was supplied. Posting
    with App credentials requires owner/repo and a scoped refreshable token; failures
    abort. Credentials are owned by this session and never mutate process-global state.
    """
    source_environment = dict(
        os.environ if base_environment is None else base_environment
    )
    try:
        credentials = resolve_credentials(source_environment)
    except ValueError as exc:
        raise GitHubAppError(str(exc)) from exc
    inherited_auth: git_ops.GitHubAuth
    if base_environment is None:
        inherited_auth = git_ops.INHERIT_GITHUB_AUTH
    else:
        inherited_auth = git_ops.StaticGitHubAuth(source_environment)
    if credentials is None or not is_posting:
        return ResolvedGitHubSession(
            GitHubIdentity(
                resolve_user_identity(target_dir, auth=inherited_auth)
            ),
            GitHubExecutionInput(inherited_auth),
        )

    owner_repo = _owner_repo_for(pr_repo, target_dir, auth=inherited_auth)
    if owner_repo is None:
        raise GitHubAppError("Cannot determine owner/repo for installation token minting")

    owner, repo = owner_repo

    def mint() -> tuple[_InstallationToken, git_ops.StaticGitHubAuth]:
        token = _mint_installation_token(
            target_dir,
            credentials.app_id,
            credentials.private_key,
            owner,
            repo,
            base_environment=source_environment,
        )
        return token, git_ops.StaticGitHubAuth(
            build_gh_env(token.token, base_environment=source_environment)
        )

    try:
        minted, token_auth = mint()

        def refresh() -> tuple[git_ops.StaticGitHubAuth, float]:
            fresh, fresh_auth = mint()
            return fresh_auth, fresh.expires_at

        auth = git_ops.RefreshingGitHubAuth(
            token_auth,
            expires_at=minted.expires_at,
            refresh=refresh,
        )
    except Exception:
        raise GitHubAppError("App token resolution failed") from None

    return ResolvedGitHubSession(
        GitHubIdentity(minted.identity),
        GitHubExecutionInput(auth),
    )


def _owner_repo_for(
    pr_repo: str | None,
    target_dir: Path,
    *,
    auth: git_ops.GitHubAuth = git_ops.INHERIT_GITHUB_AUTH,
) -> tuple[str, str] | None:
    """Prefer a valid owner/repo override, otherwise query the repository; unresolved
    identity returns None.
    """
    if pr_repo:
        parsed = git_ops.split_owner_repo(pr_repo)
        if parsed is not None:
            return parsed
    return git_ops.gh_repo_view(target_dir, auth=auth)
