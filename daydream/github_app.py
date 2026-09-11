"""GitHub App identity: credential resolution, JWT minting, and gh env building.

Daydream can run under an operator-owned GitHub App bot identity. The operator
supplies the App credentials via the ``DAYDREAM_APP_ID`` and
``DAYDREAM_APP_PRIVATE_KEY`` environment variables; this module turns those into
a short-lived RS256 JWT, exchanges that JWT for a scoped installation access
token, and resolves the active GitHub identity for banner display.

This module also supports the App-from-manifest registration flow: exchanging
a manifest-conversion code for a newly created App's credentials
(:func:`exchange_manifest_code`) and reading an App's metadata
(:func:`get_app_metadata`).
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
    """Raised when GitHub App identity resolution must abort the run.

    Covers every hard-abort case: partial or malformed credentials,
    owner/repo undeterminable while posting, and installation-token
    minting or injection failure.
    """


@dataclass(frozen=True)
class AppCredentials:
    """Operator-supplied GitHub App credentials.

    Attributes:
        app_id: Numeric GitHub App ID.
        private_key: PEM-encoded RSA private key for RS256 JWT signing.
    """

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
    """Resolve GitHub App credentials from the environment.

    Returns:
        ``AppCredentials`` when both env vars are present and valid, or ``None``
        when both are absent (opt-in: no App identity, no behavior change).

    Raises:
        ValueError: If exactly one of the two env vars is present (partial
            misconfiguration; names the missing var), or if ``DAYDREAM_APP_ID``
            is present but not parseable as an integer.
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
    """Mint a short-lived RS256 JWT authenticating as the GitHub App.

    Args:
        app_id: Numeric GitHub App ID, used as the ``iss`` claim.
        private_key: PEM-encoded RSA private key for RS256 signing.
    """
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
    """Bind *token* to a sanitized, complete subprocess environment.

    Args:
        token: Token to inject as ``GH_TOKEN`` (App JWT or installation token).
        base_environment: Complete environment to copy and sanitize.

    Returns:
        A complete environment with ambient credential variables removed and
        the explicit token installed as ``GH_TOKEN``.
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
    """Exchange App credentials for a scoped installation access token.

    Mints an App JWT, lists the App's installations to find the one owned by
    *owner*, and exchanges that installation for a short-lived access token.

    Returns:
        An :class:`_InstallationToken` carrying the scoped access token, the
        App's ``"{slug}[bot]"`` identity from the matched installation's
        ``app_slug`` field (``"unknown"`` if the slug is absent; identity is
        cosmetic and never fails the mint), and the token expiry.

    Raises:
        ValueError: If listing installations fails, returns invalid JSON, has no
            installation for *owner*, or the token exchange fails or omits the
            ``token`` field.
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
    """Exchange an installation id for an access token scoped to *repo*.

    Raises:
        ValueError: If the API call fails, the response has missing or invalid
            token data, or its expiry is missing or invalid.
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
    """Exchange a GitHub App-manifest code for the created App's credentials.

    Completing the App-from-manifest flow yields a temporary ``code`` that is
    itself the credential; ``POST /app-manifests/{code}/conversions`` is
    unauthenticated and returns the new App's ``id``, ``pem`` private key, and
    ``slug``.

    Args:
        repo_dir: Working directory for the ``gh`` subprocess.
        code: The temporary manifest-conversion code from the callback.

    Returns:
        A ``(credentials, slug)`` tuple: the App's id/PEM as
        :class:`AppCredentials`, and its ``slug``.

    Raises:
        GitHubAppError: If the conversion call fails, or the response is missing
            an integer ``id`` or a ``pem`` field. The missing field is named and
            never substituted with a placeholder.
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
    """Read the authenticated App's metadata (``permissions``, ``slug``) via ``GET /app``.

    Mints an App JWT, then calls ``GET /app`` with matching explicit static
    auth and an ``Authorization: Bearer`` header (the only scheme GitHub
    accepts for App JWTs). No ambient credential state is changed.

    Args:
        repo_dir: Working directory for the ``gh`` subprocess.
        app_id: Numeric GitHub App ID.
        private_key: PEM-encoded RSA private key for RS256 JWT signing.

    Returns:
        The parsed ``/app`` object, carrying ``permissions`` and ``slug``.

    Raises:
        GitHubAppError: If the call fails or returns a non-object payload.
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
    """Resolve the ambient ``gh``-authenticated user login via ``GET /user``.

    Returns:
        The login string, or the literal ``"unknown"`` if the lookup fails
        for any reason. Identity display is cosmetic and must never abort a
        run, so this function never raises.
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
    """Resolve one run-owned GitHub identity and execution credential.

    Credentials are validated before the read-only decision, preserving
    configuration error behavior. Read-only and no-App paths use live parent
    inheritance unless an explicit complete base environment was supplied.
    Posting with App credentials mints one installation token and returns a
    per-session refreshing auth value; no process-global credential is read or
    changed. When posting and the owner/repo cannot be determined, or minting
    fails, that is a hard abort.

    Args:
        target_dir: Resolved target directory for ``gh repo view`` fallback.
        pr_repo: Optional ``"owner/repo"`` override, preferred when set.
        is_posting: Whether the run posts to GitHub (comments, reviews,
            feedback replies) and therefore requires a scoped token.

    Returns:
        The resolved non-secret identity and separate execution input.

    Raises:
        GitHubAppError: On partial/malformed credentials, undeterminable
            owner/repo while posting, or minting/injection failure.
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
    try:
        minted = _mint_installation_token(
            target_dir,
            credentials.app_id,
            credentials.private_key,
            owner,
            repo,
            base_environment=source_environment,
        )

        def refresh() -> tuple[git_ops.StaticGitHubAuth, float]:
            fresh = _mint_installation_token(
                target_dir,
                credentials.app_id,
                credentials.private_key,
                owner,
                repo,
                base_environment=source_environment,
            )
            return (
                git_ops.StaticGitHubAuth(
                    build_gh_env(
                        fresh.token,
                        base_environment=source_environment,
                    )
                ),
                fresh.expires_at,
            )

        auth = git_ops.RefreshingGitHubAuth(
            git_ops.StaticGitHubAuth(
                build_gh_env(
                    minted.token,
                    base_environment=source_environment,
                )
            ),
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
    """Determine ``(owner, repo)`` for installation-token minting.

    Prefers *pr_repo* (``"owner/repo"``) when set; otherwise derives it from
    ``gh repo view``. Returns None when it cannot be determined.
    """
    if pr_repo:
        parsed = git_ops.split_owner_repo(pr_repo)
        if parsed is not None:
            return parsed
    return git_ops.gh_repo_view(target_dir, auth=auth)
