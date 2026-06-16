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
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import jwt as pyjwt

from daydream import git_ops

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
    private_key: str


def resolve_credentials() -> AppCredentials | None:
    """Resolve GitHub App credentials from the environment.

    Returns:
        ``AppCredentials`` when both env vars are present and valid, or ``None``
        when both are absent (opt-in: no App identity, no behavior change).

    Raises:
        ValueError: If exactly one of the two env vars is present (partial
            misconfiguration; names the missing var), or if ``DAYDREAM_APP_ID``
            is present but not parseable as an integer.
    """
    app_id_raw = os.environ.get(APP_ID_ENV)
    private_key = os.environ.get(APP_PRIVATE_KEY_ENV)

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

    Returns:
        The encoded JWT string.
    """
    iat = int(time.time()) - 60
    payload = {
        "iss": str(app_id),
        "iat": iat,
        "exp": iat + 600,
    }
    return pyjwt.encode(payload, private_key, algorithm="RS256")


def build_gh_env(token: str) -> dict[str, str]:
    """Build a subprocess environment with ``GH_TOKEN`` set.

    Args:
        token: Token to inject as ``GH_TOKEN`` (App JWT or installation token).

    Returns:
        A dict of env-var overrides containing ``GH_TOKEN``.  Merged with the
        live ``os.environ`` at subprocess call time by
        :func:`daydream.git_ops._run_gh`.
    """
    return {"GH_TOKEN": token}


@contextmanager
def _scoped_gh_token(token: str) -> Generator[None, None, None]:
    """Temporarily set the git_ops GH token singleton, restoring it on exit.

    Args:
        token: Token to inject as ``GH_TOKEN`` for the duration of the block.
    """
    prior = git_ops.get_gh_token_env()
    git_ops.set_gh_token_env(build_gh_env(token))
    try:
        yield
    finally:
        git_ops.set_gh_token_env(prior)


def mint_installation_token(repo_dir: Path, app_id: int, private_key: str, owner: str, repo: str) -> tuple[str, str]:
    """Exchange App credentials for a scoped installation access token.

    Mints an App JWT, lists the App's installations to find the one owned by
    *owner*, and exchanges that installation for a short-lived access token.
    Both API calls authenticate with an explicit ``Authorization: Bearer``
    header carrying the JWT (the only scheme GitHub accepts for App JWTs);
    the JWT is also injected as ``GH_TOKEN`` via the ``git_ops`` token
    singleton for the duration of the two calls so ``gh`` runs without
    ambient auth, then the prior singleton value is restored.

    Args:
        repo_dir: Working directory for the ``gh`` subprocesses.
        app_id: Numeric GitHub App ID.
        private_key: PEM-encoded RSA private key for RS256 JWT signing.
        owner: Repository owner (org or user) whose installation to use.
        repo: Repository name the minted token is scoped to.

    Returns:
        A ``(token, identity)`` tuple: the scoped installation access token and
        the App's ``"{slug}[bot]"`` identity from the matched installation's
        ``app_slug`` field, or ``"unknown"`` if the slug is absent (identity is
        cosmetic and never fails the mint).

    Raises:
        ValueError: If listing installations fails, returns invalid JSON, has no
            installation for *owner*, or the token exchange fails or omits the
            ``token`` field.
    """
    jwt_token = mint_jwt(app_id, private_key)
    # GitHub only accepts App JWTs with the Bearer scheme, but gh sends the
    # token scheme for GH_TOKEN. The explicit Bearer header does the real
    # authentication; GH_TOKEN is still set so gh runs without ambient auth
    # (CI) and never falls back to a different identity.
    bearer = {"Authorization": f"Bearer {jwt_token}"}
    with _scoped_gh_token(jwt_token):
        installation_id, identity = _find_installation(repo_dir, owner, repo, bearer)
        return _exchange_for_token(repo_dir, installation_id, owner, repo, bearer), identity


def _find_installation(repo_dir: Path, owner: str, repo: str, headers: dict[str, str]) -> tuple[int, str]:
    """List App installations and return ``(id, "{slug}[bot]")`` for *owner*."""
    try:
        installations = git_ops.gh_api(repo_dir, "/app/installations", paginate=True, jq=".[]", headers=headers)
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


def _exchange_for_token(repo_dir: Path, installation_id: int, owner: str, repo: str, headers: dict[str, str]) -> str:
    """Exchange an installation id for an access token scoped to *repo*."""
    try:
        payload = git_ops.gh_api(
            repo_dir,
            f"/app/installations/{installation_id}/access_tokens",
            method="POST",
            input_data={"repositories": [repo]},
            headers=headers,
        )
    except git_ops.GitError as exc:
        raise ValueError(f"failed to mint installation token for {owner}/{repo}: {exc}") from exc

    token = payload.get("token") if isinstance(payload, dict) else None
    if not isinstance(token, str) or not token:
        raise ValueError(f"installation token response for {owner}/{repo} is missing the 'token' field")
    return token


def exchange_manifest_code(repo_dir: Path, code: str) -> tuple[AppCredentials, str]:
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
        payload = git_ops.gh_api(repo_dir, f"/app-manifests/{code}/conversions", method="POST")
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


def get_app_metadata(repo_dir: Path, app_id: int, private_key: str) -> dict:
    """Read the authenticated App's metadata (``permissions``, ``slug``) via ``GET /app``.

    Mints an App JWT, then calls ``GET /app`` with an explicit
    ``Authorization: Bearer`` header (the only scheme GitHub accepts for App
    JWTs); the JWT is injected as ``GH_TOKEN`` via the ``git_ops`` token
    singleton for the duration of the call and restored afterward.

    Args:
        repo_dir: Working directory for the ``gh`` subprocess.
        app_id: Numeric GitHub App ID.
        private_key: PEM-encoded RSA private key for RS256 JWT signing.

    Returns:
        The parsed ``/app`` object, carrying ``permissions`` and ``slug``.

    Raises:
        GitHubAppError: If the call fails or returns a non-object payload.
    """
    jwt_token = mint_jwt(app_id, private_key)
    bearer = {"Authorization": f"Bearer {jwt_token}"}
    with _scoped_gh_token(jwt_token):
        try:
            payload = git_ops.gh_api(repo_dir, "/app", headers=bearer)
        except git_ops.GitError as exc:
            raise GitHubAppError(f"failed to read App metadata: {exc}") from exc

    if not isinstance(payload, dict):
        raise GitHubAppError("App metadata response is not a JSON object")
    return payload


def resolve_user_identity(repo_dir: Path) -> str:
    """Resolve the ambient ``gh``-authenticated user login via ``GET /user``.

    Args:
        repo_dir: Working directory for the ``gh`` subprocess.

    Returns:
        The login string, or the literal ``"unknown"`` if the lookup fails
        for any reason. Identity display is cosmetic and must never abort a
        run, so this function never raises.
    """
    try:
        login = git_ops.gh_api(repo_dir, "/user").get("login")
    except Exception:  # noqa: BLE001 - identity display is cosmetic; never abort a run
        return "unknown"
    if isinstance(login, str) and login:
        return login
    return "unknown"


def resolve_run_identity(target_dir: Path, pr_repo: str | None, *, is_posting: bool) -> str:
    """Resolve the active GitHub identity for a run, minting App tokens if configured.

    Always clears any previously injected token first, so a run never
    inherits the App identity of an earlier run in the same process.
    Without App credentials, returns the ambient ``gh`` identity. With
    credentials, mints a scoped installation token, injects it into every
    ``gh`` subprocess via the ``git_ops`` token singleton, and returns the
    App identity captured during minting. When the owner/repo cannot be determined and the
    run is not posting, falls back to the ambient identity; when posting,
    that is a hard abort so ``gh`` never silently falls back to ambient auth.

    Args:
        target_dir: Resolved target directory for ``gh repo view`` fallback.
        pr_repo: Optional ``"owner/repo"`` override, preferred when set.
        is_posting: Whether the run posts to GitHub (comments, reviews,
            feedback replies) and therefore requires a scoped token.

    Returns:
        The resolved GitHub login, or ``"unknown"`` when the (cosmetic)
        identity lookup fails.

    Raises:
        GitHubAppError: On partial/malformed credentials, undeterminable
            owner/repo while posting, or minting/injection failure.
    """
    git_ops.reset_gh_token_env()
    try:
        credentials = resolve_credentials()
    except ValueError as exc:
        raise GitHubAppError(str(exc)) from exc
    if credentials is None:
        return resolve_user_identity(target_dir)

    owner_repo = _owner_repo_for(pr_repo, target_dir)
    if owner_repo is None:
        if is_posting:
            raise GitHubAppError("Cannot determine owner/repo for installation token minting")
        return resolve_user_identity(target_dir)

    owner, repo = owner_repo
    try:
        token, identity = mint_installation_token(
            target_dir, credentials.app_id, credentials.private_key, owner, repo
        )
        git_ops.set_gh_token_env(build_gh_env(token))
    except Exception as exc:
        raise GitHubAppError(f"App token resolution failed: {exc}") from exc

    return identity


def _owner_repo_for(pr_repo: str | None, target_dir: Path) -> tuple[str, str] | None:
    """Determine ``(owner, repo)`` for installation-token minting.

    Prefers *pr_repo* (``"owner/repo"``) when set; otherwise derives it from
    ``gh repo view``. Returns None when it cannot be determined.
    """
    if pr_repo:
        parsed = git_ops.split_owner_repo(pr_repo)
        if parsed is not None:
            return parsed
    return git_ops.gh_repo_view(target_dir)
