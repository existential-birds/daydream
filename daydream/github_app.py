"""GitHub App identity: credential resolution, JWT minting, and gh env building.

Daydream can run under an operator-owned GitHub App bot identity. The operator
supplies the App credentials via the ``DAYDREAM_APP_ID`` and
``DAYDREAM_APP_PRIVATE_KEY`` environment variables; this module turns those into
a short-lived RS256 JWT, exchanges that JWT for a scoped installation access
token, and resolves the active GitHub identity for banner display.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import jwt as pyjwt

from daydream.git_ops import _run_gh

APP_ID_ENV = "DAYDREAM_APP_ID"
APP_PRIVATE_KEY_ENV = "DAYDREAM_APP_PRIVATE_KEY"


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
        live ``os.environ`` at subprocess call time by :func:`run_gh_command`.
    """
    return {"GH_TOKEN": token}


@contextmanager
def _scoped_gh_token(token: str) -> Generator[None, None, None]:
    """Temporarily set the git_ops GH token singleton, restoring it on exit.

    Args:
        token: Token to inject as ``GH_TOKEN`` for the duration of the block.
    """
    from daydream import git_ops

    prior = git_ops.get_gh_token_env()
    git_ops.set_gh_token_env(build_gh_env(token))
    try:
        yield
    finally:
        git_ops.set_gh_token_env(prior)


def mint_installation_token(app_id: int, private_key: str, owner: str, repo: str) -> str:
    """Exchange App credentials for a scoped installation access token.

    Mints an App JWT, lists the App's installations to find the one owned by
    *owner*, and exchanges that installation for a short-lived access token. The
    JWT is injected into the ``gh`` subprocess environment via the ``git_ops``
    token singleton for the duration of the two API calls, then the prior
    singleton value is restored.

    Args:
        app_id: Numeric GitHub App ID.
        private_key: PEM-encoded RSA private key for RS256 JWT signing.
        owner: Repository owner (org or user) whose installation to use.
        repo: Repository name (used only for error context).

    Returns:
        The scoped installation access token string.

    Raises:
        ValueError: If listing installations fails, returns invalid JSON, has no
            installation for *owner*, or the token exchange fails or omits the
            ``token`` field.
    """
    jwt_token = mint_jwt(app_id, private_key)
    with _scoped_gh_token(jwt_token):
        installation_id = _find_installation_id(owner, repo)
        return _exchange_for_token(installation_id, owner, repo)


def _find_installation_id(owner: str, repo: str) -> int:
    """List App installations and return the id owned by *owner*."""
    # --paginate walks all pages; --jq '.[]' flattens each page's array to NDJSON
    # so the combined stdout is one JSON object per line regardless of page count.
    proc = _run_gh(Path("."), ["api", "--paginate", "--jq", ".[]", "/app/installations"])
    if proc.returncode != 0:
        raise ValueError(f"failed to list App installations: {proc.stderr.strip()}")
    try:
        installations = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ValueError(f"App installations list returned invalid JSON: {exc}") from exc

    for entry in installations:
        account = entry.get("account") or {}
        login = account.get("login")
        if isinstance(login, str) and login.lower() == owner.lower():
            installation_id = entry.get("id")
            if not isinstance(installation_id, int):
                raise ValueError(f"installation for {owner!r} is missing an integer id")
            return installation_id
    raise ValueError(f"no App installation found for owner {owner!r} (repo {owner}/{repo})")


def _exchange_for_token(installation_id: int, owner: str, repo: str) -> str:
    """Exchange an installation id for a scoped access token."""
    proc = _run_gh(
        Path("."),
        ["api", "--method", "POST", f"/app/installations/{installation_id}/access_tokens"],
    )
    if proc.returncode != 0:
        raise ValueError(f"failed to mint installation token for {owner}/{repo}: {proc.stderr.strip()}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"installation token response returned invalid JSON: {exc}") from exc

    token = payload.get("token")
    if not isinstance(token, str) or not token:
        raise ValueError(f"installation token response for {owner}/{repo} is missing the 'token' field")
    return token


def resolve_identity(
    token: str | None = None,
    credentials: AppCredentials | None = None,
) -> str:
    """Resolve the active GitHub login for banner display.

    When *credentials* are provided the lookup uses ``GET /app`` (authenticated
    with a freshly minted App JWT) and returns ``"{slug}[bot]"``.  Installation
    tokens cannot access ``GET /user``, so the token path is skipped in that
    case.  When only *token* is provided the lookup runs under that token.
    Otherwise the lookup uses the ambient ``gh`` authentication.

    Args:
        token: Optional installation/access token to authenticate the lookup.
        credentials: Optional App credentials; when present, the App slug is
            fetched via ``GET /app`` instead of ``GET /user``.

    Returns:
        The GitHub login string, or the literal ``"unknown"`` if the lookup
        fails for any reason. Identity display is cosmetic and must never abort
        a run, so this function never raises.
    """
    if credentials is not None:
        return _read_app_slug(credentials.app_id, credentials.private_key)
    if token is not None:
        with _scoped_gh_token(token):
            return _read_user_login()
    return _read_user_login()


def _read_app_slug(app_id: int, private_key: str) -> str:
    """Read ``gh api /app`` slug, returning ``"{slug}[bot]"`` or ``"unknown"`` on failure."""
    try:
        jwt_token = mint_jwt(app_id, private_key)
        with _scoped_gh_token(jwt_token):
            proc = _run_gh(Path("."), ["api", "/app"])
            if proc.returncode != 0:
                return "unknown"
            slug = json.loads(proc.stdout).get("slug")
    except Exception:  # noqa: BLE001 - identity display is cosmetic; never abort a run
        return "unknown"
    if isinstance(slug, str) and slug:
        return f"{slug}[bot]"
    return "unknown"


def _read_user_login() -> str:
    """Read ``gh api /user`` login, returning ``"unknown"`` on any failure."""
    try:
        proc = _run_gh(Path("."), ["api", "/user"])
        if proc.returncode != 0:
            return "unknown"
        login = json.loads(proc.stdout).get("login")
    except Exception:  # noqa: BLE001 - identity display is cosmetic; never abort a run
        return "unknown"
    if isinstance(login, str) and login:
        return login
    return "unknown"
