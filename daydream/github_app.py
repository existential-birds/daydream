"""GitHub App identity: credential resolution, JWT minting, and gh env building.

Daydream can run under an operator-owned GitHub App bot identity. The operator
supplies the App credentials via the ``DAYDREAM_APP_ID`` and
``DAYDREAM_APP_PRIVATE_KEY`` environment variables; this module turns those into
a short-lived RS256 JWT, exchanges that JWT for a scoped installation access
token, and resolves the active GitHub identity for banner display.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import jwt as pyjwt

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
        A copy of the parent environment with ``GH_TOKEN`` set, so ``gh`` still
        inherits ``PATH`` and other parent-env essentials.
    """
    return {**os.environ, "GH_TOKEN": token}
