"""GitHub request authentication and per-session credential refresh."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from daydream.git_ops.models import GitError

# Refresh before the credential's actual deadline so a request cannot start
# with a token that expires while GitHub is processing it.
_GH_TOKEN_REFRESH_SKEW_SECONDS = 300


class GitHubAuth(Protocol):
    """Resolve the complete environment for one ``gh`` subprocess request."""

    def environment_for_request(self) -> Mapping[str, str] | None: ...


@dataclass(frozen=True)
class InheritGitHubAuth:
    """Request live parent-process environment inheritance from ``gh``."""

    def environment_for_request(self) -> None:
        return None


INHERIT_GITHUB_AUTH = InheritGitHubAuth()

_GITHUB_CREDENTIAL_ENV_KEYS = frozenset(
    {
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GH_ENTERPRISE_TOKEN",
        "GITHUB_ENTERPRISE_TOKEN",
    }
)


@dataclass(frozen=True)
class StaticGitHubAuth:
    """An immutable complete subprocess environment containing a credential."""

    environment: Mapping[str, str] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))

    def environment_for_request(self) -> Mapping[str, str]:
        return dict(self.environment)


class RefreshingGitHubAuth:
    """A per-session credential refreshed under an instance-local lock."""

    def __init__(
        self,
        initial: StaticGitHubAuth,
        *,
        expires_at: float,
        refresh: Callable[[], tuple[StaticGitHubAuth, float]],
    ) -> None:
        self._current = initial
        self._base_environment = {
            name: value for name, value in initial.environment.items() if name not in _GITHUB_CREDENTIAL_ENV_KEYS
        }
        self._expires_at = expires_at
        self._refresh = refresh
        self._lock = threading.Lock()

    def environment_for_request(self) -> Mapping[str, str]:
        if time.time() < self._expires_at - _GH_TOKEN_REFRESH_SKEW_SECONDS:
            return self._current.environment_for_request()

        with self._lock:
            if time.time() < self._expires_at - _GH_TOKEN_REFRESH_SKEW_SECONDS:
                return self._current.environment_for_request()
            try:
                replacement, expires_at = self._refresh()
            except Exception:
                raise GitError("failed to refresh GitHub App installation token") from None
            if not isinstance(replacement, StaticGitHubAuth):
                raise GitError("GitHub App refresh returned invalid authentication")
            replacement_base = {
                name: value
                for name, value in replacement.environment.items()
                if name not in _GITHUB_CREDENTIAL_ENV_KEYS
            }
            if replacement_base != self._base_environment:
                raise GitError("GitHub App refresh changed the bound base environment")
            self._current = replacement
            self._expires_at = expires_at
            return self._current.environment_for_request()
