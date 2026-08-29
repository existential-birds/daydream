"""Credential-safe Git URL normalizer (issue #981).

Sole authority for interpreting Git remote URLs. Parses, classifies, strips
credentials, and reconstructs a canonical HTTPS URL — never serializes
userinfo or credential-like query parameters into any output form.
"""

import re
from urllib.parse import parse_qsl, unquote, urlparse

# Hosts whose owner/repo identity we trust for harvest resolution.
_DEFAULT_HOSTS = frozenset({"github.com"})

# Query-parameter keys that must never survive into any output form.
_CREDENTIAL_QUERY_KEYS = frozenset(
    {"token", "access_token", "key", "secret", "password", "credential"}
)

# The one regex case allowed by the contract: the SCP form, which
# urllib.parse cannot interpret (host:path with no scheme).
_SCP_RE = re.compile(r"^(?:([^@/]+)@)?([^:/]+):(/?.+)$")


def _parse_scp(raw: str) -> tuple[str, str, str] | None:
    """Parse ``git@host:owner/repo(.git)`` -> (user, host, path) or None."""
    match = _SCP_RE.match(raw)
    if match is None:
        return None
    user, host, path = match.groups()
    return user or "", host, path


def _split_path(path: str) -> tuple[str, str] | None:
    """Extract (owner, repo) from a URL path, stripping .git / slashes."""
    trimmed = path.strip("/")
    if trimmed.endswith(".git"):
        trimmed = trimmed[: -len(".git")]
    parts = [p for p in trimmed.split("/") if p]
    if len(parts) < 2:
        return None
    owner, repo = parts[-2], parts[-1]
    if not owner or not repo:
        return None
    return owner, repo


def normalize_remote_url(
    raw: str, *, allowed_hosts: frozenset[str] = _DEFAULT_HOSTS
) -> tuple[str | None, str | None]:
    """Normalize a Git remote URL to (identity, credential-free canonical URL).

    Returns (owner/repo, https://host/owner/repo). Identity and URL are None
    when the input is unparseable; identity alone is None when the host is
    not on the allowlist (URL is still returned, credential-stripped).
    Percent-decodes userinfo before classification; never decodes the path.
    Never raises on malformed input.
    """
    if not raw or not raw.strip():
        return None, None

    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        host = parsed.hostname or ""
        # Percent-decode userinfo before classification (user%40corp, p%40ss).
        path = parsed.path or ""
    else:
        scp = _parse_scp(raw)
        if scp is None:
            return None, None
        _, host, path = scp

    owner_repo = _split_path(path)
    if owner_repo is None:
        return None, None
    owner, repo = owner_repo

    # Canonical form carries no query string and no userinfo: credential-like
    # params would leak, userinfo is stripped, and the identity is the path.
    canonical = f"https://{host.lower()}/{owner}/{repo}"

    if host.lower() not in allowed_hosts:
        return None, canonical
    return f"{owner}/{repo}", canonical


def classify_remote_url(raw: str) -> list[str]:
    """Return triage categories for credential exposure in a remote URL.

    Labels among "userinfo" and "query". Empty list for benign URLs.
    """
    if not raw or not raw.strip():
        return []
    categories: list[str] = []
    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        user = unquote(parsed.username) if parsed.username else ""
        password = unquote(parsed.password) if parsed.password else ""
        # Percent-encoded userinfo (e.g. user%40corp:p%40ss) counts too;
        # token-only user@ is credential-bearing only on http(s) (the
        # x-access-token form); a lone SSH login user (ssh://git@...) is not.
        if password or (user and parsed.scheme.lower() in {"http", "https"}):
            categories.append("userinfo")
        if parsed.query:
            keys = {k.lower() for k, _ in parse_qsl(parsed.query, keep_blank_values=True)}
            if keys & _CREDENTIAL_QUERY_KEYS:
                categories.append("query")
    else:
        scp = _parse_scp(raw)
        # A plain SCP username (git@host:path) is a login, not a credential;
        # only an embedded user:pass slot (colon in the user group) carries one.
        if scp is not None and ":" in scp[0]:
            categories.append("userinfo")
    return categories
