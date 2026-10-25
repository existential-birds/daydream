"""Credential-safe Git URL normalizer (issue #981 M1/M2/M5)."""
import pytest

from daydream.archive.git_safe import normalize_remote_url


@pytest.mark.parametrize(("raw", "slug", "url_out"),
    [
        # M2: SCP form
        ("git@github.com:o/r.git", "o/r", "https://github.com/o/r"),
        # M5: benign SSH URL (ssh:// scheme)
        ("ssh://git@github.com/o/r.git", "o/r", "https://github.com/o/r"),
    ],
)
def test_normalize_strips_credentials_and_keeps_identity(raw: str, slug: str, url_out: str) -> None:
    identity, url = normalize_remote_url(raw)
    assert identity == slug
    assert url == url_out
    # M3 invariant: no userinfo, no credential query value survives
    assert "@" not in (url or "")
    assert "token=" not in (url or "")

def test_normalize_unparseable_returns_none_identity() -> None:
    identity, _ = normalize_remote_url("not a url at all")
    assert identity is None

def test_normalize_rejects_non_github_host_identity() -> None:
    identity, _ = normalize_remote_url("https://evil.example.com/o/r")
    assert identity is None  # host not in allowlist -> no trusted identity
