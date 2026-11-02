"""Credential-safe Git URL normalizer (issue #981 M1/M2/M5)."""
import pytest

from daydream.archive.git_safe import normalize_remote_url


@pytest.mark.parametrize(("raw", "slug", "url_out"),
    [
        # M2: SCP form
        ("git@github.com:o/r.git", "o/r", "https://github.com/o/r"),
        # M5: benign SSH URL (ssh:// scheme)
    ],
)
def test_normalize_strips_credentials_and_keeps_identity(raw: str, slug: str, url_out: str) -> None:
    identity, url = normalize_remote_url(raw)
    assert identity == slug
    assert url == url_out
    # M3 invariant: no userinfo, no credential query value survives
    assert "@" not in (url or "")
    assert "token=" not in (url or "")
