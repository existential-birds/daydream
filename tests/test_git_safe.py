"""Credential-safe Git URL normalizer (issue #981 M1/M2/M5)."""
from daydream.archive.git_safe import normalize_remote_url


def test_normalize_strips_credentials_and_keeps_identity() -> None:
    for raw in (
        "git@github.com:o/r.git",  # M2: SCP form
        "ssh://git@github.com/o/r.git",  # M5: benign ssh:// URL
        "https://user:ghp_secret@github.com/o/r.git?token=abc",  # M1/M3: userinfo and credential queries drop
    ):
        assert normalize_remote_url(raw) == ("o/r", "https://github.com/o/r"), raw
