"""Live trajectory redaction must satisfy the archive publication scanner."""

import json
from pathlib import Path

import pytest

from daydream.archive.scan import _scan_file, scan_run_dir
from daydream.backends import ResultEvent, TextEvent, ToolResultEvent, ToolStartEvent
from daydream.trajectory import DaydreamPhase, redact_text
from tests.harness.trajectory import make_recorder


@pytest.mark.parametrize(
    "text",
    [
        "https://opaque-canary-credential@registry.example.com/pkg",
        "HTTPS://user:opaque-canary-credential@registry.example.com/pkg",
        "deploy:opaque-canary-credential@host.example.com:/srv/app",
        "postgresql://user:opaque-canary-credential@db.example.com:5432/app",
        "https://example.com/pkg?access_token=opaque-canary-credential&ref=main",
        # Shelfspace Makefile shape: Make interpolation plus an already masked
        # password was blocked by the SCP rule after live redaction missed it.
        'postgresql://$${DB_USER}:***@$${DB_HOST}:$${DB_PORT}/$${DB_NAME}',
    ],
)
async def test_recorded_url_shapes_pass_publication_scan(tmp_path: Path, text: str) -> None:
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            invocation.observe(TextEvent(text=text))
            invocation.observe(ToolStartEvent(id="read-1", name="read", input={"path": text}))
            invocation.observe(ToolResultEvent(id="read-1", output=text, is_error=False))
            invocation.observe(ResultEvent(structured_output=None, continuation=None))

    serialized = recorder.path.read_text()
    assert "opaque-canary-credential" not in serialized
    assert not scan_run_dir(recorder.path.parent).blocking


def test_url_redaction_is_stable_and_preserves_safe_url_context() -> None:
    text = "https://user:opaque-canary-credential@example.com/pkg?token=another-canary&ref=main"
    redacted = redact_text(text)
    assert "opaque-canary-credential" not in redacted
    assert "another-canary" not in redacted
    assert "example.com/pkg" in redacted
    assert "&ref=main" in redacted
    assert redact_text(redacted) == redacted


def test_url_redaction_preserves_public_urls_and_ssh_logins() -> None:
    text = "https://example.com/pkg?ref=main git@github.com:org/repo.git"
    assert redact_text(text) == text


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com?token=opaque-canary-credential#section",
        "deploy:opaque-canary-credential@host:/app",
        "https://user:opaque-canary-credential@host/app",
    ],
)
def test_url_redaction_preserves_serialized_json(url: str) -> None:
    text = json.dumps({"url": url, "safe": "preserve"}, separators=(",", ":"))
    redacted = redact_text(text)
    parsed = json.loads(redacted)
    assert parsed["safe"] == "preserve"
    assert "opaque-canary-credential" not in parsed["url"]
    if "#section" in url:
        assert parsed["url"].endswith("#section")


@pytest.mark.parametrize(
    "text",
    [
        "https://user:can'ary@host/path",
        "https://u'ser:canary@host/path",
        "deploy:can'ary@host:/path",
        "https://example.com?token=can'ary&ref=main",
    ],
)
def test_apostrophes_in_credentials_remain_blocking_and_are_fully_redacted(text: str) -> None:
    # Apostrophes are legal RFC 3986 sub-delimiters, including in userinfo.
    assert any(f.severity == "blocking" for f in _scan_file("sample.txt", text))
    redacted = redact_text(text)
    assert "canary" not in redacted
    assert "can'ary" not in redacted
    assert "ary" not in redacted
    assert not _scan_file("sample.txt", redacted)
