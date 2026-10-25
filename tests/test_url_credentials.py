"""Live trajectory redaction must satisfy the archive publication scanner."""

import json
from pathlib import Path

import pytest

from daydream.archive.scan import _scan_file, scan_run_dir
from daydream.backends import ResultEvent, TextEvent, ToolResultEvent, ToolStartEvent
from daydream.trajectory import DaydreamPhase, redact_text
from tests.harness.trajectory import make_recorder


async def test_recorded_url_shapes_pass_publication_scan(tmp_path: Path) -> None:
    """One complete endpoint configuration retains safe context through all recorded surfaces."""
    payload = {
        "registry": "https://opaque-canary-credential@registry.example.com/pkg",
        "uppercase": "HTTPS://user:opaque-canary-credential@registry.example.com/pkg",
        "deploy": "deploy:opaque-canary-credential@host.example.com:/srv/app",
        "database": "postgresql://user:opaque-canary-credential@db.example.com:5432/app",
        "fragment": "https://example.com/pkg?access_token=opaque-canary-credential&ref=main#section",
        "template": 'postgresql://$${DB_USER}:***@$${DB_HOST}:$${DB_PORT}/$${DB_NAME}',
        "public": "https://example.com/pkg?ref=main", "ssh": "git@github.com:org/repo.git", "safe": "preserve",
    }
    expected = {
        "registry": "https://[REDACTED_USER]@registry.example.com/pkg",
        "uppercase": "HTTPS://[REDACTED_USER]:[REDACTED_API_KEY]@registry.example.com/pkg",
        "deploy": "[REDACTED_USER]:[REDACTED_API_KEY]@host.example.com:/srv/app",
        "database": "postgresql://[REDACTED_USER]:[REDACTED_API_KEY]@db.example.com:5432/app",
        "fragment": "https://example.com/pkg?access_token=[REDACTED_CREDENTIAL]&ref=main#section",
        "template": 'postgresql://[REDACTED_USER]:[REDACTED_API_KEY]@$${DB_HOST}:$${DB_PORT}/$${DB_NAME}',
        "public": "https://example.com/pkg?ref=main", "ssh": "git@github.com:org/repo.git", "safe": "preserve",
    }
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            invocation.observe(TextEvent(text=json.dumps(payload)))
            invocation.observe(ToolStartEvent(id="read-1", name="read", input=payload))
            invocation.observe(ToolResultEvent(id="read-1", output=json.dumps(payload), is_error=False))
            invocation.observe(ResultEvent(structured_output=None, continuation=None))
    serialized = recorder.path.read_text()
    step = next(step for step in json.loads(serialized)["steps"] if step["source"] == "agent")
    assert json.loads(step["message"]) == expected
    assert step["tool_calls"][0]["arguments"] == expected
    assert json.loads(step["observation"]["results"][0]["content"]) == expected
    assert redact_text(step["message"]) == step["message"]
    assert "opaque-canary-credential" not in serialized
    assert not scan_run_dir(recorder.path.parent).blocking


@pytest.mark.parametrize("text",
    ["https://user:can'ary@host/path", "https://u'ser:canary@host/path", "deploy:can'ary@host:/path",
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
