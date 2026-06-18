"""REDA-01..06 unit tests for ``daydream.trajectory.Redactor``.

Pattern: positive case (secret present, asserts replacement token in
output and raw value absent), negative case (clean input, asserts no
``[REDACTED_*]`` tokens appear). Plus REDA-04 surface coverage and
REDA-05 fail-safe.
"""

from __future__ import annotations

import pytest

from daydream.atif import ContentPart, Observation, ObservationResult, Step, ToolCall
from daydream.trajectory import Redactor, now_iso


def _user_step(message: str) -> Step:
    """Construct a minimal user Step with *message* (test helper)."""
    return Step(
        step_id=1,
        timestamp=now_iso(),
        source="user",
        message=message,
        extra={"daydream_phase": "review", "daydream_run_flow": "normal"},
    )


def _agent_step(
    message: str = "ok",
    reasoning_content: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    observation: Observation | None = None,
) -> Step:
    """Construct a minimal agent Step (test helper)."""
    return Step(
        step_id=2,
        timestamp=now_iso(),
        source="agent",
        model_name="opus",
        message=message,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        observation=observation,
        extra={"daydream_phase": "review", "daydream_run_flow": "normal"},
    )


# ---- API key patterns (REDA-01) ----


def test_redactor_scrubs_openai_api_key() -> None:
    """REDA-01: sk-* tokens replaced with [REDACTED_API_KEY]."""
    out = Redactor().redact_step(_user_step("token=sk-test-12345abcdef done"))
    assert isinstance(out.message, str)
    assert "sk-test-12345abcdef" not in out.message
    assert "[REDACTED_API_KEY]" in out.message


def test_redactor_scrubs_github_token() -> None:
    """REDA-01: ghp_* tokens replaced with [REDACTED_API_KEY]."""
    out = Redactor().redact_step(_user_step("auth=ghp_test123abcdef bearer"))
    assert isinstance(out.message, str)
    assert "ghp_test123abcdef" not in out.message
    assert "[REDACTED_API_KEY]" in out.message


def test_redactor_scrubs_slack_bot_token() -> None:
    """REDA-01: xoxb-* tokens replaced with [REDACTED_API_KEY]."""
    out = Redactor().redact_step(_user_step("slack=xoxb-test456abcdef"))
    assert isinstance(out.message, str)
    assert "xoxb-test456abcdef" not in out.message
    assert "[REDACTED_API_KEY]" in out.message


def test_redactor_scrubs_aws_access_key() -> None:
    """REDA-01: AKIA-prefixed AWS keys replaced with [REDACTED_API_KEY]."""
    out = Redactor().redact_step(_user_step("aws=AKIA0000TESTKEY00000"))
    assert isinstance(out.message, str)
    assert "AKIA0000TESTKEY00000" not in out.message
    assert "[REDACTED_API_KEY]" in out.message


# ---- JWT pattern (REDA-01) ----


def test_redactor_scrubs_jwt_token() -> None:
    """REDA-01: eyJ JWT tokens replaced with [REDACTED_JWT]."""
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.aBcDeF12345"
    out = Redactor().redact_step(_user_step(f"bearer {jwt}"))
    assert isinstance(out.message, str)
    assert jwt not in out.message
    assert "[REDACTED_JWT]" in out.message


# ---- JWT negative case (TEST-03 gap fill) ----


def test_redactor_preserves_short_eyj_non_jwt() -> None:
    """TEST-03 negative: short eyJ-prefixed strings without JWT dot structure pass through."""
    out = Redactor().redact_step(_user_step("eyJhbG is a prefix"))
    assert isinstance(out.message, str)
    assert "[REDACTED_JWT]" not in out.message
    assert "eyJhbG" in out.message


# ---- Message surface (TEST-03 gap fill: explicit Step.message redaction) ----


def test_redactor_applies_to_step_message_surface() -> None:
    """TEST-03 surface: secrets in Step.message are redacted (explicit surface test)."""
    step = _agent_step(message="key is ghp_ABCDEF1234567890abcdef1234567890abcdef")
    out = Redactor().redact_step(step)
    assert isinstance(out.message, str)
    assert "ghp_ABCDEF1234567890abcdef1234567890abcdef" not in out.message
    assert "[REDACTED_API_KEY]" in out.message


# ---- Git URL credentials (REDA-01: explicit "git remote URLs with embedded credentials") ----


def test_redactor_scrubs_git_url_credentials() -> None:
    """REDA-01: https://user:token@host credentials are scrubbed; host+path preserved."""
    url = "git+https://oauth2:ghp_realtoken123@github.com/user/repo.git"
    out = Redactor().redact_step(_user_step(url))
    assert isinstance(out.message, str)
    assert "ghp_realtoken123" not in out.message
    assert "oauth2" not in out.message
    assert "[REDACTED_USER]" in out.message
    assert "[REDACTED_API_KEY]" in out.message
    # Host and path preserved (debugging/replay value).
    assert "github.com" in out.message
    assert "/user/repo.git" in out.message


# ---- Username path patterns (REDA-02) ----


def test_redactor_scrubs_macos_username_path() -> None:
    """REDA-02: /Users/<name>/ → /Users/[REDACTED_USER]/ (project-relative tail preserved)."""
    out = Redactor().redact_step(_user_step("path=/Users/ka/github/proj/app.py"))
    assert isinstance(out.message, str)
    assert "/Users/ka" not in out.message
    assert "/Users/[REDACTED_USER]" in out.message
    assert "github/proj/app.py" in out.message


def test_redactor_scrubs_linux_username_path() -> None:
    """REDA-02: /home/<name>/ → /home/[REDACTED_USER]/."""
    out = Redactor().redact_step(_user_step("path=/home/alice/foo/bar"))
    assert isinstance(out.message, str)
    assert "/home/alice" not in out.message
    assert "/home/[REDACTED_USER]" in out.message
    assert "foo/bar" in out.message


def test_redactor_scrubs_windows_username_path() -> None:
    """REDA-02: C:\\Users\\<name>\\ → C:\\Users\\[REDACTED_USER]\\."""
    out = Redactor().redact_step(_user_step("path=C:\\Users\\bob\\repo"))
    assert isinstance(out.message, str)
    assert "Users\\bob" not in out.message
    assert "[REDACTED_USER]" in out.message


# ---- Env-var pattern (REDA-03) ----


def test_redactor_scrubs_env_var_secret() -> None:
    """REDA-03: KEY=value secret-keyname env vars get value redacted, key preserved."""
    out = Redactor().redact_step(_user_step("OPENAI_API_KEY=sk-realvalue123"))
    assert isinstance(out.message, str)
    assert "sk-realvalue123" not in out.message
    assert "OPENAI_API_KEY=[REDACTED_ENV_VAR]" in out.message


def test_redactor_scrubs_env_var_password() -> None:
    """REDA-03: PASSWORD= env vars get value redacted, key preserved."""
    out = Redactor().redact_step(_user_step("DB_PASSWORD=hunter2"))
    assert isinstance(out.message, str)
    assert "hunter2" not in out.message
    assert "DB_PASSWORD=[REDACTED_ENV_VAR]" in out.message


# ---- Negative cases ----


def test_redactor_preserves_non_secret_env_vars() -> None:
    """REDA-03 negative: DEBUG=true and APP_NAME=foo pass through unredacted."""
    out = Redactor().redact_step(_user_step("DEBUG=true\nAPP_NAME=myproject"))
    assert isinstance(out.message, str)
    assert "DEBUG=true" in out.message
    assert "APP_NAME=myproject" in out.message
    assert "[REDACTED" not in out.message


def test_redactor_preserves_clean_paths() -> None:
    """Negative: relative paths without /Users//home/ prefix pass through."""
    out = Redactor().redact_step(_user_step("./src/app.py"))
    assert isinstance(out.message, str)
    assert out.message == "./src/app.py"


def test_redactor_preserves_clean_urls() -> None:
    """Negative: URLs without embedded credentials pass through unchanged."""
    out = Redactor().redact_step(_user_step("https://github.com/user/repo"))
    assert isinstance(out.message, str)
    assert out.message == "https://github.com/user/repo"


# ---- Surface coverage (REDA-04) ----


def test_redactor_applies_to_reasoning_content() -> None:
    """REDA-04: secrets inside Step.reasoning_content are redacted."""
    step = _agent_step(reasoning_content="thought: sk-test-secret123abc")
    out = Redactor().redact_step(step)
    assert out.reasoning_content is not None
    assert "sk-test-secret123abc" not in out.reasoning_content
    assert "[REDACTED_API_KEY]" in out.reasoning_content


def test_redactor_applies_to_tool_call_arguments() -> None:
    """REDA-04: secrets inside ToolCall.arguments values are redacted."""
    call = ToolCall(
        tool_call_id="t1",
        function_name="Bash",
        arguments={"command": "echo sk-test-secret123abc"},
    )
    step = _agent_step(tool_calls=[call])
    out = Redactor().redact_step(step)
    assert out.tool_calls is not None
    args_str = str(out.tool_calls[0].arguments)
    assert "sk-test-secret123abc" not in args_str
    assert "[REDACTED_API_KEY]" in args_str


def test_redactor_applies_to_observation_content() -> None:
    """REDA-04: secrets inside ObservationResult.content are redacted."""
    obs = Observation(
        results=[ObservationResult(source_call_id="t1", content="leaked /Users/ka/.ssh/id_rsa")],
    )
    step = _agent_step(observation=obs)
    out = Redactor().redact_step(step)
    assert out.observation is not None
    first_content = out.observation.results[0].content
    assert isinstance(first_content, str)
    assert "/Users/ka" not in first_content
    assert "[REDACTED_USER]" in first_content


# ---- Fail-safe (REDA-05) ----


def test_redactor_failure_mode_replaces_with_redaction_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REDA-05: when an internal regex raises, the field becomes [REDACTION_FAILED] — never raw."""
    from daydream import trajectory as traj_mod

    class _BoomPattern:
        def sub(self, *_args: object, **_kwargs: object) -> str:
            raise RuntimeError("boom")

    # Replace the first rule so _redact_text raises on the first call.
    original_rules = traj_mod._REDACTION_RULES
    boom_rules = ((_BoomPattern(), "[REDACTED_API_KEY]"), *original_rules[1:])
    monkeypatch.setattr(traj_mod, "_REDACTION_RULES", boom_rules)

    out = Redactor().redact_step(_user_step("OPENAI_API_KEY=sk-leakthis123"))
    assert isinstance(out.message, str)
    assert "sk-leakthis123" not in out.message
    assert "[REDACTION_FAILED]" in out.message


# ---- Regression: CR-01 (inverted ternary preserved JSON-string instead of dict) ----


def test_redact_arguments_preserves_dict_structure_for_nested_values() -> None:
    """CR-01 regression: non-string nested values must round-trip through json, not stay as JSON string.

    The Edit tool's `arguments` is `{"old_string": "...", "new_string": "..."}`,
    which is a dict whose values are strings — the string branch handles those.
    But MultiEdit's `arguments` is `{"edits": [{...}, {...}]}` where the value
    is a list of dicts, and MCP tool envelopes can have arbitrary nested
    structure. Those must round-trip through json.dumps + redact + json.loads
    so ToolCall.arguments keeps its declared dict[str, Any] shape.

    Pre-fix bug: the inverted ternary stored the redacted JSON-encoded string
    (rather than the parsed Python structure) whenever redaction changed any
    text — corrupting every MultiEdit / nested MCP arguments value.
    """
    # Path triggers redaction so we exercise the redaction-changed-the-text branch (the bug only fires then).
    nested_value = [
        {"path": "/Users/alice/repo/file.py", "edit_type": "replace"},
        {"path": "/Users/alice/repo/other.py", "edit_type": "delete"},
    ]
    arguments = {"edits": nested_value}
    out = Redactor()._redact_arguments(arguments)

    # out["edits"] must stay a list of dicts, NOT a JSON-encoded string (CR-01 stored the string).
    assert isinstance(out["edits"], list), (
        f"Expected list, got {type(out['edits']).__name__}: {out['edits']!r}"
    )
    assert len(out["edits"]) == 2
    assert all(isinstance(item, dict) for item in out["edits"])
    serialized_back = str(out["edits"])
    assert "alice" not in serialized_back
    assert "[REDACTED_USER]" in serialized_back


def test_redact_arguments_passthrough_when_no_secret() -> None:
    """Non-string values without secrets must round-trip cleanly to the same Python structure."""
    arguments = {"count": 42, "flags": ["a", "b"], "config": {"x": 1, "y": [1, 2, 3]}}
    out = Redactor()._redact_arguments(arguments)
    assert out["count"] == 42
    assert out["flags"] == ["a", "b"]
    assert out["config"] == {"x": 1, "y": [1, 2, 3]}


def test_redact_arguments_invalid_json_falls_back_to_redaction_failed() -> None:
    """When regex breaks JSON syntax, value falls back to [REDACTION_FAILED]."""

    class _PoisonPattern:
        def sub(self, _repl: object, s: str) -> str:
            return "{not valid json"

    from daydream import trajectory as traj_mod

    original_rules = traj_mod._REDACTION_RULES
    poison_rules = ((_PoisonPattern(), "ignored"), *original_rules[:0])
    try:
        traj_mod._REDACTION_RULES = poison_rules
        out = Redactor()._redact_arguments({"edits": [{"x": 1}]})
        assert out["edits"] == "[REDACTION_FAILED]"
    finally:
        traj_mod._REDACTION_RULES = original_rules


# ---- Regression: WR-03 (env-var pattern over-redacted via substring match) ----


@pytest.mark.parametrize(
    "non_secret",
    [
        "MONKEY_PATCH=enabled",
        "KEYBOARD_LAYOUT=qwerty",
        "AUTHOR=alice",
        "TOKENIZED=foo",
        "KEYSTORE=path/to/store",
    ],
)
def test_env_var_pattern_does_not_match_substring_lookalikes(non_secret: str) -> None:
    """WR-03 regression: env-var redaction is segment-aware, not substring-based.

    The original pattern matched any var name containing KEY/SECRET/TOKEN/AUTH
    as a substring. The fix requires the secret keyword to appear as a full
    underscore-separated segment.
    """
    out = Redactor().redact_step(_user_step(non_secret))
    assert isinstance(out.message, str)
    assert "[REDACTED_ENV_VAR]" not in out.message
    name, _, value = non_secret.partition("=")
    assert value in out.message, f"Expected {value!r} preserved in {out.message!r}"


@pytest.mark.parametrize(
    "secret",
    [
        ("OPENAI_API_KEY=sk-leakthis", "sk-leakthis"),
        ("MY_API_KEY=value", "value"),
        ("JWT_TOKEN=abc.def.ghi", "abc.def.ghi"),
        ("DB_PASSWORD=hunter2", "hunter2"),
        ("AUTH_TOKEN=t-foo", "t-foo"),
        ("CACHE_KEY=k-bar", "k-bar"),
        ("DB_CREDENTIAL=admin:pw", "admin:pw"),
    ],
)
def test_env_var_pattern_redacts_legitimate_secret_segments(
    secret: tuple[str, str],
) -> None:
    """WR-03 regression: real secret env vars still redact correctly."""
    line, raw_value = secret
    out = Redactor().redact_step(_user_step(line))
    assert isinstance(out.message, str)
    assert raw_value not in out.message
    assert "[REDACTED_ENV_VAR]" in out.message


# ---- Regression: WR-04 (top-level fallback wiped only message, leaked others) ----


def test_redactor_failure_mode_wipes_all_text_bearing_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WR-04 regression: top-level fallback wipes message, reasoning_content, tool_calls, observation.

    Triggers the outer ``except`` by patching ``Redactor._redact_optional_text``
    to raise. Without the fix, only ``message`` is wiped and the remaining
    text-bearing fields pass through unredacted — secrets in
    ``reasoning_content`` / ``tool_calls.arguments`` / ``observation.results``
    leak.
    """
    def _boom(self: object, value: object) -> str:
        raise RuntimeError("simulated regex failure deep in pipeline")

    monkeypatch.setattr(Redactor, "_redact_optional_text", _boom, raising=True)

    step = _agent_step(
        message="OPENAI_API_KEY=sk-leak1",
        reasoning_content="thinking about sk-leak2",
        tool_calls=[
            ToolCall(
                tool_call_id="tc1",
                function_name="Edit",
                arguments={"old_string": "sk-leak3", "new_string": "x"},
            ),
        ],
        observation=Observation(
            results=[ObservationResult(content="result with sk-leak4")],
        ),
    )

    out = Redactor().redact_step(step)

    serialized = str(out.model_dump())
    for leak in ("sk-leak1", "sk-leak2", "sk-leak3", "sk-leak4"):
        assert leak not in serialized, f"{leak} leaked through fallback: {serialized!r}"
    assert out.message == "[REDACTION_FAILED]"
    assert out.reasoning_content == "[REDACTION_FAILED]"
    assert out.tool_calls is not None
    assert out.tool_calls[0].arguments == {"_redaction": "[REDACTION_FAILED]"}
    assert out.observation is not None
    assert out.observation.results[0].content == "[REDACTION_FAILED]"


# ---- Multimodal message (list[ContentPart]) ----


def test_redactor_scrubs_text_content_parts() -> None:
    """Text parts in a multimodal message must be redacted; image parts left intact."""
    from daydream.atif.models.content import ImageSource

    parts = [
        ContentPart(type="text", text="key=sk-test-secret123abc"),
        ContentPart(
            type="image",
            source=ImageSource(media_type="image/png", path="screenshot.png"),
        ),
        ContentPart(type="text", text="clean text"),
    ]
    step = Step(
        step_id=1,
        timestamp=now_iso(),
        source="user",
        message=parts,
        extra={"daydream_phase": "review", "daydream_run_flow": "normal"},
    )
    out = Redactor().redact_step(step)
    assert isinstance(out.message, list)
    assert len(out.message) == 3
    assert out.message[0].type == "text"
    first_text = out.message[0].text
    assert first_text is not None  # type='text' guarantees text is populated
    assert "sk-test-secret123abc" not in first_text
    assert "[REDACTED_API_KEY]" in first_text
    assert out.message[1].type == "image"
    image_source = out.message[1].source
    assert image_source is not None  # type='image' guarantees source is populated
    assert image_source.path == "screenshot.png"
    assert out.message[2].type == "text"
    assert out.message[2].text == "clean text"


def test_redactor_scrubs_github_installation_token() -> None:
    """ghs_* installation tokens replaced with [REDACTED_API_KEY]."""
    out = Redactor().redact_step(_user_step("token=ghs_abc123DEF456ghi789jkl012 done"))
    assert isinstance(out.message, str)
    assert "ghs_abc123DEF456ghi789jkl012" not in out.message
    assert "[REDACTED_API_KEY]" in out.message


def test_redactor_scrubs_pkcs1_private_key_block() -> None:
    """-----BEGIN RSA PRIVATE KEY----- blocks replaced with [REDACTED_PEM_KEY]."""
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF8PbnGPgjF\n"
        "-----END RSA PRIVATE KEY-----"
    )
    out = Redactor().redact_step(_user_step(f"key is {pem} ok"))
    assert isinstance(out.message, str)
    assert "MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn" not in out.message
    assert "[REDACTED_PEM_KEY]" in out.message


def test_redactor_scrubs_pkcs8_private_key_block() -> None:
    """-----BEGIN PRIVATE KEY----- (PKCS8) blocks also redacted."""
    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvgIBADANBgkqhkiG9w0BAQEFAASC\n"
        "-----END PRIVATE KEY-----"
    )
    out = Redactor().redact_step(_user_step(f"key: {pem}"))
    assert isinstance(out.message, str)
    assert "MIIEvgIBADANBgkqhkiG9w0BAQEFAASC" not in out.message
    assert "[REDACTED_PEM_KEY]" in out.message


def test_redactor_scrubs_pkcs1_private_key_in_env_assignment() -> None:
    """VAR=<PKCS1 PEM> redacts fully — PEM rule must run before the env-var rule."""
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEA0Z3VSecretBody1234567890abcdef\n"
        "-----END RSA PRIVATE KEY-----"
    )
    out = Redactor().redact_step(_user_step(f"DAYDREAM_APP_PRIVATE_KEY={pem}"))
    assert isinstance(out.message, str)
    assert "MIIEpAIBAAKCAQEA0Z3VSecretBody" not in out.message
    assert out.message == "DAYDREAM_APP_PRIVATE_KEY=[REDACTED_ENV_VAR]"


def test_redactor_scrubs_pkcs8_private_key_in_env_assignment() -> None:
    """VAR=<PKCS8 PEM> redacts fully — no base64 body survives the assignment form."""
    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvgIBADANBgkqhkiG9w0BAQEFAASC\n"
        "-----END PRIVATE KEY-----"
    )
    out = Redactor().redact_step(_user_step(f"DAYDREAM_APP_PRIVATE_KEY={pem}"))
    assert isinstance(out.message, str)
    assert "MIIEvgIBADANBgkqhkiG9w0BAQEFAASC" not in out.message
    assert out.message == "DAYDREAM_APP_PRIVATE_KEY=[REDACTED_ENV_VAR]"


def test_redactor_preserves_certificate_block() -> None:
    """BEGIN CERTIFICATE blocks are public material — not redacted."""
    cert = (
        "-----BEGIN CERTIFICATE-----\n"
        "MIIDdzCCAl+gAwIBAgIEAgAAuTANBgkq\n"
        "-----END CERTIFICATE-----"
    )
    out = Redactor().redact_step(_user_step(f"cert: {cert}"))
    assert isinstance(out.message, str)
    assert "[REDACTED_PEM_KEY]" not in out.message
    assert "BEGIN CERTIFICATE" in out.message
