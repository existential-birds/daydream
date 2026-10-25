"""Redaction positives, clean-input negatives, field coverage, and fail-closed behavior."""

from __future__ import annotations

import copy
import json
import time

import pytest

from daydream import redaction as redaction_mod
from daydream.atif import ContentPart, Observation, ObservationResult, Step, ToolCall
from daydream.atif.models.content import ImageSource
from daydream.trajectory import Redactor, now_iso, redact_structured_text, redact_value, redactor as redactor_mod


def _user_step(message: str) -> Step:
    """Construct a minimal user Step with *message* (test helper)."""
    return Step(step_id=1, timestamp=now_iso(), source="user", message=message,
        extra={"daydream_phase": "review", "daydream_run_flow": "normal"},
    )


def _agent_step(message: str = "ok", reasoning_content: str | None = None, tool_calls: list[ToolCall] | None = None,
    observation: Observation | None = None,
) -> Step:
    """Construct a minimal agent Step (test helper)."""
    return Step(step_id=2, timestamp=now_iso(), source="agent", model_name="opus", message=message,
        reasoning_content=reasoning_content, tool_calls=tool_calls, observation=observation,
        extra={"daydream_phase": "review", "daydream_run_flow": "normal"},
    )


_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.aBcDeF12345"


def test_redactor_sanitizes_provider_credentials_in_one_recorded_message() -> None:
    """Every token family is scrubbed independently while the surrounding diagnostic survives."""
    text = (
        "OpenAI sk-test-12345abcdef; GitHub ghp_test123abcdef; Slack xoxb-test456abcdef; "
        "AWS AKIA0000TESTKEY00000; installation ghs_abc123DEF456ghi789jkl012; "
        f"bearer {_JWT}; diagnostic complete."
    )
    out = Redactor().redact_step(_user_step(text))
    assert out.message == (
        "OpenAI [REDACTED_API_KEY]; GitHub [REDACTED_API_KEY]; Slack [REDACTED_API_KEY]; "
        "AWS [REDACTED_API_KEY]; installation [REDACTED_API_KEY]; bearer [REDACTED_JWT]; diagnostic complete."
    )


def test_redactor_sanitizes_environment_diagnostic_without_consuming_lookalikes() -> None:
    """A complete environment dump distinguishes secret segments from harmless substrings."""
    text = (
        "OPENAI_API_KEY=sk-leakthis\nMY_API_KEY=value\nJWT_TOKEN=abc.def.ghi\nDB_PASSWORD=hunter2\n"
        "AUTH_TOKEN=t-foo\nCACHE_KEY=k-bar\nDB_CREDENTIAL=admin:pw\nDEBUG=true\nAPP_NAME=myproject\n"
        "MONKEY_PATCH=enabled\nKEYBOARD_LAYOUT=qwerty\nAUTHOR=alice\nTOKENIZED=foo\nKEYSTORE=path/to/store"
    )
    out = Redactor().redact_step(_user_step(text))
    assert out.message == (
        "OPENAI_API_KEY=[REDACTED_ENV_VAR]\nMY_API_KEY=[REDACTED_ENV_VAR]\nJWT_TOKEN=[REDACTED_ENV_VAR]\n"
        "DB_PASSWORD=[REDACTED_ENV_VAR]\nAUTH_TOKEN=[REDACTED_ENV_VAR]\nCACHE_KEY=[REDACTED_ENV_VAR]\n"
        "DB_CREDENTIAL=[REDACTED_ENV_VAR]\nDEBUG=true\nAPP_NAME=myproject\nMONKEY_PATCH=enabled\n"
        "KEYBOARD_LAYOUT=qwerty\nAUTHOR=alice\nTOKENIZED=foo\nKEYSTORE=path/to/store"
    )


def test_redactor_sanitizes_http_observation_and_preserves_its_layout() -> None:
    """Header alternatives and embedded bearer tokens must all survive traversal into observations."""
    content = (
        "Authorization: opaque-test-only-sentinel\nauthorization: Bearer opaque-test-only-sentinel\n"
        "Proxy-Authorization: opaque-test-only-sentinel\nX-Api-Key: opaque-test-only-sentinel\n"
        "X-Auth-Token: opaque-test-only-sentinel\nCookie: session=opaque-test-only-sentinel\n"
        "Set-Cookie: opaque-test-only-sentinel\n  authorization: Bearer opaque-token-xyz\n"
        'curl -H "Authorization: Bearer opaque-token-xyz" https://api\n'
        "saw Authorization: Bearer opaque-token-xyz in the log\nnext line stays"
    )
    out = Redactor().redact_step(_agent_step(observation=Observation(results=[ObservationResult(content=content)])))
    assert out.observation is not None
    assert out.observation.results[0].content == (
        "Authorization: [REDACTED_CREDENTIAL]\nauthorization: Bearer [REDACTED_CREDENTIAL]\n"
        "Proxy-Authorization: [REDACTED_CREDENTIAL]\nX-Api-Key: [REDACTED_CREDENTIAL]\n"
        "X-Auth-Token: [REDACTED_CREDENTIAL]\nCookie: [REDACTED_CREDENTIAL]\nSet-Cookie: [REDACTED_CREDENTIAL]\n"
        "  authorization: Bearer [REDACTED_CREDENTIAL]\n"
        'curl -H "Authorization: Bearer [REDACTED_CREDENTIAL]" https://api\n'
        "saw Authorization: Bearer [REDACTED_CREDENTIAL] in the log\nnext line stays"
    )


def test_redactor_preserves_complete_tool_arguments_with_harmless_key_names() -> None:
    """Harmless key substrings cannot erase ordinary structured tool data."""
    arguments = {
        "tokenizer": "opaque-test-only-sentinel",
        "passwordless": "opaque-test-only-sentinel",
        "monkeyPatch": "opaque-test-only-sentinel",
        "keyStore": "opaque-test-only-sentinel",
        "max_tokens": "opaque-test-only-sentinel",
        "config": {"count": 42, "flags": ["a", "b"]},
    }
    out = Redactor().redact_step(
        _agent_step(
            tool_calls=[
                ToolCall(tool_call_id="t1", function_name="Bash", arguments=arguments),
            ]
        )
    )
    assert out.tool_calls is not None
    assert out.tool_calls[0].arguments == arguments


def test_redactor_preserves_short_eyj_non_jwt() -> None:
    out = Redactor().redact_step(_user_step("eyJhbG is a prefix"))
    assert isinstance(out.message, str)
    assert "[REDACTED_JWT]" not in out.message
    assert "eyJhbG" in out.message


@pytest.mark.parametrize(("text", "absent", "present", "preserved_tail"),
    [("path=/Users/ka/github/proj/app.py", "/Users/ka", "/Users/[REDACTED_USER]", "github/proj/app.py"),
        ("path=/home/alice/foo/bar", "/home/alice", "/home/[REDACTED_USER]", "foo/bar"),
        ("path=C:\\Users\\bob\\repo", "Users\\bob", "[REDACTED_USER]", None),
    ], ids=["macos", "linux", "windows"],
)
def test_redactor_scrubs_username_path(text: str, absent: str, present: str, preserved_tail: str | None) -> None:
    """REDA-02: /Users//home//C:\\Users\\ <name> → [REDACTED_USER], project-relative tail preserved."""
    out = Redactor().redact_step(_user_step(text))
    assert isinstance(out.message, str)
    assert absent not in out.message
    assert present in out.message
    if preserved_tail is not None:
        assert preserved_tail in out.message


def test_redactor_applies_to_reasoning_content() -> None:
    step = _agent_step(reasoning_content="thought: sk-test-secret123abc")
    out = Redactor().redact_step(step)
    assert out.reasoning_content is not None
    assert "sk-test-secret123abc" not in out.reasoning_content
    assert "[REDACTED_API_KEY]" in out.reasoning_content


def test_redactor_failure_mode_replaces_with_redaction_failed(monkeypatch: pytest.MonkeyPatch,) -> None:
    class _BoomPattern:
        def sub(self, *_args: object, **_kwargs: object) -> str:
            raise RuntimeError("boom")

    # Replace the first rule so redact_structured_text raises on the first call.
    original_rules = redaction_mod._REDACTION_RULES
    boom_rules = ((_BoomPattern(), "[REDACTED_API_KEY]"), *original_rules[1:])
    monkeypatch.setattr(redaction_mod, "_REDACTION_RULES", boom_rules)

    out = Redactor().redact_step(_user_step("OPENAI_API_KEY=sk-leakthis123"))
    assert isinstance(out.message, str)
    assert "sk-leakthis123" not in out.message
    assert "[REDACTION_FAILED]" in out.message


def test_redact_arguments_preserves_dict_structure_for_nested_values() -> None:
    # Path triggers redaction so we exercise the redaction-changed-the-text branch (the bug only fires then).
    nested_value = [{"path": "/Users/alice/repo/file.py", "edit_type": "replace"},
        {"path": "/Users/alice/repo/other.py", "edit_type": "delete"},
    ]
    arguments = {"edits": nested_value}
    out = Redactor()._redact_arguments(arguments)

    # out["edits"] must stay a list of dicts, NOT a JSON-encoded string (CR-01 stored the string).
    assert isinstance(out["edits"], list), (f"Expected list, got {type(out['edits']).__name__}: {out['edits']!r}")
    assert len(out["edits"]) == 2
    assert all(isinstance(item, dict) for item in out["edits"])
    serialized_back = str(out["edits"])
    assert "alice" not in serialized_back
    assert "[REDACTED_USER]" in serialized_back


def test_redact_arguments_recursive_failure_falls_back_to_redaction_failed(monkeypatch: pytest.MonkeyPatch,) -> None:
    def _boom(value: object, sensitive: bool = False) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(redactor_mod, "redact_value", _boom, raising=True)
    out = Redactor()._redact_arguments({"edits": [{"x": 1}]})
    assert out["edits"] == "[REDACTION_FAILED]"


def test_redactor_failure_mode_wipes_all_text_bearing_fields(monkeypatch: pytest.MonkeyPatch,) -> None:
    def _boom(self: object, value: object) -> str:
        raise RuntimeError("simulated regex failure deep in pipeline")

    monkeypatch.setattr(Redactor, "_redact_optional_text", _boom, raising=True)

    step = _agent_step(message="OPENAI_API_KEY=sk-leak1", reasoning_content="thinking about sk-leak2",
        tool_calls=[ToolCall(
                tool_call_id="tc1", function_name="Edit", arguments={"old_string": "sk-leak3", "new_string": "x"},
            ),
        ], observation=Observation(results=[ObservationResult(content="result with sk-leak4")],),
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


def test_redactor_scrubs_text_content_parts() -> None:
    parts = [ContentPart(type="text", text="key=sk-test-secret123abc"),
        ContentPart(type="image", source=ImageSource(media_type="image/png", path="screenshot.png"),),
        ContentPart(type="text", text="clean text"),
    ]
    step = Step(step_id=1, timestamp=now_iso(), source="user", message=parts,
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


_PKCS1_PEM = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF8PbnGPgjF\n"
    "-----END RSA PRIVATE KEY-----"
)
_PKCS8_PEM = ("-----BEGIN PRIVATE KEY-----\n" "MIIEvgIBADANBgkqhkiG9w0BAQEFAASC\n" "-----END PRIVATE KEY-----")
_ENCRYPTED_PEM = (
    "-----BEGIN ENCRYPTED PRIVATE KEY-----\n"
    "MIIFCTBHBgkqhkiG9w0BBQ0wOjANBglghkgBZQMEAwEFENCRYPTEDKEYBODY\n"
    "-----END ENCRYPTED PRIVATE KEY-----"
)
_OPENSSH_PEM = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\n"
    "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAEAAABlOPENSSHKEYBODY\n"
    "-----END OPENSSH PRIVATE KEY-----"
)
_EC_PEM = (
    "-----BEGIN EC PRIVATE KEY-----\n"
    "MHcCAQEEIBF8XZQ6ECKEYBODYwJ5fWx8+1FcQ2rY=\n"
    "-----END EC PRIVATE KEY-----"
)
_DSA_PEM = (
    "-----BEGIN DSA PRIVATE KEY-----\n"
    "MIH3AgEAAkEA8qWq6Q2DSAKEYBODYB5QhJ9zQ2nLr3U=\n"
    "-----END DSA PRIVATE KEY-----"
)

# Shared by the block and env-assignment redaction tests — one spec for the six
# PEM variants so the parametrize lists cannot drift out of sync.
_PEM_KEY_CASES: list[tuple[str, str]] = [
    (_PKCS1_PEM, "MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn"), (_PKCS8_PEM, "MIIEvgIBADANBgkqhkiG9w0BAQEFAASC"),
    (_ENCRYPTED_PEM, "ENCRYPTEDKEYBODY"), (_OPENSSH_PEM, "OPENSSHKEYBODY"), (_EC_PEM, "ECKEYBODY"),
    (_DSA_PEM, "DSAKEYBODY"),
]
_PEM_KEY_IDS = ["pkcs1", "pkcs8", "encrypted", "openssh", "ec", "dsa"]


@pytest.mark.parametrize(("pem", "body"), _PEM_KEY_CASES, ids=_PEM_KEY_IDS,)
def test_redactor_scrubs_private_key_block(pem: str, body: str) -> None:
    """PEM private-key blocks (PKCS1/RSA, PKCS8, ENCRYPTED, OPENSSH, EC, DSA) replaced with [REDACTED_PEM_KEY]."""
    out = Redactor().redact_step(_user_step(f"key is {pem} ok"))
    assert isinstance(out.message, str)
    assert body not in out.message
    assert "[REDACTED_PEM_KEY]" in out.message


@pytest.mark.parametrize(("pem", "body"), _PEM_KEY_CASES, ids=_PEM_KEY_IDS)
def test_redactor_scrubs_private_key_in_env_assignment(pem: str, body: str) -> None:
    """VAR=<PEM> redacts fully — PEM rule must run before the env-var rule, no base64 body survives."""
    out = Redactor().redact_step(_user_step(f"DAYDREAM_APP_PRIVATE_KEY={pem}"))
    assert isinstance(out.message, str)
    assert body not in out.message
    assert out.message == "DAYDREAM_APP_PRIVATE_KEY=[REDACTED_ENV_VAR]"


def test_redactor_preserves_certificate_block() -> None:
    cert = ("-----BEGIN CERTIFICATE-----\n" "MIIDdzCCAl+gAwIBAgIEAgAAuTANBgkq\n" "-----END CERTIFICATE-----")
    out = Redactor().redact_step(_user_step(f"cert: {cert}"))
    assert isinstance(out.message, str)
    assert "[REDACTED_PEM_KEY]" not in out.message
    assert "BEGIN CERTIFICATE" in out.message


@pytest.mark.parametrize("marker", ["[REDACTED_API_KEY]", "<redacted>"])
def test_structured_redaction_leaves_an_existing_marker_alone(marker: str) -> None:
    """A later pass must not swap one host redaction marker for another."""
    text = f"api_key = already {marker}"
    assert redact_value({"note": text}) == {"note": text}


def test_redact_value_recurses_redacts_keys_and_values_without_mutating() -> None:
    sentinel = "ghp_" + "x" * 16
    payload = {"token": sentinel, sentinel: "key-secret", "nested": {"path": f"/Users/{sentinel}"},
        "items": [sentinel, 42, None], "flag": True, 1: "non-string-key",
    }
    original = copy.deepcopy(payload)
    out = redact_value(payload)

    assert payload == original                      # never mutates the argument
    assert out is not payload and out["nested"] is not payload["nested"]  # fresh containers
    assert sentinel not in json.dumps(out)          # redacted in values AND keys
    assert "[REDACTED" in json.dumps(out)           # a marker replaced it
    assert out["items"] == ["[REDACTED_API_KEY]", 42, None]  # scalars preserved
    assert out["flag"] is True and out[1] == "non-string-key"  # non-string keys untouched
    assert redact_value(("sk-" + "x" * 16,)) == ("[REDACTED_API_KEY]",)  # tuple rebuilt
    opaque = object()
    assert redact_value(opaque) is opaque  # non-container leaves keep identity
    # Nested container shapes used by improve artifacts: tuple of list of dict,
    # with non-sensitive keys redacted by flat/structured text rules only.
    assert redact_value(([{"note": "credential sk-abcdef123456 in the note"}],)) == (
        [{"note": "credential [REDACTED_API_KEY] in the note"}],
    )


@pytest.mark.parametrize("text", [
    '{"credentials": {"apiKey": "opaque-test-only-sentinel"}}', "{'client_secret': 'opaque-test-only-sentinel'}",
    "apiKey: opaque-test-only-sentinel", "client-secret = opaque-test-only-sentinel",
])
def test_redactor_scrubs_sensitive_key_value_text_formats(text: str) -> None:
    out = Redactor().redact_step(_user_step(text))
    assert isinstance(out.message, str)
    assert "opaque-test-only-sentinel" not in out.message
    assert "[REDACTED_CREDENTIAL]" in out.message


@pytest.mark.parametrize("text", [
    "apiKey:\n  nested: opaque-test-only-sentinel",
    '{\n  "apiKey": {\n    "nested": "opaque-test-only-sentinel"\n  }\n}',
    '{"apiKey": {"nested": "opaque-test-only-sentinel"}}',
    "1AUTHORIZATION = opaque-test-only-sentinel",
    '1apiKey: "opaque-test-only-sentinel"',
    "1apiKey:\n  nested: opaque-test-only-sentinel\n",
])
def test_redactor_scrubs_sensitive_key_shapes(text: str) -> None:
    out = redact_structured_text(text)
    assert "opaque-test-only-sentinel" not in out
    assert "[REDACTED_CREDENTIAL]" in out


def test_redactor_preserves_structural_separator_after_bare_value() -> None:
    """Redacting a bare value must not swallow the following ',' — the JSON
    keeps its structure (issue: '{"token": null, "count": 3}' lost its comma)."""
    out = redact_structured_text('{"token": null, "count": 3}')
    assert out == '{"token": "[REDACTED_CREDENTIAL]", "count": 3}'

    assert json.loads(out)  # still parseable as JSON


@pytest.mark.parametrize("text", ["the token: is now available", "The authorization: feature is enabled now"])
def test_redactor_preserves_prose_with_sensitive_word_colon(text: str) -> None:
    """A sensitive word followed by a colon in ordinary prose is not a key-value
    pair — the following prose word is left untouched."""
    out = redact_structured_text(text)
    assert out == text
    assert "[REDACTED_CREDENTIAL]" not in out


def test_redactor_keeps_comma_in_bare_yaml_pair() -> None:
    out = redact_structured_text("token: abc, other: 1")
    assert out == 'token: "[REDACTED_CREDENTIAL]", other: 1'


def test_redactor_scrubs_scheme_pair_under_sensitive_key() -> None:
    out = redact_structured_text("token: Bearer opaque-token-xyz")
    assert "opaque-token-xyz" not in out
    assert "token: Bearer [REDACTED_CREDENTIAL]" in out


def test_redactor_linear_scan_nested_pair_in_bare_value() -> None:
    """A sensitive pair nested in a non-sensitive pair's bare value is still
    redacted under the separator-anchored scan: the value advance stops at the
    value start, so the inner pair's key is re-anchored, never skipped."""
    out = redact_structured_text("note: apiKey: sk-opaque123")
    assert "sk-opaque123" not in out
    assert "[REDACTED" in out
    assert out.startswith("note:")


def test_redactor_linear_scan_nested_pair_in_quoted_value() -> None:
    out = redact_structured_text('note: "apiKey: sk-opaque123"')
    assert "sk-opaque123" not in out
    assert "[REDACTED" in out
    assert out.startswith('note: "')


def test_redactor_linear_scan_long_non_sensitive_key_precedes_sensitive_pair() -> None:
    """The O(n^2) shape from the bug: a long non-sensitive key run followed by
    a sensitive pair. The scan must not re-match the run per character — the
    pair after it is still found and redacted."""
    text = "a" * 5000 + "=x token=sk-opaque123"
    out = redact_structured_text(text)
    assert "sk-opaque123" not in out
    assert "[REDACTED" in out
    assert out.startswith("a" * 5000)
    assert "=x " in out


def test_redactor_linear_scan_separatorless_large_text_unchanged() -> None:
    """A long run of key-shaped characters with no separator anywhere is
    returned unchanged (the engine-quadratic guard: the old scan made the
    engine try every start position over the run)."""
    text = "x" * 200_000
    out = redact_structured_text(text)
    assert out == text


def test_redactor_sensitive_suffix_scan_is_linear() -> None:
    """Bound a long nonsensitive run while still redacting its trailing sensitive pair.

    The 5s CPU ceiling excludes scheduler delays from parallel test workers;
    the credential marker proves the structured pass ran."""

    text = "a" * 200_000 + "=x token=opaque-test-only-sentinel"
    start = time.process_time()
    out = redact_structured_text(text)
    elapsed = time.process_time() - start
    assert elapsed < 5
    assert "opaque-test-only-sentinel" not in out
    assert "[REDACTED_CREDENTIAL]" in out
    assert out.startswith("a" * 200_000)
    assert "=x " in out


def test_redactor_separator_heavy_suffix_scan_is_linear() -> None:
    """A 100K-separator run must remain linear under a generous 10s ceiling."""

    seps = "_" * 100_000

    # Non-sensitive: byte-identical output, and the run must not dominate.
    text = "foo" + seps + "tail: x"
    start = time.perf_counter()
    out = redact_structured_text(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 10
    assert out == text

    # Pair pass: a sensitive suffix AFTER the run (''xapi_key'' -> 'api_key' —
    # the whole key stays non-sensitive because segments split on '_') is
    # still found and redacted once the separator run is skipped.
    text2 = "foo" + seps + "xapi_key: opaque-test-only-sentinel"
    start = time.perf_counter()
    out2 = redact_structured_text(text2)
    elapsed = time.perf_counter() - start
    assert elapsed < 10
    assert "opaque-test-only-sentinel" not in out2
    assert "[REDACTED" in out2
    assert out2.startswith("foo" + seps + "x")

    # Block pass: same long run in front of a block-style sensitive suffix.
    text3 = "foo" + seps + "xapi_key: {\n  \"nested\": \"opaque-test-only-sentinel\"\n}"
    start = time.perf_counter()
    out3 = redact_structured_text(text3)
    elapsed = time.perf_counter() - start
    assert elapsed < 10
    assert "opaque-test-only-sentinel" not in out3
    assert "[REDACTED" in out3
    assert out3.startswith("foo" + seps + "x")


@pytest.mark.parametrize("text", [
    'defauthorization: "opaque-test-only-sentinel"',
    "nullpasswd=1",
    "superclient_secret = opaque-test-only-sentinel",
    "defauthorization:\n  nested: opaque-test-only-sentinel\n",
    "nullpasswd:\n  nested: opaque-test-only-sentinel\n",
])
def test_redactor_scrubs_sensitive_suffix_in_non_sensitive_key(text: str) -> None:
    out = redact_structured_text(text)
    assert "opaque-test-only-sentinel" not in out
    assert "[REDACTED_CREDENTIAL]" in out
    assert str(text).split(":", 1)[0].split("=", 1)[0] in out


def test_redactor_gap_mechanics_compose() -> None:
    """The digit-prefix anchor and the sensitive-suffix discovery compose: a
    digit-prefixed run whose first key-START char starts a NON-sensitive key
    still lands on the embedded sensitive suffix (issue #1236 Task 2b)."""
    out = redact_structured_text("12defauthorization= opaque-test-only-sentinel")
    assert "opaque-test-only-sentinel" not in out
    assert out.startswith("12defauthorization=")
    assert "[REDACTED_CREDENTIAL]" in out
    out2 = redact_structured_text("1nullpasswd: opaque-test-only-sentinel")
    assert "opaque-test-only-sentinel" not in out2
    assert "[REDACTED_CREDENTIAL]" in out2


def test_redactor_sensitive_suffix_block_empty_value_not_redacted() -> None:
    """A sensitive suffix whose block value is EMPTY (nothing indented) is not
    a redaction in the block pass — the old scan advanced past it and kept
    looking, so the text passes through unchanged."""
    text = "defpasswd:\nplain: 1\n"
    out = redact_structured_text(text)
    assert out == text
    assert "[REDACTED_CREDENTIAL]" not in out
