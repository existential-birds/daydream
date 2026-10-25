"""Redaction positives, clean-input negatives, field coverage, and fail-closed behavior."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Sequence

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

def test_redactor_scrubs_single_token_secret(subtests: pytest.Subtests) -> None:
    """REDA-01: sk-/ghp_/xoxb-/AKIA/ghs_ tokens → [REDACTED_API_KEY], JWTs → [REDACTED_JWT]."""
    cases: Sequence[tuple[str, str, str]] = [
        ("token=sk-test-12345abcdef done", "sk-test-12345abcdef", "[REDACTED_API_KEY]"),
        ("auth=ghp_test123abcdef bearer", "ghp_test123abcdef", "[REDACTED_API_KEY]"),
        ("slack=xoxb-test456abcdef", "xoxb-test456abcdef", "[REDACTED_API_KEY]"),
        ("aws=AKIA0000TESTKEY00000", "AKIA0000TESTKEY00000", "[REDACTED_API_KEY]"),
        ("token=ghs_abc123DEF456ghi789jkl012 done", "ghs_abc123DEF456ghi789jkl012", "[REDACTED_API_KEY]"),
        (f"bearer {_JWT}", _JWT, "[REDACTED_JWT]"),
    ]
    case_ids_0 = ["openai", "github", "slack", "aws", "github_installation", "jwt"]
    for case_index, (text, raw_secret, marker) in enumerate(cases):
        with subtests.test(msg=case_ids_0[case_index], text=text, raw_secret=raw_secret, marker=marker):
            out = Redactor().redact_step(_user_step(text))
            assert isinstance(out.message, str)
            assert raw_secret not in out.message
            assert marker in out.message


def test_redaction_preserves_non_secret_prose_names_and_certificates(subtests: pytest.Subtests) -> None:
    """Credential-like substrings, prose, public certificates and empty blocks stay byte-identical."""
    cases = [
        "eyJhbG is a prefix",
        "DEBUG=true\nAPP_NAME=myproject",
        "MONKEY_PATCH=enabled",
        "KEYBOARD_LAYOUT=qwerty",
        "AUTHOR=alice",
        "TOKENIZED=foo",
        "KEYSTORE=path/to/store",
        "the token: is now available",
        "The authorization: feature is enabled now",
        "defpasswd:\nplain: 1\n",
        "-----BEGIN CERTIFICATE-----\nMIIDdzCCAl+gAwIBAgIEAgAAuTANBgkq\n-----END CERTIFICATE-----",
    ]
    for text in cases:
        with subtests.test(text=text):
            out = Redactor().redact_step(_user_step(text))
            assert out.message == text
            assert "[REDACTED" not in out.message
            assert redact_structured_text(text) == text


def test_redactor_scrubs_username_path(subtests: pytest.Subtests) -> None:
    """REDA-02: /Users//home//C:\\Users\\ <name> → [REDACTED_USER], project-relative tail preserved."""
    cases: Sequence[tuple[str, str, str, str | None]] = [
        ("path=/Users/ka/github/proj/app.py", "/Users/ka", "/Users/[REDACTED_USER]", "github/proj/app.py"),
        ("path=/home/alice/foo/bar", "/home/alice", "/home/[REDACTED_USER]", "foo/bar"),
        ("path=C:\\Users\\bob\\repo", "Users\\bob", "[REDACTED_USER]", None),
    ]
    case_ids_0 = ["macos", "linux", "windows"]
    for case_index, (text, absent, present, preserved_tail) in enumerate(cases):
        with subtests.test(
            msg=case_ids_0[case_index],
            text=text,
            absent=absent,
            present=present,
            preserved_tail=preserved_tail,
        ):
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

def test_redact_arguments_passthrough_when_no_secret() -> None:
    arguments = {"count": 42, "flags": ["a", "b"], "config": {"x": 1, "y": [1, 2, 3]}}
    out = Redactor()._redact_arguments(arguments)
    assert out["count"] == 42
    assert out["flags"] == ["a", "b"]
    assert out["config"] == {"x": 1, "y": [1, 2, 3]}

def test_redact_arguments_recursive_failure_falls_back_to_redaction_failed(monkeypatch: pytest.MonkeyPatch,) -> None:
    def _boom(value: object, sensitive: bool = False) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(redactor_mod, "redact_value", _boom, raising=True)
    out = Redactor()._redact_arguments({"edits": [{"x": 1}]})
    assert out["edits"] == "[REDACTION_FAILED]"




def test_env_var_pattern_redacts_legitimate_secret_segments(subtests: pytest.Subtests) -> None:
    cases: Sequence[tuple[str, str]] = [
        ("OPENAI_API_KEY=sk-leakthis", "sk-leakthis"),
        ("MY_API_KEY=value", "value"),
        ("JWT_TOKEN=abc.def.ghi", "abc.def.ghi"),
        ("DB_PASSWORD=hunter2", "hunter2"),
        ("AUTH_TOKEN=t-foo", "t-foo"),
        ("CACHE_KEY=k-bar", "k-bar"),
        ("DB_CREDENTIAL=admin:pw", "admin:pw"),
    ]
    for case_index, secret in enumerate(cases):
        with subtests.test(secret=secret):
            line, raw_value = secret
            out = Redactor().redact_step(_user_step(line))
            assert isinstance(out.message, str)
            assert raw_value not in out.message
            assert "[REDACTED_ENV_VAR]" in out.message


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

# Both the text-block and environment-assignment channels use the same six
# PEM variants in the private-key contract table.
_PEM_KEY_CASES: list[tuple[str, str]] = [
    (_PKCS1_PEM, "MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn"), (_PKCS8_PEM, "MIIEvgIBADANBgkqhkiG9w0BAQEFAASC"),
    (_ENCRYPTED_PEM, "ENCRYPTEDKEYBODY"), (_OPENSSH_PEM, "OPENSSHKEYBODY"), (_EC_PEM, "ECKEYBODY"),
    (_DSA_PEM, "DSAKEYBODY"),
]
_PEM_KEY_IDS = ["pkcs1", "pkcs8", "encrypted", "openssh", "ec", "dsa"]


def test_private_key_formats_are_redacted_before_environment_assignments(subtests: pytest.Subtests) -> None:
    """Every PEM variant is removed both as a text block and as an environment value."""
    for name, (pem, body) in zip(_PEM_KEY_IDS, _PEM_KEY_CASES, strict=True):
        for assignment in (False, True):
            with subtests.test(format=name, assignment=assignment):
                text = f"DAYDREAM_APP_PRIVATE_KEY={pem}" if assignment else f"key is {pem} ok"
                out = Redactor().redact_step(_user_step(text))
                assert isinstance(out.message, str)
                assert body not in out.message
                if assignment:
                    assert out.message == "DAYDREAM_APP_PRIVATE_KEY=[REDACTED_ENV_VAR]"
                else:
                    assert "[REDACTED_PEM_KEY]" in out.message


def test_structured_redaction_leaves_an_existing_marker_alone(subtests: pytest.Subtests) -> None:
    """A later pass must not swap one host redaction marker for another."""
    cases: Sequence[str] = ["[REDACTED_API_KEY]", "<redacted>"]
    for case_index, marker in enumerate(cases):
        with subtests.test(marker=marker):
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


def test_redactor_preserves_non_sensitive_structured_keys(subtests: pytest.Subtests) -> None:
    cases: Sequence[str] = ["tokenizer", "passwordless", "monkeyPatch", "keyStore", "max_tokens"]
    for case_index, non_secret_key in enumerate(cases):
        with subtests.test(non_secret_key=non_secret_key):
            call = ToolCall(
                tool_call_id="t1",
                function_name="Bash",
                arguments={non_secret_key: "opaque-test-only-sentinel"},
            )
            out = Redactor().redact_step(_agent_step(tool_calls=[call]))
            assert out.tool_calls is not None
            args = out.tool_calls[0].arguments
            assert args[non_secret_key] == "opaque-test-only-sentinel"
            assert "[REDACTED_CREDENTIAL]" not in json.dumps(args)


def test_redactor_scrubs_sensitive_key_value_text_formats(subtests: pytest.Subtests) -> None:
    cases: Sequence[str] = [
        '{"credentials": {"apiKey": "opaque-test-only-sentinel"}}',
        "{'client_secret': 'opaque-test-only-sentinel'}",
        "apiKey: opaque-test-only-sentinel",
        "client-secret = opaque-test-only-sentinel",
    ]
    for case_index, text in enumerate(cases):
        with subtests.test(text=text):
            out = Redactor().redact_step(_user_step(text))
            assert isinstance(out.message, str)
            assert "opaque-test-only-sentinel" not in out.message
            assert "[REDACTED_CREDENTIAL]" in out.message


def test_header_redaction_preserves_names_schemes_and_following_lines(subtests: pytest.Subtests) -> None:
    """Headers redact case-insensitively even when indented or embedded in command/prose."""
    rows = [
        ("Authorization", "opaque-test-only-sentinel", None),
        ("authorization", "Bearer opaque-test-only-sentinel", "Bearer"),
        ("Proxy-Authorization", "opaque-test-only-sentinel", None),
        ("X-Api-Key", "opaque-test-only-sentinel", None),
        ("X-Auth-Token", "opaque-test-only-sentinel", None),
        ("Cookie", "session=opaque-test-only-sentinel", None),
        ("Set-Cookie", "opaque-test-only-sentinel", None),
    ]
    for header, value, scheme in rows:
        with subtests.test(header=header, scheme=scheme):
            out = Redactor().redact_step(_user_step(f"{header}: {value}\nnext line stays"))
            assert isinstance(out.message, str)
            assert "opaque-test-only-sentinel" not in out.message
            assert "[REDACTED_CREDENTIAL]" in out.message
            assert f"{header}: " in out.message
            assert "next line stays" in out.message
            if scheme is not None:
                assert f"{header}: {scheme} " in out.message
    for text in (
        'curl -H "Authorization: Bearer opaque-token-xyz" https://api',
        "  authorization: Bearer opaque-token-xyz",
        "saw Authorization: Bearer opaque-token-xyz in the log",
    ):
        with subtests.test(embedded=text):
            out = Redactor().redact_step(_user_step(text))
            assert isinstance(out.message, str)
            assert "opaque-token-xyz" not in out.message
            assert "Bearer [REDACTED_CREDENTIAL]" in out.message
            if text.startswith("curl"):
                assert "Authorization: Bearer [REDACTED_CREDENTIAL]" in out.message


def test_structured_scan_redacts_nested_pairs_prefixes_and_sensitive_suffixes(subtests: pytest.Subtests) -> None:
    """Separator/digit anchors and sensitive suffixes compose without skipping nested credentials."""
    rows = [
        ("apiKey:\n  nested: opaque-test-only-sentinel", "opaque-test-only-sentinel", "", "[REDACTED_CREDENTIAL]"),
        (
            '{\n  "apiKey": {\n    "nested": "opaque-test-only-sentinel"\n  }\n}',
            "opaque-test-only-sentinel",
            "",
            "[REDACTED_CREDENTIAL]",
        ),
        (
            '{"apiKey": {"nested": "opaque-test-only-sentinel"}}',
            "opaque-test-only-sentinel",
            "",
            "[REDACTED_CREDENTIAL]",
        ),
        ("text: apiKey: opaque-test-only-sentinel", "opaque-test-only-sentinel", "", "[REDACTED_CREDENTIAL]"),
        ("config: token=opaque-test-only-sentinel", "opaque-test-only-sentinel", "", "[REDACTED_CREDENTIAL]"),
        (
            '{"description": "use token=opaque-test-only-sentinel here"}',
            "opaque-test-only-sentinel",
            "",
            "[REDACTED_CREDENTIAL]",
        ),
        ("1apiKey=x", "opaque-test-only-sentinel", "", "[REDACTED_CREDENTIAL]"),
        ("2token= y", "opaque-test-only-sentinel", "", "[REDACTED_CREDENTIAL]"),
        ("123secret: z", "opaque-test-only-sentinel", "", "[REDACTED_CREDENTIAL]"),
        ("1AUTHORIZATION = opaque-test-only-sentinel", "opaque-test-only-sentinel", "", "[REDACTED_CREDENTIAL]"),
        ('1apiKey: "opaque-test-only-sentinel"', "opaque-test-only-sentinel", "", "[REDACTED_CREDENTIAL]"),
        ("1apiKey:\n  nested: opaque-test-only-sentinel\n", "opaque-test-only-sentinel", "", "[REDACTED_CREDENTIAL]"),
        ("token: Bearer opaque-token-xyz", "opaque-token-xyz", "token: Bearer ", "[REDACTED_CREDENTIAL]"),
        ("note: apiKey: sk-opaque123", "sk-opaque123", "note:", "[REDACTED"),
        ('note: "apiKey: sk-opaque123"', "sk-opaque123", 'note: "', "[REDACTED"),
        ("a" * 5000 + "=x token=sk-opaque123", "sk-opaque123", "a" * 5000 + "=x ", "[REDACTED"),
        (
            "a" * 5000 + "=x token=opaque-test-only-sentinel",
            "opaque-test-only-sentinel",
            "a" * 5000 + "=x ",
            "[REDACTED_CREDENTIAL]",
        ),
        (
            'defauthorization: "opaque-test-only-sentinel"',
            "opaque-test-only-sentinel",
            "defauthorization",
            "[REDACTED_CREDENTIAL]",
        ),
        ("nullpasswd=1", "opaque-test-only-sentinel", "nullpasswd", "[REDACTED_CREDENTIAL]"),
        ("fooapi_key: 2", "opaque-test-only-sentinel", "fooapi_key", "[REDACTED_CREDENTIAL]"),
        ("xpassword=3", "opaque-test-only-sentinel", "xpassword", "[REDACTED_CREDENTIAL]"),
        (
            "superclient_secret = opaque-test-only-sentinel",
            "opaque-test-only-sentinel",
            "superclient_secret ",
            "[REDACTED_CREDENTIAL]",
        ),
        (
            "defauthorization:\n  nested: opaque-test-only-sentinel\n",
            "opaque-test-only-sentinel",
            "defauthorization",
            "[REDACTED_CREDENTIAL]",
        ),
        (
            "nullpasswd:\n  nested: opaque-test-only-sentinel\n",
            "opaque-test-only-sentinel",
            "nullpasswd",
            "[REDACTED_CREDENTIAL]",
        ),
        (
            "12defauthorization= opaque-test-only-sentinel",
            "opaque-test-only-sentinel",
            "12defauthorization=",
            "[REDACTED_CREDENTIAL]",
        ),
        ("1nullpasswd: opaque-test-only-sentinel", "opaque-test-only-sentinel", "", "[REDACTED_CREDENTIAL]"),
    ]
    for text, removed, prefix, marker in rows:
        with subtests.test(text=text):
            out = redact_structured_text(text)
            assert removed not in out
            assert marker in out
            assert out.startswith(prefix)


def test_structured_redaction_preserves_following_json_and_yaml_separators(subtests: pytest.Subtests) -> None:
    """Bare-value replacement cannot swallow the following comma or corrupt JSON."""
    for text, expected, parseable in (
        ('{"token": null, "count": 3}', '{"token": "[REDACTED_CREDENTIAL]", "count": 3}', True),
        ("token: abc, other: 1", 'token: "[REDACTED_CREDENTIAL]", other: 1', False),
    ):
        with subtests.test(text=text):
            out = redact_structured_text(text)
            assert out == expected
            if parseable:
                assert json.loads(out)


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
