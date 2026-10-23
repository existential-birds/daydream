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

def test_credential_rotation_diagnostic_redacts_independent_documents_and_preserves_siblings() -> None:
    """One native record retains its certificate, paths and config structure while removing every credential family."""
    certificate = "-----BEGIN CERTIFICATE-----\nMIIDdzCCAl+gAwIBAgIEAgAAuTANBgkq\n-----END CERTIFICATE-----"
    out = Redactor().redact_step(
        _agent_step(
            message=(
                f"RSA: {_PKCS1_PEM}\nPKCS8: {_PKCS8_PEM}\nEncrypted: {_ENCRYPTED_PEM}\n"
                f"SSH: {_OPENSSH_PEM}\nEC: {_EC_PEM}\nDSA: {_DSA_PEM}\nCertificate: {certificate}\n"
                "path=/Users/ka/github/proj/app.py\npath=/home/alice/foo/bar\npath=C:\\Users\\bob\\repo"
            ),
            reasoning_content=(
                f"thought: sk-test-secret123abc\nRSA_PRIVATE_KEY={_PKCS1_PEM}\nPKCS8_PRIVATE_KEY={_PKCS8_PEM}\n"
                f"ENCRYPTED_PRIVATE_KEY={_ENCRYPTED_PEM}\nSSH_PRIVATE_KEY={_OPENSSH_PEM}\n"
                f"EC_PRIVATE_KEY={_EC_PEM}\nDSA_PRIVATE_KEY={_DSA_PEM}"
            ),
            observation=Observation(
                results=[
                    ObservationResult(
                        source_call_id="json-config",
                        content=(
                            '{"credentials": {"apiKey": "opaque-test-only-sentinel"}, "profiles": ['
                            '{"apiKey": {"nested": "opaque-test-only-sentinel"}},\n'
                            '{"apiKey": {\n  "nested": "opaque-test-only-sentinel"\n}}], "enabled": true}'
                        ),
                    ),
                    ObservationResult(
                        source_call_id="python-config", content="{'client_secret': 'opaque-test-only-sentinel'}"
                    ),
                    ObservationResult(
                        source_call_id="yaml-config",
                        content=(
                            "1apiKey:\n  nested: opaque-test-only-sentinel\n"
                            "defauthorization:\n  nested: opaque-test-only-sentinel\n"
                            "nullpasswd:\n  nested: opaque-test-only-sentinel\nplain: 1\n"
                            "apiKey:\n  nested: opaque-test-only-sentinel"
                        ),
                    ),
                    ObservationResult(
                        source_call_id="rotation-log",
                        content=(
                            "apiKey: opaque-test-only-sentinel\nclient-secret = opaque-test-only-sentinel\n"
                            '1AUTHORIZATION = opaque-test-only-sentinel\n1apiKey: "opaque-test-only-sentinel"\n'
                            'defauthorization: "opaque-test-only-sentinel"\nnullpasswd=1\n'
                            "superclient_secret = opaque-test-only-sentinel"
                        ),
                    ),
                ]
            ),
        )
    )
    assert out.message == (
        "RSA: [REDACTED_PEM_KEY]\nPKCS8: [REDACTED_PEM_KEY]\nEncrypted: [REDACTED_PEM_KEY]\n"
        "SSH: [REDACTED_PEM_KEY]\nEC: [REDACTED_PEM_KEY]\nDSA: [REDACTED_PEM_KEY]\n"
        f"Certificate: {certificate}\npath=/Users/[REDACTED_USER]/github/proj/app.py\n"
        "path=/home/[REDACTED_USER]/foo/bar\npath=C:\\Users\\[REDACTED_USER]\\repo"
    )
    assert out.reasoning_content == (
        "thought: [REDACTED_API_KEY]\nRSA_PRIVATE_KEY=[REDACTED_ENV_VAR]\nPKCS8_PRIVATE_KEY=[REDACTED_ENV_VAR]\n"
        "ENCRYPTED_PRIVATE_KEY=[REDACTED_ENV_VAR]\nSSH_PRIVATE_KEY=[REDACTED_ENV_VAR]\n"
        "EC_PRIVATE_KEY=[REDACTED_ENV_VAR]\nDSA_PRIVATE_KEY=[REDACTED_ENV_VAR]"
    )
    assert out.observation is not None
    assert [(result.source_call_id, result.content) for result in out.observation.results] == [
        (
            "json-config",
            '{"credentials": "[REDACTED_CREDENTIAL]", "profiles": ['
            '{"apiKey": "[REDACTED_CREDENTIAL]"},\n{"apiKey": "[REDACTED_CREDENTIAL]"}], "enabled": true}',
        ),
        ("python-config", "{'client_secret': '[REDACTED_CREDENTIAL]'}"),
        (
            "yaml-config",
            '1apiKey:"[REDACTED_CREDENTIAL]"\ndefauthorization:"[REDACTED_CREDENTIAL]"\n'
            'nullpasswd:"[REDACTED_CREDENTIAL]"\nplain: 1\napiKey:"[REDACTED_CREDENTIAL]"',
        ),
        (
            "rotation-log",
            'apiKey: "[REDACTED_CREDENTIAL]"\nclient-secret = "[REDACTED_CREDENTIAL]"\n'
            '1AUTHORIZATION = "[REDACTED_CREDENTIAL]"\n1apiKey: "[REDACTED_CREDENTIAL]"\n'
            'defauthorization: "[REDACTED_CREDENTIAL]"\nnullpasswd="[REDACTED_CREDENTIAL]"\n'
            'superclient_secret = "[REDACTED_CREDENTIAL]"',
        ),
    ]

@pytest.mark.parametrize(("pem", "body"), [
    (_PKCS1_PEM, "MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn"), (_PKCS8_PEM, "MIIEvgIBADANBgkqhkiG9w0BAQEFAASC"),
    (_ENCRYPTED_PEM, "ENCRYPTEDKEYBODY"), (_OPENSSH_PEM, "OPENSSHKEYBODY"),
    (_EC_PEM, "ECKEYBODY"), (_DSA_PEM, "DSAKEYBODY"),
], ids=["pkcs1", "pkcs8", "encrypted", "openssh", "ec", "dsa"])

def test_redactor_scrubs_private_key_block(pem: str, body: str) -> None:
    """All supported key grammars fail closed before a streamed footer arrives.

    Complete PEM cases cannot detect key bodies exposed by an interrupted tail.
    """
    pem = pem.rsplit("-----END", 1)[0]
    out = Redactor().redact_step(_user_step(f"key is {pem} ok"))
    assert isinstance(out.message, str)
    assert body not in out.message
    assert "[REDACTED_PEM_KEY]" in out.message


@pytest.mark.parametrize("marker", ["[REDACTED_API_KEY]", "[redacted_api_key]", "<redacted>"])
def test_structured_redaction_leaves_an_existing_marker_alone(marker: str) -> None:
    """Exact markers remain idempotent without exempting another sensitive value.

    The old fixture put ``already`` under api_key before a marker and required
    that entire sensitive value to survive. That assertion allowed an opaque
    credential to survive merely because a later marker was present. Pin exact
    safe marker values/schemes and ordinary public text instead; the real
    recorder privacy rows independently cover residual credentials.
    """
    for text in (f"api_key = {marker}", f"api_key = '{marker}'", f"auth=Bearer {marker}",
                 f"api_key = {marker}[REDACTED_CREDENTIAL]", f"auth=Bearer {marker}[REDACTED_CREDENTIAL]",
                 f"api_key = {marker}; public diagnostics", f"auth=Bearer {marker}; public diagnostics",
                 f"refresh_token={marker};api_key: {marker}", f"refresh_token:{marker};api_key={marker}",
                 f"refresh_token: {marker};api_key:\t{marker}",
                 f"Public note already {marker}"):
        assert redact_value({"note": text}) == {"note": text}
    combined = marker + '[REDACTED_CREDENTIAL]'
    serialized = json.dumps({'api_key': combined})
    assert json.loads(redact_structured_text(serialized))['api_key'] == combined


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





@pytest.mark.parametrize('key', [
    'Authorization: Bearer opaque-key-canary',
    'auth=Bearer opaque-key-canary',
    'refresh_token="opaque-key-canary',
    'api_key=[REDACTED]opaque-key-canary',
    '{"Authorization: Bearer opaque-key-canary":"public"}',
])
def test_json_observation_keys_use_structured_redaction_without_retyping_values(key: str) -> None:
    """Decoded JSON keys need the same credential policy as observation text.

    Header tests pass plain text and argument tests exercise a native mapping;
    neither reaches the JSON leaf walk that formerly flat-redacted only keys.
    Keep JSON-looking public keys, marker identity, zero, null and valid wrappers.
    """
    payload = {key: '[REDACTED_RETAINED]', '{"scope":"python"}': True,
               'usage': {'input_tokens': 0, 'output_tokens': None}, 'public': 'diagnostics'}
    original = json.dumps(payload)
    step = _agent_step(observation=Observation(results=[ObservationResult(content=original)]))
    out = Redactor().redact_step(step)
    assert out.observation is not None
    content = out.observation.results[0].content
    assert isinstance(content, str)
    assert 'opaque-key-canary' not in content
    decoded = json.loads(content)
    assert decoded['{"scope":"python"}'] is True
    assert decoded['usage'] == {'input_tokens': 0, 'output_tokens': None}
    assert decoded['public'] == 'diagnostics'
    assert '[REDACTED_RETAINED]' in decoded.values()
    assert step.observation is not None and step.observation.results[0].content == original
    repeated = Redactor().redact_step(out)
    assert repeated.observation is not None and repeated.observation.results[0].content == content


@pytest.mark.parametrize('key', ['defauthorization: opaque-key-canary', 'deftoken=opaque-key-canary',
                                  'api_key=opaque-key-canary', 'X-Api-Key: opaque-key-canary',
                                  'private_key=opaque-key-canary'])
@pytest.mark.parametrize('value', [0, None, {'public': 'diagnostics'}])
def test_generated_json_key_marker_cannot_change_public_value_sensitivity(key: str, value: object) -> None:
    """A generated marker in a key cannot turn its public value into a credential.

    Structured suffix recognition can sanitize a key that whole-key sensitivity
    does not classify as secret. Marker words must not change that classification
    during the recorder's repeated sanitation, erasing zero/null/container values.
    """
    redacted = redact_structured_text(json.dumps({key: value}))
    assert 'opaque-key-canary' not in redacted
    assert list(json.loads(redacted).values()) == [value]
    assert redact_structured_text(redacted) == redacted


@pytest.mark.parametrize('key', ['api_key', 'private_key', '[REDACTED_CREDENTIAL]',
                                  'prefix_[REDACTED_TOKEN]', '[REDACTED]token',
                                  'Authorization: Bearer [REDACTED_CREDENTIAL]'])
def test_marker_words_do_not_hide_actual_json_key_sensitivity(key: str) -> None:
    """New key sanitation must preserve every previously sensitive key's masking."""
    redacted = redact_structured_text(json.dumps({key: 'opaque-value-canary'}))
    assert 'opaque-value-canary' not in redacted
    assert redact_structured_text(redacted) == redacted


@pytest.mark.parametrize(('header', 'marker', 'suffix'), [
    ('X-Api-Key', '[REDACTED_API_KEY]', ''),
    ('Cookie', '[REDACTED_CREDENTIAL]', ' (2)'),
    ('Authorization', '<redacted>', ' (2)'),
])
def test_complete_markers_in_json_header_keys_keep_their_identity(header: str, marker: str, suffix: str) -> None:
    """An already safe header-shaped key must survive the newly applied text policy.

    Complete markers before a collision suffix exposed regex token backtracking,
    which consumed only the marker prefix and added a spurious closing bracket.
    Plain secret header tests never establish safe-key identity on repeated writes.
    """
    key = f'{header}: {marker}{suffix}'
    redacted = redact_structured_text(json.dumps({key: '[REDACTED_RETAINED]'}))
    assert list(json.loads(redacted)) == [key]
    assert redact_structured_text(redacted) == redacted


@pytest.mark.parametrize(('template', 'safe'), [
    ('Authorization: Bearer {}', 'Authorization: Bearer [REDACTED_CREDENTIAL]'),
    ('api_key={}', '[REDACTED_LABEL]'),
    ('Cookie: {}', 'Cookie: [REDACTED_CREDENTIAL]'),
])
def test_json_observation_key_collisions_preserve_entries_and_existing_public_keys(template: str, safe: str) -> None:
    """Two newly sanitized keys cannot overwrite each other or a literal safe key.

    Flat key replacement already collides for shaped secrets, but adding structured
    key coverage creates collisions for opaque headers too. Keep each retained
    value and existing marker key without encoding any raw secret in its alias.
    """
    payload = {template.format('first-key-canary'): '[REDACTED_FIRST]',
               template.format('second-key-canary'): '[REDACTED_SECOND]',
               safe: '[REDACTED_EXISTING]', safe + ' (2)': '[REDACTED_RESERVED]',
               'usage': {'input_tokens': 0, 'output_tokens': None}}
    out = Redactor().redact_step(_agent_step(
        observation=Observation(results=[ObservationResult(content=json.dumps(payload))]),
    ))
    assert out.observation is not None
    content = out.observation.results[0].content
    assert isinstance(content, str)
    decoded = json.loads(content)
    assert 'first-key-canary' not in content and 'second-key-canary' not in content
    assert len(decoded) == len(payload)
    assert decoded[safe] == '[REDACTED_EXISTING]'
    assert decoded[safe + ' (2)'] == '[REDACTED_RESERVED]'
    assert set(item for item in decoded.values() if isinstance(item, str)) == {
        '[REDACTED_FIRST]', '[REDACTED_SECOND]', '[REDACTED_EXISTING]', '[REDACTED_RESERVED]',
    }
    assert decoded['usage'] == {'input_tokens': 0, 'output_tokens': None}
    repeated = Redactor().redact_step(out)
    assert repeated.observation is not None and repeated.observation.results[0].content == content



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


@pytest.mark.parametrize('fragment', [
    '-----BEGIN PRIVATE KEY-----\nleaf-secret-canary',
    '{"refresh_token": "leaf-secret-canary',
    'refresh_token="leaf-secret-canary',
    'refresh_token="leaf-secret-canary' + chr(92),
    "refresh_token='leaf-secret-canary" + chr(92),
    '{"refresh_token": "leaf-secret-canary' + chr(92),
])
@pytest.mark.parametrize('envelope', ['object', 'scalar'])
def test_unfinished_secret_leaf_redaction_preserves_complete_json_envelope(fragment: str, envelope: str) -> None:
    """A fail-closed leaf tail cannot consume its serialized JSON quote or siblings.

    Plain-field redaction cannot establish that log/receipt JSON remains parseable
    when its text leaf contains an unfinished PEM or quoted credential value.
    """
    serialized = json.dumps({'message': fragment, 'model': 'fixture-model', 'count': 0}
                            if envelope == 'object' else fragment)
    redacted = redact_structured_text(serialized)
    parsed = json.loads(redacted)
    assert 'leaf-secret-canary' not in redacted
    if envelope == 'scalar':
        assert '[REDACTED' in parsed
    else:
        assert '[REDACTED' in parsed['message']
        assert parsed['model'] == 'fixture-model' and parsed['count'] == 0


@pytest.mark.parametrize('container', [
    '["opaque-container-canary", "public"]',
    '{"entry": "opaque-container-canary", "public": true}',
])
def test_sensitive_container_redaction_is_not_replaced_by_marker_token_parsing(container: str) -> None:
    """Marker-prefixed bare values must not swallow or expose structured containers.

    Marker-only cases do not establish the array/object alternative still consumes
    a complete sensitive block after the bare-token grammar changes.
    """
    assert redact_structured_text('token = ' + container) == 'token = "[REDACTED_CREDENTIAL]"'


@pytest.mark.parametrize(('source', 'expected'), [
    ('TOKEN=opaque-idempotence-canary; public diagnostics',
     'TOKEN=[REDACTED_ENV_VAR]; public diagnostics'),
    ('TOKEN=opaque-idempotence-canary;API_KEY=opaque-second-canary',
     'TOKEN=[REDACTED_ENV_VAR];API_KEY=[REDACTED_ENV_VAR]'),
    ('TOKEN=opaque-idempotence-canary;NEXT=public',
     'TOKEN=[REDACTED_ENV_VAR];NEXT=public'),
    ('TOKEN=opaque-idempotence-canary;api_key=opaque-second-canary',
     'TOKEN=[REDACTED_ENV_VAR];api_key="[REDACTED_CREDENTIAL]"'),
    ('refresh_token=[REDACTED]opaque-first-canary;api_key=opaque-second-canary',
     'refresh_token=[REDACTED];api_key="[REDACTED_CREDENTIAL]"'),
    ('refresh_token=[REDACTED]opaque-first-canary;;api_key=opaque-second-canary',
     'refresh_token=[REDACTED];;api_key="[REDACTED_CREDENTIAL]"'),
    ('refresh_token=[REDACTED]opaque-first-canary;api_key:opaque-second-canary',
     'refresh_token=[REDACTED];api_key:"[REDACTED_CREDENTIAL]"'),
    ('refresh_token=[REDACTED]opaque-first-canary;api_key: opaque-second-canary',
     'refresh_token=[REDACTED];api_key: "[REDACTED_CREDENTIAL]"'),
    ('refresh_token:[REDACTED]opaque-first-canary;api_key:opaque-second-canary',
     'refresh_token:[REDACTED];api_key:"[REDACTED_CREDENTIAL]"'),
    ('refresh_token:[REDACTED]opaque-first-canary;api_key:opaque-second-canary;password=opaque-third-canary',
     'refresh_token:[REDACTED];api_key:"[REDACTED_CREDENTIAL]";password="[REDACTED_CREDENTIAL]"'),
])
def test_secret_assignment_redaction_preserves_semicolon_and_is_idempotent(source: str, expected: str) -> None:
    """A separator is not residual credential content after an existing marker.

    Marker-only fields without delimiters miss this composition: ENV redaction
    adds a marker, the structured pass must preserve its semicolon and following
    assignment identity (including a lowercase secret), and recorder
    re-redaction must not change the sanitized bytes again. The Pi persistence
    test separately protects the transport's bound and truncation metadata.
    """
    redacted = redact_structured_text(source)
    assert redacted == expected
    assert redact_structured_text(redacted) == redacted


@pytest.mark.parametrize('first_separator', ['=', ':'])
@pytest.mark.parametrize('second_separator', ['=', ':', ': ', ':\t'])
@pytest.mark.parametrize('boundary', [';', '; ', ' ;', ' ;\t'])
def test_sensitive_semicolon_pairs_preserve_both_separator_frontiers(
    first_separator: str, second_separator: str, boundary: str,
) -> None:
    """A later colon pair must receive the same sanitation as an assignment.

    The original adjacent-key row only exercised '=', allowing a preceding bare
    match to consume a later colon key and expose its credential. Preserve the
    supported key/value separator crossproduct without admitting ordinary prose.
    """
    source = (f'refresh_token{first_separator}[REDACTED]opaque-first-canary{boundary}'
              f'api_key{second_separator}opaque-second-canary')
    redacted = redact_structured_text(source)
    assert 'opaque-first-canary' not in redacted and 'opaque-second-canary' not in redacted
    assert f'refresh_token{first_separator}' in redacted and f'api_key{second_separator}' in redacted
    assert redact_structured_text(redacted) == redacted
    prose = 'The token: is now available;metadata: public diagnostics'
    assert redact_structured_text(prose) == prose


@pytest.mark.parametrize('name', ['api_key', 'TOKEN'])
def test_internal_semicolon_cannot_expose_sensitive_assignment_residual(name: str) -> None:
    """Keep both private fragments hidden when a semicolon is inside one bare value.

    Treating every semicolon as a delimiter fixes marker-only punctuation but
    weakens existing residual masking; this guards that distinct privacy contract.
    """
    redacted = redact_structured_text(name + '=opaque-prefix-canary;opaque-tail-canary')
    assert 'opaque-prefix-canary' not in redacted and 'opaque-tail-canary' not in redacted
    assert redact_structured_text(redacted) == redacted
