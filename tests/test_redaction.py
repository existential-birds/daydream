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

@pytest.mark.parametrize(('source', 'private', 'public'), [
    ('token=sk-test-12345abcdef done', 'sk-test-12345abcdef', '[REDACTED_API_KEY]'),
    ('auth=ghp_test123abcdef bearer', 'ghp_test123abcdef', '[REDACTED_API_KEY]'),
    ('slack=xoxb-test456abcdef', 'xoxb-test456abcdef', '[REDACTED_API_KEY]'),
    ('aws=AKIA0000TESTKEY00000', 'AKIA0000TESTKEY00000', '[REDACTED_API_KEY]'),
    ('token=ghs_abc123DEF456ghi789jkl012 done', 'ghs_abc123DEF456ghi789jkl012', '[REDACTED_API_KEY]'),
    (f'bearer {_JWT}', _JWT, '[REDACTED_JWT]'),
    ('path=/Users/ka/github/proj/app.py', '/Users/ka', '/Users/[REDACTED_USER]/github/proj/app.py'),
    ('path=/home/alice/foo/bar', '/home/alice', '/home/[REDACTED_USER]/foo/bar'),
    ('path=C:\\Users\\bob\\repo', 'Users\\bob', '[REDACTED_USER]'),
    *[(line, value, '[REDACTED_ENV_VAR]') for line, value in [
        ('OPENAI_API_KEY=sk-leakthis', 'sk-leakthis'), ('MY_API_KEY=value', 'value'),
        ('JWT_TOKEN=abc.def.ghi', 'abc.def.ghi'), ('DB_PASSWORD=hunter2', 'hunter2'),
        ('AUTH_TOKEN=t-foo', 't-foo'), ('CACHE_KEY=k-bar', 'k-bar'), ('DB_CREDENTIAL=admin:pw', 'admin:pw'),
    ]],
    *[(text, 'opaque-test-only-sentinel', '[REDACTED_CREDENTIAL]') for text in [
        '{"credentials": {"apiKey": "opaque-test-only-sentinel"}}', "{'client_secret': 'opaque-test-only-sentinel'}",
        'apiKey: opaque-test-only-sentinel', 'client-secret = opaque-test-only-sentinel',
        'apiKey:\n  nested: opaque-test-only-sentinel',
        '{\n "apiKey": {\n "nested": "opaque-test-only-sentinel"\n }\n}',
        '{"apiKey": {"nested": "opaque-test-only-sentinel"}}', 'text: apiKey: opaque-test-only-sentinel',
        'config: token=opaque-test-only-sentinel', '{"description": "use token=opaque-test-only-sentinel here"}',
        '1AUTHORIZATION = opaque-test-only-sentinel', '1apiKey: "opaque-test-only-sentinel"',
        '1apiKey:\n  nested: opaque-test-only-sentinel\n',
    ]],
    *[(text, '', '[REDACTED_CREDENTIAL]') for text in ['1apiKey=x', '2token= y', '123secret: z']],
    *[(f'{header}: {scheme}opaque-header-canary\nnext line stays', 'opaque-header-canary',
       f'{header}: {scheme if scheme == 'Bearer ' else ''}[REDACTED_CREDENTIAL]\nnext line stays')
      for header, scheme in [
        ('Authorization', ''), ('authorization', 'Bearer '), ('Proxy-Authorization', ''),
        ('X-Api-Key', ''), ('X-Auth-Token', ''), ('Cookie', 'session='), ('Set-Cookie', ''),
    ]],
    *[(text, 'opaque-token-xyz', 'Bearer [REDACTED_CREDENTIAL]') for text in [
        'curl -H "Authorization: Bearer opaque-token-xyz" https://api',
        '  authorization: Bearer opaque-token-xyz', 'saw Authorization: Bearer opaque-token-xyz in the log',
        'token: Bearer opaque-token-xyz',
    ]],
    *[(prefix + 'sk-opaque123' + suffix, 'sk-opaque123', prefix) for prefix, suffix in [
        ('note: apiKey: ', ''), ('note: "apiKey: ', '"'),
    ]],
    *[(key + separator + value, 'opaque-test-only-sentinel', key) for key, separator, value in [
        ('defauthorization', ': ', '"opaque-test-only-sentinel"'), ('nullpasswd', '=', '1'),
        ('fooapi_key', ': ', '2'), ('xpassword', '=', '3'), ('superclient_secret ', '= ', 'opaque-test-only-sentinel'),
        ('defauthorization', ':\n', '  nested: opaque-test-only-sentinel\n'),
        ('nullpasswd', ':\n', '  nested: opaque-test-only-sentinel\n'),
        ('12defauthorization', '= ', 'opaque-test-only-sentinel'), ('1nullpasswd', ': ', 'opaque-test-only-sentinel'),
    ]],
])
def test_text_redaction_preserves_public_structure(source: str, private: str, public: str) -> None:
    """Grammar belongs here; field routing and persisted bytes have separate keepers."""
    result = redact_structured_text(source)
    if private:
        assert private not in result
    assert public in result
    assert '[REDACTED' in result
    assert redact_structured_text(result) == result


@pytest.mark.parametrize('source', [
    'eyJhbG is a prefix', 'DEBUG=true\nAPP_NAME=myproject',
    'MONKEY_PATCH=enabled', 'KEYBOARD_LAYOUT=qwerty', 'AUTHOR=alice', 'TOKENIZED=foo', 'KEYSTORE=path/to/store',
    'the token: is now available', 'The authorization: feature is enabled now', 'defpasswd:\nplain: 1\n',
    'CLERK_SECRET_KEY=\nCLOUDFLARE_ACCOUNT_ID=\nCLOUDFLARE_API_TOKEN=\n'
    'CLOUDFLARE_ACCOUNT_HASH=\nINTERNAL_SERVICE_SECRET=',
    'Read ' + json.dumps('data: { "type":"record", "key": "public-quoted-key" }') + ' trailing prose',
    'Read \'data: { "type":"record", "key": "public-quoted-key" }\' trailing prose',
])
def test_public_text_redaction_is_identity(source: str) -> None:
    assert redact_structured_text(source) == source
    assert redaction_mod.redact_text(source) == source


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

    assert redaction_mod.redact_text("OPENAI_API_KEY=sk-leakthis123") == "[REDACTION_FAILED]"
    out = Redactor().redact_step(_user_step("OPENAI_API_KEY=sk-leakthis123"))
    assert isinstance(out.message, str)
    assert "sk-leakthis123" not in out.message
    assert "[REDACTION_FAILED]" in out.message


@pytest.mark.parametrize('fail_nested', [False, True])
def test_redactor_preserves_argument_structure_and_per_key_failure(
    monkeypatch: pytest.MonkeyPatch, fail_nested: bool,
) -> None:
    arguments = {'edits': [{'path': '/Users/alice/repo/file.py', 'edit_type': 'replace'},
                           {'path': '/Users/alice/repo/other.py', 'edit_type': 'delete'}],
                 'count': 42, 'flags': ['a', 'b'], 'config': {'x': 1, 'y': [1, 2, 3]}}
    original = copy.deepcopy(arguments)
    real = redactor_mod.redact_value
    def redact(value: object, sensitive: bool = False) -> object:
        if fail_nested and isinstance(value, list) and value and isinstance(value[0], dict):
            raise RuntimeError('nested argument failure')
        return real(value, sensitive=sensitive)
    monkeypatch.setattr(redactor_mod, 'redact_value', redact)
    step = Redactor().redact_step(_agent_step(tool_calls=[ToolCall(
        tool_call_id='edit', function_name='Edit', arguments=arguments,
    )]))
    assert step.tool_calls is not None
    result = step.tool_calls[0].arguments
    assert result['count'] == 42 and result['flags'] == ['a', 'b'] and result['config'] == arguments['config']
    expected = '[REDACTION_FAILED]' if fail_nested else [
        {'path': '/Users/[REDACTED_USER]/repo/file.py', 'edit_type': 'replace'},
        {'path': '/Users/[REDACTED_USER]/repo/other.py', 'edit_type': 'delete'},
    ]
    assert result['edits'] == expected
    assert arguments == original


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


_PEM_KEY_CASES = [
    (f"-----BEGIN {label}-----\nsynthetic-private-pem-body\n-----END {label}-----", "synthetic-private-pem-body")
    for label in ("RSA PRIVATE KEY", "PRIVATE KEY", "ENCRYPTED PRIVATE KEY", "OPENSSH PRIVATE KEY",
                  "EC PRIVATE KEY", "DSA PRIVATE KEY")
]
_PEM_KEY_IDS = ["pkcs1", "pkcs8", "encrypted", "openssh", "ec", "dsa"]

@pytest.mark.parametrize(("pem", "body"), _PEM_KEY_CASES, ids=_PEM_KEY_IDS)
@pytest.mark.parametrize("interrupted", [False, True])
def test_redactor_scrubs_private_key_block(pem: str, body: str, interrupted: bool) -> None:
    """All supported key grammars fail closed before a streamed footer arrives.

    Complete PEM cases cannot detect key bodies exposed by an interrupted tail.
    """
    if interrupted:
        pem = pem.rsplit("-----END", 1)[0]
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

@pytest.mark.parametrize("marker", ["[REDACTED_API_KEY]", "[redacted_api_key]", "<redacted>"])
def test_structured_redaction_leaves_an_existing_marker_alone(marker: str) -> None:
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


@pytest.mark.parametrize("non_secret_key", ["tokenizer", "passwordless", "monkeyPatch", "keyStore", "max_tokens"])
def test_redactor_preserves_non_sensitive_structured_keys(non_secret_key: str) -> None:
    call = ToolCall(tool_call_id="t1", function_name="Bash", arguments={non_secret_key: "opaque-test-only-sentinel"},)
    out = Redactor().redact_step(_agent_step(tool_calls=[call]))
    assert out.tool_calls is not None
    args = out.tool_calls[0].arguments
    assert args[non_secret_key] == "opaque-test-only-sentinel"
    assert "[REDACTED_CREDENTIAL]" not in json.dumps(args)


_JSON_KEY_PRIVACY_CASES = [
    *[(key, value, value, False) for key in [
        'defauthorization: opaque-key-canary', 'deftoken=opaque-key-canary', 'api_key=opaque-key-canary',
        'X-Api-Key: opaque-key-canary', 'private_key=opaque-key-canary',
        json.dumps({'type': 'api_key', 'key': 'opaque-key-canary'}, separators=(',', ':')),
    ] for value in [0, None, {'public': 'diagnostics'}]],
    *[(key, 'opaque-value-canary', '[REDACTED_CREDENTIAL]', False) for key in [
        'api_key', 'private_key', '[REDACTED_CREDENTIAL]', 'prefix_[REDACTED_TOKEN]',
        '[REDACTED]token', 'Authorization: Bearer [REDACTED_CREDENTIAL]',
    ]],
    *[(key, '[REDACTED_RETAINED]', '[REDACTED_RETAINED]', False) for key in [
        'Authorization: Bearer opaque-key-canary', 'auth=Bearer opaque-key-canary',
        'refresh_token="opaque-key-canary', 'api_key=[REDACTED]opaque-key-canary',
        '{"Authorization: Bearer opaque-key-canary":"public"}',
    ]],
    *[(key, '[REDACTED_RETAINED]', '[REDACTED_RETAINED]', True) for key in [
        'X-Api-Key: [REDACTED_API_KEY]', 'Cookie: [REDACTED_CREDENTIAL] (2)', 'Authorization: <redacted> (2)',
    ]],
]


@pytest.mark.parametrize(('key', 'value', 'expected', 'unchanged', 'envelope'), [
    *[(*case, 'object') for case in _JSON_KEY_PRIVACY_CASES],
    ('api_key=opaque-key-canary', 0, 0, False, 'scalar'),
    ('api_key', '[REDACTED_RETAINED]', '[REDACTED_RETAINED]', False, 'scalar'),
    ('{"type":"api_key","key":"opaque-key-canary"}', None, None, False, 'credential-parent'),
    ('Authorization: Bearer opaque-key-canary', '[REDACTED_RETAINED]', '[REDACTED_RETAINED]',
     False, 'credential-parent'),
])
def test_json_key_privacy_preserves_original_sensitivity_and_value_types(
    key: str, value: object, expected: object, unchanged: bool, envelope: str,
) -> None:
    """Bind original key sensitivity before sanitation; markers cannot retype public values."""
    payload = {key: value, '{"scope":"python"}': True, 'usage': {'input_tokens': 0, 'output_tokens': None}}
    source = json.dumps(payload)
    if envelope == 'scalar':
        source = json.dumps(source)
    elif envelope == 'credential-parent':
        source = json.dumps({'type': 'api_key', 'key': 'opaque-parent-canary', 'public': source})
    result = redact_structured_text(source)
    assert '-canary' not in result
    assert redact_structured_text(result) == result
    decoded = json.loads(result)
    decoded = json.loads(decoded) if envelope == 'scalar' else json.loads(decoded['public']) \
        if envelope == 'credential-parent' else decoded
    assert decoded['{"scope":"python"}'] is True
    assert decoded['usage'] == {'input_tokens': 0, 'output_tokens': None}
    assert [item for name, item in decoded.items() if name not in ('{"scope":"python"}', 'usage')] == [expected]
    if unchanged:
        assert key in decoded


@pytest.mark.parametrize(('template', 'safe', 'values', 'existing'), [
    ('Authorization: Bearer {}', 'Authorization: Bearer [REDACTED_CREDENTIAL]',
     ('[REDACTED_FIRST]', '[REDACTED_SECOND]'), '[REDACTED_EXISTING]'),
    ('api_key={}', '[REDACTED_LABEL]', ('[REDACTED_FIRST]', '[REDACTED_SECOND]'), '[REDACTED_EXISTING]'),
    ('Cookie: {}', 'Cookie: [REDACTED_CREDENTIAL]', ('[REDACTED_FIRST]', '[REDACTED_SECOND]'), '[REDACTED_EXISTING]'),
    ('{{"type":"api_key","key":"{}"}}', '[REDACTED_LABEL]', (0, None), 'reserved-public-value'),
])
def test_json_key_collisions_preserve_entries_and_reserved_public_keys(
    template: str, safe: str, values: tuple[object, object], existing: str,
) -> None:
    payload = {template.format('first-key-canary'): values[0], template.format('second-key-canary'): values[1],
               safe: existing, safe + ' (2)': '[REDACTED_RESERVED]',
               'usage': {'input_tokens': 0, 'output_tokens': None}}
    result = redact_structured_text(json.dumps(payload))
    assert '-canary' not in result and redact_structured_text(result) == result
    decoded = json.loads(result)
    assert len(decoded) == len(payload)
    assert decoded[safe] == existing and decoded[safe + ' (2)'] == '[REDACTED_RESERVED]'
    assert [value for key, value in decoded.items() if key not in (safe, safe + ' (2)', 'usage')] == list(values)
    assert decoded['usage'] == {'input_tokens': 0, 'output_tokens': None}


@pytest.mark.parametrize(('source', 'private', 'prefix'), [
    ('x' * 200_000, '', 'x' * 200_000),
    ('a' * 5000 + '=x token=sk-opaque123', 'sk-opaque123', 'a' * 5000 + '=x '),
    ('a' * 200_000 + '=x token=opaque-scan-canary', 'opaque-scan-canary', 'a' * 200_000 + '=x '),
    ('foo' + '_' * 100_000 + 'tail: x', '', 'foo' + '_' * 100_000 + 'tail: x'),
    ('foo' + '_' * 100_000 + 'xapi_key: opaque-scan-canary', 'opaque-scan-canary', 'foo' + '_' * 100_000 + 'x'),
    ('foo' + '_' * 100_000 + 'xapi_key: {\n "nested": "opaque-scan-canary"\n}',
     'opaque-scan-canary', 'foo' + '_' * 100_000 + 'x'),
])
def test_redaction_scan_is_bounded(source: str, private: str, prefix: str) -> None:
    start = time.process_time()
    result = redact_structured_text(source)
    assert time.process_time() - start < 5
    assert result.startswith(prefix)
    if private:
        assert private not in result and '[REDACTED' in result
    else:
        assert result == source


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


@pytest.mark.parametrize('payload', [
    {'message': chr(0xD800), 'public': 'café 😀', 'usage': {'input_tokens': 0, 'output_tokens': None}},
    {'nested': [{'message': chr(0xDFFF)}], 'public': 'café 😀'},
    {chr(0xD800) + 'clé': 'café 😀'},
    [chr(0xD800), chr(0xDFFF), 'café 😀', 0, None],
    chr(0xD800) + 'café 😀',
    {'message': 'literal ' + chr(92) + 'ud800', 'public': 'café 😀'},
    {'public': 'café 😀'},
    {'nested_json': json.dumps({'message': chr(0xD800), 'public': 'café 😀'}, ensure_ascii=False)},
    {'nested_json': json.dumps(chr(0xDFFF) + 'café 😀', ensure_ascii=False)},
])
def test_structured_json_surrogates_preserve_utf8_and_json_semantics(payload: object) -> None:
    source = json.dumps(payload)
    redacted = redact_structured_text(source)
    encoded = redacted.encode('utf-8')
    assert 'café 😀'.encode('utf-8') in encoded
    assert json.loads(redacted) == json.loads(source)
    assert redact_structured_text(redacted) == redacted


def test_plain_surrogate_text_is_safe_at_public_redaction_boundary() -> None:
    redacted = redact_structured_text('public café 😀 ' + chr(0xD800) + ' tail ' + chr(0xDFFF))
    assert redacted == 'public café 😀 ' + chr(92) + 'ud800 tail ' + chr(92) + 'udfff'
    assert redacted.encode('utf-8')
    assert redact_structured_text(redacted) == redacted


@pytest.mark.parametrize('container', [
    '["opaque-container-canary", "public"]',
    '{"entry": "opaque-container-canary", "public": true}',
])
def test_sensitive_container_redaction_is_not_replaced_by_marker_token_parsing(container: str) -> None:
    assert redact_structured_text('token = ' + container) == 'token = "[REDACTED_CREDENTIAL]"'


@pytest.mark.parametrize(('source', 'expected'), [
    ('{"token": null, "count": 3}', '{"token": "[REDACTED_CREDENTIAL]", "count": 3}'),
    ('token: abc, other: 1', 'token: "[REDACTED_CREDENTIAL]", other: 1'),
    ('API_KEY= sk-live-abc123', 'API_KEY=[REDACTED_ENV_VAR]'),
    ('TOKEN=abc\nPLAIN_SETTING=1', 'TOKEN=[REDACTED_ENV_VAR]\nPLAIN_SETTING=1'),
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
    redacted = redact_structured_text(source)
    assert redacted == expected
    assert redact_structured_text(redacted) == redacted


@pytest.mark.parametrize(('first_separator', 'second_separator', 'boundary'), [
    ('=' if (first + frontier) % 2 else ':', second, boundary)
    for first, second in enumerate(('=', ':', ': ', ':\t'))
    for frontier, boundary in enumerate((';', '; ', ' ;', ' ;\t'))
])
def test_sensitive_semicolon_pairs_preserve_both_separator_frontiers(
    first_separator: str, second_separator: str, boundary: str,
) -> None:
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
    redacted = redact_structured_text(name + '=opaque-prefix-canary;opaque-tail-canary')
    assert 'opaque-prefix-canary' not in redacted and 'opaque-tail-canary' not in redacted
    assert redact_structured_text(redacted) == redacted


@pytest.mark.parametrize(('kind', 'order', 'form', 'marked'), [
    *[(kind, 'type-first', form, False) for kind in ('api_key', 'oauth')
      for form in ('mapping', 'json', 'json-string', 'text-parts', 'arguments')],
    ('api_key', 'type-last', 'mapping', True),
    ('oauth', 'type-last', 'json', True),
])
def test_native_tagged_credentials_use_context_without_masking_public_keys(
    kind: str, order: str, form: str, marked: bool,
) -> None:
    credential: dict[str, object] = ({'type': kind, 'key': 'opaque-native-key-canary'} if kind == 'api_key' else
        {'type': kind, 'access': 'opaque-native-access-canary', 'refresh': 'opaque-native-refresh-canary',
         'accountId': 'offline-account', 'expires': 9000000000})
    field = 'key' if kind == 'api_key' else 'access'
    if marked:
        credential[field] = '[REDACTED_API_KEY]' + str(credential[field])
    if order == 'type-last':
        credential = dict(reversed(list(credential.items())))
    public = {'type': 'record', 'key': 'ordinary-public-key', 'access': 'ordinary-public-access',
              'refresh': 'ordinary-public-refresh'}
    payload = {'data': credential, 'public': public, 'usage': {'input': 0, 'output': None}}
    original = copy.deepcopy(payload)
    if form == 'mapping':
        result = redact_value(payload)
        assert redact_value(result) == result
    elif form == 'arguments':
        out = Redactor().redact_step(_agent_step(tool_calls=[ToolCall(
            tool_call_id='auth', function_name='Inspect', arguments=credential,
        )]))
        assert out.tool_calls is not None
        result = {**payload, 'data': out.tool_calls[0].arguments}
    elif form == 'text-parts':
        out = Redactor().redact_step(_agent_step(observation=Observation(results=[ObservationResult(
            content=[ContentPart(type='text', text=json.dumps(payload))],
        )])))
        assert out.observation is not None
        parts = out.observation.results[0].content
        assert isinstance(parts, list) and isinstance(parts[0].text, str)
        result = json.loads(parts[0].text)
    else:
        text = json.dumps(payload)
        text = json.dumps(text) if form == 'json-string' else text
        redacted = redact_structured_text(text)
        assert redact_structured_text(redacted) == redacted
        result = json.loads(redacted)
        result = json.loads(result) if form == 'json-string' else result
    assert '-canary' not in json.dumps(result)
    assert result['public'] == public
    assert result['usage'] == {'input': 0, 'output': None}
    assert result['data']['type'] == kind
    if marked:
        assert result['data'][field] == '[REDACTED_API_KEY]'
    if kind == 'oauth':
        assert result['data']['accountId'] == 'offline-account' and result['data']['expires'] == 9000000000
    assert payload == original
    safe = json.dumps({'type': kind, field: '[REDACTED_API_KEY]'})[:-1]
    assert redact_structured_text(safe) == safe
    assert redact_structured_text(safe[:-1]) == safe[:-1]


_NATIVE_FRAGMENT = '{"type":"KIND","FIELD":"opaque-context-canary'


_NATIVE_AUTH_FRAGMENTS = [
    *[(prefix + text, 'opaque-context-canary', ()) for prefix in ('', '{"data":') for text in (
        _NATIVE_FRAGMENT,
        '{"FIELD":"opaque-context-canary","type":"KIND"',
        r'{"ty\u0070e":"ESC_KIND","ESC_FIELD":"opaque-context-canary',
        r'{"ESC_FIELD":"opaque-context-canary","ty\u0070e":"ESC_KIND"',
    )],
    ('{"type":"KIND","FIELD":"opaque-context-canary","type":"record"}', 'opaque-context-canary', ()),
    ('{"type":"api_key","key":"opaque-context-canary","type":"oauth","access":"opaque-context-canary"}',
     'opaque-context-canary', ()),
    ('{"typed":{"type":"KIND","FIELD":"opaque-context-canary"},'
     '"public":{"key":"public-sibling","access":"public-access","refresh":"public-refresh"},'
     '"nested":{"type":"KIND","metadata":{"key":"public-nested"}}} trailing',
     'opaque-context-canary', ('public-sibling', 'public-access', 'public-refresh', 'public-nested')),
    ('{"key":"public-before-type', '', ('public-before-type',)),
    ('{"key":"public-sibling","note":"' + json.dumps('{"type":"KIND"}')[1:-1] + '"} trailing',
     '', ('public-sibling',)),
    *[(prefix + _NATIVE_FRAGMENT, 'opaque-context-canary', (prefix,)) for prefix in [
        "Here's the data: ", 'Read "data: ', "it's here\n", 'prefix: ', 'Read "', 'Read "   ', '"',
    ]],
    (json.dumps(_NATIVE_FRAGMENT)[:-1], 'opaque-context-canary', ()),
    ('{"key":"public-outer-key","message":' + json.dumps(_NATIVE_FRAGMENT)[:-1],
     'opaque-context-canary', ('public-outer-key',)),
    ("Here's the data: " + json.dumps(_NATIVE_FRAGMENT)[:-1], 'opaque-context-canary', ("Here's the data: ",)),
    *[(source, 'u004', public) for marker in ['', '[REDACTED]']
      for inner in ['{"type":"KIND","FIELD":"' + marker + chr(92) + 'u004']
      for source, public in [(inner, ()), (json.dumps(inner)[:-1], ()),
          ('{"public":{"key":"ordinary-key"},"message":' + json.dumps(inner)[:-1], ('ordinary-key',))]],
    (json.dumps('{"type":"KIND","FIELD":"[REDACTED]é').rsplit('u00e9', 1)[0] + 'u00e', 'u00e', ()),
    ('{"key":"[REDACTED]' + chr(92) + 'u004', '', (chr(92) + 'u004',)),
]


_QUOTED_NATIVE_AUTH_FRAGMENTS = [
    (kind, source, 'opaque-context-canary', ('Read ', 'data: '))
    for kind, credential in [
        ('api_key', {'type': 'api_key', 'key': 'opaque-context-canary'}),
        ('oauth', {'type': 'oauth', 'access': 'opaque-context-canary', 'refresh': 'opaque-context-canary'}),
    ]
    for source in (
        'Read ' + json.dumps('data: ' + json.dumps(credential)),
        'Read ' + json.dumps('data: ' + json.dumps(credential)) + ' trailing prose',
        'Read ' + json.dumps('data: ' + json.dumps(credential)[:-2]),
        'Read ' + json.dumps('data: ' + json.dumps(credential)[:-2])[:-1],
        'Read ' + json.dumps('outer prose: ' + json.dumps('data: ' + json.dumps(credential))),
    )
]


@pytest.mark.parametrize(('kind', 'source', 'private', 'public'), [
    *[('api_key', *case) for case in _NATIVE_AUTH_FRAGMENTS],
    *_QUOTED_NATIVE_AUTH_FRAGMENTS,
    ('oauth', _NATIVE_FRAGMENT, 'opaque-context-canary', ()),
    ('oauth', '{"FIELD":"opaque-context-canary","type":"KIND"', 'opaque-context-canary', ()),
    ('oauth', '{"type":"KIND","refresh":"opaque-context-canary', 'opaque-context-canary', ()),
])
def test_native_auth_fragments_respect_observed_scope(
    kind: str, source: str, private: str, public: tuple[str, ...],
) -> None:
    field = 'key' if kind == 'api_key' else 'access'
    text = source.replace('ESC_KIND', chr(92) + 'u%04x' % ord(kind[0]) + kind[1:]) \
        .replace('ESC_FIELD', chr(92) + 'u%04x' % ord(field[0]) + field[1:]) \
        .replace('KIND', kind).replace('FIELD', field)
    result = redact_structured_text(text)
    if private:
        assert private not in result and result != text
    else:
        assert result == text
    assert all(value in result for value in public)
    assert redact_structured_text(result) == result


@pytest.mark.parametrize('value', [0, None, ''])
def test_invalid_nonstring_native_credential_fields_remain_typed(value: object) -> None:
    payload = {'data': {'type': 'api_key', 'key': value}, 'other': {'type': 'oauth', 'access': value, 'refresh': value}}
    assert redact_value(payload) == payload
    assert json.loads(redact_structured_text(json.dumps(payload))) == payload


def test_contextual_credentials_do_not_relax_existing_sensitive_ancestor_policy() -> None:
    credential = {'type': 'api_key', 'key': 'opaque-parent-auth-canary', 'metadata': 'private-parent-value'}
    redacted = redact_value({'auth': credential})
    assert redacted['auth']['type'] == '[REDACTED_CREDENTIAL]'
    assert redacted['auth']['metadata'] == '[REDACTED_CREDENTIAL]'
    assert 'opaque-parent-auth-canary' not in json.dumps(redacted)


@pytest.mark.parametrize(('kind', 'form'), [
    ('api_key', 'mapping'), ('oauth', 'mapping'), ('api_key', 'json'), ('oauth', 'json'),
    ('api_key', 'scalar'), ('api_key', 'text-list'),
    ('api_key', 'quoted-log'), ('oauth', 'quoted-log'),
])
def test_native_only_copy_preserves_ordinary_ui_arguments_and_source(kind: str, form: str) -> None:
    field = 'key' if kind == 'api_key' else 'access'
    metadata = 'config: token=ordinary-search-pattern /home/operator/file.py'
    credential = {'type': kind, field: '[REDACTED_API_KEY]opaque-ui-context-canary', 'metadata': metadata}
    public = {'key': 'ordinary-key', 'access': 'ordinary-access', 'refresh': 'ordinary-refresh',
              'message': 'café 😀' if form == 'mapping' else chr(0xD800), chr(0xDFFF) + 'clé': 'café 😀'}
    nested = json.dumps({'message': chr(0xD800), 'public': 'café 😀'}, ensure_ascii=form == 'mapping')
    payload = {**credential, 'data': credential, 'public': public, 'nested': nested,
               'usage': {'input': 0, 'output': None}}
    source: object = payload if form == 'mapping' else json.dumps(payload)
    if form == 'scalar':
        source = json.dumps(source)
    elif form == 'text-list':
        source = [source, metadata]
    elif form == 'quoted-log':
        assert isinstance(source, str)
        source = 'Read ' + json.dumps('data: ' + source)
    original = copy.deepcopy(source)
    result = redaction_mod.redact_native_auth_value(source)
    assert 'opaque-ui-context-canary' not in json.dumps(result)
    assert redaction_mod.redact_native_auth_value(result) == result and source == original
    decoded = result
    if form == 'text-list':
        assert isinstance(decoded, list) and decoded[1] == metadata
        decoded = decoded[0]
    if isinstance(decoded, str):
        decoded.encode('utf-8')
    if form == 'quoted-log':
        assert isinstance(decoded, str) and decoded.startswith('Read ')
        decoded = json.loads(decoded.removeprefix('Read '))
        assert isinstance(decoded, str) and decoded.startswith('data: ')
        decoded = decoded.removeprefix('data: ')
    while isinstance(decoded, str):
        decoded = json.loads(decoded)
    assert decoded['public'] == public and decoded['nested'] == nested
    assert decoded['usage'] == {'input': 0, 'output': None}
    assert decoded['data']['metadata'] == metadata
    assert decoded[field] == decoded['data'][field] == '[REDACTED_API_KEY]'
    containers = [{'key': 'ordinary-key'}, ({'key': 'ordinary-tuple-key'},)]
    copied = redaction_mod.redact_native_auth_value(containers)
    assert isinstance(copied, list) and isinstance(copied[1], tuple)
    copied[0]['key'] = 'changed presentation copy'
    assert containers[0] == {'key': 'ordinary-key'}


@pytest.mark.parametrize('nonfinite', ['NaN', 'Infinity', '-Infinity'])
def test_native_only_text_recovery_withholds_unrepresentable_credential_span(nonfinite: str) -> None:
    text = ('{"data":{"type":"api_key","key":"opaque-nonfinite-auth-canary","public":' + nonfinite
            + '},"other":{"key":"ordinary-sibling","count":0}} trailing')
    redacted = redaction_mod.redact_native_auth_value(text)
    assert 'opaque-nonfinite-auth-canary' not in redacted
    assert 'ordinary-sibling' in redacted and '"count":0' in redacted
    assert redaction_mod.redact_native_auth_value(redacted) == redacted
    ordinary = '{"key":"ordinary-key","public":' + nonfinite + '}'
    assert redaction_mod.redact_native_auth_value(ordinary) == ordinary
