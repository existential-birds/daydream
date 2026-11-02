"""Redaction positives, clean-input negatives, field coverage, and fail-closed behavior."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable
from typing import Any

import pytest

from daydream import redaction as redaction_mod
from daydream.atif import ContentPart, Observation, ObservationResult, Step, ToolCall
from daydream.atif.models.content import ImageSource
from daydream.trajectory import Redactor, now_iso, redact_structured_text, redact_value, redactor as redactor_mod


def _check_case(check: Callable[..., None], case: tuple[Any, ...]) -> None:
    """Run one table row, naming it in the failure report."""
    try:
        check(*case)
    except BaseException as error:
        error.add_note(f"redaction case: {case!r}")
        raise


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


def test_redactor_fails_closed_per_rule_per_key_and_per_field(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BoomPattern:
        def sub(self, *_args: object, **_kwargs: object) -> str:
            raise RuntimeError("boom")

    # A failing first rule replaces the whole text.
    with monkeypatch.context() as patch:
        patch.setattr(redaction_mod, "_REDACTION_RULES",
                      ((_BoomPattern(), "[REDACTED_API_KEY]"), *redaction_mod._REDACTION_RULES[1:]))
        assert redaction_mod.redact_text("OPENAI_API_KEY=sk-leakthis123") == "[REDACTION_FAILED]"
        out = Redactor().redact_step(_user_step("OPENAI_API_KEY=sk-leakthis123"))
        assert isinstance(out.message, str)
        assert "sk-leakthis123" not in out.message
        assert "[REDACTION_FAILED]" in out.message

    # A failing nested argument loses only that key; siblings and the caller's mapping survive.
    arguments = {'edits': [{'path': '/Users/alice/repo/file.py', 'edit_type': 'replace'},
                           {'path': '/Users/alice/repo/other.py', 'edit_type': 'delete'}],
                 'count': 42, 'flags': ['a', 'b'], 'config': {'x': 1, 'y': [1, 2, 3]}}
    original = copy.deepcopy(arguments)
    real = redactor_mod.redact_value
    def redact(value: object, sensitive: bool = False) -> object:
        if isinstance(value, list) and value and isinstance(value[0], dict):
            raise RuntimeError('nested argument failure')
        return real(value, sensitive=sensitive)
    with monkeypatch.context() as patch:
        patch.setattr(redactor_mod, 'redact_value', redact)
        step = Redactor().redact_step(_agent_step(tool_calls=[ToolCall(
            tool_call_id='edit', function_name='Edit', arguments=arguments,
        )]))
    assert step.tool_calls is not None
    result = step.tool_calls[0].arguments
    assert result['count'] == 42 and result['flags'] == ['a', 'b'] and result['config'] == arguments['config']
    assert result['edits'] == '[REDACTION_FAILED]'
    assert arguments == original

    # A failure deep in the step pipeline wipes every text-bearing field.
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


def test_json_key_collisions_preserve_entries_and_reserved_public_keys() -> None:
    for case in (
        ('Cookie: {}', 'Cookie: [REDACTED_CREDENTIAL]', ('[REDACTED_FIRST]', '[REDACTED_SECOND]'),
         '[REDACTED_EXISTING]'),
        ('{{"type":"api_key","key":"{}"}}', '[REDACTED_LABEL]', (0, None), 'reserved-public-value'),
    ):
        _check_case(_assert_json_key_collisions_preserved, case)


def _assert_json_key_collisions_preserved(template: str, safe: str, values: tuple[object, object], existing: str,
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


def test_structured_text_boundaries_are_bounded_idempotent_and_identity_on_public_text() -> None:
    public = 'defpasswd:\nplain: 1\n'
    assert redact_structured_text(public) == public
    assert redaction_mod.redact_text(public) == public
    separators = 'foo' + '_' * 100_000 + 'tail: x'
    start = time.process_time()
    assert redact_structured_text(separators) == separators
    assert time.process_time() - start < 5
    long_key = 'a' * 200_000 + '=x token=opaque-scan-canary'
    start = time.process_time()
    long_result = redact_structured_text(long_key)
    assert time.process_time() - start < 5
    assert long_result.startswith('a' * 200_000 + '=x token=') and 'opaque-scan-canary' not in long_result
    # An internal semicolon cannot expose the residual of a sensitive assignment.
    for name in ('api_key', 'TOKEN'):
        residual = redact_structured_text(name + '=opaque-prefix-canary;opaque-tail-canary')
        assert 'opaque-' not in residual and redact_structured_text(residual) == residual, name
    assert redact_structured_text('TOKEN=a;API_KEY=b') == 'TOKEN=[REDACTED_ENV_VAR];API_KEY=[REDACTED_ENV_VAR]'
    # An unterminated secret leaf inside a JSON string keeps the envelope valid and its siblings intact.
    envelope = json.loads(redact_structured_text(json.dumps({'message': 'refresh_token="leaf-canary', 'count': 0})))
    assert 'leaf-canary' not in json.dumps(envelope) and envelope['count'] == 0
    redacted = redact_structured_text('refresh_token:[REDACTED]opaque-first-canary&api_key=opaque-second-canary')
    assert redacted == 'refresh_token:[REDACTED]&api_key="[REDACTED_CREDENTIAL]"'
    assert redact_structured_text(redacted) == redacted
    # Inline brace/bracket values are consumed whole, quote-aware, through their matching close or text end.
    for source, expected in (
        ('log apiKey: {"nested": "opaque-canary", "q": "a\\"}b", "list": [1, {"x": 2}]} tail',
         'log apiKey: "[REDACTED_CREDENTIAL]" tail'),
        ("apiKey = ['opaque-canary', {'k': ']'}] tail", 'apiKey = "[REDACTED_CREDENTIAL]" tail'),
        ('apiKey: {"unbalanced": "opaque-canary"', 'apiKey: "[REDACTED_CREDENTIAL]"'),
    ):
        assert redact_structured_text(source) == expected


_NATIVE_TAGGED_CASES = (
    *[('oauth', 'type-first', form, False) for form in ('mapping', 'json', 'json-string', 'text-parts', 'arguments')],
    ('api_key', 'type-last', 'mapping', True),
    ('oauth', 'type-last', 'json', True),
)


def test_native_tagged_credentials_use_context_without_masking_public_keys() -> None:
    for case in _NATIVE_TAGGED_CASES:
        _check_case(_assert_native_tagged_credential_redacted, case)


def _assert_native_tagged_credential_redacted(kind: str, order: str, form: str, marked: bool) -> None:
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


_NATIVE_AUTH_FRAGMENT_CASES = (
    *[('api_key', *case) for case in _NATIVE_AUTH_FRAGMENTS],
    *_QUOTED_NATIVE_AUTH_FRAGMENTS,
    ('oauth', _NATIVE_FRAGMENT, 'opaque-context-canary', ()),
    ('oauth', '{"FIELD":"opaque-context-canary","type":"KIND"', 'opaque-context-canary', ()),
    ('oauth', '{"type":"KIND","refresh":"opaque-context-canary', 'opaque-context-canary', ()),
)


def test_native_auth_fragments_respect_observed_scope() -> None:
    for case in _NATIVE_AUTH_FRAGMENT_CASES:
        _check_case(_assert_native_auth_fragment_scope, case)


def _assert_native_auth_fragment_scope(kind: str, source: str, private: str, public: tuple[str, ...]) -> None:
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


def test_native_only_copy_preserves_ordinary_ui_arguments_and_source() -> None:
    """A quoted log carrying a native OAuth record redacts only the credential, not UI data or lone surrogates."""
    metadata = 'config: token=ordinary-search-pattern /home/operator/file.py'
    credential = {'type': 'oauth', 'access': '[REDACTED_API_KEY]opaque-ui-context-canary', 'metadata': metadata}
    public = {'key': 'ordinary-key', 'access': 'ordinary-access', 'refresh': 'ordinary-refresh',
              'message': chr(0xD800), chr(0xDFFF) + 'clé': 'café 😀'}
    nested = json.dumps({'message': chr(0xD800), 'public': 'café 😀'}, ensure_ascii=False)
    payload = {**credential, 'data': credential, 'public': public, 'nested': nested,
               'usage': {'input': 0, 'output': None}}
    source = 'Read ' + json.dumps('data: ' + json.dumps(payload))
    original = copy.deepcopy(source)
    result = redaction_mod.redact_native_auth_value(source)
    assert 'opaque-ui-context-canary' not in json.dumps(result)
    assert redaction_mod.redact_native_auth_value(result) == result and source == original
    decoded = result
    assert isinstance(decoded, str) and decoded.startswith('Read ')
    decoded.encode('utf-8')
    decoded = json.loads(decoded.removeprefix('Read '))
    assert isinstance(decoded, str) and decoded.startswith('data: ')
    decoded = decoded.removeprefix('data: ')
    while isinstance(decoded, str):
        decoded = json.loads(decoded)
    assert decoded['public'] == public and decoded['nested'] == nested
    assert decoded['usage'] == {'input': 0, 'output': None}
    assert decoded['data']['metadata'] == metadata
    assert decoded['access'] == decoded['data']['access'] == '[REDACTED_API_KEY]'
    containers = [{'key': 'ordinary-key'}, ({'key': 'ordinary-tuple-key'},)]
    copied = redaction_mod.redact_native_auth_value(containers)
    assert isinstance(copied, list) and isinstance(copied[1], tuple)
    copied[0]['key'] = 'changed presentation copy'
    assert containers[0] == {'key': 'ordinary-key'}


def test_native_only_text_recovery_withholds_unrepresentable_credential_span() -> None:
    nonfinite = 'NaN'
    text = ('{"data":{"type":"api_key","key":"opaque-nonfinite-auth-canary","public":' + nonfinite
            + '},"other":{"key":"ordinary-sibling","count":0}} trailing')
    redacted = redaction_mod.redact_native_auth_value(text)
    assert 'opaque-nonfinite-auth-canary' not in redacted
    assert 'ordinary-sibling' in redacted and '"count":0' in redacted
    assert redaction_mod.redact_native_auth_value(redacted) == redacted
    ordinary = '{"key":"ordinary-key","public":' + nonfinite + '}'
    assert redaction_mod.redact_native_auth_value(ordinary) == ordinary


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


def test_credential_rotation_diagnostic_redacts_independent_documents_and_preserves_siblings() -> None:
    """One native record retains its certificate, paths and config structure while removing every credential family."""
    certificate = "-----BEGIN CERTIFICATE-----\nMIIDdzCCAl+gAwIBAgIEAgAAuTANBgkq\n-----END CERTIFICATE-----"
    out = Redactor().redact_step(
        _agent_step(
            message=(
                f"RSA: {_PKCS1_PEM}\nPKCS8: {_PKCS8_PEM}\nEncrypted: {_ENCRYPTED_PEM}\n"
                f"SSH: {_OPENSSH_PEM}\nEC: {_EC_PEM}\nDSA: {_DSA_PEM}\nCertificate: {certificate}\n"
                "path=/Users/ka/github/proj/app.py\npath=/home/alice/foo/bar\npath=C:\\Users\\bob\\repo\n"
                f"Streamed: {_OPENSSH_PEM.rsplit('-----END', 1)[0]}"
            ),
            reasoning_content=(
                f"thought: sk-test-secret123abc\nRSA_PRIVATE_KEY={_PKCS1_PEM}\nPKCS8_PRIVATE_KEY={_PKCS8_PEM}\n"
                f"ENCRYPTED_PRIVATE_KEY={_ENCRYPTED_PEM}\nSSH_PRIVATE_KEY={_OPENSSH_PEM}\n"
                f"EC_PRIVATE_KEY={_EC_PEM}\nDSA_PRIVATE_KEY={_DSA_PEM}\n"
                "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.aBcDeF12345\n"
                "installation ghs_abc123DEF456ghi789jkl012; Slack xoxb-test456abcdef\n"
                "MONKEY_PATCH=enabled\nKEYBOARD_LAYOUT=qwerty\nAUTHOR=alice\nTOKENIZED=yes\nKEYSTORE=/etc/ks"
            ),
            tool_calls=[ToolCall(tool_call_id="config", function_name="Configure", arguments={
                "password": "opaque-plain-canary", "auth": {"metadata": "opaque-nested-canary"},
                "max_tokens": 5, "tokenizer": "bpe", "passwordless": True, "keyStore": "ks",
            })],
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
        "path=/home/[REDACTED_USER]/foo/bar\npath=C:\\Users\\[REDACTED_USER]\\repo\n"
        "Streamed: [REDACTED_PEM_KEY]"
    )
    assert out.reasoning_content == (
        "thought: [REDACTED_API_KEY]\nRSA_PRIVATE_KEY=[REDACTED_ENV_VAR]\nPKCS8_PRIVATE_KEY=[REDACTED_ENV_VAR]\n"
        "ENCRYPTED_PRIVATE_KEY=[REDACTED_ENV_VAR]\nSSH_PRIVATE_KEY=[REDACTED_ENV_VAR]\n"
        "EC_PRIVATE_KEY=[REDACTED_ENV_VAR]\nDSA_PRIVATE_KEY=[REDACTED_ENV_VAR]\n"
        "jwt [REDACTED_JWT]\ninstallation [REDACTED_API_KEY]; Slack [REDACTED_API_KEY]\n"
        # Lookalike names are ordinary configuration, not credentials.
        "MONKEY_PATCH=enabled\nKEYBOARD_LAYOUT=qwerty\nAUTHOR=alice\nTOKENIZED=yes\nKEYSTORE=/etc/ks"
    )
    # Sensitive keys redact plain values (and everything beneath them); lookalike keys keep their data.
    assert out.tool_calls is not None and out.tool_calls[0].arguments == {
        "password": "[REDACTED_CREDENTIAL]", "auth": {"metadata": "[REDACTED_CREDENTIAL]"},
        "max_tokens": 5, "tokenizer": "bpe", "passwordless": True, "keyStore": "ks",
    }
    assert out.observation is not None
    assert [(result.source_call_id, result.content) for result in out.observation.results] == [
        (
            "json-config",
            '{"credentials": "[REDACTED_CREDENTIAL]", "profiles": ['
            '{"apiKey": "[REDACTED_CREDENTIAL]"}, {"apiKey": "[REDACTED_CREDENTIAL]"}], "enabled": true}',
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
