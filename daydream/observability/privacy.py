"""The content and diagnostic boundary for an owned tracing session."""

from __future__ import annotations

import json
import logging
import os
import re
import string
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from urllib.parse import unquote, urlsplit

from daydream.trajectory import redact_structured_text, redact_value

#: A secret-named variable holding one of these (``SOME_AUTH=1``) is a feature
#: flag, not a credential: harvesting it would literal-replace the token across
#: all span content (``{"ok":true}`` -> ``{"ok":[REDACTED_CREDENTIAL]}``) while
#: protecting nothing.
_FLAG_VALUE_TOKENS = frozenset(
    {"true", "false", "yes", "no", "on", "off", "1", "0", "enabled", "disabled", "null", "none"}
)

#: Resource keys whose values are treated as credentials: the key is retained
#: but the value is replaced before export. Key names themselves never carry
#: secrets; values under these names do. The stems match credential-denoting
#: names (``api_key``, ``AUTH_TOKEN``, ``secret``, ``bearer``) without claiming
#: every key-shaped name such as ``unicode.key``.
_RESOURCE_SECRET_KEY = re.compile(r"api[_-]?key|auth|token|secret|passw(or)?d|credential|bearer|private[_-]?key", re.I)

_HEX_DIGITS = frozenset(string.hexdigits)

#: Fixed diagnostic for a rejected operator resource variable. It never echoes
#: a pair, key, or value fragment.
RESOURCE_DIAGNOSTIC = (
    "OTEL_RESOURCE_ATTRIBUTES ignored: malformed, invalid, or duplicate operator resource settings; "
    "using Daydream defaults"
)


class ResourceParseError(ValueError):
    """A strict operator-resource parse failure; never carries input fragments."""


def _percent_decode(text: str) -> str:
    """Decode OTel-style percent escapes strictly as UTF-8; no lenient fallback."""
    raw = bytearray()
    index = 0
    while index < len(text):
        char = text[index]
        if char == "%":
            escape = text[index + 1 : index + 3]
            if len(escape) != 2 or any(item not in _HEX_DIGITS for item in escape):
                raise ResourceParseError()
            raw.append(int(escape, 16))
            index += 3
        else:
            raw.extend(char.encode("utf-8"))
            index += 1
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ResourceParseError() from None


def _has_control_character(text: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in text)


def parse_operator_resource_attributes(raw: str) -> dict[str, str]:
    """Parse one operator resource variable; reject the whole value on any error.

    Every accepted value stays a string. A malformed pair, malformed percent
    escape, empty or control-character key, control character in a value, or a
    duplicate key (including aliases that collide only after percent decoding)
    raises :class:`ResourceParseError` — callers discard the entire variable.
    """
    attributes: dict[str, str] = {}
    if not raw:
        return attributes
    for pair in raw.split(","):
        if not pair:
            continue
        key, separator, value = pair.partition("=")
        if not separator:
            raise ResourceParseError()
        decoded_key = _percent_decode(key)
        decoded_value = _percent_decode(value)
        if not decoded_key or _has_control_character(decoded_key) or _has_control_character(decoded_value):
            raise ResourceParseError()
        if decoded_key in attributes:
            raise ResourceParseError()
        attributes[decoded_key] = decoded_value
    return attributes


def sanitize_operator_resource_attributes(
    attributes: Mapping[str, str], policy: PrivacyPolicy
) -> dict[str, str]:
    """Return the privacy-filtered copy of accepted operator resource entries.

    Secret-like keys keep a stable key with the value replaced; all other keys
    and values are scrubbed of operator secret literals. Keys and values remain
    strings.
    """
    sanitized: dict[str, str] = {}
    for key, value in attributes.items():
        safe_key = policy.text(key)
        if _RESOURCE_SECRET_KEY.search(safe_key):
            sanitized[safe_key] = "[REDACTED_CREDENTIAL]"
        else:
            sanitized[safe_key] = policy.text(value)
    return sanitized


class PrivacyPolicy:
    """Redact structured secrets and literal credentials known to the operator."""

    def __init__(self, capture_content: bool = True, environ: Mapping[str, str] | None = None) -> None:
        self.capture_content = capture_content
        secrets: set[str] = set()
        for name, value in (os.environ if environ is None else environ).items():
            if value and value.casefold() not in _FLAG_VALUE_TOKENS and re.search(
                r"(?:KEY|TOKEN|SECRET|PASSWORD|AUTH|HEADERS)", name, re.I
            ):
                secrets.add(value)
                if "HEADERS" in name:
                    for header in value.split(","):
                        if "=" in header:
                            credential = unquote(header.split("=", 1)[1]).strip()
                            if credential:
                                secrets.add(credential)
                                if " " in credential:
                                    secrets.add(credential.split(" ", 1)[1])
                        elif header.strip():
                            # OTel logs malformed comma-delimited fragments.
                            secrets.add(unquote(header.strip()))
            if "ENDPOINT" in name or name.endswith("_URL"):
                try:
                    parsed = urlsplit(value)
                    for url_credential in (parsed.username, parsed.password):
                        if url_credential:
                            secrets.add(url_credential)
                            secrets.add(unquote(url_credential))
                except ValueError:
                    pass
        self._secrets = sorted(secrets, key=len, reverse=True)

    def text(self, value: str) -> str:
        """Scrub free text before it crosses the telemetry boundary."""
        for secret in self._secrets:
            value = value.replace(secret, "[REDACTED_CREDENTIAL]")
        return redact_structured_text(value)

    def value(self, value: Any) -> Any:
        """Build a JSON-safe sanitized copy; arbitrary objects fail closed."""
        try:
            # Serialization rejects unsupported objects and detects cycles before
            # the recursive shared redactor sees them. Never use repr/default=str.
            plain = json.loads(json.dumps(value, allow_nan=False))
            redacted = redact_value(plain)
            return self._literals(redacted)
        except Exception:
            return "[UNSERIALIZABLE]"

    def _literals(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {self.text(k): self._literals(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._literals(item) for item in value]
        return value

    def json(self, value: Any) -> str:
        """Encode safe structured content without truncating it."""
        return json.dumps(self.value(value), ensure_ascii=False, separators=(",", ":"))


_diagnostic_policy: ContextVar[PrivacyPolicy | None] = ContextVar("daydream_telemetry_diagnostics", default=None)
_record_factory_installed = False


def _install_diagnostic_boundary() -> None:
    global _record_factory_installed
    if _record_factory_installed:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        policy = _diagnostic_policy.get()
        if policy is not None and record.name.startswith(("opentelemetry", "traceloop", "requests", "urllib3")):
            if policy.capture_content:
                try:
                    message = record.getMessage()
                    if record.exc_info:
                        message += "\n" + "".join(traceback.format_exception(*record.exc_info))
                    if record.stack_info:
                        message += "\n" + record.stack_info
                    record.msg = policy.text(message)
                except Exception:
                    record.msg = "Telemetry diagnostic could not be formatted safely"
            else:
                record.msg = "Telemetry transport diagnostic (content capture disabled)"
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return record

    logging.setLogRecordFactory(factory)
    _record_factory_installed = True


@contextmanager
def diagnostic_scope(policy: PrivacyPolicy) -> Iterator[None]:
    """Sanitize only telemetry logs emitted in this operation/thread's context."""
    _install_diagnostic_boundary()
    token = _diagnostic_policy.set(policy)
    try:
        yield
    finally:
        _diagnostic_policy.reset(token)
