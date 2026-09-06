"""The content and diagnostic boundary for an owned tracing session."""

from __future__ import annotations

import json
import logging
import os
import re
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from urllib.parse import unquote, urlsplit

from daydream.trajectory import redact_structured_text, redact_value


class PrivacyPolicy:
    """Redact structured secrets and literal credentials known to the operator."""

    def __init__(self, capture_content: bool = True, environ: Mapping[str, str] | None = None) -> None:
        self.capture_content = capture_content
        secrets: set[str] = set()
        for name, value in (os.environ if environ is None else environ).items():
            if value and re.search(r"(?:KEY|TOKEN|SECRET|PASSWORD|AUTH|HEADERS)", name, re.I):
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
