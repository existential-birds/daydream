"""Fail-closed redaction of the transcript text and tool-result surfaces."""

from __future__ import annotations

from typing import Any

from daydream import ui
from daydream.redaction import (
    _is_native_credential_field as _is_native_credential_field,
    _is_sensitive_key as _is_sensitive_key,
    _redact_sensitive_text as _redact_sensitive_text,
    redact_structured_text as redact_structured_text,
    redact_value as redact_value,
)

_console = ui.create_console()

_FAILED = "[REDACTION_FAILED]"


def _redact_text_parts(parts: list[Any]) -> list[Any]:
    """Redact text parts in a message or observation, preserving non-text parts."""
    return [
        {**part, "text": redact_structured_text(part["text"]) if part.get("text") is not None else None}
        if isinstance(part, dict) and part.get("type") == "text" else part
        for part in parts
    ]


def _redact_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Redact every tool-call argument with key-aware container redaction.

    Nested credentials under lower/mixed/camel-case keys are replaced without a JSON
    round-trip; a failure on one key degrades only that key (REDA-05)."""
    out: dict[str, Any] = {}
    for key, val in arguments.items():
        try:
            out[key] = (_redact_sensitive_text(val) if _is_native_credential_field(key, arguments)
                        else redact_value(val, _is_sensitive_key(key) if isinstance(key, str) else False))
        except Exception:  # noqa: BLE001 - REDA-05 redact-or-omit
            out[key] = _FAILED
    return out


def redact_result_content(result: dict[str, Any]) -> dict[str, Any]:
    """Return *result* with its text content redacted; failures replace the content."""
    content = result.get("content")
    if isinstance(content, str):
        try:
            return {**result, "content": redact_structured_text(content)}
        except Exception:  # noqa: BLE001 - REDA-05 redact-or-omit
            return {**result, "content": _FAILED}
    if isinstance(content, list):
        return {**result, "content": _redact_text_parts(content)}
    return result


def redact_step(step: dict[str, Any]) -> dict[str, Any]:
    """Return a redacted copy of one transcript step (REDA-04, REDA-05).

    Redacts ``message``, ``reasoning_content``, every tool-call argument and every
    observation result's text. Internal failures replace every text-bearing surface
    with ``[REDACTION_FAILED]``, never raw pass-through."""
    try:
        updated = dict(step)
        message = step.get("message")
        if isinstance(message, str):
            updated["message"] = redact_structured_text(message)
        elif isinstance(message, list):
            updated["message"] = _redact_text_parts(message)
        if step.get("reasoning_content") is not None:
            updated["reasoning_content"] = redact_structured_text(step["reasoning_content"])
        if step.get("tool_calls") is not None:
            updated["tool_calls"] = [{**call, "arguments": _redact_arguments(call["arguments"])}
                                     for call in step["tool_calls"]]
        if step.get("observation") is not None:
            updated["observation"] = {**step["observation"], "results": [
                redact_result_content(result) for result in step["observation"]["results"]]}
        return updated
    except Exception as exc:  # noqa: BLE001 - REDA-05 redact-or-omit (top-level fallback)
        ui.print_warning(_console, f"Redactor failure: {type(exc).__name__}")
        # Wipe every text-bearing surface: a partial wipe could leak a secret.
        safe = dict(step, message=_FAILED)
        if step.get("reasoning_content") is not None:
            safe["reasoning_content"] = _FAILED
        if step.get("tool_calls") is not None:
            safe["tool_calls"] = [{**call, "arguments": {"_redaction": _FAILED}} for call in step["tool_calls"]]
        if step.get("observation") is not None:
            safe["observation"] = {"results": [{**result, "content": _FAILED}
                                               for result in step["observation"]["results"]]}
        return safe
