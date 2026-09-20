"""Single owner of improve's ``redact_model_value`` model-value redaction policy.

The policy does not live in ``render.py`` because that module is a declared pure
Markdown renderer; privacy policy owned there is invisible authority at every
call site.
"""

from __future__ import annotations

from typing import Any

from daydream.trajectory import redact_text


def redact_model_value(value: Any) -> Any:
    """Redact nested model-authored strings before durable host rendering."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_model_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_model_value(item) for item in value)
    if isinstance(value, dict):
        return {
            key: redact_model_value(item)
            for key, item in value.items()
        }
    return value
