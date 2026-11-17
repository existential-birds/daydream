"""Copy JSON containers back to canonical dict/list values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def thaw_json(value: Any) -> Any:
    """Copy immutable JSON containers back to ordinary dictionaries and lists."""
    if isinstance(value, Mapping):
        return {key: thaw_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(child) for child in value]
    return value
