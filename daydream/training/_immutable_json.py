"""Owned immutable JSON values with canonical dict/list serialization."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def freeze_json(value: Any) -> Any:
    """Copy JSON containers into immutable mappings and tuples recursively."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_json(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(child) for child in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    """Copy immutable JSON containers back to ordinary dictionaries and lists."""
    if isinstance(value, Mapping):
        return {key: thaw_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(child) for child in value]
    return value
