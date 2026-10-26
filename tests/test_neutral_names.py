"""Project-owned identifiers use canonical names without corpus-generation suffixes.

External protocol names, such as Anthropic's claude-pretooluse, are outside this rule.
"""

import importlib

import pytest

from daydream.backends import AUDIT_ROOT_ISOLATION


def test_audit_root_isolation_constant_neutral() -> None:
    assert AUDIT_ROOT_ISOLATION == "claude-pretooluse"
    with pytest.raises(AttributeError):
        getattr(importlib.import_module("daydream.backends"), "AUDIT_ROOT_ISOLATION_V1")
