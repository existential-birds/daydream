"""Project-owned data identifiers carry neutral names after #1093.

The v1/v2 suffixes named corpus *generations* the project invented; with no
production training run ever completed, every generation prefix is removed and
each identifier keeps one canonical name. External tool-protocol strings
(e.g. ``claude-pretooluse`` naming Anthropic's hook API) are not project-owned
and are not covered here.
"""

import pytest


def test_audit_root_isolation_constant_neutral() -> None:
    from daydream.backends import AUDIT_ROOT_ISOLATION

    assert AUDIT_ROOT_ISOLATION == "claude-pretooluse"
    with pytest.raises(AttributeError):
        import importlib

        getattr(importlib.import_module("daydream.backends"), "AUDIT_ROOT_ISOLATION_V1")


def test_calibration_schema_version_neutral() -> None:
    from daydream.training.calibration import ARTIFACT_SCHEMA_VERSION

    assert ARTIFACT_SCHEMA_VERSION == "calibration-artifact"


def test_member_alias_prefix_neutral() -> None:
    from daydream.improve.prioritize import member_alias

    assert member_alias({"title": "x"}).startswith("member:")
