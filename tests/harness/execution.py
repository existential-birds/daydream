"""Shared builders for test-execution fixtures."""

from __future__ import annotations

from typing import Any

from daydream.test_execution import TestExecutionIdentity


def test_execution_identity(**overrides: Any) -> TestExecutionIdentity:
    """Build a reusable host-passed execution identity, overriding named fields.

    The defaults describe a passed host ``uv run pytest`` run against the
    current working directory; pass ``cwd_relative``, ``interpreter``,
    ``config_digest``, ``absent_components``, ``kind``, ``outcome``, etc. to
    vary one dimension at a time.
    """
    fields: dict[str, Any] = {
        "session_id": "s",
        "argv": ("uv", "run", "pytest"),
        "cwd_relative": ".",
        "runner": "uv",
        "interpreter": None,
        "config_digest": "d" * 64,
        "absent_components": (),
        "input_tree_key": "t",
        "output_tree_key": "t",
        "head_sha": "a" * 40,
        "branch": "feature",
        "kind": "host",
        "outcome": "passed",
    }
    return TestExecutionIdentity(**{**fields, **overrides})
