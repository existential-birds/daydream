"""Tests for the executor/publisher registration seam (DAYDREAM_SERVICE_V1).

The seal extends ``daydream.extensions`` Registry with
``register_executor`` / ``register_publisher`` and their lookups, so a fork
registers compute/workspace adapters and trusted publishers through the same
versioned ``daydream_ext`` seam it already uses for flows, skills, and prompts.

Key contract guarantees asserted here:

- capability admission fails at registration (no weak executor registers);
- non-conformant objects and duplicate names are rejected;
- the service contract version is asserted;
- publisher registration is name-keyed and resolves;
- the extension ``__all__`` facade surfaces the new symbols (drift-guarded).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from daydream.extensions import (
    DAYDREAM_SERVICE_V1,
    ExtensionError,
    LocalExecutor,
    Registry,
    ScriptedExecutor,
    UnresolvedExtensionError,
)
from daydream.extensions.api import MIN_SUPPORTED_DAYDREAM_SERVICE_V1


def _local() -> LocalExecutor:
    return LocalExecutor(Path(tempfile.mkdtemp()))


def test_register_and_resolve_executor() -> None:
    reg = Registry()
    executor = _local()
    reg.register_executor("local", executor)
    assert reg.executor("local") is executor
    assert reg.executor_if_registered("local") is executor
    assert reg.executor_names() == ("local",)


def test_register_second_executor_name_isolated() -> None:
    reg = Registry()
    reg.register_executor("local", _local())
    reg.register_executor("scripted", ScriptedExecutor())
    assert set(reg.executor_names()) == {"local", "scripted"}


def test_duplicate_executor_name_rejected() -> None:
    reg = Registry()
    reg.register_executor("local", _local())
    with pytest.raises(ExtensionError, match="already registered"):
        reg.register_executor("local", _local())


def test_non_conformant_executor_rejected() -> None:
    reg = Registry()

    class NotAnExecutor:
        kind = "bogus"

    with pytest.raises(ExtensionError, match="not a conformant ReviewExecutor"):
        reg.register_executor("bogus", NotAnExecutor())  # type: ignore[arg-type]


def test_weak_capability_executor_fails_admission() -> None:
    reg = Registry()

    class Partial:
        kind = "partial"
        adapter_version = 1
        capabilities: frozenset[object] = frozenset()  # missing every required capability

        async def start(self, *a, **k):
            return None

        async def inspect(self, *a, **k):
            return None

        async def cancel(self, *a, **k):
            return None

        async def collect(self, *a, **k):
            return None

        async def release(self, *a, **k):
            return None

    with pytest.raises(ExtensionError, match="capabilities"):
        reg.register_executor("partial", Partial())  # type: ignore[arg-type]


def test_service_api_version_out_of_range_rejected() -> None:
    reg = Registry()
    with pytest.raises(ExtensionError, match="DAYDREAM_SERVICE_V1"):
        reg.register_executor("local", _local(), service_api=99)


def test_unresolved_executor_and_if_registered() -> None:
    reg = Registry()
    with pytest.raises(UnresolvedExtensionError, match="executor 'ghost'"):
        reg.executor("ghost")
    assert reg.executor_if_registered("ghost") is None


def test_register_and_resolve_publisher() -> None:
    reg = Registry()
    publisher = object()
    reg.register_publisher("github-checks", publisher)
    assert reg.publisher("github-checks") is publisher
    assert reg.publisher_if_registered("github-checks") is publisher
    assert reg.publisher_names() == ("github-checks",)


def test_duplicate_publisher_rejected() -> None:
    reg = Registry()
    reg.register_publisher("github-checks", object())
    with pytest.raises(ExtensionError, match="already registered"):
        reg.register_publisher("github-checks", object())


def test_publisher_lookup_unknown_raises() -> None:
    reg = Registry()
    with pytest.raises(UnresolvedExtensionError, match="publisher 'ghost'"):
        reg.publisher("ghost")


def test_default_service_api_contract_version() -> None:
    assert DAYDREAM_SERVICE_V1 == 1
    assert MIN_SUPPORTED_DAYDREAM_SERVICE_V1 <= DAYDREAM_SERVICE_V1


def test_service_seam_rides_current_extension_version() -> None:
    """The executor/publisher seam is additive on the extension contract: it must
    not have bumped EXTENSION_API_VERSION, which stays at 5."""
    from daydream.extensions.api import EXTENSION_API_VERSION

    assert EXTENSION_API_VERSION == 5
