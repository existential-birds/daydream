"""Tests for extension discovery, the version gate, and build_registry()."""

import pytest

from daydream.extensions import (
    ExtensionError,
    ExtensionVersionError,
    build_registry,
)
from tests.conftest import ExtDir


def test_loader_applies_extension(ext_dir: ExtDir) -> None:
    ext_dir.write_module(
        "DAYDREAM_EXT_API = 2\n"
        "def register(registry):\n"
        "    registry.override_skill('structural', 'ro-core:review-structure')\n"
    )
    assert build_registry().skill("structural") == "ro-core:review-structure"


def test_supported_nonpreferred_version_is_accepted(ext_dir: ExtDir) -> None:
    # DAYDREAM_EXT_API = 1 is within the supported range but below the preferred
    # version: the rolling-upgrade window (issue #274). Strict equality rejected it.
    ext_dir.write_module(
        "DAYDREAM_EXT_API = 1\n"
        "def register(registry):\n"
        "    registry.override_skill('structural', 'ro-core:review-structure')\n"
    )
    assert build_registry().skill("structural") == "ro-core:review-structure"


def test_version_above_ceiling_is_rejected(ext_dir: ExtDir) -> None:
    # 99 is above the ceiling.
    ext_dir.write_module("DAYDREAM_EXT_API = 99\ndef register(registry): ...\n")
    with pytest.raises(ExtensionVersionError, match=r"99.*supports 1\.\.2"):
        build_registry()


def test_version_below_floor_is_rejected(ext_dir: ExtDir) -> None:
    # Below the supported floor: a contract the tool has dropped.
    ext_dir.write_module("DAYDREAM_EXT_API = 0\ndef register(registry): ...\n")
    with pytest.raises(ExtensionVersionError, match=r"= 0;.*supports 1\.\.2"):
        build_registry()


def test_string_version_is_rejected(ext_dir: ExtDir) -> None:
    # A str declaration must not slip through the range comparison as a TypeError.
    ext_dir.write_module("DAYDREAM_EXT_API = '1'\ndef register(registry): ...\n")
    with pytest.raises(ExtensionVersionError, match=r"= '1';.*supports 1\.\.2"):
        build_registry()


def test_float_version_is_rejected(ext_dir: ExtDir) -> None:
    # 1.5 sits inside the numeric range but is not an integer contract version.
    ext_dir.write_module("DAYDREAM_EXT_API = 1.5\ndef register(registry): ...\n")
    with pytest.raises(ExtensionVersionError, match=r"= 1\.5;.*supports 1\.\.2"):
        build_registry()


def test_bool_version_is_rejected(ext_dir: ExtDir) -> None:
    # True == 1 would pass the range check; bool is not a valid API declaration.
    ext_dir.write_module("DAYDREAM_EXT_API = True\ndef register(registry): ...\n")
    with pytest.raises(ExtensionVersionError, match=r"= True;.*supports 1\.\.2"):
        build_registry()


def test_register_exception_is_wrapped_and_named(ext_dir: ExtDir) -> None:
    ext_dir.write_module("DAYDREAM_EXT_API = 2\ndef register(registry):\n    raise RuntimeError('boom')\n")
    with pytest.raises(ExtensionError, match=r"daydream_ext.*boom"):
        build_registry()


def test_supported_range_invariant() -> None:
    from daydream.extensions import (
        EXTENSION_API_VERSION,
        MIN_SUPPORTED_EXTENSION_API_VERSION,
    )

    assert 1 <= MIN_SUPPORTED_EXTENSION_API_VERSION <= EXTENSION_API_VERSION
