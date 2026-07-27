"""Tests for extension discovery, the version gate, and build_registry()."""

import pytest

from daydream.extensions import (
    ExtensionError,
    ExtensionVersionError,
    build_registry,
)
from tests.conftest import ExtDir


@pytest.mark.parametrize(
    ("version", "skill"),
    [
        pytest.param(1, "ro-core:review-structure", id="supported-floor"),
        pytest.param(2, "ro-core:review-structure", id="preferred-at-introduction"),
        pytest.param(3, "ro-core:v3-structure", id="supported-ceiling"),
    ],
)
def test_supported_extension_loads(ext_dir: ExtDir, version: int, skill: str) -> None:
    """Load supported extension API versions and apply their registry override."""
    ext_dir.write_module(
        f"DAYDREAM_EXT_API = {version}\n"
        "def register(registry):\n"
        f"    registry.override_skill('structural', '{skill}')\n"
    )
    assert build_registry().skill("structural") == skill


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        pytest.param("99", r"99.*supports 1\.\.3", id="above-ceiling"),
        pytest.param("0", r"= 0;.*supports 1\.\.3", id="below-floor"),
        pytest.param("'1'", r"= '1';.*supports 1\.\.3", id="string"),
        pytest.param("1.5", r"= 1\.5;.*supports 1\.\.3", id="float"),
        pytest.param("True", r"= True;.*supports 1\.\.3", id="bool"),
    ],
)
def test_unsupported_extension_version_is_rejected(
    ext_dir: ExtDir,
    declaration: str,
    message: str,
) -> None:
    """Reject missing, malformed, boolean, and unsupported API declarations."""
    ext_dir.write_module(f"DAYDREAM_EXT_API = {declaration}\ndef register(registry): ...\n")
    with pytest.raises(ExtensionVersionError, match=message):
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
