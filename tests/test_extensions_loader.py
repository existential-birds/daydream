"""Tests for extension discovery, the version gate, and build_registry()."""

import pytest

from daydream.extensions import (
    ExtensionError,
    ExtensionVersionError,
    build_registry,
)
from tests.conftest import ExtDir


def test_v5_extension_now_rejected_after_hard_bump(ext_dir: ExtDir) -> None:
    """M10: the hard break raises the floor — a v5 extension fails the version gate."""
    from daydream.extensions import (
        EXTENSION_API_VERSION,
        MIN_SUPPORTED_EXTENSION_API_VERSION,
    )

    assert EXTENSION_API_VERSION == 6 and MIN_SUPPORTED_EXTENSION_API_VERSION == 6
    ext_dir.write_module("def register(r): ...\n", api_version=5)
    with pytest.raises(ExtensionVersionError, match=r"supports 6\.\.6"):
        build_registry()


def test_supported_extension_loads(ext_dir: ExtDir) -> None:
    """Load the supported extension API version and apply its registry override."""
    ext_dir.write_module(
        "def _prompt():\n"
        "    return 'v6-review'\n"
        "def register(registry):\n"
        "    registry.override_prompt('review', _prompt)\n",
        api_version=6,
    )
    assert build_registry().prompt("review")() == "v6-review"


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        pytest.param("99", r"99.*supports 6\.\.6", id="above-ceiling"),
        pytest.param("0", r"= 0;.*supports 6\.\.6", id="below-floor"),
        pytest.param("5", r"= 5;.*supports 6\.\.6", id="aged-out-v5"),
        pytest.param("'1'", r"= '1';.*supports 6\.\.6", id="string"),
        pytest.param("1.5", r"= 1\.5;.*supports 6\.\.6", id="float"),
        pytest.param("True", r"= True;.*supports 6\.\.6", id="bool"),
    ],
)
def test_unsupported_extension_version_is_rejected(
    ext_dir: ExtDir,
    declaration: str,
    message: str,
) -> None:
    """Reject missing, malformed, boolean, and unsupported API declarations."""
    ext_dir.write_module("def register(registry): ...\n", api_version=declaration)
    with pytest.raises(ExtensionVersionError, match=message):
        build_registry()


def test_register_exception_is_wrapped_and_named(ext_dir: ExtDir) -> None:
    ext_dir.write_module("def register(registry):\n    raise RuntimeError('boom')\n")
    with pytest.raises(ExtensionError, match=r"daydream_ext.*boom"):
        build_registry()
