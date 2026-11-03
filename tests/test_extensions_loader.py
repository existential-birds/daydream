"""Tests for extension discovery, the version gate, and build_registry()."""

import pytest

from daydream.extensions import (
    ExtensionError,
    ExtensionVersionError,
    build_registry,
)
from tests.conftest import ExtDir


@pytest.mark.parametrize(("declaration", "message"),
    [
        pytest.param("1.5", r"= 1\.5;.*supports 8\.\.8", id="float"),
        pytest.param("True", r"= True;.*supports 8\.\.8", id="bool"),
    ],
)
def test_unsupported_extension_version_is_rejected(ext_dir: ExtDir, declaration: str, message: str,) -> None:
    """Reject missing, malformed, boolean, and unsupported API declarations."""
    ext_dir.write_module("def register(registry): ...\n", api_version=declaration)
    with pytest.raises(ExtensionVersionError, match=message):
        build_registry()

def test_register_exception_is_wrapped_and_named(ext_dir: ExtDir) -> None:
    ext_dir.write_module("def register(registry):\n    raise RuntimeError('boom')\n")
    with pytest.raises(ExtensionError, match=r"daydream_ext.*boom"):
        build_registry()
