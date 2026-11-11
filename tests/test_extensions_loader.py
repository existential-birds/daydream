"""Tests for extension discovery, the version gate, and build_registry()."""

import pytest

from daydream.extensions import ExtensionError, build_registry
from tests.conftest import ExtDir


def test_unsupported_extension_version_is_rejected(ext_dir: ExtDir) -> None:
    """The declaration must be the integer API version; an equal float is rejected."""
    ext_dir.write_module("def register(registry): ...\n", api_version="9.0")
    with pytest.raises(ExtensionError, match=r"must declare DAYDREAM_EXT_API = 9; found 9\.0 "):
        build_registry()

def test_register_exception_is_wrapped_and_named(ext_dir: ExtDir) -> None:
    ext_dir.write_module("def register(registry):\n    raise RuntimeError('boom')\n")
    with pytest.raises(ExtensionError, match=r"daydream_ext.*boom"):
        build_registry()
