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
        "DAYDREAM_EXT_API = 1\n"
        "def register(registry):\n"
        "    registry.override_skill('structural', 'ro-core:review-structure')\n"
    )
    assert build_registry().skill("structural") == "ro-core:review-structure"


def test_version_mismatch_names_both_versions(ext_dir: ExtDir) -> None:
    ext_dir.write_module("DAYDREAM_EXT_API = 99\ndef register(registry): ...\n")
    with pytest.raises(ExtensionVersionError, match=r"99.*expects 1|expects 1.*99"):
        build_registry()


def test_register_exception_is_wrapped_and_named(ext_dir: ExtDir) -> None:
    ext_dir.write_module("DAYDREAM_EXT_API = 1\ndef register(registry):\n    raise RuntimeError('boom')\n")
    with pytest.raises(ExtensionError, match=r"daydream_ext.*boom"):
        build_registry()
