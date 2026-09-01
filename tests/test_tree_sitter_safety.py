"""Unit tests for the shared tree-sitter version guard (#1087)."""

from importlib.metadata import PackageNotFoundError

import pytest

from daydream import _tree_sitter_safety as safety
from daydream._tree_sitter_safety import TreeSitterBadVersionError


def _bad_install() -> str:
    return "0.26.0"


def _good_install() -> str:
    return "0.25.2"


def _not_installed(pkg: str) -> None:
    raise PackageNotFoundError(pkg)


def test_guard_raises_on_known_bad_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(safety, "installed_tree_sitter_version", _bad_install)
    with pytest.raises(TreeSitterBadVersionError) as exc:
        safety.assert_tree_sitter_safe()
    assert "0.26.0" in str(exc.value)  # reason names the offending version (S2)


def test_guard_passes_on_known_good_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(safety, "installed_tree_sitter_version", _good_install)
    safety.assert_tree_sitter_safe()  # no raise


def test_guard_passes_when_version_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(safety, "installed_tree_sitter_version", _not_installed)
    safety.assert_tree_sitter_safe()  # not installed is the existing degrade path, not bad


def test_unavailable_reason_names_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(safety, "installed_tree_sitter_version", _bad_install)
    reason = safety.tree_sitter_unavailable_reason()
    assert reason is not None and "0.26.0" in reason
