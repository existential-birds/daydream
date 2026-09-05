"""Tests for the shared repository-path helpers.

Covers ``is_test_path``, the deterministic test-vs-production path classifier
promoted out of ``improve/assemble.py`` (issue #1113) so the improve plan gates
and grounded-diagram eligibility answer that question from one implementation.
"""
from __future__ import annotations

import pytest

from daydream import repository_paths
from daydream.repository_paths import is_test_path


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_runner.py",
        "TESTS/helper.py",
        "src/__tests__/widget.tsx",
        "app/spec/models/user.rb",
        "app/specs/models/user.rb",
        "pkg/test/fixture.go",
        "test_runner.py",
        "runner_test.go",
        "src/tests.py",
        "src/test.py",
        "src/WidgetTest.java",
        "src/WidgetTests.swift",
        "src/widget.test.ts",
        "src/widget.spec.ts",
        "deep/nesting/tests/a/b/c.py",
    ],
)
def test_conventional_test_paths_are_classified_as_tests(path: str) -> None:
    assert is_test_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "daydream/runner.py",
        "src/widget.tsx",
        "src/latest.py",
        "src/contest.py",
        "src/protest_handler.go",
        "testing/harness.py",
        "src/testable.py",
        "README.md",
        "",
    ],
)
def test_production_paths_are_not_classified_as_tests(path: str) -> None:
    assert is_test_path(path) is False


def test_test_directory_match_ignores_the_final_segment() -> None:
    """A *file* literally named ``spec`` in a production tree is matched by the
    name rule, not the directory rule — and a directory segment named after the
    file's own basename must not leak into the parent set."""
    # "spec" as a parent directory: directory rule.
    assert is_test_path("app/spec/user.py") is True
    # "specs" as the file's own stem is not a test-naming convention; only the
    # parent-directory rule covers the plural directory spelling.
    assert is_test_path("app/specs.py") is False


def test_camel_case_suffix_is_case_sensitive_but_directories_are_not() -> None:
    """``Test``/``Tests`` is a JVM/Swift naming convention that only reads as
    one in its original casing; directory names fold case."""
    assert is_test_path("src/WidgetTest.java") is True
    assert is_test_path("src/widgettest.java") is False
    assert is_test_path("SRC/Tests/widget.java") is True


def test_is_test_path_is_public_api() -> None:
    """#1113: two flows import it, so it is exported, not incidental."""
    assert "is_test_path" in repository_paths.__all__
