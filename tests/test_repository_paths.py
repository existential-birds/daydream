"""Shared repository-path confinement and deterministic test-file classification."""
from __future__ import annotations

import pytest

from daydream.repository_paths import is_test_path


def test_conventional_test_paths_are_classified_as_tests(
    subtests: pytest.Subtests,
) -> None:
    cases: list[str] = [
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
    ]
    for path in cases:
        with subtests.test(path=path):
            assert is_test_path(path) is True

def test_production_paths_are_not_classified_as_tests(
    subtests: pytest.Subtests,
) -> None:
    cases: list[str] = [
        "daydream/runner.py",
        "src/widget.tsx",
        "src/latest.py",
        "src/contest.py",
        "src/protest_handler.go",
        "testing/harness.py",
        "src/testable.py",
        "README.md",
        "",
    ]
    for path in cases:
        with subtests.test(path=path):
            assert is_test_path(path) is False

def test_test_directory_match_ignores_the_final_segment() -> None:
    """A *file* literally named ``spec`` in a production tree is matched by the name rule, not the directory rule —
    and a directory segment named after the file's own basename must not leak into the parent set."""
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
