"""Tests for detect_test_success() pattern matching."""

from daydream.agent import detect_test_success


def test_agent_emoji_summary_multiline() -> None:
    """The real reported regression: multi-line summary with emoji."""
    output = """Tests PASS ✅

Summary:
- 1,261 tests passed across 12 crates
- 0 tests failed
- 46 tests ignored
"""
    assert detect_test_success(output) is True


def test_cargo_native_output() -> None:
    """Cargo's native summary uses semicolons and a `test result: ok` sentinel."""
    output = "test result: ok. 310 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"
    assert detect_test_success(output) is True


def test_pytest_style_inline() -> None:
    """Pytest-style single-line summary."""
    output = "===== 5 passed, 0 failed in 1.23s ====="
    assert detect_test_success(output) is True


def test_pytest_counts_on_separate_lines() -> None:
    """Counts split across newlines — the main regression case."""
    output = """Ran test suite.
5 passed
0 failed
"""
    assert detect_test_success(output) is True


def test_failing_tests_nonzero_count() -> None:
    """A non-zero failure count must always return False."""
    assert detect_test_success("100 tests passed, 3 failed") is False


def test_zero_passed_nonzero_failed() -> None:
    """Pathological: 0 passed with 5 failed is still a failure."""
    assert detect_test_success("0 tests passed, 5 failed") is False


def test_n_tests_failed_wording() -> None:
    """The wording `N tests failed` must register as failure."""
    assert detect_test_success("5 tests failed during the run") is False


def test_unittest_style_failed() -> None:
    """Python unittest's FAILED (failures=N) output."""
    assert detect_test_success("FAILED (failures=3)") is False


def test_all_tests_passed_phrase() -> None:
    """Plain 'all tests passed' phrase."""
    assert detect_test_success("all tests passed") is True


def test_no_failures_phrase() -> None:
    """'no failures' sentinel."""
    assert detect_test_success("Run complete: no failures") is True


def test_traceback_in_output() -> None:
    """A traceback anywhere in the output indicates failure."""
    output = """Running tests...
Traceback (most recent call last):
  File "x.py", line 1, in <module>
    foo()
"""
    assert detect_test_success(output) is False


def test_empty_output() -> None:
    """Empty output must not be treated as success."""
    assert detect_test_success("") is False


def test_bare_passed_word_no_count() -> None:
    """Bare 'passed' without a count is not sufficient — conservative fallback."""
    assert detect_test_success("the change passed review") is False


def test_assertion_error() -> None:
    """Assertion errors indicate failure."""
    assert detect_test_success("AssertionError: expected 1, got 2") is False


def test_zero_failures_sentinel() -> None:
    """'0 failures' sentinel alone is success."""
    assert detect_test_success("Results: 0 failures") is True


def test_ten_failures_not_success() -> None:
    """Regression: '10 failures' must not match the '0 failures?' sentinel via substring."""
    assert detect_test_success("Results: 10 failures") is False


def test_comma_separated_counts_with_zero_failures() -> None:
    """Large suites format counts with commas; must still pass with zero failures."""
    assert detect_test_success("1,234 passed / 0 failures") is True


def test_bare_failures_wording() -> None:
    """The wording `N failures` (no 'failed') must register as failure."""
    assert detect_test_success("5 failures during the run") is False


def test_comma_separated_failed_count() -> None:
    """Comma-grouped failure counts must register as failure."""
    assert detect_test_success("1,002 passed, 2,500 failed") is False
