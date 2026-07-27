"""Tests for detect_test_success() pattern matching."""

import pytest

from daydream.agent import detect_test_success


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        pytest.param(
            """Tests PASS ✅

Summary:
- 1,261 tests passed across 12 crates
- 0 tests failed
- 46 tests ignored
""",
            True,
            id="agent-emoji-summary-multiline",
        ),
        pytest.param(
            "test result: ok. 310 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out",
            True,
            id="cargo-native-output",
        ),
        pytest.param(
            "===== 5 passed, 0 failed in 1.23s =====",
            True,
            id="pytest-style-inline",
        ),
        pytest.param(
            """Ran test suite.
5 passed
0 failed
""",
            True,
            id="pytest-counts-on-separate-lines",
        ),
        pytest.param("100 tests passed, 3 failed", False, id="failing-tests-nonzero-count"),
        pytest.param("0 tests passed, 5 failed", False, id="zero-passed-nonzero-failed"),
        pytest.param("5 tests failed during the run", False, id="n-tests-failed-wording"),
        pytest.param("FAILED (failures=3)", False, id="unittest-style-failed"),
        pytest.param("all tests passed", True, id="all-tests-passed-phrase"),
        pytest.param("Run complete: no failures", True, id="no-failures-phrase"),
        pytest.param(
            """Running tests...
Traceback (most recent call last):
  File "x.py", line 1, in <module>
    foo()
""",
            False,
            id="traceback-in-output",
        ),
        pytest.param("", False, id="empty-output"),
        pytest.param("the change passed review", False, id="bare-passed-word-no-count"),
        pytest.param("AssertionError: expected 1, got 2", False, id="assertion-error"),
        pytest.param("Results: 0 failures", True, id="zero-failures-sentinel"),
        pytest.param("Results: 10 failures", False, id="ten-failures-not-success"),
        pytest.param("1,234 passed / 0 failures", True, id="comma-separated-counts-with-zero-failures"),
        pytest.param("5 failures during the run", False, id="bare-failures-wording"),
        pytest.param("1,002 passed, 2,500 failed", False, id="comma-separated-failed-count"),
        pytest.param(
            """First attempt: 10 passed, 0 failed
Retry after flake: 8 passed, 5 failed
""",
            False,
            id="later-nonzero-failed-not-hidden-by-earlier-zero",
        ),
        pytest.param(
            """all tests pass
...but then:
Traceback (most recent call last):
  File "x.py", line 1, in <module>
    foo()
""",
            False,
            id="traceback-overrides-success-sentinel",
        ),
        pytest.param(
            "2528 passed, 391 deselected, 1 warning in 30.30s",
            True,
            id="pytest-deselected-passed",
        ),
        pytest.param("100 passed in 5.2s", True, id="pytest-bare-passed-no-failed"),
        pytest.param("50 passed, 3 skipped in 2.1s", True, id="pytest-skipped-passed"),
        pytest.param("10 passed, 2 xfailed", True, id="pytest-xfailed-passed"),
        pytest.param("100 tests passed, 0 failed", True, id="explicit-zero-failed-with-tests-passed"),
        pytest.param("1 passed, 2 errors in 1.0s", False, id="pytest-errors-are-not-pass"),
    ],
)
def test_detect_test_success(output: str, expected: bool) -> None:
    assert detect_test_success(output) is expected
