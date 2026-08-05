"""Tests for the dependency-neutral prompt-size policy."""


def test_inline_diff_budget_uses_utf8_bytes() -> None:
    """The shared policy accepts its boundary and rejects oversized text."""
    from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES, fits_inline_diff_budget

    assert fits_inline_diff_budget("x" * INLINE_DIFF_BUDGET_BYTES)
    assert not fits_inline_diff_budget("x" * (INLINE_DIFF_BUDGET_BYTES + 1))
    assert not fits_inline_diff_budget("あ" * (INLINE_DIFF_BUDGET_BYTES // 2))
